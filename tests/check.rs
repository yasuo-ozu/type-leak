//! Tests for `View::check()` and `Leaker::allowed_paths`.

mod common;

use common::check_in;
use syn::*;
use type_leak::{CheckResult, ContextId, Leaker, NotInternableError, Referrer, View};

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
            .unwrap()
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
}

#[test]
fn test_check_self_no_internable() {
    let test_struct: ItemStruct = parse_quote!(
        pub struct MyStruct<'l1, const N1: usize, T1, T2, T3>;
    );

    // `Self` of a struct scope is allowed by `check()` if `allow_self` is set.
    let mut leaker = Leaker::new();
    let result = leaker
        .enter_type(&test_struct.ident, &test_struct.generics, true, |view| {
            view.check(&parse_quote!(Self))
        })
        .unwrap()
        .1;
    assert_eq!(kind(&result), "Neutral");

    // With `allow_self` unset, a type mentioning `Self` cannot be interned as a whole, as it is
    // for the items.
    let item: ItemStruct = parse_quote!(
        pub struct MyStruct<T1> {
            a: Rel<Self>,
            b: T1,
        }
    );
    assert!(Leaker::new().intern_struct(&item, true).is_ok());
    assert!(Leaker::new().intern_struct(&item, false).is_err());
    // The types which do not mention it are not affected.
    let item: ItemStruct = parse_quote!(
        pub struct MyStruct<T1> {
            a: Rel<T1>,
            b: Self,
        }
    );
    assert!(Leaker::new().intern_struct(&item, false).is_ok());
}

/// README, External crates: a path that starts with `crate` is leaked, unless the allowed paths
/// have `crate` (`LeakerConfig::allow_crate()`). A real `$crate` token (which a `macro_rules!`
/// expansion gives to a proc macro) cannot be made outside one, so the `$crate` paths are pinned
/// by the fixture crate (`DollarCrate` in `tests/crates/leaker`, with `check_leak!`, and
/// `cross_crate::dollar_crate_paths`).
#[test]
fn test_check_allows_crate_paths_only_if_allowed() {
    let mut leaker = Leaker::new();
    leaker.allowed_paths.push(parse_quote!(crate));

    let item: ItemTrait = parse_quote!(
        pub trait Parse<Atom>: ::core::marker::Sized {
            type Error: crate::error::Error;
            fn parse(
                stream: impl crate::parse::into_parse_stream::IntoParseStream<Atom = Atom>,
            ) -> ::core::result::Result<Self, Self::Error>;
        }
    );
    leaker.intern_trait(&item, true).unwrap();
    leaker.reduce_roots();
    assert!(leaker.finish().is_empty());

    // Without it the same paths are leaked (here in bounds, which cannot be leaked): the allowed
    // `crate` is what lets them through.
    let err = Leaker::new().intern_trait(&item, true).unwrap_err();
    let message = Error::from(err).to_string();
    assert!(
        message.contains("use absolute path: 'crate :: error :: Error'"),
        "{}",
        message
    );
}

#[test]
fn test_check_allowed_paths() {
    let generics: Generics = parse_quote!(<'l1, const N1: usize, T1>);
    let mut leaker = Leaker::new();

    leaker.allowed_paths.push(parse_quote!(some::path));

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

    leaker.allowed_paths.push(parse_quote!(::abs::path));
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
    use type_leak::LeakerConfig;

    let generics: Generics = parse_quote!(<T1>);
    let mut config = LeakerConfig::new();
    config.allow_std_prelude().allow_primitive();
    let mut leaker = Leaker::from_config(config);

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
    let mut config = LeakerConfig::new();
    config.allow_std_prelude().allow_primitive();
    let mut leaker = Leaker::from_config(config);
    let item: ItemTrait = parse_quote! {
        trait MyTrait: Into<u8> {
            fn f<I: Iterator<Item = u8>>(&self, i: I);
            fn g<I: Iterator<Item = RelType>>(&self, i: I);
        }
    };
    leaker.intern_trait(&item, true).unwrap();
    leaker.reduce_roots();
    let leaked: std::collections::BTreeSet<String> = leaker
        .finish()
        .iter()
        .map(|t| template_quote::quote!(#t).to_string())
        .collect();
    assert_eq!(leaked, vec!["RelType".to_string()].into_iter().collect());
}

#[test]
fn test_check_conflict() {
    let generics: Generics = parse_quote!(<T>);
    let mut leaker = Leaker::new();
    let strings = |tokens: Vec<proc_macro2::TokenStream>| -> Vec<String> {
        tokens.iter().map(|t| t.to_string()).collect()
    };
    let mut conflict = |ty: Type| {
        let c = check_in(&mut leaker, &ty, &generics);
        (strings(c.must), strings(c.impossible), c.is_own)
    };
    let v = |s: &[&str]| -> Vec<String> { s.iter().map(|s| s.to_string()).collect() };

    // The type's own path depends on the type context: it cannot be decomposed.
    assert_eq!(
        conflict(parse_quote!(MyType<'_>)),
        (v(&["MyType < '_ >"]), v(&["'_"]), true)
    );
    assert_eq!(
        conflict(parse_quote!(<T as MyTrait<'_>>::Ty)),
        (v(&["MyTrait < '_ > :: Ty"]), v(&["'_"]), true)
    );
    // Only components do: it can be decomposed.
    assert_eq!(
        conflict(parse_quote!(::path::Wrap<MyType, '_>)),
        (v(&["MyType"]), v(&["'_"]), false)
    );
    assert_eq!(
        conflict(parse_quote!(<T as ::path::Trait<MyType, '_>>::Ty)),
        (v(&["MyType"]), v(&["'_"]), false)
    );
    // All the causes are collected, the outermost ones first.
    assert_eq!(
        conflict(parse_quote!(
            ::path::Wrap<'_, ::path::Wrap<(C,)>, A, B<D>, impl ::path::Tr>
        )),
        (
            v(&["A", "B < D >", "D", "C"]),
            v(&["'_", "impl :: path :: Tr"]),
            false
        )
    );
    // A bare identifier as an argument may be a const, so the type containing it is interned as a
    // whole (with the cause in its own path) if it can be, as `::path::Wrap<C>` here, but not the
    // outer one, which has impossible causes (`A` is a type of its own).
    assert_eq!(
        conflict(parse_quote!(::path::Wrap<'_, ::path::Wrap<C>, A>)),
        (v(&["C", "A"]), v(&["'_"]), false)
    );
    assert_eq!(
        conflict(parse_quote!(dyn A + B<'_>)),
        (v(&["A", "B < '_ >"]), v(&["'_"]), true)
    );
}

fn strings(tokens: &[proc_macro2::TokenStream]) -> Vec<String> {
    tokens.iter().map(|t| t.to_string()).collect()
}

fn check_ty(view: &mut View<'_>, ty: Type) -> CheckResult {
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
    let (_, results) = leaker
        .enter_mod("m", [parse_quote!(Local)], |view| {
            [
                check_ty(view, parse_quote!(Local)),
                check_ty(view, parse_quote!(self::Local)),
                check_ty(view, parse_quote!(::p::W<(Local,)>)),
                check_ty(view, parse_quote!(Other)),
                check_ty(view, parse_quote!(super::Local)),
            ]
        })
        .unwrap();
    let kinds: Vec<_> = results.iter().map(kind).collect();
    assert_eq!(
        kinds,
        ["Neutral", "Neutral", "Neutral", "MustIntern", "MustIntern"]
    );
    assert_eq!(strings(&results[3].must), ["Other"]);
    assert_eq!(strings(&results[4].must), ["super :: Local"]);
    // A module declares only its own names: not the ones of the module around.
    let (_, inner) = leaker
        .enter_mod("m", [parse_quote!(Local)], |view| {
            view.enter_mod("n", [parse_quote!(Deeper)], |view| {
                [
                    check_ty(view, parse_quote!(Deeper)),
                    check_ty(view, parse_quote!(Local)),
                ]
            })
            .unwrap()
            .1
        })
        .unwrap();
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
            .unwrap()
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
}

#[test]
fn check_self_has_no_meaning_in_generics_and_modules() {
    let generics: Generics = parse_quote!(<T>);
    let mut leaker = Leaker::new();
    let checks = |view: &mut View<'_>| {
        [
            check_ty(view, parse_quote!(Self)),
            check_ty(view, parse_quote!(Self::Item)),
            check_ty(view, parse_quote!(Rel<Self>)),
            check_ty(view, parse_quote!(::p::W<(Self,)>)),
            check_ty(view, parse_quote!(())),
        ]
    };
    let (_, in_generics) = leaker.enter_generics(&generics, checks).unwrap();
    let (_, in_module) = leaker.enter_mod("m", [], checks).unwrap();
    for results in [in_generics, in_module] {
        let [own, assoc, in_rel, in_path, plain] = results;
        // It is an impossible cause, as written, and the note tells why.
        assert_eq!(kind(&own), "MustNotIntern");
        assert_eq!(strings(&own.impossible), ["Self"]);
        let note = own.note.expect("a note");
        assert!(note.starts_with("`Self` has no meaning here"), "{}", note);
        assert_eq!(strings(&assoc.impossible), ["Self :: Item"]);
        assert!(assoc.note.is_some());
        // A type that holds it is a conflict if it is a cause itself, and its parts can be leaked
        // otherwise.
        assert_eq!(kind(&in_rel), "Conflict");
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
    let (_, results) = leaker
        .enter_type(&item.ident, &item.generics, true, |view| {
            view.enter_generics(&parse_quote!(<'a>), checks).unwrap().1
        })
        .unwrap();
    assert_eq!(kind(&results[0]), "Neutral");
    assert_eq!(kind(&results[4]), "Neutral");
    let (_, results) = leaker
        .enter_type(&item.ident, &item.generics, true, |view| {
            view.enter_mod("m", [], checks).unwrap().1
        })
        .unwrap();
    assert_eq!(kind(&results[0]), "MustNotIntern");
}

#[test]
fn check_is_not_the_error_of_the_scope() {
    let mut leaker = Leaker::new();
    // `check()` finds `Self` without leaving the error a visit leaves.
    let result = leaker.enter_generics(&Generics::default(), |view| {
        view.check(&parse_quote!(Self)).impossible.len()
    });
    assert_eq!(result.unwrap().1, 1);
    let result = leaker.enter_generics(&Generics::default(), |view| {
        view.check(&parse_quote!(Self));
        view.visit_type(&parse_quote!(Self));
        view.check(&parse_quote!(Self));
    });
    match result {
        Err(NotInternableError::SelfWithoutMeaning(selves)) => assert_eq!(selves.len(), 1),
        other => panic!("{:?}", other.map(|_| ())),
    }
    // Neither does the conflict of a type leave one.
    let result = leaker.enter_generics(&Generics::default(), |view| {
        view.check(&parse_quote!(MyType<'_>)).impossible.len()
    });
    assert_eq!(result.unwrap().1, 1);
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
    leaker.allowed_paths.push(parse_quote!(::foo));
    let result = check_in(&mut leaker, &parse_quote!(::foo::X), &generics);
    assert_eq!(kind(&result), "Neutral");
    assert!(!result.optional);
    // The names of the module are not copied by the Referrer, so a type using one is not leaked
    // alone, and its path is not an optional entry either.
    let mut leaker = Leaker::new();
    let (_, results) = leaker
        .enter_mod("m", [parse_quote!(Local)], |view| {
            [
                check_ty(view, parse_quote!(::foo::W<(Local,)>)),
                check_ty(view, parse_quote!(::foo::W<(Other,)>)),
            ]
        })
        .unwrap();
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
    let (_, (outer, inner)) = leaker
        .enter_generics(&parse_quote!(<T>), |view| {
            let outer = [
                check_ty(view, parse_quote!(Wrap<T>)),
                check_ty(view, parse_quote!((T, U))),
                check_ty(view, parse_quote!(&'a ())),
            ];
            let (_, inner) = view
                .enter_generics(&parse_quote!(<'a, U>), |view| {
                    [
                        check_ty(view, parse_quote!((T, U))),
                        check_ty(view, parse_quote!(&'a ())),
                        check_ty(view, parse_quote!(::p::W<'a, T, U>)),
                        check_ty(view, parse_quote!((T, V))),
                    ]
                })
                .unwrap();
            // Back in the scope around, which does not know what the nested one declared.
            assert_eq!(kind(&check_ty(view, parse_quote!(U))), "MustIntern");
            (outer, inner)
        })
        .unwrap();
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
            .unwrap()
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
    // It is what the scope reports for the same type.
    let err = leaker
        .enter_type(&item.ident, &item.generics, false, |view| {
            view.visit_type(&ty)
        })
        .unwrap_err();
    match err {
        NotInternableError::RelativePath(conflicts) => {
            assert_eq!(conflicts.len(), 1);
            assert_eq!(strings(&conflicts[0].must), strings(&forbidden.must));
            assert_eq!(
                strings(&conflicts[0].impossible),
                strings(&forbidden.impossible)
            );
            assert_eq!(conflicts[0].note, forbidden.note);
        }
        other => panic!("{:?}", other),
    }
}

/// Types that a visit would record in some way: entries, optional entries, conflicts, errors for
/// `Self`, the probe of a binder, and a module found in an array length.
fn recording_types() -> Vec<Type> {
    vec![
        parse_quote!(Rel<T>),
        parse_quote!(::foo::W<(Rel2<T>,)>),
        parse_quote!(MyType<'_>),
        parse_quote!(impl MyTrait),
        parse_quote!(Self),
        parse_quote!(Rel<Self::Item>),
        parse_quote!(&'a Rel<T>),
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
    let checks = |view: &mut View<'_>| {
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
    let (id, ()) = leaker
        .enter_generics(&item.generics, |view| {
            checks(view);
            view.visit_type(&parse_quote!(Rel<T>));
            checks(view);
            view.visit_type(&parse_quote!(::foo::W<(Rel2<T>,)>));
            let (id, ()) = view
                .enter_mod("m", [parse_quote!(Local)], |view| {
                    checks(view);
                    view.visit_type(&parse_quote!(Rel3));
                    checks(view);
                })
                .unwrap();
            ids.push(id);
            let (id, ()) = view
                .enter_type(&item.ident, &item.generics, true, |view| {
                    checks(view);
                    view.visit_type(&parse_quote!(Rel4<Self>));
                })
                .unwrap();
            ids.push(id);
            checks(view);
        })
        .unwrap();
    ids.push(id);
    // The scopes after it get the ids that the ones before left free.
    let (id, ()) = leaker
        .enter_mod("n", [], |view| {
            checks(view);
            view.visit_type(&parse_quote!(Rel5));
        })
        .unwrap();
    ids.push(id);
    let (id, ()) = leaker
        .enter_generics(&parse_quote!(<U>), |view| {
            view.visit_type(&parse_quote!(Rel6<U>))
        })
        .unwrap();
    ids.push(id);
    (ids, leaker.finish())
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
    let (id, ()) = leaker
        .enter_generics(&parse_quote!(<T>), |view| {
            for ty in &types {
                view.check(ty);
            }
        })
        .unwrap();
    let (other_id, ()) = Leaker::new()
        .enter_generics(&parse_quote!(<T>), |_| ())
        .unwrap();
    assert_eq!(id, other_id);
    let referrer = leaker.finish();
    assert!(referrer.is_empty());
}

//! `LeakerConfig::allow()`: accepting a conflict.
//!
//! `allow(cause)` is looked up only where a conflict is turned into an error: a conflict whose
//! error would have an equal cause (the same kind and the same tokens, whatever the spans and the
//! spacing) is not reported, and its correction stays, which is the type copied as written, whole,
//! with nothing in it leaked. Nothing else changes: a place that is no conflict never looks at the
//! list.
//!
//! - The item below (`f` and `g`): the error, the causes that do not match (the kind and the tokens
//!   count), and that the entries are those of the item without `f`.
//! - The cause that is smaller than the part that is copied (the array with a closure).
//! - Scopes: an inner `enter_*` of a `View` whose conflict is accepted or not, `Self` without
//!   meaning and its place among the errors, the other kinds of cause.
//! - The corrections that no conflict of the type itself makes: an entry that names a `Self`
//!   without meaning, or whose bounds do (in a type, a bound, a default, a module, a value), and a
//!   type that leaves a lifetime of its method unconstrained, are copied whole and have no entry.
//! - What accepting does not make compile (`Self` in a predicate beside a leaked bound).
//! - `View::check()` ignores the list, and `LeakerConfig` with it is `Eq` and `Hash`, whatever the
//!   spacing of the tokens of a cause.
//! - The Referrer's `allowed` list (the last section): the causes that accepted a conflict and the
//!   parts that the Leaker copies as written, their rollback, their serialization, and the visitor,
//!   which leaves a part as written (also beside an entry with the same tokens) and looks up no
//!   entry in it.
//!
//! Every `Reason` is accepted by `tests/error_items.rs`
//! (`allowing_the_causes_of_the_errors_of_each_reason_accepts_the_item`), and a cause with spans
//! of its own by `tests/crates/spans`.

use proc_macro2::TokenStream;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use syn::visit_mut::VisitMut;
use syn::*;
use template_quote::{quote, ToTokens};
use type_leak::*;

/// The trait that is relative in `f` and `g` below.
fn rel_tr() -> Cause {
    Cause::Trait(parse_quote!(RelTr))
}

/// A Leaker that accepts the conflicts with these causes.
fn accepting(causes: &[Cause]) -> Leaker {
    let mut config = LeakerConfig::new();
    for cause in causes {
        config.allow(cause.clone());
    }
    Leaker::from_config(config)
}

/// The `f` that is an error (`W<'_, ..>` cannot be leaked, and `dyn RelTr` is relative) and the `g`
/// that is not (`dyn RelTr` is leaked).
fn f_and_g() -> ItemImpl {
    parse_quote!(
        impl ::p::S {
            fn f(_: ::p::W<'_, dyn RelTr>) {}
            fn g(_: ::p::X<dyn RelTr>) {}
        }
    )
}

/// `g` alone.
fn g_only() -> ItemImpl {
    parse_quote!(
        impl ::p::S {
            fn g(_: ::p::X<dyn RelTr>) {}
        }
    )
}

/// The entries of the Referrer of an item: the type, whether it is required, and its generics.
fn entries(causes: &[Cause], item: &ItemImpl) -> Vec<(String, bool, String)> {
    let mut leaker = accepting(causes);
    leaker.intern_impl(item, true).unwrap();
    let referrer = leaker.finish();
    (0..referrer.iter().count())
        .map(|ix| {
            (
                referrer
                    .iter()
                    .nth(ix)
                    .unwrap()
                    .to_token_stream()
                    .to_string(),
                referrer.must(ix),
                referrer.generics(ix).to_token_stream().to_string(),
            )
        })
        .collect()
}

/// The errors of `f_and_g()` when these causes are accepted.
fn errors_of_f_and_g(causes: &[Cause]) -> Vec<ErrorItem> {
    accepting(causes).intern_impl(&f_and_g(), true).unwrap_err()
}

#[test]
fn the_item_has_one_error_without_the_option() {
    let errors = errors_of_f_and_g(&[]);
    assert_eq!(errors.len(), 1);
    assert_eq!(errors[0].cause, rel_tr());
    assert_eq!(errors[0].reason, Reason::RelativePath);
}

#[test]
fn allowing_the_cause_of_the_error_accepts_the_item() {
    let mut leaker = accepting(&[rel_tr()]);
    assert!(leaker.intern_impl(&f_and_g(), true).is_ok());
    // The cause of the error itself is the same.
    let cause = errors_of_f_and_g(&[]).remove(0).cause;
    assert!(accepting(&[cause]).intern_impl(&f_and_g(), true).is_ok());
}

#[test]
fn the_kind_and_the_tokens_of_the_cause_count() {
    // The same tokens as another kind of cause, other tokens, more tokens, an absolute path.
    for cause in [
        Cause::Type(parse_quote!(RelTr)),
        Cause::Trait(parse_quote!(OtherTr)),
        Cause::Trait(parse_quote!(RelTr<u8>)),
        Cause::Trait(parse_quote!(::RelTr)),
        Cause::Type(parse_quote!(dyn RelTr)),
        Cause::Trait(parse_quote!(::p::W<'_, dyn RelTr>)),
        Cause::Unreadable(parse_quote!(RelTr)),
    ] {
        let errors = errors_of_f_and_g(std::slice::from_ref(&cause));
        assert_eq!(errors.len(), 1, "{:?}", cause);
        assert_eq!(errors[0].cause, rel_tr());
    }
    // A wrong one beside the right one accepts it all the same.
    let wrong = Cause::Type(parse_quote!(RelTr));
    assert!(accepting(&[wrong, rel_tr()])
        .intern_impl(&f_and_g(), true)
        .is_ok());
}

#[test]
fn a_cause_written_by_hand_matches_one_that_the_analysis_made() {
    // Other values of the same tokens: parsed from a string, and read back from the serialized
    // error.
    let made = errors_of_f_and_g(&[]).remove(0);
    let reparsed: ErrorItem = parse2(made.to_token_stream()).unwrap();
    let from_string = Cause::Trait(parse_str("  RelTr ").unwrap());
    for cause in [reparsed.cause, from_string, made.cause] {
        assert_eq!(cause, rel_tr());
        assert!(accepting(&[cause]).intern_impl(&f_and_g(), true).is_ok());
    }
}

#[test]
fn the_entries_are_those_of_the_item_without_the_conflict() {
    let accepted = entries(&[rel_tr()], &f_and_g());
    assert_eq!(accepted, entries(&[], &g_only()));
    // `g` still leaks `dyn RelTr`, which is required, and `::p::X<dyn RelTr>` is optional.
    let find = |ty: &str| accepted.iter().find(|entry| entry.0 == ty).unwrap();
    assert!(find("dyn RelTr").1);
    assert!(!find(":: p :: X < dyn RelTr >").1);
}

#[test]
fn allowed_paths_would_change_more() {
    // The trait made context-free everywhere: `g` loses its required entry too.
    let mut config = LeakerConfig::new();
    config.allowed_paths.push(parse_quote!(RelTr));
    let mut leaker = Leaker::from_config(config);
    leaker.intern_impl(&f_and_g(), true).unwrap();
    let referrer = leaker.finish();
    assert!(referrer.iter().all(|ty| *ty != parse_quote!(dyn RelTr)));
    assert!(!referrer.has_required());
}

#[test]
fn a_cause_that_matches_no_conflict_changes_nothing() {
    // A trait that is a conflict nowhere, and the types that `g` holds, which are no conflict:
    // `g` has the entries it has without them, and the error of `f` stays.
    let causes = [
        Cause::Trait(parse_quote!(OtherTr)),
        Cause::Type(parse_quote!(dyn RelTr)),
        Cause::Type(parse_quote!(::p::X<dyn RelTr>)),
    ];
    assert_eq!(entries(&causes, &g_only()), entries(&[], &g_only()));
    assert_eq!(errors_of_f_and_g(&causes).len(), 1);
    // The paths that are entries in a struct without a conflict: the same entries as without.
    let item: ItemStruct = parse_quote!(
        struct S {
            a: Rel,
            b: ::p::X<Other>,
        }
    );
    let entries_of = |causes: &[Cause]| {
        let mut leaker = accepting(causes);
        leaker.intern_struct(&item, true).unwrap();
        let referrer = leaker.finish();
        referrer
            .iter()
            .enumerate()
            .map(|(ix, ty)| (ty.to_token_stream().to_string(), referrer.must(ix)))
            .collect::<Vec<_>>()
    };
    let causes = [
        Cause::Type(parse_quote!(Rel)),
        Cause::Trait(parse_quote!(Rel)),
        Cause::Type(parse_quote!(Other)),
    ];
    assert_eq!(entries_of(&causes), entries_of(&[]));
}

#[test]
fn one_cause_covers_every_conflict_with_the_same_tokens() {
    let item: ItemImpl = parse_quote!(
        impl ::p::S {
            fn f(_: ::p::W<'_, dyn RelTr>) {}
            fn h(_: ::p::Y<'_, dyn RelTr>) {}
            fn i(_: &::p::Z<'_, dyn RelTr>) {}
        }
    );
    let errors = accepting(&[]).intern_impl(&item, true).unwrap_err();
    assert_eq!(errors.len(), 3);
    assert!(errors.iter().all(|error| error.cause == rel_tr()));
    assert!(accepting(&[rel_tr()]).intern_impl(&item, true).is_ok());
}

#[test]
fn a_conflict_that_is_not_accepted_stays_beside_one_that_is() {
    let item: ItemImpl = parse_quote!(
        impl ::p::S {
            fn f(_: ::p::W<'_, dyn RelTr>) {}
            fn h(_: ::p::Y<'_, dyn Other>) {}
        }
    );
    let errors = accepting(&[rel_tr()]).intern_impl(&item, true).unwrap_err();
    assert_eq!(errors.len(), 1);
    assert_eq!(errors[0].cause, Cause::Trait(parse_quote!(Other)));
}

// ── the cause that is smaller than the part that is copied ───────────────────────────────────

#[test]
fn the_cause_of_an_array_with_a_closure_is_the_type_in_it() {
    let item: ItemImpl = parse_quote!(
        impl ::p::S {
            fn f(_: [u8; { let c = |a: u8| a; 3 }]) {}
            fn g(_: ::p::V<u8>) {}
        }
    );
    // `u8` is the cause, `a` a note: allowing `u8` copies the array whole and `a` is not reached.
    let errors = accepting(&[]).intern_impl(&item, true).unwrap_err();
    assert_eq!(errors.len(), 1);
    assert_eq!(errors[0].cause, Cause::Type(parse_quote!(u8)));
    assert_eq!(
        errors[0].note(),
        Some("it cannot be leaked as a whole because of `| a : u8 | a`")
    );
    let causes = [Cause::Type(parse_quote!(u8))];
    let mut leaker = accepting(&causes);
    leaker.intern_impl(&item, true).unwrap();
    let referrer = leaker.finish();
    // `::p::V<u8>` is a required entry, as it is without the array, and nothing is for `a`.
    let required: Vec<&Type> = referrer.required().map(|(_, ty)| ty).collect();
    assert_eq!(required, [&parse_quote!(::p::V<u8>)]);
    let only_g: ItemImpl = parse_quote!(impl ::p::S { fn g(_: ::p::V<u8>) {} });
    assert_eq!(entries(&causes, &item), entries(&[], &only_g));
    // `a` alone accepts nothing: it is not the cause of the error.
    let a = Cause::Expr(parse_quote!(a));
    assert_eq!(
        accepting(&[a]).intern_impl(&item, true).unwrap_err().len(),
        1
    );
}

// ── scopes ───────────────────────────────────────────────────────────────────────────────────

/// A type with the conflict of `f`.
fn conflict() -> Type {
    parse_quote!(::p::W<'_, dyn RelTr>)
}

#[test]
fn an_accepted_error_in_an_inner_scope_leaves_both_scopes_succeeding() {
    let mut leaker = accepting(&[rel_tr()]);
    let (_, inner) = leaker
        .enter_generics(&Generics::default(), |view| {
            view.enter_generics(&parse_quote!(<T>), |inner| {
                inner.visit_type(&conflict());
                inner.visit_type(&parse_quote!(Rel<T>));
            })
        })
        .unwrap();
    inner.unwrap();
    // The scope stays, with the entry that it also holds.
    let referrer = leaker.finish();
    let required: Vec<&Type> = referrer.required().map(|(_, ty)| ty).collect();
    assert_eq!(required, [&parse_quote!(Rel<T>)]);
}

#[test]
fn an_error_that_is_not_accepted_in_an_inner_scope_rolls_it_back() {
    for causes in [vec![], vec![Cause::Trait(parse_quote!(OtherTr))]] {
        let mut leaker = accepting(&causes);
        let (_, inner) = leaker
            .enter_generics(&Generics::default(), |view| {
                view.enter_generics(&parse_quote!(<T>), |inner| {
                    inner.visit_type(&conflict());
                    inner.visit_type(&parse_quote!(Rel<T>));
                })
            })
            .unwrap();
        let errors = inner.unwrap_err();
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].cause, rel_tr());
        // Nothing of the inner scope stays: not `Rel<T>` either.
        assert!(leaker.finish().is_empty());
    }
}

#[test]
fn the_outer_scope_fails_on_what_the_inner_one_did_not_accept() {
    let mut leaker = accepting(&[rel_tr()]);
    let errors = leaker
        .enter_generics(&Generics::default(), |view| {
            // Accepted: the inner scope succeeds.
            view.enter_generics(&parse_quote!(<T>), |inner| {
                inner.visit_type(&conflict());
                inner.visit_type(&parse_quote!(Rel<T>));
            })
            .unwrap();
            // Not accepted, in the outer scope.
            view.visit_type(&parse_quote!(::p::W<'_, dyn Other>));
        })
        .unwrap_err();
    // Only that one: the accepted conflict of the inner scope is not reported by the outer one.
    assert_eq!(errors.len(), 1);
    assert_eq!(errors[0].cause, Cause::Trait(parse_quote!(Other)));
    // All or nothing: `Rel<T>` is gone too.
    assert!(leaker.finish().is_empty());
}

#[test]
fn an_accepted_error_in_a_module_is_accepted() {
    let item: ItemMod = parse_quote!(
        mod m {
            pub struct S(::p::W<'_, dyn RelTr>, Rel);
        }
    );
    assert!(Leaker::new().intern_mod(&item).is_err());
    let mut leaker = accepting(&[rel_tr()]);
    leaker.intern_mod(&item).unwrap();
    let referrer = leaker.finish();
    let required: Vec<&Type> = referrer.required().map(|(_, ty)| ty).collect();
    assert_eq!(required, [&parse_quote!(Rel)]);
}

// ── `Self` without meaning ───────────────────────────────────────────────────────────────────

/// The scope in which `Self` has no meaning, with these types in it.
fn without_self(leaker: &mut Leaker, types: &[Type]) -> std::result::Result<(), Vec<ErrorItem>> {
    leaker
        .enter_generics(&Generics::default(), |view| {
            for ty in types {
                view.visit_type(ty);
            }
        })
        .map(|_| ())
}

/// The reasons and the causes of the errors, as text.
fn described(errors: &[ErrorItem]) -> Vec<(Reason, String)> {
    errors
        .iter()
        .map(|error| (error.reason, error.cause.to_token_stream().to_string()))
        .collect()
}

#[test]
fn a_mention_of_self_without_meaning_is_accepted_by_its_cause() {
    let ty: Type = parse_quote!(Rel<Self>);
    let errors = without_self(&mut accepting(&[]), std::slice::from_ref(&ty)).unwrap_err();
    assert_eq!(
        described(&errors),
        [(Reason::SelfWithoutMeaning, "Self".into())]
    );
    // The scope is accepted by the cause, and not by `Rel<Self>`, which is no cause.
    let self_ty = Cause::Type(parse_quote!(Self));
    assert!(without_self(
        &mut accepting(std::slice::from_ref(&self_ty)),
        std::slice::from_ref(&ty)
    )
    .is_ok());
    assert!(without_self(&mut accepting(&[Cause::Type(ty.clone())]), &[ty]).is_err());
    // `Self::Item` is another cause.
    let item: Type = parse_quote!(Wrap<Self::Item>);
    let self_item = Cause::Type(parse_quote!(Self::Item));
    assert!(without_self(&mut accepting(&[]), std::slice::from_ref(&item)).is_err());
    assert!(without_self(&mut accepting(&[self_ty]), std::slice::from_ref(&item)).is_err());
    assert!(without_self(&mut accepting(&[self_item]), &[item]).is_ok());
}

#[test]
fn a_mention_of_self_is_reported_only_if_no_other_error_is_left() {
    let types = [conflict(), parse_quote!(Rel<Self>)];
    let self_ty = Cause::Type(parse_quote!(Self));
    let left =
        |causes: &[Cause]| described(&without_self(&mut accepting(causes), &types).unwrap_err());
    // Today: only the conflict is reported.
    let conflict = vec![(Reason::RelativePath, "RelTr".to_string())];
    assert_eq!(left(&[]), conflict);
    // Accepting the conflict makes the mention appear.
    assert_eq!(
        left(&[rel_tr()]),
        [(Reason::SelfWithoutMeaning, "Self".to_string())]
    );
    // Accepting the mention alone leaves the conflict, and only it.
    assert_eq!(left(std::slice::from_ref(&self_ty)), conflict);
    // Accepting both accepts the scope.
    assert!(without_self(&mut accepting(&[rel_tr(), self_ty]), &types).is_ok());
}

/// The entries of a Referrer, as text, with whether each is required.
type Entries = Vec<(String, bool)>;

/// The entries of the Referrer.
fn listed(referrer: &Referrer) -> Entries {
    referrer
        .iter()
        .enumerate()
        .map(|(ix, ty)| (ty.to_token_stream().to_string(), referrer.must(ix)))
        .collect()
}

/// The entries of `types` visited where `Self` has no meaning, if the mentions of `Self` are
/// accepted, and the errors they are without that.
fn entries_without_self(types: &[Type]) -> (Entries, Vec<(Reason, String)>) {
    let errors = without_self(&mut accepting(&[]), types).unwrap_err();
    let causes: Vec<Cause> = errors.iter().map(|error| error.cause.clone()).collect();
    let mut leaker = accepting(&causes);
    without_self(&mut leaker, types).unwrap();
    (listed(&leaker.finish()), described(&errors))
}

/// An accepted `Self` without meaning is not left in an entry: its impl of the Repeater would read
/// it as its own implementor. The outermost type that would be an entry and names it is copied as
/// written, whole, with nothing in it leaked, and the error stays the one it was.
#[test]
fn a_type_with_a_self_without_meaning_is_copied_whole() {
    let types: [(Type, &str); 8] = [
        (parse_quote!(Rel<Self>), "Self"),
        (parse_quote!((Rel, Rel<Self>)), "Self"),
        (parse_quote!(::p::W<Rel, Self>), "Self"),
        (parse_quote!(Rel<[Self; 2]>), "Self"),
        (parse_quote!(Vec<Rel<Self>>), "Self"),
        (parse_quote!(Wrap<Self::Item>), "Self :: Item"),
        (parse_quote!(Wrap<m!(Self)>), "m ! (Self)"),
        // An optional entry: the Referrer may replace a path of another crate.
        (parse_quote!(::p::X<Self>), "Self"),
    ];
    for (ty, cause) in &types {
        let (entries, errors) = entries_without_self(std::slice::from_ref(ty));
        assert!(
            entries.is_empty(),
            "{}: {:?}",
            ty.to_token_stream(),
            entries
        );
        // The only error is that of `Self`: the correction reports nothing of its own.
        assert_eq!(errors, [(Reason::SelfWithoutMeaning, cause.to_string())]);
    }
}

#[test]
fn check_still_tells_the_causes_of_a_type_with_a_self_without_meaning() {
    // The correction is made when the scope ends, so `View::check()` sees the type as it was.
    let results = Leaker::new()
        .enter_generics(&Generics::default(), |view| {
            view.check(&parse_quote!(Rel<Self>))
        })
        .unwrap()
        .1;
    assert_eq!(results.must.len(), 1);
    assert!(results.is_own);
    assert_eq!(results.impossible, [Cause::Type(parse_quote!(Self))]);
}

#[test]
fn a_type_without_a_self_beside_one_that_has_it_stays_an_entry() {
    // In one signature, in one scope, and in a type that is not an entry (a local generic).
    let item: ItemFn = parse_quote!(
        fn f(a: Rel, b: Rel<Self>) {}
    );
    let mut leaker = accepting(&[Cause::Type(parse_quote!(Self))]);
    leaker
        .enter_generics(&Generics::default(), |view| {
            view.visit_item(&Item::Fn(item.clone()))
        })
        .unwrap();
    assert_eq!(listed(&leaker.finish()), [("Rel".to_string(), true)]);
    let (entries, _) = entries_without_self(&[parse_quote!(Rel), parse_quote!(Rel<Self>)]);
    assert_eq!(entries, [("Rel".to_string(), true)]);
    // A type that is no entry, as it names a generic of the method: its parts are, those that do not
    // name `Self`.
    let item: ItemFn = parse_quote!(
        fn g<U>(x: (Vec<U>, Rel, Self)) {}
    );
    let mut leaker = accepting(&[Cause::Type(parse_quote!(Self))]);
    leaker
        .enter_generics(&Generics::default(), |view| {
            view.visit_item(&Item::Fn(item.clone()))
        })
        .unwrap();
    let mut found = listed(&leaker.finish());
    found.sort();
    assert_eq!(
        found,
        [("Rel".to_string(), true), ("Vec < U >".to_string(), true)]
    );
}

#[test]
fn an_entry_whose_bounds_name_a_self_without_meaning_is_copied_whole() {
    let mut where_clause: Generics = parse_quote!(<T, U>);
    where_clause.where_clause = Some(parse_quote!(where T: AsRef<Self>));
    let self_ty = Cause::Type(parse_quote!(Self));
    let generics: [Generics; 2] = [parse_quote!(<T: AsRef<Self>, U>), where_clause];
    for generics in &generics {
        let visit = |leaker: &mut Leaker| {
            leaker
                .enter_generics(generics, |view| {
                    view.visit_type(&parse_quote!(Rel<T>));
                    view.visit_type(&parse_quote!(Other<U>));
                    // An optional entry has the same bound.
                    view.visit_type(&parse_quote!(::p::Opt<T>));
                })
                .map(|_| ())
        };
        let errors = visit(&mut accepting(&[])).unwrap_err();
        assert_eq!(
            described(&errors),
            [(Reason::SelfWithoutMeaning, "Self".into())]
        );
        let mut leaker = accepting(std::slice::from_ref(&self_ty));
        visit(&mut leaker).unwrap();
        // `Rel<T>` would carry the bound with `Self`; `Other<U>` has nothing to do with it.
        assert_eq!(
            listed(&leaker.finish()),
            [("Other < U >".to_string(), true)]
        );
    }
    // The scope that names `Self` in a bound is an outer one.
    let mut leaker = accepting(std::slice::from_ref(&self_ty));
    leaker
        .enter_generics(&parse_quote!(<T: AsRef<Self>>), |view| {
            view.enter_generics(&parse_quote!(<U>), |inner| {
                inner.visit_type(&parse_quote!(Rel<T>));
                inner.visit_type(&parse_quote!(Other<U>));
            })
            .unwrap();
        })
        .unwrap();
    assert_eq!(
        listed(&leaker.finish()),
        [("Other < U >".to_string(), true)]
    );
}

#[test]
fn a_default_with_a_self_without_meaning_is_copied_whole() {
    // Defaults are analyzed alone, in the scope that declares them (`finish()` and `end_scope()`).
    let self_ty = Cause::Type(parse_quote!(Self));
    let generics: Generics = parse_quote!(<T = Rel<Self>, U = Rel2>);
    let mut leaker = accepting(std::slice::from_ref(&self_ty));
    leaker.enter_generics(&generics, |_| ()).unwrap();
    assert_eq!(listed(&leaker.finish()), [("Rel2".to_string(), true)]);
    let mut leaker = accepting(std::slice::from_ref(&self_ty));
    leaker
        .enter_generics(&Generics::default(), |view| {
            view.enter_generics(&generics, |_| ()).unwrap();
        })
        .unwrap();
    assert_eq!(listed(&leaker.finish()), [("Rel2".to_string(), true)]);
}

#[test]
fn a_module_with_a_self_without_meaning_is_copied_whole() {
    let self_ty = Cause::Type(parse_quote!(Self));
    let mut leaker = accepting(std::slice::from_ref(&self_ty));
    leaker
        .enter_mod("m", Vec::new(), |view| {
            view.visit_type(&parse_quote!((Rel, Rel<Self>)));
            view.visit_type(&parse_quote!(Rel2));
        })
        .unwrap();
    assert_eq!(listed(&leaker.finish()), [("Rel2".to_string(), true)]);
}

#[test]
fn a_value_with_a_self_without_meaning_is_copied_whole() {
    // With `check_values()`: `Self` in a value is an error too, and a type there is an entry.
    let item: ItemFn = parse_quote!(
        fn f() {
            let a: Rel<Self> = Self::make();
            let b: Rel2 = ::core::default::Default::default();
        }
    );
    let mut config = LeakerConfig::new();
    config.check_values();
    let visit = |leaker: &mut Leaker| {
        leaker
            .enter_generics(&Generics::default(), |view| {
                view.visit_item(&Item::Fn(item.clone()))
            })
            .map(|_| ())
    };
    let errors = visit(&mut Leaker::from_config(config.clone())).unwrap_err();
    assert!(errors
        .iter()
        .all(|error| error.reason == Reason::SelfWithoutMeaning));
    let mut accepting = config;
    for error in &errors {
        accepting.allow(error.cause.clone());
    }
    let mut leaker = Leaker::from_config(accepting);
    visit(&mut leaker).unwrap();
    assert_eq!(listed(&leaker.finish()), [("Rel2".to_string(), true)]);
}

#[test]
fn the_errors_of_a_type_with_a_self_without_meaning_are_those_of_the_conflict() {
    // A conflict inside it is still the error, and the mention of `Self` is not told beside it.
    let types = [parse_quote!((Rel, Rel<Self>, ::p::W<'_, dyn RelTr>))];
    let left =
        |causes: &[Cause]| described(&without_self(&mut accepting(causes), &types).unwrap_err());
    let conflict = vec![(Reason::RelativePath, "RelTr".to_string())];
    assert_eq!(left(&[]), conflict);
    assert_eq!(
        left(&[rel_tr()]),
        [(Reason::SelfWithoutMeaning, "Self".to_string())]
    );
    assert!(without_self(
        &mut accepting(&[rel_tr(), Cause::Type(parse_quote!(Self))]),
        &types
    )
    .is_ok());
}

// ── the other kinds of cause ─────────────────────────────────────────────────────────────────

#[test]
fn unreadable_syntax_in_a_body_is_accepted_with_check_values() {
    let item: ItemTrait = parse_quote!(
        trait Tr {
            fn f(&self) {
                macro_rules! m {
                    () => {};
                }
                use ::inner::*;
            }
        }
    );
    let mut config = LeakerConfig::new();
    config.check_values();
    let errors = Leaker::from_config(config.clone())
        .intern_trait(&item, true)
        .unwrap_err();
    assert_eq!(errors.len(), 2);
    assert!(errors
        .iter()
        .all(|error| error.reason == Reason::Unreadable));
    // Written by hand: the tokens of the syntax, as `syn` prints them.
    let macro_rules: Item = parse_quote!(
        macro_rules! m {
            () => {};
        }
    );
    let glob: Item = parse_quote!(
        use ::inner::*;
    );
    let by_hand = [
        Cause::Unreadable(macro_rules.to_token_stream()),
        Cause::Unreadable(glob.to_token_stream()),
    ];
    for allowed in [
        errors.iter().map(|error| error.cause.clone()).collect(),
        by_hand.to_vec(),
    ] {
        let mut accepting = config.clone();
        for cause in &allowed {
            accepting.allow(cause.clone());
        }
        Leaker::from_config(accepting)
            .intern_trait(&item, true)
            .unwrap_or_else(|errors| panic!("{:?}", errors));
    }
    // One of them is not enough.
    let mut one = config;
    one.allow(by_hand[0].clone());
    let left = Leaker::from_config(one)
        .intern_trait(&item, true)
        .unwrap_err();
    assert_eq!(left.len(), 1);
    assert_eq!(left[0].cause, by_hand[1]);
}

#[test]
fn the_default_that_names_self_is_accepted_by_the_whole_default() {
    let item: ItemTrait = parse_quote!(
        trait Tr<T: ::core::default::Default = Wrap<Self>> {}
    );
    let errors = Leaker::new().intern_trait(&item, false).unwrap_err();
    assert_eq!(errors.len(), 1);
    assert_eq!(errors[0].reason, Reason::SelfInDefault);
    assert_eq!(errors[0].cause, Cause::Type(parse_quote!(Wrap<Self>)));
    assert!(accepting(&[Cause::Type(parse_quote!(Wrap<Self>))])
        .intern_trait(&item, false)
        .is_ok());
    // `Self` in it is a note, not the cause.
    let left = accepting(&[Cause::Type(parse_quote!(Self))])
        .intern_trait(&item, false)
        .unwrap_err();
    assert_eq!(left.len(), 1);
}

/// A type that leaves a lifetime of its method unconstrained (E0195) is copied whole when the
/// error is accepted, as a trait object is: the conflict is found after the type is chosen as an
/// entry, which the copy would replace.
#[test]
fn a_type_that_leaves_a_lifetime_unconstrained_is_copied_whole() {
    let item: ItemTrait = parse_quote!(
        trait Tr {
            fn f<'b>(&self, x: Wrap<'b>) -> &'b u8;
        }
    );
    let cause = Cause::Type(parse_quote!(Wrap<'b>));
    let errors = accepting(&[]).intern_trait(&item, true).unwrap_err();
    assert_eq!(errors.len(), 1);
    assert_eq!(errors[0].reason, Reason::UnconstrainedLifetime);
    assert_eq!(errors[0].cause, cause);
    let mut leaker = accepting(&[cause]);
    leaker.intern_trait(&item, true).unwrap();
    // Only `u8`: the reference is not leaked whole, and the entry for `Wrap<'b>` is gone.
    assert_eq!(listed(&leaker.finish()), [("u8".to_string(), true)]);
}

#[test]
fn every_type_that_leaves_the_lifetime_unconstrained_is_copied_whole() {
    let item: ItemTrait = parse_quote!(
        trait Tr {
            fn f<'b>(&self, x: (Wrap<'b>, Rel), y: [Other<'b>; 1]) -> &'b ::p::Out;
        }
    );
    let errors = accepting(&[]).intern_trait(&item, true).unwrap_err();
    assert_eq!(
        described(&errors),
        [
            (Reason::UnconstrainedLifetime, "Wrap < 'b >".to_string()),
            (Reason::UnconstrainedLifetime, "Other < 'b >".to_string())
        ]
    );
    // One of them accepted is not enough.
    let one = Cause::Type(parse_quote!(Wrap<'b>));
    let left = accepting(std::slice::from_ref(&one))
        .intern_trait(&item, true)
        .unwrap_err();
    assert_eq!(left.len(), 1);
    let mut leaker = accepting(&[one, Cause::Type(parse_quote!(Other<'b>))]);
    leaker.intern_trait(&item, true).unwrap();
    // `Rel` and the optional `::p::Out` do not hold the lifetime.
    assert_eq!(
        listed(&leaker.finish()),
        [
            ("Rel".to_string(), true),
            (":: p :: Out".to_string(), false)
        ]
    );
}

// ── what accepting does not make compile ─────────────────────────────────────────────────────

/// A configuration that knows the standard prelude (`Into`) and the primitive types.
fn std_config() -> LeakerConfig {
    let mut config = LeakerConfig::new();
    config.allow_std_prelude().allow_primitive();
    config
}

/// The documented limit of `LeakerConfig::allow()`: accepting `Self::A0` in a predicate copies only
/// that, as written. The bound `T: Into<R0>` of its context names a leaked type, and stays so:
/// `R0` is a required entry, and the copy fails with E0277 until `R0` is written with an absolute
/// path or allowed.
#[test]
fn accepting_self_in_a_predicate_leaves_the_bound_of_its_context_leaked() {
    let item: ItemTrait = parse_quote!(
        trait Tr<T>
        where
            T: Into<R0>,
        {
            type A0;
            fn m<X>(&self) -> X
            where
                X: Into<Self::A0>;
        }
    );
    let errors = Leaker::from_config(std_config())
        .intern_trait(&item, true)
        .unwrap_err();
    assert_eq!(
        described(&errors),
        [(Reason::UnprovableBound, "Self :: A0".to_string())]
    );
    let mut config = std_config();
    config.allow(errors[0].cause.clone());
    let mut leaker = Leaker::from_config(config);
    leaker.intern_trait(&item, true).unwrap();
    assert_eq!(listed(&leaker.finish()), [("R0".to_string(), true)]);
}

/// The same for an impl, where `Self` is the cause. A predicate on a type that is no path of `Self`
/// is fine: accepted, it is copied as written, so it no longer goes through the Repeater.
#[test]
fn accepting_a_predicate_in_an_impl_leaves_the_bound_of_its_context_leaked() {
    let on_self: ItemImpl = parse_quote!(
        impl<T: Into<R0>> S1<T> {
            fn m<Y>(a: u16)
            where
                Y: Fn(&Self) -> u8,
            {
            }
        }
    );
    let errors = Leaker::from_config(std_config())
        .intern_impl(&on_self, false)
        .unwrap_err();
    assert_eq!(
        described(&errors),
        [(Reason::UnprovableBound, "Self".to_string())]
    );
    let mut config = std_config();
    config.allow(errors[0].cause.clone());
    let mut leaker = Leaker::from_config(config);
    leaker.intern_impl(&on_self, false).unwrap();
    let entries = listed(&leaker.finish());
    assert!(entries.contains(&("R0".to_string(), true)));
    // A predicate on another type: its own cause is accepted, and `R0` is as before.
    let on_other: ItemImpl = parse_quote!(
        impl<T: Into<R0>> S1<T> {
            fn m<Y>(a: u16)
            where
                Y: Into<R1<T>>,
            {
            }
        }
    );
    let errors = Leaker::from_config(std_config())
        .intern_impl(&on_other, false)
        .unwrap_err();
    assert_eq!(
        described(&errors),
        [(Reason::UnprovableBound, "R1 < T >".to_string())]
    );
    let mut config = std_config();
    config.allow(errors[0].cause.clone());
    let mut leaker = Leaker::from_config(config);
    leaker.intern_impl(&on_other, false).unwrap();
    let entries = listed(&leaker.finish());
    assert!(entries.contains(&("R0".to_string(), true)));
    assert!(!entries.iter().any(|(ty, _)| ty.starts_with("R1")));
}

// ── the spacing of the tokens of a cause ─────────────────────────────────────────────────────

/// The hash of a cause.
fn hash_of(cause: &Cause) -> u64 {
    let mut hasher = DefaultHasher::new();
    cause.hash(&mut hasher);
    hasher.finish()
}

#[test]
fn a_cause_is_equal_whatever_the_spacing_of_its_punctuation() {
    // The syntax as the analysis reports it, and as the lexer reads the same text: `*` is joint
    // with `;` only in the second, so the printed tokens differ (`* ;` and `*;`).
    let item: ItemTrait = parse_quote!(
        trait Tr {
            fn f(&self) {
                use ::inner::*;
            }
        }
    );
    let mut config = LeakerConfig::new();
    config.check_values();
    let errors = Leaker::from_config(config)
        .intern_trait(&item, true)
        .unwrap_err();
    assert_eq!(errors.len(), 1);
    let lexed = Cause::Unreadable("use ::inner::*;".parse().unwrap());
    assert_ne!(
        errors[0].cause.to_token_stream().to_string(),
        lexed.to_token_stream().to_string()
    );
    assert_eq!(errors[0].cause, lexed);
    assert_eq!(hash_of(&errors[0].cause), hash_of(&lexed));
    // The one that was lexed accepts the conflict.
    let mut config = LeakerConfig::new();
    config.check_values().allow(lexed);
    Leaker::from_config(config)
        .intern_trait(&item, true)
        .unwrap();
}

#[test]
fn every_kind_of_cause_ignores_the_spacing_and_the_spans() {
    // `: :` and `::` are the same two characters of punctuation.
    let spaced = |text: &str| -> TokenStream { text.parse().unwrap() };
    let pairs = [
        (
            Cause::Unreadable(spaced(": : a")),
            Cause::Unreadable(spaced("::a")),
        ),
        (
            Cause::Type(parse_str("Rel < '_ , u8 >").unwrap()),
            Cause::Type(parse_quote!(Rel<'_, u8>)),
        ),
        (
            Cause::Trait(parse_str("::p::Tr<A = u8>").unwrap()),
            Cause::Trait(parse_quote!(::p::Tr<A = u8>)),
        ),
        (
            Cause::Expr(parse_str("{ a ::b() }").unwrap()),
            Cause::Expr(parse_quote!({ a::b() })),
        ),
        (
            Cause::Macro(parse_str(":: p :: m").unwrap()),
            Cause::Macro(parse_quote!(::p::m)),
        ),
        (
            Cause::Lifetime(parse_str("'a").unwrap()),
            Cause::Lifetime(parse_quote!('a)),
        ),
    ];
    for (a, b) in &pairs {
        assert_eq!(a, b);
        assert_eq!(hash_of(a), hash_of(b));
    }
    // Other tokens, other delimiters and another kind are not equal.
    let unequal = [
        (
            Cause::Unreadable(spaced("a : b")),
            Cause::Unreadable(spaced("a ; b")),
        ),
        (
            Cause::Unreadable(spaced("f ( a )")),
            Cause::Unreadable(spaced("f [ a ]")),
        ),
        (
            Cause::Unreadable(spaced("a b")),
            Cause::Unreadable(spaced("ab")),
        ),
        (
            Cause::Unreadable(spaced("1")),
            Cause::Unreadable(spaced("\"1\"")),
        ),
        (
            Cause::Type(parse_quote!(Rel)),
            Cause::Unreadable(spaced("Rel")),
        ),
    ];
    for (a, b) in &unequal {
        assert_ne!(a, b);
    }
}

// ── what the option does not touch ───────────────────────────────────────────────────────────

#[test]
fn check_reports_the_causes_whether_they_are_accepted_or_not() {
    let types: Vec<Type> = vec![
        conflict(),
        parse_quote!(Rel),
        parse_quote!(Rel<'_>),
        parse_quote!(::p::X<dyn RelTr>),
        parse_quote!(&'static u8),
        parse_quote!(Self),
    ];
    let check = |leaker: &mut Leaker| -> Vec<String> {
        leaker
            .enter_generics(&Generics::default(), |view| {
                types
                    .iter()
                    .map(|ty| format!("{:?}", view.check(ty)))
                    .collect()
            })
            .map(|(_, results)| results)
            .unwrap_or_else(|errors| panic!("{:?}", errors))
    };
    let without = check(&mut accepting(&[]));
    let with = check(&mut accepting(&[
        rel_tr(),
        Cause::Type(parse_quote!(Rel)),
        Cause::Lifetime(parse_quote!('_)),
        Cause::Type(parse_quote!(Self)),
    ]));
    assert_eq!(without, with);
    // The conflicts are in the results: the causes are told, not reported.
    assert!(without[0].contains("RelTr"));
}

#[test]
fn the_causes_make_a_config_equal_and_hash_alike() {
    let hash = |config: &LeakerConfig| {
        let mut hasher = DefaultHasher::new();
        config.hash(&mut hasher);
        hasher.finish()
    };
    let mut a = LeakerConfig::new();
    a.allow(rel_tr()).allow(Cause::Type(parse_quote!(Rel<'_>)));
    let mut b = LeakerConfig::new();
    // The same causes, built another way.
    b.allow(Cause::Trait(parse_str("RelTr").unwrap()))
        .allow(Cause::Type(parse_str("Rel < '_ >").unwrap()));
    assert_eq!(a, b);
    assert_eq!(hash(&a), hash(&b));
    // Not the same causes.
    let mut kind = LeakerConfig::new();
    kind.allow(Cause::Type(parse_quote!(RelTr)))
        .allow(Cause::Type(parse_quote!(Rel<'_>)));
    let mut tokens = LeakerConfig::new();
    tokens
        .allow(rel_tr())
        .allow(Cause::Type(parse_quote!(Rel<'a>)));
    assert_ne!(a, kind);
    assert_ne!(a, tokens);
    assert_ne!(a, LeakerConfig::new());
    // Causes that are equal hash alike.
    let hash_of = |cause: &Cause| {
        let mut hasher = DefaultHasher::new();
        cause.hash(&mut hasher);
        hasher.finish()
    };
    assert_eq!(
        hash_of(&rel_tr()),
        hash_of(&Cause::Trait(parse_str("RelTr").unwrap()))
    );
}

// ── the Referrer's `allowed` list ────────────────────────────────────────────────────────────
//
// The Referrer holds the entries only, and its visitor finds an entry by its tokens and its context
// wherever they are: also inside a part that the Leaker meant to copy as written. So the Referrer
// carries the causes that accepted a conflict and the parts that were copied as written
// (`Referrer::allowed()`), and the visitor leaves a part as written.

/// `f` is a conflict: the lifetime of `Elided` is elided and `RelTr` is relative, so the type that
/// owns the lifetime is copied as written. `g` holds the same `dyn RelTr`, which is no conflict
/// there and is leaked.
fn elided_and_bare() -> ItemTrait {
    parse_quote!(
        trait Fg {
            fn f(&self, x: ::p::Elided<'_, dyn RelTr>);
            fn g(&self, x: ::p::Bare<dyn RelTr>);
        }
    )
}

/// The Referrer of `item` and its context, with the conflicts of `causes` accepted.
fn leak_trait(config: LeakerConfig, item: &ItemTrait) -> (Referrer, ContextId) {
    let mut leaker = Leaker::from_config(config);
    let context = leaker
        .intern_trait(item, true)
        .unwrap_or_else(|errors| panic!("{:?}", errors));
    (leaker.finish(), context)
}

/// A configuration that accepts the cause of every error of `item`, and the errors it has.
fn accepting_errors_of(config: &LeakerConfig, item: &ItemTrait) -> (LeakerConfig, Vec<ErrorItem>) {
    let errors = Leaker::from_config(config.clone())
        .intern_trait(item, true)
        .unwrap_err();
    let mut config = config.clone();
    for error in &errors {
        config.allow(error.cause.clone());
    }
    (config, errors)
}

/// The copy of `item` in which each entry is replaced by `Entry<ix>`.
fn rewritten(referrer: &Referrer, context: ContextId, item: &ItemTrait) -> ItemTrait {
    let mut copy = item.clone();
    referrer
        .clone()
        .visitor(context, |_, ix| parse_quote!(Entry<#ix>))
        .visit_item_trait_mut(&mut copy);
    copy
}

/// A part of the list, as text: the part, its context, whether it is in a predicate, and its anchor
/// with the context of the anchor.
type Shown = (String, ContextId, bool, Option<(String, ContextId)>);

fn part(item: &Allowed) -> Shown {
    match item {
        Allowed::Part {
            part,
            context,
            in_predicate,
            within,
        } => (
            part.to_token_stream().to_string(),
            *context,
            *in_predicate,
            within
                .as_ref()
                .map(|(anchor, context)| (anchor.to_token_stream().to_string(), *context)),
        ),
        other => panic!("not a part: {:?}", other),
    }
}

#[test]
fn the_referrer_tells_the_cause_that_accepted_the_conflict_and_the_part_that_is_copied() {
    let (referrer, context) = leak_trait(
        {
            let mut config = LeakerConfig::new();
            config.allow(rel_tr());
            config
        },
        &elided_and_bare(),
    );
    // The part is the type that conflicts (the trait object, not the trait that is the cause),
    // inside the path that owns the elided lifetime.
    assert_eq!(
        referrer.allowed(),
        [
            Allowed::Cause(rel_tr()),
            Allowed::Part {
                part: Cause::Type(parse_quote!(dyn RelTr)),
                context,
                in_predicate: false,
                within: Some((parse_quote!(::p::Elided<'_, dyn RelTr>), context)),
            }
        ]
    );
    // `dyn RelTr` of `g` is an entry, and the optional entry of the path around it.
    assert_eq!(
        listed(&referrer),
        [
            ("dyn RelTr".to_string(), true),
            (":: p :: Bare < dyn RelTr >".to_string(), false)
        ]
    );
    // The list is empty where nothing was accepted or corrected.
    let mut leaker = Leaker::new();
    leaker
        .intern_trait(
            &parse_quote!(
                trait Tr {
                    fn f(&self) -> Rel;
                }
            ),
            true,
        )
        .unwrap();
    assert!(leaker.finish().allowed().is_empty());
    assert!(Leaker::from_config(LeakerConfig::new())
        .finish()
        .allowed()
        .is_empty());
}

#[test]
fn the_visitor_leaves_a_part_as_written_and_replaces_the_entry_with_the_same_tokens() {
    let item = elided_and_bare();
    let mut config = LeakerConfig::new();
    config.allow(rel_tr());
    let (referrer, context) = leak_trait(config, &item);
    // `f` as written, `g` through the entry `dyn RelTr`.
    let expected: ItemTrait = parse_quote!(
        trait Fg {
            fn f(&self, x: ::p::Elided<'_, dyn RelTr>);
            fn g(&self, x: ::p::Bare<Entry<0usize>>);
        }
    );
    assert_eq!(rewritten(&referrer, context, &item), expected);
    // A strict Referrer replaces the optional entry as a whole.
    let strict = referrer.allow_ambiguous_crate_name(false);
    let expected: ItemTrait = parse_quote!(
        trait Fg {
            fn f(&self, x: ::p::Elided<'_, dyn RelTr>);
            fn g(&self, x: Entry<1usize>);
        }
    );
    assert_eq!(rewritten(&strict, context, &item), expected);
}

#[test]
fn an_item_beside_the_conflict_in_the_same_context_is_still_replaced() {
    // Both are interned without generics, so they share the context 0: the `dyn RelTr` of `S` is
    // the entry that is found inside the part of `Fg`.
    let tr = elided_and_bare();
    let st: ItemStruct = parse_quote!(
        struct S {
            y: ::p::Bare<dyn RelTr>,
        }
    );
    let mut config = LeakerConfig::new();
    config.allow(rel_tr());
    let mut leaker = Leaker::from_config(config);
    let in_struct = leaker.intern_struct(&st, true).unwrap();
    let in_trait = leaker.intern_trait(&tr, true).unwrap();
    assert_eq!(in_struct, in_trait);
    let referrer = leaker.finish();
    assert_eq!(referrer.allowed().len(), 2);
    let mut copy = st.clone();
    referrer
        .clone()
        .visitor(in_struct, |_, ix| parse_quote!(Entry<#ix>))
        .visit_item_struct_mut(&mut copy);
    let expected: ItemStruct = parse_quote!(
        struct S {
            y: ::p::Bare<Entry<0usize>>,
        }
    );
    assert_eq!(copy, expected);
    // The trait is as written in `f` beside it.
    let expected: ItemTrait = parse_quote!(
        trait Fg {
            fn f(&self, x: ::p::Elided<'_, dyn RelTr>);
            fn g(&self, x: ::p::Bare<Entry<0usize>>);
        }
    );
    assert_eq!(rewritten(&referrer, in_trait, &tr), expected);
}

/// `a1` is an entry, and so is the `R1<Y>` in the bound of `ai`, but the copy cannot write it as a
/// path through the Repeater there: rustc proves the bound `Y: Fn(R0)` of its generics in an
/// environment where `R0` is not normalized yet (`UnprovableBound`). The same tokens in the same
/// context, so the part differs from the entry by where it is.
fn r1_in_a_bound() -> ItemTrait {
    parse_quote!(
        trait Tr {
            fn m<Y: Clone>(&self, a1: R1<Y>, ai: impl Into<R1<Y>>)
            where
                Y: Fn(R0) -> u8;
        }
    )
}

#[test]
fn a_part_in_a_bound_is_told_from_the_entry_with_the_same_tokens() {
    let item = r1_in_a_bound();
    let (config, errors) = accepting_errors_of(&std_config(), &item);
    assert_eq!(
        described(&errors),
        [(Reason::UnprovableBound, "R1 < Y >".to_string())]
    );
    let (referrer, context) = leak_trait(config, &item);
    // `R0` and `R1<Y>` (in the context of the method, which holds `Y`).
    assert_eq!(
        listed(&referrer),
        [("R0".to_string(), true), ("R1 < Y >".to_string(), true)]
    );
    assert_eq!(referrer.allowed().len(), 2);
    assert_eq!(
        referrer.allowed()[0],
        Allowed::Cause(Cause::Type(parse_quote!(R1<Y>)))
    );
    // The type that conflicts is the `R1<Y>` in the bound of the argument, inside the `impl
    // Into<R1<Y>>` that owns the cause (`impl Trait` cannot be leaked), which has the same
    // context.
    assert_eq!(
        part(&referrer.allowed()[1]),
        (
            "R1 < Y >".to_string(),
            referrer.context(1),
            true,
            Some(("impl Into < R1 < Y > >".to_string(), referrer.context(1)))
        )
    );
    let expected: ItemTrait = parse_quote!(
        trait Tr {
            fn m<Y: Clone>(&self, a1: Entry<1usize>, ai: impl Into<R1<Y>>)
            where
                Y: Fn(Entry<0usize>) -> u8;
        }
    );
    assert_eq!(rewritten(&referrer, context, &item), expected);
}

#[test]
fn a_part_in_a_predicate_is_not_the_entry_outside_it() {
    // `R1<Y>` in the where predicate is a conflict (`UnprovableBound`), whereas the argument is not.
    let item: ItemTrait = parse_quote!(
        trait Tr {
            fn m<Y: Clone>(&self, a1: R1<Y>)
            where
                Y: Fn(R0) -> u8,
                Vec<R1<Y>>: Clone;
        }
    );
    let (config, errors) = accepting_errors_of(&std_config(), &item);
    assert_eq!(
        described(&errors),
        [(Reason::UnprovableBound, "R1 < Y >".to_string())]
    );
    let (referrer, context) = leak_trait(config, &item);
    // No type around it owns a cause of the conflict: no anchor.
    assert_eq!(
        part(&referrer.allowed()[1]),
        ("R1 < Y >".to_string(), referrer.context(1), true, None)
    );
    let expected: ItemTrait = parse_quote!(
        trait Tr {
            fn m<Y: Clone>(&self, a1: Entry<1usize>)
            where
                Y: Fn(Entry<0usize>) -> u8,
                Vec<R1<Y>>: Clone;
        }
    );
    assert_eq!(rewritten(&referrer, context, &item), expected);
}

#[test]
fn the_causes_are_the_ones_that_accepted_a_conflict_in_order_each_once() {
    let rel = Cause::Type(parse_quote!(Rel<'_>));
    let mut leaker = accepting(&[Cause::Type(parse_quote!(NeverFound)), rel.clone(), rel_tr()]);
    leaker
        .enter_generics(&Generics::default(), |view| {
            view.visit_type(&parse_quote!(Rel<'_>));
            view.visit_type(&conflict());
            view.visit_type(&conflict());
            view.visit_type(&parse_quote!(Rel<'_>));
        })
        .unwrap();
    let referrer = leaker.finish();
    // In order of the conflicts, not of the configuration; `NeverFound` accepted nothing.
    let allowed = referrer.allowed();
    assert_eq!(allowed.len(), 4);
    assert_eq!(allowed[0], Allowed::Cause(rel));
    assert_eq!(allowed[1], Allowed::Cause(rel_tr()));
    // The parts are distinct: the same tokens in the same place are one. `Rel<'_>` holds its own
    // lifetime and no type owns it around, whereas the trait object is in a path that does.
    let parts: Vec<_> = allowed[2..].iter().map(part).collect();
    let context = referrer.contexts().next().unwrap();
    assert_eq!(
        parts,
        [
            ("Rel < '_ >".to_string(), context, false, None),
            (
                "dyn RelTr".to_string(),
                context,
                false,
                Some((":: p :: W < '_ , dyn RelTr >".to_string(), context))
            )
        ]
    );
    // The causes of two scopes are one list, in order of first acceptance.
    let mut leaker = accepting(&[rel_tr(), Cause::Type(parse_quote!(Rel<'_>))]);
    for ty in [parse_quote!(Rel<'_>), conflict()] {
        leaker
            .enter_generics(&Generics::default(), |view| view.visit_type(&ty))
            .unwrap();
    }
    leaker
        .enter_generics(&Generics::default(), |view| view.visit_type(&conflict()))
        .unwrap();
    let referrer = leaker.finish();
    assert_eq!(referrer.allowed().len(), 4);
    assert_eq!(
        referrer.allowed()[..2],
        [
            Allowed::Cause(Cause::Type(parse_quote!(Rel<'_>))),
            Allowed::Cause(rel_tr())
        ]
    );
}

#[test]
fn the_list_is_rolled_back_with_the_scope() {
    let other = Cause::Trait(parse_quote!(OtherTr));
    let outer: Type = parse_quote!(::p::X<'_, dyn RelTr>);
    // An inner scope that fails: what it accepted is gone, the cause too.
    let mut leaker = accepting(&[rel_tr(), other.clone()]);
    leaker
        .enter_generics(&Generics::default(), |view| {
            view.visit_type(&outer);
            let failed = view.enter_generics(&parse_quote!(<T>), |inner| {
                inner.visit_type(&conflict());
                inner.visit_type(&parse_quote!(::p::Y<'_, dyn OtherTr>));
                inner.visit_type(&parse_quote!(::p::Z<'_, dyn Unknown>));
            });
            assert!(failed.is_err());
        })
        .unwrap();
    let referrer = leaker.finish();
    assert_eq!(
        referrer.allowed()[0],
        Allowed::Cause(rel_tr()),
        "{:?}",
        referrer.allowed()
    );
    assert_eq!(referrer.allowed().len(), 2, "{:?}", referrer.allowed());
    let (ty, _, _, anchor) = part(&referrer.allowed()[1]);
    assert_eq!(ty, "dyn RelTr");
    assert_eq!(anchor.unwrap().0, ":: p :: X < '_ , dyn RelTr >");
    // An inner scope that succeeds inside an outer one that fails: nothing stays.
    let mut leaker = accepting(&[rel_tr()]);
    let errors = leaker
        .enter_generics(&Generics::default(), |view| {
            view.enter_generics(&parse_quote!(<T>), |inner| {
                inner.visit_type(&conflict());
            })
            .unwrap();
            view.visit_type(&parse_quote!(::p::W<'_, dyn Unknown>));
        })
        .unwrap_err();
    assert_eq!(errors.len(), 1);
    assert!(leaker.finish().allowed().is_empty());
    // And one that fails after the call: the Leaker keeps what its earlier scopes accepted.
    let mut leaker = accepting(&[rel_tr()]);
    leaker
        .enter_generics(&Generics::default(), |view| view.visit_type(&conflict()))
        .unwrap();
    assert!(leaker
        .enter_generics(&Generics::default(), |view| view
            .visit_type(&parse_quote!(::p::W<'_, dyn Unknown>)))
        .is_err());
    assert_eq!(leaker.finish().allowed().len(), 2);
}

#[test]
fn a_silent_correction_is_a_part_and_the_error_it_comes_with_is_the_only_cause() {
    // (1) `Self` has no meaning: the type that names it is copied whole instead of being an entry.
    let self_ty = Cause::Type(parse_quote!(Self));
    let mut leaker = accepting(std::slice::from_ref(&self_ty));
    let (context, ()) = leaker
        .enter_generics(&Generics::default(), |view| {
            view.visit_type(&parse_quote!(Rel));
            view.visit_type(&parse_quote!(Wrap<Self>));
        })
        .unwrap();
    let referrer = leaker.finish();
    assert_eq!(referrer.allowed().len(), 2);
    assert_eq!(referrer.allowed()[0], Allowed::Cause(self_ty));
    assert_eq!(
        referrer.allowed()[1],
        Allowed::Part {
            part: Cause::Type(parse_quote!(Wrap<Self>)),
            context,
            in_predicate: false,
            within: None,
        }
    );
    let mut copy: Vec<Type> = vec![parse_quote!(Rel), parse_quote!(Wrap<Self>)];
    let mut visitor = referrer
        .visitor(context, |_, ix| parse_quote!(Entry<#ix>))
        .with_self(parse_quote!(Host));
    for ty in &mut copy {
        visitor.visit_type_mut(ty);
    }
    // `Self` in the part is still replaced when the caller asks.
    let expected: Vec<Type> = vec![parse_quote!(Entry<0usize>), parse_quote!(Wrap<Host>)];
    assert_eq!(copy, expected);

    // (2) A method lifetime that the copy leaves unconstrained.
    let item: ItemTrait = parse_quote!(
        trait Tr {
            fn f<'b>(&self, x: Held<'b>) -> &'b u8;
        }
    );
    let (config, errors) = accepting_errors_of(&std_config(), &item);
    assert_eq!(errors[0].reason, Reason::UnconstrainedLifetime);
    let (referrer, context) = leak_trait(config, &item);
    assert_eq!(
        referrer.allowed()[0],
        Allowed::Cause(Cause::Type(parse_quote!(Held<'b>)))
    );
    assert_eq!(referrer.allowed().len(), 2);
    assert_eq!(part(&referrer.allowed()[1]).0, "Held < 'b >");
    // `u8` is allowed and not an entry, and `Held<'b>` is not, so nothing is replaced.
    assert_eq!(rewritten(&referrer, context, &item), item);
}

#[test]
fn an_entry_whose_bounds_name_a_self_without_meaning_is_a_part() {
    let mut leaker = accepting(&[Cause::Type(parse_quote!(Self))]);
    let (context, ()) = leaker
        .enter_generics(&parse_quote!(<T: AsRef<Self>, U>), |view| {
            view.visit_type(&parse_quote!(Rel<T>));
            view.visit_type(&parse_quote!(Other<U>));
        })
        .unwrap();
    let referrer = leaker.finish();
    assert_eq!(referrer.allowed().len(), 2);
    let (ty, part_context, in_predicate, _) = part(&referrer.allowed()[1]);
    assert_eq!((ty.as_str(), in_predicate), ("Rel < T >", false));
    // The context of the part holds the bound that names `Self`.
    assert!(referrer
        .context_generics(part_context)
        .to_token_stream()
        .to_string()
        .contains("AsRef"));
    let mut copy: Vec<Type> = vec![parse_quote!(Rel<T>), parse_quote!(Other<U>)];
    let mut visitor = referrer.visitor(context, |_, ix| parse_quote!(Entry<#ix>));
    for ty in &mut copy {
        visitor.visit_type_mut(ty);
    }
    let expected: Vec<Type> = vec![parse_quote!(Rel<T>), parse_quote!(Entry<0usize>)];
    assert_eq!(copy, expected);
}

#[test]
fn the_part_is_the_type_that_conflicts_and_the_types_beside_it_stay_entries() {
    // Only `dyn RelTr` was reported and accepted: it is copied as written, in the path that owns
    // the elided lifetime. `Rel` beside it is an entry, and replaced.
    let list = |ty: Type| -> (Entries, Vec<Shown>, Type) {
        let mut config = std_config();
        config.allow(rel_tr());
        let mut leaker = Leaker::from_config(config);
        let (context, ()) = leaker
            .enter_generics(&Generics::default(), |view| view.visit_type(&ty))
            .unwrap();
        let referrer = leaker.finish();
        let parts = referrer
            .allowed()
            .iter()
            .filter(|item| matches!(item, Allowed::Part { .. }))
            .map(part)
            .collect();
        let entries = listed(&referrer);
        let mut copy = ty;
        referrer
            .visitor(context, |_, ix| parse_quote!(Entry<#ix>))
            .visit_type_mut(&mut copy);
        (entries, parts, copy)
    };
    let shown = |part: &str, anchor: &str| -> Vec<(String, bool, Option<String>)> {
        vec![(part.to_string(), false, Some(anchor.to_string()))]
    };
    let simple = |parts: Vec<Shown>| -> Vec<(String, bool, Option<String>)> {
        parts
            .into_iter()
            .map(|(ty, _, predicate, anchor)| (ty, predicate, anchor.map(|(anchor, _)| anchor)))
            .collect()
    };
    let (entries, parts, copy) = list(parse_quote!(::p::W<'_, dyn RelTr, Rel>));
    assert_eq!(entries, [("Rel".to_string(), true)]);
    assert_eq!(
        simple(parts),
        shown("dyn RelTr", ":: p :: W < '_ , dyn RelTr , Rel >")
    );
    let expected: Type = parse_quote!(::p::W<'_, dyn RelTr, Entry<0usize>>);
    assert_eq!(copy, expected);
    // A type around the owner is no owner: the anchor is the path that holds the lifetime.
    let (entries, parts, copy) = list(parse_quote!((Rel, ::p::W<'_, dyn RelTr>)));
    assert_eq!(entries, [("Rel".to_string(), true)]);
    assert_eq!(
        simple(parts),
        shown("dyn RelTr", ":: p :: W < '_ , dyn RelTr >")
    );
    let expected: Type = parse_quote!((Entry<0usize>, ::p::W<'_, dyn RelTr>));
    assert_eq!(copy, expected);
    let (entries, parts, copy) = list(parse_quote!(Vec<::p::W<'_, dyn RelTr>>));
    assert!(entries.is_empty());
    assert_eq!(
        simple(parts),
        shown("dyn RelTr", ":: p :: W < '_ , dyn RelTr >")
    );
    let expected: Type = parse_quote!(Vec<::p::W<'_, dyn RelTr>>);
    assert_eq!(copy, expected);
    // An elided reference owns its cause too.
    let (entries, parts, copy) = list(parse_quote!((Rel, &dyn RelTr)));
    assert_eq!(entries, [("Rel".to_string(), true)]);
    assert_eq!(simple(parts), shown("dyn RelTr", "& dyn RelTr"));
    let expected: Type = parse_quote!((Entry<0usize>, &dyn RelTr));
    assert_eq!(copy, expected);
    // The innermost owner is the anchor.
    let (_, parts, copy) = list(parse_quote!(::p::V<'_, ::p::W<'_, dyn RelTr, Rel>, Rel2>));
    assert_eq!(
        simple(parts),
        shown("dyn RelTr", ":: p :: W < '_ , dyn RelTr , Rel >")
    );
    let expected: Type =
        parse_quote!(::p::V<'_, ::p::W<'_, dyn RelTr, Entry<0usize>>, Entry<1usize>>);
    assert_eq!(copy, expected);
}

#[test]
fn a_bound_whose_trait_conflicts_is_a_part_with_the_types_in_it() {
    let item: ItemTrait = parse_quote!(
        trait Tr<T: RelTrait<R0>, F: for<'a> Fn(&'a R0) -> &'a R0>
        where
            T: Other<R0>,
        {
            fn f(&self, x: R0) -> R0;
        }
    );
    let (config, errors) = accepting_errors_of(&std_config(), &item);
    assert_eq!(
        described(&errors),
        [
            (Reason::RelativePath, "RelTrait < R0 >".to_string()),
            (Reason::RelativePath, "Other < R0 >".to_string()),
        ]
    );
    let (referrer, context) = leak_trait(config, &item);
    // Only `R0` of `f` is an entry. The bounds are copied as written, with the `R0` in them. They
    // name no generic, so their context is that of `R0`.
    assert_eq!(listed(&referrer), [("R0".to_string(), true)]);
    let parts: Vec<_> = referrer.allowed()[2..].iter().map(part).collect();
    let none = referrer.context(0);
    assert_eq!(
        parts,
        [
            ("RelTrait < R0 >".to_string(), none, true, None),
            ("Other < R0 >".to_string(), none, true, None)
        ]
    );
    let expected: ItemTrait = parse_quote!(
        trait Tr<T: RelTrait<R0>, F: for<'a> Fn(&'a Entry<0usize>) -> &'a Entry<0usize>>
        where
            T: Other<R0>,
        {
            fn f(&self, x: Entry<0usize>) -> Entry<0usize>;
        }
    );
    assert_eq!(rewritten(&referrer, context, &item), expected);
}

#[test]
fn a_bound_with_a_binder_is_a_part_as_written() {
    // No node of the syntax holds a bound with a binder: its cause is its tokens, and so is the
    // part.
    let item: ItemTrait = parse_quote!(
        trait Tr<T: for<'a> RelTrait<'a, R0>> {
            fn f(&self, x: R0) -> R0;
        }
    );
    let (config, errors) = accepting_errors_of(&std_config(), &item);
    assert_eq!(
        errors[0].cause.to_token_stream().to_string(),
        "for < 'a > RelTrait < 'a , R0 >"
    );
    let (referrer, context) = leak_trait(config, &item);
    assert!(matches!(
        &referrer.allowed()[1],
        Allowed::Part {
            part: Cause::Unreadable(_),
            in_predicate: true,
            ..
        }
    ));
    let expected: ItemTrait = parse_quote!(
        trait Tr<T: for<'a> RelTrait<'a, R0>> {
            fn f(&self, x: Entry<0usize>) -> Entry<0usize>;
        }
    );
    assert_eq!(rewritten(&referrer, context, &item), expected);
}

#[test]
fn a_part_of_a_module_is_left_as_written_in_that_module_only() {
    let item: ItemMod = parse_quote!(
        mod m {
            pub trait Fg {
                fn f(&self, x: ::p::Elided<'_, dyn RelTr>);
            }
            pub struct S {
                pub y: ::p::Bare<dyn RelTr>,
            }
            mod inner {
                pub struct T {
                    pub z: ::p::Bare<dyn RelTr>,
                    pub w: ::p::Elided<'_, dyn RelTr>,
                }
            }
        }
    );
    let mut config = LeakerConfig::new();
    config.allow(rel_tr());
    let mut leaker = Leaker::from_config(config);
    let context = leaker.intern_mod(&item).unwrap();
    let referrer = leaker.finish();
    // `RelTr` is relative, so each module has its own entries and its own parts.
    let parts: Vec<_> = referrer.allowed()[1..].iter().map(part).collect();
    assert_eq!(parts.len(), 2);
    assert_ne!(parts[0].1, parts[1].1);
    for (_, part_context, _, _) in &parts {
        assert!(referrer.context_module(*part_context).is_some());
    }
    let mut items = item.content.unwrap().1;
    let mut visitor = referrer
        .clone()
        .visitor(context, |_, ix| parse_quote!(Entry<#ix>));
    for item in &mut items {
        visitor.visit_item_mut(item);
    }
    let expected: ItemMod = parse_quote!(
        mod m {
            pub trait Fg {
                fn f(&self, x: ::p::Elided<'_, dyn RelTr>);
            }
            pub struct S {
                pub y: ::p::Bare<Entry<0usize>>,
            }
            mod inner {
                pub struct T {
                    pub z: ::p::Bare<Entry<1usize>>,
                    pub w: ::p::Elided<'_, dyn RelTr>,
                }
            }
        }
    );
    assert_eq!(items, expected.content.unwrap().1);
    // The parts are written with their modules, and read back.
    let read: Referrer = parse2(referrer.to_token_stream()).unwrap();
    assert_eq!(read, referrer);
}

#[test]
fn a_path_of_a_value_that_conflicts_is_a_part_with_the_types_in_it() {
    let item: ItemTrait = parse_quote!(
        trait Tr {
            fn f(&self, x: R0) -> R0 {
                inner::make::<R0>(x)
            }
        }
    );
    let mut config = std_config();
    config.check_values();
    let (config, errors) = accepting_errors_of(&config, &item);
    assert_eq!(
        described(&errors),
        [(Reason::RelativePath, "inner :: make :: < R0 >".to_string())]
    );
    let (referrer, context) = leak_trait(config, &item);
    assert_eq!(listed(&referrer), [("R0".to_string(), true)]);
    assert_eq!(referrer.allowed().len(), 2);
    let expected: ItemTrait = parse_quote!(
        trait Tr {
            fn f(&self, x: Entry<0usize>) -> Entry<0usize> {
                inner::make::<R0>(x)
            }
        }
    );
    assert_eq!(rewritten(&referrer, context, &item), expected);
}

#[test]
fn self_is_replaced_in_a_part_when_the_caller_asks() {
    let item: ItemTrait = parse_quote!(
        trait Tr {
            fn f(&self, x: ::p::Elided<'_, dyn RelTr<Self>>, y: Rel<Self>);
        }
    );
    let (config, _) = accepting_errors_of(&LeakerConfig::new(), &item);
    let (referrer, context) = leak_trait(config, &item);
    let mut copy = item.clone();
    referrer
        .visitor(context, |_, ix| parse_quote!(Entry<#ix>))
        .with_self(parse_quote!(__S))
        .visit_item_trait_mut(&mut copy);
    // `Self` is replaced inside the part, which is otherwise as written, and the entry is replaced.
    let expected: ItemTrait = parse_quote!(
        trait Tr {
            fn f(&self, x: ::p::Elided<'_, dyn RelTr<__S>>, y: Entry<0usize>);
        }
    );
    assert_eq!(copy, expected);
}

#[test]
fn expand_leaves_a_part_as_written() {
    let item = elided_and_bare();
    let mut config = LeakerConfig::new();
    config.allow(rel_tr());
    let (referrer, context) = leak_trait(config, &item);
    let replace = |_: Type, ix: usize| -> Type { parse_quote!(Entry<#ix>) };
    let part: Type = parse_quote!(::p::Elided<'_, dyn RelTr>);
    assert_eq!(referrer.expand(context, part.clone(), replace), part);
    let other: Type = parse_quote!(::p::Bare<dyn RelTr>);
    let expected: Type = parse_quote!(::p::Bare<Entry<0usize>>);
    assert_eq!(referrer.expand(context, other, replace), expected);
}

#[test]
fn the_allowed_list_is_serialized_with_the_referrer() {
    use syn::parse::Parser;
    let item: ItemTrait = parse_quote!(
        trait Tr<T: RelTrait<R0>> {
            fn f(&self, x: ::p::Elided<'_, dyn RelTr>, y: R0);
            fn m<Y: Clone>(&self, ai: impl Into<R1<Y>>, a1: R1<Y>)
            where
                Y: Fn(R0) -> u8;
        }
    );
    let (config, _) = accepting_errors_of(&std_config(), &item);
    let (referrer, context) = leak_trait(config, &item);
    // Both kinds of item, and the three kinds of cause of a part.
    assert!(referrer
        .allowed()
        .iter()
        .any(|item| matches!(item, Allowed::Cause(_))));
    assert!(referrer.allowed().iter().any(|item| matches!(
        item,
        Allowed::Part {
            in_predicate: true,
            ..
        }
    )));
    assert!(referrer.allowed().iter().any(|item| matches!(
        item,
        Allowed::Part {
            in_predicate: false,
            ..
        }
    )));
    let tokens = quote!(#referrer, #context);
    let text = tokens.to_string();
    assert!(text.contains("cause trait"), "{}", text);
    assert!(text.contains("part type"), "{}", text);
    assert!(text.contains("predicate"), "{}", text);
    assert!(text.contains(" in ("), "{}", text);
    let read = (|input: parse::ParseStream| -> Result<(Referrer, ContextId)> {
        let referrer = input.parse()?;
        input.parse::<Token![,]>()?;
        Ok((referrer, input.parse()?))
    })
    .parse2(tokens.clone())
    .unwrap();
    assert_eq!(read.0, referrer);
    assert_eq!(read.1, context);
    assert_eq!(read.0.allowed(), referrer.allowed());
    // Writing it again gives the same tokens.
    assert_eq!(quote!(#{&read.0}, #context).to_string(), text);
    // The copy of the Referrer that was read leaves the same parts as written.
    assert_eq!(
        rewritten(&read.0, context, &item),
        rewritten(&referrer, context, &item)
    );
    // Other tokens may follow it.
    let followed = quote!(#referrer extra tokens);
    let rest = (|input: parse::ParseStream| -> Result<(Referrer, TokenStream)> {
        Ok((input.parse()?, input.parse()?))
    })
    .parse2(followed)
    .unwrap();
    assert_eq!(rest.0, referrer);
    assert_eq!(rest.1.to_string(), "extra tokens");
}

#[test]
fn a_referrer_without_a_list_writes_what_it_wrote_before() {
    let mut leaker = Leaker::new();
    leaker
        .intern_struct(
            &parse_quote!(
                struct S {
                    a: Rel,
                }
            ),
            true,
        )
        .unwrap();
    let referrer = leaker.finish();
    assert!(referrer.allowed().is_empty());
    let tag = format!("\"type-leak-format-{}\"", Referrer::FORMAT_VERSION);
    assert_eq!(
        referrer.to_token_stream().to_string(),
        format!("{} (Rel)", tag)
    );
    // Referrers that differ by the list are not equal.
    let mut config = LeakerConfig::new();
    config.allow(rel_tr());
    let (with, _) = leak_trait(config, &elided_and_bare());
    let (without, _) = {
        let mut config = LeakerConfig::new();
        config.allow(rel_tr());
        let mut leaker = Leaker::from_config(config);
        let context = leaker
            .intern_trait(
                &parse_quote!(
                    trait Fg {
                        fn g(&self, x: ::p::Bare<dyn RelTr>);
                    }
                ),
                true,
            )
            .unwrap();
        (leaker.finish(), context)
    };
    assert!(without.allowed().is_empty());
    assert_ne!(with, without);
}

#[test]
fn the_allowed_list_is_checked_when_it_is_read() {
    let tag = format!("\"type-leak-format-{}\"", Referrer::FORMAT_VERSION);
    let read = |text: &str| -> std::result::Result<Referrer, String> {
        parse_str::<Referrer>(&format!("{} {}", tag, text)).map_err(|error| error.to_string())
    };
    // Good: a cause, and a part in the context 0 (the only one).
    let good = read("[cause trait (RelTr), part type (Elided<'_, dyn RelTr>) 0] (Rel)").unwrap();
    assert_eq!(good.allowed().len(), 2);
    assert!(matches!(
        good.allowed()[1],
        Allowed::Part {
            in_predicate: false,
            ..
        }
    ));
    let predicate = read("[part type (R1<Y>) 0 predicate] (Rel)").unwrap();
    assert!(matches!(
        predicate.allowed()[0],
        Allowed::Part {
            in_predicate: true,
            ..
        }
    ));
    // The anchor: the type that owns the cause of the conflict, in a context of the groups.
    let anchored =
        read("[part type (R) 0 predicate in (Elided<'_, R>) 0, part type (R) 0 in (&R) 0] (Rel)")
            .unwrap();
    assert_eq!(
        part(&anchored.allowed()[0]),
        (
            "R".to_string(),
            anchored.contexts().next().unwrap(),
            true,
            Some((
                "Elided < '_ , R >".to_string(),
                anchored.contexts().next().unwrap()
            ))
        )
    );
    assert!(matches!(
        anchored.allowed()[1],
        Allowed::Part {
            in_predicate: false,
            within: Some(_),
            ..
        }
    ));
    // The tokens of a cause are all of the parentheses; a trailing comma is accepted.
    assert!(read("[cause trait (RelTr),] (Rel)").is_ok());
    assert!(read("[] (Rel)").is_ok());
    for (text, message) in [
        ("[what type (R)] (Rel)", "expected `cause` or `part`"),
        ("[cause nope (R)] (Rel)", "unknown kind `nope`"),
        ("[cause type (R R)] (Rel)", "unexpected token in the cause"),
        ("[part type (R)] (Rel)", "expected integer literal"),
        (
            "[part type (R) 0 nope] (Rel)",
            "expected `predicate`, `in` or `,`",
        ),
        (
            "[part type (R) 3] (Rel)",
            "the context 3 of a part is not one of the contexts",
        ),
        // The anchor is a type in parentheses, with its context.
        (
            "[part type (R) 0 in (&R)] (Rel)",
            "expected integer literal",
        ),
        ("[part type (R) 0 in &R 0] (Rel)", "expected parentheses"),
        ("[part type (R) 0 in (&R R) 0] (Rel)", "unexpected token"),
        (
            "[part type (R) 0 in (&R) 4] (Rel)",
            "the context 4 of a part is not one of the contexts",
        ),
    ] {
        let error = read(text).err().unwrap_or_else(|| panic!("read: {}", text));
        assert!(error.contains(message), "{}: {}", text, error);
    }
}

#[test]
fn the_list_is_deterministic() {
    let make = || {
        let mut config = LeakerConfig::new();
        config.allow(rel_tr());
        let item = elided_and_bare();
        let (referrer, context) = leak_trait(config, &item);
        (referrer.to_token_stream().to_string(), context)
    };
    assert_eq!(make(), make());
    let again = || {
        let item = r1_in_a_bound();
        let (config, _) = accepting_errors_of(&std_config(), &item);
        leak_trait(config, &item).0.to_token_stream().to_string()
    };
    assert_eq!(again(), again());
}

#[test]
fn a_part_with_an_anchor_is_left_as_written_only_inside_a_type_that_matches_it() {
    let tag = format!("\"type-leak-format-{}\"", Referrer::FORMAT_VERSION);
    // Two contexts: 0 has no generics, and the entry `dyn RelTr`; 1 has the generic `T`.
    let read = |anchor_context: usize| -> Referrer {
        parse_str(&format!(
            "{} [part type (dyn RelTr) 0 in (::p::Elided<'_, dyn RelTr>) {}] 2 (dyn RelTr) < T > ; ()",
            tag, anchor_context
        ))
        .unwrap()
    };
    let replace = |_: Type, ix: usize| -> Type { parse_quote!(Entry<#ix>) };
    let anchor: Type = parse_quote!(::p::Elided<'_, dyn RelTr>);
    let other: Type = parse_quote!(::p::Bare<dyn RelTr>);
    // The anchor has the context of the type that is visited: the part is left as written in it,
    // and not in a type that is not the anchor.
    let referrer = read(0);
    let context = referrer.contexts().next().unwrap();
    assert_eq!(
        referrer.expand(context, anchor.clone(), replace),
        anchor.clone()
    );
    let expected: Type = parse_quote!(::p::Bare<Entry<0usize>>);
    assert_eq!(referrer.expand(context, other.clone(), replace), expected);
    // The anchor has another context (it names `T`): the type is not it, and `dyn RelTr` is the
    // entry it is everywhere else.
    let referrer = read(1);
    let expected: Type = parse_quote!(::p::Elided<'_, Entry<0usize>>);
    assert_eq!(referrer.expand(context, anchor.clone(), replace), expected);
    // The anchor around a type that is not the part: the other types in it are looked up as usual.
    let referrer: Referrer = parse_str(&format!(
        "{} [part type (dyn RelTr) 0 in (::p::Elided<'_, dyn RelTr, Rel>) 0] (dyn RelTr, Rel)",
        tag
    ))
    .unwrap();
    let ty: Type = parse_quote!(::p::Elided<'_, dyn RelTr, Rel>);
    let expected: Type = parse_quote!(::p::Elided<'_, dyn RelTr, Entry<1usize>>);
    assert_eq!(referrer.expand(context, ty, replace), expected);
    // Outside any anchor, the same type in the same context is an entry.
    let ty: Type = parse_quote!(::p::Elided<'_, dyn RelTr, Other>);
    let expected: Type = parse_quote!(::p::Elided<'_, Entry<0usize>, Other>);
    assert_eq!(referrer.expand(context, ty, replace), expected);
}

#[test]
fn an_anchor_in_a_method_with_other_generics_does_not_activate() {
    // `ai` has the conflict of the first method, and its `impl Into<R1<Y>>` is the anchor of a
    // part. The second method has the same tokens in the same place, but `Y` has no bound that
    // names a leaked type there, so `R1<Y>` is another entry (another context) and no conflict.
    let item: ItemTrait = parse_quote!(
        trait Tr {
            fn m<Y: Clone>(&self, a1: R1<Y>, ai: impl Into<R1<Y>>)
            where
                Y: Fn(R0) -> u8;
            fn n<Y: Clone>(&self, aj: impl Into<R1<Y>>);
        }
    );
    let (config, errors) = accepting_errors_of(&std_config(), &item);
    assert_eq!(errors.len(), 1);
    let (referrer, context) = leak_trait(config, &item);
    // `R0`, `R1<Y>` with the bound of `m`, and `R1<Y>` of `n`.
    assert_eq!(referrer.iter().count(), 3);
    assert_eq!(referrer.allowed().len(), 2);
    let expected: ItemTrait = parse_quote!(
        trait Tr {
            fn m<Y: Clone>(&self, a1: Entry<1usize>, ai: impl Into<R1<Y>>)
            where
                Y: Fn(Entry<0usize>) -> u8;
            fn n<Y: Clone>(&self, aj: impl Into<Entry<2usize>>);
        }
    );
    assert_eq!(rewritten(&referrer, context, &item), expected);
}

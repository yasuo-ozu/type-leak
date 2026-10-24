//! `LeakerResult::allow()` and its filters: accepting a conflict.
//!
//! The Leaker corrects every conflict the same way, reported or not: the part is copied as written,
//! whole, with nothing in it leaked. `allow(cause)` drops the errors with an equal cause (the same
//! kind and the same tokens, whatever the spans and the spacing) and nothing else: the Referrer
//! never changes.
//!
//! - The item below (`f` and `g`): the error, the causes that do not match (the kind and the tokens
//!   count), and that the entries are those of the item without `f`.
//! - The cause that is smaller than the part that is copied (the array with a closure).
//! - Scopes: the errors of an inner `enter_*` of a `View` come once, in place, `Self` without
//!   meaning and its place among the errors, the other kinds of cause.
//! - The corrections that no conflict of the type itself makes: an entry that names a `Self`
//!   without meaning, or whose bounds do (in a type, a bound, a default, a module, a value), and a
//!   type that leaves a lifetime of its method unconstrained, are copied whole and have no entry.
//! - What accepting does not make compile (`Self` in a predicate beside a leaked bound).
//! - `allow_std_crates()`, `finish()` and `into_referrer()`.
//! - The list of errors of a Leaker: several calls, nested scopes, a panic.
//! - The parts that the Referrer carries (the last section): the visitor leaves a part as written
//!   (also beside an entry with the same tokens) and looks up no entry in it, and the serialization.
//!
//! Every `Reason` is accepted by `tests/error_items.rs`
//! (`allowing_the_causes_of_the_errors_of_each_reason_accepts_the_item`), and a cause with spans
//! of its own by `tests/crates/spans`.

mod common;

use common::allow_every_cause;
use proc_macro2::{Delimiter, TokenStream, TokenTree};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::panic::{catch_unwind, AssertUnwindSafe};
use syn::visit_mut::VisitMut;
use syn::*;
use template_quote::{quote, ToTokens};
use type_leak::*;

/// The trait that is relative in `f` and `g` below.
fn rel_tr() -> Cause {
    Cause::Trait(parse_quote!(RelTr))
}

/// The result of the analysis of an impl.
fn analyzed(item: &ItemImpl) -> LeakerResult {
    let mut leaker = Leaker::new();
    leaker.intern_impl(item, true);
    leaker.result()
}

/// `result` with these causes allowed.
fn allowing(mut result: LeakerResult, causes: &[Cause]) -> LeakerResult {
    for cause in causes {
        result.allow(cause.clone());
    }
    result
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
fn entries(item: &ItemImpl) -> Vec<(String, bool, String)> {
    let referrer = analyzed(item).into_referrer();
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

/// The errors of `f_and_g()`.
fn errors_of_f_and_g() -> Vec<ErrorItem> {
    analyzed(&f_and_g()).errors().to_vec()
}

#[test]
fn the_item_has_one_error() {
    let errors = errors_of_f_and_g();
    assert_eq!(errors.len(), 1);
    assert_eq!(errors[0].cause, rel_tr());
    assert_eq!(errors[0].reason, Reason::RelativePath);
}

#[test]
fn allowing_the_cause_of_the_error_accepts_the_item() {
    let mut result = analyzed(&f_and_g());
    assert!(result.allow(rel_tr()).errors().is_empty());
    assert!(result.finish().is_ok());
    // The cause of the error itself is the same.
    let cause = errors_of_f_and_g().remove(0).cause;
    assert!(allowing(analyzed(&f_and_g()), &[cause]).finish().is_ok());
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
        let result = allowing(analyzed(&f_and_g()), std::slice::from_ref(&cause));
        assert_eq!(result.errors().len(), 1, "{:?}", cause);
        assert_eq!(result.errors()[0].cause, rel_tr());
    }
    // A wrong one beside the right one accepts it all the same.
    let wrong = Cause::Type(parse_quote!(RelTr));
    assert!(allowing(analyzed(&f_and_g()), &[wrong, rel_tr()])
        .finish()
        .is_ok());
}

#[test]
fn a_cause_written_by_hand_matches_one_that_the_analysis_made() {
    // Other values of the same tokens: parsed from a string, and read back from the serialized
    // error.
    let made = errors_of_f_and_g().remove(0);
    let reparsed: ErrorItem = parse2(made.to_token_stream()).unwrap();
    let from_string = Cause::Trait(parse_str("  RelTr ").unwrap());
    for cause in [reparsed.cause, from_string, made.cause] {
        assert_eq!(cause, rel_tr());
        assert_eq!(hash_of(&cause), hash_of(&rel_tr()));
        assert!(allowing(analyzed(&f_and_g()), &[cause]).finish().is_ok());
    }
}

#[test]
fn allowing_a_cause_leaves_the_referrer_as_it_is() {
    let mut result = analyzed(&f_and_g());
    let referrer = result.clone().into_referrer();
    result.allow(rel_tr());
    assert!(result.errors().is_empty());
    assert_eq!(result.into_referrer(), referrer);
}

#[test]
fn the_entries_are_those_of_the_item_without_the_conflict() {
    let accepted = entries(&f_and_g());
    assert_eq!(accepted, entries(&g_only()));
    // `g` still leaks `dyn RelTr`, which is required, and `::p::X<dyn RelTr>` is optional.
    let find = |ty: &str| accepted.iter().find(|entry| entry.0 == ty).unwrap();
    assert!(find("dyn RelTr").1);
    assert!(!find(":: p :: X < dyn RelTr >").1);
}

#[test]
fn allowed_paths_would_change_more() {
    // The trait made context-free everywhere: `g` loses its required entry too.
    let mut config = LeakerConfig::new();
    config.allow_path(parse_quote!(RelTr));
    let mut leaker = Leaker::from_config(config);
    leaker.intern_impl(&f_and_g(), true);
    let referrer = leaker.result().finish().unwrap();
    assert!(referrer.iter().all(|ty| *ty != parse_quote!(dyn RelTr)));
    assert!(!referrer.has_required());
}

#[test]
fn a_cause_that_matches_no_conflict_changes_nothing() {
    // A trait that is a conflict nowhere, and the types that `g` holds, which are no conflict:
    // `g` has no error to drop, and the error of `f` stays.
    let causes = [
        Cause::Trait(parse_quote!(OtherTr)),
        Cause::Type(parse_quote!(dyn RelTr)),
        Cause::Type(parse_quote!(::p::X<dyn RelTr>)),
    ];
    let result = allowing(analyzed(&g_only()), &causes);
    assert!(result.errors().is_empty());
    assert_eq!(result.into_referrer(), analyzed(&g_only()).into_referrer());
    let result = allowing(analyzed(&f_and_g()), &causes);
    assert_eq!(result.errors().len(), 1);
    assert_eq!(result.errors()[0].cause, rel_tr());
    // The paths that are entries in a struct without a conflict: the same Referrer as without.
    let item: ItemStruct = parse_quote!(
        struct S {
            a: Rel,
            b: ::p::X<Other>,
        }
    );
    let analyze = || {
        let mut leaker = Leaker::new();
        leaker.intern_struct(&item, true);
        leaker.result()
    };
    let causes = [
        Cause::Type(parse_quote!(Rel)),
        Cause::Trait(parse_quote!(Rel)),
        Cause::Type(parse_quote!(Other)),
    ];
    let result = allowing(analyze(), &causes);
    assert_eq!(result.finish().unwrap(), analyze().finish().unwrap());
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
    let result = analyzed(&item);
    assert_eq!(result.errors().len(), 3);
    assert!(result.errors().iter().all(|error| error.cause == rel_tr()));
    assert!(allowing(result, &[rel_tr()]).finish().is_ok());
}

#[test]
fn a_conflict_that_is_not_accepted_stays_beside_one_that_is() {
    let item: ItemImpl = parse_quote!(
        impl ::p::S {
            fn f(_: ::p::W<'_, dyn RelTr>) {}
            fn h(_: ::p::Y<'_, dyn Other>) {}
        }
    );
    let result = allowing(analyzed(&item), &[rel_tr()]);
    assert_eq!(result.errors().len(), 1);
    assert_eq!(result.errors()[0].cause, Cause::Trait(parse_quote!(Other)));
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
    let result = analyzed(&item);
    assert_eq!(result.errors().len(), 1);
    assert_eq!(result.errors()[0].cause, Cause::Type(parse_quote!(u8)));
    assert_eq!(
        result.errors()[0].note(),
        Some("it cannot be leaked as a whole because of `| a : u8 | a`")
    );
    let causes = [Cause::Type(parse_quote!(u8))];
    let accepted = allowing(result.clone(), &causes);
    assert!(accepted.errors().is_empty());
    let referrer = accepted.finish().unwrap();
    // `::p::V<u8>` is a required entry, as it is without the array, and nothing is for `a`.
    let required: Vec<&Type> = referrer.required().map(|(_, ty)| ty).collect();
    assert_eq!(required, [&parse_quote!(::p::V<u8>)]);
    let only_g: ItemImpl = parse_quote!(impl ::p::S { fn g(_: ::p::V<u8>) {} });
    assert_eq!(entries(&item), entries(&only_g));
    // `a` alone accepts nothing: it is not the cause of the error.
    let a = Cause::Expr(parse_quote!(a));
    assert_eq!(allowing(result, &[a]).errors().len(), 1);
}

// ── scopes ───────────────────────────────────────────────────────────────────────────────────

/// A type with the conflict of `f`.
fn conflict() -> Type {
    parse_quote!(::p::W<'_, dyn RelTr>)
}

/// A Leaker that has visited `conflict()` and `Rel<T>` in a scope inside the scope of no generics,
/// and `::p::Bare<dyn RelTr>` too, whose `dyn RelTr` is an entry.
fn conflict_in_an_inner_scope() -> (LeakerResult, ContextId) {
    let mut leaker = Leaker::new();
    let (_, inner) = leaker.enter_generics(&Generics::default(), |view| {
        view.enter_generics(&parse_quote!(<T>), |inner| {
            inner.visit_type(&conflict());
            inner.visit_type(&parse_quote!(Rel<T>));
            inner.visit_type(&parse_quote!(::p::Bare<dyn RelTr>));
        })
        .0
    });
    (leaker.result(), inner)
}

#[test]
fn an_error_in_an_inner_scope_is_dropped_by_allowing_its_cause() {
    let (mut result, _) = conflict_in_an_inner_scope();
    assert_eq!(result.errors().len(), 1);
    result.allow(rel_tr());
    // The scope stays, with the entries that it also holds.
    let referrer = result.finish().unwrap();
    let required: Vec<&Type> = referrer.required().map(|(_, ty)| ty).collect();
    assert_eq!(required, [&parse_quote!(Rel<T>), &parse_quote!(dyn RelTr)]);
}

#[test]
fn an_error_in_an_inner_scope_is_recovered_and_recorded_once() {
    for causes in [vec![], vec![Cause::Trait(parse_quote!(OtherTr))]] {
        let (result, inner) = conflict_in_an_inner_scope();
        let result = allowing(result, &causes);
        // The only list is that of the outermost call: the error of the inner scope, once.
        assert_eq!(result.errors().len(), 1);
        assert_eq!(result.errors()[0].cause, rel_tr());
        // The inner scope keeps what it found: `Rel<T>` and `dyn RelTr` are entries, and the
        // conflict is copied as written, though it holds the same `dyn RelTr`.
        let referrer = result.into_referrer();
        let required: Vec<&Type> = referrer.required().map(|(_, ty)| ty).collect();
        assert_eq!(required, [&parse_quote!(Rel<T>), &parse_quote!(dyn RelTr)]);
        let mut copy: Vec<Type> = vec![
            conflict(),
            parse_quote!(Rel<T>),
            parse_quote!(::p::Bare<dyn RelTr>),
        ];
        let mut visitor = referrer.visitor(inner, |_, ix| parse_quote!(Entry<#ix>));
        for ty in &mut copy {
            visitor.visit_type_mut(ty);
        }
        let expected: Vec<Type> = vec![
            conflict(),
            parse_quote!(Entry<0usize>),
            parse_quote!(::p::Bare<Entry<1usize>>),
        ];
        assert_eq!(copy, expected);
    }
}

#[test]
fn the_errors_of_the_inner_and_the_outer_scope_are_one_list_in_order() {
    let mut leaker = Leaker::new();
    leaker.enter_generics(&Generics::default(), |view| {
        // The error of the inner scope, then that of the outer one.
        view.enter_generics(&parse_quote!(<T>), |inner| {
            inner.visit_type(&conflict());
            inner.visit_type(&parse_quote!(Rel<T>));
        });
        view.visit_type(&parse_quote!(::p::W<'_, dyn Other>));
    });
    let mut result = leaker.result();
    let causes: Vec<Cause> = result.errors().iter().map(|e| e.cause.clone()).collect();
    assert_eq!(causes, [rel_tr(), Cause::Trait(parse_quote!(Other))]);
    // Allowing the first leaves the second.
    result.allow(rel_tr());
    assert_eq!(result.errors().len(), 1);
    assert_eq!(result.errors()[0].cause, Cause::Trait(parse_quote!(Other)));
    // The scope keeps `Rel<T>` as well, and the conflicts it recovered from.
    let referrer = result.into_referrer();
    let required: Vec<&Type> = referrer.required().map(|(_, ty)| ty).collect();
    assert_eq!(required, [&parse_quote!(Rel<T>)]);
}

#[test]
fn an_error_in_a_module_is_dropped_by_allowing_its_cause() {
    let item: ItemMod = parse_quote!(
        mod m {
            pub struct S(::p::W<'_, dyn RelTr>, Rel);
        }
    );
    let analyze = || {
        let mut leaker = Leaker::new();
        leaker.intern_mod(&item);
        leaker.result()
    };
    assert!(analyze().finish().is_err());
    let referrer = allowing(analyze(), &[rel_tr()]).finish().unwrap();
    let required: Vec<&Type> = referrer.required().map(|(_, ty)| ty).collect();
    assert_eq!(required, [&parse_quote!(Rel)]);
}

// ── `Self` without meaning ───────────────────────────────────────────────────────────────────

/// The result of the scope in which `Self` has no meaning, with these types in it.
fn without_self(types: &[Type]) -> LeakerResult {
    let mut leaker = Leaker::new();
    leaker.enter_generics(&Generics::default(), |view| {
        for ty in types {
            view.visit_type(ty);
        }
    });
    leaker.result()
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
    let result = without_self(std::slice::from_ref(&ty));
    assert_eq!(
        described(result.errors()),
        [(Reason::SelfWithoutMeaning, "Self".into())]
    );
    // The scope is accepted by the cause, and not by `Rel<Self>`, which is no cause.
    let self_ty = Cause::Type(parse_quote!(Self));
    let accepts =
        |causes: &[Cause], types: &[Type]| allowing(without_self(types), causes).finish().is_ok();
    assert!(accepts(
        std::slice::from_ref(&self_ty),
        std::slice::from_ref(&ty)
    ));
    assert!(!accepts(&[Cause::Type(ty.clone())], &[ty]));
    // `Self::Item` is another cause.
    let item: Type = parse_quote!(Wrap<Self::Item>);
    let self_item = Cause::Type(parse_quote!(Self::Item));
    assert!(!accepts(&[], std::slice::from_ref(&item)));
    assert!(!accepts(&[self_ty], std::slice::from_ref(&item)));
    assert!(accepts(&[self_item], &[item]));
}

#[test]
fn a_mention_of_self_is_reported_with_the_other_errors() {
    let types = [conflict(), parse_quote!(Rel<Self>)];
    let self_ty = Cause::Type(parse_quote!(Self));
    let left = |causes: &[Cause]| described(allowing(without_self(&types), causes).errors());
    let conflict = vec![(Reason::RelativePath, "RelTr".to_string())];
    let mention = (Reason::SelfWithoutMeaning, "Self".to_string());
    // Both are reported, in the order in which they were found.
    assert_eq!(left(&[]), [conflict[0].clone(), mention.clone()]);
    // Accepting the conflict leaves the mention.
    assert_eq!(left(&[rel_tr()]), [mention]);
    // Accepting the mention alone leaves the conflict, and only it.
    assert_eq!(left(std::slice::from_ref(&self_ty)), conflict);
    // Accepting both accepts the scope.
    assert!(allowing(without_self(&types), &[rel_tr(), self_ty])
        .finish()
        .is_ok());
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

/// The entries of `types` visited where `Self` has no meaning, and the errors they are with.
fn entries_without_self(types: &[Type]) -> (Entries, Vec<(Reason, String)>) {
    let result = without_self(types);
    let errors = described(result.errors());
    (listed(&result.into_referrer()), errors)
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
    let mut leaker = Leaker::new();
    leaker.enter_generics(&Generics::default(), |view| {
        view.visit_item(&Item::Fn(item.clone()))
    });
    assert_eq!(
        listed(&leaker.result().into_referrer()),
        [("Rel".to_string(), true)]
    );
    let (entries, _) = entries_without_self(&[parse_quote!(Rel), parse_quote!(Rel<Self>)]);
    assert_eq!(entries, [("Rel".to_string(), true)]);
    // A type that is no entry, as it names a generic of the method: its parts are, those that do not
    // name `Self`.
    let item: ItemFn = parse_quote!(
        fn g<U>(x: (Vec<U>, Rel, Self)) {}
    );
    let mut leaker = Leaker::new();
    leaker.enter_generics(&Generics::default(), |view| {
        view.visit_item(&Item::Fn(item.clone()))
    });
    let mut found = listed(&leaker.result().into_referrer());
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
        let visit = || {
            let mut leaker = Leaker::new();
            leaker.enter_generics(generics, |view| {
                view.visit_type(&parse_quote!(Rel<T>));
                view.visit_type(&parse_quote!(Other<U>));
                // An optional entry has the same bound.
                view.visit_type(&parse_quote!(::p::Opt<T>));
            });
            leaker.result()
        };
        let result = visit();
        assert_eq!(
            described(result.errors()),
            [(Reason::SelfWithoutMeaning, "Self".into())]
        );
        // `Rel<T>` would carry the bound with `Self`; `Other<U>` has nothing to do with it.
        assert_eq!(
            listed(&result.clone().into_referrer()),
            [("Other < U >".to_string(), true)]
        );
        assert!(allowing(result, std::slice::from_ref(&self_ty))
            .finish()
            .is_ok());
    }
    // The scope that names `Self` in a bound is an outer one.
    let mut leaker = Leaker::new();
    leaker.enter_generics(&parse_quote!(<T: AsRef<Self>>), |view| {
        view.enter_generics(&parse_quote!(<U>), |inner| {
            inner.visit_type(&parse_quote!(Rel<T>));
            inner.visit_type(&parse_quote!(Other<U>));
        });
    });
    let result = leaker.result();
    assert_eq!(result.errors().len(), 1);
    assert_eq!(
        listed(&result.into_referrer()),
        [("Other < U >".to_string(), true)]
    );
}

#[test]
fn a_default_with_a_self_without_meaning_is_copied_whole() {
    // Defaults are analyzed alone, in the scope that declares them (`finish()` and `end_scope()`).
    let generics: Generics = parse_quote!(<T = Rel<Self>, U = Rel2>);
    let mut leaker = Leaker::new();
    leaker.enter_generics(&generics, |_| ());
    let result = leaker.result();
    assert_eq!(result.errors().len(), 1);
    assert_eq!(
        listed(&result.into_referrer()),
        [("Rel2".to_string(), true)]
    );
    let mut leaker = Leaker::new();
    leaker.enter_generics(&Generics::default(), |view| {
        view.enter_generics(&generics, |_| ());
    });
    let result = leaker.result();
    assert_eq!(result.errors().len(), 1);
    assert_eq!(
        listed(&result.into_referrer()),
        [("Rel2".to_string(), true)]
    );
}

#[test]
fn a_module_with_a_self_without_meaning_is_copied_whole() {
    let mut leaker = Leaker::new();
    leaker.enter_mod("m", Vec::new(), |view| {
        view.visit_type(&parse_quote!((Rel, Rel<Self>)));
        view.visit_type(&parse_quote!(Rel2));
    });
    let result = leaker.result();
    assert_eq!(result.errors().len(), 1);
    assert_eq!(
        listed(&result.into_referrer()),
        [("Rel2".to_string(), true)]
    );
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
    let mut leaker = Leaker::from_config(config);
    leaker.enter_generics(&Generics::default(), |view| {
        view.visit_item(&Item::Fn(item.clone()))
    });
    let mut result = leaker.result();
    assert!(!result.errors().is_empty());
    assert!(result
        .errors()
        .iter()
        .all(|error| error.reason == Reason::SelfWithoutMeaning));
    allow_every_cause(&mut result);
    assert_eq!(
        listed(&result.finish().unwrap()),
        [("Rel2".to_string(), true)]
    );
}

#[test]
fn the_errors_of_a_type_with_a_self_without_meaning_and_a_conflict_are_both_told() {
    // A conflict inside it is an error, and so is the mention of `Self`, in the order in which the
    // analysis finds them.
    let types = [parse_quote!((Rel, Rel<Self>, ::p::W<'_, dyn RelTr>))];
    let left = |causes: &[Cause]| described(allowing(without_self(&types), causes).errors());
    let conflict = (Reason::RelativePath, "RelTr".to_string());
    let mention = (Reason::SelfWithoutMeaning, "Self".to_string());
    assert_eq!(left(&[]), [mention.clone(), conflict.clone()]);
    assert_eq!(left(&[rel_tr()]), [mention]);
    assert_eq!(left(&[Cause::Type(parse_quote!(Self))]), [conflict]);
    assert!(allowing(
        without_self(&types),
        &[rel_tr(), Cause::Type(parse_quote!(Self))]
    )
    .finish()
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
    let analyze = || {
        let mut leaker = Leaker::from_config(config.clone());
        leaker.intern_trait(&item, true);
        leaker.result()
    };
    let result = analyze();
    assert_eq!(result.errors().len(), 2);
    assert!(result
        .errors()
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
        result
            .errors()
            .iter()
            .map(|error| error.cause.clone())
            .collect(),
        by_hand.to_vec(),
    ] {
        let left = allowing(analyze(), &allowed);
        assert!(left.errors().is_empty(), "{:?}", left.errors());
    }
    // One of them is not enough.
    let left = allowing(analyze(), &by_hand[..1]);
    assert_eq!(left.errors().len(), 1);
    assert_eq!(left.errors()[0].cause, by_hand[1]);
}

#[test]
fn the_default_that_names_self_is_accepted_by_the_whole_default() {
    let item: ItemTrait = parse_quote!(
        trait Tr<T: ::core::default::Default = Wrap<Self>> {}
    );
    let mut leaker = Leaker::new();
    leaker.intern_trait(&item, false);
    let result = leaker.result();
    assert_eq!(result.errors().len(), 1);
    assert_eq!(result.errors()[0].reason, Reason::SelfInDefault);
    assert_eq!(
        result.errors()[0].cause,
        Cause::Type(parse_quote!(Wrap<Self>))
    );
    assert!(
        allowing(result.clone(), &[Cause::Type(parse_quote!(Wrap<Self>))])
            .finish()
            .is_ok()
    );
    // `Self` in it is a note, not the cause.
    let left = allowing(result, &[Cause::Type(parse_quote!(Self))]);
    assert_eq!(left.errors().len(), 1);
}

/// A type that leaves a lifetime of its method unconstrained (E0195) is copied whole, as a trait
/// object is: the conflict is found after the type is chosen as an entry, which the copy would
/// replace.
#[test]
fn a_type_that_leaves_a_lifetime_unconstrained_is_copied_whole() {
    let item: ItemTrait = parse_quote!(
        trait Tr {
            fn f<'b>(&self, x: Wrap<'b>) -> &'b u8;
        }
    );
    let cause = Cause::Type(parse_quote!(Wrap<'b>));
    let mut leaker = Leaker::new();
    leaker.intern_trait(&item, true);
    let result = leaker.result();
    assert_eq!(result.errors().len(), 1);
    assert_eq!(result.errors()[0].reason, Reason::UnconstrainedLifetime);
    assert_eq!(result.errors()[0].cause, cause);
    // Only `u8`: the reference is not leaked whole, and the entry for `Wrap<'b>` is gone.
    assert_eq!(
        listed(&allowing(result, &[cause]).finish().unwrap()),
        [("u8".to_string(), true)]
    );
}

#[test]
fn every_type_that_leaves_the_lifetime_unconstrained_is_copied_whole() {
    let item: ItemTrait = parse_quote!(
        trait Tr {
            fn f<'b>(&self, x: (Wrap<'b>, Rel), y: [Other<'b>; 1]) -> &'b ::p::Out;
        }
    );
    let mut leaker = Leaker::new();
    leaker.intern_trait(&item, true);
    let result = leaker.result();
    assert_eq!(
        described(result.errors()),
        [
            (Reason::UnconstrainedLifetime, "Wrap < 'b >".to_string()),
            (Reason::UnconstrainedLifetime, "Other < 'b >".to_string())
        ]
    );
    // One of them accepted is not enough.
    let one = Cause::Type(parse_quote!(Wrap<'b>));
    let left = allowing(result.clone(), std::slice::from_ref(&one));
    assert_eq!(left.errors().len(), 1);
    let both = allowing(result, &[one, Cause::Type(parse_quote!(Other<'b>))]);
    // `Rel` and the optional `::p::Out` do not hold the lifetime.
    assert_eq!(
        listed(&both.finish().unwrap()),
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
    config.allow_prelude().allow_primitive();
    config
}

/// The documented limit of `LeakerResult::allow()`: accepting `Self::A0` in a predicate copies only
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
    let mut leaker = Leaker::from_config(std_config());
    leaker.intern_trait(&item, true);
    let result = leaker.result();
    assert_eq!(
        described(result.errors()),
        [(Reason::UnprovableBound, "Self :: A0".to_string())]
    );
    let cause = result.errors()[0].cause.clone();
    assert_eq!(
        listed(&allowing(result, &[cause]).finish().unwrap()),
        [("R0".to_string(), true)]
    );
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
    let mut leaker = Leaker::from_config(std_config());
    leaker.intern_impl(&on_self, false);
    let result = leaker.result();
    assert_eq!(
        described(result.errors()),
        [(Reason::UnprovableBound, "Self".to_string())]
    );
    let cause = result.errors()[0].cause.clone();
    let entries = listed(&allowing(result, &[cause]).finish().unwrap());
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
    let mut leaker = Leaker::from_config(std_config());
    leaker.intern_impl(&on_other, false);
    let result = leaker.result();
    assert_eq!(
        described(result.errors()),
        [(Reason::UnprovableBound, "R1 < T >".to_string())]
    );
    let cause = result.errors()[0].cause.clone();
    let entries = listed(&allowing(result, &[cause]).finish().unwrap());
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
    let mut leaker = Leaker::from_config(config);
    leaker.intern_trait(&item, true);
    let mut result = leaker.result();
    assert_eq!(result.errors().len(), 1);
    let lexed = Cause::Unreadable("use ::inner::*;".parse().unwrap());
    let made = result.errors()[0].cause.clone();
    assert_ne!(
        made.to_token_stream().to_string(),
        lexed.to_token_stream().to_string()
    );
    assert_eq!(made, lexed);
    assert_eq!(hash_of(&made), hash_of(&lexed));
    // The one that was lexed accepts the conflict.
    result.allow(lexed);
    assert!(result.finish().is_ok());
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

// ── what the filters do not touch ────────────────────────────────────────────────────────────

#[test]
fn check_reports_the_causes_and_records_no_error() {
    let types: Vec<Type> = vec![
        conflict(),
        parse_quote!(Rel),
        parse_quote!(Rel<'_>),
        parse_quote!(::p::X<dyn RelTr>),
        parse_quote!(&'static u8),
        parse_quote!(Self),
    ];
    let mut leaker = Leaker::new();
    let (_, results) = leaker.enter_generics(&Generics::default(), |view| {
        types
            .iter()
            .map(|ty| format!("{:?}", view.check(ty)))
            .collect::<Vec<_>>()
    });
    // The conflicts are in the results: the causes are told, not reported.
    assert!(results[0].contains("RelTr"));
    let result = leaker.result();
    assert!(result.errors().is_empty());
    assert!(result.finish().unwrap().is_empty());
}

// ── `allow_std_crates()` ─────────────────────────────────────────────────────────────────────

/// Errors whose causes are paths: a prelude name (`Clone`, `Vec<Rel<'_>>`), the paths of the
/// standard crates (`std`, `core`, `alloc`), an absolute path (whose cause is the `Rel<'_>` in it)
/// and a path of the item (`MyType<'_>`).
fn paths() -> ItemTrait {
    parse_quote!(
        trait Tr: Clone {
            fn c(&self, x: Vec<Rel<'_>>);
            fn s(&self, x: std::fmt::Formatter<'_>);
            fn o(&self, x: core::cell::Ref<'_, ::core::primitive::u8>);
            fn b(&self, x: alloc::borrow::Cow<'_, ::core::primitive::str>);
            fn a(&self, x: ::std::vec::Vec<Rel<'_>>);
            fn m(&self, x: MyType<'_>);
        }
    )
}

fn result_of_paths() -> LeakerResult {
    let mut leaker = Leaker::new();
    leaker.intern_trait(&paths(), true);
    leaker.result()
}

/// The causes of the errors, as text.
fn causes(result: &LeakerResult) -> Vec<String> {
    result
        .errors()
        .iter()
        .map(|error| error.cause.to_token_stream().to_string())
        .collect()
}

#[test]
fn allow_std_crates_drops_the_causes_that_start_with_std_core_or_alloc() {
    let mut result = result_of_paths();
    let referrer = result.clone().into_referrer();
    // `Clone` and `Vec<..>` are prelude names, not paths of a crate.
    result.allow_std_crates();
    assert_eq!(
        causes(&result),
        ["Clone", "Vec < Rel < '_ > >", "Rel < '_ >", "MyType < '_ >"]
    );
    assert_eq!(result.into_referrer(), referrer);
}

#[test]
fn allow_std_crates_chains_with_allow() {
    let mut result = result_of_paths();
    let referrer = result.clone().into_referrer();
    result.allow_std_crates();
    assert_eq!(
        causes(&result),
        ["Clone", "Vec < Rel < '_ > >", "Rel < '_ >", "MyType < '_ >"]
    );
    result
        .allow(Cause::Trait(parse_quote!(Clone)))
        .allow(Cause::Type(parse_quote!(Vec<Rel<'_>>)))
        .allow(Cause::Type(parse_quote!(Rel<'_>)))
        .allow(Cause::Type(parse_quote!(MyType<'_>)));
    assert!(result.errors().is_empty());
    assert_eq!(result.finish().unwrap(), referrer);
}

/// A path rule drops an error only if every relative path of it is allowed: `std::vec::Vec<Rel<'_>>`
/// has `Rel` too. The config makes `std::vec::Vec` context-free and reports `Rel<'_>` instead.
#[test]
fn allow_std_crates_needs_every_relative_path_of_the_cause() {
    let item: ItemTrait = parse_quote!(
        trait Tr {
            fn s(&self, x: std::vec::Vec<Rel<'_>>);
            fn t(&self, x: std::fmt::Formatter<'_>);
        }
    );
    let analyze = |config: LeakerConfig| {
        let mut leaker = Leaker::from_config(config);
        leaker.intern_trait(&item, true);
        leaker.result()
    };
    let mut result = analyze(LeakerConfig::new());
    assert_eq!(
        causes(&result),
        [
            "std :: vec :: Vec < Rel < '_ > >",
            "std :: fmt :: Formatter < '_ >"
        ]
    );
    let paths: Vec<Vec<String>> = result
        .errors()
        .iter()
        .map(|error| error.paths.iter().map(|p| quote!(#p).to_string()).collect())
        .collect();
    assert_eq!(
        paths,
        [
            vec!["std :: vec :: Vec", "Rel"],
            vec!["std :: fmt :: Formatter"]
        ]
    );
    result.allow_std_crates();
    assert_eq!(causes(&result), ["std :: vec :: Vec < Rel < '_ > >"]);
    // The config reports the part that is not allowed.
    let mut config = LeakerConfig::new();
    config.allow_std_crates();
    assert_eq!(causes(&analyze(config)), ["Rel < '_ >"]);
}

/// The path of a cause is that of a type, a trait, a macro or a value; the others have none.
#[test]
fn allow_std_crates_reads_the_path_of_a_trait_a_macro_and_a_value() {
    let item: ItemTrait = parse_quote!(
        trait Tr {
            fn f(&self) {
                std::println!("x");
                let n = core::mem::size_of::<u8>();
                let m = inner::make();
            }
        }
    );
    let mut config = LeakerConfig::new();
    config.check_values();
    let analyze = || {
        let mut leaker = Leaker::from_config(config.clone());
        leaker.intern_trait(&item, true);
        leaker.result()
    };
    let result = analyze();
    let has = |kind: fn(&Cause) -> bool| result.errors().iter().any(|e| kind(&e.cause));
    assert!(has(|cause| matches!(cause, Cause::Macro(_))));
    assert!(has(|cause| matches!(cause, Cause::Expr(_))));
    let mut left = analyze();
    left.allow_std_crates();
    // Only `inner::make` is left: the macro and the value of `core` are dropped.
    assert_eq!(causes(&left), ["inner :: make"]);
    // `allow(cause)` drops each of them, whatever its kind.
    let mut left = analyze();
    allow_every_cause(&mut left);
    assert!(left.errors().is_empty());
}

/// A cause with no path is never dropped: `Self`, an unreadable syntax, a closure.
#[test]
fn a_cause_without_a_path_is_never_dropped() {
    let mut result = without_self(&[parse_quote!(Rel<Self>)]);
    result.allow_std_crates();
    assert_eq!(result.errors().len(), 1);

    let item: ItemTrait = parse_quote!(
        trait Tr {
            fn f(&self) {
                macro_rules! m {
                    () => {};
                }
            }
        }
    );
    let mut config = LeakerConfig::new();
    config.check_values();
    let mut leaker = Leaker::from_config(config);
    leaker.intern_trait(&item, true);
    let mut result = leaker.result();
    assert_eq!(result.errors().len(), 1);
    result.allow_std_crates();
    assert_eq!(result.errors().len(), 1);
}

#[test]
fn finish_is_ok_only_if_no_error_remains() {
    let mut result = analyzed(&f_and_g());
    let errors = result.clone().finish().unwrap_err();
    assert_eq!(errors.len(), 1);
    assert_eq!(errors[0].cause, rel_tr());
    // Allowing something else leaves the error, allowing the cause leaves the Referrer.
    result.allow(Cause::Trait(parse_quote!(Other)));
    assert!(result.clone().finish().is_err());
    let recovered = result.clone().into_referrer();
    result.allow(rel_tr());
    assert_eq!(result.finish().unwrap(), recovered);
    // No error at all.
    assert!(analyzed(&g_only()).finish().is_ok());
}

#[test]
fn into_referrer_returns_the_recovered_referrer_whatever_remains() {
    let result = analyzed(&f_and_g());
    assert_eq!(result.errors().len(), 1);
    let referrer = result.into_referrer();
    // The entries of `g` (and the self type, which is another crate's), and `f` copied as written.
    assert_eq!(
        listed(&referrer),
        [
            ("dyn RelTr".to_string(), true),
            (":: p :: S".to_string(), false),
            (":: p :: X < dyn RelTr >".to_string(), false)
        ]
    );
    assert_eq!(parts_of(&referrer).len(), 1);
}

// ── the list of errors of a Leaker ───────────────────────────────────────────────────────────

#[test]
fn the_errors_of_the_calls_are_one_list_in_the_order_found_each_once() {
    let rel = Cause::Type(parse_quote!(Rel<'_>));
    let mut leaker = Leaker::new();
    leaker.enter_generics(&Generics::default(), |view| {
        view.visit_type(&parse_quote!(Rel<'_>));
        view.visit_type(&conflict());
        view.visit_type(&conflict());
        view.visit_type(&parse_quote!(Rel<'_>));
    });
    let result = leaker.result();
    // Every conflict is an error, in the order found.
    let found: Vec<Cause> = result.errors().iter().map(|e| e.cause.clone()).collect();
    assert_eq!(found, [rel.clone(), rel_tr(), rel_tr(), rel.clone()]);
    // The parts are distinct: the same tokens in the same place are one. `Rel<'_>` holds its own
    // lifetime and no type owns it around, whereas the trait object is in a path that does.
    let referrer = result.into_referrer();
    assert_eq!(
        parts_of(&referrer),
        [
            "type (Rel < '_ >) 0",
            "type (dyn RelTr) 0 in (:: p :: W < '_ , dyn RelTr >) 0"
        ]
    );
    // Two outermost calls are one list, in the order of the calls.
    let mut leaker = Leaker::new();
    for ty in [parse_quote!(Rel<'_>), conflict()] {
        leaker.enter_generics(&Generics::default(), |view| view.visit_type(&ty));
    }
    leaker.enter_generics(&Generics::default(), |view| view.visit_type(&conflict()));
    let found: Vec<Cause> = leaker
        .result()
        .errors()
        .iter()
        .map(|e| e.cause.clone())
        .collect();
    assert_eq!(found, [rel, rel_tr(), rel_tr()]);
}

#[test]
fn nested_scopes_record_each_error_once_in_the_order_found() {
    let other = Cause::Trait(parse_quote!(OtherTr));
    let unknown = Cause::Trait(parse_quote!(Unknown));
    let outer: Type = parse_quote!(::p::X<'_, dyn RelTr>);
    let mut leaker = Leaker::new();
    leaker.enter_generics(&Generics::default(), |view| {
        view.visit_type(&outer);
        view.enter_generics(&parse_quote!(<T>), |inner| {
            inner.visit_type(&conflict());
            inner.enter_generics(&parse_quote!(<U>), |innermost| {
                innermost.visit_type(&parse_quote!(::p::Y<'_, dyn OtherTr>));
            });
            inner.visit_type(&parse_quote!(::p::Z<'_, dyn Unknown>));
        });
    });
    let mut result = leaker.result();
    let found: Vec<Cause> = result.errors().iter().map(|e| e.cause.clone()).collect();
    assert_eq!(found, [rel_tr(), rel_tr(), other.clone(), unknown.clone()]);
    let referrer = result.clone().into_referrer();
    // The parts: the outer one, and those of the inner scopes.
    assert_eq!(parts_of(&referrer).len(), 4);
    // Allowing leaves the rest, and the Referrer as it is.
    result.allow(rel_tr()).allow(other);
    assert_eq!(result.errors().len(), 1);
    assert_eq!(result.errors()[0].cause, unknown);
    assert_eq!(result.clone().into_referrer(), referrer);
    result.allow(unknown);
    assert_eq!(result.finish().unwrap(), referrer);
}

#[test]
fn a_panic_in_a_closure_records_nothing() {
    let mut leaker = Leaker::new();
    // An error before the panic is rolled back with the scope.
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        leaker.enter_generics(&Generics::default(), |view| {
            view.visit_type(&conflict());
            panic!("in the closure");
        })
    }));
    assert!(outcome.is_err());
    // An inner scope that panics and is caught by the outer closure: its error is gone too.
    leaker.enter_generics(&Generics::default(), |view| {
        view.visit_type(&parse_quote!(::p::W<'_, dyn Outer>));
        let inner = catch_unwind(AssertUnwindSafe(|| {
            view.enter_generics(&parse_quote!(<T>), |inner| {
                inner.visit_type(&conflict());
                inner.visit_type(&parse_quote!(Rel<T>));
                panic!("in the inner scope");
            })
        }));
        assert!(inner.is_err());
    });
    let result = leaker.result();
    // Only the error of the outer scope remains, and no entry of the scopes that panicked.
    assert_eq!(result.errors().len(), 1);
    assert_eq!(result.errors()[0].cause, Cause::Trait(parse_quote!(Outer)));
    let referrer = result.into_referrer();
    assert!(referrer.is_empty());
    assert_eq!(parts_of(&referrer).len(), 1);
}

// ── the parts that the Referrer carries ──────────────────────────────────────────────────────
//
// The Referrer holds the entries, and its visitor finds an entry by its tokens and its context
// wherever they are: also inside a part that the Leaker meant to copy as written. So the Referrer
// carries the parts that were copied as written, and the visitor leaves a part as written.

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

/// The result of the analysis of `item` and its context.
fn leak_trait(config: LeakerConfig, item: &ItemTrait) -> (LeakerResult, ContextId) {
    let mut leaker = Leaker::from_config(config);
    let context = leaker.intern_trait(item, true);
    (leaker.result(), context)
}

/// The Referrer of `item` and its context.
fn referrer_of(config: LeakerConfig, item: &ItemTrait) -> (Referrer, ContextId) {
    let (result, context) = leak_trait(config, item);
    (result.into_referrer(), context)
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

/// The parts that the Referrer carries, as it serializes them after the format tag: `KIND (CAUSE)
/// CONTEXT [predicate] [in (TYPE) CONTEXT]`, one string for each.
fn parts_of(referrer: &Referrer) -> Vec<String> {
    let mut tokens = referrer.to_token_stream().into_iter().skip(1);
    let list = match tokens.next() {
        Some(TokenTree::Group(group)) if group.delimiter() == Delimiter::Bracket => group.stream(),
        _ => return Vec::new(),
    };
    let (mut parts, mut current) = (Vec::new(), TokenStream::new());
    for token in list {
        match &token {
            TokenTree::Punct(punct) if punct.as_char() == ',' => {
                parts.push(std::mem::take(&mut current).to_string());
            }
            _ => current.extend([token]),
        }
    }
    if !current.is_empty() {
        parts.push(current.to_string());
    }
    parts
}

/// The errors of `item` as text, with the reason.
fn errors_of(config: &LeakerConfig, item: &ItemTrait) -> Vec<(Reason, String)> {
    described(leak_trait(config.clone(), item).0.errors())
}

#[test]
fn the_referrer_carries_the_part_that_is_copied_and_the_entries_beside_it() {
    let (result, context) = leak_trait(LeakerConfig::new(), &elided_and_bare());
    assert_eq!(result.errors().len(), 1);
    assert_eq!(result.errors()[0].cause, rel_tr());
    let referrer = result.into_referrer();
    // The part is the type that conflicts (the trait object, not the trait that is the cause),
    // inside the path that owns the elided lifetime.
    assert_eq!(
        parts_of(&referrer),
        [format!(
            "type (dyn RelTr) {} in (:: p :: Elided < '_ , dyn RelTr >) {}",
            context, context
        )]
    );
    // `dyn RelTr` of `g` is an entry, and the optional entry of the path around it.
    assert_eq!(
        listed(&referrer),
        [
            ("dyn RelTr".to_string(), true),
            (":: p :: Bare < dyn RelTr >".to_string(), false)
        ]
    );
    // Nothing is carried where nothing was copied as written.
    let mut leaker = Leaker::new();
    leaker.intern_trait(
        &parse_quote!(
            trait Tr {
                fn f(&self) -> Rel;
            }
        ),
        true,
    );
    assert!(parts_of(&leaker.result().finish().unwrap()).is_empty());
    assert!(parts_of(
        &Leaker::from_config(LeakerConfig::new())
            .result()
            .into_referrer()
    )
    .is_empty());
}

#[test]
fn the_visitor_leaves_a_part_as_written_and_replaces_the_entry_with_the_same_tokens() {
    let item = elided_and_bare();
    let (referrer, context) = referrer_of(LeakerConfig::new(), &item);
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
    let mut leaker = Leaker::new();
    let in_struct = leaker.intern_struct(&st, true);
    let in_trait = leaker.intern_trait(&tr, true);
    assert_eq!(in_struct, in_trait);
    let result = leaker.result();
    assert_eq!(result.errors().len(), 1);
    let referrer = result.into_referrer();
    assert_eq!(parts_of(&referrer).len(), 1);
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
    let config = std_config();
    assert_eq!(
        errors_of(&config, &item),
        [(Reason::UnprovableBound, "R1 < Y >".to_string())]
    );
    let (referrer, context) = referrer_of(config, &item);
    // `R0` and `R1<Y>` (in the context of the method, which holds `Y`).
    assert_eq!(
        listed(&referrer),
        [("R0".to_string(), true), ("R1 < Y >".to_string(), true)]
    );
    // The type that conflicts is the `R1<Y>` in the bound of the argument, inside the `impl
    // Into<R1<Y>>` that owns the cause (`impl Trait` cannot be leaked), which has the same
    // context.
    assert_eq!(
        parts_of(&referrer),
        [format!(
            "type (R1 < Y >) {} predicate in (impl Into < R1 < Y > >) {}",
            referrer.context(1),
            referrer.context(1)
        )]
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
    let config = std_config();
    assert_eq!(
        errors_of(&config, &item),
        [(Reason::UnprovableBound, "R1 < Y >".to_string())]
    );
    let (referrer, context) = referrer_of(config, &item);
    // No type around it owns a cause of the conflict: no anchor.
    assert_eq!(
        parts_of(&referrer),
        [format!("type (R1 < Y >) {} predicate", referrer.context(1))]
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

/// The errors of two scopes are one list, and the parts they copy as written follow the errors in
/// the Referrer, each once.
#[test]
fn a_silent_correction_is_a_part_and_the_error_it_comes_with_is_the_only_error() {
    // (1) `Self` has no meaning: the type that names it is copied whole instead of being an entry.
    let mut leaker = Leaker::new();
    let (context, ()) = leaker.enter_generics(&Generics::default(), |view| {
        view.visit_type(&parse_quote!(Rel));
        view.visit_type(&parse_quote!(Wrap<Self>));
    });
    let result = leaker.result();
    assert_eq!(
        described(result.errors()),
        [(Reason::SelfWithoutMeaning, "Self".to_string())]
    );
    let referrer = result.into_referrer();
    assert_eq!(
        parts_of(&referrer),
        [format!("type (Wrap < Self >) {}", context)]
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
    let (result, context) = leak_trait(std_config(), &item);
    assert_eq!(result.errors().len(), 1);
    assert_eq!(result.errors()[0].reason, Reason::UnconstrainedLifetime);
    let referrer = result.into_referrer();
    assert_eq!(parts_of(&referrer).len(), 1);
    // `u8` is allowed and not an entry, and `Held<'b>` is not, so nothing is replaced.
    assert_eq!(rewritten(&referrer, context, &item), item);
}

#[test]
fn an_entry_whose_bounds_name_a_self_without_meaning_is_a_part() {
    let mut leaker = Leaker::new();
    let (context, ()) = leaker.enter_generics(&parse_quote!(<T: AsRef<Self>, U>), |view| {
        view.visit_type(&parse_quote!(Rel<T>));
        view.visit_type(&parse_quote!(Other<U>));
    });
    let referrer = leaker.result().into_referrer();
    assert_eq!(parts_of(&referrer).len(), 1);
    // The context of the part holds the bound that names `Self`.
    assert!(referrer.contexts().any(|part_context| referrer
        .context_generics(part_context)
        .to_token_stream()
        .to_string()
        .contains("AsRef")));
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
    // Only `dyn RelTr` is the cause of the error: it is copied as written, in the path that owns
    // the elided lifetime. `Rel` beside it is an entry, and replaced.
    let list = |ty: Type| -> (Entries, Type) {
        let mut leaker = Leaker::from_config(std_config());
        let (context, ()) =
            leaker.enter_generics(&Generics::default(), |view| view.visit_type(&ty));
        let referrer = leaker.result().into_referrer();
        let entries = listed(&referrer);
        let mut copy = ty;
        referrer
            .visitor(context, |_, ix| parse_quote!(Entry<#ix>))
            .visit_type_mut(&mut copy);
        (entries, copy)
    };
    let (entries, copy) = list(parse_quote!(::p::W<'_, dyn RelTr, Rel>));
    assert_eq!(entries, [("Rel".to_string(), true)]);
    let expected: Type = parse_quote!(::p::W<'_, dyn RelTr, Entry<0usize>>);
    assert_eq!(copy, expected);
    // A type around the owner is no owner: the anchor is the path that holds the lifetime.
    let (entries, copy) = list(parse_quote!((Rel, ::p::W<'_, dyn RelTr>)));
    assert_eq!(entries, [("Rel".to_string(), true)]);
    let expected: Type = parse_quote!((Entry<0usize>, ::p::W<'_, dyn RelTr>));
    assert_eq!(copy, expected);
    // So the same tokens beside the owner are an entry, inside the tuple that is no owner.
    let (entries, copy) = list(parse_quote!((::p::W<'_, dyn RelTr>, ::p::Bare<dyn RelTr>)));
    assert_eq!(entries[0], ("dyn RelTr".to_string(), true));
    let expected: Type = parse_quote!((::p::W<'_, dyn RelTr>, ::p::Bare<Entry<0usize>>));
    assert_eq!(copy, expected);
    let (entries, copy) = list(parse_quote!(Vec<::p::W<'_, dyn RelTr>>));
    assert!(entries.is_empty());
    let expected: Type = parse_quote!(Vec<::p::W<'_, dyn RelTr>>);
    assert_eq!(copy, expected);
    // An elided reference owns its cause too.
    let (entries, copy) = list(parse_quote!((Rel, &dyn RelTr)));
    assert_eq!(entries, [("Rel".to_string(), true)]);
    let expected: Type = parse_quote!((Entry<0usize>, &dyn RelTr));
    assert_eq!(copy, expected);
    // The innermost owner is the anchor.
    let (_, copy) = list(parse_quote!(::p::V<'_, ::p::W<'_, dyn RelTr, Rel>, Rel2>));
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
    let config = std_config();
    assert_eq!(
        errors_of(&config, &item),
        [
            (Reason::RelativePath, "RelTrait < R0 >".to_string()),
            (Reason::RelativePath, "Other < R0 >".to_string()),
        ]
    );
    let (referrer, context) = referrer_of(config, &item);
    // Only `R0` of `f` is an entry. The bounds are copied as written, with the `R0` in them. They
    // name no generic, so their context is that of `R0`.
    assert_eq!(listed(&referrer), [("R0".to_string(), true)]);
    let none = referrer.context(0);
    assert_eq!(
        parts_of(&referrer),
        [
            format!("trait (RelTrait < R0 >) {} predicate", none),
            format!("trait (Other < R0 >) {} predicate", none)
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
    let config = std_config();
    let (result, context) = leak_trait(config, &item);
    assert_eq!(
        result.errors()[0].cause.to_token_stream().to_string(),
        "for < 'a > RelTrait < 'a , R0 >"
    );
    let referrer = result.into_referrer();
    let parts = parts_of(&referrer);
    assert_eq!(parts.len(), 1);
    assert!(
        parts[0].starts_with("unreadable (for < 'a > RelTrait < 'a , R0 >)")
            && parts[0].ends_with("predicate"),
        "{}",
        parts[0]
    );
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
    let mut leaker = Leaker::new();
    let context = leaker.intern_mod(&item);
    let result = leaker.result();
    assert_eq!(result.errors().len(), 2);
    let referrer = result.into_referrer();
    // `RelTr` is relative, so each module has its own entries and its own parts.
    let parts = parts_of(&referrer);
    assert_eq!(parts.len(), 2);
    assert_ne!(parts[0], parts[1]);
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
    assert_eq!(
        errors_of(&config, &item),
        [(Reason::RelativePath, "inner :: make :: < R0 >".to_string())]
    );
    let (referrer, context) = referrer_of(config, &item);
    assert_eq!(listed(&referrer), [("R0".to_string(), true)]);
    let parts = parts_of(&referrer);
    assert_eq!(parts.len(), 1);
    assert!(
        parts[0].starts_with("expr (inner :: make :: < R0 >)"),
        "{}",
        parts[0]
    );
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
    let (referrer, context) = referrer_of(LeakerConfig::new(), &item);
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
    let (referrer, context) = referrer_of(LeakerConfig::new(), &item);
    let replace = |_: Type, ix: usize| -> Type { parse_quote!(Entry<#ix>) };
    let part: Type = parse_quote!(::p::Elided<'_, dyn RelTr>);
    assert_eq!(referrer.expand(context, part.clone(), replace), part);
    let other: Type = parse_quote!(::p::Bare<dyn RelTr>);
    let expected: Type = parse_quote!(::p::Bare<Entry<0usize>>);
    assert_eq!(referrer.expand(context, other, replace), expected);
}

#[test]
fn the_parts_are_serialized_with_the_referrer() {
    use syn::parse::Parser;
    let item: ItemTrait = parse_quote!(
        trait Tr<T: RelTrait<R0>> {
            fn f(&self, x: ::p::Elided<'_, dyn RelTr>, y: R0);
            fn m<Y: Clone>(&self, ai: impl Into<R1<Y>>, a1: R1<Y>)
            where
                Y: Fn(R0) -> u8;
        }
    );
    let (referrer, context) = referrer_of(std_config(), &item);
    // The three kinds of cause of a part, in a predicate or not, with an anchor or not.
    let parts = parts_of(&referrer);
    assert!(parts.iter().any(|part| part.starts_with("trait (")));
    assert!(parts.iter().any(|part| part.starts_with("type (")));
    assert!(parts.iter().any(|part| part.ends_with("predicate")));
    assert!(parts.iter().any(|part| part.contains(" in (")));
    assert!(parts.iter().any(|part| !part.contains("predicate")));
    let tokens = quote!(#referrer, #context);
    let text = tokens.to_string();
    // The list follows the tag, and has neither `part` nor `cause` in front of an item.
    assert!(text.starts_with("\"type-leak-format-"), "{}", text);
    assert!(
        !text.contains("cause trait") && !text.contains("part type"),
        "{}",
        text
    );
    let read = (|input: parse::ParseStream| -> Result<(Referrer, ContextId)> {
        let referrer = input.parse()?;
        input.parse::<Token![,]>()?;
        Ok((referrer, input.parse()?))
    })
    .parse2(tokens.clone())
    .unwrap();
    assert_eq!(read.0, referrer);
    assert_eq!(read.1, context);
    assert_eq!(parts_of(&read.0), parts);
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
fn a_referrer_without_parts_writes_what_it_wrote_before() {
    let mut leaker = Leaker::new();
    leaker.intern_struct(
        &parse_quote!(
            struct S {
                a: Rel,
            }
        ),
        true,
    );
    let referrer = leaker.result().finish().unwrap();
    assert!(parts_of(&referrer).is_empty());
    let tag = format!("\"type-leak-format-{}\"", Referrer::FORMAT_VERSION);
    assert_eq!(
        referrer.to_token_stream().to_string(),
        format!("{} (Rel)", tag)
    );
    // Referrers that differ by the parts are not equal.
    let (with, _) = referrer_of(LeakerConfig::new(), &elided_and_bare());
    let mut leaker = Leaker::new();
    leaker.intern_trait(
        &parse_quote!(
            trait Fg {
                fn g(&self, x: ::p::Bare<dyn RelTr>);
            }
        ),
        true,
    );
    let without = leaker.result().finish().unwrap();
    assert!(parts_of(&without).is_empty());
    assert_ne!(with, without);
}

#[test]
fn the_parts_are_checked_when_they_are_read() {
    let tag = format!("\"type-leak-format-{}\"", Referrer::FORMAT_VERSION);
    let read = |text: &str| -> std::result::Result<Referrer, String> {
        parse_str::<Referrer>(&format!("{} {}", tag, text)).map_err(|error| error.to_string())
    };
    // What is written is read, and what is read is written the same: a trait and a type in the
    // context 0 (the only one), in a predicate or not, with an anchor or not.
    for (text, parts) in [
        (
            "[trait (RelTr) 0 predicate, type (Elided<'_, dyn RelTr>) 0] (Rel)",
            2,
        ),
        ("[type (R1<Y>) 0 predicate] (Rel)", 1),
        (
            "[type (R) 0 predicate in (Elided<'_, R>) 0, type (R) 0 in (&R) 0] (Rel)",
            2,
        ),
        ("[type (R) 0] (Rel)", 1),
    ] {
        let good = read(text).unwrap_or_else(|error| panic!("{}: {}", text, error));
        assert_eq!(parts_of(&good).len(), parts, "{}", text);
        let written = good.to_token_stream().to_string();
        let again = parse_str::<Referrer>(&written).unwrap();
        assert_eq!(again, good);
        assert_eq!(again.to_token_stream().to_string(), written);
    }
    let anchored =
        read("[type (R) 0 predicate in (Elided<'_, R>) 0, type (R) 0 in (&R) 0] (Rel)").unwrap();
    let context = anchored.contexts().next().unwrap();
    assert_eq!(
        parts_of(&anchored),
        [
            format!(
                "type (R) {} predicate in (Elided < '_ , R >) {}",
                context, context
            ),
            format!("type (R) {} in (& R) {}", context, context)
        ]
    );
    // The tokens of a cause are all of the parentheses; a trailing comma is accepted.
    assert!(read("[trait (RelTr) 0,] (Rel)").is_ok());
    assert!(read("[] (Rel)").is_ok());
    for (text, message) in [
        // The words `cause` and `part` of the first form of the list are not read any more.
        ("[cause trait (RelTr) 0] (Rel)", "expected parentheses"),
        ("[part type (R) 0] (Rel)", "expected parentheses"),
        ("[what (R) 0] (Rel)", "unknown kind `what`"),
        ("[nope (R) 0] (Rel)", "unknown kind `nope`"),
        ("[type (R R) 0] (Rel)", "unexpected token in the cause"),
        ("[type (R)] (Rel)", "expected integer literal"),
        (
            "[type (R) 0 nope] (Rel)",
            "expected `sig`, `predicate`, `in` or `,`",
        ),
        (
            "[type (R) 3] (Rel)",
            "the context 3 of a part is not one of the contexts",
        ),
        // The anchor is a type in parentheses, with its context.
        ("[type (R) 0 in (&R)] (Rel)", "expected integer literal"),
        ("[type (R) 0 in &R 0] (Rel)", "expected parentheses"),
        ("[type (R) 0 in (&R R) 0] (Rel)", "unexpected token"),
        (
            "[type (R) 0 in (&R) 4] (Rel)",
            "the context 4 of a part is not one of the contexts",
        ),
    ] {
        let error = read(text).err().unwrap_or_else(|| panic!("read: {}", text));
        assert!(error.contains(message), "{}: {}", text, error);
    }
}

#[test]
fn the_parts_are_deterministic() {
    let make = || {
        let (referrer, context) = referrer_of(LeakerConfig::new(), &elided_and_bare());
        (referrer.to_token_stream().to_string(), context)
    };
    assert_eq!(make(), make());
    let again = || {
        let (referrer, _) = referrer_of(std_config(), &r1_in_a_bound());
        referrer.to_token_stream().to_string()
    };
    assert_eq!(again(), again());
}

#[test]
fn a_part_with_an_anchor_is_left_as_written_only_inside_a_type_that_matches_it() {
    let tag = format!("\"type-leak-format-{}\"", Referrer::FORMAT_VERSION);
    // Two contexts: 0 has no generics, and the entry `dyn RelTr`; 1 has the generic `T`.
    let read = |anchor_context: usize| -> Referrer {
        parse_str(&format!(
            "{} [type (dyn RelTr) 0 in (::p::Elided<'_, dyn RelTr>) {}] 2 (dyn RelTr) < T > ; ()",
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
        "{} [type (dyn RelTr) 0 in (::p::Elided<'_, dyn RelTr, Rel>) 0] (dyn RelTr, Rel)",
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
    let config = std_config();
    assert_eq!(errors_of(&config, &item).len(), 1);
    let (referrer, context) = referrer_of(config, &item);
    // `R0`, `R1<Y>` with the bound of `m`, and `R1<Y>` of `n`.
    assert_eq!(referrer.iter().count(), 3);
    assert_eq!(parts_of(&referrer).len(), 1);
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

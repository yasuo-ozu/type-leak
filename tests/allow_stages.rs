//! The `allow*()` operations of `Allow` and `LeakerResult` mean the same thing.
//!
//! `Leaker::with_rule(config.allow_X()).result()` and
//! `Leaker::with_rule(config).result().allow_X()`:
//!
//! - the cause, value and default rules give the same errors and an equal Referrer
//!   (`tests/error_items.rs` checks every trigger of the errors; the cases of a value and of a default
//!   are here);
//! - the path rules (`Rules::or_if()`, `Rules::primitive()`, `Rules::prelude()`, `Rules::std_crates()`,
//!   `Rules::crate_paths()`) make a path context-free during the analysis, which a result cannot do. So
//!   the errors that remain are those of the paths that are not allowed, and a cause can be wider
//!   after the analysis, the Referrer may keep entries that the config does not make (`Vec<u8>`),
//!   and an error can remain that the config does not have (`Self::A` in a predicate beside a leaked
//!   `String`);
//! - the errors are a flat list: a type whose own path conflicts is written as it is and what is in it
//!   is analyzed on its own (`tests/flat_errors.rs`), so `Vec<Rel<'_>>` is an error for `Vec` and
//!   another for `Rel<'_>`, and a rule on `Vec` after the analysis leaves what the config reports.
//!
//! The property test builds the copies after the result and after the config
//! (`TYPE_LEAK_PROP_FEATURES=allow_stages`, `tests/prop_cross_crate/main.rs`): what the tests below
//! say of the copies was found with it.

mod common;

use common::{starts_with, Recovered, Rule};
use syn::*;
use template_quote::{quote, ToTokens};
use type_leak::*;

/// The causes of the errors, as text.
fn causes(result: &LeakerResult) -> Vec<String> {
    result
        .errors()
        .iter()
        .map(|error| error.cause.to_token_stream().to_string())
        .collect()
}

/// The cause of an error, as text.
fn causes_text(error: &ErrorItem) -> String {
    error.cause.to_token_stream().to_string()
}

/// The causes of `errors`, as text.
fn causes_text_of(errors: &[ErrorItem]) -> Vec<String> {
    errors.iter().map(causes_text).collect()
}

/// The entries of the Referrer, as text.
fn entries(result: &LeakerResult) -> Vec<String> {
    let referrer = result.clone().recovered();
    common::types(&referrer)
        .map(|ty| quote!(#ty).to_string())
        .collect()
}

fn analyze<A: Allow>(item: &ItemTrait, config: A) -> LeakerResult {
    common::analyze_trait(item, config, false)
}

/// [`analyze()`] with `check_values`: the values are read too.
fn analyze_values<A: Allow>(item: &ItemTrait, config: A) -> LeakerResult {
    common::analyze_trait(item, config, true)
}

/// The result of `item` with `rule` given to the config (`before`) and to its result (`after`), the
/// other options being those of `base`.
fn stages<A: Allow>(
    item: &ItemTrait,
    base: impl Fn(Rules) -> A,
    rule: Rule,
) -> (LeakerResult, LeakerResult) {
    let before = rule.analyze_with(item, base(Rules::new()), false);
    let after = rule.on_result(analyze(item, base(Rules::new())));
    (before, after)
}

#[test]
fn allow_prelude_reports_the_same_error_before_and_after() {
    let item: ItemTrait = parse_quote!(
        trait Tr<T: Clone> {
            fn a(&self, x: Vec<u8>);
            fn c(&self, x: Vec<Rel<'_>>);
        }
    );
    let (before, after) = stages(&item, |c| c.primitive(), Rule::Prelude);
    // `Clone` is dropped by both. The config does not leak `Vec`, so `Rel<'_>` is the error; the
    // result has the errors of `Vec` and of `Rel<'_>`, and the rule drops the first.
    assert_eq!(causes(&before), ["Rel < '_ >"]);
    assert_eq!(causes(&after), ["Rel < '_ >"]);
    let analyzed = analyze(&item, Rules::new().primitive());
    assert_eq!(
        causes(&analyzed),
        ["Clone", "Vec < Rel < '_ > >", "Rel < '_ >"]
    );
    assert_eq!(before.errors().len(), after.errors().len());
    // The config leaves `Vec<u8>` and `Vec<Rel<'_>>` as written. After the analysis the Referrer
    // still has the entry `Vec<u8>`, which a Referrer replaces as it does any entry.
    assert!(entries(&before).is_empty());
    assert_eq!(entries(&after), ["Vec < u8 >"]);
    // The result with the same rule once more, and with the config's, is the same result.
    let twice = after.clone().allow(Rules::new().prelude());
    assert_eq!(causes(&twice), causes(&after));
}

#[test]
fn allow_primitive_drops_a_primitive_path_in_a_const_expression_at_both_stages() {
    // `u8::MAX` is a path of a value in a const expression: an error, for the path `u8::MAX`.
    let item: ItemTrait = parse_quote!(
        trait Tr: ::p::Sup<{ u8::MAX }> {
            fn c(
                &self,
                x: [u8; {
                    let f = |a: u8| a;
                    3
                }],
            );
        }
    );
    let (before, after) = stages(&item, |c| c, Rule::Primitive);
    // The closure's `a` is relative, so the array is still an error, with `u8` allowed. After the
    // analysis the cause is `u8`, the first path of the array, and `a` is the second path: the rule
    // removes `u8` from the paths of the error and leaves `a`. The error has paths, so it is
    // accepted by them and not by its cause (which the rule allows).
    assert_eq!(causes(&before), ["a"]);
    assert_eq!(causes(&after), ["u8"]);
    assert_eq!(before.errors().len(), after.errors().len());
    let texts = |error: &ErrorItem| -> Vec<String> {
        error.paths.iter().map(|p| quote!(#p).to_string()).collect()
    };
    assert_eq!(texts(&after.errors()[0]), ["a"]);
    assert_eq!(
        texts(&analyze(&item, Rules::new()).errors()[1]),
        ["u8", "a"]
    );
}

/// A path rule drops an error only if every relative path of it is allowed. `std::vec::Vec<Rel<'_>>`
/// is written as it is, with an error for `std::vec::Vec`, and `Rel<'_>` in it has an error of its
/// own: the rule drops the first, as the config makes `std::vec::Vec` context-free and reports
/// `Rel<'_>`.
#[test]
fn allow_std_crates_reports_the_same_error_before_and_after() {
    let item: ItemTrait = parse_quote!(
        trait Tr: core::fmt::Debug {
            fn s(&self, x: std::vec::Vec<Rel<'_>>);
            fn t(&self, x: std::fmt::Formatter<'_>);
        }
    );
    let (before, after) = stages(&item, |c| c.primitive(), Rule::StdCrates);
    assert_eq!(causes(&before), ["Rel < '_ >"]);
    assert_eq!(causes(&after), ["Rel < '_ >"]);
    assert_eq!(before.errors().len(), after.errors().len());
    // Without the rule: an error for the head of each path, and the one inside `Vec`.
    let plain = analyze(&item, Rules::new());
    assert_eq!(
        causes(&plain),
        [
            "core :: fmt :: Debug",
            "std :: vec :: Vec < Rel < '_ > >",
            "Rel < '_ >",
            "std :: fmt :: Formatter < '_ >"
        ]
    );
    let paths: Vec<Vec<String>> = plain
        .errors()
        .iter()
        .map(|error| error.paths.iter().map(|p| quote!(#p).to_string()).collect())
        .collect();
    assert_eq!(
        paths,
        [
            vec!["core :: fmt :: Debug"],
            vec!["std :: vec :: Vec"],
            vec!["Rel"],
            vec!["std :: fmt :: Formatter"]
        ]
    );
    // The rule alone, after the analysis and as the config, leaves the same.
    assert_eq!(
        causes(&plain.allow(Rules::new().std_crates())),
        ["Rel < '_ >"]
    );
    assert_eq!(
        causes(&analyze(&item, Rules::new().std_crates())),
        ["Rel < '_ >"]
    );
}

#[test]
fn allow_crate_agrees_where_every_path_of_an_error_is_allowed_or_not() {
    let item: ItemTrait = parse_quote!(
        trait Tr: crate::Sup {
            fn a(&self, x: crate::Foo<'_>);
            fn c(&self, x: Vec<crate::Foo<'_>>);
        }
    );
    let (before, after) = stages(&item, |c| c.primitive(), Rule::Crate);
    // `Vec` is not allowed: it is written as it is in `c`, at both stages, and `crate::Foo<'_>` in
    // it has no error.
    assert_eq!(causes(&before), ["Vec < crate :: Foo < '_ > >"]);
    assert_eq!(causes(&after), causes(&before));
}

#[test]
fn allow_path_agrees_where_every_path_of_an_error_is_allowed_or_not() {
    let item: ItemTrait = parse_quote!(
        trait Tr: my::Sup {
            fn a(&self, x: my::Foo<u8>);
            fn c(&self, x: Wrap<my::Foo<'_>>);
        }
    );
    let (before, after) = stages(&item, |c| c.primitive(), Rule::Path(parse_quote!(my)));
    assert_eq!(causes(&before), ["Wrap < my :: Foo < '_ > >"]);
    assert_eq!(causes(&after), causes(&before));
    // `my::Foo<'_>` in it is an error of its own, so allowing `Wrap` only is not enough, and the
    // rules add up in any order: the second one drops the error, as one config with both does.
    let plain = analyze(&item, Rules::new().primitive());
    let texts = |error: &ErrorItem| -> Vec<String> {
        error.paths.iter().map(|p| quote!(#p).to_string()).collect()
    };
    // The error of the supertrait `my::Sup` is dropped by the rule too: it has no other path.
    let wrapped = |result: &LeakerResult| -> Option<Vec<String>> {
        let error = result
            .errors()
            .iter()
            .find(|e| causes_text(e).starts_with("Wrap"))?;
        Some(texts(error))
    };
    assert_eq!(plain.errors().len(), 3);
    assert_eq!(wrapped(&plain).unwrap(), ["Wrap"]);
    assert_eq!(wrapped(&after).unwrap(), ["Wrap"]);
    let wrap_only = plain.clone().allow(starts_with(parse_quote!(Wrap)));
    assert_eq!(
        causes_text_of(wrap_only.errors()),
        ["my :: Sup", "my :: Foo < '_ >"]
    );
    let wrap_only = wrap_only.allow(starts_with(parse_quote!(my)));
    assert!(wrap_only.errors().is_empty());
    let both = plain
        .clone()
        .allow(starts_with(parse_quote!(my)))
        .allow(starts_with(parse_quote!(Wrap)));
    assert!(both.errors().is_empty());
    let config = Rules::new()
        .primitive()
        .or_if(starts_with(parse_quote!(my)))
        .or_if(starts_with(parse_quote!(Wrap)));
    assert!(analyze(&item, config).errors().is_empty());
}

#[test]
fn an_absolute_path_is_no_path_of_an_error() {
    // `::p::W` is context-free: an optional entry, never a path that a rule must allow.
    let item: ItemTrait = parse_quote!(
        trait Tr {
            fn f(&self, x: ::p::W<'_, dyn RelTr>);
        }
    );
    let result = analyze(&item, Rules::new());
    assert_eq!(result.errors().len(), 1);
    let paths: Vec<String> = result.errors()[0]
        .paths
        .iter()
        .map(|p| quote!(#p).to_string())
        .collect();
    assert_eq!(paths, ["RelTr"]);
    // So a path rule that allows the trait drops the error, and `Rules::prelude()` does not.
    let dropped = result.clone().allow(starts_with(parse_quote!(RelTr)));
    assert!(dropped.errors().is_empty());
    let kept = result
        .allow(Rules::new().prelude())
        .allow(Rules::new().primitive())
        .allow(Rules::new().std_crates())
        .allow(Rules::new().crate_paths());
    assert_eq!(kept.errors().len(), 1);
}

// ── values and defaults ──────────────────────────────────────────────────────────────────────

/// An item for each place that `Rules::values()` and `Rules::defaults()` read.
fn places() -> Vec<(&'static str, ItemTrait)> {
    vec![
        (
            "a default value and a default body of a trait",
            parse_quote!(
                trait Tr {
                    const N: usize = inner::N;
                    fn f(&self) {
                        let _ = inner::f();
                    }
                }
            ),
        ),
        (
            "the default of a parameter",
            parse_quote!(
                trait Tr<U = Rel<'_>> {}
            ),
        ),
        (
            "a type and a default",
            parse_quote!(
                trait Tr<U = Rel<'_>> {
                    fn f(&self, x: Wrap<'_>);
                }
            ),
        ),
        (
            "nothing",
            parse_quote!(
                trait Tr {
                    fn f(&self, x: Wrap<'_>);
                }
            ),
        ),
    ]
}

/// The rules of the Leaker that reads values (`check_values`).
fn reading_values() -> Rules<impl Allow + Clone> {
    Rules::new().primitive().prelude()
}

#[test]
fn the_values_and_defaults_rules_give_the_same_errors_and_entries_at_both_stages() {
    // The config is asked about each path of a value or of a default, so it writes them as they
    // are: no error is found. The result finds the errors and drops them; the entries are equal.
    let mut dropped = 0;
    for (name, item) in places() {
        for rule in [Rule::Values, Rule::Default] {
            let before = rule.analyze_with(&item, reading_values(), true);
            let after = analyze_values(&item, reading_values());
            let all = after.errors().len();
            let after = rule.on_result(after);
            dropped += all - after.errors().len();
            let ser = |r: &LeakerResult| {
                r.errors()
                    .iter()
                    .map(|e| e.to_token_stream().to_string())
                    .collect::<Vec<_>>()
            };
            assert_eq!(ser(&before), ser(&after), "{}", name);
            assert_eq!(entries(&before), entries(&after), "{}", name);
            // The entries and their contexts are the same. The parts are not: the result has the
            // part that each error it dropped was about, which the config never found.
            let seen = |result: LeakerResult| {
                let referrer = result.recovered();
                let found: Vec<_> = referrer
                    .iter()
                    .map(|(ix, ty, context)| (ix, ty.clone(), context.clone()))
                    .collect();
                found
            };
            assert_eq!(seen(before), seen(after), "{}", name);
        }
    }
    assert!(dropped > 0);
}

#[test]
fn a_trait_default_body_is_a_value_and_a_default() {
    let item: ItemTrait = parse_quote!(
        trait Tr {
            fn f(&self) {
                let _ = inner::f();
            }
        }
    );
    let result = analyze_values(&item, reading_values());
    let error = &result.errors()[0];
    assert!(error.place.in_value && error.place.in_default);
    // An impl's value is a value, not a default.
    let item: ItemImpl = parse_quote!(impl ::p::Tr for X { fn f(&self) { let _ = inner::f(); } });
    let mut leaker = Leaker::with_rule(reading_values());
    leaker.check_values = true;
    leaker.intern_impl(&item);
    let result = leaker.result();
    let error = &result.errors()[0];
    assert!(error.place.in_value && !error.place.in_default);
    // The default of a parameter is a default, not a value.
    let item: ItemTrait = parse_quote!(
        trait Tr<U = Rel<'_>> {}
    );
    let result = analyze_values(&item, reading_values());
    let error = &result.errors()[0];
    assert!(!error.place.in_value && error.place.in_default);
    // `Rules::values()` drops the first two and leaves the third, `Rules::defaults()` the reverse.
    let values = result.clone().allow(Rules::new().values());
    assert_eq!(values.errors().len(), 1);
    let defaults = result.allow(Rules::new().defaults());
    assert!(defaults.errors().is_empty());
}

// ── what the result cannot do ────────────────────────────────────────────────────────────────

#[test]
fn an_error_for_the_head_of_a_type_is_dropped_by_the_rule_and_leaves_what_is_in_it_leaked() {
    // With `crate` allowed, the config leaves `crate::Foo` as written, and `Rel`, a bare
    // identifier, is leaked alone. The result writes `crate::Foo` as it is, in an error whose path
    // the rule allows, and leaks `Rel` as well: the same errors and the same entries.
    let item: ItemTrait = parse_quote!(
        trait Tr {
            fn c(&self, x: crate::Foo<'_, Rel>);
        }
    );
    let (before, after) = stages(&item, |c| c.primitive(), Rule::Crate);
    assert!(before.errors().is_empty());
    assert_eq!(entries(&before), ["Rel"]);
    assert!(after.errors().is_empty());
    assert_eq!(entries(&after), ["Rel"]);
    let plain = analyze(&item, Rules::new().primitive());
    assert_eq!(causes(&plain), ["crate :: Foo < '_ , Rel >"]);
}

/// The result is the analysis without the option, whose errors it filters. The property test
/// (`TYPE_LEAK_PROP_FEATURES=allow_stages`) counts them: 27 of 1255 items of seed 1 leave another
/// number of errors, for the reasons below. It also builds the copies. The one after the result
/// fails to build where the one after the config builds for (B): `Self::A` is unprovable (E0277,
/// the documented limitation of `LeakerResult::allow()`). Where (B) meets (C), in `trait Tr<T:
/// ::p::BT1<String>> { type A0; fn m<X>(&self) where X: Into<Self::A0>; }`, the relative `String`
/// stays an entry (the config leaves it as written), and the relative `Into<Self::A0>`
/// is copied whole: the unprovable `Self::A0` in it has an error of its own, which
/// `Rules::prelude()` leaves (it left no error before, and the copy failed with E0277 all the
/// same).
#[test]
fn a_bound_with_a_relative_path_is_copied_whole_and_stays_an_error() {
    // (A) With the prelude allowed, `Fn` is context-free and `R0` is leaked as an entry. Without
    // it, the relative `Fn(R0) -> u8` is copied as written whole, `R0` in it too, so the paths of
    // the error are `Fn` and `R0`: the rule removes `Fn`, and `R0` stays an error.
    let item: ItemTrait = parse_quote!(
        trait Tr {
            fn f<X: Fn(R0) -> u8>(&self, x: X);
        }
    );
    let (before, after) = stages(&item, |c| c.primitive(), Rule::Prelude);
    assert!(before.errors().is_empty());
    assert_eq!(entries(&before), ["R0"]);
    assert_eq!(causes(&after), ["Fn (R0) -> u8"]);
    let paths: Vec<String> = after.errors()[0]
        .paths
        .iter()
        .map(|p| quote!(#p).to_string())
        .collect();
    assert_eq!(paths, ["R0"]);
    assert!(entries(&after).is_empty());
}

#[test]
fn a_prelude_type_that_stays_leaked_makes_a_projection_unprovable_after_the_analysis() {
    // (B) Without the prelude `String` is leaked, a bound that names a leaked type makes `Self::A`
    // in a predicate unprovable ([`Reason::UnprovableBound`]), and the error has no path to allow.
    let item: ItemTrait = parse_quote!(
        trait Tr<T: ::core::ops::Fn(String) -> u8> {
            type A;
            fn m<X>(&self)
            where
                X: ::core::convert::Into<Self::A>;
        }
    );
    let (before, after) = stages(&item, |c| c.primitive(), Rule::Prelude);
    assert!(before.errors().is_empty());
    assert_eq!(causes(&after), ["Self :: A"]);
    assert_eq!(after.errors()[0].reason, Reason::UnprovableBound);
    assert!(after.errors()[0].paths.is_empty());
}

#[test]
fn an_error_inside_a_bound_that_is_written_as_it_is_stays_when_it_is_dropped() {
    // (C) The relative `Fn(&Self::A0) -> u8` is written as it is, and what is in it is analyzed:
    // the config, where `Fn` is context-free, finds that `Self::A0` is unprovable beside `I1<T>`.
    // The analysis without the prelude finds it too, an error after that of the bound, which
    // `Rules::prelude()` leaves. (Neither copy builds, though: both fail with E0275.)
    let item: ItemTrait = parse_quote!(
        trait Tr<T: Fn(&Self::A0) -> u8> {
            type A0;
            fn m(&self)
            where
                I1<T>: ::core::default::Default;
        }
    );
    let (before, after) = stages(&item, |c| c.primitive(), Rule::Prelude);
    // The same two errors, in the same order.
    assert_eq!(causes(&before), ["I1 < T >", "Self :: A0"]);
    assert_eq!(causes(&after), causes(&before));
    let plain = analyze(&item, Rules::new().primitive());
    assert_eq!(
        causes(&plain),
        ["Fn (& Self :: A0) -> u8", "I1 < T >", "Self :: A0"]
    );
    assert_eq!(plain.errors()[2].reason, Reason::UnprovableBound);
}

#[test]
fn an_entry_that_the_config_does_not_make_and_a_bound_written_as_it_is_leave_an_error_and_a_copy_that_fails(
) {
    // (D) (B) meets (C). The crate `::p` is allowed. The config allows `String` and `Into`, so
    // `::p::BT1<String>` is context-free and `Self::A0` is provable. Without them `String` is a
    // relative name, which `Rules::prelude()` drops the error of and leaves as an entry. A bound that
    // names a leaked type makes `Self::A0` unprovable in a predicate, but `Into<Self::A0>` is relative
    // and written as it is: the analysis finds the problem inside it, an error of its own, which
    // `Rules::prelude()` leaves. The copy fails to build in the Referrer's crate (E0277:
    // `T: BT1<String>` is not satisfied by the host of the trait), as the error says it does.
    let item: ItemTrait = parse_quote!(
        trait Tr<T: ::p::BT1<String>> {
            type A0;
            fn m<X>(&self)
            where
                X: Into<Self::A0>;
        }
    );
    let (before, after) = stages(
        &item,
        |c| c.primitive().or_if(starts_with(parse_quote!(::p))),
        Rule::Prelude,
    );
    assert!(before.errors().is_empty());
    assert_eq!(causes(&after), ["Self :: A0"]);
    assert_eq!(after.errors()[0].reason, Reason::UnprovableBound);
    assert!(after.errors()[0].paths.is_empty());
    assert!(entries(&before).is_empty());
    assert_eq!(entries(&after), ["String"]);
    // With an absolute `Into` the result reports the same error in the first place (B).
    let item: ItemTrait = parse_quote!(
        trait Tr<T: ::p::BT1<String>> {
            type A0;
            fn m<X>(&self)
            where
                X: ::core::convert::Into<Self::A0>;
        }
    );
    let (before, after) = stages(
        &item,
        |c| c.primitive().or_if(starts_with(parse_quote!(::p))),
        Rule::Prelude,
    );
    assert!(before.errors().is_empty());
    assert_eq!(causes(&after), ["Self :: A0"]);
    assert_eq!(after.errors()[0].reason, Reason::UnprovableBound);
}

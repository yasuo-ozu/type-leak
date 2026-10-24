//! The `allow*()` operations of `LeakerConfig` and `LeakerResult` mean the same thing.
//!
//! `Leaker::from_config(config.allow_X()).result()` and
//! `Leaker::from_config(config).result().allow_X()`:
//!
//! - the cause, value and default rules give the same errors and an equal Referrer
//!   (`tests/error_items.rs` checks every trigger of the errors; the cases of a value and of a default
//!   are here);
//! - the path rules (`allow_path()`, `allow_primitive()`, `allow_prelude()`, `allow_std_crates()`,
//!   `allow_crate()`) make a path context-free during the analysis, which a result cannot do. So
//!   the errors that remain are those of the paths that are not allowed, and a cause can be wider
//!   after the analysis (`Vec<Rel<'_>>` where the config reports `Rel<'_>`), the Referrer may keep
//!   entries that the config does not make (`Vec<u8>`), and an error can remain that the config does
//!   not have (the last test).

mod common;

use common::Rule;
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

/// The entries of the Referrer, as text.
fn entries(result: &LeakerResult) -> Vec<String> {
    let referrer = result.clone().into_referrer();
    referrer.iter().map(|ty| quote!(#ty).to_string()).collect()
}

fn analyze(item: &ItemTrait, config: LeakerConfig) -> LeakerResult {
    let mut leaker = Leaker::from_config(config);
    leaker.intern_trait(item, true);
    leaker.result()
}

/// The result of `item` with `rule` given to the config (`before`) and to its result (`after`), the
/// other options being those of `base`.
fn stages(
    item: &ItemTrait,
    base: impl Fn(&mut LeakerConfig),
    rule: Rule,
) -> (LeakerResult, LeakerResult) {
    let mut config = LeakerConfig::new();
    base(&mut config);
    let mut allowed = config.clone();
    rule.on_config(&mut allowed);
    let before = analyze(item, allowed);
    let mut after = analyze(item, config);
    rule.on_result(&mut after);
    (before, after)
}

#[test]
fn allow_prelude_reports_the_part_before_and_the_whole_after() {
    let item: ItemTrait = parse_quote!(
        trait Tr<T: Clone> {
            fn a(&self, x: Vec<u8>);
            fn c(&self, x: Vec<Rel<'_>>);
        }
    );
    let (before, after) = stages(
        &item,
        |c| {
            c.allow_primitive();
        },
        Rule::Prelude,
    );
    // `Clone` is dropped by both. The config does not leak `Vec`, so `Rel<'_>` is the error; the
    // result has `Vec<Rel<'_>>` and drops it only if `Rel` were allowed too.
    assert_eq!(causes(&before), ["Rel < '_ >"]);
    assert_eq!(causes(&after), ["Vec < Rel < '_ > >"]);
    assert_eq!(before.errors().len(), after.errors().len());
    // The config leaves `Vec<u8>` and `Vec<Rel<'_>>` as written. After the analysis the Referrer
    // still has the entry `Vec<u8>`, which a Referrer replaces as it does any entry.
    assert!(entries(&before).is_empty());
    assert_eq!(entries(&after), ["Vec < u8 >"]);
    // The result with the same rule once more, and with the config's, is the same result.
    let mut twice = after.clone();
    twice.allow_prelude();
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
    let (before, after) = stages(&item, |_| (), Rule::Primitive);
    // The closure's `a` is relative, so the array is still an error, with `u8` allowed. After the
    // analysis the cause is `u8`, the first path of the array, and `a` is the second path: the rule
    // removes `u8` from the paths of the error and leaves `a`.
    assert_eq!(causes(&before), ["a"]);
    assert_eq!(causes(&after), ["u8"]);
    assert_eq!(before.errors().len(), after.errors().len());
    let texts = |error: &ErrorItem| -> Vec<String> {
        error.paths.iter().map(|p| quote!(#p).to_string()).collect()
    };
    assert_eq!(texts(&after.errors()[0]), ["a"]);
    assert_eq!(
        texts(&analyze(&item, LeakerConfig::new()).errors()[1]),
        ["u8", "a"]
    );
}

#[test]
fn allow_std_crates_reports_the_part_before_and_the_whole_after() {
    let item: ItemTrait = parse_quote!(
        trait Tr: core::fmt::Debug {
            fn s(&self, x: std::vec::Vec<Rel<'_>>);
            fn t(&self, x: std::fmt::Formatter<'_>);
        }
    );
    let (before, after) = stages(
        &item,
        |c| {
            c.allow_primitive();
        },
        Rule::StdCrates,
    );
    assert_eq!(causes(&before), ["Rel < '_ >"]);
    assert_eq!(causes(&after), ["std :: vec :: Vec < Rel < '_ > >"]);
    assert_eq!(before.errors().len(), after.errors().len());
}

#[test]
fn allow_crate_agrees_where_every_path_of_an_error_is_allowed_or_not() {
    let item: ItemTrait = parse_quote!(
        trait Tr: crate::Sup {
            fn a(&self, x: crate::Foo<'_>);
            fn c(&self, x: Vec<crate::Foo<'_>>);
        }
    );
    let (before, after) = stages(
        &item,
        |c| {
            c.allow_primitive();
        },
        Rule::Crate,
    );
    // `Vec` is not allowed: it is leaked whole in `c` and cannot be, at both stages.
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
    let (before, after) = stages(
        &item,
        |c| {
            c.allow_primitive();
        },
        Rule::Path(parse_quote!(my)),
    );
    assert_eq!(causes(&before), ["Wrap < my :: Foo < '_ > >"]);
    assert_eq!(causes(&after), causes(&before));
    // `my::Foo` is a path of the cause too, so allowing `Wrap` only is not enough, and the rules
    // add up in any order: the second one drops the error, as one config with both does.
    let plain = analyze(&item, {
        let mut config = LeakerConfig::new();
        config.allow_primitive();
        config
    });
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
    assert_eq!(plain.errors().len(), 2);
    assert_eq!(wrapped(&plain).unwrap(), ["Wrap", "my :: Foo"]);
    assert_eq!(wrapped(&after).unwrap(), ["Wrap"]);
    let mut wrap_only = plain.clone();
    wrap_only.allow_path(parse_quote!(Wrap));
    assert_eq!(wrapped(&wrap_only).unwrap(), ["my :: Foo"]);
    wrap_only.allow_path(parse_quote!(my));
    assert!(wrap_only.errors().is_empty());
    let mut both = plain.clone();
    both.allow_path(parse_quote!(my))
        .allow_path(parse_quote!(Wrap));
    assert!(both.errors().is_empty());
    let mut config = LeakerConfig::new();
    config
        .allow_primitive()
        .allow_path(parse_quote!(my))
        .allow_path(parse_quote!(Wrap));
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
    let result = analyze(&item, LeakerConfig::new());
    assert_eq!(result.errors().len(), 1);
    let paths: Vec<String> = result.errors()[0]
        .paths
        .iter()
        .map(|p| quote!(#p).to_string())
        .collect();
    assert_eq!(paths, ["RelTr"]);
    // So a path rule that allows the trait drops the error, and `allow_prelude()` does not.
    let mut dropped = result.clone();
    dropped.allow_path(parse_quote!(RelTr));
    assert!(dropped.errors().is_empty());
    let mut kept = result;
    kept.allow_prelude()
        .allow_primitive()
        .allow_std_crates()
        .allow_crate();
    assert_eq!(kept.errors().len(), 1);
}

#[test]
fn a_cause_that_is_no_path_has_none_and_no_path_rule_drops_it() {
    let item: ItemTrait = parse_quote!(
        trait Tr<T = Wrap<Self>> {}
    );
    let result = analyze(&item, LeakerConfig::new());
    assert_eq!(result.errors().len(), 1);
    let mut after = result.clone();
    after
        .allow_prelude()
        .allow_primitive()
        .allow_std_crates()
        .allow_crate();
    assert_eq!(after.errors().len(), 1);
}

// ── values and defaults ──────────────────────────────────────────────────────────────────────

/// An item for each place that `allow_values()` and `allow_default()` read.
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

fn reading_values() -> LeakerConfig {
    let mut config = LeakerConfig::new();
    config.allow_primitive().allow_prelude().check_values();
    config
}

#[test]
fn allow_values_and_allow_default_give_the_same_result_at_both_stages() {
    let mut dropped = 0;
    for (name, item) in places() {
        for rule in [Rule::Values, Rule::Default] {
            let before = {
                let mut config = reading_values();
                rule.on_config(&mut config);
                analyze(&item, config)
            };
            let mut after = analyze(&item, reading_values());
            let all = after.errors().len();
            rule.on_result(&mut after);
            dropped += all - after.errors().len();
            let ser = |r: &LeakerResult| {
                r.errors()
                    .iter()
                    .map(|e| e.to_token_stream().to_string())
                    .collect::<Vec<_>>()
            };
            assert_eq!(ser(&before), ser(&after), "{}", name);
            assert_eq!(before.into_referrer(), after.into_referrer(), "{}", name);
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
    let result = analyze(&item, reading_values());
    let error = &result.errors()[0];
    assert!(error.in_value && error.in_default);
    // An impl's value is a value, not a default.
    let item: ItemImpl = parse_quote!(impl ::p::Tr for X { fn f(&self) { let _ = inner::f(); } });
    let mut leaker = Leaker::from_config(reading_values());
    leaker.intern_impl(&item, true);
    let result = leaker.result();
    let error = &result.errors()[0];
    assert!(error.in_value && !error.in_default);
    // The default of a parameter is a default, not a value.
    let item: ItemTrait = parse_quote!(
        trait Tr<U = Rel<'_>> {}
    );
    let result = analyze(&item, reading_values());
    let error = &result.errors()[0];
    assert!(!error.in_value && error.in_default);
    // `allow_values()` drops the first two and leaves the third, `allow_default()` the reverse.
    let mut values = result.clone();
    values.allow_values();
    assert_eq!(values.errors().len(), 1);
    let mut defaults = result;
    defaults.allow_default();
    assert!(defaults.errors().is_empty());
}

// ── what the result cannot do ────────────────────────────────────────────────────────────────

#[test]
fn an_error_can_remain_after_the_analysis_that_the_config_does_not_have() {
    // With `crate` allowed, the config leaves `crate::Foo` as written, and `Rel`, a bare
    // identifier, is leaked alone. The result has leaked `crate::Foo<'_, Rel>` whole as a part
    // (nothing in it is leaked), so `Rel` is written as it is in the copy: it is a path that the
    // rule does not allow, and the error stays.
    let item: ItemTrait = parse_quote!(
        trait Tr {
            fn c(&self, x: crate::Foo<'_, Rel>);
        }
    );
    let (before, after) = stages(
        &item,
        |c| {
            c.allow_primitive();
        },
        Rule::Crate,
    );
    assert!(before.errors().is_empty());
    assert_eq!(entries(&before), ["Rel"]);
    assert_eq!(causes(&after), ["crate :: Foo < '_ , Rel >"]);
    assert!(entries(&after).is_empty());
}

/// The result is the analysis without the option, whose errors it filters. The property test
/// (`TYPE_LEAK_PROP_FEATURES=allow_stages`) counts them: 27 of 1255 items of seed 1 leave another
/// number of errors, for the three reasons below.
#[test]
fn a_bound_with_a_relative_path_in_its_arguments_is_copied_whole_and_stays_an_error() {
    // (A) With the prelude allowed, `Fn` is context-free and `R0` is leaked as an entry. Without
    // it, the relative `Fn(R0) -> u8` is copied as written whole, `R0` in it too, so the paths of
    // the error are `Fn` and `R0`: the rule removes `Fn`, and `R0` stays an error.
    let item: ItemTrait = parse_quote!(
        trait Tr {
            fn f<X: Fn(R0) -> u8>(&self, x: X);
        }
    );
    let (before, after) = stages(
        &item,
        |c| {
            c.allow_primitive();
        },
        Rule::Prelude,
    );
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
    let (before, after) = stages(
        &item,
        |c| {
            c.allow_primitive();
        },
        Rule::Prelude,
    );
    assert!(before.errors().is_empty());
    assert_eq!(causes(&after), ["Self :: A"]);
    assert_eq!(after.errors()[0].reason, Reason::UnprovableBound);
    assert!(after.errors()[0].paths.is_empty());
}

#[test]
fn an_error_inside_a_bound_that_the_analysis_copied_whole_is_not_found_after_it() {
    // (C) The relative `Fn(&Self::A0) -> u8` is copied whole, so what is in it is not analyzed:
    // the config, where `Fn` is context-free, finds that `Self::A0` is unprovable beside `I1<T>`.
    let item: ItemTrait = parse_quote!(
        trait Tr<T: Fn(&Self::A0) -> u8> {
            type A0;
            fn m(&self)
            where
                I1<T>: ::core::default::Default;
        }
    );
    let (before, after) = stages(
        &item,
        |c| {
            c.allow_primitive();
        },
        Rule::Prelude,
    );
    assert_eq!(causes(&before), ["I1 < T >", "Self :: A0"]);
    assert_eq!(causes(&after), ["I1 < T >"]);
}

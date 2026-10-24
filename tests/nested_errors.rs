//! The errors inside a part that is copied whole (`ErrorItem::nested`).
//!
//! A bound with a relative path, a type that cannot be leaked and a path of a value are copied as
//! written, whole, and nothing in them is analyzed. The analysis also looks inside them as if the
//! paths of their error were allowed, and keeps what it finds in the error. When a rule of `allow*()`
//! drops the error, the nested errors take its place: the part is copied as written, so what is
//! wrong inside it is real whichever rule accepted it.
//!
//! - the reduced case of `tests/allow_stages.rs` (D): `X: Into<Self::A0>` has `Self::A0` nested;
//! - every rule brings it up, on a config and on a result alike, and the rules add up;
//! - a part with nothing wrong inside has none, and neither does `Vec<Rel<'_>>`;
//! - the nesting is recursive, in the analysis (a value) and in what the rules do with it;
//! - the serialization writes `nested [..]` only if it is not empty, and reporting never shows it.

mod common;

use common::{allow_every_cause, Rule};
use syn::parse::Parser;
use syn::*;
use template_quote::{quote, ToTokens};
use type_leak::*;

fn analyze(item: &ItemTrait, config: LeakerConfig) -> LeakerResult {
    let mut leaker = Leaker::from_config(config);
    leaker.intern_trait(item, true);
    leaker.result()
}

/// The reduced case: the trait of `X: Into<Self::A0>` is relative, so the bound is copied whole,
/// and `T: ::p::BT1<String>` makes `Self::A0` unprovable in a predicate once `Into` is allowed.
fn reduced() -> ItemTrait {
    parse_quote!(
        trait Tr<T: ::p::BT1<String>> {
            type A0;
            fn m<X>(&self)
            where
                X: Into<Self::A0>;
        }
    )
}

fn text(tokens: &impl ToTokens) -> String {
    tokens.to_token_stream().to_string()
}

fn causes(errors: &[ErrorItem]) -> Vec<String> {
    errors.iter().map(|error| text(&error.cause)).collect()
}

fn paths(error: &ErrorItem) -> Vec<String> {
    error.paths.iter().map(text).collect()
}

/// The shape of the errors and of what they have nested: `cause [reason] { nested }`.
fn shape(errors: &[ErrorItem]) -> String {
    errors
        .iter()
        .map(|error| {
            let nested = if error.nested.is_empty() {
                String::new()
            } else {
                format!(" {{ {} }}", shape(&error.nested))
            };
            format!("{} [{:?}]{}", text(&error.cause), error.reason, nested)
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn into_cause() -> Cause {
    Cause::Trait(parse_quote!(Into<Self::A0>))
}

#[test]
fn a_bound_copied_whole_has_the_error_inside_it_nested() {
    let result = analyze(&reduced(), LeakerConfig::new());
    // `String` is an entry, and the bound is one error whose path is `Into`.
    assert_eq!(result.errors().len(), 1);
    let error = &result.errors()[0];
    assert_eq!(error.cause, into_cause());
    assert_eq!(error.reason, Reason::RelativePath);
    assert_eq!(paths(error), ["Into"]);
    // With `Into` allowed the analysis finds `Self::A0` unprovable: the nested error.
    assert_eq!(error.nested.len(), 1);
    let nested = &error.nested[0];
    assert_eq!(nested.cause, Cause::Type(parse_quote!(Self::A0)));
    assert_eq!(nested.reason, Reason::UnprovableBound);
    assert!(nested.paths.is_empty() && nested.nested.is_empty());
    assert!(!nested.in_value && !nested.in_default);
    // The same error as the analysis with an absolute `Into` reports in the first place.
    let absolute: ItemTrait = parse_quote!(
        trait Tr<T: ::p::BT1<String>> {
            type A0;
            fn m<X>(&self)
            where
                X: ::core::convert::Into<Self::A0>;
        }
    );
    let direct = analyze(&absolute, LeakerConfig::new());
    assert_eq!(direct.errors().len(), 1);
    assert_eq!(text(&direct.errors()[0]), text(nested));
    // Looking inside changes nothing else: the Referrer is the one of the absolute bound but for
    // the part.
    assert_eq!(
        result.clone().into_referrer().iter().count(),
        direct.into_referrer().iter().count()
    );
}

#[test]
fn every_rule_that_drops_the_error_brings_the_nested_error_up() {
    // The cause, the path, and the prelude (which has `Into`) drop the bound; the entry `String` is
    // there all the same, and the error is the one found inside the bound.
    let mut by_cause = analyze(&reduced(), LeakerConfig::new());
    by_cause.allow(into_cause());
    let mut by_path = analyze(&reduced(), LeakerConfig::new());
    by_path.allow_path(parse_quote!(Into));
    let mut by_prelude = analyze(&reduced(), LeakerConfig::new());
    by_prelude.allow_prelude();
    for left in [&by_cause, &by_path, &by_prelude] {
        assert_eq!(shape(left.errors()), "Self :: A0 [UnprovableBound]");
        assert!(left.errors()[0].paths.is_empty());
    }
    // A rule that drops nothing leaves the bound, with the nested error still in it.
    let mut other = analyze(&reduced(), LeakerConfig::new());
    other.allow_primitive().allow_std_crates();
    assert_eq!(
        shape(other.errors()),
        "Into < Self :: A0 > [RelativePath] { Self :: A0 [UnprovableBound] }"
    );
    // A config gives the same: the rule is applied when the analysis finishes.
    let mut config = LeakerConfig::new();
    config.allow(into_cause());
    assert_eq!(
        shape(analyze(&reduced(), config).errors()),
        "Self :: A0 [UnprovableBound]"
    );
    // A config that allows `Into` makes it context-free: the analysis finds the same error.
    let mut config = LeakerConfig::new();
    config.allow_path(parse_quote!(Into));
    assert_eq!(
        shape(analyze(&reduced(), config).errors()),
        "Self :: A0 [UnprovableBound]"
    );
    // Allowing the cause of that error drops it, and nothing else is left: `allow_every_cause()`.
    let mut all = by_cause.clone();
    assert_eq!(allow_every_cause(&mut all), 1);
    assert!(all.errors().is_empty());
}

#[test]
fn a_rule_that_drops_the_nested_error_first_leaves_nothing_when_the_bound_is_dropped() {
    // The rules add up in any order: what a rule accepts is accepted inside a bound that stays, so
    // it does not come up when the bound goes.
    let mut result = analyze(&reduced(), LeakerConfig::new());
    result.allow(Cause::Type(parse_quote!(Self::A0)));
    assert_eq!(shape(result.errors()), "Into < Self :: A0 > [RelativePath]");
    assert!(result.errors()[0].nested.is_empty());
    result.allow_prelude();
    assert!(result.errors().is_empty());
    let mut other = analyze(&reduced(), LeakerConfig::new());
    other
        .allow_prelude()
        .allow(Cause::Type(parse_quote!(Self::A0)));
    assert!(other.errors().is_empty());
    // And in one set, as a config has them.
    let mut config = LeakerConfig::new();
    config
        .allow_prelude()
        .allow(Cause::Type(parse_quote!(Self::A0)));
    assert!(analyze(&reduced(), config).errors().is_empty());
}

#[test]
fn the_cause_value_and_default_rules_are_the_same_before_and_after_the_analysis() {
    // The place rules drop the error all the same, and the nested error comes up at both stages.
    let item: ItemTrait = parse_quote!(
        trait Tr<T: ::p::BT1<String>> {
            type A0;
            fn m<X>(&self)
            where
                X: Into<Self::A0>;
            fn d(&self) {
                let _ = inner::f::<Rel<'_>>();
            }
        }
    );
    let mut base = LeakerConfig::new();
    base.check_values();
    for rule in [
        Rule::Cause(into_cause()),
        Rule::Values,
        Rule::Default,
        Rule::Path(parse_quote!(Into)),
    ] {
        let mut config = base.clone();
        rule.on_config(&mut config);
        let before = analyze(&item, config);
        let mut after = analyze(&item, base.clone());
        rule.on_result(&mut after);
        let both = |result: &LeakerResult| {
            result
                .errors()
                .iter()
                .map(|error| error.to_token_stream().to_string())
                .collect::<Vec<_>>()
        };
        match rule {
            // The config makes `Into` context-free in the analysis, and finds the errors in the
            // order of the analysis; the result puts the nested one where the bound was.
            Rule::Path(_) => {
                let sorted = |result: &LeakerResult| {
                    let mut causes = causes(result.errors());
                    causes.sort();
                    causes
                };
                assert_eq!(sorted(&before), sorted(&after), "path rule");
            }
            _ => assert_eq!(both(&before), both(&after)),
        }
    }
    // `allow_values()` drops the error in the body and what is nested in it, which is in the body
    // too; the bound stays. Allowing the cause of the path brings `Rel<'_>` up.
    let mut result = analyze(&item, base.clone());
    result.allow_values();
    assert_eq!(
        shape(result.errors()),
        "Into < Self :: A0 > [RelativePath] { Self :: A0 [UnprovableBound] }"
    );
    let mut result = analyze(&item, base);
    result.allow(Cause::Expr(parse_quote!(inner::f::<Rel<'_>>)));
    assert_eq!(
        shape(result.errors()),
        "Into < Self :: A0 > [RelativePath] { Self :: A0 [UnprovableBound] }, Rel < '_ > [RelativePath]"
    );
    assert!(result.errors()[1].in_value);
}

#[test]
fn a_part_with_nothing_wrong_inside_has_no_nested_errors() {
    let item: ItemTrait = parse_quote!(
        trait Tr {
            fn m<X>(&self)
            where
                X: Rel<u8>;
            fn n<Y: Other>(&self, a: Vec<Rel<'_>>, b: MyType<'_>);
        }
    );
    let mut config = LeakerConfig::new();
    config.allow_primitive();
    let result = analyze(&item, config);
    // The bound, the other bound, a type with `Vec` and `Rel` among its paths, and one with `MyType`.
    assert_eq!(
        causes(result.errors()),
        ["Rel < u8 >", "Other", "Vec < Rel < '_ > >", "MyType < '_ >"]
    );
    for error in result.errors() {
        assert!(
            error.nested.is_empty(),
            "{}",
            shape(std::slice::from_ref(error))
        );
    }
    assert_eq!(paths(&result.errors()[2]), ["Vec", "Rel"]);
    // Dropping them leaves nothing.
    let mut all = result.clone();
    assert_eq!(allow_every_cause(&mut all), 1);
}

#[test]
fn an_error_without_a_part_or_a_path_has_none() {
    // `Self` in a module has no meaning, a lifetime has no path: nothing to allow, nothing to look in.
    let item: ItemTrait = parse_quote!(
        trait Tr<T = Wrap<Self>> {}
    );
    let result = analyze(&item, LeakerConfig::new());
    assert_eq!(result.errors().len(), 1);
    assert!(result.errors()[0].nested.is_empty());
    let mut leaker = Leaker::new();
    leaker.enter_generics(&Generics::default(), |view| {
        view.visit_type(&parse_quote!(Wrap<Self>));
    });
    let result = leaker.result();
    assert_eq!(result.errors().len(), 1);
    assert_eq!(result.errors()[0].reason, Reason::SelfWithoutMeaning);
    assert!(result.errors()[0].nested.is_empty());
}

#[test]
fn a_type_that_cannot_be_leaked_has_what_is_wrong_inside_it_nested() {
    // `::p::Tr<Rel<Self::A0>>` is leaked whole, and its entry `Rel<Self::A0>` is unprovable in a
    // predicate beside `T: ::p::BT1<String>`: copied as written, it has `Self::A0` in it, which is
    // unprovable too. The error of the part itself is found as well.
    let item: ItemTrait = parse_quote!(
        trait Tr<T: ::p::BT1<String>> {
            type A0;
            fn m<X>(&self)
            where
                X: ::p::Tr<Rel<Self::A0>>;
        }
    );
    let result = analyze(&item, LeakerConfig::new());
    assert_eq!(
        shape(result.errors()),
        "Rel < Self :: A0 > [UnprovableBound] { Self :: A0 [UnprovableBound] }, \
         Self :: A0 [UnprovableBound]"
    );
    assert_eq!(paths(&result.errors()[0]), ["Rel"]);
    // The config that allows `Rel` has the second error and not the first: the nested error is the
    // second one of the result, which comes up when the first is dropped, so there are two with the
    // result and one with the config.
    let mut result_after = result.clone();
    result_after.allow_path(parse_quote!(Rel));
    assert_eq!(
        shape(result_after.errors()),
        "Self :: A0 [UnprovableBound], Self :: A0 [UnprovableBound]"
    );
    let mut config = LeakerConfig::new();
    config.allow_path(parse_quote!(Rel));
    assert_eq!(
        shape(analyze(&item, config).errors()),
        "Self :: A0 [UnprovableBound]"
    );
}

#[test]
fn the_nesting_is_recursive_in_the_analysis() {
    // The path of a value is copied whole with the types in its arguments. Its error has only its
    // own path (`inner::f`), so allowing that finds `Rel<'_>`, a type that conflicts, and the
    // analysis of that one is a probe inside the first.
    let item: ItemTrait = parse_quote!(
        trait Tr {
            fn d(&self) {
                let _ = inner::f::<Rel<'_>, Wrap<Rel2<'_>>>();
            }
        }
    );
    let mut config = LeakerConfig::new();
    config.check_values();
    let result = analyze(&item, config);
    assert_eq!(result.errors().len(), 1);
    let error = &result.errors()[0];
    assert_eq!(paths(error), ["inner :: f"]);
    assert!(error.in_value);
    assert_eq!(
        shape(&error.nested),
        "Rel < '_ > [RelativePath], Wrap < Rel2 < '_ > > [RelativePath]"
    );
    // Each nested error has its own paths and its own place.
    assert_eq!(paths(&error.nested[0]), ["Rel"]);
    assert_eq!(paths(&error.nested[1]), ["Wrap", "Rel2"]);
    assert!(error.nested.iter().all(|nested| nested.in_value));
    // The rules go down one level at a time: dropping the path brings up both, and a path rule that
    // allows only `Rel` drops the first of them with it.
    let mut result = result;
    result.allow_path(parse_quote!(inner));
    assert_eq!(
        shape(result.errors()),
        "Rel < '_ > [RelativePath], Wrap < Rel2 < '_ > > [RelativePath]"
    );
    result.allow_path(parse_quote!(Rel));
    assert_eq!(
        shape(result.errors()),
        "Wrap < Rel2 < '_ > > [RelativePath]"
    );
    assert_eq!(paths(&result.errors()[0]), ["Wrap", "Rel2"]);
}

/// A result with three levels of nesting, written by hand: the analysis does not make one (a probe
/// allows all the paths of its part, so what it finds has none to allow), but the rules and the
/// serialization must not depend on it.
fn deep() -> LeakerResult {
    let referrer = Leaker::new().result().to_token_stream();
    let errors: proc_macro2::TokenStream = r#"
        { trait ( A1 ) RelativePath "a" [] paths [ A1 ]
          nested [
            { type ( B1 ) RelativePath "b" [] paths [ B1 , B2 ]
              nested [
                { type ( C1 ) UnprovableBound "c1" [] },
                { type ( C2 ) UnprovableBound "c2" [] paths [ Z ]
                  nested [ { type ( D1 ) Unreadable "d" [] value } ] }
              ] }
          ] }
        { type ( E1 ) RelativePath "e" [] }
    "#
    .parse()
    .unwrap();
    parse2(quote_all(referrer, errors)).unwrap()
}

fn quote_all(
    first: proc_macro2::TokenStream,
    second: proc_macro2::TokenStream,
) -> proc_macro2::TokenStream {
    let mut all = first;
    all.extend(second);
    all
}

#[test]
fn dropping_an_error_brings_up_its_nested_errors_down_to_the_last() {
    let result = deep();
    assert_eq!(
        shape(result.errors()),
        "A1 [RelativePath] { B1 [RelativePath] { C1 [UnprovableBound], C2 [UnprovableBound] \
         { D1 [Unreadable] } } }, E1 [RelativePath]"
    );
    // One level at a time.
    let mut result = result;
    result.allow(Cause::Trait(parse_quote!(A1)));
    assert_eq!(
        shape(result.errors()),
        "B1 [RelativePath] { C1 [UnprovableBound], C2 [UnprovableBound] { D1 [Unreadable] } }, \
         E1 [RelativePath]"
    );
    result.allow(Cause::Type(parse_quote!(B1)));
    assert_eq!(
        shape(result.errors()),
        "C1 [UnprovableBound], C2 [UnprovableBound] { D1 [Unreadable] }, E1 [RelativePath]"
    );
    result.allow(Cause::Type(parse_quote!(C2)));
    assert_eq!(
        shape(result.errors()),
        "C1 [UnprovableBound], D1 [Unreadable], E1 [RelativePath]"
    );
    // A rule drops a nested error that comes up, and the error it was in too: `allow_values()` is
    // the place of `D1`.
    result.allow_values();
    assert_eq!(
        shape(result.errors()),
        "C1 [UnprovableBound], E1 [RelativePath]"
    );
    // All of them: one round for each level.
    assert_eq!(allow_every_cause(&mut deep()), 4);
}

#[test]
fn a_rule_filters_the_nested_errors_of_an_error_that_stays() {
    // `allow_values()` drops `D1` inside `C2`, which stays, so it does not come up later.
    let mut result = deep();
    result.allow_values();
    assert_eq!(
        shape(result.errors()),
        "A1 [RelativePath] { B1 [RelativePath] { C1 [UnprovableBound], C2 [UnprovableBound] } }, \
         E1 [RelativePath]"
    );
    // And a path rule removes the paths it allows from the nested errors too, so rules add up: `Z`
    // is all that `C2` has, so it goes, and `B1` loses `B2` and then goes with its own path.
    result.allow_path(parse_quote!(Z));
    assert_eq!(
        shape(result.errors()),
        "A1 [RelativePath] { B1 [RelativePath] { C1 [UnprovableBound] } }, E1 [RelativePath]"
    );
    result.allow_path(parse_quote!(B2));
    assert_eq!(paths(&result.errors()[0].nested[0]), ["B1"]);
    result.allow_path(parse_quote!(B1));
    assert_eq!(
        shape(result.errors()),
        "A1 [RelativePath] { C1 [UnprovableBound] }, E1 [RelativePath]"
    );
    // In one set, in any order, as a config has them.
    let mut first = deep();
    first.allow_values();
    let mut second = first.clone();
    first
        .allow_path(parse_quote!(Z))
        .allow_path(parse_quote!(A1));
    second
        .allow_path(parse_quote!(A1))
        .allow_path(parse_quote!(Z));
    assert_eq!(shape(first.errors()), shape(second.errors()));
    assert_eq!(
        shape(first.errors()),
        "B1 [RelativePath] { C1 [UnprovableBound] }, E1 [RelativePath]"
    );
}

// ── serialization and reporting ─────────────────────────────────────────────────────────────

#[test]
fn nested_errors_are_written_after_the_paths_only_if_there_are_some() {
    let result = analyze(&reduced(), LeakerConfig::new());
    let error = &result.errors()[0];
    let tokens = error.to_token_stream().to_string();
    assert!(
        tokens.contains("paths [Into] nested [{ type (Self :: A0) UnprovableBound"),
        "{}",
        tokens
    );
    assert!(tokens.ends_with("}] }"), "{}", tokens);
    // The nested error is an error as any other, and has none of its own to write.
    let nested = error.nested[0].to_token_stream().to_string();
    assert!(!nested.contains("nested"), "{}", nested);
    // No nested errors, no word.
    let plain: ErrorItem = parse_str(r#"{ type ( A ) RelativePath "m" [] paths [ A ] }"#).unwrap();
    assert!(plain.nested.is_empty());
    assert_eq!(
        plain.to_token_stream().to_string(),
        r#"{ type (A) RelativePath "m" [] paths [A] }"#
    );
    // An empty list is the same as none.
    let empty: ErrorItem =
        parse_str(r#"{ type ( A ) RelativePath "m" [] nested [] paths [ A ] }"#).unwrap();
    assert_eq!(
        empty.to_token_stream().to_string(),
        plain.to_token_stream().to_string()
    );
    // A trailing comma is accepted, and anything but an item is refused.
    let two: ErrorItem = parse_str(
        r#"{ type ( A ) RelativePath "m" [] nested [ { type ( B ) Unreadable "n" [] } , { trait ( C ) Unreadable "o" [] } , ] }"#,
    )
    .unwrap();
    assert_eq!(causes(&two.nested), ["B", "C"]);
    assert!(parse_str::<ErrorItem>(r#"{ type ( A ) RelativePath "m" [] nested [ B ] }"#).is_err());
    assert!(parse_str::<ErrorItem>(r#"{ type ( A ) RelativePath "m" [] nested }"#).is_err());
}

#[test]
fn nested_errors_are_written_and_read_back_with_every_field() {
    let result = analyze(&reduced(), LeakerConfig::new());
    let error = &result.errors()[0];
    let tokens = error.to_token_stream();
    let read: ErrorItem = parse2(tokens.clone()).unwrap();
    assert_eq!(read.to_token_stream().to_string(), tokens.to_string());
    assert_eq!(read.cause, error.cause);
    assert_eq!(paths(&read), paths(error));
    assert_eq!(read.nested.len(), 1);
    let (nested, original) = (&read.nested[0], &error.nested[0]);
    assert_eq!(nested.cause, original.cause);
    assert_eq!(nested.reason, original.reason);
    assert_eq!(nested.message, original.message);
    assert_eq!(nested.notes.len(), original.notes.len());
    assert_eq!(nested.help_text(), original.help_text());
    // Three levels, and the place of a nested error, are kept.
    let deep = deep();
    let read: LeakerResult = parse2(deep.to_token_stream()).unwrap();
    assert_eq!(shape(read.errors()), shape(deep.errors()));
    assert_eq!(
        read.to_token_stream().to_string(),
        deep.to_token_stream().to_string()
    );
    let d1 = &read.errors()[0].nested[0].nested[1].nested[0];
    assert!(d1.in_value && !d1.in_default);
    assert_eq!(paths(&read.errors()[0].nested[0].nested[1]), ["Z"]);
    // A result with an error that has nested errors takes the rules after the trip as before it.
    let mut after = parse2::<LeakerResult>(result.to_token_stream()).unwrap();
    let mut before = result;
    after.allow_prelude();
    before.allow_prelude();
    assert_eq!(shape(after.errors()), "Self :: A0 [UnprovableBound]");
    assert_eq!(shape(before.errors()), shape(after.errors()));
}

#[test]
fn a_result_is_read_with_the_items_that_follow_it() {
    // The tokens after the errors are not consumed, nested groups are no errors of the list.
    let result = analyze(&reduced(), LeakerConfig::new());
    let mut tokens = result.to_token_stream();
    tokens.extend(quote!(trailing tokens));
    let parsed =
        (|input: syn::parse::ParseStream| -> Result<(LeakerResult, proc_macro2::TokenStream)> {
            Ok((input.parse()?, input.parse()?))
        })
        .parse2(tokens)
        .unwrap();
    assert_eq!(parsed.0.errors().len(), 1);
    assert_eq!(parsed.0.errors()[0].nested.len(), 1);
    assert_eq!(parsed.1.to_string(), "trailing tokens");
}

#[test]
fn reporting_does_not_show_the_nested_errors() {
    let result = analyze(&reduced(), LeakerConfig::new());
    let error = &result.errors()[0];
    let nested = &error.nested[0];
    assert!(!nested.message.is_empty());
    // `Display`, `to_compile_error()`, `to_syn_error()` and `From` say the error, not what is in it.
    assert_eq!(error.to_string(), error.message);
    for shown in [
        error.to_compile_error().to_string(),
        to_syn_error(result.errors()).to_compile_error().to_string(),
        syn::Error::from(error.clone())
            .to_compile_error()
            .to_string(),
    ] {
        assert!(shown.contains("use absolute path"), "{}", shown);
        assert!(
            !shown.contains("is in a bound or where predicate"),
            "{}",
            shown
        );
        assert!(!shown.contains(&nested.message), "{}", shown);
    }
}

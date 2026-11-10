//! The errors are a flat list: a type whose own path conflicts is written as it is, and what is in
//! it is analyzed on its own.
//!
//! A type that cannot be leaked (`MyType<'_, Rel>`) and a path of a value are not copied whole:
//! only the path of the head is written as it is (it is the only path of its error), and the types
//! in its arguments are entries, or have errors of their own, after it. A bound with a relative
//! path, and a type in a predicate, are copied whole (a leaked type in a predicate may be
//! unprovable), the relative paths in them are paths of their error, and the errors in them are
//! listed after it as well. A rule after the analysis drops the errors it accepts and leaves the
//! others where they were, so
//!
//! - the reduced case of `tests/allow_stages.rs` (D): `X: Into<Self::A0>` is an error for `Into`
//!   and, after it, one for `Self::A0`;
//! - every rule that drops the first leaves the second, on a config and on a result alike;
//! - a bound or a type with nothing wrong in its arguments has one error, or one for each of its
//!   heads where it has several (`Vec<Rel<'_>>`);
//! - the arguments of a path of a value are analyzed on their own;
//! - the serialization of an error has no nested errors, and a list of errors is filtered in order,
//!   identical errors included;
//! - `ErrorItem::paths`: each relative path that the copy writes as it is, as the cause of its kind
//!   (a path in a type is a `Cause::Type`, the trait of a bound or of a qualified path a
//!   `Cause::Trait`, a path in a value a `Cause::Expr`, the path of a macro a `Cause::Macro`, and one
//!   whose kind is not known, the root of a `use` in a value, a `Cause::Unknown`; all without
//!   generic arguments). Only the head of what conflicts is written as it is.

mod common;

use common::{
    accepting, allow_every_cause, causes_accepting, starts_with, to_syn_error, AcceptCause,
    Recovered, Rule,
};
use syn::*;
use template_quote::{quote, ToTokens};
use type_leak::*;

fn analyze<A: Allow>(item: &ItemTrait, config: A) -> LeakerResult {
    common::analyze_trait(item, config, false)
}

/// [`analyze()`] with `check_values`: the values are read too.
fn analyze_values<A: Allow>(item: &ItemTrait, config: A) -> LeakerResult {
    common::analyze_trait(item, config, true)
}

/// The reduced case: the trait of `X: Into<Self::A0>` is relative, so it is written as it is, and
/// `T: ::p::BT1<String>` makes `Self::A0` unprovable in a predicate once `Into` is allowed.
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

/// The paths of the errors as `kind(path)`, the paths of an error in a list.
fn paths_of(result: &LeakerResult) -> Vec<Vec<String>> {
    result
        .errors()
        .iter()
        .map(|error| error.paths.iter().map(show).collect())
        .collect()
}

fn show(cause: &Cause) -> String {
    let kind = match cause {
        Cause::Type(_) => "type",
        Cause::Trait(_) => "trait",
        Cause::Expr(_) => "expr",
        Cause::Macro(_) => "macro",
        Cause::Unknown(_) => "unknown",
        other => panic!("{:?}", other),
    };
    format!("{}({})", kind, text(cause))
}

fn values() -> Leaker<Rules> {
    let mut leaker = Leaker::with_rule(Rules::new());
    leaker.check_values = true;
    leaker
}

/// The shape of the errors: `cause [reason]`, in order.
fn shape(errors: &[ErrorItem]) -> String {
    errors
        .iter()
        .map(|error| format!("{} [{:?}]", text(&error.cause), error.reason))
        .collect::<Vec<_>>()
        .join(", ")
}

fn into_cause() -> Cause {
    Cause::Trait(parse_quote!(Into<Self::A0>))
}

fn a0_cause() -> Cause {
    Cause::Type(parse_quote!(Self::A0))
}

#[test]
fn a_bound_written_as_it_is_has_the_error_inside_it_after_it() {
    let result = analyze(&reduced(), Rules::new());
    // `String` is an entry, and the bound is an error for its path `Into`, with the error of
    // `Self::A0` after it.
    assert_eq!(result.errors().len(), 2);
    let (bound, inside) = (&result.errors()[0], &result.errors()[1]);
    assert_eq!(bound.cause, into_cause());
    assert_eq!(bound.reason, Reason::RelativePath);
    assert_eq!(paths(bound), ["Into"]);
    // With `Into` allowed the analysis finds `Self::A0` unprovable.
    assert_eq!(inside.cause, a0_cause());
    assert_eq!(inside.reason, Reason::UnprovableBound);
    assert!(inside.paths.is_empty());
    assert!(!inside.place.in_value && !inside.place.in_default);
    // The same error as the analysis with an absolute `Into` reports in the first place.
    let absolute: ItemTrait = parse_quote!(
        trait Tr<T: ::p::BT1<String>> {
            type A0;
            fn m<X>(&self)
            where
                X: ::core::convert::Into<Self::A0>;
        }
    );
    let direct = analyze(&absolute, Rules::new());
    assert_eq!(direct.errors().len(), 1);
    // (The part that it is about is another one of another Referrer.)
    let same = &direct.errors()[0];
    assert_eq!(text(&same.cause), text(&inside.cause));
    assert_eq!(same.reason, inside.reason);
    assert_eq!(same.message, inside.message);
    assert_eq!(
        same.notes.iter().map(|(_, note)| note).collect::<Vec<_>>(),
        inside
            .notes
            .iter()
            .map(|(_, note)| note)
            .collect::<Vec<_>>()
    );
    assert_eq!(same.help_text(), inside.help_text());
    // The Referrer is the one of the absolute bound.
    assert_eq!(
        result.clone().recovered().iter().count(),
        direct.recovered().iter().count()
    );
}

#[test]
fn every_rule_that_drops_the_first_error_leaves_the_second() {
    // The cause, the path, and the prelude (which has `Into`) drop the bound; the entry `String` is
    // there all the same, and the error left is the one inside the bound.
    let by_cause = analyze(&reduced(), Rules::new()).accept_cause(&into_cause());
    let by_path = analyze(&reduced(), Rules::new()).allow(starts_with(parse_quote!(Into)));
    let by_prelude = analyze(&reduced(), Rules::new()).allow(Rules::new().prelude());
    for left in [&by_cause, &by_path, &by_prelude] {
        assert_eq!(shape(left.errors()), "Self :: A0 [UnprovableBound]");
        assert!(left.errors()[0].paths.is_empty());
    }
    // A rule that drops nothing leaves both.
    let other = analyze(&reduced(), Rules::new())
        .allow(Rules::new().primitive())
        .allow(Rules::new().std_crates());
    assert_eq!(
        shape(other.errors()),
        "Into < Self :: A0 > [RelativePath], Self :: A0 [UnprovableBound]"
    );
    // A config gives the same: the rule is applied when the analysis finishes.
    let plain = analyze(&reduced(), Rules::new());
    let config = Rules::new().or_if(accepting(plain.errors(), &into_cause()));
    assert_eq!(
        shape(analyze(&reduced(), config).errors()),
        "Self :: A0 [UnprovableBound]"
    );
    // A config that allows `Into` makes it context-free: the analysis finds the same error.
    let config = Rules::new().or_if(starts_with(parse_quote!(Into)));
    assert_eq!(
        shape(analyze(&reduced(), config).errors()),
        "Self :: A0 [UnprovableBound]"
    );
    // Allowing the cause of the error that is left drops it, and nothing else is left.
    let (all, rounds) = allow_every_cause(by_cause.clone());
    assert_eq!(rounds, 1);
    assert!(all.errors().is_empty());
    // A result takes the rules after the trip as before it.
    let trip = parse2::<LeakerResult>(analyze(&reduced(), Rules::new()).to_token_stream()).unwrap();
    assert_eq!(
        shape(trip.allow(Rules::new().prelude()).errors()),
        "Self :: A0 [UnprovableBound]"
    );
}

#[test]
fn the_rules_add_up_in_any_order() {
    // What a rule accepts is accepted whatever the other rules did before.
    let result = analyze(&reduced(), Rules::new()).accept_cause(&a0_cause());
    assert_eq!(shape(result.errors()), "Into < Self :: A0 > [RelativePath]");
    let result = result.allow(Rules::new().prelude());
    assert!(result.errors().is_empty());
    let other = analyze(&reduced(), Rules::new())
        .allow(Rules::new().prelude())
        .accept_cause(&a0_cause());
    assert!(other.errors().is_empty());
    // And in one set, as a config has them.
    let config = Rules::new().prelude().or_if(accepting(
        analyze(&reduced(), Rules::new()).errors(),
        &a0_cause(),
    ));
    assert!(analyze(&reduced(), config).errors().is_empty());
}

#[test]
fn the_place_rules_are_the_same_before_and_after_the_analysis() {
    // The place rules drop the errors of their place at both stages, and what is inside a path of a
    // value is analyzed on its own: `Rel<'_>` in the turbofish has an error of its own.
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
    let base = Rules::new();
    let plain = analyze_values(&item, base.clone());
    // The error of `Self::A0` is a cycle: it is found apart from the others, and comes after them.
    assert_eq!(
        shape(plain.errors()),
        "Into < Self :: A0 > [RelativePath], inner :: f :: < Rel < '_ > > [RelativePath], \
         Rel < '_ > [RelativePath], Self :: A0 [UnprovableBound]"
    );
    assert_eq!(paths(&plain.errors()[1]), ["inner :: f"]);
    assert_eq!(paths(&plain.errors()[2]), ["Rel"]);
    for rule in [
        Rule::Cause(causes_accepting(plain.errors(), &into_cause())),
        Rule::Values,
        Rule::Default,
        Rule::Path(parse_quote!(Into)),
    ] {
        let before = rule.analyze_with(&item, base.clone(), true);
        let after = rule.on_result(analyze_values(&item, base.clone()));
        let both = |result: &LeakerResult| {
            result
                .errors()
                .iter()
                .map(|error| error.to_token_stream().to_string())
                .collect::<Vec<_>>()
        };
        match rule {
            // The config makes `Into` context-free in the analysis, and finds the errors in the
            // order of the analysis, as the result does.
            Rule::Path(_) | Rule::Cause(_) => {
                assert_eq!(causes(before.errors()), causes(after.errors()), "path rule")
            }
            _ => assert_eq!(both(&before), both(&after)),
        }
    }
    // `Rules::values()` drops the error in the body, and the bound stays with the error inside it.
    let result = analyze_values(&item, base.clone()).allow(Rules::new().values());
    assert_eq!(
        shape(result.errors()),
        "Into < Self :: A0 > [RelativePath], Self :: A0 [UnprovableBound]"
    );
    // The path of the value is an error of its own, and `Rel<'_>` in it is another.
    let result =
        analyze_values(&item, base).accept_cause(&Cause::Expr(parse_quote!(inner::f::<Rel<'_>>)));
    assert_eq!(
        shape(result.errors()),
        "Into < Self :: A0 > [RelativePath], Rel < '_ > [RelativePath], \
         Self :: A0 [UnprovableBound]"
    );
}

#[test]
fn a_type_in_the_arguments_of_a_bound_with_nothing_wrong_has_no_error() {
    let item: ItemTrait = parse_quote!(
        trait Tr: Clone {
            fn m<X>(&self)
            where
                X: Rel<u8>;
            fn n<Y: Other>(&self, a: Vec<Rel<'_>>, b: MyType<'_>);
        }
    );
    let config = Rules::new().primitive();
    let result = analyze(&item, config);
    // The supertrait, the bound, the other bound, `Vec` and the `Rel<'_>` inside it, and `MyType`:
    // one path each. The path of a bound is a trait, that of a type is a type.
    assert_eq!(
        causes(result.errors()),
        [
            "Clone",
            "Rel < u8 >",
            "Other",
            "Vec < Rel < '_ > >",
            "Rel < '_ >",
            "MyType < '_ >"
        ]
    );
    assert_eq!(
        paths_of(&result),
        [
            ["trait(Clone)"],
            ["trait(Rel)"],
            ["trait(Other)"],
            ["type(Vec)"],
            ["type(Rel)"],
            ["type(MyType)"],
        ]
    );
    // Dropping them leaves nothing.
    assert_eq!(allow_every_cause(result.clone()).1, 1);
}

#[test]
fn an_error_without_a_path_has_none() {
    // `Self` where nothing declares it has no meaning: nothing to allow but the cause. (The default of a
    // parameter that names it is an error for the path of its type, `Wrap`.)
    let item: ItemTrait = parse_quote!(
        trait Tr<T = Wrap<Self>> {}
    );
    let result = analyze(&item, Rules::new());
    assert_eq!(result.errors().len(), 1);
    assert_eq!(paths(&result.errors()[0]), ["Wrap"]);
    let mut leaker = Leaker::new();
    leaker.enter_generics(&Generics::default(), |view| {
        view.visit_type(&parse_quote!(Wrap<Self>));
    });
    let result = leaker.result();
    assert_eq!(result.errors().len(), 1);
    assert_eq!(result.errors()[0].reason, Reason::SelfWithoutMeaning);
    assert!(result.errors()[0].paths.is_empty());
}

/// Where a part is copied whole.
enum Whole {
    Type(Type),
    Predicate(WhereClause),
}

/// A part that is copied whole (a reference with a lifetime, a type in a predicate) and names a
/// `Self` without meaning is written as it is with what is in it: the error is that of `Self`,
/// with no path, no entry is left inside, and the Visitor leaves the copy as written. A type that
/// is not whole leaks the types in it that do not name `Self`.
#[test]
fn a_whole_part_with_a_self_without_meaning_has_no_entry_inside() {
    use syn::visit_mut::VisitMut;
    // The generics, the part, the entries left, and the copy as the Visitor writes it (`None`: as
    // written).
    type Case = (Generics, Whole, &'static [&'static str], Option<Whole>);
    let cases: Vec<Case> = vec![
        (
            parse_quote!(<'a>),
            Whole::Type(parse_quote!(&'a Rel<Self>)),
            &[],
            None,
        ),
        (
            parse_quote!(<'a, T>),
            Whole::Type(parse_quote!(&'a Wrap<T, Self>)),
            &[],
            None,
        ),
        (
            parse_quote!(<T>),
            Whole::Type(parse_quote!(&'static Rel<Self, T>)),
            &[],
            None,
        ),
        // `Other` would be an entry, and so would `Rel<Wrap>` if `Self` were not there.
        (
            parse_quote!(<>),
            Whole::Type(parse_quote!(&'static (Other, Rel<Self, Wrap>))),
            &[],
            None,
        ),
        (
            parse_quote!(<>),
            Whole::Predicate(parse_quote!(where Rel<Self>: Sized)),
            &[],
            None,
        ),
        (
            parse_quote!(<T>),
            Whole::Predicate(parse_quote!(where Rel<Self, T>: Sized)),
            &[],
            None,
        ),
        (
            parse_quote!(<>),
            Whole::Predicate(parse_quote!(where Rel<Self, Other<u8>>: Sized)),
            &[],
            None,
        ),
        // Not whole: the types in it that do not name `Self` are leaked.
        (
            parse_quote!(<>),
            Whole::Type(parse_quote!(Rel<Self, Other>)),
            &["Other"],
            Some(Whole::Type(parse_quote!(Rel<Self, Entry<0usize>>))),
        ),
    ];
    for (generics, part, entries, copy) in cases {
        let mut written = match &part {
            Whole::Type(ty) => text(ty),
            Whole::Predicate(predicate) => text(predicate),
        };
        written.insert_str(0, &format!("{} ", text(&generics)));
        let mut leaker = Leaker::with_rule(Rules::new().primitive().prelude());
        let (context, ()) = leaker.enter_generics(&generics, |view| match &part {
            Whole::Type(ty) => view.visit_type(ty),
            Whole::Predicate(predicate) => view.visit_where_clause(predicate),
        });
        let result = leaker.result();
        // One error, for `Self`, whose cause is the only thing to allow.
        assert_eq!(
            shape(result.errors()),
            "Self [SelfWithoutMeaning]",
            "{}",
            written
        );
        assert_eq!(result.errors()[0].message, "`Self` has no meaning here");
        assert!(result.errors()[0].paths.is_empty(), "{}", written);
        assert!(allow_every_cause(result.clone()).0.finish().is_ok());
        let referrer = result.recovered();
        let found: Vec<String> = common::types(&referrer).map(text).collect();
        assert_eq!(found, entries, "{}", written);
        // The Visitor writes the part as it is, and the entries it has.
        let mut visitor = common::visitor(&referrer, context, |_, ix, _| parse_quote!(Entry<#ix>));
        match (part, copy) {
            (Whole::Type(mut ty), copy) => {
                let expected = match copy {
                    Some(Whole::Type(copy)) => copy,
                    _ => ty.clone(),
                };
                visitor.visit_type_mut(&mut ty);
                assert_eq!(ty, expected, "{}", written);
            }
            (Whole::Predicate(mut predicate), _) => {
                let expected = predicate.clone();
                visitor.visit_where_clause_mut(&mut predicate);
                assert_eq!(predicate, expected, "{}", written);
            }
        }
    }
}

#[test]
fn a_type_that_cannot_be_leaked_has_what_is_wrong_inside_it_listed_after_it() {
    // `::p::Tr<Rel<Self::A0>>` is leaked whole, and its entry `Rel<Self::A0>` is unprovable in a
    // predicate beside `T: ::p::BT1<String>`: copied as written, whole, it has `Self::A0` in it,
    // which is unprovable too. The error of the part itself is found as well, once.
    let item: ItemTrait = parse_quote!(
        trait Tr<T: ::p::BT1<String>> {
            type A0;
            fn m<X>(&self)
            where
                X: ::p::Tr<Rel<Self::A0>>;
        }
    );
    let result = analyze(&item, Rules::new());
    assert_eq!(
        shape(result.errors()),
        "Rel < Self :: A0 > [UnprovableBound], Self :: A0 [UnprovableBound]"
    );
    assert_eq!(paths(&result.errors()[0]), ["Rel"]);
    // The config that allows `Rel` has the second error and not the first, as the result does
    // after the rule.
    let result_after = result.clone().allow(starts_with(parse_quote!(Rel)));
    assert_eq!(shape(result_after.errors()), "Self :: A0 [UnprovableBound]");
    let config = Rules::new().or_if(starts_with(parse_quote!(Rel)));
    assert_eq!(
        shape(analyze(&item, config).errors()),
        "Self :: A0 [UnprovableBound]"
    );
}

#[test]
fn the_arguments_of_a_value_path_are_analyzed_on_their_own() {
    // The path of a value is written as it is, and the types in its arguments are analyzed: `Rel<'_>`
    // has an error of its own, and so has `Wrap<..>`, with `Rel2<'_>` in it.
    let item: ItemTrait = parse_quote!(
        trait Tr {
            fn d(&self) {
                let _ = inner::f::<Rel<'_>, Wrap<Rel2<'_>>>();
            }
        }
    );
    let config = Rules::new();
    let result = analyze_values(&item, config);
    assert_eq!(
        causes(result.errors()),
        [
            "inner :: f :: < Rel < '_ > , Wrap < Rel2 < '_ > > >",
            "Rel < '_ >",
            "Wrap < Rel2 < '_ > >",
            "Rel2 < '_ >"
        ]
    );
    assert_eq!(paths(&result.errors()[0]), ["inner :: f"]);
    assert!(result.errors()[0].place.in_value);
    assert_eq!(paths(&result.errors()[1]), ["Rel"]);
    assert_eq!(paths(&result.errors()[2]), ["Wrap"]);
    assert_eq!(paths(&result.errors()[3]), ["Rel2"]);
    // The rules drop the errors one by one, each with its own path.
    let result = result.allow(starts_with(parse_quote!(inner)));
    assert_eq!(result.errors().len(), 3);
    let result = result.allow(starts_with(parse_quote!(Rel)));
    assert_eq!(
        causes(result.errors()),
        ["Wrap < Rel2 < '_ > >", "Rel2 < '_ >"]
    );
    let result = result.allow(starts_with(parse_quote!(Wrap)));
    assert_eq!(causes(result.errors()), ["Rel2 < '_ >"]);
    let result = result.allow(starts_with(parse_quote!(Rel2)));
    assert!(result.errors().is_empty());
    // The config that allows the path leaves the types in the arguments: the same errors.
    let config = Rules::new().or_if(starts_with(parse_quote!(inner)));
    assert_eq!(
        shape(analyze_values(&item, config).errors()),
        "Rel < '_ > [RelativePath], Wrap < Rel2 < '_ > > [RelativePath], \
         Rel2 < '_ > [RelativePath]"
    );
}

// ── a list of errors, written by hand ───────────────────────────────────────────────────────

/// A result with the errors `errors`, written by hand in the serialization.
fn with_errors(errors: &str) -> LeakerResult {
    let referrer = Leaker::new().result().to_token_stream();
    let mut all = referrer;
    all.extend(quote!(@ [#{errors.parse::<proc_macro2::TokenStream>().unwrap()}]));
    parse2(all).unwrap()
}

/// The error `cause` (a type), `UnprovableBound`, with the message `m`, the note `(a) "n"` and the help.
fn plain(cause: &str) -> String {
    format!(r#"{{ type ( {cause} ) UnprovableBound "m" [ ( a ) "n" ] ( h ) "help" }}"#)
}

/// A relative path `P` as the error `cause`.
fn outer(cause: &str) -> String {
    format!(r#"{{ type ( {cause} ) RelativePath "o" [] paths [ type ( P ) ] }}"#)
}

#[test]
fn a_rule_drops_the_errors_it_accepts_and_leaves_the_others_in_order() {
    let (s, t) = (plain("S"), plain("T"));
    // The error with the path `P` goes, the others stay where they were.
    let result = with_errors(&format!(
        "{} {} {} {t} {s}",
        plain("A"),
        outer("O"),
        plain("C")
    ))
    .allow(starts_with(parse_quote!(P)));
    assert_eq!(
        shape(result.errors()),
        "A [UnprovableBound], C [UnprovableBound], T [UnprovableBound], S [UnprovableBound]"
    );
    // Errors that are identical are kept both: nothing is merged.
    let result =
        with_errors(&format!("{s} {s} {}", outer("O"))).allow(starts_with(parse_quote!(P)));
    assert_eq!(
        shape(result.errors()),
        "S [UnprovableBound], S [UnprovableBound]"
    );
    let result = with_errors(&format!("{s} {s}")).allow(starts_with(parse_quote!(P)));
    assert_eq!(result.errors().len(), 2);
    // The analysis reports two identical errors side by side: both stay, with a rule that drops
    // nothing and with one that drops something else.
    let item: ItemTrait = parse_quote!(
        trait Tr {
            fn m(&self, a: Wrap<'_>, b: Wrap<'_>);
        }
    );
    let result = analyze(&item, Rules::new());
    assert_eq!(result.errors().len(), 2);
    assert_eq!(text(&result.errors()[0]), text(&result.errors()[1]));
    let other = result
        .clone()
        .allow(Rules::new().prelude())
        .accept_cause(&Cause::Type(parse_quote!(Other)));
    assert_eq!(other.errors().len(), 2);
}

#[test]
fn a_path_rule_trims_the_paths_it_allows_from_the_errors_that_stay() {
    // An error with several paths stays until every path is allowed, by whatever rules, and keeps
    // those that are not.
    let several =
        r#"{ type ( K ) RelativePath "o" [] paths [ type ( P ) , type ( Q ) , type ( R ) ] }"#;
    let result = with_errors(several).allow(starts_with(parse_quote!(P)));
    assert_eq!(paths(&result.errors()[0]), ["Q", "R"]);
    let result = result.allow(starts_with(parse_quote!(R)));
    assert_eq!(paths(&result.errors()[0]), ["Q"]);
    let result = result.allow(starts_with(parse_quote!(Q)));
    assert!(result.errors().is_empty());
    // In one set, in any order, as a config has them.
    let first = with_errors(several)
        .allow(starts_with(parse_quote!(Q)))
        .allow(starts_with(parse_quote!(P)));
    let second = with_errors(several)
        .allow(starts_with(parse_quote!(P)))
        .allow(starts_with(parse_quote!(Q)));
    assert_eq!(shape(first.errors()), shape(second.errors()));
    assert_eq!(paths(&first.errors()[0]), ["R"]);
    // A rule on the place drops an error of that place, whatever its paths.
    let in_value = r#"{ type ( K ) RelativePath "o" [] value paths [ type ( P ) ] }"#;
    let result = with_errors(&format!("{in_value} {}", outer("O"))).allow(Rules::new().values());
    assert_eq!(shape(result.errors()), "O [RelativePath]");
}

// ── serialization and reporting ─────────────────────────────────────────────────────────────

#[test]
fn an_error_is_written_with_its_paths_and_has_no_nested_errors() {
    let result = analyze(&reduced(), Rules::new());
    let error = &result.errors()[0];
    let tokens = error.to_token_stream().to_string();
    assert!(tokens.contains("paths [trait (Into)]"), "{}", tokens);
    assert!(!tokens.contains("nested"), "{}", tokens);
}

#[test]
fn each_error_is_reported_on_its_own() {
    let result = analyze(&reduced(), Rules::new());
    let (error, inside) = (&result.errors()[0], &result.errors()[1]);
    // `Display`, `to_compile_error()`, `to_syn_error()` and `From` say the error.
    assert_eq!(error.to_string(), error.message);
    for shown in [
        error.to_compile_error().to_string(),
        syn::Error::from(error.clone())
            .to_compile_error()
            .to_string(),
    ] {
        assert!(shown.contains("use absolute path"), "{}", shown);
        assert!(!shown.contains(&inside.message), "{}", shown);
    }
    // All of them, each at its own place.
    let all = to_syn_error(result.errors()).to_compile_error().to_string();
    assert!(all.contains("use absolute path"), "{}", all);
    assert!(all.contains(&inside.message), "{}", all);
    assert_eq!(to_syn_error(result.errors()).into_iter().count(), 2);
}

// ── the kind of each path ───────────────────────────────────────────────────────────────────

#[test]
fn the_trait_of_a_bound_and_a_trait_object_is_a_trait_and_its_arguments_are_types() {
    let item: ItemStruct = parse_quote!(
        struct S<T: RelTr<Arg>> {
            a: ::p::W<'_, dyn OtherTr<Item = Assoc>>,
            b: T,
        }
    );
    let mut leaker = Leaker::new();
    leaker.intern_struct(&item);
    let result = leaker.result();
    // The bound is a predicate, so it is copied whole, `Arg` in it too: it is a path of its error. The
    // trait object is written as it is, and its argument is an entry.
    assert_eq!(
        paths_of(&result),
        [vec!["trait(RelTr)", "type(Arg)"], vec!["trait(OtherTr)"]]
    );
    let referrer = result.recovered();
    let entries: Vec<String> = common::types(&referrer)
        .map(|ty| ty.to_token_stream().to_string())
        .collect();
    assert_eq!(entries, ["Assoc"]);
}

#[test]
fn the_qualified_path_in_the_arguments_of_what_conflicts_is_leaked() {
    let item: ItemStruct = parse_quote!(
        struct S {
            a: Wrap<'_, <Rel as m::Tr<u8>>::X>,
        }
    );
    let mut leaker = Leaker::new();
    leaker.intern_struct(&item);
    let result = leaker.result();
    assert_eq!(paths_of(&result), [vec!["type(Wrap)"]]);
    let entries: Vec<String> = result
        .recovered()
        .iter()
        .map(|(_, ty, _)| ty.to_token_stream().to_string())
        .collect();
    assert_eq!(entries, ["< Rel as m :: Tr < u8 > > :: X"]);
}

#[test]
fn the_macro_of_a_type_is_no_path_of_the_error() {
    let item: ItemStruct = parse_quote!(
        struct S {
            a: Wrap<'_, m!(u8)>,
        }
    );
    let mut leaker = Leaker::new();
    leaker.intern_struct(&item);
    let result = leaker.result();
    // The type macro is a type of its own, leaked whole: no `macro` path (that of a value is one,
    // see `tests/error_items.rs`).
    assert_eq!(paths_of(&result), [vec!["type(Wrap)"]]);
}

#[test]
fn a_path_in_a_value_is_an_expression_and_its_arguments_are_leaked() {
    let item: ItemConst = parse_quote!(
        const C: u8 = rel::f::<Arg>(1);
    );
    let mut leaker = values();
    leaker.enter_generics(&Generics::default(), |view| {
        view.visit_item(&Item::Const(item));
    });
    let result = leaker.result();
    assert_eq!(paths_of(&result), [vec!["expr(rel :: f)"]]);
    let entries: Vec<String> = result
        .recovered()
        .iter()
        .map(|(_, ty, _)| ty.to_token_stream().to_string())
        .collect();
    // The type of the constant is leaked too.
    assert_eq!(entries, ["Arg", "u8"]);
}

#[test]
fn the_root_of_a_use_in_a_value_has_no_known_kind() {
    let item: ItemFn = parse_quote!(
        fn f() {
            use inner::X;
        }
    );
    let mut leaker = values();
    leaker.enter_generics(&Generics::default(), |view| {
        view.visit_item(&Item::Fn(item));
    });
    let result = leaker.result();
    assert_eq!(result.errors().len(), 1);
    assert_eq!(paths_of(&result), [vec!["unknown(inner)"]]);
}

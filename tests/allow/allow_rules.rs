//! `Allow`, `Rules` and `LeakerResult::allow()`.
//!
//! The rules: closures, tuples and the flags, which the convenience methods are made of. The tests
//! are about what the rule is asked (each path as the cause of its kind, never `Self` or a generic,
//! never arguments in the analysis) and about a rule that panics (on a result, and inside the
//! closure of a scope).
//!
//! Accepting a conflict (from the section "accepting a conflict" on). The Leaker corrects every
//! conflict the same way, reported or not: the part is copied as written, whole, with nothing in it
//! leaked. `allow(cause)` drops the errors with an equal cause (the same kind and the same tokens,
//! whatever the spans and the spacing) and nothing else: the Referrer never changes.
//!
//! - The item below (`f` and `g`): the error, the causes that do not match (the kind and the tokens
//!   count), and that the entries are those of the item without `f`.
//! - The cause that is smaller than the part that is copied (the array with a closure).
//! - Scopes: the errors of an inner `enter_*` of a `View` come once, in place, `Self` without
//!   meaning and its place among the errors, the other kinds of cause.
//! - The corrections that no conflict of the type itself makes: an entry that names a `Self`
//!   without meaning, or whose bounds do (in a type, a bound, a default, a value), and a type that
//!   leaves a lifetime of its method unconstrained, are copied whole and have no entry.
//! - What accepting does not make compile (`Self` in a predicate beside a leaked bound).
//! - `Rules::std_crates()`.
//! - The list of errors of a Leaker: several calls, nested scopes.
//! - The parts that the Referrer carries: the visitor leaves a part as written (also beside an
//!   entry with the same tokens) and looks up no entry in it, and the serialization.
//!
//! Every `Reason` is accepted by `tests/errors/error_items.rs`
//! (`allowing_the_causes_of_the_errors_of_each_reason_accepts_the_item`), and a cause with spans
//! of its own by `tests/crates/spans`.
//!
//! The last section, the module `names`, is about the names that `Rules::primitive()` and
//! `Rules::prelude()` allow, one by one.

use crate::common;

use common::{allow_every_cause, analyze_trait, bare_path, starts_with, AcceptCause, Recovered};
use proc_macro2::TokenStream;
use std::cell::{Cell, RefCell};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::rc::Rc;
use syn::visit_mut::VisitMut;
use syn::*;
use template_quote::{quote, ToTokens};
use type_leak::*;

/// A rule that captures nothing. Each closure is a type of its own, and each `Allow` type of a
/// Leaker is one more copy of the analyzer to compile: the rules of this type share one.
type FnRule = fn(&Cause, Place) -> bool;

/// `Place` is non-exhaustive.
fn place(in_value: bool, in_default: bool) -> Place {
    let mut place = Place::default();
    place.in_value = in_value;
    place.in_default = in_default;
    place
}

fn texts(items: &[ErrorItem]) -> Vec<String> {
    items
        .iter()
        .map(|item| item.to_token_stream().to_string())
        .collect()
}

/// The result of the analysis of the struct `item` with `config`.
fn analyze<A: Allow>(item: &ItemStruct, config: A) -> LeakerResult {
    let mut leaker = Leaker::with_rule(config);
    leaker.intern_struct(item);
    leaker.result()
}

#[test]
fn a_path_rule_leaves_the_paths_it_accepts_as_written() {
    let item: ItemStruct = parse_quote!(
        struct S {
            a: Vec<u8>,
            b: ::serde::W,
        }
    );
    // Without a rule `Vec<u8>` is an entry, and so is `::serde::W`.
    let result = analyze(&item, Rules::new());
    assert!(result.errors().is_empty());
    assert_eq!(
        common::type_texts(&result.finish().unwrap()),
        ["Vec < u8 >".to_string(), ":: serde :: W".to_string()]
    );
    // A rule for each of them: nothing is leaked, and there is no error.
    let config = Rules::new()
        .prelude()
        .primitive()
        .or_if(starts_with(parse_quote!(::serde)));
    let result = analyze(&item, config);
    assert!(result.errors().is_empty());
    assert!(common::type_texts(&result.finish().unwrap()).is_empty());
}

#[test]
fn rules_add_up_on_a_result() {
    // The three traits of the trait object are written as they are: they are the paths of its one
    // error.
    let item: ItemStruct = parse_quote!(
        struct S<'a> {
            a: &'a (dyn Rel<'_> + Send + std::fmt::Debug),
        }
    );
    let result = analyze(&item, Rules::new());
    assert_eq!(result.errors().len(), 1);
    assert_eq!(
        common::path_texts(&result.errors()[0]),
        ["Rel", "Send", "std :: fmt :: Debug"]
    );
    // The rules take the paths off one by one, as the rule of one call that accepts them all.
    let result = result.allow(Rules::new().prelude());
    assert_eq!(
        common::path_texts(&result.errors()[0]),
        ["Rel", "std :: fmt :: Debug"]
    );
    let result = result.allow(Rules::new().std_crates());
    assert_eq!(common::path_texts(&result.errors()[0]), ["Rel"]);
    // The error goes with the last.
    let result = result.allow(starts_with(parse_quote!(Rel)));
    assert!(result.errors().is_empty());
}

#[test]
fn an_error_is_accepted_by_its_paths_if_it_has_any_and_by_its_cause_if_it_has_none() {
    let item: ItemStruct = parse_quote!(
        struct S<'a> {
            a: &'a (dyn Base<'_> + Local),
        }
    );
    let result = analyze(&item, Rules::new());
    assert_eq!(result.errors().len(), 1);
    let error = &result.errors()[0];
    assert_eq!(common::path_texts(error), ["Base", "Local"]);
    // A rule for the cause, as written, never accepts it, and is not even asked: the paths are.
    let seen = Rc::new(Seen::default());
    let cause = error.cause.clone();
    let seen_rule = seen.rule();
    let by_cause = result.clone().allow(move |c: &Cause, place: Place| {
        seen_rule(c, place);
        *c == cause
    });
    assert_eq!(texts(by_cause.errors()), texts(result.errors()));
    assert!(!seen.asked.borrow().iter().any(|(c, _)| *c == error.cause));
    assert!(seen.contains("Base") && seen.contains("Local"));
    assert!(seen.paths().iter().all(|p| p == "Base" || p == "Local"));
    // A rule that allows a cause that is a path and some of the paths trims the others: the error
    // stays with those that are not allowed.
    let some = result.clone().allow(starts_with(parse_quote!(Base)));
    assert_eq!(paths_of(&some), [["Local"]]);
    // It goes when all the paths are allowed, by whatever rule.
    let all = some.allow(starts_with(parse_quote!(Local)));
    assert!(all.errors().is_empty());
    let whole = error.cause.clone();
    assert!(result.accept_cause(&whole).errors().is_empty());
    // An error without paths is accepted by its cause: F79 (`tests/findings/findings.rs`), and
    // `a_mention_of_self_without_meaning_is_accepted_by_its_cause` below.
}

#[test]
fn the_rules_may_borrow_what_is_around() {
    let result = three_errors();
    let causes: Vec<Cause> = result.errors().iter().map(|e| e.cause.clone()).collect();
    assert_eq!(causes.len(), 3);
    // A rule of a result borrows a list of paths that is not 'static.
    let first_two: Vec<Cause> = result.errors()[..2]
        .iter()
        .flat_map(|e| e.paths.clone())
        .collect();
    let left = result
        .clone()
        .allow(|cause: &Cause, _: Place| first_two.contains(cause));
    assert_eq!(left.errors().len(), 1);
    assert_eq!(left.errors()[0].cause, causes[2]);
    // So does a rule of a config, which the Leaker keeps while it analyzes.
    let allowed: Vec<Path> = vec![parse_quote!(MyType)];
    let own = result.errors()[0].paths.clone();
    let config = Rules::new().or_if(|cause: &Cause, place: Place| {
        allowed.iter().any(|a| starts_with(a.clone())(cause, place)) || own.contains(cause)
    });
    let by_config = analyze(&three_errors_item(), config);
    assert_eq!(by_config.errors().len(), 1);
    assert_eq!(by_config.errors()[0].cause, causes[2]);
}

#[test]
fn rules_are_clone_and_debug_where_their_rule_is() {
    let rules = Rules::new().primitive().defaults();
    assert_eq!(
        format!("{:?}", rules.clone()),
        "Rules { primitive: true, prelude: false, std_crates: false, external_crates: false, \
         crate_paths: false, values: false, defaults: true, custom: () }"
    );
    // A function is `Debug` and `Clone`; a closure that captures is `Clone` if its captures are.
    let _ = format!("{:?}", Rules::new().or(()));
    let owned: Path = parse_quote!(::serde);
    let rules = Rules::new().or_if(move |cause: &Cause, _: Place| cause.starts_with(&owned));
    let _clone = rules.clone();
}

#[test]
fn a_rule_of_a_config_is_called_on_every_check_of_a_path() {
    let item: ItemStruct = parse_quote!(
        struct S {
            a: Foo,
            b: Foo,
        }
    );
    let calls = Rc::new(Cell::new(0usize));
    let saw_foo = Rc::new(Cell::new(false));
    let rule = {
        let (calls, saw_foo) = (calls.clone(), saw_foo.clone());
        move |cause: &Cause, _: Place| {
            calls.set(calls.get() + 1);
            if bare_path(cause).map_or(false, |path| path.is_ident("Foo")) {
                saw_foo.set(true);
            }
            false
        }
    };
    let result = analyze(&item, Rules::new().or_if(rule));
    assert!(result.errors().is_empty());
    // `Foo` is checked for each of the fields, and nothing is remembered between the checks.
    assert!(calls.get() >= 2, "{} calls", calls.get());
    assert!(saw_foo.get());
}

// ── what a rule is asked, and what a panic of a rule does ────────────────────────────────────

/// What a rule was asked: the causes, with the place.
#[derive(Default)]
struct Seen {
    asked: RefCell<Vec<(Cause, Place)>>,
}

impl Seen {
    /// A rule that records what it is asked, and accepts nothing.
    fn rule(self: &Rc<Self>) -> impl Fn(&Cause, Place) -> bool {
        let seen = self.clone();
        move |cause: &Cause, place: Place| {
            seen.asked.borrow_mut().push((cause.clone(), place));
            false
        }
    }

    /// The paths that a rule saw, as tokens.
    fn paths(&self) -> Vec<String> {
        self.asked
            .borrow()
            .iter()
            .filter_map(|(cause, _)| cause.path())
            .map(|path| path.to_token_stream().to_string())
            .collect()
    }

    fn contains(&self, path: &str) -> bool {
        self.paths().iter().any(|seen| seen == path)
    }

    /// Whether a path that starts with the segment `head` was asked about.
    fn starts_with(&self, head: &str) -> bool {
        let longer = format!("{} ::", head);
        self.paths()
            .iter()
            .any(|seen| seen == head || seen.starts_with(&longer))
    }

    /// Whether the rule was asked about a cause that is a path with generic arguments, but for the
    /// causes of `errors` (and the errors in them), which the filter asks about as written.
    fn with_arguments(&self, errors: &[ErrorItem]) -> bool {
        let errors_causes: Vec<Cause> = errors.iter().map(|error| error.cause.clone()).collect();
        self.asked.borrow().iter().any(|(cause, _)| {
            cause.path().is_some() && bare_path(cause).is_none() && !errors_causes.contains(cause)
        })
    }
}

/// The item of [`three_errors()`].
fn three_errors_item() -> ItemStruct {
    parse_quote!(
        struct S<'a> {
            a: Rel<'a, '_>,
            b: MyType<'_>,
            c: Other<'_>,
        }
    )
}

/// Three errors, each of a relative trait or type, with the paths `Rel`, `MyType` and `Other`.
fn three_errors() -> LeakerResult {
    let result = analyze(&three_errors_item(), Rules::new());
    assert_eq!(paths_of(&result), [["Rel"], ["MyType"], ["Other"]]);
    result
}

fn paths_of(result: &LeakerResult) -> Vec<Vec<String>> {
    result.errors().iter().map(common::path_texts).collect()
}

#[test]
fn a_rule_that_panics_on_a_result_panics_the_call_and_loses_that_result() {
    let result = three_errors();
    let before = texts(result.errors());

    // A rule that accepts the first cause and panics on the second.
    let calls = Cell::new(0);
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        result.clone().allow(|_: &Cause, _: Place| {
            calls.set(calls.get() + 1);
            if calls.get() == 2 {
                panic!("the rule on causes");
            }
            true
        })
    }));
    assert!(outcome.is_err());
    assert_eq!(calls.get(), 2);
    // The result that the call took is gone with the panic; the one it was cloned from is as it was.
    assert_eq!(texts(result.errors()), before);
    let all: FnRule = |_, _| true;
    assert!(result.clone().allow(all).errors().is_empty());

    // A rule on paths, which accepts `Rel` and panics on `MyType`.
    let rel_or_panic: FnRule = |cause, _| {
        let path = bare_path(cause);
        if path.map_or(false, |path| path.is_ident("MyType")) {
            panic!("the rule on paths");
        }
        path.map_or(false, |path| path.is_ident("Rel"))
    };
    let outcome = catch_unwind(AssertUnwindSafe(|| result.clone().allow(rel_or_panic)));
    assert!(outcome.is_err());
    assert_eq!(texts(result.errors()), before);
    assert!(result.clone().finish().is_err());
    let not_rel: FnRule = |cause, _| {
        bare_path(cause).map_or(false, |path| {
            path.is_ident("MyType") || path.is_ident("Other")
        })
    };
    let by_path = result.allow(not_rel);
    assert_eq!(paths_of(&by_path), [["Rel"]]);
}

/// The item whose items name `Self`, `Self::A`, `T` and `T::Item`, and the relative `Rel<T>`.
fn own_names() -> ItemTrait {
    parse_quote!(
        trait Tr<T: Iterator, Vec> {
            type A;
            fn f(&self, a: Self::A, t: T::Item, v: Vec) -> Rel<T>;
            fn g(&self, a: Self::A) {
                let _: T::Item = Self::make(a);
                T::new();
            }
        }
    )
}

/// The paths of `errors`, as tokens.
fn all_paths(errors: &[ErrorItem], found: &mut Vec<String>) {
    for error in errors {
        found.extend(common::path_texts(error));
    }
}

#[test]
fn the_rule_is_not_asked_about_self_or_a_generic() {
    for check_values in [false, true] {
        let seen = Rc::new(Seen::default());
        let mut leaker = Leaker::with_rule(Rules::new().or_if(seen.rule()));
        leaker.check_values = check_values;
        leaker.check_bodies = check_values;
        leaker.intern_trait(&own_names());
        let result = leaker.result();
        let what = format!("check_values {}", check_values);
        assert!(seen.contains("Rel"), "{}: {:?}", what, seen.paths());
        for head in ["Self", "T", "Vec", "Iterator"] {
            // `Iterator` is the bound of `T`, a relative path that the rule may see. The causes of
            // the errors are asked about as written, so `Self` and `T` may be among them.
            let asked = seen
                .asked
                .borrow()
                .iter()
                .filter(|(cause, _)| bare_path(cause).is_some())
                .any(|(cause, _)| {
                    let path = cause.path().unwrap();
                    path.segments.first().map_or(false, |seg| seg.ident == head)
                });
            assert_eq!(asked, head == "Iterator", "{}: {:?}", what, seen.paths());
        }
        assert!(
            !seen.with_arguments(result.errors()),
            "{}: {:?}",
            what,
            seen.paths()
        );
        // The paths of the errors are the paths that the rule may see: none of those either.
        let mut found = Vec::new();
        all_paths(result.errors(), &mut found);
        assert!(!found.is_empty(), "{}", what);
        for head in ["Self", "T", "Vec"] {
            let longer = format!("{} ::", head);
            let headed = |path: &String| path == head || path.starts_with(&longer);
            assert!(!found.iter().any(headed), "{}: {:?}", what, found);
        }
    }
}

#[test]
fn a_rule_that_accepts_self_does_not_hide_an_error_about_it() {
    // `Self::A` of a struct cannot be written in the Repeater's impl, so a type that names it cannot
    // be an entry.
    fn by<A: Allow>(source: &str, allow: A) -> LeakerResult {
        let item: ItemStruct = parse_str(source).unwrap();
        let mut leaker = Leaker::with_rule(allow);
        leaker.intern_struct(&item);
        leaker.result()
    }
    for (source, relative) in [
        ("struct S { a: Rel<Self::A>, b: Self }", "Rel"),
        ("struct S { a: Box<Self::A>, b: Self }", "Box"),
    ] {
        let plain = by(source, Rules::new());
        assert_eq!(plain.errors().len(), 1, "{}", source);
        assert!(plain.clone().recovered().is_empty());
        // A rule for every path of one segment but the relative one accepts `Self` and a generic
        // too, if it is asked: it is not, neither in the analysis nor as a path of the error.
        let seen = Rc::new(Seen::default());
        let seen_rule = seen.rule();
        let short = move |cause: &Cause, place: Place| {
            seen_rule(cause, place);
            bare_path(cause).map_or(false, |path| {
                path.segments.len() == 1 && !path.is_ident(relative)
            })
        };
        let by_rule = by(source, Rules::new().or_if(short));
        assert_eq!(texts(by_rule.errors()), texts(plain.errors()), "{}", source);
        assert!(by_rule.recovered().is_empty(), "{}", source);
        assert!(!seen.starts_with("Self"), "{}: {:?}", source, seen.paths());
        // Accepting the paths of `Self` only, on a result.
        let paths_of_self: FnRule = |cause, _| {
            cause
                .path()
                .and_then(|path| path.segments.first())
                .map_or(false, |seg| seg.ident == "Self")
        };
        let on_result = plain.clone().allow(paths_of_self);
        assert_eq!(
            texts(on_result.errors()),
            texts(plain.errors()),
            "{}",
            source
        );
    }
}

#[test]
fn the_filter_does_not_ask_the_rule_about_a_path_of_self() {
    // An error written by hand, with a path that starts with `Self`, read as a serialized result.
    let base = three_errors();
    let mut error = base.errors()[0].clone();
    error.paths = vec![Cause::Type(parse_quote!(Self::A))];
    let mut tokens = base.recovered().to_token_stream();
    tokens.extend(quote!(@ [#error]));
    let result: LeakerResult = parse2(tokens).unwrap();
    assert_eq!(paths_of(&result), [["Self :: A"]]);
    let seen = Rc::new(Seen::default());
    let result = result.allow(seen.rule());
    // The error has a path, so its cause is not asked about, and the path of `Self` is not either.
    assert!(seen.asked.borrow().is_empty(), "{:?}", seen.paths());
    // A rule that accepts everything does not accept it either.
    let all: FnRule = |_, _| true;
    let result = result.allow(all);
    assert_eq!(paths_of(&result), [["Self :: A"]]);
}

#[test]
fn the_rule_sees_a_path_without_arguments_in_both_forms() {
    // The path compared as a whole, or by its segments.
    let exact: FnRule = |cause, _| {
        cause.path().map_or(false, |path| {
            let path = path.to_token_stream().to_string();
            path == "m :: Bound" || path == "helper :: f"
        })
    };
    // In a bound, and in a turbofish in a value, which `check_values` reads.
    let cases: [(ItemTrait, &str, bool); 2] = [
        (
            parse_quote!(
                trait Tr<T: m::Bound<Rel>> {
                    fn f(&self, t: T);
                }
            ),
            "m::Bound",
            false,
        ),
        (
            parse_quote!(
                trait Tr {
                    fn f(&self) {
                        helper::f::<Rel>();
                    }
                }
            ),
            "helper::f",
            true,
        ),
    ];
    for (item, prefix, values) in cases {
        let what = item.to_token_stream().to_string();
        let prefix = starts_with(parse_str(prefix).unwrap());
        let by_exact = analyze_trait(&item, Rules::new().or_if(exact), values);
        let by_prefix = analyze_trait(&item, Rules::new().or_if(prefix), values);
        assert!(by_exact.errors().is_empty(), "{}", what);
        assert_eq!(
            texts(by_exact.errors()),
            texts(by_prefix.errors()),
            "{}",
            what
        );
        // `Rel` in the arguments is still analyzed: it is the one entry.
        let (by_exact, by_prefix) = (by_exact.finish().unwrap(), by_prefix.finish().unwrap());
        assert_eq!(
            common::type_texts(&by_exact),
            ["Rel".to_string()],
            "{}",
            what
        );
        assert_eq!(
            common::type_texts(&by_exact),
            common::type_texts(&by_prefix),
            "{}",
            what
        );
    }

    // Whatever it is given in the analysis has no arguments; on a result it is given the causes of
    // the errors as written, and then their paths without arguments.
    let seen = Rc::new(Seen::default());
    let item: ItemTrait = parse_quote!(
        trait Tr<T: m::Bound<Rel>> {
            fn f(&self) -> m::Out<Rel2<'_>> {
                helper::f::<Rel3>()
            }
        }
    );
    let result = analyze_trait(&item, Rules::new().or_if(seen.rule()), true);
    assert!(seen.contains("m :: Bound") && seen.contains("helper :: f"));
    assert!(!seen.with_arguments(result.errors()), "{:?}", seen.paths());
    seen.asked.borrow_mut().clear();
    let result = result.allow(seen.rule());
    assert!(!seen.asked.borrow().is_empty());
    assert!(!seen.with_arguments(result.errors()), "{:?}", seen.paths());
}

#[test]
fn the_rule_is_asked_about_absolute_paths_of_types_except_core() {
    let seen = Rc::new(Seen::default());
    let item: ItemStruct = parse_quote!(
        struct S {
            a: ::std::vec::Vec<u8>,
            b: ::core::option::Option<u8>,
            c: ::serde::W,
            d: Rel,
        }
    );
    let result = analyze(&item, Rules::new().or_if(seen.rule()));
    assert!(result.errors().is_empty());
    // `::core` is never optional, and `$crate` is context-free: the rule is not asked.
    assert!(seen.contains(":: std :: vec :: Vec"));
    assert!(seen.contains(":: serde :: W"));
    assert!(seen.contains("Rel"));
    assert!(!seen.starts_with(":: core"));
}

/// A rule that panics when it is asked about `Boom`.
fn boom(cause: &Cause, _: Place) -> bool {
    if bare_path(cause).map_or(false, |path| path.is_ident("Boom")) {
        panic!("the rule is asked about Boom");
    }
    false
}

#[test]
fn a_rule_that_panics_is_a_panic_of_the_closure() {
    let mut leaker = Leaker::with_rule(Rules::new().or_if(boom as FnRule));
    let item: ItemStruct = parse_quote!(
        struct S {
            a: Rel,
            b: Boom,
        }
    );
    // The panic goes on, and the entry of `Rel` is rolled back.
    let outcome = catch_unwind(AssertUnwindSafe(|| leaker.intern_struct(&item)));
    assert!(outcome.is_err());
    let fine: ItemStruct = parse_quote!(
        struct S {
            a: Rel2,
        }
    );
    leaker.intern_struct(&fine);
    let result = leaker.result();
    assert!(result.errors().is_empty());
    assert_eq!(
        common::type_texts(&result.finish().unwrap()),
        ["Rel2".to_string()]
    );
}

// ── each path is asked about as the cause of its kind ────────────────────────────────────────

/// What a rule was asked about `item`, as `kind(path)` in the order asked, without the repeats.
fn asked_as(item: &ItemTrait, values: bool) -> Vec<String> {
    let seen = Rc::new(Seen::default());
    let mut leaker = Leaker::with_rule(Rules::new().or_if(seen.rule()));
    leaker.check_values = values;
    leaker.check_bodies = values;
    leaker.intern_trait(item);
    let _ = leaker.result();
    let mut found: Vec<String> = Vec::new();
    for (cause, _) in seen.asked.borrow().iter() {
        let text = common::kinded(cause);
        // The causes of the errors are asked about as written, after the analysis: leave them.
        if bare_path(cause).is_some() && !found.contains(&text) {
            found.push(text);
        }
    }
    found
}

#[test]
fn the_rule_is_asked_about_each_path_as_its_kind() {
    // A type, a trait, a value and a macro each as its kind. A qualified path is asked as its
    // trait, or as unknown if it has none.
    type Case<'a> = (&'a str, ItemTrait, bool, &'a [&'a str], &'a [&'a str]);
    let cases: [Case; 2] = [
        (
            "a type, a trait, a value and a macro",
            parse_quote!(
                trait Tr: SuperTr {
                    fn f(&self, a: TyArg, b: Box<dyn ObjTr>, c: ty_mac!(u8)) -> <Rel as QualTr>::X
                    where
                        Rel: WhereTr;
                    fn g(&self) {
                        let _: LetTy = val_mod::func(ValArg);
                        mac_path!();
                    }
                }
            ),
            true,
            &[
                "trait(SuperTr)",
                "type(TyArg)",
                "type(Box)",
                "trait(ObjTr)",
                "macro(ty_mac !)",
                "type(Rel)",
                "trait(QualTr)",
                "trait(WhereTr)",
                "type(LetTy)",
                "expr(val_mod :: func)",
                "macro(mac_path !)",
                "expr(ValArg)",
            ],
            // Never as another kind: a trait is not asked about as a type.
            &[
                "type(SuperTr)",
                "type(ObjTr)",
                "type(QualTr)",
                "type(WhereTr)",
                "trait(Rel)",
                "type(val_mod :: func)",
                "type(ValArg)",
                "trait(TyArg)",
            ],
        ),
        (
            "a qualified path",
            parse_quote!(
                trait Tr {
                    fn f(&self, a: <Rel as QualTr<u8>>::X, b: <Other>::Y);
                }
            ),
            false,
            &["trait(QualTr)", "unknown(:: Y)"],
            &["type(:: Y)", "trait(:: Y)", "unknown(QualTr)", "unknown(X)"],
        ),
    ];
    for (name, item, values, expected, unexpected) in cases {
        let asked = asked_as(&item, values);
        for expected in expected {
            assert!(
                asked.iter().any(|a| a == expected),
                "{}: {}: {:?}",
                name,
                expected,
                asked
            );
        }
        for unexpected in unexpected {
            assert!(
                !asked.iter().any(|a| a == unexpected),
                "{}: {}: {:?}",
                name,
                unexpected,
                asked
            );
        }
    }
}

#[test]
fn a_rule_for_one_kind_leaves_a_path_of_another_kind_leaked() {
    let item: ItemTrait = parse_quote!(
        trait Tr {
            fn f(&self, a: Same) -> u8
            where
                u8: Same2;
        }
    );
    // `Same` is a type: a rule for traits named `Same` does not allow it, one for types does.
    let only_traits: FnRule = |cause, _| matches!(cause, Cause::Trait(_));
    let only_types: FnRule = |cause, _| matches!(cause, Cause::Type(_));
    let by_traits = analyze_trait(&item, Rules::new().primitive().or_if(only_traits), false);
    let by_types = analyze_trait(&item, Rules::new().primitive().or_if(only_types), false);
    // Types allowed: no entry for `Same`; the trait `Same2` is an error.
    assert!(common::type_texts(&by_types.clone().recovered()).is_empty());
    assert_eq!(by_types.errors().len(), 1);
    assert!(matches!(
        by_types.errors()[0].cause,
        Cause::Type(_) | Cause::Trait(_)
    ));
    // Traits allowed: the type `Same` is an entry, and the trait is fine.
    assert!(by_traits.errors().is_empty());
    assert_eq!(
        common::type_texts(&by_traits.finish().unwrap()),
        ["Same".to_string()]
    );
}

#[test]
fn a_path_of_unknown_kind_is_asked_as_unknown_and_no_other_kind() {
    // The root of a `use` in a body is a path of unknown kind: a module, a type or a trait.
    let item: ItemTrait = parse_quote!(
        trait Tr {
            fn f(&self) {
                use inner::Thing;
            }
        }
    );
    let plain = analyze_trait(&item, Rules::new(), true);
    assert_eq!(plain.errors().len(), 1);
    assert!(matches!(plain.errors()[0].paths[..], [Cause::Unknown(_)]));
    // Each check asks once, as an unknown path, and not as a type and as a trait.
    let seen = Rc::new(Seen::default());
    let _ = analyze_trait(&item, Rules::new().or_if(seen.rule()), true);
    let asked: Vec<Cause> = seen
        .asked
        .borrow()
        .iter()
        .map(|(cause, _)| cause.clone())
        .filter(|cause| cause.starts_with(&parse_quote!(inner)))
        .collect();
    assert!(!asked.is_empty());
    assert!(asked
        .iter()
        .all(|cause| *cause == Cause::Unknown(parse_quote!(inner))));
    let only_unknown: FnRule =
        |cause, _| matches!(cause, Cause::Unknown(_)) && cause.starts_with(&parse_quote!(inner));
    // A rule for another kind does not match it.
    let only_traits: FnRule =
        |cause, _| matches!(cause, Cause::Trait(_)) && cause.starts_with(&parse_quote!(inner));
    let only_types: FnRule =
        |cause, _| matches!(cause, Cause::Type(_)) && cause.starts_with(&parse_quote!(inner));
    let only_macros: FnRule =
        |cause, _| matches!(cause, Cause::Macro(_)) && cause.starts_with(&parse_quote!(inner));
    // In the analysis.
    assert!(analyze_trait(&item, Rules::new().or_if(only_unknown), true)
        .errors()
        .is_empty());
    assert_eq!(
        analyze_trait(&item, Rules::new().or_if(only_traits), true)
            .errors()
            .len(),
        1
    );
    assert_eq!(
        analyze_trait(&item, Rules::new().or_if(only_types), true)
            .errors()
            .len(),
        1
    );
    assert_eq!(
        analyze_trait(&item, Rules::new().or_if(only_macros), true)
            .errors()
            .len(),
        1
    );
    // And on the result, also after a round trip through the serialization.
    let serialized: LeakerResult = parse2(plain.to_token_stream()).unwrap();
    for result in [plain.clone(), serialized] {
        assert!(result.clone().allow(only_unknown).errors().is_empty());
        assert_eq!(result.clone().allow(only_traits).errors().len(), 1);
        assert_eq!(result.clone().allow(only_types).errors().len(), 1);
        assert_eq!(result.allow(only_macros).errors().len(), 1);
    }
}

#[test]
fn the_built_in_rules_allow_a_path_of_unknown_kind_by_its_head() {
    let item: ItemTrait = parse_quote!(
        trait Tr {
            fn f(&self) {
                use core::mem;
                use inner::Thing;
            }
        }
    );
    // `std_crates()` allows the root `core`, and the error of `inner` stays.
    let result = analyze_trait(&item, Rules::new().std_crates(), true);
    assert_eq!(result.errors().len(), 1);
    assert!(matches!(result.errors()[0].paths[..], [Cause::Unknown(_)]));
    assert!(result.errors()[0].paths[0].starts_with(&parse_quote!(inner)));
}

// ── the built-in rules ───────────────────────────────────────────────────────────────────────

#[test]
fn the_built_in_rules_take_relative_paths_by_their_head() {
    let paths: &[(Path, [bool; 4])] = &[
        // primitive, prelude, std, crate
        (parse_quote!(u8), [true, false, false, false]),
        (parse_quote!(str), [true, false, false, false]),
        (parse_quote!(u8::MAX), [true, false, false, false]),
        (parse_quote!(f64), [true, false, false, false]),
        (parse_quote!(Vec), [false, true, false, false]),
        (parse_quote!(Clone), [false, true, false, false]),
        (parse_quote!(Rel), [false, false, false, false]),
        (parse_quote!(TryFrom), [false, false, false, false]),
        (parse_quote!(std), [false, false, true, false]),
        (parse_quote!(std::vec::Vec), [false, false, true, false]),
        (parse_quote!(core::fmt), [false, false, true, false]),
        (parse_quote!(alloc::vec), [false, false, true, false]),
        (parse_quote!(stdx::Rel), [false, false, false, false]),
        (parse_quote!(crate), [false, false, false, true]),
        (parse_quote!(crate::m::T), [false, false, false, true]),
        (parse_quote!(self::m::T), [false, false, false, false]),
        (parse_quote!(super::T), [false, false, false, false]),
        // A leading `::` is another crate, never one of these, except that `std_crates()` takes the
        // crates `std`, `core` and `alloc` also so.
        (parse_quote!(::u8), [false; 4]),
        (parse_quote!(::std::vec::Vec), [false, false, true, false]),
        (parse_quote!(::alloc::vec::Vec), [false, false, true, false]),
        (
            parse_quote!(::core::primitive::u8),
            [false, false, true, false],
        ),
        (parse_quote!(::stdx::Vec), [false; 4]),
        (parse_quote!(::crate::T), [false; 4]),
    ];
    let place = Place::default();
    for (path, expected) in paths {
        let causes = [
            Cause::Type(Type::Path(TypePath {
                qself: None,
                path: path.clone(),
            })),
            Cause::Trait(path.clone()),
            Cause::Macro(path.clone()),
            Cause::Expr(Expr::Path(ExprPath {
                attrs: Vec::new(),
                qself: None,
                path: path.clone(),
            })),
        ];
        // Whatever the kind of the cause is, as long as it carries a path.
        for cause in &causes {
            let found = [
                Rules::new().primitive().allow(cause, place),
                Rules::new().prelude().allow(cause, place),
                Rules::new().std_crates().allow(cause, place),
                Rules::new().crate_paths().allow(cause, place),
            ];
            assert_eq!(&found, expected, "{}", cause.to_token_stream());
        }
        // With generic arguments the cause is not a path of one of these: its paths are.
        let mut with_arguments = path.clone();
        if let Some(last) = with_arguments.segments.last_mut() {
            last.arguments = PathArguments::AngleBracketed(parse_quote!(<u8>));
        }
        let cause = Cause::Trait(with_arguments);
        let all = Rules::new()
            .primitive()
            .prelude()
            .std_crates()
            .crate_paths();
        assert!(!all.allow(&cause, place), "{}", cause.to_token_stream());
    }
    // A cause that is no path: not allowed by the paths.
    let all = Rules::new()
        .primitive()
        .prelude()
        .std_crates()
        .crate_paths();
    for cause in [
        Cause::Type(parse_quote!(&u8)),
        Cause::Type(parse_quote!(<u8 as Tr>::X)),
        Cause::Lifetime(parse_quote!('a)),
        Cause::Unreadable(quote!(u8)),
    ] {
        assert!(!all.allow(&cause, place), "{}", cause.to_token_stream());
    }
}

/// `external_crates()` allows the paths with a leading `::` (any crate, of any kind of cause that
/// carries a path), and no relative path.
#[test]
fn external_crates_allows_the_absolute_paths_only() {
    let place = Place::default();
    let rules = Rules::new().external_crates();
    for (path, expected) in [
        (parse_quote!(::foo::X), true),
        (parse_quote!(::foo), true),
        (parse_quote!(::std::vec::Vec), true),
        (parse_quote!(::core::primitive::u8), true),
        (parse_quote!(::u8), true),
        (parse_quote!(foo::X), false),
        (parse_quote!(u8), false),
        (parse_quote!(std::vec::Vec), false),
        (parse_quote!(crate::X), false),
        (parse_quote!(self::X), false),
    ] {
        let path: Path = path;
        let causes = [
            Cause::Type(Type::Path(TypePath {
                qself: None,
                path: path.clone(),
            })),
            Cause::Trait(path.clone()),
            Cause::Macro(path.clone()),
            Cause::Unknown(path.clone()),
            Cause::Expr(Expr::Path(ExprPath {
                attrs: Vec::new(),
                qself: None,
                path: path.clone(),
            })),
        ];
        for cause in &causes {
            assert_eq!(
                rules.allow(cause, place),
                expected,
                "{}",
                cause.to_token_stream()
            );
        }
        // With generic arguments the cause is not a bare path: its paths are asked.
        let mut with_arguments = path.clone();
        if let Some(last) = with_arguments.segments.last_mut() {
            last.arguments = PathArguments::AngleBracketed(parse_quote!(<u8>));
        }
        assert!(!rules.allow(&Cause::Trait(with_arguments), place));
    }
    // A cause that is no path is not allowed, and nothing else is allowed by it.
    for cause in [
        Cause::Type(parse_quote!(&u8)),
        Cause::Lifetime(parse_quote!('a)),
        Cause::Unreadable(quote!(::foo::X)),
    ] {
        assert!(!rules.allow(&cause, place), "{}", cause.to_token_stream());
    }
    // It chains with the other rules like the flags do.
    let both = Rules::new().external_crates().primitive();
    assert!(both.allow(&Cause::Trait(parse_quote!(u8)), place));
    assert!(both.allow(&Cause::Trait(parse_quote!(::foo::X)), place));
    assert!(!both.allow(&Cause::Trait(parse_quote!(Rel)), place));
}

#[test]
fn values_and_defaults_allow_everything_in_their_place() {
    let cause = Cause::Type(parse_quote!(&Rel<'_>));
    let rules = |values: bool, defaults: bool| {
        let mut rules = Rules::new();
        if values {
            rules = rules.values();
        }
        if defaults {
            rules = rules.defaults();
        }
        rules
    };
    for (in_value, in_default) in [(false, false), (true, false), (false, true), (true, true)] {
        let place = place(in_value, in_default);
        assert_eq!(rules(true, false).allow(&cause, place), in_value);
        assert_eq!(rules(false, true).allow(&cause, place), in_default);
        assert_eq!(
            rules(true, true).allow(&cause, place),
            in_value || in_default
        );
        assert!(!rules(false, false).allow(&cause, place));
    }
}

#[test]
fn values_writes_every_relative_path_in_a_value_as_it_is() {
    let item: ItemTrait = parse_quote!(
        trait Tr {
            const N: usize = inner::N;
            fn f(&self, a: Sig) {
                let x: Local<Arg> = inner::make::<Turbo>();
                helper!(x);
            }
        }
    );
    // Values are read: a relative type in a `let` is an entry, and the paths of the value are errors.
    let plain = analyze_trait(&item, Rules::new(), true);
    assert!(!plain.errors().is_empty());
    let leaked: Vec<String> = common::type_texts(&plain.clone().recovered())
        .into_iter()
        .collect();
    assert!(
        leaked.contains(&"Local < Arg >".to_string()),
        "{:?}",
        leaked
    );
    // With the flag everything in a value is written as it is, the types too: no entry for them,
    // and no error. A type of the signature is not in a value, and stays an entry (`usize` too:
    // the primitive types are not allowed).
    let by_flag = analyze_trait(&item, Rules::new().values(), true);
    assert!(by_flag.errors().is_empty());
    let kept: Vec<String> = common::type_texts(&by_flag.finish().unwrap())
        .into_iter()
        .collect();
    assert_eq!(kept, ["Sig", "usize"]);
    // Only the values of the item are read: without `check_values` the flag changes nothing.
    let without = analyze_trait(&item, Rules::new().values(), false);
    let plain_without = analyze_trait(&item, Rules::new(), false);
    assert_eq!(
        without.to_token_stream().to_string(),
        plain_without.to_token_stream().to_string()
    );
    // After the analysis it only drops the errors, and leaves the entries.
    let on_result = plain.allow(Rules::new().values());
    assert!(on_result.errors().is_empty());
    assert!(common::type_texts(&on_result.finish().unwrap())
        .iter()
        .any(|ty| ty == "Local < Arg >"));
}

#[test]
fn defaults_writes_every_relative_path_in_a_default_as_it_is() {
    let item: ItemStruct = parse_quote!(
        struct S<T = Rel<Other>, U = Wrap<'static>> {
            a: T,
            b: U,
            c: Plain,
        }
    );
    let plain = analyze(&item, Rules::new());
    let leaked: Vec<String> = common::type_texts(&plain.finish().unwrap())
        .into_iter()
        .collect();
    assert!(
        leaked.contains(&"Rel < Other >".to_string()),
        "{:?}",
        leaked
    );
    // The types of the defaults are written as they are, and the field `c` is an entry still.
    let by_flag = analyze(&item, Rules::new().defaults());
    assert!(by_flag.errors().is_empty());
    let kept: Vec<String> = common::type_texts(&by_flag.finish().unwrap())
        .into_iter()
        .collect();
    assert_eq!(kept, ["Plain"]);
}

#[test]
fn cause_starts_with_compares_identifiers_segment_by_segment() {
    let starts = |path: Path, prefix: Path| Cause::Trait(path).starts_with(&prefix);
    assert!(starts(parse_quote!(a::b::c), parse_quote!(a)));
    assert!(starts(parse_quote!(a::b::c), parse_quote!(a::b)));
    assert!(starts(parse_quote!(a::b::c), parse_quote!(a::b::c)));
    assert!(!starts(parse_quote!(a::b), parse_quote!(a::b::c)));
    assert!(!starts(parse_quote!(a::b::c), parse_quote!(b)));
    assert!(!starts(parse_quote!(ab::c), parse_quote!(a)));
    // Both have a leading `::`, or neither.
    assert!(starts(parse_quote!(::a::b), parse_quote!(::a)));
    assert!(!starts(parse_quote!(a::b), parse_quote!(::a)));
    assert!(!starts(parse_quote!(::a::b), parse_quote!(a)));
    // The arguments of the segments are not compared.
    assert!(starts(parse_quote!(a::B<u8>::c), parse_quote!(a::B)));
    assert!(starts(parse_quote!(a::B), parse_quote!(a::B<u8>)));
    // Every cause that is a path has one, and no other has.
    let path: Path = parse_quote!(a::b);
    assert_eq!(Cause::Trait(path.clone()).path(), Some(&path));
    assert_eq!(Cause::Macro(path.clone()).path(), Some(&path));
    assert_eq!(Cause::Type(parse_quote!(a::b)).path(), Some(&path));
    assert_eq!(Cause::Expr(parse_quote!(a::b)).path(), Some(&path));
    assert_eq!(Cause::Type(parse_quote!(&a::b)).path(), None);
    assert_eq!(Cause::Type(parse_quote!(<u8 as a>::b)).path(), None);
    assert_eq!(Cause::Expr(parse_quote!(a::b())).path(), None);
    assert_eq!(Cause::Lifetime(parse_quote!('a)).path(), None);
    assert!(!Cause::Type(parse_quote!(&a::b)).starts_with(&parse_quote!(a)));
}

// ── composition ──────────────────────────────────────────────────────────────────────────────

/// The serialized result of `item` (with `values`) after `allow` given to the config.
fn by_config<A: Allow>(item: &ItemTrait, allow: A, values: bool) -> String {
    analyze_trait(item, allow, values)
        .to_token_stream()
        .to_string()
}

#[test]
fn a_rule_built_by_flags_closures_tuples_or_or_is_one_rule() {
    let items: Vec<ItemTrait> = vec![
        parse_quote!(
            trait Tr: Clone {
                fn f(&self) -> u8;
            }
        ),
        parse_quote!(
            trait Tr {
                fn f(&self, a: Vec<u8>, b: Option<Rel>) -> bool;
            }
        ),
        parse_quote!(
            trait Tr: std::fmt::Debug {
                fn f(&self, a: core::option::Option<u8>);
            }
        ),
        parse_quote!(
            trait Tr {
                const C: usize = std::mem::size_of::<u8>() + crate::N;
                fn f(&self) -> usize {
                    core::cmp::max(1, Rel::N)
                }
            }
        ),
    ];
    // Whether something changes for some item: the comparison is not vacuous.
    let mut changes = [false; 4];
    for item in &items {
        for values in [false, true] {
            let plain = by_config(item, Rules::new(), values);
            let flags: [Rules; 4] = [
                Rules::new().primitive(),
                Rules::new().prelude(),
                Rules::new().std_crates(),
                Rules::new().crate_paths(),
            ];
            for (n, flag) in flags.iter().enumerate() {
                let what = format!("{} in {}", n, item.to_token_stream());
                let expected = by_config(item, flag.clone(), values);
                changes[n] |= expected != plain;
                // The same flag, as a rule of a closure, of a tuple, or added by `or()`.
                let inner = flag.clone();
                let closure = move |cause: &Cause, place: Place| inner.allow(cause, place);
                assert_eq!(
                    by_config(item, closure.clone(), values),
                    expected,
                    "{}",
                    what
                );
                assert_eq!(
                    by_config(item, Rules::new().or_if(closure), values),
                    expected,
                    "{}",
                    what
                );
                assert_eq!(
                    by_config(item, (flag.clone(), ()), values),
                    expected,
                    "{}",
                    what
                );
                assert_eq!(
                    by_config(item, ((), flag.clone()), values),
                    expected,
                    "{}",
                    what
                );
                assert_eq!(
                    by_config(item, Rules::new().or(flag.clone()), values),
                    expected,
                    "{}",
                    what
                );
                // On the result of the analysis without it: the same errors, however it is built.
                let plain = analyze_trait(item, Rules::new(), values);
                let on_result = plain.clone().allow(flag.clone());
                let inner = flag.clone();
                let by_closure = plain
                    .clone()
                    .allow(move |cause: &Cause, place: Place| inner.allow(cause, place));
                let by_tuple = plain.allow(((), flag.clone()));
                assert_eq!(
                    on_result.to_token_stream().to_string(),
                    by_closure.to_token_stream().to_string(),
                    "{}",
                    what
                );
                assert_eq!(
                    on_result.to_token_stream().to_string(),
                    by_tuple.to_token_stream().to_string(),
                    "{}",
                    what
                );
            }
        }
    }
    assert_eq!(changes, [true; 4]);
}

#[test]
fn a_run_time_choice_through_rules_keeps_one_type() {
    let item: ItemTrait = parse_quote!(
        trait Tr: Clone {
            fn f(&self, a: u8) -> Vec<char>;
        }
    );
    for (primitive, prelude) in [(false, false), (true, false), (false, true), (true, true)] {
        // The choice is made at run time, and the type of `config` does not depend on it.
        let mut config = Rules::new();
        if primitive {
            config = config.primitive();
        }
        if prelude {
            config = config.prelude();
        }
        let chosen = analyze_trait(&item, config.clone(), false);
        let allow = Rules::new();
        let expected = match (primitive, prelude) {
            (true, true) => analyze_trait(&item, allow.primitive().prelude(), false),
            (true, false) => analyze_trait(&item, allow.primitive(), false),
            (false, true) => analyze_trait(&item, allow.prelude(), false),
            (false, false) => analyze_trait(&item, allow, false),
        };
        let what = format!("primitive: {}, prelude: {}", primitive, prelude);
        assert_eq!(
            chosen.to_token_stream().to_string(),
            expected.to_token_stream().to_string(),
            "{}",
            what
        );
        // `Clone` is an error, `u8` and `Vec<char>` are entries, unless the flags allow them.
        assert_eq!(chosen.errors().is_empty(), prelude, "{}", what);
    }
    // The result takes the same type.
    let mut config = Rules::new();
    if item.supertraits.len() == 1 {
        config = config.prelude();
    }
    let result = analyze_trait(&item, Rules::new(), false).allow(config.primitive());
    assert!(result.errors().is_empty());
}

#[test]
fn or_if_takes_a_closure_without_annotations_and_or_takes_any_rule() {
    let item: ItemTrait = parse_quote!(
        trait Tr {
            fn f(&self, a: Rel) -> Other;
        }
    );
    // No type annotation on the closure.
    let by_closure = analyze_trait(
        &item,
        Rules::new().or_if(|cause, _| cause.starts_with(&parse_quote!(Rel))),
        false,
    );
    assert_eq!(
        common::type_texts(&by_closure.finish().unwrap()),
        ["Other".to_string()]
    );
    // Rules add up, in any order, and a struct of the user is a rule.
    struct OnlyOther;
    impl Allow for OnlyOther {
        fn allow(&self, cause: &Cause, _: Place) -> bool {
            cause.starts_with(&parse_quote!(Other))
        }
    }
    let both = Rules::new()
        .or_if(|cause, _| cause.starts_with(&parse_quote!(Rel)))
        .or(OnlyOther)
        .primitive();
    let result = analyze_trait(&item, both, false);
    assert!(result.errors().is_empty());
    assert!(result.finish().unwrap().is_empty());
    // And on a result: an annotated closure, a function, a tuple, `()`.
    let errored: ItemTrait = parse_quote!(
        trait Tr: Rel {}
    );
    let result = analyze_trait(&errored, Rules::new(), false);
    assert_eq!(result.errors().len(), 1);
    assert_eq!(result.clone().allow(()).errors().len(), 1);
    fn trait_rel(cause: &Cause, _: Place) -> bool {
        matches!(cause, Cause::Trait(_)) && cause.starts_with(&parse_quote!(Rel))
    }
    assert!(result.clone().allow(trait_rel).errors().is_empty());
    assert!(result
        .clone()
        .allow(|cause: &Cause, _: Place| cause.starts_with(&parse_quote!(Rel)))
        .errors()
        .is_empty());
    assert!(result.allow((OnlyOther, trait_rel)).errors().is_empty());
}

// ── accepting a conflict ─────────────────────────────────────────────────────────────────────

/// The trait that is relative in `f` and `g` below.
fn rel_tr() -> Cause {
    Cause::Trait(parse_quote!(RelTr))
}

/// The result of the analysis of an impl.
fn analyzed(item: &ItemImpl) -> LeakerResult {
    let mut leaker = Leaker::new();
    leaker.intern_impl(item);
    leaker.result()
}

/// `result` with these causes allowed.
fn allowing(mut result: LeakerResult, causes: &[Cause]) -> LeakerResult {
    for cause in causes {
        result = result.accept_cause(cause);
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

/// The entries of the Referrer of an item: the type and its generics.
fn entries(item: &ItemImpl) -> Vec<(String, String)> {
    let referrer = analyzed(item).recovered();
    (0..referrer.iter().count())
        .map(|ix| {
            (
                common::types(&referrer)
                    .nth(ix)
                    .unwrap()
                    .to_token_stream()
                    .to_string(),
                common::entry(&referrer, ix)
                    .generics
                    .to_token_stream()
                    .to_string(),
            )
        })
        .collect()
}

/// The errors of `f_and_g()`.
fn errors_of_f_and_g() -> Vec<ErrorItem> {
    analyzed(&f_and_g()).errors().to_vec()
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
fn the_entries_are_those_of_the_item_without_the_conflict() {
    let accepted = entries(&f_and_g());
    assert_eq!(accepted, entries(&g_only()));
    // `g` still leaks `::p::X<dyn RelTr>` as a whole, as `dyn RelTr` is in it, and the self type.
    let has = |ty: &str| accepted.iter().any(|entry| entry.0 == ty);
    assert!(has(":: p :: X < dyn RelTr >"));
    assert!(has(":: p :: S"));
}

#[test]
fn allowed_paths_would_change_more() {
    // The trait made context-free everywhere: `g` loses its required entry too.
    let config = Rules::new().or_if(starts_with(parse_quote!(RelTr)));
    let mut leaker = Leaker::with_rule(config);
    leaker.intern_impl(&f_and_g());
    let referrer = leaker.result().finish().unwrap();
    assert!(common::types(&referrer).all(|ty| *ty != parse_quote!(dyn RelTr)));
    // `g` loses its entry too: only the types of the crate `p` are left.
    let left: Vec<&Type> = common::types(&referrer).collect();
    assert_eq!(
        left,
        [&parse_quote!(::p::S), &parse_quote!(::p::X<dyn RelTr>)]
    );
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
    assert_eq!(
        result.finish().unwrap(),
        analyzed(&g_only()).finish().unwrap()
    );
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
        leaker.intern_struct(&item);
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
            fn f(_: [u8; { let c = |a: u8| h(a); 3 }]) {}
            fn g(_: ::p::V<u8>) {}
        }
    );
    // `u8` is the cause, `h` a note: allowing `u8` copies the array whole and `h` is not reached.
    // (`a` is the closure's own name.)
    let result = analyzed(&item);
    assert_eq!(result.errors().len(), 1);
    assert_eq!(result.errors()[0].cause, Cause::Type(parse_quote!(u8)));
    assert_eq!(
        result.errors()[0].notes.first().map(|(_, n)| n.as_str()),
        Some("it cannot be leaked as a whole because of `| a : u8 | h (a)`")
    );
    let causes = [Cause::Type(parse_quote!(u8))];
    let accepted = allowing(result.clone(), &causes);
    assert!(accepted.errors().is_empty());
    let referrer = accepted.finish().unwrap();
    // `::p::V<u8>` is an entry, as it is without the array, and nothing is for `h`; so is the self
    // type.
    let found: Vec<&Type> = common::types(&referrer).collect();
    assert_eq!(found, [&parse_quote!(::p::V<u8>), &parse_quote!(::p::S)]);
    let only_g: ItemImpl = parse_quote!(impl ::p::S { fn g(_: ::p::V<u8>) {} });
    assert_eq!(entries(&item), entries(&only_g));
    // `h` alone accepts nothing: it is not the cause of the error.
    let h = Cause::Expr(parse_quote!(h));
    assert_eq!(allowing(result, &[h]).errors().len(), 1);
}

// ── scopes ───────────────────────────────────────────────────────────────────────────────────

/// A type with the conflict of `f`.
fn conflict() -> Type {
    parse_quote!(::p::W<'_, dyn RelTr>)
}

/// A Leaker that has visited `conflict()` and `Rel<T>` in a scope inside the scope of no generics,
/// and `::core::marker::PhantomData<dyn RelTr>` too, whose `dyn RelTr` is an entry.
fn conflict_in_an_inner_scope() -> (LeakerResult, ContextId) {
    let mut leaker = Leaker::new();
    let (_, inner) = leaker.enter_generics(&Generics::default(), |view| {
        view.enter_generics(&parse_quote!(<T>), |inner| {
            inner.visit_type(&conflict());
            inner.visit_type(&parse_quote!(Rel<T>));
            inner.visit_type(&parse_quote!(::core::marker::PhantomData<dyn RelTr>));
        })
        .0
    });
    (leaker.result(), inner)
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
        // conflict is copied as written, though it holds the same `dyn RelTr`. The entries are in
        // the order of their contexts, which are numbered as the types are found: `dyn RelTr`
        // inside the conflict comes first.
        let referrer = result.recovered();
        let required: Vec<&Type> = common::types(&referrer).collect();
        assert_eq!(required, [&parse_quote!(dyn RelTr), &parse_quote!(Rel<T>)]);
        let mut copy: Vec<Type> = vec![
            conflict(),
            parse_quote!(Rel<T>),
            parse_quote!(::core::marker::PhantomData<dyn RelTr>),
        ];
        let mut visitor = common::visitor(&referrer, inner, |_, ix, _| parse_quote!(Entry<#ix>));
        for ty in &mut copy {
            visitor.visit_type_mut(ty);
        }
        let expected: Vec<Type> = vec![
            conflict(),
            parse_quote!(Entry<1usize>),
            parse_quote!(::core::marker::PhantomData<Entry<0usize>>),
        ];
        assert_eq!(copy, expected);
    }
    // Allowing the cause drops the error, and the scope stays, with the entries that it also holds.
    let (result, _) = conflict_in_an_inner_scope();
    let recovered = result.clone().recovered();
    assert_eq!(result.accept_cause(&rel_tr()).finish().unwrap(), recovered);
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
    // The error of `Rel<Self>` and its cause `Self`, which accepts it, are F79's
    // (`tests/findings/findings.rs`). `Rel<Self>` is no cause, so it does not.
    let ty: Type = parse_quote!(Rel<Self>);
    let self_ty = Cause::Type(parse_quote!(Self));
    let accepts =
        |causes: &[Cause], types: &[Type]| allowing(without_self(types), causes).finish().is_ok();
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
    let self_ty = Cause::Type(parse_quote!(Self));
    let conflict_error = (Reason::RelativePath, "RelTr".to_string());
    let mention_error = (Reason::SelfWithoutMeaning, "Self".to_string());
    // Two types, the conflict first, and one type that holds both, the mention first (a conflict
    // inside it is an error, too): the errors come in the order in which the analysis finds them.
    // The conflict and the type that names `Self` are copied as written: what is left is the
    // entries.
    type Case = (Vec<Type>, [(Reason, String); 2], &'static [&'static str]);
    let cases: [Case; 2] = [
        (
            vec![conflict(), parse_quote!(Rel<Self>)],
            [conflict_error.clone(), mention_error.clone()],
            &[],
        ),
        (
            vec![parse_quote!((Rel, Rel<Self>, ::p::W<'_, dyn RelTr>))],
            [mention_error.clone(), conflict_error.clone()],
            &["Rel"],
        ),
    ];
    for (types, both, entries) in cases {
        let case = types
            .iter()
            .map(|ty| ty.to_token_stream().to_string())
            .collect::<Vec<_>>()
            .join(" ; ");
        let left = |causes: &[Cause]| described(allowing(without_self(&types), causes).errors());
        // Both are reported, in the order in which they were found.
        assert_eq!(left(&[]), both, "{}", case);
        let recovered = without_self(&types).recovered();
        assert_eq!(common::type_texts(&recovered), entries, "{}", case);
        // Accepting the conflict leaves the mention.
        assert_eq!(
            left(&[rel_tr()]),
            std::slice::from_ref(&mention_error),
            "{}",
            case
        );
        // Accepting the mention alone leaves the conflict, and only it.
        assert_eq!(
            left(std::slice::from_ref(&self_ty)),
            std::slice::from_ref(&conflict_error),
            "{}",
            case
        );
        // Accepting both accepts the scope, with the recovered Referrer.
        assert_eq!(
            allowing(without_self(&types), &[rel_tr(), self_ty.clone()])
                .finish()
                .unwrap(),
            recovered,
            "{}",
            case
        );
    }
}

/// The entries of a Referrer, as text.
type Entries = Vec<String>;

/// The entries of `types` visited where `Self` has no meaning, and the errors they are with.
fn entries_without_self(types: &[Type]) -> (Entries, Vec<(Reason, String)>) {
    let result = without_self(types);
    let errors = described(result.errors());
    (common::type_texts(&result.recovered()), errors)
}

/// An accepted `Self` without meaning is not left in an entry: its impl of the Repeater would read
/// it as its own implementor. The outermost type that would be an entry and names it is copied as
/// written, and the types in it that do not name it are leaked as usual (`Rel` in `(Rel,
/// Rel<Self>)`); the error stays the one it was.
#[test]
fn a_type_with_a_self_without_meaning_is_copied_as_written() {
    let types: [(Type, &str, &[&str]); 8] = [
        (parse_quote!(Rel<Self>), "Self", &[]),
        (parse_quote!((Rel, Rel<Self>)), "Self", &["Rel"]),
        (parse_quote!(::p::W<Rel, Self>), "Self", &["Rel"]),
        (parse_quote!(Rel<[Self; 2]>), "Self", &[]),
        (parse_quote!(Vec<Rel<Self>>), "Self", &[]),
        (parse_quote!(Wrap<Self::Item>), "Self :: Item", &[]),
        (parse_quote!(Wrap<m!(Self)>), "m ! (Self)", &[]),
        // An optional entry: the Referrer may replace a path of another crate.
        (parse_quote!(::p::X<Self>), "Self", &[]),
    ];
    for (ty, cause, expected) in &types {
        let (entries, errors) = entries_without_self(std::slice::from_ref(ty));
        let expected: Entries = expected.iter().map(|ty| ty.to_string()).collect();
        assert_eq!(entries, expected, "{}", ty.to_token_stream());
        // The only error is that of `Self`: the correction reports nothing of its own.
        assert_eq!(errors, [(Reason::SelfWithoutMeaning, cause.to_string())]);
    }
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
        common::type_texts(&leaker.result().recovered()),
        ["Rel".to_string()]
    );
    // A type that is no entry, as it names a generic of the method: its parts are, those that do not
    // name `Self`.
    let item: ItemFn = parse_quote!(
        fn g<U>(x: (Vec<U>, Rel, Self)) {}
    );
    let mut leaker = Leaker::new();
    leaker.enter_generics(&Generics::default(), |view| {
        view.visit_item(&Item::Fn(item.clone()))
    });
    let mut found = common::type_texts(&leaker.result().recovered());
    found.sort();
    assert_eq!(found, ["Rel".to_string(), "Vec < U >".to_string()]);
}

#[test]
fn an_entry_whose_bounds_name_a_self_without_meaning_is_copied_whole() {
    let mut where_clause: Generics = parse_quote!(<T, U>);
    where_clause.where_clause = Some(parse_quote!(where T: AsRef<Self>));
    let self_ty = Cause::Type(parse_quote!(Self));
    let generics: [Generics; 2] = [parse_quote!(<T: AsRef<Self>, U>), where_clause];
    for generics in &generics {
        let mut leaker = Leaker::new();
        let (context, ()) = leaker.enter_generics(generics, |view| {
            view.visit_type(&parse_quote!(Rel<T>));
            view.visit_type(&parse_quote!(Other<U>));
            // An optional entry has the same bound.
            view.visit_type(&parse_quote!(::p::Opt<T>));
        });
        let result = leaker.result();
        assert_eq!(
            described(result.errors()),
            [(Reason::SelfWithoutMeaning, "Self".into())]
        );
        // `Rel<T>` would carry the bound with `Self`; `Other<U>` has nothing to do with it.
        let referrer = result.clone().recovered();
        assert_eq!(common::type_texts(&referrer), ["Other < U >".to_string()]);
        // No entry has the tokens and the context of `Rel<T>`, so it is no part either, and the
        // visitor leaves it, and the optional entry, as written.
        assert!(common::parts_of(&referrer).is_empty());
        let mut copy: Vec<Type> = vec![
            parse_quote!(Rel<T>),
            parse_quote!(Other<U>),
            parse_quote!(::p::Opt<T>),
        ];
        let mut visitor = common::visitor(&referrer, context, |_, ix, _| parse_quote!(Entry<#ix>));
        for ty in &mut copy {
            visitor.visit_type_mut(ty);
        }
        let expected: Vec<Type> = vec![
            parse_quote!(Rel<T>),
            parse_quote!(Entry<0usize>),
            parse_quote!(::p::Opt<T>),
        ];
        assert_eq!(copy, expected, "{}", generics.to_token_stream());
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
        common::type_texts(&result.recovered()),
        ["Other < U >".to_string()]
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
        common::type_texts(&result.recovered()),
        ["Rel2".to_string()]
    );
    let mut leaker = Leaker::new();
    leaker.enter_generics(&Generics::default(), |view| {
        view.enter_generics(&generics, |_| ());
    });
    let result = leaker.result();
    assert_eq!(result.errors().len(), 1);
    assert_eq!(
        common::type_texts(&result.recovered()),
        ["Rel2".to_string()]
    );
}

#[test]
fn a_value_with_a_self_without_meaning_is_copied_whole() {
    // With `check_values`: `Self` in a value is an error too, and a type there is an entry.
    let item: ItemFn = parse_quote!(
        fn f() {
            let a: Rel<Self> = Self::make();
            let b: Rel2 = ::core::default::Default::default();
        }
    );
    let mut leaker = Leaker::with_rule(Rules::new());
    leaker.check_values = true;
    leaker.check_bodies = true;
    leaker.enter_generics(&Generics::default(), |view| {
        view.visit_item(&Item::Fn(item.clone()))
    });
    let result = leaker.result();
    assert!(!result.errors().is_empty());
    assert!(result
        .errors()
        .iter()
        .all(|error| error.reason == Reason::SelfWithoutMeaning));
    let (result, _) = allow_every_cause(result);
    assert_eq!(
        common::type_texts(&result.finish().unwrap()),
        ["Rel2".to_string()]
    );
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
    let analyze = || {
        let mut leaker = Leaker::with_rule(Rules::new());
        leaker.check_values = true;
        leaker.check_bodies = true;
        leaker.intern_trait(&item);
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
    leaker.intern_trait(&item);
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

#[test]
fn every_type_that_leaves_the_lifetime_unconstrained_is_copied_whole() {
    let item: ItemTrait = parse_quote!(
        trait Tr {
            fn f<'b>(&self, x: (Wrap<'b>, Rel), y: [Other<'b>; 1]) -> &'b ::p::Out;
        }
    );
    let mut leaker = Leaker::new();
    leaker.intern_trait(&item);
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
        common::type_texts(&both.finish().unwrap()),
        ["Rel".to_string(), ":: p :: Out".to_string()]
    );
}

// ── what accepting does not make compile ─────────────────────────────────────────────────────

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
    let mut leaker = Leaker::with_rule(common::std_config());
    leaker.intern_trait(&item);
    let result = leaker.result();
    assert_eq!(
        described(result.errors()),
        [(Reason::UnprovableBound, "Self :: A0".to_string())]
    );
    let cause = result.errors()[0].cause.clone();
    assert_eq!(
        common::type_texts(&allowing(result, &[cause]).finish().unwrap()),
        ["R0".to_string()]
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
    let mut leaker = Leaker::with_rule(common::std_config());
    leaker.intern_impl(&on_self);
    let result = leaker.result();
    assert_eq!(
        described(result.errors()),
        [(Reason::UnprovableBound, "Self".to_string())]
    );
    let cause = result.errors()[0].cause.clone();
    let entries = common::type_texts(&allowing(result, &[cause]).finish().unwrap());
    assert!(entries.contains(&"R0".to_string()));
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
    let mut leaker = Leaker::with_rule(common::std_config());
    leaker.intern_impl(&on_other);
    let result = leaker.result();
    assert_eq!(
        described(result.errors()),
        [(Reason::UnprovableBound, "R1 < T >".to_string())]
    );
    let cause = result.errors()[0].cause.clone();
    let entries = common::type_texts(&allowing(result, &[cause]).finish().unwrap());
    assert!(entries.contains(&"R0".to_string()));
    assert!(!entries.iter().any(|ty| ty.starts_with("R1")));
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
    let mut leaker = Leaker::with_rule(Rules::new());
    leaker.check_values = true;
    leaker.check_bodies = true;
    leaker.intern_trait(&item);
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
    result = result.accept_cause(&lexed);
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
    // A cause read back from a serialized error is the one the analysis made.
    let made = errors_of_f_and_g().remove(0);
    let read: ErrorItem = parse2(made.to_token_stream()).unwrap();
    assert_eq!(read.cause, made.cause);
    assert_eq!(hash_of(&read.cause), hash_of(&made.cause));
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

// ── `Rules::std_crates()` ─────────────────────────────────────────────────────────────────────

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
    leaker.intern_trait(&paths());
    leaker.result()
}

#[test]
fn allow_std_crates_chains_with_allow() {
    let mut result = result_of_paths();
    let referrer = result.clone().recovered();
    // `Clone` and `Vec<..>` are prelude names, not paths of a crate. `Rel<'_>` is the error of the
    // type in `Vec<Rel<'_>>`, and of the one in the absolute `::std::vec::Vec<Rel<'_>>`.
    result = result.allow(Rules::new().std_crates());
    assert_eq!(
        common::causes_of(&result),
        [
            "Clone",
            "Vec < Rel < '_ > >",
            "Rel < '_ >",
            "Rel < '_ >",
            "MyType < '_ >"
        ]
    );
    assert_eq!(result.clone().recovered(), referrer);
    result = result
        .accept_cause(&Cause::Trait(parse_quote!(Clone)))
        .accept_cause(&Cause::Type(parse_quote!(Vec<Rel<'_>>)))
        .accept_cause(&Cause::Type(parse_quote!(Rel<'_>)))
        .accept_cause(&Cause::Type(parse_quote!(MyType<'_>)));
    assert!(result.errors().is_empty());
    assert_eq!(result.finish().unwrap(), referrer);
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
    // `u8` in the arguments of the value is a path of it, which a rule has to allow too (F50).
    let analyze = || {
        let config = Rules::new().primitive();
        let mut leaker = Leaker::with_rule(config);
        leaker.check_values = true;
        leaker.check_bodies = true;
        leaker.intern_trait(&item);
        leaker.result()
    };
    let result = analyze();
    let has = |kind: fn(&Cause) -> bool| result.errors().iter().any(|e| kind(&e.cause));
    assert!(has(|cause| matches!(cause, Cause::Macro(_))));
    assert!(has(|cause| matches!(cause, Cause::Expr(_))));
    let left = analyze().allow(Rules::new().std_crates());
    // Only `inner::make` is left: the macro and the value of `core` are dropped.
    assert_eq!(common::causes_of(&left), ["inner :: make"]);
    // `allow(cause)` drops each of them, whatever its kind.
    let (left, _) = allow_every_cause(analyze());
    assert!(left.errors().is_empty());
}

/// No built-in flag drops an error whose paths it does not allow: a cause with no path (`Self`, an
/// unreadable syntax, a closure), and one whose path is relative (`Wrap` of `Wrap<Self>` in a
/// default).
#[test]
fn no_built_in_flag_drops_an_error_whose_paths_it_does_not_allow() {
    let in_a_body: ItemTrait = parse_quote!(
        trait Tr {
            fn f(&self) {
                macro_rules! m {
                    () => {};
                }
            }
        }
    );
    let mut leaker = Leaker::with_rule(Rules::new());
    leaker.check_values = true;
    leaker.check_bodies = true;
    leaker.intern_trait(&in_a_body);
    let in_a_default: ItemTrait = parse_quote!(
        trait Tr<T = Wrap<Self>> {}
    );
    let results = [
        ("Self", without_self(&[parse_quote!(Rel<Self>)])),
        ("macro_rules! in a body", leaker.result()),
        (
            "Wrap<Self> in a default",
            leak_trait(Rules::new(), &in_a_default).0,
        ),
    ];
    for (name, result) in results {
        assert_eq!(result.errors().len(), 1, "{}", name);
        let flags = [
            ("prelude", Rules::new().prelude()),
            ("primitive", Rules::new().primitive()),
            ("std_crates", Rules::new().std_crates()),
            ("crate_paths", Rules::new().crate_paths()),
        ];
        let mut all = result.clone();
        for (flag, rules) in flags {
            let left = result.clone().allow(rules.clone());
            assert_eq!(left.errors().len(), 1, "{} with {}", name, flag);
            all = all.allow(rules);
        }
        assert_eq!(all.errors().len(), 1, "{} with all the flags", name);
    }
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
    // lifetime and is the same conflict wherever it is, so it is a part only because its error is
    // about it (the last, after those that the Referrer needs), whereas the trait object is in a
    // path that owns the cause, which tells it from the entry of the same tokens.
    let referrer = result.recovered();
    assert_eq!(
        common::parts_of(&referrer),
        [
            "type (dyn RelTr) 0 in (:: p :: W < '_ , dyn RelTr >) 0",
            "type (Rel < '_ >) 0"
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
    let referrer = result.clone().recovered();
    // The parts: the outer one, and those of the inner scopes.
    assert_eq!(common::parts_of(&referrer).len(), 4);
    // Allowing leaves the rest, and the Referrer as it is.
    result = result.accept_cause(&rel_tr()).accept_cause(&other);
    assert_eq!(result.errors().len(), 1);
    assert_eq!(result.errors()[0].cause, unknown);
    assert_eq!(result.clone().recovered(), referrer);
    result = result.accept_cause(&unknown);
    assert_eq!(result.finish().unwrap(), referrer);
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
            fn g(&self, x: ::core::marker::PhantomData<dyn RelTr>);
        }
    )
}

/// The result of the analysis of `item` and its context.
fn leak_trait<A: Allow>(config: A, item: &ItemTrait) -> (LeakerResult, ContextId) {
    let mut leaker = Leaker::with_rule(config);
    let context = leaker.intern_trait(item);
    (leaker.result(), context)
}

/// The Referrer of `item` and its context.
fn referrer_of<A: Allow>(config: A, item: &ItemTrait) -> (Referrer, ContextId) {
    let (result, context) = leak_trait(config, item);
    (result.recovered(), context)
}

/// The copy of `item` in which each entry is replaced by `Entry<ix>`.
fn rewritten(referrer: &Referrer, context: ContextId, item: &ItemTrait) -> ItemTrait {
    let mut copy = item.clone();
    common::visitor(referrer, context, |_, ix, _| parse_quote!(Entry<#ix>))
        .visit_item_trait_mut(&mut copy);
    copy
}

/// The errors of `item` as text, with the reason.
fn errors_of<A: Allow>(config: A, item: &ItemTrait) -> Vec<(Reason, String)> {
    described(leak_trait(config, item).0.errors())
}

#[test]
fn the_referrer_carries_the_part_that_is_copied_and_the_entries_beside_it() {
    let (result, context) = leak_trait(Rules::new(), &elided_and_bare());
    assert_eq!(result.errors().len(), 1);
    assert_eq!(result.errors()[0].cause, rel_tr());
    let referrer = result.recovered();
    // The part is the type that conflicts (the trait object, not the trait that is the cause),
    // inside the path that owns the elided lifetime. They are in the context of the types, which
    // use no generics (the id 0), not in the top-level one (the id 1).
    assert_eq!(context.index(), 1);
    assert_eq!(
        common::parts_of(&referrer),
        ["type (dyn RelTr) 0 in (:: p :: Elided < '_ , dyn RelTr >) 0".to_string()]
    );
    // `dyn RelTr` of `g` is an entry.
    assert_eq!(common::type_texts(&referrer), ["dyn RelTr".to_string()]);
    // Nothing is carried where nothing was copied as written.
    let mut leaker = Leaker::new();
    leaker.intern_trait(&parse_quote!(
        trait Tr {
            fn f(&self) -> Rel;
        }
    ));
    assert!(common::parts_of(&leaker.result().finish().unwrap()).is_empty());
    assert!(
        common::parts_of(&Leaker::with_rule(Rules::new()).result().finish().unwrap()).is_empty()
    );
}

#[test]
fn the_visitor_leaves_a_part_as_written_and_replaces_the_entry_with_the_same_tokens() {
    let item = elided_and_bare();
    let (result, context) = leak_trait(Rules::new(), &item);
    // `f` as written, `g` through the entry `dyn RelTr`: the same after a rule that allows the
    // crates (it changes no entry), as there is no optional entry.
    let expected: ItemTrait = parse_quote!(
        trait Fg {
            fn f(&self, x: ::p::Elided<'_, dyn RelTr>);
            fn g(&self, x: ::core::marker::PhantomData<Entry<0usize>>);
        }
    );
    let allowed = result.clone().allow(Rules::new().external_crates());
    assert_eq!(rewritten(&allowed.recovered(), context, &item), expected);
    assert_eq!(rewritten(&result.recovered(), context, &item), expected);
}

#[test]
fn an_item_beside_the_conflict_in_the_same_context_is_still_replaced() {
    // Both are interned without generics, so they share the context 0: the `dyn RelTr` of `S` is
    // the entry that is found inside the part of `Fg`.
    let tr = elided_and_bare();
    let st: ItemStruct = parse_quote!(
        struct S {
            y: ::core::marker::PhantomData<dyn RelTr>,
        }
    );
    let mut leaker = Leaker::new();
    let in_struct = leaker.intern_struct(&st);
    let in_trait = leaker.intern_trait(&tr);
    assert_eq!(in_struct, in_trait);
    let result = leaker.result();
    assert_eq!(result.errors().len(), 1);
    let referrer = result.recovered();
    assert_eq!(common::parts_of(&referrer).len(), 1);
    let mut copy = st.clone();
    common::visitor(&referrer, in_struct, |_, ix, _| parse_quote!(Entry<#ix>))
        .visit_item_struct_mut(&mut copy);
    let expected: ItemStruct = parse_quote!(
        struct S {
            y: ::core::marker::PhantomData<Entry<0usize>>,
        }
    );
    assert_eq!(copy, expected);
    // The trait is as written in `f` beside it.
    let expected: ItemTrait = parse_quote!(
        trait Fg {
            fn f(&self, x: ::p::Elided<'_, dyn RelTr>);
            fn g(&self, x: ::core::marker::PhantomData<Entry<0usize>>);
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
    assert_eq!(
        errors_of(common::std_config(), &item),
        [(Reason::UnprovableBound, "R1 < Y >".to_string())]
    );
    let (referrer, context) = referrer_of(common::std_config(), &item);
    // `R0` and `R1<Y>` (in the context of the method, which holds `Y`).
    assert_eq!(
        common::type_texts(&referrer),
        ["R0".to_string(), "R1 < Y >".to_string()]
    );
    // The type that conflicts is the `R1<Y>` in the bound of the argument, inside the `impl
    // Into<R1<Y>>` that owns the cause (`impl Trait` cannot be leaked), which has the same
    // context.
    assert_eq!(
        common::parts_of(&referrer),
        [format!(
            "type (R1 < Y >) {} whole predicate in (impl Into < R1 < Y > >) {}",
            common::entry_id(&referrer, 1),
            common::entry_id(&referrer, 1)
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
    assert_eq!(
        errors_of(common::std_config(), &item),
        [(Reason::UnprovableBound, "R1 < Y >".to_string())]
    );
    let (referrer, context) = referrer_of(common::std_config(), &item);
    // No type around it owns a cause of the conflict: no anchor.
    assert_eq!(
        common::parts_of(&referrer),
        [format!(
            "type (R1 < Y >) {} whole predicate",
            common::entry_id(&referrer, 1)
        )]
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
fn a_silent_correction_is_no_error_of_its_own_and_a_signature_makes_it_a_part() {
    // (1) `Self` has no meaning: the type that names it is copied as written instead of being an
    // entry. No entry has its tokens and context, so the Referrer has no part for it.
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
    let referrer = result.recovered();
    assert!(common::parts_of(&referrer).is_empty());
    let mut copy: Vec<Type> = vec![parse_quote!(Rel), parse_quote!(Wrap<Self>)];
    let mut visitor = common::visitor(&referrer, context, |_, ix, _| parse_quote!(Entry<#ix>))
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
    let (result, context) = leak_trait(common::std_config(), &item);
    assert_eq!(result.errors().len(), 1);
    assert_eq!(result.errors()[0].reason, Reason::UnconstrainedLifetime);
    let referrer = result.recovered();
    assert_eq!(common::parts_of(&referrer).len(), 1);
    // `u8` is allowed and not an entry, and `Held<'b>` is not, so nothing is replaced.
    assert_eq!(rewritten(&referrer, context, &item), item);
}

#[test]
fn the_part_is_the_type_that_conflicts_and_the_types_beside_it_stay_entries() {
    // Only `dyn RelTr` is the cause of the error: it is copied as written, in the path that owns
    // the elided lifetime. `Rel` beside it is an entry, and replaced.
    let list = |ty: Type| -> (Entries, Type) {
        let mut leaker = Leaker::with_rule(common::std_config());
        let (context, ()) =
            leaker.enter_generics(&Generics::default(), |view| view.visit_type(&ty));
        let result = leaker.result();
        let referrer = result.recovered();
        let entries = common::type_texts(&referrer);
        let mut copy = ty;
        common::visitor(&referrer, context, |_, ix, _| parse_quote!(Entry<#ix>))
            .visit_type_mut(&mut copy);
        (entries, copy)
    };
    let (entries, copy) = list(parse_quote!(::p::W<'_, dyn RelTr, Rel>));
    assert_eq!(entries, ["Rel".to_string()]);
    let expected: Type = parse_quote!(::p::W<'_, dyn RelTr, Entry<0usize>>);
    assert_eq!(copy, expected);
    // A type around the owner is no owner: the anchor is the path that holds the lifetime.
    let (entries, copy) = list(parse_quote!((Rel, ::p::W<'_, dyn RelTr>)));
    assert_eq!(entries, ["Rel".to_string()]);
    let expected: Type = parse_quote!((Entry<0usize>, ::p::W<'_, dyn RelTr>));
    assert_eq!(copy, expected);
    // So the same tokens beside the owner are an entry, inside the tuple that is no owner.
    let (entries, copy) = list(parse_quote!((
        ::p::W<'_, dyn RelTr>,
        ::core::marker::PhantomData<dyn RelTr>
    )));
    assert_eq!(entries[0], "dyn RelTr".to_string());
    let expected: Type = parse_quote!((
        ::p::W<'_, dyn RelTr>,
        ::core::marker::PhantomData<Entry<0usize>>
    ));
    assert_eq!(copy, expected);
    let (entries, copy) = list(parse_quote!(Vec<::p::W<'_, dyn RelTr>>));
    assert!(entries.is_empty());
    let expected: Type = parse_quote!(Vec<::p::W<'_, dyn RelTr>>);
    assert_eq!(copy, expected);
    // An elided reference owns its cause too.
    let (entries, copy) = list(parse_quote!((Rel, &dyn RelTr)));
    assert_eq!(entries, ["Rel".to_string()]);
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
    assert_eq!(
        errors_of(common::std_config(), &item),
        [
            (Reason::RelativePath, "RelTrait < R0 >".to_string()),
            (Reason::RelativePath, "Other < R0 >".to_string()),
        ]
    );
    let (referrer, context) = referrer_of(common::std_config(), &item);
    // Only `R0` of `f` is an entry. The bounds are copied as written, whole, with the `R0` in
    // them: a bound is a predicate of the item, and a leaked type in it may not be normalized. They
    // name no generic, so their context is that of `R0`.
    assert_eq!(common::type_texts(&referrer), ["R0".to_string()]);
    let none = common::entry_id(&referrer, 0);
    assert_eq!(
        common::parts_of(&referrer),
        [
            format!("trait (RelTrait < R0 >) {} whole predicate", none),
            format!("trait (Other < R0 >) {} whole predicate", none)
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
    let config = common::std_config();
    let (result, context) = leak_trait(config, &item);
    assert_eq!(
        result.errors()[0].cause.to_token_stream().to_string(),
        "for < 'a > RelTrait < 'a , R0 >"
    );
    let referrer = result.recovered();
    let parts = common::parts_of(&referrer);
    assert_eq!(parts.len(), 1);
    assert!(
        parts[0].starts_with("unreadable (for < 'a > RelTrait < 'a , R0 >)")
            && parts[0].ends_with("whole predicate"),
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
fn a_path_of_a_value_that_conflicts_is_written_as_it_is_and_the_types_in_it_are_leaked() {
    let item: ItemTrait = parse_quote!(
        trait Tr {
            fn f(&self, x: R0) -> R0 {
                inner::make::<R0>(x)
            }
        }
    );
    let leak = || {
        let mut leaker = Leaker::with_rule(common::std_config());
        leaker.check_values = true;
        leaker.check_bodies = true;
        let context = leaker.intern_trait(&item);
        (leaker.result(), context)
    };
    assert_eq!(
        described(leak().0.errors()),
        [(Reason::RelativePath, "inner :: make :: < R0 >".to_string())]
    );
    let (result, context) = leak();
    let referrer = result.recovered();
    assert_eq!(common::type_texts(&referrer), ["R0".to_string()]);
    assert!(common::parts_of(&referrer).is_empty());
    let expected: ItemTrait = parse_quote!(
        trait Tr {
            fn f(&self, x: Entry<0usize>) -> Entry<0usize> {
                inner::make::<Entry<0usize>>(x)
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
    let (referrer, context) = referrer_of(Rules::new(), &item);
    let mut copy = item.clone();
    common::visitor(&referrer, context, |_, ix, _| parse_quote!(Entry<#ix>))
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
    let (referrer, context) = referrer_of(common::std_config(), &item);
    // The two kinds of cause of a part (the path of a value is no part: it is written as it is, and
    // the types in it are analyzed), in a predicate or not, with an anchor or not, whole or not.
    let parts = common::parts_of(&referrer);
    assert!(parts.iter().any(|part| part.starts_with("trait (")));
    assert!(parts.iter().any(|part| part.starts_with("type (")));
    assert!(parts.iter().any(|part| part.contains(" predicate")));
    assert!(parts.iter().any(|part| part.contains(" in (")));
    assert!(parts.iter().any(|part| !part.contains("predicate")));
    assert!(parts.iter().any(|part| part.contains(" whole ")));
    assert!(parts.iter().any(|part| !part.contains(" whole ")));
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
    assert_eq!(common::parts_of(&read.0), parts);
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
fn a_referrer_without_parts_writes_no_list_of_parts() {
    let mut leaker = Leaker::new();
    leaker.intern_struct(&parse_quote!(
        struct S {
            a: Rel,
        }
    ));
    let referrer = leaker.result().finish().unwrap();
    assert!(common::parts_of(&referrer).is_empty());
    let tag = format!("\"type-leak-format-{}\"", Referrer::FORMAT_VERSION);
    assert_eq!(
        referrer.to_token_stream().to_string(),
        format!("{} 2 (Rel) declared < > ; ()", tag)
    );
    // Referrers that differ by the parts are not equal.
    let (with, _) = referrer_of(Rules::new(), &elided_and_bare());
    let mut leaker = Leaker::new();
    leaker.intern_trait(&parse_quote!(
        trait Fg {
            fn g(&self, x: ::core::marker::PhantomData<dyn RelTr>);
        }
    ));
    let without = leaker.result().finish().unwrap();
    assert!(common::parts_of(&without).is_empty());
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
        assert_eq!(common::parts_of(&good).len(), parts, "{}", text);
        let written = good.to_token_stream().to_string();
        let again = parse_str::<Referrer>(&written).unwrap();
        assert_eq!(again, good);
        assert_eq!(again.to_token_stream().to_string(), written);
    }
    let anchored =
        read("[type (R) 0 predicate in (Elided<'_, R>) 0, type (R) 0 in (&R) 0] (Rel)").unwrap();
    let context = common::context_ids(&anchored).into_iter().next().unwrap();
    assert_eq!(
        common::parts_of(&anchored),
        [
            format!(
                "type (R) {} predicate in (Elided < '_ , R >) {}",
                context, context
            ),
            format!("type (R) {} in (& R) {}", context, context)
        ]
    );
    // The tokens of a cause are all of the parentheses; a trailing comma is accepted. (F43 has more
    // refusals.)
    assert!(read("[trait (RelTr) 0,] (Rel)").is_ok());
    assert!(read("[type (R) 0 whole sig ('a) predicate in (&R) 0] (Rel)").is_ok());
    assert!(read("[] (Rel)").is_ok());
    for (text, message) in [
        // The words `cause` and `part` of the first form of the list are not read any more.
        ("[cause trait (RelTr) 0] (Rel)", "expected parentheses"),
        ("[part type (R) 0] (Rel)", "expected parentheses"),
        ("[what (R) 0] (Rel)", "unknown kind `what`"),
        ("[nope (R) 0] (Rel)", "unknown kind `nope`"),
        ("[type (R R) 0] (Rel)", "unexpected token in the cause"),
        (
            "[type (R) 3] (Rel)",
            "the context 3 of a part is not one of the contexts",
        ),
        // The anchor is a type in parentheses, with its context.
        ("[type (R) 0 in &R 0] (Rel)", "expected parentheses"),
        ("[type (R) 0 in (&R R) 0] (Rel)", "unexpected token"),
    ] {
        let error = read(text).err().unwrap_or_else(|| panic!("read: {}", text));
        assert!(error.contains(message), "{}: {}", text, error);
    }
}

#[test]
fn the_parts_are_deterministic() {
    let make = || {
        let (referrer, context) = referrer_of(Rules::new(), &elided_and_bare());
        (referrer.to_token_stream().to_string(), context)
    };
    assert_eq!(make(), make());
    let again = || {
        let (referrer, _) = referrer_of(common::std_config(), &r1_in_a_bound());
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
    let replace = |_: Type, ix: EntryId, _: &Context| -> Type { parse_quote!(Entry<#ix>) };
    let anchor: Type = parse_quote!(::p::Elided<'_, dyn RelTr>);
    let other: Type = parse_quote!(::core::marker::PhantomData<dyn RelTr>);
    // The anchor has the context of the type that is visited: the part is left as written in it,
    // and not in a type that is not the anchor.
    let referrer = read(0);
    let context = common::context_ids(&referrer).into_iter().next().unwrap();
    assert_eq!(
        common::expand(&referrer, context, anchor.clone(), replace),
        anchor.clone()
    );
    let expected: Type = parse_quote!(::core::marker::PhantomData<Entry<0usize>>);
    assert_eq!(
        common::expand(&referrer, context, other.clone(), replace),
        expected
    );
    // The anchor has another context (it names `T`): the type is not it, and `dyn RelTr` is the
    // entry it is everywhere else.
    let referrer = read(1);
    let expected: Type = parse_quote!(::p::Elided<'_, Entry<0usize>>);
    assert_eq!(
        common::expand(&referrer, context, anchor.clone(), replace),
        expected
    );
    // The anchor around a type that is not the part: the other types in it are looked up as usual.
    let referrer: Referrer = parse_str(&format!(
        "{} [type (dyn RelTr) 0 in (::p::Elided<'_, dyn RelTr, Rel>) 0] (dyn RelTr, Rel)",
        tag
    ))
    .unwrap();
    let ty: Type = parse_quote!(::p::Elided<'_, dyn RelTr, Rel>);
    let expected: Type = parse_quote!(::p::Elided<'_, dyn RelTr, Entry<1usize>>);
    assert_eq!(common::expand(&referrer, context, ty, replace), expected);
    // Outside any anchor, the same type in the same context is an entry.
    let ty: Type = parse_quote!(::p::Elided<'_, dyn RelTr, Other>);
    let expected: Type = parse_quote!(::p::Elided<'_, Entry<0usize>, Other>);
    assert_eq!(common::expand(&referrer, context, ty, replace), expected);
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
    assert_eq!(errors_of(common::std_config(), &item).len(), 1);
    let (referrer, context) = referrer_of(common::std_config(), &item);
    // `R0`, `R1<Y>` with the bound of `m`, and `R1<Y>` of `n`.
    assert_eq!(referrer.iter().count(), 3);
    assert_eq!(common::parts_of(&referrer).len(), 1);
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

// ── the names of the lists ───────────────────────────────────────────────────────────────────

/// The names that `Rules::primitive()` and `Rules::prelude()` allow, and the absolute paths that the
/// help of an error suggests for them (`KNOWN_NAMES` of `src/util/names.rs`).
///
/// An allowed path is the same on both sides, so it is not leaked: a bound that names it is fine,
/// and a type that names it has no entry. Without the option the same name is a relative path: an
/// error in a bound or a const expression, with a help that says what to write (``use
/// `::std::vec::Vec` instead of `Vec` ``), and an entry in a type. Dropping a name from a list
/// makes the Leaker leak a type that the Referrer's crate has anyway, so everything compiles still
/// and only the entries tell. So each name is checked in every form, with its option and without
/// it (audit mutants `cfg_primitive_f64`, `cfg_primitive_f32`, `cfg_primitive_i128`,
/// `cfg_prelude_drop`, `cfg_prelude_toowned` and `cfg_prelude_ord` drop one name of a list).
///
/// The table is written once, by the macro `absolute_paths!`, which makes both what the tests
/// compare with (the bare name, the absolute path as a string and the option that allows it) and
/// the code that uses the path, as a type, a trait bound or a value: a path with a typo fails to
/// compile, instead of being suggested to a user. The table of the library is private, so the help
/// of each name is asserted, and fails for a name whose path differs from the one here.
mod names {
    use crate::common::to_syn_error;
    use std::collections::BTreeSet;
    use syn::*;
    use type_leak::*;

    /// The option that allows a name: `Rules::primitive()`, `Rules::prelude()` (the prelude of every
    /// edition, rustc 1.56), or none (a later prelude, 2021, 2024, rustc 1.80: the crate of the
    /// copy may be of another edition or version).
    #[derive(Clone, Copy, PartialEq, Debug)]
    enum Group {
        Primitive,
        Prelude,
        Later,
    }

    /// `name = path, group, kind [what follows the path]`: `kind` is how the path is used, to check
    /// that it is what the name is: `ty` (a type), `trait` (a bound), or `value` (a function or a
    /// constructor).
    macro_rules! absolute_paths {
        ($($name:ident = $($seg:ident)::+, $group:ident, $kind:ident [$($rest:tt)*];)*) => {
            /// The bare names, the absolute path of each as the help writes it, and the option.
            fn table() -> Vec<(&'static str, String, Group)> {
                vec![$((
                    stringify!($name),
                    format!("::{}", [$(stringify!($seg)),+].join("::")),
                    Group::$group,
                )),*]
            }

            /// Every path used as what the name is. It only has to compile.
            #[test]
            #[allow(unused, drop_bounds)]
            fn every_absolute_path_is_valid() {
                $( absolute_paths!(@check $kind ($($seg)::+) [$($rest)*]); )*
            }
        };
        (@check ty ($($seg:ident)::+) [$($rest:tt)*]) => {{
            fn check(_: &::$($seg)::+ $($rest)*) {}
        }};
        (@check trait ($($seg:ident)::+) [$($rest:tt)*]) => {{
            fn check<T: ::$($seg)::+ $($rest)*>() {}
        }};
        (@check value ($($seg:ident)::+) [$($rest:tt)*]) => {{
            let _ = ::$($seg)::+ $($rest)*;
        }};
    }

    absolute_paths! {
        Box = std::boxed::Box, Prelude, ty [<u8>];
        String = std::string::String, Prelude, ty [];
        Vec = std::vec::Vec, Prelude, ty [<u8>];
        Option = core::option::Option, Prelude, ty [<u8>];
        Result = core::result::Result, Prelude, ty [<u8, u8>];
        Copy = core::marker::Copy, Prelude, trait [];
        Send = core::marker::Send, Prelude, trait [];
        Sized = core::marker::Sized, Prelude, trait [];
        Sync = core::marker::Sync, Prelude, trait [];
        Unpin = core::marker::Unpin, Prelude, trait [];
        Drop = core::ops::Drop, Prelude, trait [];
        Fn = core::ops::Fn, Prelude, trait [(u8)];
        FnMut = core::ops::FnMut, Prelude, trait [(u8)];
        FnOnce = core::ops::FnOnce, Prelude, trait [(u8)];
        Clone = core::clone::Clone, Prelude, trait [];
        PartialEq = core::cmp::PartialEq, Prelude, trait [];
        PartialOrd = core::cmp::PartialOrd, Prelude, trait [];
        Eq = core::cmp::Eq, Prelude, trait [];
        Ord = core::cmp::Ord, Prelude, trait [];
        AsRef = core::convert::AsRef, Prelude, trait [<str>];
        AsMut = core::convert::AsMut, Prelude, trait [<str>];
        Into = core::convert::Into, Prelude, trait [<u8>];
        From = core::convert::From, Prelude, trait [<u8>];
        Default = core::default::Default, Prelude, trait [];
        Iterator = core::iter::Iterator, Prelude, trait [];
        IntoIterator = core::iter::IntoIterator, Prelude, trait [];
        DoubleEndedIterator = core::iter::DoubleEndedIterator, Prelude, trait [];
        ExactSizeIterator = core::iter::ExactSizeIterator, Prelude, trait [];
        Extend = core::iter::Extend, Prelude, trait [<u8>];
        FromIterator = core::iter::FromIterator, Later, trait [<u8>];
        ToOwned = std::borrow::ToOwned, Prelude, trait [];
        ToString = std::string::ToString, Prelude, trait [];
        drop = core::mem::drop, Prelude, value [::<u8>];
        TryFrom = core::convert::TryFrom, Later, trait [<u8>];
        TryInto = core::convert::TryInto, Later, trait [<u8>];
        Future = core::future::Future, Later, trait [];
        // From rustc 1.64.
        IntoFuture = core::future::IntoFuture, Later, trait [];
        size_of = core::mem::size_of, Later, value [::<u8>];
        size_of_val = core::mem::size_of_val, Later, value [::<u8>];
        align_of = core::mem::align_of, Later, value [::<u8>];
        align_of_val = core::mem::align_of_val, Later, value [::<u8>];
        Some = core::option::Option::Some, Prelude, value [::<u8>];
        None = core::option::Option::None, Prelude, value [::<u8>];
        Ok = core::result::Result::Ok, Prelude, value [::<u8, u8>];
        Err = core::result::Result::Err, Prelude, value [::<u8, u8>];
        bool = core::primitive::bool, Primitive, ty [];
        char = core::primitive::char, Primitive, ty [];
        str = core::primitive::str, Primitive, ty [];
        u8 = core::primitive::u8, Primitive, ty [];
        u16 = core::primitive::u16, Primitive, ty [];
        u32 = core::primitive::u32, Primitive, ty [];
        u64 = core::primitive::u64, Primitive, ty [];
        u128 = core::primitive::u128, Primitive, ty [];
        usize = core::primitive::usize, Primitive, ty [];
        i8 = core::primitive::i8, Primitive, ty [];
        i16 = core::primitive::i16, Primitive, ty [];
        i32 = core::primitive::i32, Primitive, ty [];
        i64 = core::primitive::i64, Primitive, ty [];
        i128 = core::primitive::i128, Primitive, ty [];
        isize = core::primitive::isize, Primitive, ty [];
        f32 = core::primitive::f32, Primitive, ty [];
        f64 = core::primitive::f64, Primitive, ty [];
    }

    /// The names of the option `group`.
    fn names(group: Group) -> Vec<&'static str> {
        table()
            .into_iter()
            .filter(|(_, _, other)| *other == group)
            .map(|(name, _, _)| name)
            .collect()
    }

    /// Whether the config allows the primitive types and the prelude.
    type Names = (bool, bool);

    /// Where the name is written, and the headline of the error where that is one.
    struct Form {
        label: &'static str,
        source: String,
        /// The headline of the error without the option.
        headline: Option<String>,
        /// The entry without the option, if it is no error.
        entry: Option<String>,
    }

    fn forms(name: &str) -> Vec<Form> {
        vec![
            // The value of a const argument of a bound.
            Form {
                label: "const expression",
                source: format!("trait Tr: ::p::Sup<{{ {} }}> {{}}", name),
                headline: Some(format!("use absolute path: '{}'", name)),
                entry: None,
            },
            // The path of a const expression with more segments.
            Form {
                label: "path of a const expression",
                source: format!("trait Tr: ::p::Sup<{{ {}::X }}> {{}}", name),
                headline: Some(format!("use absolute path: '{}::X'", name)),
                entry: None,
            },
            Form {
                label: "supertrait",
                source: format!("trait Tr: {} {{}}", name),
                headline: Some(format!("use absolute path: '{}'", name)),
                entry: None,
            },
            // A type with an elided lifetime in an argument, which cannot be leaked as a whole.
            Form {
                label: "type with an elided lifetime",
                source: format!("trait Tr {{ fn f(&self, x: {}<&'_ ()>); }}", name),
                headline: Some(format!("use absolute path: '{}<&'_ ()>'", name)),
                entry: None,
            },
            Form {
                label: "type",
                source: format!("trait Tr {{ fn f(&self, x: {}<()>); }}", name),
                headline: None,
                entry: Some(format!("{} < () >", name)),
            },
            Form {
                label: "bare type",
                source: format!("struct S({});", name),
                headline: None,
                entry: Some(name.to_string()),
            },
        ]
    }

    /// The result of the analysis of the trait or struct `source` with `config`.
    fn leak<A: Allow>(config: A, source: &str) -> LeakerResult {
        let mut leaker = Leaker::with_rule(config);
        crate::common::intern_item(&mut leaker, &parse_str(source).unwrap());
        leaker.result()
    }

    /// The entries of the Referrer, or the headline of the error, with the options of `options`.
    fn analyze(options: Names, source: &str) -> std::result::Result<Vec<String>, String> {
        let (primitive, prelude) = options;
        let mut config = Rules::new();
        if primitive {
            config = config.primitive();
        }
        if prelude {
            config = config.prelude();
        }
        analyze_config(config, source)
    }

    fn analyze_config<A: Allow>(
        config: A,
        source: &str,
    ) -> std::result::Result<Vec<String>, String> {
        let referrer = leak(config, source).finish().map_err(|err| {
            err.iter()
                .map(|item| item.message.as_str())
                .collect::<Vec<_>>()
                .join("\n")
        })?;
        Ok(crate::common::type_texts(&referrer))
    }

    /// `form` is not allowed by `options`: an error, or an entry for the type.
    fn assert_not_allowed(options: Names, form: &Form, context: &str) {
        let result = analyze(options, &form.source);
        match (&form.headline, &form.entry) {
            (Some(headline), _) => assert_eq!(
                result.as_ref().err(),
                Some(headline),
                "{} is not allowed",
                context
            ),
            (_, Some(entry)) => assert_eq!(
                result.as_ref().ok(),
                Some(&vec![entry.clone()]),
                "{} is not allowed",
                context
            ),
            _ => unreachable!(),
        }
    }

    /// `names` are allowed by the config `allowing`, and not by the other config, nor by none.
    fn check(names: &[&str], allowing: Names, others: &[Names]) {
        for name in names {
            for form in forms(name) {
                let context = format!("{} in {}: {}", name, form.label, form.source);
                // Allowed: no error, and nothing leaked.
                assert_eq!(
                    analyze(allowing, &form.source),
                    Ok(Vec::new()),
                    "{}",
                    context
                );
                for options in others.iter().copied().chain([(false, false)]) {
                    assert_not_allowed(options, &form, &context);
                }
            }
        }
    }

    #[test]
    fn allow_primitive_allows_every_primitive_type() {
        let names = names(Group::Primitive);
        assert_eq!(names.len(), 17);
        check(&names, (true, false), &[(false, true)]);
    }

    #[test]
    fn allow_prelude_allows_every_name_of_the_prelude() {
        let names = names(Group::Prelude);
        assert_eq!(names.len(), 36);
        check(&names, (false, true), &[(true, false)]);
    }

    #[test]
    fn the_names_of_a_later_prelude_and_other_names_are_not_allowed_by_both_options() {
        // The names of a later prelude are not allowed by either option, nor by both: a crate of an
        // earlier edition has no such name. Another name is a relative path still, with both options.
        let later = names(Group::Later);
        assert_eq!(later.len(), 9);
        for name in later.into_iter().chain(["Foo", "Local", "Rel", "tryfrom"]) {
            for form in forms(name) {
                let context = format!("{} in {}: {}", name, form.label, form.source);
                assert_not_allowed((true, true), &form, &context);
            }
        }
    }

    /// The help gives the absolute path of every name of the prelude, of a later prelude and of a
    /// primitive type (`DoubleEndedIterator` had "e.g. `::my_crate::DoubleEndedIterator`"), in each
    /// form that is an error, as the help of the first error and not as a note.
    #[test]
    fn a_bare_name_gets_the_help_with_its_absolute_path() {
        let table = table();
        assert_eq!(table.len(), 62);
        assert_eq!(
            table
                .iter()
                .map(|(name, _, _)| name)
                .collect::<BTreeSet<_>>()
                .len(),
            62
        );
        for (name, absolute, _) in table {
            let help = format!("use `{}` instead of `{}`", absolute, name);
            for form in forms(name).iter().filter(|form| form.headline.is_some()) {
                let message = form.headline.as_ref().unwrap();
                let result = leak((), &form.source);
                let err = result.errors();
                let context = format!("{} as a {}: {}", name, form.label, form.source);
                assert_eq!(err.len(), 1, "{}", context);
                assert_eq!(&err[0].message, message, "{}", context);
                // It is the help of the error. It is no note: the notes say why.
                assert_eq!(err[0].help_text(), Some(help.as_str()), "{}", context);
                assert!(
                    err[0].notes.iter().all(|(_, note)| *note != help),
                    "{}",
                    context
                );
                // And it is in the `syn::Error`, which stable Rust gets: the notes follow the
                // headline, then the help.
                let rendered = to_syn_error(err).to_string();
                assert!(rendered.starts_with(message), "{}: {}", context, rendered);
                assert!(
                    rendered.ends_with(&format!("\n  = help: {}\n", help)),
                    "{}: {}",
                    context,
                    rendered
                );
            }
        }
    }

    #[test]
    fn allow_crate_and_allow_std_crates_allow_the_paths_that_start_with_their_roots() {
        // In a supertrait and in a type: the root decides, and the rest of the path does not.
        let analyze_with = |root: &str, source: &str| {
            let config = Rules::new();
            match root {
                "crate" => analyze_config(config.crate_paths(), source),
                "std crates" => analyze_config(config.std_crates(), source),
                _ => analyze_config(config, source),
            }
        };
        let roots: [(&str, &[&str]); 2] = [
            ("crate", &["crate::Foo", "crate::a::b::Foo"]),
            (
                "std crates",
                &["std::fmt::Debug", "core::fmt::Debug", "alloc::vec::Vec"],
            ),
        ];
        for (label, paths) in roots {
            for path in paths {
                for source in [
                    format!("trait Tr: {} {{}}", path),
                    format!("trait Tr {{ fn f(&self, x: {}<()>); }}", path),
                ] {
                    // Allowed: no error, and nothing leaked.
                    assert_eq!(analyze_with(label, &source), Ok(Vec::new()), "{}", source);
                    // Not by another root, nor by none.
                    let none = analyze_with("none", &source);
                    assert!(
                        none != Ok(Vec::new()),
                        "{} ({}) is leaked or an error",
                        source,
                        label
                    );
                }
            }
        }
        // A root decides by its first segment only: `my::core::X` is not `core::X`, and the absolute
        // `::core::X` is context-free whatever is allowed.
        let relative = analyze_config(
            Rules::new().std_crates().crate_paths(),
            "trait Tr { fn f(&self, x: my::core::X<()>); }",
        );
        assert_eq!(relative, Ok(vec!["my :: core :: X < () >".to_string()]));
        let absolute = analyze((false, false), "trait Tr: ::core::fmt::Debug {}");
        assert_eq!(absolute, Ok(Vec::new()));
    }
}

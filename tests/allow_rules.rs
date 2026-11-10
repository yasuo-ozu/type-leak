//! `Allow` and `Rules`: closures, tuples and the flags, which the convenience methods are made
//! of. The last tests are about what the rule is asked (each path as the cause of its kind, never
//! `Self` or a generic, never arguments in the analysis) and about a rule that panics (on a result,
//! and inside the closure of a scope).

mod common;

use common::{analyze_trait, bare_path, starts_with, AcceptCause};
use std::cell::{Cell, RefCell};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::rc::Rc;
use syn::*;
use template_quote::{quote, ToTokens};
use type_leak::*;

/// `Place` is non-exhaustive.
fn place(in_value: bool, in_default: bool) -> Place {
    let mut place = Place::default();
    place.in_value = in_value;
    place.in_default = in_default;
    place
}

fn entries(referrer: &Referrer) -> Vec<(String, bool)> {
    referrer
        .iter()
        .enumerate()
        .map(|(ix, ty)| (ty.to_token_stream().to_string(), referrer.must(ix)))
        .collect()
}

fn texts(items: &[ErrorItem]) -> Vec<String> {
    items
        .iter()
        .map(|item| item.to_token_stream().to_string())
        .collect()
}

fn paths(error: &ErrorItem) -> Vec<String> {
    error
        .paths
        .iter()
        .map(|path| path.to_token_stream().to_string())
        .collect()
}

/// The result of the analysis of the struct `item` with `config`.
fn analyze<A: Allow>(item: &ItemStruct, config: A) -> LeakerResult {
    let mut leaker = Leaker::from_config(config, Options::new());
    leaker.intern_struct(item, true);
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
    // Without a rule `Vec<u8>` is an entry, and `::serde::W` an optional one.
    let result = analyze(&item, Rules::new());
    assert!(result.errors().is_empty());
    assert_eq!(
        entries(&result.into_referrer()),
        [
            ("Vec < u8 >".to_string(), true),
            (":: serde :: W".to_string(), false)
        ]
    );
    // A rule for each of them: nothing is leaked, and there is no error.
    let config = Rules::new()
        .prelude()
        .primitive()
        .or_if(starts_with(parse_quote!(::serde)));
    let result = analyze(&item, config);
    assert!(result.errors().is_empty());
    assert!(entries(&result.into_referrer()).is_empty());
}

#[test]
fn a_rule_on_the_place_accepts_what_the_flag_does() {
    let item: ItemStruct = parse_quote!(
        struct S<'a, U = &'a Rel<'_>> {
            a: &'a Rel<'_>,
            u: U,
        }
    );
    let result = analyze(&item, Rules::new());
    // The type of the field and the default are two errors of one cause, one of them in a default.
    assert_eq!(result.errors().len(), 2);
    assert_eq!(result.errors()[0].cause, result.errors()[1].cause);
    let in_default: Vec<bool> = result.errors().iter().map(|e| e.in_default).collect();
    assert_eq!(in_default.iter().filter(|&&d| d).count(), 1);
    for error in result.errors() {
        assert_eq!(error.place(), place(error.in_value, error.in_default));
    }
    // A rule on the place drops the one in the default, as `defaults()` does.
    let by_rule = result
        .clone()
        .allow(|_: &Cause, place: Place| place.in_default);
    assert_eq!(by_rule.errors().len(), 1);
    assert!(!by_rule.errors()[0].in_default);
    let by_method = result.allow(Rules::new().defaults());
    assert_eq!(texts(by_rule.errors()), texts(by_method.errors()));
    // The same in the config: the default is analyzed with everything allowed, so its path `Rel` is
    // written as it is, and the field stays an error.
    let by_config = analyze(&item, Rules::new().defaults());
    assert_eq!(by_config.errors().len(), 1);
    assert!(!by_config.errors()[0].in_default);
    assert_eq!(texts(by_config.errors()), texts(by_rule.errors()));
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
        paths(&result.errors()[0]),
        ["Rel", "Send", "std :: fmt :: Debug"]
    );
    // The rules take the paths off one by one, as the rule of one call that accepts them all.
    let result = result.allow(Rules::new().prelude());
    assert_eq!(paths(&result.errors()[0]), ["Rel", "std :: fmt :: Debug"]);
    let result = result.allow(Rules::new().std_crates());
    assert_eq!(paths(&result.errors()[0]), ["Rel"]);
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
    assert_eq!(paths(error), ["Base", "Local"]);
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

    // An error without paths is accepted by its cause.
    let mut leaker = Leaker::new();
    leaker.enter_generics(&Generics::default(), |view| {
        view.visit_type(&parse_quote!(Self))
    });
    let result = leaker.result();
    assert_eq!(result.errors().len(), 1);
    assert!(result.errors()[0].paths.is_empty());
    let cause = Cause::Type(parse_quote!(Self));
    assert_eq!(result.errors()[0].cause, cause);
    assert!(result
        .clone()
        .allow(|c: &Cause, _: Place| *c == cause)
        .errors()
        .is_empty());
    assert_eq!(result.allow(()).errors().len(), 1);
}

#[test]
fn the_rules_may_borrow_what_is_around() {
    let item: ItemStruct = parse_quote!(
        struct S<'a> {
            a: Rel<'a, '_>,
            b: MyType<'_>,
            c: Other<'_>,
        }
    );
    let result = analyze(&item, Rules::new());
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
    let by_config = analyze(&item, config);
    assert_eq!(by_config.errors().len(), 1);
    assert_eq!(by_config.errors()[0].cause, causes[2]);
}

#[test]
fn a_rule_that_accepts_everything_leaves_no_error_and_dropping_the_bound_leaves_the_one_inside() {
    // The bound `Into<Self::A0>` is written as it is, and `Self::A0` in it has an error of its own,
    // which stays when the bound is dropped.
    let item: ItemTrait = parse_quote!(
        trait Tr<T: ::p::BT1<String>> {
            type A0;
            fn m<X>(&self)
            where
                X: Into<Self::A0>;
        }
    );
    let mut leaker = Leaker::new();
    leaker.intern_trait(&item, true);
    let result = leaker.result();
    assert_eq!(result.errors().len(), 2);
    let all = result.clone().allow(|_: &Cause, _: Place| true);
    assert!(all.errors().is_empty());
    // Dropping the bound alone leaves the error of `Self::A0`.
    let left = result.accept_cause(&Cause::Trait(parse_quote!(Into<Self::A0>)));
    assert_eq!(left.errors().len(), 1);
    assert_eq!(left.errors()[0].reason, Reason::UnprovableBound);
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

/// Three errors, each of a relative trait or type, with the paths `Rel`, `MyType` and `Other`.
fn three_errors() -> LeakerResult {
    let item: ItemStruct = parse_quote!(
        struct S<'a> {
            a: Rel<'a, '_>,
            b: MyType<'_>,
            c: Other<'_>,
        }
    );
    let result = analyze(&item, Rules::new());
    assert_eq!(paths_of(&result), [["Rel"], ["MyType"], ["Other"]]);
    result
}

fn paths_of(result: &LeakerResult) -> Vec<Vec<String>> {
    result.errors().iter().map(paths).collect()
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
    assert!(result
        .clone()
        .allow(|_: &Cause, _: Place| true)
        .errors()
        .is_empty());

    // A rule on paths, which accepts `Rel` and panics on `MyType`.
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        result.clone().allow(|cause: &Cause, _: Place| {
            let path = bare_path(cause);
            if path.map_or(false, |path| path.is_ident("MyType")) {
                panic!("the rule on paths");
            }
            path.map_or(false, |path| path.is_ident("Rel"))
        })
    }));
    assert!(outcome.is_err());
    assert_eq!(texts(result.errors()), before);
    assert!(result.clone().finish().is_err());
    let by_path = result.allow(|cause: &Cause, _: Place| {
        bare_path(cause).map_or(false, |path| {
            path.is_ident("MyType") || path.is_ident("Other")
        })
    });
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
        found.extend(paths(error));
    }
}

#[test]
fn the_rule_is_not_asked_about_self_or_a_generic() {
    for (allow_self, check_values) in [(true, false), (false, false), (true, true), (false, true)] {
        let seen = Rc::new(Seen::default());
        let mut leaker = Leaker::from_config(
            Rules::new().or_if(seen.rule()),
            if check_values {
                Options::new().check_values()
            } else {
                Options::new()
            },
        );
        leaker.intern_trait(&own_names(), allow_self);
        let result = leaker.result();
        let what = format!("allow_self {}, check_values {}", allow_self, check_values);
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
    // With `allow_self = false` a type that names `Self` cannot be an entry.
    fn by<A: Allow>(source: &str, allow: A) -> LeakerResult {
        let item: ItemStruct = parse_str(source).unwrap();
        let mut leaker = Leaker::from_config(allow, Options::new());
        leaker.intern_struct(&item, false);
        leaker.result()
    }
    for (source, relative) in [
        ("struct S { a: Rel<Self>, b: Self }", "Rel"),
        ("struct S { a: Box<Self>, b: Self }", "Box"),
    ] {
        let plain = by(source, Rules::new());
        assert_eq!(plain.errors().len(), 1, "{}", source);
        assert!(plain.clone().into_referrer().is_empty());
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
        assert!(by_rule.into_referrer().is_empty(), "{}", source);
        assert!(!seen.starts_with("Self"), "{}: {:?}", source, seen.paths());
        // Accepting `Self` only, on a result.
        let on_result = plain.clone().allow(|cause: &Cause, _: Place| {
            cause.path().map_or(false, |path| path.is_ident("Self"))
        });
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
    let mut tokens = base.into_referrer().to_token_stream();
    tokens.extend(quote!(@ [#error]));
    let result: LeakerResult = parse2(tokens).unwrap();
    assert_eq!(paths_of(&result), [["Self :: A"]]);
    let seen = Rc::new(Seen::default());
    let result = result.allow(seen.rule());
    // The error has a path, so its cause is not asked about, and the path of `Self` is not either.
    assert!(seen.asked.borrow().is_empty(), "{:?}", seen.paths());
    // A rule that accepts everything does not accept it either.
    let result = result.allow(|_: &Cause, _: Place| true);
    assert_eq!(paths_of(&result), [["Self :: A"]]);
}

#[test]
fn the_rule_sees_a_path_without_arguments_in_both_forms() {
    let item: ItemTrait = parse_quote!(
        trait Tr<T: m::Bound<Rel>> {
            fn f(&self, t: T);
        }
    );
    let exact = |cause: &Cause, _: Place| {
        cause.path().map_or(false, |path| {
            path.to_token_stream().to_string() == "m :: Bound"
        })
    };
    let prefix = starts_with(parse_quote!(m::Bound));
    let by_exact = analyze_trait(&item, Rules::new().or_if(exact), false);
    let by_prefix = analyze_trait(&item, Rules::new().or_if(prefix), false);
    assert!(by_exact.errors().is_empty());
    assert_eq!(texts(by_exact.errors()), texts(by_prefix.errors()));
    // `Rel` in the arguments is still analyzed: it is the one entry.
    let (by_exact, by_prefix) = (by_exact.into_referrer(), by_prefix.into_referrer());
    assert_eq!(entries(&by_exact), [("Rel".to_string(), true)]);
    assert_eq!(entries(&by_exact), entries(&by_prefix));

    // A turbofish in a value, which `check_values` reads.
    let item: ItemTrait = parse_quote!(
        trait Tr {
            fn f(&self) {
                helper::f::<Rel>();
            }
        }
    );
    let exact = |cause: &Cause, _: Place| {
        cause.path().map_or(false, |path| {
            path.to_token_stream().to_string() == "helper :: f"
        })
    };
    let prefix = starts_with(parse_quote!(helper::f));
    let by_exact = analyze_trait(&item, Rules::new().or_if(exact), true);
    let by_prefix = analyze_trait(&item, Rules::new().or_if(prefix), true);
    assert!(by_exact.errors().is_empty());
    assert_eq!(texts(by_exact.errors()), texts(by_prefix.errors()));
    let (by_exact, by_prefix) = (by_exact.into_referrer(), by_prefix.into_referrer());
    assert_eq!(entries(&by_exact), [("Rel".to_string(), true)]);
    assert_eq!(entries(&by_exact), entries(&by_prefix));

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
    let mut leaker = Leaker::from_config(Rules::new().or_if(boom), Options::new());
    let item: ItemStruct = parse_quote!(
        struct S {
            a: Rel,
            b: Boom,
        }
    );
    // The panic goes on, and the entry of `Rel` is rolled back.
    let outcome = catch_unwind(AssertUnwindSafe(|| leaker.intern_struct(&item, true)));
    assert!(outcome.is_err());
    let fine: ItemStruct = parse_quote!(
        struct S {
            a: Rel2,
        }
    );
    leaker.intern_struct(&fine, true);
    let result = leaker.result();
    assert!(result.errors().is_empty());
    assert_eq!(
        entries(&result.into_referrer()),
        [("Rel2".to_string(), true)]
    );
}

// ── each path is asked about as the cause of its kind ────────────────────────────────────────

/// What a rule was asked about `item`, as `kind(path)` in the order asked, without the repeats.
fn asked_as(item: &ItemTrait, values: bool, nested: bool) -> Vec<String> {
    let seen = Rc::new(Seen::default());
    let mut leaker = Leaker::from_config(
        Rules::new().or_if(seen.rule()),
        if values {
            Options::new().check_values()
        } else {
            Options::new()
        },
    );
    if nested {
        leaker.enter_mod("m", [], |view| view.visit_item(&Item::Trait(item.clone())));
    } else {
        leaker.intern_trait(item, true);
    }
    let _ = leaker.result();
    let mut found: Vec<String> = Vec::new();
    for (cause, _) in seen.asked.borrow().iter() {
        let kind = match cause {
            Cause::Type(_) => "type",
            Cause::Trait(_) => "trait",
            Cause::Expr(_) => "expr",
            Cause::Macro(_) => "macro",
            Cause::Unknown(_) => "unknown",
            other => panic!("{:?}", other),
        };
        let text = format!("{}({})", kind, cause.to_token_stream());
        // The causes of the errors are asked about as written, after the analysis: leave them.
        if bare_path(cause).is_some() && !found.contains(&text) {
            found.push(text);
        }
    }
    found
}

#[test]
fn the_rule_is_asked_about_a_type_a_trait_a_value_and_a_macro_each_as_its_kind() {
    let item: ItemTrait = parse_quote!(
        trait Tr: SuperTr {
            fn f(&self, a: TyArg, b: Box<dyn ObjTr>, c: ty_mac!(u8)) -> <Rel as QualTr>::X
            where
                Rel: WhereTr;
            fn g(&self) {
                let _: LetTy = val_mod::func(ValArg);
                mac_path!();
            }
        }
    );
    let asked = asked_as(&item, true, false);
    for expected in [
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
    ] {
        assert!(
            asked.iter().any(|a| a == expected),
            "{}: {:?}",
            expected,
            asked
        );
    }
    // Never as another kind: a trait is not asked about as a type.
    for unexpected in [
        "type(SuperTr)",
        "type(ObjTr)",
        "type(QualTr)",
        "type(WhereTr)",
        "trait(Rel)",
        "type(val_mod :: func)",
        "type(ValArg)",
        "trait(TyArg)",
    ] {
        assert!(
            !asked.iter().any(|a| a == unexpected),
            "{}: {:?}",
            unexpected,
            asked
        );
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
    let only_traits = |cause: &Cause, _: Place| matches!(cause, Cause::Trait(_));
    let only_types = |cause: &Cause, _: Place| matches!(cause, Cause::Type(_));
    let by_traits = analyze_trait(&item, Rules::new().primitive().or_if(only_traits), false);
    let by_types = analyze_trait(&item, Rules::new().primitive().or_if(only_types), false);
    // Types allowed: no entry for `Same`; the trait `Same2` is an error.
    assert!(entries(&by_types.clone().into_referrer()).is_empty());
    assert_eq!(by_types.errors().len(), 1);
    assert!(matches!(
        by_types.errors()[0].cause,
        Cause::Type(_) | Cause::Trait(_)
    ));
    // Traits allowed: the type `Same` is an entry, and the trait is fine.
    assert!(by_traits.errors().is_empty());
    assert_eq!(
        entries(&by_traits.into_referrer()),
        [("Same".to_string(), true)]
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
    let only_unknown = |cause: &Cause, _: Place| {
        matches!(cause, Cause::Unknown(_)) && cause.starts_with(&parse_quote!(inner))
    };
    // A rule for another kind does not match it.
    let only_traits = |cause: &Cause, _: Place| {
        matches!(cause, Cause::Trait(_)) && cause.starts_with(&parse_quote!(inner))
    };
    let only_types = |cause: &Cause, _: Place| {
        matches!(cause, Cause::Type(_)) && cause.starts_with(&parse_quote!(inner))
    };
    let only_macros = |cause: &Cause, _: Place| {
        matches!(cause, Cause::Macro(_)) && cause.starts_with(&parse_quote!(inner))
    };
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
fn a_qualified_path_is_asked_as_its_trait_or_as_unknown_without_one() {
    let item: ItemTrait = parse_quote!(
        trait Tr {
            fn f(&self, a: <Rel as QualTr<u8>>::X, b: <Other>::Y);
        }
    );
    let asked = asked_as(&item, false, false);
    for expected in ["trait(QualTr)", "unknown(:: Y)"] {
        assert!(
            asked.iter().any(|a| a == expected),
            "{}: {:?}",
            expected,
            asked
        );
    }
    for unexpected in ["type(:: Y)", "trait(:: Y)", "unknown(QualTr)", "unknown(X)"] {
        assert!(
            !asked.iter().any(|a| a == unexpected),
            "{}: {:?}",
            unexpected,
            asked
        );
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
    let leaked: Vec<String> = entries(&plain.clone().into_referrer())
        .into_iter()
        .map(|(ty, _)| ty)
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
    let kept: Vec<String> = entries(&by_flag.into_referrer())
        .into_iter()
        .map(|(ty, _)| ty)
        .collect();
    assert_eq!(kept, ["Sig", "usize"]);
    // Only the values of the item are read: without `check_values()` the flag changes nothing.
    let without = analyze_trait(&item, Rules::new().values(), false);
    let plain_without = analyze_trait(&item, Rules::new(), false);
    assert_eq!(
        without.to_token_stream().to_string(),
        plain_without.to_token_stream().to_string()
    );
    // After the analysis it only drops the errors, and leaves the entries.
    let on_result = plain.allow(Rules::new().values());
    assert!(on_result.errors().is_empty());
    assert!(entries(&on_result.into_referrer())
        .iter()
        .any(|(ty, _)| ty == "Local < Arg >"));
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
    let leaked: Vec<String> = entries(&plain.into_referrer())
        .into_iter()
        .map(|(ty, _)| ty)
        .collect();
    assert!(
        leaked.contains(&"Rel < Other >".to_string()),
        "{:?}",
        leaked
    );
    // The types of the defaults are written as they are, and the field `c` is an entry still.
    let by_flag = analyze(&item, Rules::new().defaults());
    assert!(by_flag.errors().is_empty());
    let kept: Vec<String> = entries(&by_flag.into_referrer())
        .into_iter()
        .map(|(ty, _)| ty)
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
        entries(&by_closure.into_referrer()),
        [("Other".to_string(), true)]
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
    assert!(result.into_referrer().is_empty());
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

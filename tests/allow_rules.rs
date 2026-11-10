//! The rules of `allow()` and `allow_path_if()` of `Allow` and `LeakerResult`: closures,
//! which the convenience methods (`allow_prelude()`, `allow_default()`, ..) are made of. The last
//! tests are about what a path rule is asked (never `Self` or a generic, never arguments) and about
//! a rule that panics (on a result, and inside the closure of a scope).

mod common;

use common::{analyze_trait, starts_with};
use std::cell::{Cell, RefCell};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::rc::Rc;
use syn::*;
use template_quote::ToTokens;
use type_leak::*;

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
fn analyze<P: Fn(&Path) -> bool, E: Fn(&ErrorItem) -> bool>(
    item: &ItemStruct,
    config: Allow<P, E>,
) -> LeakerResult {
    let mut leaker = Leaker::from_config(config, false, false);
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
    let result = analyze(&item, Allow::new());
    assert!(result.errors().is_empty());
    assert_eq!(
        entries(&result.into_referrer()),
        [
            ("Vec < u8 >".to_string(), true),
            (":: serde :: W".to_string(), false)
        ]
    );
    // A rule for each of them: nothing is leaked, and there is no error.
    let config = Allow::new()
        .allow_prelude()
        .allow_primitive()
        .allow_path_if(starts_with(parse_quote!(::serde)));
    let result = analyze(&item, config);
    assert!(result.errors().is_empty());
    assert!(entries(&result.into_referrer()).is_empty());
}

#[test]
fn an_error_rule_accepts_what_the_convenience_method_does() {
    let item: ItemStruct = parse_quote!(
        struct S<'a, U = &'a Rel<'_>> {
            a: &'a Rel<'_>,
            u: U,
        }
    );
    let result = analyze(&item, Allow::new());
    // The type of the field and the default are two errors of one cause, one of them in a default.
    assert_eq!(result.errors().len(), 2);
    assert_eq!(result.errors()[0].cause, result.errors()[1].cause);
    let in_default: Vec<bool> = result.errors().iter().map(|e| e.in_default).collect();
    assert_eq!(in_default.iter().filter(|&&d| d).count(), 1);
    // A rule on the errors drops the one in the default, as `allow_default()` does.
    let mut by_rule = result.clone();
    by_rule.allow(|error| error.in_default);
    assert_eq!(by_rule.errors().len(), 1);
    assert!(!by_rule.errors()[0].in_default);
    let mut by_method = result;
    by_method.allow_default();
    assert_eq!(texts(by_rule.errors()), texts(by_method.errors()));
    // The same in the config.
    let by_config = analyze(&item, Allow::new().allow_default());
    assert_eq!(by_config.errors().len(), 1);
    assert!(!by_config.errors()[0].in_default);
    assert_eq!(texts(by_config.errors()), texts(by_rule.errors()));
}

#[test]
fn path_rules_add_up_on_a_result() {
    let item: ItemStruct = parse_quote!(
        struct S<'a> {
            a: &'a Rel<'_, Vec<u8>>,
        }
    );
    let mut result = analyze(&item, Allow::new());
    assert_eq!(result.errors().len(), 1);
    assert_eq!(paths(&result.errors()[0]), ["Rel", "Vec", "u8"]);
    // The rules take the paths off one by one, as the rule of one call that accepts them all.
    result.allow_prelude();
    assert_eq!(paths(&result.errors()[0]), ["Rel", "u8"]);
    result.allow_primitive();
    assert_eq!(paths(&result.errors()[0]), ["Rel"]);
    // The error goes with the last.
    result.allow_path_if(starts_with(parse_quote!(Rel)));
    assert!(result.errors().is_empty());
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
    let result = analyze(&item, Allow::new());
    let causes: Vec<Cause> = result.errors().iter().map(|e| e.cause.clone()).collect();
    assert_eq!(causes.len(), 3);
    // A rule of a result borrows a list of causes that is not 'static.
    let mut left = result.clone();
    left.allow(|error| causes[..2].contains(&error.cause));
    assert_eq!(left.errors().len(), 1);
    assert_eq!(left.errors()[0].cause, causes[2]);
    // So does a rule of a config, which the Leaker keeps while it analyzes.
    let allowed: Vec<Path> = vec![parse_quote!(MyType)];
    let own = [causes[0].clone()];
    let config = Allow::new()
        .allow_path_if(|path| allowed.iter().any(|a| starts_with(a.clone())(path)))
        .allow(|error| own.contains(&error.cause));
    let by_config = analyze(&item, config);
    assert_eq!(by_config.errors().len(), 1);
    assert_eq!(by_config.errors()[0].cause, causes[2]);
}

#[test]
fn a_rule_that_accepts_everything_leaves_no_error_and_no_nested_error() {
    // The bound `Into<Self::A0>` is copied whole, with an error inside it that comes up when the
    // bound is dropped.
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
    assert!(result.errors().iter().any(|error| !error.nested.is_empty()));
    let mut all = result.clone();
    all.allow(|_| true);
    assert!(all.errors().is_empty());
    // Dropping the bound alone brings the nested error up.
    let mut brought_up = result;
    brought_up.allow(|error| error.cause == Cause::Trait(parse_quote!(Into<Self::A0>)));
    assert_eq!(brought_up.errors().len(), 1);
}

#[test]
fn a_config_with_rules_has_a_debug_form() {
    let config = Allow::new()
        .allow_primitive()
        .allow_default()
        .allow_path_if(starts_with(parse_quote!(::serde)))
        .allow(|error| error.in_value);
    assert_eq!(format!("{:?}", config), "Allow { .. }");
    assert_eq!(format!("{:?}", Allow::new()), "Allow { .. }");
}

#[test]
fn a_path_rule_of_a_config_is_called_on_every_check_of_a_path() {
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
        move |path: &Path| {
            calls.set(calls.get() + 1);
            if path.is_ident("Foo") {
                saw_foo.set(true);
            }
            false
        }
    };
    let result = analyze(&item, Allow::new().allow_path_if(rule));
    assert!(result.errors().is_empty());
    // `Foo` is checked for each of the fields, and nothing is remembered between the checks.
    assert!(calls.get() >= 2, "{} calls", calls.get());
    assert!(saw_foo.get());
}

// ── what a path rule is asked, and what a panic of a rule does ───────────────────────────────

/// The paths that a rule saw, as tokens, and whether one of them had generic arguments.
#[derive(Default)]
struct Seen {
    paths: RefCell<Vec<String>>,
    with_arguments: Cell<bool>,
}

impl Seen {
    /// A rule that records what it is asked, and accepts nothing.
    fn rule(self: &Rc<Self>) -> impl Fn(&Path) -> bool {
        let seen = self.clone();
        move |path: &Path| {
            seen.paths
                .borrow_mut()
                .push(path.to_token_stream().to_string());
            if path.segments.iter().any(|seg| !seg.arguments.is_none()) {
                seen.with_arguments.set(true);
            }
            false
        }
    }

    fn contains(&self, path: &str) -> bool {
        self.paths.borrow().iter().any(|seen| seen == path)
    }

    /// Whether a path that starts with the segment `head` was asked about.
    fn starts_with(&self, head: &str) -> bool {
        let longer = format!("{} ::", head);
        self.paths
            .borrow()
            .iter()
            .any(|seen| seen == head || seen.starts_with(&longer))
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
    let result = analyze(&item, Allow::new());
    assert_eq!(paths_of(&result), [["Rel"], ["MyType"], ["Other"]]);
    result
}

fn paths_of(result: &LeakerResult) -> Vec<Vec<String>> {
    result.errors().iter().map(paths).collect()
}

#[test]
fn a_rule_that_panics_on_a_result_leaves_the_errors_as_they_were() {
    let result = three_errors();
    let before = texts(result.errors());

    // A rule on errors, which accepts the first and panics on the second.
    let mut by_error = result.clone();
    let calls = Cell::new(0);
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        by_error.allow(|_| {
            calls.set(calls.get() + 1);
            if calls.get() == 2 {
                panic!("the rule on errors");
            }
            true
        });
    }));
    assert!(outcome.is_err());
    assert_eq!(calls.get(), 2);
    assert_eq!(texts(by_error.errors()), before);
    // The result is still usable.
    by_error.allow(|_| true);
    assert!(by_error.errors().is_empty());

    // A rule on paths, which accepts `Rel` and panics on `MyType`.
    let mut by_path = result;
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        by_path.allow_path_if(|path| {
            if path.is_ident("MyType") {
                panic!("the rule on paths");
            }
            path.is_ident("Rel")
        });
    }));
    assert!(outcome.is_err());
    assert_eq!(texts(by_path.errors()), before);
    assert!(by_path.clone().finish().is_err());
    by_path.allow_path_if(|path| path.is_ident("MyType") || path.is_ident("Other"));
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

/// The paths of `errors` and of the errors inside them, as tokens.
fn all_paths(errors: &[ErrorItem], found: &mut Vec<String>) {
    for error in errors {
        found.extend(paths(error));
        all_paths(&error.nested, found);
    }
}

#[test]
fn the_path_rule_is_not_asked_about_self_or_a_generic() {
    for (allow_self, check_values) in [(true, false), (false, false), (true, true), (false, true)] {
        let seen = Rc::new(Seen::default());
        let mut leaker =
            Leaker::from_config(Allow::new().allow_path_if(seen.rule()), check_values, false);
        leaker.intern_trait(&own_names(), allow_self);
        let result = leaker.result();
        let what = format!("allow_self {}, check_values {}", allow_self, check_values);
        assert!(seen.contains("Rel"), "{}: {:?}", what, seen.paths);
        for head in ["Self", "T", "Vec", "Iterator"] {
            // `Iterator` is the bound of `T`, a relative path that the rule may see.
            let asked = seen.starts_with(head);
            assert_eq!(asked, head == "Iterator", "{}: {:?}", what, seen.paths);
        }
        assert!(!seen.with_arguments.get(), "{}: {:?}", what, seen.paths);
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
    fn by<P: Fn(&Path) -> bool>(source: &str, allow: Allow<P>) -> LeakerResult {
        let item: ItemStruct = parse_str(source).unwrap();
        let mut leaker = Leaker::from_config(allow, false, false);
        leaker.intern_struct(&item, false);
        leaker.result()
    }
    for (source, relative) in [
        ("struct S { a: Rel<Self>, b: Self }", "Rel"),
        ("struct S { a: Box<Self>, b: Self }", "Box"),
    ] {
        let plain = by(source, Allow::new());
        assert_eq!(plain.errors().len(), 1, "{}", source);
        assert!(plain.clone().into_referrer().is_empty());
        // A rule for every path of one segment but the relative one accepts `Self` and a generic
        // too, if it is asked.
        let short = |path: &Path| path.segments.len() == 1 && !path.is_ident(relative);
        let by_rule = by(source, Allow::new().allow_path_if(short));
        assert_eq!(texts(by_rule.errors()), texts(plain.errors()), "{}", source);
        assert!(by_rule.into_referrer().is_empty(), "{}", source);
        // Accepting `Self` only, in a config or on a result.
        let by_rule = by(
            source,
            Allow::new().allow_path_if(|path| path.is_ident("Self")),
        );
        assert_eq!(texts(by_rule.errors()), texts(plain.errors()), "{}", source);
        let mut on_result = plain.clone();
        on_result.allow_path_if(|path| path.is_ident("Self"));
        assert_eq!(
            texts(on_result.errors()),
            texts(plain.errors()),
            "{}",
            source
        );
    }
}

#[test]
fn the_filter_does_not_ask_the_rule_about_self() {
    // An error written by hand, with a path that starts with `Self`, read as a serialized result.
    let base = three_errors();
    let mut error = base.errors()[0].clone();
    error.paths = vec![parse_quote!(Self::A)];
    let mut tokens = base.into_referrer().to_token_stream();
    tokens.extend(error.to_token_stream());
    let mut result: LeakerResult = parse2(tokens).unwrap();
    assert_eq!(paths_of(&result), [["Self :: A"]]);
    let seen = Rc::new(Seen::default());
    result.allow_path_if(seen.rule());
    assert!(seen.paths.borrow().is_empty());
    // A rule that accepts everything does not accept it either.
    result.allow_path_if(|_| true);
    assert_eq!(paths_of(&result), [["Self :: A"]]);
}

#[test]
fn the_rule_sees_a_path_without_arguments_in_both_forms() {
    let item: ItemTrait = parse_quote!(
        trait Tr<T: m::Bound<Rel>> {
            fn f(&self, t: T);
        }
    );
    let exact = |path: &Path| path.to_token_stream().to_string() == "m :: Bound";
    let prefix = starts_with(parse_quote!(m::Bound));
    let by_exact = analyze_trait(&item, Allow::new().allow_path_if(exact), false);
    let by_prefix = analyze_trait(&item, Allow::new().allow_path_if(prefix), false);
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
    let exact = |path: &Path| path.to_token_stream().to_string() == "helper :: f";
    let prefix = starts_with(parse_quote!(helper::f));
    let by_exact = analyze_trait(&item, Allow::new().allow_path_if(exact), true);
    let by_prefix = analyze_trait(&item, Allow::new().allow_path_if(prefix), true);
    assert!(by_exact.errors().is_empty());
    assert_eq!(texts(by_exact.errors()), texts(by_prefix.errors()));
    let (by_exact, by_prefix) = (by_exact.into_referrer(), by_prefix.into_referrer());
    assert_eq!(entries(&by_exact), [("Rel".to_string(), true)]);
    assert_eq!(entries(&by_exact), entries(&by_prefix));

    // Whatever it is given has no arguments, in the analysis and on a result.
    let seen = Rc::new(Seen::default());
    let item: ItemTrait = parse_quote!(
        trait Tr<T: m::Bound<Rel>> {
            fn f(&self) -> m::Out<Rel2<'_>> {
                helper::f::<Rel3>()
            }
        }
    );
    let result = analyze_trait(&item, Allow::new().allow_path_if(seen.rule()), true);
    assert!(seen.contains("m :: Bound") && seen.contains("helper :: f"));
    assert!(!seen.with_arguments.get(), "{:?}", seen.paths);
    let mut result = result;
    seen.paths.borrow_mut().clear();
    result.allow_path_if(seen.rule());
    assert!(!seen.paths.borrow().is_empty());
    assert!(!seen.with_arguments.get(), "{:?}", seen.paths);
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
    let result = analyze(&item, Allow::new().allow_path_if(seen.rule()));
    assert!(result.errors().is_empty());
    // `::core` is never optional, and `$crate` is context-free: the rule is not asked.
    assert!(seen.contains(":: std :: vec :: Vec"));
    assert!(seen.contains(":: serde :: W"));
    assert!(seen.contains("Rel"));
    assert!(!seen.starts_with(":: core"));
}

/// A rule that panics when it is asked about `Boom`.
fn boom(path: &Path) -> bool {
    if path.is_ident("Boom") {
        panic!("the rule is asked about Boom");
    }
    false
}

#[test]
fn a_rule_that_panics_in_a_caught_visit_poisons_the_call() {
    let mut leaker = Leaker::from_config(Allow::new().allow_path_if(boom), false, false);
    let generics: Generics = parse_quote!(<T>);
    let where_clause: WhereClause = parse_quote!(where T: m::Tr<Rel2>);
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        leaker.enter_generics(&generics, |view| {
            view.visit_type(&parse_quote!(Rel<T>));
            // The panic is caught here: the visit stops half way.
            let caught = catch_unwind(AssertUnwindSafe(|| {
                view.visit_type(&parse_quote!(Wrap<Boom<T>>))
            }));
            assert!(caught.is_err());
            // It would be accepted silently (and `Rel2` would be an entry) with the frames as
            // the panic left them.
            view.visit_where_clause(&where_clause);
        })
    }));
    // The call panics, as if the closure had.
    let payload = outcome.unwrap_err();
    assert!(payload.downcast_ref::<&str>().is_some());
    // Nothing of it is recorded, and the Leaker is usable.
    leaker.enter_generics(&generics, |view| view.visit_type(&parse_quote!(Rel3<T>)));
    let result = leaker.result();
    assert!(result.errors().is_empty());
    let referrer = result.finish().unwrap();
    assert_eq!(entries(&referrer), [("Rel3 < T >".to_string(), true)]);
}

#[test]
fn a_rule_that_panics_in_a_caught_inner_scope_poisons_the_outer_call() {
    let mut leaker = Leaker::from_config(Allow::new().allow_path_if(boom), false, false);
    let generics: Generics = parse_quote!(<T>);
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        leaker.enter_generics(&generics, |view| {
            view.visit_type(&parse_quote!(Rel<T>));
            let inner = catch_unwind(AssertUnwindSafe(|| {
                view.enter_generics(&parse_quote!(<U>), |inner| {
                    inner.visit_type(&parse_quote!(Boom<U>))
                })
            }));
            assert!(inner.is_err());
            // The scope that the panic left is not the one that is open any more.
            view.visit_type(&parse_quote!(Rel2<T>));
        })
    }));
    assert!(outcome.is_err());
    let result = leaker.result();
    assert!(result.errors().is_empty());
    assert!(result.into_referrer().is_empty());
}

#[test]
fn a_rule_that_panics_is_a_panic_of_the_closure() {
    let mut leaker = Leaker::from_config(Allow::new().allow_path_if(boom), false, false);
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

// The rules of the convenience methods are public functions: `allow_x()` is
// `allow_path_if(is_x_path)`, and a run-time choice goes into one closure.

const RULES: [&str; 4] = ["primitive", "prelude", "std", "crate"];

/// The analysis of `item` with the wrapper of `Allow` that has `rule` as its name (none if `None`),
/// or with `allow_path_if()` of the function of `rule`.
fn analyze_rule(
    rule: Option<&str>,
    function: bool,
    item: &ItemTrait,
    values: bool,
) -> LeakerResult {
    let allow = Allow::new();
    match (rule, function) {
        (None, _) => analyze_trait(item, allow, values),
        (Some("primitive"), false) => analyze_trait(item, allow.allow_primitive(), values),
        (Some("primitive"), true) => {
            analyze_trait(item, allow.allow_path_if(is_primitive_path), values)
        }
        (Some("prelude"), false) => analyze_trait(item, allow.allow_prelude(), values),
        (Some("prelude"), true) => {
            analyze_trait(item, allow.allow_path_if(is_prelude_path), values)
        }
        (Some("std"), false) => analyze_trait(item, allow.allow_std_crates(), values),
        (Some("std"), true) => analyze_trait(item, allow.allow_path_if(is_std_crate_path), values),
        (Some("crate"), false) => analyze_trait(item, allow.allow_crate(), values),
        (Some("crate"), true) => analyze_trait(item, allow.allow_path_if(is_crate_path), values),
        _ => unreachable!(),
    }
}

/// `result` after the wrapper of `LeakerResult` named `rule`, or after `allow_path_if()` of the
/// function of `rule`.
fn on_result(rule: &str, function: bool, mut result: LeakerResult) -> LeakerResult {
    match (rule, function) {
        ("primitive", false) => result.allow_primitive(),
        ("primitive", true) => result.allow_path_if(is_primitive_path),
        ("prelude", false) => result.allow_prelude(),
        ("prelude", true) => result.allow_path_if(is_prelude_path),
        ("std", false) => result.allow_std_crates(),
        ("std", true) => result.allow_path_if(is_std_crate_path),
        ("crate", false) => result.allow_crate(),
        ("crate", true) => result.allow_path_if(is_crate_path),
        _ => unreachable!(),
    };
    result
}

/// The Referrer and the errors of `result`, as one text.
fn serialized(result: &LeakerResult) -> String {
    result.to_token_stream().to_string()
}

#[test]
fn the_wrappers_are_the_path_functions() {
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
                type A: alloc::fmt::Debug + crate::m::Bound;
                fn f(&self, a: Self::A);
            }
        ),
        parse_quote!(
            trait Tr {
                fn f(&self, a: crate::m::T<u8>, b: crate::T) -> Rel<char>;
            }
        ),
        parse_quote!(
            trait Tr<T: Sized + Default> {
                fn f(&self, a: u8::Foo, b: ::std::vec::Vec<T>);
            }
        ),
        parse_quote!(
            trait Tr: Iterator<Item = u8> {
                fn f(&self, a: f64, b: &str, c: ::core::primitive::u8);
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
    // Whether the rule changes something for some item: the comparison is not vacuous.
    let mut changes = [false; 4];
    for (n, rule) in RULES.iter().enumerate() {
        for item in &items {
            let what = item.to_token_stream().to_string();
            for values in [false, true] {
                let by_wrapper = analyze_rule(Some(rule), false, item, values);
                let by_function = analyze_rule(Some(rule), true, item, values);
                let what = format!("{} (values: {}) in {}", rule, values, what);
                assert_eq!(
                    serialized(&by_wrapper),
                    serialized(&by_function),
                    "{}",
                    what
                );
                assert_eq!(
                    texts(by_wrapper.errors()),
                    texts(by_function.errors()),
                    "{}",
                    what
                );
                let plain = analyze_rule(None, false, item, values);
                changes[n] |= serialized(&plain) != serialized(&by_wrapper);
            }
            // After the analysis: the same errors, and the same Referrer.
            let plain = analyze_rule(None, false, item, false);
            let by_wrapper = on_result(rule, false, plain.clone());
            let by_function = on_result(rule, true, plain);
            let what = format!("{} on the result of {}", rule, what);
            assert_eq!(
                serialized(&by_wrapper),
                serialized(&by_function),
                "{}",
                what
            );
        }
    }
    assert_eq!(changes, [true; 4], "{:?}", RULES);
}

#[test]
fn the_path_functions_take_relative_paths_by_their_head() {
    let paths: &[(Path, [bool; 4])] = &[
        // primitive, prelude, std, crate
        (parse_quote!(u8), [true, false, false, false]),
        (parse_quote!(str), [true, false, false, false]),
        (parse_quote!(u8::MAX), [true, false, false, false]),
        (parse_quote!(f64), [true, false, false, false]),
        (parse_quote!(Vec), [false, true, false, false]),
        (parse_quote!(Option<u8>), [false, true, false, false]),
        (parse_quote!(Clone), [false, true, false, false]),
        (parse_quote!(Rel), [false, false, false, false]),
        (parse_quote!(TryFrom), [false, false, false, false]),
        (parse_quote!(std), [false, false, true, false]),
        (parse_quote!(std::vec::Vec<u8>), [false, false, true, false]),
        (parse_quote!(core::fmt), [false, false, true, false]),
        (parse_quote!(alloc::vec), [false, false, true, false]),
        (parse_quote!(stdx::Rel), [false, false, false, false]),
        (parse_quote!(crate), [false, false, false, true]),
        (parse_quote!(crate::m::T), [false, false, false, true]),
        (parse_quote!(self::m::T), [false, false, false, false]),
        (parse_quote!(super::T), [false, false, false, false]),
        // A leading `::` is another crate, never one of these.
        (parse_quote!(::u8), [false; 4]),
        (parse_quote!(::std::vec::Vec), [false; 4]),
        (parse_quote!(::core::primitive::u8), [false; 4]),
        (parse_quote!(::crate::T), [false; 4]),
    ];
    for (path, expected) in paths {
        let found = [
            is_primitive_path(path),
            is_prelude_path(path),
            is_std_crate_path(path),
            is_crate_path(path),
        ];
        assert_eq!(&found, expected, "{}", path.to_token_stream());
    }
}

#[test]
fn path_starts_with_compares_identifiers_segment_by_segment() {
    let starts = |path: Path, prefix: Path| path_starts_with(&path, &prefix);
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
}

#[test]
fn a_run_time_choice_goes_into_one_closure() {
    let item: ItemTrait = parse_quote!(
        trait Tr: Clone {
            fn f(&self, a: u8) -> Vec<char>;
        }
    );
    for (primitive, prelude) in [(false, false), (true, false), (false, true), (true, true)] {
        let config = Allow::new().allow_path_if(move |path| {
            (primitive && is_primitive_path(path)) || (prelude && is_prelude_path(path))
        });
        let chosen = analyze_trait(&item, config, false);
        let allow = Allow::new();
        let expected = match (primitive, prelude) {
            (true, true) => analyze_trait(&item, allow.allow_primitive().allow_prelude(), false),
            (true, false) => analyze_trait(&item, allow.allow_primitive(), false),
            (false, true) => analyze_trait(&item, allow.allow_prelude(), false),
            (false, false) => analyze_trait(&item, allow, false),
        };
        let what = format!("primitive: {}, prelude: {}", primitive, prelude);
        assert_eq!(serialized(&chosen), serialized(&expected), "{}", what);
        // `Clone` is an error, `u8` and `Vec<char>` are entries, unless the flags allow them.
        assert_eq!(chosen.errors().is_empty(), prelude, "{}", what);
    }
    // The result takes the same closure.
    let mut result = analyze_trait(&item, Allow::new(), false);
    result.allow_path_if(|path| is_prelude_path(path) || is_primitive_path(path));
    assert!(result.errors().is_empty());
}

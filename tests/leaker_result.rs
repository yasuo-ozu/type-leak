//! `LeakerResult` as tokens: the `Referrer`, then, if errors remain, `@ [ {error} {error} .. ]`.
//!
//! A Leaker's macro emits the result and the Referrer's macro reads it and decides what to allow.
//!
//! - Without errors: the tokens are the Referrer's own.
//! - With errors: the errors, the Referrer and the tokens that are written again are the same.
//! - `allow*` drops the same errors before and after the trip.
//! - The tokens after the result are not consumed (a brace group among them is no error: only an
//!   `@` and a bracket group are read), a wrong format tag fails as the Referrer's `Parse` does,
//!   and a malformed `@ [ .. ]` fails with a message of its own.
//! - `finish()` gives the Referrer, or a `LeakerResultError` with the errors that remain: it reads
//!   as a slice, is a `std::error::Error`, writes a `compile_error!` for each error as tokens (not
//!   the serialized result), and becomes one `syn::Error` for `?`.
//!
//! The spans of the errors that are read back are checked by `tests/crates/spans`
//! (`a_serialized_leaker_result_keeps_its_spans`).

mod common;

use common::{allow_every_cause, starts_with, AcceptCause, Recovered};
use proc_macro2::{Delimiter, Literal, TokenStream, TokenTree};
use std::cell::{Cell, RefCell};
use syn::parse::Parser;
use syn::*;
use template_quote::{quote, ToTokens};
use type_leak::*;

/// A trait and a struct that go wrong in several ways: a trait in a bound, a path of the standard
/// library in a bound, a type with an elided lifetime, a path of the prelude with one, a trait
/// object, and a trait in an argument. There are entries, a required one and an optional one.
fn with_errors() -> LeakerResult {
    let tr: ItemTrait = parse_quote!(
        trait Tr<T>: Clone + core::fmt::Debug {
            fn f(&self, a: MyType<'_>, b: Vec<Rel<'_>>, c: Box<dyn RelA + RelB<'_>>) -> Wrap<T>;
            fn g(&self, x: ::foo::X) -> Rel;
        }
    );
    let st: ItemStruct = parse_quote!(
        struct S<T> {
            a: Wrap<T>,
            b: ::p::W<'_, dyn RelTr>,
        }
    );
    let mut leaker = Leaker::new();
    leaker.intern_trait(&tr);
    leaker.intern_struct(&st);
    leaker.result()
}

/// An item with entries and no error.
fn without_errors() -> LeakerResult {
    let item: ItemTrait = parse_quote!(
        trait Tr<T> {
            fn f(&self, t: T) -> Wrap<T>;
            fn g(&self) -> Rel;
        }
    );
    let mut leaker = Leaker::new();
    leaker.intern_trait(&item);
    leaker.result()
}

/// Parses a result.
fn read(tokens: TokenStream) -> Result<LeakerResult> {
    parse2(tokens)
}

/// The fields of an error as text: the kind and the tokens of the cause, the reason, the message,
/// and the notes and the help with the tokens they are about.
fn describe(item: &ErrorItem) -> String {
    let notes: Vec<String> = item
        .notes
        .iter()
        .map(|(at, note)| format!("({}) {}", at, note))
        .collect();
    let help = item
        .help
        .as_ref()
        .map(|(at, help)| format!("({}) {}", at, help));
    let paths: Vec<String> = item.paths.iter().map(text).collect();
    format!(
        "{:?} {} | {:?} | {} | {:?} | {:?} | value {} default {} paths {:?}",
        std::mem::discriminant(&item.cause),
        item.cause.to_token_stream(),
        item.reason,
        item.message,
        notes,
        help,
        item.place.in_value,
        item.place.in_default,
        paths
    )
}

fn describe_all(result: &LeakerResult) -> Vec<String> {
    result.errors().iter().map(describe).collect()
}

/// The tokens as text, which compares them whatever their spans.
fn text(tokens: &impl ToTokens) -> String {
    tokens.to_token_stream().to_string()
}

/// A result without errors is the tokens of its Referrer and nothing more, and it is read back:
/// `finish()` is Ok and returns a Referrer equal to the original.
#[test]
fn a_result_without_errors_is_written_and_read_back() {
    let result = without_errors();
    assert!(result.errors().is_empty());
    let referrer = result.clone().finish().unwrap();
    assert_eq!(referrer.iter().count(), 2);
    let tokens = result.to_token_stream();
    assert_eq!(text(&tokens), text(&referrer));
    let read = read(tokens.clone()).unwrap();
    assert!(read.errors().is_empty());
    assert_eq!(text(&read), text(&tokens));
    assert_eq!(read.finish().unwrap(), referrer);
}

/// A result with several errors is the tokens of its Referrer, then `@` and one bracket group with
/// the tokens of each error in order. It is read back with the same errors, item by item, and a
/// Referrer equal to the original; writing it twice, or writing the one that was read, gives the
/// same tokens.
#[test]
fn a_result_with_errors_is_written_and_read_back() {
    let result = with_errors();
    assert!(result.errors().len() >= 5, "{:?}", describe_all(&result));
    let referrer = result.clone().recovered();

    // The form: the Referrer, then `@`, then one bracket group with one brace group for each error,
    // in the order of `errors()`.
    let tokens = result.to_token_stream();
    let mut list = TokenStream::new();
    for error in result.errors() {
        let group = error.to_token_stream();
        assert_eq!(group.clone().into_iter().count(), 1);
        list.extend(group);
    }
    let mut expected = referrer.to_token_stream();
    expected.extend(quote!(@ [#list]));
    assert_eq!(text(&tokens), text(&expected));
    // And nothing else: the tokens of the Referrer, an `@`, and the bracket group of the errors.
    let trees: Vec<TokenTree> = tokens.clone().into_iter().collect();
    let n = referrer.to_token_stream().into_iter().count();
    assert_eq!(trees.len(), n + 2);
    assert!(matches!(&trees[n], TokenTree::Punct(p) if p.as_char() == '@'));
    match &trees[n + 1] {
        TokenTree::Group(group) => {
            assert_eq!(group.delimiter(), Delimiter::Bracket);
            let groups: Vec<TokenTree> = group.stream().into_iter().collect();
            assert_eq!(groups.len(), result.errors().len());
            assert!(groups.iter().all(
                |tree| matches!(tree, TokenTree::Group(g) if g.delimiter() == Delimiter::Brace)
            ));
        }
        other => panic!("{}", other),
    }
    // It is deterministic.
    assert_eq!(text(&tokens), text(&result.to_token_stream()));
    assert_eq!(text(&tokens), text(&result.clone()));

    let read = read(tokens.clone()).unwrap();
    assert_eq!(describe_all(&read), describe_all(&result));
    for (read, error) in read.errors().iter().zip(result.errors()) {
        assert_eq!(read.cause, error.cause);
    }
    assert_eq!(read.clone().recovered(), referrer);
    assert_eq!(text(&read), text(&tokens));
    // `finish()` returns the errors, in order, when some remain.
    let errors = read.finish().unwrap_err();
    assert_eq!(
        errors.iter().map(describe).collect::<Vec<_>>(),
        describe_all(&result)
    );
}

/// `allow(cause)` and `Rules::std_crates()` drop the same errors after the trip as before it, each
/// step and all of them, and so does writing the rest again. Accepting every cause leaves a Referrer
/// that `finish()` returns, equal to the original.
#[test]
fn allowing_after_the_round_trip_drops_the_same_errors() {
    let result = with_errors();
    let referrer = result.clone().recovered();
    let read_back = read(result.to_token_stream()).unwrap();
    let causes: Vec<Cause> = result.errors().iter().map(|e| e.cause.clone()).collect();

    for cause in &causes {
        let (mut before, mut after) = (result.clone(), read_back.clone());
        before = before.accept_cause(cause);
        after = after.accept_cause(cause);
        assert!(before.errors().len() < result.errors().len());
        assert_eq!(describe_all(&after), describe_all(&before));
    }
    let (mut before, mut after) = (result.clone(), read_back.clone());
    before = before.allow(Rules::new().std_crates());
    after = after.allow(Rules::new().std_crates());
    // `core::fmt::Debug` is the only cause that starts with the name of a crate of the library.
    let remaining: Vec<String> = before.errors().iter().map(|e| text(&e.cause)).collect();
    assert_eq!(remaining.len(), result.errors().len() - 1);
    assert!(!remaining.contains(&"core :: fmt :: Debug".to_string()));
    assert_eq!(describe_all(&after), describe_all(&before));

    // What remains is written and read again, with the same Referrer.
    let rest = read(after.to_token_stream()).unwrap();
    assert_eq!(describe_all(&rest), describe_all(&before));
    assert_eq!(rest.recovered(), referrer);

    let (all, _) = allow_every_cause(read_back);
    assert!(all.errors().is_empty());
    assert_eq!(all.finish().unwrap(), referrer);
}

/// The path, value and default rules (`Rules::or_if()` and the others) give the same
/// errors on the other side of the trip. A path rule leaves the paths it does not allow in the
/// error, which is serialized with them, so rules given in two macros add up as they do in one.
#[test]
fn the_path_value_and_default_rules_give_the_same_errors_after_the_round_trip() {
    let item: ItemTrait = parse_quote!(
        trait Tr<U = Rel<'_>>: core::fmt::Debug {
            const N: usize = inner::N;
            fn f(&self, a: Box<dyn Rel2<'_> + Send>, b: my::W<'_, Rel3>) {
                let _ = inner::f();
            }
        }
    );
    let config = Rules::new().primitive();
    let mut leaker = Leaker::with_rule(config);
    leaker.check_values = true;
    leaker.check_bodies = true;
    leaker.intern_trait(&item);
    let result = leaker.result();
    assert!(result
        .errors()
        .iter()
        .any(|e| e.place.in_value && e.place.in_default));
    assert!(result.errors().iter().any(|e| e.paths.len() > 1));
    let read_back = read(result.to_token_stream()).unwrap();
    assert_eq!(describe_all(&read_back), describe_all(&result));

    type Rule = fn(LeakerResult) -> LeakerResult;
    let rules: [(&str, Rule); 7] = [
        ("path", |r| r.allow(starts_with(parse_quote!(my)))),
        ("prelude", |r| r.allow(Rules::new().prelude())),
        ("std crates", |r| r.allow(Rules::new().std_crates())),
        ("crate", |r| r.allow(Rules::new().crate_paths())),
        ("primitive", |r| r.allow(Rules::new().primitive())),
        ("values", |r| r.allow(Rules::new().values())),
        ("default", |r| r.allow(Rules::new().defaults())),
    ];
    let mut dropped_something = 0;
    for (name, rule) in rules {
        let (before, after) = (rule(result.clone()), rule(read_back.clone()));
        assert_eq!(describe_all(&after), describe_all(&before), "{}", name);
        if before.errors().len() < result.errors().len()
            || describe_all(&before) != describe_all(&result)
        {
            dropped_something += 1;
        }
        // The rest is written and read again, and a second rule on the other side of that trip
        // gives what both rules give on one side.
        for (second_name, second) in rules {
            let together = second(before.clone());
            let trip = second(read(before.to_token_stream()).unwrap());
            assert_eq!(
                describe_all(&trip),
                describe_all(&together),
                "{} then {}",
                name,
                second_name
            );
        }
    }
    assert!(dropped_something >= 3);
}

/// `finish()` is the only way to the Referrer, and `allow()` drops errors and changes no entry:
/// with every error allowed, `finish()` gives the Referrer that the result wrote before, whatever
/// errors the analysis found.
#[test]
fn allowing_every_error_gives_the_referrer_of_the_analysis() {
    let result = with_errors();
    assert!(!result.errors().is_empty());
    let errors = result.clone().finish().unwrap_err();
    assert_eq!(
        describe_all(&result),
        errors.iter().map(describe).collect::<Vec<_>>()
    );
    // The tokens of the result start with the tokens of the Referrer.
    let written = result.to_token_stream().to_string();
    let (all, _) = allow_every_cause(result.clone());
    assert!(all.errors().is_empty());
    let referrer = all.clone().finish().unwrap();
    let own = referrer.to_token_stream().to_string();
    assert!(
        written.starts_with(&own) && written[own.len()..].trim_start().starts_with('@'),
        "{}\n{}",
        written,
        own
    );
    assert_eq!(referrer, result.clone().recovered());
    // The same holds when only some of the errors are allowed first.
    for error in result.errors() {
        let rest = result
            .clone()
            .allow(common::accepting(result.errors(), &error.cause));
        let (rest, _) = allow_every_cause(rest);
        assert_eq!(rest.finish().unwrap(), referrer);
    }
}

/// `allow()` takes and returns the result, so it chains; a rule is applied at once, so it may count
/// (through a `Cell`).
#[test]
fn the_rules_chain_and_a_rule_may_count() {
    let result = with_errors();
    let asked = Cell::new(0);
    let seen = RefCell::new(Vec::new());
    let chained = result
        .clone()
        .allow(|_: &Cause, _: Place| {
            asked.set(asked.get() + 1);
            false
        })
        .allow(|cause: &Cause, _: Place| {
            if let Some(path) = cause.path() {
                seen.borrow_mut().push(text(path));
            }
            false
        })
        .allow(Rules::new().std_crates());
    // The rule is asked about the cause of each error, and then about the paths of those it did not
    // accept: the counting rule saw at least one question for each error.
    assert!(asked.get() >= result.errors().len());
    assert!(seen.borrow().iter().any(|path| path == "Rel"), "{:?}", seen);
    // Chained: the same errors as one call after the other. No rule changes an entry, so the
    // Referrer replaces the optional entry `::foo::X` whether or not a rule allows its crate.
    let one_by_one = result
        .clone()
        .allow(Rules::new().std_crates())
        .allow(Rules::new().external_crates());
    assert_eq!(describe_all(&chained), describe_all(&one_by_one));
    let expand = |result: LeakerResult| {
        let referrer = result.recovered();
        let ix = common::types(&referrer)
            .position(|ty| text(ty) == ":: foo :: X")
            .unwrap();
        let ty = common::expand(
            &referrer,
            common::entry_id(&referrer, ix),
            parse_quote!(::foo::X),
            |_, ix, _| parse_quote!(P<#ix>),
        );
        (ix, text(&ty))
    };
    let (ix, replaced) = expand(chained);
    assert_eq!(replaced, format!("P < {}usize >", ix));
    assert_eq!(expand(one_by_one), (ix, replaced));
}

/// The tokens after the result are left for the caller, whatever they are. A result with errors
/// and one without, and a Referrer (which parses as a result without errors), are each followed by
/// tokens: an ident, a brace group (`#result { item }`, which is no error: only an `@` and a
/// bracket group are read), a group that has the shape of an error (it was read as one without the
/// `@`), an `@` with anything but a bracket group after it, and an empty list.
#[test]
fn the_tokens_after_the_result_are_left_to_the_caller() {
    /// A result, then the rest.
    fn then_rest(input: parse::ParseStream) -> Result<(LeakerResult, String)> {
        let result: LeakerResult = input.parse()?;
        let rest: TokenStream = input.parse()?;
        Ok((result, rest.to_string()))
    }
    // The errors and the Referrer that are read back.
    struct Expected(Vec<String>, Referrer);
    let of_result = |result: LeakerResult| {
        let tokens = result.to_token_stream();
        (tokens, Expected(describe_all(&result), result.recovered()))
    };
    let of_referrer = |result: LeakerResult| {
        let referrer = result.recovered();
        (referrer.to_token_stream(), Expected(Vec::new(), referrer))
    };
    let error_group = with_errors().errors()[0].to_token_stream();
    // The label, the result as tokens with what it reads back as, what follows it, and what is left
    // of that (all of it, unless the result takes some).
    type Row = (
        &'static str,
        (TokenStream, Expected),
        TokenStream,
        Option<&'static str>,
    );
    let rows: Vec<Row> = vec![
        (
            "an ident and a `;` after the errors",
            of_result(with_errors()),
            quote!(; after),
            None,
        ),
        (
            "an ident and a `;` after a result without errors",
            of_result(without_errors()),
            quote!(; after),
            None,
        ),
        (
            "a brace group after the errors",
            of_result(with_errors()),
            quote!({
                struct Copy;
            }),
            None,
        ),
        (
            "a brace group after a result without errors",
            of_result(without_errors()),
            quote!({
                struct Copy;
            }),
            None,
        ),
        (
            "a group that has the shape of an error",
            of_result(without_errors()),
            error_group.clone(),
            None,
        ),
        (
            "a brace group after the Referrer of a result with errors",
            of_referrer(with_errors()),
            quote!({ item }),
            None,
        ),
        (
            "a brace group after a Referrer",
            of_referrer(without_errors()),
            quote!({ item }),
            None,
        ),
        ("an `@` alone", of_referrer(with_errors()), quote!(@), None),
        (
            "an `@` and an ident before a bracket group",
            of_referrer(with_errors()),
            quote!(@ x [ y ]),
            None,
        ),
        (
            "an `@` and a brace group",
            of_referrer(with_errors()),
            quote!(@ { }),
            None,
        ),
        (
            "an `@` and a paren group",
            of_referrer(with_errors()),
            quote!(@ ( )),
            None,
        ),
        (
            "a bracket group without an `@`",
            of_referrer(with_errors()),
            quote!([]),
            None,
        ),
        (
            "an empty list of errors",
            of_result(without_errors()),
            quote!(@ [ ] after),
            Some("after"),
        ),
    ];
    for (label, (mut tokens, Expected(errors, referrer)), after, rest) in rows {
        tokens.extend(after.clone());
        let (read, left) = then_rest.parse2(tokens).unwrap();
        assert_eq!(describe_all(&read), errors, "{}", label);
        assert_eq!(read.recovered(), referrer, "{}", label);
        // What follows is left as it is, unless the result takes some of it.
        let rest = rest.map_or_else(|| after.to_string(), str::to_string);
        assert_eq!(left, rest, "{}", label);
    }
    // Two results one after the other: the first stops where the second starts.
    let (first, second) = (without_errors(), with_errors());
    let mut tokens = first.to_token_stream();
    tokens.extend(second.to_token_stream());
    let (a, b) = (|input: parse::ParseStream| -> Result<(LeakerResult, LeakerResult)> {
        Ok((input.parse()?, input.parse()?))
    })
    .parse2(tokens)
    .unwrap();
    assert!(a.errors().is_empty());
    assert_eq!(describe_all(&b), describe_all(&second));
}

/// The tag is read by the Referrer: a different format, and no tag at all, fail with its message.
#[test]
fn a_wrong_format_tag_fails_as_the_referrer_does() {
    let tokens = with_errors().to_token_stream();
    let mut trees = tokens.into_iter();
    let tag = trees.next().unwrap();
    assert!(matches!(tag, TokenTree::Literal(_)));
    let rest: TokenStream = trees.collect();
    let other = TokenTree::Literal(Literal::string("type-leak-format-15"));
    let unrecognized = TokenTree::Literal(Literal::string("something else"));
    let tagged =
        |tag: TokenTree| -> TokenStream { std::iter::once(tag).chain(rest.clone()).collect() };

    for (tokens, expected) in [
        (tagged(other), "format 15"),
        (tagged(unrecognized), "an unrecognized tag"),
        (rest.clone(), "no format tag"),
    ] {
        let from_result = read(tokens.clone()).unwrap_err().to_string();
        let from_referrer = parse2::<Referrer>(tokens).unwrap_err().to_string();
        assert_eq!(from_result, from_referrer);
        assert!(from_result.contains(expected), "{}", from_result);
        assert!(
            from_result.contains("incompatible type-leak"),
            "{}",
            from_result
        );
    }
}

/// A Referrer parses as a result without errors, and is written again as it was (the tokens after
/// it are left to the caller: see `the_tokens_after_the_result_are_left_to_the_caller`).
#[test]
fn a_referrer_parses_as_a_result_without_errors() {
    for result in [with_errors(), without_errors()] {
        let referrer = result.recovered();
        let tokens = referrer.to_token_stream();
        let read = read(tokens.clone()).unwrap();
        assert!(read.errors().is_empty());
        assert_eq!(text(&read), text(&tokens));
        assert_eq!(read.finish().unwrap(), referrer);
    }
}

/// An `@ [ .. ]` that holds a token which is not a brace group fails with a message that says what
/// is expected; a brace group that is no error fails with the message of `ErrorItem`'s `Parse`.
#[test]
fn a_malformed_error_list_is_refused() {
    let result = with_errors();
    let referrer = result.clone().recovered().to_token_stream();
    let first = result.errors()[0].to_token_stream();
    let with = |list: TokenStream| -> TokenStream {
        let mut tokens = referrer.clone();
        tokens.extend(quote!(@ [#list]));
        tokens
    };

    for list in [
        quote!(nothing),
        quote!(#first , #first),
        quote!(#first ( #first )),
        quote!(#first [ #first ]),
        quote!(#first x),
    ] {
        let message = read(with(list.clone())).unwrap_err().to_string();
        assert_eq!(
            message, "expected an error `{ .. }` in `@ [ .. ]`",
            "{}",
            list
        );
    }

    let expected = parse2::<ErrorItem>(quote!({ nothing }))
        .unwrap_err()
        .to_string();
    for list in [quote!({ nothing }), quote!(#first { nothing })] {
        let message = read(with(list.clone())).unwrap_err().to_string();
        assert_eq!(message, expected, "{}", list);
        assert!(!message.contains("incompatible type-leak"), "{}", message);
    }

    // The list of the result itself is no such list.
    assert_eq!(
        describe_all(&read(result.to_token_stream()).unwrap()),
        describe_all(&result)
    );
}

/// `finish()` is `Ok` without errors, and `Err` with the errors of `errors()`, in the same order,
/// which are never none.
#[test]
fn finish_gives_the_errors_of_the_result_in_order() {
    assert!(without_errors().finish().is_ok());

    let result = with_errors();
    let expected = result.errors().to_vec();
    assert!(expected.len() > 1);
    let errors: LeakerResultError = result.finish().unwrap_err();
    assert_eq!(errors.len(), expected.len());
    assert_eq!(&*errors, &expected[..]);
    assert_eq!(errors.clone().into_vec(), expected);
    assert_eq!(errors, errors.clone());

    // One error is enough, and the others are not made up.
    let mut leaker = Leaker::new();
    leaker.intern_struct(&parse_quote!(
        struct S(MyType<'_>);
    ));
    let one = leaker.result().finish().unwrap_err();
    assert_eq!(one.len(), 1);
}

/// `into_vec()`, `Deref` to a slice, and both `IntoIterator`s give the errors in order.
#[test]
fn the_errors_read_as_a_vec_a_slice_and_an_iterator() {
    let expected = with_errors().errors().to_vec();
    let errors = with_errors().finish().unwrap_err();

    // `Deref`: `len()`, `[..]`, `iter()` and `&errors` as a slice.
    fn first_of(slice: &[ErrorItem]) -> &ErrorItem {
        &slice[0]
    }
    assert_eq!(errors.len(), expected.len());
    assert!(!errors.is_empty());
    assert_eq!(errors[0], expected[0]);
    assert_eq!(errors[expected.len() - 1], expected[expected.len() - 1]);
    assert_eq!(first_of(&errors), &expected[0]);
    assert_eq!(
        errors.iter().collect::<Vec<_>>(),
        expected.iter().collect::<Vec<_>>()
    );
    assert_eq!(errors.first(), expected.first());

    // `IntoIterator` for a reference.
    let mut seen = Vec::new();
    for error in &errors {
        seen.push(error.clone());
    }
    assert_eq!(seen, expected);

    // `IntoIterator` by value, and `into_vec()`.
    assert_eq!(
        errors.clone().into_iter().collect::<Vec<ErrorItem>>(),
        expected
    );
    assert_eq!(errors.into_vec(), expected);
}

/// `Display` is the messages of the errors, one on each line.
#[test]
fn a_result_error_displays_the_messages_one_per_line() {
    let errors = with_errors().finish().unwrap_err();
    let messages: Vec<&str> = errors.iter().map(|error| error.message.as_str()).collect();
    assert!(messages.len() > 1);
    assert!(messages.iter().all(|message| !message.contains('\n')));
    assert_eq!(errors.to_string(), messages.join("\n"));
    assert_eq!(errors.to_string().lines().count(), messages.len());

    let mut leaker = Leaker::new();
    leaker.intern_struct(&parse_quote!(
        struct S(MyType<'_>);
    ));
    let one = leaker.result().finish().unwrap_err();
    assert_eq!(one.to_string(), "use absolute path: 'MyType<'_>'");
}

/// It is a `std::error::Error`: a `Box<dyn Error>`, and `?` into one, hold it.
#[test]
fn a_result_error_is_a_std_error() {
    let errors = with_errors().finish().unwrap_err();
    let boxed: Box<dyn std::error::Error> = Box::new(errors.clone());
    assert_eq!(boxed.to_string(), errors.to_string());
    assert_eq!(boxed.downcast_ref::<LeakerResultError>(), Some(&errors));
    assert!(boxed.source().is_none());

    fn analyze() -> std::result::Result<Referrer, Box<dyn std::error::Error>> {
        Ok(with_errors().finish()?)
    }
    let boxed = analyze().unwrap_err();
    assert_eq!(boxed.to_string(), errors.to_string());
    assert_eq!(
        boxed.downcast_ref::<LeakerResultError>().map(|e| e.len()),
        Some(errors.len())
    );
}

/// `ToTokens` writes each error's `to_compile_error()`, in order: not the serialized result, which
/// `LeakerResult` writes.
#[test]
fn a_result_error_writes_the_compile_errors_as_tokens() {
    let result = with_errors();
    let errors = result.clone().finish().unwrap_err();
    let mut expected = TokenStream::new();
    for error in result.errors() {
        expected.extend(error.to_compile_error());
    }
    assert_eq!(text(&errors), text(&expected));
    assert_eq!(
        errors
            .to_token_stream()
            .to_string()
            .matches("compile_error")
            .count(),
        errors.len()
    );
    // The tokens of the result are the serialized form, which holds an `@` and no `compile_error`.
    assert_ne!(text(&errors), text(&result));
    assert!(!text(&result).contains("compile_error"));
}

/// `syn::Error::from` makes one error with an item for each error, in order, written as the
/// conversion of the error does, so `?` works in a function that returns `syn::Result`.
#[test]
fn a_result_error_becomes_one_syn_error_with_an_item_for_each() {
    let result = with_errors();
    let errors = result.clone().finish().unwrap_err();
    let expected: Vec<String> = result
        .errors()
        .iter()
        .map(|error| Error::from(error.clone()).to_string())
        .collect();

    let combined = Error::from(errors);
    let items: Vec<String> = combined
        .clone()
        .into_iter()
        .map(|e| e.to_string())
        .collect();
    assert_eq!(items.len(), result.errors().len());
    assert_eq!(items, expected);
    assert_eq!(combined.into_iter().count(), result.errors().len());

    fn analyze() -> Result<Referrer> {
        Ok(with_errors().finish()?)
    }
    let error = analyze().unwrap_err();
    assert_eq!(
        error.into_iter().map(|e| e.to_string()).collect::<Vec<_>>(),
        expected
    );

    // `map_err(Error::from)` is the same.
    let error = with_errors().finish().map_err(Error::from).unwrap_err();
    assert_eq!(error.into_iter().count(), result.errors().len());
}

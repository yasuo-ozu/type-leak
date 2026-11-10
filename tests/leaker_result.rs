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
//!
//! The spans of the errors that are read back are checked by `tests/crates/spans`
//! (`a_serialized_leaker_result_keeps_its_spans`).

mod common;

use common::{allow_every_cause, starts_with};
use proc_macro2::{Delimiter, Literal, TokenStream, TokenTree};
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
    leaker.intern_trait(&tr, true);
    leaker.intern_struct(&st, true);
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
    leaker.intern_trait(&item, true);
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
        item.in_value,
        item.in_default,
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
    let referrer = result.clone().into_referrer();

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
    assert_eq!(read.clone().into_referrer(), referrer);
    assert_eq!(text(&read), text(&tokens));
    // `finish()` returns the errors, in order, when some remain.
    let errors = read.finish().unwrap_err();
    assert_eq!(
        errors.iter().map(describe).collect::<Vec<_>>(),
        describe_all(&result)
    );
}

/// `allow(cause)` and `allow_std_crates()` drop the same errors after the trip as before it, each
/// step and all of them, and so does writing the rest again. Accepting every cause leaves a Referrer
/// that `finish()` returns, equal to the original.
#[test]
fn allowing_after_the_round_trip_drops_the_same_errors() {
    let result = with_errors();
    let referrer = result.clone().into_referrer();
    let read_back = read(result.to_token_stream()).unwrap();
    let causes: Vec<Cause> = result.errors().iter().map(|e| e.cause.clone()).collect();

    for cause in &causes {
        let (mut before, mut after) = (result.clone(), read_back.clone());
        before = before.allow(|e| e.cause == cause.clone());
        after = after.allow(|e| e.cause == cause.clone());
        assert!(before.errors().len() < result.errors().len());
        assert_eq!(describe_all(&after), describe_all(&before));
    }
    let (mut before, mut after) = (result.clone(), read_back.clone());
    before = before.allow_std_crates();
    after = after.allow_std_crates();
    // `core::fmt::Debug` is the only cause that starts with the name of a crate of the library.
    let remaining: Vec<String> = before.errors().iter().map(|e| text(&e.cause)).collect();
    assert_eq!(remaining.len(), result.errors().len() - 1);
    assert!(!remaining.contains(&"core :: fmt :: Debug".to_string()));
    assert_eq!(describe_all(&after), describe_all(&before));

    // What remains is written and read again, with the same Referrer.
    let rest = read(after.to_token_stream()).unwrap();
    assert_eq!(describe_all(&rest), describe_all(&before));
    assert_eq!(rest.into_referrer(), referrer);

    let (all, _) = allow_every_cause(read_back);
    assert!(all.errors().is_empty());
    assert_eq!(all.finish().unwrap(), referrer);
}

/// The path, value and default rules (`LeakerResult::allow_path_if()` and the others) give the same
/// errors on the other side of the trip. A path rule leaves the paths it does not allow in the
/// error, which is serialized with them, so rules given in two macros add up as they do in one.
#[test]
fn the_path_value_and_default_rules_give_the_same_errors_after_the_round_trip() {
    let item: ItemTrait = parse_quote!(
        trait Tr<U = Rel<'_>>: core::fmt::Debug {
            const N: usize = inner::N;
            fn f(&self, a: Vec<Rel2<'_>>, b: my::W<'_, Rel3>) {
                let _ = inner::f();
            }
        }
    );
    let config = Allow::new().allow_primitive();
    let mut leaker = Leaker::from_config(config, Options::new().check_values());
    leaker.intern_trait(&item, true);
    let result = leaker.result();
    assert!(result.errors().iter().any(|e| e.in_value && e.in_default));
    assert!(result.errors().iter().any(|e| e.paths.len() > 1));
    let read_back = read(result.to_token_stream()).unwrap();
    assert_eq!(describe_all(&read_back), describe_all(&result));

    type Rule = fn(LeakerResult) -> LeakerResult;
    let rules: [(&str, Rule); 7] = [
        ("path", |r| r.allow_path_if(starts_with(parse_quote!(my)))),
        ("prelude", |r| r.allow_prelude()),
        ("std crates", |r| r.allow_std_crates()),
        ("crate", |r| r.allow_crate()),
        ("primitive", |r| r.allow_primitive()),
        ("values", |r| r.allow_values()),
        ("default", |r| r.allow_default()),
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

/// `into_parts()` is `into_referrer()` and `errors()` in one call.
#[test]
fn into_parts_returns_the_referrer_and_the_errors_that_remain() {
    let result = with_errors();
    assert!(!result.errors().is_empty());
    let (referrer, errors) = result.clone().into_parts();
    assert_eq!(referrer, result.clone().into_referrer());
    assert_eq!(errors.len(), result.errors().len());
    let described: Vec<String> = errors.iter().map(describe).collect();
    assert_eq!(described, describe_all(&result));
    // With every error dropped, the errors are none and the Referrer is the same.
    let (all, _) = allow_every_cause(result);
    let (again, none) = all.into_parts();
    assert!(none.is_empty());
    assert_eq!(again, referrer);
}

/// The `allow*()` methods and `deny_ambiguous_crate_name()` take and return the result, so they
/// chain; a rule is an `FnMut`, applied at once, so it may count.
#[test]
fn the_rules_chain_and_a_rule_may_count() {
    let result = with_errors();
    let mut asked = 0;
    let mut seen = Vec::new();
    let chained = result
        .clone()
        .allow(|_| {
            asked += 1;
            false
        })
        .allow_path_if(|path| {
            seen.push(text(path));
            false
        })
        .allow_std_crates()
        .deny_ambiguous_crate_name();
    // The rule on errors is asked once for each error, and the rule on paths for the paths.
    assert_eq!(asked, result.errors().len());
    assert!(seen.iter().any(|path| path == "Rel"), "{:?}", seen);
    // Chained: the same errors as one call after the other, and a strict Referrer.
    let one_by_one = result.clone().allow_std_crates();
    assert_eq!(describe_all(&chained), describe_all(&one_by_one));
    // Only the strict Referrer replaces the optional entry `::foo::X` (`==` ignores strictness).
    let expand = |result: LeakerResult| {
        let referrer = result.into_referrer();
        let ix = referrer
            .iter()
            .position(|ty| text(ty) == ":: foo :: X")
            .unwrap();
        let ty = referrer.expand(
            referrer.context(ix),
            parse_quote!(::foo::X),
            |_, ix| parse_quote!(P<#ix>),
        );
        (ix, text(&ty))
    };
    let (ix, strict) = expand(chained);
    assert_eq!(strict, format!("P < {}usize >", ix));
    let one_by_one = expand(one_by_one).1;
    assert_eq!(one_by_one, ":: foo :: X");
}

/// The tokens after the result are left for the caller: an ident and a `;` after the errors, and
/// the same after a result without errors.
#[test]
fn the_tokens_after_the_result_are_not_consumed() {
    /// A result, then the `;` and the ident that follow it, and nothing else.
    fn then_semi_and_ident(input: parse::ParseStream) -> Result<(LeakerResult, Ident)> {
        let result: LeakerResult = input.parse()?;
        input.parse::<Token![;]>()?;
        let ident: Ident = input.parse()?;
        assert!(input.is_empty());
        Ok((result, ident))
    }
    for result in [with_errors(), without_errors()] {
        let mut tokens = result.to_token_stream();
        tokens.extend(quote!(; after));
        let (read, ident) = then_semi_and_ident.parse2(tokens).unwrap();
        assert_eq!(ident, "after");
        assert_eq!(describe_all(&read), describe_all(&result));
        assert_eq!(read.into_referrer(), result.into_referrer());
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

/// A result and then a brace group (`#result { item }`): the group is left to the caller, with
/// errors or without, and the errors are those of the result.
#[test]
fn a_brace_group_after_the_result_is_left_to_the_caller() {
    /// A result, then a brace group with its content, and nothing else.
    fn result_then_group(input: parse::ParseStream) -> Result<(LeakerResult, String)> {
        let result: LeakerResult = input.parse()?;
        let group;
        braced!(group in input);
        let content: TokenStream = group.parse()?;
        assert!(input.is_empty());
        Ok((result, content.to_string()))
    }
    for result in [with_errors(), without_errors()] {
        let tokens = quote!(#result { struct Copy; });
        let (read, item) = result_then_group.parse2(tokens).unwrap();
        assert_eq!(item, "struct Copy ;");
        assert_eq!(describe_all(&read), describe_all(&result));
        assert_eq!(read.into_referrer(), result.into_referrer());
    }
    // A group that has the shape of an error is left too (it was read as one without the `@`).
    let error = with_errors().errors()[0].to_token_stream();
    let mut tokens = without_errors().to_token_stream();
    tokens.extend(error.clone());
    let (read, item) = (|input: parse::ParseStream| -> Result<(LeakerResult, TokenStream)> {
        Ok((input.parse()?, input.parse()?))
    })
    .parse2(tokens)
    .unwrap();
    assert!(read.errors().is_empty());
    assert_eq!(item.to_string(), error.to_string());
}

/// A Referrer parses as a result without errors, alone or with tokens after it.
#[test]
fn a_referrer_parses_as_a_result_without_errors() {
    for result in [with_errors(), without_errors()] {
        let referrer = result.into_referrer();
        let tokens = referrer.to_token_stream();
        let read = read(tokens.clone()).unwrap();
        assert!(read.errors().is_empty());
        assert_eq!(text(&read), text(&tokens));
        assert_eq!(read.into_referrer(), referrer);
        let (read, rest) = (|input: parse::ParseStream| -> Result<(LeakerResult, TokenStream)> {
            Ok((input.parse()?, input.parse()?))
        })
        .parse2(quote!(#referrer { item }))
        .unwrap();
        assert!(read.errors().is_empty());
        assert_eq!(rest.to_string(), "{ item }");
    }
}

/// Only an `@` followed by a bracket group is read as the errors: an `@` with anything else after
/// it is left to the caller, as are the tokens when there is no `@`.
#[test]
fn an_at_without_a_bracket_group_is_left_to_the_caller() {
    fn then_rest(input: parse::ParseStream) -> Result<(LeakerResult, String)> {
        let result: LeakerResult = input.parse()?;
        let rest: TokenStream = input.parse()?;
        Ok((result, rest.to_string()))
    }
    for rest in [
        quote!(@),
        quote!(@ x [ y ]),
        quote!(@ { }),
        quote!(@ ( )),
        quote!([]),
    ] {
        let mut tokens = with_errors().into_referrer().to_token_stream();
        tokens.extend(rest.clone());
        let (read, left) = then_rest.parse2(tokens).unwrap();
        assert!(read.errors().is_empty(), "{}", rest);
        assert_eq!(left, rest.to_string());
    }
    // An empty list is a list without errors.
    let mut tokens = without_errors().to_token_stream();
    tokens.extend(quote!(@ [ ] after));
    let (read, left) = then_rest.parse2(tokens).unwrap();
    assert!(read.errors().is_empty());
    assert_eq!(left, "after");
}

/// An `@ [ .. ]` that holds a token which is not a brace group fails with a message that says what
/// is expected; a brace group that is no error fails with the message of `ErrorItem`'s `Parse`.
#[test]
fn a_malformed_error_list_is_refused() {
    let result = with_errors();
    let referrer = result.clone().into_referrer().to_token_stream();
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

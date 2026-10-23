//! `LeakerResult` as tokens: the `Referrer`, then one brace group for each remaining error.
//!
//! A Leaker's macro emits the result and the Referrer's macro reads it and decides what to allow.
//!
//! - Without errors: the tokens are the Referrer's own.
//! - With errors: the errors, the Referrer and the tokens that are written again are the same.
//! - `allow*` drops the same errors before and after the trip.
//! - The tokens after the result are not consumed, and a wrong format tag fails as the Referrer's
//!   `Parse` does.
//!
//! The spans of the errors that are read back are checked by `tests/crates/spans`
//! (`a_serialized_leaker_result_keeps_its_spans`).

use proc_macro2::{Literal, TokenStream, TokenTree};
use syn::parse::Parser;
use syn::*;
use template_quote::{quote, ToTokens};
use type_leak::*;

/// A trait and a struct that go wrong in several ways: a trait in a bound, a type with an elided
/// lifetime, a path of the prelude with one, a trait object, and a trait in an argument. There are
/// entries, a required one and an optional one.
fn with_errors() -> LeakerResult {
    let tr: ItemTrait = parse_quote!(
        trait Tr<T>: Clone {
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
    format!(
        "{:?} {} | {:?} | {} | {:?} | {:?}",
        std::mem::discriminant(&item.cause),
        item.cause.to_token_stream(),
        item.reason,
        item.message,
        notes,
        help
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

/// A result with several errors is the tokens of its Referrer, then the tokens of each error in
/// order. It is read back with the same errors, item by item, and a Referrer equal to the original;
/// writing it twice, or writing the one that was read, gives the same tokens.
#[test]
fn a_result_with_errors_is_written_and_read_back() {
    let result = with_errors();
    assert!(result.errors().len() >= 5, "{:?}", describe_all(&result));
    let referrer = result.clone().into_referrer();

    // The form: the Referrer, then one brace group for each error, in the order of `errors()`.
    let tokens = result.to_token_stream();
    let mut expected = referrer.to_token_stream();
    for error in result.errors() {
        let group = error.to_token_stream();
        assert_eq!(group.clone().into_iter().count(), 1);
        expected.extend(group);
    }
    assert_eq!(text(&tokens), text(&expected));
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

/// `allow(cause)`, `allow_prelude()` and `allow_std_crates()` drop the same errors after the trip as
/// before it, each step and all of them, and so does writing the rest again. Accepting every cause
/// leaves a Referrer that `finish()` returns, equal to the original.
#[test]
fn allowing_after_the_round_trip_drops_the_same_errors() {
    let result = with_errors();
    let referrer = result.clone().into_referrer();
    let read_back = read(result.to_token_stream()).unwrap();
    let causes: Vec<Cause> = result.errors().iter().map(|e| e.cause.clone()).collect();

    for cause in &causes {
        let (mut before, mut after) = (result.clone(), read_back.clone());
        before.allow(cause.clone());
        after.allow(cause.clone());
        assert!(before.errors().len() < result.errors().len());
        assert_eq!(describe_all(&after), describe_all(&before));
    }
    let (mut before, mut after) = (result.clone(), read_back.clone());
    before.allow_prelude();
    after.allow_prelude();
    // `Clone`, `Vec<Rel<'_>>` and `Box<dyn ..>` start with names of the prelude, `MyType<'_>` and
    // `dyn RelTr` do not.
    let remaining: Vec<String> = before.errors().iter().map(|e| text(&e.cause)).collect();
    assert_eq!(remaining, ["MyType < '_ >", "RelTr"]);
    assert_eq!(describe_all(&after), describe_all(&before));
    before.allow_std_crates();
    after.allow_std_crates();
    assert_eq!(describe_all(&after), describe_all(&before));

    // What remains is written and read again, with the same Referrer.
    let rest = read(after.to_token_stream()).unwrap();
    assert_eq!(describe_all(&rest), describe_all(&before));
    assert_eq!(rest.into_referrer(), referrer);

    let mut all = read_back;
    for cause in causes {
        all.allow(cause);
    }
    assert!(all.errors().is_empty());
    assert_eq!(all.finish().unwrap(), referrer);
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

/// A brace group after the Referrer is read as an error: one that is not fails with the message of
/// `ErrorItem`'s `Parse`.
#[test]
fn a_group_that_is_not_an_error_is_refused() {
    let mut tokens = without_errors().to_token_stream();
    tokens.extend(quote!({ nothing }));
    let message = read(tokens).unwrap_err().to_string();
    let expected = parse2::<ErrorItem>(quote!({ nothing }))
        .unwrap_err()
        .to_string();
    assert_eq!(message, expected);
    assert!(!message.contains("incompatible type-leak"), "{}", message);
}

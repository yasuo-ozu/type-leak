//! The spans of the errors: a macro reports each error at the tokens that cause it.
//!
//! `ErrorItem::span()` and the spans of the notes and the help are where a proc macro puts its
//! diagnostic, so a wrong one points the user at the wrong place, and nothing else
//! shows it: the messages are the same. These tests parse the code from a string with
//! `syn::parse_str()`, so that every token has a line and a column, and compare the span of an
//! error with the place of the offending path in the string. They need the feature
//! `span-locations` of `proc-macro2`, which the package of these tests asks for, and the others
//! cannot have (see `Cargo.toml`): `tests/spans.rs` of `type-leak` runs them.

mod support;

use support::{assert_span_at, location_of};
use syn::spanned::Spanned;
use syn::*;
use template_quote::ToTokens;
use type_leak::*;

/// The messages of the errors, one per line.
fn messages(errors: &[ErrorItem]) -> String {
    errors
        .iter()
        .map(|item| item.message.as_str())
        .collect::<Vec<_>>()
        .join("\n")
}

/// A guard for the others: without a location, every span is at the line 0.
#[test]
fn spans_parsed_from_a_string_have_a_location() {
    let ty: Type = parse_str("A\n    <B>").unwrap();
    let span = syn::spanned::Spanned::span(&ty);
    assert_eq!(
        (span.start().line, span.start().column),
        (1, 0),
        "proc-macro2 needs the feature `span-locations` for these tests"
    );
    assert_eq!((span.end().line, span.end().column), (2, 7));
}

/// The errors are at the path that is relative, the help at the first segment of the path (the
/// suggestion), and the notes at the causes.
#[test]
fn the_span_of_a_relative_path_is_the_path() {
    let source = "trait Tr {\n    fn f(&self, a: MyType<'_>, b: Vec<&'_ u8>);\n}";
    let item: ItemTrait = parse_str(source).unwrap();
    let items = Leaker::new().intern_trait(&item, true).unwrap_err();
    assert_eq!(items.len(), 2, "{}", messages(&items));
    // The error is at the whole type, in the order of the source.
    assert_eq!(items[0].message, "use absolute path: 'MyType < '_ >'");
    assert_span_at(items[0].span(), source, "MyType<'_>", 0);
    assert_eq!(items[1].message, "use absolute path: 'Vec < & '_ u8 >'");
    assert_span_at(items[1].span(), source, "Vec<&'_ u8>", 0);
    // The first one: the note is the cause, at the lifetime that stops the type from being leaked
    // as a whole, and the help is at the name of the path, not at the whole type.
    assert_eq!(items[0].notes.len(), 1);
    assert_eq!(
        items[0].notes[0].1,
        "it cannot be leaked as a whole because of `'_`"
    );
    assert_span_at(items[0].notes[0].0.span(), source, "'_", 0);
    let (at, help) = items[0].help.clone().unwrap();
    assert!(
        help.starts_with("write `MyType` with an absolute path"),
        "{}",
        help
    );
    assert_span_at(at.span(), source, "MyType", 0);
    // The second one: the absolute path of `Vec` is the help, at the first segment of the path,
    // not the whole type, and the cause is the note, at the lifetime of the reference.
    assert_eq!(items[1].notes.len(), 1);
    assert_eq!(
        items[1].notes[0].1,
        "it cannot be leaked as a whole because of `'_`"
    );
    assert_span_at(items[1].notes[0].0.span(), source, "'_", 1);
    let (at, help) = items[1].help.clone().unwrap();
    assert_eq!(help, "use `::std::vec::Vec` instead of `Vec`");
    assert_span_at(at.span(), source, "Vec", 0);
}

/// The span of an error in a bound is that of the path of the trait.
#[test]
fn the_span_of_a_relative_trait_is_the_bound() {
    let source = "trait Tr: Clone {\n    fn f<T: ToString>(&self, t: T);\n}";
    let item: ItemTrait = parse_str(source).unwrap();
    let items = Leaker::new().intern_trait(&item, true).unwrap_err();
    assert_eq!(items.len(), 2, "{}", messages(&items));
    assert_span_at(items[0].span(), source, "Clone", 0);
    assert_span_at(items[0].help.as_ref().unwrap().0.span(), source, "Clone", 0);
    assert_span_at(items[1].span(), source, "ToString", 0);
    assert_span_at(
        items[1].help.as_ref().unwrap().0.span(),
        source,
        "ToString",
        0,
    );
}

/// Each mention of `Self` where it has no meaning is an error at its own tokens.
#[test]
fn the_span_of_self_without_meaning_is_each_mention() {
    let source = "struct S<T: ::p::Tr<Self>, U: ::p::Tr<Self::X>>(T, U) where Self: 'static;";
    let item: ItemStruct = parse_str(source).unwrap();
    let items = Leaker::new()
        .enter_generics(&item.generics, |_| ())
        .unwrap_err();
    assert!(items
        .iter()
        .all(|item| item.reason == Reason::SelfWithoutMeaning));
    assert_eq!(items.len(), 3, "{}", messages(&items));
    let expected = [("Self", 0), ("Self::X", 0), ("Self", 2)];
    for (item, (needle, nth)) in items.iter().zip(expected) {
        assert_eq!(item.message, "`Self` has no meaning here");
        assert_span_at(item.span(), source, needle, nth);
        // The note and the help are about the same tokens.
        assert_eq!(item.notes.len(), 1);
        assert_span_at(item.notes[0].0.span(), source, needle, nth);
        assert_span_at(item.help.as_ref().unwrap().0.span(), source, needle, nth);
    }
    // In a type that the View visits, in a module.
    let source = "Wrap<Self>";
    let ty: Type = parse_str(source).unwrap();
    let items = Leaker::new()
        .enter_mod("m", [], |view| view.visit_type(&ty))
        .unwrap_err();
    assert_span_at(items[0].span(), source, "Self", 0);
}

/// `syn::Error` has one error per item, each at the span of the item.
#[test]
fn the_syn_error_has_the_span_of_each_item() {
    let source = "trait Tr {\n    fn f(&self, a: MyType<'_>, b: Vec<&'_ u8>);\n}";
    let item: ItemTrait = parse_str(source).unwrap();
    let items = Leaker::new().intern_trait(&item, true).unwrap_err();
    let errors: Vec<syn::Error> = to_syn_error(&items).into_iter().collect();
    assert_eq!(errors.len(), 2);
    for (error, item) in errors.iter().zip(&items) {
        let (start, end) = (error.span().start(), error.span().end());
        assert_eq!((start, end), (item.span().start(), item.span().end()));
    }
    assert_span_at(errors[0].span(), source, "MyType<'_>", 0);
    assert_span_at(errors[1].span(), source, "Vec<&'_ u8>", 0);
}

/// A conflict that is not about the path is reported at the type that has it.
#[test]
fn the_span_of_a_conflict_of_a_type_in_a_bound_is_the_type() {
    let source = "trait Tr<T: Fn(Rel)> {\n    fn f(&self, a: Wrap<T>)\n    where\n        Wrap<T>: Sized;\n}";
    let item: ItemTrait = parse_str(source).unwrap();
    let mut config = LeakerConfig::new();
    config.allow_primitive().allow_std_prelude();
    let items = Leaker::from_config(config)
        .intern_trait(&item, true)
        .unwrap_err();
    assert_eq!(
        messages(&items),
        "`Wrap < T >` is in a bound or where predicate, and a bound of its generics names a leaked \
         type"
    );
    assert_eq!(items[0].reason, Reason::UnprovableBound);
    // The type is reported where it is in the predicate, not in the position of the argument.
    assert_span_at(items[0].span(), source, "Wrap<T>", 1);
    let (start, _) = location_of(source, "Wrap<T>", 1);
    assert_eq!(items[0].span().start(), start);
}

/// The line and the column where `tokens` start and end.
fn places(tokens: &proc_macro2::TokenStream) -> (usize, usize, usize, usize) {
    let span = tokens.span();
    (
        span.start().line,
        span.start().column,
        span.end().line,
        span.end().column,
    )
}

/// Reads the errors that [`ToTokens`] wrote, one after the other.
fn parse_items(tokens: proc_macro2::TokenStream) -> Vec<ErrorItem> {
    use syn::parse::Parser;
    (|input: parse::ParseStream| -> Result<Vec<ErrorItem>> {
        let mut items = Vec::new();
        while !input.is_empty() {
            items.push(input.parse()?);
        }
        Ok(items)
    })
    .parse2(tokens)
    .unwrap()
}

/// The tokens of the cause, of the notes and of the help are written as they are, so a macro that
/// writes its errors for another to read reports them at the same places.
#[test]
fn a_serialized_error_keeps_its_spans() {
    let source =
        "trait Tr: Clone {\n    fn f(&self, a: MyType<'_>, b: Box<dyn RelA + RelB<'_>>);\n}";
    let item: ItemTrait = parse_str(source).unwrap();
    let items = Leaker::new().intern_trait(&item, true).unwrap_err();
    assert_eq!(items.len(), 3, "{}", messages(&items));
    let tokens: proc_macro2::TokenStream =
        items.iter().map(|item| item.to_token_stream()).collect();
    let parsed = parse_items(tokens);
    assert_eq!(parsed.len(), items.len());
    for (parsed, item) in parsed.iter().zip(&items) {
        assert_eq!(parsed.message, item.message);
        assert_eq!(parsed.reason, item.reason);
        assert_eq!(
            places(&parsed.cause.to_token_stream()),
            places(&item.cause.to_token_stream())
        );
        assert_eq!(parsed.notes.len(), item.notes.len());
        for ((at, note), (expected_at, expected_note)) in parsed.notes.iter().zip(&item.notes) {
            assert_eq!(note, expected_note);
            assert_eq!(places(at), places(expected_at));
        }
        let (at, help) = parsed.help.as_ref().unwrap();
        let (expected_at, expected_help) = item.help.as_ref().unwrap();
        assert_eq!(help, expected_help);
        assert_eq!(places(at), places(expected_at));
    }
    // They are where the source has the offending tokens, and the `syn::Error`s are at the same
    // places as those of the errors that were written.
    assert_span_at(parsed[0].span(), source, "Clone", 0);
    assert_span_at(parsed[1].span(), source, "MyType<'_>", 0);
    assert_span_at(parsed[2].span(), source, "Box<dyn RelA + RelB<'_>>", 0);
    assert_span_at(parsed[2].help.as_ref().unwrap().0.span(), source, "Box", 0);
    assert_span_at(parsed[2].notes[1].0.span(), source, "RelA", 0);
    let direct: Vec<_> = to_syn_error(&items).into_iter().collect();
    let again: Vec<_> = to_syn_error(&parsed).into_iter().collect();
    assert_eq!(direct.len(), again.len());
    for (direct, again) in direct.iter().zip(&again) {
        assert_eq!(direct.to_string(), again.to_string());
        assert_eq!(
            (direct.span().start(), direct.span().end()),
            (again.span().start(), again.span().end())
        );
    }
}

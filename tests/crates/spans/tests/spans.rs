//! The spans of the errors: a macro reports each error at the tokens that cause it.
//!
//! `NotInternableError::span()`, `ErrorItem::span` and the spans of the notes and the help are where a proc
//! macro puts its diagnostic, so a wrong one points the user at the wrong place, and nothing else
//! shows it: the messages are the same. These tests parse the code from a string with
//! `syn::parse_str()`, so that every token has a line and a column, and compare the span of an
//! error with the place of the offending path in the string. They need the feature
//! `span-locations` of `proc-macro2`, which the package of these tests asks for, and the others
//! cannot have (see `Cargo.toml`): `tests/spans.rs` of `type-leak` runs them.

mod support;

use support::{assert_span_at, location_of};
use syn::*;
use type_leak::*;

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
    let err = Leaker::new().intern_trait(&item, true).unwrap_err();
    let items: Vec<_> = err.items().collect();
    assert_eq!(items.len(), 2, "{}", err);
    // The error is at the whole type, in the order of the source.
    assert_eq!(items[0].message, "use absolute path: 'MyType < '_ >'");
    assert_span_at(items[0].span, source, "MyType<'_>", 0);
    assert_eq!(items[1].message, "use absolute path: 'Vec < & '_ u8 >'");
    assert_span_at(items[1].span, source, "Vec<&'_ u8>", 0);
    // `NotInternableError::span()` is that of the first one.
    assert_span_at(err.span(), source, "MyType<'_>", 0);
    // The first one: the note is the cause, at the lifetime that stops the type from being leaked
    // as a whole, and the help is at the name of the path, not at the whole type.
    assert_eq!(items[0].notes.len(), 1);
    assert_eq!(
        items[0].notes[0].1,
        "it cannot be leaked as a whole because of `'_`"
    );
    assert_span_at(items[0].notes[0].0, source, "'_", 0);
    let (span, help) = items[0].help.clone().unwrap();
    assert!(help.starts_with("write `MyType` with an absolute path"), "{}", help);
    assert_span_at(span, source, "MyType", 0);
    // The second one: the absolute path of `Vec` is the help, at the first segment of the path,
    // not the whole type, and the cause is the note, at the lifetime of the reference.
    assert_eq!(items[1].notes.len(), 1);
    assert_eq!(
        items[1].notes[0].1,
        "it cannot be leaked as a whole because of `'_`"
    );
    assert_span_at(items[1].notes[0].0, source, "'_", 1);
    let (span, help) = items[1].help.clone().unwrap();
    assert_eq!(help, "use `::std::vec::Vec` instead of `Vec`");
    assert_span_at(span, source, "Vec", 0);
}

/// The span of an error in a bound is that of the path of the trait.
#[test]
fn the_span_of_a_relative_trait_is_the_bound() {
    let source = "trait Tr: Clone {\n    fn f<T: ToString>(&self, t: T);\n}";
    let item: ItemTrait = parse_str(source).unwrap();
    let err = Leaker::new().intern_trait(&item, true).unwrap_err();
    let items: Vec<_> = err.items().collect();
    assert_eq!(items.len(), 2, "{}", err);
    assert_span_at(items[0].span, source, "Clone", 0);
    assert_span_at(items[0].help.as_ref().unwrap().0, source, "Clone", 0);
    assert_span_at(items[1].span, source, "ToString", 0);
    assert_span_at(items[1].help.as_ref().unwrap().0, source, "ToString", 0);
    assert_span_at(err.span(), source, "Clone", 0);
}

/// Each mention of `Self` where it has no meaning is an error at its own tokens.
#[test]
fn the_span_of_self_without_meaning_is_each_mention() {
    let source = "struct S<T: ::p::Tr<Self>, U: ::p::Tr<Self::X>>(T, U) where Self: 'static;";
    let item: ItemStruct = parse_str(source).unwrap();
    let err = Leaker::new()
        .enter_generics(&item.generics, |_| ())
        .unwrap_err();
    assert!(matches!(err, NotInternableError::SelfWithoutMeaning(_)));
    let items: Vec<_> = err.items().collect();
    assert_eq!(items.len(), 3, "{}", err);
    let expected = [("Self", 0), ("Self::X", 0), ("Self", 2)];
    for (item, (needle, nth)) in items.iter().zip(expected) {
        assert_eq!(item.message, "`Self` has no meaning here");
        assert_span_at(item.span, source, needle, nth);
        // The note and the help are about the same tokens.
        assert_eq!(item.notes.len(), 1);
        assert_span_at(item.notes[0].0, source, needle, nth);
        assert_span_at(item.help.as_ref().unwrap().0, source, needle, nth);
    }
    assert_span_at(err.span(), source, "Self", 0);
    // In a type that the View visits, in a module.
    let source = "Wrap<Self>";
    let ty: Type = parse_str(source).unwrap();
    let err = Leaker::new()
        .enter_mod("m", [], |view| view.visit_type(&ty))
        .unwrap_err();
    assert_span_at(err.span(), source, "Self", 0);
}

/// `syn::Error` has one error per item, each at the span of the item.
#[test]
fn the_syn_error_has_the_span_of_each_item() {
    let source = "trait Tr {\n    fn f(&self, a: MyType<'_>, b: Vec<&'_ u8>);\n}";
    let item: ItemTrait = parse_str(source).unwrap();
    let err = Leaker::new().intern_trait(&item, true).unwrap_err();
    let items: Vec<_> = err.items().collect();
    let errors: Vec<syn::Error> = syn::Error::from(err).into_iter().collect();
    assert_eq!(errors.len(), 2);
    for (error, item) in errors.iter().zip(&items) {
        let (start, end) = (error.span().start(), error.span().end());
        assert_eq!((start, end), (item.span.start(), item.span.end()));
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
    let err = Leaker::from_config(config)
        .intern_trait(&item, true)
        .unwrap_err();
    assert_eq!(
        err.to_string(),
        "`Wrap < T >` is in a bound or where predicate, and a bound of its generics names a leaked \
         type"
    );
    // The type is reported where it is in the predicate, not in the position of the argument.
    assert_span_at(err.span(), source, "Wrap<T>", 1);
    let (start, _) = location_of(source, "Wrap<T>", 1);
    assert_eq!(err.items().next().unwrap().span.start(), start);
}

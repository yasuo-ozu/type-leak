//! The spans of the errors: a macro reports each error at the tokens that cause it.
//!
//! `ErrorItem::span()` and the spans of the notes and the help are where a proc macro puts its
//! diagnostic, so a wrong one points the user at the wrong place, and nothing else
//! shows it: the messages are the same. These tests parse the code from a string with
//! `syn::parse_str()`, so that every token has a line and a column, and compare the span of an
//! error with the place of the offending path in the string. They need the feature
//! `span-locations` of `proc-macro2`, which the package of these tests asks for, and the others
//! cannot have (see `Cargo.toml`): `tests/diagnostic.rs` of `type-leak` runs them.

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

/// The errors as one `syn::Error`, one for each in order. Panics if there is none.
fn to_syn_error(items: &[ErrorItem]) -> syn::Error {
    let mut errors = items.iter().cloned().map(syn::Error::from);
    let mut combined = errors.next().expect("at least one error");
    for error in errors {
        combined.combine(error);
    }
    combined
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
    let mut leaker = Leaker::new();
    leaker.intern_trait(&item);
    let result = leaker.result();
    let items = result.errors();
    assert_eq!(items.len(), 2, "{}", messages(items));
    // The error is at the whole type, in the order of the source.
    assert_eq!(items[0].message, "use absolute path: 'MyType<'_>'");
    assert_span_at(items[0].span(), source, "MyType<'_>", 0);
    assert_eq!(items[1].message, "use absolute path: 'Vec<&'_ u8>'");
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
    let mut leaker = Leaker::new();
    leaker.intern_trait(&item);
    let result = leaker.result();
    let items = result.errors();
    assert_eq!(items.len(), 2, "{}", messages(items));
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
    let mut leaker = Leaker::new();
    leaker.enter_generics(&item.generics, |_| ());
    let result = leaker.result();
    let items = result.errors();
    assert!(items
        .iter()
        .all(|item| item.reason == Reason::SelfWithoutMeaning));
    assert_eq!(items.len(), 3, "{}", messages(items));
    let expected = [("Self", 0), ("Self::X", 0), ("Self", 2)];
    for (item, (needle, nth)) in items.iter().zip(expected) {
        assert_eq!(item.message, "`Self` has no meaning here");
        assert_span_at(item.span(), source, needle, nth);
        // The note and the help are about the same tokens.
        assert_eq!(item.notes.len(), 1);
        assert_span_at(item.notes[0].0.span(), source, needle, nth);
        assert_span_at(item.help.as_ref().unwrap().0.span(), source, needle, nth);
    }
    // In a type that the View visits, in a scope without generics.
    let source = "Wrap<Self>";
    let ty: Type = parse_str(source).unwrap();
    let mut leaker = Leaker::new();
    leaker.enter_generics(&Generics::default(), |view| view.visit_type(&ty));
    let result = leaker.result();
    let items = result.errors();
    assert_span_at(items[0].span(), source, "Self", 0);
}

/// `syn::Error` has one error per item, each at the span of the item.
#[test]
fn the_syn_error_has_the_span_of_each_item() {
    let source = "trait Tr {\n    fn f(&self, a: MyType<'_>, b: Vec<&'_ u8>);\n}";
    let item: ItemTrait = parse_str(source).unwrap();
    let mut leaker = Leaker::new();
    leaker.intern_trait(&item);
    let result = leaker.result();
    let items = result.errors();
    let errors: Vec<syn::Error> = to_syn_error(items).into_iter().collect();
    assert_eq!(errors.len(), 2);
    for (error, item) in errors.iter().zip(items) {
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
    let config = Rules::new().primitive().prelude();
    let mut leaker = Leaker::with_rule(config);
    leaker.intern_trait(&item);
    let result = leaker.result();
    let items = result.errors();
    assert_eq!(
        messages(items),
        "`Wrap<T>` is in a bound or where predicate, and a bound of its generics names a leaked \
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
    let mut leaker = Leaker::new();
    leaker.intern_trait(&item);
    let result = leaker.result();
    let items = result.errors();
    // `Box` and the trait object in it, which has an error of its own.
    assert_eq!(items.len(), 4, "{}", messages(items));
    let tokens: proc_macro2::TokenStream =
        items.iter().map(|item| item.to_token_stream()).collect();
    let parsed = parse_items(tokens);
    assert_eq!(parsed.len(), items.len());
    for (parsed, item) in parsed.iter().zip(items) {
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
    assert_span_at(parsed[3].span(), source, "RelA", 0);
    assert_span_at(parsed[3].notes[1].0.span(), source, "RelB<'_>", 0);
    let direct: Vec<_> = to_syn_error(items).into_iter().collect();
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

/// The same through a `LeakerResult`: the cause of each error is where the source has the offending
/// tokens after the trip, and the errors are read back in order.
#[test]
fn a_serialized_leaker_result_keeps_its_spans() {
    let source =
        "trait Tr: Clone {\n    fn f(&self, a: MyType<'_>, b: Box<dyn RelA + RelB<'_>>);\n}";
    let item: ItemTrait = parse_str(source).unwrap();
    let mut leaker = Leaker::new();
    leaker.intern_trait(&item);
    let result = leaker.result();
    assert_eq!(result.errors().len(), 4, "{}", messages(result.errors()));
    let read: LeakerResult = parse2(result.to_token_stream()).unwrap();
    assert_eq!(read.errors().len(), 4);
    for (read, item) in read.errors().iter().zip(result.errors()) {
        assert_eq!(read.message, item.message);
        assert_eq!(
            places(&read.cause.to_token_stream()),
            places(&item.cause.to_token_stream())
        );
    }
    assert_span_at(read.errors()[0].span(), source, "Clone", 0);
    assert_span_at(read.errors()[1].span(), source, "MyType<'_>", 0);
    assert_span_at(
        read.errors()[2].span(),
        source,
        "Box<dyn RelA + RelB<'_>>",
        0,
    );
    assert_span_at(
        read.errors()[2].help.as_ref().unwrap().0.span(),
        source,
        "Box",
        0,
    );
    assert_span_at(read.errors()[3].span(), source, "RelA", 0);
}

/// A cause is the same whatever its spans: the one that is written elsewhere (other lines and
/// columns) drops the error (`LeakerResult::allow()`), is equal to the cause of the error, and
/// hashes the same.
#[test]
fn an_accepted_cause_is_the_same_whatever_its_spans() {
    use std::hash::{Hash, Hasher};
    let source = "impl ::p::S {\n    fn f(_: ::p::W<'_, dyn RelTr>) {}\n}";
    let item: ItemImpl = parse_str(source).unwrap();
    let mut leaker = Leaker::new();
    leaker.intern_impl(&item);
    let result = leaker.result();
    let errors = result.errors();
    assert_eq!(errors.len(), 1, "{}", messages(errors));
    assert_span_at(errors[0].span(), source, "RelTr", 0);
    let elsewhere: Path = parse_str("\n\n      RelTr").unwrap();
    assert_ne!(
        places(&elsewhere.to_token_stream()),
        places(&errors[0].cause.to_token_stream())
    );
    let cause = Cause::Trait(elsewhere);
    assert_eq!(cause, errors[0].cause);
    let hash = |cause: &Cause| {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        cause.hash(&mut hasher);
        hasher.finish()
    };
    assert_eq!(hash(&cause), hash(&errors[0].cause));
    let allowed = result.clone();
    assert!(allowed
        .allow(|c: &Cause, _: Place| *c == cause)
        .errors()
        .is_empty());
    // Another cause drops nothing.
    let other = result.clone();
    assert_eq!(
        other
            .allow(|c: &Cause, _: Place| *c == Cause::Trait(parse_quote!(Other)))
            .errors()
            .len(),
        1
    );
}

/// The line and the column where each token of `tokens` starts and ends, the groups included.
fn token_places(tokens: proc_macro2::TokenStream) -> Vec<(String, (usize, usize), (usize, usize))> {
    use proc_macro2::TokenTree;
    let mut places = Vec::new();
    for tree in tokens {
        let span = tree.span();
        places.push((
            tree.to_string(),
            (span.start().line, span.start().column),
            (span.end().line, span.end().column),
        ));
        if let TokenTree::Group(group) = tree {
            places.extend(token_places(group.stream()));
        }
    }
    places
}

/// Asserts that every token of `tokens` is in `source`, on its first line, and spans some of it. A
/// token of `Span::call_site()` is not: it is the empty span at the start of the first line here,
/// and the whole attribute in a proc macro.
fn assert_in_source(tokens: proc_macro2::TokenStream, source: &str) {
    for (token, start, end) in token_places(tokens) {
        assert!(
            start.0 == 1 && end.0 == 1 && start.1 < end.1 && end.1 <= source.chars().count(),
            "`{}` at {:?}..{:?} is not in {:?}",
            token,
            start,
            end,
            source
        );
    }
}

/// The error of the first macro cause of the item `source`, with the macro `two!()` in an array
/// length. `u8` is allowed, so `two` is the first error.
fn macro_error(source: &str) -> ErrorItem {
    let item: ItemStruct = parse_str(source).unwrap();
    let config = Rules::new().primitive();
    let mut leaker = Leaker::with_rule(config);
    leaker.intern_struct(&item);
    let result = leaker.result();
    assert_eq!(result.errors().len(), 1, "{}", messages(result.errors()));
    result.errors()[0].clone()
}

/// The `!` of a macro cause has the span of the last name of its path (the call site before), so
/// every token of the cause is in the item and the error is reported there, not at the attribute.
#[test]
fn the_bang_of_a_macro_cause_is_at_the_name_of_the_macro() {
    let source = "struct S([u8; two!()]);";
    let error = macro_error(source);
    assert!(matches!(error.cause, Cause::Macro(_)));
    let tokens = error.cause.to_token_stream();
    assert_in_source(tokens.clone(), source);
    // `two` and the `!` have the span of `two`.
    let places = token_places(tokens);
    assert_eq!(places.len(), 2);
    assert_eq!(places[0].0, "two");
    assert_eq!(places[1].0, "!");
    assert_eq!(places[0].1, places[1].1);
    assert_eq!(places[0].2, places[1].2);
    assert_span_at(error.span(), source, "two", 0);
    // A path of several segments: the last name.
    let source = "struct S([u8; p::m::two!()]);";
    let error = macro_error(source);
    assert_in_source(error.cause.to_token_stream(), source);
    let places = token_places(error.cause.to_token_stream());
    assert_eq!(places.last().unwrap().0, "!");
    let (start, end) = location_of(source, "two", 0);
    assert_eq!(
        places.last().unwrap().1,
        (start.line, start.column),
        "{:?}",
        places
    );
    assert_eq!(places.last().unwrap().2, (end.line, end.column));
    assert_eq!(error.span().end(), end);
    assert_eq!(error.span().start(), location_of(source, "p::m::two", 0).0);
}

/// Every way to report the error of a macro cause is at the name of the macro: `span()`,
/// `syn::Error` (`From`, also combined), `to_compile_error()`, the notes and the help.
#[test]
fn an_error_of_a_macro_cause_is_reported_at_the_name_of_the_macro() {
    let source = "struct S([u8; two!()]);";
    let error = macro_error(source);
    let (start, end) = location_of(source, "two", 0);
    // `ErrorItem::span()`.
    assert_eq!((error.span().start(), error.span().end()), (start, end));
    // `From<ErrorItem> for syn::Error`, and the one combined.
    let converted: syn::Error = error.clone().into();
    assert_eq!(
        (converted.span().start(), converted.span().end()),
        (start, end)
    );
    let combined = to_syn_error(std::slice::from_ref(&error));
    assert_eq!(
        (combined.span().start(), combined.span().end()),
        (start, end)
    );
    // `compile_error!` of `to_compile_error()`: every token of it is in the source.
    assert_in_source(error.to_compile_error(), source);
    // The note is at the call (`two!()`) and the help at the name of the macro.
    assert_eq!(error.notes.len(), 1);
    assert_in_source(error.notes[0].0.clone(), source);
    assert_span_at(error.notes[0].0.span(), source, "two!()", 0);
    let (at, _) = error.help.as_ref().unwrap();
    assert_in_source(at.clone(), source);
    assert_span_at(at.span(), source, "two", 0);
    // Written for another macro to read, and read back: the same places.
    let read = parse_items(error.to_token_stream());
    assert_eq!(read.len(), 1);
    assert_span_at(read[0].span(), source, "two", 0);
    assert_in_source(read[0].cause.to_token_stream(), source);
    assert_eq!(read[0].cause, error.cause);
}

/// The macro cause of a body (`check_bodies`), and of a mention of `Self` that has no meaning
/// (`Self::m!()`), has no token at the call site either.
#[test]
fn a_macro_cause_of_a_value_and_of_self_is_in_the_item() {
    let source = "trait Tr {\n    fn f(&self) -> u8 {\n        two!() + 1\n    }\n}";
    let item: ItemTrait = parse_str(source).unwrap();
    let config = Rules::new().primitive();
    let mut leaker = Leaker::with_rule(config);
    leaker.check_bodies = true;
    leaker.intern_trait(&item);
    let result = leaker.result();
    let items = result.errors();
    assert_eq!(items.len(), 1, "{}", messages(items));
    assert!(matches!(items[0].cause, Cause::Macro(_)));
    assert_span_at(items[0].span(), source, "two", 0);
    for (token, start, end) in token_places(items[0].cause.to_token_stream()) {
        assert_eq!((start.0, end.0), (3, 3), "{}", token);
    }
    // `Self` where a scope gives it no meaning, in the path of a macro.
    let source = "Wrap<[u8; Self::two!()]>";
    let ty: Type = parse_str(source).unwrap();
    let mut leaker = Leaker::new();
    leaker.enter_generics(&Generics::default(), |view| view.visit_type(&ty));
    let result = leaker.result();
    let selfs: Vec<&ErrorItem> = result
        .errors()
        .iter()
        .filter(|item| item.reason == Reason::SelfWithoutMeaning)
        .collect();
    assert!(!selfs.is_empty(), "{}", messages(result.errors()));
    assert!(selfs
        .iter()
        .any(|item| matches!(item.cause, Cause::Macro(_))));
    for item in selfs {
        assert_in_source(item.cause.to_token_stream(), source);
        assert_in_source(item.to_compile_error(), source);
        for (at, _) in item.notes.iter().chain(item.help.iter()) {
            assert_in_source(at.clone(), source);
        }
    }
}

/// `Leaker::assertions()` writes the check of each primitive name at the name, so that rustc
/// reports a shadowed one (`struct u8;`) at the type in the item, not at the macro.
#[test]
fn the_check_of_a_primitive_name_is_at_the_name() {
    let source = "struct S {\n    a: u8,\n    b: u16,\n    c: u8,\n}";
    let item: ItemStruct = parse_str(source).unwrap();
    let mut leaker = Leaker::with_rule(Rules::new().primitive());
    leaker.intern_struct(&item);
    let checks: ItemConst = parse2(leaker.assertions()).unwrap();
    let block = match &*checks.expr {
        Expr::Block(block) => &block.block,
        other => panic!("{}", other.to_token_stream()),
    };
    let mut names = Vec::new();
    for stmt in &block.stmts {
        let check = match stmt {
            Stmt::Item(Item::Fn(check)) => check,
            other => panic!("{}", other.to_token_stream()),
        };
        let name = check.sig.ident.to_string();
        let primitive = name
            .trim_start_matches("__type_leak_assert_")
            .trim_end_matches("_is_primitive")
            .to_string();
        // The first place that names it.
        let (start, end) = location_of(source, &primitive, 0);
        for (token, at_start, at_end) in token_places(check.to_token_stream()) {
            assert_eq!(
                (at_start, at_end),
                ((start.line, start.column), (end.line, end.column)),
                "`{}` in the check of `{}`",
                token,
                primitive
            );
        }
        names.push(primitive);
    }
    assert_eq!(names, ["u8", "u16"]);
}

/// The error of a `Self` in the self type of an impl is at the `Self` of `Self: 'a`, which ties the
/// entry to `Self`, not at the first `Self` of the where clause (`Self::Y`, whose predicate has
/// nothing to do with the entry).
#[test]
fn the_span_of_the_self_that_ties_an_entry_is_the_one_of_its_outlives_predicate() {
    let source = "impl<'a, T, U> Tr<U> for W<'a, Rel<'a, T>> where U: Into<Self::Y>, Self: 'a {}";
    let item: ItemImpl = parse_str(source).unwrap();
    let mut leaker = Leaker::new();
    leaker.intern_impl(&item);
    let result = leaker.result();
    let selfs: Vec<&ErrorItem> = result
        .errors()
        .iter()
        .filter(|item| item.reason == Reason::SelfInImplSelfType)
        .collect();
    assert!(!selfs.is_empty(), "{}", messages(result.errors()));
    for item in selfs {
        // The headline is about the type, and the last note, which says what stops the type from
        // being leaked whole, about the `Self` that ties it.
        let (at, note) = item.notes.last().unwrap();
        assert_eq!(note, "it cannot be leaked as a whole because of `Self`");
        assert_span_at(at.span(), source, "Self", 1);
    }
}

/// A `syn::Error` of an item (and the `compile_error!` of it) starts at the first token of the
/// cause and ends at the last, so that rustc underlines the whole cause on stable, where the
/// `Span::join` of the span of the item is not available: `new_spanned()`, not `new()` with one
/// span.
#[test]
fn the_syn_error_starts_at_the_first_token_of_the_cause_and_ends_at_the_last() {
    let source = "trait Tr<T: inner::Base> {}";
    let item: ItemTrait = parse_str(source).unwrap();
    let mut leaker = Leaker::new();
    leaker.intern_trait(&item);
    let result = leaker.result();
    let items = result.errors();
    assert_eq!(items.len(), 1, "{}", messages(items));
    let (first, last) = (
        location_of(source, "inner", 0),
        location_of(source, "Base", 0),
    );
    let at = |(start, end): (proc_macro2::LineColumn, proc_macro2::LineColumn)| {
        ((start.line, start.column), (end.line, end.column))
    };
    for compile_error in [
        items[0].to_compile_error(),
        syn::Error::from(items[0].clone()).to_compile_error(),
    ] {
        // `compile_error ! { "message" }` with the two colons before it: the name and the `!` are
        // at the first token of the cause, and the braces at the last.
        let places = token_places(compile_error);
        let name = places
            .iter()
            .position(|place| place.0 == "compile_error")
            .unwrap_or_else(|| panic!("{:?}", places));
        for place in &places[name..name + 2] {
            assert_eq!((place.1, place.2), at(first), "{:?}", places);
        }
        assert_eq!(places[name + 1].0, "!", "{:?}", places);
        assert_eq!(
            (places[name + 2].1, places[name + 2].2),
            at(last),
            "{:?}",
            places
        );
    }
    // `syn::Error::span()` joins them: the whole cause.
    assert_span_at(
        syn::Error::from(items[0].clone()).span(),
        source,
        "inner::Base",
        0,
    );
}

//! Where the parts of a string are, to compare with the spans of code parsed from it.

/// The start and the end of the `nth` (from 0) occurrence of `needle` in `source`, as the spans of
/// `proc_macro2` locate them when the code is parsed from a string (`syn::parse_str()`): the line
/// counts from 1, the column from 0, in characters. The `needle` is on one line.
///
/// The spans have a location only if `proc-macro2` is built with the feature `span-locations`,
/// which `Cargo.toml` of this package asks for.
#[allow(dead_code)]
pub fn location_of(
    source: &str,
    needle: &str,
    nth: usize,
) -> (proc_macro2::LineColumn, proc_macro2::LineColumn) {
    assert!(!needle.contains('\n'), "{:?}", needle);
    let offset = source
        .match_indices(needle)
        .nth(nth)
        .unwrap_or_else(|| panic!("no occurrence {} of {:?} in {:?}", nth, needle, source))
        .0;
    let before = &source[..offset];
    let line = 1 + before.matches('\n').count();
    let column = before.chars().rev().take_while(|&c| c != '\n').count();
    (
        proc_macro2::LineColumn { line, column },
        proc_macro2::LineColumn {
            line,
            column: column + needle.chars().count(),
        },
    )
}

/// Asserts that `span` covers the `nth` occurrence of `needle` in `source` (see [`location_of()`]).
#[allow(dead_code)]
pub fn assert_span_at(span: proc_macro2::Span, source: &str, needle: &str, nth: usize) {
    let (start, end) = location_of(source, needle, nth);
    assert_eq!(
        (span.start(), span.end()),
        (start, end),
        "the span does not cover occurrence {} of {:?} in {:?}",
        nth,
        needle,
        source
    );
}

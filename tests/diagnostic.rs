//! Checks how rustc renders the errors of the Leaker (`ErrorItem`) emitted by a proc macro with
//! `ErrorItem::to_compile_error()`, by building `tests/crates/diagnostic` (which fails to compile).
//! The first items of `tests/crates/diagnostic_serialized` are the same, but the macro writes its
//! errors as tokens, reads them back and then reports them: rustc renders them in the same way, at
//! the same places.
//!
//! On nightly, the notes and the help are those of the diagnostic, each with its own span and
//! source preview; on stable, they are appended to the message (`= note: ..`, then `= help: ..`, as
//! rustc writes its own).
//!
//! The fixture has the errors of
//!
//! - an item with one offending path (`Prelude`, `Bound`, `Nested`),
//! - an item with several (`Multi`): each is an error of its own, at its own path, in order,
//! - `Self` where no scope declares it (`NoSelf`, `Leaker::enter_generics()`): one for each
//!   mention, and
//! - a Referrer that a macro cannot read, because it was serialized by another version of
//!   `type-leak` (`Tag*`), whatever it carries as the format tag.
//!
//! The build must report exactly these errors, and no other.
//!
//! `tests/crates/diagnostic_result` is a Referrer's crate whose macro got the errors of the Leaker's
//! macro in another crate (the serialized `LeakerResult`), allowed some, and reports the others. By
//! default rustc points an error at the invocation, as it does for every span inside a macro of
//! another crate; on nightly with `-Zmacro-backtrace` it points at the cause, in the Leaker's crate.

use type_leak::Referrer;

#[path = "common/nested.rs"]
mod nested;

/// Builds the fixture crate, returning the rendered diagnostics.
fn build_fixture() -> String {
    let built = nested::cargo("diagnostic", &["check"], &[]);
    assert!(!built.success, "{}", built.stderr);
    built.stderr
}

/// An error as rustc printed it.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Reported {
    /// `error: use absolute path: 'Clone'`, without whitespace.
    headline: String,
    /// `line:column` of the first source preview, which is the span of the error.
    at: String,
    /// The notes of the diagnostic (rustc's own, such as the origin of an error, excluded).
    notes: Vec<String>,
    /// The helps of the diagnostic.
    helps: Vec<String>,
}

/// The text without the whitespace at the end of each line.
fn trimmed(text: &str) -> String {
    text.lines()
        .map(str::trim_end)
        .collect::<Vec<_>>()
        .join("\n")
}

fn squash(text: &str) -> String {
    text.split_whitespace().collect()
}

/// The errors of the output of rustc. A diagnostic starts at a line `error..` and goes on to the
/// next one. A note is `note: ..` (nightly: a diagnostic of its own, below the error) or `= note:
/// ..` (stable: in the message).
fn reported(stderr: &str) -> Vec<Reported> {
    let mut errors = Vec::new();
    let mut lines = stderr.lines().peekable();
    while let Some(line) = lines.next() {
        if !line.starts_with("error") || line.starts_with("error: could not compile") {
            continue;
        }
        let mut at = String::new();
        let mut notes = Vec::new();
        let mut helps = Vec::new();
        while let Some(next) = lines.peek() {
            if next.starts_with("error") || next.starts_with("warning") {
                break;
            }
            let next = lines.next().unwrap().trim();
            if let Some(position) = next.strip_prefix("--> lib.rs:") {
                if at.is_empty() {
                    at = position.to_string();
                }
            }
            let note = next
                .strip_prefix("= note: ")
                .or_else(|| next.strip_prefix("note: "));
            if let Some(note) = note.filter(|n| !n.starts_with("this error originates")) {
                notes.push(note.to_string());
            }
            let help = next
                .strip_prefix("= help: ")
                .or_else(|| next.strip_prefix("help: "));
            if let Some(help) = help {
                helps.push(help.to_string());
            }
        }
        errors.push(Reported {
            headline: squash(line),
            at,
            notes,
            helps,
        });
    }
    errors
}

fn error(headline: &str, at: &str, notes: &[&str], helps: &[&str]) -> Reported {
    Reported {
        headline: squash(&format!("error: {}", headline)),
        at: at.to_string(),
        notes: notes.iter().map(|note| note.to_string()).collect(),
        helps: helps.iter().map(|help| help.to_string()).collect(),
    }
}

/// The error of a Referrer whose format tag is `found`.
fn format_error(found: &str, at: &str) -> Reported {
    error(
        &format!(
            "this Referrer was serialized by an incompatible type-leak ({}, expected {}): the \
             macros which emit and parse it must use the same version of type-leak",
            found,
            Referrer::FORMAT_VERSION
        ),
        at,
        &[],
        &[],
    )
}

#[test]
fn note_is_rendered() {
    let stderr = build_fixture();
    let expected: &[&str] = if nested::rustc_is_nightly() {
        &[
            // The help has its own span: the first segment of the offending path.
            "\
error: use absolute path: 'Clone'
 --> lib.rs:2:20
  |
2 | pub trait Prelude: Clone {}
  |                    ^^^^^
  |
help: use `::core::clone::Clone` instead of `Clone`
 --> lib.rs:2:20
  |
2 | pub trait Prelude: Clone {}
  |                    ^^^^^
",
            // The error spans the whole bound, while the help spans `Fn`.
            "
 --> lib.rs:5:18
  |
5 | pub trait Bound: for<'a> Fn(&'a u8) {}
  |                  ^^^^^^^^^^^^^^^^^^
  |
help: use `::core::ops::Fn` instead of `Fn`
 --> lib.rs:5:26
  |
5 | pub trait Bound: for<'a> Fn(&'a u8) {}
  |                          ^^
",
            // Each note has its own span: why the type cannot be leaked as a whole, and the other
            // paths inside it. The help comes last.
            "\
note: it cannot be leaked as a whole because of `'_`
 --> lib.rs:9:40
  |
9 |     fn f(&self, x: Box<dyn RelA + RelB<'_>>);
  |                                        ^^
note: `RelA` also depends on the type context
 --> lib.rs:9:28
  |
9 |     fn f(&self, x: Box<dyn RelA + RelB<'_>>);
  |                            ^^^^
note: `RelB < '_ >` also depends on the type context
 --> lib.rs:9:35
  |
9 |     fn f(&self, x: Box<dyn RelA + RelB<'_>>);
  |                                   ^^^^^^^^
help: use `::std::boxed::Box` instead of `Box`
 --> lib.rs:9:20
  |
9 |     fn f(&self, x: Box<dyn RelA + RelB<'_>>);
  |                    ^^^
",
        ]
    } else {
        &[
            "\
error: use absolute path: 'Clone'

         = help: use `::core::clone::Clone` instead of `Clone`

 --> lib.rs:2:20
  |
2 | pub trait Prelude: Clone {}
  |                    ^^^^^
",
            "
         = help: use `::core::ops::Fn` instead of `Fn`

 --> lib.rs:5:18
",
            // The notes (why) come first, then the help (what to write).
            "\
error: use absolute path: 'Box < dyn RelA + RelB < '_ > >'

         = note: it cannot be leaked as a whole because of `'_`
         = note: `RelA` also depends on the type context
         = note: `RelB < '_ >` also depends on the type context
         = help: use `::std::boxed::Box` instead of `Box`

 --> lib.rs:9:20
",
        ]
    };
    // The lines that rustc pads with spaces (the empty line between a message and its notes) are
    // compared without them.
    let stderr = trimmed(&stderr);
    for expected in expected {
        let expected = trimmed(expected);
        assert!(
            stderr.contains(&expected),
            "expected:\n{}\nin:\n{}",
            expected,
            stderr
        );
    }
}

/// The build reports exactly the errors of the fixture: each offending path of an item is an error
/// at its own span (an item with five has five, in the order of the source), and the notes of each
/// are its own.
#[test]
fn every_offending_path_is_an_error_of_its_own() {
    let stderr = build_fixture();
    let got = reported(&stderr);

    let clone_help = "use `::core::clone::Clone` instead of `Clone`";
    let because = "it cannot be leaked as a whole because of `'_`";
    let self_note = "`Self` has no meaning here: it is declared only by a struct, an enum, a union, an impl or a trait, not by a module or generics";
    let self_help = "write the type instead of `Self`, or enter a scope that declares `Self` around it (`Leaker::enter_type`, `enter_impl` or `enter_trait`)";
    let expected_notes_of_multi = [
        error("use absolute path: 'Clone'", "14:18", &[], &[clone_help]),
        error(
            "use absolute path: 'Send'",
            "14:26",
            &[],
            &["use `::core::marker::Send` instead of `Send`"],
        ),
        error(
            "use absolute path: 'MyType < '_ >'",
            "15:20",
            &[because],
            &["write `MyType` with an absolute path (e.g. `::my_crate::MyType`), or allow it with `LeakerConfig::allowed_paths`"],
        ),
        error(
            "use absolute path: 'Other < '_, :: core :: primitive :: u8 >'",
            "15:35",
            &[because],
            &["write `Other` with an absolute path (e.g. `::my_crate::Other`), or allow it with `LeakerConfig::allowed_paths`"],
        ),
        error(
            "use absolute path: 'Sync'",
            "16:13",
            &[],
            &["use `::core::marker::Sync` instead of `Sync`"],
        ),
    ];
    // The errors of `Multi` come out in the order of the paths in the source, and are the only
    // ones at lines 14 to 16.
    let multi: Vec<&Reported> = got
        .iter()
        .filter(|r| matches!(r.at.split(':').next(), Some("14") | Some("15") | Some("16")))
        .collect();
    assert_eq!(multi.len(), expected_notes_of_multi.len(), "{}", stderr);
    for (got, expected) in multi.iter().zip(&expected_notes_of_multi) {
        assert_eq!(*got, expected, "{}", stderr);
    }

    // Every error of the fixture, and no other: the three single ones, the five of `Multi`, the two
    // mentions of `Self`, and the four unreadable Referrers.
    let mut expected = vec![
        error("use absolute path: 'Clone'", "2:20", &[], &[clone_help]),
        error(
            "use absolute path: 'for < 'a > Fn(& 'a u8)'",
            "5:18",
            &[],
            &["use `::core::ops::Fn` instead of `Fn`"],
        ),
        error(
            "use absolute path: 'Box < dyn RelA + RelB < '_ > >'",
            "9:20",
            &[
                because,
                "`RelA` also depends on the type context",
                "`RelB < '_ >` also depends on the type context",
            ],
            &["use `::std::boxed::Box` instead of `Box`"],
        ),
        error(
            "`Self` has no meaning here",
            "22:29",
            &[self_note],
            &[self_help],
        ),
        error(
            "`Self` has no meaning here",
            "22:38",
            &[self_note],
            &[self_help],
        ),
        format_error("format 7", "33:1"),
        format_error("format 9", "39:1"),
        format_error("no format tag", "45:1"),
        format_error("an unrecognized tag \"not-a-tag\"", "51:1"),
    ];
    // Not the "format 7" of the previous version, but the one that is, whatever the version is.
    expected[5] = format_error(&format!("format {}", Referrer::FORMAT_VERSION - 1), "33:1");
    expected[6] = format_error(&format!("format {}", Referrer::FORMAT_VERSION + 1), "39:1");
    expected.extend(expected_notes_of_multi);
    // The first lines of a nightly diagnostic differ, but not its headline, position and notes.
    let mut got = got;
    got.sort();
    expected.sort();
    assert_eq!(got, expected, "{}", stderr);

    // And the summary of cargo agrees (the current format, `TagCurrent`, is no error).
    assert!(
        stderr.contains(&format!("due to {} previous errors", expected.len())),
        "{}",
        stderr
    );
}

/// The diagnostics of the output of rustc for the first `lines` lines of the source, as text, in
/// order. What differs in the two fixtures is left out: the note that names the macro, and the
/// summary of cargo.
fn diagnostics_up_to(stderr: &str, lines: usize) -> Vec<String> {
    let mut found: Vec<String> = Vec::new();
    let mut current: Option<Vec<&str>> = None;
    for line in stderr.lines().map(str::trim_end) {
        if line.starts_with("error") || line.starts_with("warning") {
            found.extend(current.take().map(|block| block.join("\n")));
            if !line.starts_with("error: could not compile") {
                current = Some(Vec::new());
            }
        }
        if let Some(block) = &mut current {
            if !line.contains("this error originates") {
                block.push(line);
            }
        }
    }
    found.extend(current.map(|block| block.join("\n")));
    // The place of a diagnostic is the first `--> lib.rs:<line>:` in it.
    found
        .into_iter()
        .filter(|block| {
            block
                .lines()
                .find_map(|line| line.trim().strip_prefix("--> lib.rs:"))
                .and_then(|place| place.split(':').next()?.parse::<usize>().ok())
                .map_or(false, |line| line <= lines)
        })
        .collect()
}

/// The errors that a macro writes as tokens, reads back and reports are reported as those it
/// reports directly: the tokens of the cause, of the notes and of the help keep their spans, so
/// rustc points at the same places, with the same text.
#[test]
fn serialized_errors_are_reported_at_the_same_places() {
    let direct = diagnostics_up_to(&build_fixture(), 23);
    let built = nested::cargo("diagnostic_serialized", &["check"], &[]);
    assert!(!built.success, "{}", built.stderr);
    let serialized = diagnostics_up_to(&built.stderr, 23);
    // The three items with one error, the five of `Multi`, and the two mentions of `Self`.
    assert_eq!(direct.len(), 10, "{}", direct.join("\n"));
    assert_eq!(serialized, direct);
    // And no other error: the fixture is the first lines of the other.
    let errors = built
        .stderr
        .lines()
        .filter(|line| line.starts_with("error") && !line.starts_with("error: could not compile"))
        .count();
    assert_eq!(errors, direct.len(), "{}", built.stderr);
}

/// The first position of each error of the output of rustc: `path:line:column` of its first `-->`.
fn first_positions(stderr: &str) -> Vec<String> {
    let mut positions = Vec::new();
    let mut waiting = false;
    for line in stderr.lines() {
        if line.starts_with("error") {
            waiting = !line.starts_with("error: could not compile");
        } else if let Some(position) = line.trim().strip_prefix("--> ").filter(|_| waiting) {
            positions.push(position.to_string());
            waiting = false;
        }
    }
    positions
}

/// `line:column` of the first `needle` after the first `marker` in the source of the Leaker's crate.
fn position_in_leaker(marker: &str, needle: &str) -> String {
    let source = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/crates/leaker/lib.rs"
    ))
    .unwrap();
    let start = source.find(marker).unwrap();
    let at = start + source[start..].find(needle).unwrap();
    let line = source[..at].matches('\n').count() + 1;
    let column = source[..at].rsplit('\n').next().unwrap().chars().count() + 1;
    format!("{}:{}", line, column)
}

/// The errors of the item `Denied` (`id = 9193` in the Leaker's crate) that the invocation does not
/// allow are reported in the Referrer's crate, and the one it allows (`Clone`, with `allow(prelude)`)
/// is not: `core::fmt::Debug` is not the prelude, and `Held<'_>` is no path at all. They are at the
/// invocation, as rustc has it for the spans of a macro of another crate.
#[test]
fn errors_the_referrers_macro_does_not_allow_are_reported_in_its_crate() {
    let built = nested::cargo("diagnostic_result", &["check"], &[]);
    assert!(!built.success, "{}", built.stderr);
    let mut expected = vec![
        error(
            "use absolute path: 'Held < '_ >'",
            "11:1",
            &["it cannot be leaked as a whole because of `'_`"],
            &["write `Held` with an absolute path (e.g. `::my_crate::Held`), or allow it with `LeakerConfig::allowed_paths`"],
        ),
        error(
            "use absolute path: 'core :: fmt :: Debug'",
            "11:1",
            &[],
            &["write `core::fmt::Debug` with an absolute path (e.g. `::my_crate::core::fmt::Debug`), or allow it with `LeakerConfig::allowed_paths`"],
        ),
    ];
    let mut got = reported(&built.stderr);
    got.sort();
    expected.sort();
    assert_eq!(got, expected, "{}", built.stderr);
    assert!(
        built.stderr.contains("due to 2 previous errors"),
        "{}",
        built.stderr
    );
}

/// With `-Zmacro-backtrace` (nightly) the spans are not moved to the invocation: the errors are at
/// the causes in the Leaker's crate, which the serialization and the `macro_rules!` of the other
/// crate kept. Stable has no such flag; the test above is what it can check.
#[test]
fn the_spans_of_the_causes_survive_the_trip_between_the_crates() {
    if !nested::rustc_is_nightly() {
        return;
    }
    let built = nested::cargo_rustc("diagnostic_result", &[], &["-Zmacro-backtrace"]);
    assert!(!built.success, "{}", built.stderr);
    let leaker = "/tests/crates/leaker/lib.rs";
    let expected = [
        position_in_leaker("id = 9193", "Held<'_>"),
        position_in_leaker("id = 9193", "core::fmt::Debug"),
    ];
    let got: Vec<String> = first_positions(&built.stderr)
        .iter()
        .map(|position| {
            let (path, rest) = position.split_once(leaker).expect(position);
            assert!(path.starts_with('/'), "{}", position);
            rest.trim_start_matches(':').to_string()
        })
        .collect();
    assert_eq!(got, expected, "{}", built.stderr);
}

/// The copy is made when an error remains, with the recovered Referrer: the host of `Denied` and
/// its impl are in the expansion, with the two parts that conflict as written. (rustc stops after
/// the expansion that reported an error, so nothing but the expansion shows it; `-Zunpretty` is
/// nightly's.)
#[test]
fn the_copy_is_made_when_an_error_remains() {
    if !nested::rustc_is_nightly() {
        return;
    }
    let built = nested::cargo_rustc("diagnostic_result", &[], &["-Zunpretty=expanded"]);
    assert!(!built.success, "{}", built.stderr);
    let expanded = squash(&built.stdout);
    for part in [
        "pubstructHost9193;",
        "impl<T:Clone>::type_leak_test_leaker::kinds::Denied<T>forHost9193{",
        "fnheld(&self,x:Held<'_>)->",
        "fnshow<U:core::fmt::Debug>(&self,",
    ] {
        assert!(expanded.contains(part), "{} in:\n{}", part, built.stdout);
    }
}

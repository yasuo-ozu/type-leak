//! Checks how rustc renders a `NotInternableError` emitted by a proc macro with its `ToTokens`
//! implementation, by building `tests/crates/diagnostic` (which fails to compile).
//!
//! On nightly, the note is a note of the diagnostic with its own span and source preview; on
//! stable, it is appended to the message.
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
        }
        errors.push(Reported {
            headline: squash(line),
            at,
            notes,
        });
    }
    errors
}

fn error(headline: &str, at: &str, notes: &[&str]) -> Reported {
    Reported {
        headline: squash(&format!("error: {}", headline)),
        at: at.to_string(),
        notes: notes.iter().map(|note| note.to_string()).collect(),
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
    )
}

#[test]
fn note_is_rendered() {
    let stderr = build_fixture();
    let expected: &[&str] = if nested::rustc_is_nightly() {
        &[
            // The note has its own span: the first segment of the offending path.
            "\
error: use absolute path: 'Clone'
 --> lib.rs:2:20
  |
2 | pub trait Prelude: Clone {}
  |                    ^^^^^
  |
note: use `::core::clone::Clone` instead of `Clone`
 --> lib.rs:2:20
  |
2 | pub trait Prelude: Clone {}
  |                    ^^^^^
",
            // The error spans the whole bound, while the note spans `Fn`.
            "
 --> lib.rs:5:18
  |
5 | pub trait Bound: for<'a> Fn(&'a u8) {}
  |                  ^^^^^^^^^^^^^^^^^^
  |
note: use `::core::ops::Fn` instead of `Fn`
 --> lib.rs:5:26
  |
5 | pub trait Bound: for<'a> Fn(&'a u8) {}
  |                          ^^
",
            // Each note has its own span: why the type cannot be leaked as a whole, and the other
            // paths inside it.
            "\
note: use `::std::boxed::Box` instead of `Box`
 --> lib.rs:9:20
  |
9 |     fn f(&self, x: Box<dyn RelA + RelB<'_>>);
  |                    ^^^
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
",
        ]
    } else {
        &[
            "\
error: use absolute path: 'Clone'

         = note: use `::core::clone::Clone` instead of `Clone`

 --> lib.rs:2:20
  |
2 | pub trait Prelude: Clone {}
  |                    ^^^^^
",
            "
         = note: use `::core::ops::Fn` instead of `Fn`

 --> lib.rs:5:18
",
            "\
error: use absolute path: 'Box < dyn RelA + RelB < '_ > >'

         = note: use `::std::boxed::Box` instead of `Box`
         = note: it cannot be leaked as a whole because of `'_`
         = note: `RelA` also depends on the type context
         = note: `RelB < '_ >` also depends on the type context

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

    let clone_note = "use `::core::clone::Clone` instead of `Clone`";
    let because = "it cannot be leaked as a whole because of `'_`";
    let self_note = "`Self` has no meaning here: it is declared only by a struct, an enum, a union, an impl or a trait, not by a module or generics";
    let expected_notes_of_multi = [
        error("use absolute path: 'Clone'", "14:18", &[clone_note]),
        error(
            "use absolute path: 'Send'",
            "14:26",
            &["use `::core::marker::Send` instead of `Send`"],
        ),
        error("use absolute path: 'MyType < '_ >'", "15:20", &[because]),
        error(
            "use absolute path: 'Other < '_, :: core :: primitive :: u8 >'",
            "15:35",
            &[because],
        ),
        error(
            "use absolute path: 'Sync'",
            "16:13",
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
        error("use absolute path: 'Clone'", "2:20", &[clone_note]),
        error(
            "use absolute path: 'for < 'a > Fn(& 'a u8)'",
            "5:18",
            &["use `::core::ops::Fn` instead of `Fn`"],
        ),
        error(
            "use absolute path: 'Box < dyn RelA + RelB < '_ > >'",
            "9:20",
            &[
                "use `::std::boxed::Box` instead of `Box`",
                because,
                "`RelA` also depends on the type context",
                "`RelB < '_ >` also depends on the type context",
            ],
        ),
        error("`Self` has no meaning here", "22:29", &[self_note]),
        error("`Self` has no meaning here", "22:38", &[self_note]),
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

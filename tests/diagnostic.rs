//! The tests that build a crate outside this package with a nested `cargo` and check what comes
//! out: the crates under `tests/crates` and the example `examples/two-macros`, which the published
//! package has not (see `exclude` of `Cargo.toml`). The fixtures are built on their own
//! (`tests/common/nested.rs`), so that a failure of one does not break the other tests. They are
//! in this one binary, in a module each:
//!
//! - the top level: how rustc renders the errors of the Leaker (below);
//! - `spans`: the spans of the errors, by the tests of `tests/crates/spans`;
//! - `trusting`: a Referrer in a crate without the dependency of an optional entry;
//! - `outlives`: the outlives bounds in the Repeater's implementations and the copies;
//! - `two_macros`: the example of the README, with a macro in each of two crates.
//!
//! The top level checks how rustc renders the errors of the Leaker (`ErrorItem`) reported by a proc
//! macro with `ErrorItem::emit()`, by building `tests/crates/diagnostic` (which fails to compile).
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
//! `tests/crates/diagnostic_pure` is a macro that builds the tokens of `to_compile_error()` and drops
//! them: it reports nothing, on any toolchain.
//!
//! `tests/crates/diagnostic_result` is a Referrer's crate whose macro got the errors of the Leaker's
//! macro in another crate (the serialized `LeakerResult`), allowed some, and reports the others. By
//! default rustc points an error at the invocation, as it does for every span inside a macro of
//! another crate; on nightly with `-Zmacro-backtrace` it points at the cause, in the Leaker's crate.

use std::sync::OnceLock;
use type_leak::Referrer;

#[path = "common/nested.rs"]
mod nested;

/// Builds the fixture crate (once: a build that fails is made again by every `cargo check`),
/// returning the rendered diagnostics. The tests are not built with the MSRV (the job of the MSRV
/// builds the library only).
#[allow(clippy::incompatible_msrv)]
fn build_fixture() -> &'static str {
    static STDERR: OnceLock<String> = OnceLock::new();
    STDERR.get_or_init(|| {
        let built = nested::cargo("diagnostic", &["check"], &[]);
        assert!(!built.success, "{}", built.stderr);
        built.stderr
    })
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
            // paths of the same type (`RelB<'_>`, beside `RelA`, in the error of the trait object).
            // The help comes last.
            "\
note: it cannot be leaked as a whole because of `'_`
 --> lib.rs:9:40
  |
9 |     fn f(&self, x: Box<dyn RelA + RelB<'_>>);
  |                                        ^^
help: use `::std::boxed::Box` instead of `Box`
 --> lib.rs:9:20
  |
9 |     fn f(&self, x: Box<dyn RelA + RelB<'_>>);
  |                    ^^^
",
            "\
note: `RelB<'_>` also depends on the type context
 --> lib.rs:9:35
  |
9 |     fn f(&self, x: Box<dyn RelA + RelB<'_>>);
  |                                   ^^^^^^^^
help: write `RelA` with an absolute path (e.g. `::my_crate::RelA`)
 --> lib.rs:9:28
  |
9 |     fn f(&self, x: Box<dyn RelA + RelB<'_>>);
  |                            ^^^^
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
error: use absolute path: 'Box<dyn RelA + RelB<'_>>'

         = note: it cannot be leaked as a whole because of `'_`
         = help: use `::std::boxed::Box` instead of `Box`

 --> lib.rs:9:20
",
            "\
error: use absolute path: 'RelA'

         = note: it cannot be leaked as a whole because of `'_`
         = note: `RelB<'_>` also depends on the type context
         = help: write `RelA` with an absolute path (e.g. `::my_crate::RelA`)

 --> lib.rs:9:28
",
        ]
    };
    // The lines that rustc pads with spaces (the empty line between a message and its notes) are
    // compared without them.
    let stderr = trimmed(stderr);
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
    let got = reported(stderr);

    let clone_help = "use `::core::clone::Clone` instead of `Clone`";
    let because = "it cannot be leaked as a whole because of `'_`";
    let self_note = "`Self` has no meaning here: it is declared only by a struct, an enum, a union, an impl or a trait, not by a module or generics";
    let self_help = "write the type instead of `Self`";
    let expected_notes_of_multi = [
        error("use absolute path: 'Clone'", "14:18", &[], &[clone_help]),
        error(
            "use absolute path: 'Send'",
            "14:26",
            &[],
            &["use `::core::marker::Send` instead of `Send`"],
        ),
        error(
            "use absolute path: 'MyType<'_>'",
            "15:20",
            &[because],
            &["write `MyType` with an absolute path (e.g. `::my_crate::MyType`)"],
        ),
        error(
            "use absolute path: 'Other<'_, ::core::primitive::u8>'",
            "15:35",
            &[because],
            &["write `Other` with an absolute path (e.g. `::my_crate::Other`)"],
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

    // Every error of the fixture, and no other: the three single ones and the trait object in the
    // second, the five of `Multi`, the two mentions of `Self`, and the four unreadable Referrers.
    let mut expected = vec![
        error("use absolute path: 'Clone'", "2:20", &[], &[clone_help]),
        error(
            "use absolute path: 'for<'a> Fn(&'a u8)'",
            "5:18",
            &[],
            &["use `::core::ops::Fn` instead of `Fn`"],
        ),
        error(
            "use absolute path: 'Box<dyn RelA + RelB<'_>>'",
            "9:20",
            &[because],
            &["use `::std::boxed::Box` instead of `Box`"],
        ),
        error(
            "use absolute path: 'RelA'",
            "9:28",
            &[because, "`RelB<'_>` also depends on the type context"],
            &["write `RelA` with an absolute path (e.g. `::my_crate::RelA`)"],
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
    expected[6] = format_error(&format!("format {}", Referrer::FORMAT_VERSION - 1), "33:1");
    expected[7] = format_error(&format!("format {}", Referrer::FORMAT_VERSION + 1), "39:1");
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
    let direct = diagnostics_up_to(build_fixture(), 23);
    let built = nested::cargo("diagnostic_serialized", &["check"], &[]);
    assert!(!built.success, "{}", built.stderr);
    let serialized = diagnostics_up_to(&built.stderr, 23);
    // The three items with one error and the trait object in the second, the five of `Multi`, and
    // the two mentions of `Self`.
    assert_eq!(direct.len(), 11, "{}", direct.join("\n"));
    assert_eq!(serialized, direct);
    // And no other error: the fixture is the first lines of the other.
    let errors = built
        .stderr
        .lines()
        .filter(|line| line.starts_with("error") && !line.starts_with("error: could not compile"))
        .count();
    assert_eq!(errors, direct.len(), "{}", built.stderr);
}

/// `ErrorItem::to_compile_error()` only builds tokens: a macro that drops them reports nothing,
/// also on nightly, where `emit()` reports a `Diagnostic` as a side effect.
#[test]
fn to_compile_error_alone_reports_nothing() {
    let built = nested::cargo("diagnostic_pure", &["check"], &[]);
    assert!(built.success, "{}", built.stderr);
    assert_eq!(built.stderr.trim(), "", "{}", built.stderr);
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

/// The errors of the item `Denied` of the Leaker's crate that the invocation does not allow are
/// reported in the Referrer's crate, and the one it allows (`Clone`, with `allow(trait Clone)`) is
/// not: `core::fmt::Debug` and `Held<'_>` are. They are at the invocation, as rustc has it for the
/// spans of a macro of another crate.
#[test]
fn errors_the_referrers_macro_does_not_allow_are_reported_in_its_crate() {
    let built = nested::cargo("diagnostic_result", &["check"], &[]);
    assert!(!built.success, "{}", built.stderr);
    let mut expected = vec![
        error(
            "use absolute path: 'Held<'_>'",
            "11:1",
            &["it cannot be leaked as a whole because of `'_`"],
            &["write `Held` with an absolute path (e.g. `::my_crate::Held`)"],
        ),
        error(
            "use absolute path: 'core::fmt::Debug'",
            "11:1",
            &[],
            &["write `core::fmt::Debug` with an absolute path (e.g. `::core::fmt::Debug`)"],
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

mod spans {
    //! Runs the tests of the spans of the errors, `tests/crates/spans`.
    //!
    //! The spans of the errors are where a macro reports them (`ErrorItem::span()`, the tokens of its
    //! notes and help, `EncodeError::span()`), and only a test that knows the line and the column of
    //! a span can tell a wrong one. Code parsed from a string has them if `proc-macro2` is built with
    //! the feature `span-locations`. A feature is shared by the whole build of a package, and this one
    //! makes the spans larger, which is enough for the deep recursion of other tests
    //! (`tests/binder_output.rs`) to overflow the stack of a test thread. So the tests are a package of
    //! their own, built on its own as the other fixtures are.

    use std::process::Command;

    #[test]
    fn the_spans_of_the_errors_are_at_the_offending_tokens() {
        let manifest = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/crates/spans/Cargo.toml");
        // The published package has no `tests/crates` (see `exclude` of `Cargo.toml`).
        if !std::path::Path::new(manifest).exists() {
            return;
        }
        let output = Command::new(env!("CARGO"))
            .args(["test", "--quiet", "--manifest-path"])
            .arg(manifest)
            .env(
                "CARGO_TARGET_DIR",
                concat!(env!("CARGO_TARGET_TMPDIR"), "/spans"),
            )
            .env("CARGO_TERM_COLOR", "never")
            .output()
            .unwrap();
        let stdout = String::from_utf8(output.stdout).unwrap();
        let stderr = String::from_utf8(output.stderr).unwrap();
        assert!(output.status.success(), "{}\n{}", stdout, stderr);
        // The tests ran (the other targets of the package have none): 12 of the errors of the Leaker,
        // and 2 of `EncodeError`.
        assert!(stdout.contains("test result: ok. 12 passed"), "{}", stdout);
        assert!(stdout.contains("test result: ok. 2 passed"), "{}", stdout);
    }
}

mod trusting {
    //! A Leaker that allows the crate of a type, and one that does not, in a crate that has no
    //! dependency on that crate (`tests/crates/trusting`).
    //!
    //! `type_leak_test_leaker::x::ExtDep` names `::type_leak_test_ext::Ext<..>`. It is an entry
    //! unless the Leaker allows the crate: `ExtDep` is analyzed by one that does not, so the macro
    //! replaces the whole type with the Repeater's projection, and the crate does not need it;
    //! `ExtDepAllowed` is analyzed by one that does, so the copy names a crate that the Referrer's
    //! crate does not have. `tests/cross_crate.rs` has the text of both copies. This is the
    //! second, which rustc rejects, and the same crate with the feature `replace`, which it accepts
    //! and whose own tests (the copies of a struct and of a fragment of an impl, and an impl that
    //! forwards) pass: the outcome differs by that flag alone.

    use super::nested;

    #[test]
    fn a_crate_that_the_leaker_allows_needs_the_dependency() {
        // The methods that name the crate fail alike (rustc prints the same error once).
        let allowed = nested::cargo("trusting", &["check"], &[]);
        assert!(!allowed.success, "{}", allowed.stderr);
        assert_eq!(
            nested::error_codes(&allowed.stderr),
            ["E0433"],
            "{}",
            allowed.stderr
        );
        assert!(
            allowed
                .stderr
                .contains("cannot find `type_leak_test_ext` in"),
            "{}",
            allowed.stderr
        );
    }

    #[test]
    fn a_crate_that_it_does_not_allow_does_not() {
        // Nothing else is wrong with the crate, so the flag alone makes the difference.
        let replaced = nested::cargo("trusting", &["check"], &["replace"]);
        assert!(replaced.success, "{}", replaced.stderr);
        assert_eq!(replaced.stderr, "");
        // And what it wrote works.
        let tests = nested::cargo("trusting", &["test", "--lib"], &["replace"]);
        assert!(tests.success, "{}{}", tests.stdout, tests.stderr);
        assert!(
            tests.stdout.contains("test result: ok. 3 passed; 0 failed"),
            "{}{}",
            tests.stdout,
            tests.stderr
        );
    }
}

mod outlives {
    //! Builds `tests/crates/outlives`, a crate with the Leaker's side and the Referrer's side of types
    //! whose generics carry an outlives bound, or are lifetimes of `for<..>` binders, and checks that
    //! it compiles.
    //!
    //! Only the bounds which are exact are encoded: those which are written (`T: 'a`, `where Self:
    //! 'a`) and those which a reference implies (`&'a X` requires `X: 'a`, whatever `X` is, and what
    //! a reference to a type must satisfy is passed on to the types inside it). A path with a lifetime
    //! argument is not assumed to require anything: `Rel<'a, T>`, where `struct Rel<'a, T>(&'a T)` gets
    //! `T: 'a` inferred (RFC 2093), is interned as a whole, with the generics `<'a, T>` it uses, but
    //! `struct Unrelated<'a, T>(&'a (), T)` does not tie `T` to `'a`, and the definition cannot be
    //! seen. An assumed bound would make the copy stricter than the original (a caller whose `T` is
    //! shorter than `'a` would be rejected). So where the definition needs `T: 'a`, the signature
    //! writes it, as `fn f<'a, T: 'a>(&self, x: Rel<'a, T>)`: the Repeater's implementation which the
    //! downstream macro emits is
    //!
    //! ```ignore
    //! impl<'a, T> Repeater<(&'a (), T, &'a T), KEY> for S { type Type = Rel<'a, T>; }
    //! ```
    //!
    //! whose header encodes `'a` as `&'a ()`, `T` as `T` and the written bound as `&'a T`, which
    //! implies `T: 'a` wherever it is written (the copy of the signature
    //! `<S as Repeater<(&'a (), T, &'a T), ..>>::Type` has the same encoding). Without the bound
    //! written (`fn f<'a, T>(&self, x: Rel<'a, T>)`) the header implies nothing about `T: 'a`, and
    //! rustc rejects the implementation with `error[E0309]: the parameter type `T` may not live long
    //! enough`. The fixture has this failing case disabled (the feature `negative-unbounded`, `m::
    //! Unbounded`), and the same for `Self` (`negative-unbounded-self`, `m::UnboundedSelf`, E0309 for
    //! `__S`) and for the field of a struct (`negative-unbounded-around`, `m::UnboundedAround`, see
    //! below). Each is built here with its feature, and must fail with exactly the error written
    //! above it (E0309 on stable and nightly); so must the alternative that is not taken for a trait
    //! object (`negative-alone`, E0308 or E0478, and E0521).
    //!
    //! The fixture also has the cases around it: the control `Loose<'a, T>` (which does not require
    //! the bound, so nothing is written and the copy is the same as the original), references and
    //! paths with a lifetime around a relative type (`&'a Rel2<T>`, `Ref<'a, Rel2<T>>`), and trait
    //! objects as arguments of a path with a lifetime (`Ref<'a, dyn Iterator<Item = Plain>>`, of which
    //! only `Plain` is leaked, and `Ref<'a, dyn RelTr>`, which is leaked as a whole) and the
    //! references to them (`&'a dyn Iterator<Item = Plain>`, likewise), with the implementations of
    //! the Referrer's side in the same crate.
    //!
    //! And the lifetimes of `for<..>` binders: `Hr<'x>` is leaked with `'x` if the output of the binder
    //! does not mention it, or if an input keeps it where it is written (`&'x u8` of `Fn(&'x u8,
    //! Hr<'x>) -> &'x u8`), and the function pointer or the trait object holding the binder is leaked
    //! as a whole otherwise, as the copy `for<'x> fn(<..>::Type) -> &'x u8` is rejected (E0581, or
    //! E0582 for a bound or a trait object).
    //!
    //! An anchor in a path of `::core` (`Option<&'x u8>`) stays where it is written, so `Hr<'x>` is leaked
    //! alone as before, but one in the type of an external crate (`::std::boxed::Box<dyn
    //! Iterator<Item = &'x u8>>`) is not: the Referrer replaces that type as a whole (it is an
    //! entry), so the function pointer or the trait object holding the binder is leaked as a whole
    //! (a bound is an error), and the copies compile (`Binders`: `core_*`, `external_*`, and the
    //! `check_leak!` next to `AnchoredInside`, where `::std` is allowed, so the types of it are no
    //! entries and anchor as before).
    //!
    //! And `Self: 'a`, which a reference implies (`&'a Rel2<Self>`, `&'a Rel2<Self::Item>`, `SelfOutlives`)
    //! and a path with a lifetime does not, so `Rel<'a, Self>` and `Rel<'a, Self::Item>` write it
    //! (`where Self: 'a`, `SelfBound`): the trait arguments of the Repeater encode it as `&'a __S`,
    //! which implies it in the implementation (E0309 otherwise, "the parameter type `__S` may not live
    //! long enough"), in the copy of a method, a free function and the field of a struct (the
    //! implementation must not declare `__S: 'a`, which nothing proves there).
    //!
    //! And the lifetimes of the references around a type leaked alone (`Around`, `SelfAround`):
    //! `Rel2<T>` of `&'a (u8, [Rel2<T>; 2])` is leaked with `<'a, T: 'a>`, so that the copy of the
    //! field of a struct `<S as Repeater<(&'a (), T, &'a T), ..>>::Type` has the `T: 'a` which the
    //! original has inferred (RFC 2093), whereas the one without is rejected with E0309: around a
    //! reference, a tuple, an array, a path, a function pointer and a trait object, in the field of a
    //! struct and in the variant of an enum, and for `Self`. Not across a projection (`<X as Tr>::A<..>`,
    //! and a path starting with a type parameter, `T::A<..>`, which the copy of the field of a struct
    //! must not require `U: 'a` for either, as the original is well-formed without it), nor for
    //! `'static` (which the struct declares as the original does, RFC 2093 infers no such bound) or
    //! the lifetime of a `for<..>` binder, whose copies compile as they are. A path with a lifetime
    //! (`Ref<'a, Rel2<T>>`) passes on nothing, so `Rel2<T>` of it is leaked with the generics which
    //! are written (`<'a, T: 'a>` if the signature and the struct write `T: 'a`, and `<T>` if not,
    //! where the copy of the field of the struct is rejected with E0309, `m::UnboundedAround`).
    //!
    //! And a trait object borrowed through a lifetime of the method (`Locals`, `LocalBinders`): `&'b
    //! dyn RelTr` is leaked as a whole, with `'b` among its generics, so rustc gives the trait object
    //! its default where the Repeater is implemented, and the copy is `<Self as Repeater<(&'b (),),
    //! ..>>::Type`, which has `'b` only in a projection: the return type may mention `'b` only if
    //! another argument keeps it as written (`&'b self`, or `y: &'b dyn ::core::fmt::Debug`, a
    //! reference which is not leaked as a whole), a bound makes it early-bound in both, or no argument
    //! mentions it (E0195 where the method is implemented otherwise, which is an error of the Leaker), with the
    //! forwarding implementations called through `&dyn Locals`. A path `Ref<'b, dyn RelTr>` is the
    //! same. And a trait with a lifetime bound of its own (`StaticTr: 'static`, `Statics`), whose
    //! default is `'static` whatever the reference is, which leaking the trait object alone with the
    //! lifetime of the reference written out would get wrong (E0478).
    //!
    //! And a `Self: 'a` which the item writes (`SelfBound`, `SelfBoundTrait`): it is encoded like the
    //! implied one and never declared, so a free function and the field of a struct, where nothing
    //! proves it but the header, compile as well.
    //!
    //! The fixture is built on its own, so that a failure of it does not break the other cross-crate
    //! tests (the Leaker's crate in `tests/crates/leaker` is a dependency of all of them). The tests
    //! check what rustc printed, not only whether it succeeded: the warnings of the good build, and
    //! the errors of each negative control.

    use super::nested;

    /// The good build compiles, with no warning but the ones the fixture is known to have: a
    /// parenthesized trait object (`&'a (dyn Iterator<Item = Plain>)`) in the copy of `Dyns`, which the
    /// copy keeps as it is written. Any other lint (an unused lifetime or import, a dead item) in a
    /// copy or in a Repeater's implementation would be new.
    #[test]
    fn implied_outlives_bounds_are_kept() {
        let built = nested::cargo("outlives", &["build"], &[]);
        assert!(built.success, "{}", built.stderr);
        assert!(
            nested::headlines(&built.stderr, "error").is_empty(),
            "{}",
            built.stderr
        );
        for warning in nested::headlines(&built.stderr, "warning") {
            assert_eq!(
                warning, "warning: unnecessary parentheses around type",
                "{}",
                built.stderr
            );
        }
    }

    /// Builds the fixture with a negative control, which must fail to compile with exactly these
    /// errors (their codes, and the headline of the first).
    fn negative(feature: &str, codes: &[&str], headline: &str) {
        let built = nested::cargo("outlives", &["check"], &[feature]);
        assert!(
            !built.success,
            "`{}` compiles, but is a negative control:\n{}",
            feature, built.stderr
        );
        let mut got = nested::error_codes(&built.stderr);
        // The error of a trait object whose lifetime was written where the path makes it `'static` is
        // E0308 (`method not compatible with trait`) on stable, and E0478 (`lifetime bound not
        // satisfied`) on nightly.
        for code in &mut got {
            if code == "E0478" {
                *code = "E0308".to_string();
            }
        }
        got.sort();
        assert_eq!(got, codes, "`{}`:\n{}", feature, built.stderr);
        assert!(
            built.stderr.contains(headline),
            "`{}` has no `{}`:\n{}",
            feature,
            headline,
            built.stderr
        );
    }

    /// The negative controls: each feature of the fixture enables an item that does not compile, which
    /// must fail with exactly these errors (the codes, and the headline of the first), because
    ///
    /// - `Rel<'a, T>` without `T: 'a` written: the Repeater's implementation, whose header encodes `T`
    ///   but implies nothing about `'a`, is rejected (`m::Unbounded`);
    /// - the same for `Self` (`m::UnboundedSelf`): `Rel<'a, Self>` needs `Self: 'a`, and the header
    ///   encodes `__S` only;
    /// - the field of a struct (`m::UnboundedAround`): `Rel2<T>` is leaked with the generics `<T>` its
    ///   signature writes, so the copy `Ref<'a, <Wrapper as Repeater<(T,), ..>>::Type>` is not
    ///   well-formed without `T: 'a`, which nothing infers through the projection;
    /// - the alternative which is not taken for a trait object without a lifetime bound as the argument
    ///   of a path with a lifetime: leaking it alone with `+ 'a` written, and keeping the path around
    ///   the projection in the copy. The object lifetime of `Inferred` and `Unrelated` is `'static`, so
    ///   the copy returns another type (E0308 on stable, E0478 on nightly, for each of the two), and
    ///   the forwarding body hands out a borrow that does not live that long (E0521).
    #[test]
    fn the_negative_controls_fail_with_their_errors() {
        for (feature, codes, headline) in [
            (
                "negative-unbounded",
                &["E0309"][..],
                "error[E0309]: the parameter type `T` may not live long enough",
            ),
            (
                "negative-unbounded-self",
                &["E0309"][..],
                "error[E0309]: the parameter type `__S` may not live long enough",
            ),
            (
                "negative-unbounded-around",
                &["E0309"][..],
                "error[E0309]: the parameter type `T` may not live long enough",
            ),
            (
                "negative-alone",
                &["E0308", "E0308", "E0521", "E0521"][..],
                "borrowed data escapes outside of method",
            ),
        ] {
            negative(feature, codes, headline);
        }
    }
}

mod two_macros {
    //! Builds and tests the two-macro example, `examples/two-macros`: a trait that a crate exports with
    //! one proc macro and another crate implements with a second one, with `type-leak` between them as
    //! the README describes. Its crates are not part of this package, so they are built on their own,
    //! as the fixtures of `tests/crates` are.

    use std::path::Path;
    use std::process::Command;

    /// Runs `cargo <args> --quiet` on the workspace of the example, if this package has it (the
    /// published one does not: see `exclude` of `Cargo.toml`). Returns `(success, stdout, stderr)`.
    fn cargo(args: &[&str]) -> Option<(bool, String, String)> {
        let manifest = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/examples/two-macros/Cargo.toml"
        );
        if !Path::new(manifest).exists() {
            return None;
        }
        let output = Command::new(env!("CARGO"))
            .args(args)
            .args(["--quiet", "--manifest-path", manifest])
            .env(
                "CARGO_TARGET_DIR",
                concat!(env!("CARGO_TARGET_TMPDIR"), "/two-macros"),
            )
            .env("CARGO_TERM_COLOR", "never")
            .output()
            .unwrap();
        Some((
            output.status.success(),
            String::from_utf8(output.stdout).unwrap(),
            String::from_utf8(output.stderr).unwrap(),
        ))
    }

    /// The wrappers of `app` forward, with the types of `repo` that `app` never names.
    #[test]
    fn the_example_builds_and_its_wrappers_forward() {
        let Some((success, stdout, stderr)) = cargo(&["test", "--workspace"]) else {
            return;
        };
        assert!(success, "{}\n{}", stdout, stderr);
        assert!(stdout.contains("test result: ok. 2 passed"), "{}", stdout);
    }

    /// `#[export_trait]` reports each path that cannot be leaked at its own span, with what to write.
    #[test]
    fn the_example_reports_what_cannot_be_exported() {
        let Some((success, _, stderr)) = cargo(&["check", "-p", "repo", "--features", "bad"])
        else {
            return;
        };
        assert!(!success, "{}", stderr);
        assert!(
            stderr.contains("error: use absolute path: 'Clone'"),
            "{}",
            stderr
        );
        assert!(
            stderr.contains("error: use absolute path: 'Record<'_>'"),
            "{}",
            stderr
        );
        assert!(stderr.contains("repo/lib.rs:50:16"), "{}", stderr);
        // The help says what to write.
        assert!(
            stderr.contains("help: use `::core::clone::Clone` instead of `Clone`"),
            "{}",
            stderr
        );
        assert!(
            stderr.contains("help: write `Record` with an absolute path"),
            "{}",
            stderr
        );
    }
}

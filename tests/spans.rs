//! Runs the tests of the spans of the errors, `tests/crates/spans`.
//!
//! The spans of the errors are where a macro reports them (`ErrorItem::span()`, the tokens of its
//! notes and help, `EncodeError::span()`), and only a test that knows the line and the column of
//! a span can tell a wrong one. Code parsed from a string has them if `proc-macro2` is built with
//! the feature `span-locations`. A feature is shared by the whole build of a package, and this one
//! makes the spans larger, which is enough for the deep recursion of other tests
//! (`tests/binder_output.rs`) to overflow the stack of a test thread. So the tests are a package
//! of their own, built on its own as `tests/diagnostic.rs` builds its fixture.

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
    // The tests ran (the other targets of the package have none): 8 of the errors of the Leaker,
    // and 2 of `EncodeError`.
    assert!(stdout.contains("test result: ok. 8 passed"), "{}", stdout);
    assert!(stdout.contains("test result: ok. 2 passed"), "{}", stdout);
}

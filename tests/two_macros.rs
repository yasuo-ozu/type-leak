//! Builds and tests the two-macro example, `examples/two-macros`: a trait that a crate exports with
//! one proc macro and another crate implements with a second one, with `type-leak` between them as
//! the README describes. Its crates are not part of this package, so they are built on their own,
//! as the fixtures of `tests/crates` are (`tests/trusting.rs`).

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

/// `#[export_trait]` reports each path that cannot be leaked at its own span.
#[test]
fn the_example_reports_what_cannot_be_exported() {
    let Some((success, _, stderr)) = cargo(&["check", "-p", "repo", "--features", "bad"]) else {
        return;
    };
    assert!(!success, "{}", stderr);
    assert!(
        stderr.contains("error: use absolute path: 'Clone'"),
        "{}",
        stderr
    );
    assert!(
        stderr.contains("error: use absolute path: 'Record < '_ >'"),
        "{}",
        stderr
    );
    assert!(stderr.contains("repo/lib.rs:50:16"), "{}", stderr);
}

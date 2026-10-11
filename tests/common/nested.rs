//! Builds a fixture crate of `tests/crates` with a nested `cargo`, for the tests which check what
//! rustc says about it (tests/outlives.rs, tests/diagnostic.rs, tests/trusting.rs).
//!
//! The fixtures are built on their own, so that a failure of one does not break the other
//! tests (the crates they depend on are shared by all of them). They share one target directory,
//! so that `syn` and `type-leak` are built once for all of them.

#![allow(dead_code)]

use std::process::Command;

/// What a build printed.
pub struct Built {
    pub success: bool,
    pub stdout: String,
    pub stderr: String,
}

/// Whether the compiler that builds the fixtures (`rustc` of the toolchain that runs the tests) is a
/// nightly: `proc_macro::Diagnostic` is available then.
pub fn rustc_is_nightly() -> bool {
    let output = Command::new("rustc").arg("-V").output().unwrap();
    let version = String::from_utf8(output.stdout).unwrap();
    version.contains("nightly") || version.contains("-dev")
}

/// Runs `cargo <args> --quiet` on the fixture `tests/crates/<fixture>` with these features.
/// `check` is enough to know whether a crate compiles (its dependencies are built in full).
pub fn cargo(fixture: &str, args: &[&str], features: &[&str]) -> Built {
    let mut command = Command::new(env!("CARGO"));
    command
        .args(args)
        .args(["--quiet", "--manifest-path"])
        .arg(format!(
            "{}/tests/crates/{}/Cargo.toml",
            env!("CARGO_MANIFEST_DIR"),
            fixture
        ))
        .env(
            "CARGO_TARGET_DIR",
            concat!(env!("CARGO_TARGET_TMPDIR"), "/nested"),
        )
        .env("CARGO_TERM_COLOR", "never");
    if !features.is_empty() {
        command.arg("--features").arg(features.join(","));
    }
    let output = command.output().unwrap();
    Built {
        success: output.status.success(),
        stdout: String::from_utf8(output.stdout).unwrap(),
        stderr: String::from_utf8(output.stderr).unwrap(),
    }
}

/// The codes of the errors (`error[E0309]`), sorted.
pub fn error_codes(stderr: &str) -> Vec<String> {
    let mut codes: Vec<String> = stderr
        .lines()
        .filter_map(|line| line.strip_prefix("error[")?.split(']').next())
        .map(str::to_string)
        .collect();
    codes.sort();
    codes
}

/// The headlines of the errors and warnings (`error[E0309]: the parameter type ..`,
/// `warning: ..`), without the final summary of cargo (`error: could not compile ..`) and the
/// count of rustc (`warning: 2 warnings emitted`).
pub fn headlines(stderr: &str, level: &str) -> Vec<String> {
    stderr
        .lines()
        .filter(|line| line.starts_with(level))
        .filter(|line| !line[level.len()..].starts_with(": could not compile"))
        .filter(|line| !line[level.len()..].starts_with(": aborting"))
        .filter(|line| !line.contains(" generated ") && !line.contains(" emitted"))
        .map(str::to_string)
        .collect()
}

//! Checks how rustc renders a `NotInternableError` emitted by a proc macro with its `ToTokens`
//! implementation, by building `tests/crates/diagnostic` (which fails to compile).
//!
//! On nightly, the note is a note of the diagnostic with its own span and source preview; on
//! stable, it is appended to the message.

use std::process::Command;

fn rustc_is_nightly() -> bool {
    let output = Command::new("rustc").arg("-V").output().unwrap();
    let version = String::from_utf8(output.stdout).unwrap();
    version.contains("nightly") || version.contains("-dev")
}

/// Builds the fixture crate, returning the rendered diagnostics.
fn build_fixture() -> String {
    let output = Command::new(env!("CARGO"))
        .args(["build", "--quiet", "--manifest-path"])
        .arg(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/crates/diagnostic/Cargo.toml"
        ))
        .env(
            "CARGO_TARGET_DIR",
            concat!(env!("CARGO_TARGET_TMPDIR"), "/diagnostic"),
        )
        .env("CARGO_TERM_COLOR", "never")
        .output()
        .unwrap();
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(!output.status.success(), "{}", stderr);
    stderr
}

#[test]
fn note_is_rendered() {
    let stderr = build_fixture();
    let expected: &[&str] = if rustc_is_nightly() {
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
    for expected in expected {
        assert!(
            stderr.contains(expected),
            "expected:\n{}\nin:\n{}",
            expected,
            stderr
        );
    }
}

//! Builds `tests/crates/outlives`, a crate with the Leaker's side and the Referrer's side of types
//! whose generics carry an outlives bound that is only implied, and checks that it compiles.
//!
//! `Rel<'a, T>`, where `struct Rel<'a, T>(&'a T)` gets `T: 'a` inferred (RFC 2093), is interned as
//! a whole, with the generics `<'a, T>` it uses. In the original signature `fn f<'a, T>(&self, x:
//! Rel<'a, T>)`, `T: 'a` is an implied bound, but the Repeater's implementation which the
//! downstream macro emits would be
//!
//! ```ignore
//! impl<'a, T> Repeater<(&'a (), T), ID, NTH> for S { type Type = Rel<'a, T>; }
//! ```
//!
//! whose header encodes `'a` as `&'a ()` and `T` as `T`, which implies nothing about `T: 'a`. So
//! the bound from the definition site would be missing and rustc rejects the implementation with
//! `error[E0309]: the parameter type `T` may not live long enough`, and so the copy of the
//! signature `<S as Repeater<(&'a (), T), ..>>::Type`. The Leaker adds the bound to the generics
//! of the type, and the encoders encode it as `&'a T`, which implies it wherever it is written.
//!
//! The fixture also has the cases around it: the control `Loose<'a, T>` (which does not require
//! the bound, but is assumed to), references and paths with a lifetime around a relative type
//! (`&'a Rel2<T>`, `Ref<'a, Rel2<T>>`), and trait objects as arguments of a path with a lifetime
//! (`Ref<'a, dyn Iterator<Item = Plain>>`, of which only `Plain` is leaked, and `Ref<'a, dyn
//! RelTr>`, which is leaked as a whole) and the references to them (`&'a dyn Iterator<Item =
//! Plain>`, likewise), with the implementations of the Referrer's side in the same crate.
//!
//! The fixture is built on its own, so that a failure of it does not break the other cross-crate
//! tests (the Leaker's crate in `tests/crates/leaker` is a dependency of all of them).

use std::process::Command;

#[test]
fn implied_outlives_bounds_are_kept() {
    let output = Command::new(env!("CARGO"))
        .args(["build", "--quiet", "--manifest-path"])
        .arg(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/crates/outlives/Cargo.toml"
        ))
        .env(
            "CARGO_TARGET_DIR",
            concat!(env!("CARGO_TARGET_TMPDIR"), "/outlives"),
        )
        .env("CARGO_TERM_COLOR", "never")
        .output()
        .unwrap();
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(output.status.success(), "{}", stderr);
}

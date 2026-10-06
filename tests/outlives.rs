//! Builds `tests/crates/outlives`, a crate with the Leaker's side and the Referrer's side of types
//! whose generics carry an outlives bound that is only implied, or are lifetimes of `for<..>`
//! binders, and checks that it compiles.
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
//! And the lifetimes of `for<..>` binders: `Hr<'x>` is leaked with `'x` if the output of the binder
//! does not mention it, or if an input keeps it where it is written (`&'x u8` of `Fn(&'x u8,
//! Hr<'x>) -> &'x u8`), and the function pointer or the trait object holding the binder is leaked
//! as a whole otherwise, as the copy `for<'x> fn(<..>::Type) -> &'x u8` is rejected (E0581, or
//! E0582 for a bound or a trait object).
//!
//! And `Self: 'a`, which `Rel<'a, Self>` and `Rel<'a, Self::Item>` imply like `Rel<'a, T>` does
//! `T: 'a`: the trait arguments of the Repeater encode it as `&'a __S`, which implies it in the
//! implementation (E0309 otherwise, "the parameter type `__S` may not live long enough"), in the
//! copy of a method, a free function and the field of a struct (the implementation must not
//! declare `__S: 'a`, which nothing proves there).
//!
//! And the lifetimes of the types around a type leaked alone (`Around`, `SelfAround`): `Rel2<T>` of
//! `Ref<'a, Rel2<T>>` is leaked with `<'a, T: 'a>`, so that the copy of the field of a struct
//! `Ref<'a, <S as Repeater<(&'a (), T, &'a T), ..>>::Type>` has the `T: 'a` which the original has
//! inferred (RFC 2093), whereas the one without is rejected with E0309: around a reference, a
//! tuple, an array, a path, a function pointer and a trait object, in the field of a struct and in
//! the variant of an enum, and for `Self`. Not across a projection, nor for `'static` (which the
//! struct declares as the original does, RFC 2093 infers no such bound) or the lifetime of a
//! `for<..>` binder, whose copies compile as they are.
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

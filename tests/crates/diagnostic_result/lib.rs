//! A Referrer's crate for the errors that a Leaker's macro passes on (`#[prop_leak_result]`, in
//! `type_leak_test_leaker::kinds`). `Denied` (9193) has three. The invocation allows the trait
//! `Clone`, so it is not reported, and the `core::fmt::Debug` and the `Held<'_>` are. The copy is
//! made all the same, with these two as written: the host implements the trait.
//!
//! rustc stops after the expansion that reported an error, so nothing here shows the copy but its
//! expansion (`-Zunpretty=expanded`, tests/diagnostic.rs).

use type_leak_test_leaker::kinds::Held;

type_leak_test_leaker::__tl_prop_9193!(
    path = ::type_leak_test_leaker::kinds,
    accept,
    allow(trait Clone)
);

//! A Referrer which trusts the names of the crates, and one which does not, in a crate that has
//! no dependency on the crate of an optional entry (`tests/crates/trusting`).
//!
//! `type_leak_test_leaker::x::ExtDep` names `::type_leak_test_ext::Ext<..>`. It is an optional
//! entry (see `Referrer::must()`): the Referrer leaves it as written by default, and the copy then
//! names a crate that the Referrer's crate does not have; with `allow_ambiguous_crate_name(false)`
//! the whole type is replaced with the Repeater's projection, and the crate does not need it.
//! `tests/cross_crate.rs` has the text of both copies. This is the trusting one, which rustc
//! rejects, and the same crate with the strict flag, which it accepts and whose own tests (the
//! copies of a struct and of a fragment of an impl, and an impl that forwards) pass: the outcome
//! differs by that flag alone.

#[path = "common/nested.rs"]
mod nested;

#[test]
fn a_trusting_referrer_needs_the_dependency() {
    // The methods that name the crate fail alike (rustc prints the same error once).
    let trusting = nested::cargo("trusting", &["check"], &[]);
    assert!(!trusting.success, "{}", trusting.stderr);
    assert_eq!(
        nested::error_codes(&trusting.stderr),
        ["E0433"],
        "{}",
        trusting.stderr
    );
    assert!(
        trusting
            .stderr
            .contains("cannot find `type_leak_test_ext` in"),
        "{}",
        trusting.stderr
    );
}

#[test]
fn a_strict_referrer_does_not() {
    // Nothing else is wrong with the crate, so the flag alone makes the difference.
    let strict = nested::cargo("trusting", &["check"], &["strict"]);
    assert!(strict.success, "{}", strict.stderr);
    assert_eq!(strict.stderr, "");
    // And what it wrote works.
    let tests = nested::cargo("trusting", &["test", "--lib"], &["strict"]);
    assert!(tests.success, "{}{}", tests.stdout, tests.stderr);
    assert!(
        tests.stdout.contains("test result: ok. 3 passed; 0 failed"),
        "{}{}",
        tests.stdout,
        tests.stderr
    );
}

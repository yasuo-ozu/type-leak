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
//! impl<'a, T> Repeater<(&'a (), T, &'a T), ID, NTH> for S { type Type = Rel<'a, T>; }
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
//! Iterator<Item = &'x u8>>`) is not: the Referrer may replace that type as a whole (it is an
//! optional entry, see `Referrer::must()`), so the function pointer or the trait object holding the
//! binder is leaked as a whole (a bound is an error), and the copies compile whichever the Referrer
//! does (`Binders`: `core_*`, `external_*`, and the `check_leak!` next to `AnchoredInside`, where
//! `::std` is allowed, so the types of it are no optional entries and anchor as before).
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

#[path = "common/nested.rs"]
mod nested;

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

/// `Rel<'a, T>` without `T: 'a` written: the Repeater's implementation, whose header encodes `T`
/// but implies nothing about `'a`, is rejected (`m::Unbounded`).
#[test]
fn the_bound_of_a_path_with_a_lifetime_must_be_written() {
    negative(
        "negative-unbounded",
        &["E0309"],
        "error[E0309]: the parameter type `T` may not live long enough",
    );
}

/// The same for `Self` (`m::UnboundedSelf`): `Rel<'a, Self>` needs `Self: 'a`, and the header
/// encodes `__S` only.
#[test]
fn the_bound_of_self_in_a_path_with_a_lifetime_must_be_written() {
    negative(
        "negative-unbounded-self",
        &["E0309"],
        "error[E0309]: the parameter type `__S` may not live long enough",
    );
}

/// The field of a struct (`m::UnboundedAround`): `Rel2<T>` is leaked with the generics `<T>` its
/// signature writes, so the copy `Ref<'a, <Wrapper as Repeater<(T,), ..>>::Type>` is not
/// well-formed without `T: 'a`, which nothing infers through the projection.
#[test]
fn the_bound_around_a_type_leaked_alone_must_be_written() {
    negative(
        "negative-unbounded-around",
        &["E0309"],
        "error[E0309]: the parameter type `T` may not live long enough",
    );
}

/// The alternative which is not taken for a trait object without a lifetime bound as the argument
/// of a path with a lifetime: leaking it alone with `+ 'a` written, and keeping the path around
/// the projection in the copy. The object lifetime of `Inferred` and `Unrelated` is `'static`, so
/// the copy returns another type (E0308 on stable, E0478 on nightly, for each of the two), and
/// the forwarding body hands out a borrow that does not live that long (E0521).
#[test]
fn a_trait_object_must_be_leaked_with_its_path() {
    negative(
        "negative-alone",
        &["E0308", "E0308", "E0521", "E0521"],
        "borrowed data escapes outside of method",
    );
}

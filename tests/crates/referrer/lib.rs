//! The Referrer's crate: implements the Leaker's traits for wrappers, without importing any of
//! the types relative to `type_leak_test_leaker::m`. This crate compiles only if `type-leak`
//! rewrites every such type.

use type_leak_test_leaker::m::Inner;

/// Concrete arguments, including a negative `i32`.
pub struct Concrete(pub Inner);
type_leak_test_leaker::m::Consts!(impl Consts<u8, 3, true, 'x', -3> for Concrete => 0);

/// Generic arguments, including a bare const parameter and a const expression.
pub struct Generic<const B: bool, const N: usize>(pub Inner);
type_leak_test_leaker::m::Consts!(
    impl<const B: bool, const N: usize> Consts<u16, N, B, 'y', { 1 + 2 }> for Generic<B, N> => 0
);

// All defaulted arguments omitted: `Defaults<u8, 2, true>`.
type_leak_test_leaker::m::Defaults!(impl Defaults<u8> for Concrete => 0);

// Only some defaulted arguments given: `Defaults<u16, N, true>`.
type_leak_test_leaker::m::Defaults!(
    impl<const B: bool, const N: usize> Defaults<u16, N> for Generic<B, N> => 0
);

// Where clauses, typed receivers, argument patterns, HRTB, elided lifetimes and RPITIT.
type_leak_test_leaker::m::Edge!(impl Edge<u8> for Concrete => 0);

// A trait without generic arguments.
type_leak_test_leaker::m::Plain!(impl Plain for Concrete => 0);

/// The omitted default `RelDefault<u8>` is relative to the Leaker's module.
pub struct Omitted(pub Inner);
type_leak_test_leaker::m::Defaulted!(impl Defaulted<u8> for Omitted => 0);

/// The default is given explicitly.
pub struct Explicit(pub Inner);
type_leak_test_leaker::m::Defaulted!(impl Defaulted<u8, u32> for Explicit => 0);

/// A generic argument is used by the default.
pub struct Generic2<T>(pub Inner, pub core::marker::PhantomData<T>);
type_leak_test_leaker::m::Defaulted!(impl<T> Defaulted<T> for Generic2<T> => 0);

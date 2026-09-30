//! Uses `type-leak` across crate boundaries through a proc macro, as a downstream crate would.
//!
//! - `tests/crates/macro`: the proc macro using `type-leak` (`Leaker`, `Referrer` and
//!   `RepeaterSlots`).
//! - `tests/crates/leaker`: the Leaker's crate, defining traits with relative types.
//! - `tests/crates/referrer`: the Referrer's crate, implementing the traits for wrappers which
//!   forward to `Inner`.

use type_leak_test_leaker::m::{Consts, Defaults, Holder, Inner};
use type_leak_test_referrer::{Concrete, Generic};

#[test]
fn concrete_args() {
    let c = Concrete(Inner);
    assert_eq!(c.relative().0, 42);
    assert_eq!(
        c.array().iter().map(|r| r.0).collect::<Vec<_>>(),
        vec![1, 2, 3]
    );
    let (flag, letter) = c.flags();
    assert!(flag.get());
    assert_eq!(letter.get(), 'x');
    assert_eq!(c.value(5u8), (5u8, -3));
}

#[test]
fn generic_args() {
    let g = Generic::<false, 2>(Inner);
    assert_eq!(g.array().len(), 2);
    let (flag, letter) = g.flags();
    assert!(!flag.get());
    assert_eq!(letter.get(), 'y');
    assert_eq!(g.value(7u16), (7u16, 3));
}

#[test]
fn local_generics() {
    let c = Concrete(Inner);
    assert!(c.local_flag::<true>().get());
    assert!(!c.local_flag::<false>().get());
    assert_eq!(
        c.local_array::<4>().iter().map(|r| r.0).collect::<Vec<_>>(),
        vec![4, 4, 4, 4]
    );
    let n = 9;
    assert_eq!(c.holder(Holder(&n)), 9);
}

#[test]
fn omitted_default_args() {
    let c = Concrete(Inner);
    assert_eq!(c.default_array().len(), 2);
    assert!(c.default_flag().get());

    let g = Generic::<false, 5>(Inner);
    assert_eq!(g.default_array().len(), 5);
    assert!(g.default_flag().get());
}

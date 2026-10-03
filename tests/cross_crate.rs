//! Uses `type-leak` across crate boundaries through a proc macro, as a downstream crate would.
//!
//! - `tests/crates/macro`: the proc macro using `type-leak` (`Leaker`, `Referrer` and
//!   `RepeaterSlots`).
//! - `tests/crates/leaker`: the Leaker's crate, defining traits with relative types.
//! - `tests/crates/referrer`: the Referrer's crate, implementing the traits for wrappers which
//!   forward to `Inner`.

use type_leak_test_leaker::m::{
    Consts, Defaulted, Defaults, DollarCrate, Edge, Holder, Inner, Plain, RelDefault, RelType,
};
use type_leak_test_referrer::{Concrete, Explicit, Generic, Generic2, Omitted};

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

#[test]
fn omitted_relative_default() {
    // The omitted `U` is the Leaker's `RelDefault<u8>`, which the Referrer's crate cannot name.
    let r: RelDefault<u8> = <Omitted as Defaulted<u8>>::get(&Omitted(Inner), 7);
    assert_eq!(r, RelDefault(7));
    // The default mentions the argument given for `T`.
    let r: RelDefault<&str> =
        <Generic2<&str> as Defaulted<&str>>::get(&Generic2(Inner, Default::default()), "x");
    assert_eq!(r, RelDefault("x"));
    // An explicit argument is used as it is.
    assert_eq!(
        <Explicit as Defaulted<u8, u32>>::get(&Explicit(Inner), 5),
        105
    );
}

#[test]
fn edge_cases() {
    let c = Concrete(Inner);
    let pair = Edge::<u8>::e_where(&c, 7u8);
    assert_eq!((pair.0, pair.1), (7, 7));
    assert_eq!(Edge::<u8>::e_where_rel(&c, 9u32), 9);
    assert_eq!(Edge::<u8>::e_ref(&c).0, 2);
    assert_eq!(Edge::<u8>::e_pat(&c, RelType(0), 4), 5);
    assert_eq!(Edge::<u8>::e_hrtb(&c, |r: &RelType| r.0 * 10), 30);
    assert_eq!(Edge::<u8>::e_elided(&c, &RelType(6)), 6);
    assert_eq!(
        Edge::<u8>::e_impl(&c).map(|r| r.0).collect::<Vec<_>>(),
        vec![4]
    );
    assert_eq!(Edge::<u8>::e_box(Box::new(Concrete(Inner))).0, 1);
    assert_eq!(c.p().0, 5);
}

/// The `$crate` paths of `DollarCrate` (a supertrait, types, an array length and a bound), which
/// the Referrer's crate copies, resolve to the Leaker's crate.
///
/// The exact results of the Leaker for such paths (with and without
/// `allow_external_crates`) are checked when the Leaker's crate is compiled, by
/// `check_leak!` in `tests/crates/leaker`.
#[test]
fn dollar_crate_paths() {
    use type_leak_test_leaker::m_portable::{Portable, Wrap};
    let c = Concrete(Inner);
    let _: Portable = c.d_ret();
    assert_eq!(c.d_arg(Wrap(4)), 5);
    assert_eq!(c.d_arr(), [7, 7]);
    assert_eq!(c.d_rel(Portable).0, 11);
}

/// `Self`, its associated types (also as a GAT and in a bound) and the generics of the trait in
/// the leaked types of `type_leak_test_leaker::s`.
#[test]
fn self_and_associated_types() {
    use type_leak_test_leaker::s::{Assoc, SelfSized, Wrap};
    let c = Concrete(Inner);
    // `Wrap<Self>` where `Self: Sized`, implemented by the Referrer's crate.
    let me: Wrap<Concrete> = c.me();
    let _ = me;
    assert_eq!(c.plain(), Wrap(1));
    // `Self::Item` is `u8` for `Concrete`.
    let item: Wrap<u8> = c.item();
    assert_eq!(item, Wrap(8));
    assert_eq!(c.u(3u8), Wrap(3u8));
    let v = 5u16;
    let g: Wrap<&u16> = c.gat(&v);
    assert_eq!(g, Wrap(&5));
}

#[test]
fn lifetimes_trait_objects_and_projections() {
    use type_leak_test_leaker::s::{DynRoot, Out, RelImpl, Wrap, Q};
    let c = Concrete(Inner);
    let x = 4u64;
    assert_eq!(Out::f(&c, &x), Wrap(&4));
    assert_eq!(c.boxed().rel(), 31);
    assert_eq!(c.borrowed(&RelImpl), 32);
    assert_eq!(c.lent().0.rel(), 31);
    // `<T>::Item` with `T` = `Range<u32>`.
    assert_eq!(Q::<::core::ops::Range<u32>>::f(&c, 6), Wrap(6));
}

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

// A trait using `$crate` paths (a supertrait, types, an array length and a bound): they resolve to
// the Leaker's crate from here.
impl type_leak_test_leaker::m_portable::PortTr for Concrete {}
type_leak_test_leaker::m::DollarCrate!(impl DollarCrate for Concrete => 0);

// `Self` in the leaked types: the implementations of methods which return `Self` are given
// verbatim (the forwarded value would be the field's type), and so are the associated types.
type_leak_test_leaker::s::SelfSized!(impl SelfSized for Concrete => 0 {
    fn me(&self) -> type_leak_test_leaker::s::Wrap<Self>
    where
        Self: ::core::marker::Sized,
    {
        type_leak_test_leaker::s::Wrap(Concrete(Inner))
    }
});
type_leak_test_leaker::s::Assoc!(impl Assoc for Concrete => 0 {
    type Item = u8;
    type G<'a, V: 'a> = &'a V;
});
type_leak_test_leaker::s::Out!(impl<'x, T> Out<'x, T> for Concrete => 0);
type_leak_test_leaker::s::DynRoot!(impl<'x> DynRoot<'x> for Concrete => 0);
type_leak_test_leaker::s::Q!(impl<I: ::core::iter::Iterator> Q<I> for Concrete => 0);

// A trait object inheriting the lifetime of its path, a bare identifier argument which is a const
// and types in expressions, which are leaked as a whole. The implementors of the last are generic, where a
// projection (as in `[u8; size_of::<<Self as Repeater<..>>::Type>()]`) is not permitted in an
// anonymous constant.
type_leak_test_leaker::s::PathDyn!(impl<'x> PathDyn<'x> for Concrete => 0);
type_leak_test_leaker::s::ConstArg!(impl ConstArg for Concrete => 0);
type_leak_test_leaker::s::ExprTypes!(impl ExprTypes for Concrete => 0);

/// A generic implementor.
pub struct GenericExpr<const B: bool>(pub Inner);
type_leak_test_leaker::s::ExprTypes!(impl<const B: bool> ExprTypes for GenericExpr<B> => 0);

/// The types of external crates are optional entries: the Referrer leaves them as written, or
/// replaces them with the Repeater's projections (`strict`, the Referrer does not trust the names
/// of the crates), with the relative types in them replaced either way.
pub struct Trusting(pub Inner);
type_leak_test_leaker::m::External!(impl<T> External<T> for Trusting => 0 {
    fn receiver(self: ::std::boxed::Box<Self>) -> usize {
        type_leak_test_leaker::m::External::<T>::receiver(::std::boxed::Box::new(self.0))
    }
});

/// The same, replacing the optional entries as well.
pub struct Strict(pub Inner);
type_leak_test_leaker::m::External!(strict impl<T> External<T> for Strict => 0 {
    fn receiver(self: ::std::boxed::Box<Self>) -> usize {
        type_leak_test_leaker::m::External::<T>::receiver(::std::boxed::Box::new(self.0))
    }
});

// What the two Referrers write for the same items, as text: the trusting one names the crate (its
// relative types are replaced), the strict one replaces the whole type of it. This crate has no
// dependency on that crate (`type_leak_test_ext`), so it cannot compile either copy: the strict one
// is compiled in `tests/crates/trusting`, which has the feature `strict` for it.
type_leak_test_leaker::x::ExtDep!(show TRUSTING_COPY [Wrapper] {
    pub fn f(x: ::type_leak_test_ext::Ext<(RelX,)>, y: ::type_leak_test_ext::Ext<u8>) {}
});
type_leak_test_leaker::x::ExtDep!(strict show STRICT_COPY [Wrapper] {
    pub fn f(x: ::type_leak_test_ext::Ext<(RelX,)>, y: ::type_leak_test_ext::Ext<u8>) {}
});

/// Fragments of impls analyzed with the scope API (`Leaker::enter_impl()`): the Referrer's copy of
/// each is checked against the original with a witness, and by finding the impl of the Leaker's
/// crate for the rewritten types.
pub mod scoped {
    // `coinduction`'s shape: `ImplSelf::Marker`.
    type_leak_test_leaker::__scope_apply_7001!(path = ::type_leak_test_leaker::scoped);
    // `newer-type`'s shape: `ImplSelf::Param`, `Self` is the self type.
    type_leak_test_leaker::__scope_apply_7002!(path = ::type_leak_test_leaker::scoped);
    type_leak_test_leaker::__scope_apply_7003!(path = ::type_leak_test_leaker::scoped);
    // A where clause of the impl, which the Repeater's impl has too.
    type_leak_test_leaker::__scope_apply_7006!(path = ::type_leak_test_leaker::scoped);
}

/// Items of every kind, copied by the Referrer: each copy has a witness for each type position, so
/// this crate compiles only if every rewritten type is the original one.
pub mod kinds {
    use type_leak_test_leaker::kinds::{Rel, Wrap};
    pub use type_leak_test_leaker::kinds::Pair;
    #[allow(unused_imports)]
    use Rel as _;
    #[allow(unused_imports)]
    use Wrap as _;

    type_leak_test_leaker::__tl_prop_9001!(path = ::type_leak_test_leaker::kinds);
    type_leak_test_leaker::__tl_prop_9002!(path = ::type_leak_test_leaker::kinds);
    type_leak_test_leaker::__tl_prop_9005!(path = ::type_leak_test_leaker::kinds);
    type_leak_test_leaker::__tl_prop_9011!(path = ::type_leak_test_leaker::kinds);
    type_leak_test_leaker::__tl_prop_9012!(path = ::type_leak_test_leaker::kinds);
    type_leak_test_leaker::__tl_prop_9021!(path = ::type_leak_test_leaker::kinds);
    type_leak_test_leaker::__tl_prop_9022!(path = ::type_leak_test_leaker::kinds);
    type_leak_test_leaker::__tl_prop_9031!(path = ::type_leak_test_leaker::kinds);
    type_leak_test_leaker::__tl_prop_9032!(path = ::type_leak_test_leaker::kinds);
    type_leak_test_leaker::__tl_prop_9033!(path = ::type_leak_test_leaker::kinds);
    type_leak_test_leaker::__tl_prop_9034!(path = ::type_leak_test_leaker::kinds);
    type_leak_test_leaker::__tl_prop_9041!(path = ::type_leak_test_leaker::kinds);
    type_leak_test_leaker::__tl_prop_9051!(path = ::type_leak_test_leaker::kinds);
    // The rejected ones expand to nothing.
    type_leak_test_leaker::__tl_prop_9003!(path = ::type_leak_test_leaker::kinds);
    type_leak_test_leaker::__tl_prop_9052!(path = ::type_leak_test_leaker::kinds);
    // A trait implemented for an unsized type: `dyn Host9053`.
    type_leak_test_leaker::__tl_prop_9053!(path = ::type_leak_test_leaker::kinds, dyn_host);
}

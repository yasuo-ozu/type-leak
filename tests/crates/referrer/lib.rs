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

/// A braced argument for a const parameter whose braced use is in a default (F73): the completed
/// default is `[u8; { M }]`.
pub struct G9246<const M: usize>(pub Inner);
type_leak_test_leaker::m::Braced9246!(impl<const M: usize> Braced9246<{ M }> for G9246<M> => 0);

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
// The trait's generic is named like the parameter of the Repeater's impl for the implementor (F76).
type_leak_test_leaker::s::Clash9248!(impl<T> Clash9248<T> for Concrete => 0 {
    type Item = u8;
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
/// replaces them with the Repeater's projections (the Referrer does not trust the names
/// of the crates), with the relative types in them replaced either way.
pub struct Trusting(pub Inner);
type_leak_test_leaker::m::External!(impl<T> External<T> for Trusting => 0 {
    fn receiver(self: ::std::boxed::Box<Self>) -> usize {
        type_leak_test_leaker::m::External::<T>::receiver(::std::boxed::Box::new(self.0))
    }
});

/// The same, replacing the optional entries as well.
pub struct Strict(pub Inner);
type_leak_test_leaker::m::External!(impl<T> External<T> for Strict => 0 {
    fn receiver(self: ::std::boxed::Box<Self>) -> usize {
        type_leak_test_leaker::m::External::<T>::receiver(::std::boxed::Box::new(self.0))
    }
});

// What the macro writes for the same items after a Leaker that allows the crate and one that does
// not, as text: the first names the crate, the second replaces the whole type of it. This crate has
// no dependency on that crate (`type_leak_test_ext`), so it cannot compile either copy: the second
// is compiled in `tests/crates/trusting`, which has the feature `replace` for it.
type_leak_test_leaker::x::ExtDepAllowed!(show ALLOWED_COPY [Wrapper] {
    pub fn f(x: ::type_leak_test_ext::Ext<(RelX,)>, y: ::type_leak_test_ext::Ext<u8>) {}
});
type_leak_test_leaker::x::ExtDep!(show REPLACED_COPY [Wrapper] {
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
    // A bare identifier that is a const, in a module: beside a name that the module declares, an
    // absolute `use` declares it (the copy keeps it), and a relative one is rejected (it expands to
    // nothing).
    type_leak_test_leaker::__tl_prop_9101!(path = ::type_leak_test_leaker::kinds);
    type_leak_test_leaker::__tl_prop_9102!(path = ::type_leak_test_leaker::kinds);
    type_leak_test_leaker::__tl_prop_9051!(path = ::type_leak_test_leaker::kinds);
    // A trait implemented for an unsized type: `dyn Host9053`.
    type_leak_test_leaker::__tl_prop_9053!(path = ::type_leak_test_leaker::kinds, dyn_host);
    // A bare identifier that is a const: the path that holds it is an entry as a whole.
    type_leak_test_leaker::__tl_prop_9081!(path = ::type_leak_test_leaker::kinds);
    type_leak_test_leaker::__tl_prop_9083!(path = ::type_leak_test_leaker::kinds);
    type_leak_test_leaker::__tl_prop_9084!(path = ::type_leak_test_leaker::kinds);
    // Items nested in a module that name their own `Self`.
    type_leak_test_leaker::__tl_prop_9061!(path = ::type_leak_test_leaker::kinds);
    type_leak_test_leaker::__tl_prop_9062!(path = ::type_leak_test_leaker::kinds);
    type_leak_test_leaker::__tl_prop_9063!(path = ::type_leak_test_leaker::kinds);
    // Items nested in a module whose generics have a bound that names their own `Self`: the copy
    // has the bound for its own type.
    type_leak_test_leaker::__tl_prop_9111!(path = ::type_leak_test_leaker::kinds);
    type_leak_test_leaker::__tl_prop_9112!(path = ::type_leak_test_leaker::kinds);
    type_leak_test_leaker::__tl_prop_9113!(path = ::type_leak_test_leaker::kinds);
    type_leak_test_leaker::__tl_prop_9114!(path = ::type_leak_test_leaker::kinds);
    // This one names `Self` in a field too, and expands to nothing.
    type_leak_test_leaker::__tl_prop_9115!(path = ::type_leak_test_leaker::kinds);
    type_leak_test_leaker::__tl_prop_9116!(path = ::type_leak_test_leaker::kinds);
    type_leak_test_leaker::__tl_prop_9117!(path = ::type_leak_test_leaker::kinds);
    type_leak_test_leaker::__tl_prop_9118!(path = ::type_leak_test_leaker::kinds);
    type_leak_test_leaker::__tl_prop_9119!(path = ::type_leak_test_leaker::kinds);
    type_leak_test_leaker::__tl_prop_9120!(path = ::type_leak_test_leaker::kinds);
    // The entry is the last field of a struct, or a field of an enum: the impl is `?Sized` unless
    // a bound needs `Self: Sized`.
    type_leak_test_leaker::__tl_prop_9125!(path = ::type_leak_test_leaker::kinds);
    type_leak_test_leaker::__tl_prop_9126!(path = ::type_leak_test_leaker::kinds);
    type_leak_test_leaker::__tl_prop_9127!(path = ::type_leak_test_leaker::kinds);
    type_leak_test_leaker::__tl_prop_9128!(path = ::type_leak_test_leaker::kinds);
    // An absolute type that names their own `Self` is copied as written, by a strict Referrer too.
    type_leak_test_leaker::__tl_prop_9121!(path = ::type_leak_test_leaker::kinds);
    type_leak_test_leaker::__tl_prop_9122!(path = ::type_leak_test_leaker::kinds);
    type_leak_test_leaker::__tl_prop_9124!(path = ::type_leak_test_leaker::kinds);
    // The defaults of a struct and an enum nested in a module, whose generic has a bound that names
    // their `Self`: the entry has no `Self`, and the copy writes the default through the Repeater.
    type_leak_test_leaker::__tl_prop_9151!(path = ::type_leak_test_leaker::kinds);
    type_leak_test_leaker::__tl_prop_9152!(path = ::type_leak_test_leaker::kinds);
    type_leak_test_leaker::__tl_prop_9153!(path = ::type_leak_test_leaker::kinds);
    type_leak_test_leaker::__tl_prop_9154!(path = ::type_leak_test_leaker::kinds);
    // A type that names the `Self` of such an item too, with `nested_self_entries`: for any type,
    // and the copy passes its own.
    type_leak_test_leaker::__tl_prop_9161!(path = ::type_leak_test_leaker::kinds);
    type_leak_test_leaker::__tl_prop_9162!(path = ::type_leak_test_leaker::kinds);
    type_leak_test_leaker::__tl_prop_9163!(path = ::type_leak_test_leaker::kinds);
    type_leak_test_leaker::__tl_prop_9164!(path = ::type_leak_test_leaker::kinds);
    // A bound that names a leaked type that names the `Self` of the nested struct is rejected, as
    // `intern_struct()` rejects it (F51): it expands to nothing.
    type_leak_test_leaker::__tl_prop_9206!(path = ::type_leak_test_leaker::kinds);
    // A bound of a generic that names it, next to one that fixes an associated type: left out of
    // the Repeater's impls of the entries. The one invoked as `strict` replaces the alias too.
    type_leak_test_leaker::__tl_prop_9171!(path = ::type_leak_test_leaker::kinds);
    type_leak_test_leaker::__tl_prop_9172!(path = ::type_leak_test_leaker::kinds);
    type_leak_test_leaker::__tl_prop_9173!(path = ::type_leak_test_leaker::kinds);
    type_leak_test_leaker::__tl_prop_9174!(path = ::type_leak_test_leaker::kinds);
    type_leak_test_leaker::__tl_prop_9175!(path = ::type_leak_test_leaker::kinds);
    type_leak_test_leaker::__tl_prop_9176!(path = ::type_leak_test_leaker::kinds);
    type_leak_test_leaker::__tl_prop_9177!(path = ::type_leak_test_leaker::kinds);
    type_leak_test_leaker::__tl_prop_9178!(path = ::type_leak_test_leaker::kinds);
    type_leak_test_leaker::__tl_prop_9179!(path = ::type_leak_test_leaker::kinds);
    type_leak_test_leaker::__tl_prop_9180!(path = ::type_leak_test_leaker::kinds);
    // The self type of an impl and `Self` of a struct keep the bounds of their item.
    type_leak_test_leaker::__tl_prop_9182!(path = ::type_leak_test_leaker::kinds);
    type_leak_test_leaker::__tl_prop_9183!(path = ::type_leak_test_leaker::kinds);
    // The bound of another generic is not left out: the Leaker rejects it (it expands to nothing).
    type_leak_test_leaker::__tl_prop_9181!(path = ::type_leak_test_leaker::kinds);
    // A bound that names its generic through an alias stays in the context of an entry that names
    // `Self`, and where a method equates the `Item`: the Leaker rejects the entries in a predicate
    // (it expands to nothing, as the copy fails to build).
    type_leak_test_leaker::__tl_prop_9223!(path = ::type_leak_test_leaker::kinds);
    // The copy is used at an instantiation that leaves the defaults out and passes a const
    // parameter by its own name (F65, F66): the use site completes the arguments and writes the
    // braced const parameters once.
    type_leak_test_leaker::__tl_prop_9224!(path = ::type_leak_test_leaker::kinds, inst(2));
    type_leak_test_leaker::__tl_prop_9225!(path = ::type_leak_test_leaker::kinds, inst(2));
    type_leak_test_leaker::__tl_prop_9226!(path = ::type_leak_test_leaker::kinds, inst(2));
    type_leak_test_leaker::__tl_prop_9227!(path = ::type_leak_test_leaker::kinds, inst(2));
    // `Self: 'x` is no reason to keep the bound that is left out (F47).
    type_leak_test_leaker::__tl_prop_9200!(path = ::type_leak_test_leaker::kinds);
    type_leak_test_leaker::__tl_prop_9201!(path = ::type_leak_test_leaker::kinds);
    // The defaults of a nested trait whose generic has a bound that names its `Self`: the copy is
    // declared here, with the defaults written through the host.
    type_leak_test_leaker::__tl_prop_9131!(path = ::type_leak_test_leaker::kinds);
    type_leak_test_leaker::__tl_prop_9132!(path = ::type_leak_test_leaker::kinds);
    // The entries that do not name `Self` are for a primitive type, the others for the host.
    type_leak_test_leaker::__tl_prop_9209!(path = ::type_leak_test_leaker::kinds);
    // The entries that do not need `Self` are for a marker type, the others for the host. The
    // default of `MarkerDefaultCopy` is written through the host: an impl that takes it needs
    // `Owner9210: AsRef<Host9210>` (and would need `AsRef<Marker9210>` if it were the marker).
    type_leak_test_leaker::__tl_prop_9210!(path = ::type_leak_test_leaker::kinds);
    pub struct Owner9210;
    impl AsRef<Host9210> for Owner9210 {
        fn as_ref(&self) -> &Host9210 {
            unimplemented!()
        }
    }
    impl MarkerDefaultCopy<Owner9210> for Host9210 {
        fn get(&self, _t: Owner9210, _u: ::type_leak_test_leaker::kinds::Defd<Owner9210>) {}
        fn plain(&self) -> ::type_leak_test_leaker::kinds::Rel {
            ::type_leak_test_leaker::kinds::Rel(0)
        }
    }
    // The types in the values of an item, with `check_values`: the copy compiles only if every
    // type in a value that this crate cannot name is replaced.
    type_leak_test_leaker::__tl_prop_9141!(path = ::type_leak_test_leaker::kinds);
    type_leak_test_leaker::__tl_prop_9142!(path = ::type_leak_test_leaker::kinds);
    // The names of the prelude of every edition in the values and the bounds (issue 3).
    type_leak_test_leaker::__tl_prop_9208!(path = ::type_leak_test_leaker::kinds);
    // The bindings of a discriminant and of the default of a const parameter are local (F56). The
    // relative macro and struct literal are rejected: they expand to nothing, as they could not be
    // resolved here.
    type_leak_test_leaker::__tl_prop_9215!(path = ::type_leak_test_leaker::kinds);
    type_leak_test_leaker::__tl_prop_9216!(path = ::type_leak_test_leaker::kinds);
    type_leak_test_leaker::__tl_prop_9217!(path = ::type_leak_test_leaker::kinds);
    type_leak_test_leaker::__tl_prop_9218!(path = ::type_leak_test_leaker::kinds);
    // The scopes that share names: every entry has the generics of its own scope.
    type_leak_test_leaker::__tl_prop_9071!(path = ::type_leak_test_leaker::kinds);
    type_leak_test_leaker::__tl_prop_9072!(path = ::type_leak_test_leaker::kinds);
    type_leak_test_leaker::__tl_prop_9073!(path = ::type_leak_test_leaker::kinds);
    type_leak_test_leaker::__tl_prop_9074!(path = ::type_leak_test_leaker::kinds);
    type_leak_test_leaker::__tl_prop_9075!(path = ::type_leak_test_leaker::kinds);
    type_leak_test_leaker::__tl_prop_9077!(path = ::type_leak_test_leaker::kinds);
    // Paths that end with the name of a generic, and a module that declares one.
    type_leak_test_leaker::__tl_prop_9091!(path = ::type_leak_test_leaker::kinds);
    type_leak_test_leaker::__tl_prop_9092!(path = ::type_leak_test_leaker::kinds);
    type_leak_test_leaker::__tl_prop_9093!(path = ::type_leak_test_leaker::kinds);
    // The cyclic bound is rejected: it expands to nothing.
    type_leak_test_leaker::__tl_prop_9076!(path = ::type_leak_test_leaker::kinds);
    // An optional entry that holds a method lifetime is copied as written, by a strict Referrer too
    // (F18): the path of this crate resolves here.
    type_leak_test_leaker::__tl_prop_9194!(path = ::type_leak_test_leaker::kinds);
    // The types of `::std` around a trait object bounded by a method lifetime are written as they
    // are, and the lifetime stays in an input (F67).
    type_leak_test_leaker::__tl_prop_9228!(path = ::type_leak_test_leaker::kinds);
    type_leak_test_leaker::__tl_prop_9229!(path = ::type_leak_test_leaker::kinds);
    // The optional self type of an impl is written as it is, though a strict Referrer replaces
    // the optional entries: a bound that names `Self` is next to one that names an optional type
    // (F68).
    type_leak_test_leaker::__tl_prop_9230!(path = ::type_leak_test_leaker::kinds);
    // The copy is used at an instantiation that gives the type parameter a type with a lifetime,
    // where it is in the inputs of a function pointer or `Fn(..)` whose output has an elided
    // lifetime: the use site names that lifetime (F69). The last leaves the default out.
    type_leak_test_leaker::__tl_prop_9231!(path = ::type_leak_test_leaker::kinds, inst(1), lifetime_args);
    type_leak_test_leaker::__tl_prop_9232!(path = ::type_leak_test_leaker::kinds, inst(1), lifetime_args);
    type_leak_test_leaker::__tl_prop_9233!(path = ::type_leak_test_leaker::kinds, inst(2), lifetime_args);
    // The same where an entry is in the inputs, which the strict Referrer replaces: the path through
    // the Repeater names the lifetime (F70).
    type_leak_test_leaker::__tl_prop_9234!(path = ::type_leak_test_leaker::kinds, inst(1), lifetime_args);
    type_leak_test_leaker::__tl_prop_9235!(path = ::type_leak_test_leaker::kinds, inst(1), lifetime_args);
    // A type of a crate that the rule does not allow, in a where predicate that the context of an
    // entry carries, is no entry: the copy writes it as it is (F71). The last is the control, whose
    // type names no generic and stays an entry.
    type_leak_test_leaker::__tl_prop_9237!(path = ::type_leak_test_leaker::kinds);
    type_leak_test_leaker::__tl_prop_9238!(path = ::type_leak_test_leaker::kinds);
    type_leak_test_leaker::__tl_prop_9239!(path = ::type_leak_test_leaker::kinds);
    type_leak_test_leaker::__tl_prop_9240!(path = ::type_leak_test_leaker::kinds);
    type_leak_test_leaker::__tl_prop_9241!(path = ::type_leak_test_leaker::kinds);
    type_leak_test_leaker::__tl_prop_9242!(path = ::type_leak_test_leaker::kinds);
    type_leak_test_leaker::__tl_prop_9243!(path = ::type_leak_test_leaker::kinds);
    type_leak_test_leaker::__tl_prop_9244!(path = ::type_leak_test_leaker::kinds);
    // Const generics of type `u64` and `u128`: the slot `U64` and the `KEY` of the Repeater.
    type_leak_test_leaker::__tl_prop_9236!(path = ::type_leak_test_leaker::kinds);
}

/// Items that the Leaker rejects and accepts with `LeakerResult::allow()` of the causes of their
/// errors: the part that conflicts is copied as written, so this module names what the copies name
/// (`Held`, `ElidedTr`, `SharedTr`, `Wrap`) and invokes them with `accept`. Everything else is
/// replaced as usual.
pub mod accepted {
    #[allow(unused_imports)]
    use type_leak_test_leaker::kinds::{ElidedTr, Held, Leaf, SharedTr, Two, Wrap};

    type_leak_test_leaker::__tl_prop_9184!(path = ::type_leak_test_leaker::kinds, accept);
    type_leak_test_leaker::__tl_prop_9185!(path = ::type_leak_test_leaker::kinds, accept);
    type_leak_test_leaker::__tl_prop_9186!(path = ::type_leak_test_leaker::kinds, accept);
    type_leak_test_leaker::__tl_prop_9187!(path = ::type_leak_test_leaker::kinds, accept);
    type_leak_test_leaker::__tl_prop_9188!(path = ::type_leak_test_leaker::kinds, accept);
    type_leak_test_leaker::__tl_prop_9189!(path = ::type_leak_test_leaker::kinds, accept);
    // Items whose macro carries the errors (`prop_leak_result`) and whose invocation says what it
    // allows: all the errors are dropped, or the crate would not compile. The copies keep `Clone`,
    // `core::fmt::Debug`, `PlainTr` and `Held<'_>` as written (the crate names the last two).
    #[allow(unused_imports)]
    use type_leak_test_leaker::kinds::PlainTr;
    // An optional entry in the predicate of a trait, next to an unrelated error that copies a part
    // with `Self::A0` as written (F52). A strict Referrer, which the crate does not depend on `ext`
    // for: the types are of the Leaker's crate, which it depends on.
    #[allow(unused_imports)]
    use type_leak_test_leaker::kinds::GenTr;

    type_leak_test_leaker::__tl_prop_9207!(path = ::type_leak_test_leaker::kinds, accept);

    type_leak_test_leaker::__tl_prop_9190!(
        path = ::type_leak_test_leaker::kinds,
        accept,
        allow(trait Clone)
    );
    type_leak_test_leaker::__tl_prop_9191!(
        path = ::type_leak_test_leaker::kinds,
        accept,
        allow(std_crates)
    );
    type_leak_test_leaker::__tl_prop_9192!(
        path = ::type_leak_test_leaker::kinds,
        accept,
        allow(trait PlainTr, type Held)
    );

    // Parts that name `Self` or have no context, found by the visitor (F43 to F45).
    type_leak_test_leaker::__tl_prop_9196!(path = ::type_leak_test_leaker::kinds, accept);
    type_leak_test_leaker::__tl_prop_9197!(path = ::type_leak_test_leaker::kinds, accept);
    type_leak_test_leaker::__tl_prop_9198!(path = ::type_leak_test_leaker::kinds, accept);
    type_leak_test_leaker::__nested_apply_9199!(path = ::type_leak_test_leaker::kinds);
    // Flat recovery: the types in what is written as it is are entries.
    type_leak_test_leaker::__tl_prop_9220!(path = ::type_leak_test_leaker::kinds, accept);
    // The types in a type written as it is that hold a lifetime of a method are written as they are.
    type_leak_test_leaker::__tl_prop_9221!(path = ::type_leak_test_leaker::kinds, accept);
    type_leak_test_leaker::__tl_prop_9222!(path = ::type_leak_test_leaker::kinds, accept);

    // The traits of two items analyzed together: `f` keeps the type as written, `Held` and `Wrap`
    // are named here (F48).
    type_leak_test_leaker::__shared_apply_9202!(path = ::type_leak_test_leaker::kinds, copy(0));
    type_leak_test_leaker::__shared_apply_9203!(path = ::type_leak_test_leaker::kinds, copy(0));
    type_leak_test_leaker::__shared_apply_9204!(path = ::type_leak_test_leaker::kinds, copy(0));
    // The traits with the culprits keep `Held` as written, where an input keeps no `'b` (F72).
    type_leak_test_leaker::__shared_apply_9245!(
        path = ::type_leak_test_leaker::kinds,
        copy(0),
        copy(2)
    );
}

/// The structs of the two items analyzed together: this module names neither `Held` nor `Wrap`
/// (E0412 if a copy kept them), the entry is written through the Repeater (F48). So is `Held` in
/// the methods that keep their lifetime in another input (F72).
pub mod unnamed {
    type_leak_test_leaker::__shared_apply_9202!(path = ::type_leak_test_leaker::kinds, copy(1));
    type_leak_test_leaker::__shared_apply_9203!(path = ::type_leak_test_leaker::kinds, copy(1));
    type_leak_test_leaker::__shared_apply_9204!(path = ::type_leak_test_leaker::kinds, copy(1));
    type_leak_test_leaker::__shared_apply_9245!(path = ::type_leak_test_leaker::kinds, copy(1));
}

/// The copies that `stages = true` makes after `Rules::std_crates()` given to the result (F50): the
/// Leaker's `ValuePathArguments` has `core::mem::size_of::<*const Rel>()` in a default method. This
/// module names `Rel`, so the copy that keeps the path as written (the error stays, and its cause is
/// allowed) builds here.
pub mod stage_copies {
    #[allow(unused_imports)]
    use type_leak_test_leaker::kinds::Rel;

    type_leak_test_leaker::__tl_st_std_res_9205!(path = ::type_leak_test_leaker::kinds, accept);
}

/// A crate of no name of the Leaker's (`clean`): the macro makes the copy only if the rule left no
/// error. If the rule dropped the path with `Rel` in its arguments, the copy would write `Rel`
/// here, where nothing names it (E0412).
pub mod clean {
    type_leak_test_leaker::__tl_st_std_res_9205!(clean path = ::type_leak_test_leaker::kinds, accept);
}

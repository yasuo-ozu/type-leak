//! A Leaker's crate whose leaked types have outlives bounds, and the Referrer's side of them, for
//! tests/outlives.rs.
//!
//! Only the bounds which are exact are encoded: the ones which are written (`T: 'a`, `where Self:
//! 'a`) and the ones which a reference implies (`&'a X` requires `X: 'a`, which a type inside it
//! must satisfy too, whatever it is). A path with a lifetime argument `R<'a, T>` is not assumed to
//! require `T: 'a` (its definition is not visible, and `struct Unrelated<'a, T>(&'a (), T)` does
//! not), so a bound which only the definition of `R` implies (`struct Rel<'a, T>(&'a T)`) is
//! written in the signature (`fn f<'a, T: 'a>(x: Rel<'a, T>)`): the Repeater's implementation
//! `impl<'a, T> Repeater<(&'a (), T, &'a T), ..> for S { type Type = Rel<'a, T>; }` has `T: 'a` by
//! the encoding of the generics (`&'a T` in the trait arguments), and is rejected with E0309
//! without it, as is the copy of a field of a struct. See `Bounds`, `Around` and `SelfBound`, and
//! `Unbounded`, `UnboundedSelf` and `UnboundedAround` (disabled, as they do not compile).
//!
//! It does not compile if the encoding of a bound is lost (E0309 in the Repeater's
//! implementations or in the copies of the signatures), nor if a type is leaked with the lifetime
//! of a `for<..>` binder whose output mentions it and no input keeps it (E0581 for a function
//! pointer, E0582 for a bound or a trait object), see `Binders`, nor if `Self: 'a` is not encoded
//! (E0309 for `__S`), see `SelfOutlives`, nor if a type leaked alone does not have the bounds
//! which the types around it imply (E0309 in the field of a struct), see `Around`.
//!
//! It does not compile either if a trait object borrowed through a lifetime of a method is leaked
//! so that the copy leaves the lifetime only in a projection, where it constrains nothing (E0195
//! for the method, E0581 and E0582 for a binder), see `Locals` and `LocalBinders`, nor if the
//! trait object is leaked alone with the lifetime of the reference written out as its bound,
//! where a trait with a lifetime bound of its own (`trait StaticTr: 'static`) rejects it (E0478),
//! see `Statics`.
//!
//! Nor does it if a trait object without a lifetime bound, as the argument of a path with a
//! lifetime (`Inferred<'a, dyn RelTr>`), is leaked alone with the lifetime written out (`dyn RelTr +
//! 'a`) where the definition of the path makes it `'static`: E0308 on stable (E0478 on nightly) in
//! the return type of the copy of a method, see `ObjectDefaults`.

#[doc(hidden)]
pub use type_leak_test_macro as __macro;

type_leak_test_macro::define_repeater!(pub Repeater);

// So that the crate can name itself with an absolute path (`::type_leak_test_outlives::Gat`).
extern crate self as type_leak_test_outlives;

/// A trait with a GAT, for the projections which start with a type parameter (`T::A<X>`).
pub trait Gat {
    type A<X>;
}

/// An implementation of [`Gat`], for the tests.
pub struct GatOf;

impl Gat for GatOf {
    type A<X> = ::std::vec::Vec<X>;
}

/// An implementation of [`Gat`] whose projection does not mention its argument, so that the
/// field of a struct is well-formed with an argument which does not outlive its lifetime.
pub struct GatUnit;

impl Gat for GatUnit {
    type A<X> = ();
}

pub mod m {
    use type_leak_test_macro::{check_leak, leak};

    /// `T: 'a` is inferred from the field (RFC 2093), so it is an implied bound wherever
    /// `Rel<'a, T>` is written in a signature, but it is not written anywhere. The Leaker cannot
    /// see it, so it is not encoded: `Rel<'a, T>` is leaked with the bounds which are written
    /// (`fn f<'a, T: 'a>(x: Rel<'a, T>)`), see `Bounds`, and the Repeater's implementation is
    /// rejected otherwise (E0309), see `Unbounded`.
    pub struct Rel<'a, T>(pub &'a T);

    /// The control: nothing relates `T` to `'a`, and no bound is assumed either, so the leaked
    /// type has the generics which are written, and the copy is the same as the original (not
    /// stricter: a caller with a `T` shorter than `'a` is accepted).
    pub struct Loose<'a, T>(pub &'a (), pub T);

    /// The default object lifetime of `R<'a, dyn Tr>` is `'a` only if the definition of `R` WRITES
    /// the bound `T: 'a`, as it does here. See `ObjectDefaults` and the functions below.
    pub struct Explicit<'a, T: ?Sized + 'a>(pub &'a T);

    /// `T: 'a` is only inferred here (RFC 2093) from the field, which does not count: the default
    /// object lifetime of `Inferred<'a, dyn Tr>` is `'static`.
    pub struct Inferred<'a, T: ?Sized>(pub &'a T);

    /// No bound between `T` and `'a`: the default object lifetime of `Unrelated<'a, dyn Tr>` is
    /// `'static`, as with a bound which is only inferred.
    pub struct Unrelated<'a, T: ?Sized>(pub &'a (), pub Box<T>);

    /// A relative type which is not generic over a lifetime: `&'a Rel2<T>` and `::core::cell::Ref<'a, Rel2<T>>`
    /// are well-formed only if `Rel2<T>: 'a`, i.e. `T: 'a`.
    pub struct Rel2<T>(pub T);

    impl<T> ::core::ops::Deref for Rel2<T> {
        type Target = T;
        fn deref(&self) -> &T {
            &self.0
        }
    }

    /// A relative type without generics.
    pub struct Plain(pub u8);

    /// A relative type with a lifetime, for the lifetimes of `for<..>` binders.
    pub struct Hr<'x>(pub &'x u8);

    pub trait RelTr {
        fn rel(&self) -> usize;
    }

    impl RelTr for Plain {
        fn rel(&self) -> usize {
            self.0 as usize
        }
    }

    /// Another relative trait, for the methods with several trait objects.
    pub trait RelTr2 {
        fn rel2(&self) -> usize;
    }

    impl RelTr2 for Plain {
        fn rel2(&self) -> usize {
            self.0 as usize + 100
        }
    }

    /// A relative trait with an associated type, for the associated type bindings of the trait
    /// objects in the inputs of the binders (`dyn RelIt<Item = &'x u8>`, leaked as a whole).
    pub trait RelIt {
        type Item;
    }

    /// A relative trait with a lifetime bound of its own: the default of the lifetime bound of
    /// `dyn StaticTr` is `'static` whatever the reference is (`&'a dyn StaticTr` is `&'a (dyn
    /// StaticTr + 'static)`), which the Leaker cannot see.
    pub trait StaticTr: 'static {
        fn stat(&self) -> usize;
    }

    impl StaticTr for Plain {
        fn stat(&self) -> usize {
            self.0 as usize + 200
        }
    }

    /// Leaked types whose generics are those of the methods.
    #[leak(
        helper = crate::__macro,
        repeater = crate::Repeater,
        module = crate::m,
    )]
    pub trait Bounds {
        /// `T: 'a` is written, as `Rel<'a, T>` requires it (by its definition, which is not
        /// visible, so the bound is not assumed). The Repeater's implementation for `Rel<'a, T>`
        /// is `impl<'a, T> Repeater<(&'a (), T, &'a T), ..> for S`, and the copy of the signature
        /// has the same encoding, which implies `T: 'a` there as well. Without it, the
        /// implementation is rejected with E0309, see `Unbounded`.
        fn by_ref<'a, T: 'a>(&self, x: Rel<'a, T>) -> usize;
        /// `Loose` is not assumed to require `T: 'a` like `Rel` (we cannot see its definition),
        /// and it does not: nothing is written, and the copy is the same as the original (a
        /// caller with a `T` shorter than `'a` is accepted).
        fn loose<'a, T>(&self, x: Loose<'a, T>) -> usize;
        /// The bounds which are written: `Rel<'a, &'b T>` requires `&'b T: 'a`, which `'b: 'a`
        /// and `T: 'b` give.
        fn written<'a, 'b: 'a, T: 'b>(&self, x: Rel<'a, &'b T>) -> usize;
        /// A bound to a lifetime which is not a parameter is not encoded.
        fn forever<T: 'static>(&self, x: Rel<'static, T>) -> usize;
    }

    /// The same with the generics of the trait: they are the generics of the implementation as
    /// well, so `T: 'a` is written in the trait (E0309 in the implementation without it).
    #[leak(
        helper = crate::__macro,
        repeater = crate::Repeater,
        module = crate::m,
    )]
    pub trait Outlives<'a, T: 'a> {
        fn field(&self, x: Rel<'a, T>) -> usize;
        /// The generics of the trait and of the method together: `Rel<'a, (T, &'b u8, U)>`
        /// requires `T: 'a` (of the trait), `'b: 'a` and `U: 'a`, which the method writes as the
        /// bounds of its parameters.
        fn mixed<'b: 'a, U: 'a>(&self, x: Rel<'a, (T, &'b u8, U)>) -> usize;
    }

    /// The lifetime is a generic of the trait, and the type is a generic of the method: the
    /// Referrer sees only the latter declared where it copies the signature, and has to find
    /// `T: 'a` as well (`'a` is not declared there).
    #[leak(
        helper = crate::__macro,
        repeater = crate::Repeater,
        module = crate::m,
    )]
    pub trait Local<'a> {
        fn local<T: 'a>(&self, x: Rel<'a, T>) -> usize;
        /// The other way round: the lifetime of the argument is a generic of the method, bounded
        /// by the lifetime of the trait (`Rel<'a, &'b u8>` requires `'b: 'a`).
        fn lifetime<'b: 'a>(&self, x: Rel<'a, &'b u8>) -> usize;
    }

    /// References and paths with a lifetime around a relative type: a reference is leaked as a
    /// whole if its lifetime is a generic of the trait (`&'a Rel2<T>`, with `T: 'a`), and the
    /// relative type alone otherwise, as for a path (`::core::cell::Ref<'a, Rel2<T>>`). A
    /// reference requires `Rel2<T>: 'a`, so the type in it has `T: 'a` (`dyn_ref`), whereas a path
    /// does not require anything of it which the Leaker knows (`cell`: `Ref<'a, X>` does require
    /// `X: 'a` where it is written with a bound in its definition, but that is not assumed, see
    /// `Around`): `Rel2<T>` is leaked with the generics `<T>`, and the copy of a method signature
    /// is well-formed (the original's implied bound is implied there as well). The same type found
    /// without a type around it (`plain`) is the same entry as that of `cell`, and another entry
    /// than that of `dyn_ref`, which has the bound.
    #[leak(
        helper = crate::__macro,
        repeater = crate::Repeater,
        module = crate::m,
    )]
    pub trait Descend<'a, T> {
        /// Leaked as a whole, with `T: 'a`.
        fn by_ref(&self, x: &'a Rel2<T>) -> usize;
        /// `Rel2<T>` alone is leaked, with no bound: a path with a lifetime does not imply one.
        fn cell(&self, x: ::core::cell::Ref<'a, Rel2<T>>) -> usize;
        fn local_ref<'b>(&self, x: &'b Rel2<T>) -> usize;
        fn local_cell<'b>(&self, x: ::core::cell::Ref<'b, Rel2<T>>) -> usize;
        /// A trait object in a reference requires its arguments to outlive the reference as well
        /// (`dyn Iterator<Item = X> + 'a: 'a` requires `X: 'a`), so `Rel2<T>` is leaked alone, with
        /// `T: 'a` as well, whereas `&'a Rel2<T>` is leaked as a whole.
        fn dyn_ref(&self, x: &'a dyn ::core::iter::Iterator<Item = Rel2<T>>) -> usize;
        /// The same type with nothing around it: the entry of `cell`, and not that of `dyn_ref`,
        /// which has `T: 'a` as well (the Referrer tells them apart when the generics of the trait
        /// are not declared where it finds them).
        fn plain(&self, x: Rel2<T>) -> usize;
    }

    /// The bounds of the references around a type leaked alone are passed to it, as rustc
    /// derives them: `&'a X` is well-formed only if `X: 'a`, which the copy `&'a <S as
    /// Repeater<(T,), ..>>::Type` has only if the context of `X` carries it, as nothing infers `T:
    /// 'a` through a projection (E0309 in the field of a struct). So `Rel2<T>` is leaked with
    /// the generics `<'a, T: 'a>`, which the copy of the field encodes in the arguments of the
    /// Repeater (`(&'a (), T, &'a T)`).
    ///
    /// A path with a lifetime argument (`Ref<'a, X>`) does not pass one: it is well-formed only
    /// if `X: 'a` where its definition says so, which is not visible, so the bound is not
    /// assumed, and it is written where it is needed (`T: 'a` in the signature, `fn cell<'a, T:
    /// 'a>` here, and in the struct which holds the copy, see `user`), which makes it the
    /// generics of `Rel2<T>`. Without it the copy of the field of a struct is rejected with
    /// E0309, see `UnboundedAround`.
    ///
    /// The requirement of a reference is carried by tuples, arrays, slices, references, raw
    /// pointers, the type arguments of paths, the inputs and outputs of function pointers, and the
    /// arguments of a trait object (`dyn Tr<X> + 'b: 'a` requires `X: 'a`), but not by a
    /// projection (nothing infers `T: 'a` from `&'a <Rel2<T> as Deref>::Target` in the original
    /// either, and the same goes for a path starting with a type parameter, `&'a
    /// T::A<Rel2<U>>`), nor across the lifetime of a `for<..>` binder or `'static`, which are not
    /// generics. See `tests/outlives_ancestors.rs` for the contexts of each.
    #[leak(
        helper = crate::__macro,
        repeater = crate::Repeater,
        module = crate::m,
    )]
    pub trait Around {
        /// The bound is written, as `Ref<'a, Rel2<T>>` requires it (E0309 in the copy of the field
        /// of a struct without it, see `user`). So it is for the entries below which have a
        /// path with a lifetime and no reference around the type: `T: 'a` is part of the
        /// generics of `Rel2<T>`, which the struct writes the same way.
        fn cell<'a, T: 'a>(&self, x: ::core::cell::Ref<'a, Rel2<T>>) -> usize;
        fn by_ref<'a, T>(&self, x: &'a Rel2<T>) -> usize;
        fn by_mut<'a, T>(&self, x: &'a mut Rel2<T>) -> usize;
        /// Further paths: `Vec<X>` requires `X: 'a` as well.
        fn nested<'a, T: 'a>(&self, x: ::core::cell::Ref<'a, ::std::vec::Vec<Rel2<T>>>) -> usize;
        fn tuple<'a, T>(&self, x: &'a (u8, [Rel2<T>; 2])) -> usize;
        fn function<'a, T>(&self, x: &'a fn(Rel2<T>) -> usize) -> usize;
        /// A trait object: `Ref<'a, dyn Iterator<Item = X> + 'a>` requires `X: 'a`.
        fn object<'a, T: 'a>(
            &self,
            x: ::core::cell::Ref<'a, dyn ::core::iter::Iterator<Item = Rel2<T>>>,
        ) -> usize;
        /// The lifetimes are bounded too.
        fn lifetime<'a, 'c>(&self, x: &'a Rel2<&'c u8>) -> usize;
        /// Both lifetimes bound `T`.
        fn both<'a, 'b, T>(&self, x: &'a &'b Rel2<T>) -> usize;
        /// `'static` is written as well: rustc infers no `T: 'static` for a struct (RFC 2093), so
        /// the original declares it, and so does the copy of the field (the Repeater's
        /// implementation declares the bound too, which is a written bound).
        fn forever<T: 'static>(&self, x: ::core::cell::Ref<'static, Rel2<T>>) -> usize;
        /// Not for the lifetime of a binder.
        fn binder<T>(&self, f: for<'x> fn(&'x Rel2<T>) -> usize) -> usize;
        /// Not through a projection.
        fn projection<'a, T>(&self, x: &'a <Rel2<T> as ::core::ops::Deref>::Target) -> usize;
        /// With nothing around it.
        fn alone<T>(&self, x: Rel2<T>) -> usize;
        /// Not through a path starting with a type parameter (`T::A<X>`), which is a projection
        /// like `<T as Gat>::A<X>`: the original field of a struct requires no `U: 'a` for
        /// `Ref<'a, T::A<Rel2<U>>>` either, whereas the copy with `Rel2<U>: 'a` encoded would be
        /// stricter (a caller with a `U` shorter than `'a` is rejected), and a path with a
        /// lifetime does not imply one in any case.
        fn gat<'a, T: ::type_leak_test_outlives::Gat, U>(
            &self,
            x: ::core::cell::Ref<'a, T::A<Rel2<U>>>,
        ) -> usize;
        fn gat_nested<'a, T: ::type_leak_test_outlives::Gat, U>(
            &self,
            x: &'a ::std::vec::Vec<T::A<Rel2<U>>>,
        ) -> usize;
    }

    /// `Self: 'a` for a type leaked alone, like `T: 'a`: the reference implies it (`by_ref`, the
    /// bound is that of the reference around `Rel2<Self>`), and a path with a lifetime does not,
    /// so it is written (`cell`: `Ref<'a, Rel2<Self>>` requires it, E0309 in the copy of the
    /// field of a struct without it, see `UnboundedAround`).
    #[leak(
        helper = crate::__macro,
        repeater = crate::Repeater,
        module = crate::m,
    )]
    pub trait SelfAround {
        fn cell<'a>(&self, x: ::core::cell::Ref<'a, Rel2<Self>>) -> usize
        where
            Self: Sized + 'a;
        fn by_ref<'a>(&self, x: &'a Rel2<Self>) -> usize
        where
            Self: Sized;
    }

    /// Trait objects as arguments of a path with a lifetime: their lifetime bound defaults to
    /// that lifetime.
    #[leak(
        helper = crate::__macro,
        repeater = crate::Repeater,
        module = crate::m,
    )]
    pub trait Dyns<'a> {
        /// Only `Plain` is leaked: the copy keeps the trait object where it is, so it keeps the
        /// lifetime bound it defaults to.
        fn items(
            &self,
            x: ::core::cell::Ref<'a, dyn ::core::iter::Iterator<Item = Plain>>,
        ) -> usize;
        /// The trait object cannot be leaked alone (its trait is relative, and its lifetime bound
        /// would be `'static`), so the path is.
        fn whole(&self, x: ::core::cell::Ref<'a, dyn RelTr>) -> usize;
        /// The lifetime of a method: only `Plain` is leaked.
        fn local<'b>(
            &self,
            x: ::core::cell::Ref<'b, dyn ::core::iter::Iterator<Item = Plain>>,
        ) -> usize;
        /// A reference to a trait object with a lifetime of the trait, which the trait object
        /// defaults to: only `Plain` is leaked, and the copy keeps the reference and the trait
        /// object where they are.
        fn by_ref(&self, x: &'a dyn ::core::iter::Iterator<Item = Plain>) -> usize;
        #[allow(unused_parens)]
        fn by_paren(&self, x: &'a (dyn ::core::iter::Iterator<Item = Plain>)) -> usize;
        fn by_mut(&self, x: &'a mut dyn ::core::iter::Iterator<Item = Plain>) -> usize;
        /// The lifetime bound is written.
        fn bounded(&self, x: &'a (dyn ::core::iter::Iterator<Item = Plain> + 'static)) -> usize;
        fn forever(&self, x: &'static dyn ::core::iter::Iterator<Item = Plain>) -> usize;
        fn local_ref<'b>(&self, x: &'b dyn ::core::iter::Iterator<Item = Plain>) -> usize;
        /// The trait object has to be leaked (its trait is relative), and its lifetime bound is
        /// `'a`, or that of the trait if it has one (`Statics`), which the Leaker cannot see: it
        /// cannot be leaked alone, so the whole reference is.
        fn whole_ref(&self, x: &'a dyn RelTr) -> usize;
        fn whole_mut(&self, x: &'a mut dyn RelTr) -> usize;
        /// A bare identifier in the arguments may be a const, which cannot be leaked on its own, so
        /// the reference is leaked as a whole as well.
        fn bare_ref(&self, x: &'a dyn ::core::borrow::Borrow<Plain>) -> usize;
        /// With a reference to a trait object inside, which is transparent as well.
        fn nested(
            &self,
            x: &'a dyn ::core::iter::Iterator<Item = &'a dyn ::core::iter::Iterator<Item = Plain>>,
        ) -> usize;
    }

    /// A trait object borrowed through a lifetime of the method: the whole reference is leaked,
    /// `&'b dyn RelTr` with `'b` among its generics, so rustc gives the trait object its default
    /// where the Repeater is implemented (`&'b (dyn RelTr + 'b)`, or `'static` for a trait with a
    /// lifetime bound of its own, see `Statics`). The copy has `'b` only in a projection,
    /// `<Self as Repeater<(&'b (),), ..>>::Type`, which constrains nothing, so the copy of a method
    /// whose return type mentions `'b` is rejected where it is implemented (E0195: early-bound
    /// where the trait's is late-bound) unless another argument keeps `'b` as written (`&'b self`, a
    /// reference with it, also to an absolute trait object), a bound makes it early-bound in both,
    /// or no argument mentions it (it is early-bound in the original too): the others are an error
    /// (see the `check_leak!` below). A path `Ref<'b, dyn RelTr>` is the same.
    ///
    /// The trait is dyn-compatible, and called through `&dyn Locals` where it is implemented.
    #[leak(
        helper = crate::__macro,
        repeater = crate::Repeater,
        module = crate::m,
    )]
    pub trait Locals {
        fn by_ref<'b>(&self, x: &'b dyn RelTr) -> usize;
        fn by_mut<'b>(&self, x: &'b mut dyn RelTr) -> usize;
        /// The lifetime is in the return type too, so another argument keeps it as written
        /// (`y`, or the receiver).
        fn returned<'b>(&self, x: &'b dyn RelTr, y: &'b u8) -> &'b dyn RelTr;
        fn anchored<'b>(&'b self, x: &'b dyn RelTr) -> &'b dyn RelTr;
        /// A bound makes it early-bound in both.
        fn bounded<'b, 'c: 'b>(&self, x: &'b dyn RelTr, y: &'c u8) -> &'b u8;
        /// Several lifetimes, bounds and trait objects (`RelTr` once, for each lifetime).
        fn several<'b, 'c: 'b>(&self, x: &'b dyn RelTr, y: &'c dyn RelTr2) -> usize;
        fn clause<'b, 'c>(&self, x: &'b dyn RelTr, y: &'c dyn RelTr) -> usize
        where
            'c: 'b;
        /// The innermost reference is leaked.
        fn nested<'b, 'c>(&self, x: &'b &'c dyn RelTr) -> usize;
        fn cell<'b>(&self, x: ::core::cell::Ref<'b, dyn RelTr>) -> usize;
        /// Another argument keeps `'b` as written, or a bound makes it early-bound in both, so
        /// the return type may mention it.
        fn cell_anchored<'b>(&'b self, x: ::core::cell::Ref<'b, dyn RelTr>) -> &'b u8;
        fn cell_argument<'b>(&self, x: ::core::cell::Ref<'b, dyn RelTr>, y: &'b u8) -> &'b u8;
        fn cell_bound<'b, 'c: 'b>(&self, x: ::core::cell::Ref<'b, dyn RelTr>, y: &'c u8) -> &'b u8;
        /// No argument mentions `'b`: it is in the return type only, so it is early-bound in the
        /// original, and so it is in the copy, where it is only in a projection. Nothing needs
        /// to keep it, for a reference and for a path alike.
        fn unmentioned<'b>(&self) -> &'b dyn RelTr;
        fn unmentioned_cell<'b>(&self) -> ::core::cell::Ref<'b, dyn RelTr>;
        /// A reference to an absolute trait object is never leaked as a whole, so it stays in the
        /// copy and keeps `'b` as written, whether or not what is inside the trait object is
        /// leaked (`Plain` is, alone).
        fn kept_by_absolute<'b>(&self, x: &'b dyn RelTr, y: &'b dyn ::core::fmt::Debug) -> &'b u8;
        fn kept_by_component<'b>(
            &self,
            x: &'b dyn RelTr,
            y: &'b dyn ::core::iter::Iterator<Item = Plain>,
        ) -> &'b u8;
    }

    /// A trait with a lifetime bound of its own: the default of the lifetime bound of the trait
    /// object is `'static`, whatever the reference is. The whole reference is leaked, with its
    /// lifetime (of the trait, or of the method) among its generics, so rustc computes the default
    /// where the Repeater is implemented (`type Type = &'a dyn StaticTr;`). Leaking the trait
    /// object alone with the lifetime of the reference written out (`dyn StaticTr + 'a`) is
    /// rejected there (E0478: lifetime bound not satisfied).
    #[leak(
        helper = crate::__macro,
        repeater = crate::Repeater,
        module = crate::m,
    )]
    pub trait Statics<'a> {
        fn by_ref(&self, x: &'a dyn StaticTr) -> usize;
        fn by_mut(&self, x: &'a mut dyn StaticTr) -> usize;
        /// A lifetime of the method, which the receiver keeps as written in the copy.
        fn local<'b>(&'b self, x: &'b dyn StaticTr) -> &'b u8;
        fn local_mut<'b>(&'b self, x: &'b mut dyn StaticTr) -> usize;
        fn returned(&self, x: &'a dyn StaticTr) -> &'a dyn StaticTr;
    }

    /// The default object lifetime of `R<'a, dyn Tr>`, a trait object without a lifetime bound as
    /// the argument of a path with a lifetime, is `'a` only if the definition of `R` WRITES the
    /// bound `T: 'a`. An outlives bound which is only inferred (RFC 2093), as `Inferred` has from its
    /// `&'a T` field, does not count, and with no bound at all it is `'static`. Only the definition
    /// of the path tells which, so a relative trait object like this one has to be leaked with the
    /// whole path (see `ObjectDefaults`), and rustc reads the real definition in the Repeater's
    /// implementation. Each function compiles only if the type is identical to the other (a
    /// return type is not widened by subtyping the way an argument is).
    pub fn explicit_default<'a>(x: Explicit<'a, dyn RelTr>) -> Explicit<'a, dyn RelTr + 'a> {
        x
    }

    /// `Explicit<'a, T: 'a>` writes the bound, so the default is `'a`, and not `'static`: the
    /// reverse of [`explicit_default()`] is the one which a default of `'static` would reject.
    pub fn explicit_default_back<'a>(x: Explicit<'a, dyn RelTr + 'a>) -> Explicit<'a, dyn RelTr> {
        x
    }

    /// `Inferred<'a, T>` has `T: 'a` only inferred, so the default is `'static`, and not `'a` (this
    /// function would be rejected if it were: `dyn RelTr + 'a` does not outlive `'static`).
    pub fn inferred_default<'a>(x: Inferred<'a, dyn RelTr>) -> Inferred<'a, dyn RelTr + 'static> {
        x
    }

    /// The reverse of [`inferred_default()`], which is valid for a default of `'a` as well, as
    /// `Inferred` is covariant in `T`.
    pub fn inferred_default_back<'a>(
        x: Inferred<'a, dyn RelTr + 'static>,
    ) -> Inferred<'a, dyn RelTr> {
        x
    }

    /// `Unrelated<'a, T>` has no bound between `T` and `'a`, so the default is `'static`.
    pub fn unrelated_default<'a>(
        x: Unrelated<'a, dyn RelTr>,
    ) -> Unrelated<'a, dyn RelTr + 'static> {
        x
    }

    /// The reverse of [`unrelated_default()`], valid whatever the default is (covariance).
    pub fn unrelated_default_back<'a>(
        x: Unrelated<'a, dyn RelTr + 'static>,
    ) -> Unrelated<'a, dyn RelTr> {
        x
    }

    /// A trait object without a lifetime bound as the argument of a path with a lifetime: its
    /// default object lifetime comes from the definition of the path (see `explicit_default()`),
    /// which the Leaker cannot see. `Explicit<'a, dyn RelTr>` is `Explicit<'a, dyn RelTr + 'a>`,
    /// whereas `Inferred<'a, dyn RelTr>` and `Unrelated<'a, dyn RelTr>` are `+ 'static`. The trait
    /// object is relative, so it has to be leaked, and the whole path is (`Explicit<'a, dyn RelTr>`
    /// with `'a` among its generics, and `Inferred<'b, dyn RelTr>` for the lifetime of `local`):
    /// rustc reads the real definition of the path where the Repeater is implemented, and the copy
    /// `<Self as Repeater<(&'a (),), ..>>::Type` is the same type as the original.
    ///
    /// The return types are the ones which matter: an argument of the copy may be wider than the
    /// original's (`+ 'a` where the trait has `+ 'static` is accepted for the caller's `'static`,
    /// by covariance), but a return type may not, so a wrong object lifetime in the copy is
    /// rejected there. Leaking `dyn RelTr + 'a` alone, the alternative written out in
    /// `user::alone` (disabled), is rejected for `Inferred` and `Unrelated`, whose real default is
    /// `'static`: the copy returns `Inferred<'a, dyn RelTr + 'a>` where the trait returns
    /// `Inferred<'a, dyn RelTr + 'static>` (E0308 on stable, E0478 on nightly). It is right for
    /// `Explicit` only, so the trait object is never leaked alone.
    #[leak(
        helper = crate::__macro,
        repeater = crate::Repeater,
        module = crate::m,
    )]
    pub trait ObjectDefaults<'a> {
        fn explicit(&self, x: Explicit<'a, dyn RelTr>) -> Explicit<'a, dyn RelTr>;
        fn inferred(&self, x: Inferred<'a, dyn RelTr>) -> Inferred<'a, dyn RelTr>;
        fn unrelated(&self, x: Unrelated<'a, dyn RelTr>) -> Unrelated<'a, dyn RelTr>;
        /// A lifetime of the method: `&'b self` keeps `'b` as written in the copy, where the path
        /// has it only in a projection, so the return type may mention it.
        fn local<'b>(&'b self, x: Inferred<'b, dyn RelTr>) -> Inferred<'b, dyn RelTr>;
    }

    /// The same for the lifetime of a `for<..>` binder. If the output of the binder does not
    /// mention it, the reference is leaked as a whole, `&'x dyn RelTr` with `'x` among its
    /// generics, and the copy `for<'x> fn(<Self as Repeater<(&'x (),), ..>>::Type) -> usize`
    /// compiles. If it does, the reference is no anchor (`'x` is only in a projection of the copy,
    /// as in the original of a method, E0581 or E0582), so the function pointer or the trait
    /// object holding the binder is leaked as a whole instead, and a bound or a where predicate
    /// holding it is an error (see the `check_leak!` below). A path whose output does not mention
    /// it is leaked as a whole, as before.
    #[leak(
        helper = crate::__macro,
        repeater = crate::Repeater,
        module = crate::m,
    )]
    pub trait LocalBinders {
        fn pointer(&self, f: for<'x> fn(&'x dyn RelTr) -> usize) -> usize;
        fn bound<F: for<'x> Fn(&'x dyn RelTr) -> usize>(&self, f: F) -> usize;
        fn predicate<F>(&self, f: F) -> usize
        where
            for<'x> F: Fn(&'x dyn RelTr) -> usize;
        /// Another input keeps `'x` as written, so the reference is leaked alone.
        fn anchored_pointer(&self, f: for<'x> fn(&'x dyn RelTr, &'x u8) -> &'x u8) -> usize;
        fn anchored_bound<F: for<'x> Fn(&'x dyn RelTr, &'x u8) -> &'x u8>(&self, f: F) -> usize;
        fn lost_pointer(&self, f: for<'x> fn(&'x dyn RelTr) -> &'x u8) -> usize;
        fn lost_dynamic(&self, f: Box<dyn for<'x> Fn(&'x dyn RelTr) -> &'x u8>) -> usize;
        fn path(&self, f: for<'x> fn(::core::cell::Ref<'x, dyn RelTr>) -> usize) -> usize;
        fn path_bound<F: for<'x> Fn(::core::cell::Ref<'x, dyn RelTr>) -> usize>(
            &self,
            f: F,
        ) -> usize;
    }

    /// The lifetime `'x` of a `for<..>` binder. If the output of the binder (the return type, or an
    /// associated type binding) does not mention it, `Hr<'x>` is leaked with `'x` among its
    /// generics, and the copy `for<'x> fn(<Self as Repeater<(&'x (),), ..>>::Type) -> usize`
    /// compiles. If it does, the copy would not (E0581 for a function pointer and E0582 for a
    /// bound or a trait object: a lifetime which occurs only in a projection is not constrained by
    /// the inputs), unless an input keeps it where it is written (`&'x u8`: an anchor, also in a
    /// trait object's associated type binding, lifetime bound or `Fn(..)`, and in a function
    /// pointer), so `Hr<'x>` is leaked alone again. Otherwise the function pointer or the trait
    /// object holding the binder is leaked as a whole, and `Hr<'x>` is not.
    #[leak(
        helper = crate::__macro,
        repeater = crate::Repeater,
        module = crate::m,
    )]
    pub trait Binders {
        fn pointer(&self, f: for<'x> fn(Hr<'x>) -> usize) -> usize;
        fn bound<F: for<'x> Fn(Hr<'x>) -> usize>(&self, f: F) -> usize;
        fn predicate<F>(&self, f: F) -> usize
        where
            for<'x> F: Fn(Hr<'x>) -> usize;
        fn dynamic(&self, f: Box<dyn for<'x> Fn(Hr<'x>) -> usize>) -> usize;
        /// The return type mentions `'x` and no input keeps it: the function pointer is leaked as
        /// a whole.
        fn whole(&self, f: for<'x> fn(Hr<'x>) -> &'x u8) -> usize;
        /// `Hr<'x>` is in the return type, and `&'x u8` keeps `'x`: only `Hr<'x>` is leaked.
        fn returning(&self, f: for<'x> fn(&'x u8) -> Hr<'x>) -> usize;
        /// The same for a trait object, which is leaked as a whole.
        fn object(&self, f: Box<dyn for<'x> Fn(Hr<'x>) -> &'x u8>) -> usize;
        /// And for an associated type binding (`'x` is in the inputs of the trait, as it must be in
        /// the original as well, and it is kept by `&'x u8`).
        fn binding(&self, f: Box<dyn for<'x> ::core::ops::Index<&'x u8, Output = Hr<'x>>>)
            -> usize;
        /// `&'x u8` keeps `'x`, whatever `Hr<'x>` is replaced by.
        fn anchored<F: for<'x> Fn(&'x u8, Hr<'x>) -> &'x u8>(&self, f: F) -> usize;
        fn anchored_after<F: for<'x> Fn(Hr<'x>, &'x u8) -> &'x u8>(&self, f: F) -> usize;
        fn anchored_predicate<F>(&self, f: F) -> usize
        where
            for<'x> F: Fn(&'x u8, Hr<'x>) -> &'x u8;
        fn anchored_pointer(&self, f: for<'x> fn(&'x u8, Hr<'x>) -> &'x u8) -> usize;
        fn anchored_object(&self, f: Box<dyn for<'x> Fn(&'x u8, Hr<'x>) -> &'x u8>) -> usize;
        fn anchored_impl(&self, f: impl for<'x> Fn(&'x u8, Hr<'x>) -> &'x u8) -> usize;
        /// The anchor is in a tuple (which mentions `'x`, so it is never leaked, however many
        /// times it is used): the copy keeps `&'x u8` in it.
        fn tuple<F: for<'x> Fn((&'x u8, Hr<'x>, Plain)) -> &'x u8>(&self, f: F) -> usize;
        fn tuple_again<F: for<'x> Fn((&'x u8, Hr<'x>, Plain)) -> &'x u8>(&self, f: F) -> usize;
        /// `Rel2` is leaked as a whole, with the `&'x u8` in it: not an anchor, so the pointer is
        /// leaked as a whole.
        fn lost(&self, f: for<'x> fn(Rel2<&'x u8>) -> &'x u8) -> usize;
        /// Another input keeps `'x`, so `Rel2<&'x u8>` is leaked alone.
        fn kept(&self, f: for<'x> fn(Rel2<&'x u8>, &'x u8) -> &'x u8) -> usize;
        /// An associated type binding of a trait object in an input keeps `'x` as well (rustc
        /// counts it as constraining), in a bound, a predicate, a function pointer, a trait object
        /// and an `impl Trait`, so only `Hr<'x>` is leaked.
        fn binding_bound<F: for<'x> Fn(Box<dyn Iterator<Item = &'x u8>>, Hr<'x>) -> &'x u8>(
            &self,
            f: F,
        ) -> usize;
        fn binding_predicate<F>(&self, f: F) -> usize
        where
            for<'x> F: Fn(Box<dyn Iterator<Item = &'x u8>>, Hr<'x>) -> &'x u8;
        fn binding_pointer(
            &self,
            f: for<'x> fn(Box<dyn Iterator<Item = &'x u8>>, Hr<'x>) -> &'x u8,
        ) -> usize;
        fn binding_object(
            &self,
            f: Box<dyn for<'x> Fn(Box<dyn Iterator<Item = &'x u8>>, Hr<'x>) -> &'x u8>,
        ) -> usize;
        fn binding_impl(
            &self,
            f: impl for<'x> Fn(Box<dyn Iterator<Item = &'x u8>>, Hr<'x>) -> &'x u8,
        ) -> usize;
        /// The lifetime bound of a trait object.
        fn lifetime_bound<F: for<'x> Fn(Box<dyn ::core::fmt::Debug + 'x>, Hr<'x>) -> &'x u8>(
            &self,
            f: F,
        ) -> usize;
        /// The arguments and the output of the `Fn` trait object (`'x` is the outer binder's).
        fn nested_args<F: for<'x> Fn(Box<dyn Fn(&'x u8)>, Hr<'x>) -> &'x u8>(&self, f: F) -> usize;
        fn nested_output<F: for<'x> Fn(Box<dyn Fn() -> &'x u8>, Hr<'x>) -> &'x u8>(
            &self,
            f: F,
        ) -> usize;
        /// A trait object, or a function pointer, with a binder of its own.
        fn nested_binder<F: for<'x> Fn(Box<dyn for<'y> Fn(&'x u8, &'y u8)>, Hr<'x>) -> &'x u8>(
            &self,
            f: F,
        ) -> usize;
        fn nested_pointer<F: for<'x> Fn(fn(&'x u8), Hr<'x>) -> &'x u8>(&self, f: F) -> usize;
        fn nested_pointer_output<F: for<'x> Fn(fn() -> &'x u8, Hr<'x>) -> &'x u8>(
            &self,
            f: F,
        ) -> usize;
        fn nested_pointer_binder<F: for<'x> Fn(for<'y> fn(&'x u8, &'y u8), Hr<'x>) -> &'x u8>(
            &self,
            f: F,
        ) -> usize;
        /// The trait object holding the binding is leaked as a whole (`RelIt` is relative), so
        /// `'x` is only in a projection of the copy: it is no anchor, and the function pointer is
        /// leaked as a whole, like for the lifetime bound of `RelTr`.
        fn lost_binding(
            &self,
            f: for<'x> fn(Box<dyn RelIt<Item = &'x u8>>, Hr<'x>) -> &'x u8,
        ) -> usize;
        fn lost_bound(&self, f: for<'x> fn(Box<dyn RelTr + 'x>, Hr<'x>) -> &'x u8) -> usize;
        /// Another trait object keeps `'x`: `dyn RelIt<Item = &'x u8>` is leaked alone, with `'x`.
        fn kept_binding(
            &self,
            f: for<'x> fn(
                Box<dyn RelIt<Item = &'x u8>>,
                Box<dyn Iterator<Item = &'x u8>>,
            ) -> &'x u8,
        ) -> usize;
        /// An anchor in a path of `::core` stays where it is written, whatever the Referrer
        /// does, so only `Hr<'x>` is leaked.
        fn core_bound<F: for<'x> Fn(::core::option::Option<&'x u8>, Hr<'x>) -> &'x u8>(
            &self,
            f: F,
        ) -> usize;
        fn core_pointer(&self, f: for<'x> fn(::core::cell::Ref<'x, u8>, Hr<'x>) -> &'x u8)
            -> usize;
        fn core_object(
            &self,
            f: Box<dyn for<'x> Fn(::core::option::Option<(u8, &'x u8)>, Hr<'x>) -> &'x u8>,
        ) -> usize;
        /// The only anchor is in the type of an external crate (`::std::boxed::Box`, as `::std` is
        /// not allowed by this Leaker), which the Referrer may replace as a whole, and the Leaker
        /// cannot tell whether it does (old: it anchored `'x`, and `Hr<'x>` was leaked alone). So
        /// it is no anchor, and the function pointer or the trait object is leaked as a whole
        /// instead of `Hr<'x>` (a bound would be an error, see the `check_leak!` below), whichever
        /// the Referrer does.
        fn external_pointer(
            &self,
            f: for<'x> fn(::std::boxed::Box<dyn Iterator<Item = &'x u8>>, Hr<'x>) -> &'x u8,
        ) -> usize;
        fn external_object(
            &self,
            f: Box<
                dyn for<'x> Fn(::std::boxed::Box<dyn Iterator<Item = &'x u8>>, Hr<'x>) -> &'x u8,
            >,
        ) -> usize;
    }

    /// `Self: 'a` is implied by a reference: `&'a Rel2<Self>` and `&'a Rel2<Self::Item>` are
    /// well-formed only if `Self: 'a` (a language rule, which holds whatever `Rel2` is). The
    /// Repeater's implementation is `impl<'a, __S: ?Sized + SelfOutlives> Repeater<(&'a (), &'a
    /// __S), ..> for __S where __S: Sized`, where the header implies `__S: 'a`, which it does not
    /// declare, and so do the copies, `<Self as Repeater<(&'a (), &'a Self), ..>>::Type`, in a
    /// method (where `&'a self` implies it as well), a free function and the field of a struct.
    /// A path with a lifetime (`Rel<'a, Self>`) implies nothing which the Leaker knows, so the
    /// bound is written then, see `SelfBound`.
    #[leak(
        helper = crate::__macro,
        repeater = crate::Repeater,
        module = crate::m,
    )]
    pub trait SelfOutlives {
        type Item: Sized;
        fn me<'a>(&'a self, x: &'a Rel2<Self>) -> usize
        where
            Self: Sized;
        fn it<'a>(&'a self, x: &'a Rel2<Self::Item>) -> usize
        where
            Self: Sized;
        /// Without a receiver of the lifetime, so the bound is implied by the encoding only.
        fn arg<'a>(&self, x: &'a Rel2<Self>) -> usize
        where
            Self: Sized;
    }

    /// `Self: 'a` written in the where clause, as `Rel<'a, Self>` requires it (by its definition,
    /// which is not visible, so the bound is not assumed): the context is the same as that of a
    /// bound which a reference implies (see `SelfOutlives`, and `implied`), as the written bound is
    /// encoded like the implied one (`&'a __S`) and not declared: the Repeater's implementation is
    /// `impl<'a, __S: ?Sized + SelfBound> Repeater<(&'a (), &'a __S), ..> for __S where __S:
    /// Sized`, with the `Sized` of the same predicate (`Self: Sized + 'a`) kept. So the copies
    /// compile where nothing proves `Self: 'a` but the header, the free function and the field of
    /// a struct (which write the bound as well, `where Self: 'a`, for the copy to find the entry).
    /// Without it the implementation is rejected with E0309 (`__S` may not live long enough), see
    /// `UnboundedSelf`.
    #[leak(
        helper = crate::__macro,
        repeater = crate::Repeater,
        module = crate::m,
    )]
    pub trait SelfBound {
        type Item: Sized;
        fn me<'a>(&'a self) -> Rel<'a, Self>
        where
            Self: Sized + 'a;
        fn it<'a>(&'a self) -> Rel<'a, Self::Item>
        where
            Self: Sized + 'a;
        /// The bounds in separate predicates, and without a receiver of the lifetime.
        fn split<'a>(&self, x: Rel<'a, Self>) -> usize
        where
            Self: Sized,
            Self: 'a;
        /// A reference implies the bound which the others write (`&'a Rel<'a, Self>` requires
        /// `Rel<'a, Self>: 'a`, so `Self: 'a`), so this is the same context as `me`, whichever way
        /// it is written.
        fn implied<'a>(&self, x: &'a Rel<'a, Self>) -> usize
        where
            Self: Sized;
    }

    /// The same bound in the where clause of the trait, where the copy of the signature of a
    /// method does not repeat it.
    #[leak(
        helper = crate::__macro,
        repeater = crate::Repeater,
        module = crate::m,
    )]
    pub trait SelfBoundTrait<'a>
    where
        Self: Sized + 'a,
    {
        fn me(&'a self) -> Rel<'a, Self>;
        fn arg(&self, x: Rel<'a, Self>) -> usize;
    }

    // The leaked types, checked when this crate is compiled.
    check_leak! {
        { Hr<'x> }
        { }
        trait Inputs {
            fn pointer(&self, f: for<'x> fn(Hr<'x>) -> usize) -> usize;
            fn bound<F: for<'x> ::core::ops::Fn(Hr<'x>) -> usize>(&self, f: F) -> usize;
        }
    }
    check_leak! {
        {
            for<'x> fn(Hr<'x>) -> &'x u8,
            dyn for<'x> ::core::ops::Fn(Hr<'x>) -> &'x u8
        }
        { allow = ::std }
        trait Outputs {
            fn whole(&self, f: for<'x> fn(Hr<'x>) -> &'x u8) -> usize;
            fn object(&self, f: ::std::boxed::Box<dyn for<'x> ::core::ops::Fn(Hr<'x>) -> &'x u8>) -> usize;
        }
    }
    // An input keeps `'x` (an anchor), so only `Hr<'x>` is leaked.
    check_leak! {
        { Hr<'x> }
        { allow = ::std }
        trait Anchored {
            fn returning(&self, f: for<'x> fn(&'x u8) -> Hr<'x>) -> usize;
            fn binding(&self, f: ::std::boxed::Box<dyn for<'x> ::core::ops::Index<&'x u8, Output = Hr<'x>>>) -> usize;
            fn bound<F: for<'x> ::core::ops::Fn(&'x u8, Hr<'x>) -> &'x u8>(&self, f: F) -> usize;
            fn after<F: for<'x> ::core::ops::Fn(Hr<'x>, &'x u8) -> &'x u8>(&self, f: F) -> usize;
            fn predicate<F>(&self, f: F) -> usize where for<'x> F: ::core::ops::Fn(&'x u8, Hr<'x>) -> &'x u8;
            fn pointer(&self, f: for<'x> fn(&'x u8, Hr<'x>) -> &'x u8) -> usize;
            fn object(&self, f: ::std::boxed::Box<dyn for<'x> ::core::ops::Fn(&'x u8, Hr<'x>) -> &'x u8>) -> usize;
            fn imp(&self, f: impl for<'x> ::core::ops::Fn(&'x u8, Hr<'x>) -> &'x u8) -> usize;
            fn tuple<F: for<'x> ::core::ops::Fn((&'x u8, Hr<'x>)) -> &'x u8>(&self, f: F) -> usize;
            fn tuple_again<F: for<'x> ::core::ops::Fn((&'x u8, Hr<'x>)) -> &'x u8>(&self, f: F) -> usize;
        }
    }
    // What is in a trait object or a function pointer in an input keeps `'x` too (an associated
    // type binding, the lifetime bound, the output of `Fn(..)`, a binder of its own), so only
    // `Hr<'x>` is leaked.
    check_leak! {
        { Hr<'x> }
        { allow = ::std }
        trait AnchoredInside {
            fn binding<F: for<'x> ::core::ops::Fn(::std::boxed::Box<dyn ::core::iter::Iterator<Item = &'x u8>>, Hr<'x>) -> &'x u8>(&self, f: F) -> usize;
            fn predicate<F>(&self, f: F) -> usize where for<'x> F: ::core::ops::Fn(::std::boxed::Box<dyn ::core::iter::Iterator<Item = &'x u8>>, Hr<'x>) -> &'x u8;
            fn pointer(&self, f: for<'x> fn(::std::boxed::Box<dyn ::core::iter::Iterator<Item = &'x u8>>, Hr<'x>) -> &'x u8) -> usize;
            fn object(&self, f: ::std::boxed::Box<dyn for<'x> ::core::ops::Fn(::std::boxed::Box<dyn ::core::iter::Iterator<Item = &'x u8>>, Hr<'x>) -> &'x u8>) -> usize;
            fn imp(&self, f: impl for<'x> ::core::ops::Fn(::std::boxed::Box<dyn ::core::iter::Iterator<Item = &'x u8>>, Hr<'x>) -> &'x u8) -> usize;
            fn bound<F: for<'x> ::core::ops::Fn(::std::boxed::Box<dyn ::core::fmt::Debug + 'x>, Hr<'x>) -> &'x u8>(&self, f: F) -> usize;
            fn args<F: for<'x> ::core::ops::Fn(::std::boxed::Box<dyn ::core::ops::Fn(&'x u8)>, Hr<'x>) -> &'x u8>(&self, f: F) -> usize;
            fn output<F: for<'x> ::core::ops::Fn(::std::boxed::Box<dyn ::core::ops::Fn() -> &'x u8>, Hr<'x>) -> &'x u8>(&self, f: F) -> usize;
            fn binder<F: for<'x> ::core::ops::Fn(::std::boxed::Box<dyn for<'y> ::core::ops::Fn(&'x u8, &'y u8)>, Hr<'x>) -> &'x u8>(&self, f: F) -> usize;
            fn fn_pointer<F: for<'x> ::core::ops::Fn(fn(&'x u8), Hr<'x>) -> &'x u8>(&self, f: F) -> usize;
            fn fn_pointer_output<F: for<'x> ::core::ops::Fn(fn() -> &'x u8, Hr<'x>) -> &'x u8>(&self, f: F) -> usize;
            fn fn_pointer_binder<F: for<'x> ::core::ops::Fn(for<'y> fn(&'x u8, &'y u8), Hr<'x>) -> &'x u8>(&self, f: F) -> usize;
        }
    }
    // An anchor in a path of `::core` stays where it is written, so only `Hr<'x>` is leaked (the
    // optional entry is the `Box` of an external crate around the trait object).
    check_leak! {
        { Hr<'x>, ? ::std::boxed::Box<dyn for<'x> ::core::ops::Fn(::core::option::Option<(u8, &'x u8)>, Hr<'x>) -> &'x u8> }
        { }
        trait CoreAnchored {
            fn bound<F: for<'x> ::core::ops::Fn(::core::option::Option<&'x u8>, Hr<'x>) -> &'x u8>(&self, f: F) -> usize;
            fn predicate<F>(&self, f: F) -> usize where for<'x> F: ::core::ops::Fn(::core::cell::Ref<'x, u8>, Hr<'x>) -> &'x u8;
            fn pointer(&self, f: for<'x> fn(::core::cell::Ref<'x, u8>, Hr<'x>) -> &'x u8) -> usize;
            fn object(&self, f: ::std::boxed::Box<dyn for<'x> ::core::ops::Fn(::core::option::Option<(u8, &'x u8)>, Hr<'x>) -> &'x u8>) -> usize;
        }
    }
    // The only anchor is in the type of an external crate, which the Referrer may replace as a
    // whole, so it is no anchor (old: `Hr<'x>` alone, see `AnchoredInside`, where `::std` is
    // allowed): the function pointer or the trait object is leaked as a whole, and a bound is an
    // error. The optional entries are the types of `::std` which do not use `'x` themselves, or
    // which hold the binder.
    check_leak! {
        {
            for<'x> fn(::std::boxed::Box<dyn ::core::iter::Iterator<Item = &'x u8>>, Hr<'x>) -> &'x u8,
            dyn for<'x> ::core::ops::Fn(::std::boxed::Box<dyn ::core::iter::Iterator<Item = &'x u8>>, Hr<'x>) -> &'x u8,
            ? ::std::boxed::Box<dyn for<'x> ::core::ops::Fn(::std::boxed::Box<dyn ::core::iter::Iterator<Item = &'x u8>>, Hr<'x>) -> &'x u8>
        }
        { }
        trait ExternalAnchor {
            fn pointer(&self, f: for<'x> fn(::std::boxed::Box<dyn ::core::iter::Iterator<Item = &'x u8>>, Hr<'x>) -> &'x u8) -> usize;
            fn object(&self, f: ::std::boxed::Box<dyn for<'x> ::core::ops::Fn(::std::boxed::Box<dyn ::core::iter::Iterator<Item = &'x u8>>, Hr<'x>) -> &'x u8>) -> usize;
        }
    }
    check_leak! {
        { error: "use absolute path: 'Hr < 'x >'" }
        { }
        trait ExternalAnchorBound {
            fn bound<F: for<'x> ::core::ops::Fn(::std::boxed::Box<dyn ::core::iter::Iterator<Item = &'x u8>>, Hr<'x>) -> &'x u8>(&self, f: F) -> usize;
        }
    }
    // The type of an external crate does not anchor `'x`, but it is not leaked with it either
    // (`'x` would be only in a projection): the function pointer holding the binder is leaked
    // instead, as an optional entry, and a bound, which is not a type, is copied as it is (no error).
    check_leak! {
        { ? for<'x> fn(::std::boxed::Box<dyn ::core::iter::Iterator<Item = &'x u8>>) -> &'x u8 }
        { }
        trait ExternalOnly {
            fn pointer(&self, f: for<'x> fn(::std::boxed::Box<dyn ::core::iter::Iterator<Item = &'x u8>>) -> &'x u8) -> usize;
            fn bound<F: for<'x> ::core::ops::Fn(::std::boxed::Box<dyn ::core::iter::Iterator<Item = &'x u8>>) -> &'x u8>(&self, f: F) -> usize;
        }
    }
    // The trait object holding the anchor is a cause itself (`RelIt`, `RelTr`), so it is leaked as a
    // whole, and the function pointer holding it in its place of `Hr<'x>`.
    check_leak! {
        {
            for<'x> fn(::std::boxed::Box<dyn RelIt<Item = &'x u8>>, Hr<'x>) -> &'x u8,
            for<'x> fn(::std::boxed::Box<dyn RelTr + 'x>, Hr<'x>) -> &'x u8
        }
        { allow = ::std }
        trait LostInside {
            fn binding(&self, f: for<'x> fn(::std::boxed::Box<dyn RelIt<Item = &'x u8>>, Hr<'x>) -> &'x u8) -> usize;
            fn bound(&self, f: for<'x> fn(::std::boxed::Box<dyn RelTr + 'x>, Hr<'x>) -> &'x u8) -> usize;
        }
    }
    // Another trait object keeps it: the one which is a cause is leaked alone, with `'x`.
    check_leak! {
        { dyn RelIt<Item = &'x u8> }
        { allow = ::std }
        trait KeptInside {
            fn binding(&self, f: for<'x> fn(::std::boxed::Box<dyn RelIt<Item = &'x u8>>, ::std::boxed::Box<dyn ::core::iter::Iterator<Item = &'x u8>>) -> &'x u8) -> usize;
        }
    }
    // The anchor is in a type leaked as a whole (`Rel2`), so it is not one.
    check_leak! {
        { for<'x> fn(Rel2<&'x u8>) -> &'x u8, Rel2<&'x u8> }
        { }
        trait Lost {
            fn lost(&self, f: for<'x> fn(Rel2<&'x u8>) -> &'x u8) -> usize;
            fn kept(&self, f: for<'x> fn(Rel2<&'x u8>, &'x u8) -> &'x u8) -> usize;
        }
    }
    check_leak! {
        { error: "use absolute path: 'Rel2 < & 'x u8 >'" }
        { }
        trait LostBound {
            fn f<F: for<'x> ::core::ops::Fn(Rel2<&'x u8>) -> &'x u8>(&self, f: F) -> usize;
        }
    }
    // A bound, which cannot be leaked as a whole, is an error, as the copy of `Hr<'x>` would not
    // compile.
    check_leak! {
        { error: "use absolute path: 'Hr < 'x >'" }
        { }
        trait BoundOutput {
            fn f<F: for<'x> ::core::ops::Fn(Hr<'x>) -> &'x u8>(&self, f: F) -> usize;
        }
    }
    // A function pointer with an elided lifetime is leaked as a whole: the elided lifetime is its
    // own, and the impl of the Repeater writes it as the Leaker does.
    check_leak! {
        { for<'x> fn(Hr<'x>, &u8) -> &'x u8 }
        { }
        trait ElidedPointer {
            fn f(&self, f: for<'x> fn(Hr<'x>, &u8) -> &'x u8) -> usize;
        }
    }
    check_leak! {
        { Plain }
        { }
        trait Items<'a> {
            fn items(&self, x: ::core::cell::Ref<'a, dyn ::core::iter::Iterator<Item = Plain>>) -> usize;
        }
    }
    check_leak! {
        { Plain }
        { }
        trait Items {
            fn local<'b>(&self, x: ::core::cell::Ref<'b, (dyn ::core::iter::Iterator<Item = Plain>)>) -> usize;
        }
    }
    check_leak! {
        { ::core::cell::Ref<'a, dyn RelTr> }
        { }
        trait Whole<'a> {
            fn whole(&self, x: ::core::cell::Ref<'a, dyn RelTr>) -> usize;
        }
    }
    check_leak! {
        { Plain }
        { }
        trait RefItems<'a> {
            fn by_ref(&self, x: &'a dyn ::core::iter::Iterator<Item = Plain>) -> usize;
            #[allow(unused_parens)]
            fn by_paren(&self, x: &'a (dyn ::core::iter::Iterator<Item = Plain>)) -> usize;
            fn by_mut(&self, x: &'a mut dyn ::core::iter::Iterator<Item = Plain>) -> usize;
            fn bounded(&self, x: &'a (dyn ::core::iter::Iterator<Item = Plain> + 'static)) -> usize;
            fn forever(&self, x: &'static dyn ::core::iter::Iterator<Item = Plain>) -> usize;
        }
    }
    check_leak! {
        { Plain }
        { }
        trait RefItems {
            fn local_ref<'b>(&self, x: &'b dyn ::core::iter::Iterator<Item = Plain>) -> usize;
        }
    }
    // The trait object has to be leaked, and its default is the lifetime of the reference unless
    // the trait has a bound of its own (`Statics`), which the Leaker cannot see, so the whole
    // reference is leaked.
    check_leak! {
        { &'a dyn RelTr, &'a mut dyn RelTr }
        { }
        trait RefWhole<'a> {
            fn whole_ref(&self, x: &'a dyn RelTr) -> usize;
            fn whole_mut(&self, x: &'a mut dyn RelTr) -> usize;
        }
    }
    check_leak! {
        { &'a dyn ::core::borrow::Borrow<Plain> }
        { }
        trait RefBare<'a> {
            fn bare_ref(&self, x: &'a dyn ::core::borrow::Borrow<Plain>) -> usize;
        }
    }
    // With a lifetime bound written the reference is the type which carries the bound.
    check_leak! {
        { &'a (dyn RelTr + 'static) }
        { }
        trait RefBounded<'a> {
            fn bounded(&self, x: &'a (dyn RelTr + 'static)) -> usize;
        }
    }
    check_leak! {
        { &'a dyn StaticTr, &'a mut dyn StaticTr }
        { }
        trait RefStatic<'a> {
            fn by_ref(&self, x: &'a dyn StaticTr) -> usize;
            fn by_mut(&self, x: &'a mut dyn StaticTr) -> usize;
        }
    }
    // The same for a lifetime of the method, and for the one of a binder: the whole reference,
    // with the lifetime among its generics.
    check_leak! {
        { &'b (dyn RelTr), &'b dyn RelTr, &'b mut dyn RelTr, &'c dyn RelTr, &'c dyn RelTr2 }
        { }
        trait LocalRefs {
            fn by_ref<'b>(&self, x: &'b dyn RelTr) -> usize;
            fn by_mut<'b>(&self, x: &'b mut dyn RelTr) -> usize;
            #[allow(unused_parens)]
            fn by_paren<'b>(&self, x: &'b (dyn RelTr)) -> usize;
            fn returned<'b>(&'b self, x: &'b dyn RelTr) -> &'b dyn RelTr;
            fn several<'b, 'c: 'b>(&self, x: &'b dyn RelTr, y: &'c dyn RelTr2) -> usize;
            fn nested<'b, 'c>(&self, x: &'b &'c dyn RelTr) -> usize;
        }
    }
    check_leak! {
        { &'b dyn StaticTr, &'b mut dyn StaticTr }
        { }
        trait LocalStatics {
            fn by_ref<'b>(&'b self, x: &'b dyn StaticTr) -> &'b u8;
            fn by_mut<'b>(&'b self, x: &'b mut dyn StaticTr) -> usize;
        }
    }
    check_leak! {
        { &'x dyn RelTr }
        { }
        trait LocalBinder {
            fn pointer(&self, f: for<'x> fn(&'x dyn RelTr) -> usize) -> usize;
            fn bound<F: for<'x> ::core::ops::Fn(&'x dyn RelTr) -> usize>(&self, f: F) -> usize;
        }
    }
    // Another input keeps it as written: the reference is leaked alone, with `'x`.
    check_leak! {
        { &'x dyn RelTr }
        { }
        trait LocalBinderAnchored {
            fn pointer(&self, f: for<'x> fn(&'x dyn RelTr, &'x u8) -> &'x u8) -> usize;
            fn bound<F: for<'x> ::core::ops::Fn(&'x dyn RelTr, &'x u8) -> &'x u8>(&self, f: F) -> usize;
        }
    }
    // The output of the binder mentions it, and the reference is no anchor (it is leaked as a
    // whole): the function pointer or the trait object holding the binder is leaked instead, and a
    // bound is an error.
    check_leak! {
        { for<'x> fn(&'x dyn RelTr) -> &'x u8, dyn for<'x> ::core::ops::Fn(&'x dyn RelTr) -> &'x u8 }
        { allow = ::std }
        trait LocalBinderLost {
            fn pointer(&self, f: for<'x> fn(&'x dyn RelTr) -> &'x u8) -> usize;
            fn dynamic(&self, f: ::std::boxed::Box<dyn for<'x> ::core::ops::Fn(&'x dyn RelTr) -> &'x u8>) -> usize;
        }
    }
    check_leak! {
        { error: "use absolute path: 'RelTr'" }
        { }
        trait LocalBinderBound {
            fn bound<F: for<'x> ::core::ops::Fn(&'x dyn RelTr) -> &'x u8>(&self, f: F) -> usize;
        }
    }
    // The return type mentions the lifetime of the method, and no other argument keeps it as
    // written (E0195 where the copy is implemented): the reference cannot be leaked.
    check_leak! {
        { error: "use absolute path: 'RelTr'" }
        { }
        trait LocalUnconstrainedRef {
            fn f<'b>(&self, x: &'b dyn RelTr) -> &'b u8;
        }
    }
    check_leak! {
        { error: "use absolute path: 'RelTr'\nuse absolute path: 'RelTr2'" }
        { }
        trait LocalUnconstrainedRefs {
            fn f<'b>(&self, x: &'b dyn RelTr, y: &'b dyn RelTr2) -> &'b u8;
        }
    }
    // No argument mentions the lifetime: it is early-bound in the original and in the copy alike,
    // so the reference or the path is leaked whatever the return type is.
    check_leak! {
        { &'b dyn RelTr }
        { }
        trait LocalUnmentionedRef {
            fn f<'b>(&self) -> &'b dyn RelTr;
            fn g<'b>(&self) -> &'b u8;
            fn h<'b>(&self, x: u8) -> (&'b dyn RelTr, &'b u8);
        }
    }
    check_leak! {
        { ::core::cell::Ref<'b, dyn RelTr> }
        { }
        trait LocalUnmentionedPath {
            fn f<'b>(&self) -> ::core::cell::Ref<'b, dyn RelTr>;
        }
    }
    // But some argument which mentions it, without keeping it as written, does not exempt it.
    check_leak! {
        { error: "use absolute path: 'RelTr'" }
        { }
        trait LocalMentionedAlone {
            fn f<'b>(&self, x: ::std::vec::Vec<&'b u8>) -> &'b dyn RelTr;
        }
    }
    // A reference to a trait object which is not a cause itself (an absolute trait, with or
    // without what is leaked inside it) is never leaked as a whole, so it keeps the lifetime.
    check_leak! {
        { &'b dyn RelTr, ::core::cell::Ref<'b, dyn RelTr> }
        { }
        trait LocalKeptByAbsolute {
            fn f<'b>(&self, x: &'b dyn RelTr, y: &'b dyn ::core::fmt::Debug) -> &'b u8;
            fn g<'b>(&self, x: &'b dyn RelTr, y: &'b mut dyn ::core::fmt::Debug) -> &'b u8;
            fn h<'b>(&self, x: ::core::cell::Ref<'b, dyn RelTr>, y: &'b dyn ::core::fmt::Debug) -> &'b u8;
        }
    }
    check_leak! {
        { &'b dyn RelTr, Plain }
        { }
        trait LocalKeptByComponent {
            fn f<'b>(&self, x: &'b dyn RelTr, y: &'b dyn ::core::iter::Iterator<Item = Plain>) -> &'b u8;
        }
    }
    // One which is a cause itself may be leaked as a whole, so it does not (conservative): a
    // relative trait, or a bare identifier which may be a const (`y` is then not leaked as a whole,
    // the identifier alone is, but `x` is left with nothing which keeps `'b`).
    check_leak! {
        { error: "use absolute path: 'RelTr'\nuse absolute path: 'RelTr2'" }
        { }
        trait LocalNotKeptByRelative {
            fn f<'b>(&self, x: &'b dyn RelTr, y: &'b dyn RelTr2) -> &'b u8;
        }
    }
    check_leak! {
        { error: "use absolute path: 'RelTr'" }
        { }
        trait LocalNotKeptByConst {
            fn f<'b>(&self, x: &'b dyn RelTr, y: &'b dyn ::core::convert::AsRef<SIZE>) -> &'b u8;
        }
    }
    // The absolute path of an external crate is not an anchor either, as the Referrer may replace
    // the reference to it (old: only with `allow_external_crates` unset, the other did not need
    // the reference to be leaked, see `LocalKeptByAbsolute`). Only `RelTr` is an error: the
    // other type can only be an optional entry, which cannot be leaked there (E0195 for it as
    // well), so it is copied as it is.
    check_leak! {
        { error: "use absolute path: 'RelTr'" }
        { }
        trait LocalNotKeptByExternal {
            fn f<'b>(&self, x: &'b dyn RelTr, y: &'b dyn ::other::Tr) -> &'b u8;
        }
    }
    // Unless the crate is allowed, as `::core` is.
    check_leak! {
        { &'b dyn RelTr }
        { allow = ::other }
        trait LocalKeptByAllowed {
            fn f<'b>(&self, x: &'b dyn RelTr, y: &'b dyn ::other::Tr) -> &'b u8;
        }
    }
    // Another argument keeps it as written, or a bound makes it early-bound in both.
    check_leak! {
        { &'b dyn RelTr }
        { }
        trait LocalConstrainedRef {
            fn anchored<'b>(&'b self, x: &'b dyn RelTr) -> &'b u8;
            fn argument<'b>(&self, x: &'b dyn RelTr, y: &'b u8) -> &'b u8;
            fn bounded<'b, 'c: 'b>(&self, x: &'b dyn RelTr, y: &'c u8) -> &'b u8;
        }
    }
    // A path is leaked as a whole, with the lifetime among its generics, unless its return type
    // mentions it and no other argument keeps it as written (E0195): the trait object cannot be
    // alone then, and the path cannot be leaked.
    check_leak! {
        { ::core::cell::Ref<'b, dyn RelTr> }
        { }
        trait LocalPaths {
            fn cell<'b>(&self, x: ::core::cell::Ref<'b, dyn RelTr>) -> usize;
            fn anchored<'b>(&'b self, x: ::core::cell::Ref<'b, dyn RelTr>) -> &'b u8;
            fn argument<'b>(&self, x: ::core::cell::Ref<'b, dyn RelTr>, y: &'b u8) -> &'b u8;
            fn bound<'b, 'c: 'b>(&self, x: ::core::cell::Ref<'b, dyn RelTr>, y: &'c u8) -> &'b u8;
        }
    }
    check_leak! {
        { error: "use absolute path: 'RelTr'" }
        { }
        trait LocalUnconstrained {
            fn f<'b>(&self, x: ::core::cell::Ref<'b, dyn RelTr>) -> &'b u8;
        }
    }
    check_leak! {
        { error: "use absolute path: 'RelTr'" }
        { }
        trait LocalElided {
            fn f(&self, x: &dyn RelTr) -> usize;
        }
    }
    check_leak! {
        { error: "use absolute path: 'RelTr'" }
        { }
        trait LocalElidedPath {
            fn f(&self, x: ::core::cell::Ref<'_, dyn RelTr>) -> usize;
        }
    }
    check_leak! {
        { Plain }
        { }
        trait RefNested<'a> {
            fn nested(&self, x: &'a dyn ::core::iter::Iterator<Item = &'a dyn ::core::iter::Iterator<Item = Plain>>) -> usize;
        }
    }
    check_leak! {
        { Rel2<T> }
        { }
        trait RefGeneric<'a, T> {
            fn dyn_ref(&self, x: &'a dyn ::core::iter::Iterator<Item = Rel2<T>>) -> usize;
        }
    }
    check_leak! {
        { Rel2<T> }
        { }
        trait Descending<'a, T> {
            fn cell(&self, x: ::core::cell::Ref<'a, Rel2<T>>) -> usize;
        }
    }
    check_leak! {
        { &'a Rel2<T> }
        { }
        trait Referencing<'a, T> {
            fn by_ref(&self, x: &'a Rel2<T>) -> usize;
        }
    }

    // A trait object without a lifetime bound as the argument of a path with a lifetime: the whole
    // path is leaked, whichever the definition of the path (which the Leaker cannot see) makes of the
    // default object lifetime (`'a` for `Explicit`, `'static` for `Inferred` and `Unrelated`).
    check_leak! {
        { Explicit<'a, dyn RelTr>, Inferred<'a, dyn RelTr>, Unrelated<'a, dyn RelTr> }
        { }
        trait ObjectDefaultsPaths<'a> {
            fn explicit(&self, x: Explicit<'a, dyn RelTr>) -> Explicit<'a, dyn RelTr>;
            fn inferred(&self, x: Inferred<'a, dyn RelTr>) -> Inferred<'a, dyn RelTr>;
            fn unrelated(&self, x: Unrelated<'a, dyn RelTr>) -> Unrelated<'a, dyn RelTr>;
        }
    }
    // The same for a lifetime of the method, which `&'b self` keeps as written.
    check_leak! {
        { Inferred<'b, dyn RelTr> }
        { }
        trait ObjectDefaultsLocal {
            fn local<'b>(&'b self, x: Inferred<'b, dyn RelTr>) -> Inferred<'b, dyn RelTr>;
        }
    }

    // What the bounds which are written above avoid, disabled (`#[cfg(any())]`) as it does not
    // compile: a path with a lifetime (`Rel<'a, T>`, `Ref<'a, Rel2<T>>`) is not assumed to
    // require `T: 'a`, so the Repeater's implementation does not have it, unless the signature
    // writes it. Enabling these gives the errors of the comments (each checked on stable and
    // nightly).

    /// `error[E0309]: the parameter type `T` may not live long enough`, at the implementation
    /// `impl<'a, T> Repeater<(&'a (), T), ..> for __S { type Type = Rel<'a, T>; }`: `T: 'a` is
    /// only implied in the original, by the definition of `Rel`, which the Leaker cannot see.
    #[cfg(any())]
    #[leak(
        helper = crate::__macro,
        repeater = crate::Repeater,
        module = crate::m,
    )]
    pub trait Unbounded {
        fn by_ref<'a, T>(&self, x: Rel<'a, T>) -> usize;
    }

    /// `error[E0309]: the parameter type `__S` may not live long enough`, as `Rel<'a, Self>`
    /// needs `Self: 'a`, which `where Self: Sized + 'a` writes (see `SelfBound`).
    #[cfg(any())]
    #[leak(
        helper = crate::__macro,
        repeater = crate::Repeater,
        module = crate::m,
    )]
    pub trait UnboundedSelf {
        fn me<'a>(&'a self) -> Rel<'a, Self>
        where
            Self: Sized;
    }

    /// `Rel2<T>` is leaked with the generics `<T>`, as the path `Ref<'a, ..>` does not imply a
    /// bound. The copy of the field of a struct, `pub struct Unbounded<'a, T>(pub Ref<'a,
    /// Rel2<T>>)` (see `user`), is rejected with `error[E0309]: the parameter type `T` may not live long
    /// enough` (`Rel2<T>: 'a`): the original is accepted, as RFC 2093 infers `T: 'a` for the
    /// struct from the field, but nothing infers it through the projection `<S as Repeater<(T,),
    /// ..>>::Type`. Writing `T: 'a` in the signature and the struct makes it the generics of
    /// `Rel2<T>` and the copy compile (see `Around`).
    #[cfg(any())]
    #[leak(
        helper = crate::__macro,
        repeater = crate::Repeater,
        module = crate::m,
    )]
    pub trait UnboundedAround {
        fn cell<'a, T>(&self, x: ::core::cell::Ref<'a, Rel2<T>>) -> usize;
    }

    pub struct Inner;

    impl Bounds for Inner {
        fn by_ref<'a, T: 'a>(&self, x: Rel<'a, T>) -> usize {
            core::mem::size_of_val(x.0)
        }
        fn loose<'a, T>(&self, _x: Loose<'a, T>) -> usize {
            1
        }
        fn written<'a, 'b: 'a, T: 'b>(&self, x: Rel<'a, &'b T>) -> usize {
            core::mem::size_of_val(x.0)
        }
        fn forever<T: 'static>(&self, x: Rel<'static, T>) -> usize {
            core::mem::size_of_val(x.0)
        }
    }

    impl<'a, T: 'a> Outlives<'a, T> for Inner {
        fn field(&self, x: Rel<'a, T>) -> usize {
            core::mem::size_of_val(x.0)
        }
        fn mixed<'b: 'a, U: 'a>(&self, _x: Rel<'a, (T, &'b u8, U)>) -> usize {
            2
        }
    }

    impl<'a> Local<'a> for Inner {
        fn local<T: 'a>(&self, x: Rel<'a, T>) -> usize {
            core::mem::size_of_val(x.0)
        }
        fn lifetime<'b: 'a>(&self, _x: Rel<'a, &'b u8>) -> usize {
            7
        }
    }

    impl<'a, T> Descend<'a, T> for Inner {
        fn by_ref(&self, _x: &'a Rel2<T>) -> usize {
            3
        }
        fn cell(&self, _x: ::core::cell::Ref<'a, Rel2<T>>) -> usize {
            4
        }
        fn local_ref<'b>(&self, _x: &'b Rel2<T>) -> usize {
            5
        }
        fn local_cell<'b>(&self, _x: ::core::cell::Ref<'b, Rel2<T>>) -> usize {
            6
        }
        fn dyn_ref(&self, x: &'a dyn ::core::iter::Iterator<Item = Rel2<T>>) -> usize {
            x.size_hint().0
        }
        fn plain(&self, _x: Rel2<T>) -> usize {
            11
        }
    }

    impl Around for Inner {
        fn cell<'a, T: 'a>(&self, _x: ::core::cell::Ref<'a, Rel2<T>>) -> usize {
            1
        }
        fn by_ref<'a, T>(&self, _x: &'a Rel2<T>) -> usize {
            2
        }
        fn by_mut<'a, T>(&self, _x: &'a mut Rel2<T>) -> usize {
            3
        }
        fn nested<'a, T: 'a>(&self, x: ::core::cell::Ref<'a, ::std::vec::Vec<Rel2<T>>>) -> usize {
            x.len()
        }
        fn tuple<'a, T>(&self, x: &'a (u8, [Rel2<T>; 2])) -> usize {
            x.0 as usize
        }
        fn function<'a, T>(&self, _x: &'a fn(Rel2<T>) -> usize) -> usize {
            5
        }
        fn object<'a, T: 'a>(
            &self,
            x: ::core::cell::Ref<'a, dyn ::core::iter::Iterator<Item = Rel2<T>>>,
        ) -> usize {
            x.size_hint().0
        }
        fn lifetime<'a, 'c>(&self, x: &'a Rel2<&'c u8>) -> usize {
            *x.0 as usize
        }
        fn both<'a, 'b, T>(&self, _x: &'a &'b Rel2<T>) -> usize {
            6
        }
        fn forever<T: 'static>(&self, _x: ::core::cell::Ref<'static, Rel2<T>>) -> usize {
            7
        }
        fn binder<T>(&self, _f: for<'x> fn(&'x Rel2<T>) -> usize) -> usize {
            8
        }
        fn projection<'a, T>(&self, _x: &'a <Rel2<T> as ::core::ops::Deref>::Target) -> usize {
            9
        }
        fn alone<T>(&self, _x: Rel2<T>) -> usize {
            10
        }
        fn gat<'a, T: ::type_leak_test_outlives::Gat, U>(
            &self,
            _x: ::core::cell::Ref<'a, T::A<Rel2<U>>>,
        ) -> usize {
            12
        }
        fn gat_nested<'a, T: ::type_leak_test_outlives::Gat, U>(
            &self,
            _x: &'a ::std::vec::Vec<T::A<Rel2<U>>>,
        ) -> usize {
            13
        }
    }

    impl SelfAround for Inner {
        fn cell<'a>(&self, _x: ::core::cell::Ref<'a, Rel2<Self>>) -> usize
        where
            Self: Sized + 'a,
        {
            1
        }
        fn by_ref<'a>(&self, _x: &'a Rel2<Self>) -> usize {
            2
        }
    }

    impl Binders for Inner {
        fn pointer(&self, f: for<'x> fn(Hr<'x>) -> usize) -> usize {
            f(Hr(&3))
        }
        fn bound<F: for<'x> Fn(Hr<'x>) -> usize>(&self, f: F) -> usize {
            f(Hr(&4))
        }
        fn predicate<F>(&self, f: F) -> usize
        where
            for<'x> F: Fn(Hr<'x>) -> usize,
        {
            f(Hr(&5))
        }
        fn dynamic(&self, f: Box<dyn for<'x> Fn(Hr<'x>) -> usize>) -> usize {
            f(Hr(&6))
        }
        fn whole(&self, f: for<'x> fn(Hr<'x>) -> &'x u8) -> usize {
            *f(Hr(&7)) as usize
        }
        fn returning(&self, f: for<'x> fn(&'x u8) -> Hr<'x>) -> usize {
            *f(&8).0 as usize
        }
        fn object(&self, f: Box<dyn for<'x> Fn(Hr<'x>) -> &'x u8>) -> usize {
            *f(Hr(&9)) as usize
        }
        fn binding(
            &self,
            f: Box<dyn for<'x> ::core::ops::Index<&'x u8, Output = Hr<'x>>>,
        ) -> usize {
            *f[&10].0 as usize
        }
        fn anchored<F: for<'x> Fn(&'x u8, Hr<'x>) -> &'x u8>(&self, f: F) -> usize {
            *f(&11, Hr(&12)) as usize
        }
        fn anchored_after<F: for<'x> Fn(Hr<'x>, &'x u8) -> &'x u8>(&self, f: F) -> usize {
            *f(Hr(&13), &14) as usize
        }
        fn anchored_predicate<F>(&self, f: F) -> usize
        where
            for<'x> F: Fn(&'x u8, Hr<'x>) -> &'x u8,
        {
            *f(&15, Hr(&16)) as usize
        }
        fn anchored_pointer(&self, f: for<'x> fn(&'x u8, Hr<'x>) -> &'x u8) -> usize {
            *f(&17, Hr(&18)) as usize
        }
        fn anchored_object(&self, f: Box<dyn for<'x> Fn(&'x u8, Hr<'x>) -> &'x u8>) -> usize {
            *f(&19, Hr(&20)) as usize
        }
        fn anchored_impl(&self, f: impl for<'x> Fn(&'x u8, Hr<'x>) -> &'x u8) -> usize {
            *f(&32, Hr(&33)) as usize
        }
        fn tuple<F: for<'x> Fn((&'x u8, Hr<'x>, Plain)) -> &'x u8>(&self, f: F) -> usize {
            *f((&21, Hr(&22), Plain(23))) as usize
        }
        fn tuple_again<F: for<'x> Fn((&'x u8, Hr<'x>, Plain)) -> &'x u8>(&self, f: F) -> usize {
            *f((&24, Hr(&25), Plain(26))) as usize
        }
        fn lost(&self, f: for<'x> fn(Rel2<&'x u8>) -> &'x u8) -> usize {
            *f(Rel2(&27)) as usize
        }
        fn kept(&self, f: for<'x> fn(Rel2<&'x u8>, &'x u8) -> &'x u8) -> usize {
            *f(Rel2(&28), &29) as usize
        }
        fn binding_bound<F: for<'x> Fn(Box<dyn Iterator<Item = &'x u8>>, Hr<'x>) -> &'x u8>(
            &self,
            f: F,
        ) -> usize {
            *f(Box::new(::core::iter::once(&40)), Hr(&41)) as usize
        }
        fn binding_predicate<F>(&self, f: F) -> usize
        where
            for<'x> F: Fn(Box<dyn Iterator<Item = &'x u8>>, Hr<'x>) -> &'x u8,
        {
            *f(Box::new(::core::iter::once(&42)), Hr(&43)) as usize
        }
        fn binding_pointer(
            &self,
            f: for<'x> fn(Box<dyn Iterator<Item = &'x u8>>, Hr<'x>) -> &'x u8,
        ) -> usize {
            *f(Box::new(::core::iter::once(&44)), Hr(&45)) as usize
        }
        fn binding_object(
            &self,
            f: Box<dyn for<'x> Fn(Box<dyn Iterator<Item = &'x u8>>, Hr<'x>) -> &'x u8>,
        ) -> usize {
            *f(Box::new(::core::iter::once(&46)), Hr(&47)) as usize
        }
        fn binding_impl(
            &self,
            f: impl for<'x> Fn(Box<dyn Iterator<Item = &'x u8>>, Hr<'x>) -> &'x u8,
        ) -> usize {
            *f(Box::new(::core::iter::once(&48)), Hr(&49)) as usize
        }
        fn lifetime_bound<F: for<'x> Fn(Box<dyn ::core::fmt::Debug + 'x>, Hr<'x>) -> &'x u8>(
            &self,
            f: F,
        ) -> usize {
            *f(Box::new(50u8), Hr(&51)) as usize
        }
        fn nested_args<F: for<'x> Fn(Box<dyn Fn(&'x u8)>, Hr<'x>) -> &'x u8>(&self, f: F) -> usize {
            *f(Box::new(|_| ()), Hr(&52)) as usize
        }
        fn nested_output<F: for<'x> Fn(Box<dyn Fn() -> &'x u8>, Hr<'x>) -> &'x u8>(
            &self,
            f: F,
        ) -> usize {
            *f(Box::new(|| &53), Hr(&54)) as usize
        }
        fn nested_binder<F: for<'x> Fn(Box<dyn for<'y> Fn(&'x u8, &'y u8)>, Hr<'x>) -> &'x u8>(
            &self,
            f: F,
        ) -> usize {
            *f(Box::new(|_, _| ()), Hr(&55)) as usize
        }
        fn nested_pointer<F: for<'x> Fn(fn(&'x u8), Hr<'x>) -> &'x u8>(&self, f: F) -> usize {
            *f(|_| (), Hr(&56)) as usize
        }
        fn nested_pointer_output<F: for<'x> Fn(fn() -> &'x u8, Hr<'x>) -> &'x u8>(
            &self,
            f: F,
        ) -> usize {
            *f(|| &57, Hr(&58)) as usize
        }
        fn nested_pointer_binder<F: for<'x> Fn(for<'y> fn(&'x u8, &'y u8), Hr<'x>) -> &'x u8>(
            &self,
            f: F,
        ) -> usize {
            *f(|_, _| (), Hr(&59)) as usize
        }
        fn lost_binding(
            &self,
            f: for<'x> fn(Box<dyn RelIt<Item = &'x u8>>, Hr<'x>) -> &'x u8,
        ) -> usize {
            let _ = f;
            60
        }
        fn lost_bound(&self, f: for<'x> fn(Box<dyn RelTr + 'x>, Hr<'x>) -> &'x u8) -> usize {
            *f(Box::new(Plain(61)), Hr(&62)) as usize
        }
        fn kept_binding(
            &self,
            f: for<'x> fn(
                Box<dyn RelIt<Item = &'x u8>>,
                Box<dyn Iterator<Item = &'x u8>>,
            ) -> &'x u8,
        ) -> usize {
            let _ = f;
            63
        }
        fn core_bound<F: for<'x> Fn(::core::option::Option<&'x u8>, Hr<'x>) -> &'x u8>(
            &self,
            f: F,
        ) -> usize {
            *f(Some(&41), Hr(&42)) as usize
        }
        fn core_pointer(
            &self,
            f: for<'x> fn(::core::cell::Ref<'x, u8>, Hr<'x>) -> &'x u8,
        ) -> usize {
            let cell = ::core::cell::RefCell::new(43u8);
            *f(cell.borrow(), Hr(&44)) as usize
        }
        fn core_object(
            &self,
            f: Box<dyn for<'x> Fn(::core::option::Option<(u8, &'x u8)>, Hr<'x>) -> &'x u8>,
        ) -> usize {
            *f(Some((1, &45)), Hr(&46)) as usize
        }
        fn external_pointer(
            &self,
            f: for<'x> fn(::std::boxed::Box<dyn Iterator<Item = &'x u8>>, Hr<'x>) -> &'x u8,
        ) -> usize {
            *f(Box::new(::core::iter::once(&47)), Hr(&48)) as usize
        }
        fn external_object(
            &self,
            f: Box<
                dyn for<'x> Fn(::std::boxed::Box<dyn Iterator<Item = &'x u8>>, Hr<'x>) -> &'x u8,
            >,
        ) -> usize {
            *f(Box::new(::core::iter::once(&49)), Hr(&50)) as usize
        }
    }

    impl SelfOutlives for Inner {
        type Item = u8;
        fn me<'a>(&'a self, x: &'a Rel2<Self>) -> usize
        where
            Self: Sized,
        {
            let _ = x;
            1
        }
        fn it<'a>(&'a self, x: &'a Rel2<Self::Item>) -> usize
        where
            Self: Sized,
        {
            x.0 as usize
        }
        fn arg<'a>(&self, x: &'a Rel2<Self>) -> usize
        where
            Self: Sized,
        {
            let _ = x;
            0
        }
    }

    impl SelfBound for Inner {
        type Item = u8;
        fn me<'a>(&'a self) -> Rel<'a, Self>
        where
            Self: Sized + 'a,
        {
            Rel(self)
        }
        fn it<'a>(&'a self) -> Rel<'a, Self::Item>
        where
            Self: Sized + 'a,
        {
            Rel(&1)
        }
        fn split<'a>(&self, _x: Rel<'a, Self>) -> usize
        where
            Self: Sized,
            Self: 'a,
        {
            1
        }
        fn implied<'a>(&self, _x: &'a Rel<'a, Self>) -> usize
        where
            Self: Sized,
        {
            2
        }
    }

    impl<'a> SelfBoundTrait<'a> for Inner {
        fn me(&'a self) -> Rel<'a, Self> {
            Rel(self)
        }
        fn arg(&self, _x: Rel<'a, Self>) -> usize {
            3
        }
    }

    impl<'a> Dyns<'a> for Inner {
        fn items(
            &self,
            x: ::core::cell::Ref<'a, dyn ::core::iter::Iterator<Item = Plain>>,
        ) -> usize {
            x.size_hint().0
        }
        fn whole(&self, x: ::core::cell::Ref<'a, dyn RelTr>) -> usize {
            x.rel()
        }
        fn local<'b>(
            &self,
            x: ::core::cell::Ref<'b, dyn ::core::iter::Iterator<Item = Plain>>,
        ) -> usize {
            x.size_hint().0
        }
        fn by_ref(&self, x: &'a dyn ::core::iter::Iterator<Item = Plain>) -> usize {
            x.size_hint().0
        }
        #[allow(unused_parens)]
        fn by_paren(&self, x: &'a (dyn ::core::iter::Iterator<Item = Plain>)) -> usize {
            x.size_hint().0
        }
        fn by_mut(&self, x: &'a mut dyn ::core::iter::Iterator<Item = Plain>) -> usize {
            x.count()
        }
        fn bounded(&self, x: &'a (dyn ::core::iter::Iterator<Item = Plain> + 'static)) -> usize {
            x.size_hint().0
        }
        fn forever(&self, x: &'static dyn ::core::iter::Iterator<Item = Plain>) -> usize {
            x.size_hint().0
        }
        fn local_ref<'b>(&self, x: &'b dyn ::core::iter::Iterator<Item = Plain>) -> usize {
            x.size_hint().0
        }
        fn whole_ref(&self, x: &'a dyn RelTr) -> usize {
            x.rel()
        }
        fn whole_mut(&self, x: &'a mut dyn RelTr) -> usize {
            x.rel() + 1
        }
        fn bare_ref(&self, x: &'a dyn ::core::borrow::Borrow<Plain>) -> usize {
            ::core::borrow::Borrow::<Plain>::borrow(x).0 as usize
        }
        fn nested(
            &self,
            x: &'a dyn ::core::iter::Iterator<Item = &'a dyn ::core::iter::Iterator<Item = Plain>>,
        ) -> usize {
            x.size_hint().0
        }
    }

    impl Locals for Inner {
        fn by_ref<'b>(&self, x: &'b dyn RelTr) -> usize {
            x.rel()
        }
        fn by_mut<'b>(&self, x: &'b mut dyn RelTr) -> usize {
            x.rel() + 1
        }
        fn returned<'b>(&self, x: &'b dyn RelTr, y: &'b u8) -> &'b dyn RelTr {
            let _ = y;
            x
        }
        fn anchored<'b>(&'b self, x: &'b dyn RelTr) -> &'b dyn RelTr {
            x
        }
        fn bounded<'b, 'c: 'b>(&self, x: &'b dyn RelTr, y: &'c u8) -> &'b u8 {
            let _ = x;
            y
        }
        fn several<'b, 'c: 'b>(&self, x: &'b dyn RelTr, y: &'c dyn RelTr2) -> usize {
            x.rel() + y.rel2()
        }
        fn clause<'b, 'c>(&self, x: &'b dyn RelTr, y: &'c dyn RelTr) -> usize
        where
            'c: 'b,
        {
            x.rel() + y.rel() + 3
        }
        fn nested<'b, 'c>(&self, x: &'b &'c dyn RelTr) -> usize {
            x.rel() + 4
        }
        fn cell<'b>(&self, x: ::core::cell::Ref<'b, dyn RelTr>) -> usize {
            x.rel() + 5
        }
        fn cell_anchored<'b>(&'b self, x: ::core::cell::Ref<'b, dyn RelTr>) -> &'b u8 {
            let _ = x;
            &6
        }
        fn cell_argument<'b>(&self, x: ::core::cell::Ref<'b, dyn RelTr>, y: &'b u8) -> &'b u8 {
            let _ = x;
            y
        }
        fn cell_bound<'b, 'c: 'b>(&self, x: ::core::cell::Ref<'b, dyn RelTr>, y: &'c u8) -> &'b u8 {
            let _ = x;
            y
        }
        fn unmentioned<'b>(&self) -> &'b dyn RelTr {
            &Plain(9)
        }
        fn unmentioned_cell<'b>(&self) -> ::core::cell::Ref<'b, dyn RelTr> {
            unimplemented!()
        }
        fn kept_by_absolute<'b>(&self, x: &'b dyn RelTr, y: &'b dyn ::core::fmt::Debug) -> &'b u8 {
            let _ = (x, y);
            &10
        }
        fn kept_by_component<'b>(
            &self,
            x: &'b dyn RelTr,
            y: &'b dyn ::core::iter::Iterator<Item = Plain>,
        ) -> &'b u8 {
            let _ = (x, y);
            &11
        }
    }

    impl LocalBinders for Inner {
        fn pointer(&self, f: for<'x> fn(&'x dyn RelTr) -> usize) -> usize {
            f(&Plain(1))
        }
        fn bound<F: for<'x> Fn(&'x dyn RelTr) -> usize>(&self, f: F) -> usize {
            f(&Plain(2))
        }
        fn predicate<F>(&self, f: F) -> usize
        where
            for<'x> F: Fn(&'x dyn RelTr) -> usize,
        {
            f(&Plain(3))
        }
        fn anchored_pointer(&self, f: for<'x> fn(&'x dyn RelTr, &'x u8) -> &'x u8) -> usize {
            *f(&Plain(8), &9) as usize
        }
        fn anchored_bound<F: for<'x> Fn(&'x dyn RelTr, &'x u8) -> &'x u8>(&self, f: F) -> usize {
            *f(&Plain(10), &11) as usize
        }
        fn lost_pointer(&self, f: for<'x> fn(&'x dyn RelTr) -> &'x u8) -> usize {
            *f(&Plain(4)) as usize
        }
        fn lost_dynamic(&self, f: Box<dyn for<'x> Fn(&'x dyn RelTr) -> &'x u8>) -> usize {
            *f(&Plain(5)) as usize
        }
        fn path(&self, f: for<'x> fn(::core::cell::Ref<'x, dyn RelTr>) -> usize) -> usize {
            let cell = ::core::cell::RefCell::new(Box::new(Plain(6)) as Box<dyn RelTr>);
            f(::core::cell::Ref::map(cell.borrow(), |b| &**b))
        }
        fn path_bound<F: for<'x> Fn(::core::cell::Ref<'x, dyn RelTr>) -> usize>(
            &self,
            f: F,
        ) -> usize {
            let cell = ::core::cell::RefCell::new(Box::new(Plain(7)) as Box<dyn RelTr>);
            f(::core::cell::Ref::map(cell.borrow(), |b| &**b))
        }
    }

    impl<'a> Statics<'a> for Inner {
        fn by_ref(&self, x: &'a dyn StaticTr) -> usize {
            x.stat()
        }
        fn by_mut(&self, x: &'a mut dyn StaticTr) -> usize {
            x.stat() + 1
        }
        fn local<'b>(&'b self, x: &'b dyn StaticTr) -> &'b u8 {
            let _ = x;
            &8
        }
        fn local_mut<'b>(&'b self, x: &'b mut dyn StaticTr) -> usize {
            x.stat() + 2
        }
        fn returned(&self, x: &'a dyn StaticTr) -> &'a dyn StaticTr {
            x
        }
    }

    impl<'a> ObjectDefaults<'a> for Inner {
        fn explicit(&self, x: Explicit<'a, dyn RelTr>) -> Explicit<'a, dyn RelTr> {
            x
        }
        fn inferred(&self, x: Inferred<'a, dyn RelTr>) -> Inferred<'a, dyn RelTr> {
            x
        }
        fn unrelated(&self, x: Unrelated<'a, dyn RelTr>) -> Unrelated<'a, dyn RelTr> {
            x
        }
        fn local<'b>(&'b self, x: Inferred<'b, dyn RelTr>) -> Inferred<'b, dyn RelTr> {
            x
        }
    }
}

/// The Referrer's side: implements the traits by forwarding to `Inner`, without importing any of
/// the types relative to `crate::m`. The copies of the signatures are compiled here.
pub mod user {
    /// The Referrer which does not trust the names of the external crates: it replaces their types
    /// (the optional entries) as well, so the function pointers and the trait objects of `Binders`
    /// which hold an anchor of an external crate, and the `::std::vec::Vec<..>` of `Around`, are
    /// copied by the Repeater too, and compile.
    pub struct StrictWrapper(pub crate::m::Inner);

    crate::m::Binders!(strict impl Binders for StrictWrapper => 0);
    crate::m::Around!(strict impl Around for StrictWrapper => 0);
    crate::m::Dyns!(strict impl<'a> Dyns<'a> for StrictWrapper => 0);
    crate::m::Locals!(strict impl Locals for StrictWrapper => 0);
    crate::m::LocalBinders!(strict impl LocalBinders for StrictWrapper => 0);

    /// The copies as the fields of structs, with the `::std::vec::Vec<..>` replaced by its entry,
    /// which has the bounds that the struct writes (`T: 'a`, as `Ref<'a, X>` requires it, E0309
    /// otherwise).
    pub mod strict_copies {
        use crate::m::Rel2;

        crate::m::Around!(strict copy [StrictWrapper] {
            pub struct Nested<'a, T: 'a>(pub ::core::cell::Ref<'a, ::std::vec::Vec<Rel2<T>>>);
            // The entry `Vec<T::A<Rel2<U>>>` has the where predicate `T::A<Rel2<U>>: 'a` (the
            // projection is a unit, which the original gets by RFC 2093 from its field). The
            // copy of the field does not infer it through the projection of the Repeater: write it
            // (with `Rel2` in scope, as the predicate is not rewritten, being no entry's context).
            pub struct GatNested<'a, T: ::type_leak_test_outlives::Gat, U>(
                pub &'a ::std::vec::Vec<T::A<Rel2<U>>>,
            )
            where
                T::A<Rel2<U>>: 'a;
        });
        pub struct StrictWrapper;
    }

    pub struct Wrapper(pub crate::m::Inner);

    crate::m::Bounds!(impl Bounds for Wrapper => 0);
    crate::m::Outlives!(impl<'a, T: 'a> Outlives<'a, T> for Wrapper => 0);
    crate::m::Local!(impl<'a> Local<'a> for Wrapper => 0);
    crate::m::Descend!(impl<'a, T> Descend<'a, T> for Wrapper => 0);
    crate::m::Dyns!(impl<'a> Dyns<'a> for Wrapper => 0);
    crate::m::Binders!(impl Binders for Wrapper => 0);
    // A trait object borrowed through a lifetime of the method: the forwarding implementations, with
    // the reference kept in the signature around the leaked trait object (`E0195` if the
    // lifetime were left only in a projection).
    crate::m::Locals!(impl Locals for Wrapper => 0);
    crate::m::LocalBinders!(impl LocalBinders for Wrapper => 0);
    // A trait with a lifetime bound of its own, borrowed through a lifetime of the trait and of
    // the method: the whole reference is leaked, so rustc computes the default of the trait object
    // (`'static`) in the implementation of the Repeater (E0478 if it were written out as `'a`).
    crate::m::Statics!(impl<'a> Statics<'a> for Wrapper => 0);
    crate::m::Around!(impl Around for Wrapper => 0);
    // The copies of the types around `Rel2<T>` as the fields of structs and a free function, where
    // the bounds which the original has by RFC 2093 are not implied unless they are written, or
    // the context of the leaked type carries them (the reference around it implies `T: 'a`, and a
    // path with a lifetime does not: E0309 without `T: 'a` written, see `UnboundedAround`).
    // `Wrapper` is `Self`, but nothing mentions it.
    crate::m::Around!(copy [Wrapper] {
        pub struct Cell<'a, T: 'a>(pub ::core::cell::Ref<'a, Rel2<T>>);
        pub struct ByRef<'a, T>(pub &'a Rel2<T>);
        pub struct ByMut<'a, T>(pub &'a mut Rel2<T>);
        pub struct Nested<'a, T: 'a>(pub ::core::cell::Ref<'a, ::std::vec::Vec<Rel2<T>>>);
        pub struct Tuple<'a, T>(pub &'a (u8, [Rel2<T>; 2]));
        pub struct Function<'a, T>(pub &'a fn(Rel2<T>) -> usize);
        pub struct Object<'a, T: 'a>(
            pub ::core::cell::Ref<'a, dyn ::core::iter::Iterator<Item = Rel2<T>>>,
        );
        pub struct Lifetime<'a, 'c>(pub &'a Rel2<&'c u8>);
        pub struct Both<'a, 'b, T>(pub &'a &'b Rel2<T>);
        pub struct Forever<T: 'static>(pub ::core::cell::Ref<'static, Rel2<T>>);
        pub struct Binder<T>(pub for<'x> fn(&'x Rel2<T>) -> usize);
        pub struct Projection<'a, T>(pub &'a <Rel2<T> as ::core::ops::Deref>::Target);
        pub struct Alone<T>(pub Rel2<T>);
        // A path starting with a type parameter is a projection: no `U: 'a` is declared (and the
        // original needs none), so the copy must not require it either.
        pub struct GatCell<'a, T: ::type_leak_test_outlives::Gat, U>(
            pub ::core::cell::Ref<'a, T::A<Rel2<U>>>,
        );
        pub struct GatNested<'a, T: ::type_leak_test_outlives::Gat, U>(
            pub &'a ::std::vec::Vec<T::A<Rel2<U>>>,
        );
        // The original is well-formed with a `U` which does not outlive `'a` (`Rel2<U>` is not
        // required to, as the projection is `()`), and so is the copy, unless it requires
        // `Rel2<U>: 'a` of the type inside the projection, as the Repeater's arguments would
        // encode (`&'a U`).
        pub fn gat_short_lived<'x>(cell: ::core::cell::Ref<'static, ()>, _short: &'x u8) {
            let _held: GatCell<'static, ::type_leak_test_outlives::GatUnit, &'x u8> =
                GatCell(cell);
        }
        pub fn around_free<'a, T: 'a>(_x: ::core::cell::Ref<'a, Rel2<T>>, _y: &'a Rel2<T>) {}
        pub enum Choice<'a, T: 'a> {
            Cell(::core::cell::Ref<'a, Rel2<T>>),
            Alone(Rel2<T>),
        }
    });
    // The field of a struct without the bound written, disabled as it does not compile (E0309, see
    // `m::UnboundedAround`): `Rel2<T>` has the generics `<T>` of its signature, and the copy
    // `Ref<'a, <Wrapper as Repeater<(T,), ..>>::Type>` is well-formed only with `T: 'a`, which the
    // struct has to write (see `Around`).
    #[cfg(any())]
    crate::m::UnboundedAround!(copy [Wrapper] {
        pub struct Unbounded<'a, T>(pub ::core::cell::Ref<'a, Rel2<T>>);
    });
    // The copies of a trait object borrowed through a lifetime of a function, as a free function
    // and the fields of a struct and an enum (`Wrapper` is the type of the projection).
    crate::m::Locals!(copy [Wrapper] {
        pub fn locals_free<'b, 'c: 'b>(
            _x: &'b dyn RelTr,
            _y: &'b mut dyn RelTr,
            _z: &'c &'c dyn RelTr,
            _w: ::core::cell::Ref<'b, dyn RelTr>,
        ) -> &'b dyn RelTr {
            loop {}
        }
        pub struct HoldsRef<'b>(pub &'b dyn RelTr);
        pub struct HoldsMut<'b>(pub &'b mut dyn RelTr);
        pub struct HoldsCell<'b>(pub ::core::cell::Ref<'b, dyn RelTr>);
        pub enum HoldsEither<'b> {
            Ref(&'b dyn RelTr),
            Cell(::core::cell::Ref<'b, dyn RelTr>),
        }
    });
    // The copies of it as a free function and the fields of a struct and an enum.
    crate::m::Statics!(copy [Wrapper] {
        pub fn statics_free<'a, 'b>(
            _x: &'a dyn StaticTr,
            _y: &'a mut dyn StaticTr,
            _z: &'b dyn StaticTr,
        ) -> &'a dyn StaticTr {
            loop {}
        }
        pub struct HoldsStatic<'a>(pub &'a dyn StaticTr);
        pub struct HoldsStaticMut<'a>(pub &'a mut dyn StaticTr);
        pub enum HoldsStaticEither<'a> {
            Ref(&'a dyn StaticTr),
            Mut(&'a mut dyn StaticTr),
        }
    });
    // A trait object without a lifetime bound as the argument of a path with a lifetime: the copies
    // return the same path as the original, whose default object lifetime rustc takes from the
    // definition of the path (E0308 in the return types if it were leaked alone with `+ 'a`).
    crate::m::ObjectDefaults!(impl<'a> ObjectDefaults<'a> for Wrapper => 0);
    // The copies of them as a free function and the fields of a struct and an enum.
    crate::m::ObjectDefaults!(copy [Wrapper] {
        pub fn defaults_free<'a>(
            x: Explicit<'a, dyn RelTr>,
            y: Inferred<'a, dyn RelTr>,
            z: Unrelated<'a, dyn RelTr>,
        ) -> (
            Explicit<'a, dyn RelTr>,
            Inferred<'a, dyn RelTr>,
            Unrelated<'a, dyn RelTr>,
        ) {
            (x, y, z)
        }
        pub fn defaults_local<'b>(x: Inferred<'b, dyn RelTr>) -> Inferred<'b, dyn RelTr> {
            x
        }
        pub struct HoldsDefaults<'a> {
            pub explicit: Explicit<'a, dyn RelTr>,
            pub inferred: Inferred<'a, dyn RelTr>,
            pub unrelated: Unrelated<'a, dyn RelTr>,
        }
        pub enum HoldsDefaultsEither<'a> {
            Explicit(Explicit<'a, dyn RelTr>),
            Inferred(Inferred<'a, dyn RelTr>),
            Unrelated(Unrelated<'a, dyn RelTr>),
        }
    });
    // The alternative which is NOT taken, written out and disabled: leaking the trait object `dyn
    // RelTr + 'a` alone, and keeping the path around `<Self as Repeater<..>>::Type` in the copy.
    // `Inferred<'a, dyn RelTr>` and `Unrelated<'a, dyn RelTr>` are `+ 'static` (their definitions
    // do not write `T: 'a`), so the copy returns a type whose object lifetime is `'a` where the
    // trait returns `'static`, which rustc rejects (E0308 `method not compatible with trait` on
    // stable, E0478 `lifetime bound not satisfied` on nightly, for `inferred` and `unrelated`, and
    // E0521 in the forwarding body); it compiles for `explicit` only. The Repeater's whole path
    // above is read by rustc from the real definition instead.
    #[cfg(any())]
    mod alone {
        use crate::m::{Explicit, Inferred, Unrelated};

        pub struct Alone(pub crate::m::Inner);

        impl<'a, __S: ?Sized> crate::Repeater<(&'a (),), 0, 0> for __S {
            type Type = dyn crate::m::RelTr + 'a;
        }

        impl<'a> crate::m::ObjectDefaults<'a> for Alone {
            fn explicit(
                &self,
                x: Explicit<'a, <Self as crate::Repeater<(&'a (),), 0, 0>>::Type>,
            ) -> Explicit<'a, <Self as crate::Repeater<(&'a (),), 0, 0>>::Type> {
                crate::m::ObjectDefaults::<'a>::explicit(&self.0, x)
            }
            fn inferred(
                &self,
                x: Inferred<'a, <Self as crate::Repeater<(&'a (),), 0, 0>>::Type>,
            ) -> Inferred<'a, <Self as crate::Repeater<(&'a (),), 0, 0>>::Type> {
                crate::m::ObjectDefaults::<'a>::inferred(&self.0, x)
            }
            fn unrelated(
                &self,
                x: Unrelated<'a, <Self as crate::Repeater<(&'a (),), 0, 0>>::Type>,
            ) -> Unrelated<'a, <Self as crate::Repeater<(&'a (),), 0, 0>>::Type> {
                crate::m::ObjectDefaults::<'a>::unrelated(&self.0, x)
            }
            // `local` is left out: the other methods say it all.
        }
    }
    /// The forwarding implementations are called through `&dyn Locals`, and the explicit
    /// lifetime arguments of the original are late-bound (not allowed for the original and the copy
    /// alike, E0794), so the call is the same.
    pub fn through_dyn(
        locals: &dyn crate::m::Locals,
        x: &dyn crate::m::RelTr,
        y: &dyn crate::m::RelTr2,
    ) -> usize {
        let cell =
            ::core::cell::RefCell::new(Box::new(crate::m::Plain(9)) as Box<dyn crate::m::RelTr>);
        locals.by_ref(x)
            + locals.returned(x, &7).rel()
            + locals.anchored(x).rel()
            + locals.nested(&x)
            + locals.clause(x, x)
            + locals.cell(::core::cell::Ref::map(cell.borrow(), |b| &**b))
            + *locals.cell_argument(::core::cell::Ref::map(cell.borrow(), |b| &**b), &7) as usize
            + (y.rel2() & 0)
    }
    pub fn forwarded() -> usize {
        use crate::m::{Locals, RelTr};
        let wrapper = Wrapper(crate::m::Inner);
        let inner = crate::m::Inner;
        let mut plain = crate::m::Plain(3);
        let boxed: Box<dyn Locals> = Box::new(Wrapper(crate::m::Inner));
        let mut total = 0;
        {
            let x: &dyn RelTr = &plain;
            total += through_dyn(&wrapper, x, &plain) + through_dyn(&inner, x, &plain);
            total += boxed.by_ref(x) + wrapper.several(x, &plain);
        }
        total += wrapper.by_mut(&mut plain);
        let cell = ::core::cell::RefCell::new(Box::new(plain) as Box<dyn RelTr>);
        total += *wrapper.cell_bound(::core::cell::Ref::map(cell.borrow(), |b| &**b), &1) as usize;
        total
    }
    // The methods mentioning `Self` are given with the types of the Leaker (the forwarded value
    // would be the field's type), and their signatures are copied by the macro.
    crate::m::SelfOutlives!(copy [] {
        impl crate::m::SelfOutlives for Wrapper {
            type Item = u8;
            fn me<'a>(&'a self, x: &'a Rel2<Self>) -> usize
            where
                Self: Sized,
            {
                let _ = x;
                1
            }
            fn it<'a>(&'a self, x: &'a Rel2<Self::Item>) -> usize
            where
                Self: Sized,
            {
                x.0 as usize
            }
            fn arg<'a>(&self, x: &'a Rel2<Self>) -> usize
            where
                Self: Sized,
            {
                let _ = x;
                0
            }
        }
    });
    // A free function and a struct, where `Self` is a type parameter: the reference implies
    // `Self: 'a` (the encoding of the entry has it as `&'a Self`).
    crate::m::SelfOutlives!(copy [S] {
        pub fn free<'a, S: crate::m::SelfOutlives + Sized>(_x: &'a Rel2<Self>, _y: &'a Rel2<Self::Item>) {}
        pub struct Holder<'a, S: crate::m::SelfOutlives + Sized> {
            pub me: &'a Rel2<Self>,
            pub it: &'a Rel2<Self::Item>,
        }
    });
    // `Self: 'a` of a type leaked alone: `Self` is the type parameter. The reference implies it
    // (`by_ref`), and the path writes it (`where Self: 'a`, E0309 for the field without).
    crate::m::SelfAround!(copy [S] {
        pub struct SelfHolder<'a, S: crate::m::SelfAround + Sized>
        where
            Self: 'a,
        {
            pub cell: ::core::cell::Ref<'a, Rel2<Self>>,
            pub by_ref: &'a Rel2<Self>,
        }
        pub fn self_free<'a, S: crate::m::SelfAround + Sized>(
            _x: ::core::cell::Ref<'a, Rel2<Self>>,
            _y: &'a Rel2<Self>,
        ) where
            Self: 'a,
        {
        }
    });
    crate::m::SelfBound!(copy [] {
        impl crate::m::SelfBound for Wrapper {
            type Item = u8;
            fn me<'a>(&'a self) -> Rel<'a, Self>
            where
                Self: Sized + 'a,
            {
                crate::m::Rel(self)
            }
            fn it<'a>(&'a self) -> Rel<'a, Self::Item>
            where
                Self: Sized + 'a,
            {
                crate::m::Rel(&1)
            }
            fn split<'a>(&self, x: Rel<'a, Self>) -> usize
            where
                Self: Sized,
                Self: 'a,
            {
                let _ = x;
                0
            }
            fn implied<'a>(&self, x: &'a Rel<'a, Self>) -> usize
            where
                Self: Sized,
            {
                let _ = x;
                0
            }
        }
    });
    // The same entries in a free function and the field of a struct, where `Self` is a type
    // parameter: the bound is written as a predicate on `Self`, as the entry has it (`Rel<'a,
    // Self>` is not assumed to imply it), and it is the one the original would need.
    crate::m::SelfBound!(copy [S] {
        pub fn written_free<'a, S: crate::m::SelfBound + Sized>(
            _x: Rel<'a, Self>,
            _y: Rel<'a, Self::Item>,
        ) where
            Self: 'a,
        {
        }
        pub struct WrittenHolder<'a, S: crate::m::SelfBound + Sized>
        where
            Self: 'a,
        {
            pub me: Rel<'a, Self>,
            pub it: Rel<'a, Self::Item>,
        }
    });
    // The where clause of the trait is not in the copy of the implementation, where `'a` is
    // declared by the `impl`: the copy writes `Self: 'a` again (the implementation needs it as
    // the trait does, and the copy finds the entry by it).
    crate::m::SelfBoundTrait!(copy [] {
        impl<'a> crate::m::SelfBoundTrait<'a> for Wrapper
        where
            Self: 'a,
        {
            fn me(&'a self) -> Rel<'a, Self> {
                crate::m::Rel(self)
            }
            fn arg(&self, x: Rel<'a, Self>) -> usize {
                let _ = x;
                0
            }
        }
    });
}

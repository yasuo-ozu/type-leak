//! A Leaker's crate whose leaked types have outlives bounds which are only implied by the
//! definition-site signature (RFC 2093), and the Referrer's side of them, for tests/outlives.rs.
//!
//! It does not compile if the Repeater's implementations (E0309) or the copies of the signatures
//! (E0309 as well) lack the bounds, which the encoding of the generics carries (`&'a T` in the
//! trait arguments implies `T: 'a`).

#[doc(hidden)]
pub use type_leak_test_macro as __macro;

type_leak_test_macro::define_repeater!(pub Repeater);

pub mod m {
    use type_leak_test_macro::{check_leak, leak};

    /// `T: 'a` is inferred from the field (RFC 2093), so it is an implied bound wherever
    /// `Rel<'a, T>` is written in a signature, but it is not written anywhere.
    pub struct Rel<'a, T>(pub &'a T);

    /// The control: nothing relates `T` to `'a`, so there is no implied bound to lose.
    pub struct Loose<'a, T>(pub &'a (), pub T);

    /// A relative type which is not generic over a lifetime: `&'a Rel2<T>` and `::core::cell::Ref<'a, Rel2<T>>`
    /// are well-formed only if `Rel2<T>: 'a`, i.e. `T: 'a`.
    pub struct Rel2<T>(pub T);

    /// A relative type without generics.
    pub struct Plain(pub u8);

    pub trait RelTr {
        fn rel(&self) -> usize;
    }

    impl RelTr for Plain {
        fn rel(&self) -> usize {
            self.0 as usize
        }
    }

    /// Leaked types whose generics are those of the methods.
    #[leak(
        helper = crate::__macro,
        repeater = crate::Repeater,
        module = crate::m,
    )]
    pub trait Bounds {
        /// `T: 'a` is an implied bound of the signature. The Repeater's implementation for
        /// `Rel<'a, T>` is `impl<'a, T> Repeater<(&'a (), T, &'a T), ..> for S`, and the copy of
        /// the signature has the same encoding, which implies `T: 'a` there as well.
        fn by_ref<'a, T>(&self, x: Rel<'a, T>) -> usize;
        /// `Loose` is assumed to require `T: 'a` like `Rel` (we cannot see its definition), so the
        /// copy is stricter than the original here (a caller with a `T` shorter than `'a` is
        /// rejected where `'a` cannot be shortened), but it compiles. It would not if the
        /// implementation declared `T: 'a` as well (E0309 at the copy, where nothing implies it),
        /// so the macro declares the generics without the lifetime bounds that are encoded.
        fn loose<'a, T>(&self, x: Loose<'a, T>) -> usize;
        /// A bound which is written, and one which the type implies.
        fn written<'a, 'b: 'a, T: 'b>(&self, x: Rel<'a, &'b T>) -> usize;
        /// A bound to a lifetime which is not a parameter is not encoded.
        fn forever<T: 'static>(&self, x: Rel<'static, T>) -> usize;
    }

    /// The same with the generics of the trait: they are the generics of the implementation as
    /// well, so `T: 'a` is lost in the same way.
    #[leak(
        helper = crate::__macro,
        repeater = crate::Repeater,
        module = crate::m,
    )]
    pub trait Outlives<'a, T> {
        fn field(&self, x: Rel<'a, T>) -> usize;
        /// The generics of the trait and of the method together.
        fn mixed<'b, U>(&self, x: Rel<'b, (&'a T, U)>) -> usize;
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
        fn local<T>(&self, x: Rel<'a, T>) -> usize;
        /// The other way round: the type is a generic of the trait.
        fn lifetime<'b>(&self, x: Rel<'b, &'a u8>) -> usize;
    }

    /// References and paths with a lifetime around a relative type: a reference is leaked as a
    /// whole if its lifetime is a generic of the trait (`&'a Rel2<T>`, with `T: 'a`), and the
    /// relative type alone otherwise, as for a path (`::core::cell::Ref<'a, Rel2<T>>`). In the
    /// latter case `Rel2<T>` has no bound, and the copy `Ref<'a, <Self as Repeater<(T,),
    /// ..>>::Type>` compiles in a method signature (the original's implied bound is implied there
    /// as well), but would not in the field of a struct (E0309): the bounds of the types around a
    /// leaked type are not passed to it.
    #[leak(
        helper = crate::__macro,
        repeater = crate::Repeater,
        module = crate::m,
    )]
    pub trait Descend<'a, T> {
        /// Leaked as a whole, with `T: 'a`.
        fn by_ref(&self, x: &'a Rel2<T>) -> usize;
        /// `Rel2<T>` alone is leaked, with no bound.
        fn cell(&self, x: ::core::cell::Ref<'a, Rel2<T>>) -> usize;
        fn local_ref<'b>(&self, x: &'b Rel2<T>) -> usize;
        fn local_cell<'b>(&self, x: ::core::cell::Ref<'b, Rel2<T>>) -> usize;
        /// A trait object in a reference needs nothing of its arguments to outlive the reference,
        /// so `Rel2<T>` is leaked alone (with no bound), unlike `&'a Rel2<T>`.
        fn dyn_ref(&self, x: &'a dyn ::core::iter::Iterator<Item = Rel2<T>>) -> usize;
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
        /// The trait object cannot be leaked alone (its trait is relative, and its lifetime bound
        /// would be `'static`), so the reference is.
        fn whole_ref(&self, x: &'a dyn RelTr) -> usize;
        /// A bare identifier in the arguments may be a const, which cannot be leaked alone, so
        /// the reference is leaked as a whole as well.
        fn bare_ref(&self, x: &'a dyn ::core::borrow::Borrow<Plain>) -> usize;
        /// With a reference to a trait object inside, which is transparent as well.
        fn nested(
            &self,
            x: &'a dyn ::core::iter::Iterator<Item = &'a dyn ::core::iter::Iterator<Item = Plain>>,
        ) -> usize;
    }

    // The leaked types, checked when this crate is compiled.
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
    check_leak! {
        { &'a dyn RelTr }
        { }
        trait RefWhole<'a> {
            fn whole_ref(&self, x: &'a dyn RelTr) -> usize;
        }
    }
    check_leak! {
        { &'a dyn ::core::borrow::Borrow<Plain> }
        { }
        trait RefBare<'a> {
            fn bare_ref(&self, x: &'a dyn ::core::borrow::Borrow<Plain>) -> usize;
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

    pub struct Inner;

    impl Bounds for Inner {
        fn by_ref<'a, T>(&self, x: Rel<'a, T>) -> usize {
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

    impl<'a, T> Outlives<'a, T> for Inner {
        fn field(&self, x: Rel<'a, T>) -> usize {
            core::mem::size_of_val(x.0)
        }
        fn mixed<'b, U>(&self, _x: Rel<'b, (&'a T, U)>) -> usize {
            2
        }
    }

    impl<'a> Local<'a> for Inner {
        fn local<T>(&self, x: Rel<'a, T>) -> usize {
            core::mem::size_of_val(x.0)
        }
        fn lifetime<'b>(&self, _x: Rel<'b, &'a u8>) -> usize {
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
}

/// The Referrer's side: implements the traits by forwarding to `Inner`, without importing any of
/// the types relative to `crate::m`. The copies of the signatures are compiled here.
pub mod user {
    pub struct Wrapper(pub crate::m::Inner);

    crate::m::Bounds!(impl Bounds for Wrapper => 0);
    crate::m::Outlives!(impl<'a, T> Outlives<'a, T> for Wrapper => 0);
    crate::m::Local!(impl<'a> Local<'a> for Wrapper => 0);
    crate::m::Descend!(impl<'a, T> Descend<'a, T> for Wrapper => 0);
    crate::m::Dyns!(impl<'a> Dyns<'a> for Wrapper => 0);
}

//! A Leaker's crate whose leaked types have outlives bounds which are only implied by the
//! definition-site signature (RFC 2093), and the Referrer's side of them, for tests/outlives.rs.
//!
//! It does not compile if the Repeater's implementations (E0309) or the copies of the signatures
//! (E0309 as well) lack the bounds, which the encoding of the generics carries (`&'a T` in the
//! trait arguments implies `T: 'a`).
//!
//! It does not compile either if a type is leaked with the lifetime of a `for<..>` binder whose
//! output mentions it and no input keeps it (E0581 for a function pointer, E0582 for a bound or a
//! trait object), see `Binders`, nor if `Self: 'a` is not encoded (E0309 for `__S`), see
//! `SelfOutlives`, nor if a type leaked alone does not have the bounds which the types around it
//! imply (E0309 in the field of a struct), see `Around`.

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
    /// latter case `Rel2<T>` has the bounds which the type around it implies (`T: 'a`, see
    /// `Around`), so the copy `Ref<'a, <Self as Repeater<(&'a (), T, &'a T), ..>>::Type>` is
    /// well-formed in the field of a struct too, whereas the copy of a method signature would be
    /// without (the original's implied bound is implied there as well). The same type found
    /// without a type around it (`plain`) is another entry, without the bound.
    #[leak(
        helper = crate::__macro,
        repeater = crate::Repeater,
        module = crate::m,
    )]
    pub trait Descend<'a, T> {
        /// Leaked as a whole, with `T: 'a`.
        fn by_ref(&self, x: &'a Rel2<T>) -> usize;
        /// `Rel2<T>` alone is leaked, with the bound `T: 'a` which `Ref<'a, ..>` implies.
        fn cell(&self, x: ::core::cell::Ref<'a, Rel2<T>>) -> usize;
        fn local_ref<'b>(&self, x: &'b Rel2<T>) -> usize;
        fn local_cell<'b>(&self, x: ::core::cell::Ref<'b, Rel2<T>>) -> usize;
        /// A trait object in a reference requires its arguments to outlive the reference as well
        /// (`dyn Iterator<Item = X> + 'a: 'a` requires `X: 'a`), so `Rel2<T>` is leaked alone, with
        /// `T: 'a` as well, whereas `&'a Rel2<T>` is leaked as a whole.
        fn dyn_ref(&self, x: &'a dyn ::core::iter::Iterator<Item = Rel2<T>>) -> usize;
        /// The same type with nothing around it: not the entry of `cell`, which has `T: 'a` as
        /// well (the Referrer tells them apart when the generics of the trait are not declared
        /// where it finds them).
        fn plain(&self, x: Rel2<T>) -> usize;
    }

    /// The bounds of the types around a type leaked alone are passed to it, as rustc derives
    /// them: `Ref<'a, X>` is well-formed only if `X: 'a`, which the original field of a struct
    /// has by RFC 2093, but not the copy `Ref<'a, <S as Repeater<(T,), ..>>::Type>`, where nothing
    /// infers `T: 'a` through a projection (E0309). So `Rel2<T>` is leaked with the generics `<'a,
    /// T: 'a>`, which the copy of the field encodes in the arguments of the Repeater (`(&'a (), T,
    /// &'a T)`).
    ///
    /// The requirement is carried by tuples, arrays, slices, references, raw pointers, the type
    /// arguments of paths, the inputs and outputs of function pointers, and the arguments of a trait
    /// object (`dyn Tr<X> + 'b: 'a` requires `X: 'a`), but not by a projection (nothing infers
    /// `T: 'a` from `&'a <Rel2<T> as Deref>::Target` in the original either), nor across the
    /// lifetime of a `for<..>` binder or `'static`, which are not generics. See
    /// `tests/outlives_ancestors.rs` for the contexts of each.
    #[leak(
        helper = crate::__macro,
        repeater = crate::Repeater,
        module = crate::m,
    )]
    pub trait Around {
        fn cell<'a, T>(&self, x: ::core::cell::Ref<'a, Rel2<T>>) -> usize;
        fn by_ref<'a, T>(&self, x: &'a Rel2<T>) -> usize;
        fn by_mut<'a, T>(&self, x: &'a mut Rel2<T>) -> usize;
        /// Further paths: `Vec<X>` requires `X: 'a` as well.
        fn nested<'a, T>(&self, x: ::core::cell::Ref<'a, ::std::vec::Vec<Rel2<T>>>) -> usize;
        fn tuple<'a, T>(&self, x: &'a (u8, [Rel2<T>; 2])) -> usize;
        fn function<'a, T>(&self, x: &'a fn(Rel2<T>) -> usize) -> usize;
        /// A trait object: `Ref<'a, dyn Iterator<Item = X> + 'a>` requires `X: 'a`.
        fn object<'a, T>(
            &self,
            x: ::core::cell::Ref<'a, dyn ::core::iter::Iterator<Item = Rel2<T>>>,
        ) -> usize;
        /// The lifetimes are bounded too.
        fn lifetime<'a, 'c>(&self, x: &'a Rel2<&'c u8>) -> usize;
        /// Both lifetimes bound `T`.
        fn both<'a, 'b, T>(&self, x: &'a &'b Rel2<T>) -> usize;
        /// Not for `'static`: rustc infers no `T: 'static` for a struct (RFC 2093), so the
        /// original declares it, and so does the copy of the field (the Repeater's implementation
        /// declares the bound too, which is a written bound).
        fn forever<T: 'static>(&self, x: ::core::cell::Ref<'static, Rel2<T>>) -> usize;
        /// Not for the lifetime of a binder.
        fn binder<T>(&self, f: for<'x> fn(&'x Rel2<T>) -> usize) -> usize;
        /// Not through a projection.
        fn projection<'a, T>(&self, x: &'a <Rel2<T> as ::core::ops::Deref>::Target) -> usize;
        /// With nothing around it.
        fn alone<T>(&self, x: Rel2<T>) -> usize;
    }

    /// `Self: 'a` for a type leaked alone, like `T: 'a`.
    #[leak(
        helper = crate::__macro,
        repeater = crate::Repeater,
        module = crate::m,
    )]
    pub trait SelfAround {
        fn cell<'a>(&self, x: ::core::cell::Ref<'a, Rel2<Self>>) -> usize
        where
            Self: Sized;
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

    /// The lifetime `'x` of a `for<..>` binder. If the output of the binder (the return type, or an
    /// associated type binding) does not mention it, `Hr<'x>` is leaked with `'x` among its
    /// generics, and the copy `for<'x> fn(<Self as Repeater<(&'x (),), ..>>::Type) -> usize`
    /// compiles. If it does, the copy would not (E0581 for a function pointer and E0582 for a
    /// bound or a trait object: a lifetime which occurs only in a projection is not constrained by
    /// the inputs), unless an input keeps it where it is written (`&'x u8`: an anchor), so `Hr<'x>`
    /// is leaked alone again. Otherwise the function pointer or the trait object holding the binder
    /// is leaked as a whole, and `Hr<'x>` is not.
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
        fn binding(&self, f: Box<dyn for<'x> ::core::ops::Index<&'x u8, Output = Hr<'x>>>) -> usize;
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
    }

    /// `Self: 'a` is implied by `Rel<'a, Self>` and by `Rel<'a, Self::Item>`, like `T: 'a` is by
    /// `Rel<'a, T>` (RFC 2093): the Repeater's implementation is `impl<'a, __S: ?Sized +
    /// SelfOutlives> Repeater<(&'a (), &'a __S), ..> for __S where __S: Sized`, where the header
    /// implies `__S: 'a`, which it does not declare, and so do the copies, `<Self as
    /// Repeater<(&'a (), &'a Self), ..>>::Type`, in a method (where `&'a self` implies it
    /// as well), a free function and the field of a struct.
    #[leak(
        helper = crate::__macro,
        repeater = crate::Repeater,
        module = crate::m,
    )]
    pub trait SelfOutlives {
        type Item: Sized;
        fn me<'a>(&'a self) -> Rel<'a, Self>
        where
            Self: Sized;
        fn it<'a>(&'a self) -> Rel<'a, Self::Item>
        where
            Self: Sized;
        /// Without a receiver of the lifetime, so the bound is implied by the encoding only.
        fn arg<'a>(&self, x: Rel<'a, Self>) -> usize
        where
            Self: Sized;
    }

    /// `Self: 'a` written in the where clause as well: it is a bound of the signature, so the
    /// Repeater's implementation declares it (`where __S: Sized + 'a`) besides the encoding, and
    /// the copies of the signature, where the bound is written too, compile.
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
        { }
        trait Outputs {
            fn whole(&self, f: for<'x> fn(Hr<'x>) -> &'x u8) -> usize;
            fn object(&self, f: ::std::boxed::Box<dyn for<'x> ::core::ops::Fn(Hr<'x>) -> &'x u8>) -> usize;
        }
    }
    // An input keeps `'x` (an anchor), so only `Hr<'x>` is leaked.
    check_leak! {
        { Hr<'x> }
        { }
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
    // A bound, or a function pointer which cannot be leaked as a whole (it has an elided lifetime),
    // is an error, as the copy of `Hr<'x>` would not compile.
    check_leak! {
        { error: "use absolute path: 'Hr < 'x >'" }
        { }
        trait BoundOutput {
            fn f<F: for<'x> ::core::ops::Fn(Hr<'x>) -> &'x u8>(&self, f: F) -> usize;
        }
    }
    check_leak! {
        { error: "use absolute path: 'Hr < 'x >'" }
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
        fn plain(&self, _x: Rel2<T>) -> usize {
            11
        }
    }

    impl Around for Inner {
        fn cell<'a, T>(&self, _x: ::core::cell::Ref<'a, Rel2<T>>) -> usize {
            1
        }
        fn by_ref<'a, T>(&self, _x: &'a Rel2<T>) -> usize {
            2
        }
        fn by_mut<'a, T>(&self, _x: &'a mut Rel2<T>) -> usize {
            3
        }
        fn nested<'a, T>(&self, x: ::core::cell::Ref<'a, ::std::vec::Vec<Rel2<T>>>) -> usize {
            x.len()
        }
        fn tuple<'a, T>(&self, x: &'a (u8, [Rel2<T>; 2])) -> usize {
            x.0 as usize
        }
        fn function<'a, T>(&self, _x: &'a fn(Rel2<T>) -> usize) -> usize {
            5
        }
        fn object<'a, T>(
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
    }

    impl SelfAround for Inner {
        fn cell<'a>(&self, _x: ::core::cell::Ref<'a, Rel2<Self>>) -> usize {
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
        fn binding(&self, f: Box<dyn for<'x> ::core::ops::Index<&'x u8, Output = Hr<'x>>>) -> usize {
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
    }

    impl SelfOutlives for Inner {
        type Item = u8;
        fn me<'a>(&'a self) -> Rel<'a, Self>
        where
            Self: Sized,
        {
            Rel(self)
        }
        fn it<'a>(&'a self) -> Rel<'a, Self::Item>
        where
            Self: Sized,
        {
            Rel(&1)
        }
        fn arg<'a>(&self, x: Rel<'a, Self>) -> usize
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
    crate::m::Binders!(impl Binders for Wrapper => 0);
    crate::m::Around!(impl Around for Wrapper => 0);
    // The copies of the types around `Rel2<T>` as the fields of structs and a free function, where
    // the bounds which the original has by RFC 2093 are not implied unless the context of the leaked
    // type carries them (E0309 otherwise). `Wrapper` is `Self`, but nothing mentions it.
    crate::m::Around!(copy [Wrapper] {
        pub struct Cell<'a, T>(pub ::core::cell::Ref<'a, Rel2<T>>);
        pub struct ByRef<'a, T>(pub &'a Rel2<T>);
        pub struct ByMut<'a, T>(pub &'a mut Rel2<T>);
        pub struct Nested<'a, T>(pub ::core::cell::Ref<'a, ::std::vec::Vec<Rel2<T>>>);
        pub struct Tuple<'a, T>(pub &'a (u8, [Rel2<T>; 2]));
        pub struct Function<'a, T>(pub &'a fn(Rel2<T>) -> usize);
        pub struct Object<'a, T>(
            pub ::core::cell::Ref<'a, dyn ::core::iter::Iterator<Item = Rel2<T>>>,
        );
        pub struct Lifetime<'a, 'c>(pub &'a Rel2<&'c u8>);
        pub struct Both<'a, 'b, T>(pub &'a &'b Rel2<T>);
        pub struct Forever<T: 'static>(pub ::core::cell::Ref<'static, Rel2<T>>);
        pub struct Binder<T>(pub for<'x> fn(&'x Rel2<T>) -> usize);
        pub struct Projection<'a, T>(pub &'a <Rel2<T> as ::core::ops::Deref>::Target);
        pub struct Alone<T>(pub Rel2<T>);
        pub fn around_free<'a, T>(_x: ::core::cell::Ref<'a, Rel2<T>>, _y: &'a Rel2<T>) {}
        pub enum Choice<'a, T> {
            Cell(::core::cell::Ref<'a, Rel2<T>>),
            Alone(Rel2<T>),
        }
    });
    // The methods mentioning `Self` are given with the types of the Leaker (the forwarded value
    // would be the field's type), and their signatures are copied by the macro.
    crate::m::SelfOutlives!(copy [] {
        impl crate::m::SelfOutlives for Wrapper {
            type Item = u8;
            fn me<'a>(&'a self) -> Rel<'a, Self>
            where
                Self: Sized,
            {
                crate::m::Rel(self)
            }
            fn it<'a>(&'a self) -> Rel<'a, Self::Item>
            where
                Self: Sized,
            {
                crate::m::Rel(&1)
            }
            fn arg<'a>(&self, x: Rel<'a, Self>) -> usize
            where
                Self: Sized,
            {
                let _ = x;
                0
            }
        }
    });
    // A free function and a struct, where `Self` is a type parameter.
    crate::m::SelfOutlives!(copy [S] {
        pub fn free<'a, S: crate::m::SelfOutlives + Sized>(_x: Rel<'a, Self>, _y: Rel<'a, Self::Item>) {}
        pub struct Holder<'a, S: crate::m::SelfOutlives + Sized> {
            pub me: Rel<'a, Self>,
            pub it: Rel<'a, Self::Item>,
        }
    });
    // `Self: 'a` of a type leaked alone: `Self` is the type parameter.
    crate::m::SelfAround!(copy [S] {
        pub struct SelfHolder<'a, S: crate::m::SelfAround + Sized> {
            pub cell: ::core::cell::Ref<'a, Rel2<Self>>,
            pub by_ref: &'a Rel2<Self>,
        }
        pub fn self_free<'a, S: crate::m::SelfAround + Sized>(
            _x: ::core::cell::Ref<'a, Rel2<Self>>,
            _y: &'a Rel2<Self>,
        ) {
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
        }
    });
}

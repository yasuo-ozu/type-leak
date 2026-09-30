//! The Leaker's crate: traits whose signatures use types relative to `mod m`, which the
//! Referrer's crate (`type-leak-test-referrer`) cannot name.

#[doc(hidden)]
pub use type_leak_test_macro as __macro;

/// The Repeater. Its type parameter carries the encoded generics (including local generics).
pub trait Repeater<T: ?Sized, const ID: u64, const NTH: usize> {
    type Type: ?Sized;
}

/// Marker types of `type_leak::ConstMarkers`.
pub mod markers {
    type_leak_test_macro::const_markers!();
}

pub mod m {
    use type_leak_test_macro::leak;

    pub struct RelType(pub u32);

    pub struct Flag<const B: bool>;
    impl<const B: bool> Flag<B> {
        pub fn get(&self) -> bool {
            B
        }
    }

    pub struct Letter<const C: char>;
    impl<const C: char> Letter<C> {
        pub fn get(&self) -> char {
            C
        }
    }

    pub struct Holder<'a>(pub &'a u32);

    /// Const generics of `usize`, `bool`, `char` and `i32`, and method-local generics.
    #[leak(
        helper = crate::__macro,
        repeater = crate::Repeater,
        markers = crate::markers,
        module = crate::m,
    )]
    pub trait Consts<T, const N: usize, const B: bool, const C: char, const X: i32> {
        /// A relative type.
        fn relative(&self) -> RelType;
        /// The Leaker's own const parameter used in an expression.
        fn array(&self) -> [RelType; N];
        /// Const parameters of non-`usize` types.
        fn flags(&self) -> (Flag<B>, Letter<C>);
        /// A type parameter and a const parameter used directly.
        fn value(&self, t: T) -> (T, i32);
        /// A relative type constructor applied to a local const generic.
        fn local_flag<const M: bool>(&self) -> Flag<M>;
        /// A type mentioning a local const generic, decomposed into `RelType`.
        fn local_array<const M: usize>(&self) -> [RelType; M];
        /// A relative type constructor applied to a local lifetime.
        fn holder<'b>(&self, h: Holder<'b>) -> u32;
    }

    /// Implements [`Consts`] for any arguments; the Referrer's wrappers forward to it.
    pub struct Inner;

    impl<T, const N: usize, const B: bool, const C: char, const X: i32> Consts<T, N, B, C, X>
        for Inner
    {
        fn relative(&self) -> RelType {
            RelType(42)
        }
        fn array(&self) -> [RelType; N] {
            let mut i = 0;
            [(); N].map(|()| {
                i += 1;
                RelType(i)
            })
        }
        fn flags(&self) -> (Flag<B>, Letter<C>) {
            (Flag, Letter)
        }
        fn value(&self, t: T) -> (T, i32) {
            (t, X)
        }
        fn local_flag<const M: bool>(&self) -> Flag<M> {
            Flag
        }
        fn local_array<const M: usize>(&self) -> [RelType; M] {
            [(); M].map(|()| RelType(M as u32))
        }
        fn holder<'b>(&self, h: Holder<'b>) -> u32 {
            *h.0
        }
    }
}

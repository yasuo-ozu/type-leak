//! The Leaker's crate: traits whose signatures use types relative to `mod m`, which the
//! Referrer's crate (`type-leak-test-referrer`) cannot name.

#[doc(hidden)]
pub use type_leak_test_macro as __macro;

// The Repeater, with const slots to encode const generics of any primitive type.
type_leak_test_macro::define_repeater!(pub Repeater);

/// Defines a leaked trait using `$crate` paths, which are real tokens since this is a
/// `macro_rules!`: they resolve to this crate, also from the Referrer's crate.
macro_rules! define_dollar_crate_trait {
    () => {
        #[leak(
            helper = crate::__macro,
            repeater = crate::Repeater,
            module = crate::m,
        )]
        pub trait DollarCrate: $crate::m_portable::PortTr {
            /// A `$crate` type in the return type.
            fn d_ret(&self) -> $crate::m_portable::Portable;
            /// A `$crate` type constructor applied to a primitive.
            fn d_arg(&self, a: $crate::m_portable::Wrap<u8>) -> u8;
            /// A `$crate` value in an array length.
            fn d_arr(&self) -> [u8; $crate::m_portable::LEN];
            /// A `$crate` bound, with a relative type, which is leaked.
            fn d_rel<T: $crate::m_portable::PortTr>(&self, t: T) -> RelType;
        }
    };
}

pub mod m {
    use type_leak_test_macro::leak;

    define_dollar_crate_trait!();

    impl crate::m_portable::PortTr for Inner {}

    impl DollarCrate for Inner {
        fn d_ret(&self) -> crate::m_portable::Portable {
            crate::m_portable::Portable
        }
        fn d_arg(&self, a: crate::m_portable::Wrap<u8>) -> u8 {
            a.0 + 1
        }
        fn d_arr(&self) -> [u8; crate::m_portable::LEN] {
            [7; crate::m_portable::LEN]
        }
        fn d_rel<T: crate::m_portable::PortTr>(&self, _t: T) -> RelType {
            RelType(11)
        }
    }

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

    /// Defaulted const parameters, which the Referrer may omit.
    #[leak(
        helper = crate::__macro,
        repeater = crate::Repeater,
        module = crate::m,
    )]
    pub trait Defaults<T, const N: usize = 2, const B: bool = true> {
        fn default_array(&self) -> [RelType; N];
        fn default_flag(&self) -> Flag<B>;
    }

    /// The default of a type parameter, which is relative to this module.
    #[derive(Debug, PartialEq)]
    pub struct RelDefault<T>(pub T);

    /// A defaulted type parameter whose default is a relative type, which the Referrer may omit.
    #[leak(
        helper = crate::__macro,
        repeater = crate::Repeater,
        module = crate::m,
    )]
    pub trait Defaulted<T, U = RelDefault<T>> {
        fn get(&self, t: T) -> U;
    }

    pub struct Pair<V>(pub V, pub V);

    /// Where clauses, typed receivers, argument patterns, HRTB, elided lifetimes and RPITIT.
    #[leak(
        helper = crate::__macro,
        repeater = crate::Repeater,
        module = crate::m,
    )]
    pub trait Edge<T>
    where
        T: ::core::clone::Clone,
    {
        /// Outer and local where predicates.
        fn e_where<V>(&self, v: V) -> Pair<V>
        where
            V: ::core::marker::Copy;
        /// A leaked type in a where clause.
        fn e_where_rel<U>(&self, u: U) -> u32
        where
            U: ::core::convert::Into<RelType>;
        fn e_box(self: Box<Self>) -> RelType;
        fn e_ref(self: &Self) -> RelType;
        /// Argument patterns.
        fn e_pat(&self, _: RelType, mut x: u32) -> u32 {
            x += 1;
            x
        }
        /// A higher-ranked lifetime.
        fn e_hrtb<F: for<'a> ::core::ops::Fn(&'a RelType) -> u32>(&self, f: F) -> u32;
        /// An elided lifetime.
        fn e_elided(&self, x: &RelType) -> u32;
        /// A return type `impl Trait`.
        fn e_impl(&self) -> impl ::core::iter::Iterator<Item = RelType>;
    }

    /// A trait without generic parameters.
    #[leak(
        helper = crate::__macro,
        repeater = crate::Repeater,
        module = crate::m,
    )]
    pub trait Plain {
        fn p(&self) -> RelType;
    }

    /// Implements the traits above for any arguments; the Referrer's wrappers forward to it.
    pub struct Inner;

    impl<T: ::core::clone::Clone> Edge<T> for Inner {
        fn e_where<V>(&self, v: V) -> Pair<V>
        where
            V: ::core::marker::Copy,
        {
            Pair(v, v)
        }
        fn e_where_rel<U>(&self, u: U) -> u32
        where
            U: ::core::convert::Into<RelType>,
        {
            u.into().0
        }
        fn e_box(self: Box<Self>) -> RelType {
            RelType(1)
        }
        fn e_ref(self: &Self) -> RelType {
            RelType(2)
        }
        fn e_hrtb<F: for<'a> ::core::ops::Fn(&'a RelType) -> u32>(&self, f: F) -> u32 {
            f(&RelType(3))
        }
        fn e_elided(&self, x: &RelType) -> u32 {
            x.0
        }
        fn e_impl(&self) -> impl ::core::iter::Iterator<Item = RelType> {
            ::core::iter::once(RelType(4))
        }
    }

    impl Plain for Inner {
        fn p(&self) -> RelType {
            RelType(5)
        }
    }

    impl From<u32> for RelType {
        fn from(n: u32) -> Self {
            RelType(n)
        }
    }

    impl<T> Defaulted<T, RelDefault<T>> for Inner {
        fn get(&self, t: T) -> RelDefault<T> {
            RelDefault(t)
        }
    }

    impl Defaulted<u8, u32> for Inner {
        fn get(&self, t: u8) -> u32 {
            t as u32 + 100
        }
    }

    impl<T, const N: usize, const B: bool> Defaults<T, N, B> for Inner {
        fn default_array(&self) -> [RelType; N] {
            [(); N].map(|()| RelType(0))
        }
        fn default_flag(&self) -> Flag<B> {
            Flag
        }
    }

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

/// Items reached through `$crate`, which are portable (resolved from the Referrer's crate).
pub mod m_portable {
    pub struct Portable;
    pub struct Wrap<T>(pub T);
    pub trait PortTr {}
    impl PortTr for Portable {}
    pub const LEN: usize = 2;
    pub type Idx = usize;
}

/// Exact results of the Leaker for paths starting with `$crate`. They are checked when this
/// crate is compiled: `check_leak!` expands to a `compile_error!` on a mismatch. The `$crate`
/// tokens are real, since they come from a `macro_rules!` expansion.
///
/// (`$crate::m::..` below stands for any item reached through the Leaker's crate; the items
/// are not resolved by the checks.)
macro_rules! check_dollar_crate {
    () => {
        use type_leak_test_macro::check_leak;

        // Types, generic arguments (still analyzed), bounds, supertraits and values.
        check_leak! {
            { }
            { }
            trait T1 { fn f(&self) -> $crate::m::Portable; }
        }
        // A generic argument which is relative makes the whole type leaked, as it does for
        // `::core::option::Option<Rel>`.
        check_leak! {
            { $crate::m::Wrap<Rel> }
            { }
            trait T2 { fn g(&self) -> $crate::m::Wrap<Rel>; }
        }
        check_leak! {
            { }
            { }
            trait T3<T: $crate::m::PortTr>: $crate::m::PortTr {
                fn h<U: $crate::m::PortTr>(&self, u: U) -> T;
            }
        }
        check_leak! {
            { }
            { }
            trait T4 { fn a(&self) -> [u8; $crate::m::LEN]; }
        }
        check_leak! {
            { }
            { }
            trait T5<const N: $crate::m::Idx> { fn a(&self) -> [u8; N]; }
        }
        check_leak! {
            { }
            { }
            struct S1<const N: $crate::m::Idx>([u8; N], $crate::m::Portable);
        }

        // `$crate` paths are portable even if external crates are not allowed, unlike the
        // other absolute paths.
        check_leak! {
            { }
            { allow_external_crates = false }
            trait N1 { fn f(&self) -> $crate::m::Portable; }
        }
        check_leak! {
            { $crate::m::Wrap<Rel> }
            { allow_external_crates = false }
            trait N2 { fn g(&self) -> $crate::m::Wrap<Rel>; }
        }
        check_leak! {
            { }
            { allow_external_crates = false }
            trait N3<T: $crate::m::PortTr>: $crate::m::PortTr {}
        }
        check_leak! {
            { }
            { allow_external_crates = false }
            trait N4<const N: $crate::m::Idx> { fn a(&self) -> [u8; $crate::m::LEN]; }
        }
        check_leak! {
            { }
            { allow_external_crates = true }
            trait E1 { fn f(&self) -> ::foo::X; }
        }
        // A type of another crate is leaked (requiring that crate), instead of being an error.
        check_leak! {
            { ::foo::X }
            { allow_external_crates = false }
            trait E2 { fn f(&self) -> ::foo::X; }
        }
        check_leak! {
            { error: "use a path starting with `$crate`: ':: foo :: Tr'" }
            { allow_external_crates = false }
            trait E3<T: ::foo::Tr> {}
        }

        check_leak! {
            { ::core::option::Option<Rel> }
            { }
            trait C1 { fn g(&self) -> ::core::option::Option<Rel>; }
        }
        // Modules: `use $crate::..` is portable.
        check_leak! {
            { Rel }
            { }
            mod mm {
                use $crate::m::Portable;
                pub struct S { a: Portable, b: Rel }
            }
        }
        check_leak! {
            { Rel }
            { allow_external_crates = false }
            mod mm {
                use $crate::m::Portable;
                pub struct S { a: Portable, b: Rel }
            }
        }

        // A relative path next to it is still an error.
        check_leak! {
            { error: "use absolute path: 'RelTrait'" }
            { }
            trait R1<T: RelTrait> { fn f(&self) -> $crate::m::Portable; }
        }
    };
}

mod dollar_crate_checks {
    check_dollar_crate!();
}

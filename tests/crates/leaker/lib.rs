//! The Leaker's crate: traits whose signatures use types relative to `mod m`, which the
//! Referrer's crate (`type-leak-test-referrer`) cannot name.

// The Repeater's implementations copy the types as they are written, including the parentheses
// of `(dyn RelTr)` in `s::PathDyn`.
#![allow(unused_parens)]

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

    /// Types of external crates (`::std`), which the Leaker does not allow: they are optional
    /// entries, whose Repeater's implementations are written like the others. The Referrer
    /// leaves them as written, or replaces them (`strict`), with the relative types in them
    /// replaced either way.
    #[leak(
        helper = crate::__macro,
        repeater = crate::Repeater,
        module = crate::m,
    )]
    pub trait External<T> {
        /// An external type with a relative one in it: `RelType` is required, and the `Vec` is
        /// optional.
        fn wrapped(&self) -> ::std::vec::Vec<(RelType,)>;
        /// The generics of the trait in it.
        fn generic(&self, t: ::std::vec::Vec<T>) -> usize;
        /// A lifetime of the method, behind a reference (kept as written in either).
        fn borrowed<'a>(&self, x: &'a ::std::string::String) -> &'a str;
        /// One optional type around a required one, around another optional one.
        fn nested(&self) -> ::std::boxed::Box<(::std::vec::Vec<u8>, RelType)>;
        /// The type of a receiver is never replaced.
        fn receiver(self: ::std::boxed::Box<Self>) -> usize;
        /// A trait object of an external crate.
        fn dynamic(&self, f: ::std::boxed::Box<dyn ::core::ops::Fn(RelType) -> u32>) -> u32;
    }

    /// Implements the traits above for any arguments; the Referrer's wrappers forward to it.
    pub struct Inner;

    impl<T> External<T> for Inner {
        fn wrapped(&self) -> ::std::vec::Vec<(RelType,)> {
            vec![(RelType(1),), (RelType(2),)]
        }
        fn generic(&self, t: ::std::vec::Vec<T>) -> usize {
            t.len()
        }
        fn borrowed<'a>(&self, x: &'a ::std::string::String) -> &'a str {
            &x[1..]
        }
        fn nested(&self) -> ::std::boxed::Box<(::std::vec::Vec<u8>, RelType)> {
            ::std::boxed::Box::new((vec![3, 4], RelType(5)))
        }
        fn receiver(self: ::std::boxed::Box<Self>) -> usize {
            6
        }
        fn dynamic(&self, f: ::std::boxed::Box<dyn ::core::ops::Fn(RelType) -> u32>) -> u32 {
            f(RelType(7))
        }
    }

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

/// Types which mention `Self`, an associated type of it, or the generics of the trait (a lifetime
/// and a type with a projection), in a module of their own.
pub mod s {
    use type_leak_test_macro::leak;

    pub use crate::m::Inner;

    #[derive(Debug, PartialEq)]
    pub struct Wrap<U>(pub U);

    pub trait RelTr {
        fn rel(&self) -> u32;
    }

    pub struct RelImpl;

    impl RelTr for RelImpl {
        fn rel(&self) -> u32 {
            31
        }
    }

    /// `Self` in a leaked type, with a bound `Self: Sized` in a where clause.
    #[leak(
        helper = crate::__macro,
        repeater = crate::Repeater,
        module = crate::s,
    )]
    pub trait SelfSized {
        fn me(&self) -> Wrap<Self>
        where
            Self: ::core::marker::Sized;
        fn plain(&self) -> Wrap<u8>;
    }

    impl SelfSized for Inner {
        fn me(&self) -> Wrap<Self> {
            Wrap(Inner)
        }
        fn plain(&self) -> Wrap<u8> {
            Wrap(1)
        }
    }

    /// An associated type of `Self` in leaked types, also in a bound and as a GAT.
    #[leak(
        helper = crate::__macro,
        repeater = crate::Repeater,
        module = crate::s,
    )]
    pub trait Assoc {
        type Item;
        type G<'a, V: 'a>: ::core::marker::Sized
        where
            Self: 'a;
        fn item(&self) -> Wrap<Self::Item>;
        fn u<U: ::core::convert::Into<Self::Item>>(&self, u: U) -> Wrap<U>;
        fn gat<'a, V: 'a>(&'a self, v: &'a V) -> Wrap<Self::G<'a, V>>
        where
            Self: 'a;
    }

    impl Assoc for Inner {
        type Item = u8;
        type G<'a, V: 'a> = &'a V;
        fn item(&self) -> Wrap<Self::Item> {
            Wrap(8)
        }
        fn u<U: ::core::convert::Into<Self::Item>>(&self, u: U) -> Wrap<U> {
            Wrap(u)
        }
        fn gat<'a, V: 'a>(&'a self, v: &'a V) -> Wrap<Self::G<'a, V>>
        where
            Self: 'a,
        {
            Wrap(v)
        }
    }

    /// A lifetime parameter of the trait in a leaked type.
    #[leak(
        helper = crate::__macro,
        repeater = crate::Repeater,
        module = crate::s,
    )]
    pub trait Out<'a, T> {
        fn f(&self, x: &'a T) -> Wrap<&'a T>;
    }

    impl<'a, T> Out<'a, T> for Inner {
        fn f(&self, x: &'a T) -> Wrap<&'a T> {
            Wrap(x)
        }
    }

    /// A trait object of a relative trait with the lifetime of the trait, boxed and borrowed.
    #[leak(
        helper = crate::__macro,
        repeater = crate::Repeater,
        module = crate::s,
    )]
    pub trait DynRoot<'a> {
        fn boxed(&self) -> ::std::boxed::Box<dyn RelTr + 'a>;
        fn borrowed(&self, x: &'a dyn RelTr) -> u32;
        fn lent(&self) -> Wrap<&'a dyn RelTr>;
    }

    impl<'a> DynRoot<'a> for Inner {
        fn boxed(&self) -> ::std::boxed::Box<dyn RelTr + 'a> {
            ::std::boxed::Box::new(RelImpl)
        }
        fn borrowed(&self, x: &'a dyn RelTr) -> u32 {
            x.rel() + 1
        }
        fn lent(&self) -> Wrap<&'a dyn RelTr> {
            Wrap(&RelImpl)
        }
    }

    /// A projection of a type parameter of the trait, `<T>::Item`.
    #[leak(
        helper = crate::__macro,
        repeater = crate::Repeater,
        module = crate::s,
    )]
    pub trait Q<T: ::core::iter::Iterator> {
        fn f(&self, x: <T>::Item) -> Wrap<<T>::Item>;
    }

    impl<T: ::core::iter::Iterator> Q<T> for Inner {
        fn f(&self, x: <T>::Item) -> Wrap<<T>::Item> {
            Wrap(x)
        }
    }

    /// A trait object whose lifetime bound defaults to the lifetime of the path it is an argument
    /// of (`Ref<'a, T: 'a>`), so it must not be leaked alone (it would be `dyn RelTr + 'static`):
    /// the path is leaked as a whole if the trait object has to be (it names a relative trait),
    /// and what is inside it is leaked otherwise (the copy keeps the trait object where it is).
    #[leak(
        helper = crate::__macro,
        repeater = crate::Repeater,
        module = crate::s,
    )]
    pub trait PathDyn<'a> {
        fn path(&self, x: ::core::cell::Ref<'a, dyn RelTr>) -> u32;
        fn paren(&self, x: ::core::cell::Ref<'a, (dyn RelTr)>) -> u32;
        /// With the bound written, only the trait object is leaked.
        fn bound(&self, x: ::core::cell::Ref<'a, dyn RelTr + 'a>) -> u32;
        /// Not the argument of the path, so it defaults to `'static`.
        fn boxed(&self, x: ::core::cell::Ref<'a, ::std::boxed::Box<dyn RelTr>>) -> u32;
        /// Nothing of the trait object itself is relative: only `RelImpl` is leaked.
        fn inside(&self, x: ::core::cell::Ref<'a, dyn ::core::ops::Deref<Target = RelImpl>>)
            -> u32;
    }

    impl<'a> PathDyn<'a> for Inner {
        fn path(&self, x: ::core::cell::Ref<'a, dyn RelTr>) -> u32 {
            x.rel() + 1
        }
        fn paren(&self, x: ::core::cell::Ref<'a, (dyn RelTr)>) -> u32 {
            x.rel() + 2
        }
        fn bound(&self, x: ::core::cell::Ref<'a, dyn RelTr + 'a>) -> u32 {
            x.rel() + 3
        }
        fn boxed(&self, x: ::core::cell::Ref<'a, ::std::boxed::Box<dyn RelTr>>) -> u32 {
            x.rel() + 4
        }
        fn inside(
            &self,
            x: ::core::cell::Ref<'a, dyn ::core::ops::Deref<Target = RelImpl>>,
        ) -> u32 {
            ::core::ops::Deref::deref(&*x).rel() + 5
        }
    }

    // The leaked types of `PathDyn`, checked when this crate is compiled.
    type_leak_test_macro::check_leak! {
        {
            ::core::cell::Ref<'a, dyn RelTr>,
            ::core::cell::Ref<'a, (dyn RelTr)>,
            dyn RelTr + 'a,
            dyn RelTr,
            RelImpl,
            ? ::std::boxed::Box<dyn RelTr>
        }
        { }
        trait PathDynLeaks<'a> {
            fn path(&self, x: ::core::cell::Ref<'a, dyn RelTr>) -> u32;
            fn paren(&self, x: ::core::cell::Ref<'a, (dyn RelTr)>) -> u32;
            fn bound(&self, x: ::core::cell::Ref<'a, dyn RelTr + 'a>) -> u32;
            fn boxed(&self, x: ::core::cell::Ref<'a, ::std::boxed::Box<dyn RelTr>>) -> u32;
            fn inside(&self, x: ::core::cell::Ref<'a, dyn ::core::ops::Deref<Target = RelImpl>>)
                -> u32;
        }
    }

    pub const SIZE: usize = 4;
    /// A const which is not named in `SCREAMING_SNAKE_CASE`: there is no naming convention to tell
    /// it from a type.
    #[allow(non_upper_case_globals)]
    pub const Len: usize = 5;

    pub struct Big(pub [u8; 3]);

    /// A bare identifier as an argument, which is a const here (which syn parses as a type), and
    /// types in expressions, which cannot be leaked alone: the types around them are leaked as a
    /// whole.
    macro_rules! define_const_traits {
        () => {
            #[leak(
                helper = crate::__macro,
                repeater = crate::Repeater,
                module = crate::s,
            )]
            pub trait ConstArg {
                fn bare(&self) -> $crate::m_portable::CW<SIZE>;
                /// Named like a type.
                fn bare_mixed_case(&self) -> $crate::m_portable::CW<Len>;
            }

            #[leak(
                helper = crate::__macro,
                repeater = crate::Repeater,
                module = crate::s,
            )]
            pub trait ExprTypes {
                /// A type in an array length.
                fn expr_array(&self) -> Option<[u8; ::core::mem::size_of::<Big>()]>;
                /// A type in a const argument.
                fn expr_arg(&self) -> $crate::m_portable::CW<{ ::core::mem::size_of::<Big>() }>;
            }
        };
    }

    define_const_traits!();

    impl ConstArg for Inner {
        fn bare(&self) -> crate::m_portable::CW<SIZE> {
            crate::m_portable::CW
        }
        fn bare_mixed_case(&self) -> crate::m_portable::CW<Len> {
            crate::m_portable::CW
        }
    }

    impl ExprTypes for Inner {
        fn expr_array(&self) -> Option<[u8; ::core::mem::size_of::<Big>()]> {
            Some([1, 2, 3])
        }
        fn expr_arg(&self) -> crate::m_portable::CW<{ ::core::mem::size_of::<Big>() }> {
            crate::m_portable::CW
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

    pub struct CW<const N: usize>;
    impl<const N: usize> CW<N> {
        pub fn n(&self) -> usize {
            N
        }
    }
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
        // A generic argument which is relative is leaked (and the wrapper is rebuilt around it), as it
        // is for `::core::option::Option<(Rel,)>`. (A bare `Rel` as the argument may be a const, so
        // the wrapper is leaked as a whole instead.)
        check_leak! {
            { Rel }
            { }
            trait T2 { fn g(&self) -> $crate::m::Wrap<(Rel,)>; }
        }
        check_leak! {
            { $crate::m::Wrap<Rel> }
            { }
            trait T2b { fn g(&self) -> $crate::m::Wrap<Rel>; }
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

        // `$crate` paths are portable, unlike the other absolute paths.
        check_leak! {
            { }
            { }
            trait N1 { fn f(&self) -> $crate::m::Portable; }
        }
        check_leak! {
            { Rel }
            { }
            trait N2 { fn g(&self) -> $crate::m::Wrap<(Rel,)>; }
        }
        check_leak! {
            { }
            { }
            trait N3<T: $crate::m::PortTr>: $crate::m::PortTr {}
        }
        check_leak! {
            { }
            { }
            trait N4<const N: $crate::m::Idx> { fn a(&self) -> [u8; $crate::m::LEN]; }
        }
        // An absolute path of an external crate is an optional entry, unless it is allowed.
        check_leak! {
            { }
            { allow = ::foo }
            trait E1 { fn f(&self) -> ::foo::X; }
        }
        check_leak! {
            { ? ::foo::X }
            { }
            trait E2 { fn f(&self) -> ::foo::X; }
        }
        // A trait bound is copied as it is, which is not an error.
        check_leak! {
            { }
            { }
            trait E3<T: ::foo::Tr> {}
        }

        check_leak! {
            { Rel }
            { }
            trait C1 { fn g(&self) -> ::core::option::Option<(Rel,)>; }
        }
        check_leak! {
            { ::core::option::Option<Rel> }
            { }
            trait C2 { fn g(&self) -> ::core::option::Option<Rel>; }
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

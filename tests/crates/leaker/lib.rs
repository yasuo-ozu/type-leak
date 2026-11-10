//! The Leaker's crate: traits whose signatures use types relative to `mod m`, which the
//! Referrer's crate (`type-leak-test-referrer`) cannot name.

// The Repeater's implementations copy the types as they are written, including the parentheses
// of `(dyn RelTr)` in `s::PathDyn`, and the braces of a const parameter (`[u8; { N }]`).
#![allow(unused_parens, unused_braces)]

#[doc(hidden)]
pub use type_leak_test_macro as __macro;

// So that the crate can name itself with an absolute path (`::type_leak_test_leaker::kinds::Mark`).
extern crate self as type_leak_test_leaker;

// The Repeater, with const slots to encode const generics of any primitive type.
type_leak_test_macro::define_repeater!(pub Repeater);

/// A type of the root of the crate named like a generic (`crate::U`, `kinds::PathsLikeGeneric`).
pub struct U(pub u8);

/// Types in a module that the Referrer's crate does not have, so `crate::base::B1<X>` left as
/// written there would not compile (`kinds::declared_like_generic`).
pub mod base {
    pub struct B1<A: ?Sized>(pub core::marker::PhantomData<A>);
    pub struct B2<A, B>(pub A, pub B);
}

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
            RelImpl,
            ::std::boxed::Box<dyn RelTr>
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
        // An absolute path of an external crate is an entry, unless it is allowed.
        check_leak! {
            { }
            { allow = ::foo }
            trait E1 { fn f(&self) -> ::foo::X; }
        }
        check_leak! {
            { ::foo::X }
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

/// The crate which the types below name and the Referrer's crate does not depend on.
#[doc(hidden)]
pub use type_leak_test_ext as __ext;

/// Types of a crate which the Referrer's crate has no dependency on (`::type_leak_test_ext::Ext<..>`):
/// the Leaker makes entries of them, which the copy replaces with the Repeater's projections, so
/// that it does not name the crate; with a Leaker that allows the crate (`ExtDepAllowed`) they are no
/// entries, and the copy is no use in a crate without the dependency (the module `trusting` of
/// `tests/diagnostic.rs`).
pub mod x {
    use type_leak_test_macro::leak;

    pub use crate::m::Inner;

    pub struct RelX(pub u32);

    #[leak(
        helper = crate::__macro,
        repeater = crate::Repeater,
        module = crate::x,
    )]
    pub trait ExtDep {
        /// An external type with a relative one in it: one entry, which the copy replaces whole.
        fn wrapped(&self) -> ::type_leak_test_ext::Ext<(RelX,)>;
        /// An external type without any relative one: an entry too.
        fn plain(&self, x: ::type_leak_test_ext::Ext<u8>) -> u8;
    }

    /// The same for a Leaker that allows the crate: its types are no entries, so the copy names
    /// the crate (and the relative type in it is replaced).
    #[leak(
        helper = crate::__macro,
        repeater = crate::Repeater,
        module = crate::x,
        allow = ::type_leak_test_ext,
    )]
    pub trait ExtDepAllowed {
        fn wrapped(&self) -> ::type_leak_test_ext::Ext<(RelX,)>;
        fn plain(&self, x: ::type_leak_test_ext::Ext<u8>) -> u8;
    }

    impl ExtDepAllowed for Inner {
        fn wrapped(&self) -> ::type_leak_test_ext::Ext<(RelX,)> {
            ::type_leak_test_ext::Ext((RelX(21),))
        }
        fn plain(&self, x: ::type_leak_test_ext::Ext<u8>) -> u8 {
            x.0 + 1
        }
    }

    impl ExtDep for Inner {
        fn wrapped(&self) -> ::type_leak_test_ext::Ext<(RelX,)> {
            ::type_leak_test_ext::Ext((RelX(21),))
        }
        fn plain(&self, x: ::type_leak_test_ext::Ext<u8>) -> u8 {
            x.0 + 1
        }
    }
}

/// Fragments of impls, analyzed with the scope API: `scope_leak!` is the shape of `coinduction`
/// (`Leaker::enter_impl(.., false, ..)` with `visit_path_arguments()`, `ImplSelf::Marker`) and, with
/// `allow_self = true`, of `newer-type` (`ImplSelf::Param`).
pub mod scoped {
    use core::marker::PhantomData;
    use type_leak_test_macro::{scope_errors, scope_leak};

    pub struct Rel<T>(pub T);
    pub struct Rel2;
    pub struct Rel3;
    pub struct Wrap<T: ?Sized>(pub PhantomData<T>);
    pub struct Holder<T, U>(pub T, pub U);

    pub trait Tr<A, B> {}

    // The impls of the Leaker's crate, which the rewritten types of the Referrer's crate must find.
    impl<'a, T: ::core::clone::Clone> Tr<Rel<T>, &'a Rel2> for Holder<T, Rel3> {}
    impl<T: ::core::clone::Clone> Tr<Wrap<Holder<T, u8>>, Rel<T>> for Holder<T, u8> {}
    impl Tr<Rel2, Rel3> for Holder<Rel2, Rel3> {}

    // `coinduction`: the trait and the self type are resolved by name, so only their arguments are
    // leaked. `Self` is denied. Three required entries: `Rel3` (no generics, so the marker's impl is
    // plain), `Rel<T>` and `&'a Rel2`.
    scope_leak! {
        id = 7001, marker = M7001,
        impl<'a, T: ::core::clone::Clone> crate::scoped::Tr<Rel<T>, &'a Rel2>
            for crate::scoped::Holder<T, Rel3> {}
    }

    // `newer-type`: `Self` is the self type, and `Wrap<Self>` is an entry of it (the Repeater's impl
    // is for the self type, `ImplSelf::Param`), next to `Rel<T>` (the impl is for the marker).
    scope_leak! {
        id = 7002, marker = M7002, allow_self = true,
        impl<T: ::core::clone::Clone> crate::scoped::Tr<Wrap<Self>, Rel<T>>
            for crate::scoped::Holder<T, u8> {}
    }

    // Only entries which need no generics: the plain case of the marker.
    scope_leak! {
        id = 7003, marker = M7003,
        impl crate::scoped::Tr<Rel2, Rel3> for crate::scoped::Holder<Rel2, Rel3> {}
    }

    // An absolute path of another crate is an entry: the macro emits its impl (the report says
    // `entries=1 impls=1`), and the copy replaces it, so that the Referrer's crate needs no
    // dependency on the crate of the type.
    scope_leak! {
        id = 7004, marker = M7004,
        impl crate::scoped::Tr<::type_leak_test_ext::Ext<u8>, u8> for crate::scoped::Holder<u8, u8> {}
    }

    // The same next to a type that depends on the context: two entries and two impls.
    impl Tr<::type_leak_test_ext::Ext<u8>, Rel<u8>> for Holder<u8, u8> {}
    scope_leak! {
        id = 7005, marker = M7005,
        impl crate::scoped::Tr<::type_leak_test_ext::Ext<u8>, Rel<u8>> for crate::scoped::Holder<u8, u8> {}
    }

    // With a Leaker that allows the absolute paths (`external = true`), the type of the other crate
    // is no entry: only `Rel<u8>` gets an impl, and the copy writes the other as it is.
    scope_leak! {
        id = 7007, marker = M7007, external = true,
        impl crate::scoped::Tr<::type_leak_test_ext::Ext<u8>, Rel<u8>> for crate::scoped::Holder<u8, u8> {}
    }

    // A where predicate on a type that is not a parameter belongs to the generics it names: it is in
    // the context of `Cl<T>`, and the Repeater's impl has it in its where clause (`Cl<T>` is
    // well-formed only with it).
    pub struct Cl<T>(pub T)
    where
        ::std::vec::Vec<T>: ::core::clone::Clone;
    impl<T> Tr<Cl<T>, u8> for Holder<T, u8> where ::std::vec::Vec<T>: ::core::clone::Clone {}
    scope_leak! {
        id = 7006, marker = M7006,
        impl<T> crate::scoped::Tr<Cl<T>, u8> for crate::scoped::Holder<T, u8>
        where ::std::vec::Vec<T>: ::core::clone::Clone {}
    }

    // The errors of the analysis (`Vec<ErrorItem>`), reported through a macro:
    // several offending paths (a lifetime that is elided in a path), each with its span and notes.
    scope_errors! {
        name = SCOPE_ERRORS_MULTI,
        impl<T> crate::scoped::Tr<One<'_>, Two<'_, T>> for crate::scoped::Holder<Three<'_>, u8> {}
    }
    // `Self` is denied by `allow_self = false`.
    scope_errors! {
        name = SCOPE_ERRORS_SELF_DENIED, allow_self = false,
        impl crate::scoped::Tr<Wrap<Self>, u8> for crate::scoped::Holder<u8, u8> {}
    }
    // `Self` has no meaning in generics: two mentions, two items.
    scope_errors! {
        name = SCOPE_ERRORS_SELF_WITHOUT_MEANING, scope = generics,
        impl crate::scoped::Tr<Wrap<Self>, Rel<Self::Item>> for crate::scoped::Holder<u8, u8> {}
    }
}

/// Items of every kind, analyzed by the matching `intern_*` (`#[prop_leak]`, see
/// `tests/crates/macro/prop.rs`): the Referrer's crate invokes `__tl_prop_<id>!` for each and gets
/// a copy, whose every type position it compares with a witness of this crate, and
/// `__tl_report_<id>!` says what the Leaker found (tests/cross_crate.rs).
pub mod kinds {
    use core::marker::PhantomData;
    use type_leak_test_macro::{prop_leak, prop_leak_accept, prop_leak_result};

    #[derive(Debug, Clone, Copy, PartialEq, Default)]
    pub struct Rel(pub u8);

    pub struct Wrap<T: ?Sized>(pub PhantomData<T>);
    impl<T: ?Sized> Clone for Wrap<T> {
        fn clone(&self) -> Self {
            *self
        }
    }
    impl<T: ?Sized> Copy for Wrap<T> {}
    impl<T: ?Sized> Default for Wrap<T> {
        fn default() -> Self {
            Wrap(PhantomData)
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Default)]
    pub struct Pair<A, B>(pub A, pub B);

    pub struct Lt<'a, T>(pub &'a T);

    /// A pair whose first type is behind a pointer, so that `Link<Self, T>` is no recursive type.
    #[derive(Debug)]
    pub struct Link<A: ?Sized, B>(pub PhantomData<*const A>, pub B);

    /// A const, and a path that takes one as an argument: `CW<A, LEN>` holds a bare identifier
    /// that is a const (which `syn` parses as a type), so the path is leaked as a whole.
    pub const LEN: usize = 3;

    pub struct CW<A: ?Sized, const N: usize>(pub PhantomData<A>);

    pub trait Sink<A: ?Sized> {}

    /// A type alias that hides a projection (F26), a struct of the same shape, a trait whose
    /// supertrait fixes an `Item` (F25), and one with an associated type of its own.
    pub type WIt<T> = <T as Iterator>::Item;
    pub struct WFoo<T>(pub PhantomData<T>);
    pub trait WIter: Iterator<Item = u8> {}
    pub trait Cont {
        type Out;
    }

    pub trait Mark<A> {
        type Out;
        fn mark(&self, a: A) -> Self::Out;
    }

    // Structs.

    /// `Self` (the type) is an entry of it, as are `Rel`, `Wrap<T>`, `Pair<Rel, T>` and `&'a Rel`.
    #[prop_leak(id = 9001, allow_self = true, prelude = true)]
    pub struct Node<'a, T: 'a> {
        pub rel: Rel,
        pub wrapped: Wrap<T>,
        pub pair: Pair<Rel, T>,
        pub borrowed: &'a Rel,
        pub next: Option<Box<Wrap<Self>>>,
    }

    /// `Self` is denied, and the struct does not name it.
    #[prop_leak(id = 9002, allow_self = false)]
    pub struct Plain<T> {
        pub rel: Rel,
        pub wrapped: Wrap<T>,
        pub pair: Pair<T, Rel>,
    }

    /// `Self` is denied, and the struct names it: the Leaker rejects it.
    #[prop_leak(id = 9003, allow_self = false)]
    pub struct SelfDenied {
        pub me: Wrap<Self>,
    }

    /// An optional entry (a type of another crate), which a strict Referrer replaces. The crate
    /// is not a dependency of the Referrer's crate.
    #[prop_leak(id = 9004, allow_self = true)]
    pub struct Holds {
        pub wrapped: ::type_leak_test_ext::Ext<(Rel,)>,
        pub plain: ::type_leak_test_ext::Ext<u8>,
        pub rel: Rel,
    }

    /// The where clause of a type that is not a parameter (`Vec<T>: Clone`) is in the context of the
    /// entry `Cl<T>`, and in the where clause of the Repeater's impl.
    pub struct Cl<T>(pub T)
    where
        ::std::vec::Vec<T>: ::core::clone::Clone;

    #[prop_leak(id = 9005, allow_self = false, allow = "::std")]
    pub struct Needs<T>
    where
        ::std::vec::Vec<T>: ::core::clone::Clone,
    {
        pub cl: Cl<T>,
    }

    // Enums.

    #[prop_leak(id = 9011, allow_self = true)]
    pub enum Choice<T> {
        Rel(Rel),
        Wrapped(Wrap<T>),
        Pair { pair: Pair<Rel, T>, flag: bool },
        Me(Wrap<Self>),
        Nothing,
    }

    #[prop_leak(id = 9012, allow_self = false)]
    pub enum Borrow<'a, T: 'a> {
        Lt(Lt<'a, T>),
        Rel(Rel),
    }

    // Unions.

    #[prop_leak(id = 9021, allow_self = false)]
    pub union Bits<T: ::core::marker::Copy> {
        pub rel: Rel,
        pub wrapped: Wrap<T>,
        pub pair: Pair<Rel, Rel>,
    }

    #[prop_leak(id = 9022, allow_self = true)]
    pub union BitsSelf {
        pub rel: Rel,
        pub me: Wrap<Self>,
    }

    // Impls.

    pub struct Holder<T>(pub T);

    /// An inherent impl: `Self` is `Holder<T>`.
    #[prop_leak(id = 9031, allow_self = true)]
    impl<T> Holder<T> {
        pub fn rel(&self) -> Rel {
            Rel(1)
        }
        pub fn wrapped(&self) -> Wrap<T> {
            Wrap(PhantomData)
        }
        pub fn me(&self) -> Wrap<Self> {
            Wrap(PhantomData)
        }
        pub fn pair(&self, p: Pair<Rel, T>) -> Pair<T, Rel> {
            Pair(p.1, p.0)
        }
    }

    /// An impl of a trait: the trait is `::type_leak_test_leaker::kinds::Mark`, an absolute path (its arguments must
    /// be too, as a bound is copied as written), and the associated type and the method are leaked.
    #[prop_leak(id = 9032, allow_self = true)]
    impl<T> ::type_leak_test_leaker::kinds::Mark<u8> for Holder<T> {
        type Out = Pair<Rel, Wrap<T>>;
        fn mark(&self, a: u8) -> Pair<Rel, Wrap<T>> {
            Pair(Rel(a), Wrap(PhantomData))
        }
    }

    /// A relative argument of the trait is leaked too (the trait itself is resolved by name).
    #[prop_leak(id = 9034, allow_self = true)]
    impl<T> ::type_leak_test_leaker::kinds::Mark<Rel> for Holder<Wrap<T>> {
        type Out = Rel;
        fn mark(&self, a: Rel) -> Rel {
            a
        }
    }

    /// `Self` is denied for the impl, which does not name it.
    #[prop_leak(id = 9033, allow_self = false)]
    impl<T> Holder<Wrap<T>> {
        pub fn inner(&self) -> Rel {
            Rel(3)
        }
    }

    // Modules.

    /// A module: what it declares (`Local`) is copied with it and is not leaked, the types it gets
    /// from outside (`super::Rel`) are, and a nested module is analyzed too.
    #[prop_leak(id = 9041)]
    pub mod nested {
        use super::{Pair, Rel, Wrap};

        pub struct Local(pub u8);
        pub struct Holds<T> {
            pub rel: Rel,
            pub wrapped: Wrap<T>,
            pub pair: Pair<Rel, T>,
            pub local: Local,
        }
        pub mod deeper {
            use super::super::Rel;
            pub struct Deep(pub Rel);
        }
    }

    /// A bare identifier that is a const, in a path of another crate that names `Local`, which the
    /// module declares (the copy declares it again, so the path is not leaked as a whole). `LEN2`
    /// is imported with an absolute path, which declares it: it is copied as written, and the
    /// copy compiles. `Rel` is leaked as usual.
    #[prop_leak(id = 9101)]
    pub mod declared_const {
        use super::Rel;
        use ::type_leak_test_leaker::kinds::LEN as LEN2;

        pub struct Local(pub u8);
        pub struct Holds {
            pub rel: Rel,
            pub len: ::type_leak_test_leaker::kinds::CW<Local, LEN2>,
        }
        pub mod deeper {
            use super::super::LEN as LEN3;
            pub struct Deep(pub ::type_leak_test_leaker::kinds::CW<u8, LEN3>);
        }
    }

    /// The same, with `LEN4` imported from outside the module (`use super::LEN`): the Leaker cannot
    /// leak the path that holds it, and it would leak `LEN4` as a type, which fails for a const, so
    /// it rejects the module (the macro expands to nothing).
    #[prop_leak(id = 9102)]
    pub mod imported_const {
        use super::LEN as LEN4;

        pub struct Local(pub u8);
        pub struct Holds {
            pub len: ::type_leak_test_leaker::kinds::CW<Local, LEN4>,
        }
    }

    // Traits.

    /// `Self` is denied for the trait (`sumtype`'s way), and it does not name `Self`.
    #[prop_leak(id = 9051, allow_self = false)]
    pub trait NoSelf {
        fn wrapped(&self) -> Wrap<Rel>;
        fn pair<T>(&self, x: Pair<T, Rel>) -> Rel;
    }

    /// `Self` is denied, and the trait names it: the Leaker rejects it.
    #[prop_leak(id = 9052, allow_self = false)]
    pub trait SelfDeniedTrait {
        fn me(&self) -> Wrap<Self>;
    }

    /// A trait which is implemented for an unsized type in the Referrer's crate (`dyn_host`): the
    /// Repeater's impls of the entries that name `Self` must declare `__S: ?Sized`.
    #[prop_leak(id = 9053, allow_self = true)]
    pub trait Unsized {
        fn me(&self) -> Wrap<Self>;
        fn borrowed<'a>(&'a self, x: &'a Self) -> Wrap<&'a Self>;
        fn rel(&self) -> Rel;
        fn pair(&self, x: Pair<Rel, Wrap<Self>>) -> Rel;
    }

    // A bare identifier that is a const (`LEN`).

    /// `Self` is allowed: the path that holds the const is an entry as a whole (`CW<[Self], LEN>`),
    /// so the const is never read as a type, whatever the path holds beside it.
    #[prop_leak(id = 9081, allow_self = true)]
    pub struct ConstSelf {
        pub me: ::type_leak_test_leaker::kinds::CW<[Self], LEN>,
        pub rel: ::type_leak_test_leaker::kinds::CW<Rel, LEN>,
    }

    /// `Self` is denied, and the path that holds the const names it: it cannot be leaked as a
    /// whole, and the const must not be leaked alone as a type (it would not compile): the Leaker
    /// rejects it.
    #[prop_leak(id = 9082, allow_self = false)]
    pub struct ConstSelfDenied {
        pub me: ::type_leak_test_leaker::kinds::CW<[Self], LEN>,
    }

    /// A trait: `Self` is the implementor, and `CW<Self, LEN>` is an entry for its implementors.
    #[prop_leak(id = 9083, allow_self = true)]
    pub trait ConstTrait {
        fn me(&self) -> ::type_leak_test_leaker::kinds::CW<Self, LEN>;
        fn rel(&self, x: ::type_leak_test_leaker::kinds::CW<Rel, LEN>) -> Rel;
    }

    /// A bare identifier that is a type, where the path that holds it cannot be leaked (an elided
    /// lifetime, `impl Trait`, a trait object under an elided reference, a trait bound): it is
    /// leaked alone as a type, which works for a type.
    #[prop_leak(id = 9084, allow_self = false)]
    pub trait SoftIdent {
        fn by_dyn(&self, x: &dyn ::type_leak_test_leaker::kinds::Sink<Rel>);
        fn by_impl(&self, x: impl ::type_leak_test_leaker::kinds::Sink<Rel>);
        fn by_elided(&self, x: &u8) -> ::type_leak_test_leaker::kinds::Lt<'_, Rel>;
        fn by_bound<T: ::type_leak_test_leaker::kinds::Sink<Rel>>(&self, x: T);
    }

    // Items nested in a module (`scope = "mod"`: `enter_mod(..)` and `View::visit_item()`, as the
    // items of `intern_mod()` are). The item has a `Self` of its own, and the types that name it
    // are relative (`Wrap<Self>`), so the Repeater's impl of such an entry is for the item itself,
    // here. The copy cannot name that type as `Self`: `Visitor::with_nested_self()` gives the macro
    // the item to write it with.

    /// A struct that names its own `Self` in relative types, next to ones that do not.
    #[prop_leak(id = 9061, allow_self = true, scope = "mod")]
    pub struct Links<T> {
        pub rel: Rel,
        pub wrapped: Wrap<T>,
        pub me: Wrap<Self>,
        pub pair: Pair<Rel, Wrap<Self>>,
        pub next: Option<Box<Wrap<Self>>>,
    }

    /// An enum, with `Self` in the generics of a variant that a bound names too.
    #[prop_leak(id = 9062, allow_self = true, scope = "mod")]
    pub enum Step<T> {
        Rel(Rel),
        Wrapped(Wrap<T>),
        Me(Wrap<Self>),
        Pair { pair: Pair<Wrap<Self>, T> },
        Nothing,
    }

    /// A trait: `Self` is the implementor, and the Referrer's crate implements the trait for a host.
    #[prop_leak(id = 9063, allow_self = true, scope = "mod")]
    pub trait Walk<T> {
        fn me(&self) -> Wrap<Self>;
        fn rel(&self, x: Pair<Rel, T>) -> Wrap<T>;
        fn pair(&self, x: Pair<Rel, Wrap<Self>>) -> Rel;
    }

    // Items nested in a module whose generics have a bound that names their own `Self` (`T:
    // AsRef<Self>`), and whose fields use the generic (`Wrap<T>`, which does not name `Self`). The
    // copy declares a type of its own, which has the bound for itself, so the Repeater's impl of
    // the entry is for any type that the bound holds for, and `Visitor::with_nested_self()` lets
    // the copy pass its own type (`SelfRequirement::Any`).

    /// The bound is on the parameter.
    #[prop_leak(id = 9111, allow_self = true, scope = "mod", prelude = true)]
    pub struct Owned<T: AsRef<Self>> {
        pub rel: Rel,
        pub wrapped: Wrap<T>,
        pub pair: Pair<Rel, T>,
    }

    /// The bound is a where predicate, and names `Self` among other types.
    #[prop_leak(id = 9112, allow_self = true, scope = "mod")]
    pub struct Tagged<T>(pub Wrap<T>, pub u8)
    where
        T: ::core::convert::AsRef<(Self, u8)>;

    /// The report of the property test: an enum with a lifetime, a const parameter with a default,
    /// and a bound that names `Self` twice and a relative type (`Lt<'a, Rel>`).
    #[prop_leak(id = 9113, allow_self = true, scope = "mod")]
    pub enum Reported<'a, T: ::core::marker::Send + 'a, const V: usize = 3>
    where
        T: ::core::convert::AsRef<::std::collections::BTreeMap<(Self, Self), self::Lt<'a, Rel>>>,
    {
        V0(&'a super::base::B2<T, ::core::result::Result<u32, u16>>),
        V1,
    }

    /// A union.
    #[prop_leak(id = 9114, allow_self = true, scope = "mod")]
    pub union Overlaid<T: ::core::marker::Copy + ::core::convert::AsRef<Self>> {
        pub wrapped: Wrap<T>,
        pub rel: Rel,
    }

    /// A type that names `Self` and has a generic with a bound that names it: it is for the
    /// Leaker's type, which the bound does not hold for in the copy. The Leaker rejects it (the
    /// macro expands to nothing); `Wrap<T>`, which does not name `Self`, is the supported case.
    #[prop_leak(id = 9115, allow_self = true, scope = "mod", prelude = true)]
    pub struct Both<T: AsRef<Self>> {
        pub me: Wrap<Self>,
        pub wrapped: Wrap<T>,
    }

    /// The entry is in a bound of another generic (`U: Into<Wrap<T>>`), where the copy's `Self` is
    /// the type being declared.
    #[prop_leak(id = 9116, allow_self = true, scope = "mod", prelude = true)]
    pub struct Chain<T: AsRef<Self>, U: Into<Wrap<T>>>(pub T, pub U);

    /// The entry is in a where predicate.
    #[prop_leak(id = 9117, allow_self = true, scope = "mod", prelude = true)]
    pub struct InPredicate<T: AsRef<Self>>(pub T)
    where
        Wrap<T>: Sized;

    /// The entry has a lifetime, which the bound `T: 'a` keeps.
    #[prop_leak(id = 9118, allow_self = true, scope = "mod", prelude = true)]
    pub struct Outlives<'a, T: AsRef<Self> + 'a>(pub &'a Wrap<T>, pub T);

    /// `Self: 'a` is written, so the arguments of the Repeater encode `&'a Self`: the copy's own.
    #[prop_leak(id = 9119, allow_self = true, scope = "mod", prelude = true)]
    pub struct SelfOutlived<'a, T: AsRef<Self>>(pub &'a Wrap<T>, pub T)
    where
        Self: 'a;

    /// A predicate on `Self` alone is in the context of the entry too.
    #[prop_leak(id = 9120, allow_self = true, scope = "mod", prelude = true)]
    pub struct SelfSized<T: AsRef<Self>>(pub Wrap<T>, pub T)
    where
        Self: Sized;

    // The entry for any type is the last field of a struct, or a field of an enum. The Repeater's
    // impl is `?Sized` where the bounds do not need `Self: Sized`: with `Sized`, rustc before 1.70
    // cannot prove that the copy is `Sized` (E0275, `Last<T>: Sized` needs the impl, which needs
    // the type to be `Sized`), which shows on those toolchains only.

    /// The bound is `AsRef<Self>`: the impl is `?Sized`, the entry is the last field.
    #[prop_leak(id = 9125, allow_self = true, scope = "mod")]
    pub struct Last<T: ::core::convert::AsRef<Self>> {
        pub rel: Rel,
        pub wrapped: Wrap<T>,
    }

    /// An enum: its variants are checked as fields (on rustc 1.56 too).
    #[prop_leak(id = 9126, allow_self = true, scope = "mod")]
    pub enum Alternative<T: ::core::convert::AsRef<Self>> {
        Plain(u8),
        Wrapped(Wrap<T>),
    }

    /// `Self` is behind a `Box`, and the argument of `Borrow`: still `?Sized`.
    #[prop_leak(id = 9127, allow_self = true, scope = "mod")]
    pub struct Pointed<
        T: ::core::convert::AsRef<::std::boxed::Box<Self>> + ::core::borrow::Borrow<Self>,
    >(pub u8, pub Wrap<T>);

    /// `Self` in a tuple: the impl keeps `Sized` (the tuple needs it), the entry is not the last
    /// field.
    #[prop_leak(id = 9128, allow_self = true, scope = "mod")]
    pub struct Tupled<T: ::core::convert::AsRef<(Self, u8)>> {
        pub wrapped: Wrap<T>,
        pub tag: u8,
    }

    // Items nested in a module that name their own `Self` in an absolute type of a crate that both
    // crates have (`::type_leak_test_leaker`, `::std`). Such a type is no entry: it is copied as
    // written, and `Self` in it is the copy's own type, also for a Referrer that replaces the
    // entries. The relative types beside it are entries.

    /// A bound that names `Self`, an absolute type that names it too, and a relative type that uses
    /// the generic (`Wrap<T>`, for any type that the bound holds for).
    #[prop_leak(id = 9121, allow_self = true, scope = "mod", prelude = true)]
    pub struct Pathed<T: AsRef<Self>> {
        pub path: ::type_leak_test_leaker::kinds::Pair<::core::marker::PhantomData<Self>, T>,
        pub wrapped: Wrap<T>,
        pub rel: Rel,
    }

    /// A recursive type: `Self` in a box.
    #[prop_leak(id = 9122, allow_self = true, scope = "mod", prelude = true)]
    pub struct List {
        pub next: Option<::std::boxed::Box<Self>>,
        pub rel: Rel,
    }

    /// An enum with `Self` in a box in a vector, and in a pair next to a relative type.
    #[prop_leak(id = 9124, allow_self = true, scope = "mod", prelude = true)]
    pub enum Tree<T> {
        Leaf(Rel),
        Branch(Vec<::std::boxed::Box<Self>>),
        Tagged(::type_leak_test_leaker::kinds::Pair<::core::marker::PhantomData<Self>, Wrap<T>>),
    }

    // Traits nested in a module whose generic has a bound that names their own `Self` (`T:
    // AsRef<Self>`), and whose parameter defaults use it (`U = Defd<T>`). The default does not name
    // `Self`, so it is analyzed alone, as that of a trait that a scope enters: its entry is for any
    // type that the bound holds for, and the Visitor writes it with the ordinary closure, which
    // picks the host (it would keep the default as written, `Defd<T>`, which the Referrer's crate
    // cannot resolve, if it were not an entry). A parameter default exists only in a declaration,
    // so the Referrer's crate declares the copy (`redeclare`), and checks that its defaults are
    // the types of the original.

    /// Types that the Referrer's crate does not import: a default that it keeps as written does not
    /// resolve there (E0412).
    pub struct Defd<T: ?Sized>(pub PhantomData<T>);
    pub struct Defd2<A, B>(pub A, pub B);

    /// The impls of the entries that do not name `Self` (`Rel`, `Defd2<Rel, u8>`, `Defd<V>`) are
    /// for a primitive type that the id and the index choose (`ImplSelf::Bucket`), and those that
    /// do (`Wrap<Self>`) are for the implementor. The default `Defd<T>` is one of the first.
    #[prop_leak(id = 9209, allow_self = true, scope = "mod", prelude = true, redeclare = true)]
    pub trait BucketMix<T, U = Defd<T>> {
        fn plain(&self) -> Rel;
        fn plain_pair(&self) -> Defd2<Rel, u8>;
        fn own(&self) -> Wrap<Self>;
        fn generic<V>(&self, v: V) -> Defd<V>;
    }

    /// The marker type of `MarkerDefault`.
    pub struct Marker9210;

    /// The impls of the entries that name no `Self`, nor do their generics, are for a marker type of
    /// this crate (`ImplSelf::Marker`; `marker = ..`), not a primitive type. The trait is the item
    /// itself (not nested in a module), so its default `Defd<T>` does not name `Self` while the
    /// bound of `T` does (`Context::self_ty` is `None`): the entry is for the implementor of the
    /// trait, which is the host in the copy. For the marker it would bound `T: AsRef<Marker9210>`,
    /// which the types that the Referrer's crate uses for `T` do not satisfy
    /// (`impl MarkerDefaultCopy<Owner9210> for Host9210`, E0277).
    #[prop_leak(
        id = 9210,
        allow_self = true,
        prelude = true,
        redeclare = true,
        marker = "::type_leak_test_leaker::kinds::Marker9210"
    )]
    pub trait MarkerDefault<T: AsRef<Self>, U = Defd<T>> {
        fn get(&self, t: T, u: U);
        fn plain(&self) -> Rel;
    }

    /// One default, written for the host: `T: AsRef<Host>` holds for the types that it is used with.
    #[prop_leak(id = 9131, allow_self = true, scope = "mod", prelude = true, redeclare = true)]
    pub trait SelfDefault<T: AsRef<Self>, U = Defd<T>> {
        fn get(&self, t: T, u: U);
    }

    /// Two defaults (the second has an entry of its own, `Defd2<Rel, T>`), for a Referrer that
    /// replaces the optional entries. (A method of the copy cannot name an entry: the declaration
    /// has `T: AsRef<Self>`, and the path through the host needs `T: AsRef<Host>`.)
    #[prop_leak(
        id = 9132,
        allow_self = true,
        scope = "mod",
        prelude = true,
        redeclare = true
    )]
    pub trait SelfDefaults<T: AsRef<Self>, U = Defd<T>, V = Defd2<Rel, T>> {
        fn get(&self, t: T, u: U, v: V);
    }

    // Items nested in a module whose generics have a bound that names their own `Self` (`T:
    // AsRef<Self>`), and whose parameter defaults use the generic (`U = Defd<T>`, a type that the
    // Referrer's crate does not import). The copy cannot write `Self` in a default (E0735), so the
    // entry in a default is analyzed without `Self` and without the bounds that name it: the
    // Repeater's impl is for any type, with the other bounds (`T: Clone`), and the copy writes an
    // ordinary default through it. rustc checks a default that uses a generic where it is used, so
    // the default is the type of the original there.

    /// The bound names `Self` next to one that does not (`Clone`, which stays in the impl).
    #[prop_leak(id = 9151, allow_self = true, scope = "mod", prelude = true)]
    pub struct SelfBoundDefault<T: Clone + AsRef<Self>, U = Defd<T>>(pub T, pub U);

    /// An enum, the bound in a where predicate.
    #[prop_leak(id = 9152, allow_self = true, scope = "mod", prelude = true)]
    pub enum SelfBoundDefaultEnum<T, U = Defd<T>>
    where
        T: Clone + AsRef<Self>,
    {
        Plain(T),
        Defaulted(U),
    }

    /// The default and a field have the same type: the default is an entry without `Self`, and the
    /// field one for any type (the copy passes its own).
    #[prop_leak(id = 9153, allow_self = true, scope = "mod", prelude = true)]
    pub struct DefaultAndField<T: AsRef<Self>, U = Defd<T>>(pub T, pub U, pub Defd<T>);

    /// Two defaults, the second names the first.
    #[prop_leak(id = 9154, allow_self = true, scope = "mod", prelude = true)]
    pub struct TwoDefaults<T: AsRef<Self>, U = Defd<T>, V = Defd2<Rel, U>>(pub T, pub U, pub V);

    // Items nested in a module whose generics have a bound that names their own `Self`, and a type
    // that names it too (`nested_self_entries`, `nested_self_entries = true`). The
    // entry is for any type that the bound holds for, with `Self` as the implementor in the type
    // too, and the copy passes its own: the field is `Link<MyStruct<T>, T>`, as in the original.
    // The Repeater's impl keeps `Sized` (the leaked type may need it). An entry whose bound is
    // not on `Self` is for the Leaker's type, as without the option. (The same struct without the
    // option is rejected by the Leaker.)

    /// The bound is on the parameter. The entry is not the last field: rustc before 1.70 cannot
    /// prove that the copy is `Sized` otherwise (E0275).
    #[prop_leak(id = 9161, allow_self = true, scope = "mod", prelude = true, nested_self_entries = true)]
    pub struct Linked<T: AsRef<Self>> {
        pub link: Link<Self, T>,
        pub rel: Rel,
    }

    /// An enum, with `where Self: Sized` for rustc 1.56, and an entry that does not use the generic.
    #[prop_leak(id = 9162, allow_self = true, scope = "mod", prelude = true, nested_self_entries = true)]
    pub enum Variant<T: AsRef<Self>>
    where
        Self: Sized,
    {
        Plain(Rel),
        Me(Link<Self, Rel>),
        Held(T),
    }

    /// The bound is a where predicate that names `Self` among other types.
    #[prop_leak(id = 9163, allow_self = true, scope = "mod", prelude = true, nested_self_entries = true)]
    pub struct Tuple<T>(pub Vec<Link<Self, u8>>, pub T, pub Rel)
    where
        T: AsRef<(Self, u8)>;

    /// No bound names `Self`: the entry is for the Leaker's type, as without the option.
    #[prop_leak(id = 9164, allow_self = true, scope = "mod", prelude = true, nested_self_entries = true)]
    pub struct Unbound<T: Clone> {
        pub link: Link<Self, T>,
        pub rel: Rel,
    }

    // A bound of a generic that names the generic, next to a bound of it that fixes an associated
    // type (F26, F25, F2): the Repeater's impl of an entry that uses the generic leaves it out,
    // and the copy, which has it in its environment as written, proves the impl (rustc cannot
    // normalize `WIt<T>` to `u8` in the environment while it proves the bound of the impl, E0277).

    /// The `Item` is fixed by a bound, and a bound of the generic names it through an alias that
    /// the Leaker cannot tell from a struct.
    #[prop_leak(id = 9171, allow_self = false, prelude = true)]
    pub trait AliasItem<T: Iterator<Item = u8>>
    where
        T: Fn(::type_leak_test_leaker::kinds::WIt<T>),
    {
        fn m(&self)
        where
            Wrap<T>: Default;
    }

    /// The same for a strict Referrer, which replaces the alias too: its entry has no cycle.
    #[prop_leak(id = 9172, allow_self = false, prelude = true)]
    pub trait AliasItemStrict<T: Iterator<Item = u8>>
    where
        T: Fn(::type_leak_test_leaker::kinds::WIt<T>),
    {
        fn m(&self)
        where
            Wrap<T>: Default;
    }

    /// The projection written (the Leaker rejected it before).
    #[prop_leak(id = 9173, allow_self = false, prelude = true)]
    pub trait WrittenItem<T: Iterator<Item = u8>>
    where
        T: Fn(<T as Iterator>::Item),
    {
        fn m(&self)
        where
            Wrap<T>: Default;
    }

    /// A struct instead of the alias: it compiled before, and the Leaker cannot tell.
    #[prop_leak(id = 9174, allow_self = false, prelude = true)]
    pub trait StructItem<T: Iterator<Item = u8>>
    where
        T: Fn(::type_leak_test_leaker::kinds::WFoo<T>),
    {
        fn m(&self)
        where
            Wrap<T>: Default;
    }

    /// The `Item` is fixed by a supertrait of a bound (F25), in the where clause of the trait and
    /// in the generics of a method.
    #[prop_leak(id = 9175, allow_self = false, prelude = true)]
    pub trait SuperItem<T: ::type_leak_test_leaker::kinds::WIter>
    where
        T: Fn(T::Item),
    {
        fn m(&self)
        where
            Wrap<T>: Default;
        fn n<U: ::type_leak_test_leaker::kinds::WIter + Fn(U::Item)>(&self)
        where
            Wrap<(Rel, U)>: Default;
    }

    /// A trait with an associated type of its own, which the Leaker cannot tell from `WIter`.
    #[prop_leak(id = 9176, allow_self = false, prelude = true)]
    pub trait OwnAssoc<T: ::type_leak_test_leaker::kinds::Cont>
    where
        T: Fn(T::Out),
    {
        fn m(&self)
        where
            Wrap<T>: Default;
    }

    /// A struct with the predicates, and a field.
    #[prop_leak(id = 9177, allow_self = false, prelude = true)]
    pub struct AliasStruct<T: Iterator<Item = u8>>(pub PhantomData<Wrap<T>>)
    where
        T: Fn(::type_leak_test_leaker::kinds::WIt<T>),
        Wrap<T>: Default;

    /// The bound that names the leaked type is the one that is left out (F2): the entry would be
    /// in a cycle with itself otherwise.
    #[prop_leak(id = 9178, allow_self = false, prelude = true)]
    pub struct IntoEntry<T: Iterator<Item = u8> + Into<Wrap<T>>>(pub T);

    /// The impl of F19: `Self` in a bound is the self type, which has the entry `Wrap<I>` (and
    /// `Pair` is allowed, so the self type is no entry as a whole).
    #[prop_leak(
        id = 9179,
        allow_self = true,
        prelude = true,
        allow = "::type_leak_test_leaker::kinds::Pair"
    )]
    impl<I: Iterator<Item = u8> + Fn(I::Item), V: AsRef<Self>>
        ::type_leak_test_leaker::kinds::Pair<Wrap<I>, V>
    {
        pub fn m(&self, _a: Wrap<I>) {}
    }

    /// A type that the Leaker knows is declared with the bounds that it keeps whole: the self type
    /// of an impl (the entry `Bounded<T>` is well-formed only with `T: Fn(WIt<T>)`), and `Self` of
    /// a struct (the entry `Wrap<Self>`), though the entries that do not name them (`Wrap<T>`)
    /// leave the bound out.
    pub struct Bounded<T: Iterator<Item = u8>>(pub T)
    where
        T: Fn(::type_leak_test_leaker::kinds::WIt<T>);

    #[prop_leak(id = 9182, allow_self = true, prelude = true)]
    impl<T: Iterator<Item = u8>> Bounded<T>
    where
        T: Fn(::type_leak_test_leaker::kinds::WIt<T>),
    {
        pub fn m(&self, _x: Wrap<T>) {}
    }

    #[prop_leak(id = 9183, allow_self = true, prelude = true)]
    pub struct SelfBounded<T: Iterator<Item = u8>>(pub T, pub Wrap<Self>, pub Wrap<T>)
    where
        T: Fn(::type_leak_test_leaker::kinds::WIt<T>);

    /// Control: nothing fixes the `Item`, so the bound stays in the impl of the entry.
    #[prop_leak(id = 9180, allow_self = false, prelude = true)]
    pub trait RigidItem<T: Iterator>
    where
        T: Fn(T::Item),
    {
        fn m(&self)
        where
            Wrap<T>: Default;
    }

    /// Control: a bound of another generic that names a fixed `Item` is not left out (the rule is
    /// about the generic that the bound names itself), and the Leaker rejects it.
    #[prop_leak(id = 9181, allow_self = false, prelude = true)]
    pub trait OtherGeneric<T: Iterator<Item = u8>, U: Fn(T::Item)> {
        fn m(&self)
        where
            Wrap<U>: Default;
    }

    // `Self: 'x` brings `Self` into the context of an entry that does not name it (F47): the bound
    // of the method's generic that F26 leaves out stays out, as the entry does not need it.

    /// `Y: 'x` pulls `Self: 'x` into the context of `Wrap<Y>`.
    #[prop_leak(id = 9200, allow_self = true, prelude = true)]
    pub trait SelfOutlives {
        fn m<'x, Y: 'x + Iterator<Item = u8>>(&self)
        where
            Self: 'x,
            Y: Fn(::type_leak_test_leaker::kinds::WIt<Y>) -> u8,
            Wrap<Y>: Default;
    }

    /// The same for an impl of a trait: `Self` is its self type.
    pub trait SelfOutlivesTr<'a> {
        fn m<'x, Y: 'x + Iterator<Item = u8>>(&'x self) -> u8
        where
            Self: 'x,
            Y: Fn(::type_leak_test_leaker::kinds::WIt<Y>) -> u8,
            ::type_leak_test_leaker::kinds::Wrap<Y>: Default;
    }

    pub struct SelfOutlivesHost;

    #[prop_leak(id = 9201, allow_self = true, prelude = true)]
    impl<'a> ::type_leak_test_leaker::kinds::SelfOutlivesTr<'a>
        for ::type_leak_test_leaker::kinds::SelfOutlivesHost
    {
        fn m<'x, Y: 'x + Iterator<Item = u8>>(&'x self) -> u8
        where
            Self: 'x,
            Y: Fn(::type_leak_test_leaker::kinds::WIt<Y>) -> u8,
            Wrap<Y>: Default,
        {
            loop {}
        }
    }

    // The scopes of an item that share names, from the findings of a downstream crate
    // (`announce`, tests/downstream_findings.rs): the Repeater's impl of every entry has the
    // generics, bounds and where clause of the scope the entry is written in, and no others.

    /// A type that is well-formed only for `T: Clone`.
    pub struct Cloned<T: Clone>(pub T);

    /// A type with a const parameter.
    pub struct Arr<const N: usize>;

    /// A type whose name a method generic can have too.
    pub struct Local(pub u8);

    pub mod inner {
        /// A type whose name a generic can have too, written with a path.
        pub struct U(pub u8);
    }

    /// Two methods declare the generic `X` with different bounds (the entries `Cloned<X>` and
    /// `Wrap<X>`); a method generic is named like the relative type `Local` (its `Local` is no
    /// entry, the one in the return type of the other method is); a path ends with the name of a
    /// generic of a method (`inner::U` next to `<U>`); and two methods declare `N` as a type and as
    /// a const.
    #[prop_leak(id = 9071, allow_self = false, prelude = true)]
    pub trait SameNames {
        fn with_bound<X: Clone>(&self, x: Cloned<X>) -> Cloned<X>;
        fn without_bound<X>(&self, x: Wrap<X>) -> Wrap<X>;
        fn generic_local<Local>(&self, x: Local) -> Local;
        fn type_local(&self) -> Local;
        fn segment<U>(&self, x: Wrap<U>) -> inner::U;
        fn type_n<N>(&self, x: Wrap<N>) -> Wrap<N>;
        fn const_n<const N: usize>(&self, x: Arr<N>) -> Arr<N>;
    }

    /// A bound that names a parameter the entry `Wrap<T>` does not use (`V`).
    #[prop_leak(id = 9072, allow_self = false, prelude = true)]
    pub struct ForeignBound<T: Into<V>, V>(pub Wrap<T>, pub V);

    /// The same in a trait: the bound of a method generic names the trait's `T`.
    #[prop_leak(id = 9073, allow_self = false, prelude = true)]
    pub trait ForeignBoundTrait<T> {
        fn convert<U: Into<T>>(&self, x: Wrap<U>) -> Wrap<U>;
    }

    /// `T: 'a`, a bound that names a lifetime the entry does not use.
    #[prop_leak(id = 9074, allow_self = false, prelude = true)]
    pub struct OutlivesBound<'a, T: 'a>(pub Wrap<T>, pub &'a u8);

    /// A where clause on the parameter that the entry `Cloned<T>` needs.
    #[prop_leak(id = 9075, allow_self = false, prelude = true)]
    pub struct WhereBound<T>(pub Cloned<T>)
    where
        T: Clone;

    /// A cyclic bound: the Leaker rejects it.
    #[prop_leak(id = 9076, allow_self = false, prelude = true)]
    pub struct CyclicBound<T: From<Wrap<T>>>(pub T);

    /// A `for<'b>` binder whose lifetime only the entry `Lt<'b, u8>` names: the impl declares it.
    #[prop_leak(id = 9077, allow_self = false, prelude = true)]
    pub struct Binder<F>(pub F)
    where
        F: for<'b> Fn(Lt<'b, u8>);

    // Names that a generic shares with a path or with an item the module declares (the findings of
    // the property test, tests/shadowed_names.rs).

    /// A type named like a generic: `self::U` is this one, and `crate::U` the one at the root of
    /// the crate (`crate::U`).
    pub struct U(pub u8);

    /// The paths `self::U` and `crate::U` end with the name of the generic `U`: they are relative
    /// types, which the Referrer's crate cannot name, not the generic.
    #[prop_leak(id = 9091, allow_self = false, prelude = true)]
    pub struct PathsLikeGeneric<U> {
        pub generic: U,
        pub here: self::U,
        pub root: crate::U,
        pub wrapped: Wrap<self::U>,
        pub both: Pair<crate::U, U>,
    }

    /// The same in the methods of a trait: the generic is the method's or the trait's.
    #[prop_leak(id = 9092, allow_self = false, prelude = true)]
    pub trait PathsLikeGenericTrait<T> {
        fn here<U>(&self, x: U, y: self::U) -> crate::U;
        fn wrapped<U>(&self, x: Wrap<U>) -> Wrap<self::U>;
        fn trait_generic(&self, x: T, y: self::U) -> Pair<crate::U, T>;
    }

    /// A module that declares `X`, next to a struct whose generic is named `X`: `B1<X>` of the
    /// generic is an entry of no module (a type of another module of this crate), and the one of
    /// the declared `X` is an entry of the module. Both are written `crate::base::B1<X>`.
    #[prop_leak(id = 9093)]
    pub mod declared_like_generic {
        pub struct X(pub u8);
        pub struct Generic<X>(pub crate::base::B1<X>, pub crate::base::B2<X, u8>);
        pub struct Declared(pub crate::base::B1<X>);
        pub mod inner {
            pub struct Deeper<X>(pub crate::base::B1<X>);
        }
    }

    // The types in the values of an item (`check_values`, `values = true`): the
    // default of an associated const, a default method, the const, static and function of a
    // module. The Leaker leaks the type in a cast, a `let`, a closure, a turbofish and a qualified
    // path, and the copy replaces it. The Referrer's crate names none of these types, so it
    // compiles only if every one is replaced; the tests compare what the two crates compute.

    /// Types that neither the Referrer's crate nor its `kinds` imports.
    pub mod vals {
        /// Seven bytes: the size of it is the value of the const of the copy, whichever crate it is in.
        #[derive(Debug, Clone, Copy, PartialEq, Default)]
        pub struct Local(pub [u8; 7]);

        impl Local {
            pub const BYTES: usize = 7;

            pub fn count(&self) -> usize {
                self.0.len()
            }
        }

        /// A type alias to cast to.
        pub type Count = usize;

        /// A generic type: the entries that use a generic of a method, or of the trait.
        #[derive(Debug, Clone, Copy, PartialEq, Default)]
        pub struct Holder<T>(pub T);
    }

    /// The values of a trait: a const that sizes a relative type, a const that sizes a relative
    /// type in an array, and default methods that name a relative type in a `let`, a turbofish, a
    /// qualified path, a closure, a cast, with and without the generics of the trait and of the
    /// method. The Referrer's crate declares the copy (`ValueTypesCopy`).
    #[prop_leak(
        id = 9141,
        allow_self = true,
        scope = "mod",
        prelude = true,
        redeclare = true,
        values = true
    )]
    pub trait ValueTypes<T: Default> {
        const N: usize = ::core::mem::size_of::<vals::Local>();
        const M: usize = ::core::mem::size_of::<[vals::Local; 3]>();

        fn size(&self) -> usize {
            let x: vals::Local = ::core::default::Default::default();
            let v = Vec::<vals::Local>::new();
            let y = <vals::Local as ::core::default::Default>::default();
            x.count() + v.len() + y.count() + <vals::Local>::BYTES
        }

        fn closure(&self) -> usize {
            let c = |a: vals::Local, b: vals::Count| -> usize { a.count() + b };
            c(::core::default::Default::default(), 1u8 as vals::Count)
        }

        fn generic<U: Default>(&self, _u: U) -> usize {
            let h: vals::Holder<U> = ::core::default::Default::default();
            let k: vals::Holder<(T, U)> = ::core::default::Default::default();
            ::core::mem::size_of_val(&h)
                + ::core::mem::size_of_val(&k) * 100
                + ::core::mem::size_of::<vals::Holder<T>>() * 10_000
        }
    }

    /// The values of a module: a const, a static and a function that read a type the module does
    /// not declare, and one it does (copied with it).
    #[prop_leak(id = 9142, prelude = true, values = true)]
    pub mod values_in_module {
        pub struct Declared(pub u8);

        pub const SIZE: usize = ::core::mem::size_of::<Declared>()
            + ::core::mem::size_of::<crate::kinds::vals::Local>();

        pub static TAG: usize = ::core::mem::size_of::<crate::kinds::vals::Holder<u16>>();

        pub fn total() -> usize {
            let a: crate::kinds::vals::Local = ::core::default::Default::default();
            let b: Declared = Declared(1);
            a.count() + b.0 as usize + <crate::kinds::vals::Local>::BYTES
        }
    }

    // Conflicts that the Leaker rejects, accepted with `LeakerResult::allow()` of their causes
    // (`prop_leak_accept`): the part that conflicts is copied as written, whole, and the rest is
    // leaked as usual. The Referrer's crate invokes them with `accept`, in a module that names the
    // types and the trait that the copies keep as written (`accepted`).

    /// A type of the module that has a lifetime.
    pub struct Held<'b>(pub &'b u8);

    /// The lifetime `'b` of the method is in the output, and the only input that names it is
    /// `Held<'b>`, a path that is not a trait object. Leaked, the copy would have `'b` in no input
    /// that constrains it (E0195 in the impl of the host), so the Leaker rejects it
    /// (`UnconstrainedLifetime`, the cause is the type). Accepted, `Held<'b>` is copied as
    /// written.
    #[prop_leak_accept(id = 9184, allow_self = true, prelude = true)]
    pub trait HeldInput {
        fn f<'b>(&self, x: Held<'b>) -> &'b u8;
    }

    /// Two traits of the module, a type that takes a trait object with an elided lifetime in its
    /// generics, and one that takes it without.
    pub trait ElidedTr {}
    pub trait PlainTr {}
    pub struct Elided<'e, T: ?Sized + 'e>(pub PhantomData<&'e T>);
    pub struct Bare<T: ?Sized>(pub PhantomData<T>);

    /// `f` takes `Elided<'_, dyn ElidedTr>`: the trait object inherits the lifetime of its path,
    /// which is elided and cannot be named in the impl of the Repeater, so the type cannot be
    /// leaked, and `ElidedTr` is relative (`RelativePath`, the cause is the trait). Accepted, `f` is
    /// copied as written. `g` takes `Bare<dyn PlainTr>`, where `dyn PlainTr` is leaked as usual, and
    /// its entry is still replaced in the copy. (The traits differ: the Visitor finds a leaked type
    /// by its tokens, and would replace a `dyn ElidedTr` in `f` by the entry of `g`, with another
    /// lifetime.)
    #[prop_leak_accept(
        id = 9185,
        allow_self = true,
        prelude = true,
        allow = "::type_leak_test_leaker::kinds::Bare"
    )]
    pub trait ElidedAndPlain {
        fn f(&self, x: ::type_leak_test_leaker::kinds::Elided<'_, dyn ElidedTr>);
        fn g(&self, x: ::type_leak_test_leaker::kinds::Bare<dyn PlainTr>);
    }

    /// A struct analyzed in `enter_generics`, where `Self` has no meaning: a type that names it is
    /// copied as written. `rel` is leaked as usual.
    #[prop_leak_accept(id = 9186, allow_self = true, prelude = true, scope = "generics")]
    pub struct NoSelfMeaning {
        pub rel: Rel,
        pub me: Wrap<Self>,
    }

    /// The same trait object in `f` and `g`, which differ by the path that holds it. In `f` the
    /// path has an elided lifetime, so the trait object takes it, and the type cannot be leaked:
    /// it is copied as written, whole (`RelativePath`, the cause is the trait). In `g` the path has
    /// none, and `dyn SharedTr` is leaked as usual. The Visitor finds a leaked type by its tokens and
    /// its context, and the `dyn SharedTr` of `f` has both: it must leave the part as written, or
    /// `f` would take the entry of `g`, a trait object with another lifetime (E0308 in the impl of
    /// the host).
    pub trait SharedTr {}
    #[prop_leak_accept(
        id = 9187,
        allow_self = true,
        prelude = true,
        allow = "::type_leak_test_leaker::kinds::Bare"
    )]
    pub trait ElidedAndBare {
        fn f(&self, x: ::type_leak_test_leaker::kinds::Elided<'_, dyn SharedTr>);
        fn g(&self, x: ::type_leak_test_leaker::kinds::Bare<dyn SharedTr>);
    }

    /// Like `ElidedAndBare`, with a second type argument in `f` that is relative too. Only the trait
    /// object is reported and accepted (`RelativePath`, the cause is the trait): `dyn SharedTr` in
    /// `f` is copied as written, and `Rel` beside it is an entry like any other, so the Referrer's
    /// crate, which does not name `Rel`, replaces it (E0412 if it were copied as written).
    pub struct ElidedPair<'e, T: ?Sized + 'e, U>(pub PhantomData<(&'e T, U)>);
    #[prop_leak_accept(
        id = 9189,
        allow_self = true,
        prelude = true,
        allow = "::type_leak_test_leaker::kinds::Bare"
    )]
    pub trait ElidedWithSibling {
        fn f(&self, x: ::type_leak_test_leaker::kinds::ElidedPair<'_, dyn SharedTr, Rel>);
        fn g(&self, x: ::type_leak_test_leaker::kinds::Bare<dyn SharedTr>);
    }

    /// `a1` is an entry (`Wrap<Y>`, with the bound `Y: Fn(Rel)` in its context). In `ai`, `impl
    /// Into<Wrap<Y>>` is a generic of the method whose bound names it, and the copy would write the
    /// entry as a path through the Repeater there, in a predicate of the method, where rustc cannot
    /// prove the bound `Y: Fn(Rel)` of its generics: the Leaker rejects it (`UnprovableBound`, the
    /// cause is the type). Accepted, the `impl Into<Wrap<Y>>` is copied as written, and `a1` and
    /// the `Rel` of the where predicate are replaced. The tokens and the context of the `Wrap<Y>`
    /// in `ai` are those of `a1`'s, so the Visitor tells them by where they are, or it would write
    /// the path in the bound of `ai` (E0277).
    #[prop_leak_accept(id = 9188, allow_self = true, prelude = true)]
    pub trait EntryAndBound {
        fn m<Y: Clone>(&self, a1: Wrap<Y>, ai: impl Into<Wrap<Y>>)
        where
            Y: Fn(Rel) -> u8;
    }

    // Items analyzed together, whose conflicts are accepted: a part that a signature made is copied
    // as written in a signature like it only (F48). The Referrer's crate copies each item alone,
    // `accepted` where it names `Held` and `Wrap`, `unnamed` where it names nothing.

    // E0195: `Held<'b>` is the only input of `f` that names `'b`, which its output names too. `g`
    // has no output, and `HeldHolder` is another item, where `Held<'b>` is an entry.
    type_leak_test_macro::shared_leak! {
        id = 9202;
        items [
            pub trait HeldInTwoMethods {
                fn f<'b>(&self, x: Held<'b>) -> &'b u8;
                fn g<'b>(&self, x: Held<'b>);
            }
            pub struct HeldHolder<'b> {
                pub held: Held<'b>,
            }
        ];
    }

    // E0106: the entry `Wrap<T>` adds `'a` to the one lifetime of the inputs of `f`, whose output
    // is elided (`T: 'a` is in its context). `g` and `k` have a receiver, `h` has no output.
    type_leak_test_macro::shared_leak! {
        id = 9203;
        items [
            pub trait ElidedInOneMethod<'a, T: 'a> {
                fn f(x: &u8, y: Wrap<T>) -> &u8;
                fn g(&self, y: Wrap<T>);
                fn h(y: Wrap<T>);
                fn k(&self, y: Wrap<T>) -> &u8;
            }
            pub struct ElidedHolder<'a, T: 'a>(pub PhantomData<&'a T>, pub Wrap<T>);
        ];
    }

    // E0106: the lifetime `'a` that `x` keeps is in the projection of the entry `Wrap<T>` in `y`
    // too, and rustc counts the lifetimes of each input apart (F49). `g` has the type in the input
    // that keeps `'a`, `h` names `'a` in the output, and `k` has a receiver.
    type_leak_test_macro::shared_leak! {
        id = 9204;
        items [
            #[allow(unknown_lints, mismatched_lifetime_syntaxes)]
            pub trait KeptInOtherInput<'a, T: 'a> {
                fn f(x: &'a u8, y: Wrap<T>) -> &u8;
                fn g(x: (&'a u8, Wrap<T>)) -> &u8;
                fn h(x: &'a u8, y: Wrap<T>) -> &'a u8;
                fn k(&self, x: &'a u8, y: Wrap<T>) -> &u8;
            }
            pub struct KeptHolder<'a, T: 'a>(pub PhantomData<&'a T>, pub Wrap<T>);
        ];
    }

    // Errors that the Leaker does not decide on (`prop_leak_result`): its macro carries them with the
    // Referrer, and the Referrer's macro allows some (`allow(..)`), reports the others, and copies
    // the item either way. The Leaker does not allow the prelude, so a relative `Clone` or
    // `core::fmt::Debug` in a bound is an error (`RelativePath`, the cause is the trait). The first
    // three are copied in the Referrer's crate (`accepted`), `Denied` in
    // `tests/crates/diagnostic_result`.

    /// `Clone` as the bound of the trait's generic, as a supertrait and in the bound of a method:
    /// three errors with one cause, dropped by `allow(trait Clone)`. The copy keeps the three as
    /// written.
    #[prop_leak_result(id = 9190, allow_self = true)]
    pub trait PreludeBounds<T: Clone>: Clone {
        fn rel(&self, t: T) -> Rel;
        fn dup<U: Clone>(&self, u: U) -> Pair<Wrap<T>, Wrap<U>>;
    }

    /// A path headed by `core` or `std` with no leading `::`, in a bound of the trait's generic and in
    /// a where predicate: two errors, dropped by `allow(std_crates)`.
    #[prop_leak_result(id = 9191, allow_self = true)]
    pub trait StdBounds<T: core::fmt::Debug> {
        fn rel(&self, t: T) -> Rel;
        fn show<U>(&self, u: Wrap<U>) -> Rel
        where
            U: std::fmt::Display;
    }

    /// A trait of the module in a bound, twice, and a type of the module with an elided lifetime:
    /// two errors (`PlainTr`, `Held<'_>`), allowed by writing their paths, `allow(trait PlainTr, type
    /// Held)`.
    #[prop_leak_result(id = 9192, allow_self = true)]
    pub trait ExplicitCauses<T: PlainTr> {
        fn held(&self, x: Held<'_>) -> Rel;
        fn plain<U: PlainTr>(&self, u: U) -> Wrap<T>;
    }

    /// Three errors, of which the Referrer's macro allows one (`allow(trait Clone)`) and reports
    /// two: `core::fmt::Debug` and `Held<'_>`. The copy is made either way.
    #[prop_leak_result(id = 9193, allow_self = true)]
    pub trait Denied<T: Clone> {
        fn held(&self, x: Held<'_>) -> Rel;
        fn show<U: core::fmt::Debug>(&self, u: Wrap<U>) -> Wrap<T>;
    }

    // Parts that the Referrer must find to leave them as written (tests/findings.rs F18, F43 to
    // F45). Their trait objects take their lifetime from the path around them (`ObjOf`), and the
    // entry `dyn SharedTr` of `Bare<dyn SharedTr>` is `+ 'static`: a copy that replaced the one in a
    // part would not be the original type (E0308 in the Referrer's crate). The paths of `ObjOf` and
    // `Bare` are absolute, as the names of this crate: the copies of the parts resolve in the
    // Referrer's crate.

    /// A path whose third parameter has a lifetime bound: a trait object there is `+ 'a`.
    pub struct ObjOf<'a, T: ?Sized, O: ?Sized + 'a>(pub PhantomData<(&'a (), *const T, *const O)>);
    pub trait ObjMarker {}
    impl<'a, S: ?Sized> ObjMarker for ObjOf<'a, S, dyn SharedTr + 'a> {}

    /// `Held<'b>` is the only input of `f` that holds the lifetime of the method, and it is the
    /// type of a crate (this one, by its own name): an optional entry, which a strict Referrer
    /// replaces. The copy would have `'b` in no input that constrains it, early-bound where the
    /// trait has it late-bound (E0195 in the impl of the host). The Leaker copies it as written for
    /// every Referrer (F18), and the Referrer's crate resolves the path. `g` has it in a box, and in
    /// the output; `Rel` is an entry beside it.
    #[prop_leak(id = 9194, allow_self = true, prelude = true)]
    pub trait OptionalHoldsLifetime {
        fn f<'b>(&self, x: ::type_leak_test_leaker::kinds::Held<'b>, r: Rel) -> &'b u8;
        fn g<'c>(
            &self,
            x: Box<::type_leak_test_leaker::kinds::Held<'c>>,
        ) -> ::type_leak_test_leaker::kinds::Held<'c>;
    }

    /// `Self` is denied (`allow_self = false`), and `T: PartialEq<Self>` names it, so the type that
    /// uses `T` is a conflict (`DeniedSelfInBounds`), copied as written, `dyn SharedTr + 'a`
    /// included. No entry names `Self`: the top-level context of the trait has to record it for
    /// the Visitor to find the part (F44). `g` has the entry `dyn SharedTr`, `+ 'static`.
    #[prop_leak_accept(
        id = 9195,
        allow_self = false,
        prelude = true,
        allow = "::type_leak_test_leaker::kinds::Bare"
    )]
    pub trait DeniedSelfObject<'a> {
        fn f<T: PartialEq<Self>>(
            &self,
            x: ::type_leak_test_leaker::kinds::ObjOf<'a, T, dyn SharedTr>,
        );
        fn g(&self, y: ::type_leak_test_leaker::kinds::Bare<dyn SharedTr>);
    }

    /// Flat recovery: a type whose own path conflicts is written as it is, and the types in it are
    /// leaked as usual (`Leaf` in `Two<T, Leaf>`, beside the entry `Leaf` of `g`). `Two` is
    /// relative, so the Referrer's crate names it. Nothing in it is a trait object that takes its
    /// lifetime from the path, so the part is not whole.
    pub struct Two<A: ?Sized, B: ?Sized>(pub PhantomData<(*const A, *const B)>);
    pub struct Leaf;

    /// `Self` is denied and `T: PartialEq<Self>` names it (`DeniedSelfInBounds`): `Two<T, Leaf>` is
    /// written as it is, `Two` with it, and `Leaf` in it is an entry, as in `g`.
    #[prop_leak_accept(id = 9219, allow_self = false, prelude = true)]
    pub trait DeniedSelfFlat {
        fn f<T: PartialEq<Self>>(&self, x: Two<T, Leaf>);
        fn g(&self, y: Leaf);
    }

    /// The self type of the impl has a bound that names `Self` (`SelfInImplSelfType`, and the `Self`
    /// of the bound, `UnprovableBound`): `Two<T, Leaf>` is written as it is, and `Leaf` in it is an
    /// entry, as in the method.
    #[prop_leak_accept(id = 9220, allow_self = true, prelude = true)]
    impl<T: ::core::convert::AsRef<Option<Self>>> Two<T, Leaf> {
        pub fn g(&self, _y: Leaf) {}
    }

    /// The only inputs that name the lifetime of a method are `Two` and `Held<'b>` in it
    /// (`UnconstrainedLifetime`): leaked, `'b` would be in no input that constrains it, early-bound
    /// where the trait has it late-bound (E0195 in the impl of the host). Both are written as
    /// they are, so `'b` stays in the copy, and `Leaf` is an entry. In `g` every one of three
    /// types is in the way of the lifetime, in `h` two ways, each through a `Held<'d>`.
    #[prop_leak_accept(id = 9221, allow_self = true, prelude = true)]
    pub trait NestedMethodLifetime {
        fn f<'b>(&self, x: Two<Leaf, *mut Held<'b>>) -> &'b u8;
        fn g<'c>(&self, x: Two<Two<Leaf, Held<'c>>, Leaf>) -> &'c u8;
        fn h<'d>(&self, x: Two<Held<'d>, *mut Held<'d>>) -> &'d u8;
    }

    /// The same where an optional entry is on the way to the lifetime: `Wrap<Held<'b>>` of this
    /// crate by its own name is replaced by a strict Referrer unless it is the only one that keeps
    /// `'b` constrained, and then it is written as it is, with `Held<'b>` in it.
    #[prop_leak_accept(id = 9222, allow_self = true, prelude = true)]
    pub trait NestedOptionalLifetime {
        fn f<'b>(
            &self,
            x: Two<Leaf, ::type_leak_test_leaker::kinds::Wrap<Held<'b>>>,
        ) -> &'b u8;
    }

    /// The where predicate of the trait names `Self` (`SelfInTraitPredicate`): the type in it is
    /// copied as written, and the impl of the host must satisfy the predicate with the
    /// original types (F44).
    #[prop_leak_accept(
        id = 9196,
        allow_self = true,
        prelude = true,
        allow = "::type_leak_test_leaker::kinds::Bare"
    )]
    pub trait PredicateObject<'a>
    where
        ::type_leak_test_leaker::kinds::ObjOf<'a, Self, dyn SharedTr>:
            ::type_leak_test_leaker::kinds::ObjMarker,
    {
        fn g(&self, y: ::type_leak_test_leaker::kinds::Bare<dyn SharedTr>);
    }

    /// The same for the self type of an impl, whose bound names `Self` (`SelfInImplSelfType`, and
    /// the `Self` of the bound, `UnprovableBound`): both are copied as written (F44).
    #[prop_leak_accept(
        id = 9197,
        allow_self = true,
        prelude = true,
        allow = "::type_leak_test_leaker::kinds::Bare"
    )]
    impl<'a, T: ::core::convert::AsRef<Option<Self>>>
        ::type_leak_test_leaker::kinds::ObjOf<'a, T, dyn SharedTr>
    {
        pub fn g(&self, _y: ::type_leak_test_leaker::kinds::Bare<dyn SharedTr>) {}
    }

    /// A struct nested in a module (`scope = "mod"`) with a generic whose bound names its `Self`,
    /// and a field that names it too: `NestedSelfInBounds`, copied as written. The macro writes no
    /// entry that names the `Self` of the struct, so the visitor has no `with_nested_self()` (F45):
    /// it must leave the part all the same. `bare` is an entry beside it.
    #[prop_leak_accept(
        id = 9198,
        allow_self = true,
        prelude = true,
        scope = "mod",
        no_nested_self = true,
        allow = "::type_leak_test_leaker::kinds::Bare"
    )]
    pub struct NestedSelfObject<'a, T: ::core::convert::AsRef<Self>> {
        pub obj: ::type_leak_test_leaker::kinds::ObjOf<'a, (Self, T), dyn SharedTr>,
        pub bare: ::type_leak_test_leaker::kinds::Bare<dyn SharedTr>,
    }

    /// An impl nested in the scope of a trait (`View::enter_impl()` in `enter_trait()`), with the
    /// generic of the trait, whose bound names the `Self` of the trait: that `Self` has no name in
    /// the impl, so the type that uses `T` has no context (`UnnameableSelf`), and is copied as
    /// written (F43). The Referrer's macro nests the impl in a method of the trait, as the Leaker
    /// nested the scopes. `Bare<dyn SharedTr>` is an entry beside it.
    pub struct NestedHost;
    type_leak_test_macro::nested_leak! {
        id = 9199;
        scope trait Tr<T: ?Sized + ::core::convert::AsRef<Self>>;
        nested impl<'a> ::type_leak_test_leaker::kinds::NestedHost;
        types [
            <'a, T: ?Sized> => ::type_leak_test_leaker::kinds::ObjOf<'a, T, dyn SharedTr>;
            <> => ::type_leak_test_leaker::kinds::Bare<dyn SharedTr>;
        ];
    }

    // F50 to F52 (`tests/findings.rs`): what a rule after the analysis, a nested item and an
    // unrelated error used to get wrong. Each fails to build on the code before its fix.

    /// F50: the path of the value is relative (`core::mem::size_of`, no leading `::`), and `Rel` is
    /// in its arguments. The path is written as it is and `Rel` is an entry, so
    /// `Rules::std_crates()` after the analysis drops the only error, and the copy builds in a
    /// crate that does not name `Rel` (`stages = true`: the macro makes the copy after each rule,
    /// in a crate of the Referrer that invokes it `clean`, and gets it only if the rule left no
    /// error).
    #[prop_leak_accept(
        id = 9205,
        allow_self = true,
        scope = "mod",
        prelude = true,
        redeclare = true,
        values = true,
        stages = true
    )]
    pub trait ValuePathArguments {
        fn size(&self) -> usize {
            core::mem::size_of::<*const Rel>()
        }
    }

    /// F51: `Wrap<Self>` in the bound of `T` is a leaked type, beside the entry `Link<Self, T>`
    /// that uses `T` and so carries the bound. The copy writes the bound as a path through the
    /// Repeater, and rustc cannot prove the bound of the impl of the entry in the environment that
    /// has the rewritten form (E0277, for the entry in the last field: rustc proves the type is
    /// `Sized` there). `intern_struct()` rejects it (`UnprovableBound`); the nested item with
    /// `nested_self_entries` has to as well.
    #[prop_leak(
        id = 9206,
        allow_self = true,
        scope = "mod",
        prelude = true,
        nested_self_entries = true
    )]
    pub struct LeakedInBound<T: AsRef<Wrap<Self>>> {
        pub rel: Rel,
        pub link: Link<Self, T>,
    }

    /// A relative trait that takes a type: an unrelated error in a method of `OptionalNextToError`.
    pub trait GenTr<A: ?Sized> {}

    /// F52: the type of this crate by its own name (`Wrap<(T, Wrap<u8>)>`, `Wrap<u8>`) is an
    /// optional entry, which a strict Referrer replaces. The where predicate of the trait names
    /// both in an `Fn`, and `Y: GenTr<Self::A0>` is copied as written (`GenTr` is relative): its
    /// `Self::A0` is resolved by the impl of the host, whose predicate must be the one of the
    /// trait. So the optional entries in the predicate stay as written, as they do beside
    /// `Y: Into<Self::A0>` (E0277 "expected an `Fn(&Wrap<..>)` closure, found `T`" otherwise).
    #[prop_leak_accept(id = 9207, allow_self = true, prelude = true)]
    pub trait OptionalNextToError<T>
    where
        T: ::core::ops::Fn(
            &::type_leak_test_leaker::kinds::Wrap<(T, ::type_leak_test_leaker::kinds::Wrap<u8>)>,
        ) -> u8,
    {
        type A0;
        fn m<Y>(&self)
        where
            Y: GenTr<Self::A0>;
    }

    // Issue 3 of `tests/downstream_findings.rs`: the names `Rules::prelude()` took for the prelude.

    /// Issue 3: `None`, `Some`, `Ok` and `Err` in values, and
    /// `DoubleEndedIterator + ExactSizeIterator` in a bound, are names of the prelude of every
    /// edition. `Rules::prelude()` refused them (`use absolute path: 'None'`), so the Leaker
    /// rejected the item, which the Referrer's crate then lacks (`PreludeNamesCopy`). The copy
    /// writes them as they are; the entries are the relative types.
    #[prop_leak(
        id = 9208,
        allow_self = true,
        scope = "mod",
        prelude = true,
        redeclare = true,
        values = true,
        bodies = true
    )]
    pub trait PreludeNames {
        const NOTHING: Option<vals::Local> = None;

        fn pick(&self, n: u8) -> Option<vals::Local> {
            match n {
                0 => None,
                _ => Some(Default::default()),
            }
        }

        fn check(&self, n: u8) -> Result<vals::Local, u8> {
            if n > 0 {
                Ok(Default::default())
            } else {
                Err(n)
            }
        }

        fn count<I: DoubleEndedIterator + ExactSizeIterator>(&self, it: I) -> usize {
            it.len() + it.rev().count()
        }
    }

    // F56 (`tests/findings.rs`): an enum discriminant and the default of a const parameter are read
    // like a value, with or without `check_values`. Each fails to build on the code before the fix.

    /// F56: the discriminants bind names (`x`, `y`, `a`, `b`, `z`, `K`): they are local, not
    /// relative paths. The Leaker rejected them (`use absolute path: 'x'`), so the Referrer's crate
    /// lacked `DiscriminantBindings`. The copy writes them as they are, in a crate that has none of
    /// the Leaker's names.
    #[prop_leak(id = 9215, allow_self = true, scope = "mod")]
    pub enum DiscriminantBindings {
        Let = { let x = 3; x },
        Arm = match 5 { y => y + 1 },
        Pair = { let (a, b) = (1, 2); a * 10 + b },
        Typed = { let z: u8 = 7; z as isize },
        Item = { const K: isize = 40; K + 2 },
    }

    /// F56: the default of the const parameter binds a name (`n`). It was rejected as well.
    #[prop_leak(id = 9216, allow_self = true, scope = "mod")]
    pub struct DefaultBinding<const N: usize = { let n = 2; n * 2 }>(pub [u8; N]);

    /// A macro of this crate, which the Referrer's crate does not have.
    macro_rules! rel_two {
        () => {
            2
        };
    }

    /// F56: a discriminant that calls a macro of this crate (`rel_two!`) by a relative path. The
    /// Leaker did not report it, and the copy in the Referrer's crate could not resolve the macro.
    /// It is rejected now: the Referrer's macro expands to nothing.
    #[prop_leak(id = 9217, allow_self = true, scope = "mod")]
    pub enum RelativeMacroDiscriminant {
        Two = rel_two!(),
    }

    /// A struct of this crate, which the Referrer's crate does not have.
    pub struct RelPoint {
        pub a: usize,
    }

    /// F56: the default of a const parameter that writes a struct of this crate as a literal
    /// (`RelPoint { a: 1 }`). Not reported before, and the copy could not resolve `RelPoint`.
    #[prop_leak(id = 9218, allow_self = true, scope = "mod")]
    pub struct LiteralDefault<const N: usize = { RelPoint { a: 1 }.a }>(pub [u8; N]);

    // F64 (`tests/findings.rs`): a bound that stays in the context of an entry and names its
    // generic through a type alias, next to an equation of the generic. It fails to build on the
    // code before the fix.

    /// F64: the bounds of the generics name them through the alias `WIt`, which the Leaker cannot
    /// tell from a struct. `Wrap<Self>` in the predicate of `m` keeps `U: Fn(WIt<U>)` whole, and
    /// `Wrap<V>` in the predicate of `n`, next to the equation of `V` there, has `V: Fn(WIt<V>)`:
    /// rustc proves neither in the environment of the copy (E0277 "expected an `Fn(u8)` closure,
    /// found `U`"). The Leaker rejects both, so the Referrer's macro expands to nothing. The rule
    /// of the Leaker allows the alias, as it does where the crate has rules of its own for its
    /// paths.
    #[prop_leak(
        id = 9223,
        allow_self = true,
        prelude = true,
        allow = "::type_leak_test_leaker::kinds::WIt"
    )]
    pub trait AliasBound<T, U: Iterator<Item = u8>, V: Iterator>
    where
        U: Fn(::type_leak_test_leaker::kinds::WIt<U>) -> u8,
        V: Fn(::type_leak_test_leaker::kinds::WIt<V>) -> u8,
    {
        fn m(&self, a: u8)
        where
            Wrap<Self>: Default;

        fn n(&self, a: u8)
        where
            V: Iterator<Item = u8>,
            Wrap<V>: Default;
    }

    // F65 and F66 (`tests/findings.rs`): the Referrer's crate uses the copy at an instantiation
    // that leaves the defaults out (`inst(..)`). Each fails to build on the code before the fix.

    /// F65: the default `Defd<T>` of a trait nested in a module, whose bound `T: AsRef<Self>` names
    /// `Self`. `Referrer::complete_args()` found no entry for it and left it as written, a type that
    /// the Referrer's crate does not import (E0412). The trait is not declared again there: the
    /// use site completes the arguments.
    #[prop_leak(id = 9224, allow_self = true, scope = "mod", prelude = true)]
    pub trait NestedSelfBoundDefault<T: AsRef<Self>, U = Defd<T>> {
        fn get(&self, t: T, u: U);
    }

    /// F66: a braced const parameter as the length of an array, in a type that is no entry. The use
    /// site writes the argument of `N` instead of it: `[u8; { { N } }]` is rejected by rustc.
    /// `N` is named by a bound, so the use site passes it by its own name.
    #[prop_leak(id = 9225, allow_self = true, prelude = true)]
    pub struct BracedLength<const N: usize, T: Into<[u8; N]>>(pub T, pub [u8; { N }], pub Rel);

    /// F66: the same in the arguments of a path (`IntoIter<u8, { N }>`).
    #[prop_leak(id = 9226, allow_self = true, prelude = true)]
    pub struct BracedArgument<const N: usize, T: Into<[u8; N]>>(
        pub T,
        pub ::core::array::IntoIter<u8, { N }>,
        pub Rel,
    );

    /// F66: the default of a type parameter that names a braced const parameter, which
    /// `Referrer::complete_args()` substitutes.
    #[prop_leak(id = 9227, allow_self = true, prelude = true)]
    pub struct BracedDefault<const N: usize, T: Into<[u8; N]>, U = [u8; { N }]>(
        pub T,
        pub U,
        pub Rel,
    );

    // F67 (`tests/findings.rs`): a trait object bounded by a method lifetime under a type of a crate
    // that the Leaker does not allow. They fail to build on the code before the fix.

    /// F67: `Box<dyn Fn(Rel) + 'x>` is the only input of `f` that holds the lifetime, in the
    /// arguments of a `Vec`; both are of `::std`, which the rule does not allow. The Leaker leaked
    /// the box, so the copy had `'x` in a projection only, early-bound where the trait has it
    /// late-bound (E0195 in the impl of the host). Both are written as they are now, and the only
    /// entry is `Rel`.
    #[prop_leak(id = 9228, allow_self = true, prelude = true)]
    pub trait ObjectInVec {
        fn f<'x>(
            &self,
            a: ::std::vec::Vec<::std::boxed::Box<dyn ::core::ops::Fn(Rel) + 'x>>,
        ) -> &'x u8;
    }

    /// F67: the same under an `Option`, under two boxes, and in a tuple.
    #[prop_leak(id = 9229, allow_self = true, prelude = true)]
    pub trait ObjectUnderTypes {
        fn f<'a>(
            &self,
            x: ::std::vec::Vec<::std::option::Option<::std::boxed::Box<dyn ::core::ops::Fn(Rel) + 'a>>>,
        ) -> &'a u8;
        fn g<'b>(
            &self,
            x: ::std::vec::Vec<::std::boxed::Box<::std::boxed::Box<dyn ::core::ops::Fn(Rel) + 'b>>>,
        ) -> &'b u8;
        fn h<'c>(
            &self,
            x: ::std::vec::Vec<(::std::boxed::Box<dyn ::core::ops::Fn(Rel) + 'c>, u8)>,
        ) -> &'c u8;
    }

    // F68 (`tests/findings.rs`): the self type of an impl is a type of this crate by an absolute path
    // that the rule does not allow, an optional entry; a bound of a generic names an optional type,
    // another names `Self`, and a predicate is on an entry that uses the generic. A Referrer that
    // replaced the self type had E0277 in the copy. It fails to build on the code before the fix.

    /// The self type of the impl below, and the type that a bound names.
    pub struct OptSelf<const N: usize>;
    pub struct OptBound;

    /// F68: `Wrap<X>: Default` is a predicate on an entry whose context has `X: Fn(&Self)`, and
    /// `Self` is the optional type `OptSelf<0>`, which the bound `X: AsRef<(&'static OptBound,
    /// bool)>` hid from the check. The strict copy was E0277 ("expected an `Fn(&OptSelf<0>)`
    /// closure, found `X`"). `OptSelf<0>` is no entry now, and the copy writes it as it is.
    #[prop_leak(id = 9230, allow_self = true, prelude = true)]
    impl ::type_leak_test_leaker::kinds::OptSelf<0> {
        pub fn m<X>(&self, _a0: X)
        where
            X: ::core::convert::AsRef<(&'static ::type_leak_test_leaker::kinds::OptBound, bool)>,
            X: ::core::ops::Fn(&Self) -> u8,
            Wrap<X>: ::core::default::Default,
        {
        }
    }

    // F69 (`tests/findings.rs`): the copy is used at an instantiation that gives the type parameter
    // a type with a lifetime (`inst(..)` with `lifetime_args`: `&'static str`, `(&'static str, u8)`,
    // ..), where the parameter is in the inputs of a function pointer or `Fn(..)` whose output has an
    // elided lifetime. The use site must name that lifetime. Each fails to build on the code before
    // the fix.

    /// F69: a function pointer over a type of `::core` (no entry), so the use site writes the
    /// argument of `T` into it, and one with a binder whose input names the lifetime. `Rel` is the
    /// only entry.
    #[prop_leak(id = 9231, allow_self = true, prelude = true)]
    pub struct ElidedPointer<T>(
        pub fn(&::core::cell::Cell<[T; 1]>) -> &u8,
        pub for<'e> fn(&'e T) -> &u8,
        pub fn(&T, u8) -> &u8,
        pub Rel,
    );

    /// F69: the same in a trait object and a function pointer, in the arguments of a method (the host
    /// implements the trait for the instantiation).
    #[prop_leak(id = 9232, allow_self = true, prelude = true)]
    pub trait ElidedMethods<T> {
        fn call(&self, f: &dyn ::core::ops::Fn(&T) -> &u8, g: fn(&T) -> &u8) -> Rel;
    }

    /// F69: the default of a parameter that is such a function pointer, which
    /// `Referrer::complete_args()` completes when the use site leaves it out.
    #[prop_leak(id = 9233, allow_self = true, prelude = true)]
    pub struct ElidedDefault<T, U = fn(&T) -> &u8>(pub T, pub U, pub Rel);

    // F70 (`tests/findings.rs`): the same where an entry is in the inputs, and the Referrer replaces
    // it (`strict`): the path through the Repeater names the lifetime that the use site gives `T`.
    // Each fails to build on the code before the fix.

    /// F70: `Vec<T>` of `::std` is an optional entry that a strict Referrer replaces, in a function
    /// pointer. The Leaker leaks it alone.
    #[prop_leak(id = 9234, allow_self = true, prelude = true)]
    pub struct ElidedEntry<T>(pub fn(&::std::vec::Vec<T>) -> &u8, pub Rel);

    /// F70: the same in the sugar of `Fn(..)` of a trait object, which no reference or box holds
    /// (it would be the entry).
    #[prop_leak(id = 9235, allow_self = true, prelude = true)]
    pub struct ElidedEntryObject<T>(
        pub ::core::marker::PhantomData<dyn ::core::ops::Fn(&::std::vec::Vec<T>) -> &u8>,
        pub Rel,
    );

    /// A type with const parameters of the two types that `RepeaterSlots` puts where no other goes.
    pub struct Keyed<const N: u64, const M: u128>;

    /// The marker type of 9236.
    pub struct Marker9236;

    /// 9236: const generics of type `u64` and `u128` in the arguments of the Repeater. `RepeaterSlots`
    /// puts a `u128` into the `KEY` argument (with the id and the index, `repeater_key()`) and a
    /// `u64` into the slot `U64`, so the entries use one, the other and both.
    #[prop_leak(
        id = 9236,
        allow_self = false,
        prelude = true,
        marker = "::type_leak_test_leaker::kinds::Marker9236"
    )]
    pub struct KeySlots<const N: u64, const M: u128> {
        pub both: Keyed<N, M>,
        pub only_u64: Wrap<Keyed<N, 0>>,
        pub only_u128: Wrap<Keyed<0, M>>,
        pub rel: Rel,
    }
}

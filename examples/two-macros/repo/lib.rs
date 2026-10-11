//! Crate A of the two-macro example: it defines a trait whose signatures name types that mean
//! something only here (a relative path, a generic wrapper, `Self`), and exports it with
//! `#[export_trait]`, so that `app` can implement it without naming any of those types.

/// The macros, for the `macro_rules!` that `#[export_trait]` writes (`$crate::__macros::..`):
/// the crates that implement the trait need no dependency on the crate of the macros.
#[doc(hidden)]
pub use repo_macros as __macros;
pub use repo_macros::{export_trait, forward};

/// The one trait that both crates see. `#[export_trait]` implements it, in this crate, for the
/// types the signatures of an exported trait name; the copy in another crate names them through
/// it. It must be at the root of this crate, and local to it (the impls are for any type).
///
/// `T` encodes the generics the type uses, `ID` the trait the entry is from, and `NTH` which
/// entry of it.
pub trait Repeater<T: ?Sized, const ID: u64, const NTH: usize> {
    type Type: ?Sized;
}

/// A type in a sibling module: a relative path (`model::Record`) in the trait below.
pub mod model {
    #[derive(Debug, PartialEq)]
    pub struct Record(pub u32);
}

/// A generic wrapper: `Page<K>` uses the generic `K` of the trait.
#[derive(Debug, PartialEq)]
pub struct Page<T>(pub Vec<T>);

/// A trait with the three kinds of types that cannot be copied as they are.
#[export_trait]
pub trait Repo<K> {
    type Error;

    /// A generic wrapper of a generic of the trait.
    fn get(&self, key: K) -> Page<K>;
    /// A type of a sibling module.
    fn schema(&self) -> model::Record;
    /// A type that names `Self`.
    fn failures(&mut self) -> Page<Self::Error>;
    /// A generic wrapper of a generic of the method.
    fn many<U>(&self, items: Vec<U>) -> Page<U>;
}

/// A trait that cannot be exported: `Clone` is a relative path in a bound, and `Record<'_>` has
/// an elided lifetime. `cargo check --features bad` reports both.
#[cfg(feature = "bad")]
#[export_trait]
pub trait Bad: Clone {
    fn get(&self, record: Record<'_>);
}

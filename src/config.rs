//! Configuration of the [`Leaker`](crate::Leaker).

use syn::*;

/// The paths that a [`Leaker`](crate::Leaker) leaves as written instead of leaking them. Pass it
/// to [`Leaker::from_config()`](crate::Leaker::from_config).
///
/// ```
/// # use type_leak::*;
/// # use syn::*;
/// let mut config = LeakerConfig::new();
/// config.allow_primitive().allow_std_prelude();
/// config.allowed_paths.push(parse_quote!(::serde)); // you promise it is the same crate on both sides
/// let mut leaker = Leaker::from_config(config);
/// // `Vec` and `u8` are allowed, so only `MyType` is leaked.
/// let item: ItemStruct = parse_quote!(struct S { a: Vec<u8>, b: MyType });
/// leaker.intern_struct(&item, true).unwrap();
/// assert_eq!(leaker.finish().iter().count(), 1);
/// ```
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct LeakerConfig {
    /// The allowed paths. See [`Leaker::allowed_paths`](crate::Leaker::allowed_paths). Allow only
    /// crates that are the same on both sides ([External crates](crate::rules#external-crates)).
    pub allowed_paths: Vec<Path>,
}

impl LeakerConfig {
    /// Creates a configuration that allows no path.
    pub fn new() -> Self {
        Default::default()
    }

    /// Allows the primitive types (`u8`, `bool`, `str`, ...).
    pub fn allow_primitive(&mut self) -> &mut Self {
        self.allowed_paths.extend([
            parse_quote!(bool),
            parse_quote!(char),
            parse_quote!(str),
            parse_quote!(u8),
            parse_quote!(u16),
            parse_quote!(u32),
            parse_quote!(u64),
            parse_quote!(u128),
            parse_quote!(usize),
            parse_quote!(i8),
            parse_quote!(i16),
            parse_quote!(i32),
            parse_quote!(i64),
            parse_quote!(i128),
            parse_quote!(isize),
            parse_quote!(f32),
            parse_quote!(f64),
        ]);
        self
    }

    /// Allows the names of the standard prelude (`Vec`, `Option`, `Clone`, ...).
    ///
    /// Without it, a relative `Clone` in `trait Tr: Clone` is an error, and `Vec<T>` is leaked
    /// whole. You promise that the copy does not shadow the names.
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// let item: ItemTrait = parse_quote!(trait Tr: Clone { fn f(&self) -> Vec<Rel>; });
    /// assert!(Leaker::new().intern_trait(&item, true).is_err());
    ///
    /// let mut config = LeakerConfig::new();
    /// config.allow_std_prelude();
    /// let mut leaker = Leaker::from_config(config);
    /// leaker.intern_trait(&item, true).unwrap();
    /// // `Clone` and `Vec` are left as written. The only entry is `Vec<Rel>`: a bare `Rel` may be
    /// // a const, so the type that holds it is leaked whole.
    /// assert_eq!(leaker.finish().iter().count(), 1);
    /// ```
    pub fn allow_std_prelude(&mut self) -> &mut Self {
        self.allowed_paths.extend([
            parse_quote!(Box),
            parse_quote!(String),
            parse_quote!(Vec),
            parse_quote!(Option),
            parse_quote!(Result),
            parse_quote!(Copy),
            parse_quote!(Send),
            parse_quote!(Sized),
            parse_quote!(Sync),
            parse_quote!(Unpin),
            parse_quote!(Drop),
            parse_quote!(Fn),
            parse_quote!(FnMut),
            parse_quote!(FnOnce),
            parse_quote!(Clone),
            parse_quote!(PartialEq),
            parse_quote!(PartialOrd),
            parse_quote!(Eq),
            parse_quote!(Ord),
            parse_quote!(AsRef),
            parse_quote!(AsMut),
            parse_quote!(Into),
            parse_quote!(From),
            parse_quote!(Default),
            parse_quote!(Iterator),
            parse_quote!(IntoIterator),
            parse_quote!(Extend),
            parse_quote!(FromIterator),
            parse_quote!(ToOwned),
            parse_quote!(ToString),
            parse_quote!(drop),
        ]);
        self
    }

    /// Allows paths that start with `crate`.
    ///
    /// Use it only if the copy is in the same crate as the item: `crate::Foo` names another type
    /// in another crate.
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// let item: ItemStruct = parse_quote!(struct S { a: crate::Foo });
    /// // `crate::Foo` is leaked by default...
    /// let mut leaker = Leaker::new();
    /// leaker.intern_struct(&item, true).unwrap();
    /// assert_eq!(leaker.finish().iter().count(), 1);
    /// // ...and left as written if the copy is in the same crate.
    /// let mut config = LeakerConfig::new();
    /// config.allow_crate();
    /// let mut leaker = Leaker::from_config(config);
    /// leaker.intern_struct(&item, true).unwrap();
    /// assert!(leaker.finish().is_empty());
    /// ```
    pub fn allow_crate(&mut self) -> &mut Self {
        self.allowed_paths.push(parse_quote!(crate));
        self
    }
}

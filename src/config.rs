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
    /// The allowed paths. See [`Leaker::allowed_paths`](crate::Leaker::allowed_paths), which also
    /// tells what happens to an absolute path that is not allowed. Allow only crates that are the
    /// same on both sides.
    pub allowed_paths: Vec<Path>,
    /// Set by [`LeakerConfig::check_values()`].
    pub(crate) check_values: bool,
}

impl LeakerConfig {
    /// Creates a configuration that allows no path and does not check values.
    pub fn new() -> Self {
        Default::default()
    }

    /// Makes the [`Leaker`](crate::Leaker) read the values of an item too, and reject a relative
    /// path in them.
    ///
    /// The copy writes a value as it is: it cannot go through the Repeater, which carries types.
    /// So a path in a value that depends on where it is written breaks the copy, and the Leaker
    /// does not tell unless this option is on. It then reads:
    ///
    /// - the default value of an associated const of a trait, and the value of an associated const
    ///   of an impl;
    /// - the value of a const or static item (of a module, or nested);
    /// - the body of a function: a default method of a trait, a method of an impl, a function item.
    ///
    /// An enum discriminant and an array length are read with or without the option: a relative
    /// path in a discriminant is an error, and a type in it or in a length is leaked, as ever.
    ///
    /// A path in a value is rejected unless it means the same wherever it is written: it is
    /// absolute or starts with `$crate`, it is allowed ([`allowed_paths`](LeakerConfig::allowed_paths),
    /// also the names of [`allow_primitive()`](LeakerConfig::allow_primitive) and
    /// [`allow_std_prelude()`](LeakerConfig::allow_std_prelude): `Some` and `None` are not among
    /// them, write `::core::option::Option::Some`), it starts with a generic or `Self` in scope,
    /// or with a name that the value itself declares (a `let` binding, a closure or function
    /// parameter, a `match`, `if let` or `for` pattern, an item or a `use` in a block). The name
    /// of the item that holds the value is relative like any other. (A name in a pattern is a
    /// binding unless it starts with a capital letter, as the lint `non_snake_case` expects: then
    /// it is a path to a constant, a unit struct or a variant.) A type in a value
    /// (`size_of::<Rel>()`, `let x: Rel`) is rejected too: it is not leaked, since the copy writes
    /// a value as it is. Attributes (`#[allow(..)]`, `#[derive(..)]`) are not read, in a value or
    /// anywhere else in the item.
    ///
    /// The path of a macro call is checked like any other (`::std::println!`, not `println!`), and
    /// its arguments are read as expressions, and the names a format string captures (`"{x}"`) as
    /// paths. A macro whose arguments are not expressions (`cfg!`, `stringify!` and `matches!` are
    /// the exceptions), a `macro_rules!`, a glob import and syntax that `syn` does not parse are
    /// rejected, since the Leaker cannot see what they name.
    ///
    /// The error is the one of a relative path, and its note says which value it is in:
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// let item: ItemTrait = parse_quote! {
    ///     trait Tr { const N: usize = ::core::mem::size_of::<inner::Local>(); }
    /// };
    /// // Without the option the value is not read: the copy breaks without an error.
    /// assert!(Leaker::new().intern_trait(&item, true).is_ok());
    ///
    /// let mut config = LeakerConfig::new();
    /// config.check_values();
    /// let err = Leaker::from_config(config).intern_trait(&item, true).unwrap_err();
    /// let error = err.items().next().unwrap();
    /// assert_eq!(error.message, "use absolute path: 'inner :: Local'");
    /// assert_eq!(
    ///     error.note(),
    ///     Some("it is in the default value of `N`, which the copy writes as it is")
    /// );
    /// ```
    pub fn check_values(&mut self) -> &mut Self {
        self.check_values = true;
        self
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
    /// `Sized` is one of them, as a bound (`where Self: Sized`): a trait named `Sized` in the copy
    /// would be the one required. The relaxed bound `?Sized` needs no option, since rustc accepts
    /// `?` only on the language's `Sized` (a local trait of that name makes it an error, or before
    /// Rust 1.86 a warning that ignores it): it is left as written in a bound of a generic, a
    /// where predicate, a supertrait, an associated type and `impl Trait`.
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// let relaxed: ItemTrait = parse_quote!(trait Tr<T: ?Sized> { fn f(&self, t: &T); });
    /// assert!(Leaker::new().intern_trait(&relaxed, true).is_ok());
    /// let required: ItemTrait = parse_quote!(trait Tr { fn f(&self) where Self: Sized; });
    /// let err = Leaker::new().intern_trait(&required, true).unwrap_err();
    /// assert_eq!(err.to_string(), "use absolute path: 'Sized'");
    /// ```
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

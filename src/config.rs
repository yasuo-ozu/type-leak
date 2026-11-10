//! Configuration of the [`Leaker`](crate::Leaker).

use crate::allow::Allow;
use crate::error::Cause;
#[cfg(doc)]
use crate::view::View;
use syn::*;

/// What a [`Leaker`](crate::Leaker) reads, leaves as written and accepts. Pass it to
/// [`Leaker::from_config()`](crate::Leaker::from_config).
///
/// The `allow*` methods accept the same errors as those of [`LeakerResult`](crate::LeakerResult).
/// Here the path rules also keep the paths out of the entries; on a result they only drop errors.
///
/// ```
/// # use type_leak::*;
/// # use syn::*;
/// let mut config = LeakerConfig::new();
/// config.allow_primitive().allow_prelude();
/// config.allow_path(parse_quote!(::serde)); // you promise it is the same crate on both sides
/// let mut leaker = Leaker::from_config(config);
/// // `Vec` and `u8` are allowed, so only `MyType` is leaked.
/// let item: ItemStruct = parse_quote!(struct S { a: Vec<u8>, b: MyType });
/// leaker.intern_struct(&item, true);
/// assert_eq!(leaker.result().finish().unwrap().iter().count(), 1);
/// ```
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct LeakerConfig {
    /// Set by the `allow*()` methods.
    pub(crate) allow: Allow,
    /// Set by [`LeakerConfig::check_values()`].
    pub(crate) check_values: bool,
    /// Set by [`LeakerConfig::nested_self_entries()`].
    pub(crate) nested_self_entries: bool,
}

impl LeakerConfig {
    /// Creates a configuration that allows nothing and does not check values.
    pub fn new() -> Self {
        Default::default()
    }

    /// Makes the [`Leaker`](crate::Leaker) read the values of an item too: it leaks the types in
    /// type positions of a value and rejects a relative path that is not a type.
    ///
    /// Without it a value is not read, so a relative path in it breaks the copy and the Leaker does
    /// not tell. It reads the values of consts and statics, the defaults and values of associated
    /// consts, and the bodies of functions and methods. Enum discriminants and array lengths are
    /// read either way.
    ///
    /// **Types** in a type position (a cast, a `let`, a closure parameter or output, the generic
    /// arguments of a path or method call, the self type of a qualified path) are analyzed like any
    /// type of the item, and are entries if their path is relative
    /// ([`Referrer::required()`](crate::Referrer::required) lists them). A bare identifier argument
    /// is read as a type: write a const in braces with an absolute path. A type with `_` is not
    /// leaked whole, but its parts are.
    ///
    /// **Other paths** (a function, a constant, a struct literal, a variant, the trait of a
    /// qualified path) are rejected unless they mean the same wherever they are written: they are
    /// absolute or start with `$crate`, they are allowed
    /// ([`allow_path()`](LeakerConfig::allow_path) and the like), or they start with a generic or
    /// `Self` in scope or a name that the value itself declares.
    /// [`LeakerResult::allow()`](crate::LeakerResult::allow) accepts such a trait, and the types in
    /// it are still replaced.
    ///
    /// Types that name something the value declares, types in macro arguments and types in nested
    /// items are read, not leaked: a relative path in them is rejected as above. Attributes are not
    /// read. The path of a macro is checked like any other (`::std::println!`) and its arguments
    /// are read as expressions. A macro whose arguments are not expressions (`cfg!`, `stringify!`
    /// and `matches!` excepted), a `macro_rules!`, a glob import and syntax that `syn` does not
    /// parse are rejected. The note of an error says which value it is in.
    pub fn check_values(&mut self) -> &mut Self {
        self.check_values = true;
        self
    }

    /// Accepts a type that names the `Self` of a struct, an enum or a union nested in the analyzed
    /// syntax ([`View::visit_item()`](crate::View::visit_item), the items of
    /// [`Leaker::intern_mod()`](crate::Leaker::intern_mod)) when a bound of the generics it uses
    /// names that `Self` too: `Rel2<Self, T>` in `struct S<T: AsRef<Self>>`.
    ///
    /// Without it such an entry is an error
    /// ([`Reason::NestedSelfInBounds`](crate::Reason::NestedSelfInBounds)). With it the entry is
    /// for any type that the bound holds for
    /// ([`SelfRequirement::Any`](crate::SelfRequirement::Any)), and the copy passes its own type as
    /// the implementor. A default of a parameter is not changed (E0735), and a bound that names a
    /// leaked type that names `Self` (`T: AsRef<Rel<Self>>`) is still an error.
    ///
    /// You promise two things:
    ///
    /// - The declaration of the leaked type has no trait bound on the parameter that receives
    ///   `Self` (`struct Rel2<A: Clone, B>`), else the impl fails in the Leaker's crate (E0277).
    ///   Write the type with an absolute path, or analyze the item with its own `intern_*` call.
    /// - rustc before 1.70 fails with E0275 if the copy's field is the last field of a struct: put
    ///   it first, or add `where Self: Sized` to the item.
    pub fn nested_self_entries(&mut self) -> &mut Self {
        self.nested_self_entries = true;
        self
    }

    /// Accepts the errors with an equal cause (the same kind and tokens, spans ignored): the part
    /// is copied as written. See [`LeakerResult::allow()`](crate::LeakerResult::allow).
    pub fn allow(&mut self, cause: Cause) -> &mut Self {
        self.allow.cause(cause);
        self
    }

    /// Leaves the paths that start with `path` as written instead of leaking them, such as
    /// `::serde`.
    ///
    /// A path matches if it starts with `path` and both have a leading `::` or neither. Its generic
    /// arguments are still analyzed. You promise that an allowed crate is the same on both sides.
    ///
    /// Another absolute path (`::serde::Wrap<(Rel,)>`) is leaked as an *optional entry*
    /// ([`Referrer::must()`](crate::Referrer::must) is `false`), which the Referrer replaces or not
    /// ([`allow_ambiguous_crate_name()`](crate::LeakerResult::allow_ambiguous_crate_name)).
    /// `::core::..` and `$crate::..` are never optional. An absolute type that names the `Self` of
    /// a nested struct, enum or union (`::serde::Wrap<Self>` in [`View::visit_item()`]) is copied
    /// as written whatever the Referrer chooses
    /// ([`Visitor::with_nested_self()`](crate::Visitor::with_nested_self)).
    pub fn allow_path(&mut self, path: Path) -> &mut Self {
        self.allow.path(path);
        self
    }

    /// Allows the primitive types (`u8`, `bool`, `str`, ...).
    pub fn allow_primitive(&mut self) -> &mut Self {
        self.allow.primitive();
        self
    }

    /// Allows the names of the standard prelude (`Vec`, `Option`, `Clone`, ...).
    ///
    /// Without it, a relative `Clone` in `trait Tr: Clone` is an error, and `Vec<T>` is leaked
    /// whole. You promise that the copy does not shadow the names.
    ///
    /// The names are those of the prelude of every edition, as of Rust 1.56. A name that a later
    /// prelude added is left out, as the copy may be older: `TryFrom`, `TryInto`, `FromIterator`,
    /// `Future`, `IntoFuture`, `size_of`, `size_of_val`, `align_of` and `align_of_val`. The help
    /// gives their absolute paths. `Sized` is allowed as a bound; `?Sized` needs no option.
    pub fn allow_prelude(&mut self) -> &mut Self {
        self.allow.prelude();
        self
    }

    /// Allows the relative paths that start with `std`, `core` or `alloc`. A path with a leading
    /// `::` is context-free in any case.
    pub fn allow_std_crates(&mut self) -> &mut Self {
        self.allow.std_crates();
        self
    }

    /// Allows paths that start with `crate`.
    ///
    /// Use it only if the copy is in the same crate as the item: `crate::Foo` names another type
    /// in another crate.
    pub fn allow_crate(&mut self) -> &mut Self {
        self.allow.crate_root();
        self
    }

    /// Accepts the errors found in a value: a const or static value, the body of a function, a
    /// trait's default value or body. It has an effect with
    /// [`check_values()`](LeakerConfig::check_values) only.
    pub fn allow_values(&mut self) -> &mut Self {
        self.allow.values();
        self
    }

    /// Accepts the errors found in a default: a default body or an associated const default of a
    /// trait, and the default of a generic parameter (`U = Rel<'_>`, `T = Wrap<Self>`).
    pub fn allow_default(&mut self) -> &mut Self {
        self.allow.defaults();
        self
    }
}

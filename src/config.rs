//! Configuration of the [`Leaker`](crate::Leaker).

use crate::allow::Allow;
use crate::error::Cause;
#[cfg(doc)]
use crate::view::View;
use syn::*;

/// What a [`Leaker`](crate::Leaker) reads, leaves as written and accepts. Pass it to
/// [`Leaker::from_config()`](crate::Leaker::from_config).
///
/// The `allow*` methods are those of [`LeakerResult`](crate::LeakerResult), and accept the same
/// errors: `from_config(config.allow_prelude())` and `from_config(config).result().allow_prelude()`
/// both drop them. Here the path rules also make the paths context-free during the analysis, so
/// they are not leaked at all; on a result they only drop the errors.
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
    /// them and rejects a relative path that is not a type.
    ///
    /// The copy writes a value as it is: an expression cannot go through the Repeater, which
    /// carries types. A type in it can, since the [`Visitor`](crate::Visitor) replaces a type where
    /// it stands. Without this option a value is not read, so a relative path in it, a type or
    /// not, breaks the copy and the Leaker does not tell. With it, the Leaker reads:
    ///
    /// - the default value of an associated const of a trait, and the value of an associated const
    ///   of an impl;
    /// - the value of a const or static item (of a module, or nested);
    /// - the body of a function: a default method of a trait, a method of an impl, a function item.
    ///
    /// An enum discriminant and an array length are read with or without the option: a relative
    /// path in a discriminant is an error, and a type in it or in a length is leaked, as ever.
    ///
    /// **A type in a type position is leaked.** The positions are the ones that the visitor
    /// replaces a type in: the target of a cast (`x as Rel`), the type of a `let` (`let x: Rel`),
    /// the parameters and the output of a closure, the generic arguments of a path and of a method
    /// call (`size_of::<Rel>()`, `Vec::<Rel>::new()`, `x.f::<Rel>()`), and the self type of a
    /// qualified path (`<Rel as ::core::default::Default>::default()`, not the trait). The type is
    /// analyzed as any type of the item is, with the generics of the item and of the method, and
    /// `Self`, in scope: the rules for `Self`, for lifetimes and for what a type can be written
    /// with are the same. It is an entry if its path is relative, a required one, and an optional
    /// one if only a path of another crate makes it so
    /// ([`allow_path()`](LeakerConfig::allow_path)), as a type of a signature is. So the
    /// macro emits the impl of the Repeater for it as for the others, and
    /// [`Referrer::required()`](crate::Referrer::required) lists it. A bare identifier argument
    /// (`size_of::<Rel>()`) is read as a type, as in an enum discriminant: if it is a const
    /// (`f::<N>()`), the copy fails, so write the const in braces with an absolute path. A type
    /// with `_` (`let v: Vec<_>`) is not leaked whole, and a relative path that holds one is
    /// rejected; the parts of it are leaked.
    ///
    /// **A path that is not a type is rejected** unless it means the same wherever it is written:
    /// it is absolute or starts with `$crate`, it is allowed ([`allow_path()`](LeakerConfig::allow_path),
    /// also the names of [`allow_primitive()`](LeakerConfig::allow_primitive) and
    /// [`allow_prelude()`](LeakerConfig::allow_prelude): `Some` and `None` are not among
    /// them, write `::core::option::Option::Some`), it starts with a generic or `Self` in scope,
    /// or with a name that the value itself declares (a `let` binding, a closure or function
    /// parameter, a `match`, `if let` or `for` pattern, an item or a `use` in a block). The name
    /// of the item that holds the value is relative like any other. (A name in a pattern is a
    /// binding unless it starts with a capital letter, as the lint `non_snake_case` expects: then
    /// it is a path to a constant, a unit struct or a variant.) Such are a function, a constant, a
    /// struct literal, a variant (`inner::f()`, `Rel::new()`, `Rel { .. }`) and the trait of a
    /// qualified path (`<u8 as Rel>::f()`). If [`LeakerResult::allow()`](crate::LeakerResult::allow)
    /// accepts such a trait, it is copied as written, and the types in it (`R1` in `<X as
    /// inner::Tr<R1>>::f()`) are replaced like elsewhere, which is equivalent.
    ///
    /// Three kinds of type are read, not leaked, and a relative path in them is rejected as
    /// above:
    ///
    /// - a type that names something the value declares: an item, a `use`, a generic or a lifetime
    ///   of an item in a block, a lifetime of a closure. The copy writes the declaration with the
    ///   value, which the Repeater cannot refer to. (Such a type with no relative path is fine:
    ///   `let x: Local` for a `struct Local` of the block.)
    /// - a type in the arguments of a macro (`::core::assert_eq!(size_of::<Rel>(), 4)`): the
    ///   visitor does not rewrite the tokens of a macro. The path of the macro and its arguments
    ///   are read as paths and expressions (below).
    /// - a type in an item that the value declares (a nested `struct`, `fn` or `impl`).
    ///
    /// Attributes (`#[allow(..)]`, `#[derive(..)]`) are not read, in a value or anywhere else in
    /// the item.
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
    /// # use syn::visit_mut::VisitMut;
    /// let item: ItemTrait = parse_quote! {
    ///     trait Tr { const N: usize = ::core::mem::size_of::<inner::Local>(); }
    /// };
    /// // Without the option the value is not read: the copy breaks without an error.
    /// let mut config = LeakerConfig::new();
    /// config.allow_primitive();
    /// let mut leaker = Leaker::from_config(config.clone());
    /// leaker.intern_trait(&item, true);
    /// assert!(leaker.result().finish().unwrap().is_empty());
    ///
    /// // With it the type is an entry, which the visitor replaces in the copy of the value.
    /// config.check_values();
    /// let mut leaker = Leaker::from_config(config.clone());
    /// let context = leaker.intern_trait(&item, true);
    /// let referrer = leaker.result().finish().unwrap();
    /// assert_eq!(referrer.required().count(), 1);
    /// let mut copy = item.clone();
    /// referrer
    ///     .visitor(context, |_, ix| parse_quote!(<() as Repeater<#ix>>::Type))
    ///     .visit_item_trait_mut(&mut copy);
    /// let expected: ItemTrait = parse_quote! {
    ///     trait Tr {
    ///         const N: usize = ::core::mem::size_of::<<() as Repeater<0usize>>::Type>();
    ///     }
    /// };
    /// assert_eq!(copy, expected);
    ///
    /// // A path that is not a type cannot be replaced.
    /// let item: ItemTrait = parse_quote!(trait Tr { const N: usize = inner::N; });
    /// let mut leaker = Leaker::from_config(config);
    /// leaker.intern_trait(&item, true);
    /// let result = leaker.result();
    /// let error = &result.errors()[0];
    /// assert_eq!(error.message, "use absolute path: 'inner :: N'");
    /// assert_eq!(
    ///     error.note(),
    ///     Some("it is in the default value of `N`, which the copy writes as it is")
    /// );
    /// ```
    pub fn check_values(&mut self) -> &mut Self {
        self.check_values = true;
        self
    }

    /// Accepts a type that names the `Self` of a struct, an enum or a union nested in the analyzed
    /// syntax ([`View::visit_item()`](crate::View::visit_item), the items of
    /// [`Leaker::intern_mod()`](crate::Leaker::intern_mod)) when a bound of the generics it uses
    /// names that `Self` too: `Rel2<Self, T>` in `struct S<T: AsRef<Self>>`. You promise the two
    /// things below.
    ///
    /// Without the option such an entry is an error. The copy declares a type of its own (`Self`
    /// is the copy's `S<T>`), so the bound `T: AsRef<Self>` is proved there for the copy's type,
    /// and the Repeater's impl for the Leaker's `S<T>` would need it for another (E0277). With the
    /// option the entry is for any type that the bound holds for
    /// ([`SelfRequirement::Any`](crate::SelfRequirement::Any)), and the type keeps `Self` as the
    /// parameter of the impl: `impl<T: AsRef<__S>, __S> Repeater<(T,), ..> for __S { type Type =
    /// Rel2<__S, T>; }`. The copy passes its own type as the implementor and writes the field as
    /// `<Self as Repeater<(T,), ..>>::Type`, which is `Rel2<S<T>, T>`, the type of the original.
    /// Only the entries that the option changes are the ones that are errors without it: a type
    /// that names `Self` and has no bound that does (`Rel2<Self, T>` in `struct S<T: Clone>`) is
    /// for the Leaker's type as before, which also accepts a leaked type with a bound on the
    /// parameter that receives `Self`.
    ///
    /// A default of a parameter is not changed: the copy cannot write `Self` there (E0735).
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// let item: Item = parse_quote! {
    ///     pub struct S<T: ::core::convert::AsRef<Self>> { pub b: Rel2<Self, T> }
    /// };
    /// let nested = |leaker: &mut Leaker| {
    ///     leaker.enter_mod("m", [parse_quote!(S)], |view| view.visit_item(&item));
    /// };
    /// // Without the option it is an error: the bound and the type name `Self`.
    /// let mut leaker = Leaker::new();
    /// nested(&mut leaker);
    /// assert_eq!(leaker.result().errors().len(), 1);
    ///
    /// let mut config = LeakerConfig::new();
    /// config.nested_self_entries();
    /// let mut leaker = Leaker::from_config(config);
    /// nested(&mut leaker);
    /// let referrer = leaker.result().finish().unwrap();
    /// // The entry is for any type that the bound holds for, with `Self` as its implementor.
    /// assert!(matches!(
    ///     referrer.context_self(referrer.context(0)),
    ///     Some(SelfRequirement::Any)
    /// ));
    /// let param: Ident = parse_quote!(__S);
    /// let r = referrer.repeater_impl(0, ImplSelf::Param(&param)).unwrap();
    /// let (generics, args, ty) = (&r.generics, &r.args, &r.ty);
    /// let imp: ItemImpl = parse_quote! {
    ///     impl #generics Repeater<#args, 7u64, 0usize> for __S { type Type = #ty; }
    /// };
    /// let expected: ItemImpl = parse_quote! {
    ///     impl<T: ::core::convert::AsRef<__S>, __S> Repeater<(T,), 7u64, 0usize> for __S {
    ///         type Type = Rel2<__S, T>;
    ///     }
    /// };
    /// assert_eq!(imp, expected);
    /// ```
    ///
    /// **Risks**
    ///
    /// - If the declaration of the leaked type has a trait bound on the parameter that receives
    ///   `Self` (`struct Rel2<A: Clone, B>`), the impl in the Leaker's crate fails (E0277: `__S:
    ///   Clone` is not satisfied), and no copy could satisfy it anyway: the copy is generated
    ///   before its impls are known, and the parameter is any type. The Leaker cannot see the
    ///   declaration of a relative path. A way out is an absolute path for the type (it is copied
    ///   as written, and its `Self` is the copy's own), or the analysis of the item with its own
    ///   `intern_*` call. Without the option such an entry is rejected, so this is a failure at
    ///   build time, in the Leaker's crate.
    /// - rustc before 1.70 (tested on 1.56 to 1.66) fails with E0275 (overflow evaluating the
    ///   requirement `S<T>: Sized`) when the copy's field is the last field of a struct: the type
    ///   needs the impl to be `Sized`, and the impl, whose `__S` stays `Sized` since `Rel2<__S,
    ///   T>` needs it, needs the type to be (see [`SelfRequirement::Any`](crate::SelfRequirement::Any)).
    ///   Put the field first, or add `where Self: Sized` to the item.
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
    /// A path matches if it starts with `path` and both have a leading `::` or neither:
    /// `parse_quote!(::serde)` allows `::serde::..` and not `serde::..`. Its generic arguments are
    /// still analyzed. You promise that an allowed crate is the same on both sides.
    ///
    /// Another absolute path, such as `::serde::Wrap<(Rel,)>`, may name a different crate in the
    /// copy, which the Leaker cannot tell. So it is leaked as an *optional entry*
    /// ([`Referrer::must()`](crate::Referrer::must) is `false`), and the Referrer chooses whether a
    /// copy replaces it ([`Referrer::allow_ambiguous_crate_name()`](crate::Referrer::allow_ambiguous_crate_name)).
    /// `::core::..` and `$crate::..` are never optional, and `::std` and `::alloc` are like any
    /// other crate unless allowed.
    /// [`Referrer::external_crates_require()`](crate::Referrer::external_crates_require) lists the
    /// crates an entry names.
    ///
    /// Not so for an absolute type that names the `Self` of a struct, an enum or a union nested in
    /// the analyzed syntax (`::serde::Wrap<Self>` in [`View::visit_item()`]): it is copied as
    /// written whatever the Referrer chooses, so that its `Self` is the copy's own type (see
    /// [`Visitor::with_nested_self()`](crate::Visitor::with_nested_self)).
    ///
    /// On a [`LeakerResult`](crate::LeakerResult) an absolute path changes nothing: the entries stay.
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
    /// let mut leaker = Leaker::new();
    /// leaker.intern_trait(&relaxed, true);
    /// assert!(leaker.result().finish().is_ok());
    /// let required: ItemTrait = parse_quote!(trait Tr { fn f(&self) where Self: Sized; });
    /// let mut leaker = Leaker::new();
    /// leaker.intern_trait(&required, true);
    /// assert_eq!(leaker.result().errors()[0].message, "use absolute path: 'Sized'");
    /// ```
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// let item: ItemTrait = parse_quote!(trait Tr: Clone { fn f(&self) -> Vec<Rel>; });
    /// let mut leaker = Leaker::new();
    /// leaker.intern_trait(&item, true);
    /// assert!(leaker.result().finish().is_err());
    ///
    /// let mut config = LeakerConfig::new();
    /// config.allow_prelude();
    /// let mut leaker = Leaker::from_config(config);
    /// leaker.intern_trait(&item, true);
    /// // `Clone` and `Vec` are left as written. The only entry is `Vec<Rel>`: a bare `Rel` may be
    /// // a const, so the type that holds it is leaked whole.
    /// assert_eq!(leaker.result().finish().unwrap().iter().count(), 1);
    /// ```
    pub fn allow_prelude(&mut self) -> &mut Self {
        self.allow.prelude();
        self
    }

    /// Allows the relative paths that start with `std`, `core` or `alloc`.
    ///
    /// A path with a leading `::` is not one of them, and is context-free in any case.
    pub fn allow_std_crates(&mut self) -> &mut Self {
        self.allow.std_crates();
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
    /// leaker.intern_struct(&item, true);
    /// assert_eq!(leaker.result().finish().unwrap().iter().count(), 1);
    /// // ...and left as written if the copy is in the same crate.
    /// let mut config = LeakerConfig::new();
    /// config.allow_crate();
    /// let mut leaker = Leaker::from_config(config);
    /// leaker.intern_struct(&item, true);
    /// assert!(leaker.result().finish().unwrap().is_empty());
    /// ```
    pub fn allow_crate(&mut self) -> &mut Self {
        self.allow.crate_root();
        self
    }

    /// Accepts the errors found in a value: a const or static value, the body of a function, a
    /// trait's default value or body. It has an effect with [`check_values()`](LeakerConfig::check_values)
    /// only.
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

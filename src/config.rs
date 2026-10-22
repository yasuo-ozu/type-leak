//! Configuration of the [`Leaker`](crate::Leaker).

use crate::error::Cause;
#[cfg(doc)]
use crate::error::{ErrorItem, Reason};
#[cfg(doc)]
use crate::referrer::{Allowed, Referrer};
#[cfg(doc)]
use crate::view::View;
use syn::*;

/// The paths that a [`Leaker`](crate::Leaker) leaves as written instead of leaking them, and the
/// conflicts that it accepts ([`LeakerConfig::allow()`]). Pass it to
/// [`Leaker::from_config()`](crate::Leaker::from_config).
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
    /// Set by [`LeakerConfig::nested_self_entries()`].
    pub(crate) nested_self_entries: bool,
    /// Set by [`LeakerConfig::allow()`].
    pub(crate) allowed_causes: Vec<Cause>,
}

impl LeakerConfig {
    /// Creates a configuration that allows no path and does not check values.
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
    /// ([`allowed_paths`](crate::Leaker::allowed_paths)), as a type of a signature is. So the
    /// macro emits the impl of the Repeater for it as for the others, and
    /// [`Referrer::required()`](crate::Referrer::required) lists it. A bare identifier argument
    /// (`size_of::<Rel>()`) is read as a type, as in an enum discriminant: if it is a const
    /// (`f::<N>()`), the copy fails, so write the const in braces with an absolute path. A type
    /// with `_` (`let v: Vec<_>`) is not leaked whole, and a relative path that holds one is
    /// rejected; the parts of it are leaked.
    ///
    /// **A path that is not a type is rejected** unless it means the same wherever it is written:
    /// it is absolute or starts with `$crate`, it is allowed ([`allowed_paths`](LeakerConfig::allowed_paths),
    /// also the names of [`allow_primitive()`](LeakerConfig::allow_primitive) and
    /// [`allow_std_prelude()`](LeakerConfig::allow_std_prelude): `Some` and `None` are not among
    /// them, write `::core::option::Option::Some`), it starts with a generic or `Self` in scope,
    /// or with a name that the value itself declares (a `let` binding, a closure or function
    /// parameter, a `match`, `if let` or `for` pattern, an item or a `use` in a block). The name
    /// of the item that holds the value is relative like any other. (A name in a pattern is a
    /// binding unless it starts with a capital letter, as the lint `non_snake_case` expects: then
    /// it is a path to a constant, a unit struct or a variant.) Such are a function, a constant, a
    /// struct literal, a variant (`inner::f()`, `Rel::new()`, `Rel { .. }`) and the trait of a
    /// qualified path (`<u8 as Rel>::f()`).
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
    /// leaker.intern_trait(&item, true).unwrap();
    /// assert!(leaker.finish().is_empty());
    ///
    /// // With it the type is an entry, which the visitor replaces in the copy of the value.
    /// config.check_values();
    /// let mut leaker = Leaker::from_config(config.clone());
    /// let context = leaker.intern_trait(&item, true).unwrap();
    /// let referrer = leaker.finish();
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
    /// let err = Leaker::from_config(config).intern_trait(&item, true).unwrap_err();
    /// let error = &err[0];
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
    ///     leaker.enter_mod("m", [parse_quote!(S)], |view| view.visit_item(&item)).map(|(id, ())| id)
    /// };
    /// // Without the option it is an error: the bound and the type name `Self`.
    /// assert!(nested(&mut Leaker::new()).is_err());
    ///
    /// let mut config = LeakerConfig::new();
    /// config.nested_self_entries();
    /// let mut leaker = Leaker::from_config(config);
    /// nested(&mut leaker).unwrap();
    /// let referrer = leaker.finish();
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

    /// Accepts a conflict: the analysis does not report the error whose [`cause`](ErrorItem::cause)
    /// equals `cause` (the same kind and the same tokens, whatever the spans). The part that
    /// conflicts is copied as written, whole, as the error says it must be, and nothing in it is
    /// leaked: a relative path in it stays as it is. You promise that it can be written as it is in
    /// the copy; if it cannot, the copy fails to build, in the crate of the copy.
    ///
    /// The cause is looked up only where a conflict is turned into an error. A place that is no
    /// conflict never looks at it, so the option changes nothing else. It is not a second
    /// [`allowed_paths`](LeakerConfig::allowed_paths): that makes a name context-free everywhere,
    /// and this accepts the copy at the conflicts the name causes. Every conflict with the same
    /// cause is accepted. If no error is left, the analysis succeeds: a scope that held only
    /// accepted conflicts is not rolled back, and its other entries stay.
    ///
    /// The [`Referrer`] keeps the causes that accepted a conflict and the parts that were copied as
    /// written ([`Referrer::allowed()`], [`Allowed`]), and its [`Visitor`](crate::Visitor) leaves
    /// each part as written. It would find an entry inside the part otherwise, by its tokens and
    /// its context, as it does everywhere: the `dyn RelTr` of `f` below is the entry of `g`.
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// # use syn::visit_mut::VisitMut;
    /// let item: ItemImpl = parse_quote! {
    ///     impl ::p::S {
    ///         fn f(_: ::p::W<'_, dyn RelTr>) {}  // the lifetime is elided, and `RelTr` is relative
    ///         fn g(_: ::p::X<dyn RelTr>) {}      // fine: `dyn RelTr` is leaked
    ///     }
    /// };
    /// // The entries of an item, with whether each is required.
    /// let entries = |config: LeakerConfig, item: &ItemImpl| {
    ///     let mut leaker = Leaker::from_config(config);
    ///     leaker.intern_impl(item, true).map(|_| {
    ///         let referrer = leaker.finish();
    ///         let entries: Vec<_> = referrer.iter().cloned().enumerate().collect();
    ///         entries.into_iter().map(|(ix, ty)| (ty, referrer.must(ix))).collect::<Vec<_>>()
    ///     })
    /// };
    ///
    /// // The error is about the trait, which is smaller than the type `W<'_, dyn RelTr>` it is in.
    /// let errors = Leaker::new().intern_impl(&item, true).unwrap_err();
    /// assert_eq!(errors.len(), 1);
    /// assert_eq!(errors[0].cause, Cause::Trait(parse_quote!(RelTr)));
    ///
    /// // Accept it: `f` is copied as written, and the entries are those of `g` alone, which still
    /// // leaks `dyn RelTr` (required).
    /// let mut config = LeakerConfig::new();
    /// config.allow(errors[0].cause.clone());
    /// let accepted = entries(config, &item).unwrap();
    /// let g_only: ItemImpl = parse_quote!(impl ::p::S { fn g(_: ::p::X<dyn RelTr>) {} });
    /// assert_eq!(accepted, entries(LeakerConfig::new(), &g_only).unwrap());
    /// assert!(accepted.contains(&(parse_quote!(dyn RelTr), true)));
    ///
    /// // `allowed_paths` makes `RelTr` context-free in `g` too, which loses its entry.
    /// let mut config = LeakerConfig::new();
    /// config.allowed_paths.push(parse_quote!(RelTr));
    /// let entries = entries(config, &item).unwrap();
    /// assert!(!entries.iter().any(|(ty, _)| *ty == parse_quote!(dyn RelTr)));
    ///
    /// // The Referrer lists the cause that accepted the conflict, and the part of `f` that is copied
    /// // as written. Its visitor leaves `f` as it is and replaces the `dyn RelTr` of `g`.
    /// let mut config = LeakerConfig::new();
    /// config.allow(errors[0].cause.clone());
    /// let mut leaker = Leaker::from_config(config);
    /// let context = leaker.intern_impl(&item, true).unwrap();
    /// let referrer = leaker.finish();
    /// assert_eq!(referrer.allowed().len(), 2);
    /// assert_eq!(referrer.allowed()[0], Allowed::Cause(Cause::Trait(parse_quote!(RelTr))));
    /// assert!(matches!(
    ///     &referrer.allowed()[1],
    ///     Allowed::Part { part: Cause::Type(ty), .. } if *ty == parse_quote!(dyn RelTr)
    /// ));
    /// let mut copy = item.clone();
    /// referrer
    ///     .visitor(context, |_, ix| parse_quote!(<S as Repeater<(), 7, #ix>>::Type))
    ///     .visit_item_impl_mut(&mut copy);
    /// let expected: ItemImpl = parse_quote! {
    ///     impl ::p::S {
    ///         fn f(_: ::p::W<'_, dyn RelTr>) {}
    ///         fn g(_: ::p::X<<S as Repeater<(), 7, 0usize>>::Type>) {}
    ///     }
    /// };
    /// assert_eq!(copy, expected);
    ///
    /// // A cause of another kind, or with other tokens, accepts nothing.
    /// let mut config = LeakerConfig::new();
    /// config.allow(Cause::Type(parse_quote!(RelTr)));
    /// config.allow(Cause::Trait(parse_quote!(OtherTr)));
    /// assert!(Leaker::from_config(config).intern_impl(&item, true).is_err());
    /// ```
    ///
    /// The cause may be smaller than the part that is copied. In `[u8; { let c = |a: u8| a; 3 }]`
    /// the closure stops the array from being leaked, the error is about `u8` (`a` is a note of it),
    /// and accepting `u8` copies the array whole: `a` is never reached.
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// let item: ItemImpl = parse_quote! {
    ///     impl ::p::S {
    ///         fn f(_: [u8; { let c = |a: u8| a; 3 }]) {}
    ///         fn g(_: ::p::V<u8>) {}
    ///     }
    /// };
    /// let errors = Leaker::new().intern_impl(&item, true).unwrap_err();
    /// assert_eq!(errors.len(), 1);
    /// assert_eq!(errors[0].cause, Cause::Type(parse_quote!(u8)));
    ///
    /// let mut config = LeakerConfig::new();
    /// config.allow(Cause::Type(parse_quote!(u8)));
    /// let mut leaker = Leaker::from_config(config);
    /// leaker.intern_impl(&item, true).unwrap();
    /// // `g` leaks `::p::V<u8>` as it does without the array, and nothing is leaked for `a`.
    /// let referrer = leaker.finish();
    /// let required: Vec<_> = referrer.required().map(|(_, ty)| ty).collect();
    /// assert_eq!(required, [&parse_quote!(::p::V<u8>)]);
    /// ```
    ///
    /// The mentions of `Self` that have no meaning ([`Reason::SelfWithoutMeaning`]) are reported
    /// only if no other error is, so accepting the others makes them appear. The outermost type
    /// that would be an entry and names one, or has a bound in its context that does, is copied as
    /// written too, whole: the impl of the Repeater would read that `Self` as its implementor. So
    /// is a type that leaves a lifetime of its method unconstrained
    /// ([`Reason::UnconstrainedLifetime`]). [`View::check()`] reports causes, not errors, and
    /// ignores the option.
    ///
    /// **What accepting does not make compile**
    ///
    /// - `Self::A`, or `Self` as a type, in a bound or where predicate when a bound in its context
    ///   names a leaked type ([`Reason::UnprovableBound`]). Only the cause is copied as written:
    ///   the bound stays leaked, so the copy still fails with E0277 until the type of the bound is
    ///   written with an absolute path or allowed with
    ///   [`allowed_paths`](LeakerConfig::allowed_paths). It is the same in an impl (`impl<T:
    ///   Into<R0>> Tr<T> for S1` with `where Y: Into<Self::A1>` in a method, or `impl<T: Into<R0>>
    ///   S1<T>` with `where Y: Fn(&Self) -> u8` and `allow_self = false`). A predicate on a type
    ///   that is no path of `Self` (`X: Into<R1<T>>`) is fine: accepted, it is copied as written,
    ///   and no longer goes through the Repeater.
    ///
    ///   ```
    ///   # use type_leak::*;
    ///   # use syn::*;
    ///   let item: ItemTrait = parse_quote! {
    ///       trait Tr<T> where T: Into<R0> {
    ///           type A0;
    ///           fn m<X>(&self) -> X where X: Into<Self::A0>;
    ///       }
    ///   };
    ///   let mut config = LeakerConfig::new();
    ///   config.allow_std_prelude();
    ///   let errors = Leaker::from_config(config.clone()).intern_trait(&item, true).unwrap_err();
    ///   assert_eq!(errors[0].reason, Reason::UnprovableBound);
    ///   assert_eq!(errors[0].cause, Cause::Type(parse_quote!(Self::A0)));
    ///
    ///   // Accepted, `Self::A0` is copied as written, and `R0` in the bound is still an entry.
    ///   config.allow(errors[0].cause.clone());
    ///   let mut leaker = Leaker::from_config(config);
    ///   leaker.intern_trait(&item, true).unwrap();
    ///   let referrer = leaker.finish();
    ///   let required: Vec<_> = referrer.required().map(|(_, ty)| ty).collect();
    ///   assert_eq!(required, [&parse_quote!(R0)]);
    ///   ```
    /// - A trait with a generic whose bound names an associated type that another bound fixes
    ///   (`T: Iterator<Item = (R0, u8)> + Fn(<T as Iterator>::Item) -> u8`, also through an alias
    ///   of the projection), and a method predicate that names `Self::A`: rustc cannot implement it
    ///   with a generic impl at all, even with nothing leaked (E0277 "expected an `Fn((R0, u8))`
    ///   closure, found `T`" in the impl), so no copy compiles, whatever is accepted. See [Bounds
    ///   that name leaked types](crate::Leaker#bounds-that-name-leaked-types).
    /// - [`Visitor::with_self()`](crate::Visitor::with_self) replaces `Self` inside an accepted part
    ///   too, which can change the elision of a lifetime. With `allow_self = false` and `Self`
    ///   accepted, `struct S<'a>(PhantomData<&'a ()>, fn(::core::slice::Iter<u8>, *mut Self) ->
    ///   &u32)` is copied with `Self` replaced by `S<'a>`: the function pointer then has two input
    ///   lifetimes, and the elided output fails (E0106).
    pub fn allow(&mut self, cause: Cause) -> &mut Self {
        self.allowed_causes.push(cause);
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
    /// assert_eq!(err[0].message, "use absolute path: 'Sized'");
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

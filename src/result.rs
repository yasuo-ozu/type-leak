//! What a [`Leaker`] found, the [`LeakerResult`], and what it may write as it is:
//! the [`Allow`] trait, the [`Rules`] that build one, and the filter that applies it to errors.

use crate::encode::EncodeError;
#[cfg(doc)]
use crate::error::Reason;
use crate::error::{Cause, ErrorItem};
#[cfg(doc)]
use crate::leaker::Leaker;
#[cfg(doc)]
use crate::referrer::EntryId;
use crate::referrer::Referrer;
#[cfg(doc)]
use crate::repeater::ImplSelf;
use crate::repeater::{Projection, RepeaterImpl};
use crate::util::{is_std_crate_path, self_head, starts_with_known, Known};
use proc_macro2::{Span, TokenStream};
use std::collections::HashSet;
use syn::parse::{Parse, ParseStream};
use syn::*;
use template_quote::{quote, ToTokens};

/// Where in an item a path or an error is: [`Allow::allow()`] is asked at a place, and
/// [`ErrorItem::place`] is the place of an error. See the example there.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct Place {
    /// In a value or a body that [`Leaker::check_values`](crate::Leaker::check_values) or
    /// [`Leaker::check_bodies`](crate::Leaker::check_bodies) reads: the value of a const or a
    /// static, the body of a function, a default value or body of a trait. Not an enum
    /// discriminant.
    pub in_value: bool,
    /// In a default: a default value or body of a trait, or the default of a generic parameter
    /// (`U = Rel<'_>`).
    pub in_default: bool,
}

/// A rule that says which paths the copy may write as written. Give it to
/// [`Leaker::with_rule()`](crate::Leaker::with_rule) or to
/// [`LeakerResult::allow()`](crate::LeakerResult::allow).
///
/// `allow(cause, place)` returns `true` if the copy may write `cause` as written at `place`. You
/// promise that the path means the same in the copy: for an absolute path, that the crate is the
/// same in crate B.
///
/// The Leaker asks the rule about each relative path, and about each absolute path in a type
/// (except `::core::..` and `$crate::..`). It asks with the path alone, without its generic
/// arguments (they are analyzed on their own), as a [`Cause`] of its kind:
///
/// - [`Cause::Type`]: a path in a type.
/// - [`Cause::Trait`]: the trait of a bound or of a qualified path.
/// - [`Cause::Expr`]: a path in a value.
/// - [`Cause::Macro`]: the path of a macro.
/// - [`Cause::Unknown`]: a path of unknown kind (the root of a `use`, a path in the tokens of a
///   macro, a path that an attribute names).
///
/// It does not ask about a path that starts with `Self`, a generic in scope, or a name that a
/// block declares. It does not leak an allowed path. It calls the rule for every check of a path,
/// so keep the rule cheap. If the rule panics, the call to the Leaker is rolled back and panics.
///
/// [`LeakerResult::allow()`](crate::LeakerResult::allow) asks the rule at the
/// [`place`](ErrorItem::place) of each error. It drops an error with [`paths`](ErrorItem::paths)
/// if the rule allows each path (never one that starts with `Self`); else the error stays without
/// the allowed paths. It drops an error without paths if the rule allows its cause, as written,
/// with its generic arguments, so a rule on a whole cause (`Wrap<Rel>`) never drops an error with
/// paths. The Leaker drops errors in the same way when its outermost call returns. The error of
/// `Self` without a meaning ([`Reason::SelfWithoutMeaning`]) has no paths, so a rule that allows
/// its cause (`Self`, `Self::Item`) drops it in both places.
///
/// `()` allows nothing, a closure `Fn(&Cause, Place) -> bool` is a rule, and `(A, B)` allows what
/// either allows. [`Rules`] builds the common rules. A closure needs the type of its first
/// parameter (`|cause: &Cause, _| ..`) wherever you give it (E0282, E0631);
/// [`Rules::or_if()`] needs none.
///
/// ```
/// # use type_leak::*;
/// # use syn::*;
/// // You promise that `::serde` is the same crate in crate B.
/// let rule = Rules::new().or_if(|cause, _| cause.starts_with(&parse_quote!(::serde)));
/// let mut leaker = Leaker::with_rule(rule);
/// leaker.intern_struct(&parse_quote!(struct S { a: ::serde::Value, b: MyType }));
/// // Only `MyType` is leaked.
/// assert_eq!(leaker.result().finish().unwrap().iter().count(), 1);
/// ```
pub trait Allow {
    /// Returns `true` if the copy may write `cause` as written at `place`.
    fn allow(&self, cause: &Cause, place: Place) -> bool;
}

/// Allows nothing.
impl Allow for () {
    fn allow(&self, _: &Cause, _: Place) -> bool {
        false
    }
}

impl<F: Fn(&Cause, Place) -> bool> Allow for F {
    fn allow(&self, cause: &Cause, place: Place) -> bool {
        self(cause, place)
    }
}

/// Allows what either allows.
impl<A: Allow, B: Allow> Allow for (A, B) {
    fn allow(&self, cause: &Cause, place: Place) -> bool {
        self.0.allow(cause, place) || self.1.allow(cause, place)
    }
}

/// The built-in rules and your own, as one [`Allow`]: it allows what any of them allows.
///
/// Each method sets a flag and returns the same type, so a choice at run time keeps one type.
/// [`or()`] and [`or_if()`] add a rule of your own and change `A`. The built-in path rules allow
/// only a cause that is a path without generic arguments ([`Cause::path()`]), as the Leaker asks.
///
/// ```
/// # use type_leak::*;
/// # use syn::*;
/// # let flag = true;
/// let mut rules = Rules::new().primitive();
/// if flag {
///     rules = rules.prelude(); // the same type
/// }
/// let (u8_ty, vec) = (Cause::Type(parse_quote!(u8)), Cause::Type(parse_quote!(Vec)));
/// assert!(rules.allow(&u8_ty, Place::default()) && rules.allow(&vec, Place::default()));
/// ```
///
/// [`or()`]: Rules::or
/// [`or_if()`]: Rules::or_if
#[derive(Clone, Debug, Default)]
#[must_use]
pub struct Rules<A = ()> {
    primitive: bool,
    prelude: bool,
    std_crates: bool,
    external_crates: bool,
    crate_paths: bool,
    values: bool,
    defaults: bool,
    custom: A,
}

impl Rules {
    /// Creates rules that allow nothing.
    pub fn new() -> Self {
        Self::default()
    }
}

impl<A> Rules<A> {
    /// Allows the paths that start with a primitive type (`bool`, `char`, `str`, `u8` to `u128`,
    /// `usize`, `i8` to `i128`, `isize`, `f32`, `f64`). You promise that the copy does not shadow
    /// them; [`Leaker::assertions()`] checks crate A.
    pub fn primitive(mut self) -> Self {
        self.primitive = true;
        self
    }

    /// Allows the paths that start with a name of the prelude of every edition, as of Rust 1.56:
    /// `Box`, `Option`, `Some`, `None`, `Result`, `Ok`, `Err`, `String`, `Vec`, `drop`, `Copy`,
    /// `Send`, `Sized`, `Sync`, `Unpin`, `Drop`, `Fn`, `FnMut`, `FnOnce`, `Clone`, `PartialEq`,
    /// `PartialOrd`, `Eq`, `Ord`, `AsRef`, `AsMut`, `Into`, `From`, `Default`, `Iterator`,
    /// `Extend`, `IntoIterator`, `DoubleEndedIterator`, `ExactSizeIterator`, `ToOwned`, `ToString`.
    /// You promise that the copy does not shadow them.
    ///
    /// The names of later preludes are not allowed, as the copy may use an older edition or Rust:
    /// `TryFrom`, `TryInto`, `FromIterator`, `Future`, `IntoFuture`, `size_of`, `size_of_val`,
    /// `align_of`, `align_of_val`. The help of their errors gives their absolute paths.
    ///
    /// A derive is a name like any other: this allows `#[derive(Clone, Copy, Default, Eq, Ord,
    /// PartialEq, PartialOrd)]`, but not `Debug` or `Hash`, whose traits are not in the prelude.
    /// Without this rule, `Clone` in `trait Tr: Clone` and `Sized` in `Self: Sized` are errors (a
    /// bare `?Sized` needs no rule).
    pub fn prelude(mut self) -> Self {
        self.prelude = true;
        self
    }

    /// Allows the paths that start with `std`, `core` or `alloc`, relative (`std::vec::Vec`) or
    /// absolute (`::std::vec::Vec`). In the Leaker, they are then no entries, and the copy writes
    /// them as written.
    ///
    /// You promise that the copy does not shadow the relative names. An absolute `::core` path
    /// needs no rule: it is never an entry. [`external_crates()`](Rules::external_crates) allows
    /// every absolute path.
    pub fn std_crates(mut self) -> Self {
        self.std_crates = true;
        self
    }

    /// Allows every absolute path (`::foo::X`), `::std` included. In the Leaker, a type of another
    /// crate is then no entry, and the copy writes it as written. You promise that each crate is
    /// the same crate in crate B.
    pub fn external_crates(mut self) -> Self {
        self.external_crates = true;
        self
    }

    /// Allows the paths that start with `crate`.
    ///
    /// Use it only if the copy is in the same crate as the item: in another crate, `crate::Foo`
    /// names another type.
    pub fn crate_paths(mut self) -> Self {
        self.crate_paths = true;
        self
    }

    /// Allows everything in a value or a body ([`Place::in_value`]): the copy writes every relative
    /// path there as written, the type of a `let` too, instead of leaking it. It changes nothing
    /// unless [`Leaker::check_values`](crate::Leaker::check_values) or
    /// [`Leaker::check_bodies`](crate::Leaker::check_bodies) is on.
    pub fn values(mut self) -> Self {
        self.values = true;
        self
    }

    /// Allows everything in a default ([`Place::in_default`]): the copy writes every relative path
    /// in a default as written (`U = Rel<'_>`, `T = Wrap<Self>`) instead of leaking it.
    pub fn defaults(mut self) -> Self {
        self.defaults = true;
        self
    }

    /// Adds `rule`: the rules then allow what it allows too. For a closure, use
    /// [`or_if()`](Rules::or_if), which needs no type annotations.
    pub fn or<B: Allow>(self, rule: B) -> Rules<(A, B)> {
        Rules {
            primitive: self.primitive,
            prelude: self.prelude,
            std_crates: self.std_crates,
            external_crates: self.external_crates,
            crate_paths: self.crate_paths,
            values: self.values,
            defaults: self.defaults,
            custom: (self.custom, rule),
        }
    }

    /// Adds the closure `rule`, like [`or()`](Rules::or).
    pub fn or_if<F: Fn(&Cause, Place) -> bool>(self, rule: F) -> Rules<(A, F)> {
        self.or(rule)
    }

    /// Returns `true` if a built-in path rule allows the bare `path`.
    fn allows_path(&self, path: &Path) -> bool {
        path.segments.iter().all(|seg| seg.arguments.is_none())
            && ((self.primitive && starts_with_known(path, Known::Primitive))
                || (self.prelude && starts_with_known(path, Known::Prelude))
                || (self.std_crates && is_std_crate_path(path))
                || (self.external_crates && path.leading_colon.is_some())
                || (self.crate_paths && is_crate_path(path)))
    }
}

impl<A: Allow> Allow for Rules<A> {
    fn allow(&self, cause: &Cause, place: Place) -> bool {
        (self.values && place.in_value)
            || (self.defaults && place.in_default)
            || cause.path().map_or(false, |path| self.allows_path(path))
            || self.custom.allow(cause, place)
    }
}

/// Returns `true` if `path` is relative and starts with `crate`.
fn is_crate_path(path: &Path) -> bool {
    path.leading_colon.is_none()
        && path
            .segments
            .first()
            .map_or(false, |seg| seg.ident == "crate")
}

/// Returns `true` if `rule` allows `path`, one of [`ErrorItem::paths`], at `place`. A path that
/// starts with `Self` is not allowed, and the rule is not asked about it. (The error of a `Self`
/// without meaning has no paths: `accepts()` asks its cause.)
fn allows_listed_path<A: Allow + ?Sized>(rule: &A, path: &Cause, place: Place) -> bool {
    path.path().map_or(false, |p| self_head(p).is_none()) && rule.allow(path, place)
}

/// Returns `true` if `rule` accepts `error`: if it has relative paths ([`ErrorItem::paths`]), it
/// allows each; else it allows the cause. If only some paths are allowed, the error stays without
/// them ([`filter()`]).
fn accepts<A: Allow + ?Sized>(rule: &A, error: &ErrorItem) -> bool {
    let place = error.place;
    if error.paths.is_empty() {
        rule.allow(&error.cause, place)
    } else {
        error
            .paths
            .iter()
            .all(|path| allows_listed_path(rule, path, place))
    }
}

/// Drops the errors that `rule` accepts. An error that stays loses the paths that the rule allows,
/// so that rules given one after the other add up as they do in one set (also in the serialized
/// error, which carries its paths), and the errors inside its part that are dropped.
pub(crate) fn filter<A: Allow + ?Sized>(rule: &A, errors: &mut Vec<ErrorItem>) {
    errors.retain(|error| !accepts(rule, error));
    for error in errors.iter_mut() {
        let place = error.place;
        error
            .paths
            .retain(|path| !allows_listed_path(rule, path, place));
    }
    forget_dropped(errors);
}

/// Makes each error hold only the errors inside its part that are in `errors`: after errors were
/// dropped, the others do not hold them.
fn forget_dropped(errors: &mut [ErrorItem]) {
    let present: HashSet<usize> = errors.iter().filter_map(|error| error.link.id).collect();
    for error in errors {
        error.link.inner.retain(|id| present.contains(id));
    }
}

/// What a [`Leaker`] found: the [`Referrer`] and the errors that remain.
///
/// [`finish()`](LeakerResult::finish) returns the Referrer only when no error remains. First
/// accept the errors that you can, and report the others ([`ErrorItem::emit()`]):
///
/// - [`allow()`](LeakerResult::allow) drops the errors that an [`Allow`] rule accepts.
/// - [`ignore()`](LeakerResult::ignore) drops one error, and the copy writes its whole part as
///   written. [`ignore_if()`](LeakerResult::ignore_if) does this for each error that a closure
///   chooses.
///
/// [`repeater_impls()`](LeakerResult::repeater_impls) returns the Repeater impls for crate A.
///
/// ```
/// # use {syn::*, type_leak::*};
/// let mut leaker = Leaker::new();
/// leaker.intern_struct(&parse_quote!(struct S(::p::W<'_, dyn RelTr>);));
/// let result = leaker.result();
/// assert_eq!(result.errors()[0].cause, Cause::Trait(parse_quote!(RelTr)));
/// // Allowing the error copies the part as written, and `finish()` is `Ok`.
/// let rel_tr = Cause::Trait(parse_quote!(RelTr));
/// let result = result.allow(|cause: &Cause, _: Place| *cause == rel_tr);
/// assert!(result.finish().is_ok());
/// ```
#[derive(Clone, Debug)]
pub struct LeakerResult {
    /// The Referrer of the analysis.
    pub(crate) referrer: Referrer,
    /// The errors that remain.
    pub(crate) errors: Vec<ErrorItem>,
}

impl LeakerResult {
    /// Drops the errors that `rule` accepts, and keeps the others in order. The copy writes the
    /// part of each dropped error as written.
    ///
    /// The rule accepts an error by its [`paths`](ErrorItem::paths), or by its cause if it has
    /// none: see [`Allow`]. Compare a cause with `==`, which compares the kind and the tokens, not
    /// the spans. A closure needs the type of its first parameter, also when written in the call.
    /// You promise that the part compiles as written in the copy.
    ///
    /// - The part may be larger than the cause: in `[u8; { let c = |a: u8| a; 3 }]`, the error is
    ///   about `u8`, and the copy writes the whole array as written.
    /// - Allowing some errors does not make the copy compile ([`Reason::UnprovableBound`]).
    /// - It changes no entry. An absolute path is never an error, so a rule for `::serde` here
    ///   changes nothing. To have the copy write a type of another crate as written, give
    ///   [`Rules::external_crates()`](crate::Rules::external_crates) to [`Leaker::with_rule()`].
    /// - A rule here only approximates the same rule given to the Leaker: more errors may remain,
    ///   and the copy may fail to compile. Give rules to the Leaker when you can.
    ///
    /// To have the copy write the whole part as written, use [`ignore()`](LeakerResult::ignore).
    pub fn allow(mut self, rule: impl Allow) -> Self {
        filter(&rule, &mut self.errors);
        self
    }

    /// Accepts `error`, one of [`errors()`](LeakerResult::errors): the copy writes the whole part
    /// of the error as written, and finds no entry in it.
    ///
    /// The errors about the same part go too, and so do the errors found inside the part. For
    /// `MyType<'_, Rel>`, [`allow()`](LeakerResult::allow) has the copy write `MyType` as written
    /// and replace the entry `Rel`; `ignore()` has it write all of `MyType<'_, Rel>` as written.
    /// An entry that is also used at another place is still replaced there. You promise that the
    /// part compiles as written.
    ///
    /// For the error of an entry ([`Reason::NestedSelfInBounds`]), it removes the entry instead.
    /// The entry gets no impl ([`repeater_impls()`](LeakerResult::repeater_impls)),
    /// [`Referrer::iter()`] skips it, and the copy writes its type as written. The other entries
    /// keep their [`EntryId`]s, so you can call it before or after you write the impls.
    ///
    /// - An error about no part (`Self` without a meaning, a path in a value) is only dropped.
    /// - An error about an entry inside the part stays: the entry may be used elsewhere, so ignore
    ///   that error on its own.
    /// - If `error` is not one of the errors, nothing changes.
    /// - Call it before [`allow()`](LeakerResult::allow) for the same error.
    ///
    /// ```
    /// # use {syn::*, type_leak::*};
    /// # use syn::visit_mut::VisitMut;
    /// let item: ItemStruct = parse_quote!(struct S<'a>(MyType<'_, Inner<'_>, Rel>, Rel, &'a u8););
    /// let mut leaker = Leaker::new();
    /// let top_level = leaker.intern_struct(&item);
    /// let result = leaker.result();
    /// // `MyType` and, inside it, `Inner<'_>`.
    /// assert_eq!(result.errors().len(), 2);
    ///
    /// let error = result.errors()[0].clone();
    /// let result = result.ignore(&error);
    /// assert!(result.errors().is_empty());
    /// let referrer = result.finish().unwrap();
    ///
    /// let projection = Projection::new(
    ///     parse_quote!(Repeater),
    ///     7,
    ///     ImplSelf::Param(parse_quote!(__S)),
    /// );
    /// let mut copy = item.clone();
    /// referrer.visitor(top_level, &projection).visit_item_struct_mut(&mut copy);
    /// // `Rel` is replaced in the second field only.
    /// let (e0, e1) = (projection.key(parse_quote!(0)), projection.key(parse_quote!(1)));
    /// assert_eq!(copy, parse_quote! {
    ///     struct S<'a>(
    ///         MyType<'_, Inner<'_>, Rel>,
    ///         <Self as Repeater<(), #e0>>::Type,
    ///         <Self as Repeater<(&'a (),), #e1>>::Type
    ///     );
    /// });
    /// ```
    pub fn ignore(mut self, error: &ErrorItem) -> Self {
        if let Some(at) = self.errors.iter().position(|known| known == error) {
            self.ignore_at(at);
        }
        self
    }

    /// Calls `f` for each error that remains, in the order found, and ignores each error for which
    /// it returns `true`, as [`ignore()`](LeakerResult::ignore) does.
    ///
    /// `f` is not called for the errors that go with an ignored one (about the same part or entry,
    /// or inside its part).
    ///
    /// ```
    /// # use {syn::*, type_leak::*};
    /// let item: Item = parse_quote!(
    ///     struct N<T: ::core::convert::AsRef<Self>>(Rel2<Self, T>, Wrap<T>);
    /// );
    /// let mut leaker = Leaker::new();
    /// leaker.enter_generics(&Generics::default(), |view| view.visit_item(&item));
    /// let result = leaker.result();
    /// assert_eq!(result.errors()[0].reason, Reason::NestedSelfInBounds);
    /// assert!(result.clone().finish().is_err());
    ///
    /// // The entry `Rel2<Self, T>` is removed, and the error with it.
    /// let result = result.ignore_if(|error| error.reason == Reason::NestedSelfInBounds);
    /// assert!(result.errors().is_empty());
    /// assert_eq!(result.finish().unwrap().iter().count(), 1);
    /// ```
    pub fn ignore_if<F: FnMut(&ErrorItem) -> bool>(mut self, mut f: F) -> Self {
        // `asked[ix]`: `errors[ix]` was asked about. The errors that remain keep their order.
        let mut asked = vec![false; self.errors.len()];
        while let Some(at) = asked.iter().position(|asked| !asked) {
            asked[at] = true;
            if f(&self.errors[at]) {
                let mut gone = self.ignore_at(at).into_iter();
                asked.retain(|_| !gone.next().unwrap_or(false));
            }
        }
        self
    }

    /// Ignores `errors[at]` and what goes with it. Returns whether each error, as they were
    /// before, was dropped.
    fn ignore_at(&mut self, at: usize) -> Vec<bool> {
        let (part, entry) = (self.errors[at].link.part, self.errors[at].link.entry);
        // The errors about the same part, or entry, go together.
        let same = |known: &ErrorItem| {
            (part.is_some() && known.link.part == part)
                || (entry.is_some() && known.link.entry == entry)
        };
        let mut inside = HashSet::new();
        for (ix, known) in self.errors.iter().enumerate() {
            if ix == at || same(known) {
                if let Some(part) = known.link.part {
                    self.referrer.make_whole(part);
                }
                if let Some(entry) = known.link.entry {
                    self.referrer.remove(entry);
                }
                inside.extend(known.link.inner.iter().copied());
            }
        }
        // An error about an entry inside the part stays: the entry may be used elsewhere.
        let gone: Vec<bool> = self
            .errors
            .iter()
            .enumerate()
            .map(|(ix, known)| {
                let revealed = known.link.entry.is_none()
                    && known.link.id.map_or(false, |id| inside.contains(&id));
                ix == at || same(known) || revealed
            })
            .collect();
        let mut dropped = gone.iter();
        self.errors.retain(|_| dropped.next() != Some(&true));
        forget_dropped(&mut self.errors);
        gone
    }

    /// Returns the errors that remain, in the order found.
    pub fn errors(&self) -> &[ErrorItem] {
        &self.errors
    }

    /// Returns the Referrer if no error remains, else the errors that remain, in the order found.
    ///
    /// It is the only way to get the Referrer: first drop the errors that you accept with
    /// [`allow()`](LeakerResult::allow) or [`ignore()`](LeakerResult::ignore), and report the
    /// others. In a proc macro, `?` turns the errors into one [`syn::Error`] that holds them all.
    ///
    /// ```
    /// # use {syn::*, type_leak::*};
    /// let mut leaker = Leaker::new();
    /// leaker.intern_struct(&parse_quote!(struct S(MyType<'_>);));
    /// let result = leaker.result();
    /// let expected = result.errors().to_vec();
    ///
    /// let errors = result.finish().unwrap_err();
    /// assert_eq!(errors.into_vec(), expected);
    /// ```
    pub fn finish(self) -> std::result::Result<Referrer, LeakerResultError> {
        if self.errors.is_empty() {
            Ok(self.referrer)
        } else {
            Err(LeakerResultError(self.errors))
        }
    }

    /// Returns the Repeater impl of each entry, for crate A.
    ///
    /// Write one impl for every entry, next to the analyzed items. The impls come in the order of
    /// [`Referrer::iter()`], so without the entries that [`ignore()`](LeakerResult::ignore)
    /// removed. Each has its [`EntryId`] in [`RepeaterImpl::entry`].
    ///
    /// Give the same `projection` that crate B gives to [`Projection::of()`]. Its
    /// [`path`](Projection::path) and [`id`](Projection::id) give the trait path and the `KEY` of
    /// each impl ([`RepeaterImpl`]), and its [`impl_self`](Projection::impl_self) says what type
    /// each impl is for ([`ImplSelf`]).
    ///
    /// Fails with the first of these errors, and returns no impl:
    ///
    /// - [`EncodeError::NeedsSelf`] for [`ImplSelf::Marker`], if an entry or its context names
    ///   `Self`. The error holds the first such entry.
    /// - [`EncodeError::UnsupportedConstType`] for the first const parameter whose type is not a
    ///   primitive type, in the order of the entries and of their parameters.
    ///
    /// # Panics
    ///
    /// If the path of `projection` has arguments (`crate::Repeater<u8>`): it must name the trait
    /// alone, or with an empty `<>`. Also if `projection` is [`Projection::unused()`] and there is
    /// an entry.
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// let mut leaker = Leaker::new();
    /// leaker.intern_struct(&parse_quote!(struct S<T>(Wrap<T>);));
    /// let result = leaker.result();
    ///
    /// let projection = Projection::new(
    ///     parse_quote!(crate::Repeater),
    ///     7,
    ///     ImplSelf::Param(parse_quote!(__S)),
    /// );
    /// let impls = result
    ///     .repeater_impls(&projection)
    ///     .unwrap();
    /// assert_eq!(impls[0].ty, parse_quote!(Wrap<T>));
    /// // `path` has no `KEY`, and `ToTokens` writes the one of the id 7.
    /// assert_eq!(impls[0].path, parse_quote!(crate::Repeater<(T,)>));
    /// ```
    ///
    /// An entry that names `Self` has no impl for a marker, so the call fails:
    ///
    /// ```
    /// # use type_leak::*;
    /// # use syn::*;
    /// let mut leaker = Leaker::new();
    /// leaker.intern_struct(&parse_quote!(struct S(Wrap<Self>);));
    /// let result = leaker.result();
    ///
    /// let projection = Projection::new(
    ///     parse_quote!(Repeater),
    ///     7,
    ///     ImplSelf::Marker(parse_quote!(Marker)),
    /// );
    /// let err = result.repeater_impls(&projection).err();
    /// assert!(matches!(err, Some(EncodeError::NeedsSelf(_))));
    /// ```
    pub fn repeater_impls(
        &self,
        projection: &Projection,
    ) -> std::result::Result<Vec<RepeaterImpl>, EncodeError> {
        self.referrer.repeater_impls(projection)
    }
}

/// Writes the [`Referrer`], then, if errors remain, `@ [ {error} {error} .. ]`: an `@` and one
/// bracket group with a brace group for each [`ErrorItem`], in the order found.
impl ToTokens for LeakerResult {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        self.referrer.to_tokens(tokens);
        if !self.errors.is_empty() {
            tokens.extend(quote! {
                @ [#(for error in &self.errors) { #error }]
            });
        }
    }
}

/// Reads a [`Referrer`], then the errors of `@ [ {error} .. ]` if they follow, and leaves any
/// other tokens (a brace group, for example). A Referrer alone reads as a result with no errors.
/// Fails as the [`Referrer`] does (a missing or different format tag), on an `@ [ .. ]` that holds
/// something else than errors, and on an error that names a part, an entry or an error that is
/// not there.
impl Parse for LeakerResult {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let referrer: Referrer = input.parse()?;
        let mut errors = Vec::new();
        let mut spans = Vec::new();
        if input.peek(Token![@]) && input.peek2(token::Bracket) {
            input.parse::<Token![@]>()?;
            let list;
            bracketed!(list in input);
            while !list.is_empty() {
                if !list.peek(token::Brace) {
                    return Err(list.error("expected an error `{ .. }` in `@ [ .. ]`"));
                }
                spans.push(list.span());
                errors.push(list.parse()?);
            }
        }
        check_links(&referrer, &errors, &spans)?;
        Ok(LeakerResult { referrer, errors })
    }
}

/// Fails at the span of the first error that is about a part, an entry or an error that is not
/// there, or has the `id` of another.
fn check_links(referrer: &Referrer, errors: &[ErrorItem], spans: &[Span]) -> syn::Result<()> {
    let mut ids = HashSet::new();
    for (error, span) in errors.iter().zip(spans) {
        let link = &error.link;
        if let Some(part) = link.part.filter(|&part| part >= referrer.part_count()) {
            let message = format!("the part {} of an error is not one of the parts", part);
            return Err(syn::Error::new(*span, message));
        }
        if let Some(entry) = link.entry.filter(|&entry| entry >= referrer.entry_count()) {
            let message = format!("the entry {} of an error is not one of the entries", entry);
            return Err(syn::Error::new(*span, message));
        }
        if let Some(id) = link.id.filter(|&id| !ids.insert(id)) {
            let message = format!("the id {} is the id of another error", id);
            return Err(syn::Error::new(*span, message));
        }
    }
    for (error, span) in errors.iter().zip(spans) {
        if let Some(id) = error.link.inner.iter().find(|id| !ids.contains(id)) {
            let message = format!(
                "an error inside an error has the id {}: no error has it",
                id
            );
            return Err(syn::Error::new(*span, message));
        }
    }
    Ok(())
}

/// The errors that [`LeakerResult::finish()`] returns: at least one, in the order found.
///
/// - It reads as a slice of [`ErrorItem`]s (`len()`, `[0]`, `iter()`), or as a `Vec` with
///   [`into_vec()`](LeakerResultError::into_vec).
/// - It is a `std::error::Error` whose text is the messages of the errors, one per line.
/// - [`ToTokens`] writes a `compile_error!` for each error.
/// - `From` makes one [`syn::Error`] of them, so a proc macro can use `?`.
///
/// ```
/// # use {syn::*, type_leak::*};
/// # use template_quote::ToTokens;
/// fn analyze(item: &ItemStruct) -> Result<Referrer> {
///     let mut leaker = Leaker::new();
///     leaker.intern_struct(item);
///     Ok(leaker.result().finish()?)
/// }
///
/// let item = parse_quote!(struct S(MyType<'_>, Other<'_>););
/// let error = analyze(&item).unwrap_err();
/// // One error for each of the two types, in the order found.
/// assert_eq!(error.into_iter().count(), 2);
///
/// let mut leaker = Leaker::new();
/// leaker.intern_struct(&item);
/// let errors = leaker.result().finish().unwrap_err();
/// assert_eq!(errors.len(), 2);
/// assert_eq!(errors[0].cause, Cause::Type(parse_quote!(MyType<'_>)));
/// assert_eq!(
///     errors.to_string(),
///     "use absolute path: 'MyType<'_>'\nuse absolute path: 'Other<'_>'"
/// );
/// // `compile_error!` for each error.
/// assert_eq!(errors.to_token_stream().to_string().matches("compile_error").count(), 2);
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LeakerResultError(Vec<ErrorItem>);

impl LeakerResultError {
    /// Returns the errors, in the order found. The vector is never empty.
    ///
    /// ```
    /// # use {syn::*, type_leak::*};
    /// let mut leaker = Leaker::new();
    /// leaker.intern_struct(&parse_quote!(struct S(MyType<'_>);));
    /// let errors: Vec<ErrorItem> = leaker.result().finish().unwrap_err().into_vec();
    /// assert_eq!(errors.len(), 1);
    /// ```
    pub fn into_vec(self) -> Vec<ErrorItem> {
        self.0
    }
}

/// Reads the errors as a slice: `errors.len()`, `errors[0]`, `errors.iter()`, `&errors` as
/// `&[ErrorItem]`.
///
/// ```
/// # use {syn::*, type_leak::*};
/// let mut leaker = Leaker::new();
/// leaker.intern_struct(&parse_quote!(struct S(MyType<'_>);));
/// let errors = leaker.result().finish().unwrap_err();
/// let slice: &[ErrorItem] = &errors;
/// assert_eq!(slice.len(), 1);
/// assert_eq!(errors[0].reason, Reason::RelativePath);
/// ```
impl std::ops::Deref for LeakerResultError {
    type Target = [ErrorItem];

    fn deref(&self) -> &[ErrorItem] {
        &self.0
    }
}

/// Goes through the errors, in the order found.
///
/// ```
/// # use {syn::*, type_leak::*};
/// let mut leaker = Leaker::new();
/// leaker.intern_struct(&parse_quote!(struct S(MyType<'_>);));
/// let errors = leaker.result().finish().unwrap_err();
/// let causes: Vec<Cause> = errors.into_iter().map(|error| error.cause).collect();
/// assert_eq!(causes, [Cause::Type(parse_quote!(MyType<'_>))]);
/// ```
impl IntoIterator for LeakerResultError {
    type Item = ErrorItem;
    type IntoIter = std::vec::IntoIter<ErrorItem>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

/// Goes through the errors by reference, in the order found.
///
/// ```
/// # use {syn::*, type_leak::*};
/// let mut leaker = Leaker::new();
/// leaker.intern_struct(&parse_quote!(struct S(MyType<'_>);));
/// let errors = leaker.result().finish().unwrap_err();
/// let mut count = 0;
/// for error in &errors {
///     assert_eq!(error.reason, Reason::RelativePath);
///     count += 1;
/// }
/// assert_eq!(count, 1);
/// ```
impl<'a> IntoIterator for &'a LeakerResultError {
    type Item = &'a ErrorItem;
    type IntoIter = std::slice::Iter<'a, ErrorItem>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

/// Writes the messages of the errors, one per line.
///
/// ```
/// # use {syn::*, type_leak::*};
/// let mut leaker = Leaker::new();
/// leaker.intern_struct(&parse_quote!(struct S(MyType<'_>, Other<'_>);));
/// let errors = leaker.result().finish().unwrap_err();
/// assert_eq!(
///     errors.to_string(),
///     "use absolute path: 'MyType<'_>'\nuse absolute path: 'Other<'_>'"
/// );
/// ```
impl std::fmt::Display for LeakerResultError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (ix, error) in self.0.iter().enumerate() {
            if ix > 0 {
                f.write_str("\n")?;
            }
            std::fmt::Display::fmt(error, f)?;
        }
        Ok(())
    }
}

/// Lets `?` pass the errors of `finish()` on as a `Box<dyn Error>`.
///
/// ```
/// # use {syn::*, type_leak::*};
/// fn analyze(item: &ItemStruct) -> std::result::Result<Referrer, Box<dyn std::error::Error>> {
///     let mut leaker = Leaker::new();
///     leaker.intern_struct(item);
///     Ok(leaker.result().finish()?)
/// }
/// let error = analyze(&parse_quote!(struct S(MyType<'_>);)).unwrap_err();
/// assert_eq!(error.to_string(), "use absolute path: 'MyType<'_>'");
/// ```
impl std::error::Error for LeakerResultError {}

/// Writes the [`ErrorItem::to_compile_error()`] of each error, in the order found. To report the
/// errors from a proc macro with the nightly diagnostic, call [`ErrorItem::emit()`] for each.
///
/// ```
/// # use {syn::*, type_leak::*};
/// # use template_quote::ToTokens;
/// let mut leaker = Leaker::new();
/// leaker.intern_struct(&parse_quote!(struct S(MyType<'_>, Other<'_>);));
/// let errors = leaker.result().finish().unwrap_err();
/// let written: proc_macro2::TokenStream =
///     errors.iter().map(|error| error.to_compile_error()).collect();
/// assert_eq!(errors.to_token_stream().to_string(), written.to_string());
/// ```
impl ToTokens for LeakerResultError {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        for error in &self.0 {
            tokens.extend(error.to_compile_error());
        }
    }
}

/// Makes one [`syn::Error`] of the errors: each error is converted as an [`ErrorItem`] is, and
/// they are joined in order with [`syn::Error::combine()`].
///
/// ```
/// # use {syn::*, type_leak::*};
/// let mut leaker = Leaker::new();
/// leaker.intern_struct(&parse_quote!(struct S(MyType<'_>, Other<'_>);));
/// let errors = leaker.result().finish().unwrap_err();
/// let error = Error::from(errors);
/// let messages: Vec<String> = error.into_iter().map(|error| error.to_string()).collect();
/// assert_eq!(messages.len(), 2);
/// assert!(messages[0].contains("MyType"));
/// assert!(messages[1].contains("Other"));
/// ```
impl From<LeakerResultError> for syn::Error {
    fn from(errors: LeakerResultError) -> Self {
        let mut errors = errors.0.into_iter().map(syn::Error::from);
        // `LeakerResultError` is never empty.
        let mut combined = errors
            .next()
            .expect("a LeakerResultError holds at least one error");
        for error in errors {
            combined.combine(error);
        }
        combined
    }
}

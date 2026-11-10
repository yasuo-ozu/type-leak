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

/// Where in an item a path or an error is. [`Allow::allow()`] is asked at a place; it is
/// [`ErrorItem::place`] for an error.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct Place {
    /// While a value is read ([`Leaker::check_values`](crate::Leaker::check_values)): the value
    /// of a const or static, the body of a function, a trait's default value or body.
    pub in_value: bool,
    /// In a default: a default body or an associated const default of a trait, and the default of
    /// a generic parameter (`U = Rel<'_>`, `T = Wrap<Self>`).
    pub in_default: bool,
}

/// What a [`Leaker`] may write as it is. Pass it to
/// [`Leaker::with_rule()`](crate::Leaker::with_rule), or to
/// [`LeakerResult::allow()`](crate::LeakerResult::allow).
///
/// `allow()` means "the copy may write `cause` as it is at `place`", and is the one question at both
/// stages. The analysis asks it about each relative path, and about each absolute path in a type
/// (`::core::..` and `$crate::..` excepted), as the cause of its kind: a path in a type is a
/// [`Cause::Type`], the trait of a bound or of a qualified path a [`Cause::Trait`], a path in a
/// value a [`Cause::Expr`], the path of a macro a [`Cause::Macro`], a path of unknown kind (the
/// root of a `use`, a path in the tokens of a macro) a [`Cause::Unknown`]. The paths are bare:
/// their generic arguments are analyzed on their own. A path that starts with `Self` or a generic
/// in scope is not asked as the analysis walks. An allowed path is not leaked, and you promise that
/// it means the same in the copy (an absolute path: that the crate is the same on both sides). The
/// rule is called on every check of a path: keep it cheap. A rule that panics rolls the analysis
/// back.
///
/// After the analysis, [`LeakerResult::allow()`](crate::LeakerResult::allow) accepts an error by
/// its [`paths`](ErrorItem::paths) if it has any (the rule must allow each, at the error's
/// field [`place`](ErrorItem::place); the error stays without those it allows), else by its cause,
/// asked as written, with its generic arguments. A rule on a whole cause (`Wrap<Rel>`) never
/// accepts an error that has paths. That only drops errors: it changes no entry, and a type of
/// another crate that the rule allows is still an entry. That is the approximation of the rule of
/// the Leaker: give the rule to the Leaker when you can.
///
/// The Leaker drops the errors of a scope the same way when the scope ends. The error of a `Self`
/// that has no meaning ([`Reason::SelfWithoutMeaning`]) has no
/// paths, so a rule that allows its cause (`Self`, `Self::Item`) accepts it at both stages, though
/// a path that starts with `Self` is not asked as the analysis walks.
///
/// A closure `Fn(&Cause, Place) -> bool` is a rule, `()` allows nothing, and `(A, B)` allows what
/// either does. [`Rules`] builds the common ones. A closure must annotate its first parameter
/// (`|cause: &Cause, _| ..`) wherever it is passed (E0282, E0631); [`Rules::or_if()`] needs none.
///
/// ```
/// # use type_leak::*;
/// # use syn::*;
/// // You promise that `::serde` is the same crate on both sides.
/// let rule = Rules::new().or_if(|cause, _| cause.starts_with(&parse_quote!(::serde)));
/// let mut leaker = Leaker::with_rule(rule);
/// leaker.intern_struct(&parse_quote!(struct S { a: ::serde::Value, b: MyType }));
/// // Only `MyType` is leaked.
/// assert_eq!(leaker.result().finish().unwrap().iter().count(), 1);
/// ```
pub trait Allow {
    /// Returns `true` if the copy may write `cause` as it is, at `place`.
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

/// The built-in rules and your own, as one [`Allow`]. Rules add up: it allows what any of them
/// does.
///
/// The methods set a flag and return the same type, so a choice at run time keeps it; [`or()`] and
/// [`or_if()`] add a rule of your own and change `A`. The built-in path rules allow a cause that
/// is a path without generic arguments ([`Cause::path()`]). On a result, an error is accepted by
/// its paths, or by its cause if it has none.
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
    /// Allows the primitive types (`u8`, `bool`, `str`, ...).
    pub fn primitive(mut self) -> Self {
        self.primitive = true;
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
    /// gives their absolute paths. `Sized` is allowed as a bound; `?Sized` needs no rule.
    pub fn prelude(mut self) -> Self {
        self.prelude = true;
        self
    }

    /// Allows the paths that start with `std`, `core` or `alloc`, also those with a leading `::`
    /// (`::std::vec::Vec`), which are no entry in the Leaker: the copy writes them as they are. On
    /// a [`LeakerResult`] it only drops the errors of relative paths.
    ///
    /// You promise that the copy does not shadow the relative names. [`external_crates()`] allows
    /// the other absolute paths. A `::core` path needs no rule: it is never an entry.
    ///
    /// [`external_crates()`]: Rules::external_crates
    pub fn std_crates(mut self) -> Self {
        self.std_crates = true;
        self
    }

    /// Allows every absolute path (`::foo::X`), `::std` included.
    ///
    /// In a Leaker, a type of another crate (`::foo::X`) is no entry: the copy writes it as it is,
    /// and not through the Repeater. Without it, such a type is an entry and is replaced. On a
    /// [`LeakerResult`] it changes no entry, as `allow()` only drops errors.
    /// You promise that each crate is the same crate on both sides.
    pub fn external_crates(mut self) -> Self {
        self.external_crates = true;
        self
    }

    /// Allows the paths that start with `crate`.
    ///
    /// Use it only if the copy is in the same crate as the item: `crate::Foo` names another type
    /// in another crate.
    pub fn crate_paths(mut self) -> Self {
        self.crate_paths = true;
        self
    }

    /// Allows everything found in a value (see [`Place::in_value`]): the Leaker writes every
    /// relative path in a value as it is, also the type of a `let`, instead of leaking it. It has
    /// an effect only with [`Leaker::check_values`](crate::Leaker::check_values).
    pub fn values(mut self) -> Self {
        self.values = true;
        self
    }

    /// Allows everything found in a default (see [`Place::in_default`]): the Leaker writes every
    /// relative path in a default as it is (`U = Rel<'_>`, `T = Wrap<Self>`) instead of leaking
    /// it.
    pub fn defaults(mut self) -> Self {
        self.defaults = true;
        self
    }

    /// Adds `rule`: the rules allow what it allows too. Pass a closure to
    /// [`or_if()`](Rules::or_if), which needs no annotations.
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

    /// Adds the closure `rule` like [`or()`](Rules::or).
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
/// [`finish()`](LeakerResult::finish) gives the Referrer only when no error remains: accept each
/// error with [`allow()`](LeakerResult::allow) or [`ignore()`](LeakerResult::ignore), and report the
/// others ([`ErrorItem::emit()`]).
///
/// [`allow()`](LeakerResult::allow) takes and returns the result, so it chains. It drops the errors
/// that an [`Allow`] accepts (you promise that the part compiles as written). It leaves the
/// entries as they are: to remove entries, give the [`Allow`] to [`Leaker::with_rule()`]. A rule
/// here only approximates the one of the Leaker (more errors may remain, and a copy may fail to
/// build): give rules to the Leaker when you can. [`ignore()`](LeakerResult::ignore) accepts one
/// error, and has the copy write its whole part as written, or removes its entry;
/// [`ignore_if()`](LeakerResult::ignore_if) does it for each error that a closure picks.
///
/// ```
/// # use {syn::*, type_leak::*};
/// let mut leaker = Leaker::new();
/// leaker.intern_struct(&parse_quote!(struct S(::p::W<'_, dyn RelTr>);));
/// let result = leaker.result();
/// assert_eq!(result.errors()[0].cause, Cause::Trait(parse_quote!(RelTr)));
/// // Allowing the error copies the part as written, and `finish()` is `Ok`.
/// let result = result.allow(|cause: &Cause, _: Place| *cause == Cause::Trait(parse_quote!(RelTr)));
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
    /// Drops the errors that `rule` accepts: the part is copied as written. An error is accepted
    /// by its [`paths`](ErrorItem::paths) if it has any, else by its cause. The other errors stay,
    /// in order.
    ///
    /// You promise that the part can be written as it is in the copy. The cause may be smaller
    /// than the part that is copied: in `[u8; { let c = |a: u8| a; 3 }]` the error is about `u8`,
    /// and the array is copied whole. Allowing a [`Reason::UnprovableBound`] does not make the copy
    /// compile (E0277). To allow a cause, compare it with `==` (the same kind and tokens, spans
    /// ignored). An absolute path is never an error, so a rule for `::serde` changes no error. It
    /// changes no entry either: to have the copy write a type of another crate as it is, give
    /// [`Rules::external_crates()`](crate::Rules::external_crates) to
    /// [`Leaker::with_rule()`].
    ///
    /// See [`Allow`] for what the rule is asked. A closure must annotate its first parameter
    /// (`|cause: &Cause, _| ..`), even written in the call; [`Rules::or_if()`](crate::Rules::or_if)
    /// needs no annotation. To have the copy write the whole part as written, use
    /// [`ignore()`](LeakerResult::ignore).
    pub fn allow(mut self, rule: impl Allow) -> Self {
        filter(&rule, &mut self.errors);
        self
    }

    /// Accepts `error`, one of [`errors()`](LeakerResult::errors): the copy writes the part that it
    /// is about as written, whole, and nothing is looked up in it. The error goes, and so do the
    /// errors revealed inside that part and the errors about the same part.
    ///
    /// [`allow()`](LeakerResult::allow) writes only the head that is wrong as written (`MyType` in
    /// `MyType<'_, Rel>`) and replaces the entries in the arguments; `ignore()` writes all of
    /// `MyType<'_, Rel>` as written, so `Rel` is not replaced there. An entry that is used at
    /// another place is still one there. You promise that the part compiles as written.
    ///
    /// The error of an entry ([`Reason::NestedSelfInBounds`]) removes the entry instead: it gets no
    /// impl ([`repeater_impls()`](LeakerResult::repeater_impls)), [`Referrer::iter()`] skips it,
    /// and the copy writes its type as written wherever it would have been replaced. The other
    /// entries keep their [`EntryId`]s, so it can be called before or after the impls are written.
    ///
    /// An error that is about no part (a `Self` without meaning, a path in a value) is only
    /// dropped. An error about an entry inside the part stays: the entry may be used elsewhere, so
    /// ignore it on its own. If `error` is not one of the errors, nothing changes, so calling it
    /// again is harmless. Call it before [`allow()`](LeakerResult::allow) for the same error.
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

    /// Calls `f` for each error that remains, in the order found, and ignores the ones it returns
    /// `true` for, as [`ignore()`](LeakerResult::ignore) does.
    ///
    /// The errors that go with an ignored one (the same part or entry, and those inside its part)
    /// are dropped and not asked about.
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

    /// Returns the Referrer if no error remains, else the errors that remain, in the order found,
    /// as a [`LeakerResultError`].
    ///
    /// It is the only way to the Referrer: drop each error with [`allow()`](LeakerResult::allow)
    /// or [`ignore()`](LeakerResult::ignore) first, and report the others. In a proc macro,
    /// `?` turns the error into a [`syn::Error`] that holds each of them.
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

    /// Describes the Repeater impl of each entry: its generics, trait path and type.
    ///
    /// It is the Leaker's side that writes the impls: write one for every entry, next to the
    /// analyzed items. They come in the order of the entries
    /// ([`Referrer::iter()`], so without those that [`ignore()`](LeakerResult::ignore) removed),
    /// each with its [`EntryId`] in [`RepeaterImpl::entry`], which the
    /// Repeater's `KEY` ([`Projection::key()`]) and [`Projection::of()`] take.
    ///
    /// `projection` is the one that the copy's side uses too ([`Projection::of()`]): its
    /// [`path`](Projection::path) becomes [`RepeaterImpl::path`], with the first argument, and its
    /// [`id`](Projection::id) is in the `KEY` that `ToTokens` of the item writes
    /// ([`Projection::key(entry)`](Projection::key)); [`ImplSelf::Bucket`] chooses the type from the
    /// same `id`. Its [`impl_self`](Projection::impl_self) says what every impl is for
    /// ([`ImplSelf`]). The consts of the entries are encoded as [`Projection::of()`] does.
    ///
    /// The call fails with the first of these:
    ///
    /// - [`EncodeError::NeedsSelf`], for [`ImplSelf::Marker`], if some entry names `Self`, or its
    ///   context's generics do. The error has the first such entry, and no impl is made.
    /// - The first [`EncodeError::UnsupportedConstType`], in the order of the entries and of their
    ///   parameters.
    ///
    /// # Panics
    ///
    /// If the path of `projection` has arguments (`crate::Repeater<u8>`): it names the trait alone,
    /// or with an empty `<>`. Also if `projection` is [`Projection::unused()`] and there is an
    /// entry.
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

/// Writes the [`Referrer`], then, if errors remain, `@ [ {error} {error} .. ]`: one brace group for
/// each [`ErrorItem`], in the order found, in one bracket group after an `@`.
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

/// Reads a [`Referrer`], then the errors of `@ [ {error} .. ]` if that follows. Anything else
/// after the Referrer is left to the caller: only an `@` followed by a bracket group is read, so a
/// brace group after the result is not an error. Fails as the [`Referrer`] does (a missing or
/// different format tag), and on an `@ [ .. ]` that holds something else than error groups. A
/// serialized Referrer parses with no errors.
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
/// It reads as a slice of [`ErrorItem`]s (`len()`, `[0]`, `iter()`), or as a `Vec` with
/// [`into_vec()`](LeakerResultError::into_vec). It is a `std::error::Error` whose text is the
/// messages of the errors, one per line. [`ToTokens`] writes a `compile_error!` for each error,
/// and `From` makes one [`syn::Error`] of them, so a proc macro can use `?`.
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

/// Lets `finish()` go through `?` in a function that returns a `Box<dyn Error>`.
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

/// Writes the [`ErrorItem::to_compile_error()`] of each error, in the order found. It is pure; to
/// report from a proc macro with a diagnostic on nightly, call [`ErrorItem::emit()`] for each
/// error.
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

/// Makes one [`syn::Error`] of the errors, with [`syn::Error::combine()`]: each error is
/// converted as [`ErrorItem`] is, in order.
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

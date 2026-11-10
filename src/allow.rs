//! What a [`Leaker`](crate::Leaker) may write as it is: the [`Allow`] trait, the [`Rules`] that
//! build one, and the filter that applies it to errors. [`LeakerResult`](crate::LeakerResult)
//! applies it too.

use crate::error::{Cause, ErrorItem};
use crate::util::{heads_self, starts_with_known, without_arguments, Known, PathOf, STD_CRATES};
use proc_macro2::TokenStream;
use syn::Path;
use template_quote::ToTokens;

/// Where in an item a path or an error is. [`Allow::allow()`] is asked at a place; it is
/// [`ErrorItem::place()`] for an error.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct Place {
    /// While a value is read ([`check_values()`](crate::Options::check_values)): the value of a
    /// const or static, the body of a function, a trait's default value or body.
    pub in_value: bool,
    /// In a default: a default body or an associated const default of a trait, and the default of
    /// a generic parameter (`U = Rel<'_>`, `T = Wrap<Self>`).
    pub in_default: bool,
}

/// What a [`Leaker`](crate::Leaker) may write as it is. Pass it to
/// [`Leaker::from_config()`](crate::Leaker::from_config), or to
/// [`LeakerResult::allow()`](crate::LeakerResult::allow).
///
/// `allow()` means "the copy may write `cause` as it is at `place`", and is the one question at both
/// stages. The analysis asks it about each relative path, and about each absolute path in a type
/// (`::core::..` and `$crate::..` excepted), as the cause of its kind: a path in a type is a
/// [`Cause::Type`], the trait of a bound or of a qualified path a [`Cause::Trait`], a path in a
/// value a [`Cause::Expr`], the path of a macro a [`Cause::Macro`]. A path of unknown kind (the
/// root of a `use` in a value) is allowed if the rule allows it as a type or as a trait. The
/// paths are bare: their generic arguments are analyzed on their own. A path that starts with
/// `Self` or a generic in scope is never asked. An allowed path is not leaked, and you promise
/// that it means the same in the copy (an absolute path: that the crate is the same on both
/// sides). The rule is called on every check of a path: keep it cheap. A rule that panics rolls the
/// analysis back.
///
/// After the analysis, [`LeakerResult::allow()`](crate::LeakerResult::allow) accepts an error by
/// its [`paths`](ErrorItem::paths) if it has any (the rule must allow each, at the error's
/// [`place()`](ErrorItem::place); the error stays without those it allows), else by its cause,
/// asked as written, with its generic arguments. A rule on a whole cause (`Wrap<Rel>`) never
/// accepts an error that has paths. That only drops errors; to keep paths out of the entries, give
/// the rule to the Leaker.
///
/// A closure `Fn(&Cause, Place) -> bool` is a rule, `()` allows nothing, and `(A, B)` allows what
/// either does. [`Rules`] builds the common ones.
///
/// ```
/// # use type_leak::*;
/// # use syn::*;
/// // You promise that `::serde` is the same crate on both sides.
/// let allow = Rules::new()
///     .primitive()
///     .prelude()
///     .or_if(|cause, _| cause.starts_with(&parse_quote!(::serde)));
/// let mut leaker = Leaker::from_config(allow, Options::new());
/// // `Vec` and `u8` are allowed, so only `MyType` is leaked.
/// let item: ItemStruct = parse_quote!(struct S { a: Vec<u8>, b: MyType });
/// leaker.intern_struct(&item, true);
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
/// # let flag = true;
/// let mut rules = Rules::new().primitive();
/// if flag {
///     rules = rules.prelude();
/// }
/// // A rule of your own; `or_if()` needs no annotations.
/// let rules = rules.or_if(|cause, place| place.in_default && cause.path().is_some());
/// # let _ = rules;
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

    /// Allows the relative paths that start with `std`, `core` or `alloc`. A path with a leading
    /// `::` is context-free in any case.
    pub fn std_crates(mut self) -> Self {
        self.std_crates = true;
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
    /// an effect only with [`Options::check_values()`](crate::Options::check_values).
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

/// Returns `true` if `path` is relative and starts with `std`, `core` or `alloc`.
fn is_std_crate_path(path: &Path) -> bool {
    path.leading_colon.is_none()
        && path
            .segments
            .first()
            .map_or(false, |seg| STD_CRATES.iter().any(|name| seg.ident == name))
}

/// Returns `true` if `path` is relative and starts with `crate`.
fn is_crate_path(path: &Path) -> bool {
    path.leading_colon.is_none()
        && path
            .segments
            .first()
            .map_or(false, |seg| seg.ident == "crate")
}

/// Asks `rule` about `path`, which is the path of `of`, without the generic arguments of its
/// segments, wherever the rule is asked: as the cause of that kind, or if the kind is not known as
/// a type or as a trait.
pub(crate) fn ask<A: Allow + ?Sized>(rule: &A, path: &Path, of: PathOf, place: Place) -> bool {
    match of.entry(path) {
        Cause::Unreadable(_) => {
            let bare = without_arguments(path);
            rule.allow(&Cause::type_path(&bare), place) || rule.allow(&Cause::Trait(bare), place)
        }
        entry => rule.allow(&entry, place),
    }
}

/// Returns `true` if `rule` allows the entry `entry` of [`ErrorItem::paths`] at `place`. It is
/// never asked about a path that starts with `Self`.
fn allows_entry<A: Allow + ?Sized>(rule: &A, entry: &Cause, place: Place) -> bool {
    entry.entry_path().map_or(false, |path| {
        !heads_self(&path) && ask(rule, &path, PathOf::of_entry(entry), place)
    })
}

/// Returns `true` if `rule` accepts `error`: if it has relative paths ([`ErrorItem::paths`]), it
/// allows each; else it allows the cause. If only some paths are allowed, the error stays without
/// them ([`filter()`]).
fn accepts<A: Allow + ?Sized>(rule: &A, error: &ErrorItem) -> bool {
    let place = error.place();
    if error.paths.is_empty() {
        rule.allow(&error.cause, place)
    } else {
        error
            .paths
            .iter()
            .all(|entry| allows_entry(rule, entry, place))
    }
}

/// Drops the errors that `rule` accepts, and puts their nested errors in their place
/// ([`ErrorItem::nested`]): the part is copied as written, so what is wrong inside it is real
/// whichever rule accepted it. The nested errors are filtered the same way, down to the last. An
/// error that stays loses the paths that the rule allows, so that rules given one after the other
/// add up as they do in one set (and in the serialization of the error, which carries its paths);
/// its nested errors are filtered too, for when it is dropped later.
///
/// A nested error that comes up is skipped if it is identical but for its spans to an error that
/// stays, before it or after it ([`same()`]). Errors that did not come up, and errors that come
/// up, are never merged with each other.
pub(crate) fn filter<A: Allow + ?Sized>(rule: &A, errors: &mut Vec<ErrorItem>) {
    // The errors that stay, and for each group that comes up the number of those before it.
    let mut stays = Vec::new();
    let mut ups = Vec::new();
    for mut error in std::mem::take(errors) {
        if accepts(rule, &error) {
            ups.push((stays.len(), error.nested));
        } else {
            trim(rule, &mut error);
            stays.push(error);
        }
    }
    let mut came_up = Vec::new();
    for (before, nested) in ups {
        for error in nested {
            raise(rule, error, before, &stays, &mut came_up);
        }
    }
    // Back in the order of the input: what came up goes where its outer error was.
    let mut came_up = came_up.into_iter().peekable();
    for (ix, error) in stays.into_iter().enumerate() {
        while let Some((_, up)) = came_up.next_if(|(before, _)| *before <= ix) {
            errors.push(up);
        }
        errors.push(error);
    }
    errors.extend(came_up.map(|(_, up)| up));
}

/// Removes from `error`, which stays, the paths that `rule` allows, and filters its nested errors.
fn trim<A: Allow + ?Sized>(rule: &A, error: &mut ErrorItem) {
    let place = error.place();
    error
        .paths
        .retain(|entry| !allows_entry(rule, entry, place));
    filter(rule, &mut error.nested);
}

/// Puts `error`, which came up, in `came_up` after `before` errors that stay if `rule` does not
/// accept it and it is not `same()` as one of `stays`, else its nested errors that it does not
/// accept.
fn raise<A: Allow + ?Sized>(
    rule: &A,
    mut error: ErrorItem,
    before: usize,
    stays: &[ErrorItem],
    came_up: &mut Vec<(usize, ErrorItem)>,
) {
    if accepts(rule, &error) {
        for nested in error.nested {
            raise(rule, nested, before, stays, came_up);
        }
    } else {
        // Trimmed like the errors that stay, so that what is compared is what would be listed.
        trim(rule, &mut error);
        if !stays.iter().any(|other| same(other, &error)) {
            came_up.push((before, error));
        }
    }
}

/// Returns `true` if the errors are identical but for their spans, which cannot be compared
/// reliably: the same cause ([`Cause`] equality), reason, message, place, paths, notes, help and
/// nested errors, the tokens compared as text. So that a skipped error is never one that a rule
/// would treat differently from the one it equals.
fn same(left: &ErrorItem, right: &ErrorItem) -> bool {
    fn text(tokens: &impl ToTokens) -> String {
        tokens.to_token_stream().to_string()
    }
    fn note(left: &(TokenStream, String), right: &(TokenStream, String)) -> bool {
        left.1 == right.1 && text(&left.0) == text(&right.0)
    }
    left.cause == right.cause
        && left.reason == right.reason
        && left.message == right.message
        && left.place() == right.place()
        && left.paths == right.paths
        && left.notes.len() == right.notes.len()
        && left.notes.iter().zip(&right.notes).all(|(l, r)| note(l, r))
        && match (&left.help, &right.help) {
            (Some(left), Some(right)) => note(left, right),
            (None, None) => true,
            _ => false,
        }
        && left.nested.len() == right.nested.len()
        && left
            .nested
            .iter()
            .zip(&right.nested)
            .all(|(l, r)| same(l, r))
}

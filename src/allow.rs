//! The rules of what a [`Leaker`](crate::Leaker) leaves as written and accepts: [`Allow`], and the
//! filter that applies them to errors. [`LeakerResult`](crate::LeakerResult) applies them too.

#[cfg(doc)]
use crate::error::Cause;
use crate::error::ErrorItem;
use crate::util::{ask_rule, heads_self, starts_with_known, Known, STD_CRATES};
#[cfg(doc)]
use crate::view::View;
use proc_macro2::TokenStream;
use std::fmt;
use syn::Path;
use template_quote::ToTokens;

/// What a [`Leaker`](crate::Leaker) leaves as written and accepts. Pass it to
/// [`Leaker::from_config()`](crate::Leaker::from_config).
///
/// The methods accept the same errors as those of [`LeakerResult`](crate::LeakerResult). Here the
/// path rules also keep the paths out of the entries; on a result they only drop errors. `P` and
/// `E` are the types of its rules on paths and on errors, which the methods build.
///
/// ```
/// # use type_leak::*;
/// # use syn::*;
/// // You promise that `::serde` is the same crate on both sides.
/// let allow = Allow::new()
///     .allow_primitive()
///     .allow_prelude()
///     .allow_path_if(|p| p.leading_colon.is_some() && p.segments[0].ident == "serde");
/// let mut leaker = Leaker::from_config(allow, false, false);
/// // `Vec` and `u8` are allowed, so only `MyType` is leaked.
/// let item: ItemStruct = parse_quote!(struct S { a: Vec<u8>, b: MyType });
/// leaker.intern_struct(&item, true);
/// assert_eq!(leaker.result().finish().unwrap().iter().count(), 1);
/// ```
///
/// Each method returns another type, so choose a built-in rule at run time inside one closure with
/// the functions [`is_primitive_path()`] and the like:
///
/// ```
/// # use type_leak::*;
/// # let primitive = true;
/// let allow = Allow::new().allow_path_if(move |p| primitive && is_primitive_path(p));
/// # let _ = Leaker::from_config(allow, false, false);
/// ```
#[derive(Clone)]
pub struct Allow<P = fn(&Path) -> bool, E = fn(&ErrorItem) -> bool> {
    /// Set by [`Allow::allow_path_if()`] and the like.
    pub(crate) paths: P,
    /// Set by [`Allow::allow()`] and the like.
    pub(crate) errors: E,
}

impl<P, E> fmt::Debug for Allow<P, E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Allow").finish_non_exhaustive()
    }
}

impl Allow {
    /// Creates rules that allow nothing.
    pub fn new() -> Self {
        let paths: fn(&Path) -> bool = |_| false;
        let errors: fn(&ErrorItem) -> bool = |_| false;
        Self { paths, errors }
    }
}

impl Default for Allow {
    fn default() -> Self {
        Self::new()
    }
}

impl<P: Fn(&Path) -> bool, E: Fn(&ErrorItem) -> bool> Allow<P, E> {
    /// Accepts the errors for which `rule` returns `true`: the part is copied as written. Rules add
    /// up: an error is accepted if any of them accepts it. See
    /// [`LeakerResult::allow()`](crate::LeakerResult::allow).
    pub fn allow<F: Fn(&ErrorItem) -> bool>(
        self,
        rule: F,
    ) -> Allow<P, impl Fn(&ErrorItem) -> bool> {
        let errors = self.errors;
        Allow {
            paths: self.paths,
            errors: move |error: &ErrorItem| errors(error) || rule(error),
        }
    }

    /// Leaves the paths for which `rule` returns `true` as written instead of leaking them, such
    /// as `::serde`. Rules add up.
    ///
    /// The rule sees each path without its generic arguments (which are still analyzed), as
    /// [`ErrorItem::paths`] has it: a relative path, or an absolute one in a type, but not
    /// `::core::..` or `$crate::..`. It is never asked about a path that starts with `Self` or a
    /// generic in scope. It is called on every check of a path: keep it cheap. You promise that an
    /// allowed crate is the same on both sides. An error whose relative paths are all allowed is
    /// accepted.
    ///
    /// Another absolute path (`::serde::Wrap<(Rel,)>`) is leaked as an *optional entry*
    /// ([`Referrer::must()`](crate::Referrer::must) is `false`), which the Referrer replaces or not
    /// ([`deny_ambiguous_crate_name()`](crate::LeakerResult::deny_ambiguous_crate_name)).
    /// `::core::..` and `$crate::..` are never optional. An absolute type that names the `Self` of
    /// a nested struct, enum or union (`::serde::Wrap<Self>` in [`View::visit_item()`]) is copied
    /// as written whatever the Referrer chooses
    /// ([`Visitor::with_nested_self()`](crate::Visitor::with_nested_self)).
    pub fn allow_path_if<F: Fn(&Path) -> bool>(self, rule: F) -> Allow<impl Fn(&Path) -> bool, E> {
        let paths = self.paths;
        Allow {
            paths: move |path: &Path| paths(path) || rule(path),
            errors: self.errors,
        }
    }

    /// Allows the primitive types (`u8`, `bool`, `str`, ...): [`is_primitive_path()`].
    pub fn allow_primitive(self) -> Allow<impl Fn(&Path) -> bool, E> {
        self.allow_path_if(is_primitive_path)
    }

    /// Allows the names of the standard prelude (`Vec`, `Option`, `Clone`, ...):
    /// [`is_prelude_path()`].
    ///
    /// Without it, a relative `Clone` in `trait Tr: Clone` is an error, and `Vec<T>` is leaked
    /// whole. You promise that the copy does not shadow the names.
    ///
    /// The names are those of the prelude of every edition, as of Rust 1.56. A name that a later
    /// prelude added is left out, as the copy may be older: `TryFrom`, `TryInto`, `FromIterator`,
    /// `Future`, `IntoFuture`, `size_of`, `size_of_val`, `align_of` and `align_of_val`. The help
    /// gives their absolute paths. `Sized` is allowed as a bound; `?Sized` needs no option.
    pub fn allow_prelude(self) -> Allow<impl Fn(&Path) -> bool, E> {
        self.allow_path_if(is_prelude_path)
    }

    /// Allows the relative paths that start with `std`, `core` or `alloc`: [`is_std_crate_path()`].
    /// A path with a leading `::` is context-free in any case.
    pub fn allow_std_crates(self) -> Allow<impl Fn(&Path) -> bool, E> {
        self.allow_path_if(is_std_crate_path)
    }

    /// Allows paths that start with `crate`: [`is_crate_path()`].
    ///
    /// Use it only if the copy is in the same crate as the item: `crate::Foo` names another type
    /// in another crate.
    pub fn allow_crate(self) -> Allow<impl Fn(&Path) -> bool, E> {
        self.allow_path_if(is_crate_path)
    }

    /// Accepts the errors found in a value: a const or static value, the body of a function, a
    /// trait's default value or body. It has an effect only with the `check_values` argument of
    /// [`Leaker::from_config()`](crate::Leaker::from_config).
    pub fn allow_values(self) -> Allow<P, impl Fn(&ErrorItem) -> bool> {
        self.allow(|error| error.in_value)
    }

    /// Accepts the errors found in a default: a default body or an associated const default of a
    /// trait, and the default of a generic parameter (`U = Rel<'_>`, `T = Wrap<Self>`).
    pub fn allow_default(self) -> Allow<P, impl Fn(&ErrorItem) -> bool> {
        self.allow(|error| error.in_default)
    }
}

/// The rule of [`Allow::allow_primitive()`]: a relative path that starts with a primitive type name.
///
/// A rule sees paths without generic arguments ([`Allow::allow_path_if()`]).
pub fn is_primitive_path(path: &Path) -> bool {
    starts_with_known(path, Known::Primitive)
}

/// The rule of [`Allow::allow_prelude()`]: a relative path that starts with a name of the prelude.
///
/// A rule sees paths without generic arguments ([`Allow::allow_path_if()`]).
pub fn is_prelude_path(path: &Path) -> bool {
    starts_with_known(path, Known::Prelude)
}

/// The rule of [`Allow::allow_std_crates()`]: a relative path that starts with `std`, `core` or
/// `alloc`.
///
/// A rule sees paths without generic arguments ([`Allow::allow_path_if()`]).
pub fn is_std_crate_path(path: &Path) -> bool {
    path.leading_colon.is_none()
        && path
            .segments
            .first()
            .map_or(false, |seg| STD_CRATES.iter().any(|name| seg.ident == name))
}

/// The rule of [`Allow::allow_crate()`]: a relative path that starts with `crate`.
///
/// A rule sees paths without generic arguments ([`Allow::allow_path_if()`]).
pub fn is_crate_path(path: &Path) -> bool {
    path.leading_colon.is_none()
        && path
            .segments
            .first()
            .map_or(false, |seg| seg.ident == "crate")
}

/// Returns `true` if `path` starts with `prefix`, segment by segment by identifier.
///
/// Both must have a leading `::` or neither; the generic arguments of the segments are ignored. A
/// rule sees paths without generic arguments ([`Allow::allow_path_if()`]).
pub fn path_starts_with(path: &Path, prefix: &Path) -> bool {
    prefix.leading_colon.is_some() == path.leading_colon.is_some()
        && prefix.segments.len() <= path.segments.len()
        && prefix
            .segments
            .iter()
            .zip(path.segments.iter())
            .all(|(prefix_seg, seg)| prefix_seg.ident == seg.ident)
}

impl<P: Fn(&Path) -> bool, E: Fn(&ErrorItem) -> bool> Allow<P, E> {
    /// Returns `true` if the rules accept `error`: the rule on errors accepts it, or it has
    /// relative paths ([`ErrorItem::paths`]) and the rule on paths accepts each. If only some are
    /// accepted, the error stays without them ([`filter()`](Self::filter)).
    pub(crate) fn accepts(&self, error: &ErrorItem) -> bool {
        (self.errors)(error)
            || (!error.paths.is_empty() && error.paths.iter().all(|path| self.allows_path(path)))
    }

    /// Returns `true` if the rule on paths accepts `path`, which is as [`ErrorItem::paths`] has it.
    /// It is never asked about a path that starts with `Self`.
    fn allows_path(&self, path: &Path) -> bool {
        !heads_self(path) && ask_rule(&self.paths, path)
    }

    /// Drops the errors that the rules accept, and puts their nested errors in their place
    /// ([`ErrorItem::nested`]): the part is copied as written, so what is wrong inside it is real
    /// whichever rule accepted it. The nested errors are filtered the same way, down to the last.
    /// An error that stays loses the paths that the rules allow, so that rules given one after the
    /// other add up as they do in one set (and in the serialization of the error, which carries its
    /// paths); its nested errors are filtered too, for when it is dropped later.
    ///
    /// A nested error that comes up is skipped if it is identical but for its spans to an error
    /// that stays, before it or after it ([`same()`]). Errors that did not come up, and errors that
    /// come up, are never merged with each other.
    pub(crate) fn filter(&self, errors: &mut Vec<ErrorItem>) {
        // The errors that stay, and for each group that comes up the number of those before it.
        let mut stays = Vec::new();
        let mut ups = Vec::new();
        for mut error in std::mem::take(errors) {
            if self.accepts(&error) {
                ups.push((stays.len(), error.nested));
            } else {
                self.trim(&mut error);
                stays.push(error);
            }
        }
        let mut came_up = Vec::new();
        for (before, nested) in ups {
            for error in nested {
                self.raise(error, before, &stays, &mut came_up);
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

    /// Removes from `error`, which stays, the paths that the rules allow, and filters its nested
    /// errors.
    fn trim(&self, error: &mut ErrorItem) {
        error.paths.retain(|path| !self.allows_path(path));
        self.filter(&mut error.nested);
    }

    /// Puts `error`, which came up, in `came_up` after `before` errors that stay if the rules do
    /// not accept it and it is not `same()` as one of `stays`, else its nested errors that they do
    /// not accept.
    fn raise(
        &self,
        mut error: ErrorItem,
        before: usize,
        stays: &[ErrorItem],
        came_up: &mut Vec<(usize, ErrorItem)>,
    ) {
        if self.accepts(&error) {
            for nested in error.nested {
                self.raise(nested, before, stays, came_up);
            }
        } else {
            // Trimmed like the errors that stay, so that what is compared is what would be listed.
            self.trim(&mut error);
            if !stays.iter().any(|other| same(other, &error)) {
                came_up.push((before, error));
            }
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
        && left.in_value == right.in_value
        && left.in_default == right.in_default
        && left.paths.len() == right.paths.len()
        && left
            .paths
            .iter()
            .zip(&right.paths)
            .all(|(l, r)| text(l) == text(r))
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

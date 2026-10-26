//! What the Leaker accepts of its errors: the one set of rules behind `allow*()` of both
//! [`LeakerConfig`](crate::LeakerConfig) and [`LeakerResult`](crate::LeakerResult).

use crate::error::{Cause, ErrorItem};
use crate::util::{path_starts_with, primitive_paths, std_crate_paths, std_prelude_paths};
use proc_macro2::TokenStream;
use syn::Path;
use template_quote::ToTokens;

/// Rules that accept errors. An error is accepted when any rule applies:
///
/// - its cause equals one of `causes` (the same kind and tokens, spans ignored);
/// - `values` is set and it was found in a value ([`ErrorItem::in_value`]);
/// - `defaults` is set and it was found in a default ([`ErrorItem::in_default`]);
/// - it has relative paths ([`ErrorItem::paths`]) and each starts with one of `paths`; if only some
///   do, the error stays without them.
///
/// The `paths` also make those paths context-free during the analysis, when the rules are those of
/// a `LeakerConfig`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub(crate) struct Allow {
    paths: Vec<Path>,
    causes: Vec<Cause>,
    values: bool,
    defaults: bool,
}

impl Allow {
    pub(crate) fn cause(&mut self, cause: Cause) {
        self.causes.push(cause);
    }

    pub(crate) fn path(&mut self, path: Path) {
        self.paths.push(path);
    }

    pub(crate) fn primitive(&mut self) {
        self.paths.extend(primitive_paths());
    }

    pub(crate) fn prelude(&mut self) {
        self.paths.extend(std_prelude_paths());
    }

    pub(crate) fn std_crates(&mut self) {
        self.paths.extend(std_crate_paths());
    }

    pub(crate) fn crate_root(&mut self) {
        self.paths.push(syn::parse_quote!(crate));
    }

    pub(crate) fn values(&mut self) {
        self.values = true;
    }

    pub(crate) fn defaults(&mut self) {
        self.defaults = true;
    }

    /// The allowed paths, which are context-free for the analysis of a Leaker.
    pub(crate) fn paths(&self) -> &[Path] {
        &self.paths
    }

    /// Returns `true` if the path `path` of an error is allowed.
    fn allows_path(&self, path: &Path) -> bool {
        self.paths
            .iter()
            .any(|allowed| path_starts_with(path, allowed))
    }

    /// Returns `true` if the rules accept `error`.
    pub(crate) fn accepts(&self, error: &ErrorItem) -> bool {
        self.causes.contains(&error.cause)
            || (self.values && error.in_value)
            || (self.defaults && error.in_default)
            || (!error.paths.is_empty() && error.paths.iter().all(|path| self.allows_path(path)))
    }

    /// Drops the errors that the rules accept, and puts their nested errors in their place
    /// ([`ErrorItem::nested`]): the part is copied as written, so what is wrong inside it is real
    /// whichever rule accepted it. The nested errors are filtered the same way, down to the last.
    /// An error that stays loses the paths that the rules allow, so that rules given one after the
    /// other add up as they do in one set (and in the serialization of the error, which carries its
    /// paths); its nested errors are filtered too, for when it is dropped later.
    ///
    /// A nested error that comes up is skipped if it is identical but for its spans to an error that
    /// stays, before it or after it ([`same()`]). Errors that did not come up, and errors that come
    /// up, are never merged with each other.
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

    /// Puts `error`, which came up, in `came_up` after `before` errors that stay if the rules do not
    /// accept it and it is not `same()` as one of `stays`, else its nested errors that they do not
    /// accept.
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

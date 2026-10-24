//! What the Leaker accepts of its errors: the one set of rules behind `allow*()` of both
//! [`LeakerConfig`](crate::LeakerConfig) and [`LeakerResult`](crate::LeakerResult).

use crate::error::{Cause, ErrorItem};
use crate::util::{path_starts_with, primitive_paths, std_crate_paths, std_prelude_paths};
use syn::Path;

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
    pub(crate) fn filter(&self, errors: &mut Vec<ErrorItem>) {
        let mut kept = Vec::new();
        for error in std::mem::take(errors) {
            self.keep(error, &mut kept);
        }
        *errors = kept;
    }

    /// Pushes `error` to `kept` if the rules do not accept it, else its nested errors that they do
    /// not accept.
    fn keep(&self, mut error: ErrorItem, kept: &mut Vec<ErrorItem>) {
        if self.accepts(&error) {
            for nested in error.nested {
                self.keep(nested, kept);
            }
        } else {
            error.paths.retain(|path| !self.allows_path(path));
            self.filter(&mut error.nested);
            kept.push(error);
        }
    }
}

//! What the analysis of the [`Leaker`] found: the [`CheckResult`] of a type, and the effects of
//! visiting it, which the [`Analysis`] turns into the graphs of the entries.

use crate::context::{Context, Contexts, SelfRequirement};
use crate::error::NotInternableError;
use crate::graph::{NodeKey, TypeGraph};
#[cfg(doc)]
use crate::leaker::Leaker;
#[cfg(doc)]
use crate::view::View;
use proc_macro2::TokenStream;
use std::collections::HashSet;
use syn::{Ident, Type};
use template_quote::quote;

/// What [`View::check()`] found in a type: causes that force it to be leaked, and causes that
/// forbid it.
///
/// - Both empty: copy the type as it is.
/// - Only `must`: leak the type, or only its parts unless `is_own` (`MyType`,
///   `::path::Wrap<MyType>`).
/// - Only `impossible`: the type cannot be leaked and need not be (`impl ::path::Trait`).
/// - Both: a conflict (`MyType<'_>`). Unless `is_own`, the parts can be leaked instead.
///
/// `Self` where it has no meaning (in a module, or in generics that no struct, enum, union, impl or
/// trait encloses) is an `impossible` cause, with the mention of `Self` as written and a `note`.
/// A scope fails on it, so it can be neither leaked nor copied. See [Self rules](crate#self-rules).
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct CheckResult {
    /// Causes that depend on the context (`A` and `B` in `::path::Wrap<A, B>`). Outermost first,
    /// then in order of appearance. In a conflict, a path in a type inside the type that is leaked
    /// alone is not listed: it is no error once the others are fixed (`u8` in `<Self as
    /// Tr<u8>>::X`, which cannot be leaked whole because of `Self`).
    pub must: Vec<TokenStream>,
    /// Causes that cannot be leaked (an elided lifetime, `impl Trait`), in order of appearance,
    /// then the mentions of `Self` that have no meaning.
    pub impossible: Vec<TokenStream>,
    /// `true` if the first of `must` is the type's own path (`MyType` in `MyType<'_>`), so the type
    /// must be leaked whole. `false` if `must` is empty or only its parts are causes
    /// (`::path::Wrap<MyType, '_>`). A bare identifier argument counts as a cause of the type
    /// holding it. See [Bare identifiers and consts](crate#bare-identifiers-and-consts).
    pub is_own: bool,
    /// `true` if the type is an optional entry: it has no cause of its own, and its own path names
    /// an external crate (`::path::X`, `::path::Wrap<(MyType,)>`), which the Referrer may not
    /// share. `must` holds only its parts, if any. A type with an impossible cause is not one, as
    /// it is copied as written. See [External crates](crate#external-crates).
    pub optional: bool,
    /// Explains the conflict if the causes do not (why `Self` cannot be leaked, or has no
    /// meaning). Reported first.
    pub note: Option<String>,
    /// The headline of the error, if the fault is not that the path of the type is relative, which
    /// the default `use absolute path: '..'` says. [`NotInternableError`] shows it as the message
    /// of the error. Set for a type that uses a generic whose bound mentions a `Self` that has no
    /// name here (see [Self rules](crate#self-rules)): the fault is that bound, `must` is the type
    /// that uses it, and `impossible` is the mention of `Self`. Also set for the other rules
    /// that are not about a path (the bound rules, `Self::A` where `Self` is a type), for a
    /// type in the default of a parameter that names `Self`, for a type that uses a generic whose
    /// bound mentions a `Self` that cannot be leaked, and for a lifetime of a method that the copy
    /// would leave unconstrained (E0195): the fault is the lifetime. A relative path that cannot
    /// be leaked as a whole (`MyType<'_>`, E0106) has none: the fault is the path, and `note` says
    /// why.
    pub headline: Option<String>,
}

pub(crate) struct PendingType {
    pub(crate) ty: Type,
    pub(crate) is_must_intern: bool,
    /// A candidate for the graph of the required entries: it or a part is a cause.
    pub(crate) in_graph: bool,
    /// An optional entry if the Referrer replaces it: its own path names an external crate.
    pub(crate) optional: bool,
    pub(crate) has_optional: bool,
    pub(crate) context: Context,
    /// A bare identifier argument that may be a const. Its parent is leaked whole if possible.
    pub(crate) ambiguous: bool,
    /// It cannot be an entry, but its components can: the copy would be cyclic. Like
    /// `no_optional_entry` for the optional entries.
    pub(crate) no_entry: bool,
    pub(crate) no_optional_entry: bool,
    pub(crate) components: Vec<Effect>,
}

impl PendingType {
    pub(crate) fn new(ty: &Type, context: Context, components: Vec<Effect>) -> Self {
        Self {
            ty: ty.clone(),
            is_must_intern: false,
            in_graph: false,
            optional: false,
            has_optional: false,
            context,
            ambiguous: false,
            no_entry: false,
            no_optional_entry: false,
            components,
        }
    }
}

pub(crate) enum Effect {
    Type(Box<PendingType>),
    Error(CheckResult),
    Primitive(Ident),
}

#[derive(Default)]
pub(crate) struct Analysis {
    pub(crate) types: TypeGraph,
    /// In the contexts of `Leaker::optional_contexts`.
    pub(crate) optional: TypeGraph,
    pub(crate) primitives: Vec<Ident>,
}

impl Analysis {
    pub(crate) fn apply(
        &mut self,
        contexts: &mut Contexts,
        effects: &[Effect],
        parent: Option<&NodeKey>,
    ) {
        for effect in effects {
            match effect {
                Effect::Type(pending) if !pending.in_graph => {
                    // Never leaked: not added, but the assumptions of its components stay.
                    self.apply(contexts, &pending.components, parent);
                }
                Effect::Type(pending) => {
                    let context = contexts.intern(pending.context.clone());
                    let key = self
                        .types
                        .add(context, &pending.ty, parent, pending.is_must_intern);
                    if pending.no_entry {
                        self.types.forbid(&key);
                    }
                    self.apply(contexts, &pending.components, Some(&key));
                }
                Effect::Error(_) => (),
                Effect::Primitive(ident) => self.primitives.push(ident.clone()),
            }
        }
    }

    /// Like [`Analysis::apply()`] for the optional entries, which have contexts of their own. Types
    /// leaked whole by their own cause are required, so they and their parts are not here.
    pub(crate) fn apply_optional(
        &mut self,
        contexts: &mut Contexts,
        effects: &[Effect],
        parent: Option<&NodeKey>,
    ) {
        for effect in effects {
            if let Effect::Type(pending) = effect {
                if !pending.has_optional {
                    continue;
                }
                let context = contexts.intern(pending.context.clone());
                let key = self
                    .optional
                    .add(context, &pending.ty, parent, pending.optional);
                if pending.no_optional_entry {
                    self.optional.forbid(&key);
                }
                self.apply_optional(contexts, &pending.components, Some(&key));
            }
        }
    }
}

/// Returns the error of a scope: the conflicts among `effects` and `conflicts`, or else the
/// mentions of `Self` without meaning.
pub(crate) fn scope_error(
    effects: &[Effect],
    self_errors: &[TokenStream],
    conflicts: &[CheckResult],
) -> Option<NotInternableError> {
    let mut errors = Vec::new();
    collect_errors(effects, &mut errors);
    errors.extend(conflicts.iter().cloned());
    if !errors.is_empty() {
        Some(NotInternableError::RelativePath(errors))
    } else if !self_errors.is_empty() {
        Some(NotInternableError::SelfWithoutMeaning(self_errors.to_vec()))
    } else {
        None
    }
}

fn collect_errors(effects: &[Effect], errors: &mut Vec<CheckResult>) {
    for effect in effects {
        match effect {
            Effect::Error(error) => errors.push(error.clone()),
            Effect::Type(pending) => collect_errors(&pending.components, errors),
            Effect::Primitive(_) => (),
        }
    }
}

/// Returns `true` if a type that may become an entry names `Self`, or has bounds rewritten for the
/// `Self` that `declared` is, so the Referrer needs to know what `Self` is to find it.
pub(crate) fn names_self(effects: &[Effect], declared: &SelfRequirement) -> bool {
    effects.iter().any(|effect| match effect {
        Effect::Type(pending) => {
            (pending.in_graph
                && (pending.context.self_ty().is_some() || pending.context.substitutes(declared)))
                || names_self(&pending.components, declared)
        }
        _ => false,
    })
}

/// Makes the types in `keys` (with their contexts) no optional entries.
pub(crate) fn drop_optional(effects: &mut [Effect], keys: &HashSet<(Context, String)>) {
    for effect in effects {
        if let Effect::Type(pending) = effect {
            if pending.optional {
                let ty = &pending.ty;
                if keys.contains(&(pending.context.clone(), quote!(#ty).to_string())) {
                    pending.optional = false;
                }
            }
            drop_optional(&mut pending.components, keys);
        }
    }
}

pub(crate) fn has_optional(effects: &[Effect]) -> bool {
    effects
        .iter()
        .any(|effect| matches!(effect, Effect::Type(pending) if pending.has_optional))
}

pub(crate) fn has_ambiguous(effects: &[Effect]) -> bool {
    effects.iter().any(|effect| match effect {
        Effect::Type(pending) => {
            pending.ambiguous
                || (matches!(pending.ty, Type::Group(_)) && has_ambiguous(&pending.components))
        }
        _ => false,
    })
}

pub(crate) fn primitive_effects(effects: Vec<Effect>) -> Vec<Effect> {
    let mut found = Vec::new();
    for effect in effects {
        match effect {
            Effect::Primitive(_) => found.push(effect),
            Effect::Type(pending) => found.extend(primitive_effects(pending.components)),
            Effect::Error(_) => (),
        }
    }
    found
}

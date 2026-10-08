//! What the analysis of the [`Leaker`] found: the [`CheckResult`] of a type, and the effects of
//! visiting it, which the [`Analysis`] turns into the graphs of the entries.

use crate::context::{Context, Contexts};
use crate::error::NotInternableError;
use crate::graph::{NodeKey, TypeGraph};
#[cfg(doc)]
use crate::leaker::Leaker;
use proc_macro2::TokenStream;
use syn::{Ident, Type};

/// What [`Leaker::check()`] found in a type: causes that force it to be leaked, and causes that
/// forbid it.
///
/// - Both empty: copy the type as it is.
/// - Only `must`: leak the type, or only its parts unless `is_own` (`MyType`,
///   `::path::Wrap<MyType>`).
/// - Only `impossible`: the type cannot be leaked and need not be (`impl ::path::Trait`).
/// - Both: a conflict (`MyType<'_>`). Unless `is_own`, the parts can be leaked instead.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct CheckResult {
    /// Causes that depend on the context (`A` and `B` in `::path::Wrap<A, B>`). Outermost first,
    /// then in order of appearance.
    pub must: Vec<TokenStream>,
    /// Causes that cannot be leaked (an elided lifetime, `impl Trait`), in order of appearance.
    pub impossible: Vec<TokenStream>,
    /// `true` if the first of `must` is the type's own path (`MyType` in `MyType<'_>`), so the type
    /// must be leaked whole. `false` if `must` is empty or only its parts are causes
    /// (`::path::Wrap<MyType, '_>`). A bare identifier argument counts as a cause of the type
    /// holding it. See [Bare identifiers and consts](crate#bare-identifiers-and-consts).
    pub is_own: bool,
    /// Explains the conflict if the causes do not (why `Self` cannot be leaked). Reported first.
    pub note: Option<String>,
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
                self.apply_optional(contexts, &pending.components, Some(&key));
            }
        }
    }
}

/// Returns the error of a scope: the conflicts among `effects`, or else the mentions of `Self`
/// without meaning.
pub(crate) fn scope_error(
    effects: &[Effect],
    self_errors: &[TokenStream],
) -> Option<NotInternableError> {
    let mut errors = Vec::new();
    collect_errors(effects, &mut errors);
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

/// Returns `true` if a type that may become an entry names `Self`, so the Referrer needs to know
/// what `Self` is to find it.
pub(crate) fn names_self(effects: &[Effect]) -> bool {
    effects.iter().any(|effect| match effect {
        Effect::Type(pending) => {
            (pending.in_graph && pending.context.self_ty().is_some())
                || names_self(&pending.components)
        }
        _ => false,
    })
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

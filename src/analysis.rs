//! What the analysis of the [`Leaker`] found: the [`CheckResult`] of a type, and the effects of
//! visiting it, which the [`Analysis`] turns into the graphs of the entries.

use crate::context::{Context, ContextId, Contexts, SelfRequirement};
use crate::error::{report, self_without_meaning, Cause, ErrorItem, Reason};
use crate::graph::{NodeKey, TypeGraph};
#[cfg(doc)]
use crate::leaker::Leaker;
#[cfg(doc)]
use crate::view::View;
use std::collections::HashSet;
use syn::{Ident, Type};
use template_quote::quote;

/// What [`View::check()`] found in a type: the causes that force it to be leaked, and the causes
/// that forbid it.
///
/// The causes are, for example, a relative path (`must`) and an elided lifetime (`impossible`).
/// Each is a [`Cause`] of the kind of syntax it is. The first of `must` is the cause of the
/// [`ErrorItem`] that reports the conflict.
///
/// - Both empty: copy the type as it is.
/// - Only `must`: leak the type, or only its parts unless `is_own` (`MyType`,
///   `::path::Wrap<MyType>`).
/// - Only `impossible`: the type cannot be leaked and need not be (`impl ::path::Trait`).
/// - Both: a conflict (`MyType<'_>`). Unless `is_own`, the parts can be leaked instead.
///
/// `Self` where it has no meaning (in a module, or in generics that no struct, enum, union, impl or
/// trait encloses) is an `impossible` cause, with the mention of `Self` as written, a `note` and a
/// `help`.
/// A scope reports it as an error ([`Reason::SelfWithoutMeaning`]) and copies the type that names
/// it as written, so it is no entry.
///
/// ```
/// # use type_leak::*;
/// # use syn::*;
/// let mut leaker = Leaker::new();
/// let (_, results) = leaker
///     .enter_generics(&Generics::default(), |view| {
///         [
///             view.check(&parse_quote!(::core::option::Option<::core::primitive::u8>)),
///             view.check(&parse_quote!(Rel)),            // a relative path
///             view.check(&parse_quote!(impl ::path::Trait)), // cannot be leaked
///             view.check(&parse_quote!(Rel<'_>)),        // must be leaked, and cannot be
///         ]
///     });
/// // Nothing depends on the context in `Option<u8>` of the standard library: copy it as it is.
/// assert!(results[0].must.is_empty() && results[0].impossible.is_empty());
/// // Leak `Rel`: `is_own` says that it is the path of the type itself.
/// assert_eq!(results[1].must.len(), 1);
/// assert!(results[1].is_own);
/// // `impl Trait` can be neither leaked nor needs to be.
/// assert!(results[2].must.is_empty() && !results[2].impossible.is_empty());
/// // `Rel<'_>` is a conflict: its path depends on the context, and its lifetime cannot be leaked.
/// assert!(!results[3].must.is_empty() && !results[3].impossible.is_empty());
/// assert!(matches!(&results[3].must[0], Cause::Type(_)));
/// assert!(matches!(&results[3].impossible[0], Cause::Lifetime(_)));
/// ```
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct CheckResult {
    /// Causes that depend on the context (`A` and `B` in `::path::Wrap<A, B>`). Outermost first,
    /// then in order of appearance. In a conflict, a path in a type inside the type that is leaked
    /// alone is not listed: it is no error once the others are fixed (`u8` in `<Self as
    /// Tr<u8>>::X`, which cannot be leaked whole because of `Self`).
    pub must: Vec<Cause>,
    /// Causes that cannot be leaked (an elided lifetime, `impl Trait`), in order of appearance,
    /// then the mentions of `Self` that have no meaning.
    pub impossible: Vec<Cause>,
    /// `true` if the first of `must` is the type's own path (`MyType` in `MyType<'_>`), so the type
    /// must be leaked whole. `false` if `must` is empty or only its parts are causes
    /// (`::path::Wrap<MyType, '_>`). A bare identifier argument counts as a cause of the type
    /// holding it, and a conflict if that type cannot be leaked (`::path::Wrap<MyType, Self>` where
    /// `Self` is denied). See [Bare identifiers and
    /// consts](crate::Leaker#bare-identifiers-and-consts).
    pub is_own: bool,
    /// `true` if the type is an optional entry: it has no cause of its own, and its own path names
    /// an external crate (`::path::X`, `::path::Wrap<(MyType,)>`), which the Referrer may not
    /// share. `must` holds only its parts, if any. A type with an impossible cause is not one, as
    /// it is copied as written. See [`Leaker::allowed_paths`](crate::Leaker::allowed_paths).
    pub optional: bool,
    /// Explains the conflict if the causes do not (why `Self` cannot be leaked, or has no
    /// meaning). Reported first. It says why, and `help` says what to write instead.
    pub note: Option<String>,
    /// What to write instead, with the tokens as written (for example "replace `Self::A` with
    /// `<Self as ::path::Tr>::A`"). If it is `None`, the [`ErrorItem`] suggests the absolute path
    /// of the first of `must`. See [`ErrorItem::help`](crate::ErrorItem::help).
    pub help: Option<String>,
    /// The headline of the error, if the fault is not that the path of the type is relative. It is
    /// the message of the [`ErrorItem`]; if it is `None`, the message is `use absolute path: '..'`
    /// and `note` says why the path cannot be leaked as a whole (`MyType<'_>`, E0106).
    ///
    /// It is set when the fault is something else than a path:
    ///
    /// - a type that uses a generic whose bound mentions a `Self` that has no name here (see
    ///   [`View::visit_item()`](crate::View::visit_item)): the fault is that bound, `must` is the
    ///   type that uses it, and `impossible` is the mention of `Self`;
    /// - a type that uses a generic whose bound mentions a `Self` that cannot be leaked;
    /// - a type in the default of a parameter that names `Self`;
    /// - the bound rules, and `Self::A` where `Self` is a type;
    /// - a lifetime of a method that the copy would leave unconstrained (E0195): the fault is the
    ///   lifetime;
    /// - syntax in a value that the Leaker cannot read, with
    ///   [`LeakerConfig::check_values()`](crate::LeakerConfig::check_values) ("cannot analyze .. in
    ///   the body of ..").
    pub headline: Option<String>,
    /// The cause of the [`ErrorItem`] if it is not the first of `must` (the default of a parameter
    /// that names `Self` is reported whole, with the note at the `Self` in it).
    pub(crate) cause: Option<Cause>,
    /// The [`Reason`] that goes with `headline` (`RelativePath` if there is none).
    pub(crate) reason: Reason,
    /// The part that the conflict leaves as written, if the Referrer has to know it.
    pub(crate) part: Option<Part>,
    /// The number of mentions of `Self` without meaning found when this conflict was recorded, to
    /// report them among the conflicts in the order found ([`scope_error()`]).
    pub(crate) seen: usize,
}

/// A part of the item that the Leaker copies as written, whole, which the Referrer must leave so.
/// It becomes a [`PartItem`] when its effect is applied.
#[derive(Clone, Debug)]
pub(crate) struct Part {
    /// The part: a [`Cause::Type`], the cause of a bound, or the path of a value.
    pub(crate) cause: Cause,
    /// The context that an entry of it would have.
    pub(crate) context: Context,
    /// In a bound or where predicate of the item (`AnalyzeVisitor::env_depth > 0`).
    pub(crate) in_predicate: bool,
    /// The innermost type around it that owns the cause of the conflict, with its context
    /// ([`anchor_parts()`]); the part is left as written only inside it.
    pub(crate) within: Option<(Type, Context)>,
}

/// A part that the Referrer leaves as written, with its contexts interned. Serialized as `KIND (
/// CAUSE ) CONTEXT [predicate] [in ( TYPE ) CONTEXT]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PartItem {
    /// A [`Cause::Type`], the cause of a bound, or the path of a value.
    pub(crate) part: Cause,
    /// The context that an entry of it would have.
    pub(crate) context: ContextId,
    /// In a bound or where predicate of the item.
    pub(crate) in_predicate: bool,
    /// The type that owns the cause of the conflict, with its context: the part is left as written
    /// only inside it.
    pub(crate) within: Option<(Type, ContextId)>,
}

pub(crate) struct PendingType {
    pub(crate) ty: Type,
    /// The address of the type in the visited syntax, which `ty` is a copy of.
    pub(crate) origin: usize,
    pub(crate) is_must_intern: bool,
    /// A candidate for the graph of the required entries: it or a part is a cause.
    pub(crate) in_graph: bool,
    /// An optional entry if the Referrer replaces it: its own path names an external crate.
    pub(crate) optional: bool,
    pub(crate) has_optional: bool,
    pub(crate) context: Context,
    /// A bare identifier argument that may be a const. Its parent is leaked whole if possible; if
    /// it cannot be because of a hard cause (see `Impossibles`), it is a conflict.
    pub(crate) ambiguous: bool,
    /// It cannot be an entry, but its components can: the copy would be cyclic. Like
    /// `no_optional_entry` for the optional entries.
    pub(crate) no_entry: bool,
    pub(crate) no_optional_entry: bool,
    /// The type names a `Self` without meaning, or a bound of its context does
    /// ([`correct_unmeant_self()`]).
    pub(crate) unmeant_self: bool,
    /// The type is in a bound or where predicate of the item.
    pub(crate) in_predicate: bool,
    pub(crate) components: Vec<Effect>,
}

impl PendingType {
    pub(crate) fn new(ty: &Type, context: Context, components: Vec<Effect>) -> Self {
        Self {
            ty: ty.clone(),
            origin: ty as *const Type as usize,
            is_must_intern: false,
            in_graph: false,
            optional: false,
            has_optional: false,
            context,
            ambiguous: false,
            no_entry: false,
            no_optional_entry: false,
            unmeant_self: false,
            in_predicate: false,
            components,
        }
    }

    /// The part that the Referrer leaves as written if the type is copied so instead of leaked.
    pub(crate) fn part(&self) -> Part {
        Part {
            cause: Cause::Type(self.ty.clone()),
            context: self.context.clone(),
            in_predicate: self.in_predicate,
            within: None,
        }
    }
}

impl CheckResult {
    /// The correction of a part copied as written, whole, for a fault that another error tells. It
    /// has no cause that depends on the context, so `report()` gives no error for it.
    pub(crate) fn corrected() -> Self {
        Self {
            must: Vec::new(),
            impossible: Vec::new(),
            is_own: true,
            optional: false,
            note: None,
            help: None,
            headline: None,
            cause: None,
            reason: Reason::RelativePath,
            part: None,
            seen: 0,
        }
    }
}

pub(crate) enum Effect {
    Type(Box<PendingType>),
    Error(Box<CheckResult>),
    Primitive(Ident),
}

#[derive(Default)]
pub(crate) struct Analysis {
    pub(crate) types: TypeGraph,
    /// In the contexts of `Leaker::optional_contexts`.
    pub(crate) optional: TypeGraph,
    pub(crate) primitives: Vec<Ident>,
    /// The parts that the Referrer leaves as written, in order and distinct.
    pub(crate) parts: Vec<PartItem>,
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
                Effect::Error(error) => {
                    // The contexts are interned as an entry's is.
                    if let Some(part) = &error.part {
                        let item = PartItem {
                            part: part.cause.clone(),
                            context: contexts.intern(part.context.clone()),
                            in_predicate: part.in_predicate,
                            within: part.within.as_ref().map(|(ty, context)| {
                                (ty.clone(), contexts.intern(context.clone()))
                            }),
                        };
                        if !self.parts.contains(&item) {
                            self.parts.push(item);
                        }
                    }
                }
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

/// Returns the errors of an analysis in the order found: the conflicts among `effects`, then the
/// cycles in `conflicts`, with the mentions of `Self` without meaning placed by
/// [`CheckResult::seen`].
pub(crate) fn scope_error(
    effects: &[Effect],
    self_errors: &[Cause],
    conflicts: &[CheckResult],
) -> Vec<ErrorItem> {
    let mut errors = Vec::new();
    collect_errors(effects, &mut errors);
    errors.extend(conflicts.iter().cloned());
    // The mentions, with the number of the mentions before each.
    let mut mentions = self_errors
        .iter()
        .enumerate()
        .map(|(index, cause)| (index, self_without_meaning(cause)))
        .peekable();
    let mut items: Vec<ErrorItem> = Vec::new();
    // A conflict without a cause that depends on the context is no error: `report()` gives none.
    for (conflict, item) in errors
        .iter()
        .filter_map(|conflict| Some((conflict, report(conflict)?)))
    {
        while let Some((_, mention)) = mentions.next_if(|(index, _)| *index < conflict.seen) {
            items.push(mention);
        }
        items.push(item);
    }
    items.extend(mentions.map(|(_, mention)| mention));
    items
}

/// Counts `by` more mentions of `Self` without meaning before the conflicts in `effects` (see
/// [`CheckResult::seen`]).
pub(crate) fn shift_seen(effects: &mut [Effect], by: usize) {
    for effect in effects {
        match effect {
            Effect::Error(error) => error.seen += by,
            Effect::Type(pending) => shift_seen(&mut pending.components, by),
            Effect::Primitive(_) => (),
        }
    }
}

/// Copies as written, whole, the types among `effects` (outermost first) that `corrected` picks:
/// each becomes an `Effect::Error` that reports nothing and holds the type as its part. The
/// conflicts and the primitive names in it stay.
pub(crate) fn correct_effects(
    effects: Vec<Effect>,
    corrected: &dyn Fn(&PendingType) -> bool,
) -> Vec<Effect> {
    let mut found = Vec::new();
    for effect in effects {
        match effect {
            Effect::Type(pending) if corrected(&pending) => {
                let mut error = CheckResult::corrected();
                error.part = Some(pending.part());
                found.push(Effect::Error(Box::new(error)));
                let PendingType { components, .. } = *pending;
                keep_conflicts(components, &mut found);
            }
            Effect::Type(mut pending) => {
                let components = std::mem::take(&mut pending.components);
                pending.components = correct_effects(components, corrected);
                found.push(Effect::Type(pending));
            }
            other => found.push(other),
        }
    }
    found
}

/// Returns `true` if an error is among `effects` or in the parts of the types in it.
pub(crate) fn has_error(effects: &[Effect]) -> bool {
    effects.iter().any(|effect| match effect {
        Effect::Error(_) => true,
        Effect::Type(pending) => has_error(&pending.components),
        Effect::Primitive(_) => false,
    })
}

/// Gives the parts among `effects` that have no anchor yet `ty` in `context` as their anchor: the
/// innermost type that owns the cause wins, as types visited first are inside it.
pub(crate) fn anchor_parts(effects: &mut [Effect], ty: &Type, context: &Context) {
    for effect in effects {
        match effect {
            Effect::Error(error) => {
                if let Some(part) = error.part.as_mut().filter(|part| part.within.is_none()) {
                    part.within = Some((ty.clone(), context.clone()));
                }
            }
            Effect::Type(pending) => anchor_parts(&mut pending.components, ty, context),
            Effect::Primitive(_) => (),
        }
    }
}

/// Pushes the errors and the primitive names among `effects`; the errors lose their parts.
fn keep_conflicts(effects: Vec<Effect>, found: &mut Vec<Effect>) {
    for effect in effects {
        match effect {
            Effect::Type(pending) => keep_conflicts(pending.components, found),
            Effect::Error(mut error) => {
                error.part = None;
                found.push(Effect::Error(error));
            }
            other => found.push(other),
        }
    }
}

/// Copies as written, whole, the outermost type that would be an entry and names a `Self` without
/// meaning, or has a bound in its context that does: the impl of the Repeater would read that
/// `Self` as its implementor ([`Reason::SelfWithoutMeaning`]).
pub(crate) fn correct_unmeant_self(effects: Vec<Effect>) -> Vec<Effect> {
    correct_effects(effects, &|pending| {
        pending.unmeant_self && (pending.in_graph || pending.has_optional)
    })
}

fn collect_errors(effects: &[Effect], errors: &mut Vec<CheckResult>) {
    for effect in effects {
        match effect {
            Effect::Error(error) => errors.push((**error).clone()),
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

/// Returns the bare identifiers among `effects` that may be consts: those of `has_ambiguous()`.
pub(crate) fn ambiguous_types(effects: &[Effect]) -> Vec<Type> {
    let mut found = Vec::new();
    for effect in effects {
        if let Effect::Type(pending) = effect {
            if pending.ambiguous {
                found.push(pending.ty.clone());
            } else if matches!(pending.ty, Type::Group(_)) {
                found.extend(ambiguous_types(&pending.components));
            }
        }
    }
    found
}

/// Returns the errors among `effects`, without their parts: the caller copies the type that holds
/// them as written.
pub(crate) fn error_effects(effects: Vec<Effect>) -> Vec<Effect> {
    let mut found = Vec::new();
    for effect in effects {
        match effect {
            Effect::Error(mut error) => {
                error.part = None;
                found.push(Effect::Error(error));
            }
            Effect::Type(pending) => found.extend(error_effects(pending.components)),
            Effect::Primitive(_) => (),
        }
    }
    found
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

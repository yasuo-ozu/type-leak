//! What the analysis of the [`Leaker`] found: the [`CheckResult`] of a type, and the effects of
//! visiting it, which the [`Analysis`] turns into the graph of the entries.

use crate::analyze::{
    input_type, unconstrained_in_copy, unconstrained_in_output, Anchoring, Elision,
};
use crate::context::{Context, ContextId, Contexts, SelfRequirement};
use crate::error::{report, self_without_meaning, Cause, ErrorItem, Reason};
use crate::graph::{is_whole_reference, NodeKey, TypeGraph};
#[cfg(doc)]
use crate::leaker::Leaker;
use crate::result::Place;
#[cfg(doc)]
use crate::view::View;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use syn::{FnArg, Ident, Lifetime, Signature, Type};
use template_quote::quote;

/// What [`View::check()`] found in a type: the causes that force it to be leaked, and the causes
/// that forbid it.
///
/// Each is a [`Cause`] of the kind of syntax it is. The first of `must` is the cause of the
/// [`ErrorItem`] that reports a conflict.
///
/// - Both empty: copy the type as it is.
/// - Only `must`: leak the type, or only its parts unless `is_own` (`MyType`,
///   `::path::Wrap<MyType>`).
/// - Only `impossible`: the type cannot be leaked and need not be (`impl ::path::Trait`).
/// - Both: a conflict (`MyType<'_>`). Unless `is_own`, the parts can be leaked instead.
///
/// `Self` where it has no meaning is an `impossible` cause ([`Reason::SelfWithoutMeaning`]): a
/// scope reports it as an error and copies the type that names it as written.
///
/// ```
/// # use type_leak::*;
/// # use syn::*;
/// let mut leaker = Leaker::new();
/// let (_, (rel, conflict)) = leaker.enter_generics(&Generics::default(), |view| {
///     (view.check(&parse_quote!(Rel)), view.check(&parse_quote!(Rel<'_>)))
/// });
/// // Leak `Rel`, the path of the type itself.
/// assert!(rel.is_own && rel.must.len() == 1 && rel.impossible.is_empty());
/// // `Rel<'_>` is a conflict: its path depends on the context, and its lifetime cannot be leaked.
/// assert!(matches!(&conflict.must[0], Cause::Type(_)));
/// assert!(matches!(&conflict.impossible[0], Cause::Lifetime(_)));
/// ```
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct CheckResult {
    /// Causes that depend on the context (`A` and `B` in `::path::Wrap<A, B>`): outermost first,
    /// then in order of appearance. In a conflict, paths inside a type that is leaked alone are not
    /// listed.
    pub must: Vec<Cause>,
    /// Causes that cannot be leaked (an elided lifetime, `impl Trait`), in order of appearance,
    /// then the mentions of `Self` that have no meaning.
    pub impossible: Vec<Cause>,
    /// `true` if the type is leaked whole, so a conflict cannot be solved by leaking its parts
    /// (`MyType<'_>`; `::ext::W<Rel>` has `must` `[Rel]` and no conflict); `false` if `must` is
    /// empty or only the parts are leaked (`::path::Wrap<MyType, '_>`). A bare identifier argument
    /// counts as a cause of the type holding it ([Bare identifiers and
    /// consts](crate::Leaker#bare-identifiers-and-consts)).
    pub is_own: bool,
    /// `true` if the type is an optional entry: it has no cause of its own, and its own path names
    /// another crate (`::path::X`). It is an entry unless a rule allows the path
    /// ([`Allow`](crate::Allow)). `must` holds only its parts.
    pub optional: bool,
    /// Explains the conflict if the causes do not (why `Self` cannot be leaked, or has no meaning).
    /// Reported first.
    pub note: Option<String>,
    /// What to write instead, as text. If it is `None`, the [`ErrorItem`] suggests the absolute
    /// path of the first of `must` ([`ErrorItem::help`](crate::ErrorItem::help)).
    pub help: Option<String>,
    /// The message of the [`ErrorItem`] if the fault is not that the path of the type is relative.
    /// If it is `None`, the message is `use absolute path: '..'` and `note` says why the path
    /// cannot be leaked as a whole (`MyType<'_>`, E0106).
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
    /// Where the conflict was found ([`ErrorItem::place`]). Set where it is recorded as an error.
    pub(crate) place: Place,
    /// The relative paths that the copy writes as they are, with their kinds and without their
    /// arguments: the heads of the type that conflicts ([`ErrorItem::paths`]).
    pub(crate) paths: Vec<Cause>,
    /// Which conflict this is, inside which, and the entry it is about, so that
    /// [`LeakerResult::ignore()`](crate::LeakerResult::ignore) finds them in the result.
    pub(crate) origin: Origin,
}

/// What links a conflict to the others and to the Referrer.
#[derive(Clone, Debug, Default)]
pub(crate) struct Origin {
    /// Names the conflict: its clones share it, and no other conflict has it. Only the address is
    /// compared.
    pub(crate) token: Rc<()>,
    /// The `token` of the conflict that holds this one inside its part: the innermost one.
    pub(crate) outer: Option<Rc<()>>,
    /// The entry that the conflict is about, as its context and type: a risk of an entry, which is
    /// one all the same ([`Reason::NestedSelfInBounds`]).
    pub(crate) entry: Option<(Context, Type)>,
}

/// A mention of `Self` that has no meaning where it is written ([`Reason::SelfWithoutMeaning`]).
#[derive(Clone, Debug)]
pub(crate) struct SelfMention {
    /// The path or type that names `Self`.
    pub(crate) cause: Cause,
    /// Where it is written.
    pub(crate) place: Place,
    /// The `token` of the conflict that holds it inside its part ([`Origin::outer`]).
    pub(crate) outer: Option<Rc<()>>,
}

impl SelfMention {
    pub(crate) fn new(cause: Cause, place: Place) -> Self {
        Self {
            cause,
            place,
            outer: None,
        }
    }
}

/// A type that the Leaker copies as written, which the Referrer must not look up as an entry. It
/// becomes a [`PartItem`] when its effect is applied, if it is [`recorded()`](Part::recorded), or
/// when an error is about it.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Part {
    /// The type, as a [`Cause::Type`], or the bound whose trait path conflicts (it is no type, and
    /// whole).
    pub(crate) cause: Cause,
    /// Nothing in it is leaked (`Wrap<N>` with a bare `N` that may be a const, a reference, a type
    /// in a predicate of the item, a type that holds a trait object that takes its lifetime from
    /// it). Else only its own path is written as it is, and the analysis went on inside it.
    pub(crate) whole: bool,
    /// The context that an entry of it would have, or `None` if it cannot be written here (a
    /// bound names a `Self` that has no name: [`Reason::UnnameableSelf`]).
    pub(crate) context: Option<Context>,
    /// In a bound or where predicate of the item (`Position::env_depth > 0`).
    pub(crate) in_predicate: bool,
    /// The innermost type around it that owns the cause of the conflict, with its context
    /// ([`anchor_parts()`]); the part is left as written only inside it.
    pub(crate) within: Option<(Type, Context)>,
    /// The signature that made it a part, if one did.
    pub(crate) mark: Option<SignatureMark>,
    /// The primitive names in a whole part, which the copy writes as they are
    /// ([`Leaker::assertions()`]).
    pub(crate) primitives: Vec<Ident>,
}

impl Part {
    /// The part as the Referrer carries it, with its contexts interned.
    pub(crate) fn item(&self, contexts: &mut Contexts) -> PartItem {
        PartItem {
            part: self.cause.clone(),
            whole: self.whole,
            context: self
                .context
                .as_ref()
                .map(|context| contexts.intern(context.clone())),
            in_predicate: self.in_predicate,
            within: self
                .within
                .as_ref()
                .map(|(ty, context)| (ty.clone(), contexts.intern(context.clone()))),
            mark: self.mark.clone(),
        }
    }

    /// Returns `true` if the Referrer needs to know the part. A part that is not whole, with no
    /// anchor and no mark, has the same conflict wherever its tokens and context are found, so no
    /// entry has them and the visitor finds nothing to replace at its head; the types in it are
    /// entries as any are. A whole part (nothing in it may be replaced), or one that only a position
    /// tells from an entry (an anchor, a signature), is needed.
    pub(crate) fn recorded(&self) -> bool {
        self.whole || self.within.is_some() || self.mark.is_some()
    }
}

/// A part that the Referrer leaves as written, with its contexts interned (see
/// [`Part::recorded()`]). Serialized as `KIND (
/// CAUSE ) CONTEXT [whole] [sig ( MARK )] [predicate] [in ( TYPE ) CONTEXT]`, with `unwritable` for a
/// `CONTEXT` of `None`.
///
/// A part that is not whole, has no anchor and no mark is there only because an error is about
/// it ([`LeakerResult::ignore()`](crate::LeakerResult::ignore) makes it whole): the visitor does
/// not look at it before.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PartItem {
    /// A [`Cause::Type`].
    pub(crate) part: Cause,
    /// No entry is looked up in it. Else only the type itself is not looked up, and the types in
    /// it are.
    pub(crate) whole: bool,
    /// The context that an entry of it would have. `None`: it cannot be written, so the part is
    /// the one at a place where the Referrer cannot write its context either.
    pub(crate) context: Option<ContextId>,
    /// In a bound or where predicate of the item.
    pub(crate) in_predicate: bool,
    /// The type that owns the cause of the conflict, with its context: the part is left as written
    /// only inside it.
    pub(crate) within: Option<(Type, ContextId)>,
    /// The part is left as written only in the arguments of a signature that this covers.
    pub(crate) mark: Option<SignatureMark>,
}

/// What a signature must have for a part to be copied as written in it: the fault that the
/// signature gives the type (see [`SignatureFacts`]). Serialized as `sig ( 'x , .. )` and `sig (
/// elided )`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum SignatureMark {
    /// The method lifetimes of the type that the output names and no bound does (E0195): the copy
    /// would leave one of them unconstrained, because only types leaked whole hold it in the inputs
    /// ([`unconstrained_in_copy()`]).
    Unconstrained(Vec<Lifetime>),
    /// The output relies on elision, and the type adds lifetimes to the inputs (E0106).
    Elided,
}

/// The facts about a signature that a [`SignatureMark`] is about. It is computed from the
/// signature alone, by the Leaker where it makes a part and by the Referrer at each signature it
/// visits; the Referrer also tells which of the lifetimes the copy leaves unconstrained
/// ([`SignatureFacts::in_copy()`]).
#[derive(Debug)]
pub(crate) struct SignatureFacts {
    /// The method lifetimes that the output names and no bound does.
    unconstrained: Vec<Lifetime>,
    /// Of `unconstrained`, those that the copy leaves unconstrained: no input keeps them as
    /// written once the leaked types are replaced. `None` where it is not known (the Leaker, which
    /// decides it by its own check).
    in_copy: Option<Vec<Lifetime>>,
    /// What the output keeps by elision, if it relies on it.
    elision: Option<Elision>,
    /// The facts of a probe ([`SignatureFacts::inputs_probe()`]).
    probing: bool,
}

impl SignatureFacts {
    pub(crate) fn of(sig: &Signature) -> Self {
        Self {
            unconstrained: unconstrained_in_output(sig),
            in_copy: None,
            elision: Elision::of_signature(sig),
            probing: false,
        }
    }

    /// The facts for the probe of the inputs of a signature, which replaces the leaked types to see
    /// what is left of them ([`SignatureFacts::in_copy()`]): no part that a signature made is left as
    /// written ([`SignatureFacts::covers()`] is `false`), and the visitor takes each for a leaked
    /// type, which it is where the Leaker made it a part.
    pub(crate) fn inputs_probe() -> Self {
        Self {
            unconstrained: Vec::new(),
            in_copy: None,
            elision: None,
            probing: true,
        }
    }

    /// Returns `true` for the facts of a probe ([`SignatureFacts::inputs_probe()`]).
    pub(crate) fn probing(&self) -> bool {
        self.probing
    }

    /// The facts for the types in `input`, an argument of the signature: elision counts the
    /// lifetimes of each apart ([`Elision::for_input()`]).
    pub(crate) fn for_input(&self, input: &FnArg) -> Self {
        Self {
            unconstrained: self.unconstrained.clone(),
            in_copy: self.in_copy.clone(),
            elision: self
                .elision
                .as_ref()
                .map(|elision| elision.for_input(input_type(input))),
            probing: self.probing,
        }
    }

    /// The facts with the lifetimes that the copy leaves unconstrained, from `copy`: the inputs of
    /// the signature with the leaked types replaced ([`unconstrained_in_copy()`]). A lifetime that
    /// an input keeps as written is not one, whatever other types hold it.
    pub(crate) fn in_copy(
        mut self,
        scope: &impl Anchoring,
        sig: &Signature,
        copy: &[FnArg],
    ) -> Self {
        self.in_copy = Some(unconstrained_in_copy(scope, sig, copy));
        self
    }

    /// The method lifetimes that the output names and no bound does.
    pub(crate) fn unconstrained(&self) -> &[Lifetime] {
        &self.unconstrained
    }

    /// The mark of a type that names the lifetimes `named`: those that the output names and no
    /// bound does.
    pub(crate) fn unconstrained_mark(&self, named: &[Lifetime]) -> SignatureMark {
        SignatureMark::Unconstrained(
            self.unconstrained
                .iter()
                .filter(|lifetime| named.contains(lifetime))
                .cloned()
                .collect(),
        )
    }

    /// Returns `true` if the part with `mark` and `context` is copied as written in the arguments
    /// of this signature: it has the same fault here. For the lifetimes of the mark (all that the
    /// output names and no bound does here) the copy must leave one unconstrained where it is
    /// known ([`SignatureFacts::in_copy()`]): where an input keeps the lifetime as written, the
    /// type is an entry, not a part.
    pub(crate) fn covers(&self, mark: &SignatureMark, context: Option<&Context>) -> bool {
        match mark {
            SignatureMark::Unconstrained(lifetimes) => {
                lifetimes
                    .iter()
                    .all(|lifetime| self.unconstrained.contains(lifetime))
                    && self.in_copy.as_ref().map_or(true, |left| {
                        lifetimes.iter().any(|lifetime| left.contains(lifetime))
                    })
            }
            SignatureMark::Elided => self.elision.as_ref().map_or(false, |elision| {
                context.map_or(true, |context| !elision.added(context).is_empty())
            }),
        }
    }
}

/// A type that may be an entry, which [`Analysis::apply()`] adds to the graph.
pub(crate) struct PendingType {
    /// The type.
    pub(crate) ty: Type,
    /// The address of the type in the visited syntax, which `ty` is a copy of.
    pub(crate) origin: usize,
    /// A required entry: its own path is a cause, so it is leaked whole.
    pub(crate) required: bool,
    /// It may be a required entry: it or a part is a cause.
    pub(crate) required_candidate: bool,
    /// An optional entry if the Referrer replaces it: its own path names an external crate.
    pub(crate) optional: bool,
    /// It may be an optional entry: it or a part is `optional`.
    pub(crate) optional_candidate: bool,
    /// The context that an entry of it would have.
    pub(crate) context: Context,
    /// A bare identifier argument that may be a const. Its parent is leaked whole if possible; if
    /// it cannot be because of a hard cause (see `Impossibles`), it is a conflict.
    pub(crate) ambiguous: bool,
    /// It cannot be a required entry, but its components can: the copy would be cyclic.
    pub(crate) no_required_entry: bool,
    /// Like `no_required_entry` for an optional entry.
    pub(crate) no_optional_entry: bool,
    /// The type names a `Self` without meaning, or a bound of its context does
    /// ([`correct_unmeant_self()`]).
    pub(crate) unmeant_self: bool,
    /// The type is in a bound or where predicate of the item.
    pub(crate) in_predicate: bool,
    /// The relative paths the type depends on, if it is an entry ([`CheckResult::paths`]).
    pub(crate) paths: Vec<Cause>,
    /// The primitive names in a type that is leaked whole, which `components` does not hold: they
    /// are assumed if the type is copied as written instead ([`Part::primitives`]).
    pub(crate) primitives: Vec<Ident>,
    /// The effects of the types in it that are leaked on their own.
    pub(crate) components: Vec<Effect>,
    /// For a type leaked whole, the effects of the types in it, which `components` does not hold:
    /// the copy keeps them if the type is copied as written instead.
    pub(crate) inside: Vec<Effect>,
    /// A type in it is not leaked alone, and its meaning depends on this type (a trait object that
    /// takes its lifetime from the path): copied as written, this type is whole.
    pub(crate) adopted: bool,
}

impl PendingType {
    pub(crate) fn new(ty: &Type, context: Context, components: Vec<Effect>) -> Self {
        Self {
            ty: ty.clone(),
            origin: ty as *const Type as usize,
            required: false,
            required_candidate: false,
            optional: false,
            optional_candidate: false,
            context,
            ambiguous: false,
            no_required_entry: false,
            no_optional_entry: false,
            unmeant_self: false,
            in_predicate: false,
            paths: Vec::new(),
            primitives: Vec::new(),
            components,
            inside: Vec::new(),
            adopted: false,
        }
    }

    /// The type with its context, as the analysis names an optional entry.
    pub(crate) fn optional_key(&self) -> (Context, String) {
        let ty = &self.ty;
        (self.context.clone(), quote!(#ty).to_string())
    }

    /// The part that the Referrer leaves as written if the type is copied so instead of leaked: a
    /// reference is copied whole, as its parts cannot stand for it ([`is_whole_reference()`]), and
    /// so is a type in a predicate and a type that holds one that depends on it.
    pub(crate) fn part(&self) -> Part {
        let whole = self.adopted || self.in_predicate || is_whole_reference(&self.ty);
        let mut primitives = Vec::new();
        if whole {
            primitives = self.primitives.clone();
            primitives.extend(primitive_names(&self.components));
        }
        Part {
            cause: Cause::Type(self.ty.clone()),
            whole,
            context: Some(self.context.clone()),
            in_predicate: self.in_predicate,
            within: None,
            mark: None,
            primitives,
        }
    }
}

impl CheckResult {
    /// The conflict of a type leaked whole, with the causes `must` and `impossible` and the
    /// `reason`. The caller sets what else it has (`note`, `help`, `headline`, `paths`).
    pub(crate) fn conflict(must: Vec<Cause>, impossible: Vec<Cause>, reason: Reason) -> Self {
        Self {
            must,
            impossible,
            is_own: true,
            optional: false,
            note: None,
            help: None,
            headline: None,
            cause: None,
            reason,
            part: None,
            seen: 0,
            place: Place::default(),
            paths: Vec::new(),
            origin: Origin::default(),
        }
    }

    /// The correction of a part copied as written for a fault that another error tells. It has no
    /// cause that depends on the context, so `report()` gives no error for it.
    pub(crate) fn corrected() -> Self {
        Self::conflict(Vec::new(), Vec::new(), Reason::RelativePath)
    }
}

pub(crate) enum Effect {
    Type(Box<PendingType>),
    Error(Box<CheckResult>),
    Primitive(Ident),
}

/// What the effects of the analysis add up to.
#[derive(Default)]
pub(crate) struct Analysis {
    /// The types that may be entries.
    pub(crate) types: TypeGraph,
    /// The primitive names that the copy writes as they are.
    pub(crate) primitives: Vec<Ident>,
    /// The parts that the Referrer leaves as written, in order and distinct.
    pub(crate) parts: Vec<PartItem>,
}

impl Analysis {
    /// Adds the types among `effects` that may be entries to the graph, and records the parts and
    /// primitive names. A type that may be none is not added; its components hang on `parent`.
    /// Returns `true` if a type that depends on the context was added, or hangs on `parent`.
    pub(crate) fn apply(
        &mut self,
        contexts: &mut Contexts,
        effects: &[Effect],
        parent: Option<&NodeKey>,
    ) -> bool {
        let mut holds_required = false;
        for effect in effects {
            match effect {
                Effect::Type(pending)
                    if !(pending.required_candidate || pending.optional_candidate) =>
                {
                    // Never an entry: not added, but the assumptions of its components stay.
                    holds_required |= self.apply(contexts, &pending.components, parent);
                }
                Effect::Type(pending) => {
                    let context = contexts.intern(pending.context.clone());
                    let key = self.types.add(context, &pending.ty, parent);
                    let types = &mut self.types;
                    if pending.required {
                        types.required.insert(key.clone());
                    }
                    if pending.optional {
                        types.optional.insert(key.clone());
                    }
                    if pending.required_candidate {
                        types.required_candidates.insert(key.clone());
                    }
                    if pending.optional_candidate {
                        types.optional_candidates.insert(key.clone());
                    }
                    let below = self.apply(contexts, &pending.components, Some(&key));
                    // The kind of the type is what it covers: only the flag of that kind counts.
                    let required = pending.required || below;
                    let no_entry = match required {
                        true => pending.required_candidate && pending.no_required_entry,
                        false => pending.optional_candidate && pending.no_optional_entry,
                    };
                    if no_entry {
                        self.types.no_entry.insert(key);
                    }
                    holds_required |= required;
                }
                Effect::Error(error) => {
                    // The contexts are interned as an entry's is.
                    if let Some(part) = &error.part {
                        self.primitives.extend(part.primitives.iter().cloned());
                        if !part.recorded() {
                            continue;
                        }
                        let item = part.item(contexts);
                        if !self.parts.contains(&item) {
                            self.parts.push(item);
                        }
                    }
                }
                Effect::Primitive(ident) => self.primitives.push(ident.clone()),
            }
        }
        holds_required
    }
}

/// Returns the errors of an analysis in the order found: the conflicts among `effects`, then the
/// cycles in `conflicts`, with the mentions of `Self` without meaning placed by
/// [`CheckResult::seen`]. Each is linked to the errors revealed inside its part ([`link_errors()`]).
pub(crate) fn scope_error(
    effects: &[Effect],
    self_errors: &[SelfMention],
    conflicts: &[CheckResult],
) -> Vec<ErrorItem> {
    let mut errors = Vec::new();
    collect_errors(effects, &mut errors);
    errors.extend(conflicts.iter().cloned());
    // The mentions, with the number of the mentions before each, and the conflict around each.
    let mut mentions = self_errors
        .iter()
        .enumerate()
        .map(|(index, mention)| (index, self_without_meaning(mention), mention.outer.clone()))
        .peekable();
    let mut items: Vec<ErrorItem> = Vec::new();
    // The token of the conflict of each item, and that of the conflict around it.
    let mut origins: Vec<Held> = Vec::new();
    // A conflict without a cause that depends on the context is no error: `report()` gives none.
    for (conflict, item) in errors
        .iter()
        .filter_map(|conflict| Some((conflict, report(conflict)?)))
    {
        while let Some((_, mention, outer)) =
            mentions.next_if(|(index, _, _)| *index < conflict.seen)
        {
            items.push(mention);
            origins.push((None, outer));
        }
        items.push(item);
        origins.push((
            Some(conflict.origin.token.clone()),
            conflict.origin.outer.clone(),
        ));
    }
    for (_, mention, outer) in mentions {
        items.push(mention);
        origins.push((None, outer));
    }
    link_errors(&mut items, &origins);
    items
}

/// The token of an error's conflict, and the token of the conflict that holds it inside its part.
type Held = (Option<Rc<()>>, Option<Rc<()>>);

/// Links each error to the errors revealed inside its part, those that it holds directly or
/// through another: `inner` of the error, by the `id` of each, which an error gets if some error
/// holds it. The ids are the positions in `items`.
fn link_errors(items: &mut [ErrorItem], origins: &[Held]) {
    let positions: HashMap<*const (), usize> = origins
        .iter()
        .enumerate()
        .filter_map(|(at, (token, _))| Some((Rc::as_ptr(token.as_ref()?), at)))
        .collect();
    let mut inner: Vec<Vec<usize>> = vec![Vec::new(); items.len()];
    let mut held = vec![false; items.len()];
    for (at, (_, outer)) in origins.iter().enumerate() {
        let mut around = outer.as_ref();
        // The chain is as long as the nesting is deep, and cannot make a cycle.
        for _ in 0..origins.len() {
            let holder = match around.and_then(|token| positions.get(&Rc::as_ptr(token))) {
                Some(&holder) if holder != at => holder,
                _ => break,
            };
            inner[holder].push(at);
            held[at] = true;
            around = origins[holder].1.as_ref();
        }
    }
    for (at, item) in items.iter_mut().enumerate() {
        item.link.id = held[at].then(|| at);
        item.link.inner = std::mem::take(&mut inner[at]);
    }
}

/// Makes the errors among `effects` that no conflict holds yet errors revealed inside the part of
/// the conflict `token`, which copies them as written ([`Origin::outer`]). The innermost conflict
/// holds an error, as the types inside are decided first.
pub(crate) fn adopt(effects: &mut [Effect], token: &Rc<()>) {
    for effect in effects {
        match effect {
            Effect::Error(error) => {
                error.origin.outer.get_or_insert_with(|| token.clone());
            }
            Effect::Type(pending) => adopt(&mut pending.components, token),
            Effect::Primitive(_) => (),
        }
    }
}

/// Counts `by` more mentions of `Self` without meaning before the conflicts in `effects` (see
/// [`CheckResult::seen`]).
pub(crate) fn shift_seen(effects: &mut [Effect], by: usize) {
    for effect in effects {
        match effect {
            Effect::Error(error) => error.seen += by,
            Effect::Type(pending) => {
                shift_seen(&mut pending.components, by);
                shift_seen(&mut pending.inside, by);
            }
            Effect::Primitive(_) => (),
        }
    }
}

/// Copies as written the types among `effects` (outermost first) that `corrected` gives a part:
/// each becomes an `Effect::Error` that reports nothing and holds the type as its part. Only a
/// whole part loses what is in it but the conflicts and the primitive names; else the types in the
/// type stay, corrected in turn.
pub(crate) fn correct_effects(
    effects: Vec<Effect>,
    corrected: &dyn Fn(&PendingType) -> Option<Part>,
) -> Vec<Effect> {
    let mut found = Vec::new();
    for effect in effects {
        match effect {
            Effect::Type(mut pending) => match corrected(&pending) {
                Some(part) => {
                    let whole = part.whole;
                    let mut error = CheckResult::corrected();
                    error.part = Some(part);
                    found.push(Effect::Error(Box::new(error)));
                    let PendingType {
                        components, inside, ..
                    } = *pending;
                    let mut inner = inside;
                    inner.extend(components);
                    if whole {
                        keep_conflicts(inner, &mut found);
                    } else {
                        found.extend(correct_effects(inner, corrected));
                    }
                }
                None => {
                    let components = std::mem::take(&mut pending.components);
                    pending.components = correct_effects(components, corrected);
                    found.push(Effect::Type(pending));
                }
            },
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
    written(effects, true, &mut |pending| pending.components, found)
}

/// Copies as written the outermost type that would be an entry and names a `Self` without
/// meaning, or has a bound in its context that does: the impl of the Repeater would read that
/// `Self` as its implementor ([`Reason::SelfWithoutMeaning`]). The types in it that do not are
/// leaked.
pub(crate) fn correct_unmeant_self(effects: Vec<Effect>) -> Vec<Effect> {
    correct_effects(effects, &|pending| {
        (pending.unmeant_self && (pending.required_candidate || pending.optional_candidate))
            .then(|| pending.part())
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

/// Returns `true` if a type that may become an entry (of either kind), or a part copied as written,
/// names `Self` or has bounds rewritten for the `Self` that `declared` is, so the Referrer needs to
/// know what `Self` is to find it.
pub(crate) fn names_self(effects: &[Effect], declared: &SelfRequirement) -> bool {
    effects.iter().any(|effect| match effect {
        Effect::Type(pending) => {
            ((pending.required_candidate || pending.optional_candidate)
                && depends_on_self(&pending.context, declared))
                || names_self(&pending.components, declared)
        }
        Effect::Error(error) => part_names_self(error, declared),
        Effect::Primitive(_) => false,
    })
}

/// Returns `true` if the part of `error`, if any, has a context that depends on `Self`, or none
/// because a bound names a `Self` that has no name: the Referrer finds it by that, so it needs to
/// know what `Self` is. That holds for a part that is not recorded too (see
/// [`Part::recorded()`]): it is, if its error is not dropped, and `ignore()` makes it whole.
pub(crate) fn part_names_self(error: &CheckResult, declared: &SelfRequirement) -> bool {
    error.part.as_ref().map_or(false, |part| {
        part.context
            .as_ref()
            .map_or(true, |context| depends_on_self(context, declared))
    })
}

fn depends_on_self(context: &Context, declared: &SelfRequirement) -> bool {
    context.self_ty.is_some() || context.substitutes(declared)
}

/// Makes the types in `keys` (with their contexts) no optional entries.
pub(crate) fn drop_optional(effects: &mut [Effect], keys: &HashSet<(Context, String)>) {
    each_pending_mut(effects, &mut |pending| {
        if pending.optional && keys.contains(&pending.optional_key()) {
            pending.optional = false;
        }
    });
}

/// Calls `visit` on the types among `effects`, each before the types in its components.
pub(crate) fn each_pending(effects: &[Effect], visit: &mut impl FnMut(&PendingType)) {
    for effect in effects {
        if let Effect::Type(pending) = effect {
            visit(pending);
            each_pending(&pending.components, visit);
        }
    }
}

/// Like [`each_pending()`], for `visit` that changes the types.
pub(crate) fn each_pending_mut(effects: &mut [Effect], visit: &mut impl FnMut(&mut PendingType)) {
    for effect in effects {
        if let Effect::Type(pending) = effect {
            visit(pending);
            each_pending_mut(&mut pending.components, visit);
        }
    }
}

pub(crate) fn has_optional_candidate(effects: &[Effect]) -> bool {
    effects
        .iter()
        .any(|effect| matches!(effect, Effect::Type(pending) if pending.optional_candidate))
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

/// What `effects` become if the copy writes the syntax they are about as it is, whole: no type
/// among them is leaked, so only the errors stay, without their parts (the syntax is the part). The
/// relative paths of the types that would be entries are written as they are, so they are pushed to
/// `paths`, the paths of the error of the syntax ([`CheckResult::paths`]).
pub(crate) fn as_written(effects: Vec<Effect>, paths: &mut Vec<Cause>) -> Vec<Effect> {
    let mut found = Vec::new();
    let mut enter = |pending: PendingType| {
        if pending.required {
            for path in &pending.paths {
                if !paths.contains(path) {
                    paths.push(path.clone());
                }
            }
        }
        let mut inner = pending.inside;
        inner.extend(pending.components);
        inner
    };
    written(effects, false, &mut enter, &mut found);
    found
}

/// Returns the errors among `effects`, without their parts: the caller copies the type that holds
/// them as written.
pub(crate) fn error_effects(effects: Vec<Effect>) -> Vec<Effect> {
    let mut found = Vec::new();
    written(
        effects,
        false,
        &mut |pending| pending.components,
        &mut found,
    );
    found
}

/// Pushes to `found` the errors among `effects` without their parts (the copy writes the syntax
/// they are about as it is), and the primitive names if `primitives`. A type is replaced by what
/// `enter` returns for it: its components, or more.
fn written(
    effects: Vec<Effect>,
    primitives: bool,
    enter: &mut impl FnMut(PendingType) -> Vec<Effect>,
    found: &mut Vec<Effect>,
) {
    for effect in effects {
        match effect {
            Effect::Error(mut error) => {
                error.part = None;
                found.push(Effect::Error(error));
            }
            Effect::Type(pending) => written(enter(*pending), primitives, enter, found),
            Effect::Primitive(_) if primitives => found.push(effect),
            Effect::Primitive(_) => (),
        }
    }
}

/// The primitive names in `effects`, each once, also in the parts and in the types leaked whole:
/// the names that a copy of the syntax as written holds ([`Part::primitives`]).
pub(crate) fn primitive_names(effects: &[Effect]) -> Vec<Ident> {
    let mut found = Vec::new();
    collect_primitive_names(effects, &mut found);
    found
}

fn collect_primitive_names(effects: &[Effect], found: &mut Vec<Ident>) {
    let mut add = |names: &[Ident]| {
        for ident in names {
            if !found.contains(ident) {
                found.push(ident.clone());
            }
        }
    };
    for effect in effects {
        match effect {
            Effect::Primitive(ident) => add(std::slice::from_ref(ident)),
            Effect::Type(pending) => {
                add(&pending.primitives);
                add(&primitive_names(&pending.components));
            }
            Effect::Error(error) => {
                if let Some(part) = &error.part {
                    add(&part.primitives);
                }
            }
        }
    }
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

//! What the analysis of the [`Leaker`] found: the [`CheckResult`] of a type, and the effects of
//! visiting it, which the [`Analysis`] turns into the graph of the entries; and the facts of a
//! signature that the mark of a part is about ([`SignatureFacts`]).

use crate::context::{Context, ContextId, Contexts, SelfRequirement};
use crate::elided::Elision;
use crate::error::{report, self_without_meaning, Cause, ErrorItem, Reason};
use crate::graph::{NodeKey, TypeGraph};
#[cfg(doc)]
use crate::leaker::Leaker;
use crate::result::Place;
use crate::util::{
    binder_declares, has_lifetime, has_lifetime_bound, is_whole_reference, names_in, push_once,
    same_lifetime, same_name, tokens_text, trait_object_of,
};
#[cfg(doc)]
use crate::view::View;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use syn::visit::Visit;
use syn::{
    FnArg, GenericArgument, Ident, Lifetime, Macro, Path, PathArguments, ReturnType, Signature,
    Type, TypeGroup, TypeParamBound, TypeParen, TypePath, TypePtr, TypeSlice,
};

/// What [`View::check()`] found in a type: the causes that make it need to be leaked, and the
/// causes that keep it from being leaked.
///
/// Each cause is a [`Cause`] of its kind of syntax. The first cause in `must` is the cause of the
/// [`ErrorItem`] that reports a conflict.
///
/// - Both empty: the copy writes the type as written.
/// - Only `must`: leak the type if `is_own` (`MyType`), else only its parts (`(MyType, Rel)`).
/// - Only `impossible`: the type cannot be leaked, and need not be (`impl ::path::Trait`).
/// - Both: a conflict (`MyType<'_>`). Unless `is_own`, the parts can be leaked instead.
///
/// `Self` where it has no meaning is an `impossible` cause ([`Reason::SelfWithoutMeaning`]). A
/// scope reports it as an error, and the copy writes the type that names it as written.
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
/// // `Rel<'_>` is a conflict: its path depends on where it is written, and `'_` cannot be leaked.
/// assert!(matches!(&conflict.must[0], Cause::Type(_)));
/// assert!(matches!(&conflict.impossible[0], Cause::Lifetime(_)));
/// ```
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct CheckResult {
    /// The causes that depend on where they are written (`A` and `B` in `::path::Wrap<A, B>`): the
    /// outermost first, then in order of appearance. In a conflict, the paths inside a type that is
    /// leaked alone are not listed.
    pub must: Vec<Cause>,
    /// The causes that cannot be leaked (an elided lifetime, `impl Trait`), in order of
    /// appearance, then the mentions of `Self` that have no meaning.
    pub impossible: Vec<Cause>,
    /// `true` if the type is leaked whole, so a conflict cannot be solved by leaking its parts
    /// (`MyType<'_>`); `false` if `must` is empty or only the parts are leaked
    /// (`::path::Wrap<'_, MyType>`). A bare identifier argument counts as a cause of the type that
    /// holds it, so `::ext::W<Rel>` is leaked whole
    /// ([Bare identifiers and consts](crate::Leaker#bare-identifiers-and-consts)).
    pub is_own: bool,
    /// `true` if the type is only a type of another crate (`::path::X`): it holds no cause of its
    /// own. It is an entry unless a rule allows its path ([`Allow`](crate::Allow)). `must` holds
    /// only the causes in its parts.
    pub optional: bool,
    /// Explains the conflict where the causes do not (why `Self` cannot be leaked, or has no
    /// meaning). The error shows it first.
    pub note: Option<String>,
    /// What to write instead. If it is `None`, the [`ErrorItem`] suggests the absolute path of the
    /// first cause in `must` ([`ErrorItem::help`](crate::ErrorItem::help)).
    pub help: Option<String>,
    /// The message of the [`ErrorItem`] if the fault is not a relative path. If it is `None`, the
    /// message is `use absolute path: '..'`, and the notes of the error say why the type cannot
    /// be leaked whole (`MyType<'_>`, E0106).
    pub headline: Option<String>,
    /// The cause of the [`ErrorItem`] if it is not the first of `must` (the default of a parameter
    /// that names `Self` is reported whole, with the note at the `Self` in it).
    pub(crate) cause: Option<Cause>,
    /// The [`Reason`] that goes with `headline` (`RelativePath` if there is none).
    pub(crate) reason: Reason,
    /// The part that the conflict leaves as written, if the Referrer has to know it.
    pub(crate) part: Option<Part>,
    /// The number of mentions of `Self` without meaning found when this conflict was recorded, to
    /// report them among the conflicts in the order found ([`scope_errors()`]).
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
    /// A part with no anchor and no mark.
    pub(crate) fn new(
        cause: Cause,
        whole: bool,
        context: Option<Context>,
        in_predicate: bool,
        primitives: Vec<Ident>,
    ) -> Self {
        Self {
            cause,
            whole,
            context,
            in_predicate,
            within: None,
            mark: None,
            primitives,
        }
    }

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

    /// Returns `true` if the Referrer needs to know the part: it is whole, or only its anchor or
    /// signature tells it from an entry. Any other part has no entry of the same tokens and
    /// context, so the visitor finds nothing to replace at its head anyway.
    fn recorded(&self) -> bool {
        self.whole || self.within.is_some() || self.mark.is_some()
    }
}

/// A part that the Referrer leaves as written, with its contexts interned ([`Part::recorded()`];
/// the grammar is above `FORMAT_TAG_PREFIX` in referrer.rs). One that is not recorded is there for
/// an error only, and the visitor skips it until `LeakerResult::ignore()` makes it whole.
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
/// signature gives the type ([`SignatureFacts`]; the grammar is above `FORMAT_TAG_PREFIX` in
/// referrer.rs).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum SignatureMark {
    /// The method lifetimes of the type that the output names and no bound does: the copy would
    /// leave one unconstrained (E0195, [`unconstrained_in_copy()`]).
    Unconstrained(Vec<Lifetime>),
    /// The output relies on elision, and the type adds lifetimes to the inputs (E0106).
    Elided,
}

/// The facts about a signature that a [`SignatureMark`] is about, from the signature alone: the
/// Leaker computes them where it makes a part, the Referrer at each signature it visits, adding
/// [`SignatureFacts::in_copy()`].
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

    /// The facts for the probe of the inputs of a signature ([`SignatureFacts::in_copy()`]): it
    /// covers no part ([`SignatureFacts::covers()`]), so the visitor takes a part that a signature
    /// made for a leaked type.
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
    /// the signature with the leaked types replaced ([`unconstrained_in_copy()`]).
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
                .filter(|lifetime| has_lifetime(named, lifetime))
                .cloned()
                .collect(),
        )
    }

    /// Returns `true` if the part with `mark` and `context` has the same fault in this signature,
    /// so it is copied as written in its arguments. Where `in_copy` is known, the copy must leave
    /// one of the mark's lifetimes unconstrained.
    pub(crate) fn covers(&self, mark: &SignatureMark, context: Option<&Context>) -> bool {
        match mark {
            SignatureMark::Unconstrained(lifetimes) => {
                lifetimes
                    .iter()
                    .all(|lifetime| has_lifetime(&self.unconstrained, lifetime))
                    && self.in_copy.as_ref().map_or(true, |left| {
                        lifetimes
                            .iter()
                            .any(|lifetime| has_lifetime(left, lifetime))
                    })
            }
            SignatureMark::Elided => self.elision.as_ref().map_or(false, |elision| {
                context.map_or(true, |context| !elision.added(context).is_empty())
            }),
        }
    }
}

/// The type of an argument of a signature (of a receiver, the type it stands for).
pub(crate) fn input_type(input: &FnArg) -> &Type {
    match input {
        FnArg::Receiver(receiver) => &receiver.ty,
        FnArg::Typed(arg) => &arg.ty,
    }
}

/// What a scan for the occurrences of a lifetime ([`scan_anchors()`]) found and keeps track of.
pub(crate) struct OutputUse {
    /// The lifetime that the scan looks for.
    pub(crate) lifetime: Lifetime,
    /// Each constraining occurrence in the inputs, as the types from the input down to it.
    pub(crate) anchors: Vec<Vec<usize>>,
    /// The addresses of the types from the input down to the one being scanned.
    chain: Vec<usize>,
    /// An external type that the Leaker does not allow, which the Referrer may replace, holds no
    /// occurrence. Else the occurrences are those that rustc sees as constraining.
    careful: bool,
    /// With `careful`: the types that the scan did not enter, by address.
    pub(crate) stopped: Vec<usize>,
    /// With `careful`: the types that the scan enters although the Referrer may replace them,
    /// because they are no optional entries, by address.
    entered: Vec<usize>,
}

impl OutputUse {
    pub(crate) fn new(lifetime: Lifetime, careful: bool, entered: Vec<usize>) -> Self {
        Self {
            lifetime,
            anchors: Vec::new(),
            chain: Vec::new(),
            careful,
            stopped: Vec::new(),
            entered,
        }
    }
}

/// What a scan for the occurrences of a lifetime ([`scan_anchors()`]) asks about the place it is
/// made in: the analysis of the Leaker, or the scopes of the signature that a Referrer writes.
pub(crate) trait Anchoring {
    /// Returns `true` if `path` is an associated item of a generic in scope or of `Self` (`T::A`).
    fn is_projection(&self, path: &Path) -> bool;

    /// Returns `true` if the Referrer may replace `ty` as an optional entry, so a careful scan
    /// finds no anchor in it (a path of an external crate, an external trait object, an array
    /// whose length names a crate, or a reference to such an object).
    fn is_optional_type(&self, ty: &Type) -> bool;
}

/// Scans the inputs of a signature for the occurrences of `lifetime` (see [`scan_anchors()`]).
pub(crate) fn scan_inputs<'a>(
    scope: &impl Anchoring,
    inputs: impl IntoIterator<Item = &'a FnArg>,
    lifetime: &Lifetime,
    careful: bool,
    entered: &[usize],
) -> OutputUse {
    let mut used = OutputUse::new(lifetime.clone(), careful, entered.to_vec());
    for input in inputs {
        match input {
            FnArg::Receiver(receiver) => {
                if receiver
                    .lifetime()
                    .map_or(false, |own| same_lifetime(own, lifetime))
                {
                    used.anchors.push(Vec::new());
                }
                scan_anchors(scope, &receiver.ty, &mut used)
            }
            FnArg::Typed(arg) => scan_anchors(scope, &arg.ty, &mut used),
        }
    }
    used
}

/// Finds the occurrences of `used.lifetime` in `ty` that constrain it (not in an associated type
/// path, an array length, `impl Trait` or a type macro), each with its chain of types: it anchors
/// only if no type on it is leaked whole. See [Binder lifetimes](crate::Leaker#binder-lifetimes).
pub(crate) fn scan_anchors(scope: &impl Anchoring, ty: &Type, used: &mut OutputUse) {
    let address = ty as *const Type as usize;
    if used.careful && !used.entered.contains(&address) && scope.is_optional_type(ty) {
        used.stopped.push(address);
        return;
    }
    used.chain.push(address);
    match ty {
        Type::Reference(reference) => {
            let lifetime = reference.lifetime.as_ref();
            if lifetime.map_or(false, |lifetime| same_lifetime(lifetime, &used.lifetime)) {
                used.anchors.push(used.chain.clone());
            }
            scan_anchors(scope, &reference.elem, used);
        }
        Type::Ptr(TypePtr { elem, .. })
        | Type::Slice(TypeSlice { elem, .. })
        | Type::Paren(TypeParen { elem, .. })
        | Type::Group(TypeGroup { elem, .. }) => scan_anchors(scope, elem, used),
        Type::Array(array) => scan_anchors(scope, &array.elem, used),
        Type::Tuple(tuple) => {
            for elem in &tuple.elems {
                scan_anchors(scope, elem, used);
            }
        }
        Type::Path(TypePath { qself: None, path }) if !scope.is_projection(path) => {
            if let Some(seg) = path.segments.last() {
                scan_arguments(scope, &seg.arguments, used);
            }
        }
        Type::BareFn(f) if !binder_declares(f.lifetimes.as_ref(), &used.lifetime) => {
            for arg in &f.inputs {
                scan_anchors(scope, &arg.ty, used);
            }
            scan_output(scope, &f.output, used);
        }
        Type::TraitObject(object) => {
            for bound in &object.bounds {
                match bound {
                    TypeParamBound::Trait(bound)
                        if !binder_declares(bound.lifetimes.as_ref(), &used.lifetime) =>
                    {
                        if let Some(seg) = bound.path.segments.last() {
                            scan_arguments(scope, &seg.arguments, used);
                        }
                    }
                    TypeParamBound::Lifetime(lifetime)
                        if same_lifetime(lifetime, &used.lifetime) =>
                    {
                        used.anchors.push(used.chain.clone())
                    }
                    _ => (),
                }
            }
        }
        _ => (),
    }
    used.chain.pop();
}

fn scan_arguments(scope: &impl Anchoring, arguments: &PathArguments, used: &mut OutputUse) {
    match arguments {
        PathArguments::AngleBracketed(args) => {
            for arg in &args.args {
                match arg {
                    GenericArgument::Lifetime(lifetime)
                        if same_lifetime(lifetime, &used.lifetime) =>
                    {
                        used.anchors.push(used.chain.clone())
                    }
                    GenericArgument::Type(ty) => scan_anchors(scope, ty, used),
                    GenericArgument::AssocType(binding) => scan_anchors(scope, &binding.ty, used),
                    _ => (),
                }
            }
        }
        PathArguments::Parenthesized(args) => {
            for ty in &args.inputs {
                scan_anchors(scope, ty, used);
            }
            scan_output(scope, &args.output, used);
        }
        PathArguments::None => (),
    }
}

fn scan_output(scope: &impl Anchoring, output: &ReturnType, used: &mut OutputUse) {
    if let ReturnType::Type(_, ty) = output {
        scan_anchors(scope, ty, used);
    }
}

/// Pushes the lifetimes that `ty` (an input) keeps constrained as written in the copy. A
/// reference to a trait object without a lifetime bound, or an array, keeps none if `unsure` says
/// the copy may not write the object or the array as it is.
pub(crate) fn kept_lifetimes(
    ty: &Type,
    kept: &mut Vec<Lifetime>,
    unsure: &mut impl FnMut(&Type) -> bool,
) {
    match ty {
        Type::Reference(reference) => {
            let unbounded = trait_object_of(&reference.elem)
                .map_or(false, |object| !has_lifetime_bound(object))
                && unsure(&reference.elem);
            if !unbounded {
                kept.extend(reference.lifetime.clone());
            }
            kept_lifetimes(&reference.elem, kept, unsure);
        }
        Type::Array(array) => {
            if !unsure(ty) {
                kept_lifetimes(&array.elem, kept, unsure)
            }
        }
        Type::Ptr(TypePtr { elem, .. })
        | Type::Slice(TypeSlice { elem, .. })
        | Type::Paren(TypeParen { elem, .. })
        | Type::Group(TypeGroup { elem, .. }) => kept_lifetimes(elem, kept, unsure),
        Type::Tuple(tuple) => {
            for elem in &tuple.elems {
                kept_lifetimes(elem, kept, unsure);
            }
        }
        _ => (),
    }
}

/// The lifetimes that a visit finds, also those in the tokens of a macro.
#[derive(Default)]
pub(crate) struct Lifetimes(pub(crate) Vec<Lifetime>);

impl<'ast> Visit<'ast> for Lifetimes {
    fn visit_lifetime(&mut self, i: &'ast Lifetime) {
        self.0.push(i.clone());
    }
    fn visit_macro(&mut self, i: &'ast Macro) {
        let names = names_in(&i.tokens);
        self.0.extend(
            names
                .iter()
                .filter(|name| name.starts_with('\''))
                .map(|name| Lifetime::new(name, proc_macro2::Span::call_site())),
        );
    }
}

/// Returns the lifetimes that the return type of `sig` mentions, and those that a bound names:
/// `'b: 'c` (both), `T: 'b` and `T: Tr<'b>`. A lifetime in a bound is early-bound.
fn output_and_bounded(sig: &Signature) -> (Lifetimes, Lifetimes) {
    let mut output = Lifetimes::default();
    output.visit_return_type(&sig.output);
    let mut bounded = Lifetimes::default();
    for param in sig.generics.lifetimes() {
        if !param.bounds.is_empty() {
            bounded.0.push(param.lifetime.clone());
        }
        for bound in &param.bounds {
            bounded.visit_lifetime(bound);
        }
    }
    for param in sig.generics.type_params() {
        for bound in &param.bounds {
            bounded.visit_type_param_bound(bound);
        }
    }
    if let Some(where_clause) = &sig.generics.where_clause {
        bounded.visit_where_clause(where_clause);
    }
    (output, bounded)
}

/// The lifetimes of the method `sig` that its output names and no bound does: where only types
/// leaked whole hold them in the inputs, the copy leaves them unconstrained (E0195).
pub(crate) fn unconstrained_in_output(sig: &Signature) -> Vec<Lifetime> {
    let (output, bounded) = output_and_bounded(sig);
    sig.generics
        .lifetimes()
        .map(|param| param.lifetime.clone())
        .filter(|lifetime| has_lifetime(&output.0, lifetime) && !has_lifetime(&bounded.0, lifetime))
        .collect()
}

/// The lifetimes of `unconstrained_in_output()` that some input of the method `sig` anchors
/// ([`scan_anchors()`]) and no input of `copy` does: `copy` is the inputs with the leaked types
/// replaced by types that hold no lifetime.
pub(crate) fn unconstrained_in_copy(
    scope: &impl Anchoring,
    sig: &Signature,
    copy: &[FnArg],
) -> Vec<Lifetime> {
    fn anchored<'a>(
        scope: &impl Anchoring,
        inputs: impl IntoIterator<Item = &'a FnArg>,
        lifetime: &Lifetime,
    ) -> bool {
        !scan_inputs(scope, inputs, lifetime, false, &[])
            .anchors
            .is_empty()
    }
    unconstrained_in_output(sig)
        .into_iter()
        .filter(|lifetime| {
            anchored(scope, &sig.inputs, lifetime) && !anchored(scope, copy, lifetime)
        })
        .collect()
}

/// A type that may be an entry, which [`Analysis::apply()`] adds to the graph.
pub(crate) struct PendingType {
    /// The type.
    pub(crate) ty: Type,
    /// The address of the type in the visited syntax, which `ty` is a copy of.
    pub(crate) address: usize,
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
    /// It cannot be a required entry, but its components can: the copy would be cyclic, or the
    /// type names the `Self` of a nested item.
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
            address: ty as *const Type as usize,
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

    /// Returns `true` if the type may become an entry, of either kind.
    fn may_be_entry(&self) -> bool {
        self.required_candidate || self.optional_candidate
    }

    /// The effects of the types in it: `inside`, then `components`.
    pub(crate) fn into_inner(self) -> Vec<Effect> {
        let mut inner = self.inside;
        inner.extend(self.components);
        inner
    }

    /// The type with its context, as the analysis names an optional entry.
    fn optional_key(&self) -> (Context, String) {
        let ty = &self.ty;
        (self.context.clone(), tokens_text(ty))
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
        Part::new(
            Cause::Type(self.ty.clone()),
            whole,
            Some(self.context.clone()),
            self.in_predicate,
            primitives,
        )
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
    fn corrected() -> Self {
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
                Effect::Type(pending) if !pending.may_be_entry() => {
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
                        if part.recorded() {
                            push_once(&mut self.parts, part.item(contexts));
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
/// [`CheckResult::seen`]. Each is linked to the errors inside its part ([`link_errors()`]).
pub(crate) fn scope_errors(
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
/// each becomes an error that reports nothing and holds the part. A whole part keeps only the
/// conflicts and primitive names inside it; else the types inside are corrected in turn.
pub(crate) fn correct_effects(
    effects: Vec<Effect>,
    corrected: &mut impl FnMut(&PendingType) -> Option<Part>,
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
                    let inner = pending.into_inner();
                    if whole {
                        written(inner, true, &mut |pending| pending.components, &mut found);
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

/// Copies as written the outermost entries that name a `Self` without meaning, or whose context
/// has a bound that does: the Repeater's impl would read it as its implementor
/// ([`Reason::SelfWithoutMeaning`]). The types in them that do not are leaked.
pub(crate) fn correct_unmeant_self(effects: Vec<Effect>) -> Vec<Effect> {
    correct_effects(effects, &mut |pending| {
        (pending.unmeant_self && pending.may_be_entry()).then(|| pending.part())
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
            (pending.may_be_entry() && depends_on_self(&pending.context, declared))
                || names_self(&pending.components, declared)
        }
        Effect::Error(error) => part_names_self(error, declared),
        Effect::Primitive(_) => false,
    })
}

/// Returns `true` if the part of `error`, if any, has a context that depends on `Self`, or none (a
/// bound names a `Self` without a name): the Referrer then needs `Self` to find it. That holds for
/// a part that is not recorded too, as `ignore()` may make it whole.
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

/// Returns the types, with their contexts, that are optional entries in `effects`.
pub(crate) fn optional_keys(effects: &[Effect]) -> Vec<(Context, String)> {
    let mut keys = Vec::new();
    each_pending(effects, &mut |pending| {
        if pending.optional {
            keys.push(pending.optional_key());
        }
    });
    keys
}

/// Makes the optional entries among `effects` whose type is at one of `addresses` no optional
/// entries, and adds their keys to `found` ([`drop_optional()`]).
pub(crate) fn block_optional(
    effects: &mut [Effect],
    addresses: &HashSet<usize>,
    found: &mut Vec<(Context, String)>,
) {
    each_pending_mut(effects, &mut |pending| {
        if pending.optional && addresses.contains(&pending.address) {
            found.push(pending.optional_key());
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

/// Returns the bare identifiers among `effects` that may be consts, also those in a group.
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

/// What `effects` become if the copy writes their syntax whole, as it is: only the errors stay,
/// without their parts. The relative paths of the required entries are pushed to `paths`, the
/// paths of the error of the syntax ([`CheckResult::paths`]).
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
        pending.into_inner()
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
    let add = |found: &mut Vec<Ident>, names: &[Ident]| {
        for ident in names {
            if !found.iter().any(|known| same_name(known, ident)) {
                found.push(ident.clone());
            }
        }
    };
    for effect in effects {
        match effect {
            Effect::Primitive(ident) => add(found, std::slice::from_ref(ident)),
            Effect::Type(pending) => {
                add(found, &pending.primitives);
                collect_primitive_names(&pending.components, found);
            }
            Effect::Error(error) => {
                if let Some(part) = &error.part {
                    add(found, &part.primitives);
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

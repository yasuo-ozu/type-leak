//! The analysis of items and types by the [`Leaker`].

use crate::analysis::{
    adopt, ambiguous_types, anchor_parts, as_written, block_optional, correct_effects,
    correct_unmeant_self, each_pending, each_pending_mut, error_effects, has_error,
    has_optional_candidate, kept_lifetimes, names_self, optional_keys, part_names_self,
    primitive_effects, primitive_names, scan_anchors, scan_inputs, scope_errors, shift_seen,
    unconstrained_in_output, Anchoring, CheckResult, Effect, Lifetimes, Origin, OutputUse, Part,
    PendingType, SelfMention, SignatureFacts, SignatureMark,
};
use crate::context::{
    Context, ContextId, Contexts, Enclosing, Gat, Scope, SelfRequirement, UnnameableSelf,
};
use crate::elided::{input_type, Elision};
use crate::error::{report, Cause, ErrorItem, Reason, SELF_HELP, SELF_NOTE};
#[cfg(doc)]
use crate::leaker::Leaker;
use crate::repeater::is_const_param_ty;
use crate::result::{Allow, Place};
use crate::unstable::{
    bindings, equated_generics, hidden_projection, predicate_projection, Instability, Projections,
    Unstable, Why,
};
use crate::util::{
    binder_declares, binder_lifetimes, bound_cause, bounded_ident, declared_names,
    first_associated_self_in_generics, first_self_in_generics, first_self_in_type,
    generic_param_mentions_self, generic_param_name, has_lifetime_bound, ident_name, is_bare_self,
    is_lifetime_param, is_primitive_name, is_type_param_named, is_whole_reference, names_in,
    or_list, path_type, plain_self_head, push_once, self_head, self_in_param, self_in_predicate,
    selves_in_generics, tidy, trait_head, trait_object_of, type_mentions_self, unwrap_type_group,
    where_clause, where_predicate_mentions_self, without_arguments, without_crate_prefix,
    without_default, without_defaults,
};
use crate::view::View;
use proc_macro2::TokenStream;
use std::collections::{HashMap, HashSet};
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use std::rc::Rc;
use syn::parse::ParseStream;
use syn::parse::Parser;
use syn::punctuated::Punctuated;
use syn::visit::Visit;
use syn::*;
use template_quote::quote;

/// What the visit of a type found in it: the type around takes it in (`AnalyzeVisitor::absorb()`),
/// and `AnalyzeVisitor::decide()` decides the type by it.
struct Frame {
    /// The length of `AnalyzeVisitor::musts` when the frame started: the causes of the type are
    /// those from there on.
    must_start: usize,
    /// The same for `AnalyzeVisitor::impossibles`.
    impossible_start: usize,
    /// The length of `Records::self_errors` when the frame started: a longer one means a
    /// `Self` without meaning.
    self_start: usize,
    /// The depth of the shallowest type in it that has a cause, `usize::MAX` if none: that type
    /// owns the causes.
    min_depth: usize,
    /// The outermost scope below the root that a generic named in it is declared in, `usize::MAX`
    /// if none (see `AnalyzeVisitor::mention_scope()`).
    local_scope: usize,
    /// It names a generic of the root scope, the Leaker's own generics.
    mentions_root: bool,
    /// It names `Self`.
    mentions_self: bool,
    /// The depth of the shallowest type in it whose own path names an external crate, `usize::MAX`
    /// if none: the Referrer may replace that type (an optional entry).
    optional_depth: usize,
    /// A type not leaked alone because of a binder lifetime: its holder's depth and the advice
    /// (see [`Binder`]).
    binder: Option<(usize, Advice)>,
    /// A type not leaked alone because of an elided output lifetime: its holder's depth and the
    /// advice (see [`Elision`]).
    elision: Option<(usize, Advice)>,
    /// The method lifetime that a trait object in it would take by default, and the copy would
    /// leave unconstrained (E0195).
    unconstrained: Option<Lifetime>,
    /// The first `Self::A` in the type, where `Self` is a type: the Repeater's impl cannot write
    /// it, so a type that names it is not leaked whole.
    assoc_self: Option<Cause>,
    /// The depths of the types, enclosing the type, that are the owner of a type in it that is not
    /// leaked alone (see [`AnalyzeVisitor::decide_as_owner()`]): its meaning depends on its owner,
    /// so the owner copied as written keeps it whole.
    adopted: Vec<usize>,
    /// What the types in it add to the graph, in order.
    effects: Vec<Effect>,
}

impl Frame {
    fn new(must_start: usize, impossible_start: usize, self_start: usize) -> Self {
        Self {
            must_start,
            impossible_start,
            self_start,
            min_depth: usize::MAX,
            local_scope: usize::MAX,
            mentions_root: false,
            mentions_self: false,
            optional_depth: usize::MAX,
            binder: None,
            elision: None,
            unconstrained: None,
            assoc_self: None,
            adopted: Vec::new(),
            effects: Vec::new(),
        }
    }
}

/// A cause that depends on the context (`AnalyzeVisitor::musts`).
struct Must {
    /// The depth of the type that has it.
    depth: usize,
    /// The path or type that depends on the context.
    cause: Cause,
    /// An entry that is leaked alone has it, so it is no conflict of the type around: see
    /// [`AnalyzeVisitor::conflict_causes()`].
    covered: bool,
    /// The relative path it is, as the cause of its kind without its arguments, if it is one (not
    /// a lifetime, not `Self`): an entry of [`ErrorItem::paths`].
    path: Option<Cause>,
}

/// The causes that stop a type from being leaked whole, in order of appearance. Under a soft cause
/// (an elided lifetime, a trait object that takes it, `impl Trait`) a bare identifier argument is
/// leaked as a type; under a hard one it is an error ([`bare_conflict()`]).
#[derive(Default)]
struct Impossibles {
    /// The causes, in order of pushing.
    causes: Vec<Impossible>,
    /// The depth of the type being visited now.
    at: usize,
}

/// A cause in [`Impossibles`].
struct Impossible {
    cause: Cause,
    /// A bare identifier is an error under it.
    hard: bool,
    /// The type depth (`Position::type_depth`) when it was pushed.
    depth: usize,
}

impl Impossibles {
    fn len(&self) -> usize {
        self.causes.len()
    }

    /// Pushes a hard cause (the default: a bare identifier is an error under it).
    fn push(&mut self, cause: Cause) {
        self.push_with(cause, true);
    }

    /// Pushes a soft cause.
    fn push_soft(&mut self, cause: Cause) {
        self.push_with(cause, false);
    }

    fn push_with(&mut self, cause: Cause, hard: bool) {
        let depth = self.at;
        self.causes.push(Impossible { cause, hard, depth });
    }

    fn truncate(&mut self, len: usize) {
        self.causes.truncate(len);
    }

    fn remove(&mut self, at: usize) {
        self.causes.remove(at);
    }

    /// Returns `true` if one of the causes from `start` on was pushed by the type at `depth` itself
    /// (its elided lifetime, `impl Trait`), not by a type in it.
    fn own_since(&self, start: usize, depth: usize) -> bool {
        self.causes[start..]
            .iter()
            .any(|impossible| impossible.depth == depth)
    }

    /// The causes from `start` on.
    fn since(&self, start: usize) -> impl Iterator<Item = &Cause> {
        self.causes[start..]
            .iter()
            .map(|impossible| &impossible.cause)
    }

    /// Returns `true` if one of the causes from `start` on is hard.
    fn any_hard_since(&self, start: usize) -> bool {
        self.causes[start..]
            .iter()
            .any(|impossible| impossible.hard)
    }
}

/// Why a type cannot be leaked and what to write instead: the `note` and the `help` of a conflict.
#[derive(Clone)]
struct Advice {
    /// Why the type cannot be leaked.
    note: String,
    /// What to write instead.
    help: String,
}

impl Advice {
    /// Puts them in `conflict`.
    fn give(self, conflict: &mut CheckResult) {
        conflict.note = Some(self.note);
        conflict.help = Some(self.help);
    }
}

/// A `for<..>` binder whose output mentions lifetimes that no input keeps as written (an anchor). A
/// type using them cannot be leaked alone: its copy holds them only in an associated type path
/// (E0581, E0582). See [Binder lifetimes](crate::Leaker#binder-lifetimes).
struct Binder {
    /// The depth of the type holding the binder, leaked whole instead (0 in a bound).
    owner: usize,
    /// The lifetimes that the output of the binder names and no input keeps as written.
    output: Vec<Lifetime>,
}

/// The output and the inputs of a bound of a binder.
struct BinderGroup<'a> {
    /// The lifetimes of the binder that the output names.
    output: Vec<Lifetime>,
    /// The types and lifetimes that the inputs hold.
    inputs: Vec<BinderInput<'a>>,
}

enum BinderInput<'a> {
    Type(&'a Type),
    Lifetime(&'a Lifetime),
}

/// A scope that the Leaker or a [`View`] enters: the generics that it declares.
pub(crate) struct ScopeSpec {
    /// The generics that the scope declares.
    generics: Generics,
    /// What `Self` is inside, if the scope declares it. `None` keeps the `Self` around.
    declares: Option<Box<DeclaredSelf>>,
    /// The trait of an impl, checked on entering the scope like the root of a trait bound: it
    /// must not depend on the context, but its arguments are analyzed by the walk of the impl.
    trait_path: Option<Box<Path>>,
}

/// The `Self` that a scope declares, and what comes with it.
pub(crate) struct DeclaredSelf {
    /// What `Self` means in the scope.
    requirement: SelfRequirement,
    /// `Self` is a type that the copy writes (the self type of an impl), not the item itself.
    written: bool,
    /// The scope is that of an inherent impl: no trait declares the signatures of its methods
    /// (see [`AnalyzeVisitor::with_signature()`]).
    inherent: bool,
    /// The generic associated types of the trait, if the scope is that of a trait whose items the
    /// Leaker sees (see [`Gat`]).
    gats: Vec<Gat>,
}

impl ScopeSpec {
    pub(crate) fn declared_generics(&self) -> Generics {
        self.generics.clone()
    }

    /// The generic associated types of the trait that the scope declares, if it sees them.
    pub(crate) fn declared_gats(&self) -> Vec<Gat> {
        self.declares
            .as_ref()
            .map_or_else(Vec::new, |declared| declared.gats.clone())
    }

    /// What `Self` is in the scope, if it declares it.
    pub(crate) fn declared_self(&self) -> Option<SelfRequirement> {
        self.declares
            .as_ref()
            .map(|declared| declared.requirement.clone())
    }
}

/// Where a scope is entered: the Leaker (outermost) or a [`View`] (nested). The public `enter_*`
/// methods build the [`ScopeSpec`] here and differ only in `enter_spec()`.
pub(crate) trait Scopes {
    /// The Leaker's [`Allow`], for the paths left as written.
    type Rule: Allow;

    fn enter_spec<R>(
        &mut self,
        spec: ScopeSpec,
        f: impl FnOnce(&mut View<'_, '_, Self::Rule>) -> R,
    ) -> (ContextId, R);

    fn enter_generics<R>(
        &mut self,
        generics: &Generics,
        f: impl FnOnce(&mut View<'_, '_, Self::Rule>) -> R,
    ) -> (ContextId, R) {
        let generics = generics.clone();
        self.enter_spec(
            ScopeSpec {
                generics,
                declares: None,
                trait_path: None,
            },
            f,
        )
    }

    fn enter_declaring<R>(
        &mut self,
        generics: &Generics,
        declares: DeclaredSelf,
        trait_path: Option<&Path>,
        f: impl FnOnce(&mut View<'_, '_, Self::Rule>) -> R,
    ) -> (ContextId, R) {
        let spec = ScopeSpec {
            generics: generics.clone(),
            declares: Some(Box::new(declares)),
            trait_path: trait_path.map(|path| Box::new(path.clone())),
        };
        self.enter_spec(spec, f)
    }

    fn enter_type<R>(
        &mut self,
        ident: &Ident,
        generics: &Generics,
        f: impl FnOnce(&mut View<'_, '_, Self::Rule>) -> R,
    ) -> (ContextId, R) {
        let declares = DeclaredSelf {
            requirement: SelfRequirement::of_item(ident, generics),
            written: false,
            inherent: false,
            gats: Vec::new(),
        };
        self.enter_declaring(generics, declares, None, f)
    }

    /// An impl of the trait `trait_` (which must not depend on the context), or an inherent impl
    /// if it is `None`.
    fn enter_impl<R>(
        &mut self,
        generics: &Generics,
        self_ty: &Type,
        trait_: Option<&Path>,
        f: impl FnOnce(&mut View<'_, '_, Self::Rule>) -> R,
    ) -> (ContextId, R) {
        let declares = DeclaredSelf {
            requirement: SelfRequirement::Type(self_ty.clone()),
            written: true,
            inherent: trait_.is_none(),
            gats: Vec::new(),
        };
        self.enter_declaring(generics, declares, trait_, f)
    }

    fn enter_trait<R>(
        &mut self,
        ident: &Ident,
        generics: &Generics,
        f: impl FnOnce(&mut View<'_, '_, Self::Rule>) -> R,
    ) -> (ContextId, R) {
        self.enter_trait_of(ident, generics, Vec::new(), f)
    }

    /// Like `enter_trait()` for a trait whose items the Leaker sees, with their `gats`.
    fn enter_trait_of<R>(
        &mut self,
        ident: &Ident,
        generics: &Generics,
        gats: Vec<Gat>,
        f: impl FnOnce(&mut View<'_, '_, Self::Rule>) -> R,
    ) -> (ContextId, R) {
        let declares = DeclaredSelf {
            requirement: SelfRequirement::of_trait(ident, generics),
            written: false,
            inherent: false,
            gats,
        };
        self.enter_declaring(generics, declares, None, f)
    }
}

/// What a visitor is set to when it is made, and keeps.
struct Mode {
    /// The place of the visitor that this one is a throwaway copy of
    /// ([`AnalyzeVisitor::throwaway()`]): the rule is asked at the place of the original.
    outer_place: Place,
    /// The values of an item are read too (see
    /// [`Leaker::check_values`](crate::Leaker::check_values)).
    check_values: bool,
    /// Whether [`AnalyzeVisitor::env_cycle()`] is checked. It is not by the probe that it runs.
    check_env: bool,
}

/// What a visitor records apart from the effects, which a scope or a check cuts back with
/// [`Records::truncate()`].
#[derive(Default)]
struct Records {
    /// The Leaker's table while it analyzes (see [`AnalyzeVisitor::swap_contexts()`]), an empty one
    /// in a throwaway visitor.
    contexts: Contexts,
    /// Mentions of `Self` where it has no meaning (see [`Reason::SelfWithoutMeaning`]). Kept apart
    /// from the effects, because a type leaked whole drops those of its parts.
    self_errors: Vec<SelfMention>,
    /// The top-level contexts of the scopes left, in order.
    top_levels: Vec<ContextId>,
    /// The types, with the contexts they are in, that were found to be no optional entries.
    not_optional: Vec<(Context, String)>,
    /// The cycles that `Self` makes (see [`AnalyzeVisitor::env_cycle()`]). Kept apart from the
    /// effects, like `self_errors`.
    env_errors: Vec<CheckResult>,
}

/// The lengths of the [`Records`].
struct RecordsMark {
    /// The length of `Records::contexts`.
    contexts: usize,
    /// The length of `Records::self_errors`.
    self_errors: usize,
    /// The length of `Records::top_levels`.
    top_levels: usize,
    /// The length of `Records::not_optional`.
    not_optional: usize,
    /// The length of `Records::env_errors`.
    env_errors: usize,
}

impl Records {
    fn mark(&self) -> RecordsMark {
        RecordsMark {
            contexts: self.contexts.len(),
            self_errors: self.self_errors.len(),
            top_levels: self.top_levels.len(),
            not_optional: self.not_optional.len(),
            env_errors: self.env_errors.len(),
        }
    }

    /// The cycles recorded since `mark`.
    fn env_errors_since(&self, mark: &RecordsMark) -> &[CheckResult] {
        &self.env_errors[mark.env_errors..]
    }

    /// Removes what was recorded since `mark`.
    fn truncate(&mut self, mark: &RecordsMark) {
        self.self_errors.truncate(mark.self_errors);
        self.contexts.truncate(mark.contexts);
        self.top_levels.truncate(mark.top_levels);
        self.not_optional.truncate(mark.not_optional);
        self.env_errors.truncate(mark.env_errors);
    }
}

/// What `Self` means where the visitor is: set and put back together by a scope or an item.
#[derive(Clone, Default)]
struct SelfScope {
    /// What `Self` means here, if anything. It has none directly in a `mod` (in a type), or in
    /// generics that no item declaring it encloses, so mentioning it there is an error.
    ty: Option<SelfRequirement>,
    /// `ty` is the self type of an impl, which the copy writes (see [`DeclaredSelf`]).
    written: bool,
    /// `ty` is the `Self` of a struct, an enum or a union nested in the analyzed syntax
    /// (`View::visit_item()`, not a scope's own item): the copy's own type, not the Leaker's.
    /// - A type with a bound that names it is an entry for any type ([`SelfRequirement::Any`]),
    ///   and the copy passes its own type. If the type names `Self` too, that is a risk
    ///   ([`Reason::NestedSelfInBounds`], see `nested_self_risk()`).
    /// - An absolute type that names it is no optional entry: it is copied as written.
    /// - The copy cannot write it in a parameter default (E0735, `Position::in_nested_default`).
    nested: bool,
    /// The trait that the impl declaring `Self` implements, if it is an impl of a trait: what a
    /// `Self::A` is written through (see [`AssocSelf`]).
    trait_: Option<ImplTrait>,
    /// The number of scopes when the scope that declares `Self` is the innermost: 0 if there is no
    /// `Self`. The predicates of the item that declares `Self` are visited then, not those of its
    /// methods.
    depth: usize,
}

/// What the type or path that holds a type tells it, taken by the type at its start.
#[derive(Clone, Default)]
struct ArgHints {
    /// A referent or path argument that cannot be leaked whole (elided or several local lifetimes,
    /// or in a bound). A trait object there cannot be leaked alone.
    elided_ref: bool,
    /// Like `elided_ref`, for a method lifetime that the copy would leave unconstrained (see
    /// `signatures`). The conflict has a note of its own.
    unconstrained_ref: Option<Lifetime>,
    /// The argument is a bare identifier, which may be a const.
    ambiguous_arg: bool,
    /// A referent or path argument whose lifetime is a generic. A trait object there defaults to
    /// it, and would default to `'static` if leaked alone. What is inside it is not affected.
    inherits: Option<Inherits>,
}

/// Where the walk is in a type, a signature or a value: what a check puts back as it was.
#[derive(Clone, Default)]
struct Position {
    /// The number of types being visited, the one at the end included.
    type_depth: usize,
    /// The elision of the method whose arguments are being visited.
    signature: Option<Elision>,
    /// While a binder is probed: the addresses of the types found to be leaked whole.
    binder_probe: Option<HashSet<usize>>,
    /// The type is the default of a parameter, which cannot mention `Self`. It holds whether the
    /// declaring scope gives `Self` a meaning. `SelfScope::ty` is `None` meanwhile.
    in_default: Option<bool>,
    /// The type is the default of a parameter of a nested item (`SelfScope::nested`). An entry here
    /// has no `Self` and no bound that names it ([`AnalyzeVisitor::context_of()`]); a type that
    /// names `Self` is an error if a bound does too.
    in_nested_default: bool,
    /// The depth of the type owning the expression (array length, const argument) being visited.
    /// Types in it are not leaked alone: the copy would put an associated type path in a constant.
    expr_owner: Option<usize>,
    /// In a receiver (`self: Box<Self>`), where rustc accepts only `Self` and types dereferencing
    /// to it: no type in it is optional.
    in_receiver: bool,
    /// The lifetimes the type being visited must outlive because of the types around it (see
    /// [`Enclosing`]). The Referrer tracks the same ones to find the entry. Empty outside a type.
    enclosing: Enclosing,
    /// The depth of the bounds and where predicates being visited, which are the predicates of the
    /// item (its environment). See [`AnalyzeVisitor::env_cycle()`].
    env_depth: usize,
    /// An argument of a function or method is being visited: an `impl Trait` in it is a generic of
    /// the function, and its bounds are predicates of the item (see `env_depth`).
    in_fn_arg: bool,
    /// A value is being read ([`AnalyzeVisitor::place()`]).
    in_value: bool,
    /// The default body or the default value of an associated const of a trait is being read.
    in_trait_default: bool,
    /// The default of a generic parameter is being visited, where `in_default` and
    /// `in_nested_default` do not tell.
    in_param_default: bool,
    /// What the type or path around the next type tells it.
    hints: ArgHints,
}

/// What a scope marks to leave it ([`AnalyzeVisitor::leave_scope()`]) and to roll it back
/// ([`AnalyzeVisitor::roll_back()`]).
struct ScopeMark {
    /// The number of effects of the frame.
    effects: usize,
    /// The lengths of the records.
    records: RecordsMark,
    /// The number of generics in scope.
    scopes: usize,
    /// `AnalyzeVisitor::item_depth`.
    item_depth: usize,
    /// The number of `AnalyzeVisitor::alone_defaults`.
    alone_defaults: usize,
    /// What `Self` meant.
    self_scope: SelfScope,
    /// `AnalyzeVisitor::self_type_walk`.
    self_type_walk: Option<bool>,
    /// `AnalyzeVisitor::inherent_impl`.
    inherent_impl: bool,
}

/// The defaults of the type parameters that a scope analyzes alone, without `Self` and with the
/// parameters before them (`finish()`, `end_scope()`, `analyze_defaults_alone()`). The walk of
/// the item must not analyze them again: that would record the entry twice.
struct AloneDefaults {
    /// `scopes.len()` and `item_depth` while the walk is directly in the scope: not in a method
    /// or in a nested item, whose generics are other parameters.
    scopes: usize,
    item_depth: usize,
    /// The parameters of the scope's generics that have a default.
    idents: Vec<Ident>,
    /// What the scope declares (`Self` and its generics), if it is the scope of an item: the item
    /// that is nested in the scope with the same generics and `Self` is the item itself, visited
    /// again (see `AnalyzeVisitor::visit_item_trait()`).
    declared: Option<(SelfRequirement, Generics)>,
}

/// What [`AnalyzeVisitor::check()`] puts back as it found it. The visit replaces the frame of the
/// scope, and appends to the causes and, for an expression with an item, to the contexts. `walk`
/// is the state that only a visit in progress changes, which a panic would leave half-way.
struct Checkpoint {
    /// The marks of the scope.
    mark: ScopeMark,
    /// The frame of the scope, which the visit replaces.
    frame: Frame,
    /// The number of causes that depend on the context.
    musts: usize,
    /// The number of causes that stop a type from being leaked whole.
    impossibles: usize,
    /// The state of the walk.
    walk: Walk,
}

/// The state of the walk of a type, which is back to what it was when the walk ends: the position
/// and the lengths of the stacks.
struct Walk {
    /// Where the walk is.
    position: Position,
    /// The number of `AnalyzeVisitor::binders`.
    binders: usize,
    /// The number of `AnalyzeVisitor::elisions`.
    elisions: usize,
    /// The number of `AnalyzeVisitor::bound_elided`.
    bound_elided: usize,
    /// The number of `AnalyzeVisitor::signatures`.
    signatures: usize,
}

/// What the outermost scope hands over to the Leaker when it ends.
pub(crate) struct Finished {
    /// What the scope added to the graph, in order.
    pub(crate) effects: Vec<Effect>,
    /// The top-level contexts of the scopes left, in order.
    pub(crate) top_levels: Vec<ContextId>,
    /// The types, with their contexts, that cannot be optional entries (see
    /// [`AnalyzeVisitor::env_cycle()`]).
    pub(crate) not_optional: Vec<(Context, String)>,
}

/// The visitor that analyzes items and types for the Leaker and its views.
pub(crate) struct AnalyzeVisitor<'a, A> {
    /// The Leaker's rule, asked at every check. A throwaway visitor borrows it again.
    rule: &'a mut A,
    /// What the visitor is set to when it is made.
    mode: Mode,
    /// What it records for the Leaker, apart from the effects.
    records: Records,
    /// The generics in scope, innermost last. The first are the Leaker's.
    scopes: Vec<Scope>,
    /// The `for<..>` binders being visited, innermost last.
    binders: Vec<Binder>,
    /// The signatures whose inputs are being visited, innermost last ([`Elision`]). `None` for one
    /// that needs nothing; it hides the ones around it, as a function pointer or `Fn(..)` has its
    /// own lifetimes.
    elisions: Vec<Option<Elision>>,
    /// The function pointers and `Fn(..)` being visited, innermost last: the positions in
    /// `impossibles` of the elided lifetimes each binds (see [`AnalyzeVisitor::push_elided()`]).
    bound_elided: Vec<Vec<usize>>,
    /// For each scope entered, innermost last, the defaults that it analyzes alone (see
    /// [`AloneDefaults`]).
    alone_defaults: Vec<AloneDefaults>,
    /// What `Self` means here.
    self_scope: SelfScope,
    /// Set during [`AnalyzeVisitor::walk_item_impl()`]: whether a bare `Self` was found in a
    /// predicate of the impl, so the walk calls [`AnalyzeVisitor::block_unstable_self_types()`].
    self_type_walk: Option<bool>,
    /// The causes that depend on the context.
    musts: Vec<Must>,
    /// The causes that stop the type being visited from being leaked whole.
    impossibles: Impossibles,
    /// What the type being visited has found so far.
    frame: Frame,
    /// The number of items the walk is in: 0 outside any, so the outermost item is not nested.
    item_depth: usize,
    /// The signatures being visited, innermost last. Only those that rustc compares with a trait's
    /// have one (see [`AnalyzeVisitor::with_signature()`]).
    signatures: Vec<SignatureWalk>,
    /// The methods being visited are those of an inherent impl: no trait declares their signature,
    /// so the copy cannot differ from it (E0195). Anywhere else, a method may be one of a trait.
    inherent_impl: bool,
    /// Where the walk is in the type, signature or value being visited.
    position: Position,
}

/// A method signature being visited (see `AnalyzeVisitor::signatures`).
struct SignatureWalk {
    /// The index in `scopes` of its generics.
    at: usize,
    /// The number of effects of the frame when the walk started: the signature's types come after.
    effects: usize,
    /// Its lifetimes.
    declared: Vec<Lifetime>,
    /// Its lifetimes that the copy would leave unconstrained (E0195), if a trait object inherits
    /// them.
    risky: Vec<Lifetime>,
    /// The types leaked whole that name a lifetime of it, by address: the lifetimes they name and
    /// the conflict to report if the copy has nothing but them to constrain a lifetime.
    leaked: HashMap<usize, (Vec<Lifetime>, CheckResult)>,
}

/// What a trait object without a lifetime bound takes its lifetime from. See
/// [Trait objects](crate::Leaker#trait-objects).
#[derive(Clone, Copy)]
enum Inherits {
    /// A reference `&'a dyn Trait` (its depth). The default is `'a` unless the trait has a bound of
    /// its own, which the Leaker cannot see for a relative trait, so the object goes with the
    /// reference.
    Reference(usize),
    /// A path with lifetime arguments (`Ref<'a, dyn Trait>`): the depth of the path. Like
    /// `Reference`.
    Path(usize),
}

impl Inherits {
    fn owner(self) -> usize {
        match self {
            Inherits::Reference(owner) | Inherits::Path(owner) => owner,
        }
    }
}

/// Where a phase of `decide()` sends the frame instead of deciding the type.
enum Handoff {
    /// To the owner at that depth, the type holding the binder or the elision that stops the type.
    Owner(usize),
    /// To the parts only: the type is no optional entry, its parts may be.
    Through,
}

// `A: 'a` is not implied for the associated functions on rustc 1.56.
impl<'a, A: Allow + 'a> AnalyzeVisitor<'a, A> {
    /// `check_values`: the values of the items are read too (see
    /// [`Leaker::check_values`](crate::Leaker::check_values)).
    pub(crate) fn new(rule: &'a mut A, generics: &Generics, check_values: bool) -> Self {
        Self {
            rule,
            mode: Mode {
                outer_place: Place::default(),
                check_values,
                check_env: true,
            },
            records: Records::default(),
            scopes: vec![Scope::new(generics, None)],
            binders: Vec::new(),
            elisions: Vec::new(),
            bound_elided: Vec::new(),
            alone_defaults: Vec::new(),
            self_scope: SelfScope::default(),
            self_type_walk: None,
            musts: Vec::new(),
            impossibles: Impossibles::default(),
            frame: Frame::new(0, 0, 0),
            item_depth: 0,
            signatures: Vec::new(),
            inherent_impl: false,
            position: Position::default(),
        }
    }

    /// Exchanges the table of contexts with `other`, which cannot fail. The outermost scope of the
    /// Leaker swaps its table in before the analysis and out after it, whatever the outcome.
    pub(crate) fn swap_contexts(&mut self, other: &mut Contexts) {
        std::mem::swap(&mut self.records.contexts, other);
    }

    /// Returns the Leaker's rule, for the check that follows the analysis.
    pub(crate) fn rule(&self) -> &A {
        &*self.rule
    }

    /// Runs `f` on a throwaway visitor with the generics `generics`, which asks the rule `rule` at
    /// `place`.
    fn throwaway<'b, R>(
        rule: &'b mut A,
        generics: &Generics,
        place: Place,
        f: impl FnOnce(&mut AnalyzeVisitor<'b, A>) -> R,
    ) -> R {
        let mut visitor = AnalyzeVisitor::new(rule, generics, false);
        visitor.mode.outer_place = place;
        f(&mut visitor)
    }

    /// Analyzes the default `ty` of a type parameter, with the parameters before it as the root
    /// generics; the values in it are read if `check_values`. Returns the effects and the mentions
    /// of `Self` that have no meaning.
    fn analyze_default(
        rule: &mut A,
        ty: &Type,
        generics: &Generics,
        declares_self: bool,
        check_values: bool,
    ) -> (Vec<Effect>, Vec<SelfMention>) {
        Self::throwaway(rule, generics, Place::default(), |visitor| {
            visitor.mode.check_values = check_values;
            // Inside the scope, as the visitor that the Leaker makes is after `enter_scope()`: an
            // item in the default is nested, so its generics and its `Self` are its own.
            visitor.item_depth = 1;
            visitor.position.in_default = Some(declares_self);
            visitor.visit_type(ty);
            (
                std::mem::take(&mut visitor.frame.effects),
                std::mem::take(&mut visitor.records.self_errors),
            )
        })
    }

    fn causes(&self, must_start: usize, impossible_start: usize, depth: usize) -> CheckResult {
        let mut must: Vec<&Must> = self.musts[must_start..].iter().collect();
        must.sort_by_key(|must| must.depth);
        CheckResult {
            is_own: must.first().map_or(false, |must| must.depth == depth),
            ..CheckResult::conflict(
                must.into_iter().map(|must| must.cause.clone()).collect(),
                self.impossibles.since(impossible_start).cloned().collect(),
                Reason::RelativePath,
            )
        }
    }

    /// Like [`AnalyzeVisitor::causes()`] for a type at `depth` that cannot be leaked whole: only
    /// the causes that the type owns. A type inside has an error of its own, and a path leaked
    /// alone is no cause (`u8` in `<Self as Tr<u8>>::X`).
    fn conflict_causes(
        &self,
        must_start: usize,
        impossible_start: usize,
        depth: usize,
    ) -> CheckResult {
        let mut result = self.causes(must_start, impossible_start, depth);
        let mut must: Vec<&Must> = self.musts[must_start..]
            .iter()
            .filter(|must| !must.covered)
            .collect();
        must.sort_by_key(|must| must.depth);
        let own = must.iter().filter(|must| must.depth == depth).count();
        if own > 0 {
            must.truncate(own);
        }
        if !must.is_empty() {
            result.is_own = must.first().map_or(false, |must| must.depth == depth);
            result.must = must.iter().map(|must| must.cause.clone()).collect();
        }
        result.paths = self.own_paths(must_start, depth);
        result
    }

    /// The relative paths of the causes from `must_start` on that the type at `depth` owns: the
    /// heads that the copy writes as they are ([`CheckResult::paths`]).
    fn own_paths(&self, must_start: usize, depth: usize) -> Vec<Cause> {
        paths_of(
            self.musts[must_start..]
                .iter()
                .filter(|must| must.depth == depth),
        )
    }

    /// Visits `ty` and returns what it found at its own depth: the causes, whether it is an
    /// optional entry, and its own conflict if any. What the visit records stays (the caller rolls
    /// it back or throws the visitor away).
    fn found(&mut self, ty: &Type) -> CheckResult {
        let (must_start, impossible_start) = (self.musts.len(), self.impossibles.len());
        let effects_start = self.frame.effects.len();
        let depth = self.position.type_depth + 1;
        self.visit_type(ty);
        let mut result = self.causes(must_start, impossible_start, depth);
        match &self.frame.effects[effects_start..] {
            // The type is decided at its own depth, so the first effect of a type with a cause of
            // its own is its own (what is inside it follows): the conflict is what a scope reports
            // (with the note).
            [Effect::Error(conflict), ..] if result.is_own => result = (**conflict).clone(),
            [Effect::Type(pending)] if pending.optional && pending.ty == *ty => {
                result.optional = true
            }
            _ => (),
        }
        result
    }

    fn new_frame(&mut self) -> Frame {
        let frame = Frame::new(
            self.musts.len(),
            self.impossibles.len(),
            self.records.self_errors.len(),
        );
        std::mem::replace(&mut self.frame, frame)
    }

    /// Cuts `impossibles` to `len`, with the positions that the function pointers and `Fn(..)`
    /// being visited recorded in it.
    fn truncate_impossibles(&mut self, len: usize) {
        self.impossibles.truncate(len);
        for positions in &mut self.bound_elided {
            positions.retain(|&at| at < len);
        }
    }

    /// Records an elided lifetime (`cause`, the type or lifetime written) as a soft cause: the
    /// Repeater's impl cannot name it. The function pointer or `Fn(..)` that binds it drops it
    /// ([`AnalyzeVisitor::with_bound_elided()`]).
    fn push_elided(&mut self, cause: Cause) {
        if let Some(positions) = self.bound_elided.last_mut() {
            positions.push(self.impossibles.len());
        }
        self.impossibles.push_soft(cause);
    }

    /// Runs `f`, which visits a function pointer or `Fn(..)`: the elided lifetimes in it are its
    /// own. They stop the types inside from being leaked whole, but not the type that holds it, so
    /// their causes are dropped when `f` ends.
    fn with_bound_elided(&mut self, f: impl FnOnce(&mut Self)) {
        self.bound_elided.push(Vec::new());
        f(self);
        for at in self
            .bound_elided
            .pop()
            .unwrap_or_default()
            .into_iter()
            .rev()
        {
            self.impossibles.remove(at);
        }
    }

    /// Records a cause that depends on the context, at the depth of the type being visited. `path`
    /// is the relative path it is, if it is one (an entry of [`ErrorItem::paths`]).
    fn push_must(&mut self, must: Cause, path: Option<Cause>) {
        self.musts.push(Must {
            depth: self.position.type_depth,
            cause: must,
            covered: false,
            path,
        });
        self.frame.min_depth = self.frame.min_depth.min(self.position.type_depth);
    }

    /// Where the visitor is, for the errors it finds ([`ErrorItem::place`]) and for the rule.
    fn place(&self) -> Place {
        Place {
            in_value: self.position.in_value || self.mode.outer_place.in_value,
            in_default: self.position.in_trait_default
                || self.position.in_param_default
                || self.position.in_nested_default
                || self.position.in_default.is_some()
                || self.mode.outer_place.in_default,
        }
    }

    /// Returns `true` in a bound or where predicate of the item (`Position::env_depth`).
    fn in_predicate(&self) -> bool {
        self.position.env_depth > 0
    }

    /// Returns `true` in a predicate of the item where [`AnalyzeVisitor::env_cycle()`] is checked.
    fn checks_env(&self) -> bool {
        self.in_predicate() && self.mode.check_env
    }

    /// Returns `true` if `path` is a relative path in the sense of the Leaker: it means something
    /// else wherever it is written, so it is leaked, or rejected where it cannot be.
    fn is_relative_path(&self, cause: &Cause) -> bool {
        cause.path().map_or(false, |path| {
            path.leading_colon.is_none()
                && self_head(path).is_none()
                && !self.starts_with_generic(path)
                && !self.is_free_path(cause)
        })
    }

    fn mention_scope(&mut self, scope: usize) {
        if scope == 0 {
            self.frame.mentions_root = true;
        } else {
            self.frame.local_scope = self.frame.local_scope.min(scope);
        }
    }

    fn scope_of(&self, name: &str) -> Option<usize> {
        self.scopes.iter().rposition(|scope| {
            scope
                .params()
                .any(|param| generic_param_name(param) == name)
        })
    }

    /// Returns the context of `ty`, or why it cannot be written (see [`Context::of()`]). `any`: the
    /// entry is for any type although `ty` names the `Self` of a nested struct, enum or union (see
    /// [`Context::of_any()`]).
    fn context_of(
        &self,
        ty: &Type,
        inner: &Frame,
        any: bool,
    ) -> std::result::Result<Context, UnnameableSelf> {
        let context = if inner.mentions_root
            || inner.mentions_self
            || inner.local_scope < self.scopes.len()
        {
            if self.position.in_nested_default && !type_mentions_self(ty) {
                // The copy cannot write `Self` here (E0735): the entry has none, and the copy's own
                // type proves the bounds that name it where it is used.
                let scopes: Vec<Scope> = self
                    .scopes
                    .iter()
                    .map(|scope| {
                        let mut scope = scope.clone();
                        if scope.self_ty().is_some()
                            && scope.self_ty() == self.self_scope.ty.as_ref()
                        {
                            scope.without_self();
                        }
                        scope
                    })
                    .collect();
                Context::of(ty, &scopes, None, false, &self.position.enclosing)?
            } else {
                // A bound that names a nested `Self` makes the entry one for any type.
                Context::of_as(
                    ty,
                    &self.scopes,
                    self.self_scope.ty.as_ref(),
                    self.self_scope.nested,
                    any,
                    &self.position.enclosing,
                )?
            }
        } else {
            Context::none()
        };
        Ok(context)
    }

    /// Returns `true` if `ty`, with `context`, names the `Self` of a nested item
    /// (`SelfScope::nested`).
    fn names_nested_self(&self, ty: &Type, context: &Context) -> bool {
        self.self_scope.nested
            && match &context.self_ty {
                Some(SelfRequirement::Type(_)) => true,
                Some(SelfRequirement::Any) => type_mentions_self(ty),
                _ => false,
            }
    }

    /// Returns the first mention of a nested item's `Self` (`SelfScope::nested`) in the bounds of
    /// `context` that the environment proves ([`self_in_environment_bounds()`]), unless `context`
    /// is already for any type. The entry is then for any type (`own_context()`).
    fn self_in_bounds(&self, context: &Context) -> Option<Cause> {
        if !self.self_scope.nested {
            return None;
        }
        match &context.self_ty {
            Some(SelfRequirement::Any) => None,
            _ => self_in_environment_bounds(context),
        }
    }

    fn output_binder(&self, context: &Context) -> Option<(Lifetime, usize)> {
        self.binders.iter().find_map(|binder| {
            binder
                .output
                .iter()
                .find(|lifetime| {
                    context
                        .generics
                        .params
                        .iter()
                        .any(|param| is_lifetime_param(param, lifetime))
                })
                .map(|lifetime| (lifetime.clone(), binder.owner))
        })
    }

    /// Returns the signature whose elision the copy of a type with `context` would break, with the
    /// lifetimes it adds. It is the innermost one that has the type in its inputs (see
    /// [`Elision`]).
    fn elision_broken(&self, context: &Context) -> Option<(Elision, Vec<Lifetime>)> {
        let elision = self.elisions.last()?.as_ref()?;
        let added = elision.added(context);
        (!added.is_empty()).then(|| (elision.clone(), added))
    }

    /// Like `elision_broken()` for `ty` if it is `Self` or `Self::A`, which `Visitor::with_self()`
    /// writes as the Leaker's type (`::p::S<'a>`). Not for the `Self` of an impl or a receiver,
    /// which the copy keeps, nor where the Visitor can name the lifetime (`Elision::binder`).
    fn self_elision_broken(&self, ty: &Type) -> Option<(Elision, Vec<Lifetime>)> {
        if self.position.in_receiver || self.self_scope.written || !is_self_path(ty) {
            return None;
        }
        let self_ty = match &self.self_scope.ty {
            Some(SelfRequirement::Type(self_ty)) => self_ty,
            _ => return None,
        };
        let elision = self.elisions.last()?.as_ref()?;
        if elision.binder && !elision.hidden {
            return None;
        }
        let added = elision.added_by_self(self_ty);
        (!added.is_empty()).then(|| (elision.clone(), added))
    }

    /// Finishes the outermost scope and returns what it found, with the errors if any. Defaults of
    /// type parameters are analyzed alone, with the parameters before them, and may not leak
    /// `Self`. On an error the result is kept: the part is copied as written.
    pub(crate) fn finish(&mut self, spec: &ScopeSpec) -> (Finished, Vec<ErrorItem>) {
        let generics = &spec.generics;
        let declares_self = spec.declares.is_some();
        let mut effects = std::mem::take(&mut self.frame.effects);
        for (index, _, default) in defaulted(generics) {
            let root = without_defaults(generics, index);
            let (mut found, self_errors) = Self::analyze_default(
                &mut *self.rule,
                default,
                &root,
                declares_self,
                self.mode.check_values,
            );
            // The mentions and the conflicts of the default come after those found so far.
            shift_seen(&mut found, self.records.self_errors.len());
            effects.extend(found);
            self.records.self_errors.extend(self_errors);
        }
        if !self.records.self_errors.is_empty() {
            effects = correct_unmeant_self(effects);
        }
        let errors = scope_errors(
            &effects,
            &self.records.self_errors,
            &self.records.env_errors,
        );
        // The types that a cycle left without an entry are parts too: the Referrer leaves them as
        // written.
        effects.extend(
            std::mem::take(&mut self.records.env_errors)
                .into_iter()
                .filter(|error| error.part.is_some())
                .map(|error| Effect::Error(Box::new(error))),
        );
        let finished = Finished {
            effects,
            top_levels: std::mem::take(&mut self.records.top_levels),
            not_optional: std::mem::take(&mut self.records.not_optional),
        };
        (finished, errors)
    }

    /// Starts the outermost scope.
    pub(crate) fn start(&mut self, spec: &ScopeSpec) {
        self.enter_scope(spec, true);
    }

    /// Marks what a scope or a check records, to roll it back with
    /// [`AnalyzeVisitor::roll_back()`].
    fn mark(&self) -> ScopeMark {
        ScopeMark {
            effects: self.frame.effects.len(),
            records: self.records.mark(),
            scopes: self.scopes.len(),
            item_depth: self.item_depth,
            alone_defaults: self.alone_defaults.len(),
            self_scope: self.self_scope.clone(),
            self_type_walk: self.self_type_walk,
            inherent_impl: self.inherent_impl,
        }
    }

    fn enter_scope(&mut self, spec: &ScopeSpec, root: bool) -> ScopeMark {
        let mark = self.mark();
        let ScopeSpec {
            generics,
            declares,
            trait_path,
        } = spec;
        if let Some(declared) = declares {
            self.self_scope.ty = Some(declared.requirement.clone());
            self.self_scope.written = declared.written;
            self.self_scope.nested = false;
            self.self_scope.trait_ = trait_path.as_deref().map(|path| self.impl_trait(path));
            self.inherent_impl = declared.inherent;
        }
        let gats = spec.declared_gats();
        if root {
            self.scopes[0].set_self(self.self_scope.ty.clone());
            self.scopes[0].set_gats(gats);
        } else {
            self.scopes
                .push(Scope::new(generics, self.self_scope.ty.clone()).with_gats(gats));
        }
        if declares.is_some() {
            self.self_scope.depth = self.scopes.len();
        }
        if self.self_scope.ty.is_none() {
            let generics = without_defaults(generics, generics.params.len());
            let place = self.place();
            self.records.self_errors.extend(
                selves_in_generics(&generics)
                    .into_iter()
                    .map(|cause| SelfMention::new(cause, place)),
            );
        }
        // Checked like the root of a trait bound, before what the closure visits.
        if let Some(path) = trait_path.as_deref() {
            self.check_bound_path(path, Cause::Trait(path.clone()));
        }
        self.item_depth = self.item_depth.max(1);
        let alone = AloneDefaults {
            scopes: self.scopes.len(),
            item_depth: self.item_depth,
            idents: defaulted(generics)
                .map(|(_, ident, _)| ident.clone())
                .collect(),
            declared: declares
                .as_ref()
                .map(|declared| (declared.requirement.clone(), generics.clone())),
        };
        self.alone_defaults.push(alone);
        mark
    }

    /// Leaves a scope: what it declared (the generics, `Self`) is gone, and what it recorded
    /// stays. [`AnalyzeVisitor::roll_back()`] removes that.
    fn leave_scope(&mut self, mark: &ScopeMark) {
        self.scopes.truncate(mark.scopes);
        self.item_depth = mark.item_depth;
        self.alone_defaults.truncate(mark.alone_defaults);
        self.self_scope = mark.self_scope.clone();
        self.self_type_walk = mark.self_type_walk;
        self.inherent_impl = mark.inherent_impl;
    }

    /// Removes what a scope or a check recorded since `mark`: a check, and a scope whose closure
    /// panicked.
    fn roll_back(&mut self, mark: &ScopeMark) {
        self.frame.effects.truncate(mark.effects);
        self.records.truncate(&mark.records);
    }

    /// Starts a check: sets the frame of the scope aside for the visit to use a new one.
    fn checkpoint(&mut self) -> Checkpoint {
        let mark = self.mark();
        let walk = Walk {
            position: self.position.clone(),
            binders: self.binders.len(),
            elisions: self.elisions.len(),
            bound_elided: self.bound_elided.len(),
            signatures: self.signatures.len(),
        };
        Checkpoint {
            mark,
            walk,
            musts: self.musts.len(),
            impossibles: self.impossibles.len(),
            frame: self.new_frame(),
        }
    }

    /// Ends a check, also after a panic: everything the visit recorded is gone.
    fn restore(&mut self, checkpoint: Checkpoint) {
        let Checkpoint {
            mark,
            frame,
            musts,
            impossibles,
            walk,
        } = checkpoint;
        self.frame = frame;
        self.leave_scope(&mark);
        self.roll_back(&mark);
        self.musts.truncate(musts);
        self.truncate_impossibles(impossibles);
        self.impossibles.at = walk.position.type_depth;
        self.position = walk.position;
        self.binders.truncate(walk.binders);
        self.elisions.truncate(walk.elisions);
        self.bound_elided.truncate(walk.bound_elided);
        self.signatures.truncate(walk.signatures);
    }

    /// Leaves a nested scope and returns its top-level context. The scope keeps what it recorded,
    /// and the outermost one reports the errors, in the order found.
    fn end_scope(&mut self, mark: ScopeMark, spec: &ScopeSpec) -> ContextId {
        let generics = &spec.generics;
        for (index, _, default) in defaulted(generics) {
            self.analyze_default_inline(default, &without_defaults(generics, index));
        }
        // The entries that name a `Self` without meaning are copied whole instead.
        if !self.records.self_errors.is_empty() {
            let found = self.frame.effects.split_off(mark.effects);
            self.frame.effects.extend(correct_unmeant_self(found));
        }
        // A scope that declares nothing keeps the `Self` around, so take it before leaving. It is
        // part of the top-level context only if an entry or a part copied as written names it, or
        // has a bound rewritten for it: nothing else depends on it.
        let scope_self = self.self_scope.ty.clone().filter(|declared| {
            names_self(&self.frame.effects[mark.effects..], declared)
                || self
                    .records
                    .env_errors_since(&mark.records)
                    .iter()
                    .any(|error| part_names_self(error, declared))
        });
        // The generic associated types of that `Self`, which an entry that names them depends on.
        let scope_gats = scope_self.as_ref().map_or_else(Vec::new, |declared| {
            self.scopes
                .iter()
                .rev()
                .find(|scope| scope.self_ty() == Some(declared) && !scope.gats().is_empty())
                .map_or_else(Vec::new, |scope| scope.gats().to_vec())
        });
        self.leave_scope(&mark);
        let declared = spec.generics.clone();
        let context = Context::new(declared.clone(), scope_self)
            .with_gats(scope_gats)
            .with_declared(declared);
        let top_level = self.records.contexts.intern(context);
        self.records.top_levels.push(top_level);
        top_level
    }

    /// Analyzes alone each default of `generics`, the item's, that does not name `Self` (a trait's
    /// `Rhs = Self` is for the implementors). Returns the parameters whose default the walk of the
    /// item must not analyze again.
    fn analyze_defaults_alone(&mut self, generics: &Generics) -> Vec<Ident> {
        let mut alone = Vec::new();
        for (index, ident, default) in defaulted(generics) {
            if !type_mentions_self(default) {
                self.analyze_default_inline(default, &without_defaults(generics, index));
                alone.push(ident.clone());
            }
        }
        alone
    }

    fn analyze_default_inline(&mut self, ty: &Type, before: &Generics) {
        // The root scope is always there.
        let at = self.scopes.len() - 1;
        let scope = std::mem::replace(&mut self.scopes[at], Scope::new(before, None));
        let self_ty = self.self_scope.ty.take();
        let in_default = self.position.in_default.replace(self_ty.is_some());
        self.visit_type(ty);
        self.position.in_default = in_default;
        self.self_scope.ty = self_ty;
        self.scopes[at] = scope;
    }

    /// Records `cause`, a mention of `Self`, as an error if `Self` has no meaning here, also in the
    /// default of a parameter of a scope that declares nothing.
    fn check_self_has_meaning(&mut self, cause: Cause) {
        if !self.self_has_meaning() {
            self.records
                .self_errors
                .push(SelfMention::new(cause, self.place()));
        }
    }

    /// Returns `true` if `Self` means something here: a scope declares it, also in the default of a
    /// parameter of a scope that does (which cannot name it, but is not an error for that).
    fn self_has_meaning(&self) -> bool {
        self.position
            .in_default
            .unwrap_or_else(|| self.self_scope.ty.is_some())
    }

    /// Returns `true` if the type whose frame is `inner`, an entry in `context`, names a `Self`
    /// without meaning, or a bound of `context` does: the impl of the Repeater would read it as its
    /// implementor (see `correct_unmeant_self()`).
    fn names_unmeant_self(&self, inner: &Frame, context: &Context) -> bool {
        self.records.self_errors.len() > inner.self_start
            || (!self.self_has_meaning() && !selves_in_generics(&context.generics).is_empty())
    }

    /// Like `decide()` for a type that cannot be leaked alone (a trait object inheriting a path
    /// lifetime, a type in an array length): its causes become those of `owner`, the enclosing type
    /// leaked whole.
    fn decide_as_owner(&mut self, owner: usize, mut inner: Frame) {
        if inner.min_depth != usize::MAX {
            self.own_causes(inner.must_start, owner);
            inner.min_depth = owner;
        }
        if inner.optional_depth != usize::MAX {
            inner.optional_depth = owner;
        }
        if inner.min_depth != usize::MAX || inner.optional_depth != usize::MAX {
            inner.adopted.push(owner);
        }
        self.decide_through(inner)
    }

    /// Like `decide()` for a type that cannot be leaked alone and has no cause: only its parts are
    /// leaked.
    fn decide_through(&mut self, mut inner: Frame) {
        self.absorb(&inner);
        self.frame.effects.append(&mut inner.effects);
    }

    fn hand_off(&mut self, to: Handoff, inner: Frame) {
        match to {
            Handoff::Owner(owner) => self.decide_as_owner(owner, inner),
            Handoff::Through => self.decide_through(inner),
        }
    }

    fn absorb(&mut self, inner: &Frame) {
        self.frame.min_depth = self.frame.min_depth.min(inner.min_depth);
        self.frame.optional_depth = self.frame.optional_depth.min(inner.optional_depth);
        self.frame.local_scope = self.frame.local_scope.min(inner.local_scope);
        self.frame.mentions_root |= inner.mentions_root;
        self.frame.mentions_self |= inner.mentions_self;
        if self.frame.assoc_self.is_none() {
            self.frame.assoc_self = inner.assoc_self.clone();
        }
        // Kept for the enclosing type, unless that is the type holding the binder.
        if self.frame.binder.is_none() {
            let depth = self.position.type_depth;
            self.frame.binder = inner.binder.clone().filter(|(owner, _)| *owner < depth);
        }
        if self.frame.elision.is_none() {
            let depth = self.position.type_depth;
            self.frame.elision = inner.elision.clone().filter(|(owner, _)| *owner < depth);
        }
        for &owner in &inner.adopted {
            push_once(&mut self.frame.adopted, owner);
        }
    }

    fn own_causes(&mut self, must_start: usize, depth: usize) {
        for must in &mut self.musts[must_start..] {
            must.depth = depth;
        }
    }

    /// Returns the context of `ty` at `depth` if it can be leaked whole for its own path, else the
    /// conflict (its path should be absolute).
    #[allow(clippy::result_large_err)]
    fn own_context(
        &self,
        ty: &Type,
        depth: usize,
        inner: &Frame,
    ) -> std::result::Result<Context, CheckResult> {
        if self.impossibles.len() > inner.impossible_start {
            return Err(self.impossible_conflict(ty, depth, inner));
        }
        if self.position.in_default == Some(true) && inner.mentions_self {
            // The default is also used to compute the arguments of the trait, so the Repeater impl
            // cannot know what `Self` is.
            let paths = self.own_paths(inner.must_start, depth);
            return Err(self_in_default_conflict(ty, paths));
        }
        let context = match self.context_of(ty, inner, false) {
            Ok(context) => context,
            Err(unnameable) => {
                let paths = self.own_paths(inner.must_start, depth);
                return Err(unnameable_conflict(ty, unnameable, paths));
            }
        };
        let self_in_bounds = self.self_in_bounds(&context);
        // A bound names a nested `Self`: the entry is for any type (`SelfScope::nested`). Not in
        // a default, where the copy cannot write `Self` (E0735).
        if self_in_bounds.is_some() && !self.position.in_nested_default {
            if let Ok(any) = self.context_of(ty, inner, true) {
                return match self.associated_self_in_bounds(&any) {
                    Some(conflict) => Err(conflict.of(ty, self.own_paths(inner.must_start, depth))),
                    None => Ok(any),
                };
            }
        }
        match self_in_bounds {
            Some(self_path) if self.position.in_nested_default => {
                let paths = self.own_paths(inner.must_start, depth);
                Err(nested_default_conflict(ty, self_path, paths))
            }
            // Outside a default the `if` above has returned unless `of_any()` failed, and it names
            // fewer generics than the `of()` that gave `context`.
            Some(_) => unreachable!("of_any() fails where of() succeeded"),
            None => match self.associated_self_in_bounds(&context) {
                Some(conflict) => Err(conflict.of(ty, self.own_paths(inner.must_start, depth))),
                None => Ok(context),
            },
        }
    }

    /// The conflict of `ty` at `depth`, which an impossible cause in it keeps from being leaked
    /// whole, with the advice of the binder or the elision that `ty` holds.
    fn impossible_conflict(&self, ty: &Type, depth: usize, inner: &Frame) -> CheckResult {
        let mut conflict = self.conflict_causes(inner.must_start, inner.impossible_start, depth);
        let advice = inner
            .binder
            .iter()
            .chain(&inner.elision)
            .find(|(owner, _)| *owner == depth);
        if let Some((_, advice)) = advice {
            advice.clone().give(&mut conflict);
        }
        if let (None, Some(lifetime)) = (&conflict.note, &inner.unconstrained) {
            unconstrained_advice(&mut conflict, lifetime, true);
        }
        if let (Some(cause), Some(SelfRequirement::Type(_))) =
            (&inner.assoc_self, &self.self_scope.ty)
        {
            AssocSelf::new(cause.clone(), self.self_scope.trait_.clone()).give(
                &mut conflict,
                ty,
                false,
            );
        }
        conflict
    }

    /// Returns the first `Self::A` in the bounds of the generics of `context`, if `Self` is a
    /// type: the Repeater's impl for it would write the bound as it is, and has no trait to take
    /// `A` from. (The impl for the implementor of a trait can write it: `__S::A`.)
    fn associated_self_in_bounds(&self, context: &Context) -> Option<AssocSelf> {
        match &context.self_ty {
            Some(SelfRequirement::Type(_) | SelfRequirement::Any) => {
                first_associated_self_in_generics(&context.generics)
                    .map(|cause| AssocSelf::new(cause, self.self_scope.trait_.clone()))
            }
            _ => None,
        }
    }

    /// Returns `true` if a bare identifier argument in `ty` may be a const, and `ty` can be leaked
    /// whole.
    fn is_ambiguous_owner(&self, ty: &Type, depth: usize, inner: &Frame) -> bool {
        holds_bare_ident(ty, inner) && self.own_context(ty, depth, inner).is_ok()
    }

    /// Returns `true` if the type of `inner` cannot be leaked whole for a hard cause
    /// ([`Impossibles`]), or for one that is no impossible cause (`Self` in a bound or a default).
    fn has_hard_cause(&self, inner: &Frame) -> bool {
        self.impossibles.len() <= inner.impossible_start
            || self.impossibles.any_hard_since(inner.impossible_start)
    }

    /// Decides the type `ty` at `depth`, whose parts were visited into `inner`: an entry leaked
    /// whole, copied as written with its parts leaked, or a conflict. The phases run in order; one
    /// that finds the owner of the type hands the frame over.
    #[allow(clippy::result_large_err)]
    fn decide(&mut self, ty: &Type, depth: usize, ambiguous: bool, mut inner: Frame) {
        let self_conflict = match self.self_elision(ty, depth, &mut inner) {
            Ok(conflict) => conflict,
            Err(to) => return self.hand_off(to, inner),
        };
        let bare = self.bare_identifiers(ty, depth, &mut inner);
        let mut own = match self_conflict {
            Some(conflict) => Some(Err(conflict)),
            None => (inner.min_depth == depth).then(|| self.own_context(ty, depth, &inner)),
        };
        if let (Some(idents), Some(Err(conflict))) = (&bare, &own) {
            own = Some(Err(bare_conflict(conflict.clone(), ty, idents)));
        }
        let bare = bare.is_some();
        let mark = match self.binder_and_elision(ty, depth, &mut inner, &mut own) {
            Ok(mark) => mark,
            Err(to) => return self.hand_off(to, inner),
        };
        let optional = match self.optional_context(ty, depth, &mut inner, own.is_none()) {
            Ok(optional) => optional,
            Err(to) => return self.hand_off(to, inner),
        };
        self.absorb(&inner);
        let has_impossible = self.impossibles.len() > inner.impossible_start;
        let has_must = inner.min_depth != usize::MAX;
        let mentions_local = inner.local_scope < self.scopes.len();
        // A conflict that holds a bare identifier, which may be a const, is copied whole, as
        // `bare_identifiers()` makes one without a cause of its own.
        let holds_bare = holds_bare_ident(ty, &inner) && self.has_hard_cause(&inner);
        let effects = std::mem::take(&mut inner.effects);

        if own.is_none() && self.checks_env() && self.self_in_predicate(ty, depth, &inner, &effects)
        {
            return;
        }
        match own {
            Some(Ok(context)) => {
                self.pending_for_entry(ty, depth, ambiguous, context, &inner, effects)
            }
            Some(Err(conflict)) => {
                // The type is copied as written: the Referrer leaves it so. One whose context
                // cannot be written has none, and is found where the Referrer cannot write it.
                let context = self.context_of(ty, &inner, false).ok();
                // Whole for a bare identifier (it may be a const: not leaked alone, and the errors
                // of the parts stay), and in a predicate (a leaked type there is a projection that
                // rustc cannot normalize while it checks the bounds, `env_cycle()`).
                let whole = bare
                    || holds_bare
                    || inner.adopted.contains(&depth)
                    || self.in_predicate()
                    || is_whole_reference(ty);
                let mut conflict = self.with_part(conflict, ty, context, whole, &effects);
                if let Some(part) = conflict.part.as_mut() {
                    part.mark = mark;
                }
                if bare {
                    let mut inside = error_effects(effects);
                    self.adopt_inside(&mut inside, &conflict, inner.self_start);
                    self.frame.effects.extend(inside);
                    return self.push_error(conflict);
                }
                self.conflict_inside(conflict, inner.self_start, whole, effects)
            }
            None if has_impossible => {
                if has_must || has_optional_candidate(&effects) {
                    // Not leaked whole (elided lifetime, `impl Trait`), but its parts can be.
                    self.push_parts(ty, depth, &inner, effects);
                } else {
                    // Cannot and need not be leaked: copied, so primitive names are assumed.
                    self.frame.effects.extend(primitive_effects(effects));
                }
            }
            // Not leaked itself: it depends on local generics or on `Self` in a default. Its parts
            // can be.
            None if mentions_local
                || (self.position.in_default.is_some() && inner.mentions_self) =>
            {
                match optional {
                    Some(context) => {
                        let pending = PendingType {
                            optional: true,
                            optional_candidate: true,
                            ..PendingType::new(ty, context, effects)
                        };
                        self.push_pending(pending, None, &inner, depth)
                    }
                    None => self.frame.effects.extend(effects),
                }
            }
            None => self.pending_for_parts(ty, depth, optional, has_must, &inner, effects),
        }
    }

    /// Like [`AnalyzeVisitor::binder_and_elision()`] for a `Self` or `Self::A` that breaks the
    /// elision (`self_elision_broken()`): `Self` is no entry, so the type holding the signature is
    /// leaked whole instead (`Err`), or it is a conflict (in a bound or a method).
    fn self_elision(
        &mut self,
        ty: &Type,
        depth: usize,
        inner: &mut Frame,
    ) -> std::result::Result<Option<CheckResult>, Handoff> {
        let (elision, added) = match self.self_elision_broken(ty) {
            Some(broken) => broken,
            None => return Ok(None),
        };
        let at = if elision.owner > 0 {
            elision.owner
        } else {
            depth
        };
        self.musts.push(Must {
            depth: at,
            cause: Cause::Type(ty.clone()),
            covered: false,
            path: None,
        });
        inner.min_depth = inner.min_depth.min(at);
        if elision.owner > 0 {
            let advice = self_elision_advice(&added, &elision, true);
            inner.elision = Some((elision.owner, advice));
            return Err(Handoff::Owner(elision.owner));
        }
        let advice = self_elision_advice(&added, &elision, false);
        let conflict = self.owned_conflict(inner, depth, elision.output, advice);
        Ok(Some(conflict))
    }

    /// The conflict of the type at `depth` that owns the causes of `inner`, with `first` as its
    /// first impossible cause and `advice` given.
    fn owned_conflict(
        &self,
        inner: &Frame,
        depth: usize,
        first: Cause,
        advice: Advice,
    ) -> CheckResult {
        let mut conflict = self.conflict_causes(inner.must_start, inner.impossible_start, depth);
        conflict.impossible.insert(0, first);
        advice.give(&mut conflict);
        conflict
    }

    /// A bare identifier argument may be a const, so the type holding it is leaked whole if it can
    /// be; under a soft cause the identifier is leaked as a type ([`Impossibles`]). Returns the
    /// identifiers if the type is a conflict for them; the type owns its causes then.
    fn bare_identifiers(
        &mut self,
        ty: &Type,
        depth: usize,
        inner: &mut Frame,
    ) -> Option<Vec<Type>> {
        if inner.min_depth == depth || !holds_bare_ident(ty, inner) {
            return None;
        }
        let bare = match self.own_context(ty, depth, inner) {
            Ok(_) => None,
            Err(_) if self.has_hard_cause(inner) => Some(ambiguous_types(&inner.effects)),
            Err(_) => return None,
        };
        self.own_causes(inner.must_start, depth);
        inner.min_depth = depth;
        bare
    }

    /// A type leaked whole (`own`) that uses an output's binder lifetime ([`Binder`]) or breaks an
    /// elision ([`Elision`]): its holder is leaked instead (`Err`), or it is a conflict. Returns
    /// the mark of the part that a method's signature makes the type.
    fn binder_and_elision(
        &mut self,
        ty: &Type,
        depth: usize,
        inner: &mut Frame,
        own: &mut Option<std::result::Result<Context, CheckResult>>,
    ) -> std::result::Result<Option<SignatureMark>, Handoff> {
        if let (Some(_), Some(probe)) = (&*own, &mut self.position.binder_probe) {
            probe.insert(ty as *const Type as usize);
        }
        let binder = match &*own {
            Some(Ok(context)) => self.output_binder(context),
            _ => None,
        };
        if let Some((lifetime, owner)) = binder {
            if owner > 0 {
                inner.binder = Some((owner, binder_advice(&lifetime, ty, true)));
                return Err(Handoff::Owner(owner));
            }
            let advice = binder_advice(&lifetime, ty, false);
            let conflict = self.owned_conflict(inner, depth, Cause::Lifetime(lifetime), advice);
            *own = Some(Err(conflict));
        }
        let broken = match &*own {
            Some(Ok(context)) => self.elision_broken(context),
            _ => None,
        };
        // The signature of a method is what makes the type a part: the Referrer leaves it only in
        // a signature that relies on elision as well.
        let mut mark = None;
        if let Some((elision, added)) = broken {
            if elision.owner > 0 {
                let advice = elision_advice(&added, &elision, ty, true);
                inner.elision = Some((elision.owner, advice));
                return Err(Handoff::Owner(elision.owner));
            }
            let advice = elision_advice(&added, &elision, ty, false);
            mark = (!elision.binder).then(|| SignatureMark::Elided);
            let conflict = self.owned_conflict(inner, depth, elision.output, advice);
            *own = Some(Err(conflict));
        }
        if let Some(Ok(_)) = own {
            self.record_leaked(ty, depth, inner);
        }
        Ok(mark)
    }

    /// The context of `ty` as an optional entry, if it has no own decision (`no_own`) nor cause
    /// and can be leaked whole. If it uses a binder or elided lifetime, its holder is the optional
    /// entry instead (`Err`; only its parts are leaked here).
    fn optional_context(
        &mut self,
        ty: &Type,
        depth: usize,
        inner: &mut Frame,
        no_own: bool,
    ) -> std::result::Result<Option<Context>, Handoff> {
        let optional = if no_own && inner.optional_depth == depth && !self.position.in_receiver {
            self.own_context(ty, depth, inner).ok()
        } else {
            None
        };
        let optional_binder = optional
            .as_ref()
            .and_then(|context| self.output_binder(context));
        if let Some((_, owner)) = optional_binder {
            if owner > 0 {
                inner.optional_depth = owner;
                return Err(Handoff::Through);
            }
        }
        let optional_elision = optional
            .as_ref()
            .and_then(|context| self.elision_broken(context));
        if let Some((elision, _)) = &optional_elision {
            if elision.owner > 0 {
                inner.optional_depth = elision.owner;
                return Err(Handoff::Through);
            }
        }
        // An optional entry that would make a cycle in the copy (see `env_cycle()`) is not one: the
        // Referrer cannot replace it, so it is copied as written. That holds for the type in this
        // context everywhere, as the Referrer finds an entry by its type and context.
        Ok(
            match optional.filter(|_| optional_binder.is_none() && optional_elision.is_none()) {
                Some(context) if self.env_blocked(&context, true) => {
                    self.records
                        .not_optional
                        .push((context, quote!(#ty).to_string()));
                    None
                }
                // An absolute type that names a nested `Self` is copied as written
                // (`SelfScope::nested`). The types in it are still leaked as usual.
                Some(context) if self.names_nested_self(ty, &context) => None,
                optional => optional,
            },
        )
    }

    /// Checks `ty`, a `Self` or `Self::A` in a predicate, which the copy resolves through what it
    /// writes for `Self`. Returns whether it recorded the error that leaves `ty` without an entry,
    /// kept apart from `effects`, which the type around may drop.
    fn self_in_predicate(
        &mut self,
        ty: &Type,
        depth: usize,
        inner: &Frame,
        effects: &[Effect],
    ) -> bool {
        // The type inside parentheses or a group has been decided: it has the error, once.
        if matches!(ty, Type::Paren(_) | Type::Group(_)) {
            return false;
        }
        let projection = is_self_projection(ty);
        if !(projection || (self.self_scope.written && is_bare_self(ty))) {
            return false;
        }
        let context = match self.context_of(ty, inner, false) {
            Ok(context) => context,
            Err(_) => return false,
        };
        // The self type of an impl needs a rewrite only if it names a leaked type.
        if projection || self.self_type_names_leaked(&context, false) {
            // `Self::A` needs the bounds of its context, `Self` of an impl those of the entries in
            // the self type. One that the graph may choose, with unstable bounds, is left to its
            // parts if the walk of the impl can tell the graph (`block_unstable_self_types()`).
            let blockable = !projection && self.self_type_walk.is_some();
            if blockable {
                self.self_type_walk = Some(true);
            }
            let needs = if projection {
                vec![context.clone()]
            } else {
                self.self_type_entries(&context, false, blockable)
            };
            for needed in &needs {
                if let Some(conflict) = self.self_env_cycle(ty, needed) {
                    // The type is left without an entry, and copied as written: `Self`
                    // or `Self::A` has no types in it to leak.
                    let mut conflict = self.with_part(conflict, ty, Some(context), true, effects);
                    conflict.seen = self.records.self_errors.len();
                    conflict.place = self.place();
                    conflict.paths = self.own_paths(inner.must_start, depth);
                    self.records.env_errors.push(conflict);
                    return true;
                }
            }
            if projection {
                self.not_optional_in_bounds(&context);
            } else {
                for context in self.self_type_entries(&context, true, blockable) {
                    self.not_optional_in_bounds(&context);
                }
            }
            if let Some(impl_context) = self.impl_context(projection) {
                self.not_optional_in_bounds(&impl_context);
            }
        }
        if !projection {
            self.not_optional_in_self_type(&context);
        }
        false
    }

    /// `ty` is leaked whole in `context` for its own path: pushes the entry, with the error of the
    /// risk it has, or a conflict if it would make a cycle.
    fn pending_for_entry(
        &mut self,
        ty: &Type,
        depth: usize,
        ambiguous: bool,
        context: Context,
        inner: &Frame,
        effects: Vec<Effect>,
    ) {
        let cycle = self
            .env_cycle(ty, &context)
            .or_else(|| self.nested_self_cycle(ty, &context, inner));
        if let Some(conflict) = cycle {
            // The predicate must not hold a projection through the Repeater, also not in a
            // part of the type that conflicts (E0277: rustc does not normalize it while it
            // checks the bounds): nothing in it is leaked.
            let mut conflict = self.with_part(conflict, ty, Some(context), true, &effects);
            conflict.paths = self.own_paths(inner.must_start, depth);
            return self.conflict_inside(conflict, inner.self_start, true, effects);
        }
        self.not_optional_in_bounds(&context);
        let risk = self
            .nested_self_risk(ty, &context)
            .then(|| self.nested_self_error(ty, &context, inner));
        let pending = PendingType {
            required: true,
            required_candidate: true,
            ambiguous: ambiguous && context.is_empty(),
            primitives: primitive_names(&effects),
            inside: effects,
            // A type in it is not leaked alone: its meaning depends on this one.
            adopted: inner.adopted.contains(&depth),
            ..PendingType::new(ty, context, Vec::new())
        };
        self.push_pending(pending, risk, inner, depth)
    }

    /// `ty` is not leaked for its own path and has no conflict, but its parts have causes or
    /// optional entries: pushes it as a candidate for an entry (`optional`: an optional entry
    /// already), or only its parts if its generics cannot be written.
    fn pending_for_parts(
        &mut self,
        ty: &Type,
        depth: usize,
        optional: Option<Context>,
        has_must: bool,
        inner: &Frame,
        effects: Vec<Effect>,
    ) {
        // Not leaked itself if its generics cannot be written (`Self`).
        let context = match self.context_of(ty, inner, false) {
            Ok(context)
                if self.self_in_bounds(&context).is_none()
                    && self.associated_self_in_bounds(&context).is_none() =>
            {
                context
            }
            _ => return self.frame.effects.extend(effects),
        };
        // It may replace its parts as the entry, if the graph finds it cheaper: not if it
        // adds lifetimes where elision needs none.
        let required_candidate = has_must && self.elision_broken(&context).is_none();
        // A type that is an entry only if its parts need it makes a cycle too, but its
        // parts can be the entries instead, unless it is leaked whole in any case.
        let leaked_whole = has_must && is_whole_reference(ty);
        if leaked_whole {
            if let Some(conflict) = self.env_cycle(ty, &context) {
                let mut conflict = self.with_part(conflict, ty, Some(context), true, &effects);
                conflict.paths = self.own_paths(inner.must_start, depth);
                return self.push_error(conflict);
            }
            self.not_optional_in_bounds(&context);
        }
        let optional_candidate = optional.is_some() || has_optional_candidate(&effects);
        // A type that names a nested `Self` is no entry for its parts either: `Self` in it is the
        // copy's own (`SelfScope::nested`).
        let nested_self = self.names_nested_self(ty, &context);
        let no_required_entry = has_must && (nested_self || self.env_blocked(&context, false));
        if has_must && !no_required_entry {
            self.not_optional_in_bounds(&context);
        }
        let no_optional_entry =
            optional_candidate && (nested_self || self.env_blocked(&context, true));
        // A reference that is an entry is one as a whole, so its parts cannot stand for
        // it. If it cannot be an entry, the optional entries in it are not either.
        if no_optional_entry && is_whole_reference(ty) {
            self.records.not_optional.extend(optional_keys(&effects));
        }
        let pending = PendingType {
            required_candidate,
            optional: optional.is_some(),
            optional_candidate,
            no_required_entry,
            no_optional_entry,
            ..PendingType::new(ty, context, effects)
        };
        self.push_pending(pending, None, inner, depth)
    }

    /// Pushes `pending`, the effect of the type at `depth`, after `risk`, the error that goes
    /// before it as a conflict does.
    fn push_pending(
        &mut self,
        mut pending: PendingType,
        risk: Option<CheckResult>,
        inner: &Frame,
        depth: usize,
    ) {
        pending.unmeant_self = self.names_unmeant_self(inner, &pending.context);
        pending.in_predicate = self.in_predicate();
        // An entry leaked alone covers its paths: the type around is not in conflict for them.
        if pending.required_candidate {
            pending.paths = self.own_paths(inner.must_start, depth);
            for must in &mut self.musts[inner.must_start..] {
                must.covered = true;
            }
        }
        if let Some(risk) = risk {
            self.frame.effects.push(Effect::Error(Box::new(risk)));
        }
        self.frame.effects.push(Effect::Type(Box::new(pending)));
    }

    /// Pushes `effects`, those of the parts of `ty` at `depth`, which an own cause (an elided
    /// lifetime, `impl Trait`) stops from being leaked whole. The parts among them are copied as
    /// written inside `ty` only: it is their anchor, unless its context cannot be written.
    fn push_parts(&mut self, ty: &Type, depth: usize, inner: &Frame, mut effects: Vec<Effect>) {
        if self.impossibles.own_since(inner.impossible_start, depth) && has_error(&effects) {
            if let Ok(context) = self.context_of(ty, inner, false) {
                anchor_parts(&mut effects, ty, &context);
            }
        }
        self.frame.effects.extend(effects);
    }

    /// Gives `conflict`, about `ty` in `context` (`None`: it cannot be written), the part that the
    /// Referrer leaves as written (see [`Part`]). `whole`: nothing in `ty` is leaked, so the copy
    /// writes the primitive names of `effects`, those of its components.
    fn with_part(
        &self,
        mut conflict: CheckResult,
        ty: &Type,
        context: Option<Context>,
        whole: bool,
        effects: &[Effect],
    ) -> CheckResult {
        let primitives = if whole {
            primitive_names(effects)
        } else {
            Vec::new()
        };
        let cause = Cause::Type(ty.clone());
        conflict.part = Some(Part::new(
            cause,
            whole,
            context,
            self.in_predicate(),
            primitives,
        ));
        conflict
    }

    /// Records `conflict`, about a type copied as written, then what was found inside it
    /// (`effects`, and the mentions of `Self` from `self_start`): the types in it are leaked as
    /// usual, but in a `whole` part only their conflicts stay.
    fn conflict_inside(
        &mut self,
        mut conflict: CheckResult,
        self_start: usize,
        whole: bool,
        effects: Vec<Effect>,
    ) {
        conflict.seen = self_start;
        conflict.place = self.place();
        let mut effects = if whole {
            as_written(effects, &mut conflict.paths)
        } else {
            effects
        };
        self.adopt_inside(&mut effects, &conflict, self_start);
        self.frame.effects.push(Effect::Error(Box::new(conflict)));
        self.frame.effects.extend(effects);
    }

    /// Makes the errors in `effects`, and the mentions of `Self` without meaning from `self_start`
    /// on, errors revealed inside the part of `conflict` ([`Origin::outer`]).
    fn adopt_inside(&mut self, effects: &mut [Effect], conflict: &CheckResult, self_start: usize) {
        let token = &conflict.origin.token;
        adopt(effects, token);
        for mention in &mut self.records.self_errors[self_start..] {
            mention.outer.get_or_insert_with(|| token.clone());
        }
    }

    /// Returns the part for a bound whose trait path conflicts (`Rel<R0>` in `T: Rel<R0>`), which
    /// the copy writes as it is with the types in it. `path` stands for it as a type to compute the
    /// context (`None` if it cannot be written). `primitives`: the primitive names in it.
    fn bound_part(&self, cause: Cause, path: &Path, primitives: Vec<Ident>) -> Part {
        let ty = path_type(path);
        let context = Context::of(
            &ty,
            &self.scopes,
            self.self_scope.ty.as_ref(),
            self.self_scope.nested,
            &Enclosing::default(),
        )
        .ok();
        Part::new(cause, true, context, self.in_predicate(), primitives)
    }

    /// Returns the conflict if `ty`, leaked whole in `context`, would make a cycle in the copy: it
    /// is in a predicate of the item and a bound of its generics is unstable (see
    /// [`crate::unstable`]), or it names `Self` in a predicate of the trait that declares it.
    fn env_cycle(&mut self, ty: &Type, context: &Context) -> Option<CheckResult> {
        match self.env_unstable_bound(context, false) {
            Some(unstable) => {
                let why = match &unstable.why {
                    Instability::Leaked => format!(
                        "the bound `{}` of its generics names a type that is leaked too: the copy \
                         writes both as paths through the Repeater, and rustc cannot normalize the \
                         one in a bound or where predicate while it checks the bounds (E0277, \
                         E0275)",
                        tidy(&unstable.bound)
                    ),
                    Instability::Named(named) => format!(
                        "the bound `{}` of its generics {}",
                        tidy(&unstable.bound),
                        named.explain()
                    ),
                };
                Some(unstable_error(ty, &unstable, why))
            }
            None => {
                let bound = self.trait_cycle(context)?;
                let why = format!(
                    "the copy writes it as a path through the Repeater, whose impl is for the \
                     implementors of the trait (`{}`), and rustc cannot prove that the host \
                     implements the trait while it checks this predicate (E0275)",
                    tidy(&bound)
                );
                let mut error = cycle_error(ty, why);
                error.headline = Some(format!(
                    "`{}` names `Self` in a bound or where predicate of its trait",
                    tidy(&quote!(#ty))
                ));
                error.reason = Reason::SelfInTraitPredicate;
                // The type names `Self`, so an absolute path is no fix.
                error.help = Some(format!(
                    "write the predicate of the trait without `{0}`, or use `{0}` outside the \
                     predicates of the trait (in a method, a field or a signature)",
                    tidy(&quote!(#ty))
                ));
                Some(error)
            }
        }
    }

    /// Returns `true` if `ty`, an entry for any type in `context`, names a nested item's `Self`
    /// (`Rel2<Self, T>` with `T: AsRef<Self>`, see `SelfScope::nested`): a risk, as what `ty`
    /// needs of `Self` is not known ([`Reason::NestedSelfInBounds`]).
    fn nested_self_risk(&self, ty: &Type, context: &Context) -> bool {
        self.self_scope.nested
            && matches!(context.self_ty, Some(SelfRequirement::Any))
            && type_mentions_self(ty)
    }

    /// The error of an entry for which [`AnalyzeVisitor::nested_self_risk()`] holds. The type is
    /// leaked all the same: nothing in it is copied as written, so the error has no paths.
    fn nested_self_error(&self, ty: &Type, context: &Context, inner: &Frame) -> CheckResult {
        let self_path = self
            .context_of(ty, inner, false)
            .ok()
            .and_then(|plain| self.self_in_bounds(&plain))
            .unwrap_or_else(Cause::self_type);
        CheckResult {
            origin: Origin {
                entry: Some((context.clone(), ty.clone())),
                ..Origin::default()
            },
            note: Some(
                "it names the `Self` of an item nested in the analyzed syntax, and so do the \
                 bounds of its generics: the entry is for any type that the bound holds for, and \
                 the copy passes its own type as `Self`. That is right only if the declaration of \
                 the type has no trait bound on the parameter that receives `Self` (E0277 in the \
                 Leaker's crate), and, on rustc before 1.70, if the copy's field is not the last \
                 field of a struct (E0275)"
                    .to_string(),
            ),
            help: Some(format!(
                "write the bound of the generics of `{}` without `{}`, write the type without \
                 `Self`, or write the type with an absolute path (it is copied as written, and \
                 its `Self` is the copy's own)",
                tidy(&quote!(#ty)),
                tidy(&self_path.shown())
            )),
            headline: Some(format!(
                "`{}` names `Self` and has a generic whose bound mentions `Self`",
                tidy(&quote!(#ty))
            )),
            seen: inner.self_start,
            place: self.place(),
            ..CheckResult::conflict(
                vec![Cause::Type(ty.clone())],
                Vec::new(),
                Reason::NestedSelfInBounds,
            )
        }
    }

    /// Like `env_cycle()` for an entry for which `nested_self_risk()` holds, judged with the
    /// generics that `Self` names: `context` leaves out the generics `ty` does not use, whose
    /// bounds may name a leaked type that another entry's impl needs (E0277).
    fn nested_self_cycle(
        &mut self,
        ty: &Type,
        context: &Context,
        inner: &Frame,
    ) -> Option<CheckResult> {
        if !self.nested_self_risk(ty, context) {
            return None;
        }
        let plain = self.context_of(ty, inner, false).ok()?;
        self.env_cycle(ty, &plain)
    }

    /// Like `env_cycle()` for `Self::A` and for `Self` of an impl. `Self::A` needs the bounds of
    /// its context, and of the impl if it is in one. For `Self`, `context` is that of a type in
    /// the self type that the copy writes through the Repeater (`self_type_entries()`).
    fn self_env_cycle(&mut self, ty: &Type, context: &Context) -> Option<CheckResult> {
        let projection = is_self_projection(ty);
        let (unstable, owner) = match self.env_unstable_bound(context, false) {
            Some(unstable) => (unstable, "its generics"),
            None => {
                let impl_context = self.impl_context(projection)?;
                (self.env_unstable_bound(&impl_context, false)?, "the impl")
            }
        };
        let why = match (&unstable.why, projection) {
            (Instability::Leaked, true) => format!(
                "the copy for a concrete implementor resolves `{}` by the impl of the trait, \
                 which needs the bound `{}` of {}, where the copy writes the leaked type \
                 as a path through the Repeater, and rustc cannot normalize the one in a bound or \
                 where predicate while it checks the bounds (E0277, E0275)",
                tidy(&quote!(#ty)),
                tidy(&unstable.bound),
                owner
            ),
            (Instability::Leaked, false) => format!(
                "`Self` is the self type of the impl, which the copy writes with its leaked types \
                 as paths through the Repeater, and the bound `{}` of their generics names a type \
                 that is leaked too, and rustc cannot normalize the one in a bound or where \
                 predicate while it checks the bounds (E0277, E0275)",
                tidy(&unstable.bound)
            ),
            (Instability::Named(named), true) => format!(
                "the copy for a concrete implementor resolves `{}` by the impl of the trait, \
                 which needs the bound `{}` of {}, which {}",
                tidy(&quote!(#ty)),
                tidy(&unstable.bound),
                owner,
                named.explain()
            ),
            (Instability::Named(named), false) => format!(
                "`Self` is the self type of the impl, which the copy writes with its leaked types \
                 as paths through the Repeater, and the bound `{}` of their generics {}",
                tidy(&unstable.bound),
                named.explain()
            ),
        };
        Some(unstable_error(ty, &unstable, why))
    }

    /// Returns the generics of the impl that declares `Self`, if `Self::A` (`projection`) is in
    /// one of its predicates, as a context.
    fn impl_context(&self, projection: bool) -> Option<Context> {
        if !projection || !self.self_scope.written || self.self_scope.depth == 0 {
            return None;
        }
        let scope = self.scopes.get(self.self_scope.depth - 1)?;
        Some(Context::new(scope.generics(), self.self_scope.ty.clone()))
    }

    /// Returns `Self: Trait` if a type in `context` names `Self` of a trait, and the predicates of
    /// that trait are being visited (not those of its methods): see `env_cycle()`.
    fn trait_cycle(&self, context: &Context) -> Option<TokenStream> {
        if !self.checks_env() || self.scopes.len() != self.self_scope.depth {
            return None;
        }
        match &context.self_ty {
            Some(SelfRequirement::Bound(path)) => Some(quote!(Self: #path)),
            _ => None,
        }
    }

    /// Returns `true` if a type in `context` that is an entry would make a cycle (`env_cycle()`).
    fn env_blocked(&mut self, context: &Context, optional: bool) -> bool {
        self.env_unstable_bound(context, optional).is_some() || self.trait_cycle(context).is_some()
    }

    /// Runs `visit` on a probe visitor that has the generics and the `Self` of `context`, and
    /// checks nothing itself.
    fn in_context<R>(
        &mut self,
        context: &Context,
        visit: impl FnOnce(&mut AnalyzeVisitor<'_, A>) -> R,
    ) -> R {
        Self::throwaway(
            &mut *self.rule,
            &context.generics,
            Place::default(),
            |probe| {
                probe.mode.check_env = false;
                probe.self_scope.ty = context.self_ty.clone();
                visit(probe)
            },
        )
    }

    /// Returns `true` if the self type of `context`, which is the self type of an impl, names a
    /// type that is leaked, so the copy writes it as a path through the Repeater. With `optional`,
    /// a type that the Referrer may replace counts too.
    fn self_type_names_leaked(&mut self, context: &Context, optional: bool) -> bool {
        match self.env_self_type(context) {
            Some(ty) => {
                self.probe_names_leaked(context, optional, true, |probe| probe.visit_type(ty))
            }
            None => false,
        }
    }

    /// Returns `true` if `visit` finds a leaked type in a probe in `context`: one that must be an
    /// entry if the `environment` must prove the predicate, and an optional one with `optional`.
    fn probe_names_leaked(
        &mut self,
        context: &Context,
        optional: bool,
        environment: bool,
        visit: impl FnOnce(&mut AnalyzeVisitor<'_, A>),
    ) -> bool {
        self.in_context(context, |probe| {
            visit(probe);
            (environment && !probe.musts.is_empty())
                || (optional && has_optional_candidate(&probe.frame.effects))
        })
    }

    /// The self type of `context`, if it is that of an impl and the predicates of the item are
    /// checked (see `checks_env()`).
    fn env_self_type<'c>(&self, context: &'c Context) -> Option<&'c Type> {
        match &context.self_ty {
            Some(SelfRequirement::Type(ty)) if self.checks_env() => Some(ty),
            _ => None,
        }
    }

    /// Returns the contexts of the entries in the self type of an impl (`context`), which rustc
    /// normalizes `Self` with; with `optional`, of the optional ones too. With `blockable`, a type
    /// with unstable bounds is left to its parts (`block_unstable_self_types()`).
    fn self_type_entries(
        &mut self,
        context: &Context,
        optional: bool,
        blockable: bool,
    ) -> Vec<Context> {
        let ty = match self.env_self_type(context) {
            Some(ty) => ty,
            None => return Vec::new(),
        };
        let declared = self
            .scopes
            .get(self.self_scope.depth.wrapping_sub(1))
            .map(Scope::generics);
        // `blocked` probes too, so it runs after this probe.
        let effects = self.in_context(context, |probe| {
            probe.visit_type(ty);
            std::mem::take(&mut probe.frame.effects)
        });
        let mut blocked =
            |context: &Context| blockable && self.env_unstable_bound(context, false).is_some();
        let mut found = Vec::new();
        entry_contexts(
            &effects,
            declared.as_ref(),
            &mut blocked,
            optional,
            &mut found,
        );
        found
    }

    /// Returns the first bound of the generics of `context` that names a type that is leaked, if
    /// the type being visited is in a predicate of the item. With `optional`, an optional type (one
    /// of another crate, which the Referrer replaces) counts too, and in every where predicate.
    fn env_leaked_bound(&mut self, context: &Context, optional: bool) -> Option<TokenStream> {
        let generics = &context.generics;
        // Without bounds, nothing is named.
        if !self.checks_env() || !has_bounds(generics) {
            return None;
        }
        // A type that must be an entry counts where the environment must prove the bound, and an
        // optional one counts everywhere.
        for param in &generics.params {
            if let GenericParam::Type(param) = param {
                if self.probe_names_leaked(context, optional, true, |probe| {
                    probe.visit_type_param(param)
                }) {
                    return Some(quote!(#param));
                }
            }
        }
        // An impl may prove a where predicate that is not on a parameter, so a required entry
        // there is left out. An optional one is not: the Leaker cannot tell that an impl proves it
        // (`::std::vec::Vec<T>: Clone` it does not).
        generics
            .where_clause
            .iter()
            .flat_map(|where_clause| &where_clause.predicates)
            .find(|predicate| {
                let environment = needs_environment(predicate, context);
                (environment || optional)
                    && self.probe_names_leaked(context, optional, environment, |probe| {
                        probe.visit_where_predicate(predicate)
                    })
            })
            .map(|predicate| quote!(#predicate))
    }

    /// Returns the first unstable bound of the generics of `context` (see [`crate::unstable`]):
    /// the bounds that the Repeater's impl has, so not those that `Context::of()` left out.
    fn env_unstable_bound(&mut self, context: &Context, optional: bool) -> Option<Unstable> {
        self.env_unstable_bounds(context, optional, false)
            .into_iter()
            .next()
    }

    /// Like `env_unstable_bound()`, and with `all` every unstable bound of the generics, not the
    /// first: the one that names a leaked type, then each that names `Self` or a projection (a
    /// bound that names both is two). What one of them blames does not hide another.
    fn env_unstable_bounds(
        &mut self,
        context: &Context,
        optional: bool,
        all: bool,
    ) -> Vec<Unstable> {
        let mut found = Vec::new();
        let generics = &context.generics;
        if !self.checks_env() || !has_bounds(generics) {
            return found;
        }
        if let Some(bound) = self.env_leaked_bound(context, optional) {
            found.push(Unstable {
                bound,
                why: Instability::Leaked,
            });
            if !all {
                return found;
            }
        }
        // The equations of associated types in the environment: `T::Item` of those is not rigid.
        let bindings = bindings(
            self.params(),
            self.scopes.iter().flat_map(Scope::predicates),
        );
        // Owned, because `check` borrows the visitor to probe.
        let params: HashSet<String> = self.params().map(generic_param_name).collect();
        let is_param = |name: &str| params.contains(name);
        // Whether `Self` is the self type of an impl that the copy writes as a leaked type.
        let mut self_leaked: Option<bool> = None;
        let mut check = |bound: TokenStream, projection: Option<Why>, mentions_self: bool| {
            let mut whys: Vec<Why> = Vec::new();
            let named = projection.is_some();
            whys.extend(projection);
            if (all || !named)
                && mentions_self
                && self.self_scope.written
                && *self_leaked
                    .get_or_insert_with(|| self.self_type_names_leaked(context, optional))
            {
                whys.push(Why::SelfType);
            }
            whys.into_iter()
                .map(|why| Unstable {
                    bound: bound.clone(),
                    why: Instability::Named(why),
                })
                .collect::<Vec<_>>()
        };
        for param in &generics.params {
            if let GenericParam::Type(type_param) = param {
                let mut finder = Projections::new(&is_param, &bindings);
                finder.visit_type_param(type_param);
                let unstable = check(
                    quote!(#type_param),
                    finder.found,
                    generic_param_mentions_self(param),
                );
                found.extend(unstable);
                if !all && !found.is_empty() {
                    return found;
                }
            }
        }
        for predicate in generics
            .where_clause
            .iter()
            .flat_map(|where_clause| &where_clause.predicates)
        {
            let on_environment = needs_environment(predicate, context);
            let projection = predicate_projection(predicate, on_environment, &is_param, &bindings);
            let unstable = check(
                quote!(#predicate),
                projection,
                on_environment && where_predicate_mentions_self(predicate),
            );
            found.extend(unstable);
            if !all && !found.is_empty() {
                return found;
            }
        }
        // A bound that the context keeps (`Self`) names its generic through a path that may hide
        // the projection that an equation fixes.
        let equated = equated_generics(
            self.params(),
            self.scopes.iter().flat_map(Scope::predicates),
        );
        if all || found.is_empty() {
            found.extend(hidden_projection(generics, &equated));
        }
        found
    }

    /// Makes the optional types that the unstable bounds of `context` name no optional entries,
    /// for a type leaked in a predicate: replacing them would make a cycle. With a bound that names
    /// `Self`, so are those of the self type of an impl. Every unstable bound counts.
    fn not_optional_in_bounds(&mut self, context: &Context) {
        let unstable = self.env_unstable_bounds(context, true, true);
        if unstable.is_empty() {
            return;
        }
        let names_self = unstable
            .iter()
            .any(|unstable| matches!(unstable.why, Instability::Named(Why::SelfType)));
        let keys = self.in_context(context, |probe| {
            probe.visit_generics(&context.generics);
            // The bound that names `Self` is unstable for the self type that the copy writes.
            if let (true, Some(SelfRequirement::Type(ty))) = (names_self, &context.self_ty) {
                probe.visit_type(ty);
            }
            optional_keys(&probe.frame.effects)
        });
        self.records.not_optional.extend(keys);
    }

    /// Like `not_optional_in_bounds()` for the optional entries of the self type of an impl (in
    /// `context`), which `Self` is in a bound or where predicate.
    fn not_optional_in_self_type(&mut self, context: &Context) {
        let ty = match self.env_self_type(context) {
            Some(ty) => ty,
            None => return,
        };
        let keys = self.in_context(context, |probe| {
            probe.visit_type(ty);
            optional_keys(&probe.frame.effects)
        });
        for (context, ty) in keys {
            if self.env_leaked_bound(&context, true).is_some() {
                self.records.not_optional.push((context, ty));
            }
        }
    }

    /// Records the conflict of a relative path `cause`, which cannot be leaked as a whole because
    /// of `impossible`. `paths`: the relative paths that the copy writes as they are
    /// ([`CheckResult::paths`]).
    fn push_relative_path(&mut self, cause: Cause, impossible: Cause, paths: Vec<Cause>) {
        self.push_error(CheckResult {
            paths,
            ..CheckResult::conflict(vec![cause], vec![impossible], Reason::RelativePath)
        });
    }

    /// Records the conflict as an effect, with the number of mentions of `Self` without meaning
    /// found so far (see [`CheckResult::seen`]) and where it was found.
    fn push_error(&mut self, mut conflict: CheckResult) {
        conflict.seen = self.records.self_errors.len();
        conflict.place = self.place();
        self.frame.effects.push(Effect::Error(Box::new(conflict)));
    }

    /// Walks the parts of an impl as `syn::visit::visit_item_impl()` does, with the self type
    /// apart: the copy writes it as a type, and `Self` stands for it in the rest of the impl, so
    /// no entry in it can depend on `Self` (see [`AnalyzeVisitor::cut_self_cycles()`]).
    pub(crate) fn walk_item_impl(&mut self, i: &ItemImpl) {
        for attr in &i.attrs {
            self.visit_attribute(attr);
        }
        let outer = self.self_type_walk.replace(false);
        let trait_ = i.trait_.as_ref().map(|(_, path, _)| self.impl_trait(path));
        let outer_trait = std::mem::replace(&mut self.self_scope.trait_, trait_);
        self.visit_generics(&i.generics);
        if let Some((_, path, _)) = &i.trait_ {
            let cause = Cause::Trait(path.clone());
            self.visit_path_of(path, &cause, &cause);
        }
        let start = self.frame.effects.len();
        self.visit_type(&i.self_ty);
        let end = self.cut_self_cycles(start, &i.generics);
        let inherent = std::mem::replace(&mut self.inherent_impl, i.trait_.is_none());
        for item in &i.items {
            self.visit_impl_item(item);
        }
        self.inherent_impl = inherent;
        if self.self_type_walk == Some(true) {
            self.block_unstable_self_types(start, end);
        }
        self.self_scope.trait_ = outer_trait;
        self.self_type_walk = outer;
    }

    /// Called after the walk of an impl whose predicates name a bare `Self`, with `start..end` the
    /// effects of its self type. rustc normalizes `Self` there, so a type there that the graph may
    /// choose and whose bounds are unstable would be a cycle: its parts are the entries instead.
    fn block_unstable_self_types(&mut self, start: usize, end: usize) {
        // The bounds of a type are those of a type in a predicate.
        let mut effects = std::mem::take(&mut self.frame.effects);
        self.position.env_depth += 1;
        each_pending_mut(&mut effects[start..end], &mut |pending| {
            let candidate = pending.required_candidate
                && !pending.no_required_entry
                && !pending.required
                && !is_whole_reference(&pending.ty);
            if candidate && self.env_unstable_bound(&pending.context, false).is_some() {
                pending.no_required_entry = true;
            }
        });
        self.position.env_depth -= 1;
        self.frame.effects = effects;
    }

    /// Cuts the entries of the self type of an impl whose context depends on the `Self` that a
    /// bound of `generics` names ([`cut_conflicts()`]): the copy would write such an entry through
    /// itself (E0391). `start` is the first effect of the self type; returns the end of them.
    fn cut_self_cycles(&mut self, start: usize, generics: &Generics) -> usize {
        let mut not_optional = Vec::new();
        let seen = self.records.self_errors.len();
        let place = self.place();
        let found = self.frame.effects.split_off(start);
        let found = cut_conflicts(found, generics, seen, place, &mut not_optional);
        self.frame.effects.extend(found);
        self.records.not_optional.extend(not_optional);
        self.frame.effects.len()
    }

    /// Records an error at `cause` ([`bound_cause()`]) if a trait path is relative or starts with
    /// a `Self` without meaning: the path is written as it is, and the caller visits its arguments.
    /// Returns `true` for a relative path.
    fn check_bound_path(&mut self, path: &Path, cause: Cause) -> bool {
        if self_head(path).is_some() && self.self_scope.ty.is_none() {
            self.records
                .self_errors
                .push(SelfMention::new(cause, self.place()));
        } else if !self.is_free_path(&Cause::Trait(path.clone())) {
            let paths = vec![Cause::Trait(without_arguments(path))];
            self.push_relative_path(cause.clone(), cause, paths);
            return true;
        }
        false
    }

    /// The trait `path` of an impl, for the messages that name it.
    fn impl_trait(&self, path: &Path) -> ImplTrait {
        ImplTrait {
            path: path.clone(),
            absolute: self.is_free_path(&Cause::Trait(path.clone())),
        }
    }

    /// Returns `true` if `path` starts with `Self` or a generic in scope: the rule is
    /// never asked about it.
    fn heads_own_name(&self, path: &Path) -> bool {
        path.leading_colon.is_none()
            && matches!(path.segments.first(), Some(seg)
                if seg.ident == "Self" || self.is_generic(&seg.ident))
    }

    /// Returns `true` if the path of `cause` is absolute, starts with `$crate`, or is allowed, at
    /// the place of the visitor. A type with an absolute path may still be an optional entry (see
    /// `is_optional_path()`).
    fn is_free_path(&self, cause: &Cause) -> bool {
        cause.path().map_or(false, |path| {
            path.leading_colon.is_some()
                || matches!(
                    path.segments.iter().next(),
                    Some(PathSegment {
                        ident,
                        arguments: PathArguments::None,
                    }) if ident == "$crate"
                )
                || (!self.heads_own_name(path) && self.call_rule(cause))
        })
    }

    /// Returns `true` for an absolute path to an external crate (not `::core`, not allowed). The
    /// Referrer may replace such a type, so nothing in it is anchored (see `scan_anchors()`).
    fn is_optional_path(&self, cause: &Cause) -> bool {
        cause.path().map_or(false, |path| {
            path.leading_colon.is_some()
                && !matches!(path.segments.first(), Some(seg) if seg.ident == "core")
                && !self.call_rule(cause)
        })
    }

    /// Asks the rule about the path of `cause`, without its generic arguments, at the place of the
    /// visitor.
    fn call_rule(&self, cause: &Cause) -> bool {
        self.rule.allow(&cause.bare(), self.place())
    }

    fn is_generic(&self, ident: &Ident) -> bool {
        self.scope_of(&ident_name(ident)).is_some()
    }

    fn params(&self) -> impl Iterator<Item = &GenericParam> {
        self.scopes.iter().flat_map(Scope::params)
    }

    fn resolves_through_bounds(&self, qself: &QSelf) -> bool {
        qself.as_token.is_none()
            && match unwrap_type_group(&qself.ty) {
                Type::Path(TypePath { qself: None, path }) => {
                    path.leading_colon.is_none()
                        && path.segments.len() == 1
                        && path.segments.iter().all(|seg| {
                            seg.arguments.is_none()
                                && (seg.ident == "Self" || self.is_generic(&seg.ident))
                        })
                }
                _ => false,
            }
    }

    fn starts_with_generic(&self, path: &Path) -> bool {
        path.leading_colon.is_none()
            && path.segments.first().map_or(false, |seg| {
                seg.arguments.is_none() && self.is_generic(&seg.ident)
            })
    }

    /// Whether the default of the parameter `ident`, which the walk is visiting, was analyzed
    /// alone by the scope that it is directly in (see [`AloneDefaults`]).
    fn is_alone_default(&self, ident: &Ident) -> bool {
        self.alone_defaults.last().map_or(false, |alone| {
            alone.scopes == self.scopes.len()
                && alone.item_depth == self.item_depth
                && alone.idents.contains(ident)
        })
    }

    /// The parameters with a default, if the scope that the walk is directly in was entered for
    /// the item that declares `Self` as `self_ty` with `generics`: the walk is about to visit that
    /// item again, as a nested item.
    fn own_alone_defaults(
        &self,
        self_ty: &SelfRequirement,
        generics: &Generics,
    ) -> Option<Vec<Ident>> {
        let alone = self.alone_defaults.last()?;
        let (declared_self, declared) = alone.declared.as_ref()?;
        (alone.scopes == self.scopes.len()
            && alone.item_depth == self.item_depth
            && declared_self == self_ty
            && declared == generics)
            .then(|| alone.idents.clone())
    }

    fn with_scope(&mut self, scope: Scope, f: impl FnOnce(&mut Self)) {
        self.scopes.push(scope);
        f(self);
        self.scopes.pop();
    }

    fn with_local_generics(&mut self, generics: &Generics, f: impl FnOnce(&mut Self)) {
        self.with_scope(Scope::new(generics, self.self_scope.ty.clone()), f)
    }

    /// Runs `f` with the generics of an item: the Leaker's if outermost, local otherwise. `gats`:
    /// those of a trait (see [`Gat`]).
    fn with_item_generics(
        &mut self,
        generics: &Generics,
        gats: Vec<Gat>,
        f: impl FnOnce(&mut Self),
    ) {
        let nested = self.item_depth > 0;
        self.item_depth += 1;
        if nested {
            let scope = Scope::new(generics, self.self_scope.ty.clone()).with_gats(gats);
            self.with_scope(scope, f);
        } else {
            f(self);
        }
        self.item_depth -= 1;
    }

    /// Like `with_item_generics()`, with `Self` meaning `self_ty` (`None` in a module). A nested
    /// item has its own `Self`, which may be leaked whatever the scope around says.
    fn with_item(
        &mut self,
        generics: &Generics,
        self_ty: Option<SelfRequirement>,
        written: bool,
        gats: Vec<Gat>,
        f: impl FnOnce(&mut Self),
    ) {
        let declares = self_ty.is_some();
        let nested_type =
            self.item_depth > 0 && !written && matches!(self_ty, Some(SelfRequirement::Type(_)));
        let inner = SelfScope {
            ty: self_ty,
            written,
            nested: nested_type,
            trait_: None,
            depth: self.self_scope.depth,
        };
        let outer = std::mem::replace(&mut self.self_scope, inner);
        self.with_item_generics(generics, gats, |visitor| {
            visitor.self_scope.depth = if declares { visitor.scopes.len() } else { 0 };
            f(visitor)
        });
        self.self_scope = outer;
    }

    /// Visits the attributes and visibility only: the body or value is not analyzed.
    fn visit_head(&mut self, attrs: &[Attribute], vis: &Visibility) {
        for attr in attrs {
            self.visit_attribute(attr);
        }
        self.visit_visibility(vis);
    }

    /// Runs `f`, which visits `sig`. `item`: `sig` is that of a function item, whose generics are
    /// the item's. `compared`: rustc compares `sig` with one the copy cannot change (a method of a
    /// trait or a trait impl), so a lifetime the copy leaves unconstrained is an error (E0195).
    fn with_signature(
        &mut self,
        sig: &Signature,
        item: bool,
        compared: bool,
        f: impl FnOnce(&mut Self),
    ) {
        if compared {
            let risky = self.unconstrained_lifetimes(sig);
            self.signatures.push(SignatureWalk {
                at: self.scopes.len(),
                effects: self.frame.effects.len(),
                declared: sig
                    .generics
                    .lifetimes()
                    .map(|param| param.lifetime.clone())
                    .collect(),
                risky,
                leaked: HashMap::new(),
            });
        }
        // The generics are in scope while the lifetimes are checked, which needs them.
        let walk = |visitor: &mut Self| {
            visitor.with_arguments_of(sig, f);
            if compared {
                visitor.check_leaked_lifetimes(sig);
            }
        };
        if item {
            self.with_item_generics(&sig.generics, Vec::new(), walk)
        } else {
            self.with_local_generics(&sig.generics, walk)
        }
        if compared {
            self.signatures.pop();
        }
    }

    /// Visits the signature of a method as `visit_signature()` does, then runs `body` in the scope
    /// of its generics.
    fn method_signature(&mut self, sig: &Signature, body: impl FnOnce(&mut Self)) {
        let compared = !self.inherent_impl;
        self.with_signature(sig, false, compared, |visitor| {
            syn::visit::visit_signature(visitor, sig);
            body(visitor)
        })
    }

    /// Visits the output of a function pointer or of `Fn(..)`. It has its own elision: the
    /// signature around does not count its lifetimes.
    fn in_output(&mut self, output: &ReturnType) {
        self.elisions.push(None);
        self.visit_return_type(output);
        self.elisions.pop();
    }

    /// Runs `f`, which visits the arguments of `sig`, knowing what their elision must keep (see
    /// [`Elision`]).
    fn with_arguments_of(&mut self, sig: &Signature, f: impl FnOnce(&mut Self)) {
        let outer = std::mem::replace(&mut self.position.signature, Elision::of_signature(sig));
        f(self);
        self.position.signature = outer;
    }

    fn is_unconstrained(&self, lifetime: &Lifetime, scope: usize) -> bool {
        self.signatures
            .iter()
            .any(|walk| walk.at == scope && walk.risky.contains(lifetime))
    }

    /// Notes that `ty` is leaked whole, if it names a lifetime of the signature being visited:
    /// `check_leaked_lifetimes()` needs it. `inner` and `depth` are those of its decision.
    fn record_leaked(&mut self, ty: &Type, depth: usize, inner: &Frame) {
        // The probe of a binder walks its types again for real, which must leave nothing keyed by
        // their address.
        if self.position.binder_probe.is_some() {
            return;
        }
        let walk = match self.signatures.last() {
            Some(walk) if !walk.declared.is_empty() => walk,
            _ => return,
        };
        let mut finder = OutputLifetimes::new(walk.declared.clone());
        finder.visit_type(ty);
        let named: Vec<Lifetime> = finder
            .found
            .into_iter()
            .filter(|lifetime| self.scope_of(&lifetime.to_string()) == Some(walk.at))
            .collect();
        if named.is_empty() {
            return;
        }
        let conflict = self.conflict_causes(inner.must_start, inner.impossible_start, depth);
        if let Some(walk) = self.signatures.last_mut() {
            walk.leaked
                .insert(ty as *const Type as usize, (named, conflict));
        }
    }

    /// Reports the lifetimes of the signature just visited that only types leaked whole hold in the
    /// inputs (E0195, see [Method lifetimes](crate::Leaker#method-lifetimes)). A culprit is copied
    /// by its own path, so the next ones inside it are reported too ([`unconstraining_types()`]).
    fn check_leaked_lifetimes(&mut self, sig: &Signature) {
        let (leaked, start, risky) = match self.signatures.last_mut() {
            Some(walk) if !walk.leaked.is_empty() || !walk.risky.is_empty() => (
                std::mem::take(&mut walk.leaked),
                walk.effects,
                walk.risky.clone(),
            ),
            _ => return,
        };
        let facts = SignatureFacts::of(sig);
        // The optional types that only keep a lifetime constrained, by address.
        let mut kept_optional: HashSet<usize> = HashSet::new();
        let mut conflicts =
            self.unconstrained_conflicts(sig, &facts, &leaked, &risky, &mut kept_optional);
        let made = self.correct_leaked(start, &facts, &leaked, &conflicts);
        link_unconstrained(&mut conflicts, &made);
        // After the types copied as written are corrected: the optional types in them are among
        // the effects now.
        if !kept_optional.is_empty() {
            let from = start.min(self.frame.effects.len());
            let mut found = Vec::new();
            block_optional(&mut self.frame.effects[from..], &kept_optional, &mut found);
            self.records.not_optional.extend(found);
        }
        let seen = self.records.self_errors.len();
        let place = self.place();
        self.frame
            .effects
            .extend(conflicts.into_iter().map(|(_, mut conflict)| {
                conflict.seen = seen;
                conflict.place = place;
                Effect::Error(Box::new(conflict))
            }));
    }

    /// The conflicts of the types of `leaked` that leave a lifetime of `sig` unconstrained in the
    /// copy, each with the address of its type. Adds to `kept_optional` the optional types that
    /// only keep a `risky` lifetime constrained.
    fn unconstrained_conflicts(
        &self,
        sig: &Signature,
        facts: &SignatureFacts,
        leaked: &HashMap<usize, (Vec<Lifetime>, CheckResult)>,
        risky: &[Lifetime],
        kept_optional: &mut HashSet<usize>,
    ) -> Vec<(usize, CheckResult)> {
        let mut conflicts = Vec::new();
        for lifetime in facts.unconstrained() {
            // The occurrences that constrain the lifetime as written. An external type that the
            // Leaker does not allow stays as written unless the Referrer replaces it, so it counts.
            let used = scan_inputs(self, &sig.inputs, lifetime, false, &[]);
            // The original constrains it, and in the copy only types leaked whole hold it: the
            // first on the way to each occurrence is what to report, and then the next ones.
            let culprits = unconstraining_types(&used.anchors, |ty| leaked.contains_key(&ty));
            if risky.contains(lifetime) {
                kept_optional
                    .extend(self.optional_anchors(sig, lifetime, &used, leaked, &culprits));
            }
            for ty in culprits {
                let (named, conflict) = &leaked[&ty];
                if named.contains(lifetime) {
                    let mut conflict = conflict.clone();
                    // Each conflict is one of its own, though the type is the same.
                    conflict.origin = Origin::default();
                    conflict
                        .impossible
                        .insert(0, Cause::Lifetime(lifetime.clone()));
                    unconstrained_advice(&mut conflict, lifetime, false);
                    conflicts.push((ty, conflict));
                }
            }
        }
        conflicts
    }

    /// Copies as written whole the types of the signature (its effects from `start`) that the
    /// reported `conflicts` are about. Returns the part that each is made, and the types inside it,
    /// by address.
    fn correct_leaked(
        &mut self,
        start: usize,
        facts: &SignatureFacts,
        leaked: &HashMap<usize, (Vec<Lifetime>, CheckResult)>,
        conflicts: &[(usize, CheckResult)],
    ) -> HashMap<usize, (Part, Vec<usize>)> {
        // The types that the conflicts are about, by address.
        let corrected: HashSet<usize> = conflicts
            .iter()
            .filter(|(_, conflict)| report(conflict).is_some())
            .map(|(ty, _)| *ty)
            .collect();
        let mut made = HashMap::new();
        if corrected.is_empty() {
            return made;
        }
        // The entry that the conflict is about is copied as written whole instead. Only the types
        // of this signature are looked at: an address can be one of a type that is gone.
        let found = self
            .frame
            .effects
            .split_off(start.min(self.frame.effects.len()));
        self.frame
            .effects
            .extend(correct_effects(found, &mut |pending| {
                if !corrected.contains(&pending.address) {
                    return None;
                }
                let mut part = pending.part();
                part.mark = Some(facts.unconstrained_mark(&leaked[&pending.address].0));
                let mut inside = Vec::new();
                let mut note = |inner: &PendingType| inside.push(inner.address);
                each_pending(&pending.inside, &mut note);
                each_pending(&pending.components, &mut note);
                made.insert(pending.address, (part.clone(), inside));
                Some(part)
            }));
        made
    }

    /// Returns the optional types that alone keep `lifetime` constrained in the copy, with those
    /// inside them on the way to it (`Vec<Box<dyn Fn() + 'x>>`: both): a Referrer that replaces
    /// them leaves it in a projection only (E0195). The `culprits` keep what is in them.
    fn optional_anchors(
        &self,
        sig: &Signature,
        lifetime: &Lifetime,
        used: &OutputUse,
        leaked: &HashMap<usize, (Vec<Lifetime>, CheckResult)>,
        culprits: &[usize],
    ) -> HashSet<usize> {
        let kept = |anchor: &&Vec<usize>| {
            anchor
                .iter()
                .all(|ty| !leaked.contains_key(ty) || culprits.contains(ty))
        };
        // The scan goes on inside the types that stopped it, now that they are no entries.
        let mut blocked: Vec<usize> = Vec::new();
        loop {
            let careful = scan_inputs(self, &sig.inputs, lifetime, true, &blocked);
            if careful.anchors.iter().any(|anchor| kept(&anchor)) {
                break;
            }
            let found: Vec<usize> = used
                .anchors
                .iter()
                .filter(kept)
                .flatten()
                .copied()
                .filter(|ty| careful.stopped.contains(ty) && !blocked.contains(ty))
                .collect();
            if found.is_empty() {
                break;
            }
            for ty in found {
                push_once(&mut blocked, ty);
            }
        }
        blocked.into_iter().collect()
    }

    /// Returns the lifetimes of `sig` that the copy would leave unconstrained (E0195): the return
    /// type and some input mention them, no bound does, and no input keeps them (the receiver or a
    /// reference with it).
    fn unconstrained_lifetimes(&mut self, sig: &Signature) -> Vec<Lifetime> {
        if sig.generics.lifetimes().next().is_none() {
            return Vec::new();
        }
        let in_scope: Generics = Generics {
            params: self
                .params()
                .chain(sig.generics.params.iter())
                .cloned()
                .collect(),
            ..Default::default()
        };
        let mut not_a_cause = |object: &TypeTraitObject| self.is_not_a_cause(object, &in_scope);
        let mut mentioned = Lifetimes::default();
        let mut kept = Vec::new();
        for input in &sig.inputs {
            mentioned.visit_fn_arg(input);
            match input {
                FnArg::Receiver(receiver) => {
                    kept.extend(receiver.lifetime().cloned());
                    kept_lifetimes(&receiver.ty, &mut kept, &mut not_a_cause);
                }
                FnArg::Typed(arg) => kept_lifetimes(&arg.ty, &mut kept, &mut not_a_cause),
            }
        }
        unconstrained_in_output(sig)
            .into_iter()
            .filter(|lifetime| mentioned.0.contains(lifetime) && !kept.contains(lifetime))
            .collect()
    }

    /// Returns `true` if `object` is certainly not a cause itself nor an optional entry, so a
    /// reference to it is never leaked whole because of it. When unsure (a relative path, an
    /// undeclared name) it is `false`.
    fn is_not_a_cause(&mut self, object: &TypeTraitObject, in_scope: &Generics) -> bool {
        let ty = Type::TraitObject(object.clone());
        let place = self.place();
        let result = Self::throwaway(&mut *self.rule, in_scope, place, |visitor| {
            visitor.found(&ty)
        });
        // The Referrer may replace an external trait object, with the reference holding it.
        !result.is_own && !result.optional && result.impossible.is_empty()
    }

    /// Runs `f` with the lifetimes of a `for<..>` binder in scope. `groups` are its outputs and
    /// inputs. `f` may run twice.
    fn with_bound_lifetimes(
        &mut self,
        lifetimes: Option<&BoundLifetimes>,
        groups: Vec<BinderGroup>,
        f: impl Fn(&mut Self),
    ) {
        let lifetimes = match lifetimes {
            Some(lifetimes) => lifetimes,
            None => return f(self),
        };
        let uses = self.binder_uses(groups);
        let output = self.unanchored(lifetimes, &uses, &f);
        let tracked = !output.is_empty();
        if tracked {
            self.binders.push(Binder {
                owner: self.position.type_depth,
                output,
            });
        }
        self.with_scope(Scope::binder(lifetimes), |v| f(v));
        if tracked {
            self.binders.pop();
        }
    }

    /// Returns the lifetimes of `uses` that no input anchors (see [`Binder`]). The binder is probed
    /// first, in an isolated state, to log the types leaked whole. Binders inside a probe are not
    /// probed again: `binder_uses()` finds no anchors there.
    fn unanchored(
        &mut self,
        lifetimes: &BoundLifetimes,
        uses: &[OutputUse],
        f: &impl Fn(&mut Self),
    ) -> Vec<Lifetime> {
        // An anchor with no type in the way (an argument of the bound itself) cannot be lost.
        let to_check = uses
            .iter()
            .any(|used| used.anchors.iter().any(|anchor| !anchor.is_empty()));
        let own = if to_check {
            self.probe_binder(lifetimes, f)
        } else {
            HashSet::new()
        };
        let mut unanchored: Vec<Lifetime> = Vec::new();
        for used in uses {
            let anchored = used
                .anchors
                .iter()
                .any(|anchor| anchor.iter().all(|ty| !own.contains(ty)));
            if !anchored && !unanchored.contains(&used.lifetime) {
                unanchored.push(used.lifetime.clone());
            }
        }
        unanchored
    }

    /// Runs `f` with the binder `lifetimes` in scope without effect, returning the types leaked
    /// whole in it.
    fn probe_binder(
        &mut self,
        lifetimes: &BoundLifetimes,
        f: &impl Fn(&mut Self),
    ) -> HashSet<usize> {
        let frame = self.new_frame();
        let (musts, impossibles) = (self.musts.len(), self.impossibles.len());
        let records = self.records.mark();
        self.position.binder_probe = Some(HashSet::new());
        self.with_scope(Scope::binder(lifetimes), |v| f(v));
        self.frame = frame;
        self.musts.truncate(musts);
        self.truncate_impossibles(impossibles);
        // The contexts too: what the probe interned the walk interns again, with the numbers that
        // come next then, as if there was no probe.
        self.records.truncate(&records);
        self.position.binder_probe.take().unwrap_or_default()
    }

    fn binder_uses(&self, groups: Vec<BinderGroup>) -> Vec<OutputUse> {
        let mut uses = Vec::new();
        for group in groups {
            for lifetime in &group.output {
                let mut used = OutputUse::new(lifetime.clone(), true, Vec::new());
                if self.position.binder_probe.is_none() {
                    for input in &group.inputs {
                        match input {
                            BinderInput::Type(ty) => scan_anchors(self, ty, &mut used),
                            BinderInput::Lifetime(input) if *input == lifetime => {
                                used.anchors.push(Vec::new())
                            }
                            BinderInput::Lifetime(_) => (),
                        }
                    }
                }
                uses.push(used);
            }
        }
        uses
    }

    /// Returns `true` if `ty` is an external trait object without a lifetime bound, so the Referrer
    /// may replace what holds it.
    fn is_external_object(&self, ty: &Type) -> bool {
        trait_object_of(ty).map_or(false, |object| {
            self.names_external_crate(object) && !has_lifetime_bound(object)
        })
    }

    fn holds_external_object(&self, path: &Path) -> bool {
        match path.segments.last().map(|seg| &seg.arguments) {
            Some(PathArguments::AngleBracketed(args)) => args
                .args
                .iter()
                .any(|arg| matches!(arg, GenericArgument::Lifetime(_)))
                && args.args.iter().any(
                    |arg| matches!(arg, GenericArgument::Type(ty) if self.is_external_object(ty)),
                ),
            _ => false,
        }
    }

    fn names_external_crate_in_expr(&self, expr: &Expr) -> bool {
        let mut finder = AnyPath::new(|path| self.is_optional_path(&Cause::Unknown(path.clone())));
        finder.visit_expr(expr);
        finder.found
    }

    fn names_external_crate(&self, object: &TypeTraitObject) -> bool {
        object.bounds.iter().any(|bound| {
            matches!(bound, TypeParamBound::Trait(bound) if self.is_optional_path(&Cause::Trait(bound.path.clone())))
        })
    }
}

/// Returns `true` if `predicate` bounds a type that only the environment can prove a bound of: a
/// generic or its associated type (`T`, `T::Item`, `<T as Tr>::Item`), or `Self` in a trait. A
/// bound on another type (`Vec<T>: Clone`) is proved by an impl.
fn needs_environment(predicate: &WherePredicate, context: &Context) -> bool {
    let predicate = match predicate {
        WherePredicate::Type(predicate) => predicate,
        _ => return false,
    };
    let (qself, path) = match unwrap_type_group(&predicate.bounded_ty) {
        Type::Path(TypePath { qself, path }) => (qself, path),
        _ => return false,
    };
    let head = match qself {
        Some(qself) => bounded_ident(&qself.ty),
        None if path.leading_colon.is_none() => {
            path.segments.first().map(|seg| ident_name(&seg.ident))
        }
        None => None,
    };
    match head {
        Some(head) if head == "Self" => {
            matches!(context.self_ty, Some(SelfRequirement::Bound(_)))
        }
        Some(head) => context
            .generics
            .params
            .iter()
            .any(|param| is_type_param_named(param, &head)),
        None => false,
    }
}

/// Returns `true` if `generics` have a where clause or a type parameter with bounds.
fn has_bounds(generics: &Generics) -> bool {
    generics.where_clause.is_some()
        || generics.params.iter().any(|param| match param {
            GenericParam::Type(param) => !param.bounds.is_empty(),
            _ => false,
        })
}

/// Returns the first mention of `Self` in a bound of a generic of `context`, or in a where
/// predicate that the environment must prove (see [`needs_environment()`]).
fn self_in_environment_bounds(context: &Context) -> Option<Cause> {
    let generics = &context.generics;
    let in_params = generics.params.iter().find_map(|param| match param {
        // The default is no bound.
        GenericParam::Type(_) => self_in_param(&without_default(param.clone())).first,
        _ => None,
    });
    in_params.or_else(|| {
        generics
            .where_clause
            .iter()
            .flat_map(|where_clause| &where_clause.predicates)
            .filter(|predicate| needs_environment(predicate, context))
            .find_map(|predicate| self_in_predicate(predicate).first)
    })
}

/// Returns `true` if `ty` is `Self::A` or `<Self as Tr>::A`: an associated type of `Self`.
fn is_self_projection(ty: &Type) -> bool {
    match unwrap_type_group(ty) {
        Type::Path(TypePath { qself: None, path }) => {
            self_head(path).is_some() && path.segments.len() > 1
        }
        Type::Path(TypePath {
            qself: Some(qself), ..
        }) => is_bare_self(&qself.ty),
        _ => false,
    }
}

/// The type parameters of `generics` that have a default: index, name and default.
fn defaulted(generics: &Generics) -> impl Iterator<Item = (usize, &Ident, &Type)> {
    generics
        .params
        .iter()
        .enumerate()
        .filter_map(|(index, param)| match param {
            GenericParam::Type(TypeParam {
                ident,
                default: Some(default),
                ..
            }) => Some((index, ident, default)),
            _ => None,
        })
}

impl<A: Allow> Anchoring for AnalyzeVisitor<'_, A> {
    fn is_projection(&self, path: &Path) -> bool {
        path.segments.len() > 1 && self.heads_own_name(path)
    }

    fn is_optional_type(&self, ty: &Type) -> bool {
        match ty {
            Type::Reference(reference) => self.is_external_object(&reference.elem),
            Type::Array(array) => self.names_external_crate_in_expr(&array.len),
            Type::Path(TypePath { qself: None, path }) => {
                !self.is_projection(path)
                    && (self.is_optional_path(&Cause::type_path(path))
                        || self.holds_external_object(path))
            }
            Type::TraitObject(object) => self.names_external_crate(object),
            _ => false,
        }
    }
}

/// What a [`View`] asks of the analysis it wraps, and the visit of a path. The view forwards the
/// `visit_*` methods the analysis overrides to it.
impl<A: Allow> AnalyzeVisitor<'_, A> {
    /// Checks `ty` where the visitor is, as [`View::check()`] does. Nothing stays recorded, also
    /// if the visit panics, which goes on ([`Checkpoint`]).
    pub(crate) fn check(&mut self, ty: &Type) -> CheckResult {
        let checkpoint = self.checkpoint();
        let selves = self.records.self_errors.len();
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            let mut result = self.found(ty);
            // `Self` without meaning fails the scope, which a check does not: it is told as a cause
            // that cannot be leaked.
            let without_meaning = &self.records.self_errors[selves..];
            if !without_meaning.is_empty() {
                result
                    .impossible
                    .extend(without_meaning.iter().map(|mention| mention.cause.clone()));
                result.optional = false;
                result.note.get_or_insert_with(|| SELF_NOTE.to_string());
                result.help.get_or_insert_with(|| SELF_HELP.to_string());
            }
            result
        }));
        self.restore(checkpoint);
        match outcome {
            Ok(result) => result,
            Err(payload) => resume_unwind(payload),
        }
    }

    /// Enters the nested scope `spec`, runs `f` and leaves it (also if `f` panics, which goes on).
    pub(crate) fn enter<R>(
        &mut self,
        spec: ScopeSpec,
        f: impl FnOnce(&mut View<'_, '_, A>) -> R,
    ) -> (ContextId, R) {
        let mark = self.enter_scope(&spec, false);
        // The closure may panic and the scope around may catch it: leave the scope as if it failed.
        match catch_unwind(AssertUnwindSafe(|| f(&mut View::new(self)))) {
            Ok(result) => (self.end_scope(mark, &spec), result),
            Err(payload) => {
                self.leave_scope(&mark);
                self.roll_back(&mark);
                resume_unwind(payload)
            }
        }
    }

    /// Records the primitive name that the allowed `path` starts with, if any: it is copied as it
    /// is, and may be shadowed (`type u16 = i32;`), so `Leaker::assertions()` checks it.
    fn push_primitive(&mut self, path: &Path) {
        if let Some(ident) = primitive_head(path) {
            self.frame.effects.push(Effect::Primitive(ident.clone()));
        }
    }

    /// Visits the path `i`. `asked` is the cause that the rule is asked about; `written` is the
    /// syntax that a mention of `Self` that has no meaning, or a relative path that is leaked, is
    /// reported at.
    fn visit_path_of(&mut self, i: &Path, asked: &Cause, written: &Cause) {
        let in_type = self.position.type_depth > 0;
        let starts_with_generic = self.starts_with_generic(i);
        let free = self.is_free_path(asked);
        if in_type {
            if starts_with_generic {
                let first = i.segments.first();
                if let Some(scope) = first.and_then(|seg| self.scope_of(&ident_name(&seg.ident))) {
                    self.mention_scope(scope);
                }
            }
            if self_head(i).is_some() {
                self.frame.mentions_self = true;
                self.check_self_has_meaning(written.clone());
            }
        }
        if free && !starts_with_generic {
            self.push_primitive(i);
            // An external crate path is resolved by name, which the Referrer may not share.
            if in_type && self.is_optional_path(asked) {
                self.frame.optional_depth = self.frame.optional_depth.min(self.position.type_depth);
            }
        } else if in_type && !starts_with_generic {
            let is_self = plain_self_head(i).is_some();
            if !is_self {
                self.push_must(written.clone(), Some(asked.bare()));
            }
        }
        syn::visit::visit_path(self, i)
    }

    /// Visits the type macro `i` (`m`), which cannot be analyzed: it is leaked whole.
    fn type_macro(&mut self, i: &Type, m: &TypeMacro) {
        let cause = Cause::Macro(m.mac.path.clone());
        let path = Some(&cause)
            .filter(|cause| self.is_relative_path(cause))
            .map(Cause::bare);
        self.push_must(Cause::Type(i.clone()), path);
        // The generics (and `Self`) are matched by name.
        let names = names_in(i);
        for name in &names {
            if let Some(scope) = self.scope_of(name) {
                self.mention_scope(scope);
            }
        }
        let mentions_self = names.contains("Self");
        self.frame.mentions_self |= mentions_self;
        if mentions_self {
            self.check_self_has_meaning(Cause::Type(i.clone()));
        }
    }

    /// Sets the hints that the lifetime of the reference `i` (`reference`) gives the types in it,
    /// and records an elided one as a cause.
    fn reference_hints(&mut self, i: &Type, reference: &TypeReference) {
        match &reference.lifetime {
            Some(lifetime) if lifetime.ident == "_" => self.position.hints.elided_ref = true,
            Some(lifetime) if lifetime.ident == "static" => (),
            Some(lifetime) => {
                let local = self
                    .scope_of(&lifetime.to_string())
                    .filter(|&scope| scope > 0);
                let bounded = trait_object_of(&reference.elem).map_or(false, has_lifetime_bound);
                match local {
                    Some(_) if bounded => {}
                    Some(scope) if self.is_unconstrained(lifetime, scope) => {
                        self.position.hints.unconstrained_ref = Some(lifetime.clone())
                    }
                    _ => {
                        self.position.hints.inherits =
                            Some(Inherits::Reference(self.position.type_depth))
                    }
                }
            }
            None => {
                self.push_elided(Cause::Type(i.clone()));
                self.position.hints.elided_ref = true;
            }
        }
    }

    /// Decides the type `i` just visited, whose parts were visited into `inner`, with the hints
    /// `ambiguous` and `inherits` that it was visited with.
    fn decide_type(
        &mut self,
        i: &Type,
        ambiguous: bool,
        inherits: Option<Inherits>,
        mut inner: Frame,
    ) {
        // Also parentheses and groups. A written bound inherits nothing; the reference carries it.
        let inherits = inherits.filter(|inherits| {
            self.impossibles.len() == inner.impossible_start
                && matches!(unwrap_type_group(i), Type::TraitObject(object)
                    if matches!(inherits, Inherits::Reference(_)) || !has_lifetime_bound(object))
        });
        match (self.position.expr_owner, inherits) {
            (Some(owner), _) => self.decide_as_owner(owner, inner),
            // Not leaked alone, but its parts can be, unless the trait object must be: then the
            // reference or path is.
            (None, Some(inherits)) => {
                let owner = inherits.owner();
                let own = inner.min_depth == self.position.type_depth
                    || inner.min_depth == owner
                    || self.is_ambiguous_owner(i, self.position.type_depth, &inner);
                if own {
                    self.decide_as_owner(owner, inner)
                } else {
                    if inner.optional_depth == self.position.type_depth
                        || inner.optional_depth == owner
                    {
                        inner.optional_depth = owner;
                    }
                    self.decide_through(inner)
                }
            }
            (None, None) => self.decide(i, self.position.type_depth, ambiguous, inner),
        }
    }

    /// Visits a struct, an enum or a union `ident` with `generics` by `f`: its `Self` is its type.
    fn with_adt(&mut self, ident: &Ident, generics: &Generics, f: impl FnOnce(&mut Self)) {
        let self_ty = SelfRequirement::of_item(ident, generics);
        self.with_item(generics, Some(self_ty), false, Vec::new(), f);
    }
}

impl<'ast, A: Allow> Visit<'ast> for AnalyzeVisitor<'_, A> {
    fn visit_type_param(&mut self, i: &TypeParam) {
        let alone = i.default.is_some() && self.is_alone_default(&i.ident);
        match &i.default {
            // A default analyzed alone is not visited again. That of a nested struct, enum or
            // union is visited with its `Self`.
            Some(default) if alone || self.self_scope.nested => {
                let without = TypeParam {
                    eq_token: None,
                    default: None,
                    ..i.clone()
                };
                syn::visit::visit_type_param(self, &without);
                if !alone {
                    let outer = std::mem::replace(&mut self.position.in_nested_default, true);
                    self.visit_type(default);
                    self.position.in_nested_default = outer;
                }
            }
            // As `syn::visit::visit_type_param()`, with the default marked as one.
            Some(default) => {
                for attr in &i.attrs {
                    self.visit_attribute(attr);
                }
                self.visit_ident(&i.ident);
                for bound in &i.bounds {
                    self.visit_type_param_bound(bound);
                }
                let outer = std::mem::replace(&mut self.position.in_param_default, true);
                self.visit_type(default);
                self.position.in_param_default = outer;
            }
            None => syn::visit::visit_type_param(self, i),
        }
    }

    fn visit_trait_item_fn(&mut self, i: &TraitItemFn) {
        for attr in &i.attrs {
            self.visit_attribute(attr);
        }
        // The default body is read only with `check_values`.
        self.method_signature(&i.sig, |visitor| {
            if let Some(block) = &i.default {
                visitor.check_body(&i.sig, block, true);
            }
        })
    }

    /// The signature of a method: its generics are local, and it is checked for E0195 unless the
    /// method is that of an inherent impl. A function item declares its own generics instead
    /// (`visit_item_fn()`).
    fn visit_signature(&mut self, i: &Signature) {
        self.method_signature(i, |_| ())
    }

    fn visit_trait_item_type(&mut self, i: &TraitItemType) {
        self.with_local_generics(&i.generics, |visitor| {
            syn::visit::visit_trait_item_type(visitor, i)
        });
    }

    fn visit_foreign_item_fn(&mut self, i: &ForeignItemFn) {
        // Not a method: its signature declares nothing here.
        self.visit_head(&i.attrs, &i.vis);
        self.with_arguments_of(&i.sig, |visitor| {
            syn::visit::visit_signature(visitor, &i.sig)
        });
    }

    fn visit_fn_arg(&mut self, i: &FnArg) {
        let elision = self
            .position
            .signature
            .as_ref()
            .map(|e| e.for_input(input_type(i)));
        self.elisions.push(elision);
        let outer = std::mem::replace(&mut self.position.in_fn_arg, true);
        syn::visit::visit_fn_arg(self, i);
        self.position.in_fn_arg = outer;
        self.elisions.pop();
    }

    fn visit_receiver(&mut self, i: &Receiver) {
        let outer = std::mem::replace(&mut self.position.in_receiver, true);
        syn::visit::visit_receiver(self, i);
        self.position.in_receiver = outer;
    }

    fn visit_trait_item_const(&mut self, i: &TraitItemConst) {
        let mut stripped = i.clone();
        stripped.default = None;
        self.with_local_generics(&i.generics, |visitor| {
            syn::visit::visit_trait_item_const(visitor, &stripped);
            if let Some((_, expr)) = &i.default {
                visitor.check_value(&i.ident, true, expr);
            }
        });
    }

    fn visit_impl_item_fn(&mut self, i: &ImplItemFn) {
        self.visit_head(&i.attrs, &i.vis);
        self.method_signature(&i.sig, |visitor| {
            visitor.check_body(&i.sig, &i.block, false)
        })
    }

    fn visit_impl_item_type(&mut self, i: &ImplItemType) {
        self.with_local_generics(&i.generics, |visitor| {
            syn::visit::visit_impl_item_type(visitor, i)
        });
    }

    fn visit_impl_item_const(&mut self, i: &ImplItemConst) {
        self.with_local_generics(&i.generics, |visitor| {
            visitor.visit_head(&i.attrs, &i.vis);
            visitor.visit_generics(&i.generics);
            visitor.visit_type(&i.ty);
            visitor.check_value(&i.ident, false, &i.expr);
        });
    }

    fn visit_item_const(&mut self, i: &ItemConst) {
        self.visit_head(&i.attrs, &i.vis);
        self.visit_generics(&i.generics);
        self.visit_type(&i.ty);
        self.check_value(&i.ident, false, &i.expr);
    }

    fn visit_item_static(&mut self, i: &ItemStatic) {
        self.visit_head(&i.attrs, &i.vis);
        self.visit_type(&i.ty);
        self.check_value(&i.ident, false, &i.expr);
    }

    fn visit_item_struct(&mut self, i: &ItemStruct) {
        self.with_adt(&i.ident, &i.generics, |visitor| {
            syn::visit::visit_item_struct(visitor, i)
        });
    }

    fn visit_item_enum(&mut self, i: &ItemEnum) {
        self.with_adt(&i.ident, &i.generics, |visitor| {
            syn::visit::visit_item_enum(visitor, i)
        });
    }

    fn visit_item_union(&mut self, i: &ItemUnion) {
        self.with_adt(&i.ident, &i.generics, |visitor| {
            syn::visit::visit_item_union(visitor, i)
        });
    }

    fn visit_item_trait(&mut self, i: &ItemTrait) {
        let self_ty = SelfRequirement::of_trait(&i.ident, &i.generics);
        let gats = Gat::of_items(&i.items);
        // The scope around may be the one entered for this trait (`Leaker::enter_trait()`, then
        // `View::visit_item()`): it has analyzed the defaults alone already.
        let alone = self.own_alone_defaults(&self_ty, &i.generics);
        self.with_item(&i.generics, Some(self_ty), false, gats, |visitor| {
            // Else the defaults that do not name `Self` are analyzed alone here: a copy of the
            // trait looks up those entries (`complete_args()`). An entry for the implementors
            // would not do: `Self: Tr<T, U>` names the parameter that it is the default of.
            let idents = alone.unwrap_or_else(|| visitor.analyze_defaults_alone(&i.generics));
            visitor.alone_defaults.push(AloneDefaults {
                scopes: visitor.scopes.len(),
                item_depth: visitor.item_depth,
                idents,
                declared: None,
            });
            syn::visit::visit_item_trait(visitor, i);
            visitor.alone_defaults.pop();
        });
    }

    fn visit_item_type(&mut self, i: &ItemType) {
        self.with_item_generics(&i.generics, Vec::new(), |visitor| {
            syn::visit::visit_item_type(visitor, i)
        });
    }

    fn visit_item_fn(&mut self, i: &ItemFn) {
        self.with_signature(&i.sig, true, false, |visitor| {
            visitor.visit_head(&i.attrs, &i.vis);
            syn::visit::visit_signature(visitor, &i.sig);
            visitor.check_body(&i.sig, &i.block, false);
        });
    }

    fn visit_item_mod(&mut self, i: &ItemMod) {
        // In an expression of a type (an array length), the module goes with the type: its items
        // are nested items with local generics, and it has no `Self`.
        if self.position.type_depth > 0 {
            return self.with_item(&Generics::default(), None, false, Vec::new(), |visitor| {
                syn::visit::visit_item_mod(visitor, i)
            });
        }
        // As an item, it is not analyzed: its names would be the copy's own in it.
        let ident = &i.ident;
        let cause = Cause::Unreadable(quote!(mod #ident));
        self.push_error(CheckResult {
            headline: Some(format!("cannot analyze the module `{}`", ident)),
            note: Some(
                "the Leaker analyzes no module: nothing in it is leaked, and the copy writes it as \
                 it is"
                    .to_string(),
            ),
            help: Some(
                "analyze each item of the module with its own call, or move it out of the item"
                    .to_string(),
            ),
            ..CheckResult::conflict(vec![cause.clone()], vec![cause], Reason::Unreadable)
        });
    }

    fn visit_item_impl(&mut self, i: &ItemImpl) {
        let self_ty = SelfRequirement::Type((*i.self_ty).clone());
        self.with_item(&i.generics, Some(self_ty), true, Vec::new(), |visitor| {
            // The trait is checked like a trait bound, but its arguments are analyzed in any case.
            if let Some((_, path, _)) = &i.trait_ {
                visitor.check_bound_path(path, Cause::Trait(path.clone()));
            }
            visitor.walk_item_impl(i)
        });
    }

    fn visit_item_trait_alias(&mut self, i: &ItemTraitAlias) {
        self.with_item_generics(&i.generics, Vec::new(), |visitor| {
            syn::visit::visit_item_trait_alias(visitor, i)
        });
    }

    fn visit_type(&mut self, i: &Type) {
        let outer = self.new_frame();
        let ArgHints {
            elided_ref,
            unconstrained_ref: unconstrained,
            ambiguous_arg: ambiguous,
            inherits,
        } = std::mem::take(&mut self.position.hints);
        self.position.type_depth += 1;
        self.impossibles.at = self.position.type_depth;
        let enclosing = self.position.enclosing.clone();
        self.position.enclosing = enclosing.components_of(i, self.scopes.len());
        match i {
            Type::Macro(m) => self.type_macro(i, m),
            Type::Paren(_) | Type::Group(_) => {
                // `W<(N)>` is unambiguously a type, unlike `W<N>` (also in an invisible group).
                self.position.hints = ArgHints {
                    elided_ref,
                    unconstrained_ref: unconstrained,
                    ambiguous_arg: ambiguous && matches!(i, Type::Group(_)),
                    inherits,
                };
                syn::visit::visit_type(self, i)
            }
            Type::Reference(reference) => {
                self.reference_hints(i, reference);
                syn::visit::visit_type(self, i)
            }
            Type::TraitObject(object) if elided_ref && !has_lifetime_bound(object) => {
                self.impossibles.push_soft(Cause::Type(i.clone()));
                syn::visit::visit_type(self, i)
            }
            Type::TraitObject(object) if unconstrained.is_some() && !has_lifetime_bound(object) => {
                if let Some(lifetime) = &unconstrained {
                    self.impossibles.push(Cause::Lifetime(lifetime.clone()));
                }
                self.frame.unconstrained = unconstrained;
                syn::visit::visit_type(self, i)
            }
            // `_` (`let v: Vec<_>`) is inferred where it is written, and no impl of the Repeater
            // can write it: a type that holds it is not leaked whole, and if it must be, it is a
            // conflict.
            Type::Infer(_) => {
                self.impossibles.push(Cause::Type(i.clone()));
                syn::visit::visit_type(self, i)
            }
            Type::ImplTrait(_) | Type::Verbatim(_) => {
                self.impossibles.push_soft(Cause::Type(i.clone()));
                // `x: impl Tr<Wrap<T>>` is `x: X` with `X: Tr<Wrap<T>>`: the bounds of the
                // `impl Trait` of an argument are predicates of the item, as those of a generic
                // are. Not those of a return type, which rustc proves where the body is.
                let in_env =
                    usize::from(self.position.in_fn_arg && matches!(i, Type::ImplTrait(_)));
                self.position.env_depth += in_env;
                syn::visit::visit_type(self, i);
                self.position.env_depth -= in_env;
            }
            _ => syn::visit::visit_type(self, i),
        }
        self.position.enclosing = enclosing;
        let inner = std::mem::replace(&mut self.frame, outer);
        self.decide_type(i, ambiguous, inherits, inner);
        self.position.type_depth -= 1;
        self.impossibles.at = self.position.type_depth;
    }

    fn visit_type_bare_fn(&mut self, i: &TypeBareFn) {
        let elision = Elision::of(
            self.position.type_depth,
            true,
            i.inputs.iter().map(|arg| &arg.ty),
            &i.output,
        );
        let mut output = output_lifetimes(i.lifetimes.as_ref(), |v| v.visit_return_type(&i.output));
        // An elided output has the only lifetime of the inputs, which the binder may declare.
        let elided = elision.as_ref().and_then(|elision| elision.kept.clone());
        output.extend(
            elided.filter(|kept| {
                binder_declares(i.lifetimes.as_ref(), kept) && !output.contains(kept)
            }),
        );
        let group = BinderGroup {
            output,
            inputs: i
                .inputs
                .iter()
                .map(|arg| BinderInput::Type(&arg.ty))
                .collect(),
        };
        // As `syn::visit::visit_type_bare_fn()`, with the inputs apart from the output.
        self.with_bound_lifetimes(i.lifetimes.as_ref(), vec![group], |v| {
            v.with_bound_elided(|v| {
                if let Some(lifetimes) = &i.lifetimes {
                    v.visit_bound_lifetimes(lifetimes);
                }
                if let Some(abi) = &i.abi {
                    v.visit_abi(abi);
                }
                for arg in &i.inputs {
                    v.elisions
                        .push(elision.as_ref().map(|e| e.for_input(&arg.ty)));
                    v.visit_bare_fn_arg(arg);
                    v.elisions.pop();
                }
                if let Some(variadic) = &i.variadic {
                    v.visit_bare_variadic(variadic);
                }
                v.in_output(&i.output);
            })
        })
    }

    fn visit_parenthesized_generic_arguments(&mut self, i: &ParenthesizedGenericArguments) {
        let elision = Elision::of(self.position.type_depth, true, &i.inputs, &i.output);
        self.with_bound_elided(|visitor| {
            for input in &i.inputs {
                let elision = elision.as_ref().map(|e| e.for_input(input));
                visitor.elisions.push(elision);
                visitor.visit_type(input);
                visitor.elisions.pop();
            }
            visitor.in_output(&i.output);
        })
    }

    fn visit_type_trait_object(&mut self, i: &TypeTraitObject) {
        for bound in &i.bounds {
            match bound {
                TypeParamBound::Trait(bound) => self.visit_trait_bound(bound),
                TypeParamBound::Verbatim(tokens) => {
                    self.impossibles.push(Cause::Unreadable(tokens.clone()))
                }
                TypeParamBound::Lifetime(lifetime) => self.visit_lifetime(lifetime),
                _ => (),
            }
        }
    }

    fn visit_type_param_bound(&mut self, i: &TypeParamBound) {
        // A bound that is not in a type is a predicate of the item (`env_depth`).
        let in_env = usize::from(self.position.type_depth == 0);
        self.position.env_depth += in_env;
        syn::visit::visit_type_param_bound(self, i);
        self.position.env_depth -= in_env;
    }

    fn visit_predicate_type(&mut self, i: &PredicateType) {
        // rustc checks each bound on its own, with the bounded type among its inputs.
        let groups = i
            .bounds
            .iter()
            .filter_map(|bound| match bound {
                TypeParamBound::Trait(bound) => Some(bound_group(
                    i.lifetimes.as_ref(),
                    bound,
                    Some(&i.bounded_ty),
                )),
                _ => None,
            })
            .collect();
        self.position.env_depth += 1;
        self.with_bound_lifetimes(i.lifetimes.as_ref(), groups, |v| {
            syn::visit::visit_predicate_type(v, i)
        });
        self.position.env_depth -= 1;
    }

    fn visit_trait_bound(&mut self, i: &TraitBound) {
        // A bare `?Sized` is the language's: rustc accepts `?` on no other trait, so a local trait
        // named `Sized` cannot change it. It is copied as written.
        if is_relaxed_sized(i) {
            return;
        }
        // A trait bound with a relative path is written as it is, with an error (for `Rel`, not
        // `? Rel`). The types in its arguments are written whole too (leaked in a predicate, they
        // would make a cycle, `env_cycle()`): their relative paths are paths of the error.
        let start = self.frame.effects.len();
        let relative =
            self.position.type_depth == 0 && self.check_bound_path(&i.path, bound_cause(i));
        let group = bound_group(i.lifetimes.as_ref(), i, None);
        self.with_bound_lifetimes(i.lifetimes.as_ref(), vec![group], |v| {
            // As `syn::visit::visit_trait_bound()`, with the path told to be that of a trait.
            v.visit_trait_bound_modifier(&i.modifier);
            if let Some(lifetimes) = &i.lifetimes {
                v.visit_bound_lifetimes(lifetimes);
            }
            let cause = Cause::Trait(i.path.clone());
            v.visit_path_of(&i.path, &cause, &cause);
        });
        if relative {
            let inside = self.frame.effects.split_off(start + 1);
            let part = self.bound_part(bound_cause(i), &i.path, primitive_names(&inside));
            let head = match &mut self.frame.effects[start] {
                Effect::Error(head) => head,
                _ => unreachable!("check_bound_path() pushed the error of the path"),
            };
            let mut inside = as_written(inside, &mut head.paths);
            head.part = Some(part);
            adopt(&mut inside, &head.origin.token);
            self.frame.effects.extend(inside);
        }
    }

    fn visit_const_param(&mut self, i: &ConstParam) {
        // The type of a const parameter cannot be an associated type path, so it is never leaked.
        // It must be a primitive type or a context-free path to an alias (`::my_crate::Idx`).
        for attr in &i.attrs {
            self.visit_attribute(attr);
        }
        match unwrap_type_group(&i.ty) {
            ty @ Type::Path(TypePath { qself: None, path }) if is_const_param_ty(ty) => {
                if let Some(ident) = path.get_ident() {
                    self.frame.effects.push(Effect::Primitive(ident.clone()));
                }
            }
            Type::Path(TypePath { qself: None, path })
                if self.is_free_path(&Cause::type_path(path)) => {}
            ty => {
                // The Visitor does not rewrite the type of a const parameter: no part. Its path
                // is written as it is.
                let paths = match ty {
                    Type::Path(TypePath { qself: None, path }) => {
                        vec![Cause::type_path(&without_arguments(path))]
                    }
                    _ => Vec::new(),
                };
                self.push_relative_path(
                    Cause::Type(i.ty.clone()),
                    Cause::Unreadable(quote!(#i)),
                    paths,
                );
                return;
            }
        }
        if let Some(expr) = &i.default {
            let outer = std::mem::replace(&mut self.position.in_param_default, true);
            self.read_expr(format!("the default of `{}`", i.ident), expr);
            self.position.in_param_default = outer;
        }
    }

    fn visit_variant(&mut self, i: &Variant) {
        // As `syn::visit::visit_variant()`, with the discriminant read as a value.
        for attr in &i.attrs {
            self.visit_attribute(attr);
        }
        self.visit_ident(&i.ident);
        self.visit_fields(&i.fields);
        if let Some((_, expr)) = &i.discriminant {
            self.read_expr(format!("the discriminant of `{}`", i.ident), expr);
        }
    }

    fn visit_path(&mut self, i: &Path) {
        // A mention of `Self` in it is read as one in a type.
        self.visit_path_of(i, &Cause::Unknown(i.clone()), &Cause::type_path(i))
    }

    fn visit_macro(&mut self, i: &Macro) {
        let cause = Cause::Macro(i.path.clone());
        self.visit_path_of(&i.path, &cause, &cause)
    }

    fn visit_type_path(&mut self, i: &TypePath) {
        // `Self::A` where `Self` is a type: a type that names it is not leaked whole.
        if i.qself.is_none()
            && i.path.segments.len() > 1
            && self_head(&i.path).is_some()
            && matches!(self.self_scope.ty, Some(SelfRequirement::Type(_)))
        {
            let cause = Cause::type_path(&i.path);
            self.impossibles.push(cause.clone());
            self.frame.assoc_self.get_or_insert(cause);
        }
        match &i.qself {
            Some(qself) if self.resolves_through_bounds(qself) => {
                // `<T>::X` goes through the bounds of `T`: only its arguments depend on the
                // context.
                self.visit_qself(qself);
                for seg in &i.path.segments {
                    self.visit_path_arguments(&seg.arguments);
                }
            }
            Some(qself) => {
                // As `syn::visit::visit_type_path()`, with the path told to be that of a qualified
                // type.
                self.visit_qself(qself);
                let (asked, written) = qualified_causes(&i.path, qself);
                self.visit_path_of(&i.path, &asked, &written);
            }
            None => {
                let cause = Cause::type_path(&i.path);
                self.visit_path_of(&i.path, &cause, &cause)
            }
        }
    }

    fn visit_qself(&mut self, i: &QSelf) {
        // `<T>::Ty` cannot be leaked, unless `T` is a generic parameter (or `Self`).
        if self.position.type_depth > 0 && i.as_token.is_none() && !self.resolves_through_bounds(i)
        {
            self.impossibles.push(Cause::Type((*i.ty).clone()));
        }
        syn::visit::visit_qself(self, i)
    }

    fn visit_lifetime(&mut self, i: &Lifetime) {
        if self.position.type_depth == 0 {
            return;
        }
        if i.ident == "_" {
            self.push_elided(Cause::Lifetime(i.clone()));
        } else if let Some(scope) = self.scope_of(&i.to_string()) {
            self.mention_scope(scope);
        } else if i.ident != "static" {
            self.push_must(Cause::Lifetime(i.clone()), None);
        }
    }

    fn visit_expr(&mut self, i: &Expr) {
        if self.position.type_depth > 0 {
            if let Expr::Closure(_) | Expr::Assign(_) | Expr::Verbatim(_) | Expr::Macro(_) = i {
                self.impossibles.push(Cause::Expr(i.clone()));
            }
        }
        let outer = self.position.expr_owner;
        if self.position.type_depth > 0 && outer.is_none() {
            self.position.expr_owner = Some(self.position.type_depth);
        }
        let enclosing = std::mem::take(&mut self.position.enclosing);
        syn::visit::visit_expr(self, i);
        self.position.enclosing = enclosing;
        self.position.expr_owner = outer;
    }

    fn visit_angle_bracketed_generic_arguments(&mut self, i: &AngleBracketedGenericArguments) {
        // A trait object without a lifetime bound as an argument of a path with lifetime arguments
        // may default to one of them, so the path is leaked whole if it must be.
        let mut elided = false;
        let mut inherited = false;
        let mut unconstrained_lifetime = None;
        let lifetimes: Vec<&Lifetime> = i
            .args
            .iter()
            .filter_map(|arg| match arg {
                GenericArgument::Lifetime(lifetime) => Some(lifetime),
                _ => None,
            })
            .collect();
        for lifetime in &lifetimes {
            if lifetime.ident == "_" {
                elided = true;
            } else if let Some(scope) = self.scope_of(&lifetime.to_string()) {
                // Local lifetimes are generics of the whole path.
                if scope > 0 && lifetimes.len() > 1 {
                    elided = true;
                } else {
                    inherited = true;
                    if scope > 0 && self.is_unconstrained(lifetime, scope) {
                        unconstrained_lifetime = Some((*lifetime).clone());
                    }
                }
            } else if lifetime.ident != "static" {
                inherited = true;
            }
        }
        // The path is leaked whole, which needs a type (not a trait bound).
        let elided = elided || (inherited && self.position.type_depth == 0);
        for arg in &i.args {
            if let GenericArgument::Type(ty) = arg {
                self.position.hints.elided_ref = elided;
                self.position.hints.unconstrained_ref =
                    unconstrained_lifetime.clone().filter(|_| !elided);
                if inherited && !elided && unconstrained_lifetime.is_none() {
                    self.position.hints.inherits = Some(Inherits::Path(self.position.type_depth));
                }
                // syn parses a bare identifier that may be a const as a type (see `ambiguous_arg`).
                self.position.hints.ambiguous_arg = is_bare_ident(ty);
            }
            self.visit_generic_argument(arg);
        }
    }

    fn visit_expr_path(&mut self, i: &ExprPath) {
        let (asked, written) = expr_causes(i);
        if self.position.type_depth == 0 {
            // Values starting with a generic in scope or `Self` do not depend on the context.
            let is_self = i.qself.is_none() && plain_self_head(&i.path).is_some();
            if is_self {
                self.check_self_has_meaning(Cause::of_expr_path(i));
            }
            if !((i.qself.is_none() && self.starts_with_generic(&i.path))
                || is_self
                || self.is_free_path(&asked))
            {
                // The path is written as it is, and its arguments are analyzed as usual.
                let paths = vec![asked.bare()];
                self.push_relative_path(Cause::of_expr_path(i), Cause::of_expr_path(i), paths);
            }
        }
        // As `syn::visit::visit_expr_path()`, as a value's path.
        for attr in &i.attrs {
            self.visit_attribute(attr);
        }
        if let Some(qself) = &i.qself {
            self.visit_qself(qself);
        }
        self.visit_path_of(&i.path, &asked, &written);
    }
}

/// The causes of the path of a qualified type or expression (`Tr<u8>::X` in `<T as Tr<u8>>::X`):
/// the one the rule is asked about (the trait, or the path of unknown kind for `<T>::X`), and the
/// one, held by no node, at which a mention of `Self` is reported.
fn qualified_causes(path: &Path, qself: &QSelf) -> (Cause, Cause) {
    let asked = if qself.as_token.is_some() && qself.position > 0 {
        Cause::Trait(trait_head(path, qself.position))
    } else {
        Cause::Unknown(path.clone())
    };
    (asked, Cause::Unreadable(quote!(#path)))
}

/// The causes of the path of the expression `i` as [`qualified_causes()`] has them: a value, or the
/// trait of a qualified path (`<u8 as Tr>::f`).
fn expr_causes(i: &ExprPath) -> (Cause, Cause) {
    match &i.qself {
        None => (Cause::expr_path(&i.path), Cause::expr_path(&i.path)),
        Some(qself) => qualified_causes(&i.path, qself),
    }
}

/// The primitive name that `path` starts with (`u8` in `u8::MAX`).
fn primitive_head(path: &Path) -> Option<&Ident> {
    path.segments
        .first()
        .map(|seg| &seg.ident)
        .filter(|ident| path.leading_colon.is_none() && is_primitive_name(ident))
}

/// The distinct paths of the types among `types` that are paths (bare identifiers), without
/// arguments.
fn paths_of_types(types: &[Type]) -> Vec<Cause> {
    let mut paths: Vec<Cause> = Vec::new();
    for ty in types {
        if let Type::Path(TypePath { qself: None, path }) = unwrap_type_group(ty) {
            push_once(&mut paths, Cause::type_path(&without_arguments(path)));
        }
    }
    paths
}

/// The distinct relative paths of `musts`, in order.
fn paths_of<'a>(musts: impl Iterator<Item = &'a Must>) -> Vec<Cause> {
    let mut paths: Vec<Cause> = Vec::new();
    for path in musts.filter_map(|must| must.path.as_ref()) {
        if !paths.contains(path) {
            paths.push(path.clone());
        }
    }
    paths
}

/// Looks for a path that `pred` accepts.
struct AnyPath<F> {
    /// Which paths count.
    pred: F,
    /// A path counted.
    found: bool,
}

impl<F: Fn(&Path) -> bool> AnyPath<F> {
    fn new(pred: F) -> Self {
        Self { pred, found: false }
    }
}

impl<'ast, F: Fn(&Path) -> bool> Visit<'ast> for AnyPath<F> {
    fn visit_path(&mut self, i: &'ast Path) {
        self.found |= (self.pred)(i);
        syn::visit::visit_path(self, i)
    }
    fn visit_trait_bound(&mut self, i: &'ast TraitBound) {
        // A bare `?Sized` is the language's relaxed bound, which no module can change.
        if !is_relaxed_sized(i) {
            syn::visit::visit_trait_bound(self, i)
        }
    }
}

/// The trait that an impl implements, as written.
#[derive(Clone)]
struct ImplTrait {
    /// The path as written.
    path: Path,
    /// The path does not depend on the context (see [`AnalyzeVisitor::is_free_path()`]), so the
    /// copy can name the trait with it.
    absolute: bool,
}

/// A `Self::A` that the Repeater's impl for the type `Self` cannot write (see
/// [`AnalyzeVisitor::associated_self_in_bounds()`]).
struct AssocSelf {
    /// The path, `Self::A` or longer.
    cause: Cause,
    /// The trait of the impl that `Self` is the self type of, if it is an impl of a trait (not a
    /// struct, an enum, a union or an inherent impl).
    trait_: Option<ImplTrait>,
}

impl AssocSelf {
    fn new(cause: Cause, trait_: Option<ImplTrait>) -> Self {
        Self { cause, trait_ }
    }

    /// The conflict of the type `ty` leaked whole, with the generics that name it. `paths`: the
    /// relative paths of `ty`.
    fn of(self, ty: &Type, paths: Vec<Cause>) -> CheckResult {
        let mut conflict = CheckResult {
            paths,
            ..CheckResult::conflict(
                vec![Cause::Type(ty.clone())],
                vec![self.cause.clone()],
                Reason::SelfProjection,
            )
        };
        self.give(&mut conflict, ty, true);
        conflict
    }

    /// Puts the headline, the note and the help in `conflict`, the conflict of `ty`. `in_bounds`:
    /// the path is in a bound of the generics of `ty`, not in `ty`.
    fn give(self, conflict: &mut CheckResult, ty: &Type, in_bounds: bool) {
        // The paths as a user writes them: without the spaces of a token stream.
        let cause = tidy(&self.cause.shown());
        let path = self.cause.path().cloned();
        let item = path
            .as_ref()
            .and_then(|path| path.segments.iter().nth(1))
            .map_or_else(|| cause.clone(), |seg| seg.ident.to_string());
        let qualified = |trait_: &str| match &path {
            Some(path) => {
                let rest = Path {
                    leading_colon: None,
                    segments: path.segments.iter().skip(1).cloned().collect(),
                };
                format!("<Self as {}>::{}", trait_, tidy(&quote!(#rest)))
            }
            None => format!("<Self as {}>::..", trait_),
        };
        let ty = tidy(&quote!(#ty));
        let place = if in_bounds {
            format!(" in a bound of the generics of `{}`", ty)
        } else if ty != cause {
            format!(" in `{}`", ty)
        } else {
            String::new()
        };
        // What the headline suggests, what the help replaces it with, why, and what else to do.
        let (suggested, replacement, why, how) = match &self.trait_ {
            Some(trait_) => {
                let written = &trait_.path;
                let written_text = tidy(&quote!(#written));
                let (replacement, how) = if trait_.absolute {
                    (qualified(&written_text), String::new())
                } else {
                    (
                        qualified(&absolute_example(written)),
                        format!(", with `{}` written as an absolute path", written_text),
                    )
                };
                (
                    qualified(&written_text),
                    replacement,
                    format!("the trait of this impl (`{}`)", written_text),
                    how,
                )
            }
            None => (
                qualified("SomeTrait"),
                qualified("SomeTrait"),
                "a trait that this type implements".to_string(),
                format!(
                    ", naming the trait that defines `{}` with its absolute path",
                    item
                ),
            ),
        };
        conflict.headline = Some(format!(
            "write `{}` instead of `{}`{}: the copy cannot tell which trait `{}` comes from",
            suggested, cause, place, item
        ));
        conflict.reason = Reason::SelfProjection;
        conflict.note = Some(format!(
            "`{}` is shorthand that rustc resolves through {}. In the copy it is written in an \
             impl of the Repeater, which has no such trait (E0220, or E0223 ambiguous associated \
             type), so the trait must be written out. `Self` itself is fine",
            cause, why
        ));
        conflict.help = Some(format!("replace `{}` with `{}`{}", cause, replacement, how));
    }
}

/// An absolute path for the example of a relative one: `Tr` is `::my_crate::Tr`.
fn absolute_example(path: &Path) -> String {
    let rest = without_crate_prefix(path);
    format!("::my_crate::{}", tidy(&quote!(#rest)))
}

/// The types to report for a method lifetime that the copy would leave unconstrained: while every
/// way in `anchors` (types by address, from an input down to an occurrence) meets a type that
/// `leaked` tells, the next such type on each way. A culprit is copied by its own path only.
fn unconstraining_types(anchors: &[Vec<usize>], leaked: impl Fn(usize) -> bool) -> Vec<usize> {
    let mut culprits: Vec<usize> = Vec::new();
    if anchors.is_empty() {
        return culprits;
    }
    // Where on each way the next look starts.
    let mut from = vec![0; anchors.len()];
    loop {
        let mut next = Vec::with_capacity(anchors.len());
        for (anchor, from) in anchors.iter().zip(&from) {
            match anchor[*from..].iter().position(|ty| leaked(*ty)) {
                Some(at) => next.push(*from + at),
                None => return culprits,
            }
        }
        for ((anchor, from), at) in anchors.iter().zip(from.iter_mut()).zip(next) {
            push_once(&mut culprits, anchor[at]);
            *from = at + 1;
        }
    }
}

/// Links the conflicts of `check_leaked_lifetimes()`: each gets the part that `made` has for its
/// type (without the primitive names), and as its outer conflict that of the innermost type in
/// `made` that holds its type.
fn link_unconstrained(
    conflicts: &mut [(usize, CheckResult)],
    made: &HashMap<usize, (Part, Vec<usize>)>,
) {
    for (ty, conflict) in conflicts.iter_mut() {
        conflict.part = made.get(ty).map(|(part, _)| Part {
            primitives: Vec::new(),
            ..part.clone()
        });
    }
    let tokens: Vec<(usize, Rc<()>)> = conflicts
        .iter()
        .map(|(ty, conflict)| (*ty, conflict.origin.token.clone()))
        .collect();
    for (ty, conflict) in conflicts.iter_mut() {
        // The type with the fewest types inside it among those that hold this one.
        let around = made
            .iter()
            .filter(|(outer, (_, inside))| *outer != ty && inside.contains(ty))
            .min_by_key(|(outer, (_, inside))| (inside.len(), **outer));
        let token = around.and_then(|(outer, _)| {
            tokens
                .iter()
                .find(|(candidate, _)| candidate == outer)
                .map(|(_, token)| token.clone())
        });
        conflict.origin.outer = token;
    }
}

/// Cuts the entries in the self type of an impl that conflict with the `Self` that `declared`
/// bound (`self_type_conflict()`). A required entry, or a reference with a required part, is an
/// error, copied as written; any other is no entry, and its optional keys go to `not_optional`.
fn cut_conflicts(
    effects: Vec<Effect>,
    declared: &Generics,
    seen: usize,
    place: Place,
    not_optional: &mut Vec<(Context, String)>,
) -> Vec<Effect> {
    let mut found = Vec::new();
    for effect in effects {
        let mut pending = match effect {
            Effect::Type(pending) => pending,
            other => {
                found.push(other);
                continue;
            }
        };
        if let Some(mut error) = self_type_conflict(&pending, declared) {
            let whole = is_whole_reference(&pending.ty);
            if pending.required_candidate && (pending.required || whole) {
                // The type is copied as written instead.
                let part = pending.part();
                let whole = part.whole;
                error.part = Some(part);
                error.seen = seen;
                error.place = place;
                let inner = pending.into_inner();
                let mut inner = if whole {
                    as_written(inner, &mut error.paths)
                } else {
                    cut_conflicts(inner, declared, seen, place, not_optional)
                };
                adopt(&mut inner, &error.origin.token);
                found.push(Effect::Error(Box::new(error)));
                found.extend(inner);
                continue;
            }
            pending.no_required_entry |= pending.required_candidate;
            pending.no_optional_entry |= pending.optional_candidate;
            if pending.optional_candidate && whole {
                // The optional entries under a reference cannot stand for it either.
                not_optional.extend(optional_keys(&pending.components));
            }
            if pending.optional {
                let ty = &pending.ty;
                not_optional.push((pending.context.clone(), quote!(#ty).to_string()));
                pending.optional = false;
            }
        }
        let components = std::mem::take(&mut pending.components);
        pending.components = cut_conflicts(components, declared, seen, place, not_optional);
        found.push(Effect::Type(pending));
    }
    found
}

/// Adds to `found` the contexts of the types in `effects` that may be entries of the copy
/// ([`AnalyzeVisitor::self_type_entries()`]): not one that `declared` cuts or `blocked` refuses,
/// and not the parts of a required entry.
fn entry_contexts(
    effects: &[Effect],
    declared: Option<&Generics>,
    blocked: &mut impl FnMut(&Context) -> bool,
    optional: bool,
    found: &mut Vec<Context>,
) {
    for effect in effects {
        let pending = match effect {
            Effect::Type(pending) => pending,
            _ => continue,
        };
        let must =
            pending.required || (pending.required_candidate && is_whole_reference(&pending.ty));
        let chosen = must || {
            let cut = declared.map_or(false, |declared| {
                self_type_conflict(pending, declared).is_some()
            });
            let entry = pending.required_candidate
                && !pending.no_required_entry
                && !blocked(&pending.context);
            let optional_entry = optional && pending.optional && !pending.no_optional_entry;
            (entry || optional_entry) && !cut
        };
        if chosen && !found.contains(&pending.context) {
            found.push(pending.context.clone());
        }
        if !must {
            entry_contexts(&pending.components, declared, blocked, optional, found);
        }
    }
}

/// Returns the first mention of `Self` in the bounds that the generics `declared` of an impl give
/// to `context`: the bounds of its parameters and its where predicates.
fn declared_self_mention(declared: &Generics, context: &Context) -> Option<Cause> {
    let names: HashSet<String> = declared.params.iter().map(generic_param_name).collect();
    let written: HashSet<String> = declared
        .where_clause
        .iter()
        .flat_map(|where_clause| where_clause.predicates.iter())
        .map(|predicate| quote!(#predicate).to_string())
        .collect();
    let predicates: Vec<WherePredicate> = context
        .generics
        .where_clause
        .iter()
        .flat_map(|where_clause| where_clause.predicates.iter())
        .filter(|predicate| written.contains(&quote!(#predicate).to_string()))
        .cloned()
        .collect();
    let own = Generics {
        params: context
            .generics
            .params
            .iter()
            .filter(|param| names.contains(&generic_param_name(param)))
            .cloned()
            .collect(),
        where_clause: where_clause(predicates),
        ..Generics::default()
    };
    first_self_in_generics(&own).or_else(|| {
        // A written `Self: 'a` moved to `self_outlives`, which is encoded as `&'a Self`.
        if context.self_outlives.is_empty() {
            return None;
        }
        first_self_in_generics(&Generics {
            where_clause: declared.where_clause.clone(),
            ..Generics::default()
        })
    })
}

/// The conflict of `pending`, an entry in the self type of an impl with the generics `declared`, if
/// its context depends on the `Self` that they bound (see [`AnalyzeVisitor::cut_self_cycles()`]).
fn self_type_conflict(pending: &PendingType, declared: &Generics) -> Option<CheckResult> {
    let ty = &pending.ty;
    let context = &pending.context;
    let self_ty = match &context.self_ty {
        Some(SelfRequirement::Type(self_ty)) => self_ty,
        _ => return None,
    };
    let cause = declared_self_mention(declared, context)?;
    let plain = quote!(#ty).to_string();
    let shown = tidy(&quote!(#ty));
    let what = if quote!(#self_ty).to_string() == plain {
        format!("`Self` is `{}`, the self type of the impl", shown)
    } else {
        format!(
            "`Self` is the self type `{}` of the impl, which contains `{}`",
            tidy(&quote!(#self_ty)),
            shown
        )
    };
    let help = format!(
        "write `{}` with an absolute path, or the bound of its generics without `{}`",
        shown,
        tidy(&cause.shown())
    );
    Some(CheckResult {
        note: Some(format!(
            "{}. The copy writes `{}` as a path through the Repeater, whose impl is for `Self`, \
             so the Referrer has no type to write for `Self`: it is the one being replaced",
            what, shown
        )),
        help: Some(help),
        headline: Some(format!(
            "`{}` is in the self type of the impl, and a bound of its generics names `Self`",
            shown
        )),
        paths: pending.paths.clone(),
        ..CheckResult::conflict(
            vec![Cause::Type(ty.clone())],
            vec![cause],
            Reason::SelfInImplSelfType,
        )
    })
}

/// Makes `conflict` the error of a type that would leave `lifetime`, a lifetime of the method,
/// unconstrained in the copy (E0195). `object`: the type is a trait object that defaults to the
/// lifetime, else it names it.
fn unconstrained_advice(conflict: &mut CheckResult, lifetime: &Lifetime, object: bool) {
    conflict.note = Some(format!(
        "{} `{1}`, a lifetime of the method which its return type mentions and no other argument \
         keeps as written (e.g. `&{1} self`), so the copy would have `{1}` only in a projection, \
         and rustc rejects the implementation of a method whose lifetime is early-bound where \
         the trait's is late-bound (E0195)",
        if object {
            "the lifetime of the trait object defaults to"
        } else {
            "it names"
        },
        lifetime
    ));
    let cause = conflict.must.first().map(|cause| tidy(&cause.shown()));
    conflict.help = cause.as_ref().map(|cause| {
        let mut options = vec![
            format!(
                "keep `{0}` in another argument of the method (for example `&{0} self`)",
                lifetime
            ),
            format!("write `{}` with an absolute path", cause),
        ];
        if object {
            options.push(format!(
                "write the bound of the trait object (`dyn Trait + {}`)",
                lifetime
            ));
        }
        or_list(options)
    });
    // The fault is that the copy would leave the lifetime unconstrained, not that the path is
    // relative, so naming an absolute path is not what the headline says.
    conflict.headline = cause.map(|cause| {
        format!(
            "`{}` leaves the lifetime `{}` of the method unconstrained in the copy (E0195)",
            cause, lifetime
        )
    });
    conflict.reason = Reason::UnconstrainedLifetime;
}

/// The advice for a type `ty` that uses the `for<..>` lifetime `lifetime` of an output that no
/// input keeps. `owned`: the type holding the binder would be leaked instead.
fn binder_advice(lifetime: &Lifetime, ty: &Type, owned: bool) -> Advice {
    let note = format!(
        "`{}` is declared by a `for<..>` binder and used in its output (a return type or an \
         associated type binding), and no input keeps it as written, so a type using it cannot be \
         leaked{}: the copy would have `{}` only in a projection, which does not constrain it \
         (E0581, E0582)",
        lifetime,
        if owned {
            " alone, and the type declaring the binder would be leaked as a whole instead"
        } else {
            ""
        },
        lifetime
    );
    let help = or_list(vec![
        format!(
            "keep an input that names `{0}` as written (for example `&{0} u8`)",
            lifetime
        ),
        format!("write `{}` with an absolute path", tidy(&quote!(#ty))),
    ]);
    Advice { note, help }
}

/// Where the elided lifetime of the output comes from, as the notes say it.
fn elision_source(elision: &Elision) -> &'static str {
    if elision.hidden {
        "the only lifetime of the inputs, which a path hides (the Leaker does not see it)"
    } else {
        "the only lifetime of the inputs"
    }
}

/// What to write where the copy cannot keep the elision of the signature: the lifetimes are named
/// (or the method gets a receiver), or the leaked type `absolute` is written with an absolute path.
fn elision_options(elision: &Elision, absolute: Option<&Type>) -> Vec<String> {
    let mut options = vec![if elision.hidden {
        "write the lifetime that the path hides (`Ref<'x, u8>`), and name it in the output"
            .to_string()
    } else {
        "name the lifetime of the inputs and of the output".to_string()
    }];
    if !elision.binder {
        options.push("give the method a receiver (`&self`)".to_string());
    }
    if let Some(ty) = absolute {
        options.push(format!(
            "write `{}` with an absolute path",
            tidy(&quote!(#ty))
        ));
    }
    options
}

/// The advice for the leaked type `ty`, which adds `added` lifetimes to the inputs of a signature
/// whose output relies on elision. `owned`: the type holding the signature would be leaked instead.
fn elision_advice(added: &[Lifetime], elision: &Elision, ty: &Type, owned: bool) -> Advice {
    // The lifetime that another input keeps is one more: rustc counts them for each input apart.
    let kept = elision
        .kept
        .as_ref()
        .map_or(false, |kept| added.contains(kept));
    let added: Vec<String> = added
        .iter()
        .map(|lifetime| format!("`{}`", lifetime))
        .collect();
    let note = format!(
        "the output `{}` has an elided lifetime, which elision takes from {}, but a leaked type in \
         the inputs has {} among its generics, and the copy names {} in the arguments of its \
         projection, so the inputs have more than one lifetime{} (E0106){}",
        tidy(&elision.output.shown()),
        elision_source(elision),
        added.join(", "),
        if added.len() == 1 { "it" } else { "them" },
        if kept {
            ", as rustc counts the lifetimes of each input apart"
        } else {
            ""
        },
        if owned {
            ", so a type using them cannot be leaked alone, and the type holding the signature \
             would be leaked as a whole instead"
        } else {
            ""
        }
    );
    Advice {
        note,
        help: or_list(elision_options(elision, Some(ty))),
    }
}

/// The advice for a signature whose elision the copy breaks by writing a type with `added`
/// lifetimes for `Self`.
fn self_elision_advice(added: &[Lifetime], elision: &Elision, owned: bool) -> Advice {
    let added: Vec<String> = added
        .iter()
        .map(|lifetime| format!("`{}`", lifetime))
        .collect();
    let note = format!(
        "the output `{}` has an elided lifetime, which elision takes from {}, but `Self` in the \
         inputs is a type that has {} among its generics, and the copy writes that type for `Self`, \
         so the inputs have more than one lifetime (E0106){}",
        tidy(&elision.output.shown()),
        elision_source(elision),
        added.join(", "),
        if owned {
            ", so the type holding the signature would be leaked as a whole instead"
        } else {
            ""
        }
    );
    Advice {
        note,
        help: or_list(elision_options(elision, None)),
    }
}

/// Returns the conflict of `ty`, which cannot be leaked whole (`conflict`) and holds the bare
/// identifiers `idents`, which may be consts. The advice says what a const needs. See [Bare
/// identifiers and consts](crate::Leaker#bare-identifiers-and-consts).
fn bare_conflict(mut conflict: CheckResult, ty: &Type, idents: &[Type]) -> CheckResult {
    let names: Vec<Cause> = idents.iter().cloned().map(Cause::Type).collect();
    let name = names
        .first()
        .map(|name| tidy(&name.shown()))
        .unwrap_or_default();
    // The type itself is no cause (a default that names `Self` says so in its note), and a
    // cause that is found twice is told once.
    let holder = Cause::Type(ty.clone());
    let mut seen: Vec<Cause> = Vec::new();
    conflict.impossible.retain(|cause| {
        if *cause == holder || seen.contains(cause) {
            return false;
        }
        seen.push(cause.clone());
        true
    });
    let absolute = format!(
        "write `{0}` with an absolute path (`::my_crate::{0}`, or `{{ ::my_crate::{0} }}` for a \
         const)",
        name
    );
    conflict.help = Some(match conflict.help.take() {
        // The advice of the cause may name the path already: a const needs braces.
        Some(help) => {
            let plain = format!("write `{}` with an absolute path", name);
            if help.contains(&plain) {
                help.replacen(&plain, &absolute, 1)
            } else if help.contains(", or ") {
                // It lists the options already: ours comes first.
                format!("{}, {}", absolute, help)
            } else {
                or_list(vec![absolute, help])
            }
        }
        None => absolute,
    });
    conflict.headline = Some(format!(
        "`{}` may be a const, and the type that holds it cannot be leaked as a whole",
        name
    ));
    conflict.reason = Reason::MaybeConst;
    // The first bare identifier is the cause, as it is the first of `must`.
    conflict.paths = paths_of_types(idents);
    conflict.must = names;
    conflict.cause = None;
    conflict.is_own = true;
    conflict
}

fn cycle_error(ty: &Type, why: String) -> CheckResult {
    CheckResult {
        note: Some(why),
        help: Some(format!(
            "write the leaked type of the bound with an absolute path, or use `{}` outside \
             the bounds and where predicates",
            tidy(&quote!(#ty))
        )),
        headline: Some(format!(
            "`{}` is in a bound or where predicate, and a bound of its generics names a leaked \
             type",
            tidy(&quote!(#ty))
        )),
        ..CheckResult::conflict(
            vec![Cause::Type(ty.clone())],
            Vec::new(),
            Reason::UnprovableBound,
        )
    }
}

/// The error for `ty`, which is in a predicate and needs `unstable`: like `cycle_error()`, with
/// the headline and the advice of what the bound names.
fn unstable_error(ty: &Type, unstable: &Unstable, why: String) -> CheckResult {
    let named = match &unstable.why {
        Instability::Leaked => return cycle_error(ty, why),
        Instability::Named(named) => named,
    };
    CheckResult {
        note: Some(why),
        help: Some(format!(
            "{}, or use `{}` outside the bounds and where predicates",
            named.advice(),
            tidy(&quote!(#ty))
        )),
        headline: Some(format!(
            "`{}` is in a bound or where predicate, and a bound of its generics names {}",
            tidy(&quote!(#ty)),
            named.names()
        )),
        ..CheckResult::conflict(
            vec![Cause::Type(ty.clone())],
            Vec::new(),
            Reason::UnprovableBound,
        )
    }
}

/// The conflict of `ty`, in the default of a parameter, which names `Self`: `paths` are its own.
fn self_in_default_conflict(ty: &Type, paths: Vec<Cause>) -> CheckResult {
    CheckResult {
        note: Some("a default of a parameter cannot depend on `Self`".to_string()),
        help: Some(format!(
            "name the type instead of `Self` in `{}`",
            tidy(&quote!(#ty))
        )),
        // The fault is not that the path is relative: the first cause is `Self`.
        headline: Some(format!(
            "`{}` is in the default of a parameter, and names `Self`",
            tidy(&quote!(#ty))
        )),
        // The default as a whole is what to change, the note is at the `Self` in it.
        cause: Some(Cause::Type(ty.clone())),
        paths,
        ..CheckResult::conflict(
            vec![first_self_in_type(ty).unwrap_or_else(Cause::self_type)],
            vec![Cause::Type(ty.clone())],
            Reason::SelfInDefault,
        )
    }
}

/// The conflict of `ty`, whose context names a `Self` that cannot be written (`unnameable`).
fn unnameable_conflict(ty: &Type, unnameable: UnnameableSelf, paths: Vec<Cause>) -> CheckResult {
    CheckResult {
        note: Some(unnameable.note),
        help: Some(format!(
            "{}, or write `{}` with an absolute path, so that it is not leaked",
            unnameable.help,
            tidy(&quote!(#ty))
        )),
        headline: Some(unnameable.headline),
        paths,
        ..CheckResult::conflict(
            vec![Cause::Type(ty.clone())],
            vec![*unnameable.cause],
            Reason::UnnameableSelf,
        )
    }
}

/// The conflict of `ty`, in the default of a parameter of a nested item, where a bound of its
/// generics names `Self` at `self_path`.
fn nested_default_conflict(ty: &Type, self_path: Cause, paths: Vec<Cause>) -> CheckResult {
    CheckResult {
        note: Some(
            "it names `Self`, which a struct, an enum or a union cannot write in the \
             default of a parameter (E0735), and the bounds of its generics mention the \
             `Self` of an item nested in the analyzed syntax too"
                .to_string(),
        ),
        help: Some("write the default without `Self`".to_string()),
        headline: Some(format!(
            "`{}` is in the default of a parameter, and it and a bound of its generics \
             name `Self`",
            tidy(&quote!(#ty))
        )),
        paths,
        ..CheckResult::conflict(
            vec![Cause::Type(ty.clone())],
            vec![self_path],
            Reason::SelfInDefault,
        )
    }
}

fn output_lifetimes(
    lifetimes: Option<&BoundLifetimes>,
    f: impl FnOnce(&mut OutputLifetimes),
) -> Vec<Lifetime> {
    let declared: Vec<Lifetime> = binder_lifetimes(lifetimes).collect();
    if declared.is_empty() {
        return Vec::new();
    }
    let mut visitor = OutputLifetimes::new(declared);
    f(&mut visitor);
    visitor.found
}

/// Returns the output and the inputs of the bound `bound` of the binder `lifetimes`. The inputs are
/// `bounded` and the trait's generic arguments. The bindings and the return type are the output.
fn bound_group<'a>(
    lifetimes: Option<&BoundLifetimes>,
    bound: &'a TraitBound,
    bounded: Option<&'a Type>,
) -> BinderGroup<'a> {
    let mut output = output_lifetimes(lifetimes, |v| v.visit_bound_output(bound));
    let mut inputs: Vec<_> = bounded.map(BinderInput::Type).into_iter().collect();
    match bound.path.segments.last().map(|seg| &seg.arguments) {
        Some(PathArguments::AngleBracketed(args)) => {
            inputs.extend(args.args.iter().filter_map(|arg| match arg {
                GenericArgument::Lifetime(lifetime) => Some(BinderInput::Lifetime(lifetime)),
                GenericArgument::Type(ty) => Some(BinderInput::Type(ty)),
                _ => None,
            }))
        }
        Some(PathArguments::Parenthesized(args)) => {
            inputs.extend(args.inputs.iter().map(BinderInput::Type));
            // An elided output has the only lifetime of the inputs, which the binder may declare.
            let kept =
                Elision::of(0, true, &args.inputs, &args.output).and_then(|elision| elision.kept);
            output.extend(
                kept.filter(|kept| binder_declares(lifetimes, kept) && !output.contains(kept)),
            );
        }
        _ => (),
    }
    BinderGroup { output, inputs }
}

/// The lifetimes of a binder that an output names.
struct OutputLifetimes {
    /// The lifetimes that the binder declares.
    declared: Vec<Lifetime>,
    /// The lifetimes that a binder inside declares again, where the walk is.
    shadowed: Vec<Lifetime>,
    /// The declared lifetimes that the output names, in order.
    found: Vec<Lifetime>,
}

impl OutputLifetimes {
    fn new(declared: Vec<Lifetime>) -> Self {
        Self {
            declared,
            shadowed: Vec::new(),
            found: Vec::new(),
        }
    }

    fn mention(&mut self, lifetime: &Lifetime) {
        if self.declared.contains(lifetime)
            && !self.shadowed.contains(lifetime)
            && !self.found.contains(lifetime)
        {
            self.found.push(lifetime.clone());
        }
    }

    fn shadowing(&mut self, lifetimes: Option<&BoundLifetimes>, walk: impl FnOnce(&mut Self)) {
        let outer = self.shadowed.len();
        self.shadowed.extend(binder_lifetimes(lifetimes));
        walk(self);
        self.shadowed.truncate(outer);
    }

    fn scan(&mut self, tokens: TokenStream) {
        let names = names_in(tokens);
        for lifetime in self.declared.clone() {
            if names.contains(&lifetime.to_string()) {
                self.mention(&lifetime);
            }
        }
    }

    /// Visits the output of a bound: the return type of `Fn(A) -> R` and the associated type
    /// bindings, not bounds on an associated type (`Iterator<Item: Tr<'x>>`).
    fn visit_bound_output(&mut self, bound: &TraitBound) {
        for seg in &bound.path.segments {
            match &seg.arguments {
                PathArguments::Parenthesized(args) => self.visit_return_type(&args.output),
                PathArguments::AngleBracketed(args) => {
                    for arg in &args.args {
                        if let GenericArgument::AssocType(binding) = arg {
                            self.visit_type(&binding.ty);
                        }
                    }
                }
                PathArguments::None => (),
            }
        }
    }
}

impl<'ast> Visit<'ast> for OutputLifetimes {
    fn visit_lifetime(&mut self, i: &'ast Lifetime) {
        self.mention(i);
    }
    fn visit_macro(&mut self, i: &'ast Macro) {
        self.scan(i.tokens.clone());
        syn::visit::visit_macro(self, i)
    }
    fn visit_type(&mut self, i: &'ast Type) {
        if let Type::Verbatim(tokens) = i {
            self.scan(tokens.clone());
        }
        syn::visit::visit_type(self, i)
    }
    fn visit_type_bare_fn(&mut self, i: &'ast TypeBareFn) {
        self.shadowing(i.lifetimes.as_ref(), |v| {
            syn::visit::visit_type_bare_fn(v, i)
        })
    }
    fn visit_trait_bound(&mut self, i: &'ast TraitBound) {
        self.shadowing(i.lifetimes.as_ref(), |v| {
            syn::visit::visit_trait_bound(v, i)
        })
    }
    fn visit_predicate_type(&mut self, i: &'ast PredicateType) {
        self.shadowing(i.lifetimes.as_ref(), |v| {
            syn::visit::visit_predicate_type(v, i)
        })
    }
}

/// Returns `true` if `bound` is a bare `?Sized`: the language's relaxed bound.
fn is_relaxed_sized(bound: &TraitBound) -> bool {
    matches!(bound.modifier, TraitBoundModifier::Maybe(_))
        && bound.lifetimes.is_none()
        && bound.path.leading_colon.is_none()
        && bound.path.segments.len() == 1
        && matches!(bound.path.segments.first(), Some(seg)
            if seg.ident == "Sized" && seg.arguments.is_none())
}

/// Returns `true` if `ty` is `Self` or `Self::A`, written as a path: not `<Self as Tr>::A`, whose
/// `Self` is a type of its own.
fn is_self_path(ty: &Type) -> bool {
    match ty {
        Type::Path(TypePath { qself: None, path }) => plain_self_head(path).is_some(),
        _ => false,
    }
}

/// Returns `true` if a bare identifier argument in `ty` may be a const, so `ty` is to be leaked
/// whole. Not so for a group or parentheses.
fn holds_bare_ident(ty: &Type, inner: &Frame) -> bool {
    !matches!(ty, Type::Group(_) | Type::Paren(_)) && !ambiguous_types(&inner.effects).is_empty()
}

/// Returns `true` if `ty` is a single identifier (`SIZE`), which may be a const as a generic
/// argument. Invisible groups (`$t:ty`) are transparent, parentheses are not.
fn is_bare_ident(ty: &Type) -> bool {
    match ty {
        Type::Group(TypeGroup { elem, .. }) => is_bare_ident(elem),
        Type::Path(TypePath { qself: None, path }) => path.get_ident().is_some(),
        _ => false,
    }
}

// The check of the values of an item (`Leaker::check_values`): const and static values, default
// consts and function bodies; enum discriminants and const parameter defaults always. The copy
// writes a value as it is, so a path in it that depends on where it is written is an error of a
// relative path. The walk follows the scopes of a body (`scopes`, `barrier`): bindings, block
// items, `use`s, generics and `Self` are local. A type in a type position (cast, `let`, closure,
// turbofish, qualified self type) is leaked as any type of the item, unless it names what the
// value declares or is in a macro or a block item (`reads`). What the walk cannot read (macro
// arguments that are no expressions, `macro_rules!`, a glob `use`, unparsed syntax) is rejected.

impl<'r, A: Allow> AnalyzeVisitor<'r, A> {
    /// Checks the value `expr` of the const or static `name`, if values are read. `default`: it is
    /// the default of an associated const of a trait.
    fn check_value(&mut self, name: &Ident, default: bool, expr: &Expr) {
        self.read_value("value", name, default, |walk| walk.visit_expr(expr));
    }

    /// Reads `expr` as a value, whether values are read or not: an enum discriminant or a const
    /// parameter default. `what` says where it is, as the notes do. `in_value` is not set: it marks
    /// only the values that `check_values` reads.
    fn read_expr(&mut self, what: String, expr: &Expr) {
        ValueWalk::new(self, what).visit_expr(expr);
    }

    /// Checks the body `block` of the function `sig`, if values are read. `default`: it is the
    /// default body of a method of a trait. The generics of `sig` are in scope.
    fn check_body(&mut self, sig: &Signature, block: &Block, default: bool) {
        self.read_value("body", &sig.ident, default, |walk| {
            for input in &sig.inputs {
                if let FnArg::Typed(arg) = input {
                    walk.visit_pat(&arg.pat);
                }
            }
            walk.visit_block(block);
        });
    }

    /// Reads the `kind` of `name` ("value", "body") with `walk`, if values are read: what it finds
    /// is in a value, and in the default of a trait item if `default`.
    fn read_value(
        &mut self,
        kind: &str,
        name: &Ident,
        default: bool,
        walk: impl FnOnce(&mut ValueWalk<'_, 'r, A>),
    ) {
        if !self.mode.check_values {
            return;
        }
        let what = format!(
            "the {}{} of `{}`",
            if default { "default " } else { "" },
            kind,
            name
        );
        let in_value = std::mem::replace(&mut self.position.in_value, true);
        let in_default = std::mem::replace(&mut self.position.in_trait_default, default);
        walk(&mut ValueWalk::new(self, what));
        self.position.in_value = in_value;
        self.position.in_trait_default = in_default;
    }
}

/// The names that a scope of a value declares.
#[derive(Default)]
struct ValueScope {
    /// The bindings of patterns: a `let`, a parameter, a match arm, a `for`.
    bindings: HashSet<String>,
    /// The items, the `use` declarations and the generics of items: named wherever they are seen.
    items: HashSet<String>,
}

/// The walk of a value: it rejects the relative paths and leaks the types in type positions.
struct ValueWalk<'a, 'r, A> {
    visitor: &'a mut AnalyzeVisitor<'r, A>,
    /// Where the walk is, as the notes say it: "the default value of `N`".
    what: String,
    /// Never empty. The first is the scope of the value itself.
    scopes: Vec<ValueScope>,
    /// The first scope whose bindings are seen. An item in a block sees the items around it, but
    /// not the bindings.
    barrier: usize,
    /// Greater than 0 where the walk reads a type itself instead of leaking it: in an item that a
    /// block declares, in the arguments of a macro (the visitor does not rewrite tokens), and in
    /// a type that names something the value declares.
    reads: usize,
    /// While such a type is read: the name it has that the value declares.
    declared_name: Option<String>,
}

impl<'a, 'r, A: Allow> ValueWalk<'a, 'r, A> {
    fn new(visitor: &'a mut AnalyzeVisitor<'r, A>, what: String) -> Self {
        Self {
            visitor,
            what,
            scopes: vec![ValueScope::default()],
            barrier: 0,
            reads: 0,
            declared_name: None,
        }
    }

    fn scope(&mut self) -> &mut ValueScope {
        self.scopes.last_mut().expect("a scope")
    }

    fn bind(&mut self, ident: &Ident) {
        self.scope().bindings.insert(ident_name(ident));
    }

    fn declare(&mut self, ident: &Ident) {
        self.scope().items.insert(ident_name(ident));
    }

    /// Declares a name that is not an identifier: a lifetime (`'a`).
    fn declare_name(&mut self, name: String) {
        self.scope().items.insert(name);
    }

    /// Returns `true` if the value declares `name` (an item, a generic, a lifetime or a binding),
    /// wherever it appears: in a path, a lifetime or the tokens of a macro.
    fn declares(&self, name: &str) -> bool {
        self.scopes.iter().any(|scope| scope.items.contains(name))
            || self.scopes[self.barrier..]
                .iter()
                .any(|scope| scope.bindings.contains(name))
    }

    /// Runs `f` in a new scope.
    fn scoped(&mut self, f: impl FnOnce(&mut Self)) {
        self.scopes.push(ValueScope::default());
        f(self);
        self.scopes.pop();
    }

    /// Returns `true` if the path starts with a name that the value declares: an item (a path of
    /// any length), or a binding (a single name).
    fn is_local(&self, path: &Path) -> bool {
        let first = match path
            .segments
            .first()
            .filter(|_| path.leading_colon.is_none())
        {
            Some(first) => first,
            None => return false,
        };
        let name = ident_name(&first.ident);
        self.scopes.iter().any(|scope| scope.items.contains(&name))
            || (path.segments.len() == 1
                && first.arguments.is_none()
                && self.scopes[self.barrier..]
                    .iter()
                    .any(|scope| scope.bindings.contains(&name)))
    }

    /// Returns `true` if the path of `cause` means the same wherever the value is written: it
    /// starts with `Self`, a generic, a name the value declares, or is absolute or allowed.
    fn is_known(&mut self, cause: &Cause) -> bool {
        let path = match cause.path() {
            Some(path) => path,
            None => return false,
        };
        // The receiver.
        if path.leading_colon.is_none() && path.get_ident().map_or(false, |ident| ident == "self") {
            return true;
        }
        if self_head(path).is_some() {
            self.visitor.check_self_has_meaning(Cause::expr_path(path));
            return true;
        }
        if self.is_local(path) || self.visitor.starts_with_generic(path) {
            return true;
        }
        if !self.visitor.is_free_path(cause) {
            return false;
        }
        self.visitor.push_primitive(path);
        true
    }

    /// Rejects the relative path `cause`, with `help` for what to write if it has to say more than
    /// the absolute path. The copy writes the path as it is.
    fn reject(&mut self, cause: Cause, help: Option<String>) {
        let entry = cause.path().map(|_| cause.bare());
        self.reject_as(cause, entry, help)
    }

    /// Like `reject()`. `entry`: the path of `cause`, for
    /// [`ErrorItem::paths`](crate::ErrorItem::paths).
    fn reject_as(&mut self, cause: Cause, entry: Option<Cause>, help: Option<String>) {
        // The error names a macro in its own way, with its `!`, whereas the path of a macro in a
        // type is shown by its path alone.
        let headline = match &cause {
            Cause::Macro(_) => Some(format!("use absolute path: '{}!'", tidy(&cause.shown()))),
            _ => None,
        };
        let mut note = format!("it is in {}, which the copy writes as it is", self.what);
        if let Some(name) = &self.declared_name {
            note.push_str(&format!(
                "; the type names `{}`, which the value declares, so it is not leaked",
                name
            ));
        }
        // The path is relative: it names no generic, no `Self` and no name of the value.
        let paths: Vec<Cause> = entry.into_iter().collect();
        self.visitor.push_error(CheckResult {
            note: Some(note),
            help,
            headline,
            paths,
            ..CheckResult::conflict(vec![cause.clone()], vec![cause], Reason::RelativePath)
        });
    }

    /// Rejects the syntax `tokens`, which the walk cannot read: `why` is what it is.
    fn cannot_read(&mut self, tokens: TokenStream, why: &str) {
        let shown = tidy(&tokens);
        let note = format!(
            "{}. The Leaker must see every path in {}, which the copy writes as it is",
            why, self.what
        );
        self.visitor.push_error(CheckResult {
            headline: Some(format!("cannot analyze `{}` in {}", shown, self.what)),
            note: Some(note),
            help: Some(
                "write it with syntax the Leaker can read (paths and expressions), or move it \
                 out of the item"
                    .to_string(),
            ),
            ..CheckResult::conflict(
                vec![Cause::Unreadable(tokens.clone())],
                vec![Cause::Unreadable(tokens)],
                Reason::Unreadable,
            )
        });
    }

    /// Declares the names that `item` introduces in the block around it, where they are visible
    /// from the first statement.
    fn declare_item(&mut self, item: &Item) {
        for name in declared_names(item) {
            self.declare(&name);
        }
    }

    /// Handles a type in a type position of the value: it is leaked as any type of the item is,
    /// unless it names something the value declares or holds syntax that `syn` leaves unparsed.
    /// Then it is read here, and a relative path in it is rejected.
    fn type_position(&mut self, ty: &Type) {
        let mut scan = Scan {
            walk: self,
            declared: None,
            unreadable: false,
        };
        scan.visit_type(ty);
        let (declared, unreadable) = (scan.declared, scan.unreadable);
        // A value inside a type (the length of an array) is part of that type, which is leaked
        // whole or not at all: its types are not entries of their own.
        let inside_type = self.visitor.position.type_depth > 0;
        if declared.is_none() && !unreadable && !inside_type {
            return self.leak(ty);
        }
        let outer = std::mem::replace(&mut self.declared_name, declared);
        self.reads += 1;
        self.visit_type(ty);
        self.reads -= 1;
        self.declared_name = outer;
    }

    /// Analyzes `ty` as a type of the item, where the visitor is, but outside the signatures
    /// around: a type of a body is no input, so it does not decide E0195.
    fn leak(&mut self, ty: &Type) {
        let signatures = std::mem::take(&mut self.visitor.signatures);
        let outcome = catch_unwind(AssertUnwindSafe(|| self.visitor.visit_type(ty)));
        self.visitor.signatures = signatures;
        if let Err(payload) = outcome {
            resume_unwind(payload);
        }
    }

    /// Checks a path that is a name alone, as a `use` root, a macro argument or a pattern.
    fn check_name(&mut self, ident: &Ident) {
        let cause = Cause::expr_path(&Path::from(ident.clone()));
        if !self.is_known(&cause) {
            self.reject(cause, None);
        }
    }

    /// Checks the path `<ty as Trait>::A` or `<ty>::A`: the trait and the arguments, and the type.
    fn qualified<'ast>(&mut self, qself: Option<&'ast QSelf>, path: &'ast Path) {
        let qself = match qself {
            Some(qself) => qself,
            None => return self.visit_path(path),
        };
        self.visit_type(&qself.ty);
        // The first `position` segments are the trait, the others are names of the type's items.
        let position = if qself.as_token.is_some() {
            qself.position
        } else {
            0
        };
        if position > 0 {
            let cause = Cause::Trait(trait_head(path, position));
            if !self.is_known(&cause) {
                self.reject(cause, None);
            }
            for seg in path.segments.iter().take(position) {
                self.visit_path_arguments(&seg.arguments);
            }
        }
        for seg in path.segments.iter().skip(position) {
            self.visit_path_arguments(&seg.arguments);
        }
    }

    /// Checks the root of a `use` tree, which names a crate, a module or an item of the value.
    fn use_roots(&mut self, tree: &UseTree, absolute: bool) {
        let ident = match tree {
            UseTree::Path(p) => &p.ident,
            UseTree::Name(n) => &n.ident,
            UseTree::Rename(r) => &r.ident,
            UseTree::Glob(_) => return,
            UseTree::Group(g) => {
                for tree in &g.items {
                    self.use_roots(tree, absolute);
                }
                return;
            }
        };
        let path = Path::from(ident.clone());
        let root = Cause::Unknown(path.clone());
        if !absolute && !self.is_local(&path) && !self.visitor.is_free_path(&root) {
            // What the root names (a crate, a module, a type, a trait) is not known.
            self.reject_as(Cause::expr_path(&path), Some(root), None);
        }
    }

    /// Checks a macro call: the path, then the arguments, which are read as expressions. A type in
    /// them is read, not leaked: the visitor does not rewrite the tokens of a macro.
    fn macro_call(&mut self, mac: &Macro) {
        let path = &mac.path;
        let cause = Cause::Macro(path.clone());
        if !self.is_known(&cause) {
            let help = macro_help(path);
            self.reject(cause, Some(help));
        }
        self.reads += 1;
        self.macro_tokens(mac);
        self.reads -= 1;
    }

    fn macro_tokens(&mut self, mac: &Macro) {
        let path = &mac.path;
        let tokens = &mac.tokens;
        // These name no item in their arguments (`cfg!(unix)`, `stringify!(a b)`).
        if is_std_macro(path, &["cfg", "stringify"]) {
            return;
        }
        if is_std_macro(path, &["matches"]) {
            if let Ok((expr, pat, guard)) = parse_matches.parse2(tokens.clone()) {
                self.visit_expr(&expr);
                self.scoped(|walk| {
                    walk.visit_pat(&pat);
                    if let Some(guard) = &guard {
                        walk.visit_expr(guard);
                    }
                });
                return;
            }
        }
        match macro_exprs(tokens) {
            Some(args) => self.macro_arguments(&args, format_string(path)),
            None => self.cannot_read(
                quote!(#path ! (#tokens)),
                "the arguments of the macro are not expressions",
            ),
        }
    }

    /// Checks the arguments of a macro. A name in `name = value` is the name of an argument, which
    /// a format string captures as `{name}`, as it captures a variable that is in scope.
    /// `format` says which string literal is a format string.
    fn macro_arguments(&mut self, args: &[Expr], format: FormatString) {
        let named: Vec<String> = args.iter().filter_map(argument_name).collect();
        for (at, arg) in args.iter().enumerate() {
            match arg {
                Expr::Assign(assign) if argument_name(arg).is_some() => {
                    self.visit_expr(&assign.right)
                }
                Expr::Lit(ExprLit {
                    lit: Lit::Str(literal),
                    ..
                }) if format.is_at(at) => {
                    for name in captured_names(&literal.value()) {
                        if named.contains(&name) {
                            continue;
                        }
                        self.check_name(&Ident::new(&name, literal.span()));
                    }
                }
                _ => self.visit_expr(arg),
            }
        }
    }
}

impl<'ast, A: Allow> Visit<'ast> for ValueWalk<'_, '_, A> {
    fn visit_attribute(&mut self, _: &'ast Attribute) {}

    fn visit_visibility(&mut self, _: &'ast Visibility) {}

    /// Any path that no other method takes is told as a value, as its error is.
    fn visit_path(&mut self, i: &'ast Path) {
        if !self.is_known(&Cause::expr_path(i)) {
            // Copied as written: the types in its arguments are analyzed as usual.
            self.reject(Cause::expr_path(i), None);
        }
        syn::visit::visit_path(self, i)
    }

    fn visit_expr_path(&mut self, i: &'ast ExprPath) {
        self.qualified(i.qself.as_ref(), &i.path)
    }

    fn visit_type_path(&mut self, i: &'ast TypePath) {
        self.qualified(i.qself.as_ref(), &i.path)
    }

    fn visit_macro(&mut self, i: &'ast Macro) {
        self.macro_call(i)
    }

    fn visit_expr(&mut self, i: &'ast Expr) {
        match i {
            Expr::Verbatim(tokens) => self.cannot_read(tokens.clone(), "syn does not parse it"),
            _ => syn::visit::visit_expr(self, i),
        }
    }

    fn visit_type(&mut self, i: &'ast Type) {
        if self.reads == 0 {
            return self.type_position(i);
        }
        match i {
            Type::Verbatim(tokens) => self.cannot_read(tokens.clone(), "syn does not parse it"),
            _ => syn::visit::visit_type(self, i),
        }
    }

    fn visit_type_param_bound(&mut self, i: &'ast TypeParamBound) {
        match i {
            TypeParamBound::Verbatim(tokens) => {
                self.cannot_read(tokens.clone(), "syn does not parse it")
            }
            _ => syn::visit::visit_type_param_bound(self, i),
        }
    }

    fn visit_pat(&mut self, i: &'ast Pat) {
        match i {
            Pat::Verbatim(tokens) => self.cannot_read(tokens.clone(), "syn does not parse it"),
            _ => syn::visit::visit_pat(self, i),
        }
    }

    /// A name in a pattern is a binding, unless it is a path to a constant, a unit struct or a
    /// variant. Those are written with a capital letter (the lint `non_snake_case` makes a binding
    /// with one a warning), and a `ref` or `mut` name or one with `@` is a binding for certain.
    fn visit_pat_ident(&mut self, i: &'ast PatIdent) {
        let name = ident_name(&i.ident);
        let capital = name.chars().next().map_or(false, char::is_uppercase);
        let binding =
            i.by_ref.is_some() || i.mutability.is_some() || i.subpat.is_some() || !capital;
        if binding {
            self.bind(&i.ident);
        } else {
            self.check_name(&i.ident);
        }
        if let Some((_, pat)) = &i.subpat {
            self.visit_pat(pat);
        }
    }

    fn visit_generics(&mut self, i: &'ast Generics) {
        for param in &i.params {
            match param {
                GenericParam::Type(param) => self.declare(&param.ident),
                GenericParam::Const(param) => self.declare(&param.ident),
                GenericParam::Lifetime(param) => self.declare_name(param.lifetime.to_string()),
            }
        }
        syn::visit::visit_generics(self, i)
    }

    fn visit_block(&mut self, i: &'ast Block) {
        self.scoped(|walk| {
            // An item is visible in the whole block.
            for stmt in &i.stmts {
                if let Stmt::Item(item) = stmt {
                    walk.declare_item(item);
                }
            }
            for stmt in &i.stmts {
                walk.visit_stmt(stmt);
            }
        })
    }

    /// The initializer does not see the bindings of the pattern.
    fn visit_local(&mut self, i: &'ast Local) {
        if let Some(init) = &i.init {
            self.visit_expr(&init.expr);
            if let Some((_, diverge)) = &init.diverge {
                self.visit_expr(diverge);
            }
        }
        self.visit_pat(&i.pat);
    }

    fn visit_item(&mut self, i: &'ast Item) {
        // The item sees the items around it, and the generics of its own, not the bindings.
        let outer = std::mem::replace(&mut self.barrier, self.scopes.len());
        // The types of an item that a block declares are read, not leaked.
        self.reads += 1;
        self.scoped(|walk| match i {
            Item::Use(item) => {
                if has_glob(&item.tree) {
                    walk.cannot_read(
                        quote!(#item),
                        "a glob import hides which names it brings into scope",
                    );
                }
                walk.use_roots(&item.tree, item.leading_colon.is_some());
            }
            Item::Macro(item) => walk.cannot_read(
                quote!(#item),
                "a macro in item position may declare any name",
            ),
            Item::Verbatim(tokens) => walk.cannot_read(tokens.clone(), "syn does not parse it"),
            _ => syn::visit::visit_item(walk, i),
        });
        self.reads -= 1;
        self.barrier = outer;
    }

    fn visit_item_mod(&mut self, i: &'ast ItemMod) {
        if let Some((_, items)) = &i.content {
            for item in items {
                self.declare_item(item);
            }
        }
        syn::visit::visit_item_mod(self, i)
    }

    fn visit_impl_item(&mut self, i: &'ast ImplItem) {
        self.scoped(|walk| syn::visit::visit_impl_item(walk, i))
    }

    fn visit_trait_item(&mut self, i: &'ast TraitItem) {
        self.scoped(|walk| syn::visit::visit_trait_item(walk, i))
    }

    fn visit_expr_closure(&mut self, i: &'ast ExprClosure) {
        self.scoped(|walk| {
            // `for<'a> |x: &'a u8| ..` declares a lifetime that only the closure knows.
            for param in i.lifetimes.iter().flat_map(|binder| &binder.lifetimes) {
                if let GenericParam::Lifetime(param) = param {
                    walk.declare_name(param.lifetime.to_string());
                }
            }
            for input in &i.inputs {
                walk.visit_pat(input);
            }
            walk.visit_return_type(&i.output);
            walk.visit_expr(&i.body);
        })
    }

    /// The bindings of the pattern are those of the guard and the body.
    fn visit_arm(&mut self, i: &'ast Arm) {
        self.scoped(|walk| {
            walk.visit_pat(&i.pat);
            if let Some((_, guard)) = &i.guard {
                walk.visit_expr(guard);
            }
            walk.visit_expr(&i.body);
        })
    }

    /// The bindings of `if let` are those of the first branch.
    fn visit_expr_if(&mut self, i: &'ast ExprIf) {
        self.scoped(|walk| {
            walk.visit_expr(&i.cond);
            walk.visit_block(&i.then_branch);
        });
        if let Some((_, branch)) = &i.else_branch {
            self.visit_expr(branch);
        }
    }

    fn visit_expr_while(&mut self, i: &'ast ExprWhile) {
        self.scoped(|walk| {
            walk.visit_expr(&i.cond);
            walk.visit_block(&i.body);
        })
    }

    fn visit_expr_let(&mut self, i: &'ast ExprLet) {
        self.visit_expr(&i.expr);
        self.visit_pat(&i.pat);
    }

    fn visit_expr_for_loop(&mut self, i: &'ast ExprForLoop) {
        self.visit_expr(&i.expr);
        self.scoped(|walk| {
            walk.visit_pat(&i.pat);
            walk.visit_block(&i.body);
        })
    }
}

/// What a type of a value names that decides how it is read: a name the value declares, and syntax
/// that `syn` leaves unparsed.
struct Scan<'w, 'a, 'r, A> {
    walk: &'w ValueWalk<'a, 'r, A>,
    /// The first name that the value declares (an item, a generic or a lifetime of an item in a
    /// block, a `use`).
    declared: Option<String>,
    /// The type holds syntax that `syn` leaves unparsed (`Verbatim`), or an item.
    unreadable: bool,
}

impl<A: Allow> Scan<'_, '_, '_, A> {
    fn name(&mut self, name: &str) {
        if self.declared.is_none() && self.walk.declares(name) {
            self.declared = Some(name.to_string());
        }
    }
}

impl<'ast, A: Allow> Visit<'ast> for Scan<'_, '_, '_, A> {
    fn visit_path(&mut self, i: &'ast Path) {
        if self.declared.is_none() && self.walk.is_local(i) {
            self.declared = i.segments.first().map(|seg| ident_name(&seg.ident));
        }
        syn::visit::visit_path(self, i)
    }

    fn visit_lifetime(&mut self, i: &'ast Lifetime) {
        self.name(&i.to_string());
    }

    /// A type macro, or a macro in a length: the names in its tokens are not told apart.
    fn visit_macro(&mut self, i: &'ast Macro) {
        let mut names: Vec<String> = names_in(&i.tokens).into_iter().collect();
        names.sort();
        for name in names {
            self.name(&name);
        }
        syn::visit::visit_macro(self, i)
    }

    fn visit_type(&mut self, i: &'ast Type) {
        match i {
            Type::Verbatim(_) => self.unreadable = true,
            _ => syn::visit::visit_type(self, i),
        }
    }

    fn visit_type_param_bound(&mut self, i: &'ast TypeParamBound) {
        match i {
            TypeParamBound::Verbatim(_) => self.unreadable = true,
            _ => syn::visit::visit_type_param_bound(self, i),
        }
    }

    fn visit_expr(&mut self, i: &'ast Expr) {
        match i {
            Expr::Verbatim(_) => self.unreadable = true,
            _ => syn::visit::visit_expr(self, i),
        }
    }

    fn visit_pat(&mut self, i: &'ast Pat) {
        match i {
            Pat::Verbatim(_) => self.unreadable = true,
            _ => syn::visit::visit_pat(self, i),
        }
    }

    /// An item in the length of an array declares names this scan does not follow.
    fn visit_item(&mut self, _: &'ast Item) {
        self.unreadable = true;
    }
}

fn has_glob(tree: &UseTree) -> bool {
    match tree {
        UseTree::Path(p) => has_glob(&p.tree),
        UseTree::Glob(_) => true,
        UseTree::Group(g) => g.items.iter().any(has_glob),
        UseTree::Name(_) | UseTree::Rename(_) => false,
    }
}

/// The arguments of a macro call as expressions: `a, b` or `a; n` (`vec![0; n]`).
fn macro_exprs(tokens: &TokenStream) -> Option<Vec<Expr>> {
    if let Ok(args) = Punctuated::<Expr, Token![,]>::parse_terminated.parse2(tokens.clone()) {
        return Some(args.into_iter().collect());
    }
    let repeat = |input: ParseStream| -> Result<Vec<Expr>> {
        let value: Expr = input.parse()?;
        input.parse::<Token![;]>()?;
        let count: Expr = input.parse()?;
        Ok(vec![value, count])
    };
    repeat.parse2(tokens.clone()).ok()
}

/// Returns `true` if `path` is the macro of the standard library `names`: `cfg!` or
/// `::core::cfg!`. (Another path ending in the same name is a macro of its own.)
fn is_std_macro(path: &Path, names: &[&str]) -> bool {
    let name = match path.segments.last() {
        Some(seg) => seg.ident.to_string(),
        None => return false,
    };
    names.contains(&name.as_str())
        && match path.segments.len() {
            1 => true,
            2 => {
                path.leading_colon.is_some()
                    && matches!(path.segments.first(), Some(seg) if seg.ident == "core" || seg.ident == "std")
            }
            _ => false,
        }
}

/// The arguments of `matches!(expr, pattern if guard)`.
fn parse_matches(input: ParseStream) -> Result<(Expr, Pat, Option<Expr>)> {
    let expr: Expr = input.parse()?;
    input.parse::<Token![,]>()?;
    let pat = Pat::parse_multi_with_leading_vert(input)?;
    let guard = if input.peek(Token![if]) {
        input.parse::<Token![if]>()?;
        Some(input.parse()?)
    } else {
        None
    };
    input.parse::<Option<Token![,]>>()?;
    Ok((expr, pat, guard))
}

/// The name of an argument of the form `name = value`.
fn argument_name(arg: &Expr) -> Option<String> {
    match arg {
        Expr::Assign(assign) => match &*assign.left {
            Expr::Path(ExprPath {
                qself: None, path, ..
            }) => path.get_ident().map(ident_name),
            _ => None,
        },
        _ => None,
    }
}

/// Where the format string of a macro is among its arguments.
#[derive(Clone, Copy)]
enum FormatString {
    /// The macro is one of the standard library, and has none (`vec!`, `dbg!`).
    None,
    /// The macro is one of the standard library: the argument at this position (`println!("..")`
    /// is the first, `write!(f, "..")` the second).
    At(usize),
    /// Any macro: every string literal among the arguments may be one.
    Any,
}

impl FormatString {
    fn is_at(self, at: usize) -> bool {
        match self {
            FormatString::None => false,
            FormatString::At(position) => position == at,
            FormatString::Any => true,
        }
    }
}

/// The macros of the standard library: the crate to name for one (see `macro_help()`), and where
/// its format string is (see `format_string()`; `Any` for one that formats nothing of its own).
const STD_MACROS: &[(&str, &str, FormatString)] = &[
    ("assert", "core", FormatString::At(1)),
    ("assert_eq", "core", FormatString::At(2)),
    ("assert_ne", "core", FormatString::At(2)),
    ("cfg", "core", FormatString::Any),
    ("column", "core", FormatString::None),
    ("compile_error", "core", FormatString::None),
    ("concat", "core", FormatString::None),
    ("debug_assert", "core", FormatString::At(1)),
    ("debug_assert_eq", "core", FormatString::At(2)),
    ("debug_assert_ne", "core", FormatString::At(2)),
    ("env", "core", FormatString::None),
    ("file", "core", FormatString::None),
    ("format_args", "core", FormatString::At(0)),
    ("include_bytes", "core", FormatString::None),
    ("include_str", "core", FormatString::None),
    ("line", "core", FormatString::None),
    ("matches", "core", FormatString::Any),
    ("module_path", "core", FormatString::None),
    ("option_env", "core", FormatString::None),
    ("panic", "core", FormatString::At(0)),
    ("stringify", "core", FormatString::Any),
    ("todo", "core", FormatString::At(0)),
    ("unimplemented", "core", FormatString::At(0)),
    ("unreachable", "core", FormatString::At(0)),
    ("write", "core", FormatString::At(1)),
    ("writeln", "core", FormatString::At(1)),
    ("dbg", "std", FormatString::None),
    ("eprint", "std", FormatString::At(0)),
    ("eprintln", "std", FormatString::At(0)),
    ("format", "std", FormatString::At(0)),
    ("print", "std", FormatString::At(0)),
    ("println", "std", FormatString::At(0)),
    ("vec", "std", FormatString::None),
];

/// Tells where the format string is in the macro `path`: for the macros of the standard library
/// that format, the position; for the others of the library, none; for a macro of the user, any
/// literal may be.
fn format_string(path: &Path) -> FormatString {
    STD_MACROS
        .iter()
        .find(|(name, _, _)| is_std_macro(path, &[name]))
        .map_or(FormatString::Any, |&(_, _, format)| format)
}

/// The names that a format string captures from the scope: `{name}`, `{name:?}`, and the `width`
/// of `{:width$}`. Only ASCII names are read.
fn captured_names(format: &str) -> Vec<String> {
    let chars: Vec<char> = format.chars().collect();
    let mut names = Vec::new();
    let mut at = 0;
    while at < chars.len() {
        if chars[at] != '{' {
            at += 1;
        } else if chars.get(at + 1) == Some(&'{') {
            at += 2;
        } else {
            let end = match chars[at + 1..].iter().position(|&c| c == '}') {
                Some(n) => at + 1 + n,
                None => break,
            };
            let inner: String = chars[at + 1..end].iter().collect();
            let (arg, spec) = match inner.find(':') {
                Some(n) => (&inner[..n], &inner[n + 1..]),
                None => (&inner[..], ""),
            };
            names.push(arg.trim().to_string());
            let spec: Vec<char> = spec.chars().collect();
            for n in 0..spec.len() {
                if spec[n] != '$' {
                    continue;
                }
                let start = spec[..n]
                    .iter()
                    .rposition(|&c| !(c.is_ascii_alphanumeric() || c == '_'))
                    .map_or(0, |p| p + 1);
                names.push(spec[start..n].iter().collect());
            }
            at = end + 1;
        }
    }
    names.retain(|name| {
        name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
            && name.chars().next().map_or(false, |c| !c.is_ascii_digit())
    });
    names
}

/// What to write for the relative macro `path`: the absolute path of a macro of the standard
/// library, else how to write it.
fn macro_help(path: &Path) -> String {
    let name = tidy(&quote!(#path));
    if path.leading_colon.is_none() && path.segments.len() == 1 {
        if let Some((_, crate_name, _)) = STD_MACROS.iter().find(|(known, _, _)| *known == name) {
            return format!("use `::{}::{}!` instead of `{}!`", crate_name, name, name);
        }
    }
    format!(
        "write `{0}!` with an absolute path (e.g. `::my_crate::{0}!`)",
        name
    )
}

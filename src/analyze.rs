//! The analysis of items and types by the [`Leaker`].

use crate::analysis::{
    has_ambiguous, has_optional, names_self, primitive_effects, scope_error, CheckResult, Effect,
    PendingType,
};
use crate::context::{
    binder_lifetimes, Context, ContextId, Contexts, Enclosing, ModuleInfo, Scope, SelfRequirement,
    UnnameableSelf,
};
use crate::encode::is_const_param_ty;
use crate::error::{NotInternableError, SELF_NOTE};
#[cfg(doc)]
use crate::leaker::Leaker;
use crate::util::{
    first_self_in_generics, first_self_in_type, generic_param_name, has_lifetime_bound, ident_name,
    is_primitive_name, selves_in_generics, token_names, trait_object_of, unwrap_type_group,
    without_defaults, AllowedPaths,
};
use crate::view::View;
use proc_macro2::TokenStream;
use std::collections::{HashMap, HashSet};
use std::fmt::Display;
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use syn::visit::Visit;
use syn::*;
use template_quote::quote;

struct Frame {
    must_start: usize,
    impossible_start: usize,
    min_depth: usize,
    local_scope: usize,
    mentions_root: bool,
    mentions_self: bool,
    module_item: bool,
    soft_depth: usize,
    /// A cause is a relative path, so the type means something else in another module.
    relative: bool,
    /// A type not leaked alone because of a binder lifetime: its holder's depth and the lifetime
    /// (see [`Binder`]).
    binder: Option<(usize, Lifetime)>,
    unconstrained: Option<Lifetime>,
    effects: Vec<Effect>,
}

impl Frame {
    fn new(must_start: usize, impossible_start: usize) -> Self {
        Self {
            must_start,
            impossible_start,
            min_depth: usize::MAX,
            local_scope: usize::MAX,
            mentions_root: false,
            mentions_self: false,
            module_item: false,
            soft_depth: usize::MAX,
            relative: false,
            binder: None,
            unconstrained: None,
            effects: Vec::new(),
        }
    }
}

/// A `for<..>` binder whose output mentions lifetimes that no input keeps as written (an anchor). A
/// type using them cannot be leaked alone: its copy holds them only in an associated type path
/// (E0581, E0582). See [Binder lifetimes](crate#binder-lifetimes).
struct Binder {
    /// The depth of the type holding the binder, leaked whole instead (0 in a bound).
    owner: usize,
    output: Vec<Lifetime>,
}

struct OutputUse {
    lifetime: Lifetime,
    /// Each constraining occurrence in the inputs, as the types from the input down to it.
    anchors: Vec<Vec<usize>>,
    chain: Vec<usize>,
}

struct BinderGroup<'a> {
    output: Vec<Lifetime>,
    inputs: Vec<BinderInput<'a>>,
}

enum BinderInput<'a> {
    Type(&'a Type),
    Lifetime(&'a Lifetime),
}

struct ModuleFrame {
    id: ContextId,
    names: ModuleNames,
}

pub(crate) enum ScopeSpec {
    /// A module: `intern_mod` (with its path) or `enter_mod` (with its hint). It has no generics.
    Module {
        names: ModuleNames,
        path: Option<Vec<Ident>>,
        hint: Option<String>,
    },
    Generics {
        generics: Generics,
        /// What `Self` is inside, if the scope declares it. `None` keeps the `Self` around.
        declares: Option<DeclaredSelf>,
        /// The trait of an impl, checked on entering the scope like the root of a trait bound: it
        /// must not depend on the context, but its arguments are analyzed by the walk of the impl.
        trait_path: Option<Box<Path>>,
    },
}

pub(crate) struct DeclaredSelf {
    requirement: SelfRequirement,
    allow_self: bool,
}

impl ScopeSpec {
    pub(crate) fn declared_generics(&self) -> Generics {
        match self {
            Self::Module { .. } => Generics::default(),
            Self::Generics { generics, .. } => generics.clone(),
        }
    }

    /// What `Self` is in the scope, if it declares it.
    pub(crate) fn declared_self(&self) -> Option<SelfRequirement> {
        match self {
            Self::Generics {
                declares: Some(declared),
                ..
            } => Some(declared.requirement.clone()),
            _ => None,
        }
    }
}

pub(crate) type Res<T> = std::result::Result<T, NotInternableError>;

/// Where a scope is entered: the Leaker (outermost) or a [`View`] (nested). The public `enter_*`
/// methods build the [`ScopeSpec`] here and differ only in `enter_spec()`.
pub(crate) trait Scopes {
    fn enter_spec<R>(
        &mut self,
        spec: ScopeSpec,
        f: impl FnOnce(&mut View<'_>) -> R,
    ) -> Res<(ContextId, R)>;

    fn enter_mod<R>(
        &mut self,
        hint: impl Display,
        declared: impl IntoIterator<Item = Ident>,
        f: impl FnOnce(&mut View<'_>) -> R,
    ) -> Res<(ContextId, R)> {
        let names = ModuleNames {
            names: declared
                .into_iter()
                .map(|ident| ident_name(&ident))
                .collect(),
            modules: HashMap::new(),
        };
        let hint = Some(hint.to_string());
        self.enter_spec(
            ScopeSpec::Module {
                names,
                path: None,
                hint,
            },
            f,
        )
    }

    fn enter_generics<R>(
        &mut self,
        generics: &Generics,
        f: impl FnOnce(&mut View<'_>) -> R,
    ) -> Res<(ContextId, R)> {
        let generics = generics.clone();
        self.enter_spec(
            ScopeSpec::Generics {
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
        requirement: SelfRequirement,
        allow_self: bool,
        trait_path: Option<&Path>,
        f: impl FnOnce(&mut View<'_>) -> R,
    ) -> Res<(ContextId, R)> {
        let declares = Some(DeclaredSelf {
            requirement,
            allow_self,
        });
        let spec = ScopeSpec::Generics {
            generics: generics.clone(),
            declares,
            trait_path: trait_path.map(|path| Box::new(path.clone())),
        };
        self.enter_spec(spec, f)
    }

    fn enter_type<R>(
        &mut self,
        ident: &Ident,
        generics: &Generics,
        allow_self: bool,
        f: impl FnOnce(&mut View<'_>) -> R,
    ) -> Res<(ContextId, R)> {
        let requirement = SelfRequirement::of_item(ident, generics);
        self.enter_declaring(generics, requirement, allow_self, None, f)
    }

    fn enter_impl<R>(
        &mut self,
        generics: &Generics,
        self_ty: &Type,
        allow_self: bool,
        f: impl FnOnce(&mut View<'_>) -> R,
    ) -> Res<(ContextId, R)> {
        self.enter_impl_of(generics, self_ty, None, allow_self, f)
    }

    /// Like `enter_impl()` for an impl of the trait `trait_path`, which must not depend on the
    /// context.
    fn enter_impl_of<R>(
        &mut self,
        generics: &Generics,
        self_ty: &Type,
        trait_path: Option<&Path>,
        allow_self: bool,
        f: impl FnOnce(&mut View<'_>) -> R,
    ) -> Res<(ContextId, R)> {
        let requirement = SelfRequirement::Type(self_ty.clone());
        self.enter_declaring(generics, requirement, allow_self, trait_path, f)
    }

    fn enter_trait<R>(
        &mut self,
        ident: &Ident,
        generics: &Generics,
        allow_self: bool,
        f: impl FnOnce(&mut View<'_>) -> R,
    ) -> Res<(ContextId, R)> {
        let requirement = SelfRequirement::of_trait(ident, generics);
        self.enter_declaring(generics, requirement, allow_self, None, f)
    }
}

struct ScopeMark {
    effects: usize,
    self_errors: usize,
    contexts: usize,
    top_levels: usize,
    scopes: usize,
    modules: usize,
    item_depth: usize,
    self_ty: Option<SelfRequirement>,
    allow_self: bool,
    module: Option<ContextId>,
}

/// What [`AnalyzeVisitor::check()`] puts back as it found it. The visit replaces the frame of the
/// scope, and appends to the causes and, for an expression with an item, to the contexts. `walk`
/// is the state that only a visit in progress changes, which a panic would leave half-way.
struct Checkpoint {
    mark: ScopeMark,
    frame: Frame,
    musts: usize,
    impossibles: usize,
    walk: Walk,
}

/// The state of the walk of a type, which is back to what it was when the walk ends.
struct Walk {
    type_depth: usize,
    binders: usize,
    signatures: usize,
    probe: Option<HashSet<usize>>,
    in_default: Option<bool>,
    expr_owner: Option<usize>,
    in_receiver: bool,
    enclosing: Enclosing,
    elided_ref: bool,
    unconstrained_ref: Option<Lifetime>,
    ambiguous_arg: bool,
    inherits: Option<Inherits>,
}

pub(crate) struct Finished {
    pub(crate) effects: Vec<Effect>,
    pub(crate) top_levels: Vec<ContextId>,
    pub(crate) module: Option<ContextId>,
}

pub(crate) struct AnalyzeVisitor {
    allowed: AllowedPaths,
    /// The Leaker's table while it analyzes (see [`AnalyzeVisitor::swap_contexts()`]), an empty one
    /// in a throwaway visitor.
    contexts: Contexts,
    /// Mentions of `Self` where it has no meaning (see [`NotInternableError::SelfWithoutMeaning`]).
    /// Kept apart from the effects, because a type leaked whole drops those of its parts.
    self_errors: Vec<TokenStream>,
    top_levels: Vec<ContextId>,
    allow_self: bool,
    /// The generics in scope, innermost last. The first are the Leaker's.
    scopes: Vec<Scope>,
    binders: Vec<Binder>,
    /// While a binder is probed: the addresses of the types found to be leaked whole.
    probe: Option<HashSet<usize>>,
    root_defaulted: Vec<Ident>,
    /// What `Self` means here, if anything. It has none directly in a module, or in generics that
    /// no item declaring it encloses, so mentioning it there is an error.
    self_ty: Option<SelfRequirement>,
    root_declares_self: bool,
    /// The type is the default of a parameter, which cannot mention `Self`. It holds whether the
    /// declaring scope gives `Self` a meaning. `self_ty` is `None` meanwhile.
    in_default: Option<bool>,
    modules: Vec<ModuleFrame>,
    musts: Vec<(usize, TokenStream)>,
    impossibles: Vec<TokenStream>,
    frame: Frame,
    type_depth: usize,
    item_depth: usize,
    /// A referent or path argument that cannot be leaked whole (elided or several local lifetimes,
    /// or in a bound). A trait object there cannot be leaked alone.
    elided_ref: bool,
    /// Like `elided_ref`, for a method lifetime that the copy would leave unconstrained (see
    /// `signatures`). The conflict has a note of its own.
    unconstrained_ref: Option<Lifetime>,
    ambiguous_arg: bool,
    /// A referent or path argument whose lifetime is a generic. A trait object there defaults to
    /// it, and would default to `'static` if leaked alone. What is inside it is not affected.
    inherits: Option<Inherits>,
    /// The signatures being visited, innermost last: the index in `scopes` of their generics, and
    /// their lifetimes that the copy would leave unconstrained (E0195).
    signatures: Vec<(usize, Vec<Lifetime>)>,
    /// The depth of the type owning the expression (array length, const argument) being visited.
    /// Types in it are not leaked alone: the copy would put an associated type path in a constant.
    expr_owner: Option<usize>,
    /// In a receiver (`self: Box<Self>`), where rustc accepts only `Self` and types dereferencing
    /// to it: no type in it is optional.
    in_receiver: bool,
    /// The lifetimes the type being visited must outlive because of the types around it (see
    /// [`Enclosing`]). The Referrer tracks the same ones to find the entry. Empty outside a type.
    enclosing: Enclosing,
}

/// What a trait object without a lifetime bound takes its lifetime from. See
/// [Trait objects](crate#trait-objects).
#[derive(Clone, Copy)]
enum Inherits {
    /// A reference `&'a dyn Trait` (the depth of the reference) with a Leaker lifetime, or a local
    /// one that the copy leaves constrained. The default is `'a` unless the trait has a bound of
    /// its own (`+ 'a` is E0478), which the Leaker cannot see for a relative trait. So it is part
    /// of the reference.
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

impl AnalyzeVisitor {
    pub(crate) fn new(allowed: AllowedPaths, generics: &Generics) -> Self {
        Self {
            allowed,
            contexts: Contexts::default(),
            self_errors: Vec::new(),
            top_levels: Vec::new(),
            allow_self: true,
            scopes: vec![Scope::new(generics, None)],
            binders: Vec::new(),
            probe: None,
            root_defaulted: defaulted(generics)
                .map(|(_, ident, _)| ident.clone())
                .collect(),
            self_ty: None,
            root_declares_self: false,
            in_default: None,
            modules: Vec::new(),
            musts: Vec::new(),
            impossibles: Vec::new(),
            frame: Frame::new(0, 0),
            type_depth: 0,
            item_depth: 0,
            elided_ref: false,
            unconstrained_ref: None,
            ambiguous_arg: false,
            inherits: None,
            signatures: Vec::new(),
            expr_owner: None,
            in_receiver: false,
            enclosing: Enclosing::default(),
        }
    }

    /// Exchanges the table of contexts with `other`, which cannot fail. The outermost scope of the
    /// Leaker swaps its table in before the analysis and out after it, whatever the outcome.
    pub(crate) fn swap_contexts(&mut self, other: &mut Contexts) {
        std::mem::swap(&mut self.contexts, other);
    }

    fn alone<R>(
        allowed: AllowedPaths,
        generics: &Generics,
        f: impl FnOnce(&mut AnalyzeVisitor) -> R,
    ) -> R {
        f(&mut AnalyzeVisitor::new(allowed, generics))
    }

    /// Analyzes the default `ty` of a type parameter, with the parameters before it as the root
    /// generics. Returns the effects and the mentions of `Self` that have no meaning.
    fn analyze_default(
        allowed: AllowedPaths,
        ty: &Type,
        generics: &Generics,
        declares_self: bool,
    ) -> (Vec<Effect>, Vec<TokenStream>) {
        Self::alone(allowed, generics, |visitor| {
            visitor.in_default = Some(declares_self);
            visitor.visit_type(ty);
            (
                std::mem::take(&mut visitor.frame.effects),
                std::mem::take(&mut visitor.self_errors),
            )
        })
    }

    fn causes(&self, must_start: usize, impossible_start: usize, depth: usize) -> CheckResult {
        let mut must: Vec<&(usize, TokenStream)> = self.musts[must_start..].iter().collect();
        must.sort_by_key(|(d, _)| *d);
        CheckResult {
            is_own: must.first().map_or(false, |(d, _)| *d == depth),
            must: must.into_iter().map(|(_, m)| m.clone()).collect(),
            impossible: self.impossibles[impossible_start..].to_vec(),
            optional: false,
            note: None,
            headline: None,
        }
    }

    /// Visits `ty` and returns what it found at its own depth: the causes, whether it is an
    /// optional entry, and the conflict if it is one by itself (see
    /// [`AnalyzeVisitor::own_context()`]). What the visit records stays: a throwaway visitor does
    /// not care, [`AnalyzeVisitor::check()`] rolls it back.
    fn found(&mut self, ty: &Type) -> CheckResult {
        let (must_start, impossible_start) = (self.musts.len(), self.impossibles.len());
        let effects_start = self.frame.effects.len();
        let depth = self.type_depth + 1;
        self.visit_type(ty);
        let mut result = self.causes(must_start, impossible_start, depth);
        match &self.frame.effects[effects_start..] {
            // The type is decided at its own depth, so the only effect of a type with a cause of
            // its own is its own: the conflict is what a scope reports (with the note).
            [Effect::Error(conflict)] if result.is_own => result = conflict.clone(),
            [Effect::Type(pending)] if pending.optional && pending.ty == *ty => {
                result.optional = true
            }
            _ => (),
        }
        result
    }

    fn new_frame(&mut self) -> Frame {
        let frame = Frame::new(self.musts.len(), self.impossibles.len());
        std::mem::replace(&mut self.frame, frame)
    }

    fn push_must(&mut self, must: TokenStream) {
        self.musts.push((self.type_depth, must));
        self.frame.min_depth = self.frame.min_depth.min(self.type_depth);
    }

    fn mention(&mut self, scope: usize) {
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

    /// Returns the context of `ty`, or why it cannot be written (see [`Context::of()`]).
    fn context_of(&self, ty: &Type, inner: &Frame) -> std::result::Result<Context, UnnameableSelf> {
        let context = if inner.mentions_root
            || inner.mentions_self
            || inner.local_scope < self.scopes.len()
        {
            Context::of(ty, &self.scopes, self.self_ty.as_ref(), &self.enclosing)?
        } else {
            Context::none()
        };
        Ok(match self.modules.last() {
            Some(module) if inner.relative || self.depends_on_module(&context) => {
                context.with_module(Some(module.id))
            }
            _ => context,
        })
    }

    fn depends_on_module(&self, context: &Context) -> bool {
        // The tokens of a macro are resolved where the macro is expanded.
        let mut finder = AnyPath::new(true, |path| {
            path.leading_colon.is_none()
                && matches!(path.segments.first(), Some(seg)
                    if seg.ident != "crate" && seg.ident != "$crate" && seg.ident != "Self")
                && !self.starts_with_generic(path)
                && !self.allowed.is_context_free_path(path)
        });
        finder.visit_generics(context.generics());
        match context.self_ty() {
            Some(SelfRequirement::Bound(path)) => finder.visit_path(path),
            Some(SelfRequirement::Type(ty)) => finder.visit_type(ty),
            _ => (),
        }
        finder.found
    }

    /// Returns the first mention of `Self` in the generics of `context`, if `Self` may not be
    /// leaked. A written `Self: 'a` counts: it is encoded as `&'a Self`, which needs `Self` too.
    fn self_in_bounds(&self, context: &Context) -> Option<TokenStream> {
        if self.allow_self {
            return None;
        }
        first_self_in_generics(context.generics())
            .or_else(|| (!context.self_outlives().is_empty()).then(|| quote!(Self)))
    }

    fn output_binder(&self, context: &Context) -> Option<(Lifetime, usize)> {
        self.binders.iter().find_map(|binder| {
            binder
                .output
                .iter()
                .find(|lifetime| {
                    context.generics().params.iter().any(|param| {
                        matches!(param, GenericParam::Lifetime(param) if &param.lifetime == *lifetime)
                    })
                })
                .map(|lifetime| (lifetime.clone(), binder.owner))
        })
    }

    /// Finishes the outermost scope. Defaults of type parameters are analyzed alone, with the
    /// parameters before them, and may not leak `Self`. On an error the caller rolls back the
    /// contexts recorded since the start.
    pub(crate) fn finish(&mut self, generics: &Generics) -> Res<Finished> {
        let mut effects = std::mem::take(&mut self.frame.effects);
        for (index, _, default) in defaulted(generics) {
            let root = without_defaults(generics, index);
            let (found, self_errors) = AnalyzeVisitor::analyze_default(
                self.allowed.clone(),
                default,
                &root,
                self.root_declares_self,
            );
            effects.extend(found);
            self.self_errors.extend(self_errors);
        }
        if let Some(error) = scope_error(&effects, &self.self_errors) {
            return Err(error);
        }
        Ok(Finished {
            effects,
            top_levels: std::mem::take(&mut self.top_levels),
            module: self.modules.first().map(|module| module.id),
        })
    }

    /// Starts the outermost scope, entering it if it is a module.
    pub(crate) fn start(&mut self, spec: &ScopeSpec) {
        self.root_declares_self = matches!(
            spec,
            ScopeSpec::Generics {
                declares: Some(_),
                ..
            }
        );
        self.enter_scope(spec, true);
    }

    /// Marks what a scope or a check records, to roll it back with
    /// [`AnalyzeVisitor::leave_scope()`].
    fn mark(&self) -> ScopeMark {
        ScopeMark {
            effects: self.frame.effects.len(),
            self_errors: self.self_errors.len(),
            contexts: self.contexts.len(),
            top_levels: self.top_levels.len(),
            scopes: self.scopes.len(),
            modules: self.modules.len(),
            item_depth: self.item_depth,
            self_ty: self.self_ty.clone(),
            allow_self: self.allow_self,
            module: None,
        }
    }

    fn enter_scope(&mut self, spec: &ScopeSpec, root: bool) -> ScopeMark {
        let mut mark = self.mark();
        match spec {
            // `Self` has no meaning in a module, only in its items, which may leak it again.
            ScopeSpec::Module { names, path, hint } => {
                let parent = self.modules.last().map(|module| module.id);
                let id = self.contexts.intern_module(ModuleInfo {
                    path: path.clone(),
                    hint: hint.clone(),
                    parent,
                });
                self.modules.push(ModuleFrame {
                    id,
                    names: names.clone(),
                });
                self.self_ty = None;
                self.allow_self = true;
                mark.module = Some(id);
            }
            ScopeSpec::Generics {
                generics,
                declares,
                trait_path,
            } => {
                if let Some(declared) = declares {
                    self.self_ty = Some(declared.requirement.clone());
                    self.allow_self = declared.allow_self;
                }
                if root {
                    self.scopes[0].set_self(self.self_ty.clone());
                } else {
                    self.scopes.push(Scope::new(generics, self.self_ty.clone()));
                }
                if self.self_ty.is_none() {
                    let generics = without_defaults(generics, generics.params.len());
                    self.self_errors.extend(selves_in_generics(&generics));
                }
                // Checked like the root of a trait bound, before what the closure visits.
                if let Some(path) = trait_path.as_deref() {
                    self.check_bound_path(path, quote!(#path));
                }
            }
        }
        self.item_depth = self.item_depth.max(1);
        mark
    }

    fn leave_scope(&mut self, mark: ScopeMark, rollback: bool) {
        self.scopes.truncate(mark.scopes);
        self.modules.truncate(mark.modules);
        self.item_depth = mark.item_depth;
        self.self_ty = mark.self_ty;
        self.allow_self = mark.allow_self;
        if rollback {
            self.frame.effects.truncate(mark.effects);
            self.self_errors.truncate(mark.self_errors);
            self.contexts.truncate(mark.contexts);
            self.top_levels.truncate(mark.top_levels);
        }
    }

    /// Starts a check: sets the frame of the scope aside for the visit to use a new one.
    fn checkpoint(&mut self) -> Checkpoint {
        let mark = self.mark();
        let walk = Walk {
            type_depth: self.type_depth,
            binders: self.binders.len(),
            signatures: self.signatures.len(),
            probe: self.probe.clone(),
            in_default: self.in_default,
            expr_owner: self.expr_owner,
            in_receiver: self.in_receiver,
            enclosing: self.enclosing.clone(),
            elided_ref: self.elided_ref,
            unconstrained_ref: self.unconstrained_ref.clone(),
            ambiguous_arg: self.ambiguous_arg,
            inherits: self.inherits,
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
        self.leave_scope(mark, true);
        self.musts.truncate(musts);
        self.impossibles.truncate(impossibles);
        self.type_depth = walk.type_depth;
        self.binders.truncate(walk.binders);
        self.signatures.truncate(walk.signatures);
        self.probe = walk.probe;
        self.in_default = walk.in_default;
        self.expr_owner = walk.expr_owner;
        self.in_receiver = walk.in_receiver;
        self.enclosing = walk.enclosing;
        self.elided_ref = walk.elided_ref;
        self.unconstrained_ref = walk.unconstrained_ref;
        self.ambiguous_arg = walk.ambiguous_arg;
        self.inherits = walk.inherits;
    }

    /// Leaves a nested scope and returns its top-level context. It is all-or-nothing: on an error,
    /// everything recorded since it was entered is rolled back, and its errors are returned.
    fn end_scope(&mut self, mark: ScopeMark, spec: &ScopeSpec) -> Res<ContextId> {
        if let ScopeSpec::Generics { generics, .. } = spec {
            for (index, _, default) in defaulted(generics) {
                self.analyze_default_inline(default, &without_defaults(generics, index));
            }
        }
        let error = scope_error(
            &self.frame.effects[mark.effects..],
            &self.self_errors[mark.self_errors..],
        );
        let module = mark.module;
        // A scope that declares nothing keeps the `Self` around, so take it before leaving. It is
        // part of the top-level context only if an entry names it, or has a bound rewritten for
        // it: nothing else depends on it.
        let scope_self = self
            .self_ty
            .clone()
            .filter(|declared| names_self(&self.frame.effects[mark.effects..], declared));
        self.leave_scope(mark, error.is_some());
        if let Some(error) = error {
            return Err(error);
        }
        let top_level = module.unwrap_or_else(|| {
            // The scope is in the module around it, if any.
            let around = self.modules.last().map(|module| module.id);
            let context = Context::new(spec.declared_generics(), scope_self);
            self.contexts.intern(context.with_module(around))
        });
        self.top_levels.push(top_level);
        Ok(top_level)
    }

    fn analyze_default_inline(&mut self, ty: &Type, before: &Generics) {
        // The root scope is always there.
        let at = self.scopes.len() - 1;
        let scope = std::mem::replace(&mut self.scopes[at], Scope::new(before, None));
        let self_ty = self.self_ty.take();
        let in_default = self.in_default.replace(self_ty.is_some());
        self.visit_type(ty);
        self.in_default = in_default;
        self.self_ty = self_ty;
        self.scopes[at] = scope;
    }

    /// Records `tokens` as an error if `Self` has no meaning here, also in the default of a
    /// parameter of a scope that declares nothing.
    fn check_self_has_meaning(&mut self, tokens: TokenStream) {
        let declared = self.in_default.unwrap_or_else(|| self.self_ty.is_some());
        if !declared {
            self.self_errors.push(tokens);
        }
    }

    /// Like `decide()` for a type that cannot be leaked alone (a trait object inheriting a path
    /// lifetime, a type in an array length): its causes become those of `owner`, the enclosing type
    /// leaked whole.
    fn decide_as_owner(&mut self, owner: usize, mut inner: Frame) {
        if inner.min_depth != usize::MAX {
            self.own_causes(inner.must_start, owner);
            inner.min_depth = owner;
        }
        if inner.soft_depth != usize::MAX {
            inner.soft_depth = owner;
        }
        self.decide_through(inner)
    }

    /// Like `decide()` for a type that cannot be leaked alone and has no cause: only its parts are
    /// leaked.
    fn decide_through(&mut self, mut inner: Frame) {
        self.absorb(&inner);
        self.frame.effects.append(&mut inner.effects);
    }

    fn absorb(&mut self, inner: &Frame) {
        self.frame.min_depth = self.frame.min_depth.min(inner.min_depth);
        self.frame.soft_depth = self.frame.soft_depth.min(inner.soft_depth);
        self.frame.local_scope = self.frame.local_scope.min(inner.local_scope);
        self.frame.mentions_root |= inner.mentions_root;
        self.frame.mentions_self |= inner.mentions_self;
        self.frame.module_item |= inner.module_item;
        self.frame.relative |= inner.relative;
        // Kept for the enclosing type, unless that is the type holding the binder.
        if self.frame.binder.is_none() {
            let depth = self.type_depth;
            self.frame.binder = inner.binder.clone().filter(|(owner, _)| *owner < depth);
        }
    }

    fn own_causes(&mut self, must_start: usize, depth: usize) {
        for must in &mut self.musts[must_start..] {
            must.0 = depth;
        }
    }

    /// Returns the context of `ty` at `depth` if it can be leaked whole for its own path, else the
    /// conflict (its path should be absolute).
    fn own_context(
        &self,
        ty: &Type,
        depth: usize,
        inner: &Frame,
    ) -> std::result::Result<Context, CheckResult> {
        if self.impossibles.len() > inner.impossible_start {
            let mut conflict = self.causes(inner.must_start, inner.impossible_start, depth);
            conflict.note = inner
                .binder
                .as_ref()
                .filter(|(owner, _)| *owner == depth)
                .map(|(_, lifetime)| binder_note(lifetime, true))
                .or_else(|| inner.unconstrained.as_ref().map(unconstrained_note));
            return Err(conflict);
        }
        if self.in_default == Some(true) && inner.mentions_self {
            // The default is also used to compute the arguments of the trait, so the Repeater impl
            // cannot know what `Self` is.
            return Err(CheckResult {
                must: vec![first_self_in_type(ty).unwrap_or_else(|| quote!(Self))],
                impossible: vec![quote!(#ty)],
                is_own: true,
                optional: false,
                note: Some("a default of a parameter cannot depend on `Self`".to_string()),
                headline: None,
            });
        }
        let context = match self.context_of(ty, inner) {
            Ok(context) => context,
            Err(unnameable) => {
                return Err(CheckResult {
                    must: vec![quote!(#ty)],
                    impossible: vec![unnameable.cause],
                    is_own: true,
                    optional: false,
                    note: Some(unnameable.note),
                    headline: Some(unnameable.headline),
                })
            }
        };
        match self.self_in_bounds(&context) {
            Some(self_path) => Err(CheckResult {
                must: vec![quote!(#ty)],
                impossible: vec![self_path],
                is_own: true,
                optional: false,
                note: Some(
                    "the bounds of its generics mention `Self`, which cannot be leaked".to_string(),
                ),
                headline: None,
            }),
            None => Ok(context),
        }
    }

    /// Returns `true` if a bare identifier argument in `ty` may be a const, so `ty` is leaked
    /// whole. Not so for a type referring to an item of the module, a group or parentheses.
    fn is_ambiguous_owner(&self, ty: &Type, depth: usize, inner: &Frame) -> bool {
        !inner.module_item
            && !matches!(ty, Type::Group(_) | Type::Paren(_))
            && has_ambiguous(&inner.effects)
            && self.own_context(ty, depth, inner).is_ok()
    }

    fn decide(&mut self, ty: &Type, depth: usize, ambiguous: bool, mut inner: Frame) {
        if inner.min_depth != depth && self.is_ambiguous_owner(ty, depth, &inner) {
            self.own_causes(inner.must_start, depth);
            inner.min_depth = depth;
        }
        // A type leaked whole for its own path cannot use a binder lifetime of the output: the type
        // holding the binder is leaked instead, or it is a conflict (in a bound).
        let mut own = (inner.min_depth == depth).then(|| self.own_context(ty, depth, &inner));
        if let (Some(_), Some(probe)) = (&own, &mut self.probe) {
            probe.insert(ty as *const Type as usize);
        }
        let binder = match &own {
            Some(Ok(context)) => self.output_binder(context),
            _ => None,
        };
        if let Some((lifetime, owner)) = binder {
            if owner > 0 {
                inner.binder = Some((owner, lifetime));
                return self.decide_as_owner(owner, inner);
            }
            let mut conflict = self.causes(inner.must_start, inner.impossible_start, depth);
            conflict.impossible.insert(0, quote!(#lifetime));
            conflict.note = Some(binder_note(&lifetime, false));
            own = Some(Err(conflict));
        }
        // The same for an external type with such a lifetime: the type holding the binder is the
        // optional entry instead. Otherwise it is optional if it has no own cause and can be leaked
        // whole.
        let soft = if own.is_none()
            && inner.soft_depth == depth
            && !inner.module_item
            && !self.in_receiver
        {
            self.own_context(ty, depth, &inner).ok()
        } else {
            None
        };
        let soft_binder = soft
            .as_ref()
            .and_then(|context| self.output_binder(context));
        if let Some((_, owner)) = soft_binder {
            if owner > 0 {
                inner.soft_depth = owner;
                return self.decide_through(inner);
            }
        }
        let optional = soft.filter(|_| soft_binder.is_none());
        self.absorb(&inner);
        let has_impossible = self.impossibles.len() > inner.impossible_start;
        let has_must = inner.min_depth != usize::MAX;
        let mentions_local = inner.local_scope < self.scopes.len();
        let effects = std::mem::take(&mut inner.effects);

        let pending = match own {
            Some(Ok(context)) => PendingType {
                is_must_intern: true,
                in_graph: true,
                ambiguous: ambiguous && context.is_empty(),
                ..PendingType::new(ty, context, Vec::new())
            },
            Some(Err(conflict)) => return self.frame.effects.push(Effect::Error(conflict)),
            None if has_impossible => {
                if has_must || has_optional(&effects) {
                    // Not leaked whole (elided lifetime, `impl Trait`), but its parts can be.
                    self.frame.effects.extend(effects);
                } else {
                    // Cannot and need not be leaked: copied, so primitive names are assumed.
                    self.frame.effects.extend(primitive_effects(effects));
                }
                return;
            }
            // Not leaked itself: it depends on local generics, refers to an item of the module
            // (which the Referrer shadows in the copy) or to `Self` in a default. Its parts can be.
            None if mentions_local
                || inner.module_item
                || (self.in_default.is_some() && inner.mentions_self) =>
            {
                match optional {
                    Some(context) => PendingType {
                        optional: true,
                        has_optional: true,
                        ..PendingType::new(ty, context, effects)
                    },
                    None => return self.frame.effects.extend(effects),
                }
            }
            None => {
                // Not leaked itself if its generics cannot be written (`Self`).
                let context = match self.context_of(ty, &inner) {
                    Ok(context) if self.self_in_bounds(&context).is_none() => context,
                    _ => return self.frame.effects.extend(effects),
                };
                PendingType {
                    in_graph: has_must,
                    optional: optional.is_some(),
                    has_optional: optional.is_some() || has_optional(&effects),
                    ..PendingType::new(ty, context, effects)
                }
            }
        };
        self.frame.effects.push(Effect::Type(Box::new(pending)));
    }

    fn relative_path(&mut self, must: TokenStream, impossible: TokenStream) {
        self.frame.effects.push(Effect::Error(CheckResult {
            must: vec![must],
            impossible: vec![impossible],
            is_own: true,
            optional: false,
            note: None,
            headline: None,
        }));
    }

    /// Records an error if a trait path depends on the context or starts with `Self` without a
    /// meaning. Returns `true` if so.
    fn check_bound_path(&mut self, path: &Path, tokens: TokenStream) -> bool {
        if self_segment(path).is_some() && self.self_ty.is_none() {
            self.self_errors.push(tokens);
            true
        } else if !self.is_context_free(path) {
            self.relative_path(tokens.clone(), tokens);
            true
        } else {
            false
        }
    }

    fn is_context_free(&self, path: &Path) -> bool {
        self.allowed.is_context_free_path(path) || self.is_module_path(path)
    }

    fn is_relative(&self, path: &Path) -> bool {
        path.leading_colon.is_none()
            && !matches!(path.segments.first(), Some(seg) if seg.ident == "crate")
            && !self.is_context_free(path)
    }

    /// Returns `true` if `path` names an item of the analyzed module, as Rust resolves it.
    fn is_module_path(&self, path: &Path) -> bool {
        if self.modules.is_empty() || path.leading_colon.is_some() {
            return false;
        }
        let mut current = self.modules.len() - 1;
        let mut segments = path.segments.iter().peekable();
        let explicit = match segments.peek() {
            Some(seg) if seg.ident == "self" => {
                segments.next();
                true
            }
            Some(seg) if seg.ident == "super" => {
                while matches!(segments.peek(), Some(seg) if seg.ident == "super") {
                    segments.next();
                    match current.checked_sub(1) {
                        Some(parent) => current = parent,
                        None => return false,
                    }
                }
                true
            }
            _ => false,
        };
        let mut module = &self.modules[current].names;
        let mut first = true;
        loop {
            let seg = match segments.next() {
                Some(seg) => seg,
                None => return false,
            };
            let name = ident_name(&seg.ident);
            if !module.names.contains(&name) || (first && !explicit && self.is_generic(&seg.ident))
            {
                return false;
            }
            first = false;
            match module.modules.get(&name) {
                Some(inner) if segments.peek().is_some() => module = inner,
                _ => return true,
            }
        }
    }

    fn is_generic(&self, ident: &Ident) -> bool {
        self.scope_of(&ident_name(ident)).is_some()
    }

    fn with_module(&mut self, i: &ItemMod, f: impl FnOnce(&mut Self)) {
        let items = match &i.content {
            Some((_, items)) => items,
            None => return f(self),
        };
        let parent = self.modules.last().map(|module| module.id);
        let mut path = parent
            .and_then(|parent| self.contexts.module(parent))
            .and_then(|info| info.path.clone())
            .unwrap_or_default();
        path.push(i.ident.clone());
        let id = self.contexts.intern_module(ModuleInfo {
            path: Some(path),
            hint: None,
            parent,
        });
        self.modules.push(ModuleFrame {
            id,
            names: declared_names(items),
        });
        f(self);
        self.modules.pop();
    }

    fn params(&self) -> impl Iterator<Item = &GenericParam> {
        self.scopes.iter().flat_map(|scope| scope.params())
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

    fn with_scope(&mut self, scope: Scope, f: impl FnOnce(&mut Self)) {
        self.scopes.push(scope);
        f(self);
        self.scopes.pop();
    }

    fn with_local_generics(&mut self, generics: &Generics, f: impl FnOnce(&mut Self)) {
        self.with_scope(Scope::new(generics, self.self_ty.clone()), f)
    }

    /// Runs `f` with the generics of an item: the Leaker's if outermost, local otherwise.
    fn with_item_generics(&mut self, generics: &Generics, f: impl FnOnce(&mut Self)) {
        let nested = self.item_depth > 0;
        self.item_depth += 1;
        if nested {
            self.with_local_generics(generics, f);
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
        f: impl FnOnce(&mut Self),
    ) {
        let allow_self = self.allow_self;
        if self_ty.is_none() || self.item_depth > 0 {
            self.allow_self = true;
        }
        let outer = std::mem::replace(&mut self.self_ty, self_ty);
        self.with_item_generics(generics, f);
        self.self_ty = outer;
        self.allow_self = allow_self;
    }

    /// Visits the attributes and visibility only: the body or value is not analyzed.
    fn visit_head(&mut self, attrs: &[Attribute], vis: &Visibility) {
        for attr in attrs {
            self.visit_attribute(attr);
        }
        self.visit_visibility(vis);
    }

    fn with_signature(&mut self, sig: &Signature, item: bool, f: impl FnOnce(&mut Self)) {
        let local = !item || self.item_depth > 0;
        self.with_risky(sig, self.scopes.len(), local, |visitor| {
            if item {
                visitor.with_item_generics(&sig.generics, f)
            } else {
                visitor.with_local_generics(&sig.generics, f)
            }
        });
    }

    fn with_risky(&mut self, sig: &Signature, at: usize, local: bool, f: impl FnOnce(&mut Self)) {
        if local {
            let risky = self.unconstrained_lifetimes(sig);
            self.signatures.push((at, risky));
        }
        f(self);
        if local {
            self.signatures.pop();
        }
    }

    fn is_unconstrained(&self, lifetime: &Lifetime, scope: usize) -> bool {
        self.signatures
            .iter()
            .any(|(at, risky)| *at == scope && risky.contains(lifetime))
    }

    /// Returns the lifetimes of `sig` that the copy would leave unconstrained, making them
    /// early-bound where the original is late-bound (E0195, see [Method
    /// lifetimes](crate#method-lifetimes)). The return type and some input mention them, no bound
    /// does, and no input keeps them: the receiver or a reference with it.
    fn unconstrained_lifetimes(&self, sig: &Signature) -> Vec<Lifetime> {
        let declared: Vec<Lifetime> = sig
            .generics
            .lifetimes()
            .map(|param| param.lifetime.clone())
            .collect();
        if declared.is_empty() {
            return Vec::new();
        }
        let mut output = Lifetimes::default();
        output.visit_return_type(&sig.output);
        // A lifetime in a bound is early-bound: `'b: 'c` (both), `T: 'b` and `T: Tr<'b>`.
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
        let in_scope: Generics = Generics {
            params: self
                .params()
                .chain(sig.generics.params.iter())
                .cloned()
                .collect(),
            ..Default::default()
        };
        let not_a_cause = |object: &TypeTraitObject| self.is_not_a_cause(object, &in_scope);
        let mut mentioned = Lifetimes::default();
        let mut kept = Vec::new();
        for input in &sig.inputs {
            mentioned.visit_fn_arg(input);
            match input {
                FnArg::Receiver(receiver) => {
                    kept.extend(receiver.lifetime().cloned());
                    kept_lifetimes(&receiver.ty, &mut kept, &not_a_cause);
                }
                FnArg::Typed(arg) => kept_lifetimes(&arg.ty, &mut kept, &not_a_cause),
            }
        }
        declared
            .into_iter()
            .filter(|lifetime| {
                output.0.contains(lifetime)
                    && mentioned.0.contains(lifetime)
                    && !bounded.0.contains(lifetime)
                    && !kept.contains(lifetime)
            })
            .collect()
    }

    /// Returns `true` if `object` is certainly not a cause itself nor an optional entry, so a
    /// reference to it is never leaked whole because of it. When unsure (a relative path, an
    /// undeclared name) it is `false`.
    fn is_not_a_cause(&self, object: &TypeTraitObject, in_scope: &Generics) -> bool {
        let ty = Type::TraitObject(object.clone());
        let result = Self::alone(self.allowed.clone(), in_scope, |visitor| visitor.found(&ty));
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
                owner: self.type_depth,
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
    /// probed again.
    fn unanchored(
        &mut self,
        lifetimes: &BoundLifetimes,
        uses: &[OutputUse],
        f: &impl Fn(&mut Self),
    ) -> Vec<Lifetime> {
        let probing = self.probe.is_some();
        // An anchor with no type in the way (an argument of the bound itself) cannot be lost.
        let to_check = uses
            .iter()
            .any(|used| used.anchors.iter().any(|anchor| !anchor.is_empty()));
        let own = if !probing && to_check {
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
        let self_errors = self.self_errors.len();
        self.probe = Some(HashSet::new());
        self.with_scope(Scope::binder(lifetimes), |v| f(v));
        self.frame = frame;
        self.musts.truncate(musts);
        self.impossibles.truncate(impossibles);
        self.self_errors.truncate(self_errors);
        self.probe.take().unwrap_or_default()
    }

    fn binder_uses(&self, groups: Vec<BinderGroup>) -> Vec<OutputUse> {
        let mut uses = Vec::new();
        for group in groups {
            for lifetime in &group.output {
                let mut used = OutputUse {
                    lifetime: lifetime.clone(),
                    anchors: Vec::new(),
                    chain: Vec::new(),
                };
                if self.probe.is_none() {
                    for input in &group.inputs {
                        match input {
                            BinderInput::Type(ty) => self.scan_anchors(ty, &mut used),
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

    /// Finds the occurrences of `used.lifetime` in `ty` that constrain it as rustc sees it (all but
    /// in an associated type path, an array length, `impl Trait` or a type macro). One anchors only
    /// if no type on its chain is leaked whole. When in doubt it does not. See [Binder
    /// lifetimes](crate#binder-lifetimes).
    fn scan_anchors(&self, ty: &Type, used: &mut OutputUse) {
        used.chain.push(ty as *const Type as usize);
        match ty {
            // A reference to an external trait object is leaked whole if it is: the Referrer may
            // replace it.
            Type::Reference(reference) if self.is_external_object(&reference.elem) => {}
            Type::Reference(reference) => {
                if reference.lifetime.as_ref() == Some(&used.lifetime) {
                    used.anchors.push(used.chain.clone());
                }
                self.scan_anchors(&reference.elem, used);
            }
            Type::Ptr(TypePtr { elem, .. })
            | Type::Slice(TypeSlice { elem, .. })
            | Type::Paren(TypeParen { elem, .. })
            | Type::Group(TypeGroup { elem, .. }) => self.scan_anchors(elem, used),
            // An array whose length names an external crate may be replaced as a whole.
            Type::Array(array) if self.names_external_crate_in_expr(&array.len) => {}
            Type::Array(array) => self.scan_anchors(&array.elem, used),
            Type::Tuple(tuple) => {
                for elem in &tuple.elems {
                    self.scan_anchors(elem, used);
                }
            }
            Type::Path(TypePath { qself: None, path })
                if !self.is_projection(path)
                    && !self.allowed.is_soft_path(path)
                    && !self.holds_external_object(path) =>
            {
                if let Some(seg) = path.segments.last() {
                    self.scan_arguments(&seg.arguments, used);
                }
            }
            Type::BareFn(f) if !declares(f.lifetimes.as_ref(), &used.lifetime) => {
                for arg in &f.inputs {
                    self.scan_anchors(&arg.ty, used);
                }
                self.scan_output(&f.output, used);
            }
            Type::TraitObject(object) if !self.names_external_crate(object) => {
                for bound in &object.bounds {
                    match bound {
                        TypeParamBound::Trait(bound)
                            if !declares(bound.lifetimes.as_ref(), &used.lifetime) =>
                        {
                            if let Some(seg) = bound.path.segments.last() {
                                self.scan_arguments(&seg.arguments, used);
                            }
                        }
                        TypeParamBound::Lifetime(lifetime) if *lifetime == used.lifetime => {
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

    fn scan_arguments(&self, arguments: &PathArguments, used: &mut OutputUse) {
        match arguments {
            PathArguments::AngleBracketed(args) => {
                for arg in &args.args {
                    match arg {
                        GenericArgument::Lifetime(lifetime) if *lifetime == used.lifetime => {
                            used.anchors.push(used.chain.clone())
                        }
                        GenericArgument::Type(ty) => self.scan_anchors(ty, used),
                        GenericArgument::AssocType(binding) => self.scan_anchors(&binding.ty, used),
                        _ => (),
                    }
                }
            }
            PathArguments::Parenthesized(args) => {
                for ty in &args.inputs {
                    self.scan_anchors(ty, used);
                }
                self.scan_output(&args.output, used);
            }
            PathArguments::None => (),
        }
    }

    fn scan_output(&self, output: &ReturnType, used: &mut OutputUse) {
        if let ReturnType::Type(_, ty) = output {
            self.scan_anchors(ty, used);
        }
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
        let mut finder = AnyPath::new(false, |path| self.allowed.is_soft_path(path));
        finder.visit_expr(expr);
        finder.found
    }

    fn names_external_crate(&self, object: &TypeTraitObject) -> bool {
        object.bounds.iter().any(|bound| {
            matches!(bound, TypeParamBound::Trait(bound) if self.allowed.is_soft_path(&bound.path))
        })
    }

    /// Returns `true` if `path` is an associated item of a generic in scope or of `Self` (`T::A`).
    fn is_projection(&self, path: &Path) -> bool {
        path.leading_colon.is_none()
            && path.segments.len() > 1
            && matches!(path.segments.first(), Some(seg)
                if seg.ident == "Self" || self.is_generic(&seg.ident))
    }
}

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

/// What a [`View`] asks of the analysis it wraps. The view forwards the `visit_*` methods the
/// analysis overrides to it.
impl AnalyzeVisitor {
    /// Checks `ty` where the visitor is, as [`View::check()`] does: what the visit of `ty` finds,
    /// and nothing stays recorded.
    ///
    /// The visit runs on a new frame of the scope, and what it appends to the visitor is cut off
    /// after it (see [`Checkpoint`]), also if it panics, which goes on.
    pub(crate) fn check(&mut self, ty: &Type) -> CheckResult {
        let checkpoint = self.checkpoint();
        let selves = self.self_errors.len();
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            let mut result = self.found(ty);
            // `Self` without meaning fails the scope, which a check does not: it is told as a cause
            // that cannot be leaked.
            let without_meaning = &self.self_errors[selves..];
            if !without_meaning.is_empty() {
                result.impossible.extend(without_meaning.iter().cloned());
                result.optional = false;
                result.note.get_or_insert_with(|| SELF_NOTE.to_string());
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
        f: impl FnOnce(&mut View<'_>) -> R,
    ) -> Res<(ContextId, R)> {
        let mark = self.enter_scope(&spec, false);
        // The closure may panic and the scope around may catch it: leave the scope as if it failed.
        match catch_unwind(AssertUnwindSafe(|| f(&mut View::new(self)))) {
            Ok(result) => self.end_scope(mark, &spec).map(|context| (context, result)),
            Err(payload) => {
                self.leave_scope(mark, true);
                resume_unwind(payload)
            }
        }
    }
}

impl<'ast> Visit<'ast> for AnalyzeVisitor {
    fn visit_type_param(&mut self, i: &TypeParam) {
        let is_root_default = i.default.is_some()
            && self.item_depth <= 1
            && self.scopes.len() == 1
            && self.root_defaulted.contains(&i.ident);
        if is_root_default {
            let mut i = i.clone();
            i.eq_token = None;
            i.default = None;
            return syn::visit::visit_type_param(self, &i);
        }
        syn::visit::visit_type_param(self, i)
    }

    fn visit_trait_item_fn(&mut self, i: &TraitItemFn) {
        let mut i = i.clone();
        i.default = None;
        i.semi_token = Some(Default::default());
        syn::visit::visit_trait_item_fn(self, &i)
    }

    /// The signature of a method: its generics are local, and it is checked for E0195. A function
    /// item declares its own generics instead (`visit_item_fn()`).
    fn visit_signature(&mut self, i: &Signature) {
        self.with_signature(i, false, |visitor| syn::visit::visit_signature(visitor, i))
    }

    fn visit_trait_item_type(&mut self, i: &TraitItemType) {
        self.with_local_generics(&i.generics, |visitor| {
            syn::visit::visit_trait_item_type(visitor, i)
        });
    }

    fn visit_foreign_item_fn(&mut self, i: &ForeignItemFn) {
        // Not a method: its signature declares nothing here, as before `visit_signature()`.
        self.visit_head(&i.attrs, &i.vis);
        syn::visit::visit_signature(self, &i.sig);
    }

    fn visit_receiver(&mut self, i: &Receiver) {
        let outer = std::mem::replace(&mut self.in_receiver, true);
        syn::visit::visit_receiver(self, i);
        self.in_receiver = outer;
    }

    fn visit_trait_item_const(&mut self, i: &TraitItemConst) {
        let mut i = i.clone();
        i.default = None;
        self.with_local_generics(&i.generics, |visitor| {
            syn::visit::visit_trait_item_const(visitor, &i)
        });
    }

    fn visit_impl_item_fn(&mut self, i: &ImplItemFn) {
        self.visit_head(&i.attrs, &i.vis);
        self.visit_signature(&i.sig);
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
        });
    }

    fn visit_item_const(&mut self, i: &ItemConst) {
        self.visit_head(&i.attrs, &i.vis);
        self.visit_generics(&i.generics);
        self.visit_type(&i.ty);
    }

    fn visit_item_static(&mut self, i: &ItemStatic) {
        self.visit_head(&i.attrs, &i.vis);
        self.visit_type(&i.ty);
    }

    fn visit_item_struct(&mut self, i: &ItemStruct) {
        let self_ty = SelfRequirement::of_item(&i.ident, &i.generics);
        self.with_item(&i.generics, Some(self_ty), |visitor| {
            syn::visit::visit_item_struct(visitor, i)
        });
    }

    fn visit_item_enum(&mut self, i: &ItemEnum) {
        let self_ty = SelfRequirement::of_item(&i.ident, &i.generics);
        self.with_item(&i.generics, Some(self_ty), |visitor| {
            syn::visit::visit_item_enum(visitor, i)
        });
    }

    fn visit_item_union(&mut self, i: &ItemUnion) {
        let self_ty = SelfRequirement::of_item(&i.ident, &i.generics);
        self.with_item(&i.generics, Some(self_ty), |visitor| {
            syn::visit::visit_item_union(visitor, i)
        });
    }

    fn visit_item_trait(&mut self, i: &ItemTrait) {
        let self_ty = SelfRequirement::of_trait(&i.ident, &i.generics);
        self.with_item(&i.generics, Some(self_ty), |visitor| {
            syn::visit::visit_item_trait(visitor, i)
        });
    }

    fn visit_item_type(&mut self, i: &ItemType) {
        self.with_item_generics(&i.generics, |visitor| {
            syn::visit::visit_item_type(visitor, i)
        });
    }

    fn visit_item_fn(&mut self, i: &ItemFn) {
        self.with_signature(&i.sig, true, |visitor| {
            visitor.visit_head(&i.attrs, &i.vis);
            syn::visit::visit_signature(visitor, &i.sig);
        });
    }

    fn visit_item_mod(&mut self, i: &ItemMod) {
        // A module has no generics and no `Self`; its items are nested items with local generics.
        self.with_item(&Generics::default(), None, |visitor| {
            visitor.with_module(i, |visitor| syn::visit::visit_item_mod(visitor, i))
        });
    }

    fn visit_item_impl(&mut self, i: &ItemImpl) {
        let self_ty = SelfRequirement::Type((*i.self_ty).clone());
        self.with_item(&i.generics, Some(self_ty), |visitor| {
            // The trait is checked like a trait bound, but its arguments are analyzed in any case.
            if let Some((_, path, _)) = &i.trait_ {
                visitor.check_bound_path(path, quote!(#path));
            }
            syn::visit::visit_item_impl(visitor, i)
        });
    }

    fn visit_item_trait_alias(&mut self, i: &ItemTraitAlias) {
        self.with_item_generics(&i.generics, |visitor| {
            syn::visit::visit_item_trait_alias(visitor, i)
        });
    }

    fn visit_type(&mut self, i: &Type) {
        let outer = self.new_frame();
        let elided_ref = std::mem::take(&mut self.elided_ref);
        let unconstrained = self.unconstrained_ref.take();
        let ambiguous = std::mem::take(&mut self.ambiguous_arg);
        let inherits = self.inherits.take();
        self.type_depth += 1;
        let enclosing = self.enclosing.clone();
        self.enclosing = enclosing.components_of(i, self.scopes.len());
        match i {
            Type::Macro(m) => {
                // A type macro cannot be analyzed: it is leaked whole.
                self.push_must(quote!(#i));
                // It is expanded in the module where it is written, whatever its path is.
                self.frame.relative |= !self.modules.is_empty() || self.is_relative(&m.mac.path);
                // The generics (and `Self`) are matched by name.
                let mut names = HashSet::new();
                token_names(quote!(#i), &mut names);
                for name in &names {
                    if let Some(scope) = self.scope_of(name) {
                        self.mention(scope);
                    }
                }
                let mentions_self = names.contains("Self");
                self.frame.mentions_self |= mentions_self;
                if mentions_self {
                    self.check_self_has_meaning(quote!(#i));
                    if !self.allow_self {
                        self.impossibles.push(quote!(Self));
                    }
                }
            }
            Type::Paren(_) | Type::Group(_) => {
                self.elided_ref = elided_ref;
                self.unconstrained_ref = unconstrained;
                self.inherits = inherits;
                // `W<(N)>` is unambiguously a type, unlike `W<N>` (also in an invisible group).
                self.ambiguous_arg = ambiguous && matches!(i, Type::Group(_));
                syn::visit::visit_type(self, i)
            }
            Type::Reference(reference) => {
                match &reference.lifetime {
                    Some(lifetime) if lifetime.ident == "_" => self.elided_ref = true,
                    Some(lifetime) if lifetime.ident == "static" => (),
                    Some(lifetime) => {
                        let local = self
                            .scope_of(&lifetime.to_string())
                            .filter(|&scope| scope > 0);
                        let bounded =
                            trait_object_of(&reference.elem).map_or(false, has_lifetime_bound);
                        match local {
                            Some(_) if bounded => {}
                            Some(scope) if self.is_unconstrained(lifetime, scope) => {
                                self.unconstrained_ref = Some(lifetime.clone())
                            }
                            _ => self.inherits = Some(Inherits::Reference(self.type_depth)),
                        }
                    }
                    None => {
                        self.impossibles.push(quote!(#i));
                        self.elided_ref = true;
                    }
                }
                syn::visit::visit_type(self, i)
            }
            Type::TraitObject(object) if elided_ref && !has_lifetime_bound(object) => {
                self.impossibles.push(quote!(#i));
                syn::visit::visit_type(self, i)
            }
            Type::TraitObject(object) if unconstrained.is_some() && !has_lifetime_bound(object) => {
                if let Some(lifetime) = &unconstrained {
                    self.impossibles.push(quote!(#lifetime));
                }
                self.frame.unconstrained = unconstrained;
                syn::visit::visit_type(self, i)
            }
            Type::ImplTrait(_) | Type::Verbatim(_) => {
                self.impossibles.push(quote!(#i));
                syn::visit::visit_type(self, i);
            }
            _ => syn::visit::visit_type(self, i),
        }
        self.enclosing = enclosing;
        let mut inner = std::mem::replace(&mut self.frame, outer);
        // Also parentheses and groups. A written bound inherits nothing; the reference carries it.
        let inherits = inherits.filter(|inherits| {
            self.impossibles.len() == inner.impossible_start
                && matches!(unwrap_type_group(i), Type::TraitObject(object)
                    if matches!(inherits, Inherits::Reference(_)) || !has_lifetime_bound(object))
        });
        match (self.expr_owner, inherits) {
            (Some(owner), _) => self.decide_as_owner(owner, inner),
            // Not leaked alone, but its parts can be, unless the trait object must be: then the
            // reference or path is.
            (None, Some(inherits)) => {
                let owner = inherits.owner();
                let own = inner.min_depth == self.type_depth
                    || inner.min_depth == owner
                    || self.is_ambiguous_owner(i, self.type_depth, &inner);
                if own {
                    self.decide_as_owner(owner, inner)
                } else {
                    if inner.soft_depth == self.type_depth || inner.soft_depth == owner {
                        inner.soft_depth = owner;
                    }
                    self.decide_through(inner)
                }
            }
            (None, None) => self.decide(i, self.type_depth, ambiguous, inner),
        }
        self.type_depth -= 1;
    }

    fn visit_type_bare_fn(&mut self, i: &TypeBareFn) {
        let group = BinderGroup {
            output: output_lifetimes(i.lifetimes.as_ref(), |v| v.visit_return_type(&i.output)),
            inputs: i
                .inputs
                .iter()
                .map(|arg| BinderInput::Type(&arg.ty))
                .collect(),
        };
        self.with_bound_lifetimes(i.lifetimes.as_ref(), vec![group], |v| {
            syn::visit::visit_type_bare_fn(v, i)
        })
    }

    fn visit_type_trait_object(&mut self, i: &TypeTraitObject) {
        for bound in &i.bounds {
            match bound {
                TypeParamBound::Trait(bound) => self.visit_trait_bound(bound),
                TypeParamBound::Verbatim(_) => self.impossibles.push(quote!(#bound)),
                TypeParamBound::Lifetime(lifetime) => self.visit_lifetime(lifetime),
                _ => (),
            }
        }
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
        self.with_bound_lifetimes(i.lifetimes.as_ref(), groups, |v| {
            syn::visit::visit_predicate_type(v, i)
        })
    }

    fn visit_trait_bound(&mut self, i: &TraitBound) {
        // A trait bound with a relative path cannot be leaked.
        if self.type_depth == 0 && self.check_bound_path(&i.path, quote!(#i)) {
            return;
        }
        let group = bound_group(i.lifetimes.as_ref(), i, None);
        self.with_bound_lifetimes(i.lifetimes.as_ref(), vec![group], |v| {
            syn::visit::visit_trait_bound(v, i)
        })
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
            Type::Path(TypePath { qself: None, path }) if self.is_context_free(path) => {}
            _ => {
                self.relative_path(quote!(#{&i.ty}), quote!(#i));
                return;
            }
        }
        if let Some(expr) = &i.default {
            self.visit_expr(expr);
        }
    }

    fn visit_path(&mut self, i: &Path) {
        let in_type = self.type_depth > 0;
        let starts_with_generic = self.starts_with_generic(i);
        let free = self.allowed.is_context_free_path(i);
        let in_module = !starts_with_generic && !free && self.is_module_path(i);
        if in_type {
            if starts_with_generic {
                let first = i.segments.first();
                if let Some(scope) = first.and_then(|seg| self.scope_of(&ident_name(&seg.ident))) {
                    self.mention(scope);
                }
            }
            if self_segment(i).is_some() {
                self.frame.mentions_self = true;
                self.check_self_has_meaning(quote!(#i));
            }
            self.frame.module_item |= in_module;
        }
        if free && !starts_with_generic {
            // An allowed primitive name is copied as it is. It may be shadowed (`type u16 = i32;`),
            // so it is recorded for `Leaker::assertions()`.
            if i.leading_colon.is_none() {
                if let Some(seg) = i
                    .segments
                    .first()
                    .filter(|seg| is_primitive_name(&seg.ident))
                {
                    self.frame
                        .effects
                        .push(Effect::Primitive(seg.ident.clone()));
                }
            }
            // An external crate path is resolved by name, which the Referrer may not share.
            if in_type && self.allowed.is_soft_path(i) {
                self.frame.soft_depth = self.frame.soft_depth.min(self.type_depth);
            }
        } else if in_type && !starts_with_generic && !in_module {
            let is_self = matches!(self_segment(i), Some(seg) if seg.arguments.is_none());
            if !is_self {
                self.push_must(quote!(#i));
                self.frame.relative |= self.is_relative(i);
            } else if !self.allow_self {
                self.impossibles.push(quote!(#i));
            }
        }
        syn::visit::visit_path(self, i)
    }

    fn visit_type_path(&mut self, i: &TypePath) {
        match &i.qself {
            Some(qself) if self.resolves_through_bounds(qself) => {
                // `<T>::X` goes through the bounds of `T`: only its arguments depend on the
                // context.
                self.visit_qself(qself);
                for seg in &i.path.segments {
                    self.visit_path_arguments(&seg.arguments);
                }
            }
            _ => syn::visit::visit_type_path(self, i),
        }
    }

    fn visit_qself(&mut self, i: &QSelf) {
        // `<T>::Ty` cannot be leaked, unless `T` is a generic parameter (or `Self`).
        if self.type_depth > 0 && i.as_token.is_none() && !self.resolves_through_bounds(i) {
            let ty = &i.ty;
            self.impossibles.push(quote!(#ty));
        }
        syn::visit::visit_qself(self, i)
    }

    fn visit_lifetime(&mut self, i: &Lifetime) {
        if self.type_depth == 0 {
            return;
        }
        if i.ident == "_" {
            self.impossibles.push(quote!(#i));
        } else if let Some(scope) = self.scope_of(&i.to_string()) {
            self.mention(scope);
        } else if i.ident != "static" {
            self.push_must(quote!(#i));
        }
    }

    fn visit_expr(&mut self, i: &Expr) {
        if self.type_depth > 0 {
            if let Expr::Closure(_) | Expr::Assign(_) | Expr::Verbatim(_) | Expr::Macro(_) = i {
                self.impossibles.push(quote!(#i));
            }
        }
        let outer = self.expr_owner;
        if self.type_depth > 0 && outer.is_none() {
            self.expr_owner = Some(self.type_depth);
        }
        let enclosing = std::mem::take(&mut self.enclosing);
        syn::visit::visit_expr(self, i);
        self.enclosing = enclosing;
        self.expr_owner = outer;
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
        let elided = elided || (inherited && self.type_depth == 0);
        for arg in &i.args {
            if let GenericArgument::Type(ty) = arg {
                self.elided_ref = elided;
                self.unconstrained_ref = unconstrained_lifetime.clone().filter(|_| !elided);
                if inherited && !elided && unconstrained_lifetime.is_none() {
                    self.inherits = Some(Inherits::Path(self.type_depth));
                }
                // syn parses a bare identifier that may be a const as a type (see `ambiguous_arg`).
                self.ambiguous_arg = is_bare_ident(ty);
            }
            self.visit_generic_argument(arg);
        }
    }

    fn visit_expr_path(&mut self, i: &ExprPath) {
        if self.type_depth > 0 {
            return syn::visit::visit_expr_path(self, i);
        }
        // Values starting with a generic in scope or `Self` do not depend on the context.
        let is_self = i.qself.is_none()
            && matches!(self_segment(&i.path), Some(seg) if seg.arguments.is_none());
        if is_self {
            self.check_self_has_meaning(quote!(#i));
        }
        if (i.qself.is_none() && self.starts_with_generic(&i.path))
            || (is_self && self.allow_self)
            || self.is_context_free(&i.path)
        {
            syn::visit::visit_expr_path(self, i)
        } else {
            self.relative_path(quote!(#i), quote!(#i));
        }
    }
}

/// Pushes the lifetimes that `ty` (an input) keeps constrained as written in the copy. A reference
/// to a trait object without a lifetime bound counts only if `not_a_cause` says so.
fn kept_lifetimes(
    ty: &Type,
    kept: &mut Vec<Lifetime>,
    not_a_cause: &dyn Fn(&TypeTraitObject) -> bool,
) {
    match ty {
        Type::Reference(reference) => {
            let unbounded = trait_object_of(&reference.elem).map_or(false, |object| {
                !has_lifetime_bound(object) && !not_a_cause(object)
            });
            if !unbounded {
                kept.extend(reference.lifetime.clone());
            }
            kept_lifetimes(&reference.elem, kept, not_a_cause);
        }
        Type::Ptr(TypePtr { elem, .. })
        | Type::Slice(TypeSlice { elem, .. })
        | Type::Array(TypeArray { elem, .. })
        | Type::Paren(TypeParen { elem, .. })
        | Type::Group(TypeGroup { elem, .. }) => kept_lifetimes(elem, kept, not_a_cause),
        Type::Tuple(tuple) => {
            for elem in &tuple.elems {
                kept_lifetimes(elem, kept, not_a_cause);
            }
        }
        _ => (),
    }
}

struct AnyPath<F> {
    pred: F,
    macros: bool,
    found: bool,
}

impl<F: Fn(&Path) -> bool> AnyPath<F> {
    fn new(macros: bool, pred: F) -> Self {
        Self {
            pred,
            macros,
            found: false,
        }
    }
}

impl<'ast, F: Fn(&Path) -> bool> Visit<'ast> for AnyPath<F> {
    fn visit_path(&mut self, i: &'ast Path) {
        self.found |= (self.pred)(i);
        syn::visit::visit_path(self, i)
    }
    fn visit_macro(&mut self, i: &'ast Macro) {
        if self.macros {
            self.found = true;
        } else {
            syn::visit::visit_macro(self, i)
        }
    }
}

#[derive(Default)]
struct Lifetimes(Vec<Lifetime>);

impl<'ast> Visit<'ast> for Lifetimes {
    fn visit_lifetime(&mut self, i: &'ast Lifetime) {
        self.0.push(i.clone());
    }
    fn visit_macro(&mut self, i: &'ast Macro) {
        let mut names = HashSet::new();
        token_names(i.tokens.clone(), &mut names);
        self.0.extend(
            names
                .iter()
                .filter(|name| name.starts_with('\''))
                .map(|name| Lifetime::new(name, proc_macro2::Span::call_site())),
        );
    }
}

fn unconstrained_note(lifetime: &Lifetime) -> String {
    format!(
        "the lifetime of the trait object defaults to `{0}`, a lifetime of the method which its \
         return type mentions and no other argument keeps as written (e.g. `&{0} self`), so the \
         copy would have `{0}` only in a projection, and rustc rejects the implementation of a \
         method whose lifetime is early-bound where the trait's is late-bound (E0195)",
        lifetime
    )
}

/// `owned`: the type holding the binder would be leaked instead.
fn binder_note(lifetime: &Lifetime, owned: bool) -> String {
    format!(
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
    )
}

fn output_lifetimes(
    lifetimes: Option<&BoundLifetimes>,
    f: impl FnOnce(&mut OutputLifetimes),
) -> Vec<Lifetime> {
    let declared: Vec<Lifetime> = binder_lifetimes(lifetimes).collect();
    if declared.is_empty() {
        return Vec::new();
    }
    let mut visitor = OutputLifetimes {
        declared,
        shadowed: Vec::new(),
        found: Vec::new(),
    };
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
    let output = output_lifetimes(lifetimes, |v| v.visit_bound_output(bound));
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
            inputs.extend(args.inputs.iter().map(BinderInput::Type))
        }
        _ => (),
    }
    BinderGroup { output, inputs }
}

struct OutputLifetimes {
    declared: Vec<Lifetime>,
    shadowed: Vec<Lifetime>,
    found: Vec<Lifetime>,
}

impl OutputLifetimes {
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
        let mut names = HashSet::new();
        token_names(tokens, &mut names);
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

fn declares(lifetimes: Option<&BoundLifetimes>, wanted: &Lifetime) -> bool {
    binder_lifetimes(lifetimes).any(|lifetime| lifetime == *wanted)
}

fn self_segment(path: &Path) -> Option<&PathSegment> {
    path.segments
        .first()
        .filter(|seg| path.leading_colon.is_none() && seg.ident == "Self")
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

#[derive(Clone, Default)]
pub(crate) struct ModuleNames {
    names: HashSet<String>,
    modules: HashMap<String, ModuleNames>,
}

/// Returns the names declared directly in `items`: items, `macro_rules!` macros, and names imported
/// by `use` with an absolute path. Other imports (`use super::X`) are outside the module.
pub(crate) fn declared_names(items: &[Item]) -> ModuleNames {
    fn use_names(tree: &UseTree, last: Option<&Ident>, names: &mut HashSet<String>) {
        match tree {
            UseTree::Path(UsePath { ident, tree, .. }) => use_names(tree, Some(ident), names),
            UseTree::Name(UseName { ident }) if ident == "self" => {
                names.extend(last.map(ident_name));
            }
            UseTree::Name(UseName { ident }) => {
                names.insert(ident_name(ident));
            }
            UseTree::Rename(UseRename { rename, .. }) if rename != "_" => {
                names.insert(ident_name(rename));
            }
            UseTree::Group(UseGroup { items, .. }) => {
                for tree in items {
                    use_names(tree, last, names)
                }
            }
            _ => (),
        }
    }
    let mut module = ModuleNames::default();
    let names = &mut module.names;
    for item in items {
        match item {
            Item::Mod(ItemMod {
                ident,
                content: Some((_, items)),
                ..
            }) => {
                names.insert(ident_name(ident));
                module
                    .modules
                    .insert(ident_name(ident), declared_names(items));
            }
            Item::Struct(ItemStruct { ident, .. })
            | Item::Enum(ItemEnum { ident, .. })
            | Item::Union(ItemUnion { ident, .. })
            | Item::Trait(ItemTrait { ident, .. })
            | Item::TraitAlias(ItemTraitAlias { ident, .. })
            | Item::Type(ItemType { ident, .. })
            | Item::Const(ItemConst { ident, .. })
            | Item::Static(ItemStatic { ident, .. })
            | Item::Mod(ItemMod { ident, .. })
            | Item::Fn(ItemFn {
                sig: Signature { ident, .. },
                ..
            })
            | Item::Macro(ItemMacro {
                ident: Some(ident), ..
            }) => {
                names.insert(ident_name(ident));
            }
            Item::Use(ItemUse {
                leading_colon,
                tree,
                ..
            }) => {
                let is_absolute = leading_colon.is_some()
                    || matches!(tree, UseTree::Path(UsePath { ident, .. }) if ident == "$crate");
                if is_absolute {
                    use_names(tree, None, names);
                }
            }
            _ => (),
        }
    }
    module
}

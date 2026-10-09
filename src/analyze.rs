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
use crate::graph::is_whole_reference;
#[cfg(doc)]
use crate::leaker::Leaker;
use crate::unstable::{bindings, Projections, Unstable, Why};
use crate::util::{
    first_associated_self_in_generics, first_self_in_generics, first_self_in_type,
    generic_param_mentions_self, generic_param_name, has_lifetime_bound, ident_name,
    is_primitive_name, selves_in_generics, token_names, trait_object_of, unwrap_type_group,
    where_predicate_mentions_self, without_defaults, AllowedPaths,
};
use crate::view::View;
use proc_macro2::TokenStream;
use std::collections::{HashMap, HashSet};
use std::fmt::Display;
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use std::rc::Rc;
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
    /// A type not leaked alone because of an elided output lifetime: its holder's depth and the
    /// note (see [`Elision`]).
    elision: Option<(usize, String)>,
    unconstrained: Option<Lifetime>,
    /// The first `Self::A` in the type, where `Self` is a type: the Repeater's impl cannot write it,
    /// so a type that names it is not leaked whole.
    assoc_self: Option<TokenStream>,
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
            elision: None,
            unconstrained: None,
            assoc_self: None,
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
    /// An external type that the Leaker does not allow, which the Referrer may replace, holds no
    /// occurrence. Else the occurrences are those that rustc sees as constraining.
    careful: bool,
}

struct BinderGroup<'a> {
    output: Vec<Lifetime>,
    inputs: Vec<BinderInput<'a>>,
}

enum BinderInput<'a> {
    Type(&'a Type),
    Lifetime(&'a Lifetime),
}

/// A signature (a function pointer, the sugar of `Fn(..) -> R`, or a method) whose output has an
/// elided lifetime that elision takes from the only lifetime of the inputs (which a path may hide:
/// `Ref<u8>`, so it is the one that the inputs do not show). The copy of a type with
/// lifetimes in the inputs names them in the arguments of its projection, so the inputs have more
/// lifetimes and elision fails (E0106). Such a type is not leaked alone: the type holding the
/// signature is leaked as a whole instead if it can be, and it is an error otherwise. The same
/// holds for `Self`, which the copy of a struct, an enum or a union writes as the type itself, but
/// there the Referrer names the lifetime where it can (see
/// [`AnalyzeVisitor::self_elision_broken()`]). See [Elided lifetimes](crate#elided-lifetimes).
#[derive(Clone)]
struct Elision {
    /// The depth of the type holding the signature, leaked whole instead (0 for a method or a
    /// bound, which no type holds).
    owner: usize,
    /// The only lifetime of the inputs as written, `None` if it is elided or hidden.
    kept: Option<Lifetime>,
    /// The inputs show no lifetime, so the one that the output has is hidden by a path
    /// (`Ref<u8>` for `Ref<'a, u8>`), which the Leaker does not see: any lifetime that the copy adds
    /// to the inputs breaks the elision.
    hidden: bool,
    /// The signature of a function pointer or of `Fn(..)`, whose elided lifetime the Referrer can
    /// name with a `for<'e>` binder where `Visitor::with_self()` adds lifetimes to its inputs (see
    /// [`AnalyzeVisitor::self_elision_broken()`]). A method has none.
    binder: bool,
    /// The output, which has the elided lifetime.
    output: TokenStream,
}

impl Elision {
    /// Returns what the copy must keep for a signature with these `inputs` and `output`, if its
    /// output relies on elision: it has an elided lifetime, and the inputs have exactly one
    /// lifetime (else the original is not valid, or has no elision to keep), or none that is seen,
    /// so that a path hides it. `binder` is `true` for a function pointer or `Fn(..)`.
    fn of<'a>(
        owner: usize,
        binder: bool,
        inputs: impl IntoIterator<Item = &'a Type>,
        output: &ReturnType,
    ) -> Option<Self> {
        let output = match output {
            ReturnType::Type(_, output) => output,
            ReturnType::Default => return None,
        };
        let mut elided = ElidedOutput(false);
        elided.visit_type(output);
        if !elided.0 {
            return None;
        }
        let mut lifetimes = InputLifetimes::default();
        for input in inputs {
            lifetimes.visit_type(input);
        }
        let (kept, hidden) = match (lifetimes.elided, lifetimes.named.len()) {
            (1, 0) => (None, false),
            (0, 1) => (lifetimes.named.pop(), false),
            // An elided output with no lifetime in the inputs: the one a path hides.
            (0, 0) => (None, true),
            _ => return None,
        };
        Some(Self {
            owner,
            kept,
            hidden,
            binder,
            output: quote!(#output),
        })
    }

    /// Returns what `Elision::of()` gives for the signature of a method. A receiver that is a
    /// reference gives its lifetime to the output, whatever the inputs have.
    fn of_signature(sig: &Signature) -> Option<Self> {
        let mut inputs: Vec<&Type> = Vec::new();
        for input in &sig.inputs {
            match input {
                FnArg::Receiver(receiver) => {
                    let mut references = HasReference(receiver.reference.is_some());
                    references.visit_type(&receiver.ty);
                    if references.0 {
                        return None;
                    }
                    inputs.push(&receiver.ty);
                }
                FnArg::Typed(arg) => inputs.push(&arg.ty),
            }
        }
        Self::of(0, false, inputs, &sig.output)
    }

    /// Returns the lifetimes that the copy of a type with `context` adds to the inputs.
    fn added(&self, context: &Context) -> Vec<Lifetime> {
        let mut added: Vec<Lifetime> = Vec::new();
        let lifetimes = context
            .generics()
            .lifetimes()
            .map(|param| &param.lifetime)
            .chain(context.self_outlives());
        for lifetime in lifetimes {
            if Some(lifetime) != self.kept.as_ref() && !added.contains(lifetime) {
                added.push(lifetime.clone());
            }
        }
        added
    }

    /// Returns the lifetimes that a copy adds to the inputs by writing `self_ty` for `Self`
    /// (`Visitor::with_self()`): those it names, or an elided one (`'_`), not the one the inputs
    /// keep.
    fn added_by_self(&self, self_ty: &Type) -> Vec<Lifetime> {
        let mut lifetimes = InputLifetimes::default();
        lifetimes.visit_type(self_ty);
        let mut added: Vec<Lifetime> = Vec::new();
        if lifetimes.elided > 0 {
            added.push(Lifetime::new("'_", proc_macro2::Span::call_site()));
        }
        for lifetime in lifetimes.named {
            if Some(&lifetime) != self.kept.as_ref() && !added.contains(&lifetime) {
                added.push(lifetime);
            }
        }
        added
    }
}

/// `true` if an output has an elided lifetime: a reference or a lifetime argument without a name,
/// or `'_`. A function pointer or `Fn(..)` inside has its own elision.
pub(crate) struct ElidedOutput(pub(crate) bool);

impl<'ast> Visit<'ast> for ElidedOutput {
    fn visit_type_reference(&mut self, i: &'ast TypeReference) {
        self.0 |= i.lifetime.is_none();
        syn::visit::visit_type_reference(self, i)
    }
    fn visit_lifetime(&mut self, i: &'ast Lifetime) {
        self.0 |= i.ident == "_";
    }
    fn visit_type_bare_fn(&mut self, _: &'ast TypeBareFn) {}
    fn visit_parenthesized_generic_arguments(&mut self, _: &'ast ParenthesizedGenericArguments) {}
}

/// The lifetimes of the inputs of a signature as elision counts them: each elided one is its own,
/// the named ones count once. A function pointer or `Fn(..)` inside has its own, and so has a
/// bound with a `for<..>` binder for the lifetimes it declares.
#[derive(Default)]
pub(crate) struct InputLifetimes {
    pub(crate) elided: usize,
    pub(crate) named: Vec<Lifetime>,
    declared: Vec<Lifetime>,
}

impl<'ast> Visit<'ast> for InputLifetimes {
    fn visit_type_reference(&mut self, i: &'ast TypeReference) {
        if i.lifetime.is_none() {
            self.elided += 1;
        }
        syn::visit::visit_type_reference(self, i)
    }
    fn visit_lifetime(&mut self, i: &'ast Lifetime) {
        if i.ident == "_" {
            self.elided += 1;
        } else if !self.declared.contains(i) && !self.named.contains(i) {
            self.named.push(i.clone());
        }
    }
    fn visit_trait_bound(&mut self, i: &'ast TraitBound) {
        let outer = self.declared.len();
        self.declared.extend(binder_lifetimes(i.lifetimes.as_ref()));
        syn::visit::visit_trait_bound(self, i);
        self.declared.truncate(outer);
    }
    fn visit_type_bare_fn(&mut self, _: &'ast TypeBareFn) {}
    fn visit_parenthesized_generic_arguments(&mut self, _: &'ast ParenthesizedGenericArguments) {}
}

/// `true` if a type has a reference: a receiver (`self: &Self`, `self: Pin<&mut Self>`) gives its
/// lifetime to the output.
struct HasReference(bool);

impl<'ast> Visit<'ast> for HasReference {
    fn visit_type_reference(&mut self, _: &'ast TypeReference) {
        self.0 = true;
    }
}

#[derive(Clone)]
struct ModuleFrame {
    id: ContextId,
    names: Rc<ModuleNames>,
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
    /// `Self` is a type that the copy writes (the self type of an impl), not the item itself.
    written: bool,
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
        written: bool,
        trait_path: Option<&Path>,
        f: impl FnOnce(&mut View<'_>) -> R,
    ) -> Res<(ContextId, R)> {
        let declares = Some(DeclaredSelf {
            requirement,
            allow_self,
            written,
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
        self.enter_declaring(generics, requirement, allow_self, false, None, f)
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
        self.enter_declaring(generics, requirement, allow_self, true, trait_path, f)
    }

    fn enter_trait<R>(
        &mut self,
        ident: &Ident,
        generics: &Generics,
        allow_self: bool,
        f: impl FnOnce(&mut View<'_>) -> R,
    ) -> Res<(ContextId, R)> {
        let requirement = SelfRequirement::of_trait(ident, generics);
        self.enter_declaring(generics, requirement, allow_self, false, None, f)
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
    self_written: bool,
    self_depth: usize,
    allow_self: bool,
    self_type_walk: Option<bool>,
    inherent_impl: bool,
    not_optional: usize,
    env_errors: usize,
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
    elisions: usize,
    signature: Option<Elision>,
    bound_elided: usize,
    signatures: usize,
    probe: Option<HashSet<usize>>,
    in_default: Option<bool>,
    expr_owner: Option<usize>,
    in_receiver: bool,
    enclosing: Enclosing,
    env_depth: usize,
    in_fn_arg: bool,
    elided_ref: bool,
    unconstrained_ref: Option<Lifetime>,
    ambiguous_arg: bool,
    inherits: Option<Inherits>,
}

pub(crate) struct Finished {
    pub(crate) effects: Vec<Effect>,
    pub(crate) top_levels: Vec<ContextId>,
    pub(crate) module: Option<ContextId>,
    /// The types, with their contexts, that cannot be optional entries (see
    /// [`AnalyzeVisitor::env_cycle()`]).
    pub(crate) not_optional: Vec<(Context, String)>,
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
    /// The signatures whose inputs are being visited, innermost last: what each keeps from its
    /// inputs (see [`Elision`]). `None` for one that needs nothing, which also hides the ones around
    /// it: lifetimes in a function pointer or `Fn(..)` are not those of the signature around it.
    elisions: Vec<Option<Elision>>,
    /// The elision of the method whose arguments are being visited.
    signature: Option<Elision>,
    /// The function pointers and `Fn(..)` being visited, innermost last: the positions in
    /// `impossibles` of the elided lifetimes each binds (see [`AnalyzeVisitor::push_elided()`]).
    bound_elided: Vec<Vec<usize>>,
    /// While a binder is probed: the addresses of the types found to be leaked whole.
    probe: Option<HashSet<usize>>,
    root_defaulted: Vec<Ident>,
    /// What `Self` means here, if anything. It has none directly in a module, or in generics that
    /// no item declaring it encloses, so mentioning it there is an error.
    self_ty: Option<SelfRequirement>,
    /// `self_ty` is the self type of an impl, which the copy writes (see [`DeclaredSelf`]).
    self_written: bool,
    /// The number of scopes when the scope that declares `Self` is the innermost: 0 if there is no
    /// `Self`. The predicates of the item that declares `Self` are visited then, not those of its
    /// methods.
    self_depth: usize,
    root_declares_self: bool,
    /// Set while the walk of an impl visits it (see [`AnalyzeVisitor::walk_item_impl()`]): whether
    /// a bare `Self` in a bound or where predicate of the impl was found. The walk then tells the
    /// graph not to choose a type in the self type whose bounds are unstable (see
    /// [`AnalyzeVisitor::block_unstable_self_types()`]).
    self_type_walk: Option<bool>,
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
    /// The signatures being visited, innermost last. Only those that rustc compares with a trait's
    /// have one (see [`AnalyzeVisitor::with_signature()`]).
    signatures: Vec<SignatureWalk>,
    /// The methods being visited are those of an inherent impl: no trait declares their signature,
    /// so the copy cannot differ from it (E0195). Anywhere else, a method may be one of a trait.
    inherent_impl: bool,
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
    /// Whether [`AnalyzeVisitor::env_cycle()`] is checked. It is not by the probe that it runs.
    check_env: bool,
    /// The types, with the contexts they are in, that were found to be no optional entries.
    not_optional: Vec<(Context, String)>,
    /// The cycles that `Self` makes (see [`AnalyzeVisitor::env_cycle()`]). Kept apart from the
    /// effects, like `self_errors`.
    env_errors: Vec<CheckResult>,
}

/// A method signature being visited (see `AnalyzeVisitor::signatures`).
struct SignatureWalk {
    /// The index in `scopes` of its generics.
    at: usize,
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
            elisions: Vec::new(),
            signature: None,
            bound_elided: Vec::new(),
            probe: None,
            root_defaulted: defaulted(generics)
                .map(|(_, ident, _)| ident.clone())
                .collect(),
            self_ty: None,
            self_written: false,
            self_depth: 0,
            root_declares_self: false,
            self_type_walk: None,
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
            inherent_impl: false,
            expr_owner: None,
            in_receiver: false,
            enclosing: Enclosing::default(),
            env_depth: 0,
            in_fn_arg: false,
            check_env: true,
            not_optional: Vec::new(),
            env_errors: Vec::new(),
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

    /// Cuts `impossibles` to `len`, with the positions that the function pointers and `Fn(..)`
    /// being visited recorded in it.
    fn truncate_impossibles(&mut self, len: usize) {
        self.impossibles.truncate(len);
        for positions in &mut self.bound_elided {
            positions.retain(|&at| at < len);
        }
    }

    /// Records an elided lifetime as a cause that stops the type holding it from being leaked
    /// whole: the impl of the Repeater cannot name it. Not so for the type that holds the
    /// function pointer or `Fn(..)` which binds it, whose copy has it as written (see
    /// [`AnalyzeVisitor::with_bound_elided()`]). `tokens` is the type or lifetime written.
    fn push_elided(&mut self, tokens: TokenStream) {
        if let Some(positions) = self.bound_elided.last_mut() {
            positions.push(self.impossibles.len());
        }
        self.impossibles.push(tokens);
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
        // A type that names an item of the module (`crate::base::B1<R0>` with a declared `R0`) means
        // something only there, like one with a relative path: the Repeater impl of its entry is
        // emitted inside the module, where the name resolves, and the copy of the module declares
        // the name again, so an entry of no module is hidden by the Referrer and left as written.
        Ok(match self.modules.last() {
            Some(module)
                if inner.relative || inner.module_item || self.depends_on_module(&context) =>
            {
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

    /// Returns the signature whose elision the copy of a type with `context` would break, with the
    /// lifetimes it adds. It is the innermost one that has the type in its inputs (see
    /// [`Elision`]).
    fn elision_broken(&self, context: &Context) -> Option<(Elision, Vec<Lifetime>)> {
        let elision = self.elisions.last()?.as_ref()?;
        let added = elision.added(context);
        if added.is_empty() {
            None
        } else {
            Some((elision.clone(), added))
        }
    }

    /// Returns the signature whose elision the copy would break by writing the type for `Self` or
    /// `Self::A`, if `ty` is one, with the lifetimes it adds. The copy of a struct, an enum or a
    /// union is a new item, so `Visitor::with_self()` writes `Self` as the type of the Leaker
    /// (`::p::S<'a>`), whereas `Self` itself names no lifetime to elision. The copy of an impl
    /// keeps `Self`, and so does a receiver (`self: Box<Self>`), which the Visitor does not
    /// rewrite.
    ///
    /// The signature of a function pointer or of `Fn(..)` whose inputs show their one lifetime is
    /// not broken: the Visitor names that lifetime with a `for<'e>` binder, in the inputs and in the
    /// output, so that `Self` adds nothing to count (`Elision::binder`). What it cannot name is the
    /// one that a path hides, and a method has no binder to write: those are broken. See
    /// [`Elision`].
    fn self_elision_broken(&self, ty: &Type) -> Option<(Elision, Vec<Lifetime>)> {
        if self.in_receiver || self.self_written || !is_self_path(ty) {
            return None;
        }
        let self_ty = match &self.self_ty {
            Some(SelfRequirement::Type(self_ty)) => self_ty,
            _ => return None,
        };
        let elision = self.elisions.last()?.as_ref()?;
        if elision.binder && !elision.hidden {
            return None;
        }
        let added = elision.added_by_self(self_ty);
        if added.is_empty() {
            None
        } else {
            Some((elision.clone(), added))
        }
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
        if let Some(error) = scope_error(&effects, &self.self_errors, &self.env_errors) {
            return Err(error);
        }
        Ok(Finished {
            effects,
            top_levels: std::mem::take(&mut self.top_levels),
            module: self.modules.first().map(|module| module.id),
            not_optional: std::mem::take(&mut self.not_optional),
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
            self_written: self.self_written,
            self_depth: self.self_depth,
            allow_self: self.allow_self,
            self_type_walk: self.self_type_walk,
            inherent_impl: self.inherent_impl,
            not_optional: self.not_optional.len(),
            env_errors: self.env_errors.len(),
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
                    names: Rc::new(names.clone()),
                });
                self.self_ty = None;
                self.self_written = false;
                self.self_depth = 0;
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
                    self.self_written = declared.written;
                    self.allow_self = declared.allow_self;
                }
                if root {
                    self.scopes[0].set_self(self.self_ty.clone());
                } else {
                    self.scopes.push(Scope::new(generics, self.self_ty.clone()));
                }
                if declares.is_some() {
                    self.self_depth = self.scopes.len();
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
        self.self_written = mark.self_written;
        self.self_depth = mark.self_depth;
        self.allow_self = mark.allow_self;
        self.self_type_walk = mark.self_type_walk;
        self.inherent_impl = mark.inherent_impl;
        if rollback {
            self.frame.effects.truncate(mark.effects);
            self.self_errors.truncate(mark.self_errors);
            self.contexts.truncate(mark.contexts);
            self.top_levels.truncate(mark.top_levels);
            self.not_optional.truncate(mark.not_optional);
            self.env_errors.truncate(mark.env_errors);
        }
    }

    /// Starts a check: sets the frame of the scope aside for the visit to use a new one.
    fn checkpoint(&mut self) -> Checkpoint {
        let mark = self.mark();
        let walk = Walk {
            type_depth: self.type_depth,
            binders: self.binders.len(),
            elisions: self.elisions.len(),
            signature: self.signature.clone(),
            bound_elided: self.bound_elided.len(),
            signatures: self.signatures.len(),
            probe: self.probe.clone(),
            in_default: self.in_default,
            expr_owner: self.expr_owner,
            in_receiver: self.in_receiver,
            enclosing: self.enclosing.clone(),
            env_depth: self.env_depth,
            in_fn_arg: self.in_fn_arg,
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
        self.truncate_impossibles(impossibles);
        self.type_depth = walk.type_depth;
        self.binders.truncate(walk.binders);
        self.elisions.truncate(walk.elisions);
        self.signature = walk.signature;
        self.bound_elided.truncate(walk.bound_elided);
        self.signatures.truncate(walk.signatures);
        self.probe = walk.probe;
        self.in_default = walk.in_default;
        self.expr_owner = walk.expr_owner;
        self.in_receiver = walk.in_receiver;
        self.enclosing = walk.enclosing;
        self.env_depth = walk.env_depth;
        self.in_fn_arg = walk.in_fn_arg;
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
            &self.env_errors[mark.env_errors..],
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
        if self.frame.assoc_self.is_none() {
            self.frame.assoc_self = inner.assoc_self.clone();
        }
        // Kept for the enclosing type, unless that is the type holding the binder.
        if self.frame.binder.is_none() {
            let depth = self.type_depth;
            self.frame.binder = inner.binder.clone().filter(|(owner, _)| *owner < depth);
        }
        if self.frame.elision.is_none() {
            let depth = self.type_depth;
            self.frame.elision = inner.elision.clone().filter(|(owner, _)| *owner < depth);
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
                .or_else(|| {
                    let (_, note) = inner
                        .elision
                        .as_ref()
                        .filter(|(owner, _)| *owner == depth)?;
                    Some(note.clone())
                })
                .or_else(|| {
                    inner
                        .unconstrained
                        .as_ref()
                        .map(|lifetime| unconstrained_note(lifetime, true))
                });
            if let (Some(cause), Some(SelfRequirement::Type(self_ty))) =
                (&inner.assoc_self, &self.self_ty)
            {
                conflict.note = Some(assoc_self_note(cause, self_ty));
                conflict.headline = Some(assoc_self_headline(cause));
            }
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
            None => match self.associated_self_in_bounds(&context) {
                Some(conflict) => Err(conflict.of(ty)),
                None => Ok(context),
            },
        }
    }

    /// Returns the first `Self::A` in the bounds of the generics of `context`, if `Self` is a
    /// type: the Repeater's impl for it would write the bound as it is, and has no trait to take
    /// `A` from. (The impl for the implementor of a trait can write it: `__S::A`.)
    fn associated_self_in_bounds(&self, context: &Context) -> Option<AssocSelf> {
        match context.self_ty() {
            Some(SelfRequirement::Type(self_ty)) => {
                first_associated_self_in_generics(context.generics())
                    .map(|cause| AssocSelf::new(cause, self_ty))
            }
            _ => None,
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
        // `Self` (or `Self::A`) that the copy writes as a type with lifetimes, in the inputs of a
        // signature whose output relies on elision: `Self` is no entry, so it is the type holding
        // the signature that is leaked whole instead, or a conflict (in a bound or a method). Like
        // the entries that add lifetimes, below.
        let mut self_conflict = None;
        if let Some((elision, added)) = self.self_elision_broken(ty) {
            if elision.owner > 0 {
                self.musts.push((elision.owner, quote!(#ty)));
                let note = self_elision_note(&added, &elision, true);
                inner.elision = Some((elision.owner, note));
                inner.min_depth = inner.min_depth.min(elision.owner);
                return self.decide_as_owner(elision.owner, inner);
            }
            self.musts.push((depth, quote!(#ty)));
            inner.min_depth = inner.min_depth.min(depth);
            let mut conflict = self.causes(inner.must_start, inner.impossible_start, depth);
            conflict.impossible.insert(0, elision.output.clone());
            conflict.note = Some(self_elision_note(&added, &elision, false));
            self_conflict = Some(conflict);
        }
        if inner.min_depth != depth && self.is_ambiguous_owner(ty, depth, &inner) {
            self.own_causes(inner.must_start, depth);
            inner.min_depth = depth;
        }
        // A type leaked whole for its own path cannot use a binder lifetime of the output: the type
        // holding the binder is leaked instead, or it is a conflict (in a bound).
        let mut own = (inner.min_depth == depth).then(|| self.own_context(ty, depth, &inner));
        if let Some(conflict) = self_conflict {
            own = Some(Err(conflict));
        }
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
        // The same for a type that adds lifetimes to the inputs of a signature whose output relies
        // on elision: the type holding the signature is leaked instead, or it is a conflict (in a
        // bound or a method).
        let broken = match &own {
            Some(Ok(context)) => self.elision_broken(context),
            _ => None,
        };
        if let Some((elision, added)) = broken {
            if elision.owner > 0 {
                let note = elision_note(&added, &elision, true);
                inner.elision = Some((elision.owner, note));
                return self.decide_as_owner(elision.owner, inner);
            }
            let mut conflict = self.causes(inner.must_start, inner.impossible_start, depth);
            conflict.impossible.insert(0, elision.output.clone());
            conflict.note = Some(elision_note(&added, &elision, false));
            own = Some(Err(conflict));
        }
        if let Some(Ok(_)) = &own {
            self.record_leaked(ty, depth, &inner);
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
        let soft_elision = soft
            .as_ref()
            .and_then(|context| self.elision_broken(context));
        if let Some((elision, _)) = &soft_elision {
            if elision.owner > 0 {
                inner.soft_depth = elision.owner;
                return self.decide_through(inner);
            }
        }
        // An optional entry that would make a cycle in the copy (see `env_cycle()`) is not one: the
        // Referrer cannot replace it, so it is copied as written. That holds for the type in this
        // context everywhere, as the Referrer finds an entry by its type and context.
        let optional = match soft.filter(|_| soft_binder.is_none() && soft_elision.is_none()) {
            Some(context) if self.env_blocked(&context, true) => {
                self.not_optional.push((context, quote!(#ty).to_string()));
                None
            }
            optional => optional,
        };
        self.absorb(&inner);
        let has_impossible = self.impossibles.len() > inner.impossible_start;
        let has_must = inner.min_depth != usize::MAX;
        let mentions_local = inner.local_scope < self.scopes.len();
        let effects = std::mem::take(&mut inner.effects);

        // `Self` and its associated types are resolved by what the copy writes for `Self`, also
        // where `Self` cannot be leaked. The error is kept apart from the effects, as the type
        // around may be copied, which drops them.
        if own.is_none() && self.env_depth > 0 && self.check_env {
            let projection = is_self_projection(ty);
            if projection || (self.self_written && is_bare_self(ty)) {
                if let Ok(context) = self.context_of(ty, &inner) {
                    // The self type of an impl needs a rewrite only if it names a leaked type.
                    if projection || self.self_type_names_leaked(&context, false) {
                        // `Self::A` needs the bounds of the generics of its context. `Self` of an
                        // impl needs those of the types in the self type that the copy writes as
                        // paths through the Repeater only: rustc normalizes them in the
                        // environment, not the bounds of a generic that they do not use. A type
                        // in the self type that the graph may choose, and whose bounds are
                        // unstable, is not chosen then: its parts are (see
                        // `block_unstable_self_types()`), if the walk of the impl can tell the
                        // graph.
                        let blockable = !projection && self.self_type_walk.is_some();
                        if blockable {
                            self.self_type_walk = Some(true);
                        }
                        let needs = if projection {
                            vec![context.clone()]
                        } else {
                            self.self_type_entries(&context, false, blockable)
                        };
                        for context in &needs {
                            if let Some(conflict) = self.self_env_cycle(ty, context) {
                                return self.env_errors.push(conflict);
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
                }
            }
        }
        let pending = match own {
            Some(Ok(context)) => {
                if let Some(conflict) = self.env_cycle(ty, &context) {
                    return self.frame.effects.push(Effect::Error(conflict));
                }
                self.not_optional_in_bounds(&context);
                PendingType {
                    is_must_intern: true,
                    in_graph: true,
                    ambiguous: ambiguous && context.is_empty(),
                    ..PendingType::new(ty, context, Vec::new())
                }
            }
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
                let in_graph = has_must && self.elision_broken(&context).is_none();
                // A type that is an entry only if its parts need it makes a cycle too, but its
                // parts can be the entries instead, unless it is leaked whole in any case.
                let leaked_whole = has_must && is_whole_reference(ty);
                if leaked_whole {
                    if let Some(conflict) = self.env_cycle(ty, &context) {
                        return self.frame.effects.push(Effect::Error(conflict));
                    }
                    self.not_optional_in_bounds(&context);
                }
                let has_optional = optional.is_some() || has_optional(&effects);
                let no_entry = has_must && self.env_blocked(&context, false);
                if has_must && !no_entry {
                    self.not_optional_in_bounds(&context);
                }
                let no_optional_entry = has_optional && self.env_blocked(&context, true);
                // A reference that is an entry is one as a whole, so its parts cannot stand for
                // it. If it cannot be an entry, the optional entries in it are not either.
                if no_optional_entry && is_whole_reference(ty) {
                    self.not_optional.extend(optional_keys(&effects));
                }
                PendingType {
                    in_graph,
                    optional: optional.is_some(),
                    has_optional,
                    no_entry,
                    no_optional_entry,
                    ..PendingType::new(ty, context, effects)
                }
            }
        };
        self.frame.effects.push(Effect::Type(Box::new(pending)));
    }

    /// Returns the conflict if `ty`, leaked whole in `context`, would make a cycle in the copy: it
    /// is in a bound or a where predicate, and a bound of its generics is unstable (see
    /// [`crate::unstable`]: it names a leaked type, `Self`, `Self::A` or an associated type that
    /// rustc normalizes), or it names `Self` in a predicate of the trait that declares it.
    ///
    /// The copy writes it as a path through the Repeater. rustc normalizes it in a predicate of
    /// the item while it builds the environment of the item, in which the bounds are not
    /// normalized yet: the impl of the Repeater for `ty` needs the bound of its generics (`T:
    /// Fn(Rel)`), and the environment has it as `T: Fn(<() as Repeater<..>>::Type)`. So it
    /// cannot prove it (E0277 "expected an `Fn(Rel)` closure, found `T`", E0275, E0284). Any
    /// predicate of the item does it, a bound as well as a where predicate, also the bound that
    /// names `ty` itself (`T: Tr<Wrap<T>>`). Types in fields and signatures are fine: they are
    /// normalized in the environment once it is built. So are bounds that are stable (`T:
    /// Clone`), and predicates on a type that is not a generic, which an impl proves. The Leaker
    /// cannot rewrite it, as the Repeater is the only link between the two crates, so it is an
    /// error.
    ///
    fn env_cycle(&self, ty: &Type, context: &Context) -> Option<CheckResult> {
        match self.env_unstable_bound(context, false) {
            Some(unstable) => {
                let why = match &unstable.why {
                    Why::Leaked => format!(
                        "the bound `{}` of its generics names a type that is leaked too: the copy \
                         writes both as paths through the Repeater, and rustc cannot normalize the \
                         one in a bound or where predicate while it checks the bounds (E0277, \
                         E0275)",
                        unstable.bound
                    ),
                    _ => format!(
                        "the bound `{}` of its generics {}",
                        unstable.bound,
                        unstable.explain()
                    ),
                };
                Some(self.unstable_error(ty, &unstable, why))
            }
            None => {
                let bound = self.trait_cycle(context)?;
                let why = format!(
                    "the copy writes it as a path through the Repeater, whose impl is for the \
                     implementors of the trait (`{}`), and rustc cannot prove that the host \
                     implements the trait while it checks this predicate (E0275)",
                    bound
                );
                let mut error = self.cycle_error(ty, why);
                error.headline = Some(format!(
                    "`{}` names `Self` in a bound or where predicate of its trait",
                    quote!(#ty)
                ));
                Some(error)
            }
        }
    }

    /// Like `env_cycle()` for `Self::A` and for `Self` of an impl: they need the bounds of the
    /// generics of their context only, and `Self::A` of an impl the bounds of the impl, whose impl
    /// of the trait rustc selects to resolve it. For `Self` of an impl, `context` is that of a type
    /// in the self type that the copy writes through the Repeater
    /// ([`AnalyzeVisitor::self_type_entries()`]): the generics that no such type uses do not
    /// matter, rustc does not need their bounds to normalize `Self`.
    fn self_env_cycle(&self, ty: &Type, context: &Context) -> Option<CheckResult> {
        let projection = is_self_projection(ty);
        let (unstable, owner) = match self.env_unstable_bound(context, false) {
            Some(unstable) => (unstable, "its generics"),
            None => (self.impl_unstable_bound(projection)?, "the impl"),
        };
        let why = match (&unstable.why, projection) {
            (Why::Leaked, true) => format!(
                "the copy for a concrete implementor resolves `{}` by the impl of the trait, \
                 which needs the bound `{}` of {}, where the copy writes the leaked type \
                 as a path through the Repeater, and rustc cannot normalize the one in a bound or \
                 where predicate while it checks the bounds (E0277, E0275)",
                quote!(#ty),
                unstable.bound,
                owner
            ),
            (Why::Leaked, false) => format!(
                "`Self` is the self type of the impl, which the copy writes with its leaked types \
                 as paths through the Repeater, and the bound `{}` of their generics names a type \
                 that is leaked too, and rustc cannot normalize the one in a bound or where \
                 predicate while it checks the bounds (E0277, E0275)",
                unstable.bound
            ),
            (_, true) => format!(
                "the copy for a concrete implementor resolves `{}` by the impl of the trait, \
                 which needs the bound `{}` of {}, which {}",
                quote!(#ty),
                unstable.bound,
                owner,
                unstable.explain()
            ),
            (_, false) => format!(
                "`Self` is the self type of the impl, which the copy writes with its leaked types \
                 as paths through the Repeater, and the bound `{}` of their generics {}",
                unstable.bound,
                unstable.explain()
            ),
        };
        Some(self.unstable_error(ty, &unstable, why))
    }

    /// Returns the first bound of the impl that declares `Self` that the environment cannot prove,
    /// for `Self::A` (`projection`): rustc resolves it by the impl of the trait, which has all the
    /// generics of the impl, not only those that `Self` uses.
    fn impl_unstable_bound(&self, projection: bool) -> Option<Unstable> {
        self.env_unstable_bound(&self.impl_context(projection)?, false)
    }

    /// Returns the generics of the impl that declares `Self`, if `Self::A` (`projection`) is in
    /// one of its predicates, as a context.
    fn impl_context(&self, projection: bool) -> Option<Context> {
        if !projection || !self.self_written || self.self_depth == 0 {
            return None;
        }
        let scope = self.scopes.get(self.self_depth - 1)?;
        Some(Context::new(scope.generics(), self.self_ty.clone()))
    }

    fn cycle_error(&self, ty: &Type, why: String) -> CheckResult {
        CheckResult {
            must: vec![quote!(#ty)],
            impossible: Vec::new(),
            is_own: true,
            optional: false,
            note: Some(format!(
                "{}. Write the leaked type of the bound with an absolute path, or use `{}` outside \
                 the bounds and where predicates",
                why,
                quote!(#ty)
            )),
            headline: Some(format!(
                "`{}` is in a bound or where predicate, and a bound of its generics names a leaked \
                 type",
                quote!(#ty)
            )),
        }
    }

    /// The error for `ty`, which is in a predicate and needs `unstable`: like `cycle_error()`, with
    /// the headline and the advice of what the bound names.
    fn unstable_error(&self, ty: &Type, unstable: &Unstable, why: String) -> CheckResult {
        if matches!(unstable.why, Why::Leaked) {
            return self.cycle_error(ty, why);
        }
        CheckResult {
            must: vec![quote!(#ty)],
            impossible: Vec::new(),
            is_own: true,
            optional: false,
            note: Some(format!(
                "{}. {}, or use `{}` outside the bounds and where predicates",
                why,
                unstable.why.advice(),
                quote!(#ty)
            )),
            headline: Some(format!(
                "`{}` is in a bound or where predicate, and a bound of its generics names {}",
                quote!(#ty),
                unstable.why.names()
            )),
        }
    }

    /// Returns `Self: Trait` if a type in `context` names `Self` of a trait, and the predicates of
    /// that trait are being visited (not those of its methods): see `env_cycle()`.
    fn trait_cycle(&self, context: &Context) -> Option<TokenStream> {
        if self.env_depth == 0
            || !self.check_env
            || self.self_depth == 0
            || self.scopes.len() != self.self_depth
        {
            return None;
        }
        match context.self_ty() {
            Some(SelfRequirement::Bound(path)) => Some(quote!(Self: #path)),
            _ => None,
        }
    }

    /// Returns `true` if a type in `context` that is an entry would make a cycle (`env_cycle()`).
    fn env_blocked(&self, context: &Context, optional: bool) -> bool {
        self.env_unstable_bound(context, optional).is_some() || self.trait_cycle(context).is_some()
    }

    /// Runs `visit` on a probe visitor that has the generics and the `Self` of `context`, and
    /// checks nothing itself.
    fn probe<R>(&self, context: &Context, visit: impl FnOnce(&mut AnalyzeVisitor) -> R) -> R {
        Self::alone(self.allowed.clone(), context.generics(), |probe| {
            probe.check_env = false;
            probe.modules = self.modules.clone();
            probe.self_ty = context.self_ty().cloned();
            visit(probe)
        })
    }

    /// Returns `true` if the self type of `context`, which is the self type of an impl, names a
    /// type that is leaked, so the copy writes it as a path through the Repeater. With `optional`,
    /// a type that the Referrer may replace counts too.
    fn self_type_names_leaked(&self, context: &Context, optional: bool) -> bool {
        match context.self_ty() {
            Some(SelfRequirement::Type(ty)) if self.env_depth > 0 && self.check_env => {
                self.probe(context, |probe| {
                    probe.visit_type(ty);
                    !probe.musts.is_empty() || (optional && has_optional(&probe.frame.effects))
                })
            }
            _ => false,
        }
    }

    /// Returns the contexts of the types in the self type of `context`, which is the self type of
    /// an impl, that the copy may write as a path through the Repeater: those that must be entries
    /// and those that the graph may choose (see [`AnalyzeVisitor::cut_self_cycles()`] for those it
    /// cannot). With `optional`, the optional entries count too.
    ///
    /// rustc normalizes `Self` where a predicate names it, and the Repeater's impl of each of them
    /// needs the bounds of the generics that it uses, not those of the other generics of the impl.
    /// An entry that depends on `Self` (a bound of the impl names it) is one that must stay: it is
    /// an error there, and the caller finds the bound that names `Self`.
    ///
    /// With `blockable`, a type that is not one that must be an entry, and whose bounds are
    /// unstable (`env_unstable_bound()`), is not one the graph may choose: its parts are the entries
    /// (see [`AnalyzeVisitor::block_unstable_self_types()`], which tells the graph).
    fn self_type_entries(
        &self,
        context: &Context,
        optional: bool,
        blockable: bool,
    ) -> Vec<Context> {
        let ty = match context.self_ty() {
            Some(SelfRequirement::Type(ty)) if self.env_depth > 0 && self.check_env => ty,
            _ => return Vec::new(),
        };
        let declared = self
            .scopes
            .get(self.self_depth.wrapping_sub(1))
            .map(Scope::generics);
        let blocked =
            |context: &Context| blockable && self.env_unstable_bound(context, false).is_some();
        self.probe(context, |probe| {
            probe.visit_type(ty);
            let mut found = Vec::new();
            entry_contexts(
                &probe.frame.effects,
                declared.as_ref(),
                &blocked,
                optional,
                &mut found,
            );
            found
        })
    }

    /// Returns the first bound of the generics of `context` that names a type that is leaked, if
    /// the type being visited is in a predicate of the item. With `optional`, a type that the
    /// Referrer may replace counts too.
    fn env_leaked_bound(&self, context: &Context, optional: bool) -> Option<TokenStream> {
        if self.env_depth == 0 || !self.check_env {
            return None;
        }
        let generics = context.generics();
        // Without bounds, nothing is named.
        let bounded = generics.where_clause.is_some()
            || generics.params.iter().any(|param| match param {
                GenericParam::Type(param) => !param.bounds.is_empty(),
                _ => false,
            });
        if !bounded {
            return None;
        }
        let names_leaked = |visit: &dyn Fn(&mut AnalyzeVisitor)| {
            self.probe(context, |probe| {
                visit(probe);
                !probe.musts.is_empty() || (optional && has_optional(&probe.frame.effects))
            })
        };
        for param in &generics.params {
            if let GenericParam::Type(param) = param {
                if names_leaked(&|probe| probe.visit_type_param(param)) {
                    return Some(quote!(#param));
                }
            }
        }
        // A where predicate that is not on a parameter has the types of the leaked type in its
        // bounded type, and rustc may prove it by an impl: it does not need the environment.
        generics
            .where_clause
            .iter()
            .flat_map(|where_clause| &where_clause.predicates)
            .filter(|predicate| needs_environment(predicate, context))
            .find(|predicate| names_leaked(&|probe| probe.visit_where_predicate(predicate)))
            .map(|predicate| quote!(#predicate))
    }

    /// Returns the first bound of the generics of `context` that the environment cannot prove for
    /// the impl that rustc selects for a type in a predicate (see [`crate::unstable`]): one that
    /// names a leaked type (`env_leaked_bound()`), or `Self` if the copy writes the self type as a
    /// leaked one, `Self::A`, or an associated type that rustc normalizes.
    fn env_unstable_bound(&self, context: &Context, optional: bool) -> Option<Unstable> {
        if let Some(bound) = self.env_leaked_bound(context, optional) {
            return Some(Unstable {
                bound,
                why: Why::Leaked,
            });
        }
        if self.env_depth == 0 || !self.check_env {
            return None;
        }
        let generics = context.generics();
        let bounded = generics.where_clause.is_some()
            || generics.params.iter().any(|param| match param {
                GenericParam::Type(param) => !param.bounds.is_empty(),
                _ => false,
            });
        if !bounded {
            return None;
        }
        // The equations of associated types in the environment: `T::Item` of those is not rigid.
        let bindings = bindings(
            self.scopes.iter().flat_map(Scope::params),
            self.scopes.iter().flat_map(Scope::predicates),
        );
        let is_param = |name: &str| self.scope_of(name).is_some();
        // Whether `Self` is the self type of an impl that the copy writes as a leaked type.
        let mut self_leaked: Option<bool> = None;
        let mut check = |bound: TokenStream, found: Option<Why>, mentions_self: bool| {
            let why = match found {
                Some(why) => why,
                None if mentions_self
                    && self.self_written
                    && *self_leaked
                        .get_or_insert_with(|| self.self_type_names_leaked(context, optional)) =>
                {
                    Why::SelfType
                }
                None => return None,
            };
            Some(Unstable { bound, why })
        };
        for param in &generics.params {
            if let GenericParam::Type(type_param) = param {
                let mut finder = Projections::new(&is_param, &bindings);
                finder.visit_type_param(type_param);
                let found = check(
                    quote!(#type_param),
                    finder.found,
                    generic_param_mentions_self(param),
                );
                if found.is_some() {
                    return found;
                }
            }
        }
        for predicate in generics
            .where_clause
            .iter()
            .flat_map(|where_clause| &where_clause.predicates)
            .filter(|predicate| needs_environment(predicate, context))
        {
            // What the predicate bounds is not a bound of it: only the bounds are normalized for
            // the impl that needs them (`Self::A: Clone` is proved by an impl).
            let mut finder = Projections::new(&is_param, &bindings);
            if let WherePredicate::Type(predicate) = predicate {
                for bound in &predicate.bounds {
                    finder.visit_type_param_bound(bound);
                }
            }
            let found = check(
                quote!(#predicate),
                finder.found,
                where_predicate_mentions_self(predicate),
            );
            if found.is_some() {
                return found;
            }
        }
        None
    }

    /// Called for a type that is leaked in a bound or where predicate, in `context`, and would
    /// make a cycle if the Referrer replaced the optional entries that the bounds of the generics
    /// name (it does if it does not share the crates): they are not optional entries.
    fn not_optional_in_bounds(&mut self, context: &Context) {
        let unstable = match self.env_unstable_bound(context, true) {
            Some(unstable) => unstable,
            None => return,
        };
        let keys = self.probe(context, |probe| {
            probe.visit_generics(context.generics());
            // The bound that names `Self` is unstable for the self type that the copy writes.
            if let (Why::SelfType, Some(SelfRequirement::Type(ty))) =
                (&unstable.why, context.self_ty())
            {
                probe.visit_type(ty);
            }
            optional_keys(&probe.frame.effects)
        });
        self.not_optional.extend(keys);
    }

    /// Like `not_optional_in_bounds()` for the optional entries of the self type of an impl (in
    /// `context`), which `Self` is in a bound or where predicate.
    fn not_optional_in_self_type(&mut self, context: &Context) {
        let ty = match context.self_ty() {
            Some(SelfRequirement::Type(ty)) if self.env_depth > 0 && self.check_env => ty,
            _ => return,
        };
        let keys = self.probe(context, |probe| {
            probe.visit_type(ty);
            optional_keys(&probe.frame.effects)
        });
        for (context, ty) in keys {
            if self.env_leaked_bound(&context, true).is_some() {
                self.not_optional.push((context, ty));
            }
        }
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

    /// Walks the parts of an impl as `syn::visit::visit_item_impl()` does, with the self type
    /// apart: the copy writes it as a type, and `Self` stands for it in the rest of the impl, so
    /// no entry in it can depend on `Self` (see [`AnalyzeVisitor::cut_self_cycles()`]).
    pub(crate) fn walk_item_impl(&mut self, i: &ItemImpl) {
        for attr in &i.attrs {
            self.visit_attribute(attr);
        }
        let outer = self.self_type_walk.replace(false);
        self.visit_generics(&i.generics);
        if let Some((_, path, _)) = &i.trait_ {
            self.visit_path(path);
        }
        let start = self.frame.effects.len();
        self.visit_type(&i.self_ty);
        let end = self.frame.effects.len();
        self.cut_self_cycles(start, &i.generics);
        let inherent = std::mem::replace(&mut self.inherent_impl, i.trait_.is_none());
        for item in &i.items {
            self.visit_impl_item(item);
        }
        self.inherent_impl = inherent;
        if self.self_type_walk == Some(true) {
            self.block_unstable_self_types(start, end);
        }
        self.self_type_walk = outer;
    }

    /// Called after the walk of an impl, with the effects `start..end` those of its self type,
    /// where a bare `Self` is in a bound or where predicate.
    ///
    /// rustc normalizes `Self` there, with the impls of the Repeater for the types in the self type
    /// that are entries, and needs the bounds of the generics that each of them uses
    /// ([`AnalyzeVisitor::self_type_entries()`]). A type that the graph may choose and whose bounds
    /// are unstable (`env_unstable_bound()`: `W2<T, Rel<U>>` with `T: Fn(Rel2<u8>)`, where `Rel<U>`
    /// is an entry anyway) would be a cycle, so the graph does not choose it: its parts are the
    /// entries, whose bounds the check looks at. A type that must be an entry is an error.
    fn block_unstable_self_types(&mut self, start: usize, end: usize) {
        fn block(visitor: &AnalyzeVisitor, effects: &mut [Effect]) {
            for effect in effects {
                let pending = match effect {
                    Effect::Type(pending) => pending,
                    _ => continue,
                };
                let candidate = pending.in_graph
                    && !pending.no_entry
                    && !pending.is_must_intern
                    && !is_whole_reference(&pending.ty);
                if candidate
                    && visitor
                        .env_unstable_bound(&pending.context, false)
                        .is_some()
                {
                    pending.no_entry = true;
                }
                block(visitor, &mut pending.components);
            }
        }
        // The bounds of a type are those of a type in a predicate.
        let mut effects = std::mem::take(&mut self.frame.effects);
        self.env_depth += 1;
        block(self, &mut effects[start..end]);
        self.env_depth -= 1;
        self.frame.effects = effects;
    }

    /// Called after the self type of an impl with `generics` is visited, with `start` the first
    /// effect it made.
    ///
    /// The Repeater's impl of an entry whose context depends on `Self` (a bound of the generics
    /// of the impl names it) is for the self type, so the copy writes `<Self as Repeater<..>>::Type`
    /// with the self type, as the Referrer has it, for `Self`. An entry in the self type has none:
    /// it is the one that is being written (`T: AsRef<Option<Self>>` in `impl<T> S<T>`: `S<T>` is
    /// its own `Self`, and a path that stands for itself is a cycle, E0391). Replacing the bound
    /// of the context by one for the self type does not help, as the bound would then name a
    /// leaked type that is the entry itself (E0277, see [Bounds that name leaked
    /// types](crate#bounds-that-name-leaked-types)), and the parts of the self type depend on
    /// `Self` as it does.
    ///
    /// So a type that must be an entry is an error, and one that may be is not. An optional entry
    /// is copied as written, which the Referrer cannot do if it replaces them (see
    /// [`AnalyzeVisitor::env_cycle()`]). Only a bound that `generics` declare counts: a bound of
    /// an outer scope that names its own `Self` is written for the type that scope knows.
    fn cut_self_cycles(&mut self, start: usize, generics: &Generics) {
        let mut not_optional = Vec::new();
        cut_conflicts(
            &mut self.frame.effects[start..],
            &|pending| self_type_conflict(pending, generics),
            &mut not_optional,
        );
        self.not_optional.extend(not_optional);
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
        let mut module: &ModuleNames = &self.modules[current].names;
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
            names: Rc::new(declared_names(items)),
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
        written: bool,
        f: impl FnOnce(&mut Self),
    ) {
        let allow_self = self.allow_self;
        if self_ty.is_none() || self.item_depth > 0 {
            self.allow_self = true;
        }
        let declares = self_ty.is_some();
        let outer = std::mem::replace(&mut self.self_ty, self_ty);
        let outer_written = std::mem::replace(&mut self.self_written, written);
        let outer_depth = self.self_depth;
        self.with_item_generics(generics, |visitor| {
            visitor.self_depth = if declares { visitor.scopes.len() } else { 0 };
            f(visitor)
        });
        self.self_ty = outer;
        self.self_written = outer_written;
        self.self_depth = outer_depth;
        self.allow_self = allow_self;
    }

    /// Visits the attributes and visibility only: the body or value is not analyzed.
    fn visit_head(&mut self, attrs: &[Attribute], vis: &Visibility) {
        for attr in attrs {
            self.visit_attribute(attr);
        }
        self.visit_visibility(vis);
    }

    /// Runs `f`, which visits `sig`. `item`: it is that of a function item, whose generics are the
    /// item's, and not those of a method. `compared`: rustc compares it with a signature that the
    /// copy cannot change, so the lifetimes that the copy would leave unconstrained are an error
    /// (E0195, see [Method lifetimes](crate#method-lifetimes)). That is a method of a trait, which
    /// the impls of the original are checked against, and one of an impl of a trait, which is
    /// checked against the trait. A function item and a method of an inherent impl have no such
    /// signature, so the copy may have the lifetime early-bound where the original has it
    /// late-bound (F22).
    fn with_signature(
        &mut self,
        sig: &Signature,
        item: bool,
        compared: bool,
        f: impl FnOnce(&mut Self),
    ) {
        self.with_risky(sig, self.scopes.len(), compared, |visitor| {
            // The generics are in scope while the lifetimes are checked, which needs them.
            let walk = |visitor: &mut Self| {
                visitor.with_arguments_of(sig, f);
                if compared {
                    visitor.check_leaked_lifetimes(sig);
                }
            };
            if item {
                visitor.with_item_generics(&sig.generics, walk)
            } else {
                visitor.with_local_generics(&sig.generics, walk)
            }
        });
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
        let outer = std::mem::replace(&mut self.signature, Elision::of_signature(sig));
        f(self);
        self.signature = outer;
    }

    fn with_risky(
        &mut self,
        sig: &Signature,
        at: usize,
        compared: bool,
        f: impl FnOnce(&mut Self),
    ) {
        if compared {
            let risky = self.unconstrained_lifetimes(sig);
            self.signatures.push(SignatureWalk {
                at,
                declared: sig
                    .generics
                    .lifetimes()
                    .map(|param| param.lifetime.clone())
                    .collect(),
                risky,
                leaked: HashMap::new(),
            });
        }
        f(self);
        if compared {
            self.signatures.pop();
        }
    }

    fn is_unconstrained(&self, lifetime: &Lifetime, scope: usize) -> bool {
        self.signatures
            .iter()
            .any(|walk| walk.at == scope && walk.risky.contains(lifetime))
    }

    /// Notes that `ty` is leaked whole, if it names a lifetime of the signature being visited:
    /// `check_leaked_lifetimes()` needs it. `inner` and `depth` are those of its decision.
    fn record_leaked(&mut self, ty: &Type, depth: usize, inner: &Frame) {
        // The probe of a binder walks its types again for real.
        if self.probe.is_some() {
            return;
        }
        let walk = match self.signatures.last() {
            Some(walk) if !walk.declared.is_empty() => walk,
            _ => return,
        };
        let mut finder = OutputLifetimes {
            declared: walk.declared.clone(),
            shadowed: Vec::new(),
            found: Vec::new(),
        };
        finder.visit_type(ty);
        let named: Vec<Lifetime> = finder
            .found
            .into_iter()
            .filter(|lifetime| self.scope_of(&lifetime.to_string()) == Some(walk.at))
            .collect();
        if named.is_empty() {
            return;
        }
        let conflict = self.causes(inner.must_start, inner.impossible_start, depth);
        if let Some(walk) = self.signatures.last_mut() {
            walk.leaked
                .insert(ty as *const Type as usize, (named, conflict));
        }
    }

    /// Reports the lifetimes of the signature just visited that the copy would leave unconstrained
    /// because only types leaked whole hold them (E0195, see [Method
    /// lifetimes](crate#method-lifetimes)). The return type names the lifetime and no bound does,
    /// so it is late-bound where some input constrains it as written (`Wrap<'x>` does), and it
    /// is early-bound in the copy where every such input is leaked whole: `Wrap<'x>` becomes a
    /// projection, which does not constrain it. An input that keeps it, such as `&'x u8`, is no
    /// type leaked whole, nor is one inside it.
    fn check_leaked_lifetimes(&mut self, sig: &Signature) {
        let leaked = match self.signatures.last_mut() {
            Some(walk) if !walk.leaked.is_empty() => std::mem::take(&mut walk.leaked),
            _ => return,
        };
        let (output, bounded) = output_and_bounded(sig);
        let mut conflicts = Vec::new();
        for param in sig.generics.lifetimes() {
            let lifetime = &param.lifetime;
            if !output.0.contains(lifetime) || bounded.0.contains(lifetime) {
                continue;
            }
            // The occurrences that constrain the lifetime as written. An external type that the
            // Leaker does not allow stays as written unless the Referrer replaces it, so it counts.
            let mut used = OutputUse {
                lifetime: lifetime.clone(),
                anchors: Vec::new(),
                chain: Vec::new(),
                careful: false,
            };
            for input in &sig.inputs {
                match input {
                    FnArg::Receiver(receiver) => {
                        if receiver.lifetime() == Some(lifetime) {
                            used.anchors.push(Vec::new());
                        }
                        self.scan_anchors(&receiver.ty, &mut used)
                    }
                    FnArg::Typed(arg) => self.scan_anchors(&arg.ty, &mut used),
                }
            }
            // The original constrains it, and in the copy only types leaked whole hold it: the
            // first on the way to each occurrence is what to report.
            let mut culprits: Vec<usize> = Vec::new();
            let first_leaked =
                |anchor: &[usize]| anchor.iter().copied().find(|ty| leaked.contains_key(ty));
            if !used.anchors.is_empty() && used.anchors.iter().all(|a| first_leaked(a).is_some()) {
                for anchor in &used.anchors {
                    if let Some(ty) = first_leaked(anchor) {
                        if !culprits.contains(&ty) {
                            culprits.push(ty);
                        }
                    }
                }
            }
            for ty in culprits {
                let (named, conflict) = &leaked[&ty];
                if named.contains(lifetime) {
                    let mut conflict = conflict.clone();
                    conflict.impossible.insert(0, quote!(#lifetime));
                    conflict.note = Some(unconstrained_note(lifetime, false));
                    conflicts.push(conflict);
                }
            }
        }
        self.frame
            .effects
            .extend(conflicts.into_iter().map(Effect::Error));
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
        let (output, bounded) = output_and_bounded(sig);
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
        let (env_errors, not_optional) = (self.env_errors.len(), self.not_optional.len());
        self.probe = Some(HashSet::new());
        self.with_scope(Scope::binder(lifetimes), |v| f(v));
        self.frame = frame;
        self.musts.truncate(musts);
        self.truncate_impossibles(impossibles);
        self.self_errors.truncate(self_errors);
        self.env_errors.truncate(env_errors);
        self.not_optional.truncate(not_optional);
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
                    careful: true,
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
            Type::Reference(reference)
                if used.careful && self.is_external_object(&reference.elem) => {}
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
            Type::Array(array) if used.careful && self.names_external_crate_in_expr(&array.len) => {
            }
            Type::Array(array) => self.scan_anchors(&array.elem, used),
            Type::Tuple(tuple) => {
                for elem in &tuple.elems {
                    self.scan_anchors(elem, used);
                }
            }
            Type::Path(TypePath { qself: None, path })
                if !self.is_projection(path)
                    && !(used.careful
                        && (self.allowed.is_soft_path(path)
                            || self.holds_external_object(path))) =>
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
            Type::TraitObject(object) if !(used.careful && self.names_external_crate(object)) => {
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
        Some(qself) => match unwrap_type_group(&qself.ty) {
            Type::Path(TypePath { qself: None, path }) => path.get_ident().map(ident_name),
            _ => None,
        },
        None if path.leading_colon.is_none() => {
            path.segments.first().map(|seg| ident_name(&seg.ident))
        }
        None => None,
    };
    match head {
        Some(head) if head == "Self" => {
            matches!(context.self_ty(), Some(SelfRequirement::Bound(_)))
        }
        Some(head) => context.generics().params.iter().any(
            |param| matches!(param, GenericParam::Type(param) if ident_name(&param.ident) == head),
        ),
        None => false,
    }
}

/// Returns the types, with their contexts, that are optional entries in `effects`.
fn optional_keys(effects: &[Effect]) -> Vec<(Context, String)> {
    let mut keys = Vec::new();
    for effect in effects {
        if let Effect::Type(pending) = effect {
            if pending.optional {
                let ty = &pending.ty;
                keys.push((pending.context.clone(), quote!(#ty).to_string()));
            }
            keys.extend(optional_keys(&pending.components));
        }
    }
    keys
}

/// Returns `true` if `ty` is `Self`.
fn is_bare_self(ty: &Type) -> bool {
    matches!(
        unwrap_type_group(ty),
        Type::Path(TypePath { qself: None, path }) if path.is_ident("Self")
    )
}

/// Returns `true` if `ty` is `Self::A` or `<Self as Tr>::A`: an associated type of `Self`.
fn is_self_projection(ty: &Type) -> bool {
    match unwrap_type_group(ty) {
        Type::Path(TypePath { qself: None, path }) => {
            path.leading_colon.is_none()
                && path.segments.len() > 1
                && matches!(path.segments.first(), Some(seg) if seg.ident == "Self")
        }
        Type::Path(TypePath {
            qself: Some(qself), ..
        }) => matches!(
            unwrap_type_group(&qself.ty),
            Type::Path(TypePath { qself: None, path }) if path.is_ident("Self")
        ),
        _ => false,
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

    /// The signature of a method: its generics are local, and it is checked for E0195 unless the
    /// method is that of an inherent impl. A function item declares its own generics instead
    /// (`visit_item_fn()`).
    fn visit_signature(&mut self, i: &Signature) {
        let compared = !self.inherent_impl;
        self.with_signature(i, false, compared, |visitor| {
            syn::visit::visit_signature(visitor, i)
        })
    }

    fn visit_trait_item_type(&mut self, i: &TraitItemType) {
        self.with_local_generics(&i.generics, |visitor| {
            syn::visit::visit_trait_item_type(visitor, i)
        });
    }

    fn visit_foreign_item_fn(&mut self, i: &ForeignItemFn) {
        // Not a method: its signature declares nothing here, as before `visit_signature()`.
        self.visit_head(&i.attrs, &i.vis);
        self.with_arguments_of(&i.sig, |visitor| {
            syn::visit::visit_signature(visitor, &i.sig)
        });
    }

    fn visit_fn_arg(&mut self, i: &FnArg) {
        self.elisions.push(self.signature.clone());
        let outer = std::mem::replace(&mut self.in_fn_arg, true);
        syn::visit::visit_fn_arg(self, i);
        self.in_fn_arg = outer;
        self.elisions.pop();
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
        self.with_item(&i.generics, Some(self_ty), false, |visitor| {
            syn::visit::visit_item_struct(visitor, i)
        });
    }

    fn visit_item_enum(&mut self, i: &ItemEnum) {
        let self_ty = SelfRequirement::of_item(&i.ident, &i.generics);
        self.with_item(&i.generics, Some(self_ty), false, |visitor| {
            syn::visit::visit_item_enum(visitor, i)
        });
    }

    fn visit_item_union(&mut self, i: &ItemUnion) {
        let self_ty = SelfRequirement::of_item(&i.ident, &i.generics);
        self.with_item(&i.generics, Some(self_ty), false, |visitor| {
            syn::visit::visit_item_union(visitor, i)
        });
    }

    fn visit_item_trait(&mut self, i: &ItemTrait) {
        let self_ty = SelfRequirement::of_trait(&i.ident, &i.generics);
        self.with_item(&i.generics, Some(self_ty), false, |visitor| {
            syn::visit::visit_item_trait(visitor, i)
        });
    }

    fn visit_item_type(&mut self, i: &ItemType) {
        self.with_item_generics(&i.generics, |visitor| {
            syn::visit::visit_item_type(visitor, i)
        });
    }

    fn visit_item_fn(&mut self, i: &ItemFn) {
        self.with_signature(&i.sig, true, false, |visitor| {
            visitor.visit_head(&i.attrs, &i.vis);
            syn::visit::visit_signature(visitor, &i.sig);
        });
    }

    fn visit_item_mod(&mut self, i: &ItemMod) {
        // A module has no generics and no `Self`; its items are nested items with local generics.
        self.with_item(&Generics::default(), None, false, |visitor| {
            visitor.with_module(i, |visitor| syn::visit::visit_item_mod(visitor, i))
        });
    }

    fn visit_item_impl(&mut self, i: &ItemImpl) {
        let self_ty = SelfRequirement::Type((*i.self_ty).clone());
        self.with_item(&i.generics, Some(self_ty), true, |visitor| {
            // The trait is checked like a trait bound, but its arguments are analyzed in any case.
            if let Some((_, path, _)) = &i.trait_ {
                visitor.check_bound_path(path, quote!(#path));
            }
            visitor.walk_item_impl(i)
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
                        self.push_elided(quote!(#i));
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
                // `x: impl Tr<Wrap<T>>` is `x: X` with `X: Tr<Wrap<T>>`: the bounds of the
                // `impl Trait` of an argument are predicates of the item, as those of a generic
                // are. Not those of a return type, which rustc proves where the body is.
                let in_env = usize::from(self.in_fn_arg && matches!(i, Type::ImplTrait(_)));
                self.env_depth += in_env;
                syn::visit::visit_type(self, i);
                self.env_depth -= in_env;
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
        let elision = Elision::of(
            self.type_depth,
            true,
            i.inputs.iter().map(|arg| &arg.ty),
            &i.output,
        );
        let mut output = output_lifetimes(i.lifetimes.as_ref(), |v| v.visit_return_type(&i.output));
        // An elided output has the only lifetime of the inputs, which the binder may declare.
        let elided = elision.as_ref().and_then(|elision| elision.kept.clone());
        output.extend(
            elided.filter(|kept| declares(i.lifetimes.as_ref(), kept) && !output.contains(kept)),
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
                v.elisions.push(elision.clone());
                for arg in &i.inputs {
                    v.visit_bare_fn_arg(arg);
                }
                v.elisions.pop();
                if let Some(variadic) = &i.variadic {
                    v.visit_bare_variadic(variadic);
                }
                v.in_output(&i.output);
            })
        })
    }

    fn visit_parenthesized_generic_arguments(&mut self, i: &ParenthesizedGenericArguments) {
        let elision = Elision::of(self.type_depth, true, &i.inputs, &i.output);
        self.with_bound_elided(|visitor| {
            visitor.elisions.push(elision);
            for input in &i.inputs {
                visitor.visit_type(input);
            }
            visitor.elisions.pop();
            visitor.in_output(&i.output);
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

    fn visit_type_param_bound(&mut self, i: &TypeParamBound) {
        // A bound that is not in a type is a predicate of the item (`env_depth`).
        let in_env = usize::from(self.type_depth == 0);
        self.env_depth += in_env;
        syn::visit::visit_type_param_bound(self, i);
        self.env_depth -= in_env;
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
        self.env_depth += 1;
        self.with_bound_lifetimes(i.lifetimes.as_ref(), groups, |v| {
            syn::visit::visit_predicate_type(v, i)
        });
        self.env_depth -= 1;
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
        // `Self::A` where `Self` is a type: a type that names it is not leaked whole.
        if i.qself.is_none()
            && i.path.segments.len() > 1
            && self_segment(&i.path).is_some()
            && matches!(self.self_ty, Some(SelfRequirement::Type(_)))
        {
            let path = &i.path;
            let cause = quote!(#path);
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
            self.push_elided(quote!(#i));
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

/// A `Self::A` that the Repeater's impl for the type `Self` cannot write (see
/// [`AnalyzeVisitor::associated_self_in_bounds()`]).
struct AssocSelf {
    cause: TokenStream,
    self_ty: Type,
}

impl AssocSelf {
    fn new(cause: TokenStream, self_ty: &Type) -> Self {
        Self {
            cause,
            self_ty: self_ty.clone(),
        }
    }

    /// The conflict of the type `ty` leaked whole, with the generics that name it.
    fn of(self, ty: &Type) -> CheckResult {
        CheckResult {
            must: vec![quote!(#ty)],
            impossible: vec![self.cause.clone()],
            is_own: true,
            optional: false,
            note: Some(assoc_self_note(&self.cause, &self.self_ty)),
            headline: Some(assoc_self_headline(&self.cause)),
        }
    }
}

fn assoc_self_headline(cause: &TokenStream) -> String {
    format!("`{}` cannot be written in the impl of the Repeater", cause)
}

fn assoc_self_note(cause: &TokenStream, self_ty: &Type) -> String {
    format!(
        "`{}` is an associated item of `Self`, which is the type `{}` here, and the impl of the \
         Repeater for that type has no trait to take it from: name the trait with its absolute \
         path (`<Self as ::path::Trait>::..`)",
        cause,
        quote!(#self_ty)
    )
}

/// Cuts the entries of the self type of an impl for which `conflict` returns the conflict (see
/// [`AnalyzeVisitor::cut_self_cycles()`]). A type that must be an entry is an error, and so is a
/// reference with a lifetime that has a part that must be one, as its parts cannot stand for it
/// (`is_whole_reference()`). Another type cannot be an entry, and an optional one is not (the
/// optional entries are added to `not_optional`).
fn cut_conflicts(
    effects: &mut [Effect],
    conflict: &dyn Fn(&PendingType) -> Option<CheckResult>,
    not_optional: &mut Vec<(Context, String)>,
) {
    for effect in effects.iter_mut() {
        let pending = match &mut *effect {
            Effect::Type(pending) => pending,
            _ => continue,
        };
        if let Some(error) = conflict(pending) {
            let whole = is_whole_reference(&pending.ty);
            if pending.in_graph && (pending.is_must_intern || whole) {
                *effect = Effect::Error(error);
                continue;
            }
            pending.no_entry |= pending.in_graph;
            pending.no_optional_entry |= pending.has_optional;
            if pending.has_optional && whole {
                // The optional entries under a reference cannot stand for it either.
                not_optional.extend(optional_keys(&pending.components));
            }
            if pending.optional {
                let ty = &pending.ty;
                not_optional.push((pending.context.clone(), quote!(#ty).to_string()));
                pending.optional = false;
            }
        }
        cut_conflicts(&mut pending.components, conflict, not_optional);
    }
}

/// Adds the contexts of the types in `effects` that may be entries of the copy to `found` (see
/// [`AnalyzeVisitor::self_type_entries()`]). A type that must be an entry has no parts that are
/// entries. Another one may be, with those of its parts that may, unless its context depends on
/// the `Self` that the generics `declared` bound (`self_type_conflict()`): it is cut then, or
/// `blocked` says that it is not one the graph chooses.
fn entry_contexts(
    effects: &[Effect],
    declared: Option<&Generics>,
    blocked: &dyn Fn(&Context) -> bool,
    optional: bool,
    found: &mut Vec<Context>,
) {
    let mut push = |context: &Context| {
        if !found.contains(context) {
            found.push(context.clone());
        }
    };
    for effect in effects {
        let pending = match effect {
            Effect::Type(pending) => pending,
            _ => continue,
        };
        if pending.is_must_intern || (pending.in_graph && is_whole_reference(&pending.ty)) {
            push(&pending.context);
            continue;
        }
        let cut = declared.map_or(false, |declared| {
            self_type_conflict(pending, declared).is_some()
        });
        let entry = pending.in_graph && !pending.no_entry && !blocked(&pending.context);
        let optional_entry = optional && pending.optional && !pending.no_optional_entry;
        if (entry || optional_entry) && !cut {
            push(&pending.context);
        }
        let mut parts = Vec::new();
        entry_contexts(&pending.components, declared, blocked, optional, &mut parts);
        for context in &parts {
            push(context);
        }
    }
}

/// Returns the first mention of `Self` in the bounds that the generics `declared` of an impl give
/// to `context`: the bounds of its parameters and its where predicates.
fn declared_self_mention(declared: &Generics, context: &Context) -> Option<TokenStream> {
    let names: HashSet<String> = declared.params.iter().map(generic_param_name).collect();
    let written: HashSet<String> = declared
        .where_clause
        .iter()
        .flat_map(|where_clause| where_clause.predicates.iter())
        .map(|predicate| quote!(#predicate).to_string())
        .collect();
    let predicates: Vec<WherePredicate> = context
        .generics()
        .where_clause
        .iter()
        .flat_map(|where_clause| where_clause.predicates.iter())
        .filter(|predicate| written.contains(&quote!(#predicate).to_string()))
        .cloned()
        .collect();
    let own = Generics {
        params: context
            .generics()
            .params
            .iter()
            .filter(|param| names.contains(&generic_param_name(param)))
            .cloned()
            .collect(),
        where_clause: (!predicates.is_empty()).then(|| WhereClause {
            where_token: Default::default(),
            predicates: predicates.into_iter().collect(),
        }),
        ..Generics::default()
    };
    first_self_in_generics(&own).or_else(|| {
        // A written `Self: 'a` moved to `self_outlives`, which is encoded as `&'a Self`.
        if context.self_outlives().is_empty() {
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
    let self_ty = match context.self_ty() {
        Some(SelfRequirement::Type(self_ty)) => self_ty,
        _ => return None,
    };
    let cause = declared_self_mention(declared, context)?;
    let shown = quote!(#ty).to_string();
    let what = if quote!(#self_ty).to_string() == shown {
        format!("`Self` is `{}`, the self type of the impl", shown)
    } else {
        format!(
            "`Self` is the self type `{}` of the impl, which contains `{}`",
            quote!(#self_ty),
            shown
        )
    };
    Some(CheckResult {
        must: vec![quote!(#ty)],
        impossible: vec![cause],
        is_own: true,
        optional: false,
        note: Some(format!(
            "{}. The copy writes `{}` as a path through the Repeater, whose impl is for `Self`, \
             so the Referrer has no type to write for `Self`: it is the one being replaced. Write \
             the self type with an absolute path, or the bounds of its generics without `Self`",
            what, shown
        )),
        headline: Some(format!(
            "`{}` is in the self type of the impl, and a bound of its generics names `Self`",
            shown
        )),
    })
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

/// `object`: the type is a trait object that defaults to the lifetime, else it names it.
fn unconstrained_note(lifetime: &Lifetime, object: bool) -> String {
    format!(
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

/// Where the elided lifetime of the output comes from, as the notes say it.
fn elision_source(elision: &Elision) -> &'static str {
    if elision.hidden {
        "the only lifetime of the inputs, which a path hides (the Leaker does not see it)"
    } else {
        "the only lifetime of the inputs"
    }
}

/// What to do to the signature where the copy cannot keep its elision: the lifetimes are named.
fn elision_advice(elision: &Elision) -> &'static str {
    if elision.hidden {
        ": write the lifetime that the path hides (`Ref<'x, u8>`), and name it in the output"
    } else {
        ": name the lifetime of the inputs and of the output"
    }
}

/// `owned`: the type holding the signature would be leaked instead.
fn elision_note(added: &[Lifetime], elision: &Elision, owned: bool) -> String {
    let added: Vec<String> = added
        .iter()
        .map(|lifetime| format!("`{}`", lifetime))
        .collect();
    format!(
        "the output `{}` has an elided lifetime, which elision takes from {}, but a leaked type in \
         the inputs has {} among its generics, and the copy names {} in the arguments of its \
         projection, so the inputs have more than one lifetime (E0106){}",
        elision.output,
        elision_source(elision),
        added.join(", "),
        if added.len() == 1 { "it" } else { "them" },
        if owned {
            ", so a type using them cannot be leaked alone, and the type holding the signature \
             would be leaked as a whole instead"
        } else {
            elision_advice(elision)
        }
    )
}

/// The note of a signature whose elision the copy breaks by writing a type with `added` lifetimes
/// for `Self`.
fn self_elision_note(added: &[Lifetime], elision: &Elision, owned: bool) -> String {
    let added: Vec<String> = added
        .iter()
        .map(|lifetime| format!("`{}`", lifetime))
        .collect();
    format!(
        "the output `{}` has an elided lifetime, which elision takes from {}, but `Self` in the \
         inputs is a type that has {} among its generics, and the copy writes that type for `Self` \
         (`Visitor::with_self()`), so the inputs have more than one lifetime (E0106){}",
        elision.output,
        elision_source(elision),
        added.join(", "),
        if owned {
            ", so the type holding the signature would be leaked as a whole instead"
        } else {
            elision_advice(elision)
        }
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
            output.extend(kept.filter(|kept| declares(lifetimes, kept) && !output.contains(kept)));
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

/// Returns `true` if `ty` is `Self` or `Self::A`, written as a path: not `<Self as Tr>::A`, whose
/// `Self` is a type of its own.
fn is_self_path(ty: &Type) -> bool {
    match ty {
        Type::Path(TypePath { qself: None, path }) => {
            matches!(self_segment(path), Some(seg) if seg.arguments.is_none())
        }
        _ => false,
    }
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

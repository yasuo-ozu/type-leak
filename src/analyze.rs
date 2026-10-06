//! The analysis of items and types by the [`Leaker`].
//!
//! [`AnalyzeVisitor`] visits each type once, in post-order: the causes found in a type (see
//! [`CheckResult`]) decide whether it is interned as a whole, decomposed into its components, or
//! reported as an error, and the effects of its components (the types to intern, the errors and
//! the assumed primitives) are kept or discarded accordingly.

use crate::context::{
    binder_lifetimes, Assumed, Context, Contexts, Enclosing, Scope, SelfRequirement,
};
use crate::encode::is_const_param_ty;
use crate::graph::{NodeKey, TypeGraph};
use crate::leaker::Leaker;
use crate::util::{
    first_self_in_generics, first_self_in_type, generic_param_name, has_lifetime_bound, ident_name,
    is_primitive_name, token_names, trait_object_of, unwrap_type_group,
};
use proc_macro2::TokenStream;
use std::collections::{HashMap, HashSet};
use syn::visit::Visit;
use syn::*;
use template_quote::quote;

/// The result of [`Leaker::check()`]: the causes which make a type need to be interned, and those
/// which make it impossible to intern.
///
/// | `must` | `impossible` | the type |
/// |---|---|---|
/// | empty | empty | does not depend on the type context, and can be copied as it is |
/// | non-empty, [`is_own`](CheckResult::is_own) | empty | must be interned (e.g. `MyType`) |
/// | non-empty, not `is_own` | empty | must be interned, or its components (e.g. `::path::Wrap<MyType>`) |
/// | empty | non-empty | cannot be interned, but does not need to be (e.g. `impl ::path::Trait`) |
/// | non-empty | non-empty | must be interned but cannot be: a conflict (e.g. `MyType<'_>`, or `::path::Wrap<MyType, '_>`, whose components can be interned instead unless `is_own`) |
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct CheckResult {
    /// The causes which depend on the type context, so they must be interned (e.g. `A` and `B` in
    /// `::path::Wrap<A, B>`). The outermost ones come first (in the order of appearance among
    /// those at the same depth).
    pub must: Vec<TokenStream>,
    /// The causes which cannot be interned (e.g. an elided lifetime or `impl Trait`), in the
    /// order of appearance.
    pub impossible: Vec<TokenStream>,
    /// `true` if the first of `must` is the type's own path (e.g. `MyType` in `MyType<'_>`), so
    /// the type must be interned as a whole: it cannot be decomposed, i.e. copied while interning
    /// only its components. `false` if `must` is empty, or all of `must` are in components (e.g.
    /// `MyType` in `::path::Wrap<MyType, '_>`), which can be interned alone.
    ///
    /// A bare identifier written as a generic argument (e.g. `SIZE` in `::path::Wrap<SIZE>`) may
    /// be a const, which cannot be interned alone (syn parses it as a type), so it is treated as
    /// the cause of the type containing it, which is own, if that can be interned as a whole.
    /// Otherwise (e.g. `::path::Wrap<'_, SIZE>`) it is a component, interned alone as a type.
    pub is_own: bool,
    /// An explanation of the conflict, if it is not clear from the causes (e.g. why `Self` cannot
    /// be leaked). It is reported first.
    pub note: Option<String>,
}

/// A type to be added to the [`TypeGraph`]s, with its components.
///
/// The types are added to two graphs: that of the required entries (the types which depend on the
/// type context, see [`CheckResult::must`]), and that of the optional ones (the types whose own
/// path is an absolute one naming an external crate, see [`Leaker::is_soft_path()`]), which the
/// Referrer may or may not replace (see [`Referrer::must()`](crate::Referrer::must())).
pub(crate) struct PendingType {
    ty: Type,
    is_must_intern: bool,
    /// The type or a component of it is a cause (see [`CheckResult::must`]). Otherwise the type is
    /// never interned, so it is not added to the graph of the required entries.
    has_must: bool,
    /// The type is a candidate of the graph of the required entries (if it has a cause). It is not
    /// if it is only an optional type (an optional entry depending on local generics, which is
    /// not decomposed into a required one).
    in_graph: bool,
    /// The type is an optional entry if the Referrer replaces it: its own path names an external
    /// crate, and it can be leaked as a whole (it has no cause of its own, or it would be a
    /// required entry), so it is a red node of the graph of the optional entries.
    optional: bool,
    /// The type or a component of it is optional (see `optional`).
    has_optional: bool,
    /// The context of the type.
    context: Context,
    /// The type is a bare identifier written as a generic argument, which may be a const (syn
    /// parses both as a type). Its parent is interned as a whole if possible, otherwise it is
    /// leaked as a type.
    ambiguous: bool,
    /// The effects of its components, applied if the type is added.
    components: Vec<Effect>,
}

/// An effect of the analysis, applied (in order) when the analysis is finished.
pub(crate) enum Effect {
    /// A type to add (not as a component of another type).
    Type(Box<PendingType>),
    /// A relative path which cannot be interned.
    Error(CheckResult),
    /// The name of a primitive type copied as it is.
    Primitive(Ident),
}

/// What is found in a type (or outside any type).
///
/// The causes are kept by the visitor, in the order of appearance (see
/// [`AnalyzeVisitor::musts`]), so a frame only records where its own start.
struct Frame {
    /// The index in [`AnalyzeVisitor::musts`] of the first cause found in the type.
    must_start: usize,
    /// The index in [`AnalyzeVisitor::impossibles`] of the first cause found in the type.
    impossible_start: usize,
    /// The minimum depth of the causes found in the type (`usize::MAX` if none).
    min_depth: usize,
    /// The minimum index in the visitor's `scopes` of the local generics (not those of the Leaker,
    /// which are the scope 0) mentioned in the type (`usize::MAX` if none).
    local_scope: usize,
    /// The type mentions a generic of the Leaker.
    mentions_root: bool,
    /// The type mentions `Self`.
    mentions_self: bool,
    /// The type mentions an item declared in an analyzed module (see
    /// [`AnalyzeVisitor::is_module_path()`]).
    module_item: bool,
    /// The minimum depth of the types whose own path is an absolute one naming an external crate
    /// (see [`Leaker::is_soft_path()`]) found in the type (`usize::MAX` if none). They are not
    /// causes: the type is an optional entry if one is its own (see [`PendingType::optional`]).
    soft_depth: usize,
    /// A cause found in the type is a relative path resolved in the module where the type is
    /// written (not a generic, `Self`, an absolute path, `crate::`, or a path portable with the
    /// module), so the type means something else in another module.
    relative: bool,
    /// A type in the type was not interned alone because it mentions a lifetime of a `for<..>`
    /// binder which its output mentions (see [`Binder`]): the depth of the type holding the
    /// binder, which is interned as a whole instead, and the lifetime. It is kept until that type
    /// is decided, which reports it in the note of the conflict (see
    /// [`AnalyzeVisitor::own_context()`]).
    binder: Option<(usize, Lifetime)>,
    /// The type is a trait object without a lifetime bound, which defaults to this lifetime of a
    /// method, which the copy would leave unconstrained (see [`AnalyzeVisitor::unconstrained_lifetimes()`]), so
    /// the path or the reference holding it cannot be interned as a whole.
    unconstrained: Option<Lifetime>,
    /// The effects, which are applied if the type is decomposed.
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

/// A `for<..>` binder being visited, with the lifetimes which a type cannot be interned alone with:
/// those which its output mentions (the return type of a function pointer, or of the
/// parenthesized arguments of a trait (`Fn(A) -> R`), or an associated type binding
/// (`Iterator<Item = R>`) in the bounds of the binder) and no input keeps as written. The copy
/// `for<'x> Fn(<S as Repeater<&'x (), ..>>::Type) -> &'x u8` is rejected by rustc: a lifetime which
/// occurs only in a projection is not constrained by the inputs (E0582 for the binding of an
/// associated type, E0581 for the return type of a function pointer). If an input keeps `'x` where
/// it is written (an *anchor*, see [`AnalyzeVisitor::unanchored()`]) it is constrained, whatever the
/// types leaked alone are, and it is not among them.
struct Binder {
    /// The depth of the type holding the binder (a function pointer, a trait object or an
    /// `impl Trait`), which is interned as a whole instead (the lifetime is bound inside it), or 0
    /// if the binder is in a bound or a where predicate, where there is none.
    owner: usize,
    /// The lifetimes of the binder which its output mentions and no input anchors.
    output: Vec<Lifetime>,
}

/// A lifetime of a `for<..>` binder which the output mentions, with the occurrences of it in the
/// inputs which constrain it (see [`AnalyzeVisitor::unanchored()`]).
struct OutputUse {
    lifetime: Lifetime,
    /// Each occurrence in a constraining position of the inputs, as the types from the input
    /// holding it down to the one directly holding it (as the addresses of the syntax, the
    /// outermost first; none if it is an argument of the bound itself). It stays in the copy as
    /// written unless one of the types is interned: an anchor.
    anchors: Vec<Vec<usize>>,
}

/// The output and the inputs of a function pointer, or of one bound (`Fn(A, B) -> R`,
/// `Iterator<Item = R>`) of a trait object, an `impl Trait`, a predicate or a generic parameter,
/// which rustc checks together (see [`Binder`]).
struct BinderGroup<'a> {
    /// The lifetimes of the binder which the output mentions.
    output: Vec<Lifetime>,
    inputs: Vec<BinderInput<'a>>,
}

/// An input of a [`BinderGroup`].
enum BinderInput<'a> {
    Type(&'a Type),
    Lifetime(&'a Lifetime),
}

/// The result of applying the effects of [`AnalyzeVisitor`].
#[derive(Default)]
pub(crate) struct Analysis {
    /// The graph of the required entries.
    pub(crate) types: TypeGraph,
    /// The graph of the optional entries, in the contexts of `Leaker::optional_contexts`.
    pub(crate) optional: TypeGraph,
    /// The relative paths which cannot be interned.
    pub(crate) errors: Vec<CheckResult>,
    /// The names of the primitive types copied as they are (see [`Leaker::assertions()`]).
    pub(crate) primitives: Vec<Ident>,
}

/// Analyzes items (see [`Leaker::intern_with()`]) and types (see [`Leaker::check()`]).
pub(crate) struct AnalyzeVisitor<'a> {
    leaker: &'a Leaker,
    /// The generics in scope: those of the Leaker first (the root, which is always there), and
    /// then those declared inside the Leaker's item (e.g. by trait methods, GATs, nested items or
    /// `for<..>` binders), the innermost last. A type depending on them is interned with its
    /// [`Context`], which the Repeater's implementation declares.
    scopes: Vec<Scope>,
    /// The `for<..>` binders being visited whose output mentions some of their lifetimes, the
    /// innermost last (see [`Binder`]).
    binders: Vec<Binder>,
    /// Set while a binder is being probed (see [`AnalyzeVisitor::unanchored()`]): the addresses of
    /// the types found to be interned as a whole.
    probe: Option<HashSet<usize>>,
    /// The type parameters of the Leaker's generics with a default.
    root_defaulted: Vec<Ident>,
    /// What `Self` means where the syntax being visited is (see [`SelfRequirement`]), if known.
    self_ty: Option<SelfRequirement>,
    /// The type being analyzed is the default of a parameter (see [`Leaker::intern_with()`]),
    /// which cannot depend on `Self`.
    in_default: bool,
    /// The names declared directly in each module being analyzed (see [`Leaker::from_module()`]),
    /// the innermost last. Paths to them are copied with the module, so they are portable.
    modules: Vec<ModuleNames>,
    /// The path of the module being visited, relative to the outermost module visited (see
    /// [`Context`]): the idents of the modules entered below it.
    module_path: Vec<Ident>,
    /// The causes which depend on the type context (see [`CheckResult::must`]), each with the
    /// depth of the type it is found in, in the order of appearance.
    musts: Vec<(usize, TokenStream)>,
    /// The causes which cannot be interned (see [`CheckResult::impossible`]), in the order of
    /// appearance.
    impossibles: Vec<TokenStream>,
    /// What is found in the type being visited (or outside any type).
    frame: Frame,
    /// The number of types enclosing the syntax being visited.
    type_depth: usize,
    /// The number of items enclosing the syntax being visited. The generics of the outermost item
    /// are the `root`.
    item_depth: usize,
    /// The type being visited is the referent of a reference, or a type argument of a path, which
    /// cannot be interned as a whole (its lifetime is elided, the path has several local
    /// lifetimes, or it is in a trait bound), so a trait object without a lifetime bound, which
    /// defaults to that lifetime, cannot be interned alone.
    elided_ref: bool,
    /// The type being visited is a type argument of a path whose only lifetime argument is this
    /// lifetime of a method, or the referent of a reference with it, which the copy would leave
    /// unconstrained if the path or the reference were interned as a whole (see
    /// [`AnalyzeVisitor::signatures`]): a trait object without a lifetime bound cannot be interned
    /// alone, like with `elided_ref`, and the conflict has a note of its own.
    unconstrained_ref: Option<Lifetime>,
    /// The type being visited is (directly) a bare identifier written as a generic argument (e.g.
    /// `SIZE` in `W<SIZE>`), which may be a const (see [`PendingType::ambiguous`]).
    ambiguous_arg: bool,
    /// The type being visited is the referent of a reference, or a type argument of a path, with a
    /// lifetime which is a generic (of the Leaker, or a local one), so a trait object defaults to
    /// that lifetime, and is not the same type when interned alone (it would default to
    /// `'static`). Only the trait object is affected (also inside parentheses and groups): what
    /// is inside it is not, and can be interned.
    inherits: Option<Inherits>,
    /// The functions whose signatures are being visited, each with the index in `scopes` of the
    /// scope of its generics and the lifetimes among them whose copy would be early-bound where
    /// the original is late-bound if a type using them were interned (see
    /// [`AnalyzeVisitor::unconstrained_lifetimes()`]), the innermost last.
    signatures: Vec<(usize, Vec<Lifetime>)>,
    /// The depth of the type owning the expression being visited in a type (an array length or a
    /// const argument), if any. The types in it are not interned on their own: the copy of the
    /// expression would have a projection (`<Self as Repeater<..>>::Type`) in an anonymous
    /// constant, which is not permitted if the generics (or `Self`) are in it.
    expr_owner: Option<usize>,
    /// The syntax being visited is a receiver (`self: Box<Self>`), whose type the Referrer cannot
    /// replace: rustc accepts only `Self` and the types dereferencing to it there, not a
    /// projection. So no type in it is an optional entry.
    in_receiver: bool,
    /// The lifetimes which the type being visited must outlive because of the types around it,
    /// and which its context carries (see [`Enclosing`]). The Referrer keeps track of the same
    /// lifetimes where it looks the types up, so that it finds the entry of the type. While the
    /// components of a type are visited, it is what they must outlive, and it is restored when
    /// they are done. It is empty outside any type (and in an expression of a type).
    enclosing: Enclosing,
}

/// What a trait object without a lifetime bound inherits its lifetime from (see
/// [`AnalyzeVisitor::inherits`]).
#[derive(Clone, Copy)]
enum Inherits {
    /// A reference `&'a dyn Trait` with a lifetime of the Leaker's generics, or a local one (of a
    /// method, an associated type, a nested item or a `for<..>` binder) which the copy does not
    /// leave unconstrained (see [`AnalyzeVisitor::unconstrained_lifetimes()`]), which is the type at the given
    /// depth. The default of the trait object is `'a` unless the trait has a lifetime bound of its
    /// own (`trait Trait: 'static`, which makes it `'static` whatever the reference is), which the
    /// Leaker cannot see for a relative trait: rustc computes it from the definition of the trait,
    /// so the trait object is not interned alone (`dyn Trait + 'a` would be rejected by the
    /// Repeater's implementation, E0478, for such a trait), and a trait object interned alone
    /// would be `'static`. It is a transparent part of the reference (its components are the
    /// reference's, and it is not a type to intern, see `Leaker::needs_parent()`), unless it must
    /// be interned itself (it has a cause of its own), when the reference is interned as a whole
    /// instead, with `'a` among its generics, so rustc gives the trait object its default in the
    /// implementation of the Repeater.
    ///
    /// Neither is a trait object with a lifetime bound written (`&'a (dyn Trait + 'b)`) of a
    /// lifetime of the Leaker, which could be: the reference is the type that carries the bound
    /// `'b: 'a` it implies.
    Reference(usize),
    /// A path with lifetime arguments (e.g. `Ref<'a, dyn Trait>`, where `Ref<'a, T: 'a>` makes the
    /// default `'a`), which is the type at the given depth. The default depends on the path, which
    /// we cannot resolve, so the trait object is not interned alone. It is a transparent part of
    /// the path (its components are the path's, and it is not a type to intern), unless it must
    /// be interned itself (it has a cause of its own), when the path is interned as a whole
    /// instead.
    Path(usize),
}

impl Inherits {
    /// The depth of the type which the trait object is the referent or an argument of, which is
    /// interned as a whole in its place.
    fn owner(self) -> usize {
        match self {
            Inherits::Reference(owner) | Inherits::Path(owner) => owner,
        }
    }
}

impl<'a> AnalyzeVisitor<'a> {
    /// Returns a visitor with the generics `generics` of the Leaker, for the `call`-th successful
    /// call of [`Leaker::intern_with()`] (see [`SelfRequirement::Call`]).
    pub(crate) fn new(leaker: &'a Leaker, call: usize, generics: &Generics) -> Self {
        Self {
            leaker,
            scopes: vec![Scope::new(generics)],
            binders: Vec::new(),
            probe: None,
            root_defaulted: generics
                .params
                .iter()
                .filter_map(|param| match param {
                    GenericParam::Type(TypeParam {
                        ident,
                        default: Some(_),
                        ..
                    }) => Some(ident.clone()),
                    _ => None,
                })
                .collect(),
            self_ty: Some(SelfRequirement::Call(call)),
            in_default: false,
            modules: Vec::new(),
            module_path: Vec::new(),
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

    /// Analyzes `ty` alone (see [`Leaker::check()`]).
    pub(crate) fn check(leaker: &Leaker, ty: &Type, generics: &Generics) -> CheckResult {
        let mut visitor = AnalyzeVisitor::new(leaker, 0, generics);
        visitor.visit_type(ty);
        // Everything found is in `ty`, which is at the depth 1.
        visitor.causes(0, 0, 1)
    }

    /// Analyzes the default `ty` of a type parameter, with the parameters before it (and the
    /// where predicates which do not mention the others) as the root generics `generics` (see
    /// [`Leaker::intern_with()`]).
    pub(crate) fn analyze_default(leaker: &Leaker, ty: &Type, generics: &Generics) -> Vec<Effect> {
        let mut visitor = AnalyzeVisitor::new(leaker, 0, generics);
        visitor.self_ty = None;
        visitor.in_default = true;
        visitor.visit_type(ty);
        visitor.finish()
    }

    /// Builds the [`CheckResult`] of the type at `depth` from the causes found in it, since
    /// `must_start` and `impossible_start`.
    fn causes(&self, must_start: usize, impossible_start: usize, depth: usize) -> CheckResult {
        let mut must: Vec<&(usize, TokenStream)> = self.musts[must_start..].iter().collect();
        // The outermost causes first (in the order of appearance among the same depth).
        must.sort_by_key(|(d, _)| *d);
        CheckResult {
            is_own: must.first().map_or(false, |(d, _)| *d == depth),
            must: must.into_iter().map(|(_, m)| m.clone()).collect(),
            impossible: self.impossibles[impossible_start..].to_vec(),
            note: None,
        }
    }

    /// Records a cause which depends on the type context, found in the type being visited.
    fn push_must(&mut self, must: TokenStream) {
        self.musts.push((self.type_depth, must));
        self.frame.min_depth = self.frame.min_depth.min(self.type_depth);
    }

    /// Records that the own path of the type being visited is an absolute one naming an external
    /// crate (see [`Leaker::is_soft_path()`]), which the Referrer may replace: not a cause.
    fn push_soft(&mut self) {
        self.frame.soft_depth = self.frame.soft_depth.min(self.type_depth);
    }

    /// Records a cause which cannot be interned, found in the type being visited.
    fn push_impossible(&mut self, impossible: TokenStream) {
        self.impossibles.push(impossible);
    }

    /// Records that the type being visited mentions the generic declared in the scope `scope`.
    fn mention(&mut self, scope: usize) {
        if scope == 0 {
            self.frame.mentions_root = true;
        } else {
            self.frame.local_scope = self.frame.local_scope.min(scope);
        }
    }

    /// Returns the index of the innermost scope declaring the type or const parameter named
    /// `name`, or the lifetime parameter if it starts with `'` (e.g. `'a`).
    fn scope_of(&self, name: &str) -> Option<usize> {
        self.scopes.iter().rposition(|scope| {
            scope
                .params()
                .any(|param| generic_param_name(param) == name)
        })
    }

    /// Returns the context of `ty`, in which `inner` is found.
    fn context_of(&self, ty: &Type, inner: &Frame) -> Context {
        let context = if inner.mentions_root
            || inner.mentions_self
            || inner.local_scope < self.scopes.len()
        {
            Context::of(
                ty,
                &self.scopes,
                self.self_ty.as_ref(),
                &Assumed::default(),
                &self.enclosing,
            )
        } else {
            Context::none()
        };
        if !self.modules.is_empty() && (inner.relative || self.depends_on_module(&context)) {
            context.with_module(Some(self.module_path.clone()))
        } else {
            context
        }
    }

    /// Returns `true` if `context` (the bounds and predicates of the generics, and what `Self`
    /// means) refers to a path resolved in the module where it is written, i.e. one which is not
    /// absolute, `crate::`, `$crate`, allowed by the Leaker, a generic or `Self`.
    fn depends_on_module(&self, context: &Context) -> bool {
        struct Finder<'v, 'l> {
            visitor: &'v AnalyzeVisitor<'l>,
            found: bool,
        }
        impl<'v, 'l, 'ast> Visit<'ast> for Finder<'v, 'l> {
            fn visit_path(&mut self, i: &'ast Path) {
                let local = i.leading_colon.is_none()
                    && matches!(i.segments.first(), Some(seg)
                        if seg.ident != "crate" && seg.ident != "$crate" && seg.ident != "Self")
                    && !self.visitor.starts_with_generic(i)
                    && !self.visitor.leaker.is_context_free_path(i);
                self.found |= local;
                syn::visit::visit_path(self, i)
            }
            fn visit_macro(&mut self, _: &'ast Macro) {
                // The tokens are resolved where the macro is expanded.
                self.found = true;
            }
        }
        let mut finder = Finder {
            visitor: self,
            found: false,
        };
        finder.visit_generics(context.generics());
        match context.self_ty() {
            Some(SelfRequirement::Bound(path)) => finder.visit_path(path),
            Some(SelfRequirement::Type(ty)) => finder.visit_type(ty),
            _ => (),
        }
        finder.found
    }

    /// Returns the first mention of `Self` in the generics which `context` declares, which
    /// cannot be leaked unless [`Leaker::self_ty_can_be_interned`]. A bound `Self: 'a` which the
    /// item writes is not among them (it is encoded as `&'a Self`, see
    /// [`Context::self_outlives()`]), but the implementation needs `Self` to encode it as well.
    fn self_in_bounds(&self, context: &Context) -> Option<TokenStream> {
        if self.leaker.self_ty_can_be_interned {
            return None;
        }
        first_self_in_generics(context.generics())
            .or_else(|| (!context.self_outlives().is_empty()).then(|| quote!(Self)))
    }

    /// Returns the lifetime which `context` declares and which the output of a `for<..>` binder
    /// being visited mentions (see [`Binder`]), with the depth of the type holding the binder
    /// (the outermost binder first).
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

    /// Returns the conflict of the type `ty` at `depth`, which is interned with the lifetime of a
    /// binder (see [`Binder`]) but there is no type holding it to be interned as a whole.
    fn binder_conflict(&self, inner: &Frame, depth: usize, lifetime: &Lifetime) -> CheckResult {
        let mut conflict = self.causes(inner.must_start, inner.impossible_start, depth);
        conflict.impossible.insert(0, quote!(#lifetime));
        conflict.note = Some(binder_note(lifetime, false));
        conflict
    }

    /// Finishes the analysis, returning the effects to be applied.
    pub(crate) fn finish(self) -> Vec<Effect> {
        self.frame.effects
    }

    /// Merges what is found in a type which cannot be interned on its own (e.g. a trait object
    /// inheriting the lifetime of its path which has a cause of its own, or a type in an array
    /// length) into `owner`, the depth of the enclosing type which is interned as a whole instead,
    /// in place of [`AnalyzeVisitor::decide()`]: all its causes are those of the owner, as if they
    /// were in its own path, and its components are not interned either (the owner is not
    /// decomposed).
    fn decide_as_owner(&mut self, owner: usize, mut inner: Frame) {
        if inner.min_depth != usize::MAX {
            self.own_causes(inner.must_start, owner);
            inner.min_depth = owner;
        }
        // The same goes for the types naming an external crate (see `Frame::soft_depth`).
        if inner.soft_depth != usize::MAX {
            inner.soft_depth = owner;
        }
        self.frame.min_depth = self.frame.min_depth.min(inner.min_depth);
        self.frame.soft_depth = self.frame.soft_depth.min(inner.soft_depth);
        self.frame.local_scope = self.frame.local_scope.min(inner.local_scope);
        self.frame.mentions_root |= inner.mentions_root;
        self.frame.mentions_self |= inner.mentions_self;
        self.frame.module_item |= inner.module_item;
        self.frame.relative |= inner.relative;
        self.merge_binder(&inner);
        let effects = std::mem::take(&mut inner.effects);
        self.frame.effects.extend(effects);
    }

    /// Merges what is found in a type which cannot be interned on its own, but has no cause of
    /// its own (e.g. `dyn ::p::Tr<Rel>` inheriting the lifetime of its path), into the enclosing
    /// type, in place of [`AnalyzeVisitor::decide()`]: it is not a type to intern, which a cut of
    /// the graph could choose instead of its components, but they are, as they are not affected
    /// by what is inherited (the copy keeps the trait object where it is).
    fn decide_through(&mut self, mut inner: Frame) {
        self.frame.min_depth = self.frame.min_depth.min(inner.min_depth);
        self.frame.soft_depth = self.frame.soft_depth.min(inner.soft_depth);
        self.frame.local_scope = self.frame.local_scope.min(inner.local_scope);
        self.frame.mentions_root |= inner.mentions_root;
        self.frame.mentions_self |= inner.mentions_self;
        self.frame.module_item |= inner.module_item;
        self.frame.relative |= inner.relative;
        self.merge_binder(&inner);
        let effects = std::mem::take(&mut inner.effects);
        self.frame.effects.extend(effects);
    }

    /// Keeps the binder of the type just visited (see [`Frame::binder`]) for the enclosing type,
    /// unless that is the type holding the binder.
    fn merge_binder(&mut self, inner: &Frame) {
        if self.frame.binder.is_none() {
            let depth = self.type_depth;
            self.frame.binder = inner.binder.clone().filter(|(owner, _)| *owner < depth);
        }
    }

    /// Makes the causes found since `must_start` those of the type at `depth` (see
    /// [`CheckResult::is_own`]).
    fn own_causes(&mut self, must_start: usize, depth: usize) {
        for must in &mut self.musts[must_start..] {
            must.0 = depth;
        }
    }

    /// Returns the context of the type `ty` at `depth`, given what is found in it (`inner`),
    /// if it can be interned as a whole because of its own path (see [`CheckResult::is_own`]).
    /// Otherwise returns the conflict: it must be interned but cannot be, and cannot be decomposed
    /// either, so its path should be made absolute.
    fn own_context(
        &self,
        ty: &Type,
        depth: usize,
        inner: &Frame,
    ) -> std::result::Result<Context, CheckResult> {
        if self.impossibles.len() > inner.impossible_start {
            // The type cannot be interned (e.g. `MyType<'_>`).
            let mut conflict = self.causes(inner.must_start, inner.impossible_start, depth);
            // A type in it, which is interned as a whole in its place, mentions a lifetime of a
            // binder (see `Binder`).
            conflict.note = inner
                .binder
                .as_ref()
                .filter(|(owner, _)| *owner == depth)
                .map(|(_, lifetime)| binder_note(lifetime, true))
                .or_else(|| inner.unconstrained.as_ref().map(unconstrained_note));
            return Err(conflict);
        }
        if self.in_default && inner.mentions_self {
            // The Repeater's implementation of the default cannot know what `Self` is, as the
            // default is also used to compute the arguments of the trait.
            return Err(CheckResult {
                must: vec![first_self_in_type(ty).unwrap_or_else(|| quote!(Self))],
                impossible: vec![quote!(#ty)],
                is_own: true,
                note: Some("a default of a parameter cannot depend on `Self`".to_string()),
            });
        }
        let context = self.context_of(ty, inner);
        match self.self_in_bounds(&context) {
            Some(self_path) => Err(CheckResult {
                must: vec![quote!(#ty)],
                impossible: vec![self_path],
                is_own: true,
                note: Some(
                    "the bounds of its generics mention `Self`, which cannot be leaked".to_string(),
                ),
            }),
            None => Ok(context),
        }
    }

    /// Decides how to handle the type `ty` at `depth`, given what is found in it (`inner`).
    /// `ambiguous` is `true` if it is (directly) a bare identifier written as a generic argument.
    fn decide(&mut self, ty: &Type, depth: usize, ambiguous: bool, mut inner: Frame) {
        // A bare identifier as a generic argument (e.g. `SIZE` in `W<SIZE>`) may be a const, which
        // syn parses as a type, and which cannot be interned on its own. So the type it is a
        // component of is interned as a whole, with the cause in its own path, if it can be.
        // Otherwise the identifier is a type like any other. A type referring to an item of the
        // module is not interned either, as the Referrer shadows the item in the copy of the
        // module. A group is transparent (see `has_ambiguous()`). Parentheses are not interned in
        // place of the type containing them either: a trait object inheriting a lifetime in them
        // is not interned alone (see `decide_as_owner()`).
        if inner.min_depth != depth
            && !inner.module_item
            && !matches!(ty, Type::Group(_) | Type::Paren(_))
            && has_ambiguous(&inner.effects)
            && self.own_context(ty, depth, &inner).is_ok()
        {
            self.own_causes(inner.must_start, depth);
            inner.min_depth = depth;
        }
        // A type interned as a whole (because of its own path, see `CheckResult::is_own`) with a
        // lifetime of a `for<..>` binder which the output of the binder mentions cannot be (see
        // `Binder`). The type holding the binder is interned as a whole instead, which can be
        // (unless it is a conflict itself), or it is a conflict if there is none (the binder is in
        // a bound).
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
            own = Some(Err(self.binder_conflict(&inner, depth, &lifetime)));
        }
        // The same for a type naming an external crate with a lifetime of such a binder (which no
        // input anchors, as a type of an external crate is no anchor): it cannot be an optional
        // entry alone either, so the type holding the binder is, instead. That does not make it
        // own the causes in it (the required entries do not depend on the optional ones).
        if own.is_none() && inner.soft_depth == depth && !inner.module_item && !self.in_receiver {
            if let Ok(context) = self.own_context(ty, depth, &inner) {
                if let Some((_, owner)) = self.output_binder(&context) {
                    if owner > 0 {
                        inner.soft_depth = owner;
                        return self.decide_through(inner);
                    }
                }
            }
        }
        // The causes and the local generics also belong to the enclosing type, as those of a
        // component.
        self.frame.min_depth = self.frame.min_depth.min(inner.min_depth);
        self.frame.soft_depth = self.frame.soft_depth.min(inner.soft_depth);
        self.frame.local_scope = self.frame.local_scope.min(inner.local_scope);
        self.frame.mentions_root |= inner.mentions_root;
        self.frame.mentions_self |= inner.mentions_self;
        self.frame.module_item |= inner.module_item;
        self.frame.relative |= inner.relative;
        self.merge_binder(&inner);
        let has_impossible = self.impossibles.len() > inner.impossible_start;
        let has_must = inner.min_depth != usize::MAX;
        // The scopes declared inside the type (e.g. by a binder) are not in scope any more.
        let mentions_local = inner.local_scope < self.scopes.len();
        // The type is an optional entry if its own path names an external crate (see
        // `Leaker::is_soft_path()`) and no cause is its own (it is a required entry then), and it
        // can be leaked as a whole. It is no conflict otherwise: it is just not leaked, and
        // copied as it is.
        let optional = if own.is_none()
            && inner.soft_depth == depth
            && !inner.module_item
            && !self.in_receiver
        {
            self.optional_context(ty, depth, &inner)
        } else {
            None
        };
        let effects = std::mem::take(&mut inner.effects);

        if let Some(own) = own {
            match own {
                Ok(context) => {
                    // If the type depends on both the type context and generics, it cannot be
                    // decomposed further. It is interned together with the generics it needs,
                    // which the Repeater's implementation must declare (see
                    // [`Referrer::generics()`](crate::Referrer::generics())).
                    self.frame.effects.push(Effect::Type(Box::new(PendingType {
                        ty: ty.clone(),
                        is_must_intern: true,
                        has_must: true,
                        in_graph: true,
                        optional: false,
                        has_optional: false,
                        ambiguous: ambiguous && context.is_empty(),
                        context,
                        components: Vec::new(),
                    })));
                }
                // The type must be interned (because of its own path, the first of `must`, which
                // depends on the type context) but cannot be, and cannot be decomposed either.
                Err(conflict) => self.frame.effects.push(Effect::Error(conflict)),
            }
        } else if has_impossible {
            if has_must {
                // The type cannot be interned as a whole (e.g. `&MyType` with an elided lifetime,
                // or `impl Iterator<Item = MyType>`), but its components can.
                self.frame.effects.extend(effects);
            } else if has_optional(&effects) {
                // The same for the types which may be replaced (`impl Iterator<Item =
                // ::path::MyType>`), as they have no cause: the primitive types are assumed.
                self.frame.effects.extend(effects);
            } else {
                // Otherwise, the type cannot be interned, but does not need to be (e.g.
                // `impl ::path::Trait`). It is copied as it is, so the primitive types in it are
                // assumed.
                self.frame.effects.extend(primitive_effects(effects));
            }
        } else if mentions_local || inner.module_item || (self.in_default && inner.mentions_self) {
            // The type itself is not interned because it depends on local generics (or refers to
            // an item of the module, which the Referrer shadows in the copy of the module, or to
            // `Self` in a default), but its components can be. It may be replaced if it is an
            // optional entry, which does not decompose into the required ones.
            match optional {
                Some(context) => {
                    self.frame.effects.push(Effect::Type(Box::new(PendingType {
                        ty: ty.clone(),
                        is_must_intern: false,
                        has_must,
                        in_graph: false,
                        optional: true,
                        has_optional: true,
                        context,
                        ambiguous: false,
                        components: effects,
                    })));
                }
                None => self.frame.effects.extend(effects),
            }
        } else {
            // The type may be interned as a whole, or its components.
            let context = self.context_of(ty, &inner);
            if self.self_in_bounds(&context).is_some() {
                self.frame.effects.extend(effects);
            } else {
                self.frame.effects.push(Effect::Type(Box::new(PendingType {
                    ty: ty.clone(),
                    is_must_intern: false,
                    has_must,
                    in_graph: true,
                    optional: optional.is_some(),
                    has_optional: optional.is_some() || has_optional(&effects),
                    context,
                    ambiguous: false,
                    components: effects,
                })));
            }
        }
    }

    /// Returns the context of the type `ty` at `depth` (see `decide()`) if it can be an optional
    /// entry: it can be leaked as a whole (see [`AnalyzeVisitor::own_context()`]), and does not use
    /// a lifetime of a `for<..>` binder which its output mentions (see [`Binder`]), which `decide()`
    /// leaves to the type holding the binder.
    fn optional_context(&self, ty: &Type, depth: usize, inner: &Frame) -> Option<Context> {
        let context = self.own_context(ty, depth, inner).ok()?;
        match self.output_binder(&context) {
            Some(_) => None,
            None => Some(context),
        }
    }

    /// Records that the relative path `must` cannot be interned (because of `impossible`, e.g. a
    /// trait bound or a value containing it), so it should be absolute.
    fn relative_path(&mut self, must: TokenStream, impossible: TokenStream) {
        self.frame.effects.push(Effect::Error(CheckResult {
            must: vec![must],
            impossible: vec![impossible],
            is_own: true,
            note: None,
        }));
    }

    /// Returns `true` if `path` does not depend on the type context: it is context-free for the
    /// Leaker (see [`Leaker::is_context_free_path()`]), or it is portable in the analyzed module
    /// (see [`AnalyzeVisitor::is_module_path()`]).
    fn is_context_free(&self, path: &Path) -> bool {
        self.leaker.is_context_free_path(path) || self.is_module_path(path)
    }

    /// Returns `true` if `path`, which is not a generic or `Self`, is resolved in the module where
    /// it is written: it is relative, and neither starts with `crate` nor is portable with the
    /// module.
    fn is_relative(&self, path: &Path) -> bool {
        path.leading_colon.is_none()
            && !matches!(path.segments.first(), Some(seg) if seg.ident == "crate")
            && !self.is_context_free(path)
    }

    /// Returns `true` if `path` refers to an item declared in the module being analyzed (see
    /// [`Leaker::from_module()`]), which is copied with it, like Rust resolves it: it starts with
    /// a name declared in the current module (and not shadowed by a generic parameter in scope),
    /// or with `self::` or `super::` (repeated) followed by a name declared in the module it
    /// refers to, which is not outside the root module.
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
                        // It leaves the root module.
                        None => return false,
                    }
                }
                true
            }
            _ => false,
        };
        let mut module = &self.modules[current];
        let mut first = true;
        loop {
            let seg = match segments.next() {
                Some(seg) => seg,
                None => return false,
            };
            let name = ident_name(&seg.ident);
            // The name must be declared (not imported by a relative `use`) in the module it is
            // looked up in.
            if !module.names.contains(&name) || (first && !explicit && self.is_generic(&seg.ident))
            {
                return false;
            }
            first = false;
            match module.modules.get(&name) {
                // The path goes through a module, which declares the next name.
                Some(inner) if segments.peek().is_some() => module = inner,
                // The rest (if any) is resolved through the item.
                _ => return true,
            }
        }
    }

    /// Returns `true` if `ident` is a type or const parameter in scope.
    fn is_generic(&self, ident: &Ident) -> bool {
        let name = ident_name(ident);
        self.params().any(|param| match param {
            GenericParam::Type(TypeParam { ident, .. })
            | GenericParam::Const(ConstParam { ident, .. }) => ident_name(ident) == name,
            GenericParam::Lifetime(_) => false,
        })
    }

    /// Runs `f` with the names declared in the module `i` in scope.
    fn with_module(&mut self, i: &ItemMod, f: impl FnOnce(&mut Self)) {
        let items = match &i.content {
            Some((_, items)) => items,
            None => return f(self),
        };
        // The outermost module is the root of the paths.
        let nested = !self.modules.is_empty();
        if nested {
            self.module_path.push(i.ident.clone());
        }
        self.modules.push(declared_names(items));
        f(self);
        self.modules.pop();
        if nested {
            self.module_path.pop();
        }
    }

    /// Returns the generic parameters in scope, those of the Leaker and the local ones.
    fn params(&self) -> impl Iterator<Item = &GenericParam> {
        self.scopes.iter().flat_map(|scope| scope.params())
    }

    /// Returns `true` if the associated item of `qself` (e.g. `<T>::X`, without `as`) is resolved
    /// through the bounds of its self type, a generic parameter in scope or `Self`.
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

    /// Returns `true` if `path` starts with a generic parameter (e.g. `T` or `T::Item`).
    fn starts_with_generic(&self, path: &Path) -> bool {
        path.leading_colon.is_none()
            && path.segments.first().map_or(false, |seg| {
                seg.arguments.is_none() && self.is_generic(&seg.ident)
            })
    }

    /// Runs `f` with `scope` pushed.
    fn with_scope(&mut self, scope: Scope, f: impl FnOnce(&mut Self)) {
        self.scopes.push(scope);
        f(self);
        self.scopes.pop();
    }

    /// Runs `f` with the generics of a trait or impl item (a method, a type or a const) in scope,
    /// as local generics.
    fn with_local_generics(&mut self, generics: &Generics, f: impl FnOnce(&mut Self)) {
        self.with_scope(Scope::new(generics), f)
    }

    /// Runs `f` with the generics of an item. They are those of the Leaker for the outermost
    /// item, and local generics for a nested one.
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

    /// Like [`AnalyzeVisitor::with_item_generics()`], with `Self` meaning `self_ty` (if given) in
    /// the item. Unlike the generics, this applies to a nested item as well.
    fn with_item(
        &mut self,
        generics: &Generics,
        self_ty: Option<Option<SelfRequirement>>,
        f: impl FnOnce(&mut Self),
    ) {
        let outer = match self_ty {
            Some(self_ty) => Some(std::mem::replace(&mut self.self_ty, self_ty)),
            None => None,
        };
        self.with_item_generics(generics, f);
        if let Some(outer) = outer {
            self.self_ty = outer;
        }
    }

    /// Runs `f` with the generics of the function `sig` (a method, or an item if `item`) in scope,
    /// and the lifetimes which the copy of its signature would leave unconstrained (see
    /// [`AnalyzeVisitor::unconstrained_lifetimes()`]).
    fn with_signature(&mut self, sig: &Signature, item: bool, f: impl FnOnce(&mut Self)) {
        // The generics of the outermost item are those of the Leaker (the scope 0).
        let local = !item || self.item_depth > 0;
        if local {
            let risky = self.unconstrained_lifetimes(sig);
            self.signatures.push((self.scopes.len(), risky));
        }
        if item {
            self.with_item_generics(&sig.generics, f);
        } else {
            self.with_local_generics(&sig.generics, f);
        }
        if local {
            self.signatures.pop();
        }
    }

    /// Returns `true` if `lifetime`, declared in the scope `scope`, is one of a method which the
    /// copy of its signature would leave unconstrained if a type using it were interned as a whole
    /// (see [`AnalyzeVisitor::unconstrained_lifetimes()`]).
    fn is_unconstrained(&self, lifetime: &Lifetime, scope: usize) -> bool {
        self.signatures
            .iter()
            .any(|(at, risky)| *at == scope && risky.contains(lifetime))
    }

    /// Runs `f` with the lifetimes of a `for<..>` binder (e.g. `'a` in `for<'a> Fn(&'a T)`) in
    /// scope. They are local generics: a type using one is interned with it among its generics,
    /// which the Repeater's implementation declares. Unless the output of the binder mentions it
    /// and no input anchors it (`uses` are the lifetimes of the binder which the output mentions,
    /// see [`Binder`] and [`AnalyzeVisitor::unanchored()`]): a type using it is not interned alone
    /// then, but with the type holding the binder.
    ///
    /// `f` may be run twice (see [`AnalyzeVisitor::unanchored()`]).
    fn with_bound_lifetimes(
        &mut self,
        lifetimes: Option<&BoundLifetimes>,
        uses: Vec<OutputUse>,
        f: impl Fn(&mut Self),
    ) {
        let lifetimes = match lifetimes {
            Some(lifetimes) => lifetimes,
            None => return f(self),
        };
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

    /// Returns the lifetimes of `uses` (those of the binder `lifetimes` which its output mentions)
    /// which no input anchors: the types using them cannot be interned alone (see [`Binder`]).
    ///
    /// rustc requires a lifetime of the output to be constrained by the inputs, where a projection
    /// constrains nothing, and the copy replaces the types leaked by projections. So a lifetime is
    /// fine if an input keeps it where it is written in the copy: an *anchor*, an occurrence of it
    /// in a position which constrains it (anywhere but in a projection, see
    /// [`AnalyzeVisitor::scan_anchors()`]), with none of the types from the input down to the
    /// occurrence interned. The types interned alone are those which depend on the type
    /// context by their own path (they are interned as a whole), and those holding the binder
    /// (see [`Binder`]), and no other, as a type which mentions a lifetime of the binder is never
    /// interned unless it has a cause of its own (see [`AnalyzeVisitor::decide()`]: it depends on
    /// local generics, so its components are interned instead), so the types on the way down to the
    /// occurrence are decomposed, never interned whole.
    ///
    /// Whether a type has a cause of its own is known only when it is visited, and the anchor must
    /// be known before the inputs are (a type using the lifetime may precede the anchor, as in
    /// `Fn(Rel<'x>, &'x str) -> &'x str`, or the type holding the anchor may be interned itself, as
    /// `Rel2<&'x str>` of `Fn(Rel2<&'x str>) -> &'x str` is). So the binder is probed first when
    /// some lifetime has an occurrence: the same code visits it in an isolated state (restored
    /// afterwards) which logs the types found to be interned as a whole. The binders inside it are
    /// not probed again (the cost of a binder is that of its content, not exponential in the
    /// nesting): they take all their lifetimes of the output as unanchored there, which can only
    /// make more types interned, i.e. fewer anchors (a function pointer or a trait object with a
    /// binder of its own on the way to an anchor is one of them if a type using the lifetime of its
    /// binder is interned in its place). Only the types on the way to an anchor count (see
    /// [`AnalyzeVisitor::scan_anchors()`]).
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

    /// Runs `f` with the binder `lifetimes` in scope, without effect on the analysis, and returns
    /// the addresses of the types found to be interned as a whole in it (see
    /// [`AnalyzeVisitor::unanchored()`]).
    fn probe_binder(
        &mut self,
        lifetimes: &BoundLifetimes,
        f: &impl Fn(&mut Self),
    ) -> HashSet<usize> {
        let frame = Frame::new(self.musts.len(), self.impossibles.len());
        let frame = std::mem::replace(&mut self.frame, frame);
        let (musts, impossibles) = (self.musts.len(), self.impossibles.len());
        self.probe = Some(HashSet::new());
        self.with_scope(Scope::binder(lifetimes), |v| f(v));
        self.frame = frame;
        self.musts.truncate(musts);
        self.impossibles.truncate(impossibles);
        self.probe.take().unwrap_or_default()
    }

    /// Returns the lifetimes of the binder which the output of each group mentions
    /// (see [`Binder`]), with their anchors in its inputs.
    fn binder_uses(&self, groups: Vec<BinderGroup>) -> Vec<OutputUse> {
        let mut uses = Vec::new();
        for group in groups {
            for lifetime in &group.output {
                let mut anchors = Vec::new();
                // The binders inside a probe do not look for anchors (see `unanchored()`).
                if self.probe.is_none() {
                    for input in &group.inputs {
                        match input {
                            BinderInput::Type(ty) => {
                                self.scan_anchors(ty, lifetime, &mut Vec::new(), &mut anchors)
                            }
                            BinderInput::Lifetime(input) if *input == lifetime => {
                                anchors.push(Vec::new())
                            }
                            BinderInput::Lifetime(_) => (),
                        }
                    }
                }
                uses.push(OutputUse {
                    lifetime: lifetime.clone(),
                    anchors,
                });
            }
        }
        uses
    }

    /// Finds the occurrences of the lifetime `wanted` in `ty` which constrain it in rustc's sense
    /// (an input of a function pointer or of `Fn(..)` constrains the lifetimes it mentions, except
    /// in a projection), pushing to `anchors` the types from the root of the input (`chain`, which
    /// is the way down to `ty`, exclusive) down to the one holding it.
    ///
    /// Everything constrains but a projection (`<T as Tr<'x>>::A`, `T::A<'x>`, `Self::A<'x>`), the
    /// length of an array (an unevaluated constant), `impl Trait` and a type macro, which are not
    /// looked into. rustc checks the input as a type: whatever it holds counts, however deep, and
    /// whether it is a generic argument, an associated type binding (`dyn Iterator<Item = &'x u8>`),
    /// the output of `Fn(..) -> R`, the lifetime bound of a trait object (`dyn Tr + 'x`), or inside a
    /// function pointer or a trait object with a binder of its own (`dyn for<'y> Fn(&'x u8, &'y u8)`),
    /// while a projection is skipped wherever it is (also in a binding). Only the bound itself,
    /// whose associated type bindings are its output (see [`bound_inputs()`]), differs. A path whose
    /// definition is not seen is assumed to be a struct, an enum or a union (a type alias of a
    /// projection, or one which drops the lifetime, does not constrain, and is not told from them).
    ///
    /// The occurrence is an anchor only if it stays in the copy as written: none of the types on the
    /// chain is interned as a whole (see [`AnalyzeVisitor::unanchored()`]), which the chain
    /// records. That includes the trait objects, the function pointers and the paths holding it,
    /// which are not leaked if they mention `'x` unless they have a cause of their own (a relative
    /// trait or path). When in doubt it is not an anchor, the rule of [`Binder`] applies then.
    ///
    /// The Referrer may replace a type with the path of an external crate (`::other::Wrap<&'x
    /// u8>`, see [`Leaker::is_soft_path()`]) as a whole, and the Leaker does not know whether it
    /// does (see `Referrer::allow_ambiguous_crate_name()`), so such a type, and a trait object with
    /// such a path in a bound, are on the chain of a type which may be leaked: nothing in them is an
    /// anchor, whichever the Referrer does. The paths of `::core` and those allowed by the Leaker
    /// stay where they are, so what is in them counts.
    fn scan_anchors(
        &self,
        ty: &Type,
        wanted: &Lifetime,
        chain: &mut Vec<usize>,
        anchors: &mut Vec<Vec<usize>>,
    ) {
        chain.push(ty as *const Type as usize);
        match ty {
            // A reference to a trait object of an external crate without a lifetime bound is leaked
            // as a whole if the trait object is (it inherits the lifetime of the reference, see
            // `Inherits::Reference`), which the Referrer may replace.
            Type::Reference(reference) if self.is_external_object(&reference.elem) => {}
            Type::Reference(reference) => {
                if reference.lifetime.as_ref() == Some(wanted) {
                    anchors.push(chain.clone());
                }
                self.scan_anchors(&reference.elem, wanted, chain, anchors);
            }
            Type::Ptr(ptr) => self.scan_anchors(&ptr.elem, wanted, chain, anchors),
            Type::Slice(slice) => self.scan_anchors(&slice.elem, wanted, chain, anchors),
            // The length of an array naming an external crate makes the array a type which may be
            // replaced as a whole.
            Type::Array(array) if self.names_external_crate_in_expr(&array.len) => {}
            Type::Array(array) => self.scan_anchors(&array.elem, wanted, chain, anchors),
            Type::Paren(paren) => self.scan_anchors(&paren.elem, wanted, chain, anchors),
            Type::Group(group) => self.scan_anchors(&group.elem, wanted, chain, anchors),
            Type::Tuple(tuple) => {
                for elem in &tuple.elems {
                    self.scan_anchors(elem, wanted, chain, anchors);
                }
            }
            // The path of an external crate (`::other::Wrap<&'x u8>`) may be leaked as a whole by
            // the Referrer (see `Leaker::is_soft_path()`), so nothing in it is an anchor.
            Type::Path(TypePath { qself: None, path })
                if !self.is_projection(path)
                    && !self.leaker.is_soft_path(path)
                    && !self.holds_external_object(path) =>
            {
                if let Some(seg) = path.segments.last() {
                    self.scan_arguments(&seg.arguments, wanted, chain, anchors);
                }
            }
            Type::BareFn(f) if !declares(f.lifetimes.as_ref(), wanted) => {
                for arg in &f.inputs {
                    self.scan_anchors(&arg.ty, wanted, chain, anchors);
                }
                self.scan_output(&f.output, wanted, chain, anchors);
            }
            // The same goes for a trait object naming an external crate.
            Type::TraitObject(object) if !self.names_external_crate(object) => {
                for bound in &object.bounds {
                    match bound {
                        TypeParamBound::Trait(bound)
                            if !declares(bound.lifetimes.as_ref(), wanted) =>
                        {
                            if let Some(seg) = bound.path.segments.last() {
                                self.scan_arguments(&seg.arguments, wanted, chain, anchors);
                            }
                        }
                        TypeParamBound::Lifetime(lifetime) if lifetime == wanted => {
                            anchors.push(chain.clone())
                        }
                        _ => (),
                    }
                }
            }
            _ => (),
        }
        chain.pop();
    }

    /// Finds the occurrences of `wanted` in the generic arguments of a path in an input, including
    /// the associated type bindings and the output of `Fn(..) -> R` (it is not the bound being
    /// checked, whose output does not constrain), see [`AnalyzeVisitor::scan_anchors()`].
    fn scan_arguments(
        &self,
        arguments: &PathArguments,
        wanted: &Lifetime,
        chain: &mut Vec<usize>,
        anchors: &mut Vec<Vec<usize>>,
    ) {
        match arguments {
            PathArguments::AngleBracketed(args) => {
                for arg in &args.args {
                    match arg {
                        GenericArgument::Lifetime(lifetime) if lifetime == wanted => {
                            anchors.push(chain.clone())
                        }
                        GenericArgument::Type(ty) => self.scan_anchors(ty, wanted, chain, anchors),
                        GenericArgument::AssocType(binding) => {
                            self.scan_anchors(&binding.ty, wanted, chain, anchors)
                        }
                        _ => (),
                    }
                }
            }
            PathArguments::Parenthesized(args) => {
                for ty in &args.inputs {
                    self.scan_anchors(ty, wanted, chain, anchors);
                }
                self.scan_output(&args.output, wanted, chain, anchors);
            }
            PathArguments::None => (),
        }
    }

    /// Finds the occurrences of `wanted` in the return type of a function pointer or of `Fn(..)`
    /// in an input, see [`AnalyzeVisitor::scan_anchors()`].
    fn scan_output(
        &self,
        output: &ReturnType,
        wanted: &Lifetime,
        chain: &mut Vec<usize>,
        anchors: &mut Vec<Vec<usize>>,
    ) {
        if let ReturnType::Type(_, ty) = output {
            self.scan_anchors(ty, wanted, chain, anchors);
        }
    }

    /// Returns `true` if `ty` (also in parentheses and groups) is a trait object without a
    /// lifetime bound which names an external crate: it inherits the lifetime of the reference or
    /// the path around it, which is leaked as a whole instead of it if it is (see
    /// `Inherits`), so what is around it may be replaced by the Referrer.
    fn is_external_object(&self, ty: &Type) -> bool {
        trait_object_of(ty).map_or(false, |object| {
            self.names_external_crate(object) && !has_lifetime_bound(object)
        })
    }

    /// Returns `true` if the last segment of `path` has a lifetime argument and an argument which
    /// is such a trait object (see [`AnalyzeVisitor::is_external_object()`]): the path is leaked as
    /// a whole if the trait object is.
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

    /// Returns `true` if a path in the expression `expr` is the path of an external crate (see
    /// [`Leaker::is_soft_path()`]).
    fn names_external_crate_in_expr(&self, expr: &Expr) -> bool {
        struct Finder<'v, 'l> {
            visitor: &'v AnalyzeVisitor<'l>,
            found: bool,
        }
        impl<'v, 'l, 'ast> Visit<'ast> for Finder<'v, 'l> {
            fn visit_path(&mut self, i: &'ast Path) {
                self.found |= self.visitor.leaker.is_soft_path(i);
                syn::visit::visit_path(self, i)
            }
        }
        let mut finder = Finder {
            visitor: self,
            found: false,
        };
        finder.visit_expr(expr);
        finder.found
    }

    /// Returns `true` if a bound of the trait object is the path of an external crate (see
    /// [`Leaker::is_soft_path()`]): the trait object may be leaked as a whole by the Referrer.
    fn names_external_crate(&self, object: &TypeTraitObject) -> bool {
        object.bounds.iter().any(|bound| {
            matches!(bound, TypeParamBound::Trait(bound) if self.leaker.is_soft_path(&bound.path))
        })
    }

    /// Returns `true` if `path` is a projection: an associated item of a generic parameter in
    /// scope or of `Self` (`T::A`, `Self::A`), which constrains nothing (`<T as Tr>::A` has a
    /// qualified self).
    fn is_projection(&self, path: &Path) -> bool {
        path.leading_colon.is_none()
            && path.segments.len() > 1
            && matches!(path.segments.first(), Some(seg)
                if seg.ident == "Self" || self.is_generic(&seg.ident))
    }
}

impl Analysis {
    /// Applies `effects` in order, adding the types as components of `parent` (if any), each in
    /// its context (numbered in the order of the first appearance), to the graph of the required
    /// entries.
    pub(crate) fn apply(
        &mut self,
        contexts: &mut Contexts,
        effects: &[Effect],
        parent: Option<&NodeKey>,
    ) {
        for effect in effects {
            match effect {
                Effect::Type(pending) if !pending.has_must || !pending.in_graph => {
                    // A type with no cause in it is never interned, so it is not added (nor its
                    // context), but the assumptions of its components are kept.
                    self.apply(contexts, &pending.components, parent);
                }
                Effect::Type(pending) => {
                    let context = contexts.intern(pending.context.clone());
                    let key = self
                        .types
                        .add(context, &pending.ty, parent, pending.is_must_intern);
                    self.apply(contexts, &pending.components, Some(&key));
                }
                Effect::Error(error) => self.errors.push(error.clone()),
                Effect::Primitive(ident) => self.primitives.push(ident.clone()),
            }
        }
    }

    /// Applies `effects` in order, like [`Analysis::apply()`], adding the types to the graph of the
    /// optional entries instead: the types which may be replaced (see [`PendingType::optional`]),
    /// and the types containing them, in contexts of their own (`contexts`, not those of the
    /// required entries). The types interned as a whole by their own cause are required (leaked in
    /// both graphs), so they are not in this one, nor the types in them.
    pub(crate) fn apply_optional(
        &mut self,
        contexts: &mut Contexts,
        effects: &[Effect],
        parent: Option<&NodeKey>,
    ) {
        for effect in effects {
            if let Effect::Type(pending) = effect {
                if pending.is_must_intern || !(pending.optional || pending.has_optional) {
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

/// Returns `true` if a type among `effects` is optional, or contains one (see
/// [`PendingType::optional`]).
fn has_optional(effects: &[Effect]) -> bool {
    effects
        .iter()
        .any(|effect| matches!(effect, Effect::Type(pending) if pending.optional || pending.has_optional))
}

impl<'a, 'ast> Visit<'ast> for AnalyzeVisitor<'a> {
    fn visit_type_param(&mut self, i: &TypeParam) {
        // The default of a type parameter of the root generics is analyzed on its own, with the
        // parameters before it (see [`Leaker::intern_with()`]), not copied in this one.
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
        self.with_signature(&i.sig, false, |visitor| {
            syn::visit::visit_trait_item_fn(visitor, &i)
        });
    }

    fn visit_trait_item_type(&mut self, i: &TraitItemType) {
        self.with_local_generics(&i.generics, |visitor| {
            syn::visit::visit_trait_item_type(visitor, i)
        });
    }

    fn visit_receiver(&mut self, i: &Receiver) {
        let outer = std::mem::replace(&mut self.in_receiver, true);
        syn::visit::visit_receiver(self, i);
        self.in_receiver = outer;
    }

    fn visit_trait_item_const(&mut self, i: &TraitItemConst) {
        // Like the bodies of methods, defaults of associated consts are not copied.
        let mut i = i.clone();
        i.default = None;
        self.with_local_generics(&i.generics, |visitor| {
            syn::visit::visit_trait_item_const(visitor, &i)
        });
    }

    fn visit_impl_item_fn(&mut self, i: &ImplItemFn) {
        // The body is not analyzed, as it is not a type: the copy of the function is rewritten by
        // the Referrer, which does not need its types to be leaked.
        self.with_signature(&i.sig, false, |visitor| {
            for attr in &i.attrs {
                visitor.visit_attribute(attr);
            }
            visitor.visit_visibility(&i.vis);
            visitor.visit_signature(&i.sig);
        });
    }

    fn visit_impl_item_type(&mut self, i: &ImplItemType) {
        self.with_local_generics(&i.generics, |visitor| {
            syn::visit::visit_impl_item_type(visitor, i)
        });
    }

    fn visit_impl_item_const(&mut self, i: &ImplItemConst) {
        // The value is not analyzed, like the body of a function.
        self.with_local_generics(&i.generics, |visitor| {
            for attr in &i.attrs {
                visitor.visit_attribute(attr);
            }
            visitor.visit_visibility(&i.vis);
            visitor.visit_generics(&i.generics);
            visitor.visit_type(&i.ty);
        });
    }

    fn visit_item_const(&mut self, i: &ItemConst) {
        // The value is not analyzed, like the body of a function.
        for attr in &i.attrs {
            self.visit_attribute(attr);
        }
        self.visit_visibility(&i.vis);
        self.visit_generics(&i.generics);
        self.visit_type(&i.ty);
    }

    fn visit_item_static(&mut self, i: &ItemStatic) {
        // The value is not analyzed, like the body of a function.
        for attr in &i.attrs {
            self.visit_attribute(attr);
        }
        self.visit_visibility(&i.vis);
        self.visit_type(&i.ty);
    }

    fn visit_item_struct(&mut self, i: &ItemStruct) {
        let self_ty = SelfRequirement::of_item(&i.ident, &i.generics);
        self.with_item(&i.generics, Some(Some(self_ty)), |visitor| {
            syn::visit::visit_item_struct(visitor, i)
        });
    }

    fn visit_item_enum(&mut self, i: &ItemEnum) {
        let self_ty = SelfRequirement::of_item(&i.ident, &i.generics);
        self.with_item(&i.generics, Some(Some(self_ty)), |visitor| {
            syn::visit::visit_item_enum(visitor, i)
        });
    }

    fn visit_item_union(&mut self, i: &ItemUnion) {
        let self_ty = SelfRequirement::of_item(&i.ident, &i.generics);
        self.with_item(&i.generics, Some(Some(self_ty)), |visitor| {
            syn::visit::visit_item_union(visitor, i)
        });
    }

    fn visit_item_trait(&mut self, i: &ItemTrait) {
        let self_ty = SelfRequirement::of_trait(&i.ident, &i.generics);
        self.with_item(&i.generics, Some(Some(self_ty)), |visitor| {
            syn::visit::visit_item_trait(visitor, i)
        });
    }

    fn visit_item_type(&mut self, i: &ItemType) {
        self.with_item_generics(&i.generics, |visitor| {
            syn::visit::visit_item_type(visitor, i)
        });
    }

    fn visit_item_fn(&mut self, i: &ItemFn) {
        // The body is not analyzed, like that of a method.
        self.with_signature(&i.sig, true, |visitor| {
            for attr in &i.attrs {
                visitor.visit_attribute(attr);
            }
            visitor.visit_visibility(&i.vis);
            visitor.visit_signature(&i.sig);
        });
    }

    fn visit_item_mod(&mut self, i: &ItemMod) {
        // A module has no generics, but the items in it are nested items (see
        // [`Leaker::from_module()`]), whose generics are local.
        // `Self` has no meaning in a module, only in the items in it.
        self.with_item(&Generics::default(), Some(None), |visitor| {
            visitor.with_module(i, |visitor| syn::visit::visit_item_mod(visitor, i))
        });
    }

    fn visit_item_impl(&mut self, i: &ItemImpl) {
        let self_ty = SelfRequirement::Type((*i.self_ty).clone());
        self.with_item(&i.generics, Some(Some(self_ty)), |visitor| {
            // The trait is checked like a trait bound, but its generic arguments are analyzed in
            // any case.
            if let Some((_, path, _)) = &i.trait_ {
                if !visitor.is_context_free(path) {
                    visitor.relative_path(quote!(#path), quote!(#path));
                }
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
        let frame = Frame::new(self.musts.len(), self.impossibles.len());
        let outer = std::mem::replace(&mut self.frame, frame);
        // Whether this is (directly) the referent of a reference, or an argument of a path, which
        // cannot be interned as a whole (its lifetime is elided or local), or one with a lifetime
        // of the Leaker (see `inherits`).
        let elided_ref = std::mem::take(&mut self.elided_ref);
        let unconstrained = self.unconstrained_ref.take();
        // Whether this is a bare identifier written as a generic argument (see `ambiguous_arg`).
        let ambiguous = std::mem::take(&mut self.ambiguous_arg);
        let inherits = self.inherits.take();
        self.type_depth += 1;
        // What the type must outlive is that of the components while they are visited.
        let enclosing = self.enclosing.clone();
        self.enclosing = enclosing.components_of(i, self.scopes.len());
        match i {
            Type::Macro(m) => {
                // The tokens of a type macro cannot be analyzed, so it is interned as a whole (it
                // is expanded in the Leaker's context).
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
                self.frame.mentions_self |= names.contains("Self");
                if names.contains("Self") && !self.leaker.self_ty_can_be_interned {
                    self.push_impossible(quote!(Self));
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
                    // `'_` is a cause itself, and keeps the reference from being interned as a
                    // whole, so the trait object would be alone.
                    Some(lifetime) if lifetime.ident == "_" => self.elided_ref = true,
                    Some(lifetime) if lifetime.ident == "static" => (),
                    Some(lifetime) => {
                        let local = self
                            .scope_of(&lifetime.to_string())
                            .filter(|&scope| scope > 0);
                        let bounded =
                            trait_object_of(&reference.elem).map_or(false, has_lifetime_bound);
                        match local {
                            // A lifetime of the Leaker (or one no scope declares).
                            None => self.inherits = Some(Inherits::Reference(self.type_depth)),
                            // A local one, with a lifetime bound written on the trait object: it
                            // is interned alone, with its bound.
                            Some(_) if bounded => {}
                            // A local one which the copy would leave unconstrained (see
                            // `unconstrained_ref`).
                            Some(scope) if self.is_unconstrained(lifetime, scope) => {
                                self.unconstrained_ref = Some(lifetime.clone())
                            }
                            Some(_) => self.inherits = Some(Inherits::Reference(self.type_depth)),
                        }
                    }
                    None => {
                        self.push_impossible(quote!(#i));
                        self.elided_ref = true;
                    }
                }
                syn::visit::visit_type(self, i)
            }
            Type::TraitObject(object)
                if elided_ref
                    && !object
                        .bounds
                        .iter()
                        .any(|bound| matches!(bound, TypeParamBound::Lifetime(_))) =>
            {
                // Its lifetime defaults to that of the reference, which is not interned as a whole,
                // so it cannot be interned alone (see [`NotInternableError`]).
                self.push_impossible(quote!(#i));
                syn::visit::visit_type(self, i)
            }
            Type::TraitObject(object) if unconstrained.is_some() && !has_lifetime_bound(object) => {
                // Its lifetime defaults to a lifetime of the path which the copy would leave
                // unconstrained, so the path cannot be interned as a whole, nor can it alone.
                if let Some(lifetime) = &unconstrained {
                    self.push_impossible(quote!(#lifetime));
                }
                self.frame.unconstrained = unconstrained;
                syn::visit::visit_type(self, i)
            }
            Type::ImplTrait(_) | Type::Verbatim(_) => {
                self.push_impossible(quote!(#i));
                syn::visit::visit_type(self, i);
            }
            _ => syn::visit::visit_type(self, i),
        }
        self.enclosing = enclosing;
        let mut inner = std::mem::replace(&mut self.frame, outer);
        // Also parentheses and groups around the trait object, which are not interned alone
        // either. One with a lifetime bound inherits nothing, but the reference is still the type
        // to carry the bound (see `Inherits::Reference`).
        let inherits = inherits.filter(|inherits| {
            self.impossibles.len() == inner.impossible_start
                && matches!(unwrap_type_group(i), Type::TraitObject(object)
                    if matches!(inherits, Inherits::Reference(_)) || !has_lifetime_bound(object))
        });
        match (self.expr_owner, inherits) {
            // Not interned alone (see `expr_owner`).
            (Some(owner), _) => self.decide_as_owner(owner, inner),
            // Not interned alone (see `inherits`), but its components can be, unless it has to be
            // (it has a cause of its own, or a bare identifier in it may be a const, which cannot
            // be interned alone): then the reference or the path is interned as a whole instead,
            // with all its causes. The copy keeps the trait object where it is, so it keeps the
            // lifetime it defaults to.
            (None, Some(inherits)) => {
                let owner = inherits.owner();
                let own = inner.min_depth == self.type_depth
                    || inner.min_depth == owner
                    || (!matches!(i, Type::Group(_) | Type::Paren(_))
                        && !inner.module_item
                        && has_ambiguous(&inner.effects)
                        && self.own_context(i, self.type_depth, &inner).is_ok());
                if own {
                    self.decide_as_owner(owner, inner)
                } else {
                    // A trait object naming an external crate is not an optional entry alone
                    // either: the reference or the path is, if it can be.
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
        let output = output_lifetimes(i.lifetimes.as_ref(), |v| v.visit_return_type(&i.output));
        let group = BinderGroup {
            output,
            inputs: i
                .inputs
                .iter()
                .map(|arg| BinderInput::Type(&arg.ty))
                .collect(),
        };
        let uses = self.binder_uses(vec![group]);
        self.with_bound_lifetimes(i.lifetimes.as_ref(), uses, |v| {
            syn::visit::visit_type_bare_fn(v, i)
        })
    }

    fn visit_type_trait_object(&mut self, i: &TypeTraitObject) {
        // All the trait bounds are checked (e.g. both `A` and `B` in `dyn A + B`).
        for bound in &i.bounds {
            match bound {
                TypeParamBound::Trait(bound) => self.visit_trait_bound(bound),
                TypeParamBound::Verbatim(_) => self.push_impossible(quote!(#bound)),
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
                TypeParamBound::Trait(bound) => Some(bound),
                _ => None,
            })
            .map(|bound| {
                let output =
                    output_lifetimes(i.lifetimes.as_ref(), |v| v.visit_bound_output(bound));
                let mut inputs = vec![BinderInput::Type(&i.bounded_ty)];
                bound_inputs(bound, &mut inputs);
                BinderGroup { output, inputs }
            })
            .collect();
        let uses = self.binder_uses(groups);
        self.with_bound_lifetimes(i.lifetimes.as_ref(), uses, |v| {
            syn::visit::visit_predicate_type(v, i)
        })
    }

    fn visit_trait_bound(&mut self, i: &TraitBound) {
        // A trait bound with a relative path cannot be interned (in a type, e.g. `dyn MyTrait`,
        // the path is a cause of the type instead).
        if self.type_depth == 0 && !self.is_context_free(&i.path) {
            self.relative_path(quote!(#i), quote!(#i));
            return;
        }
        let output = output_lifetimes(i.lifetimes.as_ref(), |v| v.visit_bound_output(i));
        let mut inputs = Vec::new();
        bound_inputs(i, &mut inputs);
        let uses = self.binder_uses(vec![BinderGroup { output, inputs }]);
        self.with_bound_lifetimes(i.lifetimes.as_ref(), uses, |v| {
            syn::visit::visit_trait_bound(v, i)
        })
    }

    fn visit_const_param(&mut self, i: &ConstParam) {
        // The type of a const parameter cannot be a projection (such as
        // `<MyLeakerType as Repeater<0>>::Type`), so it is never interned. It must be a primitive
        // type, or a context-free path to a type alias of it (e.g. `::my_crate::Idx`).
        for attr in &i.attrs {
            self.visit_attribute(attr);
        }
        match unwrap_type_group(&i.ty) {
            ty @ Type::Path(TypePath { qself: None, path }) if is_const_param_ty(ty) => {
                // A bare primitive name (e.g. `u8`, not `::core::primitive::u8`) is copied as it
                // is, assuming that it is not shadowed.
                if let Some(ident) = path.get_ident() {
                    self.frame.effects.push(Effect::Primitive(ident.clone()));
                }
            }
            Type::Path(TypePath { qself: None, path }) if self.is_context_free(path) => {}
            _ => {
                // The type of a const parameter cannot be interned.
                self.relative_path(quote!(#{&i.ty}), quote!(#i));
                return;
            }
        }
        if let Some(expr) = &i.default {
            self.visit_expr(expr);
        }
    }

    fn visit_path(&mut self, i: &Path) {
        let starts_with_generic = self.starts_with_generic(i);
        if self.type_depth > 0 && starts_with_generic {
            if let Some(seg) = i.segments.first() {
                if let Some(scope) = self.scope_of(&ident_name(&seg.ident)) {
                    self.mention(scope);
                }
            }
        }
        if self.type_depth > 0
            && i.leading_colon.is_none()
            && matches!(i.segments.first(), Some(seg) if seg.ident == "Self")
        {
            self.frame.mentions_self = true;
        }
        let in_module =
            !starts_with_generic && !self.leaker.is_context_free_path(i) && self.is_module_path(i);
        if in_module && self.type_depth > 0 {
            self.frame.module_item = true;
        }
        if self.leaker.is_context_free_path(i) || starts_with_generic || in_module {
            // A primitive type allowed by `allowed_paths` (e.g. by
            // `LeakerConfig::allow_primitive()`) is copied as it is, assuming that its name is not
            // shadowed. The name alone cannot prove it (e.g. `type u16 = i32;` in the Leaker's
            // module), so it is recorded for `Leaker::assertions()`, which fails to compile there
            // if the name is shadowed.
            if !starts_with_generic && !in_module && i.leading_colon.is_none() {
                if let Some(seg) = i.segments.first() {
                    if is_primitive_name(&seg.ident) {
                        self.frame
                            .effects
                            .push(Effect::Primitive(seg.ident.clone()));
                    }
                }
            }
            // The path of an external crate (`::foo::X`) is not resolved against the type context
            // but by the name of the crate, which the Referrer may not name as the Leaker does: the
            // type is an optional entry (see `PendingType::optional`).
            if self.type_depth > 0
                && !starts_with_generic
                && !in_module
                && self.leaker.is_soft_path(i)
            {
                self.push_soft();
            }
        } else if self.type_depth > 0 {
            let is_self = i.leading_colon.is_none()
                && matches!(i.segments.first(), Some(seg) if seg.ident == "Self" && seg.arguments.is_none());
            if !is_self {
                // The path depends on the type context.
                self.push_must(quote!(#i));
                self.frame.relative |= self.is_relative(i);
            } else if !self.leaker.self_ty_can_be_interned {
                self.push_impossible(quote!(#i));
            }
        }
        // The generic arguments are checked in any case (e.g. `RelType` in
        // `::core::option::Option<RelType>`).
        syn::visit::visit_path(self, i)
    }

    fn visit_type_path(&mut self, i: &TypePath) {
        match &i.qself {
            Some(qself) if self.resolves_through_bounds(qself) => {
                // `<T>::X` is resolved through the bounds of `T`, so `X` does not depend on the
                // type context (only the generic arguments in the path do).
                self.visit_qself(qself);
                for seg in &i.path.segments {
                    self.visit_path_arguments(&seg.arguments);
                }
            }
            _ => syn::visit::visit_type_path(self, i),
        }
    }

    fn visit_qself(&mut self, i: &QSelf) {
        // `<T>::Ty` cannot be interned, unless `T` is a generic parameter (or `Self`).
        if self.type_depth > 0 && i.as_token.is_none() && !self.resolves_through_bounds(i) {
            let ty = &i.ty;
            self.push_impossible(quote!(#ty));
        }
        syn::visit::visit_qself(self, i)
    }

    fn visit_lifetime(&mut self, i: &Lifetime) {
        if self.type_depth == 0 {
            return;
        }
        if i.ident == "_" {
            self.push_impossible(quote!(#i));
        } else if let Some(scope) = self.scope_of(&i.to_string()) {
            self.mention(scope);
        } else if i.ident != "static" {
            // The lifetime depends on the type context.
            self.push_must(quote!(#i));
        }
    }

    fn visit_expr(&mut self, i: &Expr) {
        if self.type_depth > 0 {
            if let Expr::Closure(_) | Expr::Assign(_) | Expr::Verbatim(_) | Expr::Macro(_) = i {
                self.push_impossible(quote!(#i));
            }
        }
        // The types in an expression of a type belong to the type (see `expr_owner`), and are not
        // components of the types around it.
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
        // A trait object without a lifetime bound, as an argument of a path with lifetime
        // arguments, may default to one of them (e.g. `Ref<'a, dyn Trait>` is `Ref<'a, dyn Trait +
        // 'a>`, because of the bound `T: 'a` of `Ref<'a, T>`), which only the definition of the
        // path tells. So the path is interned as a whole if the trait object has to be, which
        // needs a type and lifetimes it can have among its generics: not `'_`, and not a local
        // lifetime if there are several lifetime arguments, or the copy would leave it
        // unconstrained (see `unconstrained_lifetimes()`).
        let mut elided = false;
        let mut inherited = false;
        let mut unconstrained_lifetime = None;
        let lifetimes = i
            .args
            .iter()
            .filter(|arg| matches!(arg, GenericArgument::Lifetime(_)))
            .count();
        for arg in &i.args {
            if let GenericArgument::Lifetime(lifetime) = arg {
                if lifetime.ident == "_" {
                    elided = true;
                } else if let Some(scope) = self.scope_of(&lifetime.to_string()) {
                    // The local lifetimes of a method, an associated type, a nested item or a
                    // binder are generics of the whole path.
                    if scope > 0 && lifetimes > 1 {
                        elided = true;
                    } else {
                        inherited = true;
                        if scope > 0 && self.is_unconstrained(lifetime, scope) {
                            unconstrained_lifetime = Some(lifetime.clone());
                        }
                    }
                } else if lifetime.ident != "static" {
                    inherited = true;
                }
            }
        }
        // The path is interned as a whole, which needs a type (not a trait bound, for instance).
        let elided = elided || (inherited && self.type_depth == 0);
        for arg in &i.args {
            if let GenericArgument::Type(ty) = arg {
                self.elided_ref = elided;
                self.unconstrained_ref = unconstrained_lifetime.clone().filter(|_| !elided);
                if inherited && !elided && unconstrained_lifetime.is_none() {
                    self.inherits = Some(Inherits::Path(self.type_depth));
                }
                // A bare identifier may be a const, which syn parses as a type (see
                // `ambiguous_arg`). Not so the types in the other syntax, such as associated type
                // bindings or the elements of a tuple.
                self.ambiguous_arg = is_bare_ident(ty);
            }
            self.visit_generic_argument(arg);
        }
    }

    fn visit_expr_path(&mut self, i: &ExprPath) {
        if self.type_depth > 0 {
            // In a type (e.g. `[u8; MY_LEN]`), the path is a cause of the type.
            return syn::visit::visit_expr_path(self, i);
        }
        // Values starting with a generic parameter in scope (e.g. `N` or `T::N` where
        // `const N: usize` or `T` is a generic parameter) or `Self` do not depend on the type
        // context.
        let is_self = i.qself.is_none()
            && i.path.leading_colon.is_none()
            && matches!(i.path.segments.first(), Some(seg) if seg.ident == "Self" && seg.arguments.is_none());
        if (i.qself.is_none() && self.starts_with_generic(&i.path))
            || (is_self && self.leaker.self_ty_can_be_interned)
            || self.is_context_free(&i.path)
        {
            syn::visit::visit_expr_path(self, i)
        } else {
            // A value with a relative path, which cannot be interned
            self.relative_path(quote!(#i), quote!(#i));
        }
    }
}

impl<'a> AnalyzeVisitor<'a> {
    /// Returns the lifetimes declared by the generics of the function `sig` which the copy of its
    /// signature would leave unconstrained if a type using them in the inputs were interned as a
    /// whole (a path with the lifetime as its only lifetime argument, or a reference with it to a
    /// trait object without a lifetime bound: the copy has it only in a projection, which does not
    /// constrain it), in a way which makes the copy early-bound where the original is late-bound:
    /// rustc rejects the implementation of a method whose lifetime parameters differ in whether
    /// they are early- or late-bound from those of the trait's (E0195). A lifetime is late-bound
    /// if an input constrains it, or if it is not in the return type and in no bound or where
    /// clause, and early-bound otherwise, so it is the lifetimes which the return type mentions
    /// (also inside a type, e.g. `Box<dyn Tr + 'b>`), which no bound or where predicate mentions
    /// (that makes it early-bound in both), which some input mentions, and which no input keeps
    /// constrained where it is written: the receiver and the arguments which are references with it
    /// (`&'b self`, `x: &'b X`, also inside a tuple, an array, a slice or a pointer: the types
    /// which are never interned as a whole because of their own path, so they stay in the copy), but
    /// for a reference to a trait object without a lifetime bound, `&'b dyn Tr`, which is interned
    /// as a whole if the trait object has to be (it is a cause itself, see
    /// `AnalyzeVisitor::is_not_a_cause()`). Anything else, e.g. `Vec<&'b X>`, may be interned, so
    /// it is not counted (conservative).
    ///
    /// A lifetime which no input mentions is not affected: it is early-bound in the original (it
    /// is in the return type only), and so it is in the copy, where it is in the return type only
    /// as well (`fn f<'b>(&self) -> &'b dyn RelTr`, `-> Ref<'b, dyn RelTr>`).
    fn unconstrained_lifetimes(&self, sig: &Signature) -> Vec<Lifetime> {
        let declared: Vec<Lifetime> = sig
            .generics
            .params
            .iter()
            .filter_map(|param| match param {
                GenericParam::Lifetime(param) => Some(param.lifetime.clone()),
                _ => None,
            })
            .collect();
        if declared.is_empty() {
            return Vec::new();
        }
        let mut output = Lifetimes::default();
        output.visit_return_type(&sig.output);
        // The lifetimes in bounds are early-bound: `'b: 'c` (both), `T: 'b` and `T: Tr<'b>`.
        let mut bounded = Lifetimes::default();
        for param in &sig.generics.params {
            match param {
                GenericParam::Lifetime(param) => {
                    if !param.bounds.is_empty() {
                        bounded.0.push(param.lifetime.clone());
                    }
                    for bound in &param.bounds {
                        bounded.visit_lifetime(bound);
                    }
                }
                GenericParam::Type(param) => {
                    for bound in &param.bounds {
                        bounded.visit_type_param_bound(bound);
                    }
                }
                GenericParam::Const(_) => (),
            }
        }
        if let Some(where_clause) = &sig.generics.where_clause {
            bounded.visit_where_clause(where_clause);
        }
        // The generics in scope where the trait objects of the inputs are, for `is_not_a_cause()`.
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

    /// Returns `true` if the trait object `object` is certainly not a cause itself (an own path
    /// which depends on the type context, an elided lifetime, a bare identifier which may be a
    /// const, ...: see [`CheckResult::is_own`]), nor an optional entry (its own path is the one of
    /// an external crate, which the Referrer may replace: see [`Leaker::is_soft_path()`]), so
    /// that a reference to it is never interned as a whole because of it (see
    /// `Inherits::Reference`), and the reference stays in the copy as it is written. `in_scope`
    /// are the generics in scope where it is. It is analyzed on its own like [`Leaker::check()`]
    /// does, so it is `false` when unsure: a path which is relative (or names a module of the
    /// analyzed one, which the analysis of the object alone does not see), a type or a lifetime
    /// which `in_scope` does not declare, and anything impossible to leak.
    fn is_not_a_cause(&self, object: &TypeTraitObject, in_scope: &Generics) -> bool {
        let ty = Type::TraitObject(object.clone());
        let mut visitor = AnalyzeVisitor::new(self.leaker, 0, in_scope);
        visitor.visit_type(&ty);
        // Everything found is in `ty`, which is at the depth 1.
        let result = visitor.causes(0, 0, 1);
        // A trait object naming an external crate may be leaked by the Referrer (with the reference
        // holding it), which keeps nothing as written.
        let optional = visitor.frame.soft_depth == 1;
        !result.is_own && !optional && result.impossible.is_empty()
    }
}

/// Pushes the lifetimes which `ty` (an input) keeps constrained where they are written in the copy
/// (see [`AnalyzeVisitor::unconstrained_lifetimes()`]): those of the references in it, outside any
/// path, except the references to a trait object without a lifetime bound, which are interned as a
/// whole if the trait object has to be (see `Inherits::Reference`): they count only if
/// `not_a_cause` says the trait object is certainly not a cause itself.
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

/// Collects the lifetimes mentioned in the syntax visited, in the tokens of a macro as well.
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

/// The note of the conflict of a trait object defaulting to the lifetime of a method which the copy
/// would leave unconstrained (see [`AnalyzeVisitor::unconstrained_lifetimes()`]).
fn unconstrained_note(lifetime: &Lifetime) -> String {
    format!(
        "the lifetime of the trait object defaults to `{0}`, a lifetime of the method which its \
         return type mentions and no other argument keeps as written (e.g. `&{0} self`), so the \
         copy would have `{0}` only in a projection, and rustc rejects the implementation of a \
         method whose lifetime is early-bound where the trait's is late-bound (E0195)",
        lifetime
    )
}

/// The note of the conflict of a type using the lifetime of a `for<..>` binder (see [`Binder`]).
/// `owned` is `true` if the type holding the binder would be interned as a whole instead, which is
/// a conflict as well.
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

/// Returns the lifetimes declared by the `for<..>` binder `lifetimes` which are mentioned by the
/// output visited by `f` (see [`Binder`]).
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

/// Pushes the inputs of the bound `bound` to `inputs`: the generic arguments of the trait
/// (`Fn(A, B)` or `Tr<'x, A>`), not the associated type bindings or the return type, which are the
/// output (see [`Binder`]).
fn bound_inputs<'a>(bound: &'a TraitBound, inputs: &mut Vec<BinderInput<'a>>) {
    let seg = match bound.path.segments.last() {
        Some(seg) => seg,
        None => return,
    };
    match &seg.arguments {
        PathArguments::AngleBracketed(args) => {
            for arg in &args.args {
                match arg {
                    GenericArgument::Lifetime(lifetime) => {
                        inputs.push(BinderInput::Lifetime(lifetime))
                    }
                    GenericArgument::Type(ty) => inputs.push(BinderInput::Type(ty)),
                    _ => (),
                }
            }
        }
        PathArguments::Parenthesized(args) => {
            inputs.extend(args.inputs.iter().map(BinderInput::Type));
        }
        PathArguments::None => (),
    }
}

/// Finds the lifetimes of a binder (`declared`) which are mentioned in the syntax visited, unless
/// a binder inside it declares the same name (`shadowed`).
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

    /// Scans the tokens, which are not parsed (e.g. of a macro), for the lifetimes by name.
    fn scan(&mut self, tokens: TokenStream) {
        let mut names = HashSet::new();
        token_names(tokens, &mut names);
        for lifetime in self.declared.clone() {
            if names.contains(&lifetime.to_string()) {
                self.mention(&lifetime);
            }
        }
    }

    /// Visits the output of a bound: the return type of the parenthesized arguments (`Fn(A) -> R`)
    /// and the associated type bindings (`Iterator<Item = R>`). The bounds on an associated type
    /// (`Iterator<Item: Tr<'x>>`) are not, they are where predicates which the lifetime is
    /// declared for.
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
        let outer = self.shadowed.len();
        self.shadowed.extend(binder_lifetimes(i.lifetimes.as_ref()));
        syn::visit::visit_type_bare_fn(self, i);
        self.shadowed.truncate(outer);
    }
    fn visit_trait_bound(&mut self, i: &'ast TraitBound) {
        let outer = self.shadowed.len();
        self.shadowed.extend(binder_lifetimes(i.lifetimes.as_ref()));
        syn::visit::visit_trait_bound(self, i);
        self.shadowed.truncate(outer);
    }
    fn visit_predicate_type(&mut self, i: &'ast PredicateType) {
        let outer = self.shadowed.len();
        self.shadowed.extend(binder_lifetimes(i.lifetimes.as_ref()));
        syn::visit::visit_predicate_type(self, i);
        self.shadowed.truncate(outer);
    }
}

/// Returns `true` if the `for<..>` binder `lifetimes` declares `wanted`, which then is another
/// lifetime inside it (rustc rejects such a shadowing, but a macro may produce it).
fn declares(lifetimes: Option<&BoundLifetimes>, wanted: &Lifetime) -> bool {
    binder_lifetimes(lifetimes).any(|lifetime| lifetime == *wanted)
}

/// Returns `true` if `ty` is a single identifier (e.g. `SIZE`), which may be a const when it is a
/// generic argument. Invisible groups (e.g. from `$t:ty` in `macro_rules!`) are transparent, but
/// parentheses are not: `(SIZE)` is a type.
fn is_bare_ident(ty: &Type) -> bool {
    match ty {
        Type::Group(TypeGroup { elem, .. }) => is_bare_ident(elem),
        Type::Path(TypePath { qself: None, path }) => path.get_ident().is_some(),
        _ => false,
    }
}

/// Returns `true` if a type among `effects` is ambiguous (see [`PendingType::ambiguous`]), also
/// inside a group, which is transparent.
fn has_ambiguous(effects: &[Effect]) -> bool {
    effects.iter().any(|effect| match effect {
        Effect::Type(pending) => {
            pending.ambiguous
                || (matches!(pending.ty, Type::Group(_)) && has_ambiguous(&pending.components))
        }
        _ => false,
    })
}

/// Returns the effects which assume the primitive types, among `effects` (and those of the types
/// in them).
fn primitive_effects(effects: Vec<Effect>) -> Vec<Effect> {
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

/// The names declared in a module (see [`declared_names()`]), and those of the modules declared
/// in it.
#[derive(Default)]
struct ModuleNames {
    names: HashSet<String>,
    modules: HashMap<String, ModuleNames>,
}

/// Returns the names declared directly in `items`: items, `macro_rules!` macros, and the names
/// imported by `use` declarations with an absolute path (starting with `::` or `$crate`), which
/// are portable. The names imported by the others (e.g. `use super::X`, `use crate::X`) refer to
/// the items outside the module.
fn declared_names(items: &[Item]) -> ModuleNames {
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
            | Item::Macro(ItemMacro {
                ident: Some(ident), ..
            }) => {
                names.insert(ident_name(ident));
            }
            Item::Fn(ItemFn { sig, .. }) => {
                names.insert(ident_name(&sig.ident));
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

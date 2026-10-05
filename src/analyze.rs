//! The analysis of items and types by the [`Leaker`].
//!
//! [`AnalyzeVisitor`] visits each type once, in post-order: the causes found in a type (see
//! [`CheckResult`]) decide whether it is interned as a whole, decomposed into its components, or
//! reported as an error, and the effects of its components (the types to intern, the errors and
//! the assumed primitives) are kept or discarded accordingly.

use crate::context::{Context, Contexts, Scope, SelfRequirement};
use crate::encode::is_const_param_ty;
use crate::graph::{NodeKey, TypeGraph};
use crate::leaker::Leaker;
use crate::util::{
    first_self_in_generics, first_self_in_type, generic_param_name, ident_name, is_primitive_name,
    token_names, unwrap_type_group,
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

/// A type to be added to the [`TypeGraph`], with its components.
pub(crate) struct PendingType {
    ty: Type,
    is_must_intern: bool,
    /// A cause of the type is not the path of an external crate (see
    /// [`Referrer::must()`](crate::Referrer::must())).
    is_hard: bool,
    /// The type or a component of it is a cause (see [`CheckResult::must`]). Otherwise the type is
    /// never interned, so it is not added.
    has_must: bool,
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
    /// A cause found in the type is not the path of an external crate (an absolute path whose
    /// first segment is not `core`).
    hard: bool,
    /// A cause found in the type is a relative path resolved in the module where the type is
    /// written (not a generic, `Self`, an absolute path, `crate::`, or a path portable with the
    /// module), so the type means something else in another module.
    relative: bool,
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
            hard: false,
            relative: false,
            effects: Vec::new(),
        }
    }
}

/// The result of applying the effects of [`AnalyzeVisitor`].
#[derive(Default)]
pub(crate) struct Analysis {
    pub(crate) types: TypeGraph,
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
    /// cannot be interned as a whole (its lifetime is elided or local, or the path is in a trait
    /// bound), so a trait object without a lifetime bound, which defaults to that lifetime, cannot
    /// be interned alone.
    elided_ref: bool,
    /// The type being visited is (directly) a bare identifier written as a generic argument (e.g.
    /// `SIZE` in `W<SIZE>`), which may be a const (see [`PendingType::ambiguous`]).
    ambiguous_arg: bool,
    /// The type being visited is the referent of a reference, or a type argument of a path, with a
    /// lifetime of the Leaker's generics, so a trait object without a lifetime bound defaults to
    /// that lifetime, and is not the same type when interned alone (it would default to
    /// `'static`). Only the trait object is affected (also inside parentheses and groups): what
    /// is inside it is not, and can be interned.
    inherits: Option<Inherits>,
    /// The depth of the type owning the expression being visited in a type (an array length or a
    /// const argument), if any. The types in it are not interned on their own: the copy of the
    /// expression would have a projection (`<Self as Repeater<..>>::Type`) in an anonymous
    /// constant, which is not permitted if the generics (or `Self`) are in it.
    expr_owner: Option<usize>,
}

/// What a trait object without a lifetime bound inherits its lifetime from (see
/// [`AnalyzeVisitor::inherits`]).
#[derive(Clone, Copy)]
enum Inherits {
    /// A reference `&'a dyn Trait`, which is the type at the given depth. The default is `'a`, so
    /// the trait object is not interned alone. Neither is one with a lifetime bound written
    /// (`&'a (dyn Trait + 'b)`), which could be: the reference is the type that carries the bound
    /// `'b: 'a` it implies. It is a transparent part of the reference (its components are the
    /// reference's, and it is not a type to intern, see `Leaker::needs_parent()`), unless it must
    /// be interned itself (it has a cause of its own), when the reference is interned as a whole
    /// instead.
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
            ambiguous_arg: false,
            inherits: None,
            expr_owner: None,
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
    fn push_must(&mut self, must: TokenStream, hard: bool) {
        self.frame.hard |= hard;
        self.musts.push((self.type_depth, must));
        self.frame.min_depth = self.frame.min_depth.min(self.type_depth);
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
            Context::of(ty, &self.scopes, self.self_ty.as_ref(), &[])
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
    /// cannot be leaked unless [`Leaker::self_ty_can_be_interned`].
    fn self_in_bounds(&self, context: &Context) -> Option<TokenStream> {
        if self.leaker.self_ty_can_be_interned {
            return None;
        }
        first_self_in_generics(context.generics())
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
        self.frame.min_depth = self.frame.min_depth.min(inner.min_depth);
        self.frame.local_scope = self.frame.local_scope.min(inner.local_scope);
        self.frame.mentions_root |= inner.mentions_root;
        self.frame.mentions_self |= inner.mentions_self;
        self.frame.module_item |= inner.module_item;
        self.frame.hard |= inner.hard;
        self.frame.relative |= inner.relative;
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
        self.frame.local_scope = self.frame.local_scope.min(inner.local_scope);
        self.frame.mentions_root |= inner.mentions_root;
        self.frame.mentions_self |= inner.mentions_self;
        self.frame.module_item |= inner.module_item;
        self.frame.hard |= inner.hard;
        self.frame.relative |= inner.relative;
        let effects = std::mem::take(&mut inner.effects);
        self.frame.effects.extend(effects);
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
            return Err(self.causes(inner.must_start, inner.impossible_start, depth));
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
        // The causes and the local generics also belong to the enclosing type, as those of a
        // component.
        self.frame.min_depth = self.frame.min_depth.min(inner.min_depth);
        self.frame.local_scope = self.frame.local_scope.min(inner.local_scope);
        self.frame.mentions_root |= inner.mentions_root;
        self.frame.mentions_self |= inner.mentions_self;
        self.frame.module_item |= inner.module_item;
        self.frame.hard |= inner.hard;
        self.frame.relative |= inner.relative;
        let has_impossible = self.impossibles.len() > inner.impossible_start;
        let has_must = inner.min_depth != usize::MAX;
        let is_own = inner.min_depth == depth;
        // The scopes declared inside the type (e.g. by a binder) are not in scope any more.
        let mentions_local = inner.local_scope < self.scopes.len();
        let effects = std::mem::take(&mut inner.effects);

        if is_own {
            match self.own_context(ty, depth, &inner) {
                Ok(context) => {
                    // If the type depends on both the type context and generics, it cannot be
                    // decomposed further. It is interned together with the generics it needs,
                    // which the Repeater's implementation must declare (see
                    // [`Referrer::generics()`](crate::Referrer::generics())).
                    self.frame.effects.push(Effect::Type(Box::new(PendingType {
                        ty: ty.clone(),
                        is_must_intern: true,
                        is_hard: inner.hard,
                        has_must: true,
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
            } else {
                // Otherwise, the type cannot be interned, but does not need to be (e.g.
                // `impl ::path::Trait`). It is copied as it is, so the primitive types in it are
                // assumed.
                self.frame.effects.extend(primitive_effects(effects));
            }
        } else if mentions_local || inner.module_item || (self.in_default && inner.mentions_self) {
            // The type itself is not interned because it depends on local generics (or refers to
            // an item of the module, which the Referrer shadows in the copy of the module, or to
            // `Self` in a default), but its components can be.
            self.frame.effects.extend(effects);
        } else {
            // The type may be interned as a whole, or its components.
            let context = self.context_of(ty, &inner);
            if self.self_in_bounds(&context).is_some() {
                self.frame.effects.extend(effects);
            } else {
                self.frame.effects.push(Effect::Type(Box::new(PendingType {
                    ty: ty.clone(),
                    is_must_intern: false,
                    is_hard: inner.hard,
                    has_must,
                    context,
                    ambiguous: false,
                    components: effects,
                })));
            }
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

    /// Runs `f` with the lifetimes of a `for<..>` binder (e.g. `'a` in `for<'a> Fn(&'a T)`) in
    /// scope. They are treated as local generics, which the Repeater's implementation cannot
    /// refer to.
    fn with_bound_lifetimes(
        &mut self,
        lifetimes: Option<&BoundLifetimes>,
        f: impl FnOnce(&mut Self),
    ) {
        match lifetimes {
            Some(lifetimes) => self.with_scope(Scope::binder(lifetimes), f),
            None => f(self),
        }
    }
}

impl Analysis {
    /// Applies `effects` in order, adding the types as components of `parent` (if any), each in
    /// its context (numbered in the order of the first appearance).
    pub(crate) fn apply(
        &mut self,
        contexts: &mut Contexts,
        effects: Vec<Effect>,
        parent: Option<&NodeKey>,
    ) {
        for effect in effects {
            match effect {
                Effect::Type(pending) if !pending.has_must => {
                    // A type with no cause in it is never interned, so it is not added (nor its
                    // context), but the assumptions of its components are kept.
                    self.apply(contexts, pending.components, parent);
                }
                Effect::Type(pending) => {
                    let context = contexts.intern(pending.context);
                    let key = self.types.add(
                        context,
                        &pending.ty,
                        parent,
                        pending.is_must_intern,
                        pending.is_hard,
                    );
                    self.apply(contexts, pending.components, Some(&key));
                }
                Effect::Error(error) => self.errors.push(error),
                Effect::Primitive(ident) => self.primitives.push(ident),
            }
        }
    }
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
        self.with_local_generics(&i.sig.generics, |visitor| {
            syn::visit::visit_trait_item_fn(visitor, &i)
        });
    }

    fn visit_trait_item_type(&mut self, i: &TraitItemType) {
        self.with_local_generics(&i.generics, |visitor| {
            syn::visit::visit_trait_item_type(visitor, i)
        });
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
        self.with_local_generics(&i.sig.generics, |visitor| {
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
        self.with_item_generics(&i.sig.generics, |visitor| {
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
        // Whether this is a bare identifier written as a generic argument (see `ambiguous_arg`).
        let ambiguous = std::mem::take(&mut self.ambiguous_arg);
        let inherits = self.inherits.take();
        self.type_depth += 1;
        match i {
            Type::Macro(m) => {
                // The tokens of a type macro cannot be analyzed, so it is interned as a whole (it
                // is expanded in the Leaker's context).
                self.push_must(quote!(#i), true);
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
                self.inherits = inherits;
                // `W<(N)>` is unambiguously a type, unlike `W<N>` (also in an invisible group).
                self.ambiguous_arg = ambiguous && matches!(i, Type::Group(_));
                syn::visit::visit_type(self, i)
            }
            Type::Reference(TypeReference { lifetime, .. }) => {
                match lifetime {
                    // A local lifetime (or `'_`, a cause itself) keeps the reference from being
                    // interned as a whole, so the trait object would be alone.
                    Some(lifetime) => {
                        self.elided_ref = lifetime.ident == "_"
                            || self
                                .scope_of(&lifetime.to_string())
                                .map_or(false, |scope| scope > 0);
                        if !self.elided_ref && lifetime.ident != "static" {
                            self.inherits = Some(Inherits::Reference(self.type_depth));
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
            Type::ImplTrait(_) | Type::Verbatim(_) => {
                self.push_impossible(quote!(#i));
                syn::visit::visit_type(self, i);
            }
            _ => syn::visit::visit_type(self, i),
        }
        let inner = std::mem::replace(&mut self.frame, outer);
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
                    self.decide_through(inner)
                }
            }
            (None, None) => self.decide(i, self.type_depth, ambiguous, inner),
        }
        self.type_depth -= 1;
    }

    fn visit_type_bare_fn(&mut self, i: &TypeBareFn) {
        self.with_bound_lifetimes(i.lifetimes.as_ref(), |v| {
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
        self.with_bound_lifetimes(i.lifetimes.as_ref(), |v| {
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
        self.with_bound_lifetimes(i.lifetimes.as_ref(), |v| {
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
        } else if self.type_depth > 0 {
            let is_self = i.leading_colon.is_none()
                && matches!(i.segments.first(), Some(seg) if seg.ident == "Self" && seg.arguments.is_none());
            if !is_self {
                // The path depends on the type context.
                let external = i.leading_colon.is_some()
                    && matches!(i.segments.first(), Some(seg) if seg.ident != "core");
                self.push_must(quote!(#i), !external);
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
            self.push_must(quote!(#i), true);
        }
    }

    fn visit_expr(&mut self, i: &Expr) {
        if self.type_depth > 0 {
            if let Expr::Closure(_) | Expr::Assign(_) | Expr::Verbatim(_) | Expr::Macro(_) = i {
                self.push_impossible(quote!(#i));
            }
        }
        // The types in an expression of a type belong to the type (see `expr_owner`).
        let outer = self.expr_owner;
        if self.type_depth > 0 && outer.is_none() {
            self.expr_owner = Some(self.type_depth);
        }
        syn::visit::visit_expr(self, i);
        self.expr_owner = outer;
    }

    fn visit_angle_bracketed_generic_arguments(&mut self, i: &AngleBracketedGenericArguments) {
        // A trait object without a lifetime bound, as an argument of a path with lifetime
        // arguments, may default to one of them (e.g. `Ref<'a, dyn Trait>` is `Ref<'a, dyn Trait +
        // 'a>`, because of the bound `T: 'a` of `Ref<'a, T>`), which only the definition of the
        // path tells. So it is treated like the referent of a reference with the lifetime.
        let mut elided = false;
        let mut inherited = false;
        for arg in &i.args {
            if let GenericArgument::Lifetime(lifetime) = arg {
                if lifetime.ident == "_"
                    || self
                        .scope_of(&lifetime.to_string())
                        .map_or(false, |scope| scope > 0)
                {
                    elided = true;
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
                if inherited && !elided {
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

/// Returns `true` if the trait object has a lifetime bound (e.g. `dyn Trait + 'a`), so it does not
/// inherit one.
fn has_lifetime_bound(object: &TypeTraitObject) -> bool {
    object
        .bounds
        .iter()
        .any(|bound| matches!(bound, TypeParamBound::Lifetime(_)))
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

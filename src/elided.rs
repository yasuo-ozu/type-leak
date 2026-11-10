//! The elided lifetime of a signature ([`Elision`]): what the Leaker keeps of it, and how a copy
//! names it where what it writes in the inputs adds lifetimes (E0106).
//!
//! A copy writes the host for `Self`, an argument for a type parameter and a path through the
//! Repeater for an entry: with the host `::p::S<'a>`, `fn(&Self) -> &u8` becomes
//! `for<'e> fn(&'e ::p::S<'a>) -> &'e u8`. Only a signature whose inputs show their one lifetime
//! is named, as the Leaker expects ([`AnalyzeVisitor::self_elision_broken()`]). See
//! [Elided lifetimes](crate::Leaker#elided-lifetimes).

#[cfg(doc)]
use crate::analyze::AnalyzeVisitor;
use crate::context::{Context, Scope};
use crate::error::Cause;
use crate::util::{
    binder_declares, binder_lifetimes, has_lifetime, ident_name, is_bare_self, lifetime_name,
    names_in, same_lifetime, token_names, type_mentions_self, unwrap_type_group, ReplaceSelf,
};
use proc_macro2::Span;
use std::collections::HashSet;
use syn::fold::Fold;
use syn::visit::Visit;
use syn::visit_mut::VisitMut;
use syn::*;
use template_quote::ToTokens;

/// A source of the lifetime that a copy binds.
trait Fresh {
    /// Returns a lifetime that nothing at the place of the signature names, and that `binder` does
    /// not declare.
    fn fresh(&mut self, binder: Option<&BoundLifetimes>) -> Lifetime;
}

/// What a copy writes for the types in the inputs of a signature, before it folds them.
trait Writer: Fresh {
    /// Returns `ty` as the copy writes it. It may be called more than once for a type.
    fn written(&mut self, ty: &Type) -> Type;
}

/// The first lifetime `'__tl_e<n>` that none of the generics in `scopes` declares, nor `binder`,
/// and that is not in `taken`: what the Visitor binds.
fn visitor_lifetime(
    scopes: &[Scope],
    taken: &HashSet<String>,
    binder: Option<&BoundLifetimes>,
) -> Lifetime {
    (0..)
        .map(|n| Lifetime::new(&format!("'__tl_e{}", n), Span::call_site()))
        .find(|name| {
            let text = name.to_string();
            !scopes.iter().any(|scope| scope.declares(&text))
                && !taken.contains(&text)
                && !binder_declares(binder, name)
        })
        .expect("an unbounded range")
}

/// The `Writer` of `Visitor::with_self()`: `Self` is `self_ty`, and the generics in `scopes` are in
/// scope, whose lifetimes the binder must not shadow.
struct SelfWriter<'a> {
    self_ty: &'a Type,
    scopes: &'a [Scope],
}

impl Fresh for SelfWriter<'_> {
    fn fresh(&mut self, binder: Option<&BoundLifetimes>) -> Lifetime {
        visitor_lifetime(self.scopes, &HashSet::new(), binder)
    }
}

impl Writer for SelfWriter<'_> {
    fn written(&mut self, ty: &Type) -> Type {
        ReplaceSelf {
            self_ty: self.self_ty,
        }
        .fold_type(ty.clone())
    }
}

/// The fresh lifetimes of a signature that the Visitor has written, with the paths through the
/// Repeater in it: `'__tl_e<n>` as for `Self`, and none that the signature names already (what the
/// Visitor wrote may name any: a binder inside it is not in `scopes`).
struct WrittenNames<'a> {
    scopes: &'a [Scope],
    taken: HashSet<String>,
}

impl<'a> WrittenNames<'a> {
    /// Names for the signature `written` in `scopes`.
    fn new(scopes: &'a [Scope], written: impl ToTokens) -> Self {
        Self {
            scopes,
            taken: names_in(written),
        }
    }
}

impl Fresh for WrittenNames<'_> {
    fn fresh(&mut self, binder: Option<&BoundLifetimes>) -> Lifetime {
        visitor_lifetime(self.scopes, &self.taken, binder)
    }
}

/// The fresh lifetimes of a copy at a use site (`substitute_args()`, `Substitute`): `'__tl_s<n>`
/// with the first `n` that nothing [reserved](FreshLifetimes::reserve) takes, counting the
/// signatures around, so that a signature in the binder of another gets another name (E0496).
#[derive(Default)]
pub(crate) struct FreshLifetimes {
    taken: HashSet<String>,
    next: usize,
}

impl FreshLifetimes {
    /// Notes the lifetimes that `tokens` name, so that no fresh name is one of them.
    pub(crate) fn reserve(&mut self, tokens: impl ToTokens) {
        token_names(tokens.into_token_stream(), &mut self.taken);
    }

    /// Returns `true` if `lifetime` is a name that this made and has not given back: the library's
    /// own binder, which a site is not asked about.
    pub(crate) fn made(&self, lifetime: &Lifetime) -> bool {
        let ident = ident_name(&lifetime.ident);
        ident
            .strip_prefix("__tl_s")
            .and_then(|n| n.parse::<usize>().ok())
            .map_or(false, |n| {
                n < self.next && !self.taken.contains(&format!("'{}", ident))
            })
    }

    /// Returns the count of the signatures around, to give the names back with
    /// [`FreshLifetimes::release()`] after the signature being copied.
    fn mark(&self) -> usize {
        self.next
    }

    fn release(&mut self, mark: usize) {
        self.next = mark;
    }

    /// Returns a name that is not taken and that `binder` does not declare, and counts it.
    fn next_name(&mut self, binder: Option<&BoundLifetimes>) -> Lifetime {
        loop {
            let name = format!("'__tl_s{}", self.next);
            self.next += 1;
            if !self.taken.contains(&name)
                && !binder_lifetimes(binder).any(|declared| lifetime_name(&declared) == name)
            {
                return Lifetime::new(&name, Span::call_site());
            }
        }
    }
}

/// A substitution that writes a copy at a use site: it replaces names in the types it folds, and
/// names the elided lifetime of a signature where that adds lifetimes to its inputs
/// ([`fold_bare_fn()`], [`fold_bound()`]).
pub(crate) trait Site: Fold + Sized {
    fn names(&mut self) -> &mut FreshLifetimes;
}

impl<S: Site> Fresh for S {
    fn fresh(&mut self, binder: Option<&BoundLifetimes>) -> Lifetime {
        self.names().next_name(binder)
    }
}

impl<S: Site> Writer for S {
    fn written(&mut self, ty: &Type) -> Type {
        // The names that the trial takes are given back: the real fold takes them again.
        let mark = self.names().mark();
        let written = self.fold_type(ty.clone());
        self.names().release(mark);
        written
    }
}

/// Folds the function pointer `i` for [`Site`], with its elided lifetime named if needed.
pub(crate) fn fold_bare_fn(site: &mut impl Site, mut i: TypeBareFn) -> TypeBareFn {
    let mark = site.names().mark();
    name_elided(site, &mut i);
    let i = syn::fold::fold_type_bare_fn(site, i);
    site.names().release(mark);
    i
}

/// Like [`fold_bare_fn()`], for a bound that may be the sugar of `Fn(..) -> R`.
pub(crate) fn fold_bound(site: &mut impl Site, mut i: TraitBound) -> TraitBound {
    let mark = site.names().mark();
    name_elided(site, &mut i);
    let i = syn::fold::fold_trait_bound(site, i);
    site.names().release(mark);
    i
}

/// The binder, the inputs and the output of a signature.
type SigParts<'a> = (
    &'a mut Option<BoundLifetimes>,
    Vec<&'a mut Type>,
    &'a mut ReturnType,
);

/// A function pointer, or a bound that may be `Fn(..) -> R`: a signature whose output may take its
/// elided lifetime from its inputs.
pub(crate) trait FnSig: ToTokens {
    /// The lifetimes that it binds.
    fn binder(&self) -> Option<&BoundLifetimes>;

    /// Its parts, or `None` for a bound that is not `Fn(..) -> R`.
    fn parts(&mut self) -> Option<SigParts<'_>>;
}

impl FnSig for TypeBareFn {
    fn binder(&self) -> Option<&BoundLifetimes> {
        self.lifetimes.as_ref()
    }

    fn parts(&mut self) -> Option<SigParts<'_>> {
        let inputs = self.inputs.iter_mut().map(|arg| &mut arg.ty).collect();
        Some((&mut self.lifetimes, inputs, &mut self.output))
    }
}

impl FnSig for TraitBound {
    fn binder(&self) -> Option<&BoundLifetimes> {
        self.lifetimes.as_ref()
    }

    fn parts(&mut self) -> Option<SigParts<'_>> {
        match self.path.segments.last_mut().map(|seg| &mut seg.arguments) {
            Some(PathArguments::Parenthesized(args)) => Some((
                &mut self.lifetimes,
                args.inputs.iter_mut().collect(),
                &mut args.output,
            )),
            _ => None,
        }
    }
}

/// Before the Visitor folds `sig` in `scopes`: names its elided lifetime if the host `self_ty` adds
/// lifetimes to its inputs, then returns its elision (none once named) for [`after_fold()`], read
/// before the entries are written.
pub(crate) fn before_fold(
    sig: &mut impl FnSig,
    self_ty: Option<&Type>,
    scopes: &[Scope],
) -> Option<Elision> {
    if let Some(self_ty) = self_ty {
        name_elided(&mut SelfWriter { self_ty, scopes }, sig);
    }
    let (_, inputs, output) = sig.parts()?;
    Elision::of_copy(inputs.iter().map(|input| &**input), output)
}

/// After the Visitor folded `sig` in `scopes`: names the elided lifetime of `elision` if the inputs
/// as written show more than their one lifetime.
pub(crate) fn after_fold(sig: &mut impl FnSig, elision: Option<Elision>, scopes: &[Scope]) {
    if let Some(elision) = elision {
        let mut names = WrittenNames::new(scopes, &*sig);
        elision.name_written(&mut names, sig);
    }
}

/// Names the elided lifetime of `sig`, if what `writer` writes for its inputs adds lifetimes to
/// them.
fn name_elided(writer: &mut impl Writer, sig: &mut impl FnSig) {
    let (binder, inputs, output) = match sig.parts() {
        Some(parts) => parts,
        None => return,
    };
    let elision = match Elision::of_copy(inputs.iter().map(|input| &**input), output) {
        Some(elision) => elision,
        None => return,
    };
    // The copy's inputs show that lifetime and no other.
    let written: Vec<Type> = inputs.iter().map(|input| writer.written(input)).collect();
    if InputLifetimes::of(&written).is_one() {
        return;
    }
    elision.name(writer, binder, inputs, output)
}

/// A signature (a function pointer, `Fn(..) -> R` or a method) whose output takes its elided
/// lifetime from the one lifetime of the inputs, which a copy that adds lifetimes to them breaks
/// (E0106). See [Elided lifetimes](crate::Leaker#elided-lifetimes).
#[derive(Clone, Debug)]
pub(crate) struct Elision {
    /// The depth of the type holding the signature, leaked whole instead (0 for a method or a
    /// bound, which no type holds).
    pub(crate) owner: usize,
    /// The only lifetime of the inputs as written, `None` if it is elided or hidden.
    pub(crate) kept: Option<Lifetime>,
    /// The inputs show no lifetime, so the one that the output has is hidden by a path
    /// (`Ref<u8>` for `Ref<'a, u8>`), which the Leaker does not see: any lifetime that the copy
    /// adds to the inputs breaks the elision.
    pub(crate) hidden: bool,
    /// The signature of a function pointer or of `Fn(..)`, whose elided lifetime the Referrer can
    /// name with a `for<'e>` binder where `Visitor::with_self()` adds lifetimes to its inputs (see
    /// [`AnalyzeVisitor::self_elision_broken()`]). A method has none.
    pub(crate) binder: bool,
    /// The input being visited names `kept` ([`Elision::for_input()`]).
    in_kept: bool,
    /// The output, which has the elided lifetime.
    pub(crate) output: Cause,
}

impl Elision {
    /// The elision of a signature with these `inputs` and `output`, if its output has an elided
    /// lifetime and the inputs show exactly one lifetime, or none (a path hides it). `binder`: a
    /// function pointer or `Fn(..)`.
    pub(crate) fn of<'a>(
        owner: usize,
        binder: bool,
        inputs: impl IntoIterator<Item = &'a Type>,
        output: &ReturnType,
    ) -> Option<Self> {
        let output = match output {
            ReturnType::Type(_, output) => output,
            ReturnType::Default => return None,
        };
        // The output has an elided lifetime (a function pointer or `Fn(..)` inside has its own).
        if InputLifetimes::of(std::iter::once(&**output)).elided == 0 {
            return None;
        }
        let mut lifetimes = InputLifetimes::of(inputs);
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
            in_kept: false,
            output: Cause::Type((**output).clone()),
        })
    }

    /// Returns this for the types in `input`, an input of the signature. rustc counts the
    /// lifetimes of each input apart, so `kept` in the copy of a type in another input is one more.
    pub(crate) fn for_input(&self, input: &Type) -> Self {
        let lifetimes = InputLifetimes::of(std::iter::once(input));
        Self {
            in_kept: self
                .kept
                .as_ref()
                .map_or(false, |kept| has_lifetime(&lifetimes.named, kept)),
            ..self.clone()
        }
    }

    /// `Elision::of()` for a method: none if the receiver has a reference to `Self` (`&self`),
    /// which gives the output its lifetime. Any other receiver is left out of the count, as rustc
    /// before 1.81 does (`self: &Box<Self>`).
    pub(crate) fn of_signature(sig: &Signature) -> Option<Self> {
        let mut inputs: Vec<&Type> = Vec::new();
        for input in &sig.inputs {
            match input {
                FnArg::Receiver(receiver) => {
                    let mut references = RefersToSelf(receiver.reference.is_some());
                    references.visit_type(&receiver.ty);
                    if references.0 {
                        return None;
                    }
                }
                FnArg::Typed(arg) => inputs.push(&arg.ty),
            }
        }
        Self::of(0, false, inputs, &sig.output)
    }

    /// Returns the lifetimes that the copy of a type with `context` adds to the input it is in
    /// ([`Elision::for_input()`]): those it names, not `kept` in the input that names it.
    pub(crate) fn added(&self, context: &Context) -> Vec<Lifetime> {
        let mut added: Vec<Lifetime> = Vec::new();
        let lifetimes = context
            .generics
            .lifetimes()
            .map(|param| &param.lifetime)
            .chain(&context.self_outlives);
        for lifetime in lifetimes {
            let kept = self.in_kept
                && self
                    .kept
                    .as_ref()
                    .map_or(false, |kept| same_lifetime(kept, lifetime));
            if !kept && !has_lifetime(&added, lifetime) {
                added.push(lifetime.clone());
            }
        }
        added
    }

    /// Returns the lifetimes that a copy adds to the inputs by writing `self_ty` for `Self`
    /// (`Visitor::with_self()`): those it names, or an elided one (`'_`), not the one the inputs
    /// keep.
    pub(crate) fn added_by_self(&self, self_ty: &Type) -> Vec<Lifetime> {
        let lifetimes = InputLifetimes::of(std::iter::once(self_ty));
        let mut added: Vec<Lifetime> = Vec::new();
        if lifetimes.elided > 0 {
            added.push(Lifetime::new("'_", proc_macro2::Span::call_site()));
        }
        for lifetime in lifetimes.named {
            let kept = self
                .kept
                .as_ref()
                .map_or(false, |kept| same_lifetime(kept, &lifetime));
            if !kept && !has_lifetime(&added, &lifetime) {
                added.push(lifetime);
            }
        }
        added
    }
}

/// The copy's side of an [`Elision`]: it names the lifetime where the inputs as written do not show
/// it alone ([`before_fold()`], [`after_fold()`]).
impl Elision {
    /// `Elision::of()` of a function pointer or `Fn(..)`, unless a path hides the lifetime
    /// (`Ref<u8>`): it cannot be named, and the Leaker leaks such a signature whole or rejects it.
    fn of_copy<'a>(
        inputs: impl IntoIterator<Item = &'a Type>,
        output: &ReturnType,
    ) -> Option<Self> {
        Self::of(0, true, inputs, output).filter(|elision| !elision.hidden)
    }

    /// Names the elided lifetime of `sig`, which the copy has written (what the paths through the
    /// Repeater are in it), if its inputs show more than their one lifetime.
    fn name_written(&self, fresh: &mut impl Fresh, sig: &mut impl FnSig) {
        let (binder, inputs, output) = match sig.parts() {
            Some(parts) => parts,
            None => return,
        };
        let shown = InputLifetimes::of(inputs.iter().map(|input| &**input));
        if shown.is_one() {
            return;
        }
        match &self.kept {
            // The lifetime that the inputs named is in what the Visitor wrote (the entry held it),
            // as the site writes it: what the output takes is not known.
            Some(kept) if !has_lifetime(&shown.named, kept) => return,
            // Only the one elided lifetime that the original had can be named: what was written
            // may have an elided lifetime of its own (`&str` in an argument), which cannot be told
            // from it, and naming both would make them one.
            None if shown.elided != 1 => return,
            _ => (),
        }
        self.name(fresh, binder, inputs, output)
    }

    /// Names the elided lifetime: the output takes the lifetime that the inputs name, else it and
    /// the elided one of the inputs get a lifetime that `binder` declares.
    fn name(
        &self,
        fresh: &mut impl Fresh,
        binder: &mut Option<BoundLifetimes>,
        mut inputs: Vec<&mut Type>,
        output: &mut ReturnType,
    ) {
        let output = match output {
            ReturnType::Type(_, output) => output,
            ReturnType::Default => return,
        };
        let name = match &self.kept {
            Some(kept) => kept.clone(),
            None => {
                let name = fresh.fresh(binder.as_ref());
                let param = GenericParam::Lifetime(LifetimeParam::new(name.clone()));
                match binder {
                    Some(binder) => binder.lifetimes.push(param),
                    None => {
                        *binder = Some(BoundLifetimes {
                            for_token: Default::default(),
                            lt_token: Default::default(),
                            lifetimes: std::iter::once(param).collect(),
                            gt_token: Default::default(),
                        })
                    }
                }
                for input in &mut inputs {
                    Name(&name).visit_type_mut(input);
                }
                name
            }
        };
        Name(&name).visit_type_mut(output);
    }
}

/// The lifetimes of the inputs of a signature as elision counts them: each elided one is its own,
/// the named ones count once. A function pointer or `Fn(..)` inside has its own, and so has a
/// bound with a `for<..>` binder for the lifetimes it declares.
#[derive(Default)]
struct InputLifetimes {
    elided: usize,
    named: Vec<Lifetime>,
    declared: Vec<Lifetime>,
}

impl InputLifetimes {
    /// The lifetimes of `inputs`.
    fn of<'a>(inputs: impl IntoIterator<Item = &'a Type>) -> Self {
        let mut lifetimes = Self::default();
        for input in inputs {
            lifetimes.visit_type(input);
        }
        lifetimes
    }

    /// `true` if the lifetimes are one, as elision counts them: an elided one, or a name (used any
    /// number of times).
    fn is_one(&self) -> bool {
        self.elided + self.named.len() == 1
    }
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
        } else if !has_lifetime(&self.declared, i) && !has_lifetime(&self.named, i) {
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

/// `true` if the type of a receiver has a reference to `Self` (`self: Pin<&mut Self>`), or to
/// another type, taken for the type of the impl (`&S`). A reference to a type that holds `Self`
/// (`&Box<Self>`) is not one before rustc 1.81.
struct RefersToSelf(bool);

impl<'ast> Visit<'ast> for RefersToSelf {
    fn visit_type_reference(&mut self, i: &'ast TypeReference) {
        let elem = unwrap_type_group(&i.elem);
        self.0 |= is_bare_self(elem) || !type_mentions_self(elem);
        syn::visit::visit_type_reference(self, i)
    }
}

/// Replaces the elided lifetimes that elision counts for the signature around (`&` with none,
/// `'_`) with a name. A function pointer or `Fn(..)` inside has its own.
struct Name<'a>(&'a Lifetime);

impl<'a> VisitMut for Name<'a> {
    fn visit_type_reference_mut(&mut self, i: &mut TypeReference) {
        if i.lifetime.is_none() {
            i.lifetime = Some(self.0.clone());
        }
        syn::visit_mut::visit_type_reference_mut(self, i)
    }
    fn visit_lifetime_mut(&mut self, i: &mut Lifetime) {
        if i.ident == "_" {
            *i = self.0.clone();
        }
    }
    fn visit_type_bare_fn_mut(&mut self, _: &mut TypeBareFn) {}
    fn visit_parenthesized_generic_arguments_mut(&mut self, _: &mut ParenthesizedGenericArguments) {
    }
}

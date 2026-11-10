//! The copy of a signature whose elided lifetime the types written in its inputs would break.
//!
//! The output of a function pointer or of `Fn(..) -> R` may have an elided lifetime, which
//! elision takes from the only lifetime of the inputs. A name in the inputs that the copy writes as
//! a type with lifetimes adds some, and rustc cannot tell which one the output has (E0106). That is
//! `Self` for `Visitor::with_self()`: `fn(&Self) -> &u8` is valid in `struct S<'a>`, but
//! `fn(&::p::S<'a>) -> &u8` has two lifetimes in its inputs. It is a type parameter that
//! `substitute_args()` or `complete_params()` replaces, too: `fn(&U) -> &u8` with
//! `U := &'static str`. And it is the path through the Repeater that replaces an entry, whose
//! arguments name the lifetimes of the site: `fn(&Vec<U>) -> &u8` with the entry `Vec<U>` is
//! `fn(&<X as Repeater<(&'static str,), ..>>::Type) -> &u8`, which the Visitor cannot know until
//! `type_fn` has written it. The copy names the lifetime that elision gives instead, with a fresh
//! `for<'e>` binder: `for<'e> fn(&'e ::p::S<'a>) -> &'e u8`.
//!
//! What the copy writes is known before it folds the signature for `Self` and the type parameters
//! (the lifetimes are named first), and after for the entries (they are named once written, in the
//! signature that the original relied on: [`Elision`]). The Leaker does not count such a signature as
//! broken for `Self` (`AnalyzeVisitor::self_elision_broken()`), so the two agree on which signatures
//! this handles: those whose inputs show their one lifetime. See
//! [Elided lifetimes](crate::Leaker#elided-lifetimes).

use crate::analyze::{Elision, InputLifetimes};
use crate::context::{binder_lifetimes, Scope};
use crate::util::ReplaceSelf;
use proc_macro2::{Span, TokenStream, TokenTree};
use std::collections::HashSet;
use syn::fold::Fold;
use syn::visit::Visit;
use syn::visit_mut::VisitMut;
use syn::*;
use template_quote::ToTokens;

/// A source of the lifetime that a copy binds.
pub(crate) trait Fresh {
    /// Returns a lifetime that nothing at the place of the signature names, and that `binder` does
    /// not declare.
    fn fresh(&mut self, binder: Option<&BoundLifetimes>) -> Lifetime;
}

/// What a copy writes for the types in the inputs of a signature, before it folds them.
pub(crate) trait Writer: Fresh {
    /// Returns `ty` as the copy writes it. It may be called more than once for a type.
    fn written(&mut self, ty: &Type) -> Type;
}

/// The first lifetime `'__tl_e<n>` that none of the generics in `scopes` declares, nor `binder`, and
/// that is not in `taken`: what the Visitor binds.
fn visitor_name(
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
                && !binder_lifetimes(binder).any(|declared| declared == *name)
        })
        .expect("an unbounded range")
}

/// The `Writer` of `Visitor::with_self()`: `Self` is `self_ty`, and the generics in `scopes` are in
/// scope, whose lifetimes the binder must not shadow.
pub(crate) struct SelfWriter<'a> {
    pub(crate) self_ty: &'a Type,
    pub(crate) scopes: &'a [Scope],
}

impl Fresh for SelfWriter<'_> {
    fn fresh(&mut self, binder: Option<&BoundLifetimes>) -> Lifetime {
        visitor_name(self.scopes, &HashSet::new(), binder)
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

/// The fresh lifetimes of a signature that the Visitor has written, with the types that `type_fn`
/// returned in it: `'__tl_e<n>` as for `Self`, and none that the signature names already (what
/// `type_fn` wrote may name any: a binder inside it is not in `scopes`).
pub(crate) struct WrittenNames<'a> {
    scopes: &'a [Scope],
    taken: HashSet<String>,
}

impl<'a> WrittenNames<'a> {
    /// Names for the signature `written` in `scopes`.
    pub(crate) fn new(scopes: &'a [Scope], written: impl ToTokens) -> Self {
        let mut taken = HashSet::new();
        collect_lifetimes(written.into_token_stream(), &mut taken);
        Self { scopes, taken }
    }
}

impl Fresh for WrittenNames<'_> {
    fn fresh(&mut self, binder: Option<&BoundLifetimes>) -> Lifetime {
        visitor_name(self.scopes, &self.taken, binder)
    }
}

/// Adds the lifetimes that `tokens` name to `names`.
fn collect_lifetimes(tokens: TokenStream, names: &mut HashSet<String>) {
    let mut tokens = tokens.into_iter().peekable();
    while let Some(tree) = tokens.next() {
        match tree {
            // A lifetime is a `'` joined to the identifier after it.
            TokenTree::Punct(punct) if punct.as_char() == '\'' => {
                if let Some(TokenTree::Ident(ident)) = tokens.peek() {
                    names.insert(format!("'{}", ident));
                }
            }
            TokenTree::Group(group) => collect_lifetimes(group.stream(), names),
            _ => (),
        }
    }
}

/// The fresh lifetimes of the copy of a type at a use site (`substitute_args()`,
/// `complete_params()`).
///
/// The generics in scope there are not known, so a name is `'__tl_s<n>` for the first `n` that no
/// lifetime in what the copy is made of (the arguments, what replaces a name) is named
/// like ([`FreshLifetimes::reserve()`]); the prefix keeps it from the lifetimes that a macro declares and
/// from the Visitor's (`'__tl_e<n>`). `n` counts the signatures around, so that a signature inside
/// the binder of another gets another name (E0496), and the output is the same for the same input.
#[derive(Default)]
pub(crate) struct FreshLifetimes {
    taken: HashSet<String>,
    next: usize,
}

impl FreshLifetimes {
    /// Notes the lifetimes that `tokens` name, so that no fresh name is one of them.
    pub(crate) fn reserve(&mut self, tokens: impl ToTokens) {
        collect_lifetimes(tokens.into_token_stream(), &mut self.taken);
    }

    /// Returns `true` if `lifetime` is a name that this made and has not given back: the library's
    /// own binder, which a site is not asked about.
    pub(crate) fn made(&self, lifetime: &Lifetime) -> bool {
        let ident = lifetime.ident.to_string();
        ident
            .strip_prefix("__tl_s")
            .and_then(|n| n.parse::<usize>().ok())
            .map_or(false, |n| {
                n < self.next && !self.taken.contains(&format!("'{}", ident))
            })
    }

    /// Returns the count of the signatures around, to give the names back with
    /// [`FreshLifetimes::release()`] after the signature being copied.
    pub(crate) fn mark(&self) -> usize {
        self.next
    }

    pub(crate) fn release(&mut self, mark: usize) {
        self.next = mark;
    }

    /// Returns a name that is not taken and that `binder` does not declare, and counts it.
    fn next_name(&mut self, binder: Option<&BoundLifetimes>) -> Lifetime {
        loop {
            let name = format!("'__tl_s{}", self.next);
            self.next += 1;
            if !self.taken.contains(&name)
                && !binder_lifetimes(binder).any(|declared| declared.to_string() == name)
            {
                return Lifetime::new(&name, Span::call_site());
            }
        }
    }
}

impl Fresh for FreshLifetimes {
    fn fresh(&mut self, binder: Option<&BoundLifetimes>) -> Lifetime {
        self.next_name(binder)
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
    name_elided_in_bare_fn(site, &mut i);
    let i = syn::fold::fold_type_bare_fn(site, i);
    site.names().release(mark);
    i
}

/// Like [`fold_bare_fn()`], for a bound that may be the sugar of `Fn(..) -> R`.
pub(crate) fn fold_bound(site: &mut impl Site, mut i: TraitBound) -> TraitBound {
    let mark = site.names().mark();
    name_elided_in_bound(site, &mut i);
    let i = syn::fold::fold_trait_bound(site, i);
    site.names().release(mark);
    i
}

/// Names the elided lifetime of the signature of the function pointer `i`, if what `writer` writes
/// for its inputs adds lifetimes to them.
pub(crate) fn name_elided_in_bare_fn(writer: &mut impl Writer, i: &mut TypeBareFn) {
    name_elided(
        writer,
        &mut i.lifetimes,
        i.inputs.iter_mut().map(|arg| &mut arg.ty).collect(),
        &mut i.output,
    )
}

/// Like [`name_elided_in_bare_fn()`], for the signature of `Fn(..) -> R` (the last segment of the
/// path of `i` has parenthesized arguments), which `i` binds.
pub(crate) fn name_elided_in_bound(writer: &mut impl Writer, i: &mut TraitBound) {
    if let Some(PathArguments::Parenthesized(args)) =
        i.path.segments.last_mut().map(|seg| &mut seg.arguments)
    {
        name_elided(
            writer,
            &mut i.lifetimes,
            args.inputs.iter_mut().collect(),
            &mut args.output,
        )
    }
}

fn name_elided(
    writer: &mut impl Writer,
    binder: &mut Option<BoundLifetimes>,
    inputs: Vec<&mut Type>,
    output: &mut ReturnType,
) {
    let elision = match Elision::of_copy(inputs.iter().map(|input| &**input), output) {
        Some(elision) => elision,
        None => return,
    };
    // The copy's inputs show that lifetime and no other.
    let mut written = InputLifetimes::default();
    for input in &inputs {
        written.visit_type(&writer.written(input));
    }
    if shows_one(&written) {
        return;
    }
    elision.name(writer, binder, inputs, output)
}

/// `true` if the lifetimes of the inputs are one, as elision counts them: an elided one, or a name
/// (used any number of times).
fn shows_one(lifetimes: &InputLifetimes) -> bool {
    lifetimes.elided + lifetimes.named.len() == 1
}

/// What the copy does with an [`Elision`] (found before it writes the signature): it checks that the
/// inputs still show their one lifetime ([`name_elided_in_bare_fn()`] before it folds them,
/// [`Elision::name_written_bare_fn()`] after), and names it if not.
impl Elision {
    /// The elision of the function pointer `i`, if its signature relies on it.
    pub(crate) fn of_bare_fn(i: &TypeBareFn) -> Option<Self> {
        Self::of_copy(i.inputs.iter().map(|arg| &arg.ty), &i.output)
    }

    /// Like [`Elision::of_bare_fn()`], for the signature of `Fn(..) -> R` that `i` binds.
    pub(crate) fn of_bound(i: &TraitBound) -> Option<Self> {
        match i.path.segments.last().map(|seg| &seg.arguments) {
            Some(PathArguments::Parenthesized(args)) => {
                Self::of_copy(args.inputs.iter(), &args.output)
            }
            _ => None,
        }
    }

    /// The inputs show their one lifetime. With none, a path hides it (`Ref<u8>`): it cannot be
    /// named, and the Leaker leaks such a signature whole or rejects it. With more, the original is
    /// not valid.
    fn of_copy<'a>(
        inputs: impl IntoIterator<Item = &'a Type>,
        output: &ReturnType,
    ) -> Option<Self> {
        Self::of(0, true, inputs, output).filter(|elision| !elision.hidden)
    }

    /// Names the elided lifetime of the function pointer `i`, which the copy has written (what
    /// `type_fn` returned is in it), if its inputs show more than their one lifetime.
    pub(crate) fn name_written_bare_fn(&self, fresh: &mut impl Fresh, i: &mut TypeBareFn) {
        self.name_written(
            fresh,
            &mut i.lifetimes,
            i.inputs.iter_mut().map(|arg| &mut arg.ty).collect(),
            &mut i.output,
        )
    }

    /// Like [`Elision::name_written_bare_fn()`], for the signature of `Fn(..) -> R` that `i` binds.
    pub(crate) fn name_written_bound(&self, fresh: &mut impl Fresh, i: &mut TraitBound) {
        if let Some(PathArguments::Parenthesized(args)) =
            i.path.segments.last_mut().map(|seg| &mut seg.arguments)
        {
            self.name_written(
                fresh,
                &mut i.lifetimes,
                args.inputs.iter_mut().collect(),
                &mut args.output,
            )
        }
    }

    fn name_written(
        &self,
        fresh: &mut impl Fresh,
        binder: &mut Option<BoundLifetimes>,
        inputs: Vec<&mut Type>,
        output: &mut ReturnType,
    ) {
        let mut shown = InputLifetimes::default();
        for input in &inputs {
            shown.visit_type(input);
        }
        if shows_one(&shown) {
            return;
        }
        match &self.kept {
            // The lifetime that the inputs named is in what `type_fn` wrote (the entry held it), as
            // the site writes it: what the output takes is not known.
            Some(kept) if !shown.named.contains(kept) => return,
            // Only the one elided lifetime that the original had can be named: `type_fn` may have
            // written an elided lifetime of its own (`&str` in an argument), which cannot be told
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

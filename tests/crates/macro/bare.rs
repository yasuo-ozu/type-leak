//! The bare and the absolute spelling of a trait of the standard prelude, for `prop.rs`.
//!
//! The generator writes `Sized`, `?Sized`, `Send`, `Fn(..)`, `Into<..>`, `Iterator`, ... in bounds,
//! where clauses, supertraits and trait objects, sometimes bare and sometimes in full
//! (`::core::marker::Sized`). The Leaker leaves a bare one as written only if the configuration
//! allows the names of the prelude (`Rules::prelude()`), so:
//!
//! - With the prelude allowed, the item with the bare spelling and the same item with the absolute
//!   one (its twin) must give the same entries, the same contexts and the same errors: the spelling
//!   of a context-free trait is not a reason to leak anything else.
//! - Without it, the bare spelling is a relative path that cannot be leaked: the item is rejected,
//!   and for every bare name in it there is a "use absolute path" error that names it.
//!
//! A bare `?Sized` is the exception (`?` only applies to the real `Sized`): the Leaker accepts it
//! whatever the prelude, so its twin is compared too.

use std::collections::BTreeSet;
use syn::punctuated::Punctuated;
use syn::visit::Visit;
use syn::visit_mut::VisitMut;
use syn::*;
use template_quote::quote;
use type_leak::*;

/// The traits the generator spells bare, and their absolute paths. The generator's own table
/// (`PRELUDE_TRAITS` in `tests/prop_cross_crate/gen.rs`) is a subset; a unit test of the harness
/// checks it.
pub(crate) const PRELUDE_TRAITS: &[(&str, &str)] = &[
    ("Sized", "::core::marker::Sized"),
    ("Send", "::core::marker::Send"),
    ("Sync", "::core::marker::Sync"),
    ("Unpin", "::core::marker::Unpin"),
    ("Copy", "::core::marker::Copy"),
    ("Clone", "::core::clone::Clone"),
    ("Default", "::core::default::Default"),
    ("Drop", "::core::ops::Drop"),
    ("Fn", "::core::ops::Fn"),
    ("FnMut", "::core::ops::FnMut"),
    ("FnOnce", "::core::ops::FnOnce"),
    ("Into", "::core::convert::Into"),
    ("From", "::core::convert::From"),
    ("AsRef", "::core::convert::AsRef"),
    ("AsMut", "::core::convert::AsMut"),
    ("Iterator", "::core::iter::Iterator"),
    ("IntoIterator", "::core::iter::IntoIterator"),
    ("Extend", "::core::iter::Extend"),
    ("DoubleEndedIterator", "::core::iter::DoubleEndedIterator"),
    ("ExactSizeIterator", "::core::iter::ExactSizeIterator"),
    ("PartialEq", "::core::cmp::PartialEq"),
    ("PartialOrd", "::core::cmp::PartialOrd"),
    ("Eq", "::core::cmp::Eq"),
    ("Ord", "::core::cmp::Ord"),
];

/// The name of the trait that a one-segment relative path spells, if it is one of the prelude's.
fn bare_name(segment: &PathSegment) -> Option<&'static str> {
    PRELUDE_TRAITS
        .iter()
        .find(|(name, _)| segment.ident == name)
        .map(|(name, _)| *name)
}

fn bare_bound(bound: &TraitBound) -> Option<&'static str> {
    if bound.path.leading_colon.is_some() || bound.path.segments.len() != 1 {
        return None;
    }
    bare_name(bound.path.segments.first()?)
}

/// `name`, or `?name` for a `?Sized`.
fn spelled(name: &str, bound: &TraitBound) -> String {
    match bound.modifier {
        TraitBoundModifier::Maybe(_) => format!("?{}", name),
        TraitBoundModifier::None => name.to_string(),
    }
}

/// The bare trait of a qualified path (`<T as Iterator>::Item`).
fn bare_qualified(ty: &TypePath) -> Option<&'static str> {
    let qself = ty.qself.as_ref()?;
    if qself.position != 1 || ty.path.leading_colon.is_some() {
        return None;
    }
    bare_name(ty.path.segments.first()?)
}

/// The bare spellings in an item.
#[derive(Default)]
pub(crate) struct Spellings {
    /// In the bounds of a generic, a where predicate, a supertrait, the bounds of an associated
    /// type: `Sized`, `?Sized`, `Fn`, ... The copy has them as written, so without the prelude
    /// they are relative paths that the Leaker rejects.
    pub(crate) bounds: BTreeSet<String>,
    /// In a type: the bound of a trait object or of an `impl Trait`, the trait of a qualified path.
    /// A trait object that the Leaker leaks as a whole (the lifetime in its sugar of `Fn` is
    /// elided) is written by the Repeater's impl only, so there the Leaker accepts them without
    /// the prelude. They are compared with their twin like the others.
    pub(crate) types: BTreeSet<String>,
}

impl Spellings {
    fn is_empty(&self) -> bool {
        self.bounds.is_empty() && self.types.is_empty()
    }
}

/// The bare spellings in `item`.
///
/// With `fragments_only`, in the fragments of an impl that `enter_impl` visits (the self type and
/// the arguments of the trait): its generics and items are not analyzed there.
pub(crate) fn spellings(item: &Item, fragments_only: bool) -> Spellings {
    #[derive(Default)]
    struct Found {
        found: Spellings,
        /// How many trait objects and `impl Trait` the visit is in.
        in_type: usize,
    }
    impl<'ast> Visit<'ast> for Found {
        fn visit_trait_bound(&mut self, i: &'ast TraitBound) {
            if let Some(name) = bare_bound(i) {
                let set = if self.in_type > 0 {
                    &mut self.found.types
                } else {
                    &mut self.found.bounds
                };
                set.insert(spelled(name, i));
            }
            syn::visit::visit_trait_bound(self, i)
        }
        fn visit_type_trait_object(&mut self, i: &'ast TypeTraitObject) {
            self.in_type += 1;
            syn::visit::visit_type_trait_object(self, i);
            self.in_type -= 1;
        }
        fn visit_type_impl_trait(&mut self, i: &'ast TypeImplTrait) {
            self.in_type += 1;
            syn::visit::visit_type_impl_trait(self, i);
            self.in_type -= 1;
        }
        fn visit_type_path(&mut self, i: &'ast TypePath) {
            if let Some(name) = bare_qualified(i) {
                self.found.types.insert(name.to_string());
            }
            syn::visit::visit_type_path(self, i)
        }
    }
    let mut found = Found::default();
    match item {
        Item::Impl(i) if fragments_only => {
            found.visit_type(&i.self_ty);
            if let Some((_, path, _)) = &i.trait_ {
                for seg in &path.segments {
                    found.visit_path_arguments(&seg.arguments);
                }
            }
        }
        _ => found.visit_item(item),
    }
    found.found
}

/// Writes the bare traits in full. With `only_relaxed_sized`, only a bare `?Sized`.
pub(crate) struct Absolute {
    pub(crate) only_relaxed_sized: bool,
}

fn absolute_path(name: &str) -> Path {
    let (_, absolute) = PRELUDE_TRAITS
        .iter()
        .find(|(n, _)| *n == name)
        .expect("a prelude trait");
    parse_str(absolute).expect("an absolute path")
}

impl VisitMut for Absolute {
    fn visit_trait_bound_mut(&mut self, i: &mut TraitBound) {
        syn::visit_mut::visit_trait_bound_mut(self, i);
        let Some(name) = bare_bound(i) else { return };
        let maybe = matches!(i.modifier, TraitBoundModifier::Maybe(_));
        if self.only_relaxed_sized && !(maybe && name == "Sized") {
            return;
        }
        let arguments = i.path.segments.first().unwrap().arguments.clone();
        let mut path = absolute_path(name);
        path.segments.last_mut().unwrap().arguments = arguments;
        i.path = path;
    }

    fn visit_type_path_mut(&mut self, i: &mut TypePath) {
        syn::visit_mut::visit_type_path_mut(self, i);
        let Some(name) = bare_qualified(i) else {
            return;
        };
        if self.only_relaxed_sized {
            return;
        }
        let mut path = absolute_path(name);
        path.segments.last_mut().unwrap().arguments =
            i.path.segments.first().unwrap().arguments.clone();
        let position = path.segments.len();
        path.segments
            .extend(i.path.segments.iter().skip(1).cloned());
        i.path = path;
        i.qself.as_mut().unwrap().position = position;
    }
}

pub(crate) fn absolute_item(item: &Item, only_relaxed_sized: bool) -> Item {
    let mut item = item.clone();
    Absolute { only_relaxed_sized }.visit_item_mut(&mut item);
    item
}

fn absolute_text<T: Clone>(
    value: &T,
    only_relaxed_sized: bool,
    visit: impl FnOnce(&mut Absolute, &mut T),
) -> T {
    let mut value = value.clone();
    visit(&mut Absolute { only_relaxed_sized }, &mut value);
    value
}

/// The text with the spaces removed, and the absolute paths of the traits written bare, so that two
/// spellings give the same text. The parentheses go: the Leaker parenthesizes the output of a `Fn`
/// bound (`Fn(A) -> (&'static dyn X) + Y`) where another bound follows it, which depends on the order
/// of the bounds.
fn bare_text(text: &str) -> String {
    let mut text: String = text.replace(char::is_whitespace, "");
    for (name, absolute) in PRELUDE_TRAITS {
        text = text.replace(absolute, name);
    }
    text.replace(['(', ')'], "")
}

/// The bounds in the order of their text: the Leaker orders them by what they are spelled like.
fn sorted_bounds<P: Default>(
    bounds: &Punctuated<TypeParamBound, P>,
) -> Punctuated<TypeParamBound, P> {
    let mut bounds: Vec<TypeParamBound> = bounds.iter().cloned().collect();
    bounds.sort_by_key(|bound| bare_text(&quote!(#bound).to_string()));
    // The Leaker drops a bound that is written twice the same way: `Clone + Clone` is one, where
    // `Clone + ::core::clone::Clone` are two.
    bounds.dedup_by_key(|bound| bare_text(&quote!(#bound).to_string()));
    bounds.into_iter().collect()
}

/// The generics of an entry as text, with the traits spelled bare and in an order that does not
/// depend on the spelling: the bounds of each parameter and predicate, and the predicates.
fn generics_text(generics: &Generics, only_relaxed_sized: bool) -> String {
    let mut generics = absolute_text(generics, only_relaxed_sized, |v, g| v.visit_generics_mut(g));
    for param in generics.params.iter_mut() {
        if let GenericParam::Type(t) = param {
            t.bounds = sorted_bounds(&t.bounds);
        }
    }
    let mut predicates: Vec<String> = Vec::new();
    for predicate in generics
        .where_clause
        .iter_mut()
        .flat_map(|clause| clause.predicates.iter_mut())
    {
        match predicate {
            WherePredicate::Type(t) => t.bounds = sorted_bounds(&t.bounds),
            WherePredicate::Lifetime(_) => (),
            _ => (),
        }
        predicates.push(bare_text(&quote!(#predicate).to_string()));
    }
    predicates.sort();
    let params = &generics.params;
    bare_text(&format!("{} ; {}", quote!(#params), predicates.join(",")))
}

/// What a Referrer is made of, one line per entry, sorted, with the traits spelled bare (the
/// entries of two spellings of an item are compared with it).
fn summary(referrer: &Referrer, only_relaxed_sized: bool) -> Vec<String> {
    let mut lines: Vec<String> = referrer
        .iter()
        .map(|(_, ty, context)| {
            let ty = absolute_text(&ty.clone(), only_relaxed_sized, |v, t| v.visit_type_mut(t));
            let generics = generics_text(&context.generics, only_relaxed_sized);
            let self_text = match context.self_ty.as_ref() {
                None => "-".to_string(),
                Some(SelfRequirement::Bound(path)) => format!("bound {}", quote!(#path)),
                Some(SelfRequirement::Type(ty)) => {
                    let ty = absolute_text(ty, only_relaxed_sized, |v, t| v.visit_type_mut(t));
                    format!("type {}", quote!(#ty))
                }
                Some(_) => "other".to_string(),
            };
            bare_text(&format!(
                "{} | {} | self={} | outlives={}",
                quote!(#ty),
                generics,
                self_text,
                context.self_outlives.len(),
            ))
        })
        .collect();
    // The entries are counted per context, and the contexts are keyed by the text of their generics:
    // a bound spelled bare in one item and in full in another makes two contexts, and two equal
    // lines, where the twin with the traits in full has one. So the set of lines is compared.
    lines.sort();
    lines.dedup();
    lines
}

/// The messages of an error, in the text of `bare_text()`, sorted. The headline of a type in a
/// bound that names what it depends on says which of the bounds it found first, which depends on
/// the order the bounds are in: the message stops before it. So does the `Self::A` that a bound
/// of the generics names (the first one in the order of their text): only where it is is kept.
fn messages(error: &[ErrorItem]) -> Vec<String> {
    let mut messages: Vec<String> = error
        .iter()
        .map(|item| {
            let mut text = bare_text(&item.message);
            if let Some(at) = text.find(",andaboundofitsgenericsnames") {
                text.truncate(at);
            }
            if let Some(at) = text.find("inaboundofthegenericsof") {
                if text.starts_with("write`<Selfas") {
                    let end = text.find(":thecopycannottell").unwrap_or(text.len());
                    text = format!("write`<Selfas..>::..`{}", &text[at..end]);
                }
            }
            text
        })
        .collect();
    messages.sort();
    messages
}

/// The result of an analysis, as `prop::analyze()` returns it.
pub(crate) type Analysis =
    std::result::Result<(Referrer, ContextId, proc_macro2::TokenStream), Vec<ErrorItem>>;

fn disagree(why: String) -> Error {
    Error::new(
        proc_macro2::Span::call_site(),
        format!("BARE AND ABSOLUTE SPELLINGS DISAGREE: {}", why),
    )
}

/// Checks the bare spellings of `item` against what the configuration says. It returns whether the
/// item was compared with its twin.
///
/// `analyze` runs the analysis of the item with the arguments of the macro (the item it is given is
/// the item itself, or its twin). `prelude`: the Leaker allows the names of the prelude. A bare
/// `?Sized` is always accepted.
pub(crate) fn check(
    item: &Item,
    fragments_only: bool,
    prelude: bool,
    result: &Analysis,
    analyze: impl Fn(&Item) -> Analysis,
) -> Result<bool> {
    let found = spellings(item, fragments_only);
    if found.is_empty() {
        return Ok(false);
    }
    let bare: BTreeSet<&String> = found.bounds.iter().chain(&found.types).collect();
    // Spelled bare in a bound, the names that the configuration does not allow are relative
    // paths.
    let denied: BTreeSet<&String> = found
        .bounds
        .iter()
        .filter(|name| !prelude && name.as_str() != "?Sized")
        .collect();
    if !denied.is_empty() {
        let error = match result {
            Ok(_) => {
                return Err(disagree(format!(
                    "the item is accepted, with the bare {:?} and without the prelude",
                    denied
                )))
            }
            Err(error) => error,
        };
        let reported: Vec<String> = error
            .iter()
            .filter(|item| item.message.starts_with("use absolute path: '"))
            .map(|item| {
                item.message["use absolute path: '".len()..].replace(char::is_whitespace, "")
            })
            // The tokens of a bound with a binder start with it (`for<'h>Fn(..)`).
            .map(|text| match text.strip_prefix("for<") {
                Some(rest) => rest
                    .split_once('>')
                    .map_or(rest, |(_, rest)| rest)
                    .to_string(),
                None => text,
            })
            .collect();
        for name in denied {
            let named = reported.iter().any(|text| {
                text.strip_prefix(name.as_str()).is_some_and(|rest| {
                    !rest.starts_with(|c: char| c.is_alphanumeric() || c == '_')
                })
            });
            if !named {
                return Err(disagree(format!(
                    "the bare `{}` without the prelude is not reported as a relative path: the \
                     errors are {:?}",
                    name,
                    error.iter().map(|item| &item.message).collect::<Vec<_>>()
                )));
            }
        }
        return Ok(false);
    }
    // Every bare name is allowed: the twin that spells them in full must give the same entries.
    let only_relaxed_sized = !prelude;
    let twin = analyze(&absolute_item(item, only_relaxed_sized));
    match (result, &twin) {
        (Ok((bare_referrer, ..)), Ok((twin_referrer, ..))) => {
            let (a, b) = (
                summary(bare_referrer, only_relaxed_sized),
                summary(twin_referrer, only_relaxed_sized),
            );
            if a != b {
                let first = a
                    .iter()
                    .zip(&b)
                    .find(|(x, y)| x != y)
                    .map(|(x, y)| format!("`{}` and `{}`", x, y))
                    .unwrap_or_else(|| format!("{} and {} lines", a.len(), b.len()));
                return Err(disagree(format!(
                    "the entries of {:?} differ from those of its twin with the traits in full, \
                     first at {}",
                    bare, first
                )));
            }
        }
        (Err(a), Err(b)) => {
            if messages(a) != messages(b) {
                return Err(disagree(format!(
                    "the errors of {:?} are {:?}, and its twin's are {:?}",
                    bare,
                    messages(a),
                    messages(b)
                )));
            }
        }
        (Ok(_), Err(error)) | (Err(error), Ok(_)) => {
            return Err(disagree(format!(
                "one of {:?} and its twin with the traits in full is accepted, and the other is \
                 rejected with {:?}",
                bare,
                messages(error)
            )))
        }
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The Referrer of the structs, analyzed by one Leaker.
    fn referrer(items: &[Item]) -> Referrer {
        let mut leaker = Leaker::with_rule(Rules::new().prelude());
        for item in items {
            match item {
                Item::Struct(item) => leaker.intern_struct(item),
                _ => unreachable!("not a struct"),
            };
        }
        leaker.result().finish().unwrap()
    }

    /// Two items spell the same bound differently, so the bare items have two contexts with the
    /// entry `Vec<U>` where their twins with the traits in full have one: the same set of lines.
    #[test]
    fn contexts_that_differ_in_spelling_give_one_line() {
        let items: Vec<Item> = vec![
            parse_quote!(
                pub struct A<U: ::core::default::Default>(::std::vec::Vec<U>);
            ),
            parse_quote!(
                pub struct B<U: Default>(::std::vec::Vec<U>);
            ),
        ];
        let twins: Vec<Item> = items
            .iter()
            .map(|item| absolute_item(item, false))
            .collect();
        let bare = summary(&referrer(&items), false);
        let twin = summary(&referrer(&twins), false);
        assert!(!bare.is_empty());
        assert_eq!(bare, twin);
    }
}

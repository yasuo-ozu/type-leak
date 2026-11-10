//! A name that a generic shares with something else, and the Leaker and the Referrer's visitor
//! agreeing on which is meant. The findings of the property test (`tests/prop_cross_crate`), each as
//! a round trip Leaker -> Referrer (through tokens) -> visitor, which must replace every entry the
//! Leaker recorded: an entry that the visitor misses stays in the copy as written, and the
//! Referrer's crate cannot resolve it.
//!
//! - `path_named_like_generic`: a path whose last segment is the name of a generic in scope
//!   (`self::U`, `crate::U`, `super::U` next to a generic `U`) is a relative type, not the generic. The
//!   Leaker records it with no generics, so the Referrer must compute the same context for it: it
//!   takes the names a type can mention a generic by (the first segment of a path), not every
//!   identifier of the type. (Format 9 fixed it: the Referrer found the generic `U` in `self::U`,
//!   computed another context than the Leaker's, found no entry and left `self::U` in the copy,
//!   E0425 in the Referrer's crate.)
//! - `declared_named_like_generic`: a file that declares a type `X` hides the entries that name
//!   `X`, because the copy declares `X` again and the entry was written for another `X`. But `X` in
//!   `crate::base::B1<X>` of `fn f<X>` is the generic of the entry, which the declared item does
//!   not hide where the entry is written. (Format 10 fixed it: the visitor hid the entry, and the
//!   Referrer's crate kept `crate::base`, E0433.)
//!
//! The same shapes are copied across crates in `tests/crates/leaker` (`kinds`), where the Repeater
//! impls and the copies are compiled: `tests/crates/referrer/tests/cross_crate.rs`.

use crate::common;

use common::{messages, s};

use std::cell::RefCell;
use std::collections::BTreeSet;
use syn::visit_mut::VisitMut;
use syn::*;
use template_quote::quote;
use type_leak::*;

/// The Referrer as the macro of the Referrer's crate reads it.
fn through_tokens(referrer: Referrer) -> Referrer {
    let tokens = quote!(#referrer);
    let read: Referrer = parse2(tokens).unwrap();
    assert_eq!(read, referrer);
    read
}

fn set(indices: &[usize]) -> BTreeSet<usize> {
    indices.iter().copied().collect()
}

/// The indices of the entries, which are the ones a visitor replaces.
fn all_entries(referrer: &Referrer) -> BTreeSet<usize> {
    (0..referrer.iter().count()).collect()
}

/// The index of the entry `ty` with `generics` (as tokens).
fn index(referrer: &Referrer, ty: &str, generics: &str) -> usize {
    let (ty, generics) = (s(&parse_str::<Type>(ty).unwrap()), generics.to_string());
    let found: Vec<usize> = (0..referrer.iter().count())
        .filter(|&ix| {
            s(&common::types(referrer).nth(ix).unwrap()) == ty
                && s(&common::entry(referrer, ix).generics).replace(' ', "")
                    == generics.replace(' ', "")
        })
        .collect();
    assert_eq!(
        found.len(),
        1,
        "entry `{}` with `{}` in {:?}",
        ty,
        generics,
        referrer
            .iter()
            .map(|(_, ty, context)| (ty, &context.generics))
            .map(|(ty, generics)| (s(ty), s(&generics)))
            .collect::<Vec<_>>()
    );
    found[0]
}

/// Replaces the entries a visitor of `referrer` finds in what `visit` visits, with `Replaced<ix>`,
/// and returns the indices it found.
fn replace(
    referrer: &Referrer,
    context: ContextId,
    visit: impl FnOnce(&mut common::Replacing<'_, &mut dyn FnMut(Type, EntryId, &Context) -> Type>),
) -> BTreeSet<usize> {
    let found = RefCell::new(BTreeSet::new());
    let mut write = |_: Type, ix: EntryId, _: &Context| -> Type {
        found.borrow_mut().insert(ix.index());
        parse_quote!(Replaced<#ix>)
    };
    let mut visitor = common::visitor(
        referrer,
        context,
        &mut write as &mut dyn FnMut(Type, EntryId, &Context) -> Type,
    );
    visit(&mut visitor);
    drop(visitor);
    found.into_inner()
}

/// How an item is analyzed: by its `intern_*` method, or through the scope of a macro.
#[derive(Clone, Copy, Debug)]
enum Scope {
    Intern,
    /// `enter_generics(&Generics::default(), |view| view.visit_item(&item))`: the item is nested,
    /// and the copy is a `mod` that holds it.
    Nested,
    /// `enter_generics(&generics, ..)` for a struct, `enter_trait(..)` for a trait.
    Generics,
}

/// Analyzes the item `source` in `scope`, copies it with the visitor of the Referrer, and checks
/// that the visitor found every required entry, so that the copy has no path left.
fn round_trip(source: &str, scope: Scope) {
    let item: Item = parse_str(source).unwrap();
    let mut leaker = Leaker::with_rule(common::std_config());
    let context = match (scope, &item) {
        (Scope::Intern, _) => common::intern_item(&mut leaker, &item),
        (Scope::Nested, _) => {
            leaker
                .enter_generics(&Generics::default(), |view| view.visit_item(&item))
                .0
        }
        (Scope::Generics, Item::Struct(i)) => {
            leaker
                .enter_generics(&i.generics, |view| syn::visit::visit_item_struct(view, i))
                .0
        }
        (Scope::Generics, Item::Trait(i)) => {
            leaker
                .enter_trait(&i.ident, &i.generics, |view| {
                    syn::visit::visit_item_trait(view, i)
                })
                .0
        }
        _ => unreachable!(),
    };
    let result = leaker.result();
    let errors = result.errors();
    assert!(
        errors.is_empty(),
        "{:?} {}: {}",
        scope,
        source,
        messages(errors)
    );
    let referrer = through_tokens(result.finish().unwrap());
    assert!(!all_entries(&referrer).is_empty(), "{:?} {}", scope, source);
    let mut copy = item.clone();
    let found = replace(&referrer, context, |visitor| match scope {
        Scope::Nested => {
            let mut module: ItemMod = parse_quote!(mod w { #copy });
            visitor.visit_item_mod_mut(&mut module);
            copy = module.content.unwrap().1.remove(0);
        }
        _ => visitor.visit_item_mut(&mut copy),
    });
    assert_eq!(
        found,
        all_entries(&referrer),
        "{:?}: the visitor misses an entry of {}\n  copy: {}",
        scope,
        source,
        s(&copy)
    );
    assert!(
        !s(&copy).contains("::"),
        "{:?}: a path is left in the copy of {}\n  copy: {}",
        scope,
        source,
        s(&copy)
    );
}

// ── path_named_like_generic ──────────────────────────────────────────────────────────────────

/// The paths that end with the name of a generic.
const PATHS: &[&str] = &[
    "self::U",
    "crate::U",
    "super::U",
    "inner::U",
    "self::inner::U",
    "crate::inner::U",
];

/// The types that hold the path `{P}`, next to the generic `U`.
const SHAPES: &[&str] = &[
    "{P}",
    "Vec<{P}>",
    "({P}, U)",
    "&'static {P}",
    "[{P}; 2]",
    "Option<Vec<{P}>>",
    "fn({P}) -> U",
];

/// A struct with the generic `U` and a field of the path `self::U`: the repro of the property test
/// (`struct S<U>(U, self::U)`: entry `self::U`, and the copy kept it).
#[test]
fn a_path_named_like_a_generic_is_replaced_in_a_struct() {
    for path in PATHS {
        for shape in SHAPES {
            let ty = shape.replace("{P}", path);
            for scope in [Scope::Intern, Scope::Nested, Scope::Generics] {
                round_trip(&format!("struct S<U>(U, {});", ty), scope);
                round_trip(&format!("struct S<U, X>(X, {});", ty), scope);
                round_trip(
                    &format!("struct S<U> where U: Clone {{ a: {}, b: U }}", ty),
                    scope,
                );
            }
        }
    }
}

/// The same in the methods of a trait: the generic is the method's or the trait's.
#[test]
fn a_path_named_like_a_generic_is_replaced_in_a_trait() {
    for path in PATHS {
        for shape in SHAPES {
            let ty = shape.replace("{P}", path);
            for scope in [Scope::Intern, Scope::Nested, Scope::Generics] {
                round_trip(
                    &format!("trait T {{ fn f<U>(&self, x: U, y: {}); }}", ty),
                    scope,
                );
                round_trip(
                    &format!("trait T<U> {{ fn f(&self, x: U, y: {}); }}", ty),
                    scope,
                );
                round_trip(
                    &format!("trait T {{ fn f<U: Clone>(&self, y: {}) -> U; }}", ty),
                    scope,
                );
                round_trip(
                    &format!("trait T {{ type A; fn f<U>(&self) -> {}; }}", ty),
                    scope,
                );
            }
        }
    }
}

/// An inherent impl: `impl<U> S<U> { fn f(b: crate::U) {} }`.
#[test]
fn a_path_named_like_a_generic_is_replaced_in_an_impl() {
    for path in PATHS {
        for shape in SHAPES {
            let ty = shape.replace("{P}", path);
            round_trip(
                &format!("impl<U> S<U> {{ fn f(b: {}) {{}} }}", ty),
                Scope::Intern,
            );
            round_trip(
                &format!("impl<U> S<U> {{ fn f<X>(b: {}, c: X) {{}} }}", ty),
                Scope::Intern,
            );
        }
    }
}

/// The shapes that name the generic too (`crate::U<U>`) are entries with the generic in their
/// context, found with it; the path alone is an entry with none.
#[test]
fn the_context_of_a_path_named_like_a_generic_has_only_what_the_type_names() {
    for path in ["self::U", "crate::U"] {
        let item: ItemStruct = parse_str(&format!("struct S<U>(U, {0}, {0}<U>);", path)).unwrap();
        let mut leaker = Leaker::with_rule(common::std_config());
        let context = leaker.intern_struct(&item);
        let referrer = through_tokens(leaker.result().finish().unwrap());
        let alone = index(&referrer, path, "");
        let with_generic = index(&referrer, &format!("{} < U >", path), "< U >");
        let mut copy = item.clone();
        let found = replace(&referrer, context, |visitor| {
            visitor.visit_item_struct_mut(&mut copy)
        });
        assert_eq!(found, set(&[alone, with_generic]));
        let expected: ItemStruct =
            parse_quote!(struct S<U>(U, Replaced<#alone>, Replaced<#with_generic>););
        assert_eq!(copy, expected);
    }
}

/// `impl<U> S<U> { fn f(b: crate::U) {} }` with the exact copy: both the self type and the argument
/// are replaced, each by its own entry.
#[test]
fn the_copy_of_an_impl_has_the_entry_of_the_path_named_like_a_generic() {
    for path in ["crate::U", "self::U"] {
        let item: ItemImpl =
            parse_str(&format!("impl<U> S<U> {{ fn f(b: {}) {{}} }}", path)).unwrap();
        let mut leaker = Leaker::with_rule(common::std_config());
        let context = leaker.intern_impl(&item);
        let referrer = through_tokens(leaker.result().finish().unwrap());
        let self_ty = index(&referrer, "S < U >", "< U >");
        let arg = index(&referrer, path, "");
        let mut copy = item.clone();
        replace(&referrer, context, |visitor| {
            visitor.visit_item_impl_mut(&mut copy)
        });
        let expected: ItemImpl =
            parse_quote!(impl<U> Replaced<#self_ty> { fn f(b: Replaced<#arg>) {} });
        assert_eq!(copy, expected);
    }
}

/// The fragments of an impl of a macro that resolves the trait by its name (`coinduction`): the
/// arguments of the trait are visited in `enter_impl(..)`, and written by the visitor.
#[test]
fn a_path_named_like_a_generic_is_replaced_in_the_arguments_of_an_impl() {
    for path in PATHS {
        let item: ItemImpl = parse_str(&format!(
            "impl<U> ::tr::Tr<{0}, Vec<{0}>, U> for ::tr::S<U> {{}}",
            path
        ))
        .unwrap();
        let (_, trait_path, _) = item.trait_.clone().unwrap();
        let mut leaker = Leaker::with_rule(common::std_config());
        let (context, ()) = leaker.enter_impl(&item.generics, &item.self_ty, None, |view| {
            for seg in &trait_path.segments {
                view.visit_path_arguments(&seg.arguments);
            }
        });
        let referrer = through_tokens(leaker.result().finish().unwrap());
        let mut copy = trait_path.clone();
        let found = replace(&referrer, context, |visitor| {
            visitor.visit_path_mut(&mut copy)
        });
        assert_eq!(found, all_entries(&referrer), "{}", path);
        assert!(
            !s(&copy).contains("inner") && !s(&copy).contains("U ::"),
            "{}",
            s(&copy)
        );
        assert!(!s(&copy).contains(path), "{}: {}", path, s(&copy));
    }
}

// ── declared_named_like_generic ──────────────────────────────────────────────────────────────

/// The repro of the property test: a copy that declares `X`, and a function with the generic `X`
/// whose argument is a type of another module of the crate.
#[test]
fn a_file_that_declares_the_name_of_a_generic_keeps_the_entry_of_the_generic() {
    let items: [Item; 2] = [
        parse_quote!(
            pub struct X;
        ),
        parse_quote!(
            pub fn f<X>(a: crate::base::B1<X>) {}
        ),
    ];
    let mut leaker = Leaker::with_rule(common::std_config());
    let (context, ()) = leaker.enter_generics(&Generics::default(), |view| {
        for item in &items {
            view.visit_item(item);
        }
    });
    let referrer = through_tokens(leaker.result().finish().unwrap());
    let mut file = File {
        shebang: None,
        attrs: Vec::new(),
        items: items.to_vec(),
    };
    let found = replace(&referrer, context, |visitor| {
        visitor.visit_file_mut(&mut file)
    });
    let entry = index(&referrer, "crate :: base :: B1 < X >", "< X >");
    assert_eq!(found, set(&[entry]));
    let expected: File = parse_quote! {
        pub struct X;
        pub fn f<X>(a: Replaced<#entry>) {}
    };
    assert_eq!(file, expected);
}

/// What a `File` declares still hides an entry that names it as an item and no generic: the entry
/// was written for the `X` of the Leaker's crate, and the copy declares another.
#[test]
fn a_declared_name_still_hides_the_entries_that_do_not_name_a_generic() {
    // A trait: its `X` is a type of the Leaker's crate.
    let item: ItemTrait = parse_quote!(
        trait T {
            fn g(&self, a: crate::base::B1<X>);
        }
    );
    let mut leaker = Leaker::with_rule(common::std_config());
    let context = leaker.intern_trait(&item);
    let referrer = through_tokens(leaker.result().finish().unwrap());
    let items = |declare: bool| {
        let mut items = vec![Item::Trait(item.clone())];
        if declare {
            items.insert(
                0,
                Item::Struct(parse_quote!(
                    pub struct X;
                )),
            );
        }
        File {
            shebang: None,
            attrs: Vec::new(),
            items,
        }
    };
    // The copy without the declaration: replaced.
    let mut file = items(false);
    let found = replace(&referrer, context, |v| v.visit_file_mut(&mut file));
    assert_eq!(found, all_entries(&referrer));
    assert!(!all_entries(&referrer).is_empty());
    assert!(!s(&file).contains("base"), "{}", s(&file));
    // The copy that declares `X`: its `X` is not the entry's.
    let mut file = items(true);
    let found = replace(&referrer, context, |v| v.visit_file_mut(&mut file));
    assert!(found.is_empty(), "{:?}", found);
    assert!(
        s(&file).contains("crate :: base :: B1 < X >"),
        "{}",
        s(&file)
    );
}

// ── a prefix does not change what the Leaker decides ─────────────────────────────────────────

/// The spellings of a relative path to a type (`Rl<'a>`). `self::Rl` and the others name the same
/// item as the bare `Rl` does, so the Leaker must make the same entries from them, and give the
/// same errors, notes and help, apart from the prefix in the text.
const PREFIXES: &[&str] = &["", "self::", "crate::", "super::", "self::inner::"];

/// What the Leaker decided about an item, with the prefix taken out of every text.
#[derive(Clone, Debug, PartialEq)]
enum Outcome {
    /// The entries: the type and its generics.
    Accepted { entries: Vec<(String, String)> },
    Rejected {
        headline: String,
        note: Option<String>,
        help: Option<String>,
    },
}

/// `text` without spaces and without the prefixes of the paths.
fn unprefixed(text: &str) -> String {
    let mut text: String = text.chars().filter(|c| !c.is_whitespace()).collect();
    for prefix in ["self::inner::", "self::", "crate::", "super::"] {
        text = text.replace(prefix, "");
    }
    text
}

/// Analyzes `item` in `scope`.
fn outcome(item: &Item, scope: Scope) -> Outcome {
    let mut leaker = Leaker::with_rule(common::std_config());
    match (scope, item) {
        (Scope::Intern, Item::Struct(i)) => {
            leaker.intern_struct(i);
        }
        (Scope::Intern, Item::Enum(i)) => {
            leaker.intern_enum(i);
        }
        (Scope::Intern, Item::Trait(i)) => {
            leaker.intern_trait(i);
        }
        (Scope::Intern, Item::Impl(i)) => {
            leaker.intern_impl(i);
        }
        (Scope::Nested, _) => {
            leaker.enter_generics(&Generics::default(), |view| view.visit_item(item));
        }
        (Scope::Generics, Item::Struct(i)) => {
            leaker.enter_generics(&i.generics, |view| syn::visit::visit_item_struct(view, i));
        }
        (Scope::Generics, Item::Enum(i)) => {
            leaker.enter_generics(&i.generics, |view| syn::visit::visit_item_enum(view, i));
        }
        (Scope::Generics, Item::Trait(i)) => {
            leaker.enter_trait(&i.ident, &i.generics, |view| {
                syn::visit::visit_item_trait(view, i)
            });
        }
        _ => unreachable!(),
    };
    match leaker.result().finish() {
        Ok(referrer) => Outcome::Accepted {
            entries: referrer
                .iter()
                .map(|(_, ty, context)| (ty, &context.generics))
                .map(|(ty, generics)| {
                    (
                        unprefixed(&s(ty)),
                        unprefixed(&format!("{} {}", s(&generics), s(&generics.where_clause))),
                    )
                })
                .collect(),
        },
        Err(error) => Outcome::Rejected {
            headline: unprefixed(&messages(&error)),
            note: error[0]
                .notes
                .first()
                .map(|(_, n)| n.as_str())
                .map(unprefixed),
            help: error[0].help_text().map(unprefixed),
        },
    }
}

/// The item `source` with `{Rl}` and `{Wrap}` spelled with `prefix`.
fn spelled(source: &str, prefix: &str) -> Item {
    let source = source
        .replace("{Rl}", &format!("{}Rl", prefix))
        .replace("{Wrap}", &format!("{}Wrap", prefix));
    parse_str(&source).unwrap_or_else(|e| panic!("{}: {}", source, e))
}

/// What the bare spelling must give.
#[derive(Clone, Copy, Debug)]
enum Expect {
    Accepted,
    /// Rejected, with a headline that contains this.
    Rejected(&'static str),
}

/// Checks that each prefixed spelling of `source` gives the outcome of the bare one in `scope`, and
/// that the bare one is `expect`ed (so that the test cannot pass by comparing rejections that it
/// did not look at).
fn same_outcome_with_every_prefix(source: &str, scope: Scope, expect: Expect) {
    let bare = outcome(&spelled(source, ""), scope);
    match (&bare, expect) {
        (Outcome::Accepted { entries, .. }, Expect::Accepted) => {
            assert!(!entries.is_empty(), "{:?} {}: no entry", scope, source)
        }
        (Outcome::Rejected { headline, .. }, Expect::Rejected(part)) => assert!(
            headline.contains(&unprefixed(part)),
            "{:?} {}: {}",
            scope,
            source,
            headline
        ),
        _ => panic!(
            "{:?} {}: expected {:?}, got {:?}",
            scope, source, expect, bare
        ),
    }
    for prefix in &PREFIXES[1..] {
        let actual = outcome(&spelled(source, prefix), scope);
        assert_eq!(actual, bare, "`{}` {:?} {}", prefix, scope, source);
    }
}

/// The repro of the property test (seed 4, case 75): an enum whose where predicate names `Self`
/// and a relative type, and whose field uses the generic of that predicate. Nested in a scope,
/// the copy declares an enum of its own: the bound holds for it, and the Repeater's impl of the
/// field's type used to be for the Leaker's enum, which has not the bound in the copy's crate
/// (E0277 in the Referrer's crate). It was rejected for that, whatever the spelling of the relative
/// type (`self::Rl` was thought to slip past a check that the bare `Rl` did not: the bare
/// spelling failed in the same way). The entry is for any type now (`SelfRequirement::Any`), and
/// the copy passes its own: accepted with every spelling.
#[test]
fn a_nested_enum_whose_bound_names_self_is_accepted_with_every_spelling_of_a_relative_type() {
    let source = "pub enum E<'a, T: ::core::marker::Send + 'a, const V: usize = 3>
        where T: ::core::convert::AsRef<::std::collections::BTreeMap<(Self, Self), {Rl}<'a>>>
        { V0(&'a super::base::B2<T, ::core::result::Result<u32, u16>>), V1 }";
    same_outcome_with_every_prefix(source, Scope::Nested, Expect::Accepted);
    // Analyzed as the item itself, `Self` is written as the Leaker's type: accepted.
    same_outcome_with_every_prefix(source, Scope::Intern, Expect::Accepted);
}

/// The bounds that name a leaked type, with the path spelled every way. These rejections and the
/// items that are accepted are those of `tests/leaker/bound_cycles.rs`.
#[test]
fn a_bound_that_names_a_leaked_type_is_decided_with_every_spelling() {
    const BOUND_CYCLE: &str = "is in a bound or where predicate";
    for scope in [Scope::Intern, Scope::Nested, Scope::Generics] {
        // `{Wrap}<T>` is in a predicate, and `T` has a bound that names `{Rl}`.
        for source in [
            "pub struct S<T: Fn({Rl})>({Wrap}<T>) where {Wrap}<T>: Sized;",
            "pub struct S<T: Fn({Rl}), U: Into<{Wrap}<T>>>(T, U);",
            "pub struct S<T: Into<{Wrap}<T>>>(T);",
            "pub struct S<T: AsRef<{Rl}>>({Wrap}<T>) where {Wrap}<T>: Sized;",
        ] {
            same_outcome_with_every_prefix(source, scope, Expect::Rejected(BOUND_CYCLE));
        }
        // In a field, the same types are fine.
        same_outcome_with_every_prefix(
            "pub struct S<T: Fn({Rl})>({Wrap}<T>);",
            scope,
            Expect::Accepted,
        );
        same_outcome_with_every_prefix(
            "pub struct S<T: Iterator<Item = {Rl}>>({Wrap}<T>);",
            scope,
            Expect::Accepted,
        );
        // In a method of a trait.
        same_outcome_with_every_prefix(
            "pub trait Tr { fn f<T: Fn({Rl})>(&self, a: {Wrap}<T>) where {Wrap}<T>: Sized; }",
            scope,
            Expect::Rejected(BOUND_CYCLE),
        );
        same_outcome_with_every_prefix(
            "pub trait Tr { fn f<T: Fn({Rl})>(&self, a: {Wrap}<T>); }",
            scope,
            Expect::Accepted,
        );
    }
}

/// `Self` in a bound or a predicate, with the path spelled every way.
#[test]
fn a_bound_that_names_self_is_decided_with_every_spelling() {
    // A trait: `{Rl}<Self>` is in a predicate of the trait, and its impl is for the implementors.
    for scope in [Scope::Intern, Scope::Generics] {
        same_outcome_with_every_prefix(
            "pub trait Tr<T> where {Rl}<Self>: Sized {}",
            scope,
            Expect::Rejected("names `Self` in a bound or where predicate of its trait"),
        );
        same_outcome_with_every_prefix(
            "pub trait Tr<T: AsRef<{Rl}<Self>>> {}",
            scope,
            Expect::Rejected("is in a bound or where predicate"),
        );
        same_outcome_with_every_prefix(
            "pub trait Tr<T> { fn f(&self) where {Rl}<Self>: Sized; }",
            scope,
            Expect::Accepted,
        );
    }
    // A method of an impl: the self type is leaked, and the bound of `X` names `Self`.
    same_outcome_with_every_prefix(
        "impl S { fn m<X>(&self) where X: Fn(Self) -> u8, {Rl}<X>: Default {} }",
        Scope::Intern,
        Expect::Rejected("is in a bound or where predicate"),
    );
    // A struct, an enum: `Self` in the bound of a generic that an entry of a field uses. As the
    // item itself, the Referrer writes `Self` as the Leaker's type, so it is fine. Nested in a
    // scope, the copy has a `Self` of its own: an entry that does not name `Self` is for any type,
    // and the copy passes its own; one that names it too is for the Leaker's type, which the bound
    // does not hold for in the copy: rejected.
    for source in [
        "pub struct S<T: AsRef<Self>>({Wrap}<T>);",
        "pub struct S<T>({Wrap}<T>) where T: AsRef<(Self, {Rl}<'static>)>;",
    ] {
        same_outcome_with_every_prefix(source, Scope::Nested, Expect::Accepted);
        same_outcome_with_every_prefix(source, Scope::Intern, Expect::Accepted);
    }
    for source in [
        "pub struct S<T: AsRef<Self>>({Wrap}<Self>, {Wrap}<T>);",
        "pub enum E<T> where T: AsRef<Self> { A({Wrap}<Self>), B(T) }",
    ] {
        same_outcome_with_every_prefix(
            source,
            Scope::Nested,
            Expect::Rejected("has a generic whose bound mentions `Self`"),
        );
        same_outcome_with_every_prefix(source, Scope::Intern, Expect::Accepted);
    }
    // Not so if `Self` is not in a bound that the environment proves (`Self: Sized`, `Vec<Self>:
    // Clone`), if the entry does not use the generic, or if `Self` is only outlived.
    for source in [
        "pub struct S<T>({Wrap}<Self>, T) where Self: Sized;",
        "pub struct S<T>({Wrap}<T>, Vec<Self>) where Vec<Self>: Clone;",
        "pub struct S<T: AsRef<Self>, U>({Wrap}<U>, T);",
        "pub struct S<'a, T: 'a>(&'a {Wrap}<Self>, &'a T);",
    ] {
        for scope in [Scope::Intern, Scope::Nested] {
            same_outcome_with_every_prefix(source, scope, Expect::Accepted);
        }
    }
}

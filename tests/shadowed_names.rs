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
//! - `declared_named_like_generic`: a module that declares a type `X` hides the entries of no module
//!   that name `X`, because the copy declares `X` again and the entry was written for another
//!   `X`. But `X` in `crate::base::B1<X>` of `fn f<X>` is the generic of the entry, which the
//!   declared item does not hide where the entry is written. (Format 10 fixed it: the visitor hid
//!   the entry, and the Referrer's crate kept `crate::base`, E0433.)
//!
//! The same shapes are copied across crates in `tests/crates/leaker` (`kinds`, the ids 9091 and
//! after), where the Repeater impls and the copies are compiled: `tests/cross_crate.rs`.

use std::cell::RefCell;
use std::collections::BTreeSet;
use syn::visit_mut::VisitMut;
use syn::*;
use template_quote::{quote, ToTokens};
use type_leak::*;

fn text<T: ToTokens>(t: &T) -> String {
    quote!(#t).to_string()
}

fn config() -> LeakerConfig {
    let mut config = LeakerConfig::new();
    config.allow_primitive().allow_std_prelude();
    config
}

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

/// The indices of the required entries, which are the ones a visitor replaces.
fn required(referrer: &Referrer) -> BTreeSet<usize> {
    (0..referrer.iter().count())
        .filter(|&ix| referrer.must(ix))
        .collect()
}

/// The index of the entry `ty` with `generics` (as tokens) and the module it has.
fn index(referrer: &Referrer, ty: &str, generics: &str, in_module: bool) -> usize {
    let (ty, generics) = (text(&parse_str::<Type>(ty).unwrap()), generics.to_string());
    let found: Vec<usize> = (0..referrer.iter().count())
        .filter(|&ix| {
            text(&referrer.iter().nth(ix).unwrap()) == ty
                && text(&referrer.generics(ix)).replace(' ', "") == generics.replace(' ', "")
                && referrer.context_module(referrer.context(ix)).is_some() == in_module
        })
        .collect();
    assert_eq!(
        found.len(),
        1,
        "entry `{}` with `{}` (module: {}) in {:?}",
        ty,
        generics,
        in_module,
        referrer
            .iter_with_generics()
            .map(|(ty, generics)| (text(ty), text(&generics)))
            .collect::<Vec<_>>()
    );
    found[0]
}

/// Replaces the entries a visitor of `referrer` finds in what `visit` visits, with `Replaced<ix>`,
/// and returns the indices it found.
fn replace(
    referrer: &Referrer,
    context: ContextId,
    visit: impl FnOnce(&mut Visitor<&mut dyn FnMut(Type, usize) -> Type>),
) -> BTreeSet<usize> {
    let found = RefCell::new(BTreeSet::new());
    let mut type_fn = |_: Type, ix: usize| -> Type {
        found.borrow_mut().insert(ix);
        parse_quote!(Replaced<#ix>)
    };
    let mut visitor = referrer
        .clone()
        .visitor(context, &mut type_fn as &mut dyn FnMut(Type, usize) -> Type);
    visit(&mut visitor);
    drop(visitor);
    found.into_inner()
}

fn analyze_item(leaker: &mut Leaker, item: &Item) -> std::result::Result<ContextId, String> {
    match item {
        Item::Struct(i) => leaker.intern_struct(i, true),
        Item::Trait(i) => leaker.intern_trait(i, true),
        Item::Impl(i) => leaker.intern_impl(i, true),
        _ => unreachable!(),
    }
    .map_err(|e| e.to_string())
}

/// How an item is analyzed: by its `intern_*` method, or through the scope of a macro.
#[derive(Clone, Copy, Debug)]
enum Scope {
    Intern,
    /// `enter_mod(.., [name], |view| view.visit_item(&item))`, the copy a `File` of the item.
    Mod,
    /// `enter_generics(&generics, ..)` for a struct, `enter_trait(..)` for a trait.
    Generics,
}

/// Analyzes the item `source` in `scope`, copies it with the visitor of the Referrer, and checks
/// that the visitor found every required entry, so that the copy has no path left.
fn round_trip(source: &str, scope: Scope) {
    let item: Item = parse_str(source).unwrap();
    let mut leaker = Leaker::from_config(config());
    let name = match &item {
        Item::Struct(i) => i.ident.clone(),
        Item::Trait(i) => i.ident.clone(),
        _ => parse_quote!(Impl),
    };
    let analyzed = match (scope, &item) {
        (Scope::Intern, _) => analyze_item(&mut leaker, &item),
        (Scope::Mod, _) => leaker
            .enter_mod("m", [name], |view| view.visit_item(&item))
            .map(|(context, ())| context)
            .map_err(|e| e.to_string()),
        (Scope::Generics, Item::Struct(i)) => leaker
            .enter_generics(&i.generics, |view| syn::visit::visit_item_struct(view, i))
            .map(|(context, ())| context)
            .map_err(|e| e.to_string()),
        (Scope::Generics, Item::Trait(i)) => leaker
            .enter_trait(&i.ident, &i.generics, true, |view| {
                syn::visit::visit_item_trait(view, i)
            })
            .map(|(context, ())| context)
            .map_err(|e| e.to_string()),
        _ => unreachable!(),
    };
    let context = analyzed.unwrap_or_else(|e| panic!("{:?} {}: {}", scope, source, e));
    let referrer = through_tokens(leaker.finish());
    assert!(!required(&referrer).is_empty(), "{:?} {}", scope, source);
    let mut copy = item.clone();
    let found = replace(&referrer, context, |visitor| match scope {
        Scope::Mod => {
            let mut file = File {
                shebang: None,
                attrs: Vec::new(),
                items: vec![copy.clone()],
            };
            visitor.visit_file_mut(&mut file);
            copy = file.items.remove(0);
        }
        _ => visitor.visit_item_mut(&mut copy),
    });
    assert_eq!(
        found,
        required(&referrer),
        "{:?}: the visitor misses an entry of {}\n  copy: {}",
        scope,
        source,
        text(&copy)
    );
    assert!(
        !text(&copy).contains("::"),
        "{:?}: a path is left in the copy of {}\n  copy: {}",
        scope,
        source,
        text(&copy)
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
            for scope in [Scope::Intern, Scope::Mod, Scope::Generics] {
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
            for scope in [Scope::Intern, Scope::Mod, Scope::Generics] {
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
        let mut leaker = Leaker::from_config(config());
        let context = leaker.intern_struct(&item, true).unwrap();
        let referrer = through_tokens(leaker.finish());
        let alone = index(&referrer, path, "", false);
        let with_generic = index(&referrer, &format!("{} < U >", path), "< U >", false);
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
        let mut leaker = Leaker::from_config(config());
        let context = leaker.intern_impl(&item, true).unwrap();
        let referrer = through_tokens(leaker.finish());
        let self_ty = index(&referrer, "S < U >", "< U >", false);
        let arg = index(&referrer, path, "", false);
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
        let mut leaker = Leaker::from_config(config());
        let (context, ()) = leaker
            .enter_impl(&item.generics, &item.self_ty, false, |view| {
                for seg in &trait_path.segments {
                    view.visit_path_arguments(&seg.arguments);
                }
            })
            .unwrap();
        let referrer = through_tokens(leaker.finish());
        let mut copy = trait_path.clone();
        let found = replace(&referrer, context, |visitor| {
            visitor.visit_path_mut(&mut copy)
        });
        assert_eq!(found, required(&referrer), "{}", path);
        assert!(
            !text(&copy).contains("inner") && !text(&copy).contains("U ::"),
            "{}",
            text(&copy)
        );
        assert!(!text(&copy).contains(path), "{}: {}", path, text(&copy));
    }
}

// ── declared_named_like_generic ──────────────────────────────────────────────────────────────

/// Analyzes the module `source` as a whole (`intern_mod`), and copies its items with the visitor of
/// the Referrer, as a `File`. Returns the Referrer, and the copy of the items.
fn module_copy(source: &str) -> (Referrer, File, BTreeSet<usize>) {
    let item: ItemMod = parse_str(source).unwrap();
    let mut leaker = Leaker::from_config(config());
    let context = leaker
        .intern_mod(&item)
        .unwrap_or_else(|e| panic!("{}: {}", source, e));
    let referrer = through_tokens(leaker.finish());
    let mut file = File {
        shebang: None,
        attrs: Vec::new(),
        items: item.content.unwrap().1,
    };
    let found = replace(&referrer, context, |visitor| {
        visitor.visit_file_mut(&mut file)
    });
    (referrer, file, found)
}

/// The repro of the property test: a module that declares `X`, and a function with the generic `X`
/// whose argument is a type of another module of the crate.
#[test]
fn a_module_that_declares_the_name_of_a_generic_keeps_the_entry_of_the_generic() {
    let (referrer, file, found) =
        module_copy("mod m { pub struct X; pub fn f<X>(a: crate::base::B1<X>) {} }");
    let entry = index(&referrer, "crate :: base :: B1 < X >", "< X >", false);
    assert_eq!(found, set(&[entry]));
    let expected: File = parse_quote! {
        pub struct X;
        pub fn f<X>(a: Replaced<#entry>) {}
    };
    assert_eq!(file, expected);
}

/// The generic hides the declared item where the entry is written, and the item hides nothing
/// elsewhere: `f` and `g` write the same tokens, and each gets the entry of its own scope.
#[test]
fn the_same_type_with_the_generic_and_with_the_declared_item_is_two_entries() {
    let (referrer, file, found) = module_copy(
        "mod m {
            pub struct X;
            pub fn f<X>(a: crate::base::B1<X>) {}
            pub fn g(a: crate::base::B1<X>) {}
        }",
    );
    // `f`: the generic; no module, as the type names nothing of the module. `g`: the declared `X`,
    // which only the module has.
    let generic = index(&referrer, "crate :: base :: B1 < X >", "< X >", false);
    let declared = index(&referrer, "crate :: base :: B1 < X >", "", true);
    assert_eq!(found, set(&[generic, declared]));
    let expected: File = parse_quote! {
        pub struct X;
        pub fn f<X>(a: Replaced<#generic>) {}
        pub fn g(a: Replaced<#declared>) {}
    };
    assert_eq!(file, expected);
}

/// The modules and items that make the declared name and the generic meet, with every path form
/// of the type (`crate::`, `self::`, `super::`, relative): every entry is found.
#[test]
fn every_entry_that_names_a_generic_of_the_name_of_a_declared_item_is_found() {
    let heads = ["crate::base::B", "B", "self::B", "super::B", "crate::B"];
    let args = [
        "X",
        "(X, U)",
        "Vec<X>",
        "&'static X",
        "[X; 2]",
        "<X as Tr>::A",
        "fn(X) -> U",
        "X::A",
        "crate::m::X",
        "self::X",
        "T",
    ];
    for head in heads {
        for arg in args {
            let ty = format!("{}<{}>", head, arg);
            for source in [
                format!("mod m {{ pub struct T; pub struct U; pub struct X; pub fn f<X, U>(a: {}) {{}} }}", ty),
                format!("mod m {{ pub struct T; pub struct U; pub struct X; pub struct S<X, U>(pub {}); }}", ty),
                format!("mod m {{ pub struct T; pub struct U; pub struct X; pub trait Tr<X> {{ fn f<U>(&self, a: {}); }} }}", ty),
                format!("mod m {{ pub struct T; pub struct U; pub struct X; pub fn f(a: {}) {{}} }}", ty),
                format!("mod m {{ pub struct T; pub struct U; pub struct X; pub fn f<X>(a: {0}) {{}} pub fn g(a: {0}) {{}} }}", ty),
                format!("mod m {{ pub struct T; pub struct U; pub struct X; pub mod n {{ pub fn f<X, U>(a: {0}) {{}} pub fn g(a: {0}) {{}} }} }}", ty),
                format!("mod m {{ pub struct T; pub mod n {{ pub struct U; pub struct X; pub fn f<X, U>(a: {0}) {{}} pub fn g(a: {0}) {{}} }} }}", ty),
                format!("mod m {{ pub struct X; pub fn f() {{ fn g<X>(a: {}) {{}} }} }}", ty),
            ] {
                let item: ItemMod = parse_str(&source).unwrap();
                let mut leaker = Leaker::from_config(config());
                // A rejected module (a relative path that cannot be leaked) is no case.
                let context = match leaker.intern_mod(&item) {
                    Ok(context) => context,
                    Err(_) => continue,
                };
                let referrer = through_tokens(leaker.finish());
                let mut file = File {
                    shebang: None,
                    attrs: Vec::new(),
                    items: item.content.unwrap().1,
                };
                let found = replace(&referrer, context, |v| v.visit_file_mut(&mut file));
                assert_eq!(
                    found,
                    required(&referrer),
                    "the visitor misses an entry of {}\n  entries: {:?}",
                    source,
                    referrer
                        .iter_with_generics()
                        .map(|(ty, generics)| (text(ty), text(&generics)))
                        .collect::<Vec<_>>()
                );
            }
        }
    }
}

/// The module is entered with `enter_mod(..)`, which declares the names: the item inside is copied
/// as a nested item of a `File`.
#[test]
fn enter_mod_declaring_the_name_of_a_generic_keeps_the_entry_of_the_generic() {
    let item: ItemFn = parse_quote!(
        pub fn f<X>(a: crate::base::B1<X>) {}
    );
    let mut leaker = Leaker::from_config(config());
    let declared: [Ident; 2] = [parse_quote!(X), item.sig.ident.clone()];
    let (context, ()) = leaker
        .enter_mod("m", declared, |view| {
            view.visit_item(&Item::Fn(item.clone()))
        })
        .unwrap();
    let referrer = through_tokens(leaker.finish());
    let entry = index(&referrer, "crate :: base :: B1 < X >", "< X >", false);
    let mut file = File {
        shebang: None,
        attrs: Vec::new(),
        items: vec![
            Item::Struct(parse_quote!(
                pub struct X;
            )),
            Item::Fn(item),
        ],
    };
    let found = replace(&referrer, context, |visitor| {
        visitor.visit_file_mut(&mut file)
    });
    assert_eq!(found, set(&[entry]));
    let expected: File = parse_quote! {
        pub struct X;
        pub fn f<X>(a: Replaced<#entry>) {}
    };
    assert_eq!(file, expected);
}

/// What a `File` declares still hides an entry of no module that names it as an item and no
/// generic: the entry was written for the `X` of the Leaker's crate, and the copy declares another.
#[test]
fn a_declared_name_still_hides_the_entries_that_do_not_name_a_generic() {
    // A trait: its `X` is a type of the Leaker's crate (an entry of no module).
    let item: ItemTrait = parse_quote!(
        trait T {
            fn g(&self, a: crate::base::B1<X>);
        }
    );
    let mut leaker = Leaker::from_config(config());
    let context = leaker.intern_trait(&item, true).unwrap();
    let referrer = through_tokens(leaker.finish());
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
    assert_eq!(found, required(&referrer));
    assert!(!required(&referrer).is_empty());
    assert!(!text(&file).contains("base"), "{}", text(&file));
    // The copy that declares `X`: its `X` is not the entry's.
    let mut file = items(true);
    let found = replace(&referrer, context, |v| v.visit_file_mut(&mut file));
    assert!(found.is_empty(), "{:?}", found);
    assert!(
        text(&file).contains("crate :: base :: B1 < X >"),
        "{}",
        text(&file)
    );
}

// ── a prefix does not change what the Leaker decides ─────────────────────────────────────────

/// The spellings of a path to a type of the module (`Rl<'a>`). `self::Rl` and the others name the
/// same item as the bare `Rl` does, so the Leaker must make the same entries from them, and give the
/// same errors, notes and help, apart from the prefix in the text.
const PREFIXES: &[&str] = &["", "self::", "crate::", "super::", "self::inner::"];

/// What the Leaker decided about an item, with the prefix taken out of every text.
#[derive(Clone, Debug, PartialEq)]
enum Outcome {
    /// The entries: the type, whether it is required and its generics, and whether each has a module.
    Accepted {
        entries: Vec<(String, bool, String)>,
        modules: Vec<bool>,
    },
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

/// Analyzes `item` in `scope` (`declared`: the names the module declares, with `Scope::Mod`).
fn outcome(item: &Item, scope: Scope, declared: &[&str]) -> Outcome {
    let mut leaker = Leaker::from_config(config());
    let name = match item {
        Item::Struct(i) => i.ident.clone(),
        Item::Enum(i) => i.ident.clone(),
        Item::Trait(i) => i.ident.clone(),
        _ => parse_quote!(Impl),
    };
    let result = match (scope, item) {
        (Scope::Intern, Item::Struct(i)) => leaker.intern_struct(i, true),
        (Scope::Intern, Item::Enum(i)) => leaker.intern_enum(i, true),
        (Scope::Intern, Item::Trait(i)) => leaker.intern_trait(i, true),
        (Scope::Intern, Item::Impl(i)) => leaker.intern_impl(i, true),
        (Scope::Mod, _) => {
            let names: Vec<Ident> = std::iter::once(name)
                .chain(declared.iter().map(|d| parse_str::<Ident>(d).unwrap()))
                .collect();
            leaker
                .enter_mod("m", names, |view| view.visit_item(item))
                .map(|(context, ())| context)
        }
        (Scope::Generics, Item::Struct(i)) => leaker
            .enter_generics(&i.generics, |view| syn::visit::visit_item_struct(view, i))
            .map(|(context, ())| context),
        (Scope::Generics, Item::Enum(i)) => leaker
            .enter_generics(&i.generics, |view| syn::visit::visit_item_enum(view, i))
            .map(|(context, ())| context),
        (Scope::Generics, Item::Trait(i)) => leaker
            .enter_trait(&i.ident, &i.generics, true, |view| {
                syn::visit::visit_item_trait(view, i)
            })
            .map(|(context, ())| context),
        _ => unreachable!(),
    };
    match result {
        Ok(_) => {
            let referrer = leaker.finish();
            Outcome::Accepted {
                entries: referrer
                    .iter_with_generics()
                    .enumerate()
                    .map(|(ix, (ty, generics))| {
                        (
                            unprefixed(&text(ty)),
                            referrer.must(ix),
                            unprefixed(&format!(
                                "{} {}",
                                text(&generics),
                                text(&generics.where_clause)
                            )),
                        )
                    })
                    .collect(),
                modules: (0..referrer.iter().count())
                    .map(|ix| referrer.context_module(referrer.context(ix)).is_some())
                    .collect(),
            }
        }
        Err(error) => Outcome::Rejected {
            headline: unprefixed(&error.to_string()),
            note: error.note().map(|note| unprefixed(&note)),
            help: error.help().map(|help| unprefixed(&help)),
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
    let bare = outcome(&spelled(source, ""), scope, &[]);
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
        let actual = outcome(&spelled(source, prefix), scope, &[]);
        assert_like_bare(&actual, &bare, prefix, &format!("{:?} {}", scope, source));
    }
}

/// Asserts that `actual`, the outcome of the spelling with `prefix`, is that of the bare one. A path
/// of `crate::` is the same in every module, so its entry has no module unless the context is
/// bound to the module otherwise (the nested `Self` is a type of the module): its Repeater impl
/// need not be written in the module where the bare name resolves. The others are relative to the
/// module, as the bare name is.
fn assert_like_bare(actual: &Outcome, bare: &Outcome, prefix: &str, what: &str) {
    match (actual, bare) {
        (
            Outcome::Accepted { entries, modules },
            Outcome::Accepted {
                entries: bare_entries,
                modules: bare_modules,
            },
        ) if prefix == "crate::" => {
            assert_eq!(entries, bare_entries, "`{}` {}", prefix, what);
            assert!(
                modules
                    .iter()
                    .zip(bare_modules)
                    .all(|(m, bare)| *bare || !*m),
                "`{}` {}: {:?} is more bound to the module than {:?}",
                prefix,
                what,
                modules,
                bare_modules
            );
        }
        _ => assert_eq!(actual, bare, "`{}` {}", prefix, what),
    }
}

/// The repro of the property test (seed 4, case 75): an enum whose where predicate names `Self`
/// and a type of the module, and whose field uses the generic of that predicate. Nested in a module,
/// the copy declares an enum of its own: the bound holds for it, and the Repeater's impl of the
/// field's type used to be for the Leaker's enum, which has not the bound in the copy's crate
/// (E0277 in the Referrer's crate). It was rejected for that, whatever the spelling of the type of
/// the module (`self::Rl` was thought to slip past a check that the bare `Rl` did not: the bare
/// spelling failed in the same way). The entry is for any type now (`SelfRequirement::Any`), and
/// the copy passes its own: accepted with every spelling.
#[test]
fn a_nested_enum_whose_bound_names_self_is_accepted_with_every_spelling_of_a_type_of_the_module() {
    let source = "pub enum E<'a, T: ::core::marker::Send + 'a, const V: usize = 3>
        where T: ::core::convert::AsRef<::std::collections::BTreeMap<(Self, Self), {Rl}<'a>>>
        { V0(&'a super::base::B2<T, ::core::result::Result<u32, u16>>), V1 }";
    same_outcome_with_every_prefix(source, Scope::Mod, Expect::Accepted);
    // Analyzed as the item itself, `Self` is written as the Leaker's type: accepted.
    same_outcome_with_every_prefix(source, Scope::Intern, Expect::Accepted);
}

/// The bounds that name a leaked type, with the path spelled every way. These rejections and the
/// items that are accepted are those of `tests/bound_cycles.rs`.
#[test]
fn a_bound_that_names_a_leaked_type_is_decided_with_every_spelling() {
    const BOUND_CYCLE: &str = "is in a bound or where predicate";
    for scope in [Scope::Intern, Scope::Mod, Scope::Generics] {
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
    // module, the copy has a `Self` of its own: an entry that does not name `Self` is for any type,
    // and the copy passes its own; one that names it too is for the Leaker's type, which the bound
    // does not hold for in the copy: rejected.
    for source in [
        "pub struct S<T: AsRef<Self>>({Wrap}<T>);",
        "pub struct S<T>({Wrap}<T>) where T: AsRef<(Self, {Rl}<'static>)>;",
    ] {
        same_outcome_with_every_prefix(source, Scope::Mod, Expect::Accepted);
        same_outcome_with_every_prefix(source, Scope::Intern, Expect::Accepted);
    }
    for source in [
        "pub struct S<T: AsRef<Self>>({Wrap}<Self>, {Wrap}<T>);",
        "pub enum E<T> where T: AsRef<Self> { A({Wrap}<Self>), B(T) }",
    ] {
        same_outcome_with_every_prefix(
            source,
            Scope::Mod,
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
        for scope in [Scope::Intern, Scope::Mod] {
            same_outcome_with_every_prefix(source, scope, Expect::Accepted);
        }
    }
}

/// A type that the module declares is not an entry, spelled bare or with `self::`. Another module's
/// type with the same name (`crate::Rl`, `super::Rl`) is one.
#[test]
fn a_declared_type_is_neutral_bare_and_with_self() {
    let source = "pub enum E<'a, T: 'a> { A({Rl}<'a>, &'a T), B }";
    let bare = outcome(&spelled(source, ""), Scope::Mod, &["Rl"]);
    assert_eq!(
        bare,
        Outcome::Accepted {
            entries: Vec::new(),
            modules: Vec::new()
        },
        "declared: no entry"
    );
    assert_eq!(
        outcome(&spelled(source, "self::"), Scope::Mod, &["Rl"]),
        bare
    );
    // Not declared: an entry for each spelling, the same but for the prefix.
    let bare = outcome(&spelled(source, ""), Scope::Mod, &[]);
    assert!(matches!(&bare, Outcome::Accepted { entries, .. } if entries.len() == 1));
    for prefix in &PREFIXES[1..] {
        let actual = outcome(&spelled(source, prefix), Scope::Mod, &[]);
        assert_like_bare(&actual, &bare, prefix, source);
    }
}

//! Tests for `Leaker::intern_mod()`.

use proc_macro2::TokenStream;
use std::collections::BTreeSet;
use syn::visit_mut::VisitMut;
use syn::*;
use template_quote::quote;
use type_leak::{ContextId, Leaker, Visitor};

/// The tag of the format of a serialized Referrer, which it starts with.
fn tag() -> String {
    format!("type-leak-format-{}", type_leak::Referrer::FORMAT_VERSION)
}

/// Visits the items of the module copy `item` one by one with `visitor`, whose context is the
/// module the copy is of. A visited `mod` item is a module nested in the context, so the items
/// are what a copy of a whole module is rewritten with.
fn visit_items<F: FnMut(Type, usize) -> Type>(mut visitor: Visitor<F>, item: &mut ItemMod) {
    for item in &mut item.content.as_mut().unwrap().1 {
        visitor.visit_item_mut(item);
    }
}

fn module() -> ItemMod {
    parse_quote! {
        mod m {
            pub struct S<T> { x: Rel<T> }
            pub trait Tr {
                fn f(&self) -> Rel2;
                fn g<const B: bool>(&self) -> Wrap<B>;
            }
            pub type A = Rel3;
        }
    }
}

#[test]
fn intern_mod_leaks_with_local_generics() {
    let mut leaker = Leaker::new();
    leaker.intern_mod(&module()).unwrap();
    let referrer = leaker.finish();
    assert_eq!(
        referrer
            .iter_with_generics()
            .map(|(ty, generics)| (quote!(#ty).to_string(), quote!(#generics).to_string()))
            .collect::<BTreeSet<_>>(),
        [
            ("Rel < T >", "< T >"),
            ("Rel2", ""),
            ("Wrap < B >", "< const B : bool >"),
            ("Rel3", ""),
        ]
        .iter()
        .map(|(a, b)| (a.to_string(), b.to_string()))
        .collect::<BTreeSet<_>>()
    );
}

#[test]
fn intern_mod_referrer_rewrites_a_copy() {
    let mut leaker = Leaker::new();
    let context = leaker.intern_mod(&module()).unwrap();
    let referrer = leaker.finish();
    let mut item = module();
    visit_items(
        referrer.clone().visitor(context, |ty, ix| {
            let ty = quote!(#ty).to_string();
            let name = ty.replace(' ', "");
            let _ = ix;
            parse_str(&format!("P<{}>", name.replace(['<', '>'], "_"))).unwrap()
        }),
        &mut item,
    );
    let expected: ItemMod = parse_quote! {
        mod m {
            pub struct S<T> { x: P<Rel_T_> }
            pub trait Tr {
                fn f(&self) -> P<Rel2>;
                fn g<const B: bool>(&self) -> P<Wrap_B_>;
            }
            pub type A = P<Rel3>;
        }
    };
    assert_eq!(quote!(#item).to_string(), quote!(#expected).to_string());
}

#[test]
fn intern_mod_errors_and_bodies() {
    // A relative bound on a nested item is an error.
    let m: ItemMod = parse_quote! {
        mod m { pub struct S<T: Clone>(T); }
    };
    assert!(Leaker::new().intern_mod(&m).is_err());

    // Function bodies are ignored.
    let m: ItemMod = parse_quote! {
        mod m {
            pub fn f<T>() -> Rel<T> { let x: RelBody = RelBody::new(); x.into() }
        }
    };
    let mut leaker = Leaker::new();
    leaker.intern_mod(&m).unwrap();
    let referrer = leaker.finish();
    assert_eq!(
        referrer
            .iter()
            .map(|t| quote!(#t).to_string())
            .collect::<Vec<_>>(),
        ["Rel < T >"]
    );

    // A module without content has nothing to analyze.
    let m: ItemMod = parse_quote!(
        mod m;
    );
    assert!({
        let mut leaker = Leaker::new();
        leaker.intern_mod(&m).unwrap();
        leaker.finish()
    }
    .is_empty());
}

/// Analyzes the module like `Leaker::intern_mod()`, allowing the primitive types and the
/// standard prelude, and returns the module context too.
fn referrer_with_prelude(m: &ItemMod) -> (type_leak::Referrer, ContextId) {
    let mut config = type_leak::LeakerConfig::new();
    config.allow_primitive().allow_std_prelude();
    let mut leaker = Leaker::from_config(config);
    let context = leaker.intern_mod(m).unwrap();
    (leaker.finish(), context)
}

fn leaked(m: &ItemMod) -> Vec<String> {
    let (referrer, _) = referrer_with_prelude(m);
    let mut leaked: Vec<String> = referrer.iter().map(|t| quote!(#t).to_string()).collect();
    leaked.sort();
    leaked
}

/// The module whose items are used by paths relative to it.
fn portable_module() -> ItemMod {
    parse_quote! {
        mod m {
            pub struct S<T>(T);
            pub trait MyTrait {}
            pub const CAP: usize = 4;
            pub type A = S<u8>;
            pub struct U {
                a: S<Rel>,
                b: A,
                c: [u8; CAP],
                d: Box<dyn MyTrait>,
            }
            pub fn f<T: MyTrait>(x: S<T>) -> Rel2 { todo!() }
            pub mod inner {
                pub struct V(super::S<u8>, self::W, Wz);
                pub struct W;
            }
        }
    }
}

#[test]
fn intern_mod_items_are_portable() {
    // The items declared in the module (and the modules in it) are copied with it, so only the
    // others are leaked. `S<Rel>` is decomposed since it mentions `S`.
    assert_eq!(leaked(&portable_module()), ["Rel", "Rel2", "Wz"]);

    // Outside the root module, a path is relative.
    let m: ItemMod = parse_quote! {
        mod m {
            pub struct S;
            pub mod inner { pub struct V(super::super::X, super::S, S); }
        }
    };
    assert_eq!(leaked(&m), ["S", "super :: super :: X"]);

    // Names of enclosing modules are not in scope in a nested module (like in Rust).
    let m: ItemMod = parse_quote! {
        mod m {
            pub struct S;
            pub mod inner { pub struct T(S, self::S); }
        }
    };
    assert_eq!(leaked(&m), ["S", "self :: S"]);
}

#[test]
fn intern_mod_generics_shadow_module_items() {
    let m: ItemMod = parse_quote! {
        mod m {
            pub struct S;
            pub struct A<S>(S, Wrap<S>);
            pub struct B<T = ()>(S);
        }
    };
    // `S` in `A` is the generic parameter. Local generics are interned with the types depending
    // on them (see `Referrer::generics()`), but `Wrap<S>` is relative either way.
    let mut leaker = Leaker::new();
    leaker.intern_mod(&m).unwrap();
    let referrer = leaker.finish();
    assert_eq!(
        referrer
            .iter_with_generics()
            .map(|(ty, generics)| (quote!(#ty).to_string(), quote!(#generics).to_string()))
            .collect::<Vec<_>>(),
        [("Wrap < S >".to_string(), "< S >".to_string())]
    );
}

#[test]
fn intern_mod_imports() {
    // An absolute `use` makes the name portable (also renamed, or in a group), a relative one does
    // not.
    let m: ItemMod = parse_quote! {
        mod m {
            use ::p::Q;
            use ::p::{R as RR, self};
            use super::Rel;
            use crate::Rel2;
            use std::fmt::Debug as D;
            pub struct S(Q, RR, p::X, Rel, Rel2, D);
        }
    };
    assert_eq!(leaked(&m), ["D", "Rel", "Rel2"]);
}

#[test]
fn intern_mod_referrer_keeps_module_items() {
    let (referrer, context) = referrer_with_prelude(&portable_module());
    let mut item = portable_module();
    visit_items(
        referrer.visitor(context, |ty, _| {
            let name = quote!(#ty).to_string();
            parse_str(&format!("P_{}", name)).unwrap()
        }),
        &mut item,
    );
    let expected: ItemMod = parse_quote! {
        mod m {
            pub struct S<T>(T);
            pub trait MyTrait {}
            pub const CAP: usize = 4;
            pub type A = S<u8>;
            pub struct U {
                a: S<P_Rel>,
                b: A,
                c: [u8; CAP],
                d: Box<dyn MyTrait>,
            }
            pub fn f<T: MyTrait>(x: S<T>) -> P_Rel2 { todo!() }
            pub mod inner {
                pub struct V(super::S<u8>, self::W, P_Wz);
                pub struct W;
            }
        }
    };
    assert_eq!(quote!(#item).to_string(), quote!(#expected).to_string());
}

fn ident(name: &str) -> Ident {
    Ident::new(name, proc_macro2::Span::call_site())
}

fn nested_module() -> ItemMod {
    parse_quote! {
        mod m {
            pub struct A { y: Rel }
            pub mod inner {
                use super::super::Rel;
                pub struct C { y: Rel }
            }
        }
    }
}

/// The modules of the entries, as `type@a::b`.
fn modules_of(referrer: &type_leak::Referrer) -> Vec<String> {
    referrer
        .iter()
        .enumerate()
        .map(|(ix, ty)| {
            let module = match referrer.context_module(referrer.context(ix)) {
                None => String::new(),
                Some(module) => match referrer.module_path(module).unwrap() {
                    [] => "self".to_string(),
                    path => {
                        let path: Vec<String> = path.iter().map(|i| i.to_string()).collect();
                        path.join("::")
                    }
                },
            };
            format!("{}@{}", quote!(#ty), module)
        })
        .collect()
}

#[test]
fn same_tokens_in_different_modules_are_different_entries() {
    let mut leaker = Leaker::new();
    let top_level = leaker.intern_mod(&nested_module()).unwrap();
    let referrer = leaker.finish();
    assert_eq!(modules_of(&referrer), ["Rel@self", "Rel@inner"]);
    // The module has no generics, and its top-level context is in no module. The contexts are
    // those of the two modules and of the two entries.
    assert_eq!(referrer.contexts().count(), 4);
    let module = |ix: usize| referrer.context_module(referrer.context(ix)).unwrap();
    assert_eq!(referrer.module_path(module(0)), Some(&[][..]));
    assert_eq!(referrer.module_path(module(1)), Some(&[ident("inner")][..]));
    // The top-level context is the module analyzed, which is in no module.
    assert_eq!(top_level, module(0));
    assert_eq!(referrer.context_module(top_level), None);
    assert!(referrer.context_generics(top_level).params.is_empty());
}

#[test]
fn deep_module_path() {
    let m: ItemMod = parse_quote! {
        mod m { pub mod a { pub mod b { pub struct S<T> { x: Wrap<T> } } } }
    };
    let mut leaker = Leaker::new();
    leaker.intern_mod(&m).unwrap();
    let referrer = leaker.finish();
    assert_eq!(modules_of(&referrer), ["Wrap < T >@a::b"]);
    assert_eq!(
        referrer.module_path(referrer.context_module(referrer.context(0)).unwrap()),
        Some(&[ident("a"), ident("b")][..])
    );
}

#[test]
fn types_not_depending_on_the_module_are_shared() {
    let m: ItemMod = parse_quote! {
        mod m {
            pub struct A { x: ::foo::X, y: ::core::option::Option<::foo::X> }
            pub mod inner { pub struct C { x: ::foo::X } }
        }
    };
    // `::foo::X` is an optional entry, which does not depend on a module.
    let mut leaker = Leaker::new();
    leaker.intern_mod(&m).unwrap();
    let referrer = leaker.finish();
    assert_eq!(modules_of(&referrer), [":: foo :: X@"]);
    // The context of `::foo::X`, and the module analyzed.
    assert_eq!(referrer.contexts().count(), 2);
    // A relative type next to it is still keyed by its module.
    let m: ItemMod = parse_quote! {
        mod m {
            pub struct A { x: ::foo::X }
            pub mod inner { pub struct C { x: ::foo::X, y: &'static Rel } }
        }
    };
    assert_eq!(
        modules_of(&{
            let mut leaker = Leaker::new();
            leaker.intern_mod(&m).unwrap();
            leaker.finish()
        }),
        ["& 'static Rel@inner", ":: foo :: X@"]
    );
}

#[test]
fn referrer_rewrites_each_module_to_its_own_entry() {
    let mut leaker = Leaker::new();
    let context = leaker.intern_mod(&nested_module()).unwrap();
    let referrer = leaker.finish();
    let mut item = nested_module();
    visit_items(
        referrer.visitor(context, |_, ix| parse_str(&format!("P{}", ix)).unwrap()),
        &mut item,
    );
    let expected: ItemMod = parse_quote! {
        mod m {
            pub struct A { y: P0 }
            pub mod inner {
                use super::super::Rel;
                pub struct C { y: P1 }
            }
        }
    };
    assert_eq!(quote!(#item).to_string(), quote!(#expected).to_string());
}

#[test]
fn referrer_visits_a_copy_in_a_module() {
    let mut leaker = Leaker::new();
    let root = leaker.intern_mod(&nested_module()).unwrap();
    let referrer = leaker.finish();
    let inner = referrer.context_module(referrer.context(1)).unwrap();
    let copy = |module: ContextId| {
        let mut visitor = referrer
            .clone()
            .visitor(module, |_, ix| parse_str(&format!("P{}", ix)).unwrap());
        let mut item: ItemStruct = parse_quote!(
            pub struct C {
                y: Rel,
            }
        );
        visitor.visit_item_struct_mut(&mut item);
        quote!(#item).to_string()
    };
    let expected = |n: usize| quote!(pub struct C { y: #{ident(&format!("P{}", n))}, }).to_string();
    // The context says which module the copy is in.
    assert_eq!(copy(inner), expected(1));
    assert_eq!(copy(root), expected(0));
    // Only `inner` has the entry `Rel`, which does not match in the analyzed module.
    let m: ItemMod = parse_quote!(
        mod m {
            pub mod inner {
                pub struct C {
                    y: Rel,
                }
            }
        }
    );
    let mut leaker = Leaker::new();
    let root = leaker.intern_mod(&m).unwrap();
    let referrer = leaker.finish();
    let inner = referrer.context_module(referrer.context(0)).unwrap();
    let visit = |module: ContextId| {
        let mut item: ItemStruct = parse_quote!(
            pub struct C {
                y: Rel,
            }
        );
        let rewrite = |_: Type, _| -> Type { parse_quote!(P) };
        referrer
            .clone()
            .visitor(module, rewrite)
            .visit_item_struct_mut(&mut item);
        quote!(#item).to_string()
    };
    let unchanged = quote!(
        pub struct C {
            y: Rel,
        }
    )
    .to_string();
    let rewritten = quote!(
        pub struct C {
            y: P,
        }
    )
    .to_string();
    assert_eq!(visit(inner), rewritten);
    assert_eq!(visit(root), unchanged);
}

#[test]
fn referrer_tracks_the_modules_it_enters() {
    // Only `inner` has `Rel`, entered from a copy of the whole module.
    let m: ItemMod = parse_quote!(
        mod m {
            pub mod inner {
                pub struct C {
                    y: Rel,
                }
            }
        }
    );
    let mut leaker = Leaker::new();
    let root = leaker.intern_mod(&m).unwrap();
    let referrer = leaker.finish();
    let rewrite = |_: Type, _| -> Type { parse_quote!(P) };
    let unchanged: ItemMod = parse_quote!(
        mod inner {
            pub struct C {
                y: Rel,
            }
        }
    );
    let rewritten: ItemMod = parse_quote!(
        mod inner {
            pub struct C {
                y: P,
            }
        }
    );
    // A copy of the items of `m`: `inner` is a module in it, found by its name.
    let mut item = m.clone();
    visit_items(referrer.clone().visitor(root, rewrite), &mut item);
    let expected: ItemMod = parse_quote!(
        mod m {
            pub mod inner {
                pub struct C {
                    y: P,
                }
            }
        }
    );
    assert_eq!(quote!(#item).to_string(), quote!(#expected).to_string());
    // A copy of `inner` alone, in the module of `m`, is `m::inner`: the `mod` item is nested in
    // the module of the context, and it is found by its name. (It was `m` itself, so it did not
    // match, when a root `mod` item was the module.)
    let mut item = unchanged.clone();
    referrer
        .clone()
        .visitor(root, rewrite)
        .visit_item_mod_mut(&mut item);
    assert_eq!(quote!(#item).to_string(), quote!(#rewritten).to_string());
    // The same through `visit_item_mut()`, and as an item of a file.
    let mut item = Item::Mod(unchanged.clone());
    referrer
        .clone()
        .visitor(root, rewrite)
        .visit_item_mut(&mut item);
    assert_eq!(quote!(#item).to_string(), quote!(#rewritten).to_string());
    let mut file = File {
        shebang: None,
        attrs: Vec::new(),
        items: vec![Item::Mod(unchanged.clone())],
    };
    referrer
        .clone()
        .visitor(root, rewrite)
        .visit_file_mut(&mut file);
    assert_eq!(quote!(#file).to_string(), quote!(#rewritten).to_string());
    // With the context of the module `m::inner`, a `mod inner` is `m::inner::inner`, which has
    // no entry (the `mod` item is never the module itself, as it was when it was the root): to
    // rewrite a copy of `m::inner`, visit its items. (It was rewritten.)
    let inner = referrer.context_module(referrer.context(0)).unwrap();
    let mut item = unchanged.clone();
    referrer
        .clone()
        .visitor(inner, rewrite)
        .visit_item_mod_mut(&mut item);
    assert_eq!(quote!(#item).to_string(), quote!(#unchanged).to_string());
    let mut item = unchanged.clone();
    visit_items(referrer.visitor(inner, rewrite), &mut item);
    assert_eq!(quote!(#item).to_string(), quote!(#rewritten).to_string());
}

#[test]
fn serialization_with_modules() {
    use template_quote::ToTokens;
    let m: ItemMod = parse_quote! {
        mod m {
            pub struct A { y: Rel }
            pub mod a { pub mod b { pub struct C<T> { y: Rel, z: Wrap<T> } } }
        }
    };
    let mut leaker = Leaker::new();
    leaker.intern_mod(&m).unwrap();
    let referrer = leaker.finish();
    assert_eq!(
        modules_of(&referrer),
        ["Rel@self", "Rel@a::b", "Wrap < T >@a::b"]
    );
    let tokens = referrer.to_token_stream();
    let version = tag();
    assert_eq!(
        tokens.to_string(),
        format!(
            "\"{}\" 6 mod {{ path = [] }} ; () mod {{ path = [a] , parent = 0 }} ; () \
             mod {{ path = [a , b] , parent = 1 }} ; () mod 0 ; (Rel) mod 2 ; (Rel) \
             < T > ; mod 2 ; (Wrap < T >)",
            version
        )
    );
    // It can be followed by other tokens.
    struct Followed(type_leak::Referrer, TokenStream);
    impl parse::Parse for Followed {
        fn parse(input: parse::ParseStream) -> Result<Self> {
            Ok(Self(input.parse()?, input.parse()?))
        }
    }
    let followed: Followed = parse2(quote!(#tokens struct Z;)).unwrap();
    assert_eq!(followed.0, referrer);
    assert_eq!(followed.1.to_string(), "struct Z ;");
    assert_eq!(parse2::<type_leak::Referrer>(tokens).unwrap(), referrer);
    // The simplest Referrer has no module (nor has one of an analysis without modules).
    let simple: ItemStruct = parse_quote!(
        pub struct A(Rel, Rel2);
    );
    let tokens = {
        let mut leaker = Leaker::new();
        leaker.intern_struct(&simple, true).unwrap();
        leaker.finish()
    }
    .to_token_stream();
    assert_eq!(tokens.to_string(), format!("\"{}\" (Rel , Rel2)", version));
}

#[test]
fn root_relative_type_does_not_match_in_a_nested_module() {
    let m: ItemMod = parse_quote!(
        mod m {
            pub struct A {
                y: Rel,
            }
        }
    );
    let mut leaker = Leaker::new();
    let root = leaker.intern_mod(&m).unwrap();
    // A module which no entry depends on: it is a context of the Referrer, but it has no entry.
    let (other, ()) = leaker.enter_mod("other", [], |_| ()).unwrap();
    let referrer = leaker.finish();
    let copy = |module: ContextId| {
        let mut visitor = referrer.clone().visitor(module, |_, _| parse_quote!(P));
        let mut item: ItemStruct = parse_quote!(
            pub struct C {
                y: Rel,
            }
        );
        visitor.visit_item_struct_mut(&mut item);
        quote!(#item).to_string()
    };
    let unchanged = quote!(
        pub struct C {
            y: Rel,
        }
    )
    .to_string();
    let rewritten = quote!(
        pub struct C {
            y: P,
        }
    )
    .to_string();
    assert_eq!(copy(root), rewritten);
    assert_eq!(copy(other), unchanged);
    // A copy of the items of the module with a nested module without an entry for it.
    let mut item: ItemMod = parse_quote!(
        mod m {
            pub struct A {
                y: Rel,
            }
            pub mod inner {
                pub struct C {
                    y: Rel,
                }
            }
        }
    );
    visit_items(referrer.visitor(root, |_, _| parse_quote!(P)), &mut item);
    let expected: ItemMod = parse_quote!(
        mod m {
            pub struct A {
                y: P,
            }
            pub mod inner {
                pub struct C {
                    y: Rel,
                }
            }
        }
    );
    assert_eq!(quote!(#item).to_string(), quote!(#expected).to_string());
}

#[test]
#[should_panic(expected = "is not one of the contexts of this Referrer")]
fn a_context_which_is_not_known_panics() {
    let m: ItemMod = parse_quote!(
        mod m {
            pub struct A {
                y: Rel,
            }
        }
    );
    let mut leaker = Leaker::new();
    leaker.intern_mod(&m).unwrap();
    let referrer = leaker.finish();
    let unknown: ContextId = parse_str("99").unwrap();
    let _ = referrer.visitor(unknown, |_, _| parse_quote!(P));
}

// ── a module found with no module around it ──────────────────────────────────────────────────

/// The path of the module the `ix`-th entry is in.
fn path_of(referrer: &type_leak::Referrer, ix: usize) -> Vec<String> {
    let module = referrer.context_module(referrer.context(ix)).unwrap();
    referrer
        .module_path(module)
        .unwrap()
        .iter()
        .map(|ident| ident.to_string())
        .collect()
}

#[test]
fn a_module_found_with_no_module_around_has_its_name_as_path() {
    // `View::visit_item()` in `enter_generics()`, which is not in a module: the module is nested in
    // nothing, so its path is its own name and the modules in it start with it.
    let inner: ItemMod = parse_quote!(
        mod inner {
            pub struct C {
                a: Rel,
            }
            pub mod deep {
                pub struct D {
                    a: Rel,
                }
            }
        }
    );
    let mut leaker = Leaker::new();
    leaker
        .enter_generics(&Generics::default(), |view| {
            view.visit_item(&Item::Mod(inner.clone()))
        })
        .unwrap();
    let referrer = leaker.finish();
    assert_eq!(referrer.iter().count(), 2);
    assert_eq!(path_of(&referrer, 0), ["inner"]);
    assert_eq!(path_of(&referrer, 1), ["inner", "deep"]);
    // It is a root, which has no parent (it is not the empty path of the module `intern_mod()`
    // analyzes), and the one in it has: `parent` is written for it only.
    let text = {
        use template_quote::ToTokens;
        referrer.to_token_stream().to_string()
    };
    assert!(text.contains("mod { path = [inner] }"), "{}", text);
    assert!(
        text.contains("mod { path = [inner , deep] , parent = "),
        "{}",
        text
    );
    // Entered by `enter_mod()` (which has no path), the path of the module found in it starts at
    // it, and the modules in that one start with it.
    let mut leaker = Leaker::new();
    leaker
        .enter_mod("outer", [], |view| {
            view.visit_item(&Item::Mod(inner.clone()))
        })
        .unwrap();
    let referrer = leaker.finish();
    assert_eq!(path_of(&referrer, 0), ["inner"]);
    assert_eq!(path_of(&referrer, 1), ["inner", "deep"]);
    // The one `intern_mod()` analyzes has the empty path, and the ones in it are relative to it.
    let outer: ItemMod = parse_quote!(mod outer { #inner });
    let mut leaker = Leaker::new();
    let top = leaker.intern_mod(&outer).unwrap();
    let referrer = leaker.finish();
    assert_eq!(referrer.module_path(top), Some(&[][..]));
    assert_eq!(path_of(&referrer, 0), ["inner"]);
    assert_eq!(path_of(&referrer, 1), ["inner", "deep"]);
}

#[test]
fn a_visitor_finds_the_roots_found_with_no_module_around_by_their_names() {
    // The module found with no module around it is a root: a visitor of a context that is in no
    // module finds the `mod` item visited by its name, among the roots.
    let inner: ItemMod = parse_quote!(
        mod inner {
            pub struct C {
                a: Rel,
            }
        }
    );
    let mut leaker = Leaker::new();
    let (scope, ()) = leaker
        .enter_generics(&Generics::default(), |view| {
            view.visit_item(&Item::Mod(inner.clone()))
        })
        .unwrap();
    let referrer = leaker.finish();
    let rewrite = |item: &ItemMod, module: ContextId| {
        let mut item = item.clone();
        referrer
            .clone()
            .visitor(module, |_, _| parse_quote!(P))
            .visit_item_mod_mut(&mut item);
        quote!(#item).to_string()
    };
    let rewritten = quote!(
        mod inner {
            pub struct C {
                a: P,
            }
        }
    )
    .to_string();
    assert_eq!(referrer.context_module(scope), None);
    assert_eq!(rewrite(&inner, scope), rewritten);
    // With the id of the module, the `mod` item is nested in it, not the module itself.
    let id = referrer.context_module(referrer.context(0)).unwrap();
    assert_eq!(rewrite(&inner, id), quote!(#inner).to_string());
    // Several roots are told apart by their names, whereas a visitor that did not know its module
    // could not tell them (nothing was rewritten then).
    let other: ItemMod = parse_quote!(
        mod other {
            pub struct C {
                a: Rel,
            }
        }
    );
    let mut leaker = Leaker::new();
    let (scope, ()) = leaker
        .enter_generics(&Generics::default(), |view| {
            view.visit_item(&Item::Mod(inner.clone()));
            view.visit_item(&Item::Mod(other.clone()));
        })
        .unwrap();
    let referrer = leaker.finish();
    let ids: Vec<ContextId> = (0..2)
        .map(|ix| referrer.context_module(referrer.context(ix)).unwrap())
        .collect();
    assert_ne!(ids[0], ids[1]);
    let rewrite = |item: &ItemMod, module: ContextId| {
        let mut item = item.clone();
        referrer
            .clone()
            .visitor(module, |_, ix| parse_str(&format!("P{}", ix)).unwrap())
            .visit_item_mod_mut(&mut item);
        quote!(#item).to_string()
    };
    assert_eq!(
        rewrite(&inner, scope),
        quote!(
            mod inner {
                pub struct C {
                    a: P0,
                }
            }
        )
        .to_string()
    );
    assert_eq!(
        rewrite(&other, scope),
        quote!(
            mod other {
                pub struct C {
                    a: P1,
                }
            }
        )
        .to_string()
    );
    // The entry of one is not used for the other: the module of the entry is the one that is found.
    let swapped: ItemMod = parse_quote!(
        mod unknown {
            pub struct C {
                a: Rel,
            }
        }
    );
    assert_eq!(rewrite(&swapped, scope), quote!(#swapped).to_string());
}

// ── the module given by the context of the visitor ──────────────────────────────────────────

#[test]
fn a_copy_of_the_second_module_is_rewritten_with_its_id() {
    let m1: ItemMod = parse_quote!(
        mod m1 {
            pub struct S {
                f: Rel,
            }
        }
    );
    let m2: ItemMod = parse_quote!(
        mod m2 {
            pub struct S {
                f: Rel,
            }
        }
    );
    let mut leaker = Leaker::new();
    let a = leaker.intern_mod(&m1).unwrap();
    let b = leaker.intern_mod(&m2).unwrap();
    assert_ne!(a, b);
    let referrer = leaker.finish();
    assert_eq!(referrer.iter().count(), 2);
    let index = |module: ContextId| {
        (0..2)
            .find(|&ix| referrer.context_module(referrer.context(ix)) == Some(module))
            .unwrap()
    };
    // The items of the copy are rewritten with the entry of the module of the context.
    let rewrite = |item: &ItemMod, module: ContextId| {
        let mut item = item.clone();
        visit_items(
            referrer
                .clone()
                .visitor(module, |_, ix| parse_str(&format!("P{}", ix)).unwrap()),
            &mut item,
        );
        quote!(#item).to_string()
    };
    assert_eq!(
        rewrite(&m2, b),
        quote!(mod m2 { pub struct S { f: #{ident(&format!("P{}", index(b)))}, } }).to_string()
    );
    assert_eq!(
        rewrite(&m1, a),
        quote!(mod m1 { pub struct S { f: #{ident(&format!("P{}", index(a)))}, } }).to_string()
    );
    // With the context of the other, the entry of the other is the one found.
    assert_eq!(
        rewrite(&m2, a),
        quote!(mod m2 { pub struct S { f: #{ident(&format!("P{}", index(a)))}, } }).to_string()
    );
    // A `mod` item is nested in the module of the context, so the copy visited as a whole is
    // `m1::m2`, which has no entry. (The root `mod` item was the module itself, and then it was
    // rewritten.)
    let whole = |item: &ItemMod, module: ContextId| {
        let mut item = item.clone();
        referrer
            .clone()
            .visitor(module, |_, ix| parse_str(&format!("P{}", ix)).unwrap())
            .visit_item_mod_mut(&mut item);
        quote!(#item).to_string()
    };
    assert_eq!(whole(&m2, b), quote!(#m2).to_string());
    assert_eq!(whole(&m1, a), quote!(#m1).to_string());
    assert_eq!(whole(&m2, a), quote!(#m2).to_string());
}

#[test]
fn the_context_is_the_module_and_a_mod_item_is_nested_in_it() {
    let m: ItemMod = parse_quote!(
        mod m {
            pub struct A {
                y: Rel,
            }
            pub mod inner {
                pub struct C {
                    y: Rel,
                }
            }
        }
    );
    let mut leaker = Leaker::new();
    let root = leaker.intern_mod(&m).unwrap();
    let referrer = leaker.finish();
    let inner = referrer.context_module(referrer.context(1)).unwrap();
    let rewritten = quote!(
        mod m {
            pub struct A {
                y: P0,
            }
            pub mod inner {
                pub struct C {
                    y: P1,
                }
            }
        }
    )
    .to_string();
    let unchanged = quote!(#m).to_string();
    let visitor = |module: ContextId| {
        referrer
            .clone()
            .visitor(module, |_, ix| parse_str(&format!("P{}", ix)).unwrap())
    };
    // The items of `m` are in the module analyzed, and `inner` is found by its name.
    let by_items = |module: ContextId| {
        let mut item = m.clone();
        visit_items(visitor(module), &mut item);
        quote!(#item).to_string()
    };
    assert_eq!(by_items(root), rewritten);
    // The same as the items of a file, which are those of the module of the context.
    let by_file = |module: ContextId| {
        let mut file = File {
            shebang: None,
            attrs: Vec::new(),
            items: m.content.clone().unwrap().1,
        };
        visitor(module).visit_file_mut(&mut file);
        let items = &file.items;
        quote!(mod m { #(#items)* }).to_string()
    };
    assert_eq!(by_file(root), rewritten);
    // `mod m` as a whole is `m::m` in the module of the context: nothing is found in it, as no
    // entry depends on it. The `mod` item is never the module itself. (It was rewritten when it
    // was the root node.)
    let by_mod = |module: ContextId, as_file: bool| {
        let mut item = m.clone();
        let mut visitor = visitor(module);
        if as_file {
            let mut file = File {
                shebang: None,
                attrs: Vec::new(),
                items: vec![Item::Mod(item)],
            };
            visitor.visit_file_mut(&mut file);
            let item = &file.items[0];
            quote!(#item).to_string()
        } else {
            visitor.visit_item_mod_mut(&mut item);
            quote!(#item).to_string()
        }
    };
    assert_eq!(by_mod(root, false), unchanged);
    assert_eq!(by_mod(root, true), unchanged);
    // With the context of the module `inner`, the items of `m` are looked up in it (`Rel` is the
    // entry of `inner`), and the `inner` in it is a module in it, which is none.
    assert_eq!(
        by_items(inner),
        quote!(
            mod m {
                pub struct A {
                    y: P1,
                }
                pub mod inner {
                    pub struct C {
                        y: Rel,
                    }
                }
            }
        )
        .to_string()
    );
}

// ── `Referrer::visitor()`: the lookup of the module ─────────────────────────────────────────

/// `item` as it is printed.
fn shown(item: Item) -> String {
    quote!(#item).to_string()
}

/// Rewrites `item`, a copy in the scope `context`, replacing the `ix`-th entry with `P<ix>`.
fn rewrite_in(referrer: &type_leak::Referrer, context: ContextId, mut item: Item) -> String {
    referrer
        .clone()
        .visitor(context, |_, ix| parse_quote!(P<#ix>))
        .visit_item_mut(&mut item);
    quote!(#item).to_string()
}

/// README, Modules: a `mod` item is a module nested in the context and is found by its name, but
/// only the modules the Leaker saw: a module nested in a module that the Leaker did not see (its
/// parent is another one) matches only the entries with no module. Audit mutants
/// `vis_module_in_unknown_outside` and `vis_module_in_elsewhere_stays`.
#[test]
fn a_module_nested_in_an_unseen_module_is_not_found_by_its_name() {
    let mut leaker = Leaker::new();
    let (context, ()) = leaker
        .enter_generics(&Generics::default(), |view| {
            view.visit_item(&parse_quote!(
                mod inner {
                    pub struct C(Rel);
                }
            ));
        })
        .unwrap();
    let referrer = leaker.finish();
    // The module `inner` is seen: its `Rel` is replaced.
    let seen = rewrite_in(
        &referrer,
        context,
        parse_quote!(
            mod inner {
                pub struct C(Rel);
            }
        ),
    );
    assert!(seen.contains("P <"), "{}", seen);
    // Nested in `other`, which the Leaker did not see, it is another module.
    for item in [
        parse_quote!(
            mod other {
                mod inner {
                    pub struct C(Rel);
                }
            }
        ),
        parse_quote!(
            mod other {
                mod mid {
                    mod inner {
                        pub struct C(Rel);
                    }
                }
            }
        ),
    ] {
        let item: Item = item;
        let unseen = rewrite_in(&referrer, context, item.clone());
        assert!(!unseen.contains("P <"), "{}", unseen);
        assert_eq!(unseen, shown(item.clone()));
    }
}

/// README, Modules: an entry of the module is preferred to an entry with no module for the same
/// type. Audit mutants `vis_rank_module_last` and `vis_rank_nomodule_equal`.
#[test]
fn the_entry_of_the_module_is_preferred_to_the_one_with_no_module() {
    let mut leaker = Leaker::new();
    // `Rel` with no module is the first entry, and the one of the module `m` is the second.
    let top = leaker
        .intern_struct(
            &parse_quote!(
                struct A {
                    x: Rel,
                }
            ),
            true,
        )
        .unwrap();
    let module = leaker
        .intern_mod(&parse_quote!(
            mod m {
                pub struct B(Rel);
            }
        ))
        .unwrap();
    let referrer = leaker.finish();
    assert_eq!(referrer.iter().count(), 2);
    assert_eq!(referrer.context_module(referrer.context(0)), None);
    assert_eq!(referrer.context_module(referrer.context(1)), Some(module));
    // The copy of `m` takes its own entry.
    assert_eq!(
        rewrite_in(
            &referrer,
            module,
            parse_quote!(
                pub struct B(Rel);
            )
        ),
        shown(parse_quote!(
            pub struct B(P<1usize>);
        ))
    );
    // A copy with no module around takes the other.
    assert_eq!(
        rewrite_in(
            &referrer,
            top,
            parse_quote!(
                struct A {
                    x: Rel,
                }
            )
        ),
        shown(parse_quote!(
            struct A {
                x: P<0usize>,
            }
        ))
    );
    // The same with the module analyzed first.
    let mut leaker = Leaker::new();
    let module = leaker
        .intern_mod(&parse_quote!(
            mod m {
                pub struct B(Rel);
            }
        ))
        .unwrap();
    let top = leaker
        .intern_struct(
            &parse_quote!(
                struct A {
                    x: Rel,
                }
            ),
            true,
        )
        .unwrap();
    let referrer = leaker.finish();
    assert_eq!(
        rewrite_in(
            &referrer,
            top,
            parse_quote!(
                struct A {
                    x: Rel,
                }
            )
        ),
        shown(parse_quote!(
            struct A {
                x: P<1usize>,
            }
        ))
    );
    assert_eq!(
        rewrite_in(
            &referrer,
            module,
            parse_quote!(
                pub struct B(Rel);
            )
        ),
        shown(parse_quote!(
            pub struct B(P<0usize>);
        ))
    );
}

/// README, Modules: the items of a module are nested items, so their generics are local even if
/// the visitor is told that the outermost generics are those of the context
/// (`with_outermost_generics_as_leakers()`). Audit mutants `vis_item_depth_module`,
/// `vis_mod_item_not_nested` and `vis_mod_item_self_kept`.
#[test]
fn the_items_of_a_module_are_nested_items() {
    // The context is the module itself.
    let mut leaker = Leaker::new();
    let module = leaker
        .intern_mod(&parse_quote!(
            mod m {
                pub struct S<T>(Wrap<T>);
            }
        ))
        .unwrap();
    let referrer = leaker.finish();
    let mut item: Item = parse_quote!(
        pub struct S<T>(Wrap<T>);
    );
    referrer
        .clone()
        .visitor(module, |_, ix| parse_quote!(P<#ix>))
        .with_outermost_generics_as_leakers()
        .visit_item_mut(&mut item);
    assert_eq!(
        quote!(#item).to_string(),
        shown(parse_quote!(
            pub struct S<T>(P<0usize>);
        ))
    );
    // The context is a scope around a `mod` item.
    let mut leaker = Leaker::new();
    let (context, ()) = leaker
        .enter_generics(&Generics::default(), |view| {
            view.visit_item(&parse_quote!(
                mod m {
                    pub struct S<T>(Wrap<T>);
                }
            ));
        })
        .unwrap();
    let referrer = leaker.finish();
    let mut item: Item = parse_quote!(
        mod m {
            pub struct S<T>(Wrap<T>);
        }
    );
    referrer
        .clone()
        .visitor(context, |_, ix| parse_quote!(P<#ix>))
        .with_outermost_generics_as_leakers()
        .visit_item_mut(&mut item);
    assert_eq!(
        quote!(#item).to_string(),
        shown(parse_quote!(
            mod m {
                pub struct S<T>(P<0usize>);
            }
        ))
    );
}

/// README, Modules: an item declared in a block hides the entry of a module that names it, as it
/// does outside a module. Audit mutant `vis_shadow_retain_module_flag`.
#[test]
fn a_block_item_hides_the_entry_of_a_module() {
    let mut leaker = Leaker::new();
    let module = leaker
        .intern_mod(&parse_quote!(
            mod m {
                pub fn f(a: Rel) {}
            }
        ))
        .unwrap();
    let referrer = leaker.finish();
    let out = rewrite_in(
        &referrer,
        module,
        parse_quote!(
            pub fn f(a: Rel) {
                struct Rel;
                let b: Rel = Rel;
            }
        ),
    );
    // `a` is replaced, `b` is the local struct.
    assert_eq!(
        out,
        shown(parse_quote!(
            pub fn f(a: P<0usize>) {
                struct Rel;
                let b: Rel = Rel;
            }
        ))
    );
}

/// README, Modules / Self rules: `Self` has no meaning in a `mod` item, even inside the scope of a
/// trait that declares it, so the entry that names the `Self` of the trait is not found in it.
/// Audit mutant `vis_mod_item_self_kept` (the `Self` of the scope around stays in a `mod` item).
#[test]
fn self_has_no_meaning_in_a_mod_item_of_a_scope_which_declares_it() {
    let mut leaker = Leaker::new();
    let context = leaker
        .intern_trait(
            &parse_quote!(
                trait Tr {
                    fn f(&self) -> Rel<Self>;
                }
            ),
            true,
        )
        .unwrap();
    let referrer = leaker.finish();
    // The trait itself declares `Self`, and the entry is found.
    let expected: Item = parse_quote!(
        trait Tr {
            fn f(&self) -> P<0usize>;
        }
    );
    assert_eq!(
        rewrite_in(
            &referrer,
            context,
            parse_quote!(
                trait Tr {
                    fn f(&self) -> Rel<Self>;
                }
            )
        ),
        quote!(#expected).to_string()
    );
    // In a module it is not the `Self` of the trait: nothing is replaced.
    for src in [
        "mod inner { type X = Rel<Self>; }",
        "mod inner { const C: Option<Rel<Self>> = None; }",
        "mod inner { fn g() -> Rel<Self> { loop {} } }",
        "mod inner { static S: Rel<Self> = loop {}; }",
    ] {
        let item: Item = parse_str(src).unwrap();
        assert_eq!(
            rewrite_in(&referrer, context, item.clone()),
            quote!(#item).to_string(),
            "{}",
            src
        );
    }
}

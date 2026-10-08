//! Tests for `Leaker::intern_mod()`.

use proc_macro2::TokenStream;
use std::collections::BTreeSet;
use syn::visit_mut::VisitMut;
use syn::*;
use template_quote::quote;
use type_leak::{ContextId, Leaker};

/// The tag of the format of a serialized Referrer, which it starts with.
fn tag() -> String {
    format!("type-leak-format-{}", type_leak::Referrer::FORMAT_VERSION)
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
    leaker.intern_mod(&module()).unwrap();
    let referrer = leaker.finish();
    let mut item = module();
    referrer
        .clone()
        .into_visitor(None, None, |ty, ix| {
            let ty = quote!(#ty).to_string();
            let name = ty.replace(' ', "");
            let _ = ix;
            parse_str(&format!("P<{}>", name.replace(['<', '>'], "_"))).unwrap()
        })
        .visit_item_mod_mut(&mut item);
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
/// standard prelude.
fn referrer_with_prelude(m: &ItemMod) -> type_leak::Referrer {
    let mut config = type_leak::LeakerConfig::new();
    config.allow_primitive().allow_std_prelude();
    let mut leaker = Leaker::from_config(config);
    leaker.intern_mod(m).unwrap();
    leaker.finish()
}

fn leaked(m: &ItemMod) -> Vec<String> {
    let referrer = referrer_with_prelude(m);
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
    let referrer = referrer_with_prelude(&portable_module());
    let mut item = portable_module();
    referrer
        .into_visitor(None, None, |ty, _| {
            let name = quote!(#ty).to_string();
            parse_str(&format!("P_{}", name)).unwrap()
        })
        .visit_item_mod_mut(&mut item);
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
    leaker.intern_mod(&nested_module()).unwrap();
    let referrer = leaker.finish();
    let mut item = nested_module();
    referrer
        .into_visitor(None, None, |_, ix| parse_str(&format!("P{}", ix)).unwrap())
        .visit_item_mod_mut(&mut item);
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
    let copy = |module: Option<ContextId>| {
        let mut visitor = referrer.clone().into_visitor(None, module, |_, ix| {
            parse_str(&format!("P{}", ix)).unwrap()
        });
        let mut item: ItemStruct = parse_quote!(
            pub struct C {
                y: Rel,
            }
        );
        visitor.visit_item_struct_mut(&mut item);
        quote!(#item).to_string()
    };
    let expected = |n: usize| quote!(pub struct C { y: #{ident(&format!("P{}", n))}, }).to_string();
    // In the module of the entry.
    assert_eq!(copy(Some(inner)), expected(1));
    assert_eq!(copy(Some(root)), expected(0));
    // The module is unknown, so the analyzed module is preferred.
    assert_eq!(copy(None), expected(0));
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
    let visit = |visitor: type_leak::Visitor<_>| {
        let mut item: ItemStruct = parse_quote!(
            pub struct C {
                y: Rel,
            }
        );
        let mut visitor = visitor;
        visitor.visit_item_struct_mut(&mut item);
        quote!(#item).to_string()
    };
    let rewrite = |_: Type, _| -> Type { parse_quote!(P) };
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
    // Unknown module: any is accepted.
    assert_eq!(
        visit(referrer.clone().into_visitor(None, None, rewrite)),
        rewritten
    );
    assert_eq!(
        visit(referrer.clone().into_visitor(None, Some(inner), rewrite)),
        rewritten
    );
    assert_eq!(
        visit(referrer.clone().into_visitor(None, Some(root), rewrite)),
        unchanged
    );
    // The module is always compared with a root.
    let root = Generics::default();
    assert_eq!(
        visit(
            referrer
                .clone()
                .into_visitor_with_root(&root, None, None, rewrite)
        ),
        unchanged
    );
    assert_eq!(
        visit(referrer.into_visitor_with_root(&root, None, Some(inner), rewrite)),
        rewritten
    );
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
    leaker.intern_mod(&m).unwrap();
    let referrer = leaker.finish();
    let mut item = m.clone();
    referrer
        .clone()
        .into_visitor_with_root(&Generics::default(), None, None, |_, _| parse_quote!(P))
        .visit_item_mod_mut(&mut item);
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
    // A copy of `inner` alone, whose first module is the root `[]`, does not match.
    let mut item: ItemMod = parse_quote!(
        mod inner {
            pub struct C {
                y: Rel,
            }
        }
    );
    referrer
        .clone()
        .into_visitor_with_root(&Generics::default(), None, None, |_, _| parse_quote!(P))
        .visit_item_mod_mut(&mut item);
    let unchanged: ItemMod = parse_quote!(
        mod inner {
            pub struct C {
                y: Rel,
            }
        }
    );
    assert_eq!(quote!(#item).to_string(), quote!(#unchanged).to_string());
    // Unless it is a copy of the module `m::inner`, given by its id: the root node visited is a
    // `mod` item, which is that module, not one nested in it (it would be `m::inner::inner`).
    let inner = referrer.context_module(referrer.context(0)).unwrap();
    let mut item: ItemMod = parse_quote!(
        mod inner {
            pub struct C {
                y: Rel,
            }
        }
    );
    referrer
        .clone()
        .into_visitor_with_root(&Generics::default(), None, Some(inner), |_, _| {
            parse_quote!(P)
        })
        .visit_item_mod_mut(&mut item);
    let rewritten: ItemMod = parse_quote!(
        mod inner {
            pub struct C {
                y: P,
            }
        }
    );
    assert_eq!(quote!(#item).to_string(), quote!(#rewritten).to_string());
    // The same through `visit_item_mut()`, but not for a `mod` which is an item of a file (or of
    // the module): that one is nested in it, and there is no `m::inner::inner`.
    let mut item = Item::Mod(unchanged.clone());
    referrer
        .clone()
        .into_visitor_with_root(&Generics::default(), None, Some(inner), |_, _| {
            parse_quote!(P)
        })
        .visit_item_mut(&mut item);
    assert_eq!(quote!(#item).to_string(), quote!(#rewritten).to_string());
    let mut file = File {
        shebang: None,
        attrs: Vec::new(),
        items: vec![Item::Mod(unchanged.clone())],
    };
    referrer
        .into_visitor_with_root(&Generics::default(), None, Some(inner), |_, _| {
            parse_quote!(P)
        })
        .visit_file_mut(&mut file);
    assert_eq!(quote!(#file).to_string(), quote!(#unchanged).to_string());
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
    let referrer = leaker.finish();
    // A module which no entry depends on.
    let other: ContextId = parse_str("99").unwrap();
    let copy = |module: Option<ContextId>| {
        let mut visitor = referrer
            .clone()
            .into_visitor(None, module, |_, _| parse_quote!(P));
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
    assert_eq!(copy(Some(root)), rewritten);
    assert_eq!(copy(None), rewritten);
    assert_eq!(copy(Some(other)), unchanged);
    // A copy of the whole module with a nested module without an entry for it.
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
    referrer
        .into_visitor_with_root(&Generics::default(), None, None, |_, _| parse_quote!(P))
        .visit_item_mod_mut(&mut item);
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
fn a_visitor_which_does_not_know_the_module_finds_the_only_root_by_its_name() {
    // The module found with no module around it is the only root: a visitor which does not know
    // its module takes the `mod` item visited to be it if it has its name.
    let inner: ItemMod = parse_quote!(
        mod inner {
            pub struct C {
                a: Rel,
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
    let rewrite = |item: &ItemMod, module: Option<ContextId>| {
        let mut item = item.clone();
        referrer
            .clone()
            .into_visitor(None, module, |_, _| parse_quote!(P))
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
    assert_eq!(rewrite(&inner, None), rewritten);
    let id = referrer.context_module(referrer.context(0)).unwrap();
    assert_eq!(rewrite(&inner, Some(id)), rewritten);
    // With another root, nothing is guessed: it may be either of them. The entry of each is not
    // preferred to the other one's (the first was preferred before).
    let other: ItemMod = parse_quote!(
        mod other {
            pub struct C {
                a: Rel,
            }
        }
    );
    let mut leaker = Leaker::new();
    leaker
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
    let rewrite = |item: &ItemMod, module: Option<ContextId>| {
        let mut item = item.clone();
        referrer
            .clone()
            .into_visitor(None, module, |_, ix| {
                parse_str(&format!("P{}", ix)).unwrap()
            })
            .visit_item_mod_mut(&mut item);
        quote!(#item).to_string()
    };
    assert_eq!(rewrite(&inner, None), quote!(#inner).to_string());
    assert_eq!(rewrite(&other, None), quote!(#other).to_string());
    assert_eq!(
        rewrite(&inner, Some(ids[0])),
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
        rewrite(&other, Some(ids[1])),
        quote!(
            mod other {
                pub struct C {
                    a: P1,
                }
            }
        )
        .to_string()
    );
}

// ── the module given when the visitor is created ─────────────────────────────────────────────

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
    let rewrite = |item: &ItemMod, module: Option<ContextId>| {
        let mut item = item.clone();
        referrer
            .clone()
            .into_visitor(None, module, |_, ix| {
                parse_str(&format!("P{}", ix)).unwrap()
            })
            .visit_item_mod_mut(&mut item);
        quote!(#item).to_string()
    };
    // The root node is the module with the id, so the whole copy is rewritten (nothing was before:
    // it was looked up as a module in it, which has no entry).
    assert_eq!(
        rewrite(&m2, Some(b)),
        quote!(mod m2 { pub struct S { f: #{ident(&format!("P{}", index(b)))}, } }).to_string()
    );
    assert_eq!(
        rewrite(&m1, Some(a)),
        quote!(mod m1 { pub struct S { f: #{ident(&format!("P{}", index(a)))}, } }).to_string()
    );
    // With the module of the other, the entry of the other is the one found.
    assert_eq!(
        rewrite(&m2, Some(a)),
        quote!(mod m2 { pub struct S { f: #{ident(&format!("P{}", index(a)))}, } }).to_string()
    );
    // A visitor which does not know the module does not guess it, there being two roots: it does
    // not take the first (`m1`) for the one the copy is, nor its entry for the entry of the copy.
    assert_eq!(rewrite(&m2, None), quote!(#m2).to_string());
    assert_eq!(rewrite(&m1, None), quote!(#m1).to_string());
    // The same with the exact contexts.
    let rewrite_exact = |item: &ItemMod, module: Option<ContextId>| {
        let mut item = item.clone();
        referrer
            .clone()
            .into_visitor_with_root(&Generics::default(), None, module, |_, ix| {
                parse_str(&format!("P{}", ix)).unwrap()
            })
            .visit_item_mod_mut(&mut item);
        quote!(#item).to_string()
    };
    assert_eq!(
        rewrite_exact(&m2, Some(b)),
        quote!(mod m2 { pub struct S { f: #{ident(&format!("P{}", index(b)))}, } }).to_string()
    );
    assert_eq!(rewrite_exact(&m2, None), quote!(#m2).to_string());
}

#[test]
fn the_root_node_is_the_module_and_the_modules_in_it_are_nested_in_it() {
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
    let visit = |module: Option<ContextId>, root_node: bool| {
        let mut item = m.clone();
        let mut visitor = referrer.clone().into_visitor(None, module, |_, ix| {
            parse_str(&format!("P{}", ix)).unwrap()
        });
        if root_node {
            visitor.visit_item_mod_mut(&mut item);
            quote!(#item).to_string()
        } else {
            // The same module as an item of a file, which is not the root node: it is nested in the
            // module, which is the one of the file.
            let mut file = File {
                shebang: None,
                attrs: Vec::new(),
                items: vec![Item::Mod(item)],
            };
            visitor.visit_file_mut(&mut file);
            let item = &file.items[0];
            quote!(#item).to_string()
        }
    };
    // The root node is the module analyzed, and the one inside it is found by its name.
    assert_eq!(visit(Some(root), true), rewritten);
    assert_eq!(visit(None, true), rewritten);
    // With the id of the module nested, the root node is it (its content is in `inner`: `Rel` is
    // the one of the entry of `inner`), and the `inner` in it is a module in it, which is none.
    assert_eq!(
        visit(Some(inner), true),
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
    // As an item of a file in the module analyzed, `m` is a module in it, which has none by that
    // name: no entry of a module is found.
    assert_eq!(visit(Some(root), false), quote!(#m).to_string());
    // A file with no module known is no root node either, so the module is not guessed (the root
    // node is no longer taken to be a root): the entries of any module match, as for any syntax
    // which is not a module, and the one of the module which is nested in none is preferred.
    assert_eq!(
        visit(None, false),
        quote!(
            mod m {
                pub struct A {
                    y: P0,
                }
                pub mod inner {
                    pub struct C {
                        y: P0,
                    }
                }
            }
        )
        .to_string()
    );
}

//! Tests for `Leaker::from_module()`.

use proc_macro2::TokenStream;
use std::collections::BTreeSet;
use syn::visit_mut::VisitMut;
use syn::*;
use template_quote::quote;
use type_leak::Leaker;

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
fn from_module_leaks_with_local_generics() {
    let referrer = Leaker::from_module(&module()).unwrap().0.finish();
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
fn from_module_referrer_rewrites_a_copy() {
    let referrer = Leaker::from_module(&module()).unwrap().0.finish();
    let mut item = module();
    referrer
        .clone()
        .into_visitor(None, |ty, ix| {
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
fn from_module_errors_and_bodies() {
    // A relative bound on a nested item is an error.
    let m: ItemMod = parse_quote! {
        mod m { pub struct S<T: Clone>(T); }
    };
    assert!(Leaker::from_module(&m).is_err());

    // Function bodies are ignored.
    let m: ItemMod = parse_quote! {
        mod m {
            pub fn f<T>() -> Rel<T> { let x: RelBody = RelBody::new(); x.into() }
        }
    };
    let referrer = Leaker::from_module(&m).unwrap().0.finish();
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
    assert!(Leaker::from_module(&m).unwrap().0.finish().is_empty());
}

/// Analyzes the module like `Leaker::from_module()`, allowing the primitive types and the
/// standard prelude.
fn referrer_with_prelude(m: &ItemMod) -> type_leak::Referrer {
    let mut config = type_leak::LeakerConfig::new();
    config.allow_primitive().allow_std_prelude();
    let mut leaker = Leaker::from_config(config);
    leaker
        .intern_with(&Generics::default(), |v| {
            syn::visit::Visit::visit_item_mod(v, m)
        })
        .unwrap();
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
fn from_module_items_are_portable() {
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
fn from_module_generics_shadow_module_items() {
    let m: ItemMod = parse_quote! {
        mod m {
            pub struct S;
            pub struct A<S>(S, Wrap<S>);
            pub struct B<T = ()>(S);
        }
    };
    // `S` in `A` is the generic parameter. Local generics are interned with the types depending
    // on them (see `Referrer::generics()`), but `Wrap<S>` is relative either way.
    let referrer = Leaker::from_module(&m).unwrap().0.finish();
    assert_eq!(
        referrer
            .iter_with_generics()
            .map(|(ty, generics)| (quote!(#ty).to_string(), quote!(#generics).to_string()))
            .collect::<Vec<_>>(),
        [("Wrap < S >".to_string(), "< S >".to_string())]
    );
}

#[test]
fn from_module_imports() {
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
fn from_module_referrer_keeps_module_items() {
    let referrer = referrer_with_prelude(&portable_module());
    let mut item = portable_module();
    referrer
        .into_visitor(None, |ty, _| {
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
                Some([]) => "self".to_string(),
                Some(path) => {
                    let path: Vec<String> = path.iter().map(|i| i.to_string()).collect();
                    path.join("::")
                }
            };
            format!("{}@{}", quote!(#ty), module)
        })
        .collect()
}

#[test]
fn same_tokens_in_different_modules_are_different_entries() {
    let (leaker, top_level) = Leaker::from_module(&nested_module()).unwrap();
    let referrer = leaker.finish();
    assert_eq!(modules_of(&referrer), ["Rel@self", "Rel@inner"]);
    // The module has no generics, and its top-level context is in no module.
    assert_eq!(referrer.contexts().count(), 3);
    assert_eq!(referrer.context_module(referrer.context(0)), Some(&[][..]));
    assert_eq!(
        referrer.context_module(referrer.context(1)),
        Some(&[ident("inner")][..])
    );
    assert_eq!(top_level.index(), 2);
    assert_eq!(referrer.context_module(top_level), None);
    assert!(referrer.context_generics(top_level).params.is_empty());
}

#[test]
fn deep_module_path() {
    let m: ItemMod = parse_quote! {
        mod m { pub mod a { pub mod b { pub struct S<T> { x: Wrap<T> } } } }
    };
    let referrer = Leaker::from_module(&m).unwrap().0.finish();
    assert_eq!(modules_of(&referrer), ["Wrap < T >@a::b"]);
    assert_eq!(
        referrer.context_module(referrer.context(0)),
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
    let referrer = Leaker::from_module(&m).unwrap().0.finish();
    assert_eq!(modules_of(&referrer), [":: foo :: X@"]);
    assert_eq!(referrer.contexts().count(), 1);
    // A relative type next to it is still keyed by its module.
    let m: ItemMod = parse_quote! {
        mod m {
            pub struct A { x: ::foo::X }
            pub mod inner { pub struct C { x: ::foo::X, y: &'static Rel } }
        }
    };
    assert_eq!(
        modules_of(&Leaker::from_module(&m).unwrap().0.finish()),
        ["& 'static Rel@inner", ":: foo :: X@"]
    );
}

#[test]
fn referrer_rewrites_each_module_to_its_own_entry() {
    let referrer = Leaker::from_module(&nested_module()).unwrap().0.finish();
    let mut item = nested_module();
    referrer
        .into_visitor(None, |_, ix| parse_str(&format!("P{}", ix)).unwrap())
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
    let referrer = Leaker::from_module(&nested_module()).unwrap().0.finish();
    let copy = |module: Option<Vec<Ident>>| {
        let mut visitor = referrer
            .clone()
            .into_visitor(None, |_, ix| parse_str(&format!("P{}", ix)).unwrap());
        if let Some(module) = module {
            visitor = visitor.in_module(module);
        }
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
    assert_eq!(copy(Some(vec![ident("inner")])), expected(1));
    assert_eq!(copy(Some(vec![])), expected(0));
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
    let referrer = Leaker::from_module(&m).unwrap().0.finish();
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
        visit(referrer.clone().into_visitor(None, rewrite)),
        rewritten
    );
    assert_eq!(
        visit(
            referrer
                .clone()
                .into_visitor(None, rewrite)
                .in_module(vec![ident("inner")])
        ),
        rewritten
    );
    assert_eq!(
        visit(
            referrer
                .clone()
                .into_visitor(None, rewrite)
                .in_module(vec![])
        ),
        unchanged
    );
    // The module is always compared with a root.
    let root = Generics::default();
    assert_eq!(
        visit(
            referrer
                .clone()
                .into_visitor_with_root(&root, None, rewrite)
        ),
        unchanged
    );
    assert_eq!(
        visit(
            referrer
                .into_visitor_with_root(&root, None, rewrite)
                .in_module(vec![ident("inner")])
        ),
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
    let referrer = Leaker::from_module(&m).unwrap().0.finish();
    let mut item = m.clone();
    referrer
        .clone()
        .into_visitor_with_root(&Generics::default(), None, |_, _| parse_quote!(P))
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
        .into_visitor_with_root(&Generics::default(), None, |_, _| parse_quote!(P))
        .visit_item_mod_mut(&mut item);
    let unchanged: ItemMod = parse_quote!(
        mod inner {
            pub struct C {
                y: Rel,
            }
        }
    );
    assert_eq!(quote!(#item).to_string(), quote!(#unchanged).to_string());
    // Unless it is a copy of the module `m::inner`.
    let mut item: ItemMod = parse_quote!(
        mod inner {
            pub struct C {
                y: Rel,
            }
        }
    );
    referrer
        .into_visitor_with_root(&Generics::default(), None, |_, _| parse_quote!(P))
        .in_module(vec![ident("inner")])
        .visit_item_mod_mut(&mut item);
    // `inner` is entered below `m::inner`, which is `m::inner::inner`.
    assert_eq!(quote!(#item).to_string(), quote!(#unchanged).to_string());
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
    let referrer = Leaker::from_module(&m).unwrap().0.finish();
    assert_eq!(
        modules_of(&referrer),
        ["Rel@self", "Rel@a::b", "Wrap < T >@a::b"]
    );
    let tokens = referrer.to_token_stream();
    let version = tag();
    assert_eq!(
        tokens.to_string(),
        format!(
            "\"{}\" 4 mod self ; (Rel) mod a :: b ; (Rel) < T > ; mod a :: b ; (Wrap < T >) ()",
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
    let tokens = Leaker::from_struct(&simple)
        .unwrap()
        .0
        .finish()
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
    let referrer = Leaker::from_module(&m).unwrap().0.finish();
    let copy = |module: Option<Vec<Ident>>| {
        let mut visitor = referrer.clone().into_visitor(None, |_, _| parse_quote!(P));
        if let Some(module) = module {
            visitor = visitor.in_module(module);
        }
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
    assert_eq!(copy(Some(vec![])), rewritten);
    assert_eq!(copy(None), rewritten);
    assert_eq!(copy(Some(vec![ident("inner")])), unchanged);
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
        .into_visitor_with_root(&Generics::default(), None, |_, _| parse_quote!(P))
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

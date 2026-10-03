//! Tests for `Leaker::from_module()`.

use std::collections::BTreeSet;
use syn::visit_mut::VisitMut;
use syn::*;
use template_quote::quote;
use type_leak::Leaker;

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
    let referrer = Leaker::from_module(&module()).unwrap().finish();
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
    let referrer = Leaker::from_module(&module()).unwrap().finish();
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
    let referrer = Leaker::from_module(&m).unwrap().finish();
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
    assert!(Leaker::from_module(&m).unwrap().finish().is_empty());
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
    // on them (see `Referrer::local_generics()`), but `Wrap<S>` is relative either way.
    let referrer = Leaker::from_module(&m).unwrap().finish();
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

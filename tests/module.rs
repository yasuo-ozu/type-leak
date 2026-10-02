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
            .iter_with_local_generics()
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
        .into_visitor(|ty, ix| {
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

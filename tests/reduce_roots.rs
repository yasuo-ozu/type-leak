//! Behavioral tests for `Leaker::reduce_roots()`.

mod common;

use common::finish_as_strings;
use std::collections::BTreeSet;
use syn::*;
use template_quote::quote;
use type_leak::Leaker;

#[test]
fn test_reduce_roots_empty_leaker() {
    // A fresh Leaker with no intern_with calls produces an empty Referrer.
    let mut leaker = Leaker::new();
    leaker.reduce_roots();
    assert!(leaker.finish().is_empty());
}

#[test]
fn test_reduce_roots_neutral_generics() {
    // All fields are the struct's own generic params → all neutral → empty.
    let item: ItemStruct = parse_quote! { struct S<T, U> { a: T, b: U } };
    let (mut leaker, _) = Leaker::from_struct(&item).unwrap();
    leaker.reduce_roots();
    assert!(leaker.finish().is_empty());
}

#[test]
fn test_reduce_roots_absolute_paths_only() {
    // Absolute-path field types are neutral → nothing interned.
    let item: ItemStruct = parse_quote! {
        struct S { a: ::std::string::String }
    };
    let (mut leaker, _) = Leaker::from_struct(&item).unwrap();
    leaker.reduce_roots();
    assert!(leaker.finish().is_empty());
}

#[test]
fn test_reduce_roots_single_must_intern() {
    // A single relative-path field is a must-intern root and must be kept.
    let item: ItemStruct = parse_quote! { struct S { a: RelType } };
    let (mut leaker, _) = Leaker::from_struct(&item).unwrap();
    leaker.reduce_roots();
    let expected: Type = parse_quote!(RelType);
    assert_eq!(
        finish_as_strings(leaker),
        BTreeSet::from([quote!(#expected).to_string()])
    );
}

#[test]
fn test_reduce_roots_multiple_must_intern() {
    // Two independent must-intern roots: both survive reduction.
    let item: ItemStruct = parse_quote! { struct S { a: TypeA, b: TypeB } };
    let (mut leaker, _) = Leaker::from_struct(&item).unwrap();
    leaker.reduce_roots();
    let ta: Type = parse_quote!(TypeA);
    let tb: Type = parse_quote!(TypeB);
    assert_eq!(
        finish_as_strings(leaker),
        BTreeSet::from([quote!(#ta).to_string(), quote!(#tb).to_string()])
    );
}

#[test]
fn test_reduce_roots_neutral_root_without_must_intern_dropped() {
    // A neutral root whose children are all neutral is dropped entirely.
    let item: ItemStruct = parse_quote! {
        struct S<T> { a: (T, ::std::string::String) }
    };
    let (mut leaker, _) = Leaker::from_struct(&item).unwrap();
    leaker.reduce_roots();
    assert!(leaker.finish().is_empty());
}

#[test]
fn test_reduce_roots_root_covers_descendant() {
    // A must-intern type nested inside a compound wrapper: the inner type is interned (it
    // is the innermost minimum cut), and the wrapper is rebuilt around it.
    let item: ItemStruct = parse_quote! {
        struct S { a: (RelType, ::std::string::String) }
    };
    let (mut leaker, _) = Leaker::from_struct(&item).unwrap();
    leaker.reduce_roots();
    let expected: Type = parse_quote!(RelType);
    let result = finish_as_strings(leaker);
    assert_eq!(result, BTreeSet::from([quote!(#expected).to_string()]));
}

#[test]
fn test_reduce_roots_mixed_neutral_and_must_intern() {
    // Generic field (neutral root) alongside a relative-path field (must-intern root):
    // only the must-intern root survives.
    let item: ItemStruct = parse_quote! { struct S<T> { a: T, b: RelType } };
    let (mut leaker, _) = Leaker::from_struct(&item).unwrap();
    leaker.reduce_roots();
    let expected: Type = parse_quote!(RelType);
    assert_eq!(
        finish_as_strings(leaker),
        BTreeSet::from([quote!(#expected).to_string()])
    );
}

#[test]
fn test_reduce_roots_deep_nesting() {
    // A relative type two levels deep: only the inner type is kept; neither the
    // intermediate Vec wrapper nor the outer tuple appears as a separate entry.
    let item: ItemStruct = parse_quote! {
        struct S { a: (::std::vec::Vec<RelInner>, ::std::string::String) }
    };
    let (mut leaker, _) = Leaker::from_struct(&item).unwrap();
    leaker.reduce_roots();
    let expected: Type = parse_quote!(RelInner);
    let result = finish_as_strings(leaker);
    assert_eq!(result, BTreeSet::from([quote!(#expected).to_string()]));
}

#[test]
fn test_reduce_roots_type_in_both_positions() {
    // A type appears as a direct must-intern root (field a) AND as a must-intern
    // descendant inside a compound root (field b). Interning it covers both.
    let item: ItemStruct = parse_quote! {
        struct S {
            a: RelType,
            b: (RelType, ::std::string::String),
        }
    };
    let (mut leaker, _) = Leaker::from_struct(&item).unwrap();
    leaker.reduce_roots();
    let ta: Type = parse_quote!(RelType);
    assert_eq!(
        finish_as_strings(leaker),
        BTreeSet::from([quote!(#ta).to_string()])
    );
}

fn leaked_of(item: ItemStruct) -> Vec<String> {
    let (mut leaker, _) = Leaker::from_struct(&item).unwrap();
    leaker.reduce_roots();
    finish_as_strings(leaker).into_iter().collect()
}

fn ty_string(ty: Type) -> String {
    quote!(#ty).to_string()
}

#[test]
fn test_reduce_roots_wrappers_of_one_type() {
    // Four wrappers of the same type: interning it takes one implementation, not four.
    let leaked = leaked_of(parse_quote! {
        struct S {
            a: ::std::vec::Vec<Rel>,
            b: ::core::option::Option<Rel>,
            c: (Rel, ::std::string::String),
            d: ::std::boxed::Box<Rel>,
        }
    });
    assert_eq!(leaked, [ty_string(parse_quote!(Rel))]);
    // With the bare type as well.
    let leaked = leaked_of(parse_quote! {
        struct S {
            a: ::std::vec::Vec<Rel>,
            b: ::core::option::Option<Rel>,
            c: (Rel, ::std::string::String),
            d: ::std::boxed::Box<Rel>,
            e: Rel,
        }
    });
    assert_eq!(leaked, [ty_string(parse_quote!(Rel))]);
}

#[test]
fn test_reduce_roots_common_ancestor() {
    // The same tuple of three types twice: the tuple (1) is cheaper than its parts (3).
    let leaked = leaked_of(parse_quote! {
        struct S {
            a: (RelA, RelB, RelC),
            b: ::std::vec::Vec<(RelA, RelB, RelC)>,
        }
    });
    assert_eq!(leaked, [ty_string(parse_quote!((RelA, RelB, RelC)))]);
    // Two tuples sharing a part: the two tuples (2) are cheaper than the three parts (3).
    let leaked = leaked_of(parse_quote! {
        struct S {
            a: (RelA, RelB),
            b: (RelA, RelC),
        }
    });
    assert_eq!(
        leaked,
        [
            ty_string(parse_quote!((RelA, RelB))),
            ty_string(parse_quote!((RelA, RelC))),
        ]
    );
}

#[test]
fn test_reduce_roots_prefers_inner_type_on_a_tie() {
    // `Vec<Rel>` and `Rel` cover the only path equally; the inner one is taken.
    let leaked = leaked_of(parse_quote! { struct S { a: ::std::vec::Vec<Rel> } });
    assert_eq!(leaked, [ty_string(parse_quote!(Rel))]);
    // A red root is interned itself.
    let leaked = leaked_of(parse_quote! { struct S { a: Rel } });
    assert_eq!(leaked, [ty_string(parse_quote!(Rel))]);
}

#[test]
fn test_reduce_roots_allowed_path_is_neutral() {
    // A type under an explicitly allowed path prefix is neutral → empty.
    let item: ItemStruct = parse_quote! { struct S { a: some_mod::SomeType } };
    let mut leaker = Leaker::new();
    leaker.allowed_paths.push(parse_quote!(some_mod));
    leaker
        .intern_with(&item.generics, |v| v.visit_item_struct(&item))
        .unwrap();
    leaker.reduce_roots();
    assert!(leaker.finish().is_empty());
}

#[test]
fn test_reduce_roots_self_assoc_type_neutral() {
    // Self::AssocType in a trait method return position is neutral (concrete to each
    // implementor).
    let item: ItemTrait = parse_quote! {
        trait MyTrait {
            type Assoc;
            fn method() -> Self::Assoc;
        }
    };
    let (mut leaker, _) = Leaker::from_trait(&item).unwrap();
    leaker.reduce_roots();
    assert!(leaker.finish().is_empty());
}

#[test]
fn test_reduce_roots_generic_assoc_type_neutral() {
    // S::Item where S is a method-level generic is neutral: the caller provides S
    // directly so its associated types are also concrete.
    let item: ItemTrait = parse_quote! {
        trait MyTrait {
            fn method<S: ::core::iter::Iterator>() -> S::Item;
        }
    };
    let (mut leaker, _) = Leaker::from_trait(&item).unwrap();
    leaker.reduce_roots();
    assert!(leaker.finish().is_empty());
}

#[test]
fn test_reduce_roots_inner_entry_is_rewritten_in_the_copy() {
    // The wrappers are not leaked, but a copy of them is rewritten around the inner entry.
    let item: ItemStruct = parse_quote! {
        struct S {
            a: ::std::vec::Vec<Rel>,
            b: (Rel, ::std::string::String),
            c: &'static Rel,
        }
    };
    let referrer = Leaker::from_struct(&item).unwrap().0.finish();
    // `&'static Rel` is not rebuilt around `Rel` (`Rel: 'static` is not known of the entry).
    assert_eq!(referrer.iter().count(), 2);
    let mut copy: ItemStruct = parse_quote! {
        struct S {
            a: ::std::vec::Vec<Rel>,
            b: (Rel, ::std::string::String),
        }
    };
    let rel_ix = referrer
        .iter()
        .position(|ty| quote!(#ty).to_string() == "Rel")
        .unwrap();
    syn::visit_mut::VisitMut::visit_item_struct_mut(
        &mut referrer.into_visitor(None, |_, ix| parse_quote!(P<#ix>)),
        &mut copy,
    );
    let expected: ItemStruct = syn::parse2(quote! {
        struct S {
            a: ::std::vec::Vec<P<#rel_ix>>,
            b: (P<#rel_ix>, ::std::string::String),
        }
    })
    .unwrap();
    assert_eq!(quote!(#copy).to_string(), quote!(#expected).to_string());
}

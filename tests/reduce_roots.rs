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
    let mut leaker = Leaker::from_struct(&item).unwrap();
    leaker.reduce_roots();
    assert!(leaker.finish().is_empty());
}

#[test]
fn test_reduce_roots_absolute_paths_only() {
    // Absolute-path field types are neutral → nothing interned.
    let item: ItemStruct = parse_quote! {
        struct S { a: ::std::string::String }
    };
    let mut leaker = Leaker::from_struct(&item).unwrap();
    leaker.reduce_roots();
    assert!(leaker.finish().is_empty());
}

#[test]
fn test_reduce_roots_single_must_intern() {
    // A single relative-path field is a must-intern root and must be kept.
    let item: ItemStruct = parse_quote! { struct S { a: RelType } };
    let mut leaker = Leaker::from_struct(&item).unwrap();
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
    let mut leaker = Leaker::from_struct(&item).unwrap();
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
    let mut leaker = Leaker::from_struct(&item).unwrap();
    leaker.reduce_roots();
    assert!(leaker.finish().is_empty());
}

#[test]
fn test_reduce_roots_root_covers_descendant() {
    // A must-intern type nested inside a compound wrapper: the wrapper root is kept
    // but the inner type must NOT appear separately.
    let item: ItemStruct = parse_quote! {
        struct S { a: (RelType, ::std::string::String) }
    };
    let mut leaker = Leaker::from_struct(&item).unwrap();
    leaker.reduce_roots();
    let expected: Type = parse_quote!((RelType, ::std::string::String));
    let result = finish_as_strings(leaker);
    assert_eq!(result, BTreeSet::from([quote!(#expected).to_string()]));
    // The inner type must not appear as its own entry.
    let inner: Type = parse_quote!(RelType);
    assert!(!result.contains(&quote!(#inner).to_string()));
}

#[test]
fn test_reduce_roots_mixed_neutral_and_must_intern() {
    // Generic field (neutral root) alongside a relative-path field (must-intern root):
    // only the must-intern root survives.
    let item: ItemStruct = parse_quote! { struct S<T> { a: T, b: RelType } };
    let mut leaker = Leaker::from_struct(&item).unwrap();
    leaker.reduce_roots();
    let expected: Type = parse_quote!(RelType);
    assert_eq!(
        finish_as_strings(leaker),
        BTreeSet::from([quote!(#expected).to_string()])
    );
}

#[test]
fn test_reduce_roots_deep_nesting() {
    // A relative type two levels deep: only the outermost root is kept; neither the
    // intermediate Vec wrapper nor the inner type appears as a separate entry.
    let item: ItemStruct = parse_quote! {
        struct S { a: (::std::vec::Vec<RelInner>, ::std::string::String) }
    };
    let mut leaker = Leaker::from_struct(&item).unwrap();
    leaker.reduce_roots();
    let expected: Type = parse_quote!((::std::vec::Vec<RelInner>, ::std::string::String));
    let result = finish_as_strings(leaker);
    assert_eq!(result, BTreeSet::from([quote!(#expected).to_string()]));
    let inner: Type = parse_quote!(RelInner);
    let vec_ty: Type = parse_quote!(::std::vec::Vec<RelInner>);
    assert!(!result.contains(&quote!(#inner).to_string()));
    assert!(!result.contains(&quote!(#vec_ty).to_string()));
}

#[test]
fn test_reduce_roots_type_in_both_positions() {
    // A type appears as a direct must-intern root (field a) AND as a must-intern
    // descendant inside a compound root (field b). Both must be kept.
    let item: ItemStruct = parse_quote! {
        struct S {
            a: RelType,
            b: (RelType, ::std::string::String),
        }
    };
    let mut leaker = Leaker::from_struct(&item).unwrap();
    leaker.reduce_roots();
    let ta: Type = parse_quote!(RelType);
    let tb: Type = parse_quote!((RelType, ::std::string::String));
    assert_eq!(
        finish_as_strings(leaker),
        BTreeSet::from([quote!(#ta).to_string(), quote!(#tb).to_string()])
    );
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
    let mut leaker = Leaker::from_trait(&item).unwrap();
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
    let mut leaker = Leaker::from_trait(&item).unwrap();
    leaker.reduce_roots();
    assert!(leaker.finish().is_empty());
}

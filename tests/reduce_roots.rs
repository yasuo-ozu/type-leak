//! Behavioral tests for `Leaker::reduce_roots()`.

mod common;

use common::finish_as_strings;
use std::collections::BTreeSet;
use syn::*;
use template_quote::quote;
use type_leak::Leaker;

#[test]
fn test_reduce_roots_empty_leaker() {
    // A fresh Leaker with nothing interned produces an empty Referrer.
    let mut leaker = Leaker::new();
    leaker.reduce_roots();
    assert!(leaker.finish().is_empty());
}

#[test]
fn test_reduce_roots_neutral_generics() {
    // All fields are the struct's own generic params → all neutral → empty.
    let item: ItemStruct = parse_quote! { struct S<T, U> { a: T, b: U } };
    let mut leaker = Leaker::new();
    leaker.intern_struct(&item, true).unwrap();
    leaker.reduce_roots();
    assert!(leaker.finish().is_empty());
}

#[test]
fn test_reduce_roots_absolute_paths_only() {
    // Absolute-path field types are neutral → nothing is required (old: nothing interned at all;
    // now the type of an external crate is an optional entry, see `Referrer::must()`).
    let item: ItemStruct = parse_quote! {
        struct S { a: ::std::string::String }
    };
    let mut leaker = Leaker::new();
    leaker.intern_struct(&item, true).unwrap();
    leaker.reduce_roots();
    let referrer = leaker.finish();
    assert_eq!(referrer.iter().count(), 1);
    assert!(!referrer.must(0));
    // `::core` is never leaked.
    let item: ItemStruct = parse_quote! {
        struct S { a: ::core::option::Option<(::core::primitive::u8, ::core::primitive::bool)> }
    };
    let mut leaker = Leaker::new();
    leaker.intern_struct(&item, true).unwrap();
    leaker.reduce_roots();
    assert!(leaker.finish().is_empty());
}

#[test]
fn test_reduce_roots_single_must_intern() {
    // A single relative-path field is a must-intern root and must be kept.
    let item: ItemStruct = parse_quote! { struct S { a: RelType } };
    let mut leaker = Leaker::new();
    leaker.intern_struct(&item, true).unwrap();
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
    let mut leaker = Leaker::new();
    leaker.intern_struct(&item, true).unwrap();
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
    // A neutral root whose children are all neutral is dropped entirely (but for the type of an
    // external crate in it, which is an optional entry: the inner one, as it is the cut closest to
    // the sink, see below).
    let item: ItemStruct = parse_quote! {
        struct S<T> { a: (T, ::std::string::String), b: (T, ::core::primitive::u8) }
    };
    let mut leaker = Leaker::new();
    leaker.intern_struct(&item, true).unwrap();
    leaker.reduce_roots();
    let referrer = leaker.finish();
    assert_eq!(
        referrer
            .iter()
            .map(|t| ty_string(t.clone()))
            .collect::<Vec<_>>(),
        [":: std :: string :: String"]
    );
    assert!(!referrer.must(0));
    let item: ItemStruct = parse_quote! {
        struct S<T> { a: (T, ::core::string::String) }
    };
    let mut leaker = Leaker::new();
    leaker.intern_struct(&item, true).unwrap();
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
    let mut leaker = Leaker::new();
    leaker.intern_struct(&item, true).unwrap();
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
    let mut leaker = Leaker::new();
    leaker.intern_struct(&item, true).unwrap();
    leaker.reduce_roots();
    let expected: Type = parse_quote!(RelType);
    assert_eq!(
        finish_as_strings(leaker),
        BTreeSet::from([quote!(#expected).to_string()])
    );
}

#[test]
fn test_reduce_roots_deep_nesting() {
    // A relative type three levels deep: only the inner type is kept; neither the intermediate
    // Vec wrapper nor the tuples appear as a separate entry. (A bare `Vec<RelInner>` would be
    // interned as a whole: see `test_reduce_roots_bare_identifier_argument`.)
    let item: ItemStruct = parse_quote! {
        struct S { a: (::std::vec::Vec<(RelInner,)>, ::std::string::String) }
    };
    let mut leaker = Leaker::new();
    leaker.intern_struct(&item, true).unwrap();
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
    let mut leaker = Leaker::new();
    leaker.intern_struct(&item, true).unwrap();
    leaker.reduce_roots();
    let ta: Type = parse_quote!(RelType);
    assert_eq!(
        finish_as_strings(leaker),
        BTreeSet::from([quote!(#ta).to_string()])
    );
}

fn leaked_of(item: ItemStruct) -> Vec<String> {
    let mut leaker = Leaker::new();
    leaker.intern_struct(&item, true).unwrap();
    leaker.reduce_roots();
    finish_as_strings(leaker).into_iter().collect()
}

fn ty_string(ty: Type) -> String {
    quote!(#ty).to_string()
}

#[test]
fn test_reduce_roots_wrappers_of_one_type() {
    // Four wrappers of the same type: interning it takes one implementation, not four. (The type
    // is not a bare identifier as an argument, which may be a const, see
    // `test_reduce_roots_bare_identifier_argument`.)
    let leaked = leaked_of(parse_quote! {
        struct S {
            a: ::std::vec::Vec<(Rel,)>,
            b: ::core::option::Option<(Rel,)>,
            c: (Rel, ::std::string::String),
            d: ::std::boxed::Box<(Rel,)>,
        }
    });
    assert_eq!(leaked, [ty_string(parse_quote!(Rel))]);
    // With the bare type as well.
    let leaked = leaked_of(parse_quote! {
        struct S {
            a: ::std::vec::Vec<(Rel,)>,
            b: ::core::option::Option<(Rel,)>,
            c: (Rel, ::std::string::String),
            d: ::std::boxed::Box<(Rel,)>,
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
    // `Vec<(Rel,)>`, `(Rel,)` and `Rel` cover the only path equally; the inner one is taken.
    let leaked = leaked_of(parse_quote! { struct S { a: ::std::vec::Vec<(Rel,)> } });
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
    leaker.intern_struct(&item, true).unwrap();
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
    let mut leaker = Leaker::new();
    leaker.intern_trait(&item, true).unwrap();
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
    let mut leaker = Leaker::new();
    leaker.intern_trait(&item, true).unwrap();
    leaker.reduce_roots();
    assert!(leaker.finish().is_empty());
}

#[test]
fn test_reduce_roots_inner_entry_is_rewritten_in_the_copy() {
    // The wrappers are not leaked, but a copy of them is rewritten around the inner entry.
    let item: ItemStruct = parse_quote! {
        struct S {
            a: ::std::vec::Vec<(Rel,)>,
            b: (Rel, ::std::string::String),
            c: &'static Rel,
        }
    };
    let mut leaker = Leaker::new();
    let context = leaker.intern_struct(&item, true).unwrap();
    let referrer = leaker.finish();
    // `&'static Rel` is not rebuilt around `Rel` (`Rel: 'static` is not known of the entry). The
    // two types of external crates are optional entries (old: none), after the required ones: the
    // `Vec` (one cut covers it, whatever else is in it) and the `String`.
    assert_eq!(
        (0..referrer.iter().count())
            .map(|ix| referrer.must(ix))
            .collect::<Vec<_>>(),
        [true, true, false, false]
    );
    let strings: Vec<String> = referrer.iter().map(|t| quote!(#t).to_string()).collect();
    assert_eq!(
        strings,
        [
            "& 'static Rel",
            "Rel",
            ":: std :: string :: String",
            ":: std :: vec :: Vec < (Rel ,) >"
        ]
    );
    let mut copy: ItemStruct = parse_quote! {
        struct S {
            a: ::std::vec::Vec<(Rel,)>,
            b: (Rel, ::std::string::String),
        }
    };
    let rel_ix = strings.iter().position(|ty| ty == "Rel").unwrap();
    let vec_ix = strings.iter().position(|ty| ty.contains("Vec")).unwrap();
    let string_ix = strings.iter().position(|ty| ty.contains("String")).unwrap();
    let mut replaced = copy.clone();
    syn::visit_mut::VisitMut::visit_item_struct_mut(
        &mut referrer
            .clone()
            .visitor(context, |_, ix| parse_quote!(P<#ix>)),
        &mut copy,
    );
    let expected: ItemStruct = syn::parse2(quote! {
        struct S {
            a: ::std::vec::Vec<(P<#rel_ix>,)>,
            b: (P<#rel_ix>, ::std::string::String),
        }
    })
    .unwrap();
    assert_eq!(quote!(#copy).to_string(), quote!(#expected).to_string());
    // The Referrer which does not trust the names of the crates replaces them as well: the `Vec`
    // as a whole, and the `String`, with `Rel` inside the tuple.
    syn::visit_mut::VisitMut::visit_item_struct_mut(
        &mut referrer
            .allow_ambiguous_crate_name(false)
            .visitor(context, |_, ix| parse_quote!(P<#ix>)),
        &mut replaced,
    );
    let expected: ItemStruct = syn::parse2(quote! {
        struct S {
            a: P<#vec_ix>,
            b: (P<#rel_ix>, P<#string_ix>),
        }
    })
    .unwrap();
    assert_eq!(quote!(#replaced).to_string(), quote!(#expected).to_string());
}

/// The required leaked types of a trait, with the generics each uses.
fn leaked_trait(item: &ItemTrait) -> Vec<(String, String)> {
    let mut leaker = Leaker::new();
    leaker.intern_trait(item, true).unwrap();
    let referrer = leaker.finish();
    referrer
        .iter_with_generics()
        .enumerate()
        .filter(|(ix, _)| referrer.must(*ix))
        .map(|(_, (ty, generics))| (ty_string(ty.clone()), quote!(#generics).to_string()))
        .collect()
}

/// The optional leaked types of a trait (those of external crates, see `Referrer::must()`), with
/// the generics each uses.
fn optional_trait(item: &ItemTrait) -> Vec<(String, String)> {
    let mut leaker = Leaker::new();
    leaker.intern_trait(item, true).unwrap();
    let referrer = leaker.finish();
    referrer
        .iter_with_generics()
        .enumerate()
        .filter(|(ix, _)| !referrer.must(*ix))
        .map(|(_, (ty, generics))| (ty_string(ty.clone()), quote!(#generics).to_string()))
        .collect()
}

/// The error of the analysis of a trait.
fn error_of(item: &ItemTrait) -> String {
    let err = Leaker::new()
        .intern_trait(item, true)
        .expect_err("expected an error");
    syn::Error::from(err).to_string()
}

/// Rewrites a trait with each leaked type replaced by `P<index>`, like the Referrer's copy.
fn rewritten(item: &ItemTrait) -> String {
    let mut leaker = Leaker::new();
    let context = leaker.intern_trait(item, true).unwrap();
    let referrer = leaker.finish();
    let mut copy = item.clone();
    syn::visit_mut::VisitMut::visit_item_trait_mut(
        &mut referrer.visitor(context, |_, ix| parse_quote!(P<#ix>)),
        &mut copy,
    );
    quote!(#copy).to_string()
}

/// `item` as the Referrer's copy prints it.
fn printed(item: ItemTrait) -> String {
    quote!(#item).to_string()
}

fn entry(ty: Type, generics: &str) -> (String, String) {
    (ty_string(ty), generics.to_string())
}

fn trait_with(ty: &Type) -> ItemTrait {
    parse_quote! { trait Tr<'a> { fn f(&self, x: #ty); } }
}

#[test]
fn test_reduce_roots_trait_object_inheriting_the_lifetime_of_its_path() {
    // `Ref<'a, T: 'a>` makes the lifetime bound of `dyn RelTr` `'a`, which `dyn RelTr` alone
    // (`dyn RelTr + 'static`) does not have, so the trait object cannot be interned alone. It has
    // to be (it names a relative trait), so the whole path is interned.
    let group = Type::Group(TypeGroup {
        group_token: Default::default(),
        elem: Box::new(parse_quote!(dyn RelTr)),
    });
    for (ty, generics) in [
        (parse_quote!(::core::cell::Ref<'a, dyn RelTr>), "< 'a >"),
        (parse_quote!(::core::cell::Ref<'a, (dyn RelTr)>), "< 'a >"),
        (parse_quote!(::core::cell::Ref<'a, #group>), "< 'a >"),
        (parse_quote!(::p::W<'a, dyn RelTr<Rel>>), "< 'a >"),
        (parse_quote!(::p::W<'a, (), dyn RelTr>), "< 'a >"),
        // `Rel` may be a const (a bare identifier as an argument), which cannot be interned
        // alone, so the type containing it has to be interned: the trait object, and so the path.
        (parse_quote!(::p::W<'a, dyn ::p::Tr<Rel>>), "< 'a >"),
        (parse_quote!(::p::W<'a, (dyn ::p::Tr<Rel>)>), "< 'a >"),
        // A lifetime which the type context provides.
        (parse_quote!(::p::W<'x, dyn RelTr>), ""),
    ] {
        let ty: Type = ty;
        let item = trait_with(&ty);
        assert_eq!(leaked_trait(&item), [entry(ty.clone(), generics)]);
        assert_eq!(
            rewritten(&item),
            printed(parse_quote! { trait Tr<'a> { fn f(&self, x: P<0usize>); } })
        );
    }
    // The argument is that of the innermost path, which is the one with a lifetime.
    let item = trait_with(&parse_quote!(::p::V<::p::W<'a, dyn RelTr>>));
    assert_eq!(
        leaked_trait(&item),
        [entry(parse_quote!(::p::W<'a, dyn RelTr>), "< 'a >")]
    );
    assert_eq!(
        rewritten(&item),
        printed(parse_quote! { trait Tr<'a> { fn f(&self, x: ::p::V<P<0usize>>); } })
    );

    // Two paths sharing the trait object: it is not leaked alone, however much is saved.
    let leaked = leaked_trait(&parse_quote! {
        trait Tr<'a> {
            fn f(
                &self,
                x: ::p::W<'a, dyn ::p::Tr<RelA, RelB>>,
                y: ::p::V<'a, dyn ::p::Tr<RelA, RelB>>,
            );
        }
    });
    assert_eq!(
        leaked,
        [
            entry(parse_quote!(::p::V<'a, dyn ::p::Tr<RelA, RelB>>), "< 'a >"),
            entry(parse_quote!(::p::W<'a, dyn ::p::Tr<RelA, RelB>>), "< 'a >"),
        ]
    );

    // Without a cause of its own (and with no bare identifier as an argument, which may be a
    // const), the trait object is not interned alone, but what is inside it is: the copy keeps
    // the trait object where it is, so it keeps the lifetime it defaults to.
    let object = Type::Group(TypeGroup {
        group_token: Default::default(),
        elem: Box::new(parse_quote!(dyn ::p::Tr<Item = Rel>)),
    });
    for ty in [
        parse_quote!(::core::cell::Ref<'a, dyn ::p::Tr<Item = Rel>>),
        parse_quote!(::core::cell::Ref<'a, (dyn ::p::Tr<Item = Rel>)>),
        parse_quote!(::core::cell::Ref<'a, #object>),
        parse_quote!(::p::W<'a, (), dyn ::p::Tr<Item = Rel>>),
        parse_quote!(::p::V<::p::W<'a, dyn ::p::Tr<Item = Rel>>>),
    ] {
        let ty: Type = ty;
        let item = trait_with(&ty);
        assert_eq!(
            leaked_trait(&item),
            [entry(parse_quote!(Rel), "")],
            "{}",
            ty_string(ty.clone())
        );
    }
    let item = trait_with(&parse_quote!(
        ::core::cell::Ref<'a, dyn ::p::Tr<Item = Rel>>
    ));
    assert_eq!(
        rewritten(&item),
        printed(parse_quote! {
            trait Tr<'a> { fn f(&self, x: ::core::cell::Ref<'a, dyn ::p::Tr<Item = P<0usize>>>); }
        })
    );
    // Also if it is a type argument of the trait (not a binding) which is not a bare identifier.
    let item = trait_with(&parse_quote!(
        ::core::cell::Ref<'a, dyn ::p::Tr<Option<Rel>>>
    ));
    assert_eq!(leaked_trait(&item), [entry(parse_quote!(Option<Rel>), "")]);
    // Three paths sharing a trait object: interning it alone would cost one type instead of the
    // two components `RelA` and `RelB` (or the three paths), but it is not a candidate.
    let leaked = leaked_trait(&parse_quote! {
        trait Tr<'a> {
            fn f(
                &self,
                x: ::p::W<'a, dyn ::p::Tr<Item = RelA, Other = RelB>>,
                y: ::p::V<'a, dyn ::p::Tr<Item = RelA, Other = RelB>>,
                z: ::p::U<'a, dyn ::p::Tr<Item = RelA, Other = RelB>>,
            );
        }
    });
    assert_eq!(
        leaked,
        [entry(parse_quote!(RelA), ""), entry(parse_quote!(RelB), "")]
    );
    // The trait object has a cause of its own as well as components: the path is interned.
    let item = trait_with(&parse_quote!(::core::cell::Ref<'a, dyn RelTr<Item = Rel>>));
    assert_eq!(
        leaked_trait(&item),
        [entry(
            parse_quote!(::core::cell::Ref<'a, dyn RelTr<Item = Rel>>),
            "< 'a >"
        )]
    );

    // `check()` agrees: the path must be interned as a whole.
    let result = Leaker::new().check(&parse_quote!(::p::W<'a, dyn RelTr>), &parse_quote!(<'a>));
    assert!(result.is_own);
    assert_eq!(result.must.len(), 1);
    // Unless only the components of the trait object have to be interned.
    let result = Leaker::new().check(
        &parse_quote!(::p::W<'a, dyn ::p::Tr<Item = Rel>>),
        &parse_quote!(<'a>),
    );
    assert!(!result.is_own);
    assert_eq!(result.must.len(), 1);
}

#[test]
fn test_reduce_roots_reference_to_a_trait_object() {
    // The reference keeps the trait object where it is in the copy, so the object lifetime bound
    // it defaults to (`'a` for `&'a dyn Trait`) is unchanged, and only what is inside the trait
    // object is interned, as with a lifetime of the method (which keeps the reference itself from
    // being interned).
    let object = group(parse_quote!(dyn ::p::Tr<Item = Rel>));
    let components: Vec<(Type, Type, &str)> = vec![
        (
            parse_quote!(&'a dyn ::p::Tr<Item = Rel>),
            parse_quote!(Rel),
            "",
        ),
        (
            parse_quote!(&'a (dyn ::p::Tr<Item = Rel>)),
            parse_quote!(Rel),
            "",
        ),
        (
            parse_quote!(&'a ((dyn ::p::Tr<Item = Rel>))),
            parse_quote!(Rel),
            "",
        ),
        (parse_quote!(&'a #object), parse_quote!(Rel), ""),
        (
            parse_quote!(&'a mut dyn ::p::Tr<Item = Rel>),
            parse_quote!(Rel),
            "",
        ),
        // The lifetime bound is written, or the lifetime is `'static`.
        (
            parse_quote!(&'a (dyn ::p::Tr<Item = Rel> + 'static)),
            parse_quote!(Rel),
            "",
        ),
        (
            parse_quote!(&'a (dyn ::p::Tr<Item = Rel> + 'a)),
            parse_quote!(Rel),
            "",
        ),
        (
            parse_quote!(&'static dyn ::p::Tr<Item = Rel>),
            parse_quote!(Rel),
            "",
        ),
        (
            parse_quote!(::p::V<&'a dyn ::p::Tr<Item = Rel>>),
            parse_quote!(Rel),
            "",
        ),
        // Not a bare identifier, which may be a const.
        (
            parse_quote!(&'a dyn ::p::Tr<Item = Option<Rel>>),
            parse_quote!(Option<Rel>),
            "",
        ),
        (
            parse_quote!(&'a dyn ::p::Tr<Option<Rel>>),
            parse_quote!(Option<Rel>),
            "",
        ),
        (
            parse_quote!(&'a dyn ::p::Tr<(Rel, u8)>),
            parse_quote!((Rel, u8)),
            "",
        ),
        // A reference inside is the type to intern if it has to be (`Rel: 'a`).
        (
            parse_quote!(&'a dyn ::p::Tr<Item = &'a Rel>),
            parse_quote!(&'a Rel),
            "< 'a >",
        ),
        (
            parse_quote!(&'a dyn ::p::Tr<Item = &'a dyn ::p::Tr<Item = Rel>>),
            parse_quote!(Rel),
            "",
        ),
    ];
    for (ty, leaked, generics) in components {
        let item = trait_with(&ty);
        assert_eq!(
            leaked_trait(&item),
            [entry(leaked, generics)],
            "{}",
            ty_string(ty.clone())
        );
    }
    // The copy is the original with the leaked type in place, with the reference and the trait
    // object where they were.
    for (ty, copy) in [
        (
            parse_quote!(&'a dyn ::p::Tr<Item = Rel>),
            parse_quote!(&'a dyn ::p::Tr<Item = P<0usize>>),
        ),
        (
            parse_quote!(&'a (dyn ::p::Tr<Item = Rel>)),
            parse_quote!(&'a (dyn ::p::Tr<Item = P<0usize>>)),
        ),
    ] {
        let (ty, copy): (Type, Type) = (ty, copy);
        assert_eq!(
            rewritten(&trait_with(&ty)),
            printed(parse_quote! { trait Tr<'a> { fn f(&self, x: #copy); } })
        );
    }
    // The same as with a lifetime of the method.
    let item: ItemTrait = parse_quote! {
        trait Tr { fn f<'b>(&self, x: &'b dyn ::p::Tr<Item = Rel>); }
    };
    assert_eq!(leaked_trait(&item), [entry(parse_quote!(Rel), "")]);
    // Two references sharing what is inside: one type is leaked.
    let leaked = leaked_trait(&parse_quote! {
        trait Tr<'a> {
            fn f(&self, x: &'a dyn ::p::Tr<Item = Rel>, y: &'a mut dyn ::p::Tr<Item = Rel>);
        }
    });
    assert_eq!(leaked, [entry(parse_quote!(Rel), "")]);
    // The generics are those of the type inside, not of the reference, with the bound which the
    // reference implies on it (`dyn Tr<Item = X> + 'a: 'a` requires `X: 'a`).
    let item: ItemTrait = parse_quote! {
        trait Tr<'a, T> { fn f(&self, x: &'a dyn ::p::Tr<Item = Rel2<T>>); }
    };
    assert_eq!(
        leaked_trait(&item),
        [entry(parse_quote!(Rel2<T>), "< 'a , T : 'a >")]
    );

    // The trait object has to be interned itself (it names a relative trait, or a bare
    // identifier in it may be a const, which cannot be interned alone): it cannot be alone,
    // because its default would be `'static` instead of `'a` (and it is the lifetime bound of the
    // trait if it has one, `trait RelTr: 'static`, which the Leaker cannot see, so the default
    // cannot be written out either), so the whole reference is. The same with a lifetime bound
    // written, as before: the reference is the type that carries the bound `'b: 'a` it implies.
    let rel_object = group(parse_quote!(dyn RelTr));
    let wholes: Vec<Type> = vec![
        parse_quote!(&'a dyn RelTr),
        parse_quote!(&'a (dyn RelTr)),
        parse_quote!(&'a mut dyn RelTr),
        parse_quote!(&'a dyn RelTr<Item = Rel>),
        parse_quote!(&'a #rel_object),
        parse_quote!(&'a dyn ::p::Tr<Rel>),
        parse_quote!(&'a (dyn ::p::Tr<Rel>)),
        parse_quote!(&'a (dyn RelTr + 'static)),
        parse_quote!(&'a (dyn RelTr + 'a)),
        parse_quote!(&'a (dyn ::p::Tr<Rel> + 'static)),
    ];
    for ty in wholes {
        let item = trait_with(&ty);
        assert_eq!(
            leaked_trait(&item),
            [entry(ty.clone(), "< 'a >")],
            "{}",
            ty_string(ty.clone())
        );
        assert_eq!(
            rewritten(&item),
            printed(parse_quote! { trait Tr<'a> { fn f(&self, x: P<0usize>); } })
        );
    }
    // With `'static`, the trait object is the same alone.
    for (ty, leaked) in [
        (parse_quote!(&'static dyn RelTr), parse_quote!(dyn RelTr)),
        (parse_quote!(&'static (dyn RelTr)), parse_quote!(dyn RelTr)),
    ] {
        let (ty, leaked): (Type, Type) = (ty, leaked);
        assert_eq!(
            leaked_trait(&trait_with(&ty)),
            [entry(leaked, "")],
            "{}",
            ty_string(ty.clone())
        );
    }

    // A reference whose referent is not a trait object is the type to intern, as before.
    let wholes: Vec<(Type, &str)> = vec![
        (parse_quote!(&'a Rel), "< 'a >"),
        (parse_quote!(&'a mut Rel), "< 'a >"),
        (parse_quote!(&'a ::p::W<Rel2>), "< 'a >"),
        (parse_quote!(&'a ::p::W<Option<Rel>>), "< 'a >"),
        (
            parse_quote!(&'a ::std::boxed::Box<dyn ::p::Tr<Item = Rel>>),
            "< 'a >",
        ),
        (parse_quote!(&'static Rel), ""),
    ];
    for (ty, generics) in wholes {
        assert_eq!(
            leaked_trait(&trait_with(&ty)),
            [entry(ty.clone(), generics)],
            "{}",
            ty_string(ty.clone())
        );
    }

    // `check()` agrees.
    let result = Leaker::new().check(&parse_quote!(&'a dyn RelTr), &parse_quote!(<'a>));
    assert!(result.is_own);
    assert_eq!(result.must.len(), 1);
    let result = Leaker::new().check(
        &parse_quote!(&'a dyn ::p::Tr<Item = Rel>),
        &parse_quote!(<'a>),
    );
    assert!(!result.is_own);
    assert_eq!(result.must.len(), 1);
}

#[test]
fn test_reduce_roots_trait_object_with_its_own_lifetime() {
    // The default does not apply with the bound written, with `'static` (which every default
    // is), or to the argument of another path.
    for (ty, generics) in [
        (parse_quote!(::p::W<'a, dyn RelTr + 'a>), "< 'a >"),
        (parse_quote!(::p::W<'a, dyn RelTr + 'static>), ""),
        (parse_quote!(::p::W<'static, dyn RelTr>), ""),
        (parse_quote!(::p::W<'a, ::std::boxed::Box<dyn RelTr>>), ""),
        (
            parse_quote!(::p::W<'a, [::std::boxed::Box<dyn RelTr>; 2]>),
            "",
        ),
    ] {
        let ty: Type = ty;
        let leaked = leaked_trait(&trait_with(&ty));
        assert_eq!(leaked.len(), 1, "{}", ty_string(ty.clone()));
        assert!(leaked[0].0.starts_with("dyn RelTr"), "{}", leaked[0].0);
        assert_eq!(leaked[0].1.replace(' ', ""), generics.replace(' ', ""));
    }
}

/// A trait whose method `f` declares the lifetime `'b` and takes `ty`.
fn local_trait(ty: &Type) -> ItemTrait {
    parse_quote! { trait Tr { fn f<'b>(&self, x: #ty); } }
}

#[test]
fn test_reduce_roots_trait_object_inheriting_a_local_lifetime() {
    // A path with a lifetime of a method (the only lifetime argument) is a whole like one with a
    // lifetime of the trait: the trait object cannot be interned alone (its lifetime bound defaults
    // to `'b`), so the whole path is, with `'b` among its generics, which the copy leaves in the
    // trait arguments of the Repeater (`<S as Repeater<(&'b (),), ..>>::Type`).
    let group = group(parse_quote!(dyn RelTr));
    for ty in [
        parse_quote!(::p::W<'b, dyn RelTr>),
        parse_quote!(::p::W<'b, (dyn RelTr)>),
        parse_quote!(::p::W<'b, #group>),
        parse_quote!(::core::cell::Ref<'b, dyn RelTr>),
        parse_quote!(::p::W<'b, (), dyn RelTr>),
        parse_quote!(::p::W<'b, dyn RelTr<Rel>>),
        // `Rel` may be a const (a bare identifier as an argument), which cannot be interned alone,
        // so the trait object has to be, and so is the path.
        parse_quote!(::p::W<'b, dyn ::p::Tr<Rel>>),
    ] {
        let ty: Type = ty;
        let item = local_trait(&ty);
        assert_eq!(
            leaked_trait(&item),
            [entry(ty.clone(), "< 'b >")],
            "{}",
            ty_string(ty.clone())
        );
        assert_eq!(
            rewritten(&item),
            printed(parse_quote! { trait Tr { fn f<'b>(&self, x: P<0usize>); } })
        );
    }
    // The argument is that of the innermost path, which is the one with a lifetime.
    let item = local_trait(&parse_quote!(::p::V<::p::W<'b, dyn RelTr>>));
    assert_eq!(
        leaked_trait(&item),
        [entry(parse_quote!(::p::W<'b, dyn RelTr>), "< 'b >")]
    );
    // A lifetime of a nested item, of an associated type or of a `for<..>` binder is local as well.
    let leaked = leaked_trait(&parse_quote! {
        trait Tr { fn f(&self, f: for<'x> fn(::p::W<'x, dyn RelTr>) -> ()); }
    });
    assert_eq!(
        leaked,
        [entry(parse_quote!(::p::W<'x, dyn RelTr>), "< 'x >")]
    );
    let leaked = leaked_trait(&parse_quote! {
        trait Tr { type A<'b>: ::p::Bound<::p::W<'b, dyn RelTr>>; }
    });
    assert_eq!(
        leaked,
        [entry(parse_quote!(::p::W<'b, dyn RelTr>), "< 'b >")]
    );

    // Not for `'_`, a path with several lifetime arguments (as before), nor in a trait bound,
    // which is not a type to be interned as a whole: the trait object cannot be alone, and the
    // path cannot be interned.
    let note =
        "the lifetime of the trait object defaults to that of the enclosing reference or path";
    for item in [
        parse_quote! { trait Tr { fn f(&self, x: ::p::W<'_, dyn RelTr>); } },
        parse_quote! { trait Tr<'a> { fn f<'b>(&self, x: ::p::W<'a, 'b, dyn RelTr>); } },
        parse_quote! { trait Tr { fn f<'b, 'c>(&self, x: ::p::W<'b, 'c, dyn RelTr>); } },
        parse_quote! { trait Tr<'a, T: ::p::Bound<'a, dyn RelTr>> {} },
        parse_quote! { trait Tr<'a>: ::p::Bound<'a, dyn RelTr> {} },
        parse_quote! { trait Tr { fn f<'b, T: ::p::Bound<'b, dyn RelTr>>(&self, t: T); } },
        parse_quote! { trait Tr { fn f<'b, T>(&self, t: T) where T: ::p::Bound<'b, dyn RelTr>; } },
        parse_quote! { trait Tr { type A<'b>: ::p::Bound<'b, dyn RelTr>; } },
    ] {
        let item: ItemTrait = item;
        let err = error_of(&item);
        assert!(err.contains("use absolute path: 'RelTr'"), "{}", err);
        assert!(err.contains(note), "{}", err);
    }
    // With nothing of the type context in the trait object, there is no error.
    for item in [
        parse_quote! { trait Tr { fn f<'b>(&self, x: ::p::W<'b, dyn ::core::any::Any>); } },
        parse_quote! { trait Tr<'a, T: ::p::Bound<'a, dyn ::core::any::Any>> {} },
        parse_quote! { trait Tr<'a, T: ::p::Bound<'static, dyn RelTr>> {} },
    ] {
        let item: ItemTrait = item;
        Leaker::new().intern_trait(&item, true).unwrap();
    }
    // Its components can still be leaked, as for a reference.
    let leaked = leaked_trait(&parse_quote! {
        trait Tr { fn f<'b>(&self, x: ::p::W<'b, dyn ::p::Tr<Item = Rel>>); }
    });
    assert_eq!(leaked, [entry(parse_quote!(Rel), "")]);
}

#[test]
fn test_reduce_roots_path_with_a_lifetime_the_copy_would_leave_unconstrained() {
    // The copy of the whole path has `'b` only in a projection, which does not constrain it. A
    // lifetime which no bound mentions is late-bound only if an input constrains it or the return
    // type does not mention it, so the copy of a method whose return type mentions `'b` has it
    // early-bound where the trait's is late-bound, which rustc rejects when the method is
    // implemented (E0195). So the path is not interned: an error, like for `'_`.
    let note = "the lifetime of the trait object defaults to `'b`, a lifetime of the method which \
                its return type mentions";
    for item in [
        parse_quote! { trait Tr { fn f<'b>(&self, x: ::p::W<'b, dyn RelTr>) -> &'b (); } },
        parse_quote! { trait Tr { fn f<'b>(&self, x: ::p::W<'b, (dyn RelTr)>) -> &'b (); } },
        parse_quote! { trait Tr { fn f<'b>(&self, x: ::p::W<'b, dyn RelTr>) -> ::p::V<'b>; } },
        parse_quote! { trait Tr { fn f<'b>(&self, x: ::p::W<'b, dyn RelTr>) -> Box<dyn Other + 'b>; } },
        // Not constrained by `Vec<&'b ()>`, which may be interned: conservative.
        parse_quote! {
            trait Tr { fn f<'b>(&self, x: ::p::W<'b, dyn RelTr>, y: Vec<&'b ()>) -> &'b (); }
        },
    ] {
        let item: ItemTrait = item;
        let err = error_of(&item);
        assert!(err.contains("use absolute path: 'RelTr'"), "{}", err);
        assert!(err.contains(note), "{}", err);
        assert!(err.contains("E0195"), "{}", err);
        assert!(
            err.contains("it cannot be leaked as a whole because of `'b`"),
            "{}",
            err
        );
        // Not the note which suggests to name the lifetime, which is named.
        assert!(!err.contains("name it"), "{}", err);
    }
    // The components are leaked if it is not the trait object which has to be interned.
    let leaked = leaked_trait(&parse_quote! {
        trait Tr { fn f<'b>(&self, x: ::p::W<'b, dyn ::p::Tr<Item = Rel>>) -> &'b (); }
    });
    assert_eq!(leaked, [entry(parse_quote!(Rel), "")]);
    // An input keeping `'b` where it is written constrains it in the copy (a reference, also in
    // a tuple, an array, a slice, a pointer or a reference to it; the receiver as well), so it is
    // late-bound in the copy as in the trait.
    for item in [
        parse_quote! { trait Tr { fn f<'b>(&'b self, x: ::p::W<'b, dyn RelTr>) -> &'b (); } },
        parse_quote! { trait Tr { fn f<'b>(&'b mut self, x: ::p::W<'b, dyn RelTr>) -> &'b (); } },
        parse_quote! { trait Tr { fn f<'b>(self: &'b Self, x: ::p::W<'b, dyn RelTr>) -> &'b (); } },
        parse_quote! { trait Tr { fn f<'b>(&self, x: ::p::W<'b, dyn RelTr>, y: &'b ()) -> &'b (); } },
        parse_quote! { trait Tr { fn f<'b>(&self, x: ::p::W<'b, dyn RelTr>, y: ((), &'b ())) -> &'b (); } },
        parse_quote! { trait Tr { fn f<'b>(&self, x: ::p::W<'b, dyn RelTr>, y: [&'b (); 2]) -> &'b (); } },
        parse_quote! { trait Tr { fn f<'b>(&self, x: ::p::W<'b, dyn RelTr>, y: &[&'b ()]) -> &'b (); } },
        parse_quote! { trait Tr { fn f<'b>(&self, x: ::p::W<'b, dyn RelTr>, y: *const &'b ()) -> &'b (); } },
        parse_quote! { trait Tr { fn f<'b>(&self, x: ::p::W<'b, dyn RelTr>, y: (&'b ())) -> &'b (); } },
        // A bound or a where predicate makes it early-bound in both.
        parse_quote! {
            trait Tr { fn f<'b, 'c: 'b>(&self, x: ::p::W<'b, dyn RelTr>, y: &'c ()) -> &'b (); }
        },
        parse_quote! {
            trait Tr { fn f<'b, T: 'b>(&self, x: ::p::W<'b, dyn RelTr>, y: T) -> &'b (); }
        },
        parse_quote! {
            trait Tr { fn f<'b>(&self, x: ::p::W<'b, dyn RelTr>) -> &'b () where Self: 'b; }
        },
        parse_quote! {
            trait Tr { fn f<'b, T>(&self, x: ::p::W<'b, dyn RelTr>, t: T) -> &'b () where T: ::p::Bound<'b>; }
        },
        // The return type does not mention it.
        parse_quote! { trait Tr { fn f<'b>(&self, x: ::p::W<'b, dyn RelTr>) -> (); } },
        // Another lifetime of the return type.
        parse_quote! { trait Tr { fn f<'b, 'c>(&self, x: ::p::W<'b, dyn RelTr>, y: &'c ()) -> &'c (); } },
    ] {
        let item: ItemTrait = item;
        let leaked = leaked_trait(&item);
        assert_eq!(leaked.len(), 1, "{}", ty_string_of(&item));
        // (Its context has the generics which a predicate mentioning `'b` brings along.)
        assert_eq!(leaked[0].0, ty_string(parse_quote!(::p::W<'b, dyn RelTr>)));
        assert!(leaked[0].1.starts_with("< 'b"), "{}", leaked[0].1);
    }
    // The lifetime of the trait is not a lifetime of the method: nothing to compare.
    let leaked = leaked_trait(&parse_quote! {
        trait Tr<'a> { fn f(&self, x: ::p::W<'a, dyn RelTr>) -> &'a (); }
    });
    assert_eq!(
        leaked,
        [entry(parse_quote!(::p::W<'a, dyn RelTr>), "< 'a >")]
    );
    // A function, which is not compared with a declaration, is not affected either: the lifetime
    // is a generic of the Leaker for the outermost item, and local for a nested one, as for a
    // method.
    let item: ItemFn = parse_quote! { fn f<'b>(x: ::p::W<'b, dyn RelTr>) -> &'b () { loop {} } };
    let mut leaker = Leaker::new();
    leaker
        .enter_generics(&item.sig.generics, |view| view.visit_signature(&item.sig))
        .unwrap();
    assert_eq!(leaker.finish().iter().count(), 1);
    // As local generics it is checked, in a scope which does not declare them, in a module and in
    // the function nested in one.
    let err = Leaker::new()
        .enter_generics(&Generics::default(), |view| view.visit_signature(&item.sig))
        .unwrap_err();
    assert!(syn::Error::from(err).to_string().contains("E0195"));
    let err = Leaker::new()
        .enter_generics(&Generics::default(), |view| {
            view.visit_item(&Item::Fn(item.clone()))
        })
        .unwrap_err();
    assert!(syn::Error::from(err).to_string().contains("E0195"));
    let item: ItemMod = parse_quote! {
        mod m { fn f<'b>(x: ::p::W<'b, dyn RelTr>) -> &'b () { loop {} } }
    };
    let err = Leaker::new().intern_mod(&item).err().unwrap();
    assert!(syn::Error::from(err).to_string().contains("E0195"));
}

/// The signature of the trait, for the messages.
fn ty_string_of(item: &ItemTrait) -> String {
    quote!(#item).to_string()
}

#[test]
fn test_reduce_roots_reference_with_a_local_lifetime() {
    // Like with a lifetime of the trait, the whole reference is interned if the trait object has
    // to be, with `'b` among its generics (the copy has `<S as Repeater<(&'b (),), ..>>::Type`),
    // so rustc gives the trait object its default in the Repeater's implementation: `'b`, or the
    // lifetime bound of the trait if it has one (`trait RelTr: 'static`), which the Leaker cannot
    // see. `fn f<'b, 'c, T>(&self, x: ..)` has no return type, so there is nothing to compare.
    let rel_object = group(parse_quote!(dyn RelTr));
    let wholes: Vec<(Type, &str)> = vec![
        (parse_quote!(&'b dyn RelTr), "< 'b >"),
        (parse_quote!(&'b mut dyn RelTr), "< 'b >"),
        (parse_quote!(&'b (dyn RelTr)), "< 'b >"),
        (parse_quote!(&'b ((dyn RelTr))), "< 'b >"),
        (parse_quote!(&'b #rel_object), "< 'b >"),
        (parse_quote!(&'b (dyn RelTr + Send)), "< 'b >"),
        (parse_quote!(&'b dyn RelTr<Item = ::p::X>), "< 'b >"),
        // A bare identifier in the arguments may be a const, which cannot be interned alone.
        (parse_quote!(&'b dyn ::p::Tr<SIZE>), "< 'b >"),
        // What is inside the trait object is bounded by the lifetime (`dyn Tr<T> + 'b: 'b`).
        (
            parse_quote!(&'b dyn RelTr<Item = ::p::X<T>>),
            "< 'b , T : 'b >",
        ),
    ];
    for (ty, generics) in wholes {
        let item: ItemTrait = parse_quote! { trait Tr { fn f<'b, 'c, T>(&self, x: #ty); } };
        assert_eq!(
            leaked_trait(&item),
            [entry(ty.clone(), generics)],
            "{}",
            ty_string(ty.clone())
        );
        assert_eq!(
            rewritten(&item),
            printed(parse_quote! { trait Tr { fn f<'b, 'c, T>(&self, x: P<0usize>); } }),
            "{}",
            ty_string(ty.clone())
        );
    }
    // The innermost reference is the type to intern, with the others bounding it.
    let item: ItemTrait = parse_quote! { trait Tr { fn f<'b, 'c>(&self, x: &'b &'c dyn RelTr); } };
    assert_eq!(
        leaked_trait(&item),
        [entry(parse_quote!(&'c dyn RelTr), "< 'b , 'c : 'b >")]
    );
    assert_eq!(
        rewritten(&item),
        printed(parse_quote! { trait Tr { fn f<'b, 'c>(&self, x: &'b P<0usize>); } })
    );
    // With a lifetime bound written, the trait object is the same alone and the reference stays
    // in the copy, as before.
    for (ty, leaked, generics) in [
        (
            parse_quote!(&'b (dyn RelTr + 'static)),
            parse_quote!(dyn RelTr + 'static),
            "",
        ),
        (
            parse_quote!(&'b (dyn RelTr + 'b)),
            parse_quote!(dyn RelTr + 'b),
            "< 'b >",
        ),
    ] {
        let (ty, leaked): (Type, Type) = (ty, leaked);
        let item: ItemTrait = parse_quote! { trait Tr { fn f<'b>(&self, x: #ty); } };
        assert_eq!(
            leaked_trait(&item),
            [entry(leaked, generics)],
            "{}",
            ty_string(ty.clone())
        );
        assert_eq!(
            rewritten(&item),
            printed(parse_quote! { trait Tr { fn f<'b>(&self, x: &'b (P<0usize>)); } }),
            "{}",
            ty_string(ty.clone())
        );
    }
    // Its components are interned if the trait object need not be, as for a lifetime of the
    // trait, and the reference and the trait object stay where they are.
    let item: ItemTrait = parse_quote! {
        trait Tr { fn f<'b>(&self, x: &'b dyn ::p::Tr<Item = Rel>); }
    };
    assert_eq!(leaked_trait(&item), [entry(parse_quote!(Rel), "")]);
    assert_eq!(
        rewritten(&item),
        printed(
            parse_quote! { trait Tr { fn f<'b>(&self, x: &'b dyn ::p::Tr<Item = P<0usize>>); } }
        )
    );
    // `check()` agrees, with the lifetime a generic of the context, as the lifetime of the trait.
    let result = Leaker::new().check(&parse_quote!(&'b dyn RelTr), &parse_quote!(<'b>));
    assert!(result.is_own);
    assert_eq!(result.must.len(), 1);
}

#[test]
fn test_reduce_roots_reference_with_a_lifetime_the_copy_would_leave_unconstrained() {
    // The same as for a path: the copy of the whole reference has `'b` only in a projection,
    // which does not constrain it, so a method whose return type mentions `'b` has it early-bound
    // in the copy where it is late-bound in the trait (if an input constrains it), which rustc
    // rejects when the method is implemented (E0195, checked with rustc on stable and nightly,
    // like for `::p::W<'b, dyn RelTr>`). So the reference is not interned: an error, which names
    // the lifetime.
    let errors: Vec<(ItemTrait, &str)> = vec![
        (
            parse_quote! { trait Tr { fn f<'b>(&self, x: &'b dyn RelTr) -> &'b (); } },
            "b",
        ),
        (
            parse_quote! { trait Tr { fn f<'b>(&self, x: &'b mut dyn RelTr) -> &'b (); } },
            "b",
        ),
        (
            parse_quote! { trait Tr { fn f<'b>(&self, x: &'b (dyn RelTr)) -> &'b (); } },
            "b",
        ),
        (
            parse_quote! { trait Tr { fn f<'b>(&self, x: &'b dyn RelTr) -> ::p::V<'b>; } },
            "b",
        ),
        (
            parse_quote! { trait Tr { fn f<'b>(&self, x: &'b dyn RelTr) -> Box<dyn Other + 'b>; } },
            "b",
        ),
        (
            parse_quote! { trait Tr { fn f<'b>(&self, x: &'b dyn RelTr) -> &'b dyn RelTr; } },
            "b",
        ),
        // The other arguments are references to a trait object, which are interned as well.
        (
            parse_quote! {
                trait Tr { fn f<'b>(&self, x: &'b dyn RelTr, y: &'b dyn RelTr2) -> &'b (); }
            },
            "b",
        ),
        // The reference around `&'c dyn RelTr` keeps `'b`, not `'c`.
        (
            parse_quote! { trait Tr { fn f<'b, 'c>(&self, x: &'b &'c dyn RelTr) -> &'c (); } },
            "c",
        ),
        // Not constrained by `Vec<&'b ()>`, which may be interned: conservative.
        (
            parse_quote! {
                trait Tr { fn f<'b>(&self, x: &'b dyn RelTr, y: Vec<&'b ()>) -> &'b (); }
            },
            "b",
        ),
    ];
    for (item, lifetime) in errors {
        let err = error_of(&item);
        assert!(err.contains("use absolute path: 'RelTr'"), "{}", err);
        assert!(
            err.contains(&format!(
                "the lifetime of the trait object defaults to `'{0}`, a lifetime of the method \
                 which its return type mentions",
                lifetime
            )),
            "{}",
            err
        );
        assert!(err.contains("E0195"), "{}", err);
        assert!(
            err.contains(&format!(
                "it cannot be leaked as a whole because of `'{}`",
                lifetime
            )),
            "{}",
            err
        );
        // Not the note which suggests to name the lifetime, which is named.
        assert!(!err.contains("name it"), "{}", err);
    }
    // The components are leaked if it is not the trait object which has to be interned.
    let leaked = leaked_trait(&parse_quote! {
        trait Tr { fn f<'b>(&self, x: &'b dyn ::p::Tr<Item = Rel>) -> &'b (); }
    });
    assert_eq!(leaked, [entry(parse_quote!(Rel), "")]);
    // So is the trait object with the lifetime bound written, which is interned alone, and the
    // reference stays in the copy.
    let leaked = leaked_trait(&parse_quote! {
        trait Tr { fn f<'b>(&self, x: &'b (dyn RelTr + 'b)) -> &'b (); }
    });
    assert_eq!(leaked, [entry(parse_quote!(dyn RelTr + 'b), "< 'b >")]);
    // An input keeping `'b` where it is written constrains it in the copy (the receiver, a
    // reference, also in a tuple, an array, a slice, a pointer or a reference to it), so it is
    // late-bound in the copy as in the trait.
    for item in [
        parse_quote! { trait Tr { fn f<'b>(&'b self, x: &'b dyn RelTr) -> &'b (); } },
        parse_quote! { trait Tr { fn f<'b>(&'b mut self, x: &'b dyn RelTr) -> &'b (); } },
        parse_quote! { trait Tr { fn f<'b>(self: &'b Self, x: &'b dyn RelTr) -> &'b (); } },
        parse_quote! { trait Tr { fn f<'b>(&self, x: &'b dyn RelTr, y: &'b ()) -> &'b (); } },
        parse_quote! { trait Tr { fn f<'b>(&self, x: &'b dyn RelTr, y: ((), &'b ())) -> &'b (); } },
        parse_quote! { trait Tr { fn f<'b>(&self, x: &'b dyn RelTr, y: [&'b (); 2]) -> &'b (); } },
        parse_quote! { trait Tr { fn f<'b>(&self, x: &'b dyn RelTr, y: &[&'b ()]) -> &'b (); } },
        parse_quote! { trait Tr { fn f<'b>(&self, x: &'b dyn RelTr, y: *const &'b ()) -> &'b (); } },
        // A bound or a where predicate makes it early-bound in both (`'c: 'b` also makes `'c`).
        parse_quote! {
            trait Tr { fn f<'b, 'c: 'b>(&self, x: &'b dyn RelTr, y: &'c ()) -> &'b (); }
        },
        parse_quote! {
            trait Tr { fn f<'b, 'c: 'b>(&self, x: &'b &'c dyn RelTr) -> &'c (); }
        },
        parse_quote! {
            trait Tr { fn f<'b, T: 'b>(&self, x: &'b dyn RelTr, y: T) -> &'b (); }
        },
        parse_quote! {
            trait Tr { fn f<'b>(&self, x: &'b dyn RelTr) -> &'b () where Self: 'b; }
        },
        // The return type does not mention it.
        parse_quote! { trait Tr { fn f<'b>(&self, x: &'b dyn RelTr) -> (); } },
        // Another lifetime of the return type.
        parse_quote! { trait Tr { fn f<'b, 'c>(&self, x: &'b dyn RelTr, y: &'c ()) -> &'c (); } },
    ] {
        let item: ItemTrait = item;
        let leaked = leaked_trait(&item);
        assert_eq!(leaked.len(), 1, "{}", ty_string_of(&item));
        assert!(
            leaked[0].0.starts_with("& 'b dyn RelTr") || leaked[0].0.starts_with("& 'c dyn RelTr"),
            "{}: {:?}",
            ty_string_of(&item),
            leaked
        );
        assert!(leaked[0].1.starts_with("< 'b"), "{}", leaked[0].1);
    }
    // A reference with a lifetime bound written on the trait object keeps `'b` (the trait object
    // is interned alone, and the reference stays in the copy).
    let mut leaked = leaked_trait(&parse_quote! {
        trait Tr { fn f<'b>(&self, x: &'b dyn RelTr, y: &'b (dyn RelTr + 'static)) -> &'b (); }
    });
    leaked.sort();
    assert_eq!(
        leaked,
        [
            entry(parse_quote!(&'b dyn RelTr), "< 'b >"),
            entry(parse_quote!(dyn RelTr + 'static), ""),
        ]
    );
    // The lifetime of the trait is not a lifetime of the method: nothing to compare.
    let leaked = leaked_trait(&parse_quote! {
        trait Tr<'a> { fn f(&self, x: &'a dyn RelTr) -> &'a (); }
    });
    assert_eq!(leaked, [entry(parse_quote!(&'a dyn RelTr), "< 'a >")]);
    // A nested function is checked like a method, an outermost one is not (see paths).
    let item: ItemFn = parse_quote! { fn f<'b>(x: &'b dyn RelTr) -> &'b () { loop {} } };
    let mut leaker = Leaker::new();
    leaker
        .enter_generics(&item.sig.generics, |view| view.visit_signature(&item.sig))
        .unwrap();
    assert_eq!(leaker.finish().iter().count(), 1);
    let err = Leaker::new()
        .enter_generics(&Generics::default(), |view| view.visit_signature(&item.sig))
        .unwrap_err();
    assert!(syn::Error::from(err).to_string().contains("E0195"));
    let item: ItemMod = parse_quote! {
        mod m { fn f<'b>(x: &'b dyn RelTr) -> &'b () { loop {} } }
    };
    let err = Leaker::new().intern_mod(&item).err().unwrap();
    assert!(syn::Error::from(err).to_string().contains("E0195"));
}

#[test]
fn test_reduce_roots_reference_to_a_trait_object_in_other_places() {
    // The reference is a type wherever it is written, so it is interned as a whole in a trait
    // bound, in a binder, in an associated type, as it is for a lifetime of the trait, while a
    // path with a local lifetime argument in a bound is an error.
    for item in [
        parse_quote! { trait Tr { fn f<'b, T: ::p::Bound<&'b dyn RelTr>>(&self, t: T); } },
        parse_quote! { trait Tr { fn f<'b, T>(&self, t: T) where T: ::p::Bound<&'b dyn RelTr>; } },
        parse_quote! { trait Tr { type A<'b>: ::p::Bound<&'b dyn RelTr>; } },
        parse_quote! { trait Tr<'b>: ::p::Bound<&'b dyn RelTr> {} },
        parse_quote! { trait Tr<'b, T: ::p::Bound<&'b mut dyn RelTr>> {} },
        parse_quote! { trait Tr { fn f(&self, f: for<'b> fn(&'b dyn RelTr) -> ()); } },
        parse_quote! { trait Tr { fn f<F: for<'b> ::core::ops::Fn(&'b dyn RelTr) -> ()>(&self, f: F); } },
        parse_quote! { trait Tr { fn f<F>(&self, f: F) where for<'b> F: ::core::ops::Fn(&'b dyn RelTr); } },
    ] {
        let item: ItemTrait = item;
        let leaked = leaked_trait(&item);
        assert_eq!(leaked.len(), 1, "{}", ty_string_of(&item));
        assert!(
            leaked[0].0.starts_with("& 'b dyn RelTr")
                || leaked[0].0.starts_with("& 'b mut dyn RelTr"),
            "{}: {:?}",
            ty_string_of(&item),
            leaked
        );
        // (The context also has what a bound mentioning the lifetime brings along.)
        assert!(leaked[0].1.starts_with("< 'b"), "{:?}", leaked);
    }
    // The path in a bound is still an error, and so is the elided reference in a bound.
    for item in [
        parse_quote! { trait Tr { fn f<'b, T: ::p::Bound<'b, dyn RelTr>>(&self, t: T); } },
        parse_quote! { trait Tr { fn f<T: ::p::Bound<&dyn RelTr>>(&self, t: T); } },
        parse_quote! { trait Tr { fn f<T: ::p::Bound<&'_ dyn RelTr>>(&self, t: T); } },
    ] {
        let item: ItemTrait = item;
        let err = error_of(&item);
        assert!(err.contains("use absolute path: 'RelTr'"), "{}", err);
    }
    // An undeclared lifetime is one of the type context, which makes the reference the type to
    // intern, as before.
    assert_eq!(
        leaked_trait(&parse_quote! { trait Tr { fn f(&self, x: &'r dyn RelTr); } }),
        [entry(parse_quote!(&'r dyn RelTr), "")]
    );
    // `'static` is the default of a trait object alone.
    assert_eq!(
        leaked_trait(&parse_quote! { trait Tr { fn f(&self, x: &'static mut dyn RelTr); } }),
        [entry(parse_quote!(dyn RelTr), "")]
    );
}

#[test]
fn test_reduce_roots_reference_with_a_local_lifetime_is_found_in_its_scope() {
    use syn::visit_mut::VisitMut;
    // The entries are the whole references (`&'b dyn RelTr` of the method, `&'a mut dyn RelTr` of
    // the trait), found where the lifetime is declared: by the method (`'b`) or by the context,
    // that is the trait (`'a`).
    let tr: ItemTrait = parse_quote! {
        trait Tr<'a> {
            fn f<'b>(&'b self, x: &'b dyn RelTr, y: &'a mut dyn RelTr) -> &'b dyn RelTr;
        }
    };
    let mut leaker = Leaker::new();
    let top_level = leaker.intern_trait(&tr, true).unwrap();
    let referrer = leaker.finish();
    let mut entries: Vec<(String, String)> = referrer
        .iter_with_generics()
        .map(|(ty, generics)| (ty_string(ty.clone()), quote!(#generics).to_string()))
        .collect();
    entries.sort();
    assert_eq!(
        entries,
        [
            entry(parse_quote!(&'a mut dyn RelTr), "< 'a >"),
            entry(parse_quote!(&'b dyn RelTr), "< 'b >"),
        ]
    );
    let expected: ItemTrait = parse_quote! {
        trait Tr<'a> {
            fn f<'b>(&'b self, x: P<{b}>, y: P<{a}>) -> P<{b}>;
        }
    };
    let index_of = |ty: &str| {
        referrer
            .iter()
            .position(|found| ty_string(found.clone()) == ty)
            .unwrap()
    };
    let (a, b) = (index_of("& 'a mut dyn RelTr"), index_of("& 'b dyn RelTr"));
    let printed_expected = {
        let text = quote!(#expected).to_string();
        text.replace("{ b }", &format!("{}usize", b))
            .replace("{ a }", &format!("{}usize", a))
    };
    let rewrite = |visitor: &mut dyn FnMut(&mut ItemTrait)| {
        let mut copy = tr.clone();
        visitor(&mut copy);
        quote!(#copy).to_string()
    };
    // The copy of the trait declares `<'a>` as the context does, so it does not declare it again.
    let by_scope = rewrite(&mut |copy| {
        referrer
            .clone()
            .visitor(top_level, |_, ix| parse_quote!(P<#ix>))
            .visit_item_trait_mut(copy)
    });
    assert_eq!(by_scope, printed_expected);
    // With the generics of the trait as those of the context, whatever the copy declares.
    let by_leakers = rewrite(&mut |copy| {
        referrer
            .clone()
            .visitor(top_level, |_, ix| parse_quote!(P<#ix>))
            .with_outermost_generics_as_leakers()
            .visit_item_trait_mut(copy)
    });
    assert_eq!(by_leakers, printed_expected);
    // The type passed to the callback is the one leaked, whichever way the trait is visited.
    for leakers in [false, true] {
        let mut seen = Vec::new();
        let mut copy = tr.clone();
        let record = |ty: Type, ix: usize| -> Type {
            seen.push(ty_string(ty));
            parse_quote!(P<#ix>)
        };
        let visitor = referrer.clone().visitor(top_level, record);
        if leakers {
            visitor
                .with_outermost_generics_as_leakers()
                .visit_item_trait_mut(&mut copy);
        } else {
            let mut visitor = visitor;
            visitor.visit_item_trait_mut(&mut copy);
        }
        seen.sort();
        seen.dedup();
        assert_eq!(seen, ["& 'a mut dyn RelTr", "& 'b dyn RelTr"]);
    }
    // The signature alone: the method declares `'b`, and the context `'a`.
    let mut sig: Signature = parse_quote! {
        fn f<'b>(&'b self, x: &'b dyn RelTr, y: &'a mut dyn RelTr) -> &'b dyn RelTr
    };
    referrer
        .clone()
        .visitor(top_level, |_, ix| parse_quote!(P<#ix>))
        .visit_signature_mut(&mut sig);
    let expected: Signature = parse_quote! {
        fn f<'b>(&'b self, x: P<{b}>, y: P<{a}>) -> P<{b}>
    };
    assert_eq!(
        quote!(#sig).to_string(),
        quote!(#expected)
            .to_string()
            .replace("{ b }", &format!("{}usize", b))
            .replace("{ a }", &format!("{}usize", a))
    );
    // `expand()` knows the lifetime of the context, not that of a method: `'b` is not declared.
    let ty: Type = parse_quote!(&'a mut dyn RelTr);
    let expanded = referrer.expand(top_level, ty, |_, ix| parse_quote!(P<#ix>));
    assert_eq!(ty_string(expanded), format!("P < {}usize >", a));
    let ty: Type = parse_quote!(&'b dyn RelTr);
    let expanded = referrer.expand(top_level, ty.clone(), |_, ix| parse_quote!(P<#ix>));
    assert_eq!(ty_string(expanded), ty_string(ty));
    // A trait object which is not under the reference is not an entry: its default is that of
    // another path or `'static`.
    let ty: Type = parse_quote!(::std::boxed::Box<dyn RelTr>);
    let expanded = referrer.expand(top_level, ty.clone(), |_, ix| parse_quote!(P<#ix>));
    assert_eq!(ty_string(expanded), ty_string(ty));
    let ty: Type = parse_quote!(&'a ::std::boxed::Box<dyn RelTr>);
    let expanded = referrer.expand(top_level, ty.clone(), |_, ix| parse_quote!(P<#ix>));
    assert_eq!(ty_string(expanded), ty_string(ty));
    // The Referrer survives the serialization.
    let tokens = quote!(#referrer);
    let parsed: type_leak::Referrer = syn::parse2(tokens).unwrap();
    assert_eq!(parsed, referrer);
    let again = parsed.expand(
        top_level,
        parse_quote!(&'a mut dyn RelTr),
        |_, ix| parse_quote!(P<#ix>),
    );
    assert_eq!(ty_string(again), format!("P < {}usize >", a));
}

/// A group around `ty`, like the one `$t:ty` of a `macro_rules!` is parsed as.
fn group(ty: Type) -> Type {
    Type::Group(TypeGroup {
        group_token: Default::default(),
        elem: Box::new(ty),
    })
}

/// The leaked types of a trait returning `ty`.
fn leaked_returning(ty: &Type) -> Vec<(String, String)> {
    leaked_trait(&parse_quote! { trait Tr { fn c(&self) -> #ty; } })
}

#[test]
fn test_reduce_roots_bare_identifier_argument() {
    // `SIZE` is parsed as a type, but may be a const: `type Type = SIZE;` would not compile, so
    // the path it is an argument of is interned as a whole if it can be. There is no naming
    // convention to tell a type and a const: `Size` and `Rel` are treated alike.
    let item: ItemTrait = parse_quote! {
        trait Tr { fn c(&self) -> ::p::W<SIZE>; }
    };
    assert_eq!(leaked_trait(&item), [entry(parse_quote!(::p::W<SIZE>), "")]);
    assert_eq!(
        rewritten(&item),
        printed(parse_quote! { trait Tr { fn c(&self) -> P<0usize>; } })
    );
    for ty in [
        parse_quote!(::p::W<SIZE>),
        parse_quote!(::p::W<Size>),
        parse_quote!(::p::W<Rel>),
        parse_quote!(::p::W<(), SIZE>),
        parse_quote!(::p::W<SIZE, Rel>),
        parse_quote!(::p::W<Rel, ::p::V<SIZE>>),
        parse_quote!(::std::vec::Vec<Rel>),
        parse_quote!(::core::option::Option<Rel>),
        parse_quote!(<() as ::p::Sup<SIZE>>::Out),
    ] {
        let ty: Type = ty;
        assert_eq!(
            leaked_returning(&ty),
            [entry(ty.clone(), "")],
            "{}",
            ty_string(ty.clone())
        );
    }
    // Whatever encloses it is not interned.
    for (ty, expected) in [
        (
            parse_quote!(::std::vec::Vec<::p::W<SIZE>>),
            parse_quote!(::p::W<SIZE>),
        ),
        (
            parse_quote!(::p::V<::p::W<SIZE>>),
            parse_quote!(::p::W<SIZE>),
        ),
        (parse_quote!((::p::W<SIZE>, ())), parse_quote!(::p::W<SIZE>)),
        // The path with an argument which cannot be interned is not, its components are.
        (
            parse_quote!(::p::W<'_, ::p::V<SIZE>>),
            parse_quote!(::p::V<SIZE>),
        ),
    ] {
        let ty: Type = ty;
        let expected: Type = expected;
        assert_eq!(
            leaked_returning(&ty),
            [entry(expected, "")],
            "{}",
            ty_string(ty.clone())
        );
    }
    // The other components are leaked as before.
    assert_eq!(
        leaked_trait(&parse_quote! {
            trait Tr { fn c(&self, x: ::p::W<SIZE>, y: ::std::vec::Vec<(Rel,)>); }
        }),
        [
            entry(parse_quote!(::p::W<SIZE>), ""),
            entry(parse_quote!(Rel), ""),
        ]
    );
    // A reference is interned as a whole already.
    assert_eq!(
        leaked_trait(&parse_quote! { trait Tr<'a> { fn c(&self) -> &'a ::p::W<SIZE>; } }),
        [entry(parse_quote!(&'a ::p::W<SIZE>), "< 'a >")]
    );
    // A group (like that of `$t:ty` in a `macro_rules!`) is transparent.
    let size = group(parse_quote!(SIZE));
    let nested = group(group(parse_quote!(SIZE)));
    for ty in [
        parse_quote!(::p::W<#size>),
        parse_quote!(::p::W<#nested>),
        parse_quote!(::std::vec::Vec<::p::W<#size>>),
    ] {
        let ty: Type = ty;
        let expected: Type = match ty_string(ty.clone()).starts_with(":: std") {
            true => parse_quote!(::p::W<#size>),
            false => ty.clone(),
        };
        assert_eq!(
            leaked_returning(&ty),
            [entry(expected, "")],
            "{}",
            ty_string(ty.clone())
        );
    }
    // The generics it uses.
    assert_eq!(
        leaked_trait(&parse_quote! { trait Tr { fn c<T>(&self) -> ::p::W<T, Rel>; } }),
        [entry(parse_quote!(::p::W<T, Rel>), "< T >")]
    );
    assert_eq!(
        leaked_trait(&parse_quote! { trait Tr { fn c<'b>(&self) -> ::p::W<'b, SIZE>; } }),
        [entry(parse_quote!(::p::W<'b, SIZE>), "< 'b >")]
    );
    // A generic or a primitive is never ambiguous: there is nothing to leak.
    assert!(leaked_trait(&parse_quote! {
        trait Tr<T> { fn c(&self) -> ::p::W<T, ::core::primitive::u8>; }
    })
    .is_empty());
    // `check()` agrees: the path must be interned as a whole.
    let result = Leaker::new().check(&parse_quote!(::p::W<SIZE>), &Generics::default());
    assert!(result.is_own);
    assert!(result.impossible.is_empty());
    assert_eq!(result.must.len(), 1);
}

#[test]
fn test_reduce_roots_bare_identifier_is_a_type_if_the_path_cannot_be_interned() {
    // The path cannot be interned, so the identifier is leaked alone, as a type, like any other
    // (and not as an error).
    for item in [
        // An elided lifetime.
        parse_quote! { trait Tr { fn c(&self) -> ::p::W<'_, SIZE>; } },
        // `impl Trait`.
        parse_quote! { trait Tr { fn c(&self, x: impl ::p::Tr<SIZE>); } },
        parse_quote! { trait Tr { fn c(&self, x: ::p::W<SIZE, impl ::p::Tr>); } },
        // Another impossible cause in the path.
        parse_quote! { trait Tr { fn c(&self, x: ::p::W<SIZE, &()>); } },
        // The default of a parameter cannot depend on `Self`.
        parse_quote! { trait Tr<X = ::p::W<Self, SIZE>> {} },
        // A trait bound is not a type.
        parse_quote! { trait Tr: ::p::Sup<SIZE> {} },
        parse_quote! { trait Tr { fn f<T: ::p::Sup<SIZE>>(&self); } },
        parse_quote! { trait Tr<T: ::p::Sup<SIZE>> {} },
    ] {
        let item: ItemTrait = item;
        assert_eq!(
            leaked_trait(&item),
            [entry(parse_quote!(SIZE), "")],
            "{}",
            printed(item.clone())
        );
    }
    // The same as for any identifier.
    assert_eq!(
        leaked_returning(&parse_quote!(::p::W<'_, Rel>)),
        [entry(parse_quote!(Rel), "")]
    );
    // A group is transparent.
    let size = group(parse_quote!(SIZE));
    assert_eq!(
        leaked_returning(&parse_quote!(::p::W<'_, #size>)),
        [entry(parse_quote!(SIZE), "")]
    );
    // `check()` agrees: the identifier is the cause, not the path.
    let result = Leaker::new().check(&parse_quote!(::p::W<'_, SIZE>), &Generics::default());
    assert!(!result.is_own);
    assert_eq!(result.impossible.len(), 1);
    assert_eq!(result.must.len(), 1);
    assert_eq!(result.must[0].to_string(), "SIZE");

    // The same if the bounds of the generics mention `Self`, which cannot be leaked.
    let item: ItemTrait = parse_quote! {
        trait Tr<T: ::p::Bound<Self>> { fn c(&self, x: ::p::W<T, SIZE>); }
    };
    let leaked = |allow_self: bool| {
        let mut leaker = Leaker::new();
        common::intern_fns(
            &mut leaker,
            &item.generics,
            quote!(
                fn c(&self, x: ::p::W<T, SIZE>);
            ),
            allow_self,
        )
        .unwrap();
        leaker
            .finish()
            .iter()
            .map(|ty| ty_string(ty.clone()))
            .collect::<Vec<_>>()
    };
    assert_eq!(leaked(false), [ty_string(parse_quote!(SIZE))]);
    // And the path is interned if it can be.
    assert_eq!(leaked(true), [ty_string(parse_quote!(::p::W<T, SIZE>))]);
}

#[test]
fn test_reduce_roots_bare_identifier_in_other_positions_is_a_type() {
    // Only an identifier written directly as a generic argument may be a const. In the other
    // positions it is a type, which is leaked alone (also if the type around it is not interned).
    let paren_size = group(parse_quote!((SIZE)));
    for ty in [
        // Parentheses: `W<(SIZE)>` is a type (unlike `W<SIZE>`).
        parse_quote!(::p::W<(SIZE)>),
        parse_quote!(::p::W<((SIZE))>),
        parse_quote!(::p::W<#paren_size>),
        // Tuples, arrays, slices and function pointers.
        parse_quote!(::p::W<(SIZE,)>),
        parse_quote!(::p::W<(SIZE, ())>),
        parse_quote!(::p::W<[SIZE; 3]>),
        parse_quote!(::p::W<[SIZE]>),
        parse_quote!(::p::W<fn(SIZE)>),
        parse_quote!(::p::W<fn() -> SIZE>),
        // The arguments of `Fn(..) -> ..`, and the binding of an associated type.
        parse_quote!(::p::W<dyn ::core::ops::Fn(SIZE) -> SIZE>),
        parse_quote!(::p::W<dyn ::core::iter::Iterator<Item = SIZE>>),
        // The qualified self type.
        parse_quote!(::p::W<<SIZE as ::p::Tr>::Out>),
        parse_quote!(<SIZE as ::p::Tr>::Out),
        // Not as an argument.
        parse_quote!(SIZE),
        parse_quote!(::std::boxed::Box<(SIZE,)>),
    ] {
        let ty: Type = ty;
        assert_eq!(
            leaked_returning(&ty),
            [entry(parse_quote!(SIZE), "")],
            "{}",
            ty_string(ty.clone())
        );
    }
    // The path of the identifier written out is not a bare identifier.
    assert_eq!(
        leaked_returning(&parse_quote!(::p::W<rel::SIZE>)),
        [entry(parse_quote!(rel::SIZE), "")]
    );
    // The binding of an associated type in a bound.
    assert_eq!(
        leaked_trait(&parse_quote! {
            trait Tr { fn f<I: ::core::iter::Iterator<Item = SIZE>>(&self); }
        }),
        [entry(parse_quote!(SIZE), "")]
    );
    // The same for any identifier.
    for ty in [
        parse_quote!(::p::W<(Rel,)>),
        parse_quote!(::p::W<[Rel; 3]>),
        parse_quote!(::p::W<fn(Rel)>),
        parse_quote!(::p::W<(Rel)>),
        parse_quote!(::std::vec::Vec<(Rel, ())>),
    ] {
        let ty: Type = ty;
        assert_eq!(
            leaked_returning(&ty),
            [entry(parse_quote!(Rel), "")],
            "{}",
            ty_string(ty.clone())
        );
    }
}

#[test]
fn test_reduce_roots_bare_identifier_in_a_trait_object_inheriting_a_lifetime() {
    // The whole path is interned, also with parentheses or a group around the trait object, which
    // are not interned (nor are the types around the identifier).
    let object = group(parse_quote!(dyn ::p::Tr<SIZE>));
    for ty in [
        parse_quote!(::p::W<'a, dyn ::p::Tr<SIZE>>),
        parse_quote!(::p::W<'a, (dyn ::p::Tr<SIZE>)>),
        parse_quote!(::p::W<'a, #object>),
    ] {
        let ty: Type = ty;
        let item: ItemTrait = parse_quote! { trait Tr<'a> { fn f(&self, x: #ty); } };
        assert_eq!(
            leaked_trait(&item),
            [entry(ty.clone(), "< 'a >")],
            "{}",
            ty_string(ty.clone())
        );
    }
    // The referent of a reference makes the reference to be interned as a whole.
    let ty: Type = parse_quote!(&'a dyn ::p::Tr<SIZE>);
    let paren: Type = parse_quote!(&'a (dyn ::p::Tr<SIZE>));
    let grouped: Type = parse_quote!(&'a #object);
    for ty in [ty, paren, grouped] {
        let item: ItemTrait = parse_quote! { trait Tr<'a> { fn f(&self, x: #ty); } };
        assert_eq!(
            leaked_trait(&item),
            [entry(ty.clone(), "< 'a >")],
            "{}",
            ty_string(ty.clone())
        );
    }
}

#[test]
fn test_reduce_roots_bare_identifier_in_a_bound_is_not_an_error() {
    // At the root, `SIZE` is leaked as a type, whatever it is.
    assert_eq!(
        leaked_trait(&parse_quote! { trait Tr: ::p::Sup<SIZE> {} }),
        [entry(parse_quote!(SIZE), "")]
    );
    assert_eq!(
        leaked_trait(&parse_quote! { trait Tr { fn f<T: ::p::Sup<SIZE>>(&self); } }),
        [entry(parse_quote!(SIZE), "")]
    );
    assert_eq!(
        leaked_trait(&parse_quote! { trait Tr { fn f<T: ::p::Sup<Size>>(&self); } }),
        [entry(parse_quote!(Size), "")]
    );
    // A const in braces, which cannot be leaked on its own, is still an error.
    let err = error_of(&parse_quote! { trait Tr: ::p::Sup<{ SIZE }> {} });
    assert!(err.contains("use absolute path: 'SIZE'"), "{}", err);
}

#[test]
fn test_reduce_roots_types_in_expressions() {
    // The copy of `[::core::primitive::u8; size_of::<Big>()]` with `Big` replaced by `<Self as Repeater<..>>::Type`
    // does not compile for a generic implementor ("generic `Self` types are currently not
    // permitted in anonymous constants"), so the array is interned as a whole.
    let item: ItemTrait = parse_quote! {
        trait Tr {
            fn k(&self) -> ::core::option::Option<[::core::primitive::u8; ::core::mem::size_of::<Big>()]>;
        }
    };
    assert_eq!(
        leaked_trait(&item),
        [entry(
            parse_quote!([::core::primitive::u8; ::core::mem::size_of::<Big>()]),
            ""
        )]
    );
    assert_eq!(
        rewritten(&item),
        printed(parse_quote! { trait Tr { fn k(&self) -> ::core::option::Option<P<0usize>>; } })
    );
    // A const argument, also nested in an expression, and the other expressions.
    for (ty, expected) in [
        (
            parse_quote!(::p::W<{ ::core::mem::size_of::<Big>() }>),
            parse_quote!(::p::W<{ ::core::mem::size_of::<Big>() }>),
        ),
        (
            parse_quote!(::std::vec::Vec<::p::W<{ ::core::mem::size_of::<Big>() }>>),
            parse_quote!(::p::W<{ ::core::mem::size_of::<Big>() }>),
        ),
        (
            parse_quote!([::core::primitive::u8; ::core::mem::size_of::<::p::V<Big>>()]),
            parse_quote!([::core::primitive::u8; ::core::mem::size_of::<::p::V<Big>>()]),
        ),
        (
            parse_quote!(
                [::core::primitive::u8;
                    ::core::mem::size_of::<[Big; ::core::mem::size_of::<Other>()]>()]
            ),
            parse_quote!(
                [::core::primitive::u8;
                    ::core::mem::size_of::<[Big; ::core::mem::size_of::<Other>()]>()]
            ),
        ),
        (
            parse_quote!(
                [::core::primitive::u8; {
                    let n: Big = ::p::make();
                    ::core::mem::size_of_val(&n)
                }]
            ),
            parse_quote!(
                [::core::primitive::u8; {
                    let n: Big = ::p::make();
                    ::core::mem::size_of_val(&n)
                }]
            ),
        ),
        // The element is in the array.
        (
            parse_quote!([Rel; ::core::mem::size_of::<Big>()]),
            parse_quote!([Rel; ::core::mem::size_of::<Big>()]),
        ),
        (
            parse_quote!((Rel, [::core::primitive::u8; ::core::mem::size_of::<Big>()])),
            parse_quote!((Rel, [::core::primitive::u8; ::core::mem::size_of::<Big>()])),
        ),
    ] {
        let ty: Type = ty;
        let expected: Type = expected;
        let leaked = leaked_trait(&parse_quote! { trait Tr { fn k(&self) -> #ty; } });
        assert_eq!(leaked, [entry(expected, "")], "{}", ty_string(ty.clone()));
    }
    // The types outside the expressions are leaked as before: an element, or another occurrence.
    assert_eq!(
        leaked_trait(&parse_quote! { trait Tr { fn k(&self) -> [Rel; 3]; } }),
        [entry(parse_quote!(Rel), "")]
    );
    assert_eq!(
        leaked_trait(&parse_quote! {
            trait Tr<const N: usize> { fn k(&self) -> ::std::vec::Vec<[Rel; N]>; }
        }),
        [entry(parse_quote!(Rel), "")]
    );
    assert_eq!(
        leaked_trait(&parse_quote! {
            trait Tr {
                fn k(
                    &self,
                    x: [::core::primitive::u8; ::core::mem::size_of::<Big>()],
                    y: ::std::vec::Vec<(Big,)>,
                );
            }
        }),
        [
            entry(parse_quote!(Big), ""),
            entry(
                parse_quote!([::core::primitive::u8; ::core::mem::size_of::<Big>()]),
                ""
            ),
        ]
    );
    // Nothing is leaked if the types in the expression do not depend on the type context.
    assert!(leaked_trait(&parse_quote! {
        trait Tr { fn k(&self) -> [::core::primitive::u8; ::core::mem::size_of::<::p::Big>()]; }
    })
    .is_empty());
    // A generic is not a cause either.
    assert!(leaked_trait(&parse_quote! {
        trait Tr<T> { fn k(&self) -> [::core::primitive::u8; ::core::mem::size_of::<T>()]; }
    })
    .is_empty());
    // `check()` agrees: the array must be interned as a whole.
    let result = Leaker::new().check(
        &parse_quote!([::core::primitive::u8; ::core::mem::size_of::<Big>()]),
        &Generics::default(),
    );
    assert!(result.is_own);
    assert_eq!(result.must.len(), 1);
    // A conflict is reported as for the own path of a type.
    let err = error_of(&parse_quote! {
        trait Tr { fn k(&self) -> [impl ::p::Tr; ::core::mem::size_of::<Big>()]; }
    });
    assert!(err.contains("use absolute path: 'Big'"), "{}", err);
}

#[test]
fn test_reduce_roots_independent_components() {
    // The groups of types which share nothing are solved on their own.
    let leaked = leaked_of(parse_quote! {
        struct S {
            // Cheaper as a whole (1 instead of 3), used twice.
            a: (RelA, RelB, RelC),
            b: ::std::vec::Vec<(RelA, RelB, RelC)>,
            // The inner type.
            c: ::std::vec::Vec<(RelD,)>,
            // Two wrappers of one type, with the type itself as a root.
            d: ::std::vec::Vec<(RelE,)>,
            e: ::core::option::Option<(RelE,)>,
            f: RelE,
            // A red root which is also a component.
            g: RelF,
            h: (RelF, RelG),
            i: ::std::vec::Vec<(RelF,)>,
        }
    });
    assert_eq!(
        leaked,
        [
            ty_string(parse_quote!((RelA, RelB, RelC))),
            ty_string(parse_quote!(RelD)),
            ty_string(parse_quote!(RelE)),
            ty_string(parse_quote!(RelF)),
            ty_string(parse_quote!(RelG)),
        ]
    );
    // The wrappers of a type are joined by it, and solved together: the tuple (1) is cut instead
    // of its two components (2).
    let leaked = leaked_of(parse_quote! {
        struct S {
            a: ::std::vec::Vec<(RelA, RelB)>,
            b: ::core::option::Option<(RelA, RelB)>,
            c: ::std::boxed::Box<(RelA, RelB)>,
        }
    });
    assert_eq!(leaked, [ty_string(parse_quote!((RelA, RelB)))]);
}

#[test]
fn test_reduce_roots_red_roots_are_interned() {
    // A red root is interned even where it is also under a wrapper, which then needs nothing.
    let leaked = leaked_of(parse_quote! {
        struct S {
            a: Rel,
            b: ::std::vec::Vec<(Rel,)>,
            c: (Rel, ::std::string::String),
            d: ::std::vec::Vec<(Rel, RelB)>,
        }
    });
    assert_eq!(
        leaked,
        [ty_string(parse_quote!(Rel)), ty_string(parse_quote!(RelB))]
    );
    // The type containing it is cheaper than its other components.
    let leaked = leaked_of(parse_quote! {
        struct S {
            a: Rel,
            b: (Rel, RelB, RelC),
        }
    });
    assert_eq!(
        leaked,
        [
            ty_string(parse_quote!((Rel, RelB, RelC))),
            ty_string(parse_quote!(Rel)),
        ]
    );
}

#[test]
fn test_reduce_roots_large_item() {
    // Many independent fields, which a flow over the whole network would take quadratic time
    // for (its search is repeated for each of them). This finishes quickly as the components are
    // solved on their own.
    let n = 2000;
    let fields = (0..n).map(|i| {
        let name = Ident::new(&format!("f{}", i), proc_macro2::Span::call_site());
        let rel = Ident::new(&format!("Rel{}", i), proc_macro2::Span::call_site());
        quote!(#name: ::core::option::Option<(#rel,)>)
    });
    let item: ItemStruct = syn::parse2(quote! { struct S { #(#fields),* } }).unwrap();
    let mut leaker = Leaker::new();
    leaker.intern_struct(&item, true).unwrap();
    let referrer = leaker.finish();
    assert_eq!(referrer.iter().count(), n);
    // The inner types are leaked.
    assert!(referrer
        .iter()
        .all(|ty| quote!(#ty).to_string().starts_with("Rel")));

    // Types of which the components are shared are solved together, and chosen as before.
    let n = 500;
    let fields = (0..n).map(|i| {
        let name = Ident::new(&format!("f{}", i), proc_macro2::Span::call_site());
        let r = Ident::new(&format!("R{}", i), proc_macro2::Span::call_site());
        let q = Ident::new(&format!("Q{}", i), proc_macro2::Span::call_site());
        quote!(#name: (::std::vec::Vec<(#r,)>, ::core::option::Option<(#q,)>, (#r, #q)))
    });
    let item: ItemStruct = syn::parse2(quote! { struct S { #(#fields),* } }).unwrap();
    let mut leaker = Leaker::new();
    leaker.intern_struct(&item, true).unwrap();
    let referrer = leaker.finish();
    // The tuple (1) is cheaper than `Ri` and `Qi` (2).
    assert_eq!(referrer.iter().count(), n);
    assert!(referrer
        .iter()
        .all(|ty| matches!(ty, Type::Tuple(tuple) if tuple.elems.len() == 3)));
}

/// A random type: `Ra`, `Rb` and `Rc` are red, as is every `Rg<..>`, and so is a reference to a
/// type with a red component (which is interned as a whole).
///
/// The red leaves are written as `m::Ra`, not as a bare identifier: as an argument of a path, it
/// may be a const, so the path would be interned as a whole instead (which makes it red as well).
#[derive(Clone)]
enum Model {
    Leaf(&'static str),
    Vec(Box<Model>),
    Tuple(Vec<Model>),
    W(Box<Model>, Box<Model>),
    Rg(Box<Model>),
    Ref(Box<Model>),
}

impl Model {
    fn random(rng: &mut u64, depth: usize) -> Self {
        let mut next = |n: u64| {
            *rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (*rng >> 33) % n
        };
        if depth > 2 || next(100) < 35 {
            return Model::Leaf(["Ra", "Rb", "Rc", "u8", "T"][next(5) as usize]);
        }
        match next(5) {
            0 => Model::Vec(Box::new(Self::random(rng, depth + 1))),
            1 => {
                let n = 2 + next(2) as usize;
                Model::Tuple((0..n).map(|_| Self::random(rng, depth + 1)).collect())
            }
            2 => Model::W(
                Box::new(Self::random(rng, depth + 1)),
                Box::new(Self::random(rng, depth + 1)),
            ),
            3 => Model::Rg(Box::new(Self::random(rng, depth + 1))),
            _ => Model::Ref(Box::new(Self::random(rng, depth + 1))),
        }
    }

    fn ty(&self) -> Type {
        match self {
            Model::Leaf("u8") => parse_quote!(::core::primitive::u8),
            Model::Leaf("T") => parse_quote!(T),
            Model::Leaf(name) => syn::parse_str(&format!("m::{}", name)).unwrap(),
            Model::Vec(a) => {
                let a = a.ty();
                parse_quote!(::std::vec::Vec<#a>)
            }
            Model::Tuple(elems) => {
                let elems = elems.iter().map(|e| e.ty());
                parse_quote!((#(#elems),*))
            }
            Model::W(a, b) => {
                let (a, b) = (a.ty(), b.ty());
                parse_quote!(::p::W<#a, #b>)
            }
            Model::Rg(a) => {
                let a = a.ty();
                parse_quote!(Rg<#a>)
            }
            Model::Ref(a) => {
                let a = a.ty();
                parse_quote!(&'a #a)
            }
        }
    }

    fn is_red(&self) -> bool {
        match self {
            Model::Leaf(name) => matches!(*name, "Ra" | "Rb" | "Rc"),
            Model::Rg(_) => true,
            // The copy of a reference needs its referent to outlive the lifetime.
            Model::Ref(a) => a.has_red(),
            _ => false,
        }
    }

    fn children(&self) -> Vec<&Model> {
        match self {
            Model::Leaf(_) | Model::Rg(_) | Model::Ref(_) => Vec::new(),
            Model::Vec(a) => vec![a],
            Model::Tuple(elems) => elems.iter().collect(),
            Model::W(a, b) => vec![a, b],
        }
    }

    fn has_red(&self) -> bool {
        match self {
            Model::Ref(a) => a.has_red(),
            _ => self.is_red() || self.children().iter().any(|c| c.has_red()),
        }
    }

    /// Adds the node of the type (if it contains a red type), and those of its components.
    fn add(&self, parent: Option<usize>, graph: &mut ModelGraph) {
        if !self.has_red() {
            return;
        }
        let name = ty_string(self.ty());
        let ix = graph.node(&name, self.is_red());
        match parent {
            Some(parent) => {
                if !graph.children[parent].contains(&ix) {
                    graph.children[parent].push(ix)
                }
            }
            None => {
                if !graph.roots.contains(&ix) {
                    graph.roots.push(ix)
                }
            }
        }
        for child in self.children() {
            child.add(Some(ix), graph);
        }
    }
}

#[derive(Default)]
struct ModelGraph {
    names: Vec<String>,
    red: Vec<bool>,
    children: Vec<Vec<usize>>,
    roots: Vec<usize>,
}

impl ModelGraph {
    fn node(&mut self, name: &str, red: bool) -> usize {
        match self.names.iter().position(|n| n == name) {
            Some(ix) => ix,
            None => {
                self.names.push(name.to_string());
                self.red.push(red);
                self.children.push(Vec::new());
                self.names.len() - 1
            }
        }
    }

    /// The nodes which the roots reach without going through the nodes of `cut` (a bit set), or
    /// `None` if a red node is reached.
    fn reached(&self, cut: u32) -> Option<u32> {
        let mut seen = 0u32;
        let mut stack: Vec<usize> = self
            .roots
            .iter()
            .copied()
            .filter(|&r| cut & (1 << r) == 0)
            .collect();
        while let Some(i) = stack.pop() {
            if seen & (1 << i) != 0 {
                continue;
            }
            seen |= 1 << i;
            if self.red[i] {
                return None;
            }
            stack.extend(
                self.children[i]
                    .iter()
                    .copied()
                    .filter(|&c| cut & (1 << c) == 0),
            );
        }
        Some(seen)
    }
}

#[test]
fn test_reduce_roots_is_the_minimum_cut_closest_to_the_sink() {
    // Compared with a search of all the sets of nodes: the Leaker leaks a smallest set which cuts
    // every path from a root to a red node, and among them the one whose source side is largest
    // (the closest to the sink).
    let mut rng = 12345u64;
    let mut checked = 0;
    for _ in 0..400 {
        let fields: Vec<Model> = (0..1 + (rng >> 20) % 4)
            .map(|_| Model::random(&mut rng, 0))
            .collect();
        let mut graph = ModelGraph::default();
        for field in &fields {
            field.add(None, &mut graph);
        }
        let n = graph.names.len();
        if n > 14 {
            continue;
        }
        let mut best: Option<(u32, u32)> = None;
        for cut in 0u32..1 << n {
            if let Some(reached) = graph.reached(cut) {
                let size = cut.count_ones();
                let better = match best {
                    None => true,
                    Some((best_cut, best_reached)) => {
                        size < best_cut.count_ones()
                            || (size == best_cut.count_ones()
                                && reached.count_ones() > best_reached.count_ones())
                    }
                };
                if better {
                    best = Some((cut, reached));
                }
            }
        }
        let (cut, _) = best.unwrap();
        let mut expected: Vec<String> = (0..n)
            .filter(|&i| cut & (1 << i) != 0)
            .map(|i| graph.names[i].clone())
            .collect();
        expected.sort();
        let fields_tokens = fields.iter().enumerate().map(|(i, f)| {
            let name = Ident::new(&format!("f{}", i), proc_macro2::Span::call_site());
            let ty = f.ty();
            quote!(#name: #ty)
        });
        let item: ItemStruct =
            syn::parse2(quote! { struct S<'a, T> { #(#fields_tokens),* } }).unwrap();
        let mut got = leaked_of(item.clone());
        got.sort();
        assert_eq!(got, expected, "{}", quote!(#item));
        checked += 1;
    }
    assert!(checked > 100, "{}", checked);
}

/// The optional entries (the types of external crates, see `Referrer::must()`) are a minimum cut
/// of their own, and the required entries are cut for free in it.
#[test]
fn test_reduce_roots_optional_entries() {
    let optional = |item: ItemTrait| -> Vec<String> {
        optional_trait(&item)
            .into_iter()
            .map(|(ty, _)| ty)
            .collect()
    };
    let required = |item: ItemTrait| -> Vec<String> {
        leaked_trait(&item).into_iter().map(|(ty, _)| ty).collect()
    };
    // One entry covers a type which is used twice with everything in it, as a cut of the graph.
    let item: ItemTrait = parse_quote! {
        trait Tr {
            fn f(&self) -> (::a::A, ::b::B, ::c::C);
            fn g(&self, x: (::a::A, ::b::B, ::c::C), y: ::d::D);
        }
    };
    assert_eq!(
        optional(item.clone()),
        ["(:: a :: A , :: b :: B , :: c :: C)", ":: d :: D"]
    );
    assert!(required(item).is_empty());
    // The inner types are preferred on a tie, as for the required ones.
    let item: ItemTrait = parse_quote! {
        trait Tr {
            fn f(&self) -> ((Rel,), ::a::A);
            fn g(&self) -> (::a::A, (Rel,), ::b::B);
        }
    };
    assert_eq!(optional(item.clone()), [":: a :: A", ":: b :: B"]);
    assert_eq!(required(item), ["Rel"]);
    // The required entries do not depend on the optional ones: the type of an external crate
    // which has a required type in it (`Rel`, and the tuple, which one cut covers) is optional
    // as a whole, and the required types in it are entries of their own.
    let item: ItemTrait = parse_quote! {
        trait Tr {
            fn f(&self) -> ::core::option::Option<(Rel, ::a::A, ::b::B)>;
            fn g(&self) -> ::core::option::Option<(Rel, ::a::A, ::b::B)>;
            fn h(&self) -> ::a::W<(Rel,), ::b::X>;
            fn i(&self) -> ::b::X;
        }
    };
    assert_eq!(
        optional(item.clone()),
        [
            "(Rel , :: a :: A , :: b :: B)",
            ":: a :: W < (Rel ,) , :: b :: X >",
            ":: b :: X",
        ]
    );
    assert_eq!(required(item), ["Rel"]);
    // A required entry which is cut covers the optional types in it: they are not entries.
    let item: ItemTrait = parse_quote! {
        trait Tr {
            fn f(&self) -> Rel<::a::A, ::b::B>;
            fn g(&self) -> Rel<::a::A, ::b::B>;
        }
    };
    assert!(optional(item.clone()).is_empty());
    assert_eq!(required(item), ["Rel < :: a :: A , :: b :: B >"]);
    // A reference with a lifetime is leaked as a whole around the type of an external crate, as it
    // is around a required one (the copy needs `Type: 'a` of the type which is interned).
    let item: ItemTrait = parse_quote! {
        trait Tr<'a> {
            fn f(&self, x: &'a ::a::A);
            fn g(&self, x: &'a (::a::A, ::b::B));
        }
    };
    assert_eq!(
        optional_trait(&item),
        [
            (
                "& 'a (:: a :: A , :: b :: B)".to_string(),
                "< 'a >".to_string()
            ),
            ("& 'a :: a :: A".to_string(), "< 'a >".to_string()),
        ]
    );
    // And the optional ones of the call come after the required ones, whatever the contexts are.
    let item: ItemTrait = parse_quote! {
        trait Tr {
            fn a(&self) -> ::z::Z;
            fn f<U>(&self, u: ::foo::V<U>) -> Rel<U>;
            fn g(&self) -> Rel2;
        }
    };
    let mut leaker = Leaker::new();
    leaker.intern_trait(&item, true).unwrap();
    let referrer = leaker.finish();
    let musts: Vec<bool> = (0..referrer.iter().count())
        .map(|ix| referrer.must(ix))
        .collect();
    assert_eq!(musts, [true, true, false, false]);
}

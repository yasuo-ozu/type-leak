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
    // A relative type three levels deep: only the inner type is kept; neither the intermediate
    // Vec wrapper nor the tuples appear as a separate entry. (A bare `Vec<RelInner>` would be
    // interned as a whole: see `test_reduce_roots_bare_identifier_argument`.)
    let item: ItemStruct = parse_quote! {
        struct S { a: (::std::vec::Vec<(RelInner,)>, ::std::string::String) }
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
            a: ::std::vec::Vec<(Rel,)>,
            b: (Rel, ::std::string::String),
            c: &'static Rel,
        }
    };
    let referrer = Leaker::from_struct(&item).unwrap().0.finish();
    // `&'static Rel` is not rebuilt around `Rel` (`Rel: 'static` is not known of the entry).
    assert_eq!(referrer.iter().count(), 2);
    let mut copy: ItemStruct = parse_quote! {
        struct S {
            a: ::std::vec::Vec<(Rel,)>,
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
            a: ::std::vec::Vec<(P<#rel_ix>,)>,
            b: (P<#rel_ix>, ::std::string::String),
        }
    })
    .unwrap();
    assert_eq!(quote!(#copy).to_string(), quote!(#expected).to_string());
}

/// The leaked types of a trait, with the generics each uses.
fn leaked_trait(item: &ItemTrait) -> Vec<(String, String)> {
    let (leaker, _) = Leaker::from_trait(item).unwrap();
    let referrer = leaker.finish();
    referrer
        .iter_with_generics()
        .map(|(ty, generics)| (ty_string(ty.clone()), quote!(#generics).to_string()))
        .collect()
}

/// The error of the analysis of a trait.
fn error_of(item: &ItemTrait) -> String {
    let err = Leaker::from_trait(item).err().expect("expected an error");
    syn::Error::from(err).to_string()
}

/// Rewrites a trait with each leaked type replaced by `P<index>`, like the Referrer's copy.
fn rewritten(item: &ItemTrait) -> String {
    let (leaker, _) = Leaker::from_trait(item).unwrap();
    let referrer = leaker.finish();
    let mut copy = item.clone();
    syn::visit_mut::VisitMut::visit_item_trait_mut(
        &mut referrer.into_visitor(None, |_, ix| parse_quote!(P<#ix>)),
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
    // (`dyn RelTr + 'static`) does not have, so the whole path is interned.
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
        // Without a cause of its own, its components are not leaked on their own either.
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

    // `check()` agrees: the path must be interned as a whole.
    let result = Leaker::new().check(&parse_quote!(::p::W<'a, dyn RelTr>), &parse_quote!(<'a>));
    assert!(result.is_own);
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

#[test]
fn test_reduce_roots_trait_object_inheriting_a_local_lifetime() {
    // Like the referent of a reference (`&'b dyn RelTr`): the lifetime of the path is local or
    // elided, so the path cannot be interned as a whole, and the trait object cannot alone.
    let note =
        "the lifetime of the trait object defaults to that of the enclosing reference or path";
    for item in [
        parse_quote! { trait Tr { fn f<'b>(&self, x: ::p::W<'b, dyn RelTr>); } },
        parse_quote! { trait Tr { fn f<'b>(&self, x: ::p::W<'b, (dyn RelTr)>); } },
        parse_quote! { trait Tr { fn f(&self, x: ::p::W<'_, dyn RelTr>); } },
        parse_quote! { trait Tr<'a> { fn f<'b>(&self, x: ::p::W<'a, 'b, dyn RelTr>); } },
        // A trait bound is not a type, so it has nothing to be interned as a whole.
        parse_quote! { trait Tr<'a, T: ::p::Bound<'a, dyn RelTr>> {} },
        parse_quote! { trait Tr<'a>: ::p::Bound<'a, dyn RelTr> {} },
    ] {
        let item: ItemTrait = item;
        let err = error_of(&item);
        assert!(err.contains("use absolute path: 'RelTr'"), "{}", err);
        assert!(err.contains(note), "{}", err);
    }
    // The same as for a reference.
    let err = error_of(&parse_quote! { trait Tr { fn f<'b>(&'b self, x: &'b dyn RelTr); } });
    assert!(err.contains(note), "{}", err);
    // With nothing of the type context in the trait object, there is no error.
    for item in [
        parse_quote! { trait Tr { fn f<'b>(&self, x: ::p::W<'b, dyn ::core::any::Any>); } },
        parse_quote! { trait Tr<'a, T: ::p::Bound<'a, dyn ::core::any::Any>> {} },
        parse_quote! { trait Tr<'a, T: ::p::Bound<'static, dyn RelTr>> {} },
    ] {
        let item: ItemTrait = item;
        let _ = Leaker::from_trait(&item).unwrap();
    }
    // Its components can still be leaked, as for a reference.
    let leaked = leaked_trait(&parse_quote! {
        trait Tr { fn f<'b>(&self, x: ::p::W<'b, dyn ::p::Tr<Rel>>); }
    });
    assert_eq!(leaked, [entry(parse_quote!(Rel), "")]);
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
    let leaked = |self_ty_can_be_interned: bool| {
        let mut leaker = Leaker::new();
        leaker.self_ty_can_be_interned = self_ty_can_be_interned;
        leaker
            .intern_with(&item.generics, |v| v.visit_item_trait(&item))
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
    let referrer = Leaker::from_struct(&item).unwrap().0.finish();
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
    let referrer = Leaker::from_struct(&item).unwrap().0.finish();
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

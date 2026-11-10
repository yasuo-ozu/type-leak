//! Behavioral tests for `Leaker::reduce_roots()`.

use crate::common;

use common::{check_in, finish_as_strings, s, to_syn_error};
use syn::*;
use template_quote::quote;
use type_leak::{ContextId, Leaker, Referrer, Rules};

/// The entries that `reduce_roots()` leaves for a small item, required ones as they are and
/// optional ones after `? `, sorted. `quiet` writes the paths of external crates as they are, so
/// no optional entry changes the cut (`test_reduce_roots_one_cut` tests both).
#[test]
fn test_reduce_roots_small_cuts() {
    // A fresh Leaker with nothing interned produces an empty Referrer.
    let mut leaker = Leaker::new();
    leaker.reduce_roots();
    assert!(leaker.result().finish().unwrap().is_empty());

    fn entries<A: type_leak::Allow>(mut leaker: Leaker<A>, item: &Item) -> Vec<String> {
        common::intern_item(&mut leaker, item);
        leaker.reduce_roots();
        let referrer = leaker.result().finish().unwrap();
        let mut entries: Vec<String> = common::type_texts(&referrer);
        entries.sort();
        entries
    }
    let cases: Vec<(&str, bool, Item, Vec<&str>)> = vec![
        (
            // All fields are the struct's own generic params: all neutral.
            "neutral generics",
            false,
            parse_quote! { struct S<T, U> { a: T, b: U } },
            vec![],
        ),
        (
            // A single relative-path field is a must-intern root and must be kept.
            "single must-intern",
            false,
            parse_quote! { struct S { a: RelType } },
            vec!["RelType"],
        ),
        (
            "two independent must-intern roots both survive",
            false,
            parse_quote! { struct S { a: TypeA, b: TypeB } },
            vec!["TypeA", "TypeB"],
        ),
        (
            // A neutral root whose children are all neutral is dropped entirely, but for the type
            // of an external crate in it, an optional entry: the inner one, as it is the cut
            // closest to the sink.
            "neutral root is dropped, the external type in it stays optional",
            false,
            parse_quote! {
                struct S<T> { a: (T, ::std::string::String), b: (T, ::core::primitive::u8) }
            },
            vec![":: std :: string :: String"],
        ),
        (
            "neutral root is dropped",
            false,
            parse_quote! { struct S<T> { a: (T, ::core::string::String) } },
            vec![],
        ),
        (
            // A must-intern type nested inside a compound wrapper: the inner type is interned (it
            // is the innermost minimum cut), and the wrapper is rebuilt around it.
            "root covers descendant",
            true,
            parse_quote! { struct S { a: (RelType, ::std::string::String) } },
            vec!["RelType"],
        ),
        (
            // Generic field (neutral root) alongside a relative-path field (must-intern root).
            "mixed neutral and must-intern",
            false,
            parse_quote! { struct S<T> { a: T, b: RelType } },
            vec!["RelType"],
        ),
        (
            // A relative type three levels deep: only the inner type is kept; neither the
            // intermediate Vec wrapper nor the tuples appear as a separate entry. (A bare
            // `Vec<RelInner>` would be interned as a whole: see
            // `test_reduce_roots_bare_identifier_argument`.)
            "deep nesting",
            true,
            parse_quote! {
                struct S { a: (::std::vec::Vec<(RelInner,)>, ::std::string::String) }
            },
            vec!["RelInner"],
        ),
        (
            // A type appears as a direct must-intern root (field a) AND as a must-intern
            // descendant inside a compound root (field b). Interning it covers both.
            "type in both positions",
            false,
            parse_quote! {
                struct S {
                    a: RelType,
                    b: (RelType, ::std::string::String),
                }
            },
            vec![":: std :: string :: String", "RelType"],
        ),
        (
            // Four wrappers of the same type: interning it takes one implementation, not four.
            // (The type is not a bare identifier as an argument, which may be a const, see
            // `test_reduce_roots_bare_identifier_argument`.)
            "wrappers of one type",
            true,
            parse_quote! {
                struct S {
                    a: ::std::vec::Vec<(Rel,)>,
                    b: ::core::option::Option<(Rel,)>,
                    c: (Rel, ::std::string::String),
                    d: ::std::boxed::Box<(Rel,)>,
                }
            },
            vec!["Rel"],
        ),
        (
            "wrappers of one type and the bare type",
            true,
            parse_quote! {
                struct S {
                    a: ::std::vec::Vec<(Rel,)>,
                    b: ::core::option::Option<(Rel,)>,
                    c: (Rel, ::std::string::String),
                    d: ::std::boxed::Box<(Rel,)>,
                    e: Rel,
                }
            },
            vec!["Rel"],
        ),
        (
            // The same tuple of three types twice: the tuple (1) is cheaper than its parts (3).
            "common ancestor",
            true,
            parse_quote! {
                struct S {
                    a: (RelA, RelB, RelC),
                    b: ::std::vec::Vec<(RelA, RelB, RelC)>,
                }
            },
            vec!["(RelA , RelB , RelC)"],
        ),
        (
            // Two tuples sharing a part: the two tuples (2) are cheaper than the three parts (3).
            "two tuples sharing a part",
            true,
            parse_quote! { struct S { a: (RelA, RelB), b: (RelA, RelC) } },
            vec!["(RelA , RelB)", "(RelA , RelC)"],
        ),
        (
            // `Vec<(Rel,)>`, `(Rel,)` and `Rel` cover the only path equally; the inner one is
            // taken.
            "the inner type is preferred on a tie",
            true,
            parse_quote! { struct S { a: ::std::vec::Vec<(Rel,)> } },
            vec!["Rel"],
        ),
        (
            // A red root is interned itself, and the type containing it is cheaper than its other
            // components.
            "a red root and a tuple that holds it",
            true,
            parse_quote! { struct S { a: Rel, b: (Rel, RelB, RelC) } },
            vec!["(Rel , RelB , RelC)", "Rel"],
        ),
        (
            // Self::AssocType in a trait method return position is neutral (concrete to each
            // implementor).
            "Self::Assoc is neutral",
            false,
            parse_quote! {
                trait MyTrait {
                    type Assoc;
                    fn method() -> Self::Assoc;
                }
            },
            vec![],
        ),
        (
            // S::Item where S is a method-level generic is neutral: the caller provides S
            // directly so its associated types are also concrete.
            "S::Item of a method generic is neutral",
            false,
            parse_quote! {
                trait MyTrait {
                    fn method<S: ::core::iter::Iterator>() -> S::Item;
                }
            },
            vec![],
        ),
    ];
    for (name, is_quiet, item, expected) in &cases {
        let got = if *is_quiet {
            entries(quiet(), item)
        } else {
            entries(Leaker::new(), item)
        };
        assert_eq!(got, *expected, "{}: {}", name, quote!(#item));
    }
}

/// A Leaker that writes the paths of external crates as they are, so that the tests of the cut
/// among the required types see no optional entries (`test_reduce_roots_one_cut` tests both).
fn quiet() -> Leaker<Rules> {
    Leaker::with_rule(Rules::new().external_crates())
}

fn leaked_of(item: ItemStruct) -> Vec<String> {
    let mut leaker = quiet();
    leaker.intern_struct(&item);
    leaker.reduce_roots();
    finish_as_strings(leaker).into_iter().collect()
}

#[test]
fn test_reduce_roots_inner_entry_is_rewritten_in_the_copy() {
    // The wrappers are not leaked, but a copy of them is rewritten around the inner entry.
    let item: ItemStruct = parse_quote! {
        struct S {
            a: ::core::option::Option<(Rel,)>,
            b: (Rel, ::core::primitive::bool),
            c: &'static Rel,
            d: ::std::string::String,
        }
    };
    let mut leaker = Leaker::new();
    let context = leaker.intern_struct(&item);
    let referrer = leaker.result().finish().unwrap();
    // `&'static Rel` is not rebuilt around `Rel` (`Rel: 'static` is not known of the entry). The
    // type of an external crate is an entry (old: none), after the others.
    let strings: Vec<String> = common::type_texts(&referrer);
    assert_eq!(
        strings,
        ["& 'static Rel", "Rel", ":: std :: string :: String"]
    );
    let copy: ItemStruct = parse_quote! {
        struct S {
            a: ::core::option::Option<(Rel,)>,
            b: (Rel, ::core::primitive::bool),
            d: ::std::string::String,
        }
    };
    let rel_ix = strings.iter().position(|ty| ty == "Rel").unwrap();
    let string_ix = strings.iter().position(|ty| ty.contains("String")).unwrap();
    let mut replaced = copy.clone();
    // A macro replaces the `String` as well.
    syn::visit_mut::VisitMut::visit_item_struct_mut(
        &mut common::visitor(&referrer, context, common::placeholder),
        &mut replaced,
    );
    let expected: ItemStruct = syn::parse2(quote! {
        struct S {
            a: ::core::option::Option<(P<#rel_ix>,)>,
            b: (P<#rel_ix>, ::core::primitive::bool),
            d: P<#string_ix>,
        }
    })
    .unwrap();
    assert_eq!(quote!(#replaced).to_string(), quote!(#expected).to_string());
}

/// The leaked types of a trait, with the generics each uses.
fn leaked_trait(item: &ItemTrait) -> Vec<(String, String)> {
    let mut leaker = quiet();
    leaker.intern_trait(item);
    let referrer = leaker.result().finish().unwrap();
    common::entries_with_generics(&referrer)
}

/// The leaked types of a trait, with the generics each uses, from a Leaker that leaks the types of
/// external crates (`leaked_trait()` has none).
fn entries_trait(item: &ItemTrait) -> Vec<(String, String)> {
    let mut leaker = Leaker::new();
    leaker.intern_trait(item);
    let referrer = leaker.result().finish().unwrap();
    common::entries_with_generics(&referrer)
}

/// The error of the analysis of a trait.
fn error_of(item: &ItemTrait) -> String {
    let mut leaker = Leaker::new();
    leaker.intern_trait(item);
    let result = leaker.result();
    assert!(!result.errors().is_empty(), "expected an error");
    to_syn_error(result.errors()).to_string()
}

/// Rewrites a trait with each leaked type replaced by `P<index>`, like the Referrer's copy. The
/// types of other crates stay as written.
fn rewritten(item: &ItemTrait) -> String {
    let mut leaker = quiet();
    let context = leaker.intern_trait(item);
    let referrer = leaker.result().finish().unwrap();
    let mut copy = item.clone();
    syn::visit_mut::VisitMut::visit_item_trait_mut(
        &mut common::visitor(&referrer, context, common::placeholder),
        &mut copy,
    );
    quote!(#copy).to_string()
}

/// `item` as the Referrer's copy prints it.
fn printed(item: ItemTrait) -> String {
    quote!(#item).to_string()
}

fn entry(ty: Type, generics: &str) -> (String, String) {
    (s(&ty), generics.to_string())
}

fn trait_with(ty: &Type) -> ItemTrait {
    parse_quote! { trait Tr<'a> { fn f(&self, x: #ty); } }
}

#[test]
fn test_reduce_roots_trait_object_inheriting_the_lifetime_of_its_path() {
    // `Ref<'a, T: 'a>` makes the lifetime bound of `dyn RelTr` `'a`, which `dyn RelTr` alone
    // (`dyn RelTr + 'static`) does not have, so the trait object cannot be interned alone. It has
    // to be (it names a relative trait), so the whole path is interned.
    let rel_object = group(parse_quote!(dyn RelTr));
    let tr_object = group(parse_quote!(dyn ::p::Tr<Rel>));
    for (ty, generics) in [
        (parse_quote!(::core::cell::Ref<'a, dyn RelTr>), "< 'a >"),
        (parse_quote!(::core::cell::Ref<'a, (dyn RelTr)>), "< 'a >"),
        (parse_quote!(::core::cell::Ref<'a, #rel_object>), "< 'a >"),
        (parse_quote!(::p::W<'a, dyn RelTr<Rel>>), "< 'a >"),
        (parse_quote!(::p::W<'a, (), dyn RelTr>), "< 'a >"),
        // `Rel` may be a const (a bare identifier as an argument), which cannot be interned
        // alone, so the type containing it has to be interned: the trait object, and so the path.
        (parse_quote!(::p::W<'a, dyn ::p::Tr<Rel>>), "< 'a >"),
        (parse_quote!(::p::W<'a, (dyn ::p::Tr<Rel>)>), "< 'a >"),
        (parse_quote!(::p::W<'a, #tr_object>), "< 'a >"),
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
            s(&ty)
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
    let result = check_in(
        &mut Leaker::new(),
        &parse_quote!(::p::W<'a, dyn RelTr>),
        &parse_quote!(<'a>),
    );
    assert!(result.is_own);
    assert_eq!(result.must.len(), 1);
    // Unless only the components of the trait object have to be interned.
    let result = check_in(
        &mut Leaker::new(),
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
    // The type, what is leaked, and its generics.
    let components: [(&str, &str, &str); 13] = [
        ("&'a dyn ::p::Tr<Item = Rel>", "Rel", ""),
        ("&'a (dyn ::p::Tr<Item = Rel>)", "Rel", ""),
        ("&'a ((dyn ::p::Tr<Item = Rel>))", "Rel", ""),
        ("&'a mut dyn ::p::Tr<Item = Rel>", "Rel", ""),
        // The lifetime bound is written, or the lifetime is `'static`.
        ("&'a (dyn ::p::Tr<Item = Rel> + 'static)", "Rel", ""),
        ("&'a (dyn ::p::Tr<Item = Rel> + 'a)", "Rel", ""),
        ("&'static dyn ::p::Tr<Item = Rel>", "Rel", ""),
        ("::p::V<&'a dyn ::p::Tr<Item = Rel>>", "Rel", ""),
        // Not a bare identifier, which may be a const.
        ("&'a dyn ::p::Tr<Item = Option<Rel>>", "Option<Rel>", ""),
        ("&'a dyn ::p::Tr<Option<Rel>>", "Option<Rel>", ""),
        ("&'a dyn ::p::Tr<(Rel, u8)>", "(Rel, u8)", ""),
        // A reference inside is the type to intern if it has to be (`Rel: 'a`).
        ("&'a dyn ::p::Tr<Item = &'a Rel>", "&'a Rel", "< 'a >"),
        (
            "&'a dyn ::p::Tr<Item = &'a dyn ::p::Tr<Item = Rel>>",
            "Rel",
            "",
        ),
    ];
    for (ty, leaked, generics) in components {
        let (ty, leaked): (Type, Type) = (parse_str(ty).unwrap(), parse_str(leaked).unwrap());
        assert_eq!(
            leaked_trait(&trait_with(&ty)),
            [entry(leaked, generics)],
            "{}",
            s(&ty)
        );
    }
    // In a group, as `$t:ty` of a `macro_rules!` is.
    let object = group(parse_quote!(dyn ::p::Tr<Item = Rel>));
    assert_eq!(
        leaked_trait(&trait_with(&parse_quote!(&'a #object))),
        [entry(parse_quote!(Rel), "")]
    );
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
    let tr_object = group(parse_quote!(dyn ::p::Tr<Rel>));
    let wholes: Vec<Type> = vec![
        parse_quote!(&'a dyn RelTr),
        parse_quote!(&'a (dyn RelTr)),
        parse_quote!(&'a mut dyn RelTr),
        parse_quote!(&'a dyn RelTr<Item = Rel>),
        parse_quote!(&'a #rel_object),
        parse_quote!(&'a dyn ::p::Tr<Rel>),
        parse_quote!(&'a (dyn ::p::Tr<Rel>)),
        parse_quote!(&'a #tr_object),
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
            s(&ty)
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
            s(&ty)
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
            s(&ty)
        );
    }

    // `check()` agrees.
    let result = check_in(
        &mut Leaker::new(),
        &parse_quote!(&'a dyn RelTr),
        &parse_quote!(<'a>),
    );
    assert!(result.is_own);
    assert_eq!(result.must.len(), 1);
    let result = check_in(
        &mut Leaker::new(),
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
        assert_eq!(leaked.len(), 1, "{}", s(&ty));
        assert!(leaked[0].0.starts_with("dyn RelTr"), "{}", leaked[0].0);
        assert_eq!(leaked[0].1.replace(' ', ""), generics.replace(' ', ""));
    }
}

/// The `Leaker` docs, Trait objects: "a lifetime bound you write changes this ... with a local
/// lifetime, the trait object alone is leaked with its bound". The reference around it has an
/// elided lifetime (`&(dyn RelTr + 'a)`), which is no obstacle once the object names its own bound,
/// while `&dyn RelTr` is an error. Audit mutant `t_dyn_elided_bounded_error` (a trait object under
/// an elided reference is an error even with a bound).
#[test]
fn test_reduce_roots_trait_object_with_a_bound_under_an_elided_reference() {
    // A lifetime of the trait, of the method, `'static`, and a mutable reference.
    for (item, expected) in [
        (
            parse_quote! { trait Tr<'a> { fn f(&self, x: &(dyn RelTr + 'a)); } },
            entry(parse_quote!(dyn RelTr + 'a), "< 'a >"),
        ),
        (
            parse_quote! { trait Tr { fn f<'b>(&self, x: &(dyn RelTr + 'b)); } },
            entry(parse_quote!(dyn RelTr + 'b), "< 'b >"),
        ),
        (
            parse_quote! { trait Tr<'a> { fn f(&self, x: &mut (dyn RelTr + 'a)); } },
            entry(parse_quote!(dyn RelTr + 'a), "< 'a >"),
        ),
        (
            parse_quote! { trait Tr { fn f(&self, x: &(dyn RelTr + 'static)); } },
            entry(parse_quote!(dyn RelTr + 'static), ""),
        ),
    ] {
        let item: ItemTrait = item;
        assert_eq!(leaked_trait(&item), [expected], "{}", s(&item));
    }
    // Without the bound the object takes the lifetime of the elided reference, which the Leaker
    // cannot name: an error.
    let err = error_of(&parse_quote! { trait Tr { fn f(&self, x: &dyn RelTr); } });
    assert!(err.contains("use absolute path: 'RelTr'"), "{}", err);
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
            s(&ty)
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
        let mut leaker = Leaker::new();
        leaker.intern_trait(&item);
        assert!(leaker.result().errors().is_empty());
    }
    // Its components can still be leaked, as for a reference.
    let leaked = leaked_trait(&parse_quote! {
        trait Tr { fn f<'b>(&self, x: ::p::W<'b, dyn ::p::Tr<Item = Rel>>); }
    });
    assert_eq!(leaked, [entry(parse_quote!(Rel), "")]);
}

/// Asserts that `item` is an error because `RelTr` leaves the lifetime `'{lifetime}` of the method
/// unconstrained in the copy (E0195): the message, and the notes that say why.
fn assert_unconstrained(item: &ItemTrait, lifetime: &str) {
    let err = error_of(item);
    for part in [
        format!(
            "`RelTr` leaves the lifetime `'{}` of the method unconstrained in the copy (E0195)",
            lifetime
        ),
        format!(
            "the lifetime of the trait object defaults to `'{}`, a lifetime of the method which \
             its return type mentions",
            lifetime
        ),
        format!("it cannot be leaked as a whole because of `'{}`", lifetime),
    ] {
        assert!(err.contains(&part), "{}", err);
    }
    // Not the note which suggests to name the lifetime, which is named.
    assert!(!err.contains("name it"), "{}", err);
}

/// Methods whose input `x: {x}` holds a trait object that defaults to `'b`, and where the copy
/// still has `'b` late-bound as the trait has it, or early-bound in both: an input keeps `'b`
/// where it is written (the receiver, a reference, also in a tuple, an array, a slice, a pointer or
/// parentheses), a bound or a where predicate names it, the return type does not mention it, or
/// another lifetime is the one of the return type.
const CONSTRAINED: [&str; 15] = [
    "fn f<'b>(&'b self, x: {x}) -> &'b ();",
    "fn f<'b>(&'b mut self, x: {x}) -> &'b ();",
    "fn f<'b>(self: &'b Self, x: {x}) -> &'b ();",
    "fn f<'b>(&self, x: {x}, y: &'b ()) -> &'b ();",
    "fn f<'b>(&self, x: {x}, y: ((), &'b ())) -> &'b ();",
    "fn f<'b>(&self, x: {x}, y: [&'b (); 2]) -> &'b ();",
    "fn f<'b>(&self, x: {x}, y: &[&'b ()]) -> &'b ();",
    "fn f<'b>(&self, x: {x}, y: *const &'b ()) -> &'b ();",
    "fn f<'b>(&self, x: {x}, y: (&'b ())) -> &'b ();",
    "fn f<'b, 'c: 'b>(&self, x: {x}, y: &'c ()) -> &'b ();",
    "fn f<'b, T: 'b>(&self, x: {x}, y: T) -> &'b ();",
    "fn f<'b>(&self, x: {x}) -> &'b () where Self: 'b;",
    "fn f<'b, T>(&self, x: {x}, t: T) -> &'b () where T: ::p::Bound<'b>;",
    "fn f<'b>(&self, x: {x}) -> ();",
    "fn f<'b, 'c>(&self, x: {x}, y: &'c ()) -> &'c ();",
];

/// Asserts that the trait with each method of [`CONSTRAINED`], `holder` written for `{x}`, leaks
/// `holder` (with the generics that a predicate naming `'b` brings along).
fn assert_constrained(holder: &str) {
    for sig in CONSTRAINED {
        let item: ItemTrait =
            parse_str(&format!("trait Tr {{ {} }}", sig.replace("{x}", holder))).unwrap();
        let leaked = leaked_trait(&item);
        assert_eq!(leaked.len(), 1, "{}", s(&item));
        assert_eq!(
            leaked[0].0,
            s(&parse_str::<Type>(holder).unwrap()),
            "{}",
            s(&item)
        );
        assert!(leaked[0].1.starts_with("< 'b"), "{}", leaked[0].1);
    }
}

#[test]
fn test_reduce_roots_path_with_a_lifetime_the_copy_would_leave_unconstrained() {
    // The copy of the whole path has `'b` only in a projection, which does not constrain it. A
    // lifetime which no bound mentions is late-bound only if an input constrains it or the return
    // type does not mention it, so the copy of a method whose return type mentions `'b` has it
    // early-bound where the trait's is late-bound, which rustc rejects when the method is
    // implemented (E0195). So the path is not interned: an error, like for `'_`.
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
        assert_unconstrained(&item, "b");
    }
    // The components are leaked if it is not the trait object which has to be interned.
    let leaked = leaked_trait(&parse_quote! {
        trait Tr { fn f<'b>(&self, x: ::p::W<'b, dyn ::p::Tr<Item = Rel>>) -> &'b (); }
    });
    assert_eq!(leaked, [entry(parse_quote!(Rel), "")]);
    assert_constrained("::p::W<'b, dyn RelTr>");
    // The lifetime of the trait is not a lifetime of the method: nothing to compare.
    let leaked = leaked_trait(&parse_quote! {
        trait Tr<'a> { fn f(&self, x: ::p::W<'a, dyn RelTr>) -> &'a (); }
    });
    assert_eq!(
        leaked,
        [entry(parse_quote!(::p::W<'a, dyn RelTr>), "< 'a >")]
    );
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
            s(&ty)
        );
        assert_eq!(
            rewritten(&item),
            printed(parse_quote! { trait Tr { fn f<'b, 'c, T>(&self, x: P<0usize>); } }),
            "{}",
            s(&ty)
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
        assert_eq!(leaked_trait(&item), [entry(leaked, generics)], "{}", s(&ty));
        assert_eq!(
            rewritten(&item),
            printed(parse_quote! { trait Tr { fn f<'b>(&self, x: &'b (P<0usize>)); } }),
            "{}",
            s(&ty)
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
    let result = check_in(
        &mut Leaker::new(),
        &parse_quote!(&'b dyn RelTr),
        &parse_quote!(<'b>),
    );
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
        assert_unconstrained(&item, lifetime);
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
    assert_constrained("&'b dyn RelTr");
    // `'c: 'b` makes `'c` early-bound too.
    let leaked = leaked_trait(&parse_quote! {
        trait Tr { fn f<'b, 'c: 'b>(&self, x: &'b &'c dyn RelTr) -> &'c (); }
    });
    assert_eq!(
        leaked,
        [entry(parse_quote!(&'c dyn RelTr), "< 'b , 'c : 'b >")]
    );
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
}

/// The `Leaker` docs, Method lifetimes: an external type that the Leaker does not allow keeps the
/// method lifetime `'b` (the Referrer copies it as written, as it is no optional entry), so the
/// method is not an error (E0195) although `Wrap<'b>`, which is leaked as a whole, names `'b` as
/// well. A reference to a trait object of such a type keeps it too. Audit mutant `m_careful_on`
/// (the careful rule for trait objects applied to the external type).
#[test]
fn test_reduce_roots_external_type_keeping_a_method_lifetime_is_no_e0195() {
    let item: ItemTrait = parse_quote! {
        trait Tr {
            fn f<'b>(&self, a: ::ext::W<'b>, b: Wrap<'b>) -> &'b ::core::primitive::u8;
        }
    };
    assert_eq!(
        leaked_trait(&item),
        [entry(parse_quote!(Wrap<'b>), "< 'b >")]
    );
    assert_eq!(entries_trait(&item), leaked_trait(&item));
    let item: ItemTrait = parse_quote! {
        trait Tr {
            fn f<'b>(&self, a: &'b dyn ::ext::Tr, b: Wrap<'b>) -> &'b ::core::primitive::u8;
        }
    };
    assert_eq!(
        leaked_trait(&item),
        [entry(parse_quote!(Wrap<'b>), "< 'b >")]
    );
    assert_eq!(entries_trait(&item), leaked_trait(&item));
    // Nothing keeps `'b` here, so the method is an error: the control.
    let err = error_of(&parse_quote! {
        trait Tr { fn f<'b>(&self, b: Wrap<'b>) -> &'b ::core::primitive::u8; }
    });
    assert!(err.contains("E0195"), "{}", err);
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
        assert_eq!(leaked.len(), 1, "{}", s(&item));
        assert!(
            leaked[0].0.starts_with("& 'b dyn RelTr")
                || leaked[0].0.starts_with("& 'b mut dyn RelTr"),
            "{}: {:?}",
            s(&item),
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
        assert_eq!(leaked_returning(&ty), [entry(ty.clone(), "")], "{}", s(&ty));
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
        assert_eq!(leaked_returning(&ty), [entry(expected, "")], "{}", s(&ty));
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
        let expected: Type = match s(&ty).starts_with(":: std") {
            true => parse_quote!(::p::W<#size>),
            false => ty.clone(),
        };
        assert_eq!(leaked_returning(&ty), [entry(expected, "")], "{}", s(&ty));
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
    let result = check_in(
        &mut quiet(),
        &parse_quote!(::p::W<SIZE>),
        &Generics::default(),
    );
    assert!(result.is_own);
    assert!(result.impossible.is_empty());
    assert_eq!(result.must.len(), 1);
}

#[test]
fn test_reduce_roots_bare_identifier_is_a_type_if_the_path_cannot_be_interned() {
    // The path cannot be interned, so the identifier is leaked alone, as a type, like any other
    // (and not as an error): for an elided lifetime or reference, `impl Trait`, and a trait bound
    // (`soft_causes_leak_the_identifier` in tests/leaker/bare_const.rs). For any other cause
    // (`Self`, a method lifetime that the copy would leave unconstrained, ..) it is an error.
    // The same as for any identifier.
    assert_eq!(
        leaked_returning(&parse_quote!(::p::W<'_, Rel>)),
        [entry(parse_quote!(Rel), "")]
    );
    assert_eq!(
        leaked_trait(&parse_quote! { trait Tr { fn f<T: ::p::Sup<Size>>(&self); } }),
        [entry(parse_quote!(Size), "")]
    );
    // A group is transparent.
    let size = group(parse_quote!(SIZE));
    assert_eq!(
        leaked_returning(&parse_quote!(::p::W<'_, #size>)),
        [entry(parse_quote!(SIZE), "")]
    );
    // `check()` agrees: the identifier is the cause, not the path.
    let result = check_in(
        &mut Leaker::new(),
        &parse_quote!(::p::W<'_, SIZE>),
        &Generics::default(),
    );
    assert!(!result.is_own);
    assert_eq!(result.impossible.len(), 1);
    assert_eq!(result.must.len(), 1);
    assert_eq!(quote!(#{&result.must[0]}).to_string(), "SIZE");

    // A bound of the generics that mentions `Self` does not stop the path from being leaked whole.
    let item: ItemTrait = parse_quote! {
        trait Tr<T: ::p::Bound<Self>> { fn c(&self, x: ::p::W<T, SIZE>); }
    };
    let mut leaker = Leaker::new();
    common::intern_fns(
        &mut leaker,
        &item.generics,
        quote!(
            fn c(&self, x: ::p::W<T, SIZE>);
        ),
    );
    let referrer = leaker.result().finish().unwrap();
    assert_eq!(common::type_texts(&referrer), [":: p :: W < T , SIZE >"]);
}

#[test]
fn test_reduce_roots_bare_identifier_in_other_positions_is_a_type() {
    // Only an identifier written directly as a generic argument may be a const. In the other
    // positions it is a type, which is leaked alone (also if the type around it is not interned).
    // (A tuple, an array, a function pointer, parentheses, a binding and the qualified self type:
    // `unchanged_cases` in tests/leaker/bare_const.rs.)
    let paren_size = group(parse_quote!((SIZE)));
    for ty in [
        parse_quote!(::p::W<((SIZE))>),
        parse_quote!(::p::W<#paren_size>),
        parse_quote!(::p::W<[SIZE]>),
        parse_quote!(::p::W<fn() -> SIZE>),
        // The arguments of `Fn(..) -> ..`.
        parse_quote!(::p::W<dyn ::core::ops::Fn(SIZE) -> SIZE>),
        // Not as an argument.
        parse_quote!(SIZE),
        parse_quote!(::std::boxed::Box<(SIZE,)>),
    ] {
        let ty: Type = ty;
        assert_eq!(
            leaked_returning(&ty),
            [entry(parse_quote!(SIZE), "")],
            "{}",
            s(&ty)
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
            s(&ty)
        );
    }
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
        assert_eq!(leaked, [entry(expected, "")], "{}", s(&ty));
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
    let result = check_in(
        &mut quiet(),
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
fn test_reduce_roots_large_item() {
    // Many independent fields, which a flow over the whole network would take quadratic time
    // for (its search is repeated for each of them). This finishes quickly as the components are
    // solved on their own.
    let n = 400;
    let fields = (0..n).map(|i| {
        let name = Ident::new(&format!("f{}", i), proc_macro2::Span::call_site());
        let rel = Ident::new(&format!("Rel{}", i), proc_macro2::Span::call_site());
        quote!(#name: ::core::option::Option<(#rel,)>)
    });
    let item: ItemStruct = syn::parse2(quote! { struct S { #(#fields),* } }).unwrap();
    let mut leaker = Leaker::new();
    leaker.intern_struct(&item);
    let referrer = leaker.result().finish().unwrap();
    assert_eq!(referrer.iter().count(), n);
    // The inner types are leaked.
    assert!(common::types(&referrer).all(|ty| quote!(#ty).to_string().starts_with("Rel")));

    // Types of which the components are shared are solved together, and chosen as before.
    let n = 100;
    let fields = (0..n).map(|i| {
        let name = Ident::new(&format!("f{}", i), proc_macro2::Span::call_site());
        let r = Ident::new(&format!("R{}", i), proc_macro2::Span::call_site());
        let q = Ident::new(&format!("Q{}", i), proc_macro2::Span::call_site());
        quote!(#name: (::std::vec::Vec<(#r,)>, ::core::option::Option<(#q,)>, (#r, #q)))
    });
    let item: ItemStruct = syn::parse2(quote! { struct S { #(#fields),* } }).unwrap();
    let mut leaker = Leaker::new();
    leaker.intern_struct(&item);
    let referrer = leaker.result().finish().unwrap();
    // The tuple (1) is cheaper than `Ri` and `Qi` (2).
    assert_eq!(referrer.iter().count(), n);
    assert!(common::types(&referrer)
        .all(|ty| matches!(ty, Type::Tuple(tuple) if tuple.elems.len() == 3)));
}

/// A random type: `Ra`, `Rb` and `Rc` are red, as is every `Rg<..>`, and so is a reference to a
/// type with a red component (which is interned as a whole). If the Leaker leaks the types of
/// external crates, so are `Vec<..>` and `W<..>`.
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

    fn is_red(&self, external: bool) -> bool {
        match self {
            Model::Leaf(name) => matches!(*name, "Ra" | "Rb" | "Rc"),
            Model::Rg(_) => true,
            Model::Vec(_) | Model::W(..) => external,
            // The copy of a reference needs its referent to outlive the lifetime.
            Model::Ref(a) => a.has_red(external),
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

    fn has_red(&self, external: bool) -> bool {
        match self {
            Model::Ref(a) => a.has_red(external),
            _ => self.is_red(external) || self.children().iter().any(|c| c.has_red(external)),
        }
    }

    /// Adds the node of the type (if it contains a red type), and those of its components.
    fn add(&self, parent: Option<usize>, graph: &mut ModelGraph, external: bool) {
        if !self.has_red(external) {
            return;
        }
        let name = s(&self.ty());
        let ix = graph.node(&name, self.is_red(external));
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
            child.add(Some(ix), graph, external);
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
    is_the_minimum_cut_closest_to_the_sink(false);
}

#[test]
fn test_reduce_roots_is_the_minimum_cut_with_the_types_of_external_crates() {
    is_the_minimum_cut_closest_to_the_sink(true);
}

/// Compared with a search of all the sets of nodes: the Leaker leaks a smallest set which cuts
/// every path from a root to a red node, and among them the one whose source side is largest (the
/// closest to the sink). `external`: the types of external crates are leaked too (one cut covers
/// the required entries and the optional ones).
fn is_the_minimum_cut_closest_to_the_sink(external: bool) {
    let mut rng = 12345u64;
    let mut checked = 0;
    for _ in 0..200 {
        let fields: Vec<Model> = (0..1 + (rng >> 20) % 4)
            .map(|_| Model::random(&mut rng, 0))
            .collect();
        let mut graph = ModelGraph::default();
        for field in &fields {
            field.add(None, &mut graph, external);
        }
        let n = graph.names.len();
        if n > 12 {
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
        let mut got = if external {
            let mut leaker = Leaker::new();
            leaker.intern_struct(&item);
            let referrer = leaker.result().finish().unwrap();
            common::type_texts(&referrer)
        } else {
            leaked_of(item.clone())
        };
        got.sort();
        assert_eq!(got, expected, "{}", quote!(#item));
        checked += 1;
    }
    assert!(checked > 100, "{}", checked);
}

/// The entries of a trait in order: the types that depend on the context, then the others.
fn all_entries(item: &ItemTrait) -> Vec<String> {
    let mut leaker = Leaker::new();
    leaker.intern_trait(item);
    let referrer = leaker.result().finish().unwrap();
    common::type_texts(&referrer)
}

/// One cut chooses the entries that depend on the context and the types of external crates, and an
/// entry that has a type that depends on the context in it comes first.
#[test]
fn test_reduce_roots_one_cut() {
    // (A type of a crate with a required type in it is one required entry, replaced as a whole, and
    // the type of a crate beside a required one or in a tuple with it:
    // tests/leaker/external_crates.rs.) The type of the crate is an entry also if `Rel` is an entry
    // of its own.
    let item: ItemTrait = parse_quote! {
        trait Tr { fn f(&self) -> ::ext::W<Rel>; fn g(&self) -> Rel; }
    };
    assert_eq!(all_entries(&item), [":: ext :: W < Rel >", "Rel"]);
    // A type of a crate in a type of a crate is not an entry.
    let item: ItemTrait = parse_quote! {
        trait Tr { fn f(&self) -> ::core::option::Option<::ext::W<::ext2::X>>; }
    };
    assert_eq!(all_entries(&item), [":: ext :: W < :: ext2 :: X >"]);
    // Unless it is an entry on its own, too.
    let item: ItemTrait = parse_quote! {
        trait Tr { fn f(&self) -> ::ext::W<::ext2::X>; fn g(&self) -> ::ext2::X; }
    };
    assert_eq!(
        all_entries(&item),
        [":: ext :: W < :: ext2 :: X >", ":: ext2 :: X"]
    );
    // A type that is leaked whole holds the types of the crates in it.
    let item: ItemTrait = parse_quote! {
        trait Tr { fn f(&self) -> Local<::ext::W<::ext2::X>>; }
    };
    assert_eq!(
        all_entries(&item),
        ["Local < :: ext :: W < :: ext2 :: X > >"]
    );
    // The listing has the types that depend on the context first, each part by the order in which
    // the contexts first appear, then the tokens.
    let item: ItemTrait = parse_quote! {
        trait Tr<T> {
            fn f(&self, a: ::ext::W<Rel2<T>>, b: ::ext::X, c: Rel);
            fn g<U>(&self, u: ::ext::V<U>) -> Rel3<U>;
        }
    };
    assert_eq!(
        all_entries(&item),
        [
            ":: ext :: W < Rel2 < T > >",
            "Rel",
            "Rel3 < U >",
            ":: ext :: X",
            ":: ext :: V < U >"
        ]
    );
}

/// A type that cannot be an entry (the copy would make a cycle) is never chosen, whatever it costs
/// to leak its parts, also if one of the kinds of entry would be fine.
#[test]
fn test_reduce_roots_a_forbidden_type_has_its_components_as_entries() {
    let entries = |config: Rules, item: ItemStruct| -> Vec<String> {
        let mut leaker = Leaker::with_rule(config);
        leaker.enter_generics(&Generics::default(), |view| {
            view.visit_item(&Item::Struct(item.clone()))
        });
        let referrer = leaker.result().finish().unwrap();
        common::type_texts(&referrer)
    };
    // `Self` of a nested struct is the copy's own type, which no entry can name, so a type that
    // names it is no entry: not one that depends on the context, whose parts would be two, and
    // not a type of another crate.
    let item: ItemStruct = parse_quote! {
        struct S { a: ::p::Pair<Self, (RelA,), (RelB,)>, b: (::p::A, Self, ::p::B) }
    };
    assert_eq!(
        entries(Rules::new(), item),
        ["RelA", "RelB", ":: p :: A", ":: p :: B"]
    );
    // (The type of an external crate whose context has a bound that names a leaked type: F24.)
}

/// The types of external crates are chosen by the same cut as the others: an entry that covers a
/// type that depends on the context comes first.
#[test]
fn test_reduce_roots_entries_of_external_crates() {
    let entries = |item: ItemTrait| -> Vec<String> {
        entries_trait(&item).into_iter().map(|(ty, _)| ty).collect()
    };
    // One entry covers a type which is used twice with everything in it, as a cut of the graph.
    let item: ItemTrait = parse_quote! {
        trait Tr {
            fn f(&self) -> (::a::A, ::b::B, ::c::C);
            fn g(&self, x: (::a::A, ::b::B, ::c::C), y: ::d::D);
        }
    };
    assert_eq!(
        entries(item),
        ["(:: a :: A , :: b :: B , :: c :: C)", ":: d :: D"]
    );
    // The inner types are preferred on a tie, as for the others.
    let item: ItemTrait = parse_quote! {
        trait Tr {
            fn f(&self) -> ::core::option::Option<(::a::A,)>;
            fn g(&self) -> ::core::option::Option<(::b::B,)>;
        }
    };
    assert_eq!(entries(item), [":: a :: A", ":: b :: B"]);
    // The type of an external crate that has a type that depends on the context in it (`Rel`, and
    // the tuple, which one cut covers) is an entry as a whole, as is the tuple. The `::b::X` in
    // `::a::W<..>` is covered by it, and is an entry for the root `::b::X` only.
    let item: ItemTrait = parse_quote! {
        trait Tr {
            fn f(&self) -> ::core::option::Option<(Rel, ::a::A, ::b::B)>;
            fn g(&self) -> ::core::option::Option<(Rel, ::a::A, ::b::B)>;
            fn h(&self) -> ::a::W<(Rel,), ::b::X>;
            fn i(&self) -> ::b::X;
        }
    };
    assert_eq!(
        entries(item),
        [
            "(Rel , :: a :: A , :: b :: B)",
            ":: a :: W < (Rel ,) , :: b :: X >",
            ":: b :: X",
        ]
    );
    // An entry which is cut covers the types of the crates in it: they are not entries.
    let item: ItemTrait = parse_quote! {
        trait Tr {
            fn f(&self) -> Rel<::a::A, ::b::B>;
            fn g(&self) -> Rel<::a::A, ::b::B>;
        }
    };
    assert_eq!(entries(item), ["Rel < :: a :: A , :: b :: B >"]);
    // A reference with a lifetime is leaked as a whole around the type of an external crate, as it
    // is around another one (the copy needs `Type: 'a` of the type which is interned).
    let item: ItemTrait = parse_quote! {
        trait Tr<'a> {
            fn f(&self, x: &'a ::a::A);
            fn g(&self, x: &'a (::a::A, ::b::B));
        }
    };
    assert_eq!(
        entries_trait(&item),
        [
            (
                "& 'a (:: a :: A , :: b :: B)".to_string(),
                "< 'a >".to_string()
            ),
            ("& 'a :: a :: A".to_string(), "< 'a >".to_string()),
        ]
    );
}

// `reduce_roots()` removes every type that is not an entry from the Leaker, and the kept entries
// stay entries: the tests below.

/// The items for the tests of the pruning.
fn prune_cases() -> Vec<Item> {
    vec![
        // Entries of both kinds. `::ext::Wrap<Rel2>` holds a type that depends on the context, so
        // it comes first, and is replaced as a whole.
        parse_quote! {
            struct S {
                a: Rel,
                b: ::std::string::String,
                c: ::ext::Wrap<Rel2>,
                d: ::ext::X,
            }
        },
        // A tuple of two relative types is required through its children, which one cut covers.
        parse_quote! { struct S { a: (R1, R2) } },
        // A whole reference with a lifetime, and the type in it, in another context.
        parse_quote! { struct S<'a> { a: &'a Rel, b: Rel } },
        parse_quote! {
            trait Tr<T> {
                fn f(&self, t: T) -> Wrap<T>;
                fn g(&self) -> (Rel, ::ext::X);
                fn h(&self) -> (::ext::X, ::ext::Y, Rel);
            }
        },
        parse_quote! {
            enum E {
                A(::ext::Wrap<(R1, R2)>),
                B(::ext::Wrap<(R1, R2)>),
            }
        },
    ]
}

/// The entries of `referrer`, in order: the type and the number of the context.
fn kept(referrer: &Referrer) -> Vec<String> {
    common::types(referrer)
        .enumerate()
        .map(|(ix, ty)| format!("{} {}", quote!(#ty), common::entry_id(referrer, ix).index()))
        .collect()
}

fn referrer_after(item: &Item, reductions: usize) -> Referrer {
    let mut leaker = Leaker::new();
    common::intern_item(&mut leaker, item);
    for _ in 0..reductions {
        leaker.reduce_roots();
    }
    leaker.result().finish().unwrap()
}

/// `reduce_roots()` is idempotent, and calling it changes nothing for `result()`: the Referrer is
/// the same, serialized too. The entries are what one reduction at the end gives.
#[test]
fn test_reduce_roots_changes_nothing_for_result() {
    let expected: [&[&str]; 5] = [
        &[
            ":: ext :: Wrap < Rel2 > 0",
            "Rel 0",
            ":: ext :: X 0",
            ":: std :: string :: String 0",
        ],
        &["(R1 , R2) 0"],
        &["& 'a Rel 0", "Rel 1"],
        &[
            "Wrap < T > 0",
            "(:: ext :: X , :: ext :: Y , Rel) 1",
            "(Rel , :: ext :: X) 1",
        ],
        &[":: ext :: Wrap < (R1 , R2) > 0"],
    ];
    for (item, expected) in prune_cases().iter().zip(expected) {
        let once = referrer_after(item, 1);
        assert_eq!(kept(&once), expected, "{}", quote!(#item));
        for reductions in [0, 2, 3] {
            let other = referrer_after(item, reductions);
            assert_eq!(other, once, "{} after {}", quote!(#item), reductions);
            assert_eq!(quote!(#other).to_string(), quote!(#once).to_string());
        }
    }
}

/// The copy of `item` that the visitor of `referrer` makes in `context`: every entry is `P<ix>`.
fn copy_of(referrer: &Referrer, context: ContextId, item: &ItemStruct) -> ItemStruct {
    let mut copy = item.clone();
    let mut visitor = common::visitor(referrer, context, common::placeholder);
    syn::visit_mut::VisitMut::visit_item_struct_mut(&mut visitor, &mut copy);
    copy
}

/// Items interned after `reduce_roots()` add to what is left. The tuple of the first item stays an
/// entry, required though the relative types that made it so are gone, so the second item, which
/// names the types of the tuple alone, has them as entries too: three entries, where one reduction
/// at the end gives two (`R1` and `R2`, and the tuple is built from them).
#[test]
fn test_reduce_roots_then_interning_adds_to_the_kept_entries() {
    let s1: ItemStruct = parse_quote! { struct S1 { a: (R1, R2) } };
    let s2: ItemStruct = parse_quote! { struct S2 { b: R1, c: R2 } };

    let mut leaker = Leaker::new();
    let c1 = leaker.intern_struct(&s1);
    leaker.reduce_roots();
    let c2 = leaker.intern_struct(&s2);
    let referrer = leaker.result().finish().unwrap();
    assert_eq!(kept(&referrer), ["(R1 , R2) 0", "R1 0", "R2 0"]);
    // The copies of both items have no relative path left.
    let (copy1, copy2): (ItemStruct, ItemStruct) = (
        parse_quote! { struct S1 { a: P<0usize> } },
        parse_quote! { struct S2 { b: P<1usize>, c: P<2usize> } },
    );
    assert_eq!(copy_of(&referrer, c1, &s1), copy1);
    assert_eq!(copy_of(&referrer, c2, &s2), copy2);

    // One reduction at the end covers both items with `R1` and `R2`.
    let mut leaker = Leaker::new();
    let c1 = leaker.intern_struct(&s1);
    let c2 = leaker.intern_struct(&s2);
    let referrer = leaker.result().finish().unwrap();
    assert_eq!(kept(&referrer), ["R1 0", "R2 0"]);
    let (copy1, copy2): (ItemStruct, ItemStruct) = (
        parse_quote! { struct S1 { a: (P<0usize>, P<1usize>) } },
        parse_quote! { struct S2 { b: P<0usize>, c: P<1usize> } },
    );
    assert_eq!(copy_of(&referrer, c1, &s1), copy1);
    assert_eq!(copy_of(&referrer, c2, &s2), copy2);
}

/// The kept entries keep their kind, and the result lists those that depend on the context first,
/// whenever each was kept. A kept entry is not added twice by a later item that names it again.
#[test]
fn test_reduce_roots_kept_entries_keep_their_kind() {
    let s1: ItemStruct = parse_quote! { struct S1 { a: ::std::string::String, b: (R1, R2) } };
    let s2: ItemStruct = parse_quote! { struct S2 { c: Rel, d: ::std::string::String } };
    let s3: ItemStruct = parse_quote! { struct S3 { e: (R1, R2), f: ::ext::X } };
    let mut leaker = Leaker::new();
    leaker.intern_struct(&s1);
    leaker.reduce_roots();
    leaker.intern_struct(&s2);
    leaker.reduce_roots();
    leaker.intern_struct(&s3);
    let referrer = leaker.result().finish().unwrap();
    // The tuple comes first, though what made it so was pruned in the first reduction, and
    // `String` comes after the others, though kept before `Rel`.
    assert_eq!(
        kept(&referrer),
        [
            "(R1 , R2) 0",
            "Rel 0",
            ":: ext :: X 0",
            ":: std :: string :: String 0",
        ]
    );
    // The same entries as one reduction at the end gives.
    let mut leaker = Leaker::new();
    for item in [&s1, &s2, &s3] {
        leaker.intern_struct(item);
    }
    let at_once = leaker.result().finish().unwrap();
    assert_eq!(
        kept(&at_once),
        [
            "(R1 , R2) 0",
            "Rel 0",
            ":: ext :: X 0",
            ":: std :: string :: String 0",
        ]
    );
}

/// A later parent of a kept entry does not replace it: the entry stays one, and the parent is
/// built from it (`((R1, R2), R3)`), or is an entry as well if it is leaked whole (`&'a (R1, R2)`,
/// `::ext::W<(R1, R2)>`, which a Referrer replaces whatever it holds).
#[test]
fn test_reduce_roots_a_later_parent_of_a_kept_entry() {
    let s1: ItemStruct = parse_quote! { struct S1 { a: (R1, R2) } };
    let s2: ItemStruct = parse_quote! {
        struct S2<'a> { b: ((R1, R2), R3), c: &'a (R1, R2), d: ::ext::W<(R1, R2)> }
    };
    let mut leaker = Leaker::new();
    let c1 = leaker.intern_struct(&s1);
    leaker.reduce_roots();
    let c2 = leaker.intern_struct(&s2);
    let referrer = leaker.result().finish().unwrap();
    assert_eq!(
        kept(&referrer),
        [
            "(R1 , R2) 0",
            ":: ext :: W < (R1 , R2) > 0",
            "R3 0",
            "& 'a (R1 , R2) 2",
        ]
    );
    let (copy1, copy2): (ItemStruct, ItemStruct) = (
        parse_quote! { struct S1 { a: P<0usize> } },
        parse_quote! {
            struct S2<'a> { b: (P<0usize>, P<2usize>), c: P<3usize>, d: P<1usize> }
        },
    );
    assert_eq!(copy_of(&referrer, c1, &s1), copy1);
    assert_eq!(copy_of(&referrer, c2, &s2), copy2);
    // Nothing differs but the order, for one reduction at the end.
    let mut leaker = Leaker::new();
    leaker.intern_struct(&s1);
    leaker.intern_struct(&s2);
    assert_eq!(kept(&leaker.result().finish().unwrap()), kept(&referrer));
}

/// The result does not depend on the hash order of the sets that the Leaker keeps.
#[test]
fn test_reduce_roots_interning_after_it_is_deterministic() {
    let run = || -> String {
        let mut leaker = Leaker::new();
        for (ix, item) in prune_cases().iter().enumerate() {
            common::intern_item(&mut leaker, item);
            if ix % 2 == 0 {
                leaker.reduce_roots();
            }
        }
        let referrer = leaker.result().finish().unwrap();
        quote!(#referrer).to_string()
    };
    let first = run();
    for _ in 0..16 {
        assert_eq!(run(), first);
    }
}

/// A kept entry stays one when a later scope forbids it: that scope may get a copy that does not
/// compile, which is the price of pruning (the doc of `reduce_roots()`).
///
/// The type `W2<W1<T, Rel<U>>, Rel3<U>>` is one required entry in a scope of its own (one cut for
/// two required types). The self type of the impl is the same type in the same context, but the
/// impl has `Self` in a predicate and a bound of `T` that names a leaked type, so rustc could not
/// normalize the impl of the Repeater for the whole self type. In that scope the Leaker chooses its
/// parts, `Rel<U>` and `Rel3<U>`, and does so when it reduces once, at the end.
#[test]
fn test_reduce_roots_a_kept_entry_stays_when_a_later_scope_forbids_it() {
    let generics: Generics = parse_quote! { <T: Fn(Rel2<u8>), U> };
    let ty: Type = parse_quote! { ::p::W2<::p::W1<T, Rel<U>>, Rel3<U>> };
    let item: ItemImpl = parse_quote! {
        impl<T: Fn(Rel2<u8>), U> ::p::Tr<U> for ::p::W2<::p::W1<T, Rel<U>>, Rel3<U>>
        where
            Self: Clone,
        {
        }
    };
    let copy_with = |reduce: bool| {
        let mut leaker = Leaker::with_rule(Rules::new().prelude().primitive());
        leaker.enter_generics(&generics, |view| view.visit_type(&ty));
        if reduce {
            leaker.reduce_roots();
        }
        let context = leaker.intern_impl(&item);
        let referrer = leaker.result().finish().unwrap();
        let mut copy = item.clone();
        let mut visitor = common::visitor(&referrer, context, common::placeholder);
        syn::visit_mut::VisitMut::visit_item_impl_mut(&mut visitor, &mut copy);
        (kept(&referrer), copy)
    };
    let (entries, copy) = copy_with(false);
    // The ids 0 to 2 are those of the first call: its type, `<U>`, and its generics as declared.
    assert_eq!(entries, ["Rel < U > 1", "Rel3 < U > 1", "Rel2 < u8 > 3"]);
    let expected: ItemImpl = parse_quote! {
        impl<T: Fn(P<2usize>), U> ::p::Tr<U> for ::p::W2<::p::W1<T, P<0usize>>, P<1usize>>
        where
            Self: Clone,
        {
        }
    };
    assert_eq!(copy, expected);

    // After `reduce_roots()` the type is kept as the entry that it was. The impl is copied with the
    // self type replaced as a whole.
    let (entries, copy) = copy_with(true);
    assert_eq!(
        entries,
        [
            ":: p :: W2 < :: p :: W1 < T , Rel < U > > , Rel3 < U > > 0",
            "Rel2 < u8 > 3"
        ]
    );
    let expected: ItemImpl = parse_quote! {
        impl<T: Fn(P<1usize>), U> ::p::Tr<U> for P<0usize>
        where
            Self: Clone,
        {
        }
    };
    assert_eq!(copy, expected);
}

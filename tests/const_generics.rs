//! Tests for const generics and method-local generics in the Leaker.

mod common;

use common::finish_as_strings;
use std::collections::BTreeSet;
use syn::visit_mut::VisitMut;
use syn::*;
use template_quote::quote;
use type_leak::{ContextId, Leaker, Referrer, Rules};

/// Leaks the types of `item`, and returns the top-level context of the trait.
fn leak(item: &ItemTrait) -> (Referrer, ContextId) {
    let mut leaker = Leaker::new();
    let context = leaker.intern_trait(item);
    (leaker.result().finish().unwrap(), context)
}

/// Returns the leaked types with their local generics (with the where clause), by index.
fn entries(referrer: &Referrer) -> Vec<(String, String)> {
    referrer
        .iter()
        .map(|(_, ty, context)| (ty, &context.generics))
        .map(|(ty, generics)| {
            let wc = &generics.where_clause;
            (quote!(#ty).to_string(), quote!(#generics #wc).to_string())
        })
        .collect()
}

fn entry(ty: Type, generics: Generics) -> (String, String) {
    let wc = &generics.where_clause;
    (quote!(#ty).to_string(), quote!(#generics #wc).to_string())
}

/// Rewrites `item`, a copy in the scope `context`, replacing the `ix`-th leaked type with `P<ix>`.
fn rewrite(referrer: Referrer, context: ContextId, item: ItemTrait) -> String {
    let mut item = Item::Trait(item);
    common::visitor(&referrer, context, |_, ix, _| parse_quote!(P<#ix>)).visit_item_mut(&mut item);
    quote!(#item).to_string()
}

fn expected(item: ItemTrait) -> String {
    quote!(#item).to_string()
}

#[test]
fn test_const_generic_param_in_expr() {
    // The Leaker's own const generic parameter used as an array length is context-free.
    let item: ItemStruct = parse_quote! {
        struct S<const N: usize> { a: [RelType; N], b: [::core::primitive::u8; N] }
    };
    let mut leaker = Leaker::new();
    leaker.intern_struct(&item);
    leaker.reduce_roots();
    // The element is interned (the innermost minimum cut), and the array is rebuilt around it.
    let expected: Type = parse_quote!(RelType);
    assert_eq!(
        finish_as_strings(leaker),
        BTreeSet::from([quote!(#expected).to_string()])
    );
}

#[test]
fn test_const_generic_param_ty_not_interned() {
    // The type of a const parameter cannot be a projection, so it must not be interned.
    let item: ItemStruct = parse_quote! { struct S<const N: usize, const B: bool>; };
    let mut leaker = Leaker::new();
    leaker.intern_struct(&item);
    leaker.reduce_roots();
    assert!(leaker.result().finish().unwrap().is_empty());

    let item: ItemTrait = parse_quote! {
        trait MyTrait { fn f<const M: ::core::primitive::u8>(&self); }
    };
    let mut leaker = Leaker::new();
    leaker.intern_trait(&item);
    leaker.reduce_roots();
    assert!(leaker.result().finish().unwrap().is_empty());
}

#[test]
fn test_method_local_generics_not_interned() {
    // Types mentioning method-local generics cannot be interned as a whole because the
    // Repeater's implementation cannot refer to them; their components are interned instead.
    let item: ItemTrait = parse_quote! {
        trait MyTrait {
            fn f<const M: usize>(&self) -> [RelType1; M];
            fn g<'b>(&self, x: &'b RelType2);
            fn h<S>(&self, x: (S, RelType3));
            fn i<const M: usize>(&self) -> [::core::primitive::u8; M];
        }
    };
    let mut leaker = Leaker::new();
    leaker.intern_trait(&item);
    leaker.reduce_roots();
    assert_eq!(
        finish_as_strings(leaker),
        BTreeSet::from([
            "RelType1".to_string(),
            "RelType2".to_string(),
            "RelType3".to_string()
        ])
    );
}

#[test]
fn test_local_generics_leaked() {
    // A relative type constructor applied to local generics cannot be decomposed, so it is
    // interned together with the local generics it depends on (including those used in their
    // bounds and where predicates), in a canonical form (bounds moved to the parameters).
    let item: ItemTrait = parse_quote! {
        trait MyTrait<T> {
            fn f<const B: bool>(&self) -> MyWrap<B>;
            fn g<'b, S: ::core::iter::Iterator<Item = &'b U>, U>(&self) -> MyVec<S::Item>
            where
                U: ::core::clone::Clone,
                T: ::core::marker::Copy;
            fn h<'c>(&self, x: MyHolder<'c>, y: [RelType; 2]);
        }
    };
    let mut leaker = Leaker::new();
    leaker.intern_trait(&item);
    leaker.reduce_roots();
    let referrer = leaker.result().finish().unwrap();
    let actual: BTreeSet<(String, String)> = referrer
        .iter()
        .map(|(_, ty, context)| (ty, &context.generics))
        .map(|(ty, generics)| {
            let wc = &generics.where_clause;
            (quote!(#ty).to_string(), quote!(#generics #wc).to_string())
        })
        .collect();
    let expected = |ty: Type, generics: Generics| {
        let wc = &generics.where_clause;
        (quote!(#ty).to_string(), quote!(#generics #wc).to_string())
    };
    let g_generics: Generics =
        parse_quote!(<'b, S: ::core::iter::Iterator<Item = &'b U>, U: ::core::clone::Clone>);
    assert_eq!(
        actual,
        BTreeSet::from([
            expected(parse_quote!(MyWrap<B>), parse_quote!(<const B: bool>)),
            expected(parse_quote!(MyVec<S::Item>), g_generics),
            expected(parse_quote!(MyHolder<'c>), parse_quote!(<'c>)),
            expected(parse_quote!(RelType), Generics::default()),
        ])
    );

    // The local generics survive a round trip through `ToTokens` and `Parse`.
    let tokens = quote!(#referrer);
    let parsed: type_leak::Referrer = parse_quote!(#tokens);
    assert_eq!(parsed, referrer);
}

#[test]
fn test_local_generics_keep_self() {
    // Bounds of local generics are kept as they are, including `Self`, which refers to the
    // implementor of the Leaker trait.
    let item: ItemTrait = parse_quote! {
        trait MyTrait {
            fn f<T: ::core::cmp::PartialEq<Self>>(&self) -> MyWrap<T>;
        }
    };
    let mut leaker = Leaker::new();
    leaker.intern_trait(&item);
    leaker.reduce_roots();
    let referrer = leaker.result().finish().unwrap();
    let (ty, generics) = referrer
        .iter()
        .map(|(_, ty, context)| (ty, &context.generics))
        .next()
        .unwrap();
    assert_eq!(quote!(#ty).to_string(), quote!(MyWrap<T>).to_string());
    let expected: Generics = parse_quote!(<T: ::core::cmp::PartialEq<Self>>);
    assert_eq!(quote!(#generics).to_string(), quote!(#expected).to_string());
}

/// The entries (type and generics) of a trait, and the trait with each entry replaced by `P<ix>`.
/// A type is an entry of the context of the local generics it depends on, in the order of the
/// contexts' first appearance.
#[test]
fn test_entries_depend_on_the_local_generics_they_use() {
    type Case = (&'static str, ItemTrait, Vec<(String, String)>, ItemTrait);
    let cases: Vec<Case> = vec![
        (
            // The same tokens depending on different local generics are separate entries, and each
            // copy is rewritten to its own.
            "the same type with different local generics",
            parse_quote! {
                trait MyTrait {
                    fn f<const B: bool>(&self) -> Wrap<B>;
                    fn g<const B: u8>(&self) -> Wrap<B>;
                }
            },
            vec![
                entry(parse_quote!(Wrap<B>), parse_quote!(<const B: bool>)),
                entry(parse_quote!(Wrap<B>), parse_quote!(<const B: u8>)),
            ],
            parse_quote! {
                trait MyTrait {
                    fn f<const B: bool>(&self) -> P<0usize>;
                    fn g<const B: u8>(&self) -> P<1usize>;
                }
            },
        ),
        (
            "the same local generics",
            parse_quote! {
                trait MyTrait {
                    fn f<const B: bool>(&self) -> Wrap<B>;
                    fn g<const B: bool>(&self, x: Wrap<B>);
                }
            },
            vec![entry(parse_quote!(Wrap<B>), parse_quote!(<const B: bool>))],
            parse_quote! {
                trait MyTrait {
                    fn f<const B: bool>(&self) -> P<0usize>;
                    fn g<const B: bool>(&self, x: P<0usize>);
                }
            },
        ),
        (
            // Here `B` is a type in `g`, which the Leaker's item does not declare.
            "a type with and without local generics",
            parse_quote! {
                trait MyTrait {
                    fn f<const B: bool>(&self) -> Wrap<B>;
                    fn g(&self) -> Wrap<B>;
                }
            },
            vec![
                entry(parse_quote!(Wrap<B>), parse_quote!(<const B: bool>)),
                entry(parse_quote!(Wrap<B>), Generics::default()),
            ],
            parse_quote! {
                trait MyTrait {
                    fn f<const B: bool>(&self) -> P<0usize>;
                    fn g(&self) -> P<1usize>;
                }
            },
        ),
        (
            // The order of the parameters and the bounds, and the place of the bounds, do not
            // matter.
            "equivalent local generics",
            parse_quote! {
                trait MyTrait {
                    fn f<T: ::core::fmt::Debug + ::core::clone::Clone, U>(&self) -> Wrap<T, U>;
                    fn g<U, T>(&self) -> Wrap<T, U> where T: ::core::clone::Clone + ::core::fmt::Debug;
                }
            },
            vec![entry(
                parse_quote!(Wrap<T, U>),
                parse_quote!(<T: ::core::clone::Clone + ::core::fmt::Debug, U>),
            )],
            parse_quote! {
                trait MyTrait {
                    fn f<T: ::core::fmt::Debug + ::core::clone::Clone, U>(&self) -> P<0usize>;
                    fn g<U, T>(&self) -> P<0usize> where T: ::core::clone::Clone + ::core::fmt::Debug;
                }
            },
        ),
        (
            "local generics used in the bounds",
            parse_quote! {
                trait MyTrait {
                    fn f<T, U: ::core::cmp::PartialEq<T>>(&self) -> Wrap<U>;
                }
            },
            vec![entry(
                parse_quote!(Wrap<U>),
                parse_quote!(<T, U: ::core::cmp::PartialEq<T>>),
            )],
            parse_quote! {
                trait MyTrait {
                    fn f<T, U: ::core::cmp::PartialEq<T>>(&self) -> P<0usize>;
                }
            },
        ),
    ];
    for (what, item, entries_expected, rewritten) in cases {
        let (referrer, context) = leak(&item);
        assert_eq!(entries(&referrer), entries_expected, "{}", what);
        assert_eq!(
            rewrite(referrer, context, item),
            expected(rewritten),
            "{}",
            what
        );
    }
}

#[test]
fn test_binder_in_type() {
    // The lifetimes of a `for<..>` binder are local generics inside it, so `&'b Rel<L>` is
    // decomposed, and `Rel<L>` is leaked with `L` only.
    let item: ItemTrait = parse_quote! {
        trait MyTrait {
            fn f<L>(&self, x: ::std::boxed::Box<dyn for<'b> ::p::F<&'b Rel<L>>>);
        }
    };
    let (referrer, context) = leak(&item);
    // The `Box` of an external crate holds `Rel<L>`, so it is the one entry, and required (`Rel<L>`
    // was the required entry, and the `Box` an optional one).
    assert_eq!(
        entries(&referrer),
        [entry(
            parse_quote!(::std::boxed::Box<dyn for<'b> ::p::F<&'b Rel<L>>>),
            parse_quote!(<L>)
        )]
    );
    assert_eq!(
        rewrite(referrer, context, item.clone()),
        expected(parse_quote! {
            trait MyTrait {
                fn f<L>(&self, x: P<0usize>);
            }
        })
    );
    // Where the Leaker allows the crates, `Rel<L>` is the entry, with `L` only.
    let mut leaker = Leaker::with_rule(Rules::new().external_crates());
    let context = leaker.intern_trait(&item);
    let referrer = leaker.result().finish().unwrap();
    assert_eq!(
        entries(&referrer),
        [entry(parse_quote!(Rel<L>), parse_quote!(<L>))]
    );
    assert_eq!(
        rewrite(referrer, context, item),
        expected(parse_quote! {
            trait MyTrait {
                fn f<L>(&self, x: ::std::boxed::Box<dyn for<'b> ::p::F<&'b P<0usize>>>);
            }
        })
    );
}

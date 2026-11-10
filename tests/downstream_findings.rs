//! The findings of a downstream crate (`announce`, `ALGORITHM_FINDINGS.md`) which concern what
//! `type-leak` produces, each with the repro items of the report.
//!
//! `announce` is a macro crate which re-emits an item in another crate. It found that its own
//! table of Repeater impls chose the generics of an entry by name across all the scopes of the item
//! (findings A, 1, 2, 3 and 4), which `type-leak` 0.8 does not do: `Referrer::repeater_impl()` and
//! `Referrer::entry_args()` give every entry the generics, bounds and where clause of the scope in
//! which it is written. The tests here pin that for each repro of the report, so a macro that
//! builds its impls from them gets them right:
//!
//! - A: two methods with a same-named generic `X` with different bounds get different impls; a
//!   method generic named like a relative type (`Local`) is not mistaken for it.
//! - 1: a bound that names a parameter outside the entry's generics brings that parameter in (the
//!   impl declares what its bounds name, nothing else).
//! - 2: the where clause of the item reaches the impl.
//! - 3: a path segment named like a generic (`inner::U` next to `fn f<U>`) is not a generic.
//! - 4: sibling scopes with a type parameter and a const parameter of the same name get one impl
//!   each, with the parameter of its own kind.
//! - 5: a type in an array length is carried (the whole array is leaked), and a relative macro
//!   call in it is rejected.
//! - 6: a cyclic bound (`T: From<inner::LocalG<T>>`) is rejected, with a clear error.
//! - 7: `for<'b> Fn(inner::LocalL<'b>)`: the impl declares `'b`.
//!
//! Finding 3 was also a bug of `type-leak` itself, in the same method: the visitor of the Referrer
//! took the `U` of `inner::U` for a mention of the generic `U` of `fn f<U>(&self) -> inner::U`,
//! computed a context other than the Leaker's, found no entry, and left the relative type in the
//! copy (E0433 in the Referrer's crate). The context now takes only the names a type can mention a
//! generic by (`the_visitor_finds_every_entry_of_the_repros`).
//!
//! The same shapes are copied across crates in `tests/crates/leaker` (`kinds`, the ids 9071 and
//! after), where the Repeater impls and the copies are compiled: `tests/cross_crate.rs`.
//!
//! The last tests are the findings 12 and 14: a bare `?Sized` and the values of an item (a type in
//! a value is carried since F40 of `tests/findings.rs`, a path that is no type is rejected).

use syn::visit_mut::VisitMut;
use syn::*;
use template_quote::{quote, ToTokens};
use type_leak::*;

fn text<T: ToTokens>(t: &T) -> String {
    quote!(#t).to_string()
}

/// The messages of `errors`, one per line.
fn messages(errors: &[ErrorItem]) -> String {
    errors
        .iter()
        .map(|item| item.message.as_str())
        .collect::<Vec<_>>()
        .join("\n")
}

fn config() -> LeakerConfig {
    let mut config = LeakerConfig::new();
    config.allow_primitive().allow_prelude();
    config
}

fn trait_of(item: ItemTrait) -> Referrer {
    let mut leaker = Leaker::from_config(config());
    leaker.intern_trait(&item, true);
    leaker.result().finish().unwrap()
}

fn struct_of(item: ItemStruct) -> Referrer {
    let mut leaker = Leaker::from_config(config());
    leaker.intern_struct(&item, true);
    leaker.result().finish().unwrap()
}

fn index(referrer: &Referrer, ty: Type) -> usize {
    referrer
        .iter()
        .position(|entry| *entry == ty)
        .unwrap_or_else(|| panic!("no entry `{}`", text(&ty)))
}

/// The Repeater impl of the entry `ix`, for any type `__S`, as a macro writes it from
/// `Referrer::repeater_impl()`. The index is written `0usize` whatever it is: the impls are compared
/// by what they declare.
fn impl_of(referrer: &Referrer, ix: usize) -> ItemImpl {
    let param: Ident = parse_quote!(__S);
    let r = referrer.repeater_impl(ix, ImplSelf::Param(&param)).unwrap();
    let (impl_generics, _, where_clause) = r.generics.split_for_impl();
    let (args, self_ty, ty) = (&r.args, &r.self_ty, &r.ty);
    parse_quote! {
        impl #impl_generics crate::Repeater<#args, 7u64, 0usize> for #self_ty #where_clause {
            type Type = #ty;
        }
    }
}

/// The impl of the entry `ty`, and the arguments that a use site in the scope of the analysis
/// writes: they are the same tokens, so a copy and its impl cannot disagree.
fn impl_and_args(referrer: &Referrer, ty: Type) -> ItemImpl {
    let ix = index(referrer, ty);
    let param: Ident = parse_quote!(__S);
    let r = referrer.repeater_impl(ix, ImplSelf::Param(&param)).unwrap();
    assert_eq!(
        referrer.entry_args(ix, &UseSite::identity()).unwrap(),
        r.args,
        "the arguments at an identity site are those of the impl"
    );
    impl_of(referrer, ix)
}

// ── A ────────────────────────────────────────────────────────────────────────────────────────

/// A: `fn f<X: Clone>` and `fn g<X>` declare the same name with different bounds. The impl of
/// `GC<X>` has `X: Clone` and the impl of `G<X>` has not (the report: the impl for `G<X>` got
/// `X: Clone`, and the consumer failed with E0277).
#[test]
fn a_methods_with_a_same_named_generic_get_the_bounds_of_their_own() {
    let referrer = trait_of(parse_quote! {
        trait T {
            fn f<X: Clone>(&self) -> inner::GC<X>;
            fn g<X>(&self) -> inner::G<X>;
        }
    });
    assert_eq!(referrer.iter().count(), 2);
    assert_eq!(
        impl_and_args(&referrer, parse_quote!(inner::GC<X>)),
        parse_quote! {
            impl<X: Clone, __S: ?::core::marker::Sized> crate::Repeater<(X,), 7u64, 0usize> for __S {
                type Type = inner::GC<X>;
            }
        }
    );
    assert_eq!(
        impl_and_args(&referrer, parse_quote!(inner::G<X>)),
        parse_quote! {
            impl<X, __S: ?::core::marker::Sized> crate::Repeater<(X,), 7u64, 0usize> for __S {
                type Type = inner::G<X>;
            }
        }
    );
    // Each entry has a context of its own: a table keyed by the type text alone would merge them.
    let a = index(&referrer, parse_quote!(inner::GC<X>));
    let b = index(&referrer, parse_quote!(inner::G<X>));
    assert_ne!(referrer.context(a), referrer.context(b));
    assert_ne!(
        text(&referrer.generics(a)),
        text(&referrer.generics(b)),
        "the generics of the contexts differ"
    );
}

/// A: the same type written in two methods with different bounds on its generic is two entries,
/// and the same type with the same generics in two methods is one.
#[test]
fn a_the_same_type_under_different_bounds_is_two_entries() {
    let referrer = trait_of(parse_quote! {
        trait T {
            fn f<X: Clone>(&self) -> inner::G<X>;
            fn g<X>(&self) -> inner::G<X>;
            fn h<X>(&self) -> inner::G<X>;
        }
    });
    let impls: Vec<ItemImpl> = (0..referrer.iter().count())
        .map(|ix| impl_of(&referrer, ix))
        .collect();
    assert_eq!(impls.len(), 2, "{:?}", referrer.iter().collect::<Vec<_>>());
    assert!(impls.contains(&parse_quote! {
        impl<X: Clone, __S: ?::core::marker::Sized> crate::Repeater<(X,), 7u64, 0usize> for __S {
            type Type = inner::G<X>;
        }
    }));
    assert!(impls.contains(&parse_quote! {
        impl<X, __S: ?::core::marker::Sized> crate::Repeater<(X,), 7u64, 0usize> for __S {
            type Type = inner::G<X>;
        }
    }));
}

/// A: `fn f<Local>(&self, x: Local)` declares a generic named `Local`; `fn g(&self) -> Local` names
/// a relative type. Only `g`'s `Local` is an entry, it has no generics, and `f`'s is the generic
/// (the report: the entry was interned with the generics `(Local,)`, and the consumer failed with
/// E0425).
#[test]
fn a_a_method_generic_named_like_a_relative_type_is_not_the_type() {
    let referrer = trait_of(parse_quote! {
        trait U {
            fn f<Local>(&self, x: Local);
            fn g(&self) -> Local;
        }
    });
    assert_eq!(referrer.iter().count(), 1);
    assert_eq!(
        impl_and_args(&referrer, parse_quote!(Local)),
        parse_quote! {
            impl<__S: ?::core::marker::Sized> crate::Repeater<(), 7u64, 0usize> for __S {
                type Type = Local;
            }
        }
    );
    // The mirror image: `Local` is a generic in both, so there is no entry.
    let referrer = trait_of(parse_quote! {
        trait U2 {
            fn f<Local>(&self, x: Local);
            fn g<Local>(&self) -> Option<Local>;
        }
    });
    assert_eq!(referrer.iter().count(), 0);
    // And the generic `Local` of one method does not hide the relative `Local` of a type
    // constructor in the other method.
    let referrer = trait_of(parse_quote! {
        trait U3 {
            fn f<Local>(&self, x: inner::Wrap<Local>);
            fn g(&self) -> Local;
        }
    });
    assert_eq!(
        impl_and_args(&referrer, parse_quote!(inner::Wrap<Local>)),
        parse_quote! {
            impl<Local, __S: ?::core::marker::Sized> crate::Repeater<(Local,), 7u64, 0usize> for __S {
                type Type = inner::Wrap<Local>;
            }
        }
    );
    assert_eq!(
        impl_and_args(&referrer, parse_quote!(Local)),
        parse_quote! {
            impl<__S: ?::core::marker::Sized> crate::Repeater<(), 7u64, 0usize> for __S {
                type Type = Local;
            }
        }
    );
}

// ── 1 ────────────────────────────────────────────────────────────────────────────────────────

/// 1: `struct BF<T: Into<V>, V>(inner::LocalG<T>, pub V)`. The bound of `T` names `V`, which the
/// entry `inner::LocalG<T>` does not use: the impl declares `V` too, so that the bound it keeps
/// names only what it declares (the report: `V` was undeclared, E0425).
#[test]
fn one_a_bound_naming_another_parameter_brings_it_in() {
    let referrer = struct_of(parse_quote! {
        struct BF<T: Into<V>, V>(pub inner::LocalG<T>, pub V);
    });
    assert_eq!(referrer.iter().count(), 1);
    assert_eq!(
        impl_and_args(&referrer, parse_quote!(inner::LocalG<T>)),
        parse_quote! {
            impl<T: Into<V>, V, __S: ?::core::marker::Sized> crate::Repeater<(T, V), 7u64, 0usize> for __S {
                type Type = inner::LocalG<T>;
            }
        }
    );
}

/// 1: `fn f<U: Into<T>>(&self, x: LocalG<U>)` in `trait Tr<T>`: the bound of the method generic
/// names the trait's `T`.
#[test]
fn one_a_method_bound_naming_a_trait_parameter_brings_it_in() {
    let referrer = trait_of(parse_quote! {
        trait Tr<T> {
            fn f<U: Into<T>>(&self, x: inner::LocalG<U>);
        }
    });
    assert_eq!(
        impl_and_args(&referrer, parse_quote!(inner::LocalG<U>)),
        parse_quote! {
            impl<T, U: Into<T>, __S: ?::core::marker::Sized> crate::Repeater<(T, U), 7u64, 0usize> for __S {
                type Type = inner::LocalG<U>;
            }
        }
    );
    // A site that passes `T` by its own name writes both arguments.
    let ix = index(&referrer, parse_quote!(inner::LocalG<U>));
    assert_eq!(
        referrer.entry_args(ix, &UseSite::identity()).unwrap(),
        parse_quote!((T, U))
    );
}

/// 1: `T: 'a` names a lifetime the entry does not use. The impl declares `'a` and encodes the bound
/// as an argument (`&'a T`), so the impl has no bound that names an undeclared lifetime.
#[test]
fn one_an_outlives_bound_is_encoded_with_its_lifetime() {
    let referrer = struct_of(parse_quote! {
        struct S<'a, T: 'a>(pub inner::LocalG<T>, pub &'a u8);
    });
    assert_eq!(
        impl_and_args(&referrer, parse_quote!(inner::LocalG<T>)),
        parse_quote! {
            impl<'a, T, __S: ?::core::marker::Sized> crate::Repeater<(&'a (), T, &'a T), 7u64, 0usize> for __S {
                type Type = inner::LocalG<T>;
            }
        }
    );
}

// ── 2 ────────────────────────────────────────────────────────────────────────────────────────

/// 2: `struct WC<T>(inner::LocalC<T>) where T: Clone`. The where clause of the item reaches the
/// impl, so the carried type is well-formed there (the report: E0277 at the definer).
#[test]
fn two_the_where_clause_reaches_the_impl() {
    let referrer = struct_of(parse_quote! {
        struct WC<T>(pub inner::LocalC<T>) where T: Clone;
    });
    assert_eq!(
        impl_and_args(&referrer, parse_quote!(inner::LocalC<T>)),
        parse_quote! {
            impl<T: Clone, __S: ?::core::marker::Sized> crate::Repeater<(T,), 7u64, 0usize> for __S {
                type Type = inner::LocalC<T>;
            }
        }
    );
    // A predicate that is not on a parameter stays a where clause of the impl.
    let referrer = struct_of(parse_quote! {
        struct WV<T>(pub inner::LocalC<T>) where Vec<T>: Clone;
    });
    assert_eq!(
        impl_and_args(&referrer, parse_quote!(inner::LocalC<T>)),
        parse_quote! {
            impl<T, __S: ?::core::marker::Sized> crate::Repeater<(T,), 7u64, 0usize> for __S
            where
                Vec<T>: Clone
            {
                type Type = inner::LocalC<T>;
            }
        }
    );
}

// ── 3 ────────────────────────────────────────────────────────────────────────────────────────

/// 3: `fn g(&self) -> inner::U` next to `fn f<U>(..)`. The last segment of a path is not a generic
/// (the report: `apply` wrote `(U,)` at `inner::U`, E0425 at the consumer). The entry `inner::U` has
/// no generics, and `inner::LocalG<U>` has `U`.
#[test]
fn three_a_path_segment_named_like_a_generic_is_not_a_generic() {
    let referrer = trait_of(parse_quote! {
        trait MD {
            fn f<U>(&self, x: inner::LocalG<U>);
            fn g(&self) -> inner::U;
        }
    });
    assert_eq!(referrer.iter().count(), 2);
    assert_eq!(
        impl_and_args(&referrer, parse_quote!(inner::U)),
        parse_quote! {
            impl<__S: ?::core::marker::Sized> crate::Repeater<(), 7u64, 0usize> for __S {
                type Type = inner::U;
            }
        }
    );
    assert_eq!(
        impl_and_args(&referrer, parse_quote!(inner::LocalG<U>)),
        parse_quote! {
            impl<U, __S: ?::core::marker::Sized> crate::Repeater<(U,), 7u64, 0usize> for __S {
                type Type = inner::LocalG<U>;
            }
        }
    );
    // The same for an associated name: `x.U` is no path, and `inner::U::V` starts with `inner`.
    let referrer = trait_of(parse_quote! {
        trait MD2 {
            fn f<U>(&self, x: U);
            fn g(&self) -> inner::U::V;
        }
    });
    assert_eq!(
        impl_and_args(&referrer, parse_quote!(inner::U::V)),
        parse_quote! {
            impl<__S: ?::core::marker::Sized> crate::Repeater<(), 7u64, 0usize> for __S {
                type Type = inner::U::V;
            }
        }
    );
}

/// 3: `<Self>::A` names the associated item `A` of the implementor, and `fn f<A: Clone>` declares a
/// generic of the same name: the entry does not use it, so the impl does not declare it.
#[test]
fn three_an_associated_name_is_not_a_generic() {
    let referrer = trait_of(parse_quote! {
        trait MD {
            type A;
            fn f<A: Clone>(&self, x: A) -> inner::Wrap<<Self>::A>;
        }
    });
    assert_eq!(referrer.iter().count(), 1);
    assert_eq!(
        impl_and_args(&referrer, parse_quote!(inner::Wrap<<Self>::A>)),
        parse_quote! {
            impl<__S: ?::core::marker::Sized + MD> crate::Repeater<(), 7u64, 0usize> for __S {
                type Type = inner::Wrap<<__S>::A>;
            }
        }
    );
}

// ── 4 ────────────────────────────────────────────────────────────────────────────────────────

/// 4: `fn f<N>(x: LocalG<N>)` next to `fn g<const N: usize>(x: Arr<N>)`. Each impl declares the
/// parameter of its own kind (the report: the impl declared the type parameter `N` for both, E0747).
#[test]
fn four_sibling_scopes_keep_the_kind_of_their_parameter() {
    let referrer = trait_of(parse_quote! {
        trait Tr {
            fn f<N>(&self, x: inner::LocalG<N>);
            fn g<const N: usize>(&self, x: inner::Arr<N>);
        }
    });
    assert_eq!(referrer.iter().count(), 2);
    assert_eq!(
        impl_and_args(&referrer, parse_quote!(inner::LocalG<N>)),
        parse_quote! {
            impl<N, __S: ?::core::marker::Sized> crate::Repeater<(N,), 7u64, 0usize> for __S {
                type Type = inner::LocalG<N>;
            }
        }
    );
    assert_eq!(
        impl_and_args(&referrer, parse_quote!(inner::Arr<N>)),
        parse_quote! {
            impl<const N: usize, __S: ?::core::marker::Sized> crate::Repeater<([(); N],), 7u64, 0usize> for __S {
                type Type = inner::Arr<N>;
            }
        }
    );
}

// ── 5 ────────────────────────────────────────────────────────────────────────────────────────

/// 5: an array length that names a relative type (`size_of::<inner::Local>()`,
/// `<inner::Local as Tr>::N`) leaks the whole array, and a relative macro call in it is rejected
/// (the report: they passed the value-path check and broke the consumer).
#[test]
fn five_array_lengths_are_carried_and_a_relative_macro_is_rejected() {
    for (length, entry) in [
        (
            "::core::mem::size_of::<inner::Local>()",
            "[u8 ; :: core :: mem :: size_of :: < inner :: Local > ()]",
        ),
        (
            "<inner::Local as ::p::Tr>::N",
            "[u8 ; < inner :: Local as :: p :: Tr > :: N]",
        ),
    ] {
        let source = format!("trait Tr {{ fn f(&self) -> [u8; {}]; }}", length);
        let referrer = trait_of(parse_str(&source).unwrap());
        assert_eq!(
            referrer.iter().map(text).collect::<Vec<_>>(),
            [entry],
            "{}",
            source
        );
        // The impl is for any type, and has the length as it is written.
        let ix = 0;
        assert_eq!(impl_of(&referrer, ix).items.len(), 1);
    }
    // A macro call in the length: relative, it cannot be leaked; absolute, it is copied as written.
    assert_eq!(
        default_entries("trait Tr { fn f(&self) -> [::core::primitive::u8; two!()]; }"),
        Err(vec!["use absolute path: 'two'".to_string()])
    );
    assert_eq!(
        default_entries("trait Tr { fn f(&self) -> [::core::primitive::u8; ::p::two!()]; }"),
        Ok(0)
    );
}

// ── 6 ────────────────────────────────────────────────────────────────────────────────────────

/// 6: `struct CY<T: From<inner::LocalG<T>>>(T)` is rejected by the Leaker, at the definer, with an
/// error that names the bound (the report: it built at the definer and failed at the consumer,
/// where the rewritten bound needs itself to normalise).
#[test]
fn six_a_cyclic_bound_is_rejected_with_a_clear_error() {
    let item: ItemStruct = parse_quote! {
        struct CY<T: From<inner::LocalG<T>>>(pub T);
    };
    let mut leaker = Leaker::from_config(config());
    leaker.intern_struct(&item, true);
    let result = leaker.result();
    let err = result.errors();
    assert!(!err.is_empty());
    assert!(err
        .iter()
        .all(|item| item.reason != Reason::SelfWithoutMeaning));
    let items = err;
    assert_eq!(items.len(), 1);
    assert_eq!(
        items[0].message,
        "`inner :: LocalG < T >` is in a bound or where predicate, and a bound of its generics \
         names a leaked type"
    );
    assert!(
        items[0].note().unwrap().starts_with(
            "the bound `T : From < inner :: LocalG < T > >` of its generics names a type that is \
             leaked too"
        ),
        "{:?}",
        items[0].note()
    );
    // Without the cycle it is fine.
    let item: ItemStruct = parse_quote! {
        struct CY<T: From<u8>>(pub inner::LocalG<T>);
    };
    let mut leaker = Leaker::from_config(config());
    leaker.intern_struct(&item, true);
    let result = leaker.result();
    assert!(result.finish().is_ok());
}

// ── 7 ────────────────────────────────────────────────────────────────────────────────────────

/// 7: `where F: for<'b> Fn(inner::LocalL<'b>)`. The impl of `inner::LocalL<'b>` declares `'b` and
/// encodes it, so the bound is `for<'b> Fn(<_ as Repeater<(&'b (),), ..>>::Type)` in the copy. (The
/// report says it is rejected; `type-leak` accepts it as long as no output names `'b`.)
#[test]
fn seven_a_binder_lifetime_is_declared_by_the_impl() {
    let referrer = struct_of(parse_quote! {
        struct W<F>(pub F) where F: for<'b> Fn(inner::LocalL<'b>);
    });
    assert_eq!(referrer.iter().count(), 1);
    assert_eq!(
        impl_and_args(&referrer, parse_quote!(inner::LocalL<'b>)),
        parse_quote! {
            impl<'b, __S: ?::core::marker::Sized> crate::Repeater<(&'b (),), 7u64, 0usize> for __S {
                type Type = inner::LocalL<'b>;
            }
        }
    );
    // An output that names `'b` is the case the copy cannot express (E0581, E0582): rejected.
    let item: ItemStruct = parse_quote! {
        struct W<F>(pub F) where F: for<'b> Fn(inner::LocalL<'b>) -> &'b u8;
    };
    let mut leaker = Leaker::from_config(config());
    leaker.intern_struct(&item, true);
    let result = leaker.result();
    let err = result.errors();
    assert_eq!(messages(err), "use absolute path: 'inner :: LocalL < 'b >'");
    assert!(
        err[0].note().unwrap().contains("`for<..>` binder"),
        "{:?}",
        err[0].note()
    );
}

// ── the Referrer finds what the Leaker found ──────────────────────────────────────────────────

/// The entries that the visitor of the Referrer replaces when it visits `item`, by index.
fn found_by_visitor(source: &str) -> (usize, std::collections::BTreeSet<usize>) {
    use std::cell::RefCell;
    use syn::visit_mut::VisitMut;
    let mut item: Item = parse_str(source).unwrap();
    let mut leaker = Leaker::from_config(config());
    let context = match &item {
        Item::Struct(i) => leaker.intern_struct(i, true),
        Item::Trait(i) => leaker.intern_trait(i, true),
        _ => unreachable!(),
    };
    let referrer = leaker
        .result()
        .finish()
        .unwrap_or_else(|e| panic!("{}: {}", source, messages(&e)));
    let count = referrer.iter().count();
    let found = RefCell::new(std::collections::BTreeSet::new());
    let mut visitor = referrer.visitor(context, |ty, ix| {
        found.borrow_mut().insert(ix);
        ty
    });
    visitor.visit_item_mut(&mut item);
    drop(visitor);
    (count, found.into_inner())
}

/// The Leaker chooses the context of a type from the generics the type mentions, and the visitor of
/// the Referrer computes it from the type alone, so they must read the type alike: a path
/// segment that is named like a generic (`U` in `inner::U`) is no mention of it. If they differed,
/// the visitor would not find the entry, and the copy would keep the relative type (E0433).
#[test]
fn the_visitor_finds_every_entry_of_the_repros() {
    for source in [
        // A
        "trait T { fn f<X: Clone>(&self) -> inner::GC<X>; fn g<X>(&self) -> inner::G<X>; }",
        "trait U { fn f<Local>(&self, x: Local); fn g(&self) -> Local; }",
        // 1, 2
        "struct BF<T: Into<V>, V>(pub inner::LocalG<T>, pub V);",
        "trait Tr<T> { fn f<U: Into<T>>(&self, x: inner::LocalG<U>); }",
        "struct S<'a, T: 'a>(pub inner::LocalG<T>, pub &'a u8);",
        "struct WC<T>(pub inner::LocalC<T>) where T: Clone;",
        // 3: the segment is named like a generic of the same method, of another method, of the trait
        "trait MD { fn f<U>(&self, x: inner::LocalG<U>); fn g(&self) -> inner::U; }",
        "trait MD { fn f<U>(&self, x: inner::LocalG<U>) -> inner::U; }",
        "trait MD { fn f<U>(&self) -> inner::U; }",
        "trait MD { fn f<U>(&self) -> inner::U::V; }",
        "trait MD { fn f<U>(&self, x: U) -> (U, inner::U, inner::U<U>); }",
        "trait MD<U> { fn f(&self, x: U) -> inner::U; }",
        "trait MD { fn f<U>(&self, x: U) -> Vec<inner::U>; }",
        "trait MD { fn f<U>(&self, x: U) -> [inner::U; 2]; }",
        "trait MD { fn f<U>(&self, x: U) -> &'static inner::U; }",
        "trait MD { fn f<U>(&self, x: U) -> <inner::T as inner::U>::A; }",
        "trait MD { fn f<U: Clone>(&self) -> inner::Pair<inner::U, U>; }",
        "trait MD { fn f<T, U>(&self, x: inner::Pair<T, inner::U>); }",
        "struct MS<U>(pub inner::U, pub U);",
        "struct MS<U: Clone>(pub inner::Pair<inner::U, U>);",
        // the associated name after `Self` or a trait is no generic either
        "trait MD { type A; fn f<A>(&self, x: A) -> inner::Wrap<<Self>::A>; }",
        "trait MD { type A; fn f<A>(&self, x: A) -> inner::Wrap<Self::A>; }",
        "trait MD { type A; fn f<A>(&self, x: A) -> inner::Wrap<<Self as MD>::A>; }",
        "trait MD { type A; fn f<MD>(&self, x: MD) -> inner::Wrap<<Self as inner::MD>::A>; }",
        "trait MD: Iterator { fn f<Item>(&self, x: Item) -> inner::Wrap<<Self as Iterator>::Item>; }",
        // 4
        "trait Tr { fn f<N>(&self, x: inner::LocalG<N>); fn g<const N: usize>(&self, x: inner::Arr<N>); }",
        "trait Tr { fn f<const N: usize>(&self, x: inner::Arr<N>) -> inner::N; }",
        // 7
        "struct W<F>(pub F) where F: for<'b> Fn(inner::LocalL<'b>);",
    ] {
        let (count, found) = found_by_visitor(source);
        assert!(count > 0, "{}", source);
        assert_eq!(
            found,
            (0..count).collect(),
            "the visitor misses an entry of {}",
            source
        );
    }
}

// ── 12 ───────────────────────────────────────────────────────────────────────────────────────

/// Analyzes `source` with the default configuration (no path allowed), and returns the number of
/// entries, or the messages of the errors.
fn default_entries(source: &str) -> std::result::Result<usize, Vec<String>> {
    let item: Item = parse_str(source).unwrap();
    let mut leaker = Leaker::new();
    match &item {
        Item::Struct(i) => leaker.intern_struct(i, true),
        Item::Enum(i) => leaker.intern_enum(i, true),
        Item::Trait(i) => leaker.intern_trait(i, true),
        Item::Impl(i) => leaker.intern_impl(i, true),
        Item::Mod(i) => leaker.intern_mod(i),
        _ => panic!("unsupported item"),
    };
    match leaker.result().finish() {
        Ok(referrer) => Ok(referrer.iter().count()),
        Err(err) => Err(err.iter().map(|item| item.message.clone()).collect()),
    }
}

/// 12: a bare `?Sized` is accepted wherever a bound is, with no option: rustc accepts `?` only
/// on the language's `Sized` (an error since Rust 1.86 for another trait, a warning that ignores it
/// before), so a local trait named `Sized` cannot change what it means.
#[test]
fn twelve_a_bare_relaxed_sized_is_accepted_in_every_position() {
    for source in [
        // a bound of a type parameter
        "trait A<T: ?Sized> { fn f(&self, t: &T); }",
        "struct S<T: ?Sized>(::core::marker::PhantomData<T>);",
        "enum E<T: ?Sized> { A(::core::marker::PhantomData<T>) }",
        "impl<T: ?Sized> ::p::Tr for ::core::marker::PhantomData<T> { fn f(&self, t: &T) {} }",
        // a bound of a method generic and of an associated type
        "trait A { fn f<T: ?Sized>(&self, t: &T); }",
        "trait A { type X: ?Sized; }",
        "trait A { type X<T: ?Sized>; }",
        // a where predicate
        "trait A<T> where T: ?Sized { fn f(&self, t: &T); }",
        "struct S<T>(::core::marker::PhantomData<T>) where T: ?Sized;",
        "trait A { fn f<T>(&self, t: &T) where T: ?Sized; }",
        "trait A where Self: ?Sized {}",
        // a supertrait
        "trait A: ?Sized {}",
        // in a list of bounds
        "trait A<T: ?Sized + ::core::fmt::Debug> { fn f(&self, t: &T); }",
        "trait A<T: ::core::fmt::Debug + ?Sized> { fn f(&self, t: &T); }",
        // `impl Trait`
        "trait A { fn f(&self, x: &(impl ::core::fmt::Debug + ?Sized)); }",
        // the absolute path
        "trait A<T: ?::core::marker::Sized> { fn f(&self, t: &T); }",
    ] {
        assert_eq!(default_entries(source), Ok(0), "{}", source);
    }
}

/// 12: `?Sized` is copied as written: the context keeps the bound, and the impl of an entry that uses
/// the generic declares it.
#[test]
fn twelve_a_relaxed_bound_is_copied_as_written() {
    let mut leaker = Leaker::new();
    let item: ItemTrait = parse_quote! {
        trait A<T: ?Sized> where T: ::core::fmt::Debug { fn f(&self, t: &Wrap<T>); }
    };
    let context = leaker.intern_trait(&item, true);
    let referrer = leaker.result().finish().unwrap();
    assert_eq!(referrer.iter().count(), 1);
    assert_eq!(
        text(&referrer.context_generics(context)),
        "< T : :: core :: fmt :: Debug + ? Sized >",
        "the generics of the context (the predicate moved to the parameter)"
    );
    assert_eq!(
        impl_and_args(&referrer, parse_quote!(Wrap<T>)),
        parse_quote! {
            impl<T: ::core::fmt::Debug + ?Sized, __S: ?::core::marker::Sized>
                crate::Repeater<(*const T,), 7u64, 0usize> for __S
            {
                type Type = Wrap<T>;
            }
        }
    );
}

/// 12: in a module, a bare `?Sized` does not tie the context of an entry to the module, as a
/// relative path would: the entries are the same as with `?::core::marker::Sized`.
#[test]
fn twelve_a_bare_relaxed_bound_does_not_make_an_entry_of_the_module() {
    let modules = |item: syn::Item| {
        let mut leaker = Leaker::new();
        leaker.enter_mod("m", [], |view| view.visit_item(&item));
        let referrer = leaker.result().finish().unwrap();
        (0..referrer.iter().count())
            .map(|ix| referrer.context_module(referrer.context(ix)).is_some())
            .collect::<Vec<_>>()
    };
    let bare = modules(parse_quote! {
        enum E<T: ?Sized + ::core::fmt::Debug> { A(::p::W<T>) }
    });
    let full = modules(parse_quote! {
        enum E<T: ?::core::marker::Sized + ::core::fmt::Debug> { A(::p::W<T>) }
    });
    assert_eq!(bare, [false]);
    assert_eq!(bare, full);
}

/// 12: a relaxed bound on anything but `Sized` is no relaxation (rustc rejects it), and the plain
/// `Sized` is a relative trait, which is rejected as `Clone` is. The headline names the path, not the
/// `?`.
#[test]
fn twelve_the_headline_names_the_path_not_the_modifier() {
    // `?Rel` and `?m::Sized`: a relative path with the modifier.
    for (source, path) in [
        ("trait A<T: ?Rel> {}", "Rel"),
        ("trait A<T> where T: ?Rel {}", "Rel"),
        ("trait A: ?Rel {}", "Rel"),
        ("trait A<T: ?m::Sized> {}", "m :: Sized"),
        ("trait A<T: ?Sized + Rel> {}", "Rel"),
        ("trait A<T: Rel + ?Sized> {}", "Rel"),
        ("trait A<T: ?Sized<u8>> {}", "Sized < u8 >"),
    ] {
        assert_eq!(
            default_entries(source),
            Err(vec![format!("use absolute path: '{}'", path)]),
            "{}",
            source
        );
    }
    // The plain `Sized` is the path `Sized`, with the help of the standard prelude.
    for source in [
        "trait A where Self: Sized {}",
        "trait A<T: Sized> {}",
        "trait A: Sized {}",
        "trait A { fn f(&self) where Self: Sized; }",
        "trait A<T> where T: Sized {}",
    ] {
        let item: ItemTrait = parse_str(source).unwrap();
        let mut leaker = Leaker::new();
        leaker.intern_trait(&item, true);
        let result = leaker.result();
        let err = result.errors();
        let items = err;
        assert_eq!(items.len(), 1, "{}", source);
        assert_eq!(items[0].message, "use absolute path: 'Sized'", "{}", source);
        assert_eq!(
            items[0].help_text(),
            Some("use `::core::marker::Sized` instead of `Sized`"),
            "{}",
            source
        );
        // Allowed by the prelude option, or written in full.
        let mut leaker = Leaker::from_config(config());
        leaker.intern_trait(&item, true);
        assert!(leaker.result().finish().is_ok(), "{}", source);
        let absolute = source.replace("Sized", "::core::marker::Sized");
        assert_eq!(default_entries(&absolute), Ok(0), "{}", absolute);
    }
}

/// 12: a `?Sized` that is not written bare is a path like another.
#[test]
fn twelve_a_relaxed_bound_is_bare_only_if_it_has_nothing_else() {
    // With a binder or arguments, or in a longer path, it is not the language's `Sized`.
    for source in [
        "trait A<T: ?Sized::X> {}",
        "trait A<T: ?self::Sized> {}",
        "trait A<T: ?crate::Sized> {}",
    ] {
        assert!(default_entries(source).is_err(), "{}", source);
    }
    // The prelude option names `Sized` only.
    let item: ItemTrait = parse_quote!(
        trait A<T: ?m::Sized> {}
    );
    let mut leaker = Leaker::from_config(config());
    leaker.intern_trait(&item, true);
    let result = leaker.result();
    assert!(result.finish().is_err());
}

// ── 14 ───────────────────────────────────────────────────────────────────────────────────────

fn values_config() -> LeakerConfig {
    let mut config = config();
    config.check_values();
    config
}

/// 14: the default of an associated const is not read unless the option is on: the item is
/// accepted with no entry, as the report found, and the copy breaks (E0433). This pins the
/// documented behaviour.
#[test]
fn fourteen_values_are_not_read_by_default() {
    let item: ItemTrait = parse_quote! {
        trait Tr {
            const N: usize = ::core::mem::size_of::<inner::Local>();
            fn f(&self) -> usize { inner::helper() }
        }
    };
    let mut leaker = Leaker::from_config(config());
    leaker.intern_trait(&item, true);
    assert_eq!(leaker.result().finish().unwrap().iter().count(), 0);
    // The same with a const and a function of a module.
    let item: ItemMod = parse_quote! {
        mod m {
            pub const N: usize = inner::X;
            pub fn f() -> usize { inner::helper() }
        }
    };
    let mut leaker = Leaker::from_config(config());
    leaker.intern_mod(&item);
    assert_eq!(leaker.result().finish().unwrap().iter().count(), 0);
}

/// The errors of `source` (a trait) with `check_values()`: the message, the note and the help of
/// each.
fn value_errors(source: &str) -> Vec<(String, String, String)> {
    let item: ItemTrait = parse_str(source).unwrap();
    let mut leaker = Leaker::from_config(values_config());
    leaker.intern_trait(&item, true);
    let result = leaker.result();
    let err = result.errors();
    err.iter()
        .map(|item| {
            (
                item.message.clone(),
                item.note().unwrap_or_default().to_string(),
                item.help_text().unwrap_or_default().to_string(),
            )
        })
        .collect()
}

fn help_for(path: &str) -> String {
    let example = path.replace(' ', "");
    format!(
        "write `{}` with an absolute path (e.g. `::my_crate::{}`), or allow it with \
         `LeakerConfig::allow_path()`",
        example, example
    )
}

/// 14: with `check_values()`, the relative type in the default of an associated const is carried:
/// it is an entry, which the Visitor replaces (F40 of `tests/findings.rs`, and the fixtures 9141 and
/// 9142 of `tests/cross_crate.rs`, whose copy compiles). It was rejected, with the value it is in,
/// like the paths that are no types still are.
#[test]
fn fourteen_a_type_in_an_associated_const_default_is_carried() {
    let item: ItemTrait = parse_quote! {
        trait Tr {
            const N: usize = ::core::mem::size_of::<inner::Local>();
        }
    };
    let mut leaker = Leaker::from_config(values_config());
    let context = leaker.intern_trait(&item, true);
    let referrer = leaker.result().finish().unwrap();
    assert_eq!(
        referrer.iter().map(text).collect::<Vec<_>>(),
        ["inner :: Local"]
    );
    assert!(referrer.must(0));
    let mut copy = item.clone();
    referrer
        .visitor(context, |_, ix| parse_quote!(P<#ix>))
        .visit_item_trait_mut(&mut copy);
    let expected: ItemTrait = parse_quote! {
        trait Tr {
            const N: usize = ::core::mem::size_of::<P<0usize>>();
        }
    };
    assert_eq!(text(&copy), text(&expected));
}

/// 14: with `check_values()`, a relative path in the default of an associated const that is no
/// type is rejected, with the value it is in.
#[test]
fn fourteen_a_relative_path_in_an_associated_const_default_is_rejected() {
    // A value path, and a constructor.
    assert_eq!(
        value_errors("trait Tr { const N: usize = inner::X; const M: Wrap = Wrap::new(); }"),
        vec![
            (
                "use absolute path: 'inner :: X'".to_string(),
                "it is in the default value of `N`, which the copy writes as it is".to_string(),
                help_for("inner::X"),
            ),
            (
                "use absolute path: 'Wrap :: new'".to_string(),
                "it is in the default value of `M`, which the copy writes as it is".to_string(),
                help_for("Wrap::new"),
            ),
        ]
    );
}

/// 14: the default body of a method is read too: a type in it is carried, a path that is no type
/// is rejected.
#[test]
fn fourteen_a_relative_path_in_a_default_method_body_is_rejected() {
    assert_eq!(
        value_errors("trait Tr { fn f(&self) -> usize { inner::helper() + ::core::mem::size_of::<inner::Local>() } }"),
        vec![(
            "use absolute path: 'inner :: helper'".to_string(),
            "it is in the default body of `f`, which the copy writes as it is".to_string(),
            help_for("inner::helper"),
        )]
    );
    // The error names the macro in its own way.
    assert_eq!(
        value_errors("trait Tr { fn f(&self) { println!(\"hi\"); } }"),
        vec![(
            "use absolute path: 'println !'".to_string(),
            "it is in the default body of `f`, which the copy writes as it is".to_string(),
            "use `::std::println!` instead of `println!`".to_string(),
        )]
    );
}

/// 14: an enum discriminant is always read: a relative path in it is rejected, and a type in it is
/// leaked (as in an array length), with the option or without it.
#[test]
fn fourteen_a_discriminant_is_always_read() {
    for config in [config(), values_config()] {
        let item: ItemEnum = parse_quote! { enum E { A = inner::X, B = 2 } };
        let mut leaker = Leaker::from_config(config.clone());
        leaker.intern_enum(&item, true);
        let result = leaker.result();
        let err = result.errors();
        let items = err;
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].message, "use absolute path: 'inner :: X'");
        assert_eq!(items[0].help_text(), Some(help_for("inner::X").as_str()));

        let item: ItemEnum = parse_quote! {
            enum E { A = ::core::mem::size_of::<inner::Local>() as isize }
        };
        let mut leaker = Leaker::from_config(config);
        leaker.intern_enum(&item, true);
        let referrer = leaker.result().finish().unwrap();
        assert_eq!(
            referrer.iter().map(text).collect::<Vec<_>>(),
            ["inner :: Local"]
        );
    }
}

/// 14: an array length in a type is read, and the whole array is leaked.
#[test]
fn fourteen_an_array_length_is_always_read() {
    for config in [config(), values_config()] {
        let item: ItemTrait = parse_quote! {
            trait Tr { fn f(&self) -> [u8; ::core::mem::size_of::<inner::Local>()]; }
        };
        let mut leaker = Leaker::from_config(config);
        leaker.intern_trait(&item, true);
        let referrer = leaker.result().finish().unwrap();
        assert_eq!(
            referrer.iter().map(text).collect::<Vec<_>>(),
            ["[u8 ; :: core :: mem :: size_of :: < inner :: Local > ()]"]
        );
    }
}

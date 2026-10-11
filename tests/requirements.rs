//! Tests for the requirements of the leaked types: the generics they use and what `Self` means,
//! which decide the entries (and contexts) of the `Referrer`.

mod common;

use syn::punctuated::Punctuated;
use syn::visit_mut::VisitMut;
use syn::*;
use template_quote::quote;
use type_leak::{ContextId, EncodeError, Leaker, Referrer, SelfRequirement, Visitor};

/// The tag of the format of a serialized Referrer, which it starts with.
fn tag() -> String {
    format!("type-leak-format-{}", type_leak::Referrer::FORMAT_VERSION)
}

fn s(x: &impl template_quote::ToTokens) -> String {
    quote!(#x).to_string()
}

/// Leaks the types of the traits (one call each).
fn leak_traits(traits: &[ItemTrait]) -> Referrer {
    leak_traits_in(traits).0
}

/// Like `leak_traits()`, with the top-level context of each trait: the scope of its copy.
fn leak_traits_in(traits: &[ItemTrait]) -> (Referrer, Vec<ContextId>) {
    let mut leaker = Leaker::new();
    let contexts = traits
        .iter()
        .map(|item| leaker.intern_trait(item, true).unwrap())
        .collect();
    (leaker.finish(), contexts)
}

fn leak_trait(item: ItemTrait) -> Referrer {
    leak_traits(&[item])
}

/// Returns the leaked types with the generics (and the where clause) they use.
fn entries(referrer: &Referrer) -> Vec<(String, String)> {
    referrer
        .iter_with_generics()
        .map(|(ty, generics)| (s(ty), s(&quote!(#generics #{&generics.where_clause}))))
        .collect()
}

fn generics_of(generics: Generics) -> String {
    s(&quote!(#generics #{&generics.where_clause}))
}

fn entry(ty: Type, generics: Generics) -> (String, String) {
    (s(&ty), generics_of(generics))
}

/// What `Self` means in each entry, as a string.
fn selfs(referrer: &Referrer) -> Vec<String> {
    (0..referrer.iter().count())
        .map(|ix| match referrer.context_self(referrer.context(ix)) {
            None => "-".to_string(),
            Some(SelfRequirement::Bound(path)) => format!("Self: {}", s(path)),
            Some(SelfRequirement::Type(ty)) => format!("Self = {}", s(ty)),
            Some(_) => unreachable!(),
        })
        .collect()
}

#[test]
fn types_are_shared_by_calls() {
    // `Rel` is the same type in every call, so it is one entry.
    let a: ItemStruct = parse_quote!(
        struct A {
            a: Rel,
            b: RelA,
        }
    );
    let b: ItemStruct = parse_quote!(
        struct B<T> {
            a: Rel,
            b: RelB<u8>,
        }
    );
    let mut leaker = Leaker::new();
    let tops: Vec<_> = [&a, &b]
        .iter()
        .map(|item| leaker.intern_struct(item, true).unwrap())
        .collect();
    let referrer = leaker.finish();
    assert_eq!(
        referrer.iter().map(s).collect::<Vec<_>>(),
        ["Rel", "RelA", "RelB < u8 >"]
    );
    // They use no generics, whichever the generics of the calls are: the generics of `A` (none)
    // are the context of the types, and those of `B` are another context without an entry.
    assert_eq!(referrer.generics(0).params.len(), 0);
    assert_eq!(referrer.contexts().count(), 2);
    assert_eq!(tops[0], referrer.context(0));
    assert_eq!(referrer.context_generics(tops[1]).params.len(), 1);
}

#[test]
fn equal_requirements_are_shared_and_different_ones_are_not() {
    let call = |generics: &str| -> ItemStruct {
        parse_str(&format!("struct S{} {{ a: Wrap<T> }}", generics)).unwrap()
    };
    let leak = |items: &[ItemStruct]| {
        let mut leaker = Leaker::new();
        for item in items {
            leaker.intern_struct(item, true).unwrap();
        }
        leaker.finish()
    };
    // The same `<T>` (here with other generics unused by the type).
    let referrer = leak(&[call("<T>"), call("<T, U>"), call("<U, T>")]);
    assert_eq!(
        entries(&referrer),
        [entry(parse_quote!(Wrap<T>), parse_quote!(<T>))]
    );
    // The same up to the order and the place of the bounds.
    let referrer = leak(&[
        call("<T: ::core::fmt::Debug + ::core::clone::Clone>"),
        call("<T> where T: ::core::clone::Clone + ::core::fmt::Debug"),
    ]);
    assert_eq!(referrer.iter().count(), 1);
    // `<T>` and `<const T: usize>` are different.
    let referrer = leak(&[call("<T>"), call("<const T: usize>")]);
    assert_eq!(
        entries(&referrer),
        [
            entry(parse_quote!(Wrap<T>), parse_quote!(<T>)),
            entry(parse_quote!(Wrap<T>), parse_quote!(<const T: usize>)),
        ]
    );
    assert_eq!(referrer.contexts().count(), 2);
    assert_eq!(
        (referrer.context(0).index(), referrer.context(1).index()),
        (0, 1)
    );
    // The bounds are different.
    let referrer = leak(&[
        call("<T: ::core::clone::Clone>"),
        call("<T: ::core::marker::Copy>"),
    ]);
    assert_eq!(referrer.iter().count(), 2);
}

/// README, Contexts: "equal declarations give equal contexts": the bounds of a lifetime parameter
/// are sorted, and a where predicate on a lifetime (`where 'b: 'a`) is the bound of the lifetime
/// (`'b: 'a`), as it is for a type parameter. Audit mutants `ctx_lifetime_bounds_unsorted` and
/// `ctx_where_lifetime_not_moved`.
#[test]
fn equal_lifetime_bounds_give_equal_contexts() {
    let leak = |generics: &[&str]| {
        let mut leaker = Leaker::new();
        for generics in generics {
            let item: ItemStruct =
                parse_str(&format!("struct S{} {{ a: Wrap<'a, 'b, 'c> }}", generics)).unwrap();
            leaker.intern_struct(&item, true).unwrap();
        }
        leaker.finish()
    };
    // The order of the bounds of a lifetime.
    let referrer = leak(&["<'a, 'b, 'c: 'a + 'b>", "<'a, 'b, 'c: 'b + 'a>"]);
    assert_eq!(referrer.iter().count(), 1);
    assert_eq!(referrer.contexts().count(), 1);
    assert_eq!(s(referrer.generics(0)), "< 'a , 'b , 'c : 'a + 'b >");
    // A where predicate on a lifetime is its bound.
    let referrer = leak(&["<'a, 'b: 'a, 'c>", "<'a, 'b, 'c> where 'b: 'a"]);
    assert_eq!(referrer.iter().count(), 1);
    assert_eq!(s(referrer.generics(0)), "< 'a , 'b : 'a , 'c >");
    assert!(referrer.generics(0).where_clause.is_none());
    // Different bounds are different contexts.
    let referrer = leak(&["<'a, 'b: 'a, 'c>", "<'a, 'b, 'c: 'a>"]);
    assert_eq!(referrer.iter().count(), 2);
}

/// README, Contexts: the output of the `Fn(..)` sugar is parenthesized only where a `+` would join
/// it (`Fn() -> (&dyn A) + B`), and however the source wrote it: equal declarations give equal
/// contexts. Audit mutants `fn_output_paren_always` and `fn_output_unparen_dropped`.
#[test]
fn the_output_of_the_fn_sugar_is_parenthesized_only_where_needed() {
    let leak = |bounds: &[&str]| {
        let mut leaker = Leaker::new();
        for bound in bounds {
            let item: ItemStruct =
                parse_str(&format!("struct S<F: {}> {{ a: Wrap<F> }}", bound)).unwrap();
            leaker.intern_struct(&item, true).unwrap();
        }
        leaker.finish()
    };
    // Alone, with the parentheses of the source or without: no parentheses, one context. Its
    // entries are `Wrap<F>` and the trait object in the bound, which is the same type twice.
    let referrer = leak(&[
        "::core::ops::Fn() -> &'static dyn ::p::A",
        "::core::ops::Fn() -> (&'static dyn ::p::A)",
    ]);
    assert_eq!(referrer.iter().count(), 2);
    assert_eq!(
        s(referrer.generics(0)),
        "< F : :: core :: ops :: Fn () -> & 'static dyn :: p :: A >"
    );
    // Before another bound the parentheses are needed, and kept.
    let referrer = leak(&[
        "::core::ops::Fn() -> (&'static dyn ::p::A) + ::zzz::Z",
        "::zzz::Z + ::core::ops::Fn() -> &'static (dyn ::p::A)",
    ]);
    assert_eq!(
        s(referrer.generics(0)),
        "< F : :: core :: ops :: Fn () -> (& 'static dyn :: p :: A) + :: zzz :: Z >"
    );
}

/// README, Contexts: the generic associated types of the trait that `intern_trait()` analyzes
/// (with outlives bounds) are part of its top-level context, so two traits that differ only in
/// them have two contexts. Audit mutant `ctx_gats_not_in_identity`.
#[test]
fn the_gats_are_part_of_the_identity_of_the_top_level_context() {
    let mut leaker = Leaker::new();
    let with_gat = leaker
        .intern_trait(
            &parse_quote!(
                trait Tr {
                    type G<'b>
                    where
                        Self: 'b;
                    fn f(&self) -> Rel<Self>;
                }
            ),
            true,
        )
        .unwrap();
    let without = leaker
        .intern_trait(
            &parse_quote!(
                trait Tr {
                    fn f(&self) -> Rel<Self>;
                }
            ),
            true,
        )
        .unwrap();
    let other_bound = leaker
        .intern_trait(
            &parse_quote!(
                trait Tr {
                    type G<'b>
                    where
                        Self: 'b + 'static;
                    fn f(&self) -> Rel<Self>;
                }
            ),
            true,
        )
        .unwrap();
    let again = leaker
        .intern_trait(
            &parse_quote!(
                trait Tr {
                    type G<'b>
                    where
                        Self: 'b;
                    fn f(&self) -> Rel<Self>;
                }
            ),
            true,
        )
        .unwrap();
    assert_ne!(with_gat, without);
    assert_ne!(with_gat, other_bound);
    assert_ne!(without, other_bound);
    // The same GAT is the same context.
    assert_eq!(with_gat, again);
    assert_eq!(leaker.finish().contexts().count(), 3);
}

#[test]
fn only_the_generics_used_are_required() {
    let referrer = leak_trait(parse_quote! {
        trait X<T, U> {
            fn f(&self) -> Wrap<T>;
        }
    });
    assert_eq!(
        entries(&referrer),
        [entry(parse_quote!(Wrap<T>), parse_quote!(<T>))]
    );
    assert!(!referrer.has_self());

    // The generics that the bounds of a used generic name are used, whether the bound is written
    // inline or in the where clause. A bound on another generic (`U: Into<T>`) is not: it is a
    // bound of `U`, and `Wrap<T>` does not use `U`.
    let referrer = leak_trait(parse_quote! {
        trait X<T: ::core::convert::Into<U>, U, V, W> where U: ::core::clone::Clone, V: ::core::convert::Into<T>, W: ::core::clone::Clone {
            fn f(&self) -> Wrap<T>;
        }
    });
    assert_eq!(
        entries(&referrer),
        [entry(
            parse_quote!(Wrap<T>),
            parse_quote!(<T: ::core::convert::Into<U>, U: ::core::clone::Clone>)
        )]
    );
    // The inline spelling is the same.
    let inline = leak_trait(parse_quote! {
        trait X<T: ::core::convert::Into<U>, U: ::core::clone::Clone, V: ::core::convert::Into<T>, W: ::core::clone::Clone> {
            fn f(&self) -> Wrap<T>;
        }
    });
    assert_eq!(entries(&inline), entries(&referrer));

    // The generics of the Leaker and of a method are used together.
    let referrer = leak_trait(parse_quote! {
        trait X<'a, T, U> {
            fn f<'b, const B: bool>(&self) -> Wrap<'a, 'b, T, B>;
        }
    });
    assert_eq!(
        entries(&referrer),
        [entry(
            parse_quote!(Wrap<'a, 'b, T, B>),
            parse_quote!(<'a, 'b, T, const B: bool>)
        )]
    );

    // A type which uses a generic of a method only is separate from the same one using also that of
    // the Leaker.
    let referrer = leak_trait(parse_quote! {
        trait X<T> {
            fn f<const B: bool>(&self) -> Wrap<B>;
            fn g(&self) -> Wrap<T>;
        }
    });
    assert_eq!(
        entries(&referrer),
        [
            entry(parse_quote!(Wrap<B>), parse_quote!(<const B: bool>)),
            entry(parse_quote!(Wrap<T>), parse_quote!(<T>)),
        ]
    );
}

#[test]
fn self_of_a_trait() {
    // `Self` is the implementor of the trait, and the parameters of the trait are used.
    let (referrer, contexts) = leak_traits_in(&[parse_quote! {
        trait X<'a, T, const N: usize, U> {
            fn f(&self) -> Rel<Self>;
            fn g(&self) -> Rel2;
            fn h(&self) -> Wrap<Self::Item, T>;
        }
    }]);
    assert_eq!(
        entries(&referrer),
        [
            entry(
                parse_quote!(Rel<Self>),
                parse_quote!(<'a, T, U, const N: usize>)
            ),
            entry(
                parse_quote!(Wrap<Self::Item, T>),
                parse_quote!(<'a, T, U, const N: usize>)
            ),
            entry(parse_quote!(Rel2), Generics::default()),
        ]
    );
    assert_eq!(
        selfs(&referrer),
        [
            "Self: X < 'a , T , N , U >",
            "Self: X < 'a , T , N , U >",
            "-"
        ]
    );
    // The two types are in one context, which is also the top-level one: it has the generics of
    // the trait and, as an entry names `Self`, what `Self` is.
    assert_eq!(referrer.contexts().count(), 2);
    assert_eq!(referrer.context(0), referrer.context(1));
    assert_eq!(referrer.context(0), contexts[0]);
    assert!(referrer.has_self());

    // A trait without parameters.
    let referrer = leak_trait(parse_quote!(
        trait X {
            fn f(&self) -> Rel<Self>;
        }
    ));
    assert_eq!(selfs(&referrer), ["Self: X"]);
}

#[test]
fn self_in_different_traits_is_different() {
    let (referrer, contexts) = leak_traits_in(&[
        parse_quote!(
            trait X<T> {
                fn f(&self) -> Rel<Self>;
                fn g(&self) -> Rel2;
            }
        ),
        parse_quote!(
            trait Y<T> {
                fn f(&self) -> Rel<Self>;
                fn g(&self) -> Rel2;
            }
        ),
    ]);
    // `Rel2` is shared, but `Rel<Self>` is not (`Self` is bound by another trait).
    assert_eq!(
        referrer.iter().map(s).collect::<Vec<_>>(),
        ["Rel < Self >", "Rel2", "Rel < Self >"]
    );
    assert_eq!(selfs(&referrer), ["Self: X < T >", "-", "Self: Y < T >"]);
    // The generics of the traits are the same (`<T>`), but the top-level context has `Self` as an
    // entry names it, so it is not shared by the calls: it is the context of `Rel<Self>`.
    assert_eq!(referrer.contexts().count(), 3);
    assert_eq!(contexts, [referrer.context(0), referrer.context(2)]);

    // The same trait twice is shared.
    let tr: ItemTrait = parse_quote!(
        trait X<T> {
            fn f(&self) -> Rel<Self>;
        }
    );
    let referrer = leak_traits(&[tr.clone(), tr]);
    assert_eq!(referrer.iter().count(), 1);

    // `Self` is used by a type only if it mentions it, and it needs a requirement if the bound of
    // a generic it uses mentions it.
    let referrer = leak_trait(parse_quote! {
        trait X {
            fn f<T: ::core::cmp::PartialEq<Self>>(&self) -> Wrap<T>;
            fn g<T: ::core::cmp::PartialEq>(&self) -> Wrap<T>;
        }
    });
    assert_eq!(
        entries(&referrer),
        [
            entry(
                parse_quote!(Wrap<T>),
                parse_quote!(<T: ::core::cmp::PartialEq<Self>>)
            ),
            entry(
                parse_quote!(Wrap<T>),
                parse_quote!(<T: ::core::cmp::PartialEq>)
            ),
        ]
    );
    assert_eq!(selfs(&referrer), ["Self: X", "-"]);
}

#[test]
fn self_of_a_struct_enum_union_and_impl() {
    let mut leaker = Leaker::new();
    let st: ItemStruct = parse_quote!(
        struct S<T> {
            a: Rel<Self>,
            b: Wrap<T>,
        }
    );
    leaker.intern_struct(&st, true).unwrap();
    let en: ItemEnum = parse_quote!(
        enum E<'a, T> {
            A(Rel<Self>, &'a T),
        }
    );
    leaker.intern_enum(&en, true).unwrap();
    let un: ItemUnion = parse_quote!(union U { a: Rel<Self> });
    leaker.intern_union(&un, true).unwrap();
    let im: ItemImpl = parse_quote! {
        impl<T> ::p::Tr for W<T> {
            fn f(&self) -> Rel<Self> { todo!() }
        }
    };
    leaker.intern_impl(&im, true).unwrap();
    let referrer = leaker.finish();
    assert_eq!(
        entries(&referrer),
        [
            entry(parse_quote!(Rel<Self>), parse_quote!(<T>)),
            // The self type of the impl is leaked, and shared with the `Wrap<T>` using `T` only.
            entry(parse_quote!(W<T>), parse_quote!(<T>)),
            entry(parse_quote!(Wrap<T>), parse_quote!(<T>)),
            entry(parse_quote!(Rel<Self>), parse_quote!(<'a, T>)),
            entry(parse_quote!(Rel<Self>), Generics::default()),
            entry(parse_quote!(Rel<Self>), parse_quote!(<T>)),
        ]
    );
    assert_eq!(
        selfs(&referrer),
        [
            "Self = S < T >",
            "-",
            "-",
            "Self = E < 'a , T >",
            "Self = U",
            "Self = W < T >"
        ]
    );
}

#[test]
fn self_of_a_module_is_that_of_its_items() {
    let m: ItemMod = parse_quote! {
        mod m {
            pub struct S<T> { a: Rel<Self, T> }
            pub trait Tr { fn f(&self) -> Rel2<Self>; }
        }
    };
    let mut leaker = Leaker::new();
    leaker.intern_mod(&m).unwrap();
    let referrer = leaker.finish();
    assert_eq!(selfs(&referrer), ["Self = S < T >", "Self: Tr"]);
    assert_eq!(
        entries(&referrer),
        [
            entry(parse_quote!(Rel<Self, T>), parse_quote!(<T>)),
            entry(parse_quote!(Rel2<Self>), Generics::default()),
        ]
    );
}

#[test]
fn self_in_the_bounds_of_the_generics() {
    let tr: ItemTrait = parse_quote! {
        trait X {
            fn f<T: ::core::cmp::PartialEq<Self>>(&self) -> Wrap<T>;
        }
    };
    // It is leaked with the meaning of `Self`, if `Self` can be leaked.
    let referrer = leak_trait(tr.clone());
    assert_eq!(selfs(&referrer), ["Self: X"]);
    assert!(referrer.has_self());

    // Otherwise it cannot be, as the Repeater's implementation copies the bounds.
    let mut leaker = Leaker::new();
    let err = common::intern_fns(
        &mut leaker,
        &tr.generics,
        quote! { fn f<T: ::core::cmp::PartialEq<Self>>(&self) -> Wrap<T>; },
        false,
    )
    .err()
    .unwrap();
    assert!(matches!(
        err,
        type_leak::NotInternableError::RelativePath(_)
    ));
    assert_eq!(
        err.note().as_deref(),
        Some("the bounds of its generics mention `Self`, which cannot be leaked")
    );
    assert!(err
        .to_string()
        .lines()
        .next()
        .unwrap()
        .contains("Wrap < T >"));
    // Nothing is added by a failed call.
    assert!(leaker.finish().is_empty());

    // A type which is not interned as a whole (it has no cause itself) is decomposed.
    let tr: ItemTrait = parse_quote! {
        trait X {
            fn f<T: ::core::cmp::PartialEq<Self>>(&self) -> (Rel, ::core::option::Option<T>);
        }
    };
    let mut leaker = Leaker::new();
    common::intern_fns(
        &mut leaker,
        &tr.generics,
        quote! { fn f<T: ::core::cmp::PartialEq<Self>>(&self) -> (Rel, ::core::option::Option<T>); },
        false,
    )
    .unwrap();
    assert_eq!(leaker.finish().iter().map(s).collect::<Vec<_>>(), ["Rel"]);
}

#[test]
fn generic_args_of_the_root_and_local_generics() {
    let tr: ItemTrait = parse_quote! {
        trait Tr<'a, T, U> {
            fn f<'b, const B: bool, V>(&self) -> Wrap<'a, 'b, T, B, V>;
        }
    };
    let referrer = leak_trait(tr.clone());
    assert_eq!(referrer.iter().count(), 1);
    // `'a` and `T` are the arguments of the trait, by position, and the others are passed by
    // their names.
    let completed: Vec<GenericArgument> =
        vec![parse_quote!('x), parse_quote!(u8), parse_quote!(u16)];
    let args = referrer
        .generic_args(0, &tr.generics.params, &completed)
        .unwrap();
    assert_eq!(
        args.iter().map(s).collect::<Vec<_>>(),
        ["'x", "'b", "u8", "V", "B"]
    );
    // A const argument of the trait is passed as it is.
    let tr: ItemTrait = parse_quote! {
        trait Tr<const N: usize, U> {
            fn f<V>(&self) -> Wrap<N, V>;
        }
    };
    let referrer = leak_trait(tr.clone());
    let completed: Vec<GenericArgument> = vec![parse_quote!({ 1 + 2 }), parse_quote!(u16)];
    let args = referrer
        .generic_args(0, &tr.generics.params, &completed)
        .unwrap();
    assert_eq!(args.iter().map(s).collect::<Vec<_>>(), ["V", "{ 1 + 2 }"]);
    // Too few arguments.
    assert!(matches!(
        referrer.generic_args(0, &tr.generics.params, &[]),
        Err(EncodeError::MissingArgument(_))
    ));
    // Without the parameters of the trait, everything is passed by name.
    let args = referrer.generic_args(0, &Punctuated::new(), &[]).unwrap();
    assert_eq!(args.iter().map(s).collect::<Vec<_>>(), ["V", "N"]);
}

fn rewrite<F: FnMut(Type, usize) -> Type>(mut visitor: Visitor<F>, sig: &str) -> String {
    let mut sig: Signature = parse_str(sig).unwrap();
    visitor.visit_signature_mut(&mut sig);
    s(&sig)
}

fn placeholder(_: Type, ix: usize) -> Type {
    parse_quote!(P<#ix>)
}

fn rewritten(sig: &str) -> String {
    s(&parse_str::<Signature>(sig).unwrap())
}

#[test]
fn matching_with_the_generics_of_the_context() {
    let tr: ItemTrait = parse_quote! {
        trait X<T: ::core::clone::Clone, U> {
            fn f<const B: bool>(&self, a: Wrap<T>, b: Wrap<B>, c: Rel, d: Wrap<U>);
        }
    };
    let z: ItemTrait = parse_quote! {
        trait Z {
            fn f(&self);
        }
    };
    let (referrer, contexts) = leak_traits_in(&[tr, z]);
    let (context, none) = (contexts[0], contexts[1]);
    // The generics which are not declared in the visited syntax are those of the context, as it
    // declares them.
    assert_eq!(
        rewrite(
            referrer.clone().visitor(context, placeholder),
            "fn f<const B: bool>(&self, a: Wrap<T>, b: Wrap<B>, c: Rel, d: Wrap<U>)"
        ),
        rewritten(
            "fn f<const B: bool>(&self, a: P<0usize>, b: P<1usize>, c: P<2usize>, d: P<3usize>)"
        )
    );
    // They are not those of another context (the generics which no scope declares are not
    // assumed to be the Leaker's), but what uses none is found.
    assert_eq!(
        rewrite(
            referrer.clone().visitor(none, placeholder),
            "fn f<const B: bool>(&self, a: Wrap<T>, b: Wrap<B>, c: Rel, d: Wrap<U>)"
        ),
        rewritten("fn f<const B: bool>(&self, a: Wrap<T>, b: P<1usize>, c: P<2usize>, d: Wrap<U>)")
    );
    // The generics declared in the visited syntax must be declared in the same way, and the same
    // generics must be used.
    for (generics, a) in [
        ("<const B: bool, T: ::core::clone::Clone>", "P<0usize>"),
        ("<const B: bool, T>", "Wrap<T>"),
        ("<const B: bool, T: ::core::marker::Copy>", "Wrap<T>"),
        ("<const B: u8, T: ::core::clone::Clone>", "P<0usize>"),
    ] {
        let sig = format!("fn f{}(&self, a: Wrap<T>)", generics);
        let expected = format!("fn f{}(&self, a: {})", generics, a);
        assert_eq!(
            rewrite(referrer.clone().visitor(context, placeholder), &sig),
            rewritten(&expected),
            "{}",
            generics
        );
    }
    // `Wrap<B>` is for a `bool`.
    assert_eq!(
        rewrite(
            referrer.clone().visitor(context, placeholder),
            "fn f<const B: bool>(&self, b: Wrap<B>)"
        ),
        rewritten("fn f<const B: bool>(&self, b: P<1usize>)")
    );
    assert_eq!(
        rewrite(
            referrer.clone().visitor(context, placeholder),
            "fn f<const B: u8>(&self, b: Wrap<B>)"
        ),
        rewritten("fn f<const B: u8>(&self, b: Wrap<B>)")
    );
}

#[test]
fn matching_is_exact_where_a_name_is_declared_or_not() {
    // `B` is a `bool` in `f`, and `B` is not declared in `g` (it is a type there).
    let tr: ItemTrait = parse_quote! {
        trait X {
            fn f<const B: bool>(&self) -> Wrap<B>;
            fn g(&self) -> Wrap<B>;
        }
    };
    let (referrer, contexts) = leak_traits_in(&[tr]);
    assert_eq!(referrer.iter().count(), 2);
    assert_eq!(
        rewrite(
            referrer.clone().visitor(contexts[0], placeholder),
            "fn g(&self) -> Wrap<B>"
        ),
        rewritten("fn g(&self) -> P<1usize>")
    );
    assert_eq!(
        rewrite(
            referrer.visitor(contexts[0], placeholder),
            "fn f<const B: bool>(&self) -> Wrap<B>"
        ),
        rewritten("fn f<const B: bool>(&self) -> P<0usize>")
    );
}

#[test]
fn matching_with_the_generics_of_each_scope() {
    // Two calls declare `T` differently, so the entries of `Wrap<T>` are told apart by the
    // context of the copy.
    let a: ItemTrait = parse_quote!(
        trait A<T: ::core::clone::Clone> {
            fn f(&self) -> Wrap<T>;
        }
    );
    let b: ItemTrait = parse_quote!(
        trait B<T: ::core::marker::Copy> {
            fn f(&self) -> Wrap<T>;
        }
    );
    let c: ItemTrait = parse_quote!(
        trait C<T: ::core::clone::Clone> {
            fn f(&self) -> Rel;
        }
    );
    let mut leaker = Leaker::new();
    let contexts: Vec<ContextId> = [&a, &b]
        .iter()
        .map(|tr| leaker.intern_trait(tr, true).unwrap())
        .collect();
    // No entry has these generics.
    let other: Generics = parse_quote!(<T>);
    let (other, ()) = leaker.enter_generics(&other, |_| ()).unwrap();
    let referrer = leaker.finish();
    assert_eq!(referrer.iter().count(), 2);
    let sig = "fn f(&self) -> Wrap<T>";
    for (context, ix) in [(contexts[0], 0usize), (contexts[1], 1)] {
        assert_eq!(
            rewrite(referrer.clone().visitor(context, placeholder), sig),
            rewritten(&format!("fn f(&self) -> P<{}usize>", ix))
        );
    }
    assert_eq!(
        rewrite(referrer.visitor(other, placeholder), sig),
        rewritten(sig)
    );
    // A type which does not use the generics is found whatever they are.
    let mut leaker = Leaker::new();
    leaker.intern_trait(&c, true).unwrap();
    let (other, ()) = leaker.enter_generics(&parse_quote!(<T>), |_| ()).unwrap();
    let referrer = leaker.finish();
    assert_eq!(
        rewrite(referrer.visitor(other, placeholder), "fn f(&self) -> Rel"),
        rewritten("fn f(&self) -> P<0usize>")
    );
}

#[test]
fn matching_self() {
    let x: ItemTrait = parse_quote!(
        trait X<T> {
            fn f(&self, a: Rel<Self>, b: Rel2, c: Wrap<T>);
        }
    );
    let y: ItemTrait = parse_quote!(
        trait Y<T> {
            fn f(&self, a: Rel<Self>, b: Rel2, c: Wrap<T>);
        }
    );
    // `Z` has no entry that names `Self`, and a scope with generics only does not declare it.
    let z: ItemTrait = parse_quote!(
        trait Z<T> {
            fn f(&self, b: Rel2, c: Wrap<T>);
        }
    );
    let mut leaker = Leaker::new();
    let contexts: Vec<ContextId> = [&x, &y, &z]
        .iter()
        .map(|tr| leaker.intern_trait(tr, true).unwrap())
        .collect();
    let (generics, ()) = leaker.enter_generics(&parse_quote!(<T>), |_| ()).unwrap();
    let referrer = leaker.finish();
    // `Rel<Self>` of X, `Rel2`, `Wrap<T>` and `Rel<Self>` of Y.
    assert_eq!(
        referrer.iter().map(s).collect::<Vec<_>>(),
        ["Rel < Self >", "Rel2", "Wrap < T >", "Rel < Self >"]
    );
    let sig = "fn f(&self, a: Rel<Self>, b: Rel2, c: Wrap<T>)";
    // `Self` is that of the context, as it records it: the entries that do not mention `Self` are
    // not affected by it.
    assert_eq!(
        rewrite(referrer.clone().visitor(contexts[0], placeholder), sig),
        rewritten("fn f(&self, a: P<0usize>, b: P<1usize>, c: P<2usize>)")
    );
    assert_eq!(
        rewrite(referrer.clone().visitor(contexts[1], placeholder), sig),
        rewritten("fn f(&self, a: P<3usize>, b: P<1usize>, c: P<2usize>)")
    );
    // The context of `Z` has no `Self` that an entry names, and the one of `enter_generics()` has
    // none at all: `Rel<Self>` is not found.
    for context in [contexts[2], generics] {
        assert_eq!(
            rewrite(referrer.clone().visitor(context, placeholder), sig),
            rewritten("fn f(&self, a: Rel<Self>, b: P<1usize>, c: P<2usize>)")
        );
    }
    // The same with the generics of the trait declared again (they are those of the context).
    let mut copy: ItemTrait = parse_quote!(
        trait Y<T> {
            fn f(&self, a: Rel<Self>, b: Rel2, c: Wrap<T>);
        }
    );
    referrer
        .visitor(contexts[1], placeholder)
        .visit_item_trait_mut(&mut copy);
    let expected: ItemTrait = parse_quote!(
        trait Y<T> {
            fn f(&self, a: P<3usize>, b: P<1usize>, c: P<2usize>);
        }
    );
    assert_eq!(s(&copy), s(&expected));
}

#[test]
fn serialization_with_self() {
    let mut leaker = Leaker::new();
    let tr: ItemTrait = parse_quote!(
        trait X<T> {
            fn f(&self) -> Rel<Self>;
            fn g(&self) -> Wrap<T>;
        }
    );
    leaker.intern_trait(&tr, true).unwrap();
    let st: ItemStruct = parse_quote!(
        struct S<T> {
            a: RelS<Self>,
        }
    );
    leaker.intern_struct(&st, true).unwrap();
    let plain: ItemStruct = parse_quote!(
        struct P {
            a: RelP,
        }
    );
    leaker.intern_struct(&plain, true).unwrap();
    let referrer = leaker.finish();
    let tokens = quote!(#referrer);
    assert_eq!(
        tokens.to_string(),
        format!(
            "\"{}\" 4 \
             < T > ; Self : X < T > ; (Rel < Self >) \
             < T > ; (Wrap < T >) \
             < T > ; Self = S < T > ; (RelS < Self >) \
             (RelP)",
            tag()
        )
    );
    let parsed: Referrer = parse2(tokens).unwrap();
    assert_eq!(parsed, referrer);
    assert_eq!(selfs(&parsed), selfs(&referrer));
    assert_eq!(entries(&parsed), entries(&referrer));
    // A Referrer can be followed by other tokens.
    struct Followed(Referrer, Ident);
    impl parse::Parse for Followed {
        fn parse(input: parse::ParseStream) -> Result<Self> {
            Ok(Followed(input.parse()?, input.parse()?))
        }
    }
    let followed: Followed = parse2(quote!(#referrer after)).unwrap();
    assert_eq!(followed.0, referrer);
    assert_eq!(followed.1, "after");
    // Only a self part (no generics).
    let version = tag();
    let parsed: Referrer = parse2(quote!(#version Self : X ; (Rel < Self >))).unwrap();
    assert_eq!(selfs(&parsed), ["Self: X"]);
}

#[test]
fn a_call_as_self_is_not_parsed() {
    // `Self` has no meaning outside the scopes which declare it, so there is no `Self` of a call
    // (`Self = # 2 ;` was one, in the first format).
    let text = format!("\"{}\" Self = # 2 ; (Rel < Self >)", tag());
    // `Self = ` is followed by a type, and `#` starts none (syn lists the tokens that do).
    let err = parse_str::<Referrer>(&text).unwrap_err();
    assert!(err.to_string().starts_with("expected one of: "), "{}", err);
}

#[test]
fn node_numbers_follow_the_first_appearance() {
    // The contexts are numbered in the order they first appear: `Zed<U>` (the generics `<U>`),
    // `Abc<T>` (`<T>`), and then `Mid` (no generics), and the entries are listed by context. The
    // order of the tokens (`Abc < T >`, `Mid`, `Zed < U >`) is none of the numbering, so a
    // numbering by tokens or in reverse is told from the first appearance.
    let referrer = leak_trait(parse_quote! {
        trait X<T, U> {
            fn f(&self) -> Zed<U>;
            fn g(&self) -> Abc<T>;
            fn h(&self) -> Mid;
        }
    });
    assert_eq!(
        referrer.iter().map(s).collect::<Vec<_>>(),
        ["Zed < U >", "Abc < T >", "Mid"]
    );
    assert_eq!(
        (0..3)
            .map(|ix| referrer.context(ix).index())
            .collect::<Vec<_>>(),
        [0, 1, 2]
    );
    assert_eq!(
        (0..3)
            .map(|ix| s(referrer.generics(ix)))
            .collect::<Vec<_>>(),
        ["< U >", "< T >", ""]
    );
    // Entries of the same context are sorted by tokens: `Rel2` appears first, and `Rel1` is
    // before it in the list.
    let referrer = leak_trait(parse_quote! {
        trait X<T> {
            fn f(&self) -> Rel2;
            fn g(&self) -> Wrap<T>;
            fn h(&self) -> Rel1;
        }
    });
    assert_eq!(
        referrer.iter().map(s).collect::<Vec<_>>(),
        ["Rel1", "Rel2", "Wrap < T >"]
    );
    assert_eq!(
        (0..3)
            .map(|ix| referrer.context(ix).index())
            .collect::<Vec<_>>(),
        [0, 0, 1]
    );
}

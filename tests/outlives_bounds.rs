//! The outlives bounds which a leaked type implies are added to the bounds of the generics it
//! uses (see `Referrer::context_generics()`), so that the Repeater's implementation and the copies of the
//! type have them: `T: 'a` for `&'a X`, where `X` mentions `T`, and `'b: 'a` where it mentions
//! `'b`. A path with a lifetime argument (`Rel<'a, T>`) implies none: its definition is not seen
//! (`struct Rel<'a, T>(&'a T)` makes `T: 'a` part of it, `struct Loose<'a, T>(&'a (), T)` does
//! not), so only a bound which is written is a bound of the generics (`fn f<'a, T: 'a>(x: Rel<'a,
//! T>)`), and the lifetime of a path does not bound what is in it, in the type or around it.
//!
//! The module `ancestors` has the tests of the lifetimes that the references around a type make it
//! outlive, and `self_outlives` those of `Self: 'a`.

mod common;

use syn::*;
use template_quote::{quote, ToTokens};
use type_leak::{
    array_const, Context, ContextId, EntryId, ImplSelf, Leaker, Options, Referrer, Rules,
};

fn s(x: &impl template_quote::ToTokens) -> String {
    quote!(#x).to_string()
}

/// The leaked types of a trait with the generics each uses (and the where clause).
fn leaked(item: ItemTrait) -> Vec<(String, String)> {
    let mut leaker = Leaker::new();
    leaker.intern_trait(&item, true);
    let referrer = leaker.result().finish().unwrap();
    entries(&referrer)
}

/// Like `leaked()`, for a Leaker that allows the paths of external crates, so that none is an entry.
fn leaked_allowing_crates(item: ItemTrait) -> Vec<(String, String)> {
    let mut leaker = Leaker::from_config(Rules::new().external_crates(), Options::new());
    leaker.intern_trait(&item, true);
    let referrer = leaker.result().finish().unwrap();
    entries(&referrer)
}

fn entries(referrer: &Referrer) -> Vec<(String, String)> {
    referrer
        .iter()
        .map(|(_, ty, context)| (ty, &context.generics))
        .map(|(ty, generics)| (s(ty), s(&quote!(#generics #{&generics.where_clause}))))
        .collect()
}

fn e(ty: &str, generics: &str) -> (String, String) {
    (ty.to_string(), generics.to_string())
}

#[test]
fn a_path_with_a_lifetime_argument_assumes_no_bound() {
    // `struct Rel<'a, T>(&'a T)` has `T: 'a` inferred, which the signature implies, but the
    // definition of the path is not seen, so no bound is recorded: the generics are as written.
    assert_eq!(
        leaked(parse_quote! { trait Tr { fn f<'a, T>(&self, x: Rel<'a, T>); } }),
        [e("Rel < 'a , T >", "< 'a , T >")]
    );
    // The generics of the trait, and a mix of both.
    assert_eq!(
        leaked(parse_quote! { trait Tr<'a, T> { fn f(&self, x: Rel<'a, T>); } }),
        [e("Rel < 'a , T >", "< 'a , T >")]
    );
    assert_eq!(
        leaked(parse_quote! { trait Tr<'a> { fn f<T>(&self, x: Rel<'a, T>); } }),
        [e("Rel < 'a , T >", "< 'a , T >")]
    );
    // Not for each type argument and each lifetime argument either, nor for every type parameter
    // in an argument.
    assert_eq!(
        leaked(parse_quote! {
            trait Tr<'a, 'b, T, U> { fn f(&self, x: Rel<'a, 'b, (T, Vec<U>)>); }
        }),
        [e(
            "Rel < 'a , 'b , (T , Vec < U >) >",
            "< 'a , 'b , T , U >"
        )]
    );
    // Nor for a lifetime argument which is not a lifetime parameter (`'static`).
    assert_eq!(
        leaked(parse_quote! { trait Tr { fn f<T>(&self, x: Rel<'static, T>); } }),
        [e("Rel < 'static , T >", "< T >")]
    );
    // Not for a const parameter, nor a type that is not a parameter.
    assert_eq!(
        leaked(parse_quote! {
            trait Tr<const N: usize> { fn f<'a>(&self, x: Rel<'a, N, u8>); }
        }),
        [e("Rel < 'a , N , u8 >", "< 'a , const N : usize >")]
    );
    // The lifetime arguments of the path are not related to each other.
    assert_eq!(
        leaked(parse_quote! { trait Tr { fn f<'a, 'b>(&self, x: Rel<'a, 'b>); } }),
        [e("Rel < 'a , 'b >", "< 'a , 'b >")]
    );
    // Nested: the inner path does not bound by its own lifetime either.
    assert_eq!(
        leaked(parse_quote! { trait Tr { fn f<'a, 'b, T>(&self, x: Rel<'a, Rel<'b, T>>); } }),
        [e("Rel < 'a , Rel < 'b , T > >", "< 'a , 'b , T >")]
    );
    // A bound which is written is a bound as any other, in the generics or in the where clause.
    assert_eq!(
        leaked(parse_quote! { trait Tr { fn f<'a, T: 'a>(&self, x: Rel<'a, T>); } }),
        [e("Rel < 'a , T >", "< 'a , T : 'a >")]
    );
    assert_eq!(
        leaked(parse_quote! { trait Tr { fn f<'a, T>(&self, x: Rel<'a, T>) where T: 'a; } }),
        [e("Rel < 'a , T >", "< 'a , T : 'a >")]
    );
    assert_eq!(
        leaked(
            parse_quote! { trait Tr<'a, 'b: 'a, T: 'b> { fn f(&self, x: Rel<'a, Rel<'b, T>>); } }
        ),
        [e(
            "Rel < 'a , Rel < 'b , T > >",
            "< 'a , 'b : 'a , T : 'b >"
        )]
    );
}

#[test]
fn a_reference_requires_its_referent_to_outlive_it() {
    // The reference is interned as a whole (its referent cannot stand for it) if its lifetime is
    // a generic of the Leaker.
    assert_eq!(
        leaked(parse_quote! { trait Tr<'a, T> { fn f(&self, x: &'a Vec<T>); } }),
        [e("& 'a Vec < T >", "< 'a , T : 'a >")]
    );
    assert_eq!(
        leaked(parse_quote! { trait Tr<'a, T> { fn f(&self, x: &'a mut Vec<T>); } }),
        [e("& 'a mut Vec < T >", "< 'a , T : 'a >")]
    );
    assert_eq!(
        leaked(parse_quote! { trait Tr<'a, T, U> { fn f(&self, x: &'a (Rel, T, [U; 2])); } }),
        [e("& 'a (Rel , T , [U ; 2])", "< 'a , T : 'a , U : 'a >")]
    );
    // A referent which mentions no type parameter has nothing to bound, but its lifetimes are.
    assert_eq!(
        leaked(parse_quote! { trait Tr<'a, 'b> { fn f(&self, x: &'a &'b Rel); } }),
        [e("& 'a & 'b Rel", "< 'a , 'b : 'a >")]
    );
    assert_eq!(
        leaked(parse_quote! { trait Tr<'a> { fn f(&self, x: &'a Rel); } }),
        [e("& 'a Rel", "< 'a >")]
    );
    // What a path in the referent mentions is bounded, whichever its definition (and its
    // lifetime arguments) say: `Rel<'b, T>: 'a` requires `'b: 'a` and `T: 'a`.
    assert_eq!(
        leaked(parse_quote! { trait Tr<'a, 'b, T> { fn f(&self, x: &'a Rel<'b, T>); } }),
        [e("& 'a Rel < 'b , T >", "< 'a , 'b : 'a , T : 'a >")]
    );
    // A component of a reference with a local lifetime is leaked alone, with the bound which the
    // reference implies on it (see the module `ancestors`).
    assert_eq!(
        leaked(parse_quote! { trait Tr { fn f<'a, T>(&self, x: &'a Vec<T>); } }),
        [e("Vec < T >", "< 'a , T : 'a >")]
    );
}

#[test]
fn a_reference_in_the_arguments_bounds_what_it_mentions_only() {
    // `R<'a, &'b T>`: the reference gives `T: 'b`, and the path gives nothing (`'b: 'a` and `T:
    // 'a` would be assumed for it).
    assert_eq!(
        leaked(parse_quote! { trait Tr { fn f<'a, 'b, T>(&self, x: Rel<'a, &'b T>); } }),
        [e("Rel < 'a , & 'b T >", "< 'a , 'b , T : 'b >")]
    );
    // The lifetimes in an argument which are `'static`, elided or not generic are not bounded.
    assert_eq!(
        leaked(parse_quote! { trait Tr { fn f<'a, T>(&self, x: Rel<'a, &'static T>); } }),
        [e("Rel < 'a , & 'static T >", "< 'a , T >")]
    );
    // A reference around the path bounds everything in it: the lifetimes and the types in the
    // arguments of the path, and in the reference inside.
    assert_eq!(
        leaked(parse_quote! { trait Tr<'a, 'b, 'c, T> { fn f(&self, x: &'a Rel<'b, &'c T>); } }),
        [e(
            "& 'a Rel < 'b , & 'c T >",
            "< 'a , 'b : 'a , 'c : 'a , T : 'a + 'c >"
        )]
    );
    // The bounds written are merged with these, without duplicates.
    assert_eq!(
        leaked(parse_quote! {
            trait Tr { fn f<'a, 'b: 'a, T: 'a + ::core::clone::Clone>(&self, x: Rel<'a, &'b T>); }
        }),
        [e(
            "Rel < 'a , & 'b T >",
            "< 'a , 'b : 'a , T : 'a + 'b + :: core :: clone :: Clone >"
        )]
    );
    assert_eq!(
        leaked(parse_quote! {
            trait Tr<'a, T> where T: 'a { fn f(&self, x: Rel<'a, T>); }
        }),
        [e("Rel < 'a , T >", "< 'a , T : 'a >")]
    );
}

#[test]
fn a_trait_object_requires_its_arguments_to_outlive_too() {
    // `dyn Tr<T> + 'b` outlives `'a` only if `'b` and `T` do (rustc checks each argument, also
    // with the lifetime bound `'static`, whatever `T: 'b` says), which a reference requires of
    // its referent.
    assert_eq!(
        leaked(parse_quote! {
            trait Tr<'a, T> { fn f(&self, x: &'a (dyn RelTr<T> + 'a)); }
        }),
        [e("& 'a (dyn RelTr < T > + 'a)", "< 'a , T : 'a >")]
    );
    assert_eq!(
        leaked(parse_quote! {
            trait Tr<'a, 'b, T> { fn f(&self, x: &'a (dyn ::p::Tr<T> + 'b)); }
        }),
        [e(
            "& 'a (dyn :: p :: Tr < T > + 'b)",
            "< 'a , 'b : 'a , T : 'a >"
        )]
    );
    // Also inside a path, which passes on what the reference requires.
    assert_eq!(
        leaked(parse_quote! {
            trait Tr<'a, T> { fn f(&self, x: &'a Rel<Box<dyn ::p::Tr<T>>>); }
        }),
        [e(
            "& 'a Rel < Box < dyn :: p :: Tr < T > > >",
            "< 'a , T : 'a >"
        )]
    );
    // A path with a lifetime does not require anything of its arguments.
    assert_eq!(
        leaked(parse_quote! { trait Tr<'a, T> { fn f(&self, x: Rel<'a, dyn ::p::Tr<T>>); } }),
        [e("Rel < 'a , dyn :: p :: Tr < T > >", "< 'a , T >")]
    );
    assert_eq!(
        leaked(parse_quote! {
            trait Tr<'a, 'b, T> { fn f(&self, x: Rel<'a, dyn ::p::Tr<T> + 'b>); }
        }),
        [e(
            "Rel < 'a , dyn :: p :: Tr < T > + 'b >",
            "< 'a , 'b , T >"
        )]
    );
    // A trait with a lifetime argument in a bound is not a path to assume a bound for.
    assert_eq!(
        leaked(parse_quote! {
            trait Tr<'a, T> { fn f(&self, x: Rel<Box<dyn ::p::Tr<'a, T>>>); }
        }),
        [e("Rel < Box < dyn :: p :: Tr < 'a , T > > >", "< 'a , T >")]
    );
}

/// The leaked types of a trait with the generics each uses, and the lifetimes `Self` must outlive
/// in the context of each (`Referrer::context_self_outlives()`).
fn with_self_outlives(referrer: &Referrer) -> Vec<(String, String, Vec<String>)> {
    referrer
        .iter()
        .map(|(_, ty, context)| {
            let generics = &context.generics;
            (
                s(ty),
                s(&quote!(#generics #{&generics.where_clause})),
                context.self_outlives.iter().map(s).collect(),
            )
        })
        .collect()
}

/// `Leaker::intern_trait()` docs (generic associated types): only the GATs of the trait that
/// `intern_trait()` analyzes are read. `<Self as ::p::Other>::G<'x>` names the GAT of another
/// trait, which the Leaker cannot see: only the predicate that the reference records is there, and
/// no `Self: 'x`, which the `where Self: 'b` of this trait's own `G` would give. Audit mutant
/// `g_other_trait_recorded` (the trait of the qualified path is not compared).
#[test]
fn the_gat_of_another_trait_records_the_predicate_only() {
    let own = |path: &str| {
        let item: ItemTrait = parse_str(&format!(
            "trait Tr {{ type G<'b> where Self: 'b; \
             fn m<'x>(&'x self, a: Rel<&'x {}<'x>>); }}",
            path
        ))
        .unwrap();
        let mut leaker = Leaker::new();
        leaker.intern_trait(&item, true);
        with_self_outlives(&leaker.result().finish().unwrap())
    };
    // Its own trait, by the path or by the qualified path: `Self: 'x`, as `G` says.
    for (path, shown) in [
        ("Self::G", "Self :: G"),
        ("<Self as Tr>::G", "< Self as Tr > :: G"),
    ] {
        assert_eq!(
            own(path),
            [(
                format!("Rel < & 'x {} < 'x > >", shown),
                format!("< 'x > where {} < 'x > : 'x", shown),
                vec!["'x".to_string()]
            )],
            "{}",
            path
        );
    }
    // Another trait's: the predicate only.
    assert_eq!(
        own("<Self as ::p::Other>::G"),
        [(
            "Rel < & 'x < Self as :: p :: Other > :: G < 'x > >".to_string(),
            "< 'x > where < Self as :: p :: Other > :: G < 'x > : 'x".to_string(),
            Vec::<String>::new()
        )]
    );
}

/// `Leaker::intern_trait()` docs: the GATs of the trait that `intern_trait()` analyzes are read,
/// also when the trait is nested in a module that `intern_mod()` analyzes. Audit mutant
/// `g_nested_trait_gats` (the GATs of a trait in a module are dropped).
#[test]
fn the_gats_of_a_trait_in_a_module_are_read() {
    let item: ItemMod = parse_quote! {
        mod m {
            pub trait Tr {
                type G<'b> where Self: 'b;
                fn m<'x>(&'x self, a: Rel<&'x Self::G<'x>>);
            }
        }
    };
    let mut leaker = Leaker::new();
    leaker.intern_mod(&item);
    assert_eq!(
        with_self_outlives(&leaker.result().finish().unwrap()),
        [(
            "Rel < & 'x Self :: G < 'x > >".to_string(),
            "< 'x > where Self :: G < 'x > : 'x".to_string(),
            vec!["'x".to_string()]
        )]
    );
}

/// The `Leaker` docs, Outlives bounds: the value of an associated type or GAT
/// (`type A<'a, T> = Ref<'a, Rel2<T>>;`) is analyzed like a type written anywhere else. A path with
/// a lifetime argument records no bound for what is inside it, so the copy fails like the original
/// without `T: 'a`, and the bound that is written (`type A<'a, T: 'a>`) is the one of the entry.
#[test]
fn the_value_of_an_associated_type_is_analyzed_like_the_original() {
    // Not written: nothing is guessed for `Rel2<T>` and the path. The path of an external crate
    // is an entry with `Rel2<T>` in it; where the crate is allowed, it is not.
    assert_eq!(
        leaked_allowing_crates(
            parse_quote! { trait Tr { type A<'a, T> = ::p::Ref<'a, Rel2<T>>; } }
        ),
        [e("Rel2 < T >", "< T >")]
    );
    assert_eq!(
        leaked(parse_quote! { trait Tr { type A<'a, T> = ::p::Ref<'a, Rel2<T>>; } }),
        [e(":: p :: Ref < 'a , Rel2 < T > >", "< 'a , T >")]
    );
    // Written, as the `Leaker` docs say to: both have it.
    assert_eq!(
        leaked_allowing_crates(
            parse_quote! { trait Tr { type A<'a, T: 'a> = ::p::Ref<'a, Rel2<T>>; } }
        ),
        [e("Rel2 < T >", "< 'a , T : 'a >")]
    );
    assert_eq!(
        leaked(parse_quote! { trait Tr { type A<'a, T: 'a> = ::p::Ref<'a, Rel2<T>>; } }),
        [e(":: p :: Ref < 'a , Rel2 < T > >", "< 'a , T : 'a >")]
    );
    // A relative path is leaked as a whole.
    assert_eq!(
        leaked(parse_quote! { trait Tr { type A<'a, T: 'a> = Ref<'a, Rel2<T>>; } }),
        [e("Ref < 'a , Rel2 < T > >", "< 'a , T : 'a >")]
    );
    // The bounds of the type are read too, with the reference they write.
    assert_eq!(
        leaked(parse_quote! {
            trait Tr { type A<'a, T: 'a>: ::core::convert::Into<&'a Rel2<T>>; }
        }),
        [e("Rel2 < T >", "< 'a , T : 'a >")]
    );
}

#[test]
fn binder_lifetimes_are_not_generics() {
    // `'x` is declared by the binder, which the Repeater's implementation cannot refer to: `T:
    // 'x` is not a bound of `T`.
    assert_eq!(
        leaked(parse_quote! { trait Tr<T> { fn f(&self, x: for<'x> fn(Rel<'x, T>)); } }),
        [e("Rel < 'x , T >", "< 'x , T >")]
    );
    assert_eq!(
        leaked(parse_quote! { trait Tr<T> { fn f(&self, x: for<'x> fn(&'x Rel<T>)); } }),
        [e("Rel < T >", "< T >")]
    );
    // Nor is `'x: 'a` for the referent of a reference with a generic lifetime.
    assert_eq!(
        leaked(parse_quote! {
            trait Tr<'a, T> { fn f(&self, x: &'a Rel<for<'x> fn(&'x T) -> &'x ()>); }
        }),
        [e(
            "& 'a Rel < for < 'x > fn (& 'x T) -> & 'x () >",
            "< 'a , T : 'a >"
        )]
    );
    // The same for a path with a generic lifetime, which bounds nothing.
    assert_eq!(
        leaked(parse_quote! {
            trait Tr<'a, T> { fn f(&self, x: Rel<'a, for<'x> fn(&'x T) -> &'x ()>); }
        }),
        [e(
            "Rel < 'a , for < 'x > fn (& 'x T) -> & 'x () >",
            "< 'a , T >"
        )]
    );
    // Nor in a `dyn for<'x> Tr<'x, ..>`, whose arguments are looked into like any other (so `T`
    // is bounded by the reference's `'a`), and in which a path with the lifetime does not bound
    // by it.
    assert_eq!(
        leaked(parse_quote! {
            trait Tr<'a, T> {
                fn f(&self, x: &'a Box<dyn for<'x> ::p::Tr<'x, Rel<'x, T>>>);
            }
        }),
        [e(
            "& 'a Box < dyn for < 'x > :: p :: Tr < 'x , Rel < 'x , T > > >",
            "< 'a , T : 'a >"
        )]
    );
    assert_eq!(
        leaked(parse_quote! {
            trait Tr<'a, T> {
                fn f(&self, x: Rel<'a, Box<dyn for<'x> ::p::Tr<'x, Rel<'x, T>>>>);
            }
        }),
        [e(
            "Rel < 'a , Box < dyn for < 'x > :: p :: Tr < 'x , Rel < 'x , T > > > >",
            "< 'a , T >"
        )]
    );
}

#[test]
fn bounds_are_part_of_the_context() {
    // The context of the type differs from the top-level context of the trait (which has the
    // generics as they are written, and which `All` uses), although it uses all of them: the
    // reference implies `T: 'a`.
    let tr: ItemTrait = parse_quote! {
        trait Tr<'a, T> { fn f(&self, x: &'a Rel<T>); fn g(&self) -> All<'a, T>; }
    };
    let mut leaker = Leaker::new();
    let top_level = leaker.intern_trait(&tr, true);
    let referrer = leaker.result().finish().unwrap();
    let at = |text: &str| {
        common::types(&referrer)
            .position(|ty| s(ty) == text)
            .unwrap()
    };
    let (rel, all) = (at("& 'a Rel < T >"), at("All < 'a , T >"));
    assert_ne!(common::entry_id(&referrer, rel), top_level);
    assert_eq!(common::entry_id(&referrer, all), top_level);
    assert_eq!(s(&common::entry(&referrer, all).generics), "< 'a , T >");
    assert_eq!(
        s(&common::entry(&referrer, rel).generics),
        "< 'a , T : 'a >"
    );
    // A path with the lifetime implies none, so the context is the top-level one.
    let tr: ItemTrait = parse_quote! { trait Tr<'a, T> { fn f(&self, x: Rel<'a, T>); } };
    let mut leaker = Leaker::new();
    let top_level = leaker.intern_trait(&tr, true);
    let referrer = leaker.result().finish().unwrap();
    assert_eq!(common::entry_id(&referrer, 0), top_level);
    assert_eq!(s(&common::entry(&referrer, 0).generics), "< 'a , T >");
    // A struct with the bound written is the same context as the type.
    let st: ItemStruct = parse_quote! { struct S<'a, T: 'a>(Rel<'a, T>); };
    let mut leaker = Leaker::new();
    let top_level = leaker.intern_struct(&st, true);
    let referrer = leaker.result().finish().unwrap();
    assert_eq!(common::entry_id(&referrer, 0), top_level);
    // Without it, the context is that of the generics as they are as well: the path implies none.
    let st: ItemStruct = parse_quote! { struct S<'a, T>(Rel<'a, T>); };
    let mut leaker = Leaker::new();
    let top_level = leaker.intern_struct(&st, true);
    let referrer = leaker.result().finish().unwrap();
    assert_eq!(common::entry_id(&referrer, 0), top_level);
    assert_eq!(s(&common::entry(&referrer, 0).generics), "< 'a , T >");
    // Two declarations of the same meaning give the same context (the bound is not duplicated).
    let a: ItemTrait = parse_quote! { trait Tr { fn f<'a, T: 'a>(&self, x: Rel<'a, T>); } };
    let b: ItemTrait = parse_quote! { trait Tr { fn f<'a, T>(&self, x: Rel<'a, T>) where T: 'a; } };
    let contexts = |item: &ItemTrait| -> Vec<ContextId> {
        let mut leaker = Leaker::new();
        leaker.intern_trait(item, true);
        let referrer = leaker.result().finish().unwrap();
        vec![common::entry_id(&referrer, 0)]
    };
    assert_eq!(contexts(&a), contexts(&b));
    assert_eq!(leaked(a), leaked(b));
    // The same for a bound which a reference implies, and the same bound written: one context,
    // so one entry of the same Leaker.
    let written: ItemTrait = parse_quote! {
        trait Tr { fn f<'a, T: 'a>(&self, x: &'a Rel<T>); }
    };
    let implied: ItemTrait = parse_quote! {
        trait Tr { fn f<'a, T>(&self, x: &'a Rel<T>); }
    };
    let mut leaker = Leaker::new();
    leaker.intern_trait(&written, true);
    leaker.intern_trait(&implied, true);
    let referrer = leaker.result().finish().unwrap();
    assert_eq!(referrer.iter().count(), 1);
    assert_eq!(s(&common::entry(&referrer, 0).generics), "< 'a , T : 'a >");
    assert_eq!(leaked(written), leaked(implied));
}

#[test]
fn the_referrer_finds_the_entries_with_the_generics_it_sees_declared() {
    // The Referrer sees the generics of the context, and those the copy declares: the bounds the
    // type implies on the declared generics must be the Leaker's.
    for tr in [
        // `T` is declared by the method and `'a` is not.
        parse_quote! { trait Tr<'a> { fn f<T>(&self, x: &'a Rel<T>); } },
        // `'b` is declared by the method and `T` and `'a` are not.
        parse_quote! { trait Tr<'a, T> { fn f<'b>(&self, x: &'b Rel<(&'a T, u8)>); } },
        // `T` is bounded by the lifetime written at the method.
        parse_quote! { trait Tr { fn f<'a, T: 'a>(&self, x: Rel<'a, T>); } },
        parse_quote! { trait Tr { fn f<'a, T: 'a>(&self, x: &'a Rel<T>); } },
        // Nothing is declared.
        parse_quote! { trait Tr<'a, T> { fn f(&self, x: &'a Rel<T>); } },
        // The path implies no bound, so the entry has none either.
        parse_quote! { trait Tr<'a> { fn f<T>(&self, x: Rel<'a, T>); } },
        parse_quote! { trait Tr<'a, T> { fn f(&self, x: Rel<'a, T>); } },
    ] {
        let tr: ItemTrait = tr;
        let mut leaker = Leaker::new();
        let context = leaker.intern_trait(&tr, true);
        let referrer = leaker.result().finish().unwrap();
        let rewrite = |visitor: &mut dyn syn::visit_mut::VisitMut| {
            let mut copy = tr.clone();
            visitor.visit_item_trait_mut(&mut copy);
            s(&copy)
        };
        let replace = |_: Type, ix: EntryId, _: &Context| -> Type { parse_quote!(P<#ix>) };
        // The copy declares the generics of the trait as they are: it is the scope itself.
        let copy = rewrite(&mut referrer.visitor(context, replace));
        assert!(!copy.contains("Rel"), "{}", copy);
        // The generics of the trait are those of the context, whatever the copy declares.
        let copy = rewrite(
            &mut referrer
                .visitor(context, replace)
                .with_outermost_generics_as_leakers(),
        );
        assert!(!copy.contains("Rel"), "{}", copy);
    }
    // A signature which declares them differently from the entry (`T` without the bound `'a`
    // which the entry has) is not rewritten, as it refers to other generics.
    let tr: ItemTrait = parse_quote! { trait Tr { fn f<'a, T: 'a>(&self, x: Rel<'a, T>); } };
    let mut leaker = Leaker::new();
    let context = leaker.intern_trait(&tr, true);
    let referrer = leaker.result().finish().unwrap();
    let mut sig: Signature = parse_quote!(fn f<'a, T: ::core::clone::Clone>(x: Rel<'a, T>));
    syn::visit_mut::VisitMut::visit_signature_mut(
        &mut referrer.visitor(context, |_, ix, _| parse_quote!(P<#ix>)),
        &mut sig,
    );
    assert!(s(&sig).contains("Rel"), "{}", s(&sig));
    // The one without the bound, for a path which implies none, is not the entry which has it.
    let mut sig: Signature = parse_quote!(fn f<'a, T>(x: Rel<'a, T>));
    syn::visit_mut::VisitMut::visit_signature_mut(
        &mut referrer.visitor(context, |_, ix, _| parse_quote!(P<#ix>)),
        &mut sig,
    );
    assert!(s(&sig).contains("Rel"), "{}", s(&sig));
    // A reference implies the bound `T: 'a`, so the signature which does not write it is the
    // entry's, and one with another bound is not.
    let tr: ItemTrait = parse_quote! { trait Tr { fn f<'a, T>(&self, x: &'a Rel<T>); } };
    let mut leaker = Leaker::new();
    let context = leaker.intern_trait(&tr, true);
    let referrer = leaker.result().finish().unwrap();
    let mut sig: Signature = parse_quote!(fn f<'a, T>(x: &'a Rel<T>));
    syn::visit_mut::VisitMut::visit_signature_mut(
        &mut referrer.visitor(context, |_, ix, _| parse_quote!(P<#ix>)),
        &mut sig,
    );
    assert!(!s(&sig).contains("Rel"), "{}", s(&sig));
    let mut sig: Signature = parse_quote!(fn f<'a, T: ::core::clone::Clone>(x: &'a Rel<T>));
    syn::visit_mut::VisitMut::visit_signature_mut(
        &mut referrer.visitor(context, |_, ix, _| parse_quote!(P<#ix>)),
        &mut sig,
    );
    assert!(s(&sig).contains("Rel"), "{}", s(&sig));
}

#[test]
fn the_bounds_survive_serialization() {
    let tr: ItemTrait = parse_quote! {
        trait Tr<'a, T> {
            fn f<'b, U>(&self, x: &'b Rel<(&'a T, U)>) -> &'a Rel2<T>;
        }
    };
    let mut leaker = Leaker::new();
    leaker.intern_trait(&tr, true);
    let referrer = leaker.result().finish().unwrap();
    let tokens = referrer.to_token_stream();
    assert!(tokens.to_string().contains("'a : 'b"), "{}", tokens);
    let parsed: Referrer = parse2(tokens).unwrap();
    assert_eq!(parsed, referrer);
    assert_eq!(entries(&parsed), entries(&referrer));
}

/// The generics of the first entry of a trait with the Repeater's implementation declaring them
/// (`repeater_impls()`), and those the trait arguments encode (`context_generics()`).
fn impl_and_context(item: ItemTrait) -> (String, String) {
    let mut leaker = Leaker::new();
    leaker.intern_trait(&item, true);
    let result = leaker.result();
    let referrer = result.clone().finish().unwrap();
    let marker: Type = parse_quote!(crate::Marker);
    let r = result
        .repeater_impls(
            &parse_quote!(crate::Repeater),
            7,
            ImplSelf::Marker(&marker),
            array_const,
        )
        .unwrap()
        .next()
        .unwrap();
    let with_where = |g: &Generics| s(&quote!(#g #{&g.where_clause}));
    (
        with_where(&r.generics),
        with_where(&common::entry(&referrer, 0).generics),
    )
}

fn pair(impl_generics: &str, context_generics: &str) -> (String, String) {
    (impl_generics.to_string(), context_generics.to_string())
}

#[test]
fn impl_generics_drop_the_encoded_bounds() {
    // `&'a Rel<T>` gives the context generics `<'a, T: 'a>`, which the trait arguments encode
    // (`&'a T`), so the implementation declares `<'a, T>`.
    assert_eq!(
        impl_and_context(parse_quote! { trait Tr<'a, T> { fn f(&self, x: &'a Rel<T>); } }),
        pair("< 'a , T >", "< 'a , T : 'a >")
    );
    // `'b: 'a` is encoded (`&'a &'b ()`) as well.
    assert_eq!(
        impl_and_context(parse_quote! {
            trait Tr<'a, 'b, T> { fn f(&self, x: &'a &'b Rel<T>); }
        }),
        pair("< 'a , 'b , T >", "< 'a , 'b : 'a , T : 'a + 'b >")
    );
    // A path with a lifetime implies no bound: nothing to drop.
    assert_eq!(
        impl_and_context(parse_quote! { trait Tr<'a, T> { fn f(&self, x: Rel<'a, T>); } }),
        pair("< 'a , T >", "< 'a , T >")
    );
    assert_eq!(
        impl_and_context(parse_quote! {
            trait Tr { fn f<'a, 'b, T>(&self, x: Rel<'a, Rel<'b, T>>); }
        }),
        pair("< 'a , 'b , T >", "< 'a , 'b , T >")
    );
    // A bound written by the Leaker is not special.
    assert_eq!(
        impl_and_context(parse_quote! { trait Tr { fn f<'a, T: 'a>(&self, x: Rel<'a, T>); } }),
        pair("< 'a , T >", "< 'a , T : 'a >")
    );
    // `'static` is not a lifetime parameter, so it is kept, as are the trait bounds, in the order
    // of the canonical form, and the other bounds of the same parameter are kept.
    let (declared, context) = impl_and_context(parse_quote! {
        trait Tr { fn f<'a, T: 'static + ::core::clone::Clone + 'a>(&self, x: Rel<'a, T>); }
    });
    assert_eq!(context.matches("'a").count(), 2, "{}", context);
    assert!(
        context.contains("'static") && context.contains("Clone"),
        "{}",
        context
    );
    assert!(
        declared.contains("'static") && declared.contains("Clone"),
        "{}",
        declared
    );
    assert_eq!(declared.matches("'a").count(), 1, "{}", declared);
    assert_eq!(
        impl_and_context(
            parse_quote! { trait Tr { fn f<T: 'static>(&self, x: Rel<'static, T>); } }
        ),
        pair("< T : 'static >", "< T : 'static >")
    );
    // A where predicate on a type which is not a parameter is kept, as is one the bounds of the
    // parameters do not take.
    assert_eq!(
        impl_and_context(parse_quote! {
            trait Tr { fn f<'a, T>(&self, x: &'a Rel<T>) where ::std::vec::Vec<T>: ::core::clone::Clone; }
        }),
        pair(
            "< 'a , T > where :: std :: vec :: Vec < T > : :: core :: clone :: Clone",
            "< 'a , T : 'a > where :: std :: vec :: Vec < T > : :: core :: clone :: Clone"
        )
    );
    assert_eq!(
        impl_and_context(parse_quote! {
            trait Tr { fn f<'a, T>(&self, x: Rel<'a, T>) where ::std::vec::Vec<T>: ::core::clone::Clone; }
        }),
        pair(
            "< 'a , T > where :: std :: vec :: Vec < T > : :: core :: clone :: Clone",
            "< 'a , T > where :: std :: vec :: Vec < T > : :: core :: clone :: Clone"
        )
    );
    // Nothing to drop: the generics as they are.
    assert_eq!(
        impl_and_context(
            parse_quote! { trait Tr<T: ::core::clone::Clone> { fn f(&self, x: Rel<T>); } }
        ),
        pair(
            "< T : :: core :: clone :: Clone >",
            "< T : :: core :: clone :: Clone >"
        )
    );
    // The generics of the top-level context are the call's (no bound is implied there).
    let tr: ItemTrait = parse_quote! { trait Tr<'a, T> { fn f(&self, x: &'a Rel<T>); } };
    let mut leaker = Leaker::new();
    leaker.intern_trait(&tr, true);
    let referrer = leaker.result().finish().unwrap();
    // The context generics are not changed.
    assert_eq!(s(&common::entry(&referrer, 0).generics), "< 'a , T : 'a >");
}

mod ancestors {
    //! The lifetimes which the references around a type make it outlive are carried by the type when it
    //! is leaked alone (see `Referrer::context_generics()`): `Rel2<T>` of `&'a Vec<Rel2<T>>` is leaked with
    //! `<'a, T: 'a>`, as `&'a X` is well-formed only if `X: 'a` (the language rule), which holds for
    //! the original but not through the projection which the copy has instead of `Rel2<T>`.
    //!
    //! Only a reference adds a lifetime: the lifetime argument of a path (`Ref<'a, Rel2<T>>`) does
    //! not, as it is well-formed without `T: 'a` unless the definition of the path says so, which is
    //! not visible (`struct Unrelated<'a, T>(&'a (), T)`), so the bound is not assumed: `Rel2<T>` of
    //! `Ref<'a, Rel2<T>>` is leaked with `<T>`, and the bound is written where it is needed (`T: 'a`
    //! in the signature, which makes it `<'a, T: 'a>` as written). A lifetime which is active (that of
    //! a reference around) is passed down through the components, whatever they are (the type
    //! arguments of a path, the elements of a tuple, the inputs and the output of a function pointer,
    //! the arguments of a trait object, ..), as rustc does.
    //!
    //! The Leaker and the Referrer compute it from the same rules (which components of a type carry
    //! what the type outlives, see the table below), so that the Referrer finds the entry, whether the
    //! generics are declared by the copy or are those of the context (the scope of the copy).

    use super::s;
    use syn::visit_mut::VisitMut;
    use syn::*;
    use template_quote::{quote, ToTokens};
    use type_leak::{
        Allow, Context, ContextId, EntryId, Leaker, Options, Referrer, Rules, SelfRequirement,
    };

    /// Replaces an entry by `P<ix>`, as a macro replaces it by its path through the Repeater.
    fn p(_: Type, ix: EntryId, _: &Context) -> Type {
        let ix = proc_macro2::Literal::usize_unsuffixed(ix.index());
        parse_quote!(P<#ix>)
    }

    /// The paths of external crates (`::std::vec::Vec`) are allowed: they are no entries, so the entry
    /// of a type is the one in them.
    fn leaker() -> Leaker<impl Allow> {
        let config = Rules::new().primitive().prelude().external_crates();
        Leaker::from_config(config, Options::new())
    }

    fn referrer_of(item: &ItemTrait) -> Referrer {
        referrer_in(item).0
    }

    /// The Referrer of a Leaker that does not allow the external crates: their paths are optional
    /// entries.
    fn referrer_with_optional_entries(item: &ItemTrait) -> Referrer {
        let mut leaker = Leaker::from_config(Rules::new().primitive().prelude(), Options::new());
        leaker.intern_trait(item, true);
        leaker.result().finish().unwrap()
    }

    /// The Referrer, with the top-level context of the trait: the scope of its copies.
    fn referrer_in(item: &ItemTrait) -> (Referrer, ContextId) {
        let mut leaker = leaker();
        let context = leaker.intern_trait(item, true);
        (leaker.result().finish().unwrap(), context)
    }

    /// The generics (and the where clause) of each leaked type, by its tokens.
    fn contexts_of(referrer: &Referrer, ty: &str) -> Vec<String> {
        referrer
            .iter()
            .map(|(_, ty, context)| (ty, &context.generics))
            .filter(|(leaked, _)| s(leaked) == ty)
            .map(|(_, generics)| s(&quote!(#generics #{&generics.where_clause})))
            .collect()
    }

    /// The contexts of `Rel2<T>` in the method `f` of a trait whose parameter is `x: construct`
    /// (`{X}` in it is `Rel2<T>`), with the generics `generics`.
    fn rel2_in(generics: &str, construct: &str) -> Vec<String> {
        let construct = construct.replace("{X}", "Rel2<T>");
        let tr: ItemTrait = parse_str(&format!(
            "trait Tr {{ fn f{}(&self, x: {}); }}",
            generics, construct
        ))
        .unwrap();
        contexts_of(&referrer_of(&tr), "Rel2 < T >")
    }

    /// The components which carry what a type must outlive down, and those which do not, each
    /// verified with rustc (the field of a struct with the type has the bound inferred, and its copy
    /// with a projection in place of `Rel2<T>` is rejected with E0309 without it): `&'a X` requires
    /// `X: 'a`, and so does what is made of types, but a projection. A path with a lifetime argument
    /// passes down what is required of it (`&'a Ref<'b, X>`), but does not add its own lifetime. See
    /// `Enclosing` in the source.
    #[test]
    fn the_requirement_goes_down_through_the_components_it_goes_through_in_rustc() {
        const G: &str = "<'a, 'b, T>";
        let bounded = "< 'a , T : 'a >";
        for construct in [
            // A reference, and a raw pointer, a slice, an array, a tuple and parentheses in it.
            "&'a {X}",
            "&'a mut {X}",
            "&'a *const {X}",
            "&'a *mut {X}",
            "&'a [{X}]",
            "&'a [{X}; 3]",
            "&'a ({X}, u8)",
            "&'a (u8, {X})",
            "&'a ({X})",
            // The type arguments of a path with a lifetime argument (the one of the reference is the
            // active lifetime), and of a path without.
            "&'a ::core::cell::Ref<'a, {X}>",
            "&'a ::core::cell::Ref<'static, ::std::vec::Vec<{X}>>",
            "&'a ::std::option::Option<::std::vec::Vec<{X}>>",
            "&'a ::core::cell::Cell<{X}>",
            "&'a mut ::std::boxed::Box<{X}>",
            // The inputs and the output of a function pointer, with a binder or not.
            "&'a fn({X})",
            "&'a fn() -> {X}",
            "&'a for<'x> fn(&'x {X}) -> &'x u8",
            "&'a fn() -> ({X}, u8)",
            // The arguments of a trait object: `dyn Tr<X> + 'b: 'a` requires `X: 'a` as well as
            // `'b: 'a`, whatever the lifetime bound is (also `'static`).
            "&'a ::core::cell::Ref<'b, dyn ::core::iter::Iterator<Item = {X}>>",
            "&'a dyn ::core::iter::Iterator<Item = {X}>",
            "&'a (dyn ::core::convert::AsRef<{X}> + 'static)",
            "&'a dyn for<'x> ::core::ops::Fn(&'x {X})",
            "&'a ::std::boxed::Box<dyn ::core::ops::Fn({X}) -> u8>",
            "&'a ::std::boxed::Box<dyn ::core::ops::Fn() -> {X}>",
            "&'a ::std::vec::Vec<::std::boxed::Box<dyn ::core::iter::Iterator<Item = {X}>>>",
        ] {
            assert_eq!(rel2_in(G, construct), [bounded], "{}", construct);
        }
        // Not through a projection: `<X as Tr>::A` outlives `'a` if its components do, but it does
        // not require `X: 'a` (it does not for the original struct either).
        for construct in [
            "&'a <{X} as ::core::ops::Deref>::Target",
            "&'a <u8 as ::core::convert::Into<{X}>>::Output",
            "&'a ::std::vec::Vec<<{X} as ::core::ops::Deref>::Target>",
        ] {
            assert_eq!(rel2_in(G, construct), ["< T >"], "{}", construct);
        }
        // Nor through the length of an array (an expression, whose types are not components).
        assert_eq!(
            contexts_of(
                &referrer_of(&parse_quote! {
                    trait Tr { fn f<'a, T>(&self, x: &'a [u8; ::core::mem::size_of::<Rel2<T>>()]); }
                }),
                "Rel2 < T >"
            ),
            Vec::<String>::new(),
            "the type in an array length is not leaked alone"
        );
    }

    /// The `Leaker` docs, Outlives bounds: `impl Trait` does not pass the lifetime on, as an associated
    /// type path does not (the type behind it is opaque to the signature), while a trait object does.
    /// Audit mutant `o_impl_trait_passes` (`impl Trait` treated as a path that passes the lifetime on).
    #[test]
    fn impl_trait_does_not_pass_the_lifetime_on() {
        // The reference, around an argument-position `impl Trait`: `Rel2<T>` in its binding is no
        // `T: 'a`.
        for construct in [
            "&'a (impl ::core::iter::Iterator<Item = {X}>)",
            "&'a mut (impl ::core::iter::Iterator<Item = {X}>)",
            "&'a (::std::vec::Vec<u8>, impl ::core::iter::Iterator<Item = {X}>)",
        ] {
            assert_eq!(rel2_in("<'a, T>", construct), ["< T >"], "{}", construct);
        }
        // The same with a trait object, which does pass it on.
        assert_eq!(
            rel2_in("<'a, T>", "&'a (dyn ::core::iter::Iterator<Item = {X}>)"),
            ["< 'a , T : 'a >"]
        );
        // And the trait of the generics (`trait Tr<'a, T>`) says the same.
        let tr: ItemTrait = parse_quote! {
            trait Tr<'a, T> {
                fn f(&self, x: &'a (impl ::core::iter::Iterator<Item = Rel2<T>>));
            }
        };
        assert_eq!(contexts_of(&referrer_of(&tr), "Rel2 < T >"), ["< T >"]);
    }

    #[test]
    fn every_lifetime_around_the_type_bounds_it() {
        // Both lifetimes of `&'a &'b X`, in the canonical order.
        assert_eq!(
            rel2_in("<'a, 'b, T>", "&'a &'b {X}"),
            ["< 'a , 'b , T : 'a + 'b >"]
        );
        assert_eq!(
            rel2_in("<'a, 'b, T>", "&'b &'a {X}"),
            ["< 'a , 'b , T : 'a + 'b >"]
        );
        // A path with its own lifetime argument under the reference passes the lifetime of the
        // reference down, and adds none (`'b` is not among the generics, nor bounds the type).
        assert_eq!(
            rel2_in(
                "<'a, 'b, T>",
                "&'a ::core::cell::Ref<'b, ::std::vec::Vec<{X}>>"
            ),
            ["< 'a , T : 'a >"]
        );
        // Nested: the inner reference's lifetime bounds the type too, as does what is outside of it,
        // and the lifetime arguments of the paths between them do not.
        assert_eq!(
            rel2_in(
                "<'a, 'b, T>",
                "&'a ::core::cell::Ref<'a, &'b ::core::cell::Ref<'b, {X}>>"
            ),
            ["< 'a , 'b , T : 'a + 'b >"]
        );
        // The lifetimes in the type are bounded as well (`'c: 'a`), by the reference only.
        let tr: ItemTrait = parse_quote! {
            trait Tr { fn f<'a, 'c>(&self, x: &'a Rel2<&'c u8>); }
        };
        assert_eq!(
            contexts_of(&referrer_of(&tr), "Rel2 < & 'c u8 >"),
            ["< 'a , 'c : 'a >"]
        );
        let tr: ItemTrait = parse_quote! {
            trait Tr { fn f<'a, 'c>(&self, y: ::core::cell::Ref<'a, Rel2<&'c u8>>); }
        };
        assert_eq!(
            contexts_of(&referrer_of(&tr), "Rel2 < & 'c u8 >"),
            ["< 'c >"],
            "a path's own lifetime is not related to the lifetimes in its argument"
        );
        // A type which mentions no generic has nothing to bound: its context has no generics.
        let tr: ItemTrait = parse_quote! {
            trait Tr { fn f<'a>(&self, x: &'a Rel, y: ::core::cell::Ref<'a, Rel>); }
        };
        assert_eq!(contexts_of(&referrer_of(&tr), "Rel"), [""]);
        // The bounds which are written are merged with them, and so are those from the inside of the
        // type (the bound is not duplicated).
        assert_eq!(
            rel2_in("<'a, T: 'a + ::core::clone::Clone>", "&'a {X}"),
            ["< 'a , T : 'a + :: core :: clone :: Clone >"]
        );
        // `Rel<'a, T>` is leaked as written inside a reference, as it is under a path.
        let tr: ItemTrait = parse_quote! {
            trait Tr { fn f<'a, T>(&self, x: &'a Rel<'a, T>); }
        };
        assert_eq!(
            contexts_of(&referrer_of(&tr), "Rel < 'a , T >"),
            ["< 'a , T : 'a >"]
        );
        let tr: ItemTrait = parse_quote! {
            trait Tr { fn f<'a, T>(&self, y: ::core::cell::Ref<'a, Rel<'a, T>>); }
        };
        assert_eq!(
            contexts_of(&referrer_of(&tr), "Rel < 'a , T >"),
            ["< 'a , T >"]
        );
        // The generics of the trait: the reference is leaked as a whole if its lifetime is a generic
        // of the trait, except when it is to a trait object.
        let tr: ItemTrait = parse_quote! {
            trait Tr<'a, T> { fn f(&self, x: &'a dyn ::core::iter::Iterator<Item = Rel2<T>>); }
        };
        assert_eq!(
            contexts_of(&referrer_of(&tr), "Rel2 < T >"),
            ["< 'a , T : 'a >"]
        );
        // A lifetime of the trait and a type of the method, and the other way round.
        let tr: ItemTrait = parse_quote! {
            trait Tr<'a> { fn f<T>(&self, x: &'a dyn ::core::iter::Iterator<Item = Rel2<T>>); }
        };
        assert_eq!(
            contexts_of(&referrer_of(&tr), "Rel2 < T >"),
            ["< 'a , T : 'a >"]
        );
        let tr: ItemTrait = parse_quote! {
            trait Tr<T> { fn f<'a>(&self, x: &'a dyn ::core::iter::Iterator<Item = Rel2<T>>); }
        };
        assert_eq!(
            contexts_of(&referrer_of(&tr), "Rel2 < T >"),
            ["< 'a , T : 'a >"]
        );
    }

    /// A path with a lifetime argument does not make the types inside it outlive the lifetime: it is
    /// well-formed without `T: 'a` unless its definition says so (it is not visible, and `struct
    /// Unrelated<'a, T>(&'a (), T)` does not), so no bound is assumed, neither for the path itself
    /// nor for the types inside it. A written bound is a bound like any other.
    #[test]
    fn a_path_with_a_lifetime_argument_adds_no_bound() {
        // The type inside the path.
        assert_eq!(rel2_in("<'a, T>", "::core::cell::Ref<'a, {X}>"), ["< T >"]);
        assert_eq!(
            rel2_in("<'a, T>", "::core::cell::Ref<'a, ::std::vec::Vec<{X}>>"),
            ["< T >"]
        );
        assert_eq!(
            rel2_in(
                "<'a, T>",
                "::core::cell::Ref<'a, dyn ::core::iter::Iterator<Item = {X}>>"
            ),
            ["< T >"]
        );
        // Also with several lifetime arguments, and in the arguments of a path inside it.
        assert_eq!(
            rel2_in("<'a, 'b, T>", "::p::Two<'a, 'b, ::p::Wrap<{X}>>"),
            ["< T >"]
        );
        // A written bound stays as it is written.
        assert_eq!(
            rel2_in("<'a, T: 'a>", "::core::cell::Ref<'a, {X}>"),
            ["< 'a , T : 'a >"]
        );
        // The path itself, which is leaked as a whole when it is relative: the generics it uses, with
        // no bound between them, whatever the lifetime and the types are. (`Rel<'a, T>`,
        // `Rel<'a, Rel<'b, T>>` and the written `T: 'a` are in
        // `a_path_with_a_lifetime_argument_assumes_no_bound`.)
        for (generics, construct, leaked, expected) in [
            (
                "<'a, 'b, T>",
                "Rel<'a, 'b, (T, u8)>",
                "Rel < 'a , 'b , (T , u8) >",
                "< 'a , 'b , T >",
            ),
            (
                "<'a, 'b>",
                "Rel<'a, &'b u8>",
                "Rel < 'a , & 'b u8 >",
                "< 'a , 'b >",
            ),
            // The lifetimes of the arguments are bounded by a reference only (the path is a
            // component of it, leaked alone since the lifetime of the reference is a local one).
            (
                "<'a, 'b, T>",
                "&'a Rel<'b, T>",
                "Rel < 'b , T >",
                "< 'a , 'b : 'a , T : 'a >",
            ),
        ] {
            let tr: ItemTrait = parse_str(&format!(
                "trait Tr {{ fn f{}(&self, x: {}); }}",
                generics, construct
            ))
            .unwrap();
            assert_eq!(
                contexts_of(&referrer_of(&tr), leaked),
                [expected],
                "{}",
                construct
            );
        }
        // The lifetime of a reference around is passed down through the path, and the path's own is
        // not added: the type inside is leaked with the first only.
        assert_eq!(
            rel2_in("<'a, 'b, T>", "&'a ::core::cell::Ref<'b, {X}>"),
            ["< 'a , T : 'a >"]
        );
        assert_eq!(
            rel2_in("<'a, 'b, T>", "&'a Vec<::core::cell::Ref<'b, {X}>>"),
            ["< 'a , T : 'a >"]
        );
        // Neither does `Self`: `Self: 'a` comes from a reference or is written.
        let referrer = referrer_of(&parse_quote! {
            trait Tr {
                fn f<'a>(&self, x: ::core::cell::Ref<'a, Rel2<Self>>, y: Rel<'a, Self>)
                where Self: Sized;
            }
        });
        for ix in 0..referrer.iter().count() {
            assert!(
                crate::common::entry(&referrer, ix)
                    .self_outlives
                    .as_slice()
                    .is_empty(),
                "{}",
                s(crate::common::types(&referrer).nth(ix).unwrap())
            );
        }
        assert_eq!(
            contexts_of(&referrer, "Rel < 'a , Self >"),
            ["< 'a > where Self : Sized"]
        );
    }

    #[test]
    fn a_const_parameter_is_not_bounded() {
        // Nothing to bound by the lifetime: it is not among the generics of the type (it would only
        // be a generic nothing refers to).
        let tr: ItemTrait = parse_quote! {
            trait Tr<const N: usize> {
                fn f<'a>(&self, x: &'a Rel2<N>, y: ::core::cell::Ref<'a, Rel2<N>>);
            }
        };
        assert_eq!(
            contexts_of(&referrer_of(&tr), "Rel2 < N >"),
            ["< const N : usize >"]
        );
        // With a type parameter next to it, only the type parameter is.
        let tr: ItemTrait = parse_quote! {
            trait Tr<const N: usize> {
                fn f<'a, T>(&self, y: &'a dyn ::core::iter::Iterator<Item = Rel2<N, T>>);
            }
        };
        assert_eq!(
            contexts_of(&referrer_of(&tr), "Rel2 < N , T >"),
            ["< 'a , T : 'a , const N : usize >"]
        );
    }

    #[test]
    fn what_is_not_a_generic_lifetime_bounds_nothing() {
        // `'static`, `'_` (which is not leaked alone).
        assert_eq!(
            rel2_in("<T: 'static>", "::core::cell::Ref<'static, {X}>"),
            ["< T : 'static >"]
        );
        assert_eq!(rel2_in("<T>", "&'static {X}"), ["< T >"]);
        // The lifetime of a `for<..>` binder: the copy cannot declare it, and rustc does not need it.
        assert_eq!(rel2_in("<T>", "for<'x> fn(&'x {X}) -> usize"), ["< T >"]);
        assert_eq!(
            rel2_in(
                "<T>",
                "::std::boxed::Box<dyn for<'x> ::core::ops::Fn(&'x {X}) -> u8>"
            ),
            ["< T >"]
        );
        // Not across it: what the types around the binder require stays (`'a` is not bound by it).
        assert_eq!(
            rel2_in("<'a, T>", "&'a for<'x> fn(&'x {X}) -> usize"),
            ["< 'a , T : 'a >"]
        );
        // A binder which shadows nothing here, with the lifetime of the trait declared as a binder.
        assert_eq!(
            rel2_in("<'a, T>", "for<'a> fn(&'a {X}) -> usize"),
            ["< T >"]
        );
        // A lifetime which no scope declares is not a generic.
        assert_eq!(rel2_in("<T>", "&'_ {X}"), ["< T >"]);
        // With nothing around it.
        assert_eq!(rel2_in("<T>", "{X}"), ["< T >"]);
        assert_eq!(
            rel2_in("<'a, T>", "(&'a u8, {X})"),
            ["< T >"],
            "the lifetime of a neighbour is not around it"
        );
    }

    #[test]
    fn the_same_type_under_different_lifetimes_is_another_entry() {
        let tr: ItemTrait = parse_quote! {
            trait Tr<'a, T> {
                fn f(
                    &self,
                    x: &'a dyn ::core::iter::Iterator<Item = Rel2<T>>,
                    y: Rel2<T>,
                    z: &'a Rel2<T>,
                );
            }
        };
        let referrer = referrer_of(&tr);
        let mut contexts = contexts_of(&referrer, "Rel2 < T >");
        contexts.sort();
        // The reference is leaked as a whole (it has a lifetime of the trait).
        assert_eq!(contexts, ["< 'a , T : 'a >", "< T >"]);
        assert_eq!(
            contexts_of(&referrer, "& 'a Rel2 < T >"),
            ["< 'a , T : 'a >"]
        );
        // A path with the lifetime is not another entry: it is the one of the type with nothing
        // around it.
        let tr: ItemTrait = parse_quote! {
            trait Tr<'a, T> {
                fn f(&self, x: ::core::cell::Ref<'a, Rel2<T>>, y: Rel2<T>);
            }
        };
        assert_eq!(contexts_of(&referrer_of(&tr), "Rel2 < T >"), ["< T >"]);
    }

    #[test]
    fn self_is_bounded_too() {
        // `Self: 'a` is part of the context of `Rel2<Self>` under a reference, like `T: 'a` of
        // `Rel2<T>`.
        let tr: ItemTrait = parse_quote! {
            trait Tr {
                fn f<'a>(&self, x: &'a dyn ::core::iter::Iterator<Item = Rel2<Self>>, y: Rel2<Self>)
                where Self: ::core::marker::Sized;
            }
        };
        let referrer = referrer_of(&tr);
        let found: Vec<(String, Vec<String>)> = referrer
            .iter()
            .map(|(_, _, context)| {
                assert!(matches!(context.self_ty, Some(SelfRequirement::Bound(_))));
                (
                    s(&context.generics.params),
                    context
                        .self_outlives
                        .iter()
                        .map(|l| l.to_string())
                        .collect(),
                )
            })
            .collect();
        assert_eq!(found.len(), 2);
        // The lifetime is a generic of the context, so that `&'a Self` can be encoded, and the entry
        // with nothing around the type has neither.
        assert!(
            found.contains(&("'a".to_string(), vec!["'a".to_string()])),
            "{:?}",
            found
        );
        assert!(found.contains(&(String::new(), vec![])), "{:?}", found);
    }

    /// Rewrites `item`, a copy in the scope `context`, replacing each leaked type by `P<ix>`.
    fn rewrite(referrer: &Referrer, context: ContextId, item: &Item) -> String {
        let mut item = item.clone();
        referrer.visitor(context, p).visit_item_mut(&mut item);
        s(&item)
    }

    /// Like `rewrite()`, with the generics of the outermost item taken as those of the context.
    fn rewrite_as_leakers(referrer: &Referrer, context: ContextId, item: &Item) -> String {
        let mut item = item.clone();
        referrer
            .visitor(context, p)
            .with_outermost_generics_as_leakers()
            .visit_item_mut(&mut item);
        s(&item)
    }

    /// The index of the entry of the type `ty` with the generics `generics`.
    fn entry(referrer: &Referrer, ty: &str, generics: &str) -> usize {
        referrer
            .iter()
            .map(|(_, ty, context)| (ty, &context.generics))
            .position(|(leaked, g)| {
                s(leaked) == ty && s(&quote!(#g #{&g.where_clause})) == generics
            })
            .unwrap_or_else(|| panic!("no entry {} {}", ty, generics))
    }

    #[test]
    fn the_referrer_finds_the_entry_of_a_type_under_the_lifetimes_around_it() {
        let tr: ItemTrait = parse_quote! {
            trait Tr {
                fn f<'a, T>(
                    &self,
                    x: &'a dyn ::core::iter::Iterator<Item = Rel2<T>>,
                    y: Rel2<T>,
                );
            }
        };
        let (referrer, context) = referrer_in(&tr);
        let around = entry(&referrer, "Rel2 < T >", "< 'a , T : 'a >");
        let alone = entry(&referrer, "Rel2 < T >", "< T >");
        assert_ne!(around, alone);
        let cases: Vec<(Item, usize)> = vec![
            // A struct field: what the copy needs. The copy declares the generics of the method.
            (
                parse_quote!(
                    struct G<'a, T>(&'a Rel2<T>);
                ),
                around,
            ),
            (
                parse_quote!(
                    struct H<'a, T>(&'a ::std::vec::Vec<Rel2<T>>);
                ),
                around,
            ),
            (
                parse_quote!(
                    struct D<'a, T>(&'a dyn ::core::iter::Iterator<Item = Rel2<T>>);
                ),
                around,
            ),
            // The lifetime of the reference is passed down through the path with a lifetime, and its
            // own is not.
            (
                parse_quote!(
                    struct K<'a, 'b, T>(&'a ::core::cell::Ref<'b, Rel2<T>>);
                ),
                around,
            ),
            // A path with a lifetime does not carry it: the type is the one with nothing around it,
            // whichever lifetimes are declared.
            (
                parse_quote!(
                    struct F<'a, T>(::core::cell::Ref<'a, Rel2<T>>);
                ),
                alone,
            ),
            (
                parse_quote!(
                    struct E<'a, T>(::core::cell::Ref<'a, ::std::vec::Vec<Rel2<T>>>);
                ),
                alone,
            ),
            (
                parse_quote!(
                    struct O<'a, T>(
                        ::core::cell::Ref<'a, dyn ::core::iter::Iterator<Item = Rel2<T>>>,
                    );
                ),
                alone,
            ),
            // Nothing around it, whichever lifetimes are declared.
            (
                parse_quote!(
                    struct V<T>(Rel2<T>);
                ),
                alone,
            ),
            (
                parse_quote!(
                    struct W<'a, T>(Rel2<T>, ::core::marker::PhantomData<&'a ()>);
                ),
                alone,
            ),
            // A projection does not carry them.
            (
                parse_quote!(
                    struct Q<'a, T>(&'a <Rel2<T> as ::core::ops::Deref>::Target);
                ),
                alone,
            ),
            // A binder lifetime does not.
            (
                parse_quote!(
                    struct B<T>(for<'x> fn(&'x Rel2<T>) -> usize);
                ),
                alone,
            ),
        ];
        for (item, ix) in cases {
            let item: Item = item;
            let copy = rewrite(&referrer, context, &item);
            assert!(!copy.contains("Rel2"), "{}", copy);
            assert!(
                copy.contains(&format!("P < {} >", ix)),
                "{} has not P<{}>",
                copy,
                ix
            );
            let other = if ix == around { alone } else { around };
            assert!(!copy.contains(&format!("P < {} >", other)), "{}", copy);
        }
        // The bound is written in the struct: the same generics as the entry of the reference's
        // component, so it is found where the generics of the struct are declared.
        let item: Item = parse_quote!(
            struct F<'a, T: 'a>(::core::cell::Ref<'a, Rel2<T>>);
        );
        let copy = rewrite(&referrer, context, &item);
        assert!(!copy.contains("Rel2"), "{}", copy);
        assert!(copy.contains(&format!("P < {} >", around)), "{}", copy);
        // The context declares none of them: the generics of the outermost item cannot be taken as
        // its, so nothing is found.
        let copy = rewrite_as_leakers(&referrer, context, &item);
        assert!(copy.contains("Rel2"), "{}", copy);
    }

    #[test]
    fn the_referrer_does_not_mistake_a_type_for_another_lifetime() {
        let tr: ItemTrait = parse_quote! {
            trait Tr {
                fn f<'a, T>(&self, x: &'a dyn ::core::iter::Iterator<Item = Rel2<T>>);
            }
        };
        let (referrer, context) = referrer_in(&tr);
        let around = entry(&referrer, "Rel2 < T >", "< 'a , T : 'a >");
        // The lifetime is named differently: the generics declared here are not those of the entry.
        let item: Item = parse_quote!(
            struct F<'b, T>(&'b dyn ::core::iter::Iterator<Item = Rel2<T>>);
        );
        let copy = rewrite(&referrer, context, &item);
        assert!(copy.contains("Rel2"), "{}", copy);
        // The same bounds, with other generics around: replaced, as the type does not name them.
        let item: Item = parse_quote!(
            struct F<'a, 'b: 'a, T>(&'a dyn ::core::iter::Iterator<Item = Rel2<T>>);
        );
        let copy = rewrite(&referrer, context, &item);
        assert!(copy.contains(&format!("P < {} >", around)), "{}", copy);
        // The lifetime of a path is not the lifetime of the entry either: it is no lifetime around
        // the type, so the entry of the type alone is the one which is found, not this one.
        let item: Item = parse_quote!(
            struct F<'a, T>(::core::cell::Ref<'a, Rel2<T>>);
        );
        let copy = rewrite(&referrer, context, &item);
        assert!(copy.contains("Rel2"), "{}", copy);
    }

    #[test]
    fn the_referrer_tells_the_entries_apart_with_the_generics_of_the_context() {
        // The generics of the trait are not declared where the signature is copied, but they are
        // those of the context: the entry is the one whose bounds are those which the type implies
        // where it is found.
        let tr: ItemTrait = parse_quote! {
            trait Tr<'a, T> {
                fn f(
                    &self,
                    x: &'a dyn ::core::iter::Iterator<Item = Rel2<T>>,
                    y: Rel2<T>,
                    z: &'a dyn ::core::iter::Iterator<Item = ::std::vec::Vec<Rel2<T>>>,
                    w: ::core::cell::Ref<'a, Rel2<T>>,
                );
            }
        };
        let (referrer, context) = referrer_in(&tr);
        let around = entry(&referrer, "Rel2 < T >", "< 'a , T : 'a >");
        let alone = entry(&referrer, "Rel2 < T >", "< T >");
        // `expand()` declares nothing but the generics of the context.
        let expand = |ty: Type| -> String { s(&referrer.expand(context, ty, p)) };
        assert_eq!(
            expand(parse_quote!(&'a dyn ::core::iter::Iterator<Item = Rel2<T>>)),
            format!(
                "& 'a dyn :: core :: iter :: Iterator < Item = P < {} > >",
                around
            )
        );
        assert_eq!(expand(parse_quote!(Rel2<T>)), format!("P < {} >", alone));
        assert_eq!(
            expand(parse_quote!(
                &'a dyn ::core::iter::Iterator<Item = ::std::vec::Vec<Rel2<T>>>
            )),
            format!(
                "& 'a dyn :: core :: iter :: Iterator < Item = :: std :: vec :: Vec < P < {} > > >",
                around
            )
        );
        // A path with the lifetime carries nothing, so it is the entry of the type alone.
        assert_eq!(
            expand(parse_quote!(::core::cell::Ref<'a, Rel2<T>>)),
            format!(":: core :: cell :: Ref < 'a , P < {} > >", alone)
        );
        // The signatures, with the generics of the trait undeclared (the visitor, for a signature,
        // declares only those of the method, and those of the trait are the context's).
        let sig: Signature = parse_quote!(
            fn f(
                &self,
                x: &'a dyn ::core::iter::Iterator<Item = Rel2<T>>,
                y: Rel2<T>,
                z: &'a dyn ::core::iter::Iterator<Item = ::std::vec::Vec<Rel2<T>>>,
                w: ::core::cell::Ref<'a, Rel2<T>>,
            )
        );
        for mut visitor in [
            referrer.visitor(context, p),
            // The signature declares no generics, which are those of the context in any case.
            referrer
                .visitor(context, p)
                .with_outermost_generics_as_leakers(),
        ] {
            let mut copy = sig.clone();
            visitor.visit_signature_mut(&mut copy);
            let copy = s(&copy);
            assert!(!copy.contains("Rel2"), "{}", copy);
            assert!(copy.contains(&format!("y : P < {} >", alone)), "{}", copy);
            assert!(
                copy.contains(&format!(
                    "x : & 'a dyn :: core :: iter :: Iterator < Item = P < {} > >",
                    around
                )),
                "{}",
                copy
            );
            assert!(
                copy.contains(&format!(
                    "Item = :: std :: vec :: Vec < P < {} > > >",
                    around
                )),
                "{}",
                copy
            );
            assert!(
                copy.contains(&format!(
                    "w : :: core :: cell :: Ref < 'a , P < {} > >",
                    alone
                )),
                "{}",
                copy
            );
        }
    }

    #[test]
    fn the_referrer_finds_the_entry_with_a_lifetime_of_the_method() {
        // The lifetime is declared by the method, and the type by the trait (the context), so the
        // generics which the entry has are declared by two scopes.
        let tr: ItemTrait = parse_quote! {
            trait Tr<T> {
                fn f<'a>(
                    &self,
                    x: &'a dyn ::core::iter::Iterator<Item = Rel2<T>>,
                    y: Rel2<T>,
                    w: ::core::cell::Ref<'a, Rel2<T>>,
                );
            }
        };
        let (referrer, context) = referrer_in(&tr);
        let around = entry(&referrer, "Rel2 < T >", "< 'a , T : 'a >");
        let alone = entry(&referrer, "Rel2 < T >", "< T >");
        let sig: Signature = parse_quote!(
            fn f<'a>(
                &self,
                x: &'a dyn ::core::iter::Iterator<Item = Rel2<T>>,
                y: Rel2<T>,
                w: ::core::cell::Ref<'a, Rel2<T>>,
            )
        );
        let mut visitor = referrer.visitor(context, p);
        let mut copy = sig.clone();
        visitor.visit_signature_mut(&mut copy);
        let copy = s(&copy);
        assert!(!copy.contains("Rel2"), "{}", copy);
        assert!(
            copy.contains(&format!(
                "x : & 'a dyn :: core :: iter :: Iterator < Item = P < {} > >",
                around
            )),
            "{}",
            copy
        );
        assert!(copy.contains(&format!("y : P < {} >", alone)), "{}", copy);
        assert!(
            copy.contains(&format!(
                "w : :: core :: cell :: Ref < 'a , P < {} > >",
                alone
            )),
            "{}",
            copy
        );
    }

    #[test]
    fn the_entries_of_the_types_around_survive_serialization() {
        let tr: ItemTrait = parse_quote! {
            trait Tr<'a, T> {
                fn f<'b, U>(
                    &self,
                    x: &'b dyn ::core::iter::Iterator<Item = Rel2<U>>,
                    y: Rel2<U>,
                    z: &'a ::std::vec::Vec<Rel2<T>>,
                );
            }
        };
        let (referrer, context) = referrer_in(&tr);
        let tokens = referrer.to_token_stream();
        let parsed: Referrer = parse2(tokens.clone()).unwrap();
        assert_eq!(parsed, referrer);
        // The lifetime of the type around is a generic of the context, written in it.
        assert!(tokens.to_string().contains("'b , U : 'b"), "{}", tokens);
        // And the entries are found in the parsed one.
        let item: Item = parse_quote!(
            struct F<'b, U>(&'b dyn ::core::iter::Iterator<Item = Rel2<U>>);
        );
        let mut copy = item.clone();
        parsed.visitor(context, p).visit_item_mut(&mut copy);
        assert!(!s(&copy).contains("Rel2"), "{}", s(&copy));
    }

    /// A path starting with a type parameter or `Self` (`T::A<X>`) is a projection, like the one with
    /// a qualified self (`<T as Tr>::A<X>`): rustc does not require `X: 'a` of `&'a T::A<X>` (the
    /// original field of a struct is well-formed without `U: 'a`, whereas the one which the copy has
    /// for `Rel2<U>` would be stricter with it), so the type inside does not carry what the types
    /// around it outlive (a reference's lifetime: a path's own lifetime is never carried). The
    /// Referrer finds the entry from the generics in scope: `T` is a type parameter if a scope
    /// declares it.
    mod projection_paths {
        use super::*;

        /// The contexts of `Rel2<U>` in a method whose parameter is `x: construct` (`{X}` in it is
        /// `Rel2<U>`), with `T: ::p::Gat`.
        fn rel2_under(construct: &str) -> Vec<String> {
            let construct = construct.replace("{X}", "Rel2<U>");
            let tr: ItemTrait = parse_str(&format!(
                "trait Tr {{ fn f<'a, 'b, T: ::p::Gat, U>(&self, x: {}); }}",
                construct
            ))
            .unwrap();
            contexts_of(&referrer_of(&tr), "Rel2 < U >")
        }

        #[test]
        fn the_type_inside_a_projection_path_has_no_bound_from_the_types_around() {
            for construct in [
                "&'a T::A<{X}>",
                "&'a mut T::A<{X}>",
                "&'a (u8, T::A<{X}>)",
                "&'a ::std::vec::Vec<T::A<{X}>>",
                "&'a ::core::cell::Ref<'b, T::A<{X}>>",
                // More segments, and arguments elsewhere in the path.
                "&'a T::A::B<{X}>",
                "&'a T::A<u8, {X}>",
                "&'a T::A<(u8, {X})>",
                // A lifetime argument of the projection does not pass its bound on either (nor does
                // the one of any path, see `a_path_with_a_lifetime_argument_adds_no_bound()`).
                "T::A<'b, {X}>",
                "&'a T::A<'b, {X}>",
                // A path with a lifetime has none to pass on, projection or not.
                "::core::cell::Ref<'a, T::A<{X}>>",
                "::core::cell::Ref<'a, ::std::vec::Vec<T::A<{X}>>>",
            ] {
                assert_eq!(rel2_under(construct), ["< U >"], "{}", construct);
            }
            // The qualified path is the same, with `T` as a parameter or not.
            for construct in [
                "&'a <T as ::p::Gat>::A<{X}>",
                "&'a <::p::X as ::p::Gat>::A<{X}>",
                "::core::cell::Ref<'a, <T as ::p::Gat>::A<{X}>>",
            ] {
                assert_eq!(rel2_under(construct), ["< U >"], "{}", construct);
            }
            // What is around the projection is not what is inside it: the type of the argument of
            // another type, and what is in a reference inside the projection, are bounded as always.
            for construct in [
                "&'a (T::A<u8>, {X})",
                "&'a ::std::vec::Vec<{X}>",
                "&'a ::core::cell::Ref<'b, (T::A<u8>, {X})>",
            ] {
                assert_eq!(rel2_under(construct), ["< 'a , U : 'a >"], "{}", construct);
            }
            // A path with a lifetime and no reference around it bounds nothing.
            for construct in [
                "::core::cell::Ref<'a, (T::A<u8>, {X})>",
                "::core::cell::Ref<'a, ::std::vec::Vec<{X}>>",
            ] {
                assert_eq!(rel2_under(construct), ["< U >"], "{}", construct);
            }
            assert_eq!(
                rel2_under("T::A<&'a {X}>"),
                ["< 'a , U : 'a >"],
                "the reference requires it itself"
            );
            // The projection as a whole is still a component: `Ref<'a, T::A<X>>` requires
            // `T::A<X>: 'a`, which an implementation which has the type as a whole needs the bounds
            // for (`T: 'a` and `U: 'a`, E0309 otherwise), so it is not the same as no bound at all.
            let tr: ItemTrait = parse_quote! {
                trait Tr { fn f<'a, T: ::p::Gat, U>(&self, x: ::p::Wrap<'a, T::A<U>>); }
            };
            // An absolute path is not a required entry. It is an optional one (old: no entry), which
            // the Referrer may replace: it has the generics it uses, with no bound which its
            // definition may imply (`T: 'a` and `U: 'a` are not assumed: they are written when they
            // are needed).
            assert_eq!(
                contexts_of(
                    &referrer_with_optional_entries(&tr),
                    ":: p :: Wrap < 'a , T :: A < U > >"
                ),
                ["< 'a , T : :: p :: Gat , U >"],
                "an absolute path is an optional entry"
            );
            let tr: ItemTrait = parse_quote! {
                trait Tr { fn f<'a, T: ::p::Gat + 'a, U: 'a>(&self, x: ::p::Wrap<'a, T::A<U>>); }
            };
            assert_eq!(
                contexts_of(
                    &referrer_with_optional_entries(&tr),
                    ":: p :: Wrap < 'a , T :: A < U > >"
                ),
                ["< 'a , T : 'a + :: p :: Gat , U : 'a >"],
                "the bounds which are written"
            );
        }

        #[test]
        fn a_path_which_does_not_start_with_a_type_parameter_is_not_a_projection() {
            // `m` is not a generic here, so it is an ordinary path (a module, say): `A<X>` is its
            // argument, which passes on what is required of it by the reference around it.
            let tr: ItemTrait = parse_quote! {
                trait Tr { fn f<'a, U>(&self, x: &'a ::p::m::A<Rel2<U>>); }
            };
            assert_eq!(
                contexts_of(&referrer_of(&tr), "Rel2 < U >"),
                ["< 'a , U : 'a >"]
            );
            // And nothing is passed on by a path with a lifetime and no reference around it.
            let tr: ItemTrait = parse_quote! {
                trait Tr { fn f<'a, U>(&self, x: ::core::cell::Ref<'a, ::p::m::A<Rel2<U>>>); }
            };
            assert_eq!(contexts_of(&referrer_of(&tr), "Rel2 < U >"), ["< U >"]);
            // A const parameter is not a type parameter either.
            let tr: ItemTrait = parse_quote! {
                trait Tr { fn f<'a, const T: usize, U>(&self, x: &'a T::A<Rel2<U>>); }
            };
            assert_eq!(
                contexts_of(&referrer_of(&tr), "Rel2 < U >"),
                ["< 'a , U : 'a >"]
            );
            // A type parameter of an inner scope which shadows the name does not change it.
            let tr: ItemTrait = parse_quote! {
                trait Tr<T> {
                    fn f<'a, U>(&self, x: &'a T::A<Rel2<U>>);
                }
            };
            assert_eq!(contexts_of(&referrer_of(&tr), "Rel2 < U >"), ["< U >"]);
        }

        #[test]
        fn self_paths_are_projections() {
            // `Self::A<X>` is an associated type of the trait, whatever `Self` is.
            let tr: ItemTrait = parse_quote! {
                trait Tr {
                    type B<X>;
                    fn f<'a, U>(&self, x: &'a Self::B<Rel2<U>>);
                    fn g<'a, U>(&self, x: &'a Self::B<'a, Rel2<U>>);
                    fn h<'a, U>(&self, x: ::core::cell::Ref<'a, Self::B<Rel2<U>>>);
                }
            };
            let referrer = referrer_of(&tr);
            assert_eq!(contexts_of(&referrer, "Rel2 < U >"), ["< U >"]);
            // `Self` is not a generic: `Self::A` mentions `Self`, but the type inside does not.
            let rel2 = (0..referrer.iter().count())
                .find(|&ix| s(crate::common::types(&referrer).nth(ix).unwrap()) == "Rel2 < U >")
                .unwrap();
            assert!(crate::common::entry(&referrer, rel2)
                .self_ty
                .as_ref()
                .is_none());
            assert!(crate::common::entry(&referrer, rel2)
                .self_outlives
                .as_slice()
                .is_empty());
        }

        /// The trait of the Leaker: `x` has the type inside a projection path, and `y` the same type
        /// under the lifetime of a reference without it. Returns its Referrer and top-level context.
        fn both() -> (Referrer, ContextId) {
            let tr: ItemTrait = parse_quote! {
                trait Tr {
                    fn f<'a, T: ::p::Gat, U>(
                        &self,
                        x: &'a T::A<Rel2<U>>,
                        y: &'a ::std::vec::Vec<Rel2<U>>,
                    );
                }
            };
            referrer_in(&tr)
        }

        #[test]
        fn the_referrer_finds_the_entry_of_a_type_in_a_projection_path() {
            let (referrer, context) = both();
            let projected = entry(&referrer, "Rel2 < U >", "< U >");
            let around = entry(&referrer, "Rel2 < U >", "< 'a , U : 'a >");
            assert_ne!(projected, around);
            let cases: Vec<(Item, usize)> = vec![
                // A struct field, which is what the copy needs.
                (
                    parse_quote!(
                        struct F<'a, T: ::p::Gat, U>(::core::cell::Ref<'a, T::A<Rel2<U>>>);
                    ),
                    projected,
                ),
                (
                    parse_quote!(
                        struct G<'a, T: ::p::Gat, U>(&'a T::A<Rel2<U>>);
                    ),
                    projected,
                ),
                (
                    parse_quote!(
                        struct H<'a, T: ::p::Gat, U>(
                            ::core::cell::Ref<'a, ::std::vec::Vec<T::A<Rel2<U>>>>,
                        );
                    ),
                    projected,
                ),
                // The same type under the lifetime of a reference without a projection is the other
                // entry, and under the lifetime of a path it is the one with nothing around it.
                (
                    parse_quote!(
                        struct V<'a, U>(&'a ::std::vec::Vec<Rel2<U>>);
                    ),
                    around,
                ),
                (
                    parse_quote!(
                        struct W<'a, U>(::core::cell::Ref<'a, ::std::vec::Vec<Rel2<U>>>);
                    ),
                    projected,
                ),
                // A qualified projection (which is one whatever its self type is).
                (
                    parse_quote!(
                        struct Q<'a, U>(::core::cell::Ref<'a, <U as ::p::Gat>::A<Rel2<U>>>);
                    ),
                    projected,
                ),
            ];
            for (item, ix) in cases {
                let copy = rewrite(&referrer, context, &item);
                assert!(!copy.contains("Rel2"), "{}", copy);
                assert!(
                    copy.contains(&format!("P < {} >", ix)),
                    "{} has not P<{}>",
                    copy,
                    ix
                );
                let other = if ix == around { projected } else { around };
                assert!(!copy.contains(&format!("P < {} >", other)), "{}", copy);
            }
        }

        #[test]
        fn the_referrer_finds_the_entry_with_the_generics_of_the_method_declared() {
            // The generics of the method are declared where its signature is copied, those of the
            // trait are the context's.
            let tr: ItemTrait = parse_quote! {
                trait Tr<T: ::p::Gat> {
                    fn f<'a, U>(&self, x: ::core::cell::Ref<'a, T::A<Rel2<U>>>);
                }
            };
            let (referrer, context) = referrer_in(&tr);
            assert_eq!(contexts_of(&referrer, "Rel2 < U >"), ["< U >"]);
            let sig: Signature = parse_quote!(
                fn f<'a, U>(&self, x: ::core::cell::Ref<'a, T::A<Rel2<U>>>)
            );
            let mut copy = sig.clone();
            referrer.visitor(context, p).visit_signature_mut(&mut copy);
            assert!(!s(&copy).contains("Rel2"), "{}", s(&copy));
        }

        #[test]
        fn a_generic_which_no_scope_declares_is_not_assumed() {
            // The generics of the method are not those of the context, so `expand()`, which has
            // nothing else, does not find the entry (`Rel2<U>` with `<U>`): `U` is no generic there,
            // and `T` is no type parameter. The Referrer used to assume what no scope declared.
            let tr: ItemTrait = parse_quote! {
                trait Tr {
                    fn f<'a, T: ::p::Gat, U>(&self, x: ::core::cell::Ref<'a, T::A<Rel2<U>>>);
                }
            };
            let (referrer, context) = referrer_in(&tr);
            let projected = entry(&referrer, "Rel2 < U >", "< U >");
            let expand = |ty: Type| -> String { s(&referrer.expand(context, ty, p)) };
            for ty in [
                parse_quote!(::core::cell::Ref<'a, T::A<Rel2<U>>>),
                parse_quote!(&'a T::A<Rel2<U>>),
                parse_quote!(&'a ::std::vec::Vec<T::A<Rel2<U>>>),
                parse_quote!(&'a T::A::B<(u8, Rel2<U>)>),
                parse_quote!(::core::cell::Ref<'a, ::std::vec::Vec<T::A<Rel2<U>>>>),
                parse_quote!(::core::cell::Ref<'a, Rel2<U>>),
            ] {
                let ty: Type = ty;
                assert!(expand(ty.clone()).contains("Rel2"), "{}", s(&ty));
            }
            // The signatures which declare the generics of the method find it.
            let rewrite_sig = |sig: Signature| -> String {
                let mut sig = sig;
                referrer.visitor(context, p).visit_signature_mut(&mut sig);
                s(&sig)
            };
            let copy = rewrite_sig(parse_quote!(
                fn f<A: Allow, 'a, T: ::p::Gat, U>(&self, x: ::core::cell::Ref<'a, T::A<Rel2<U>>>)
            ));
            assert!(!copy.contains("Rel2"), "{}", copy);
            assert!(copy.contains(&format!("P < {} >", projected)), "{}", copy);
            // Under the lifetime of a reference, a head which is declared (here by the method) is a
            // projection, which carries no bound: the entry found is the one without it. A head
            // which no scope declares is an ordinary path, so the lifetime is passed on, and there is
            // no entry for it in this trait.
            let copy = rewrite_sig(parse_quote!(
                fn f<'a, T: ::p::Gat, U>(&self, x: &'a T::A<Rel2<U>>)
            ));
            assert!(copy.contains(&format!("P < {} >", projected)), "{}", copy);
            let copy = rewrite_sig(parse_quote!(
                fn f<'a, U>(&self, x: &'a X::A<Rel2<U>>)
            ));
            assert!(copy.contains("Rel2"), "{}", copy);
        }

        #[test]
        fn the_head_of_a_projection_is_a_type_parameter_if_a_scope_declares_it() {
            // `T` and `U` are declared by the trait, so they are the context's; `'a` is the method's.
            // Both types are in the Leaker<A>.
            let tr: ItemTrait = parse_quote! {
                trait Tr<T: ::p::Gat, U> {
                    fn f<'a>(&self, x: &'a T::A<Rel2<U>>, y: &'a ::std::vec::Vec<Rel2<U>>);
                }
            };
            let (referrer, context) = referrer_in(&tr);
            let projected = entry(&referrer, "Rel2 < U >", "< U >");
            let around = entry(&referrer, "Rel2 < U >", "< 'a , U : 'a >");
            assert_ne!(projected, around);
            let rewrite_sig = |sig: Signature| -> String {
                let mut sig = sig;
                referrer.visitor(context, p).visit_signature_mut(&mut sig);
                s(&sig)
            };
            // A head which a scope declares (here, the trait) is a projection: nothing is carried
            // from the reference.
            let copy = rewrite_sig(parse_quote!(fn f<'a>(&self, x: &'a T::A<Rel2<U>>)));
            assert!(copy.contains(&format!("P < {} >", projected)), "{}", copy);
            // An undeclared one is an ordinary path, so the lifetime is passed on: the stricter entry
            // is the one (the copy of the other would be rejected by rustc if `X` was no type
            // parameter).
            let copy = rewrite_sig(parse_quote!(fn f<'a>(&self, x: &'a X::A<Rel2<U>>)));
            assert!(copy.contains(&format!("P < {} >", around)), "{}", copy);
            // With a path's lifetime there is no bound to be stricter by: the entry of the projection.
            let copy = rewrite_sig(parse_quote!(
                fn f<'a>(&self, x: ::core::cell::Ref<'a, X::A<Rel2<U>>>)
            ));
            assert!(copy.contains(&format!("P < {} >", projected)), "{}", copy);
            // `T` shadowed by the method is declared there, and it is a type parameter too.
            let copy = rewrite_sig(parse_quote!(fn f<'a, T>(&self, x: &'a T::A<Rel2<U>>)));
            assert!(copy.contains(&format!("P < {} >", projected)), "{}", copy);
            // And a const parameter of the method is not.
            let copy =
                rewrite_sig(parse_quote!(fn f<'a, const T: usize>(&self, x: &'a T::A<Rel2<U>>)));
            assert!(copy.contains(&format!("P < {} >", around)), "{}", copy);
            // `expand()` has the generics of the context only.
            let expand = |ty: Type| -> String { s(&referrer.expand(context, ty, p)) };
            assert!(expand(parse_quote!(T::A<Rel2<U>>)).contains(&format!("P < {} >", projected)));
            // The copy of the trait declares its generics as they are (so it is the scope), or with
            // other bounds, which are those of the context with the outermost generics as leakers.
            let item: Item = parse_quote!(
                trait Tr<T: ::p::Gat, U> {
                    fn f<'a>(&self, x: &'a T::A<Rel2<U>>);
                }
            );
            let copy = rewrite(&referrer, context, &item);
            assert!(copy.contains(&format!("P < {} >", projected)), "{}", copy);
            assert!(!copy.contains("Rel2"), "{}", copy);
            let item: Item = parse_quote!(
                impl<T: ::p::Gat + ::core::clone::Clone, U> Tr<T, U> for W {
                    fn f<'a>(&self, x: &'a T::A<Rel2<U>>) {}
                }
            );
            let copy = rewrite_as_leakers(&referrer, context, &item);
            assert!(copy.contains(&format!("P < {} >", projected)), "{}", copy);
            assert!(!copy.contains("Rel2"), "{}", copy);
        }

        #[test]
        fn the_entries_survive_serialization() {
            let (referrer, context) = both();
            let tokens = referrer.to_token_stream();
            let parsed: Referrer = parse2(tokens).unwrap();
            assert_eq!(parsed, referrer);
            let item: Item = parse_quote!(
                struct F<'a, T: ::p::Gat, U>(::core::cell::Ref<'a, T::A<Rel2<U>>>);
            );
            let mut copy = item.clone();
            parsed.visitor(context, p).visit_item_mut(&mut copy);
            assert!(!s(&copy).contains("Rel2"), "{}", s(&copy));
        }
    }
}

mod self_outlives {
    //! `Self: 'a`, which a leaked type implies (`&'a Rel<Self>`, `&'a (Self, T)`: a reference requires
    //! its referent to outlive it) or the item writes (`where Self: 'a`). An associated type of `Self`
    //! (`&'a Self::Item`, `&'a <Self as Tr>::Item`) implies `Self::Item: 'a` instead, which is a where
    //! predicate of the context and no `Self: 'a`: the original gets no more from the well-formedness
    //! of its signature, and a copy that asks for `Self: 'a` is stricter (F34). `Self` is
    //! not a generic parameter, so it is not among the bounds of the generics of the context, but its
    //! own part of it (`Referrer::context_self_outlives()`), which the encoders encode as `&'a Self`
    //! (`&'a __S` in the Repeater's implementation for the implementor of the trait), after the
    //! lifetime bounds. The implementation does not declare it, as nothing proves it at a use but the
    //! encoding.
    //!
    //! A path with a lifetime argument (`Rel<'a, Self>`, where `struct Rel<'a, T>(&'a T)` has `T: 'a`
    //! inferred) implies nothing: its definition is not seen, so the bound is not assumed, and it is
    //! only there if it is written.

    use super::s;
    use crate::common;
    use syn::visit_mut::VisitMut;
    use syn::*;
    use template_quote::{quote, ToTokens};
    use type_leak::{
        array_const, Context, ContextId, ImplSelf, Leaker, LeakerResult, Referrer, SelfRequirement,
        UseSite,
    };

    /// The tag of the format of a serialized Referrer, which it starts with.
    fn tag() -> String {
        format!("type-leak-format-{}", type_leak::Referrer::FORMAT_VERSION)
    }

    fn of_trait(item: ItemTrait) -> Referrer {
        {
            let mut leaker = Leaker::new();
            leaker.intern_trait(&item, true);
            leaker.result().finish().unwrap()
        }
    }

    fn lifetimes(list: &[&str]) -> Vec<Lifetime> {
        list.iter().map(|l| parse_str(l).unwrap()).collect()
    }

    /// The lifetimes which `Self` outlives in each context of an entry, with the types in it. The
    /// contexts are in the order of their first entry.
    fn outlives(referrer: &Referrer) -> Vec<(Vec<String>, Vec<String>)> {
        let mut contexts: Vec<&Context> = Vec::new();
        for (_, _, context) in referrer.iter() {
            if !contexts.contains(&context) {
                contexts.push(context);
            }
        }
        contexts
            .into_iter()
            .map(|found| {
                let types = referrer
                    .iter()
                    .filter(|(_, _, other)| *other == found)
                    .map(|(_, ty, _)| s(ty))
                    .collect();
                let outlives = found.self_outlives.iter().map(|l| l.to_string()).collect();
                (types, outlives)
            })
            .collect()
    }

    #[test]
    fn a_type_implying_it_records_it_in_the_context() {
        // `&'a Rel<Self>` needs `Self: 'a`, with a lifetime of the method or of the trait.
        for item in [
            parse_quote! { trait Tr { fn f<'a>(&self, x: &'a Rel<Self>); } },
            // `Self` apart from its associated type.
            parse_quote! { trait Tr { type Item; fn f<'a>(&self, x: &'a Rel<(Self, Self::Item)>); } },
            parse_quote! { trait Tr<'a> { fn f(&self, x: &'a Rel<Self>); } },
            parse_quote! { trait Tr { fn f<'a>(&self, x: &'a Rel<(u8, Vec<Self>)>); } },
            // The arguments of a trait object must outlive too (`dyn Tr<Self> + 'b: 'a`).
            parse_quote! { trait Tr { fn f<'a>(&self, x: &'a Rel<Box<dyn Other<Self>>>); } },
            // A reference is interned as a whole if its lifetime is a generic of the trait.
            parse_quote! { trait Tr<'a> { fn f(&self, x: &'a mut Rel<Self>); } },
        ] {
            let item: ItemTrait = item;
            let referrer = of_trait(item.clone());
            let found = outlives(&referrer);
            assert_eq!(
                found.iter().filter(|(_, o)| !o.is_empty()).count(),
                1,
                "{}: {:?}",
                s(&item),
                found
            );
            let (types, o) = found.iter().find(|(_, o)| !o.is_empty()).unwrap();
            assert_eq!(o, &["'a"], "{}", s(&item));
            assert_eq!(types.len(), 1, "{}", s(&item));
            // `Self` has a meaning in the context.
            assert!(
                matches!(
                    common::entry(&referrer, 0).self_ty.as_ref(),
                    Some(SelfRequirement::Bound(_))
                ),
                "{}",
                s(&item)
            );
            assert_eq!(
                common::entry(&referrer, 0).self_outlives.as_slice(),
                lifetimes(&["'a"])
            );
        }
        // Every lifetime, sorted, each once (the reference of a generic lifetime is the type, with the
        // lifetime of the reference inside it too).
        let referrer = of_trait(parse_quote! {
            trait Tr<'b, 'a> { fn f(&self, x: &'a &'b Rel<Self>); }
        });
        assert_eq!(
            outlives(&referrer)[0],
            (
                vec!["& 'a & 'b Rel < Self >".to_string()],
                vec!["'a".to_string(), "'b".to_string()]
            )
        );
    }

    #[test]
    fn an_associated_type_of_self_implies_its_own_outlives_and_not_self() {
        // `&'a Rel<Self::Item>`, `&'a Rel<<Self as Tr>::Item>`, `&'a Rel<<Self as Tr<T>>::Item>` and
        // `&'a Rel<Self::G<'b>>` need `<path>: 'a`, which the original has from its signature. The
        // context has it as a where predicate, and `Self: 'a` is not recorded (F34).
        for (item, predicate) in [
            (
                parse_quote! { trait Tr { type Item; fn f<'a>(&self, x: &'a Rel<Self::Item>); } },
                "Self :: Item : 'a",
            ),
            (
                parse_quote! {
                    trait Tr { type Item; fn f<'a>(&self, x: &'a Rel<<Self as Tr>::Item>); }
                },
                "< Self as Tr > :: Item : 'a",
            ),
            (
                parse_quote! {
                    trait Tr<T> { type Item; fn f<'a>(&self, x: &'a Rel<<Self as Tr<T>>::Item>); }
                },
                "< Self as Tr < T > > :: Item : 'a",
            ),
            (
                parse_quote! {
                    trait Tr<'b> { type G<'g>; fn f<'a>(&self, x: &'a Rel<Self::G<'b>>); }
                },
                "Self :: G < 'b > : 'a",
            ),
            (
                parse_quote! {
                    trait Tr<'a> { type Item; fn f(&self, x: &'a Rel<Self::Item>); }
                },
                "Self :: Item : 'a",
            ),
        ] {
            let item: ItemTrait = item;
            let referrer = of_trait(item.clone());
            for (types, outlives) in outlives(&referrer) {
                assert!(
                    outlives.is_empty(),
                    "{}: {:?} has {:?}",
                    s(&item),
                    types,
                    outlives
                );
            }
            let predicates: Vec<String> = common::entry(&referrer, 0)
                .generics
                .where_clause
                .iter()
                .flat_map(|w| &w.predicates)
                .map(s)
                .collect();
            assert_eq!(predicates, [predicate], "{}", s(&item));
        }
    }

    #[test]
    fn a_type_which_does_not_imply_it_does_not() {
        for item in [
            // No lifetime.
            parse_quote! { trait Tr { fn f(&self, x: Rel<Self>); } },
            // The lifetime of a path (`struct Rel<'a, T>(&'a T)` has `T: 'a` inferred, but its
            // definition is not seen, so nothing is assumed): not a reference.
            parse_quote! { trait Tr { fn f<'a>(&self, x: Rel<'a, Self>); } },
            parse_quote! { trait Tr { type Item; fn f<'a>(&self, x: Rel<'a, Self::Item>); } },
            parse_quote! { trait Tr { type Item; fn f<'a>(&self, x: Rel<'a, <Self as Tr>::Item>); } },
            parse_quote! { trait Tr<'a> { fn f(&self, x: Rel<'a, Self>); } },
            parse_quote! { trait Tr { fn f<'a>(&self, x: Rel<'a, (u8, Vec<Self>)>); } },
            parse_quote! { trait Tr { fn f<'a>(&self, x: Rel<'a, Box<dyn Other<Self>>>); } },
            // Nor does a path with a lifetime make the types in its arguments outlive it for the
            // types inside them.
            parse_quote! { trait Tr<'a> { fn f(&self, x: ::core::cell::Ref<'a, Rel2<Self>>); } },
            parse_quote! { trait Tr<'a> { fn f(&self, x: ::core::cell::Ref<'a, Vec<Rel2<Self>>>); } },
            parse_quote! { trait Tr<'a> { fn f(&self, x: Rel<'a, Rel<'a, Self>>); } },
            // No `Self`.
            parse_quote! { trait Tr { fn f<'a, T>(&self, x: Rel<'a, T>); } },
            // Not a lifetime parameter.
            parse_quote! { trait Tr { fn f(&self, x: Rel<'static, Self>); } },
            parse_quote! { trait Tr { fn f(&self, x: for<'x> fn(Rel<'x, Self>)); } },
            parse_quote! { trait Tr { fn f(&self, x: for<'x> fn(&'x Rel<Self>)); } },
            // A projection does not require its qualified self to outlive.
            parse_quote! {
                trait Tr { fn f<'a>(&self, x: &'a <Rel2<Self> as ::core::ops::Deref>::Target); }
            },
            // `Self` is a lifetime argument's neighbour only: not a referent.
            parse_quote! { trait Tr { fn f<'a>(&self, x: Rel<'a>, y: Rel2<Self>); } },
            // A bound of a generic (the generic is not a referent of the lifetime).
            parse_quote! { trait Tr { fn f<'a, T: ::core::convert::Into<Self>>(&self, x: Rel2<T>); } },
        ] {
            let item: ItemTrait = item;
            let referrer = of_trait(item.clone());
            for (types, outlives) in outlives(&referrer) {
                assert!(
                    outlives.is_empty(),
                    "{}: {:?} has {:?}",
                    s(&item),
                    types,
                    outlives
                );
            }
        }
    }

    #[test]
    fn the_context_of_a_struct_has_it_too() {
        // `SelfRequirement::Type`: `Self` is the struct, with its parameters.
        let item: ItemStruct = parse_quote! { struct MyStruct<'a, T> { f: &'a Rel<Self> } };
        let mut leaker = Leaker::new();
        leaker.intern_struct(&item, true);
        let result = leaker.result();
        let referrer = result.clone().finish().unwrap();
        match common::entry(&referrer, 0).self_ty.as_ref() {
            Some(SelfRequirement::Type(ty)) => assert_eq!(s(ty), "MyStruct < 'a , T >"),
            other => panic!("{:?}", other),
        }
        assert_eq!(
            common::entry(&referrer, 0).self_outlives.as_slice(),
            lifetimes(&["'a"])
        );
        // The encoding in the implementation `impl<'a, T> Repeater<(&'a (), T, &'a Self), ..> for
        // MyStruct<'a, T>`; and `T: 'a` follows from it (`MyStruct<'a, T>: 'a`).
        let param: Ident = parse_quote!(__S);
        let encoded = result
            .repeater_impls(
                &parse_quote!(crate::Repeater),
                7,
                ImplSelf::Param(&param),
                array_const,
            )
            .unwrap()
            .next()
            .unwrap()
            .args;
        let expected: Type = parse_quote!((&'a (), T, &'a Self));
        assert_eq!(s(&encoded), s(&expected));
        // A concrete type in place of `Self` is as good.
        let self_ty = match common::entry(&referrer, 0).self_ty.as_ref() {
            Some(SelfRequirement::Type(ty)) => ty,
            _ => unreachable!(),
        };
        let encoded = common::repeater_args(
            common::entry(&referrer, 0),
            &UseSite::identity(),
            Some(self_ty),
            array_const,
        )
        .unwrap();
        let expected: Type = parse_quote!((&'a (), T, &'a MyStruct<'a, T>));
        assert_eq!(s(&encoded), s(&expected));
    }

    #[test]
    fn it_is_part_of_the_identity_of_the_context() {
        // The same generics and `Self`, but only one implies `Self: 'a`: separate contexts. A path
        // with the lifetime implies nothing, so it shares the context of a type without one.
        let referrer = of_trait(parse_quote! {
            trait Tr<'a> {
                fn f(&self, x: &'a Rel<Self>, y: Rel2<Self>, z: &'a Rel3, w: Rel4<'a, Self>);
            }
        });
        let found = outlives(&referrer);
        assert!(found.contains(&(
            vec!["& 'a Rel < Self >".to_string()],
            vec!["'a".to_string()]
        )));
        assert!(found.contains(&(
            vec![
                "Rel2 < Self >".to_string(),
                "Rel4 < 'a , Self >".to_string()
            ],
            vec![]
        )));
        // Not for a type which does not mention `Self`: no `Self` in its context, so no bound.
        assert!(found.contains(&(vec!["& 'a Rel3".to_string()], vec![])));
        // The contexts are distinct.
        assert_eq!(found.len(), 3);
    }

    #[test]
    fn it_survives_serialization() {
        let referrer = of_trait(parse_quote! {
            trait Tr<'b> {
                fn f<'a>(&self, x: &'a &'b Rel<Self>, y: Rel2<Self>, z: &'a Rel<Self>);
            }
        });
        let tokens = referrer.to_token_stream();
        let text = tokens.to_string();
        assert!(text.contains("Self : 'a + 'b ;"), "{}", text);
        let parsed: Referrer = parse2(tokens.clone()).unwrap();
        assert_eq!(parsed, referrer);
        assert_eq!(outlives(&parsed), outlives(&referrer));
        // Serializing again gives the same tokens.
        assert_eq!(parsed.to_token_stream().to_string(), text);
        // The order of the clauses does not matter, and the module follows them.
        let version = tag();
        let tokens: proc_macro2::TokenStream = format!(
            "\"{}\" 2 mod {{ path = [] }} ; () < 'a > ; Self : 'a ; Self : Tr ; mod 0 ; (Rel < 'a , Self >)",
            version
        )
        .parse()
        .unwrap();
        let result: LeakerResult = parse2(tokens).unwrap();
        let parsed = result.clone().into_referrer();
        assert_eq!(
            common::entry(&parsed, 0).self_outlives.as_slice(),
            lifetimes(&["'a"])
        );
        assert!(matches!(
            common::entry(&parsed, 0).self_ty.as_ref(),
            Some(SelfRequirement::Bound(_))
        ));
        let module = common::in_module(common::entry(&parsed, 0)).unwrap();
        assert_eq!(module.index(), 0);
        assert_eq!(common::entry_module_path(&result, 0), Some(vec![]));
        // A Referrer serialized before the clause existed has none.
        let tokens: proc_macro2::TokenStream =
            format!("\"{}\" Self : Tr ; (Rel < Self >)", version)
                .parse()
                .unwrap();
        let parsed: Referrer = parse2(tokens).unwrap();
        assert!(common::entry(&parsed, 0)
            .self_outlives
            .as_slice()
            .is_empty());
    }

    #[test]
    fn the_referrer_finds_the_entries_with_the_lifetimes_declared_where_the_copy_is() {
        // `Self: 'a` is implied by the references (the first trait), and not by the paths with a
        // lifetime (the second), whose entries have none.
        for tr in [
            parse_quote! {
                trait Tr<'t> {
                    fn me<'a>(&'a self) -> &'a Rel<Self> where Self: ::core::marker::Sized;
                    fn field(&self, x: &'t Rel<Self>) where Self: ::core::marker::Sized;
                    fn other<'a>(&self, x: &'a Rel<Self>, y: Rel2<Self>) where Self: ::core::marker::Sized;
                }
            },
            parse_quote! {
                trait Tr<'t> {
                    fn me<'a>(&'a self) -> Rel<'a, Self> where Self: ::core::marker::Sized;
                    fn field(&self, x: Rel<'t, Self>) where Self: ::core::marker::Sized;
                    fn other<'a>(&self, x: Rel<'a, Self>, y: Rel2<Self>) where Self: ::core::marker::Sized;
                }
            },
        ] {
            let tr: ItemTrait = tr;
            let mut leaker = Leaker::new();
            let top_level = leaker.intern_trait(&tr, true);
            let referrer = leaker.result().finish().unwrap();
            // The copy of each method is rewritten, with the lifetimes declared there or not (`'t` is
            // a generic of the trait, which the signature does not declare).
            let mut copy = tr.clone();
            referrer
                .visitor(top_level, |_, ix, _| parse_quote!(P<#ix>))
                .visit_item_trait_mut(&mut copy);
            let text = s(&copy);
            assert_eq!(text.matches("P < ").count(), 4, "{}", text);
            assert!(!text.contains("Rel <"), "{}", text);
            // Taking the generics of the trait as those of the context agrees.
            let mut copy = tr.clone();
            referrer
                .visitor(top_level, |_, ix, _| parse_quote!(P<#ix>))
                .with_outermost_generics_as_leakers()
                .visit_item_trait_mut(&mut copy);
            let text = s(&copy);
            assert_eq!(text.matches("P < ").count(), 4, "{}", text);
        }
    }

    /// The generics (with the where clause) of the context of the entry `ix`.
    fn declared(referrer: &Referrer, ix: usize) -> String {
        let generics = &common::entry(referrer, ix).generics;
        let where_clause = &generics.where_clause;
        s(&quote!(#generics #where_clause))
    }

    #[test]
    fn a_written_bound_is_the_same_as_an_implied_one() {
        // The same entry of the same trait, with `Self: 'a` written in the method or implied by the
        // reference: one context (the same id, generics, `Self` and lifetimes), and so one entry.
        let written: ItemTrait = parse_quote! {
            trait Tr { fn f<'a>(&'a self) -> &'a Rel<Self> where Self: ::core::marker::Sized + 'a; }
        };
        let implied: ItemTrait = parse_quote! {
            trait Tr { fn f<'a>(&'a self) -> &'a Rel<Self> where Self: ::core::marker::Sized; }
        };
        let mut leaker = Leaker::new();
        let first = leaker.intern_trait(&written, true);
        let second = leaker.intern_trait(&implied, true);
        assert_eq!(first, second);
        let referrer = leaker.result().finish().unwrap();
        assert_eq!(referrer.iter().count(), 1);
        assert_eq!(
            common::entry(&referrer, 0).self_outlives.as_slice(),
            lifetimes(&["'a"])
        );
        assert_eq!(
            declared(&referrer, 0),
            "< 'a > where Self : :: core :: marker :: Sized"
        );
        // Alone, each gives the same Referrer.
        let by_written = of_trait(written.clone());
        let by_implied = of_trait(implied.clone());
        assert_eq!(by_written, by_implied);
        assert_eq!(
            s(&by_written.to_token_stream()),
            s(&by_implied.to_token_stream())
        );
        // A path with the lifetime implies nothing, so it has the bound only if it is written: the
        // context of the type is then the same as that of the one the reference implies it for.
        let written_path: ItemTrait = parse_quote! {
            trait Tr { fn f<'a>(&'a self) -> Rel<'a, Self> where Self: ::core::marker::Sized + 'a; }
        };
        let plain_path: ItemTrait = parse_quote! {
            trait Tr { fn f<'a>(&'a self) -> Rel<'a, Self> where Self: ::core::marker::Sized; }
        };
        let by_written_path = of_trait(written_path.clone());
        let by_plain_path = of_trait(plain_path.clone());
        assert_eq!(
            common::entry(&by_written_path, 0).self_outlives.as_slice(),
            lifetimes(&["'a"])
        );
        assert!(common::entry(&by_plain_path, 0)
            .self_outlives
            .as_slice()
            .is_empty());
        assert_ne!(by_written_path, by_plain_path);
        assert_eq!(
            declared(&by_written_path, 0),
            "< 'a > where Self : :: core :: marker :: Sized"
        );
        assert_eq!(
            declared(&by_plain_path, 0),
            "< 'a > where Self : :: core :: marker :: Sized"
        );
        let mut leaker = Leaker::new();
        for item in [&written_path, &implied] {
            leaker.intern_trait(item, true);
        }
        let referrer = leaker.result().finish().unwrap();
        assert_eq!(referrer.iter().count(), 2);
        assert_eq!(
            common::entry_id(&referrer, 0),
            common::entry_id(&referrer, 1)
        );

        // The predicate may be written in the where clause of the trait, or in several predicates.
        for item in [
            parse_quote! {
                trait Tr where Self: ::core::marker::Sized {
                    fn f<'a>(&'a self) -> Rel<'a, Self> where Self: 'a;
                }
            },
            parse_quote! {
                trait Tr {
                    fn f<'a>(&'a self) -> Rel<'a, Self> where Self: ::core::marker::Sized, Self: 'a;
                }
            },
            parse_quote! {
                trait Tr<'a> where Self: 'a {
                    fn f(&self) -> Rel<'a, Self> where Self: ::core::marker::Sized;
                }
            },
        ] {
            let item: ItemTrait = item;
            let referrer = of_trait(item.clone());
            assert_eq!(referrer.iter().count(), 1, "{}", s(&item));
            assert_eq!(
                common::entry(&referrer, 0).self_outlives.as_slice(),
                lifetimes(&["'a"]),
                "{}",
                s(&item)
            );
            let generics = s(&common::entry(&referrer, 0).generics);
            assert!(
                !generics.contains("Self : 'a"),
                "{}: {}",
                s(&item),
                generics
            );
            assert!(
                !declared(&referrer, 0).contains("Self : 'a"),
                "{}",
                s(&item)
            );
            assert!(
                declared(&referrer, 0).contains("Self : :: core :: marker :: Sized"),
                "{}: {}",
                s(&item),
                declared(&referrer, 0)
            );
        }
        // Every lifetime, merged with those the type implies (`'b`, by the reference).
        let referrer = of_trait(parse_quote! {
            trait Tr<'b> {
                fn f<'a, 'c>(&self, x: &'b Rel<Self>) where Self: ::core::marker::Sized + 'c + 'a;
            }
        });
        assert_eq!(
            common::entry(&referrer, 0).self_outlives.as_slice(),
            lifetimes(&["'a", "'b", "'c"])
        );
        assert_eq!(
            declared(&referrer, 0),
            "< 'a , 'b , 'c > where Self : :: core :: marker :: Sized"
        );
        // A path with a lifetime implies none: only those which are written.
        let referrer = of_trait(parse_quote! {
            trait Tr<'b> {
                fn f<'a, 'c>(&self, x: Rel<'a, 'b, Self>) where Self: ::core::marker::Sized + 'c + 'a;
            }
        });
        assert_eq!(
            common::entry(&referrer, 0).self_outlives.as_slice(),
            lifetimes(&["'a", "'c"])
        );
        assert_eq!(
            declared(&referrer, 0),
            "< 'a , 'b , 'c > where Self : :: core :: marker :: Sized"
        );
        // Written alone, with no bound which the type implies (`Rel<'a, T>` has no `Self`, and a path
        // with a lifetime implies no bound on `T` either).
        let referrer = of_trait(parse_quote! {
            trait Tr { fn f<'a, T>(&self, x: Rel<'a, T>) where Self: 'a; }
        });
        assert_eq!(
            common::entry(&referrer, 0).self_outlives.as_slice(),
            lifetimes(&["'a"])
        );
        assert_eq!(declared(&referrer, 0), "< 'a , T >");
    }

    /// A bound by `'static`, by a lifetime which is not a parameter, a bound under a `for<..>`
    /// binder, and a bound of another type (`Self::Item: 'a`, which `Self: 'a` implies but is not)
    /// stay where predicates; only a bound by a lifetime parameter moves into the lifetimes `Self`
    /// outlives (`Referrer::context_self_outlives()`). The rows are the items, the generics the
    /// context declares, and the lifetimes `Self` outlives.
    #[test]
    fn a_bound_which_is_not_a_lifetime_parameter_stays_a_predicate() {
        let rows: Vec<(ItemTrait, &str, &[&str])> = vec![
            // The reference implies `Self: 'a`.
            (
                parse_quote! {
                    trait Tr { fn f<'a>(&self, x: &'a Rel<Self>) where Self: ::core::marker::Sized + 'static; }
                },
                "< 'a > where Self : 'static + :: core :: marker :: Sized",
                &["'a"],
            ),
            (
                parse_quote! {
                    trait Tr { fn f<'a>(&self, x: &'a Rel<Self>) where Self: ::core::marker::Sized, Self: 'static; }
                },
                "< 'a > where Self : 'static , Self : :: core :: marker :: Sized",
                &["'a"],
            ),
            (
                parse_quote! {
                    trait Tr {
                        fn f<'a>(&self, x: &'a Rel<Self>)
                        where Self: ::core::marker::Sized, for<'x> Self: 'x;
                    }
                },
                "< 'a > where Self : :: core :: marker :: Sized , for < 'x > Self : 'x",
                &["'a"],
            ),
            (
                parse_quote! {
                    trait Tr {
                        type Item;
                        fn f<'a>(&self, x: &'a Rel<Self>)
                        where Self: ::core::marker::Sized, Self::Item: 'a;
                    }
                },
                "< 'a > where Self : :: core :: marker :: Sized , Self :: Item : 'a",
                &["'a"],
            ),
            // The ones which are parameters are moved, the others stay in the predicate (the same
            // for a path with a lifetime, which implies nothing: the bound is the written one).
            (
                parse_quote! {
                    trait Tr {
                        fn f<'a>(&self, x: &'a Rel<Self>)
                        where Self: ::core::marker::Sized + 'static + 'a;
                    }
                },
                "< 'a > where Self : 'static + :: core :: marker :: Sized",
                &["'a"],
            ),
            (
                parse_quote! {
                    trait Tr {
                        fn f<'a>(&self, x: Rel<'a, Self>)
                        where Self: ::core::marker::Sized + 'static + 'a;
                    }
                },
                "< 'a > where Self : 'static + :: core :: marker :: Sized",
                &["'a"],
            ),
            // The path with a lifetime does not imply it, so what is not written is not there: the
            // bounds above stay predicates, and `Self: 'a` is not among the lifetimes.
            (
                parse_quote! {
                    trait Tr { fn f<'a>(&self, x: Rel<'a, Self>) where Self: ::core::marker::Sized + 'static; }
                },
                "< 'a > where Self : 'static + :: core :: marker :: Sized",
                &[],
            ),
            (
                parse_quote! {
                    trait Tr {
                        type Item;
                        fn f<'a>(&self, x: Rel<'a, Self>)
                        where Self: ::core::marker::Sized, Self::Item: 'a;
                    }
                },
                "< 'a > where Self : :: core :: marker :: Sized , Self :: Item : 'a",
                &[],
            ),
            // Written alone, a bound by a lifetime which no type mentions: no `Self`, so no bound.
            (
                parse_quote! {
                    trait Tr { fn f<'a>(&self, x: Rel2<Self>) where Self: 'a; }
                },
                "< 'a >",
                &["'a"],
            ),
            // The bound is by the parameter `'a` here, but under a binder, so it stays a predicate,
            // written in the where clause of the trait or in the one of the method (the path with a
            // lifetime implies nothing, so `Self: 'a` is only there if it is written). Audit mutant
            // `o_written_self_forall_moves` (a written `Self: 'a` under a binder moves).
            (
                parse_quote! {
                    trait Tr<'a> where for<'x> Self: 'a { fn f(&self, x: Rel<'a, Self>); }
                },
                "< 'a > where for < 'x > Self : 'a",
                &[],
            ),
            (
                parse_quote! {
                    trait Tr<'a> { fn f(&self, x: Rel<'a, Self>) where for<'x> Self: 'a; }
                },
                "< 'a > where for < 'x > Self : 'a",
                &[],
            ),
            // The control: without the binder the bound moves.
            (
                parse_quote! {
                    trait Tr<'a> where Self: 'a { fn f(&self, x: Rel<'a, Self>); }
                },
                "< 'a >",
                &["'a"],
            ),
            // The reference implies `Self: 'a` whatever else is written, and the written predicate
            // under the binder stays too.
            (
                parse_quote! {
                    trait Tr<'a> where for<'x> Self: 'a { fn f(&self, x: &'a Rel<Self>); }
                },
                "< 'a > where for < 'x > Self : 'a",
                &["'a"],
            ),
        ];
        for (item, declared_generics, outlived) in rows {
            let referrer = of_trait(item.clone());
            assert_eq!(declared(&referrer, 0), declared_generics, "{}", s(&item));
            assert_eq!(
                common::entry(&referrer, 0).self_outlives.as_slice(),
                lifetimes(outlived),
                "{}",
                s(&item)
            );
        }
    }

    /// Rewrites a copy of `tr` in the scope `top_level` with each leaked type replaced by `P<index>`,
    /// as the lookup `how` finds them (`"declared"`: the generics of the trait are declared again by
    /// the copy, `"leakers"`: they are those of the context), and counts those it found.
    fn found(tr: &ItemTrait, referrer: &Referrer, top_level: ContextId, how: &str) -> usize {
        let mut copy = tr.clone();
        let visitor = |ix| -> Type { parse_quote!(P<#ix>) };
        match how {
            "declared" => referrer
                .visitor(top_level, |_, ix, _| visitor(ix))
                .visit_item_trait_mut(&mut copy),
            "leakers" => referrer
                .visitor(top_level, |_, ix, _| visitor(ix))
                .with_outermost_generics_as_leakers()
                .visit_item_trait_mut(&mut copy),
            other => panic!("{}", other),
        }
        s(&copy).matches("P <").count()
    }

    #[test]
    fn the_referrer_finds_an_entry_with_a_written_bound() {
        // The copy has the bound written (in the method, or in the where clause of the trait, where
        // the `impl` does not repeat it), or it is implied by a reference (or there is none, for a
        // path with a lifetime, which implies nothing): the entry is found by both lookups, with the
        // generics of the trait declared again by the copy or taken from the context.
        for tr in [
            parse_quote! {
                trait Tr {
                    fn f<'a>(&'a self) -> Rel<'a, Self> where Self: ::core::marker::Sized + 'a;
                    fn g<'a>(&self, x: Rel<'a, Self>) where Self: ::core::marker::Sized, Self: 'a;
                }
            },
            parse_quote! {
                trait Tr {
                    fn f<'a>(&'a self) -> &'a Rel<Self> where Self: ::core::marker::Sized;
                    fn g<'a>(&self, x: &'a Rel<Self>) where Self: ::core::marker::Sized;
                }
            },
            // The same with the bound written as well.
            parse_quote! {
                trait Tr {
                    fn f<'a>(&'a self) -> &'a Rel<Self> where Self: ::core::marker::Sized + 'a;
                    fn g<'a>(&self, x: &'a Rel<Self>) where Self: ::core::marker::Sized, Self: 'a;
                }
            },
            // Nothing is written or implied: the type of a path is the entry without the bound.
            parse_quote! {
                trait Tr {
                    fn f<'a>(&'a self) -> Rel<'a, Self> where Self: ::core::marker::Sized;
                    fn g<'a>(&self, x: Rel<'a, Self>) where Self: ::core::marker::Sized;
                }
            },
            // Written, and not implied by the type: it makes the context depend on `Self`.
            parse_quote! {
                trait Tr {
                    fn f<'a, T>(&self, x: Rel<'a, T>) where Self: 'a;
                    fn g<'a, T>(&self, x: Rel<'a, T>) where Self: 'a;
                }
            },
            parse_quote! {
                trait Tr<'a> where Self: ::core::marker::Sized + 'a {
                    fn f(&'a self) -> Rel<'a, Self>;
                    fn g(&self, x: Rel<'a, Self>);
                }
            },
            parse_quote! {
                trait Tr<'a> where Self: ::core::marker::Sized {
                    fn f(&'a self) -> &'a Rel<Self>;
                    fn g(&self, x: &'a Rel<Self>);
                }
            },
        ] {
            let tr: ItemTrait = tr;
            let mut leaker = Leaker::new();
            let top_level = leaker.intern_trait(&tr, true);
            let referrer = leaker.result().finish().unwrap();
            assert_eq!(referrer.iter().count(), 1, "{}", s(&tr));
            for how in ["declared", "leakers"] {
                assert_eq!(
                    found(&tr, &referrer, top_level, how),
                    2,
                    "{}: {}",
                    how,
                    s(&tr)
                );
            }
            // `expand()` takes the generics and `Self` of the context. A lifetime of a method is not
            // among them, so the entry that has it is not found there: visit the method instead.
            let ty = common::types(&referrer).next().unwrap().clone();
            let expanded = referrer.expand(top_level, ty.clone(), |_, ix, _| parse_quote!(P<#ix>));
            if tr.generics.params.is_empty() {
                assert_eq!(s(&expanded), s(&ty), "{}", s(&tr));
            } else {
                assert_eq!(s(&expanded), "P < 0usize >", "{}", s(&tr));
            }
            // And the same after the serialization.
            let parsed: Referrer = parse2(referrer.to_token_stream()).unwrap();
            assert_eq!(parsed, referrer);
            for how in ["declared", "leakers"] {
                assert_eq!(
                    found(&tr, &parsed, top_level, how),
                    2,
                    "{}: {}",
                    how,
                    s(&tr)
                );
            }
        }
        // A bound by another lifetime of the method, which the type does not mention, is part of the
        // context too (the predicate bounding `Self` is swept in, as it was): the copy is found where
        // it is written again, and an entry without it is not found there.
        let other: ItemTrait = parse_quote! {
            trait Tr { fn f<'z, 'a>(&self, x: &'a Rel<Self>) where Self: ::core::marker::Sized, Self: 'z; }
        };
        let mut leaker = Leaker::new();
        let top_level = leaker.intern_trait(&other, true);
        let referrer = leaker.result().finish().unwrap();
        assert_eq!(
            common::entry(&referrer, 0).self_outlives.as_slice(),
            lifetimes(&["'a", "'z"])
        );
        for how in ["declared", "leakers"] {
            assert_eq!(found(&other, &referrer, top_level, how), 1, "{}", how);
        }
        let implied: ItemTrait = parse_quote! {
            trait Tr { fn f<'a>(&self, x: &'a Rel<Self>) where Self: ::core::marker::Sized; }
        };
        let mut leaker = Leaker::new();
        let top_level = leaker.intern_trait(&implied, true);
        let referrer = leaker.result().finish().unwrap();
        for how in ["declared", "leakers"] {
            assert_eq!(found(&other, &referrer, top_level, how), 0, "{}", how);
        }
        // With a path, which implies `Self: 'a` for neither, the written `'z` alone is the context.
        let path: ItemTrait = parse_quote! {
            trait Tr { fn f<'z, 'a>(&self, x: Rel<'a, Self>) where Self: ::core::marker::Sized, Self: 'z; }
        };
        let mut leaker = Leaker::new();
        let top_level = leaker.intern_trait(&path, true);
        let referrer = leaker.result().finish().unwrap();
        assert_eq!(
            common::entry(&referrer, 0).self_outlives.as_slice(),
            lifetimes(&["'z"])
        );
        for how in ["declared", "leakers"] {
            assert_eq!(found(&path, &referrer, top_level, how), 1, "{}", how);
        }
    }

    #[test]
    fn a_written_bound_still_needs_self_to_be_leakable() {
        // The bound is not among the predicates of the generics any more, but the implementation
        // needs `Self` to encode it (`&'a __S`), so it cannot be leaked where `Self` cannot (it is
        // the same as when the predicate was one of the bounds of the generics which mention `Self`).
        let leak = |items: proc_macro2::TokenStream| {
            let mut leaker = Leaker::new();
            common::intern_fns(&mut leaker, &Generics::default(), items, false);
            leaker.result().finish()
        };
        let err = leak(quote! { fn f<'a, T>(&self, x: Rel<'a, T>) where Self: 'a; })
            .err()
            .unwrap();
        assert_eq!(
            err[0].notes.first().map(|(_, n)| n.as_str()),
            Some("the bounds of its generics mention `Self`, which cannot be leaked")
        );
        // Without it, or with a bound which the type does not mention, it is fine (nothing in the
        // context mentions `Self`).
        for items in [
            quote! { fn f<'a, T>(&self, x: Rel<'a, T>); },
            quote! { fn f<'a, T>(&self, x: Rel<'a, T>) where Self: 'static; },
        ] {
            let result = leak(items.clone());
            assert!(result.is_ok(), "{}", items);
        }
    }
}

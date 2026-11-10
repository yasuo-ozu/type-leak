//! The methods of the `Leaker` which analyze a whole item (`intern_struct()`, `intern_enum()`,
//! `intern_union()`, `intern_impl()`, `intern_trait()` and `intern_mod()`).
//!
//! The expected values are those of the constructors of the Leaker which analyzed an item and
//! which they replace (`Leaker::from_*()`), taken from the previous version of the crate: the item
//! is analyzed with its own generics as those of the Leaker, which is what the top-level context is
//! that of.

use syn::*;
use template_quote::{quote, ToTokens};
use type_leak::{ContextId, Leaker, LeakerConfig, Referrer, SelfRequirement};

fn s(x: &impl ToTokens) -> String {
    quote!(#x).to_string()
}

/// The entries as `type | generics | self`, in the order of the Referrer.
fn entries(referrer: &Referrer) -> Vec<String> {
    (0..referrer.iter().count())
        .map(|ix| {
            let ty = referrer.iter().nth(ix).unwrap();
            let context = referrer.context(ix);
            let generics = referrer.context_generics(context);
            let self_ty = match referrer.context_self(context) {
                None => "-".to_string(),
                Some(SelfRequirement::Bound(path)) => format!("Self: {}", s(path)),
                Some(SelfRequirement::Type(ty)) => format!("Self = {}", s(ty)),
                Some(other) => format!("{:?}", other),
            };
            format!("{} | {} | {}", s(ty), s(generics), self_ty)
        })
        .collect()
}

fn ids(referrer: &Referrer) -> Vec<usize> {
    (0..referrer.iter().count())
        .map(|ix| referrer.context(ix).index())
        .collect()
}

#[test]
fn intern_struct() {
    let item: ItemStruct = parse_quote!(
        pub struct MyStruct<'a, T1, T2: ::path::to::MyType1<MyType4>> {
            field1: MyType1,
            field2: (MyType2, MyType3<MyType1>, MyType4, MyType5),
            field3: &'a (T1, T2),
        }
    );
    let mut leaker = Leaker::new();
    let top_level = leaker.intern_struct(&item, true);
    let referrer = leaker.result().finish().unwrap();
    assert_eq!(
        entries(&referrer),
        [
            "(MyType2 , MyType3 < MyType1 > , MyType4 , MyType5) |  | -",
            "MyType1 |  | -",
            "MyType4 |  | -",
        ]
    );
    assert_eq!(ids(&referrer), [0, 0, 0]);
    assert_eq!(top_level.index(), 1);
    assert_eq!(
        s(referrer.context_generics(top_level)),
        "< 'a , T1 , T2 : :: path :: to :: MyType1 < MyType4 > >"
    );
    assert_eq!(referrer.contexts().count(), 2);
    assert!(!referrer.has_self());
}

#[test]
fn intern_enum() {
    let item: ItemEnum = parse_quote!(
        pub enum MyEnum<T, U: ::core::marker::Copy> {
            A(Rel, T),
            B { x: Wrap<U>, y: Self2 },
            C(Rel2<Self>),
        }
    );
    let mut leaker = Leaker::new();
    let top_level = leaker.intern_enum(&item, true);
    let referrer = leaker.result().finish().unwrap();
    assert_eq!(
        entries(&referrer),
        [
            "Rel |  | -",
            "Self2 |  | -",
            "Wrap < U > | < U : :: core :: marker :: Copy > | -",
            "Rel2 < Self > | < T , U : :: core :: marker :: Copy > | Self = MyEnum < T , U >",
        ]
    );
    assert_eq!(ids(&referrer), [0, 0, 1, 2]);
    // The top-level context has the generics of the enum and, as `Rel2<Self>` names `Self`, what
    // `Self` is: it is the context of that entry.
    assert_eq!(top_level.index(), 2);
    assert_eq!(top_level, referrer.context(3));
    assert!(referrer.has_self());
    // Without `Self` it is an error to leak `Rel2<Self>`, and the others are not.
    let mut leaker = Leaker::new();
    leaker.intern_enum(&item, false);
    let result = leaker.result();
    assert!(result.finish().is_err());
}

#[test]
fn intern_union() {
    let item: ItemUnion = parse_quote!(
        pub union MyUnion<T: ::core::marker::Copy> {
            a: T,
            b: MyType,
        }
    );
    let mut leaker = Leaker::new();
    let top_level = leaker.intern_union(&item, true);
    let referrer = leaker.result().finish().unwrap();
    assert_eq!(entries(&referrer), ["MyType |  | -"]);
    assert_eq!(ids(&referrer), [0]);
    assert_eq!(top_level.index(), 1);
    assert_eq!(
        s(referrer.context_generics(top_level)),
        "< T : :: core :: marker :: Copy >"
    );
    // A bound must use an absolute path, as for the others.
    let mut leaker = Leaker::new();
    leaker.intern_union(&parse_quote!(union U<T: Copy> { a: T }), true);
    let result = leaker.result();
    assert!(result.finish().is_err());
}

#[test]
fn intern_trait() {
    let item: ItemTrait = parse_quote!(
        pub trait MyTrait<T, U = Rel<T>> {
            type Out: ::core::fmt::Debug;
            fn f(&self, t: T) -> Rel2<Self>;
            fn g<V: ::core::marker::Copy>(&self, v: V) -> Wrap<V, U>;
        }
    );
    let mut leaker = Leaker::new();
    let top_level = leaker.intern_trait(&item, true);
    let referrer = leaker.result().finish().unwrap();
    assert_eq!(
        entries(&referrer),
        [
            "Rel2 < Self > | < T , U > | Self: MyTrait < T , U >",
            "Wrap < V , U > | < U , V : :: core :: marker :: Copy > | -",
            "Rel < T > | < T > | -",
        ]
    );
    assert_eq!(ids(&referrer), [0, 1, 2]);
    // The top-level context has the generics of the trait and, as `Rel2<Self>` names `Self`, what
    // `Self` is: it is the context of that entry.
    assert_eq!(top_level.index(), 0);
    assert_eq!(top_level, referrer.context(0));
    assert_eq!(s(referrer.context_generics(top_level)), "< T , U >");
    // The relative bounds and the relative supertraits are errors.
    let mut leaker = Leaker::new();
    leaker.intern_trait(
        &parse_quote!(
            trait Tr: Relative {}
        ),
        true,
    );
    let result = leaker.result();
    assert!(result.finish().is_err());
    let mut leaker = Leaker::new();
    leaker.intern_trait(
        &parse_quote!(
            trait Tr: ::path::Absolute {}
        ),
        true,
    );
    let result = leaker.result();
    assert!(result.finish().is_ok());
}

#[test]
fn intern_impl() {
    let item: ItemImpl = parse_quote!(
        impl<T> ::path::MyTrait<Rel<T>> for MyType<T> {
            type Out = Rel2<T>;
            fn f(&self, x: Rel3) -> Self::Out {
                todo!()
            }
        }
    );
    let mut leaker = Leaker::new();
    let top_level = leaker.intern_impl(&item, true);
    let referrer = leaker.result().finish().unwrap();
    // The self type, the trait with its arguments and the signatures are analyzed, and the bodies
    // are not.
    assert_eq!(
        entries(&referrer),
        [
            "MyType < T > | < T > | -",
            "Rel < T > | < T > | -",
            "Rel2 < T > | < T > | -",
            "Rel3 |  | -",
        ]
    );
    assert_eq!(ids(&referrer), [0, 0, 0, 1]);
    // The generics of the impl are those of the Leaker, which entries use exactly.
    assert_eq!(top_level.index(), 0);
    assert_eq!(s(referrer.context_generics(top_level)), "< T >");

    // `Self` is the self type, which is leaked as a whole with it.
    let item: ItemImpl = parse_quote!(
        impl<T> ::path::MyTrait for ::path::W<T> {
            fn f(&self) -> Rel<Self> {
                todo!()
            }
        }
    );
    let mut leaker = Leaker::new();
    leaker.intern_impl(&item, true);
    let referrer = leaker.result().finish().unwrap();
    assert_eq!(
        entries(&referrer),
        [
            "Rel < Self > | < T > | Self = :: path :: W < T >",
            // An optional entry: the type of an external crate.
            ":: path :: W < T > | < T > | -",
        ]
    );
    // The trait must not depend on the type context, as a bound.
    let item: ItemImpl = parse_quote!(
        impl Relative for ::path::W {}
    );
    let mut leaker = Leaker::new();
    leaker.intern_impl(&item, true);
    let result = leaker.result();
    let err = result.errors();
    assert_eq!(err.len(), 1);
    assert_eq!(err[0].message, "use absolute path: 'Relative'");
}

#[test]
fn intern_mod() {
    let item: ItemMod = parse_quote!(
        mod m {
            pub struct S<T> {
                x: Rel<T>,
            }
            pub type A = Rel2;
            pub mod inner {
                pub type B = Rel3;
                use super::super::Rel4;
                pub struct C(Rel4);
            }
        }
    );
    let mut leaker = Leaker::new();
    let module = leaker.intern_mod(&item);
    let referrer = leaker.result().finish().unwrap();
    assert_eq!(
        entries(&referrer),
        [
            "Rel < T > | < T > | -",
            "Rel2 |  | -",
            "Rel3 |  | -",
            "Rel4 |  | -",
        ]
    );
    // The module of each: the analyzed one for the first two (its path is `[]`), and `inner`.
    let paths: Vec<Vec<String>> = (0..4)
        .map(|ix| {
            let module = referrer.context_module(referrer.context(ix)).unwrap();
            referrer
                .module_path(module)
                .unwrap()
                .iter()
                .map(|i| i.to_string())
                .collect()
        })
        .collect();
    assert_eq!(paths, [vec![], vec![], vec!["inner"], vec!["inner"]]);
    // The top-level context is the module, which has no generics, and is in no module.
    assert_eq!(referrer.context_module(referrer.context(0)), Some(module));
    assert!(referrer.context_generics(module).params.is_empty());
    assert_eq!(referrer.context_module(module), None);
    // A module without a content has nothing to analyze.
    let mut leaker = Leaker::new();
    let module = leaker.intern_mod(&parse_quote!(
        mod m;
    ));
    let referrer = leaker.result().finish().unwrap();
    assert!(referrer.is_empty());
    assert_eq!(referrer.module_path(module), Some(&[][..]));
}

#[test]
fn from_config_and_the_items_accumulate() {
    // A Leaker with a configuration analyzes the items like one without it, and the contexts of
    // the items are numbered in the order they appear.
    let mut config = LeakerConfig::new();
    config.allow_primitive().allow_std_prelude();
    let mut leaker = Leaker::from_config(config);
    let first: ItemStruct = parse_quote!(
        struct A<T> {
            x: Vec<(Rel, u8)>,
            y: T,
        }
    );
    let second: ItemTrait = parse_quote!(
        trait Tr<T> {
            fn f(&self) -> Option<Wrap<T>>;
        }
    );
    let third: ItemEnum = parse_quote!(
        enum E {
            A(Rel),
            B(Rel2),
        }
    );
    let a = leaker.intern_struct(&first, true);
    let b = leaker.intern_trait(&second, true);
    let c = leaker.intern_enum(&third, true);
    // The next generics take the next id.
    let d = leaker.intern_struct(
        &parse_quote!(
            struct D<V> {
                x: Wrap<V>,
            }
        ),
        true,
    );
    assert_eq!(d, id(2));
    let referrer = leaker.result().finish().unwrap();
    // `Rel` is leaked once, in the context of no generics, which is that of `E`.
    assert_eq!(
        entries(&referrer),
        [
            "Rel |  | -",
            "Rel2 |  | -",
            "Wrap < T > | < T > | -",
            "Wrap < V > | < V > | -"
        ]
    );
    assert_eq!(a, id(1));
    // The same generics are the same context, whichever item they are the generics of.
    assert_eq!(b, a);
    assert_eq!(c, id(0));
    assert_eq!(s(referrer.context_generics(a)), "< T >");
    assert_eq!(s(referrer.context_generics(b)), "< T >");
    assert!(referrer.context_generics(c).params.is_empty());
    // `from_config()` is what `new()` is with the allowed paths.
    let mut leaker = Leaker::from_config(LeakerConfig::new());
    assert!(leaker.allowed_paths.is_empty());
    leaker.allowed_paths.push(parse_quote!(Vec));
    leaker.intern_struct(&first, true);
    // `Vec` is allowed, and `u8` is not (it is a relative path, as it is not a configured one).
    assert_eq!(
        entries(&leaker.result().finish().unwrap()),
        ["(Rel , u8) |  | -"]
    );
}

fn id(n: usize) -> ContextId {
    parse_str(&n.to_string()).unwrap()
}

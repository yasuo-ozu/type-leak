//! Tests for the types that the copy writes through the Repeater in a bound or where predicate.
//!
//! rustc builds the environment of an item from its predicates. If a predicate names a leaked
//! type, the copy writes it as `<() as Repeater<..>>::Type`, which rustc normalizes there. The impl
//! of the Repeater needs the bounds of the generics that the type uses, which name leaked types
//! too, and the environment has them not normalized yet, so rustc cannot prove them (E0277, E0275).
//! The Leaker rejects such an item, and does not make an entry of a type that would do it. A type
//! in a field or in a signature is not in the environment, and a bound that names no leaked type
//! is fine. See [Bounds that name leaked types](type_leak#bounds-that-name-leaked-types).

use syn::*;
use template_quote::quote;
use type_leak::{Leaker, LeakerConfig, NotInternableError, Referrer};

fn s(x: &impl template_quote::ToTokens) -> String {
    quote!(#x).to_string()
}

fn leaker() -> Leaker {
    let mut config = LeakerConfig::new();
    config.allow_primitive().allow_std_prelude();
    Leaker::from_config(config)
}

/// Analyzes `item` with the matching `intern_*`.
fn leak(item: Item) -> std::result::Result<Referrer, NotInternableError> {
    let mut leaker = leaker();
    match &item {
        Item::Struct(item) => leaker.intern_struct(item, true),
        Item::Trait(item) => leaker.intern_trait(item, true),
        Item::Impl(item) => leaker.intern_impl(item, true),
        _ => panic!("unsupported item"),
    }?;
    Ok(leaker.finish())
}

/// The leaked types, required then optional, with their generics.
fn entries(referrer: &Referrer) -> Vec<(String, String)> {
    referrer
        .iter_with_generics()
        .map(|(ty, generics)| (s(ty), s(&quote!(#generics #{&generics.where_clause}))))
        .collect()
}

fn types(referrer: &Referrer) -> Vec<String> {
    referrer.iter().map(s).collect()
}

/// The error, with its headline.
fn rejected(item: Item) -> (String, Option<String>) {
    let err = match leak(item) {
        Ok(referrer) => panic!("not rejected: {:?}", entries(&referrer)),
        Err(err) => err,
    };
    (err.to_string(), err.note())
}

const HEADLINE: &str =
    "is in a bound or where predicate, and a bound of its generics names a leaked type";

#[test]
fn an_entry_in_a_predicate_whose_generic_has_a_leaked_bound_is_rejected() {
    // The Repeater's impl for `Wrap<T>` has `T: Fn(Rel)`, the copy `T: Fn(<..>::Type)`: the
    // predicate `<..>::Type: Sized` of the copy needs the first with the second in the environment.
    let items: Vec<Item> = vec![
        parse_quote!(
            pub struct S<T: Fn(Rel)>(Wrap<T>)
            where
                Wrap<T>: Sized;
        ),
        // A method of a trait, with the generic of the trait or of the method.
        parse_quote!(
            pub trait Tr<T: Fn(Rel)> {
                fn f(&self, a: Wrap<T>)
                where
                    Wrap<T>: Sized;
            }
        ),
        parse_quote!(
            pub trait Tr {
                fn f<T: Fn(Rel)>(&self, a: Wrap<T>)
                where
                    Wrap<T>: Sized;
            }
        ),
        // The bound of another generic, as the where predicate was: it is in the environment too.
        parse_quote!(
            pub struct S<T: Fn(Rel), U: Into<Wrap<T>>>(T, U);
        ),
        parse_quote!(
            pub struct S<T: Fn(Rel), U>(T, U)
            where
                U: Into<Wrap<T>>;
        ),
        // The bound that names `Wrap<T>` is the one that its generic has.
        parse_quote!(
            pub struct S<T: Into<Wrap<T>>>(T);
        ),
        parse_quote!(
            pub struct S<T: Fn(Rel) + Into<Wrap<T>>>(T);
        ),
        // Any bound of a generic does it: `Iterator<Item = ..>` and `AsRef<..>` as well.
        parse_quote!(
            pub struct S<T: Iterator<Item = Rel>>(Wrap<T>)
            where
                Wrap<T>: Sized;
        ),
        parse_quote!(
            pub struct S<T: AsRef<Rel>>(Wrap<T>)
            where
                Wrap<T>: Sized;
        ),
    ];
    for item in items {
        let text = s(&item);
        let (headline, note) = rejected(item);
        assert!(headline.contains(HEADLINE), "{}: {}", text, headline);
        let note = note.unwrap();
        assert!(
            note.contains("rustc cannot normalize") && note.contains("absolute path"),
            "{}: {}",
            text,
            note
        );
    }
}

#[test]
fn the_error_names_the_type_and_the_bound() {
    let (headline, note) = rejected(parse_quote!(
        pub struct S<T: Fn(Rel)>(Wrap<T>)
        where
            Wrap<T>: Sized;
    ));
    assert_eq!(
        headline,
        "`Wrap < T >` is in a bound or where predicate, and a bound of its generics names a leaked type"
    );
    let note = note.unwrap();
    assert!(
        note.contains("the bound `T : Fn (Rel)` of its generics"),
        "{}",
        note
    );
    assert!(
        note.contains("use `Wrap < T >` outside the bounds"),
        "{}",
        note
    );
}

#[test]
fn what_needs_no_environment_is_accepted() {
    let items: Vec<Item> = vec![
        // The entry is in a field only: the environment is built before the field is normalized.
        parse_quote!(
            pub struct S<T: Fn(Rel)>(Wrap<T>);
        ),
        // A bound names `Rel`, which uses no generics, so nothing in its context has a bound.
        parse_quote!(
            pub struct S<T: Fn(Rel), U: Into<Rel>>(T, U)
            where
                Rel: Sized;
        ),
        parse_quote!(
            pub trait Tr {
                fn f<T: Fn(Rel)>(&self, a: Wrap<T>)
                where
                    T: Sized;
            }
        ),
        // The generic has no bound that names a leaked type.
        parse_quote!(
            pub struct S<T: Clone>(Wrap<T>)
            where
                Wrap<T>: Sized;
        ),
        parse_quote!(
            pub struct S<T: ::core::convert::AsRef<u8>>(Wrap<T>)
            where
                Wrap<T>: Sized;
        ),
        // A predicate on another type than a generic is proved by an impl, not by the environment.
        parse_quote!(
            pub struct S<T>(T)
            where
                Vec<Rel2<T, T>>: Clone;
        ),
        // `Self` of a struct is the struct itself, which the copy declares.
        parse_quote!(
            pub struct S<T: Fn(Rel)>(T)
            where
                Self: Sized;
        ),
        parse_quote!(
            pub struct S<T: Fn(Rel)>(T)
            where
                Vec<Self>: Clone;
        ),
    ];
    for item in items {
        let text = s(&item);
        if let Err(err) = leak(item) {
            panic!("{} is rejected: {}", text, err);
        }
    }
}

#[test]
fn the_same_bound_in_the_same_type_is_not_a_cycle() {
    // `Rel` is in two bounds, and its context has no bound.
    let referrer = leak(parse_quote!(
        pub struct S<T: Fn(Rel) + Into<Rel>>(T);
    ))
    .unwrap();
    assert_eq!(types(&referrer), ["Rel"]);
}

#[test]
fn a_lifetime_in_a_bound_is_not_a_cycle() {
    // `Rel<'a>` uses `'a`, and `T: Iterator<Item = Rel<'a>>` is a bound of `T`, whether it is
    // inline or a where predicate: it is not in the context of `Rel<'a>` (F3).
    let inline = leak(parse_quote!(
        pub struct S<'a, T: Iterator<Item = Rel<'a>>>(&'a (), T);
    ))
    .unwrap();
    let predicate = leak(parse_quote!(
        pub struct S<'a, T>(&'a (), T)
        where
            T: Iterator<Item = Rel<'a>>;
    ))
    .unwrap();
    assert_eq!(
        entries(&inline),
        [("Rel < 'a >".to_string(), "< 'a >".to_string())]
    );
    assert_eq!(entries(&predicate), entries(&inline));
}

#[test]
fn a_predicate_on_a_generic_is_a_bound_of_it() {
    // `U: Into<T>` is a bound of `U`: it is in the context of what uses `U`, inline or not.
    let inline = leak(parse_quote!(
        pub struct S<T, U: Into<T>>(Wrap<T>, Wrap2<U>);
    ))
    .unwrap();
    let predicate = leak(parse_quote!(
        pub struct S<T, U>(Wrap<T>, Wrap2<U>)
        where
            U: Into<T>;
    ))
    .unwrap();
    assert_eq!(entries(&predicate), entries(&inline));
    assert_eq!(
        entries(&predicate),
        [
            ("Wrap < T >".to_string(), "< T >".to_string()),
            (
                "Wrap2 < U >".to_string(),
                "< T , U : Into < T > >".to_string()
            ),
        ]
    );
}

#[test]
fn a_type_that_is_an_entry_for_its_parts_is_not_one_in_a_cycle() {
    // The tuple holds two leaked types and a generic whose bound names the tuple, so a cycle if it
    // was the entry: the leaked types are, as they use no generic.
    let referrer = leak(parse_quote!(
        pub struct S<T: Fn((Rel, T, Rel2))>(T);
    ))
    .unwrap();
    assert_eq!(types(&referrer), ["Rel", "Rel2"]);
    // Outside the bounds, the tuple is the entry that covers both.
    let referrer = leak(parse_quote!(
        pub struct S<T: Fn(u8)>(T, (Rel, T, Rel2));
    ))
    .unwrap();
    assert_eq!(types(&referrer), ["(Rel , T , Rel2)"]);
}

#[test]
fn a_reference_that_is_leaked_whole_in_a_cycle_is_rejected() {
    // `&'a Rel` is always an entry as a whole, so its parts cannot stand for it.
    let (headline, _) = rejected(parse_quote!(
        pub struct S<'a, T: Fn(&'a Rel<T>)>(T, &'a ());
    ));
    assert!(headline.contains(HEADLINE), "{}", headline);
}

/// A Leaker for which the types of `::ext`, a crate that it does not allow, are optional entries.
fn ext_leaker() -> Leaker {
    leaker()
}

#[test]
fn an_optional_entry_in_a_cycle_is_copied_as_written() {
    // `::ext::Wrap<T>` may be replaced by the Referrer, but then its path through the Repeater
    // would make a cycle with the bound that names it: it is copied as written, which is no error.
    let mut leaker = ext_leaker();
    leaker
        .intern_struct(
            &parse_quote!(
                pub struct S<T: Into<::ext::Wrap<T>>>(T);
            ),
            true,
        )
        .unwrap();
    assert_eq!(types(&leaker.finish()), Vec::<String>::new());
    // That holds where the type is also out of the bounds: the Referrer finds an entry by the type
    // and its context, wherever it is.
    let mut leaker = ext_leaker();
    leaker
        .intern_struct(
            &parse_quote!(
                pub struct S<T: Into<::ext::Wrap<T>>>(T, ::ext::Wrap<T>);
            ),
            true,
        )
        .unwrap();
    assert_eq!(types(&leaker.finish()), Vec::<String>::new());
    // Without the bound, it is an optional entry as ever.
    let mut leaker = ext_leaker();
    leaker
        .intern_struct(
            &parse_quote!(
                pub struct S<T: Clone>(T, ::ext::Wrap<T>);
            ),
            true,
        )
        .unwrap();
    let referrer = leaker.finish();
    assert_eq!(types(&referrer), [":: ext :: Wrap < T >"]);
    assert!(!referrer.must(0));
}

#[test]
fn an_optional_entry_in_the_bound_of_a_generic_that_a_predicate_uses() {
    // `Wrap<T>` in the predicate is an entry. The Referrer would write `::ext::X` of the bound
    // through the Repeater, where the impl of `Wrap<T>` needs it as written: it is copied as
    // written instead, which is no error.
    let mut leaker = ext_leaker();
    leaker
        .intern_struct(
            &parse_quote!(
                pub struct S<T: Fn(::ext::X)>(T)
                where
                    Wrap<T>: Sized;
            ),
            true,
        )
        .unwrap();
    let referrer = leaker.finish();
    assert_eq!(types(&referrer), ["Wrap < T >"]);
    // Without the predicate, it is an optional entry.
    let mut leaker = ext_leaker();
    leaker
        .intern_struct(
            &parse_quote!(
                pub struct S<T: Fn(::ext::X)>(T);
            ),
            true,
        )
        .unwrap();
    assert_eq!(types(&leaker.finish()), [":: ext :: X"]);
}

#[test]
fn an_entry_that_names_self_in_a_predicate_of_its_trait_is_rejected() {
    // The Repeater's impl for `Rel<Self>` is for the implementors of the trait (`__S: Tr<T>`), so
    // normalizing it for the host needs `Host: Tr<T>`, which needs this predicate.
    for item in [
        parse_quote!(
            pub trait Tr<T>
            where
                Rel<Self>: Sized,
            {
            }
        ),
        parse_quote!(
            pub trait Tr<T>: ::core::convert::Into<Rel<Self>> {}
        ),
        parse_quote!(
            pub trait Tr<T: ::core::convert::AsRef<Rel<Self>>> {}
        ),
        parse_quote!(
            pub trait Tr<T>
            where
                (u8, Rel<Self>): Clone,
            {
            }
        ),
    ] {
        let text = s(&item);
        let (headline, note) = rejected(item);
        assert!(
            headline.contains(HEADLINE)
                || headline.ends_with("names `Self` in a bound or where predicate of its trait"),
            "{}: {}",
            text,
            headline
        );
        let note = note.unwrap();
        // The bound of `T` names the entry (the third), or the predicate of the trait does.
        assert!(
            note.contains("whose impl is for the implementors of the trait (`Self : Tr < T >`)")
                || note.contains("the bound `T : :: core :: convert :: AsRef < Rel < Self > >`"),
            "{}: {}",
            text,
            note
        );
    }
    // The predicates of a method are not those of the trait.
    leak(parse_quote!(
        pub trait Tr<T> {
            fn f(&self)
            where
                Rel<Self>: Sized;
        }
    ))
    .unwrap();
    // `Self` alone is not leaked, and the `Self` of a struct is the struct, which the copy declares.
    leak(parse_quote!(
        pub trait Tr<T>
        where
            Self: Sized,
        {
        }
    ))
    .unwrap();
    leak(parse_quote!(
        pub struct S<T>(T)
        where
            Rel<Self>: Sized;
    ))
    .unwrap();
}

#[test]
fn an_associated_type_of_self_in_a_predicate_is_rejected_if_the_trait_has_a_leaked_bound() {
    // The copy for a concrete implementor resolves `Self::A` by the impl of the trait, which
    // needs its bounds, with the leaked type written through the Repeater.
    let (headline, note) = rejected(parse_quote!(
        pub trait Tr<U: AsRef<Rel>> {
            type A;
            fn f(&self)
            where
                Self::A: Clone;
        }
    ));
    assert!(headline.contains(HEADLINE), "{}", headline);
    assert!(note
        .unwrap()
        .contains("resolves `Self :: A` by the impl of the trait"),);
    // The bound of a method is no bound of the trait.
    leak(parse_quote!(
        pub trait Tr<U: Clone> {
            type A;
            fn f<X: AsRef<Rel>>(&self, x: X)
            where
                Self::A: Clone;
        }
    ))
    .unwrap();
    // And a bound that names no leaked type is fine.
    leak(parse_quote!(
        pub trait Tr<U: AsRef<u8>> {
            type A;
            fn f(&self)
            where
                Self::A: Clone;
        }
    ))
    .unwrap();
}

#[test]
fn self_in_a_predicate_of_an_impl_is_rejected_if_its_self_type_is_leaked_with_a_leaked_bound() {
    // `Self` is the self type, which the copy writes through the Repeater for `W<V>`, and the
    // impl for it needs `V: Fn(Rel)`.
    for item in [
        parse_quote!(
            impl<V: Fn(Rel)> W<V> {
                fn f(&self)
                where
                    Self: Sized,
                {
                }
            }
        ),
        parse_quote!(
            impl<V: Fn(Rel)> ::p::Tr for W<V> where Option<Self>: Clone {}
        ),
    ] {
        let (headline, note) = rejected(item);
        assert!(headline.contains(HEADLINE), "{}", headline);
        assert!(note
            .unwrap()
            .contains("`Self` is the self type of the impl"));
    }
    // The self type is not leaked, or the generic has no leaked bound.
    for item in [
        parse_quote!(
            impl<V: Fn(Rel)> Vec<V> {
                fn f(&self)
                where
                    Self: Sized,
                {
                }
            }
        ),
        parse_quote!(
            impl<V: Clone> W<V> {
                fn f(&self)
                where
                    Self: Sized,
                {
                }
            }
        ),
        // A leaked type in the self type that does not use the generic.
        parse_quote!(
            impl<V: Fn(Rel)> ::p::W for Rel2 {
                fn f(&self, v: V)
                where
                    Self: Sized,
                {
                }
            }
        ),
    ] {
        let text = s(&item);
        if let Err(err) = leak(item) {
            panic!("{} is rejected: {}", text, err);
        }
    }
}

#[test]
fn the_cycle_of_self_is_an_error_where_self_cannot_be_leaked() {
    // The tuple `(T, Self)` is copied as written where `Self` cannot be leaked, which drops what
    // the types in it found: the cycle is kept apart.
    let item: ItemImpl = parse_quote!(
        impl<V: Fn(Rel)> W<V> {
            fn f(&self)
            where
                V: Fn((V, Self)),
            {
            }
        }
    );
    let err = leaker().intern_impl(&item, false).unwrap_err();
    assert!(err.to_string().contains(HEADLINE), "{}", err);
}

#[test]
fn a_scope_that_fails_leaves_no_trace_of_its_cycle() {
    let mut leaker = leaker();
    let generics: Generics = parse_quote!(<T: Fn(Rel)>);
    let result = leaker.enter_generics(&generics, |view| {
        view.visit_where_clause(&parse_quote!(where Wrap<T>: Sized));
    });
    assert!(result.is_err());
    // The next scope is not affected.
    let result = leaker.enter_generics(&generics, |view| {
        view.visit_type(&parse_quote!(Wrap<T>));
    });
    assert!(result.is_ok());
    assert_eq!(types(&leaker.finish()), ["Wrap < T >"]);
}

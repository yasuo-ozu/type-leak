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

/// The help of the error, as `rejected()` gives the headline and the note.
fn help_of(item: Item) -> String {
    match leak(item) {
        Ok(referrer) => panic!("not rejected: {:?}", entries(&referrer)),
        Err(err) => err.help().expect("a help"),
    }
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
        let item: Item = item;
        let text = s(&item);
        let help = help_of(item.clone());
        let (headline, note) = rejected(item);
        assert!(headline.contains(HEADLINE), "{}: {}", text, headline);
        let note = note.unwrap();
        // The note says why, and the help says what to write.
        assert!(
            note.contains("rustc cannot normalize"),
            "{}: {}",
            text,
            note
        );
        assert!(
            help.starts_with("write the leaked type of the bound with an absolute path, or use `")
                && help.ends_with("` outside the bounds and where predicates"),
            "{}: {}",
            text,
            help
        );
    }
}

#[test]
fn the_error_names_the_type_and_the_bound() {
    let item: Item = parse_quote!(
        pub struct S<T: Fn(Rel)>(Wrap<T>)
        where
            Wrap<T>: Sized;
    );
    let help = help_of(item.clone());
    let (headline, note) = rejected(item);
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
    assert_eq!(
        help,
        "write the leaked type of the bound with an absolute path, or use `Wrap<T>` outside the \
         bounds and where predicates"
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

/// The headline and the note of an error, as `rejected()` gives them.
fn unstable(item: Item) -> (String, String) {
    let (headline, note) = rejected(item);
    (headline, note.unwrap())
}

#[test]
fn a_bound_that_names_an_associated_type_that_rustc_normalizes_is_unstable() {
    // The `Item` is equated, so rustc normalizes it as it proves the impl of the Repeater for
    // `Wrap<T>` (`T: Fn(u8)`), and the environment has it as written (`T: Fn(T::Item)`).
    for item in [
        // By the bound of the same generic.
        parse_quote!(
            impl<T: Iterator<Item = u8>> S<T>
            where
                T: Fn(T::Item),
            {
                fn m(&self)
                where
                    Wrap<T>: Sized,
                {
                }
            }
        ),
        // By a predicate of a method, which is not in the context of the entry (F21).
        parse_quote!(
            impl<T: Iterator> S<T>
            where
                T: Fn(T::Item),
            {
                fn m(&self)
                where
                    T: Iterator<Item = Rel<T>>,
                {
                }
            }
        ),
        parse_quote!(
            pub trait Tr<T: Iterator>
            where
                T: Fn(<T as Iterator>::Item),
            {
                fn m(&self)
                where
                    T: Iterator<Item = Rel<T>>;
            }
        ),
        // A projection that an impl defines.
        parse_quote!(
            pub struct S<T: Fn(<Vec<u8> as IntoIterator>::Item)>(T)
            where
                Wrap<T>: Sized;
        ),
        // The `Output` of `Fn(..)`.
        parse_quote!(
            pub struct S<F: Fn(u8) -> u16, G: Fn(F::Output)>(F, G)
            where
                Wrap<G>: Sized;
        ),
    ] {
        let item: Item = item;
        let text = s(&item);
        let help = help_of(item.clone());
        let (headline, note) = unstable(item);
        assert!(
            headline.contains("a bound of its generics names `"),
            "{}: {}",
            text,
            headline
        );
        assert!(
            note.contains("rustc normalizes in the impl of the Repeater"),
            "{}: {}",
            text,
            note
        );
        assert!(
            help.starts_with("write the bound without `") && help.contains("`, or use `"),
            "{}: {}",
            text,
            help
        );
    }
    // The `Item` of an iterator that nothing equates is rigid, and so is the one of a generic that
    // the type does not use.
    for item in [
        parse_quote!(
            impl<T: Iterator> S<T>
            where
                T: Fn(T::Item),
            {
                fn m(&self)
                where
                    Wrap<T>: Sized,
                {
                }
            }
        ),
        parse_quote!(
            impl<T: Iterator, U> S<T, U>
            where
                T: Iterator<Item = u8> + Fn(T::Item),
            {
                fn m(&self)
                where
                    Wrap<U>: Sized,
                {
                }
            }
        ),
        // In a position, where it is normalized after the environment is built.
        parse_quote!(
            impl<T: Iterator<Item = u8>> S<T>
            where
                T: Fn(T::Item),
            {
                fn m(&self, a: Wrap<T>) {}
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
fn self_and_its_associated_types_in_a_bound_are_unstable() {
    // F19: the self type is leaked, and the bound of the generic names `Self`.
    let (headline, note) = unstable(parse_quote!(
        impl S {
            fn m<X>(&self)
            where
                X: Fn(Self) -> u8,
                Rel<X>: Default,
            {
            }
        }
    ));
    assert_eq!(
        headline,
        "`Rel < X >` is in a bound or where predicate, and a bound of its generics names `Self`, which is a leaked type"
    );
    assert!(
        note.contains("the bound `X : Fn (Self) -> u8` of its generics names `Self`"),
        "{}",
        note
    );
    // F20: the bound names `Self::A0` of a trait.
    let item: Item = parse_quote!(
        pub trait Tr {
            type A0;
            fn m<X>(&self)
            where
                X: Fn(&Self::A0) -> u8,
                Rel<X>: Default;
        }
    );
    let help = help_of(item.clone());
    let (headline, _note) = unstable(item);
    assert!(
        headline.ends_with("a bound of its generics names `Self :: A0`"),
        "{}",
        headline
    );
    assert_eq!(
        help,
        "write the bound without `Self::A0`, or use `Rel<X>` outside the bounds and where \
         predicates"
    );
    // The self type is not leaked, or the entry is in a position, or `Self` is the struct, which
    // the copy declares itself.
    for item in [
        parse_quote!(
            impl Vec<u8> {
                fn m<X>(&self)
                where
                    X: Fn(Self) -> u8,
                    Rel<X>: Default,
                {
                }
            }
        ),
        parse_quote!(
            impl S {
                fn m<X>(&self, a: Rel<X>)
                where
                    X: Fn(Self) -> u8,
                {
                }
            }
        ),
        parse_quote!(
            pub struct S<X: Fn(Self) -> u8>(X)
            where
                Rel<X>: Default;
        ),
    ] {
        let text = s(&item);
        if let Err(err) = leak(item) {
            panic!("{} is rejected: {}", text, err);
        }
    }
}

#[test]
fn self_a_in_a_predicate_of_an_impl_is_unstable_with_a_leaked_bound_of_the_impl() {
    // F14: the context of `Self::A1` of an impl has no generics, but the impl of the trait that
    // rustc proves has all those of the impl.
    for item in [
        parse_quote!(
            impl<T> ::p::Tr<T> for u8
            where
                T: Fn(Rel),
            {
                type A1 = u8;
                fn m<X>(&self, a: Rel)
                where
                    X: Into<Self::A1>,
                {
                }
            }
        ),
        parse_quote!(
            impl<T: Fn(Rel)> ::p::Tr<T> for u8 {
                type A1 = u8;
                fn m(&self)
                where
                    u8: Into<Self::A1>,
                {
                }
            }
        ),
    ] {
        let text = s(&item);
        let (headline, note) = unstable(item);
        assert!(
            headline.starts_with("`Self :: A1` is in a bound or where predicate")
                && headline.ends_with("names a leaked type"),
            "{}: {}",
            text,
            headline
        );
        assert!(
            note.contains("resolves `Self :: A1` by the impl of the trait")
                && note.contains("of the impl"),
            "{}: {}",
            text,
            note
        );
    }
    for item in [
        parse_quote!(
            impl<T: Clone> ::p::Tr<T> for u8 {
                type A1 = u8;
                fn m<X>(&self, a: Rel)
                where
                    X: Into<Self::A1>,
                {
                }
            }
        ),
        // `Self::A1` and the leaked type are in positions only.
        parse_quote!(
            impl<T> ::p::Tr<T> for u8
            where
                T: Fn(Rel),
            {
                type A1 = u8;
                fn m<X>(&self, a: Self::A1, b: Rel)
                where
                    X: Into<Rel>,
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
fn an_optional_entry_that_an_unstable_bound_makes_cyclic_is_copied_as_written() {
    // F17: `&'a ::ext::W<T, ::ext::B0>` is in a predicate that names `'a`, so it is in its own
    // context. It is leaked as a whole, so its parts cannot stand for it either.
    let mut leaker = ext_leaker();
    leaker
        .intern_struct(
            &parse_quote!(
                pub struct S<'a, T: 'a>
                where
                    T: Fn(&'a ::ext::W<T, ::ext::B0>) -> u8,
                {
                    f0: u8,
                }
            ),
            true,
        )
        .unwrap();
    assert_eq!(types(&leaker.finish()), Vec::<String>::new());
    // Without the lifetime, it is not in its own context, and `B0` is an optional entry.
    let mut leaker = ext_leaker();
    leaker
        .intern_struct(
            &parse_quote!(
                pub struct S<'a, T: 'a>
                where
                    T: Fn(::ext::W<T, ::ext::B0>) -> u8,
                {
                    f0: &'a T,
                }
            ),
            true,
        )
        .unwrap();
    assert_eq!(types(&leaker.finish()), [":: ext :: B0"]);
    // An optional entry in a predicate, with a bound of its context that names an `Item` that is
    // equated, or `Self::A`.
    let mut leaker = ext_leaker();
    leaker
        .intern_impl(
            &parse_quote!(
                impl<T: Iterator> S<T>
                where
                    T: Fn(T::Item),
                {
                    fn m(&self)
                    where
                        T: Iterator<Item = ::ext::W<T>>,
                    {
                    }
                }
            ),
            true,
        )
        .unwrap();
    assert_eq!(types(&leaker.finish()), ["S < T >"]);
    let mut leaker = ext_leaker();
    leaker
        .intern_trait(
            &parse_quote!(
                pub trait Tr {
                    type A0;
                    fn m<X>(&self)
                    where
                        X: Fn(&Self::A0) -> u8,
                        ::ext::Rel<X>: Default;
                }
            ),
            true,
        )
        .unwrap();
    assert_eq!(types(&leaker.finish()), Vec::<String>::new());
    // The optional types of a self type that a bound of an entry names (`Self`).
    let mut leaker = ext_leaker();
    leaker
        .intern_impl(
            &parse_quote!(
                impl ::ext::S {
                    fn m<X>(&self)
                    where
                        X: Fn(Self) -> u8,
                        Rel<X>: Default,
                    {
                    }
                }
            ),
            true,
        )
        .unwrap();
    assert_eq!(types(&leaker.finish()), ["Rel < X >"]);
    // Without a predicate that names the entry, the self type is an optional entry as ever.
    let mut leaker = ext_leaker();
    leaker
        .intern_impl(
            &parse_quote!(
                impl ::ext::S {
                    fn m<X>(&self, a: Rel<X>)
                    where
                        X: Fn(Self) -> u8,
                    {
                    }
                }
            ),
            true,
        )
        .unwrap();
    let referrer = leaker.finish();
    assert_eq!(types(&referrer), ["Rel < X >", ":: ext :: S"]);
    assert!(referrer.must(0) && !referrer.must(1));
}

/// rules.md, Bounds that name leaked types: `<Self as ::p::Tr>::A0` in a bound names the associated
/// type of the host, which rustc normalizes in the impl of the Repeater but not yet in the
/// environment, as `Self::A0` does. The qualified path is as unstable as the other. Audit mutant
/// `b_selfassoc_qself_off`.
#[test]
fn a_qualified_associated_type_of_self_in_a_bound_is_unstable() {
    for (written, shown) in [
        ("Self::A0", "Self :: A0"),
        ("<Self as ::p::Tr>::A0", "< Self as :: p :: Tr > :: A0"),
    ] {
        let item: Item = parse_str(&format!(
            "pub trait Tr {{ type A0; fn m<X>(&self) where X: Fn(&{}) -> u8, Rel<X>: Default; }}",
            written
        ))
        .unwrap();
        let (headline, note) = unstable(item);
        assert_eq!(
            headline,
            format!(
                "`Rel < X >` is in a bound or where predicate, and a bound of its generics names \
                 `{}`",
                shown
            )
        );
        assert!(
            note.contains(
                "rustc normalizes to the associated type of the host in the impl of the Repeater"
            ),
            "{}",
            note
        );
    }
}

/// rules.md, Bounds that name leaked types / Self rules: a predicate on a type that is not a
/// generic and names `Self` alone (`Vec<Self>: Clone`, `Self: Clone`) is proved by an impl, so it
/// is no unstable bound: an entry that names `Self` stays accepted, and `Self` stays an entry
/// (here the self type of the impl). Audit mutants `b_selftype_pred_on_env` and
/// `b_needsenv_self_bound_only`.
#[test]
fn bare_self_predicates_stay_accepted() {
    for item in [
        parse_quote!(
            impl S
            where
                Vec<Self>: Clone,
                Rel<Self>: Sized,
            {
            }
        ),
        parse_quote!(
            impl S
            where
                Self: Clone,
                Rel<Self>: Sized,
            {
            }
        ),
    ] {
        let item: Item = item;
        let text = s(&item);
        let referrer = leak(item).unwrap_or_else(|e| panic!("{}: {}", text, e));
        assert_eq!(types(&referrer), ["Rel < Self >", "S"], "{}", text);
        assert!(referrer.must(0) && referrer.must(1), "{}", text);
    }
}

/// rules.md, Bounds that name leaked types: a predicate on a type that is not a generic
/// (`Vec<Self::A>: AsRef<Self::A>`) is proved by an impl and names no unstable bound as far as
/// `Self::A` goes, so an entry in a bound of a method of the trait is accepted. Audit mutant
/// `b_nonparam_pred_bounds_count`.
#[test]
fn a_predicate_on_a_non_generic_type_with_self_assoc_is_no_unstable_bound() {
    let referrer = leak(parse_quote!(
        pub trait Tr
        where
            Vec<Self::A>: AsRef<Self::A>,
        {
            type A;
            fn m<X: Into<Rel<Self>>>(&self);
        }
    ))
    .unwrap();
    assert_eq!(types(&referrer), ["Rel < Self >"]);
    assert!(referrer.must(0));
}

/// rules.md, Bounds that name leaked types: the self type of an impl is an optional entry (it is a
/// type of an external crate), but not where a predicate names `Self` and a bound of its generic
/// names a leaked type: the Referrer's replacement of it would make a cycle. Here the self type
/// `::ext::W<T>` has `T: Fn(Rel)` and `Self: Clone` is written: the only entry is `Rel`. Audit
/// mutant `b_not_optional_in_self_type_off` (which adds the optional `::ext::W<T>`).
#[test]
fn an_optional_self_type_is_no_entry_where_self_is_in_a_predicate() {
    let referrer = leak(parse_quote!(
        impl<T: Fn(Rel)> ::p::Tr for ::ext::W<T> where Self: Clone {}
    ))
    .unwrap();
    assert_eq!(types(&referrer), ["Rel"]);
}

/// rules.md, Bounds that name leaked types: a generic whose bounds are all traits of the standard
/// library, and none defines `Item`, keeps `T::Item` rigid: the name of each trait of the list
/// (`STD_TRAITS`) is one, and the bound `Fn(T::Item) -> u8` next to it is stable. A trait of the
/// analyzed code, or a path of another crate, may define the `Item` by a supertrait, so the bound
/// is unstable. Audit mutants `std_without_<name>` (one per name; also `b_std_without_*`).
#[test]
fn the_standard_traits_do_not_define_an_associated_type() {
    let names = [
        "Sized",
        "Send",
        "Sync",
        "Unpin",
        "Copy",
        "Clone",
        "Drop",
        "Eq",
        "PartialEq",
        "Ord",
        "PartialOrd",
        "Default",
        "Debug",
        "Display",
        "Hash",
        "AsRef",
        "AsMut",
        "Into",
        "From",
        "TryFrom",
        "TryInto",
        "ToString",
        "ToOwned",
        "Borrow",
        "BorrowMut",
        "Fn",
        "FnMut",
        "FnOnce",
        "Iterator",
        "IntoIterator",
        "DoubleEndedIterator",
        "ExactSizeIterator",
        "FusedIterator",
        "Extend",
        "FromIterator",
        "Deref",
        "DerefMut",
        "Add",
        "Sub",
        "Mul",
        "Div",
        "Rem",
        "Neg",
        "Not",
        "Index",
        "IndexMut",
        "FromStr",
        "Future",
        "IntoFuture",
        "Any",
    ];
    for name in names {
        let item: Item = parse_str(&format!(
            "pub trait Tr<T: {} + Fn(T::Item) -> u8> {{ fn m(&self) where Rel<T>: Default; }}",
            name
        ))
        .unwrap();
        // The relative names are allowed paths, as the prelude's are in `leaker()`.
        let mut leaker = leaker();
        leaker.allowed_paths.push(parse_str(name).unwrap());
        match &item {
            Item::Trait(item) => leaker.intern_trait(item, true),
            _ => unreachable!(),
        }
        .unwrap_or_else(|e| panic!("{}: {}", name, e));
        assert_eq!(types(&leaker.finish()), ["Rel < T >"], "{}", name);
    }
    // The absolute path of a trait of the standard library is one too.
    let referrer = leak(parse_quote!(
        pub trait Tr<T: ::core::marker::Sized + Fn(T::Item) -> u8> {
            fn m(&self)
            where
                Rel<T>: Default;
        }
    ))
    .unwrap();
    assert_eq!(types(&referrer), ["Rel < T >"]);
    // A trait that is not, as a bare name or as a path, may define it.
    for name in ["MyTrait", "::p::Tr", "p::Tr"] {
        let item: Item = parse_str(&format!(
            "pub trait Tr<T: {} + Fn(T::Item) -> u8> {{ fn m(&self) where Rel<T>: Default; }}",
            name
        ))
        .unwrap();
        let (headline, _) = rejected(item);
        assert!(
            headline.contains("names `T :: Item`"),
            "{}: {}",
            name,
            headline
        );
    }
}

/// rules.md, Bounds that name leaked types: `Self::A` in a bound of the impl is not written in the
/// copy: the headline says to write the trait out, `<Self as ::p::Tr>::A`. Audit mutant
/// `s_selfassoc_bounds_off` (only the other error of the item remains).
#[test]
fn self_assoc_in_a_bound_of_an_impl_cannot_be_written_in_the_repeater() {
    let item: Item = parse_quote!(
        impl<T: AsRef<Self::A>> ::p::Tr for ::p::S<T> {
            fn f(&self, x: Rel<T>) {}
        }
    );
    let err = leak(item).err().unwrap();
    let headlines: Vec<String> = err.items().map(|item| item.message).collect();
    assert_eq!(
        headlines,
        [
            "write `<Self as ::p::Tr>::A` instead of `Self::A` in a bound of the generics of \
             `Rel<T>`: the copy cannot tell which trait `A` comes from",
            "`Self :: A` is in a bound or where predicate, and a bound of its generics names \
             `Self :: A`"
        ]
    );
}

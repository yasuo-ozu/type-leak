//! Tests for the types that the copy writes through the Repeater in a bound or where predicate.
//!
//! rustc builds the environment of an item from its predicates. If a predicate names a leaked
//! type, the copy writes it as `<() as Repeater<..>>::Type`, which rustc normalizes there. The impl
//! of the Repeater needs the bounds of the generics that the type uses, which name leaked types
//! too, and the environment has them not normalized yet, so rustc cannot prove them (E0277, E0275).
//! The Leaker rejects such an item, and does not make an entry of a type that would do it. A type
//! in a field or in a signature is not in the environment, and a bound that names no leaked type
//! is fine. See [Bounds that name leaked types](type_leak#bounds-that-name-leaked-types).

use crate::common;

use common::{
    entries_with_generics as entries, intern_item, leaker_allowing, messages, s, Recovered,
};
use syn::*;
use type_leak::{Allow, Cause, Leaker, LeakerResultError, Referrer};

fn leaker() -> Leaker<impl Allow> {
    leaker_allowing(&[])
}

/// Analyzes `item` with the matching `intern_*`.
fn leak(item: Item) -> std::result::Result<Referrer, LeakerResultError> {
    let mut leaker = leaker();
    intern_item(&mut leaker, &item);
    leaker.result().finish()
}

/// The error, with its headline.
fn rejected(item: Item) -> (String, Option<String>) {
    let err = match leak(item) {
        Ok(referrer) => panic!("not rejected: {:?}", entries(&referrer)),
        Err(err) => err,
    };
    (
        messages(&err),
        err[0]
            .notes
            .first()
            .map(|(_, n)| n.as_str())
            .map(str::to_string),
    )
}

/// The help of the error, as `rejected()` gives the headline and the note.
fn help_of(item: Item) -> String {
    match leak(item) {
        Ok(referrer) => panic!("not rejected: {:?}", entries(&referrer)),
        Err(err) => err[0].help_text().expect("a help").to_string(),
    }
}

const HEADLINE: &str =
    "is in a bound or where predicate, and a bound of its generics names a leaked type";

#[test]
fn an_entry_in_a_predicate_whose_generic_has_a_leaked_bound_is_rejected() {
    // The Repeater's impl for `Wrap<T>` has `T: Fn(Rel)`, the copy `T: Fn(<..>::Type)`: the
    // predicate `<..>::Type: Sized` of the copy needs the first with the second in the environment.
    // (The struct and the two traits with that predicate are F2 of `tests/findings/findings.rs`.)
    let items: Vec<Item> = vec![
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
        // A bound of a generic names `Self`, but the self type is not leaked, or `Self` is the
        // struct, which the copy declares itself.
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
            pub struct S<X: Fn(Self) -> u8>(X)
            where
                Rel<X>: Default;
        ),
    ];
    for item in items {
        let text = s(&item);
        if let Err(err) = leak(item) {
            panic!("{} is rejected: {}", text, messages(&err));
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
    assert_eq!(common::type_texts(&referrer), ["Rel"]);
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
    assert_eq!(common::type_texts(&referrer), ["Rel", "Rel2"]);
    // Outside the bounds, the tuple is the entry that covers both.
    let referrer = leak(parse_quote!(
        pub struct S<T: Fn(u8)>(T, (Rel, T, Rel2));
    ))
    .unwrap();
    assert_eq!(common::type_texts(&referrer), ["(Rel , T , Rel2)"]);
}

#[test]
fn a_reference_that_is_leaked_whole_in_a_cycle_is_rejected() {
    // `&'a Rel` is always an entry as a whole, so its parts cannot stand for it.
    let (headline, _) = rejected(parse_quote!(
        pub struct S<'a, T: Fn(&'a Rel<T>)>(T, &'a ());
    ));
    assert!(headline.contains(HEADLINE), "{}", headline);
}

/// The types of `::ext`, a crate that the Leaker does not allow, are entries. One that a cycle needs
/// is copied as written, which is no error: the Referrer would replace it, but its path through the
/// Repeater would make a cycle with the bound that names it. (An optional entry in a predicate that
/// names a lifetime is F17 of `tests/findings/findings.rs`.)
#[test]
fn an_optional_entry_that_a_cycle_needs_is_copied_as_written() {
    let rows: Vec<(Item, &[&str])> = vec![
        // The bound names `::ext::Wrap<T>`, and so does a field (the Referrer finds an entry by the
        // type and its context, wherever it is); without the bound it is an optional entry as ever.
        (
            parse_quote!(
                pub struct S<T: Into<::ext::Wrap<T>>>(T);
            ),
            &[],
        ),
        (
            parse_quote!(
                pub struct S<T: Into<::ext::Wrap<T>>>(T, ::ext::Wrap<T>);
            ),
            &[],
        ),
        (
            parse_quote!(
                pub struct S<T: Clone>(T, ::ext::Wrap<T>);
            ),
            &[":: ext :: Wrap < T >"],
        ),
        // `Wrap<T>` in the predicate is an entry. The Referrer would write `::ext::X` of the bound
        // through the Repeater, where the impl of `Wrap<T>` needs it as written: it is copied as
        // written instead. Without the predicate, it is an optional entry.
        (
            parse_quote!(
                pub struct S<T: Fn(::ext::X)>(T)
                where
                    Wrap<T>: Sized;
            ),
            &["Wrap < T >"],
        ),
        (
            parse_quote!(
                pub struct S<T: Fn(::ext::X)>(T);
            ),
            &[":: ext :: X"],
        ),
        // An optional entry in a predicate, with a bound of its context that names an `Item` that
        // is equated, or `Self::A`.
        (
            parse_quote!(
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
            &["S < T >"],
        ),
        (
            parse_quote!(
                pub trait Tr {
                    type A0;
                    fn m<X>(&self)
                    where
                        X: Fn(&Self::A0) -> u8,
                        ::ext::Rel<X>: Default;
                }
            ),
            &[],
        ),
        // The optional types of a self type that a bound of an entry names (`Self`); without a
        // predicate that names the entry, the self type is an optional entry as ever.
        (
            parse_quote!(
                impl ::ext::S {
                    fn m<X>(&self)
                    where
                        X: Fn(Self) -> u8,
                        Rel<X>: Default,
                    {
                    }
                }
            ),
            &["Rel < X >"],
        ),
        (
            parse_quote!(
                impl ::ext::S {
                    fn m<X>(&self, a: Rel<X>)
                    where
                        X: Fn(Self) -> u8,
                    {
                    }
                }
            ),
            &["Rel < X >", ":: ext :: S"],
        ),
    ];
    for (item, expected) in rows {
        let text = s(&item);
        let referrer = leak(item).unwrap_or_else(|e| panic!("{}: {}", text, messages(&e)));
        assert_eq!(common::type_texts(&referrer), expected, "{}", text);
    }
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
            note.contains("whose impl is for the implementors of the trait (`Self : Tr<T>`)")
                || note.contains("the bound `T : ::core::convert::AsRef<Rel<Self>>`"),
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
        .contains("resolves `Self::A` by the impl of the trait"),);
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
            panic!("{} is rejected: {}", text, messages(&err));
        }
    }
}

#[test]
fn the_cycle_of_self_in_a_tuple_of_a_predicate_is_an_error() {
    // `Self` in the tuple `(V, Self)` of a predicate is the self type of the impl, whose generics
    // have the bound `V: Fn(Rel)` that names a leaked type.
    let item: ItemImpl = parse_quote!(
        impl<V: Fn(Rel)> W<V> {
            fn f(&self)
            where
                V: Fn((V, Self)),
            {
            }
        }
    );
    let mut leaker = leaker();
    leaker.intern_impl(&item);
    let result = leaker.result();
    let err = result.errors();
    assert_eq!(err.len(), 1, "{}", messages(err));
    assert!(messages(err).contains(HEADLINE), "{}", messages(err));
    assert_eq!(err[0].cause, Cause::Type(parse_quote!(Self)));
}

#[test]
fn a_scope_that_fails_leaves_no_trace_of_its_cycle() {
    let mut leaker = leaker();
    let generics: Generics = parse_quote!(<T: Fn(Rel)>);
    leaker.enter_generics(&generics, |view| {
        view.visit_where_clause(&parse_quote!(where Wrap<T>: Sized));
    });
    // The next scope is not affected.
    leaker.enter_generics(&generics, |view| {
        view.visit_type(&parse_quote!(Wrap<T>));
    });
    // The first scope has the only error, and the entry is that of the second.
    let result = leaker.result();
    assert_eq!(result.errors().len(), 1);
    assert_eq!(common::type_texts(&result.recovered()), ["Wrap < T >"]);
}

/// The headline and the note of an error, as `rejected()` gives them.
fn unstable(item: Item) -> (String, String) {
    let (headline, note) = rejected(item);
    (headline, note.unwrap())
}

#[test]
fn a_bound_that_names_an_associated_type_that_rustc_normalizes_is_unstable() {
    // rustc normalizes the projection as it proves the impl of the Repeater for `Wrap<T>`, and the
    // environment has it as written. An `Item` that a predicate of a method equates is F21's
    // (`tests/findings/findings.rs`).
    for item in [
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
    // the type does not use. The `Item` that a bound of the same generic equates is not rigid, but
    // the context of the entry leaves out `T: Fn(T::Item)` then: the Repeater's impl has `T:
    // Iterator<Item = u8>` only (F26, `tests/findings/findings.rs`).
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
            panic!("{} is rejected: {}", text, messages(&err));
        }
    }
    // By the bound of the same generic.
    let referrer = leak(parse_quote!(
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
    ))
    .unwrap();
    assert_eq!(
        entries(&referrer),
        [
            // The self type is declared with the bounds of the item: it keeps them.
            (
                "S < T >".to_string(),
                "< T : Fn (T :: Item) + Iterator < Item = u8 > >".to_string()
            ),
            (
                "Wrap < T >".to_string(),
                "< T : Iterator < Item = u8 > >".to_string()
            ),
        ]
    );
}

/// The `Leaker` docs, Bounds that name leaked types: `<Self as ::p::Tr>::A0` in a bound names the
/// associated type of the host, which rustc normalizes in the impl of the Repeater but not yet in
/// the environment, as `Self::A0` does (F20 of `tests/findings/findings.rs`). The qualified path is
/// as unstable as the other. Audit mutant `b_selfassoc_qself_off`.
#[test]
fn a_qualified_associated_type_of_self_in_a_bound_is_unstable() {
    let (headline, note) = unstable(parse_quote!(
        pub trait Tr {
            type A0;
            fn m<X>(&self)
            where
                X: Fn(&<Self as ::p::Tr>::A0) -> u8,
                Rel<X>: Default;
        }
    ));
    assert_eq!(
        headline,
        "`Rel<X>` is in a bound or where predicate, and a bound of its generics names \
         `<Self as ::p::Tr>::A0`"
    );
    assert!(
        note.contains(
            "rustc normalizes to the associated type of the host in the impl of the Repeater"
        ),
        "{}",
        note
    );
}

/// The `Leaker` docs, Bounds that name leaked types: a predicate on a type that is not a
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
        let referrer = leak(item).unwrap_or_else(|e| panic!("{}: {}", text, messages(&e)));
        assert_eq!(
            common::type_texts(&referrer),
            ["Rel < Self >", "S"],
            "{}",
            text
        );
    }
}

/// The `Leaker` docs, Bounds that name leaked types: a predicate on a type that is not a generic
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
    assert_eq!(common::type_texts(&referrer), ["Rel < Self >"]);
}

/// The `Leaker` docs, Bounds that name leaked types: the self type of an impl is an optional entry
/// (it is a type of an external crate), but not where a predicate names `Self` and a bound of its
/// generic names a leaked type: the Referrer's replacement of it would make a cycle. Here the self
/// type `::ext::W<T>` has `T: Fn(Rel)` and `Self: Clone` is written: the only entry is `Rel`. Audit
/// mutant `b_not_optional_in_self_type_off` (which adds the optional `::ext::W<T>`).
#[test]
fn an_optional_self_type_is_no_entry_where_self_is_in_a_predicate() {
    let referrer = leak(parse_quote!(
        impl<T: Fn(Rel)> ::p::Tr for ::ext::W<T> where Self: Clone {}
    ))
    .unwrap();
    assert_eq!(common::type_texts(&referrer), ["Rel"]);
}

/// The `Leaker` docs, Bounds that name leaked types: a generic whose bounds are all traits of the
/// standard library, and none defines `Item`, keeps `T::Item` rigid: the name of each trait of the
/// list (`STD_TRAITS`) is one, and the bound `Fn(T::Item) -> u8` next to it is stable. A trait of
/// the analyzed code, or a path of another crate, may define the `Item` by a supertrait, so the
/// bound is unstable. Audit mutants `std_without_<name>` (one per name; also `b_std_without_*`).
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
        let mut leaker = leaker_allowing(&[name]);
        intern_item(&mut leaker, &item);
        let referrer = leaker
            .result()
            .finish()
            .unwrap_or_else(|e| panic!("{}: {}", name, messages(&e)));
        assert_eq!(common::type_texts(&referrer), ["Rel < T >"], "{}", name);
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
    assert_eq!(common::type_texts(&referrer), ["Rel < T >"]);
    // A trait that is not, as a bare name or as a path, may define it. The bound `Fn(T::Item) ->
    // u8` that names it is left out of the context of the entry (F26), so the item is accepted.
    for name in ["MyTrait", "::p::Tr", "p::Tr"] {
        let item: Item = parse_str(&format!(
            "pub trait Tr<T: {} + Fn(T::Item) -> u8> {{ fn m(&self) where Rel<T>: Default; }}",
            name
        ))
        .unwrap();
        let mut leaker = leaker_allowing(&["MyTrait", "p::Tr", "::p"]);
        intern_item(&mut leaker, &item);
        let referrer = leaker
            .result()
            .finish()
            .unwrap_or_else(|e| panic!("{}: {}", name, messages(&e)));
        assert_eq!(
            entries(&referrer),
            [(
                "Rel < T >".to_string(),
                s(&parse_str::<Generics>(&format!("<T: {}>", name)).unwrap())
            )],
            "{}",
            name
        );
    }
    // The bound of another generic that names the `Item` is not left out, and it is unstable.
    for name in ["MyTrait", "::p::Tr", "p::Tr"] {
        let item: Item = parse_str(&format!(
            "pub trait Tr<T: {}, U: Fn(T::Item) -> u8> {{ fn m(&self) where Rel<U>: Default; }}",
            name
        ))
        .unwrap();
        let mut leaker = leaker_allowing(&["MyTrait", "p::Tr", "::p"]);
        intern_item(&mut leaker, &item);
        let result = leaker.result();
        let err = result.errors();
        assert!(
            messages(err).contains("names `T::Item`"),
            "{}: {}",
            name,
            messages(err)
        );
    }
}

/// The `Leaker` docs, Bounds that name leaked types: `Self::A` in a bound of the impl is not
/// written in the copy: the headline says to write the trait out, `<Self as ::p::Tr>::A`. Audit
/// mutant `s_selfassoc_bounds_off` (only the other error of the item remains).
#[test]
fn self_assoc_in_a_bound_of_an_impl_cannot_be_written_in_the_repeater() {
    let item: Item = parse_quote!(
        impl<T: AsRef<Self::A>> ::p::Tr for ::p::S<T> {
            fn f(&self, x: Rel<T>) {}
        }
    );
    let err = leak(item).err().unwrap();
    let headlines: Vec<String> = err.into_iter().map(|item| item.message).collect();
    assert_eq!(
        headlines,
        [
            "write `<Self as ::p::Tr>::A` instead of `Self::A` in a bound of the generics of \
             `Rel<T>`: the copy cannot tell which trait `A` comes from",
            "`Self::A` is in a bound or where predicate, and a bound of its generics names \
             `Self::A`"
        ]
    );
}

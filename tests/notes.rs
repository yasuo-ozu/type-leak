//! The exact text of the notes of the errors of the Leaker that no other test pins.
//!
//! An error has a headline and notes (`ErrorItem::message`, `ErrorItem::notes`); the notes say why
//! the item cannot be leaked, and how to write it so that it can. Their words are the only advice
//! that a user of a macro gets, and a test that only looks for a part of one (`contains("E0106")`)
//! does not see a wrong name, a missing advice, or a sentence that is cut. The tests of
//! `tests/errors.rs` pin the note that suggests an absolute path, and those of
//! `tests/bound_cycles.rs`, `tests/binder_output.rs` and `tests/elided_output.rs` the notes of
//! their rules. Here are the notes of the rules for `Self`:
//!
//! - an associated item of `Self` in an entry, where `Self` is a type (E0220);
//! - an entry in a predicate, where a bound of the impl names a leaked type, and the copy resolves
//!   `Self::A` by the impl of the trait, or writes `Self` as the self type of the impl;
//! - an entry in the self type of an impl, whose bound names `Self`;
//!
//! and the note of the lifetime that a path hides, where the holder of the signature cannot be
//! leaked.

use syn::*;
use template_quote::quote;
use type_leak::*;

/// The config of the property test, with `::p` allowed.
fn leaker() -> Leaker {
    let mut config = LeakerConfig::new();
    config.allow_primitive().allow_std_prelude();
    config.allowed_paths.push(parse_quote!(::p));
    Leaker::from_config(config)
}

/// What an error says: for each of its errors, the headline, the tokens, and the notes.
fn shown(err: &NotInternableError) -> Vec<(String, String, Vec<String>)> {
    err.items()
        .map(|item| {
            (
                item.message.clone(),
                item.tokens.to_string(),
                item.notes.iter().map(|(_, note)| note.clone()).collect(),
            )
        })
        .collect()
}

fn error(item: &ItemImpl) -> NotInternableError {
    leaker()
        .intern_impl(item, true)
        .err()
        .unwrap_or_else(|| panic!("accepted: {}", quote!(#item)))
}

/// An entry that names `Self::A`, where `Self` is a type, cannot be written by the impl of the
/// Repeater for that type: it has no trait to take `A` from (E0220).
#[test]
fn an_associated_item_of_self_has_a_note_that_names_the_trait() {
    let item: ItemImpl = parse_quote!(
        impl ::p::Tr for u8 {
            type A = u8;
            fn f(&self, x: Rel<Self::A>) {}
        }
    );
    assert_eq!(
        shown(&error(&item)),
        [(
            "`Self :: A` cannot be written in the impl of the Repeater".to_string(),
            "Rel < Self :: A >".to_string(),
            vec![
                "`Self :: A` is an associated item of `Self`, which is the type `u8` here, and \
                 the impl of the Repeater for that type has no trait to take it from: name the \
                 trait with its absolute path (`<Self as ::path::Trait>::..`)"
                    .to_string(),
                "it cannot be leaked as a whole because of `Self :: A`".to_string(),
            ]
        )]
    );
    // The same for a struct: `Self` is `S<T>` there.
    let item: ItemStruct = parse_quote!(
        struct S<T>(T, Rel<Self::A>);
    );
    let err = leaker().intern_struct(&item, true).unwrap_err();
    assert_eq!(
        shown(&err),
        [(
            "`Self :: A` cannot be written in the impl of the Repeater".to_string(),
            "Rel < Self :: A >".to_string(),
            vec![
                "`Self :: A` is an associated item of `Self`, which is the type `S < T >` here, \
                 and the impl of the Repeater for that type has no trait to take it from: name \
                 the trait with its absolute path (`<Self as ::path::Trait>::..`)"
                    .to_string(),
                "it cannot be leaked as a whole because of `Self :: A`".to_string(),
            ]
        )]
    );
}

/// `Self::A` in a predicate of a method, in an impl of a trait whose bound names a leaked type:
/// the copy resolves it by the impl of the trait, which needs that bound.
#[test]
fn self_assoc_in_a_predicate_has_a_note_on_the_impl_of_the_trait() {
    let item: ItemImpl = parse_quote!(
        impl<T> ::p::Tr<T> for u8
        where
            T: Fn(Rel),
        {
            fn m<X>(&self)
            where
                X: Into<Self::A>,
            {
            }
        }
    );
    assert_eq!(
        shown(&error(&item)),
        [(
            "`Self :: A` is in a bound or where predicate, and a bound of its generics names a \
             leaked type"
                .to_string(),
            "Self :: A".to_string(),
            vec![
                "the copy for a concrete implementor resolves `Self :: A` by the impl of the \
                 trait, which needs the bound `T : Fn (Rel)` of the impl, where the copy writes \
                 the leaked type as a path through the Repeater, and rustc cannot normalize the \
                 one in a bound or where predicate while it checks the bounds (E0277, E0275). \
                 Write the leaked type of the bound with an absolute path, or use `Self :: A` \
                 outside the bounds and where predicates"
                    .to_string()
            ]
        )]
    );
}

/// `Self` in a predicate, in an impl whose self type has an entry, and a bound of a generic of
/// that entry names a leaked type: the copy writes `Self` as the self type, with the entry as a
/// path through the Repeater.
#[test]
fn self_in_a_predicate_has_a_note_on_the_self_type_of_the_impl() {
    let item: ItemImpl = parse_quote!(
        impl<T: Fn(Rel2), V: AsRef<Self>> ::p::W<Rel<T>, T, V> {}
    );
    assert_eq!(
        shown(&error(&item)),
        [(
            "`Self` is in a bound or where predicate, and a bound of its generics names a leaked \
             type"
                .to_string(),
            "Self".to_string(),
            vec![
                "`Self` is the self type of the impl, which the copy writes with its leaked \
                 types as paths through the Repeater, and the bound `T : Fn (Rel2)` of their \
                 generics names a type that is leaked too, and rustc cannot normalize the one \
                 in a bound or where predicate while it checks the bounds (E0277, E0275). Write \
                 the leaked type of the bound with an absolute path, or use `Self` outside the \
                 bounds and where predicates"
                    .to_string()
            ]
        )]
    );
}

/// An entry in the self type of an impl, whose bound names `Self`: the Referrer has no type to
/// write for `Self`, which is the one being replaced. The note starts with what `Self` is: the
/// entry itself, or the self type that contains it. The `Self` in the bound is an error of its own,
/// with the advice to write the bound without it.
#[test]
fn the_self_type_of_an_impl_is_named_in_the_note() {
    let rest = "The copy writes `S < T >` as a path through the Repeater, whose impl is for \
                `Self`, so the Referrer has no type to write for `Self`: it is the one being \
                replaced. Write the self type with an absolute path, or the bounds of its \
                generics without `Self`";
    let item: ItemImpl = parse_quote!(
        impl<T> S<T> where T: AsRef<Option<Self>> {}
    );
    let err = error(&item);
    assert_eq!(
        err.to_string(),
        "`S < T >` is in the self type of the impl, and a bound of its generics names `Self`\n\
         `Self` is in a bound or where predicate, and a bound of its generics names `Self`, \
         which is a leaked type"
    );
    assert_eq!(
        shown(&err),
        [
            (
                "`S < T >` is in the self type of the impl, and a bound of its generics names \
                 `Self`"
                    .to_string(),
                "S < T >".to_string(),
                vec![
                    format!("`Self` is `S < T >`, the self type of the impl. {}", rest),
                    "it cannot be leaked as a whole because of `Self`".to_string(),
                ]
            ),
            (
                "`Self` is in a bound or where predicate, and a bound of its generics names \
                 `Self`, which is a leaked type"
                    .to_string(),
                "Self".to_string(),
                vec![
                    "`Self` is the self type of the impl, which the copy writes with its leaked \
                      types as paths through the Repeater, and the bound `T : AsRef < Option < \
                      Self > >` of their generics names `Self`, which the copy writes as the self \
                      type of the impl, a path through the Repeater that rustc has not \
                      normalized yet in the environment, where it must prove the bound (E0277, \
                      E0271). Write the bound without `Self`, or use `Self` outside the bounds \
                      and where predicates"
                        .to_string()
                ]
            ),
        ]
    );
    // The entry is a part of the self type: `Self` is the whole self type, which contains it.
    let item: ItemImpl = parse_quote!(
        impl<T: AsRef<Self>, U> ::p::W<Rel<T>, U> {}
    );
    let err = error(&item);
    let shown = shown(&err);
    assert_eq!(shown.len(), 2, "{}", err);
    assert_eq!(
        shown[0],
        (
            "`Rel < T >` is in the self type of the impl, and a bound of its generics names \
             `Self`"
                .to_string(),
            "Rel < T >".to_string(),
            vec![
                "`Self` is the self type `:: p :: W < Rel < T > , U >` of the impl, which \
                 contains `Rel < T >`. The copy writes `Rel < T >` as a path through the \
                 Repeater, whose impl is for `Self`, so the Referrer has no type to write for \
                 `Self`: it is the one being replaced. Write the self type with an absolute \
                 path, or the bounds of its generics without `Self`"
                    .to_string(),
                "it cannot be leaked as a whole because of `Self`".to_string(),
            ]
        )
    );
    assert_eq!(
        shown[1].2,
        [
            "`Self` is the self type of the impl, which the copy writes with its leaked types as \
          paths through the Repeater, and the bound `T : AsRef < Self >` of their generics names \
          `Self`, which the copy writes as the self type of the impl, a path through the \
          Repeater that rustc has not normalized yet in the environment, where it must prove the \
          bound (E0277, E0271). Write the bound without `Self`, or use `Self` outside the \
          bounds and where predicates"
        ]
    );
}

/// The lifetime that a path hides is the only one of the inputs, elision takes it for the output,
/// and a leaked type adds another. Where the holder of the signature cannot be leaked either (an
/// `impl Fn(..)` in an argument), the note says that the type holding the signature would be
/// leaked as a whole instead, and the error is at the entry.
#[test]
fn the_note_of_a_hidden_lifetime_says_where_the_holder_is_not_leaked() {
    let item: ItemTrait = parse_quote!(
        trait Tr<'a, T: 'a> {
            fn f(&self, x: impl Fn(::p::WRef<u8>, Rel<T>) -> &u8);
        }
    );
    let err = leaker().intern_trait(&item, true).unwrap_err();
    assert_eq!(
        shown(&err),
        [(
            "use absolute path: 'Rel < T >'".to_string(),
            "Rel < T >".to_string(),
            vec![
                "the output `& u8` has an elided lifetime, which elision takes from the only \
                 lifetime of the inputs, which a path hides (the Leaker does not see it), but a \
                 leaked type in the inputs has `'a` among its generics, and the copy names it \
                 in the arguments of its projection, so the inputs have more than one lifetime \
                 (E0106), so a type using them cannot be leaked alone, and the type holding the \
                 signature would be leaked as a whole instead"
                    .to_string(),
                "it cannot be leaked as a whole because of `impl Fn (:: p :: WRef < u8 > , Rel < \
                 T >) -> & u8`"
                    .to_string(),
            ]
        )]
    );
}

//! The exact text of the notes and helps of the errors of the Leaker that no other test pins.
//!
//! An error has a headline, notes and a help (`ErrorItem::message`, `ErrorItem::notes`,
//! `ErrorItem::help`); the notes say why the item cannot be leaked, and the help says what to write
//! so that it can. Their words are the only advice that a user of a macro gets, and a test that
//! only looks for a part of one (`contains("E0106")`) does not see a wrong name, a missing advice,
//! or a sentence that is cut. The table of the help in `tests/error_items.rs` pins the help that
//! suggests an absolute path, and those of `tests/bound_cycles.rs` and `tests/binder_output.rs`
//! (also its module `elided`) the notes and helps of their rules. Here are those of the rules for
//! `Self`:
//!
//! - an associated item of `Self` in an entry, where `Self` is a type (E0220), in an impl of a
//!   trait, in a struct, an enum or a union, and in an inherent impl, with the trait written
//!   relative or absolute, in the type or in a bound;
//! - an entry in a predicate, where a bound of the impl names a leaked type, and the copy resolves
//!   `Self::A` by the impl of the trait, or writes `Self` as the self type of the impl;
//! - an entry in the self type of an impl, whose bound names `Self`;
//!
//! and the note of the lifetime that a path hides, where the holder of the signature cannot be
//! leaked. The last test (`tidy_writes_tokens_as_one_does`) pins how the texts write the tokens
//! they quote.

mod common;

use common::{messages, starts_with};
use syn::*;
use template_quote::{quote, ToTokens};
use type_leak::*;

/// The config of the property test, with `::p` allowed.
fn leaker() -> Leaker<impl Allow> {
    let config = Rules::new()
        .primitive()
        .prelude()
        .or_if(starts_with(parse_quote!(::p)));
    Leaker::with_rule(config)
}

/// What an error says.
#[derive(Debug, PartialEq, Eq)]
struct Shown {
    message: String,
    tokens: String,
    notes: Vec<String>,
    help: String,
}

impl Shown {
    fn new(message: &str, tokens: &str, notes: &[&str], help: &str) -> Self {
        Self {
            message: message.to_string(),
            tokens: tokens.to_string(),
            notes: notes.iter().map(|note| note.to_string()).collect(),
            help: help.to_string(),
        }
    }
}

/// What an error says: for each of its errors, the headline, the tokens, the notes and the help.
fn shown(err: &[ErrorItem]) -> Vec<Shown> {
    err.iter()
        .map(|item| Shown {
            message: item.message.clone(),
            tokens: item.cause.to_token_stream().to_string(),
            notes: item.notes.iter().map(|(_, note)| note.clone()).collect(),
            help: item
                .help_text()
                .expect("every error has a help")
                .to_string(),
        })
        .collect()
}

/// The errors that `leaker` has after `analyze` ran on it (`Leaker::result()`).
fn errors_of<A: Allow, R>(
    mut leaker: Leaker<A>,
    analyze: impl FnOnce(&mut Leaker<A>) -> R,
) -> Vec<ErrorItem> {
    analyze(&mut leaker);
    leaker.result().errors().to_vec()
}

fn error(item: &ItemImpl) -> Vec<ErrorItem> {
    let errors = errors_of(leaker(), |leaker| leaker.intern_impl(item));
    assert!(!errors.is_empty(), "accepted: {}", quote!(#item));
    errors
}

/// The note that says why `Self::A` is written out. `through`: what rustc resolves it through.
fn why(cause: &str, through: &str) -> String {
    format!(
        "`{}` is shorthand that rustc resolves through {}. In the copy it is written in an impl \
         of the Repeater, which has no such trait (E0220, or E0223 ambiguous associated type), \
         so the trait must be written out. `Self` itself is fine",
        cause, through
    )
}

/// An entry that names `Self::A`, where `Self` is a type, cannot be written by the impl of the
/// Repeater for that type: it has no trait to take `A` from (E0220). In an impl of a trait the
/// error names the trait, as written.
#[test]
fn an_associated_item_of_self_in_an_impl_of_a_trait_names_the_trait() {
    let item: ItemImpl = parse_quote!(
        impl ::p::Tr for u8 {
            type A = u8;
            fn f(&self, x: Rel<Self::A>) {}
        }
    );
    assert_eq!(
        shown(&error(&item)),
        [Shown::new(
            "write `<Self as ::p::Tr>::A` instead of `Self::A` in `Rel<Self::A>`: the copy \
             cannot tell which trait `A` comes from",
            "Rel < Self :: A >",
            &[
                &why("Self::A", "the trait of this impl (`::p::Tr`)"),
                "it cannot be leaked as a whole because of `Self::A`",
            ],
            "replace `Self::A` with `<Self as ::p::Tr>::A`",
        )]
    );
    // The arguments of the trait are part of it, and so is what follows `A`.
    let item: ItemImpl = parse_quote!(
        impl<T> ::p::Tr<T, u8> for ::p::S<T> {
            type A = u8;
            fn f(&self, x: Rel<Self::A::B>, y: Rel2<Self::C<'static>>) {}
        }
    );
    let items = shown(&error(&item));
    assert_eq!(items.len(), 2, "{:?}", items);
    assert_eq!(
        items[0].message,
        "write `<Self as ::p::Tr<T, u8>>::A::B` instead of `Self::A::B` in `Rel<Self::A::B>`: \
         the copy cannot tell which trait `A` comes from"
    );
    assert_eq!(
        items[0].help,
        "replace `Self::A::B` with `<Self as ::p::Tr<T, u8>>::A::B`"
    );
    assert_eq!(
        items[1].message,
        "write `<Self as ::p::Tr<T, u8>>::C<'static>` instead of `Self::C<'static>` in \
         `Rel2<Self::C<'static>>`: the copy cannot tell which trait `C` comes from"
    );
    assert_eq!(
        items[1].help,
        "replace `Self::C<'static>` with `<Self as ::p::Tr<T, u8>>::C<'static>`"
    );
}

/// A trait that the copy cannot name, because its path is relative, is written as an absolute
/// path in the help. The path of the trait is an error of its own.
#[test]
fn a_relative_trait_of_the_impl_is_written_as_an_absolute_path() {
    let item: ItemImpl = parse_quote!(
        impl Tr for S {
            type A = u8;
            fn f(&self, x: Wrap<Self::A>) {}
        }
    );
    let items = shown(&error(&item));
    assert_eq!(items.len(), 2, "{:?}", items);
    assert_eq!(items[0].message, "use absolute path: 'Tr'");
    assert_eq!(
        items[1],
        Shown::new(
            "write `<Self as Tr>::A` instead of `Self::A` in `Wrap<Self::A>`: the copy cannot \
             tell which trait `A` comes from",
            "Wrap < Self :: A >",
            &[
                &why("Self::A", "the trait of this impl (`Tr`)"),
                "it cannot be leaked as a whole because of `Self::A`",
            ],
            "replace `Self::A` with `<Self as ::my_crate::Tr>::A`, with `Tr` written as an \
             absolute path",
        )
    );
    // A longer path, with arguments, and one from the crate: the example is the path in the crate,
    // and a path of a standard crate is the crate itself.
    for (written, example) in [
        ("m::Tr<u8>", "::my_crate::m::Tr<u8>"),
        ("crate::m::Tr", "::my_crate::m::Tr"),
        ("super::Tr", "::my_crate::Tr"),
        ("core::ops::Add<u8>", "::core::ops::Add<u8>"),
        ("std::fmt::Display", "::std::fmt::Display"),
    ] {
        let tr: Path = parse_str(written).unwrap();
        let item: ItemImpl = parse_quote!(
            impl #tr for S {
                type A = u8;
                fn f(&self, x: Wrap<Self::A>) {}
            }
        );
        let items = shown(&error(&item));
        let written = written.replace(' ', "");
        assert_eq!(
            items.last().unwrap().help,
            format!(
                "replace `Self::A` with `<Self as {}>::A`, with `{}` written as an absolute path",
                example, written
            ),
            "{}",
            written
        );
    }
    // An absolute path of an allowed crate is named as it is.
    let item: ItemImpl = parse_quote!(
        impl ::p::Tr for ::p::S {
            type A = u8;
            fn f(&self, x: Wrap<Self::A>) {}
        }
    );
    let items = shown(&error(&item));
    assert_eq!(items.len(), 1, "{:?}", items);
    assert_eq!(
        items[0].help,
        "replace `Self::A` with `<Self as ::p::Tr>::A`"
    );
}

/// A struct, an enum or a union has no trait, and neither has an inherent impl: the help tells to
/// name the trait that defines `A`.
#[test]
fn an_associated_item_of_self_in_a_type_has_no_trait_to_name() {
    let message =
        "write `<Self as SomeTrait>::A` instead of `Self::A` in `Rel<Self::A>`: the copy \
                   cannot tell which trait `A` comes from";
    let notes = [
        why("Self::A", "a trait that this type implements"),
        "it cannot be leaked as a whole because of `Self::A`".to_string(),
    ];
    let notes: Vec<&str> = notes.iter().map(|note| note.as_str()).collect();
    let help =
        "replace `Self::A` with `<Self as SomeTrait>::A`, naming the trait that defines `A` \
                with its absolute path";
    let expected = [Shown::new(message, "Rel < Self :: A >", &notes, help)];
    // A struct: `Self` is `S<T>` there.
    let item: ItemStruct = parse_quote!(
        struct S<T>(T, Rel<Self::A>);
    );
    assert_eq!(
        shown(&errors_of(leaker(), |leaker| leaker.intern_struct(&item))),
        expected
    );
    let item: ItemEnum = parse_quote!(
        enum E<T> {
            V(T, Rel<Self::A>),
        }
    );
    assert_eq!(
        shown(&errors_of(leaker(), |leaker| leaker.intern_enum(&item))),
        expected
    );
    let item: ItemUnion = parse_quote!(
        union U<T: Copy> {
            a: T,
            b: Rel<Self::A>,
        }
    );
    assert_eq!(
        shown(&errors_of(leaker(), |leaker| leaker.intern_union(&item))),
        expected
    );
    // An inherent impl.
    let item: ItemImpl = parse_quote!(
        impl S {
            fn f(&self, x: Rel<Self::A>) {}
        }
    );
    assert_eq!(shown(&error(&item)), expected);
    // An impl whose parts are visited by a scope: it does not know the trait.
    let outer = errors_of(leaker(), |leaker| {
        leaker.enter_impl(&Generics::default(), &parse_quote!(S), None, |view| {
            view.enter_generics(&Generics::default(), |view| {
                view.visit_type(&parse_quote!(Rel<Self::A>))
            })
        })
    });
    // The error of the inner scope is in the list of the call, once.
    assert_eq!(shown(&outer), expected);
}

/// A bound of the generics of the entry that names `Self::A`: the headline says where it is.
#[test]
fn an_associated_item_of_self_in_a_bound_names_the_generics() {
    let item: ItemImpl = parse_quote!(
        impl<T: ::p::Tr<Self::A>> ::p::Tr for ::p::W<T> {
            type A = u8;
            fn f(&self, x: Rel<T>) {}
        }
    );
    let err = error(&item);
    let items = shown(&err);
    // The bound is an error of its own too: rustc cannot prove it while it checks the bounds.
    assert_eq!(items.len(), 2, "{:?}", items);
    assert_eq!(
        items[0],
        Shown::new(
            "write `<Self as ::p::Tr>::A` instead of `Self::A` in a bound of the generics of \
             `Rel<T>`: the copy cannot tell which trait `A` comes from",
            "Rel < T >",
            &[
                &why("Self::A", "the trait of this impl (`::p::Tr`)"),
                "it cannot be leaked as a whole because of `Self::A`",
            ],
            "replace `Self::A` with `<Self as ::p::Tr>::A`",
        )
    );
    // Not in an impl of a trait.
    let item: ItemStruct = parse_quote!(
        struct S<T: ::p::Tr<Self::A>>(Rel<T>);
    );
    let mut leaker = leaker();
    leaker.intern_struct(&item);
    let result = leaker.result();
    let err = result.errors();
    let items = shown(err);
    assert!(!items.is_empty());
    assert_eq!(
        items[0].message,
        "write `<Self as SomeTrait>::A` instead of `Self::A` in a bound of the generics of \
         `Rel<T>`: the copy cannot tell which trait `A` comes from"
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
        [Shown::new(
            "`Self::A` is in a bound or where predicate, and a bound of its generics names a \
             leaked type",
            "Self :: A",
            &[
                "the copy for a concrete implementor resolves `Self::A` by the impl of the \
               trait, which needs the bound `T : Fn (Rel)` of the impl, where the copy writes \
               the leaked type as a path through the Repeater, and rustc cannot normalize the \
               one in a bound or where predicate while it checks the bounds (E0277, E0275)"
            ],
            "write the leaked type of the bound with an absolute path, or use `Self::A` outside \
             the bounds and where predicates",
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
        [Shown::new(
            "`Self` is in a bound or where predicate, and a bound of its generics names a leaked \
             type",
            "Self",
            &[
                "`Self` is the self type of the impl, which the copy writes with its leaked \
               types as paths through the Repeater, and the bound `T : Fn (Rel2)` of their \
               generics names a type that is leaked too, and rustc cannot normalize the one \
               in a bound or where predicate while it checks the bounds (E0277, E0275)"
            ],
            "write the leaked type of the bound with an absolute path, or use `Self` outside the \
             bounds and where predicates",
        )]
    );
}

/// An entry in the self type of an impl, whose bound names `Self`: no type can be written for
/// `Self`, which is the one being replaced. The note starts with what `Self` is: the
/// entry itself, or the self type that contains it. The `Self` in the bound is an error of its own,
/// with the advice to write the bound without it.
#[test]
fn the_self_type_of_an_impl_is_named_in_the_note() {
    let rest = "The copy writes `S<T>` as a path through the Repeater, whose impl is for \
                `Self`, so no type can be written for `Self`: it is the one being replaced";
    let item: ItemImpl = parse_quote!(
        impl<T> S<T> where T: AsRef<Option<Self>> {}
    );
    let err = error(&item);
    assert_eq!(
        messages(&err),
        "`S<T>` is in the self type of the impl, and a bound of its generics names `Self`\n\
         `Self` is in a bound or where predicate, and a bound of its generics names `Self`, \
         which is a leaked type"
    );
    assert_eq!(
        shown(&err),
        [
            Shown::new(
                "`S<T>` is in the self type of the impl, and a bound of its generics names \
                 `Self`",
                "S < T >",
                &[
                    &format!("`Self` is `S<T>`, the self type of the impl. {}", rest),
                    "it cannot be leaked as a whole because of `Self`",
                ],
                "write `S<T>` with an absolute path, or the bound of its generics without `Self`",
            ),
            Shown::new(
                "`Self` is in a bound or where predicate, and a bound of its generics names \
                 `Self`, which is a leaked type",
                "Self",
                &[
                    "`Self` is the self type of the impl, which the copy writes with its leaked \
                   types as paths through the Repeater, and the bound `T : AsRef<Option<\
                   Self>>` of their generics names `Self`, which the copy writes as the self \
                   type of the impl, a path through the Repeater that rustc has not \
                   normalized yet in the environment, where it must prove the bound (E0277, \
                   E0271)"
                ],
                "write the bound without `Self`, or use `Self` outside the bounds and where \
                 predicates",
            ),
        ]
    );
    // The entry is a part of the self type: `Self` is the whole self type, which contains it.
    let item: ItemImpl = parse_quote!(
        impl<T: AsRef<Self>, U> ::p::W<Rel<T>, U> {}
    );
    let err = error(&item);
    let shown = shown(&err);
    assert_eq!(shown.len(), 2, "{}", messages(&err));
    assert_eq!(
        shown[0],
        Shown::new(
            "`Rel<T>` is in the self type of the impl, and a bound of its generics names \
             `Self`",
            "Rel < T >",
            &[
                "`Self` is the self type `::p::W<Rel<T>, U>` of the impl, which \
                 contains `Rel<T>`. The copy writes `Rel<T>` as a path through the \
                 Repeater, whose impl is for `Self`, so no type can be written for `Self`: it \
                 is the one being replaced",
                "it cannot be leaked as a whole because of `Self`",
            ],
            "write `Rel<T>` with an absolute path, or the bound of its generics without `Self`",
        )
    );
    assert_eq!(
        shown[1].notes,
        [
            "`Self` is the self type of the impl, which the copy writes with its leaked types as \
          paths through the Repeater, and the bound `T : AsRef<Self>` of their generics names \
          `Self`, which the copy writes as the self type of the impl, a path through the \
          Repeater that rustc has not normalized yet in the environment, where it must prove the \
          bound (E0277, E0271)"
        ]
    );
    assert_eq!(
        shown[1].help,
        "write the bound without `Self`, or use `Self` outside the bounds and where predicates"
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
    let mut leaker = leaker();
    leaker.intern_trait(&item);
    let result = leaker.result();
    let err = result.errors();
    assert_eq!(
        shown(err),
        [Shown::new(
            "use absolute path: 'Rel<T>'",
            "Rel < T >",
            &[
                "the output `&u8` has an elided lifetime, which elision takes from the only \
                 lifetime of the inputs, which a path hides (the syntax does not show it), but a \
                 leaked type in the inputs has `'a` among its generics, and the copy names it \
                 in the arguments of its projection, so the inputs have more than one lifetime \
                 (E0106), so a type using them cannot be leaked alone, and the type holding the \
                 signature would be leaked as a whole instead",
                "it cannot be leaked as a whole because of `impl Fn (::p::WRef<u8>, Rel<\
                 T>) -> &u8`",
            ],
            "write the lifetime that the path hides (`Ref<'x, u8>`), and name it in the output, \
             or write `Rel<T>` with an absolute path",
        )]
    );
}

/// How a row of `TIDY` is analyzed.
#[derive(Clone, Copy)]
enum Via {
    /// A struct, trait or impl, by the `intern_*` of its kind.
    Item,
    /// The same, with the values read (`Leaker::check_values`).
    Values,
    /// A where predicate on `T` of a trait: an impl nested in it has the entry `Rel<T>`, so the
    /// predicate is copied, and its `Self` has no name there.
    Predicate,
}

/// The rows: the rule of the text, how to analyze, the input, and the renderings that the texts of
/// the errors (headline, notes and help) must contain.
const TIDY: &[(&str, Via, &str, &[&str])] = &[
    (
        "no space inside `<..>` or before `,`, `'_` before a type",
        Via::Item,
        "struct S { a: Rel<'_, u8> }",
        &["'Rel<'_, u8>'"],
    ),
    (
        "an absolute path as an argument",
        Via::Item,
        "struct S { a: Other<'_, ::core::u8> }",
        &["'Other<'_, ::core::u8>'"],
    ),
    (
        "`::` after the `>` of a type",
        Via::Item,
        "struct S { a: Rel<&'_ u8>::Item }",
        &["'Rel<&'_ u8>::Item'"],
    ),
    (
        "`::<` of a path in a value",
        Via::Values,
        "impl S { const X: usize = Rel::<&'_ u8>::new(); }",
        &["'Rel::<&'_ u8>::new'"],
    ),
    (
        "`<` after a lifetime keeps its space",
        Via::Item,
        "struct S<'a> { a: I1<'_, &'a <Self as Tr>::A> }",
        &["'I1<'_, &'a <Self as Tr>::A>'"],
    ),
    (
        "an absolute path with an argument",
        Via::Item,
        "struct S { a: Other<'_, ::p::Wrap<Rel>> }",
        &["'Other<'_, ::p::Wrap<Rel>>'"],
    ),
    (
        "no space before the `;` of a statement",
        Via::Values,
        "trait Tr { const N: usize = { use inner::*; 1 }; }",
        &["cannot analyze `use inner::*;` in the default"],
    ),
    (
        "`Self::A` in a type",
        Via::Item,
        "struct S<T>(T, Rel<Self::A>);",
        &[
            "instead of `Self::A` in `Rel<Self::A>`:",
            "because of `Self::A`",
            "replace `Self::A` with `<Self as SomeTrait>::A`,",
        ],
    ),
    (
        "`Self::A0` and a Fn sugar in a bound",
        Via::Item,
        "trait Tr { type A0; fn m<X>(&self) where X: Fn(&Self::A0) -> u8, Rel<X>: Default; }",
        &[
            "a bound of its generics names `Self::A0`",
            "the bound `X : Fn (&Self::A0) -> u8` of its generics",
            "write the bound without `Self::A0`,",
        ],
    ),
    (
        "`<Self as ::p::Tr>::A0` in a bound",
        Via::Item,
        "trait Tr { type A0; fn m<X>(&self) \
         where X: ::core::ops::Fn(&<Self as ::p::Tr>::A0) -> u8, Rel<X>: Default; }",
        &[
            "a bound of its generics names `<Self as ::p::Tr>::A0`",
            "the bound `X : ::core::ops::Fn (&<Self as ::p::Tr>::A0) -> u8` of its generics",
            "write the bound without `<Self as ::p::Tr>::A0`,",
        ],
    ),
    (
        "a trait object behind a reference, `dyn ::` and `& dyn`",
        Via::Item,
        "trait Tr { fn f(&self) -> Rel<&dyn ::core::fmt::Debug>; }",
        &[
            "'Rel<&dyn ::core::fmt::Debug>'",
            "because of `&dyn ::core::fmt::Debug`",
            "because of `dyn ::core::fmt::Debug`",
        ],
    ),
    (
        "`->` of a function pointer, before an absolute path",
        Via::Item,
        "struct S { a: Other<'_, fn() -> ::p::X> }",
        &["'Other<'_, fn () -> ::p::X>'"],
    ),
    (
        "a binder before an absolute path, and a space before a Fn sugar",
        Via::Item,
        "trait Tr { fn f(&self, x: Rel<impl for<'a> ::core::ops::Fn(&'a u8)>); }",
        &[
            "'Rel<impl for<'a> ::core::ops::Fn (&'a u8)>'",
            "because of `impl for<'a> ::core::ops::Fn (&'a u8)`",
        ],
    ),
    (
        "a binder inside the arguments before `::A`",
        Via::Item,
        "struct S { a: Other<'_, Tr<dyn for<'a> Fn(&'a u8)>::A> }",
        &["'Other<'_, Tr<dyn for<'a> Fn (&'a u8)>::A>'"],
    ),
    (
        "`!` of an absolute macro, ` : ` in the trait of the note",
        Via::Item,
        "trait Tr: ::p::Sup<::p::m!(Self)> {}",
        &["`::p::m!(Self)` names `Self`", "(`Self : Tr`)"],
    ),
    (
        "`!` of a relative macro in a type",
        Via::Item,
        "trait Tr where Wrap<m!(Self)>: Sized {}",
        &["`Wrap<m!(Self)>` names `Self`", "`m!(Self)` names `Self`"],
    ),
    (
        "` : ` of a bound that names a leaked type",
        Via::Item,
        "struct S<T> where T: ::p::Tr<Rel<T>> { a: T }",
        &["the bound `T : ::p::Tr<Rel<T>>` of its generics"],
    ),
    (
        "the trait of the note, with arguments",
        Via::Item,
        "trait Tr<T> where T: ::p::Tr<Self>, Rel<T>: Clone {}",
        &["(`Self : Tr<T>`)"],
    ),
    (
        "another path of the same type",
        Via::Item,
        "trait Tr { fn f(&self, x: Box<dyn RelA + RelB<'_, ::core::u8>>); }",
        &[
            "'RelA'",
            "`RelB<'_, ::core::u8>` also depends on the type context",
        ],
    ),
    (
        "a where predicate",
        Via::Predicate,
        "T: ::p::Tr<Self>",
        &[
            "the where predicate `T : ::p::Tr<Self>` mentions a `Self` which has no name here",
            "the `Self` of the trait `Tr<'a, T>`,",
            "write the where predicate `T : ::p::Tr<Self>` without `Self`,",
        ],
    ),
    (
        "an absolute path with an argument in a predicate",
        Via::Predicate,
        "T: ::p::Tr<::p::Wrap<Rel>, Self>",
        &["`T : ::p::Tr<::p::Wrap<Rel>, Self>`"],
    ),
    (
        "`Self::A` in a predicate",
        Via::Predicate,
        "T: ::p::Tr<Self::A>",
        &["`T : ::p::Tr<Self::A>`", "because of `Self::A`"],
    ),
    (
        "`<Self as ::p::Tr>::A` in a predicate",
        Via::Predicate,
        "T: ::p::Tr<<Self as ::p::Tr>::A>",
        &["`T : ::p::Tr<<Self as ::p::Tr>::A>`"],
    ),
    (
        "a trait object in a predicate",
        Via::Predicate,
        "T: ::p::Tr<dyn ::core::fmt::Debug, Self>",
        &["`T : ::p::Tr<dyn ::core::fmt::Debug, Self>`"],
    ),
    (
        "an elided lifetime and an absolute path in a predicate",
        Via::Predicate,
        "T: ::p::Tr<Other<'_, ::core::u8>, Self>",
        &["`T : ::p::Tr<Other<'_, ::core::u8>, Self>`"],
    ),
    (
        "`->` in a predicate",
        Via::Predicate,
        "T: ::p::Tr<fn() -> ::p::X, Self>",
        &["`T : ::p::Tr<fn () -> ::p::X, Self>`"],
    ),
    (
        "a binder and a Fn sugar in a predicate",
        Via::Predicate,
        "T: for<'a> ::core::ops::Fn(&'a u8) + ::p::Tr<Self>",
        &["`T : for<'a> ::core::ops::Fn (&'a u8) + ::p::Tr<Self>`"],
    ),
    (
        "a binder inside the arguments before `::A` in a predicate",
        Via::Predicate,
        "T: ::p::Tr<Tr<dyn for<'a> Fn(&'a u8)>::A, Self>",
        &["`T : ::p::Tr<Tr<dyn for<'a> Fn (&'a u8)>::A, Self>`"],
    ),
    (
        "a macro in a predicate",
        Via::Predicate,
        "T: ::p::Tr<m!(Self)>",
        &["`T : ::p::Tr<m!(Self)>`"],
    ),
    (
        "`<` after a lifetime in a predicate",
        Via::Predicate,
        "T: ::p::Tr<I1<&'a <Self as Tr>::A>>",
        &["`T : ::p::Tr<I1<&'a <Self as Tr>::A>>`"],
    ),
    (
        "`::` after the `>` of a type in a predicate",
        Via::Predicate,
        "T: ::p::Tr<Vec<&'_ u8>::Item, Self>",
        &["`T : ::p::Tr<Vec<&'_ u8>::Item, Self>`"],
    ),
];

/// The errors of `source` (an item), as `via` analyzes it.
fn tidy_errors(via: Via, source: &str) -> Vec<ErrorItem> {
    match via {
        Via::Item | Via::Values => {
            let item: Item = parse_str(source).unwrap();
            errors_of(leaker(), |leaker| {
                leaker.check_values = matches!(via, Via::Values);
                match &item {
                    Item::Struct(item) => leaker.intern_struct(item),
                    Item::Trait(item) => leaker.intern_trait(item),
                    Item::Impl(item) => leaker.intern_impl(item),
                    _ => panic!("not a struct, trait or impl: {}", source),
                };
            })
        }
        Via::Predicate => {
            let mut generics: Generics = parse_quote!(<'a, T>);
            generics.where_clause = Some(parse_str(&format!("where {}", source)).unwrap());
            let tr: Ident = parse_quote!(Tr);
            errors_of(leaker(), |leaker| {
                leaker.enter_trait(&tr, &generics, |view| {
                    view.visit_item(&parse_quote!(
                        impl ::p::W for X {
                            fn f(&self, x: Rel<T>) {}
                        }
                    ))
                });
            })
        }
    }
}

/// The messages tell tokens as one writes them (`Rel<'_, u8>`, `fn () -> ::p::X`, `m!(Self)`), not
/// with the spaces of a token stream (`Rel < '_ , u8 >`). The rules are those of the private
/// `tidy_str()`, and each row has a different input for one of them: no space around `<`, `>`,
/// `::` and `,`; a space after `'_` and `'a`; the binder `for<'a>`; `->`; a space before the `(` of
/// a Fn sugar; the `!` of a macro; ` : ` of a bound; no space before a `;`; `dyn ::p::Tr` keeps its
/// space. The tokens of the compiler print a macro call as `m! (..)`, which a test cannot build
/// (the tokens of a string print `m ! (..)`): `tests/diagnostic.rs` checks it.
#[test]
fn tidy_writes_tokens_as_one_does() {
    for (case, via, source, renderings) in TIDY {
        let errors = tidy_errors(*via, source);
        assert!(!errors.is_empty(), "{}: accepted: {}", case, source);
        let texts: Vec<&str> = errors
            .iter()
            .flat_map(|item| {
                let notes = item.notes.iter().map(|(_, note)| note.as_str());
                std::iter::once(item.message.as_str())
                    .chain(notes)
                    .chain(item.help_text())
            })
            .collect();
        for rendering in *renderings {
            assert!(
                texts.iter().any(|text| text.contains(rendering)),
                "{}: `{}` is not in the texts of `{}`:\n{}",
                case,
                rendering,
                source,
                texts.join("\n")
            );
        }
    }
}

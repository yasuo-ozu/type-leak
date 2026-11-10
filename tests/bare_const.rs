//! A bare identifier as an argument (`::l::W<ILEN>`) may be a type or a const, and `syn` cannot tell
//! which. The Leaker leaks the type that holds it as a whole, which is right for both. Where it
//! cannot, it must not leak the identifier as a type unless it says so: a const read as a type
//! fails in the copy (rustc: E0573 in the Repeater's impl, `type Type = ILEN;`, and E0747 where the
//! copy passes the projection as a const argument). See "Bare identifiers and consts" in the
//! `Leaker` docs.
//!
//! - The identifier is leaked as a type only where there is no other way, and the docs say so: an
//!   elided lifetime or reference in the holder, `impl Trait`, and a trait bound (which has no
//!   type to leak): `soft_causes_leak_the_identifier`.
//! - For any other cause (`Self::A` of a type, the default of a parameter that names `Self`, a
//!   method lifetime that the copy would leave unconstrained, an expression): an error, with the
//!   identifier as its cause: `hard_causes_are_errors` pins the text.
//! - The cases that are not affected: `unchanged_cases`.
//! - Every place with every cause, checked for the property that the identifier is an entry on its
//!   own only for a soft cause: `the_identifier_is_leaked_alone_only_for_a_soft_cause`.
//!
//! The Referrer's crate of `tests/cross_crate.rs` (`kinds`, and a module) compiles the copies: the
//! path that holds a const (also with `Self` in it) is an entry as a whole, a rejected one expands
//! to nothing, and the identifier that is a type is leaked alone where it is a soft cause.

mod common;

use common::{messages, starts_with};
use syn::*;
use template_quote::{quote, ToTokens};
use type_leak::*;

/// Parses the tokens of an item.
macro_rules! item {
    ($($tokens:tt)*) => {{
        let item: Item = parse_quote!($($tokens)*);
        item
    }};
}

/// Analyzes `item` with the `intern_*` of its kind.
fn analyze_with<A: Allow>(
    mut leaker: Leaker<A>,
    item: &Item,
) -> std::result::Result<Referrer, Vec<ErrorItem>> {
    match item {
        Item::Struct(item) => leaker.intern_struct(item),
        Item::Enum(item) => leaker.intern_enum(item),
        Item::Union(item) => leaker.intern_union(item),
        Item::Impl(item) => leaker.intern_impl(item),
        Item::Trait(item) => leaker.intern_trait(item),
        _ => panic!("not an item to analyze"),
    };
    leaker.result().finish()
}

fn analyze(item: &Item) -> std::result::Result<Referrer, Vec<ErrorItem>> {
    // The absolute paths are allowed, so the types of `::l` are no entries: the entries are those
    // that depend on the context.
    let config = Rules::new().primitive().external_crates();
    analyze_with(Leaker::with_rule(config), item)
}

/// The entries, as text.
fn entries(referrer: &Referrer) -> Vec<String> {
    common::types(referrer)
        .map(|ty| quote!(#ty).to_string())
        .collect()
}

/// What an error says: for each of its errors, the headline, the tokens, the notes and the help.
#[derive(Debug, PartialEq, Eq)]
struct Shown {
    message: String,
    tokens: String,
    notes: Vec<String>,
    help: String,
}

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

/// The headline for `name`.
fn headline(name: &str) -> String {
    format!(
        "`{}` may be a const, and the type that holds it cannot be leaked as a whole",
        name
    )
}

/// The first way to fix it: an absolute path, which a const needs in braces.
fn absolute(name: &str) -> String {
    format!(
        "write `{0}` with an absolute path (`::my_crate::{0}`, or `{{ ::my_crate::{0} }}` for a \
         const)",
        name
    )
}

/// The note that says what the holder cannot be leaked because of.
fn because_of(cause: &str) -> String {
    format!(
        "it cannot be leaked as a whole because of `{}`",
        plain(cause)
    )
}

/// Tokens as a message writes them: `Wrap < T >` is `Wrap<T>`, and `Self :: A` is `Self::A`.
fn plain(text: &str) -> String {
    [
        (" :: ", "::"),
        (":: ", "::"),
        (" ::", "::"),
        (" <", "<"),
        ("< ", "<"),
        (" >", ">"),
        ("& ", "&"),
        (" ,", ","),
    ]
    .iter()
    .fold(text.to_string(), |text, (from, to)| text.replace(from, to))
}

/// The error for `name` with the notes and the help.
fn error(name: &str, notes: &[&str], help: &str) -> Shown {
    Shown {
        message: headline(name),
        tokens: name.to_string(),
        notes: notes.iter().map(|note| note.to_string()).collect(),
        help: help.to_string(),
    }
}

/// The note of an error whose cause is `Self::A` of a type.
const SELF_A_NOTE: &str = "`Self::A` is shorthand that rustc resolves through a trait that this \
                           type implements. In the copy it is written in an impl of the Repeater, \
                           which has no such trait (E0220, or E0223 ambiguous associated type), so \
                           the trait must be written out. `Self` itself is fine";

/// The error for the holder of `name` that names `Self::A` of a type.
fn self_a(name: &str) -> Shown {
    error(
        name,
        &[SELF_A_NOTE, &because_of("Self::A")],
        &format!(
            "{}, or replace `Self::A` with `<Self as SomeTrait>::A`, naming the trait that \
             defines `A` with its absolute path",
            absolute(name)
        ),
    )
}

// ---------------------------------------------------------------------------------------------
// The report: the holder of `ILEN` names `Self`.
// ---------------------------------------------------------------------------------------------

/// The shape of the report: a struct whose field names `Self` in a path that holds a bare
/// identifier. `Self` can be leaked, so the path is the entry: it holds `ILEN` for a type and for a
/// const. (`ILEN` was leaked alone as a type once, `type Type = ILEN;`, which does not compile if
/// it is a const.)
#[test]
fn a_bare_identifier_beside_self_is_leaked_as_a_whole() {
    let item = item!(
        struct S {
            a: ::l::W<[Self], ILEN>,
        }
    );
    let referrer = analyze(&item).unwrap();
    assert_eq!(entries(&referrer), [":: l :: W < [Self] , ILEN >"]);
    // Written so that it is no cause: an absolute path (a const needs braces), or an item that is
    // in scope.
    for item in [
        item!(
            struct S {
                a: ::l::W<[Self], ::my_crate::ILEN>,
            }
        ),
        item!(
            struct S {
                a: ::l::W<[Self], { ::my_crate::ILEN }>,
            }
        ),
        item!(
            struct S<const ILEN: usize> {
                a: ::l::W<[Self], ILEN>,
            }
        ),
    ] {
        let referrer = analyze(&item).unwrap();
        assert!(entries(&referrer).is_empty(), "{}", quote!(#item));
    }
    // Allowed by the config: left as written, so it is no cause either.
    let config = Rules::new()
        .external_crates()
        .or_if(starts_with(parse_quote!(ILEN)));
    let referrer = analyze_with(Leaker::with_rule(config), &item).unwrap();
    assert!(entries(&referrer).is_empty());
}

/// Every kind of item, and every way the holder can name `Self`: the path that holds `ILEN` is the
/// entry.
#[test]
fn the_holder_that_names_self_is_the_entry_for_every_kind_of_item() {
    for item in [
        item!(
            struct S {
                a: ::l::W<Self, ILEN>,
            }
        ),
        item!(
            struct S {
                a: ::l::W<[Self; 2], ILEN>,
            }
        ),
        item!(
            struct S {
                a: <Self as ::l::Tr<ILEN>>::Out,
            }
        ),
        item!(
            enum E {
                A(::l::W<Self, ILEN>),
                B,
            }
        ),
        item!(
            union U {
                a: ::core::mem::ManuallyDrop<::l::W<Self, ILEN>>,
            }
        ),
        item!(
            impl ::l::Tr for ::l::Host {
                fn f(&self) -> ::l::W<Self, ILEN> {
                    loop {}
                }
            }
        ),
        item!(
            impl ::l::Host {
                fn f(&self, x: ::l::W<Self, ILEN>) {}
            }
        ),
        item!(
            trait Tr {
                fn f(&self) -> ::l::W<Self, ILEN>;
            }
        ),
        // A receiver is a type as well.
        item!(
            trait Tr {
                fn f(self: ::l::W<Self, ILEN>);
            }
        ),
        item!(
            trait Tr {
                type A;
                fn f(&self) -> ::l::W<Self::A, ILEN>;
            }
        ),
        item!(
            trait Tr {
                type G<'x>;
                fn f<'y>(&self, x: ::l::W<Self::G<'y>, ILEN>);
            }
        ),
    ] {
        let referrer =
            analyze(&item).unwrap_or_else(|err| panic!("{}: {}", quote!(#item), messages(&err)));
        let leaked = entries(&referrer);
        assert!(
            leaked.len() == 1 && leaked[0] != "ILEN" && leaked[0].contains("ILEN"),
            "{}: {:?}",
            quote!(#item),
            leaked
        );
    }
}

/// `Self::A` where `Self` is a type cannot be written in the Repeater's impl (E0220): the path is
/// not leaked, and `ILEN` is no type.
#[test]
fn an_associated_type_of_a_self_that_is_a_type_is_an_error() {
    for item in [
        item!(
            struct S {
                a: ::l::W<Self::A, ILEN>,
            }
        ),
        item!(
            impl ::l::Host {
                fn f(&self, x: ::l::W<Self::A, ILEN>) {}
            }
        ),
    ] {
        let err = analyze(&item).err().unwrap();
        assert_eq!(shown(&err), [self_a("ILEN")], "{}", quote!(#item));
    }
    // Written out with the trait, it is a path that is leaked as a whole.
    let item = item!(
        struct S {
            a: ::l::W<<Self as ::l::Tr>::A, ILEN>,
        }
    );
    let referrer = analyze(&item).unwrap();
    assert_eq!(
        entries(&referrer),
        [":: l :: W < < Self as :: l :: Tr > :: A , ILEN >"]
    );
}

/// The other causes: each is an error with the identifier as its cause.
#[test]
fn hard_causes_are_errors() {
    // The default of a parameter cannot depend on `Self`, so the path that holds the identifier is
    // not leaked.
    let item = item!(
        trait Tr<T = ::l::W<Self, ILEN>> {}
    );
    let err = analyze(&item).err().unwrap();
    assert_eq!(
        shown(&err),
        [error(
            "ILEN",
            &["a default of a parameter cannot depend on `Self`"],
            &format!(
                "{}, or name the type instead of `Self` in `::l::W<Self, ILEN>`",
                absolute("ILEN")
            )
        )]
    );

    // A bound of a generic that the path uses names `Self`: the path is leaked as a whole.
    let item = item!(
        struct S<T: ::core::convert::AsRef<Self>> {
            a: ::l::W<T, ILEN>,
        }
    );
    let referrer = analyze(&item).unwrap();
    assert_eq!(entries(&referrer), [":: l :: W < T , ILEN >"]);

    // A method lifetime that the copy would leave unconstrained (E0195): no other argument keeps
    // it as written.
    let item = item!(
        trait Tr {
            fn f<'b>(&self, y: &'b dyn ::core::convert::AsRef<ILEN>) -> &'b u8;
        }
    );
    let err = analyze(&item).err().unwrap();
    assert_eq!(
        shown(&err),
        [error(
            "ILEN",
            &[
                "the lifetime of the trait object defaults to `'b`, a lifetime of the method \
                 which its return type mentions and no other argument keeps as written (e.g. \
                 `&'b self`), so the copy would have `'b` only in a projection, and rustc \
                 rejects the implementation of a method whose lifetime is early-bound where \
                 the trait's is late-bound (E0195)",
                &because_of("'b")
            ],
            &format!(
                "keep `'b` in another argument of the method (for example `&'b self`), {}, or \
                 write the bound of the trait object (`dyn Trait + 'b`)",
                absolute("ILEN")
            )
        )]
    );
    // Anchored, the reference is leaked as a whole.
    let anchored = item!(
        trait Tr {
            fn f<'b>(&'b self, y: &'b dyn ::core::convert::AsRef<ILEN>) -> &'b u8;
        }
    );
    let referrer = analyze(&anchored).unwrap();
    assert_eq!(
        entries(&referrer),
        ["& 'b dyn :: core :: convert :: AsRef < ILEN >"]
    );

    // An expression that cannot be leaked, and a type-relative path.
    let item = item!(
        struct S {
            a: ::l::W<{ ::l::m!() }, ILEN>,
        }
    );
    let err = analyze(&item).err().unwrap();
    assert_eq!(
        shown(&err),
        [error(
            "ILEN",
            &[&because_of("::l::m!()")],
            &absolute("ILEN")
        )]
    );
    let item = item!(
        struct S {
            a: ::l::W<<::l::V>::X, ILEN>,
        }
    );
    let err = analyze(&item).err().unwrap();
    assert_eq!(
        shown(&err),
        [error("ILEN", &[&because_of("::l::V")], &absolute("ILEN"))]
    );
}

/// Every identifier that may be a const is told, and an error in a part is not lost.
#[test]
fn every_identifier_is_told() {
    // Two identifiers of one path: the second is a note of the first.
    let item = item!(
        struct S {
            a: ::l::W<ILEN, Self::A, ILEN2>,
        }
    );
    let err = analyze(&item).err().unwrap();
    assert_eq!(
        shown(&err),
        [Shown {
            notes: vec![
                SELF_A_NOTE.to_string(),
                because_of("Self::A"),
                "`ILEN2` also depends on the type context".to_string()
            ],
            ..self_a("ILEN")
        }]
    );
    // A path in a path: the inner one is an error, and so is the outer one, which holds `ILEN2`
    // and names `Self::A` in the inner one.
    let item = item!(
        struct S {
            a: ::l::W<::l::V<Self::A, ILEN>, ILEN2>,
        }
    );
    let err = analyze(&item).err().unwrap();
    assert_eq!(shown(&err), [self_a("ILEN"), self_a("ILEN2")]);
    // The outer path holds no identifier: the inner one is the only error.
    let item = item!(
        struct S {
            a: ::l::W<::l::V<Self::A, ILEN>, u8>,
        }
    );
    let err = analyze(&item).err().unwrap();
    assert_eq!(shown(&err), [self_a("ILEN")]);
    // And where the inner path is leaked as a whole, the outer path is copied as written.
    let item = item!(
        trait Tr {
            fn f(&self, x: ::l::W<::l::V<ILEN>, &()>);
        }
    );
    let referrer = analyze(&item).unwrap();
    assert_eq!(entries(&referrer), [":: l :: V < ILEN >"]);
}

/// `View::check()` says the same: the identifier is the cause of the path that holds it, which
/// cannot be leaked.
#[test]
fn check_agrees() {
    let mut leaker = Leaker::new();
    let ty: Type = parse_quote!(::l::W<Self::A, ILEN>);
    let (_, result) = leaker.enter_type(&parse_quote!(S), &Generics::default(), |view| {
        view.check(&ty)
    });
    assert!(result.is_own);
    assert_eq!(
        result
            .must
            .iter()
            .map(|cause| cause.to_token_stream().to_string())
            .collect::<Vec<_>>(),
        ["ILEN"]
    );
    assert_eq!(
        result
            .impossible
            .iter()
            .map(|cause| cause.to_token_stream().to_string())
            .collect::<Vec<_>>(),
        ["Self :: A"]
    );
    assert_eq!(result.headline.as_deref(), Some(headline("ILEN").as_str()));
    // `Self` itself can be leaked.
    let own: Type = parse_quote!(::l::W<[Self], ILEN>);
    let (_, result) = leaker.enter_type(&parse_quote!(S), &Generics::default(), |view| {
        view.check(&own)
    });
    assert!(result.is_own && result.impossible.is_empty());
    // And `visit_type()` of a scope fails with the error.
    let mut leaker = Leaker::new();
    leaker.enter_type(&parse_quote!(S), &Generics::default(), |view| {
        view.visit_type(&ty)
    });
    let result = leaker.result();
    let err = result.errors();
    assert_eq!(shown(err), [self_a("ILEN")]);
}

// ---------------------------------------------------------------------------------------------
// Where the identifier is still leaked as a type.
// ---------------------------------------------------------------------------------------------

/// The path that holds the identifier cannot be leaked because of an elided lifetime or `impl
/// Trait`, or there is no path (a trait bound): the identifier is leaked alone, as a type, and the
/// copy fails if it is a const. The docs say so.
#[test]
fn soft_causes_leak_the_identifier() {
    let items = [
        // An elided lifetime, in a path and in a reference.
        item!(
            trait Tr {
                fn f(&self, x: &u8) -> ::l::W<'_, ILEN>;
            }
        ),
        item!(
            trait Tr {
                fn f(&self, x: ::l::W<ILEN, &()>);
            }
        ),
        item!(
            trait Tr {
                fn f(&self, x: &dyn ::l::Sup<ILEN>);
            }
        ),
        // `impl Trait`, as the type and in the path.
        item!(
            trait Tr {
                fn f(&self, x: impl ::l::Sup<ILEN>);
            }
        ),
        item!(
            trait Tr {
                fn f(&self, x: ::l::W<ILEN, impl ::l::Sup>);
            }
        ),
        // A trait bound is not a type.
        item!(
            trait Tr: ::l::Sup<ILEN> {}
        ),
        item!(
            trait Tr
            where
                Self: ::l::Sup<ILEN>,
            {
            }
        ),
        item!(
            trait Tr {
                fn f<T: ::l::Sup<ILEN>>(&self, x: T);
            }
        ),
        item!(
            struct S<T: ::l::Sup<ILEN>>(T);
        ),
        item!(
            impl ::l::Tr<ILEN> for ::l::Host {}
        ),
    ];
    for item in items {
        let referrer =
            analyze(&item).unwrap_or_else(|err| panic!("{}: {}", quote!(#item), messages(&err)));
        assert_eq!(entries(&referrer), ["ILEN"], "{}", quote!(#item));
    }
}

/// An elided lifetime and `Self::A` together: `Self::A` is the cause that can be fixed.
#[test]
fn a_hard_cause_wins_over_a_soft_one() {
    let item = item!(
        impl ::l::Host {
            fn f(&self, x: &u8) -> ::l::W<'_, Self::A, ILEN> {
                loop {}
            }
        }
    );
    let err = analyze(&item).err().unwrap();
    let items = shown(&err);
    assert_eq!(items.len(), 1);
    assert_eq!(items[0].message, headline("ILEN"));
    assert!(items[0].notes.contains(&because_of("Self::A")));
    // Without `Self::A`, only the elided lifetime is left.
    let item = item!(
        impl ::l::Host {
            fn f(&self, x: &u8) -> ::l::W<'_, Self, ILEN> {
                loop {}
            }
        }
    );
    let referrer = analyze(&item).unwrap();
    assert_eq!(entries(&referrer), ["ILEN"]);
}

// ---------------------------------------------------------------------------------------------
// What does not change.
// ---------------------------------------------------------------------------------------------

#[test]
fn unchanged_cases() {
    // The path that holds the identifier is leaked as a whole, also with a lifetime and a generic.
    for item in [
        item!(
            struct S {
                a: ::l::W<ILEN>,
            }
        ),
        item!(
            struct S<'a, T> {
                a: ::l::W<'a, T, ILEN>,
            }
        ),
        item!(
            struct S<'a> {
                a: &'a ::l::W<ILEN>,
            }
        ),
        item!(
            struct S {
                a: ::l::W<fn(&u8) -> &u8, ILEN>,
            }
        ),
    ] {
        let referrer = analyze(&item).unwrap();
        assert_eq!(
            entries(&referrer).len(),
            1,
            "{}: {:?}",
            quote!(#item),
            entries(&referrer)
        );
        assert!(entries(&referrer)[0].contains("ILEN"));
    }
    // `Self` is in the entry: the trait's `Self` is the implementor.
    let item = item!(
        trait Tr {
            fn f(&self) -> ::l::W<Self, ILEN>;
        }
    );
    assert_eq!(
        entries(&analyze(&item).unwrap()),
        [":: l :: W < Self , ILEN >"]
    );
    // Only an identifier written as an argument may be a const: in a tuple, an array, a function
    // pointer, a binding or parentheses it is a type, which is leaked alone also where the path
    // cannot be leaked.
    for ty in [
        quote!(::l::W<(ILEN,), <::l::V>::X>),
        quote!(::l::W<[ILEN; 2], <::l::V>::X>),
        quote!(::l::W<fn(ILEN), <::l::V>::X>),
        quote!(::l::W<(ILEN), <::l::V>::X>),
        quote!((ILEN, <::l::V>::X)),
        quote!(dyn ::l::Tr<<::l::V>::X, Out = ILEN>),
        quote!(<ILEN as ::l::Tr<<::l::V>::X>>::Out),
    ] {
        let item = item!(
            struct S {
                a: #ty,
            }
        );
        let referrer = analyze(&item).unwrap_or_else(|err| panic!("{}: {}", ty, messages(&err)));
        assert_eq!(entries(&referrer), ["ILEN"], "{}", ty);
    }
}

// ---------------------------------------------------------------------------------------------
// Every place, with every cause.
// ---------------------------------------------------------------------------------------------

/// What a part of the path that holds the identifier does to it.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Cause {
    /// Nothing: the path can be leaked as a whole.
    None,
    /// An elided lifetime, a reference or `impl Trait`: the path cannot be leaked, and the
    /// identifier is leaked as a type (the docs name them).
    Soft,
    /// Any other: `Self::A`, a type-relative path, `Self` in the default of a parameter.
    Hard,
}

/// `::l::W<part, ILEN>` is analyzed in each place below, with `part` each of these. The identifier
/// is an entry on its own only for a soft cause, whatever the place: that is the property, and the
/// one a bug broke before.
#[test]
fn the_identifier_is_leaked_alone_only_for_a_soft_cause() {
    let parts = [
        ("u8", Cause::None),
        ("[u8; 3]", Cause::None),
        ("T", Cause::None),
        ("&'a u8", Cause::None),
        ("fn(&u8) -> &u8", Cause::None),
        ("<Self as ::l::Tr>::A", Cause::None),
        ("dyn ::l::Sup", Cause::None),
        ("&()", Cause::Soft),
        ("impl ::l::Sup", Cause::Soft),
        ("::l::V<'_>", Cause::Soft),
        ("&dyn ::l::Sup", Cause::Soft),
        ("Self", Cause::Hard),
        ("[Self]", Cause::Hard),
        ("(Self, u8)", Cause::Hard),
        ("&'a Self", Cause::Hard),
        ("Self::A", Cause::Hard),
        ("::l::V<Self>", Cause::Hard),
        ("fn(Self)", Cause::Hard),
        ("<::l::V>::X", Cause::Hard),
    ];
    // Where the path is: `$H` is `::l::W<part, ILEN>` or `::l::W<ILEN, part>`.
    let places = [
        "struct S<'a, T> { a: $H }",
        "struct S<'a, T>(Option<$H>);",
        "enum S<'a, T> { A($H), B }",
        "union S<'a, T: ::core::marker::Copy> { a: ::core::mem::ManuallyDrop<$H>, b: T }",
        "struct S<'a, T> { a: &'a $H }",
        "struct S<'a, T> { a: (u8, $H) }",
        "struct S<'a, T> { a: [$H; 2] }",
        "struct S<'a, T> { a: ::l::Z<$H, u8> }",
        "struct S<'a, T>(fn($H));",
        "struct S<'a, T>(Box<dyn ::l::Sup<$H>>);",
        "struct S<'a, T: ::l::Sup<$H>>(T);",
        "struct S<'a, T, F: ::core::ops::Fn($H)>(T, F);",
        "struct S<'a, T: ::core::convert::AsRef<Self>> { a: $H, b: T }",
        "trait S<'a, T> { fn f(&self) -> $H; }",
        "trait S<'a, T> { fn f(&self, x: $H); }",
        "trait S<'a, T> { fn f<'b>(&self, x: &'b $H) -> &'b u8; }",
        "trait S<'a, T> { fn f(&self, x: $H) -> &u8; }",
        "trait S<'a, T> { fn f(&self, x: for<'x> fn(&'x u8, $H) -> &'x u8); }",
        "trait S<'a, T> { fn f(&self) where $H: ::core::marker::Sized; }",
        "trait S<'a, T> where T: ::l::Sup<$H> {}",
        "trait S<'a, T = $H> {}",
        "trait S { fn f(&self, x: impl ::l::Sup<$H>); }",
        "trait S: ::l::Sup<$H> {}",
        "impl<'a, T> ::l::Tr for ::l::Host<'a, T> { type A = u8; fn f(&self, x: $H) {} }",
        "impl<'a, T> ::l::Host<'a, T> { fn f(&self, x: $H) {} }",
    ];
    let (mut leaked, mut rejected) = (0, 0);
    for place in places {
        for (part, cause) in &parts {
            for holder in [
                format!("::l::W<{}, ILEN>", part),
                format!("::l::W<ILEN, {}>", part),
            ] {
                let item: Item = syn::parse_str(&place.replace("$H", &holder))
                    .unwrap_or_else(|err| panic!("{} in {}: {}", holder, place, err));
                let seen = format!("{} in {}", holder, place);
                match analyze(&item) {
                    Err(_) => rejected += 1,
                    Ok(referrer) => {
                        if entries(&referrer).iter().any(|entry| entry == "ILEN") {
                            leaked += 1;
                            assert_eq!(*cause, Cause::Soft, "{}", seen);
                        }
                    }
                }
            }
        }
    }
    // The test checks something: soft causes leak it, and some places and causes reject.
    assert!(leaked > 100 && rejected > 100, "{} {}", leaked, rejected);
}

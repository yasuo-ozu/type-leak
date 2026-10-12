//! A bare identifier as an argument (`::l::W<ILEN>`) may be a type or a const, and `syn` cannot tell
//! which. The Leaker leaks the type that holds it as a whole, which is right for both. Where it
//! cannot, it must not leak the identifier as a type unless it says so: a const read as a type
//! fails in the copy (rustc: E0573 in the Repeater's impl, `type Type = ILEN;`, and E0747 where the
//! copy passes the projection as a const argument). See "Bare identifiers and consts" in the
//! `Leaker` docs.
//!
//! - The identifier is leaked as a type only where there is no other way, and the docs say so: an
//!   elided lifetime or reference in the holder, `impl Trait`, a path the module of `intern_mod`
//!   declares, and a trait bound (which has no type to leak): `soft_causes_leak_the_identifier`.
//! - For any other cause (`Self` where it is denied, `Self::A`, the default of a parameter, a bound
//!   that names `Self`, a method lifetime that the copy would leave unconstrained, an expression):
//!   an error, with the identifier as its cause: `hard_causes_are_errors` pins the text.
//! - The cases that are not affected: `unchanged_cases`.
//! - Every place with every cause, checked for the property that the identifier is an entry on its
//!   own only for a soft cause: `the_identifier_is_leaked_alone_only_for_a_soft_cause`.
//!
//! The Referrer's crate of `tests/cross_crate.rs` (`kinds`, ids 9081 to 9084) compiles the copies:
//! the path that holds a const (also with `Self` in it) is an entry as a whole, a rejected one
//! expands to nothing, and the identifier that is a type is leaked alone where it is a soft cause.

use syn::*;
use template_quote::quote;
use type_leak::*;

/// Parses the tokens of an item.
macro_rules! item {
    ($($tokens:tt)*) => {{
        let item: Item = parse_quote!($($tokens)*);
        item
    }};
}

/// Analyzes `item` with the `intern_*` of its kind.
fn analyze_with(
    mut leaker: Leaker,
    item: &Item,
    allow_self: bool,
) -> std::result::Result<Referrer, NotInternableError> {
    match item {
        Item::Struct(item) => leaker.intern_struct(item, allow_self),
        Item::Enum(item) => leaker.intern_enum(item, allow_self),
        Item::Union(item) => leaker.intern_union(item, allow_self),
        Item::Impl(item) => leaker.intern_impl(item, allow_self),
        Item::Trait(item) => leaker.intern_trait(item, allow_self),
        Item::Mod(item) => leaker.intern_mod(item),
        _ => panic!("not an item to analyze"),
    }?;
    Ok(leaker.finish())
}

fn analyze(item: &Item, allow_self: bool) -> std::result::Result<Referrer, NotInternableError> {
    let mut config = LeakerConfig::new();
    config.allow_primitive();
    analyze_with(Leaker::from_config(config), item, allow_self)
}

/// The required entries, as text.
fn entries(referrer: &Referrer) -> Vec<String> {
    referrer
        .required()
        .map(|(_, ty)| quote!(#ty).to_string())
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

fn shown(err: &NotInternableError) -> Vec<Shown> {
    err.items()
        .map(|item| Shown {
            message: item.message.clone(),
            tokens: item.tokens.to_string(),
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

const ALLOW: &str = "allow it with `LeakerConfig::allowed_paths`";
const ALLOW_SELF: &str = "analyze the item with `allow_self = true`";

/// The note that says what the holder cannot be leaked because of.
fn because_of(cause: &str) -> String {
    format!("it cannot be leaked as a whole because of `{}`", cause)
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

/// The error where `Self` is denied and the holder names it as `cause`.
fn self_denied(name: &str, cause: &str) -> Shown {
    error(
        name,
        &[&because_of(cause)],
        &format!("{}, {}, or {}", absolute(name), ALLOW, ALLOW_SELF),
    )
}

// ---------------------------------------------------------------------------------------------
// The report: `Self` is denied, and the holder of `ILEN` names it.
// ---------------------------------------------------------------------------------------------

/// The shape of the report: a struct whose field names `Self` in a path that holds a bare
/// identifier. With `allow_self = false` the path cannot be leaked as a whole, and `ILEN` was
/// leaked alone as a type (`type Type = ILEN;`), which does not compile if it is a const.
#[test]
fn a_bare_identifier_beside_a_denied_self_is_not_leaked_as_a_type() {
    let item = item!(
        struct S {
            a: ::l::W<[Self], ILEN>,
        }
    );
    let err = analyze(&item, false).err().unwrap();
    assert_eq!(shown(&err), [self_denied("ILEN", "Self")], "{}", err);
    // The exact text, as the macro shows it.
    assert_eq!(
        syn::Error::from(err).to_string(),
        "`ILEN` may be a const, and the type that holds it cannot be leaked as a whole\n\
         \n\
         \x20 = note: it cannot be leaked as a whole because of `Self`\n\
         \x20 = help: write `ILEN` with an absolute path (`::my_crate::ILEN`, or \
         `{ ::my_crate::ILEN }` for a const), allow it with `LeakerConfig::allowed_paths`, or \
         analyze the item with `allow_self = true`\n"
    );
    // With `Self` allowed the path is the entry: it holds `ILEN` for a type and for a const.
    let referrer = analyze(&item, true).unwrap();
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
        let referrer = analyze(&item, false).unwrap();
        assert!(entries(&referrer).is_empty(), "{}", quote!(#item));
    }
    // Allowed by the config: left as written, so it is no cause either.
    let mut config = LeakerConfig::new();
    config.allowed_paths.push(parse_quote!(ILEN));
    let referrer = analyze_with(Leaker::from_config(config), &item, false).unwrap();
    assert!(entries(&referrer).is_empty());
}

/// Every kind of item that `Self` can be denied in, and every way the holder can name `Self`.
#[test]
fn a_denied_self_in_the_holder_is_an_error_for_every_kind_of_item() {
    for (item, cause) in [
        (
            item!(
                struct S {
                    a: ::l::W<Self, ILEN>,
                }
            ),
            "Self",
        ),
        (
            item!(
                struct S {
                    a: ::l::W<[Self; 2], ILEN>,
                }
            ),
            "Self",
        ),
        (
            item!(
                struct S {
                    a: <Self as ::l::Tr<ILEN>>::Out,
                }
            ),
            "Self",
        ),
        (
            item!(
                enum E {
                    A(::l::W<Self, ILEN>),
                    B,
                }
            ),
            "Self",
        ),
        (
            item!(
                union U {
                    a: ::core::mem::ManuallyDrop<::l::W<Self, ILEN>>,
                }
            ),
            "Self",
        ),
        (
            item!(
                impl ::l::Tr for ::l::Host {
                    fn f(&self) -> ::l::W<Self, ILEN> {
                        loop {}
                    }
                }
            ),
            "Self",
        ),
        (
            item!(
                impl ::l::Host {
                    fn f(&self, x: ::l::W<Self, ILEN>) {}
                }
            ),
            "Self",
        ),
        (
            item!(
                trait Tr {
                    fn f(&self) -> ::l::W<Self, ILEN>;
                }
            ),
            "Self",
        ),
        // A receiver is a type as well.
        (
            item!(
                trait Tr {
                    fn f(self: ::l::W<Self, ILEN>);
                }
            ),
            "Self",
        ),
        (
            item!(
                trait Tr {
                    type A;
                    fn f(&self) -> ::l::W<Self::A, ILEN>;
                }
            ),
            "Self :: A",
        ),
        (
            item!(
                trait Tr {
                    type G<'x>;
                    fn f<'y>(&self, x: ::l::W<Self::G<'y>, ILEN>);
                }
            ),
            "Self :: G < 'y >",
        ),
    ] {
        let denied = analyze(&item, false)
            .err()
            .unwrap_or_else(|| panic!("accepted: {}", quote!(#item)));
        assert_eq!(
            shown(&denied),
            [self_denied("ILEN", cause)],
            "{}",
            quote!(#item)
        );
        // Where `Self` can be leaked, the path that holds `ILEN` is the entry.
        let referrer =
            analyze(&item, true).unwrap_or_else(|err| panic!("{}: {}", quote!(#item), err));
        let leaked = entries(&referrer);
        assert!(
            leaked.len() == 1 && leaked[0] != "ILEN" && leaked[0].contains("ILEN"),
            "{}: {:?}",
            quote!(#item),
            leaked
        );
    }
}

/// `Self::A` where `Self` is a type cannot be written in the Repeater's impl (E0220), whether or
/// not `Self` is denied: the path is not leaked, and `ILEN` is no type.
#[test]
fn an_associated_type_of_a_self_that_is_a_type_is_an_error() {
    let note = "`Self::A` is shorthand that rustc resolves through a trait that this type \
                implements. In the copy it is written in an impl of the Repeater, which has no \
                such trait (E0220, or E0223 ambiguous associated type), so the trait must be \
                written out. `Self` itself is fine";
    let replace = "replace `Self::A` with `<Self as SomeTrait>::A`, naming the trait that defines \
                   `A` with its absolute path";
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
        let err = analyze(&item, true).err().unwrap();
        assert_eq!(
            shown(&err),
            [error(
                "ILEN",
                &[note, &because_of("Self :: A")],
                &format!("{}, or {}", absolute("ILEN"), replace)
            )],
            "{}",
            quote!(#item)
        );
        // `Self` is denied as well: the advice says so.
        let err = analyze(&item, false).err().unwrap();
        assert_eq!(
            shown(&err),
            [error(
                "ILEN",
                &[note, &because_of("Self :: A")],
                &format!("{}, {}, or {}", absolute("ILEN"), replace, ALLOW_SELF)
            )],
            "{}",
            quote!(#item)
        );
        // Written out with the trait, it is a path that is leaked as a whole.
    }
    let item = item!(
        struct S {
            a: ::l::W<<Self as ::l::Tr>::A, ILEN>,
        }
    );
    let referrer = analyze(&item, true).unwrap();
    assert_eq!(
        entries(&referrer),
        [":: l :: W < < Self as :: l :: Tr > :: A , ILEN >"]
    );
}

/// The other causes: each is an error with the identifier as its cause.
#[test]
fn hard_causes_are_errors() {
    // The default of a parameter cannot depend on `Self`, so the path that holds the identifier is
    // not leaked, with `Self` allowed or not.
    for allow_self in [true, false] {
        let item = item!(
            trait Tr<T = ::l::W<Self, ILEN>> {}
        );
        let err = analyze(&item, allow_self).err().unwrap();
        assert_eq!(
            shown(&err),
            [error(
                "ILEN",
                &["a default of a parameter cannot depend on `Self`"],
                &format!(
                    "{}, or write the default without `Self`: name the type instead of `Self` in \
                     `::l::W<Self, ILEN>`",
                    absolute("ILEN")
                )
            )],
            "{}",
            allow_self
        );
    }

    // A bound of a generic that the path uses names `Self`, which is denied.
    let item = item!(
        struct S<T: ::core::convert::AsRef<Self>> {
            a: ::l::W<T, ILEN>,
        }
    );
    let err = analyze(&item, false).err().unwrap();
    assert_eq!(
        shown(&err),
        [error(
            "ILEN",
            &[
                "the bounds of its generics mention `Self`, which cannot be leaked",
                &because_of("Self")
            ],
            &format!(
                "{}, write the bound of the generics of `::l::W<T, ILEN>` without `Self`, or \
                 analyze the item with `allow_self = true`",
                absolute("ILEN")
            )
        )]
    );
    let referrer = analyze(&item, true).unwrap();
    assert_eq!(entries(&referrer), [":: l :: W < T , ILEN >"]);

    // A method lifetime that the copy would leave unconstrained (E0195): no other argument keeps
    // it as written.
    let item = item!(
        trait Tr {
            fn f<'b>(&self, y: &'b dyn ::core::convert::AsRef<ILEN>) -> &'b u8;
        }
    );
    for allow_self in [true, false] {
        let err = analyze(&item, allow_self).err().unwrap();
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
    }
    // Anchored, the reference is leaked as a whole.
    let anchored = item!(
        trait Tr {
            fn f<'b>(&'b self, y: &'b dyn ::core::convert::AsRef<ILEN>) -> &'b u8;
        }
    );
    let referrer = analyze(&anchored, false).unwrap();
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
    let err = analyze(&item, true).err().unwrap();
    assert_eq!(
        shown(&err),
        [error(
            "ILEN",
            &[&because_of(":: l :: m ! ()")],
            &format!("{}, or {}", absolute("ILEN"), ALLOW)
        )]
    );
    let item = item!(
        struct S {
            a: ::l::W<<::l::V>::X, ILEN>,
        }
    );
    let err = analyze(&item, true).err().unwrap();
    assert_eq!(
        shown(&err),
        [error(
            "ILEN",
            &[&because_of(":: l :: V")],
            &format!("{}, or {}", absolute("ILEN"), ALLOW)
        )]
    );
}

/// Every identifier that may be a const is told, and an error in a part is not lost.
#[test]
fn every_identifier_is_told() {
    // Two identifiers of one path: the second is a note of the first.
    let item = item!(
        struct S {
            a: ::l::W<ILEN, [Self], ILEN2>,
        }
    );
    let err = analyze(&item, false).err().unwrap();
    assert_eq!(
        shown(&err),
        [Shown {
            notes: vec![
                because_of("Self"),
                "`ILEN2` also depends on the type context".to_string()
            ],
            ..self_denied("ILEN", "Self")
        }]
    );
    // A path in a path: the inner one is an error, and so is the outer one, which holds `ILEN2`
    // and names `Self` in the inner one.
    let item = item!(
        trait Tr {
            fn f(&self, x: ::l::W<::l::V<[Self], ILEN>, ILEN2>);
        }
    );
    let err = analyze(&item, false).err().unwrap();
    assert_eq!(
        shown(&err),
        [self_denied("ILEN", "Self"), self_denied("ILEN2", "Self")]
    );
    // The outer path holds no identifier: the inner one is the only error.
    let item = item!(
        trait Tr {
            fn f(&self, x: ::l::W<::l::V<[Self], ILEN>, u8>);
        }
    );
    let err = analyze(&item, false).err().unwrap();
    assert_eq!(shown(&err), [self_denied("ILEN", "Self")]);
    // And where the inner path is leaked as a whole, the outer path is copied as written.
    let item = item!(
        trait Tr {
            fn f(&self, x: ::l::W<::l::V<ILEN>, [Self]>);
        }
    );
    let referrer = analyze(&item, false).unwrap();
    assert_eq!(entries(&referrer), [":: l :: V < ILEN >"]);
}

/// `View::check()` says the same: the identifier is the cause of the path that holds it, which
/// cannot be leaked.
#[test]
fn check_agrees() {
    let mut leaker = Leaker::new();
    let ty: Type = parse_quote!(::l::W<[Self], ILEN>);
    let (_, result) = leaker
        .enter_type(&parse_quote!(S), &Generics::default(), false, |view| {
            view.check(&ty)
        })
        .unwrap();
    assert!(result.is_own);
    assert_eq!(
        result
            .must
            .iter()
            .map(|cause| cause.to_string())
            .collect::<Vec<_>>(),
        ["ILEN"]
    );
    assert_eq!(
        result
            .impossible
            .iter()
            .map(|cause| cause.to_string())
            .collect::<Vec<_>>(),
        ["Self"]
    );
    assert_eq!(result.headline.as_deref(), Some(headline("ILEN").as_str()));
    // With `Self` allowed it is leaked.
    let (_, result) = leaker
        .enter_type(&parse_quote!(S), &Generics::default(), true, |view| {
            view.check(&ty)
        })
        .unwrap();
    assert!(result.is_own && result.impossible.is_empty());
    // And `visit_type()` of a scope fails with the error.
    let err = Leaker::new()
        .enter_type(&parse_quote!(S), &Generics::default(), false, |view| {
            view.visit_type(&ty)
        })
        .unwrap_err();
    assert_eq!(shown(&err), [self_denied("ILEN", "Self")]);
}

// ---------------------------------------------------------------------------------------------
// Where the identifier is still leaked as a type.
// ---------------------------------------------------------------------------------------------

/// The path that holds the identifier cannot be leaked because of an elided lifetime, `impl
/// Trait`, or a name that the module declares, or there is no path (a trait bound): the identifier
/// is leaked alone, as a type, and the copy fails if it is a const. The docs say so.
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
        // A name that the module declares.
        item!(
            mod m {
                struct Local<A>(A);
                struct S {
                    a: Local<ILEN>,
                }
            }
        ),
    ];
    for item in items {
        for allow_self in [true, false] {
            let referrer = analyze(&item, allow_self)
                .unwrap_or_else(|err| panic!("{}: {}", quote!(#item), err));
            assert_eq!(
                entries(&referrer),
                ["ILEN"],
                "{} (allow_self = {})",
                quote!(#item),
                allow_self
            );
        }
    }
}

/// An elided lifetime and `Self` together: `Self` is the cause that can be fixed.
#[test]
fn a_hard_cause_wins_over_a_soft_one() {
    let item = item!(
        trait Tr {
            fn f(&self, x: &u8) -> ::l::W<'_, Self, ILEN>;
        }
    );
    let err = analyze(&item, false).err().unwrap();
    let items = shown(&err);
    assert_eq!(items.len(), 1);
    assert_eq!(items[0].message, headline("ILEN"));
    assert!(items[0].notes.contains(&because_of("Self")));
    // With `Self` allowed, only the elided lifetime is left.
    let referrer = analyze(&item, true).unwrap();
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
        for allow_self in [true, false] {
            let referrer = analyze(&item, allow_self).unwrap();
            assert_eq!(
                entries(&referrer).len(),
                1,
                "{}: {:?}",
                quote!(#item),
                entries(&referrer)
            );
            assert!(entries(&referrer)[0].contains("ILEN"));
        }
    }
    // With `Self` allowed, `Self` is in the entry: the trait's `Self` is the implementor.
    let item = item!(
        trait Tr {
            fn f(&self) -> ::l::W<Self, ILEN>;
        }
    );
    assert_eq!(
        entries(&analyze(&item, true).unwrap()),
        [":: l :: W < Self , ILEN >"]
    );
    // Only an identifier written as an argument may be a const: in a tuple, an array, a function
    // pointer, a binding or parentheses it is a type, which is leaked alone also where the path
    // cannot be leaked.
    for ty in [
        quote!(::l::W<(ILEN,), Self>),
        quote!(::l::W<[ILEN; 2], Self>),
        quote!(::l::W<fn(ILEN), Self>),
        quote!(::l::W<(ILEN), Self>),
        quote!((ILEN, Self)),
        quote!(dyn ::l::Tr<Self, Out = ILEN>),
        quote!(<ILEN as ::l::Tr<Self>>::Out),
    ] {
        let item = item!(
            struct S {
                a: #ty,
            }
        );
        let referrer = analyze(&item, false).unwrap_or_else(|err| panic!("{}: {}", ty, err));
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
    /// Any other: `Self` where it is denied, `Self::A`, a type-relative path.
    Hard,
}

/// `::l::W<part, ILEN>` is analyzed in each place below, with `part` each of these, and with `Self`
/// allowed or denied. The identifier is an entry on its own only for a soft cause, whatever the
/// place: that is the property, and the one the bug broke (`W<[Self], ILEN>`, denied).
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
                for allow_self in [true, false] {
                    let seen = format!("{} (allow_self = {}) in {}", holder, allow_self, place);
                    match analyze(&item, allow_self) {
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
    }
    // The test checks something: soft causes leak it, and some places and causes reject.
    assert!(leaked > 100 && rejected > 100, "{} {}", leaked, rejected);
}

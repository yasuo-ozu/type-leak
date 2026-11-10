//! A bare identifier as an argument (`::l::W<ILEN>`) may be a type or a const, and `syn` cannot tell
//! which. The Leaker leaks the type that holds it as a whole, which is right for both. Where it
//! cannot, it must not leak the identifier as a type unless it says so: a const read as a type
//! fails in the copy (rustc: E0573 in the Repeater's impl, `type Type = ILEN;`, and E0747 where the
//! copy passes the projection as a const argument). See "Bare identifiers and consts" in the
//! `Leaker` docs.
//!
//! - The identifier is leaked as a type only where there is no other way, and the docs say so: an
//!   elided lifetime or reference in the holder, `impl Trait`, a holder whose path the module of
//!   `intern_mod` declares (`Local<ILEN>`), and a trait bound (which has no type to leak):
//!   `soft_causes_leak_the_identifier`.
//! - A holder with the path of another crate that names an item of the module beside the
//!   identifier (`::l::W<Local, ILEN>`) is not leaked whole, as the copy declares `Local` again,
//!   and the parameters of `W` are not known: an error, unless the module declares `ILEN` too
//!   (`a_bare_identifier_beside_a_name_the_module_declares_is_not_leaked_as_a_type`, and the
//!   tests after it).
//! - For any other cause (`Self` where it is denied, `Self::A`, the default of a parameter, a bound
//!   that names `Self`, a method lifetime that the copy would leave unconstrained, an expression):
//!   an error, with the identifier as its cause: `hard_causes_are_errors` pins the text.
//! - The cases that are not affected: `unchanged_cases`.
//! - Every place with every cause, checked for the property that the identifier is an entry on its
//!   own only for a soft cause: `the_identifier_is_leaked_alone_only_for_a_soft_cause`.
//!
//! The Referrer's crate of `tests/cross_crate.rs` (`kinds`, ids 9081 to 9084, and 9101 and 9102
//! for a module) compiles the copies: the path that holds a const (also with `Self` in it) is an
//! entry as a whole, a rejected one expands to nothing, and the identifier that is a type is leaked
//! alone where it is a soft cause.

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
fn analyze_with(
    mut leaker: Leaker,
    item: &Item,
    allow_self: bool,
) -> std::result::Result<Referrer, Vec<ErrorItem>> {
    match item {
        Item::Struct(item) => leaker.intern_struct(item, allow_self),
        Item::Enum(item) => leaker.intern_enum(item, allow_self),
        Item::Union(item) => leaker.intern_union(item, allow_self),
        Item::Impl(item) => leaker.intern_impl(item, allow_self),
        Item::Trait(item) => leaker.intern_trait(item, allow_self),
        Item::Mod(item) => leaker.intern_mod(item),
        _ => panic!("not an item to analyze"),
    };
    leaker.result().finish()
}

fn analyze(item: &Item, allow_self: bool) -> std::result::Result<Referrer, Vec<ErrorItem>> {
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

/// The messages of the errors, one per line.
fn messages(errors: &[ErrorItem]) -> String {
    errors
        .iter()
        .map(|item| item.message.as_str())
        .collect::<Vec<_>>()
        .join("\n")
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

const ALLOW: &str = "allow it with `LeakerConfig::allow_path()`";
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
    assert_eq!(
        shown(&err),
        [self_denied("ILEN", "Self")],
        "{}",
        messages(&err)
    );
    // The exact text, as the macro shows it.
    assert_eq!(
        to_syn_error(&err).to_string(),
        "`ILEN` may be a const, and the type that holds it cannot be leaked as a whole\n\
         \n\
         \x20 = note: it cannot be leaked as a whole because of `Self`\n\
         \x20 = help: write `ILEN` with an absolute path (`::my_crate::ILEN`, or \
         `{ ::my_crate::ILEN }` for a const), allow it with `LeakerConfig::allow_path()`, or \
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
    config.allow_path(parse_quote!(ILEN));
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
        let referrer = analyze(&item, true)
            .unwrap_or_else(|err| panic!("{}: {}", quote!(#item), messages(&err)));
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
                    "{}, or name the type instead of `Self` in `::l::W<Self, ILEN>`",
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
    let (_, result) = leaker.enter_type(&parse_quote!(S), &Generics::default(), false, |view| {
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
        ["Self"]
    );
    assert_eq!(result.headline.as_deref(), Some(headline("ILEN").as_str()));
    // With `Self` allowed it is leaked.
    let (_, result) = leaker.enter_type(&parse_quote!(S), &Generics::default(), true, |view| {
        view.check(&ty)
    });
    assert!(result.is_own && result.impossible.is_empty());
    // And `visit_type()` of a scope fails with the error.
    let mut leaker = Leaker::new();
    leaker.enter_type(&parse_quote!(S), &Generics::default(), false, |view| {
        view.visit_type(&ty)
    });
    let result = leaker.result();
    let err = result.errors();
    assert_eq!(shown(err), [self_denied("ILEN", "Self")]);
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
                .unwrap_or_else(|err| panic!("{}: {}", quote!(#item), messages(&err)));
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
        let referrer =
            analyze(&item, false).unwrap_or_else(|err| panic!("{}: {}", ty, messages(&err)));
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

// ---------------------------------------------------------------------------------------------
// A name that the module declares, beside the identifier.
// ---------------------------------------------------------------------------------------------

/// The module of the property test that found the bug, reduced: `RC` and `R0` are declared in the
/// module analyzed, the place is in a module nested in it (or in it), and `ILEN` is a constant
/// that `ilen` declares or imports. `$RC`, `$R0` and `$Tr0` are written as the place needs them
/// (`super::RC` in the nested module), and `$H` is the holder of `ILEN`.
fn module_with(nested: bool, ilen: &str, place: &str, holder: &str) -> Item {
    let (open, close, up) = if nested {
        ("pub mod n0 {", "}", "super::")
    } else {
        ("", "", "")
    };
    let place = place
        .replace("$H", holder)
        .replace("$RC", &format!("{}RC", up))
        .replace("$R0", &format!("{}R0", up))
        .replace("$Tr0", &format!("{}Tr0", up));
    let source = format!(
        "pub mod mm0 {{
            pub struct RC<T: ?Sized, const N: usize>(::core::marker::PhantomData<*const T>);
            pub struct R0;
            pub trait Tr0 {{}}
            {open}
                {ilen}
                {place}
            {close}
        }}",
        open = open,
        close = close,
        ilen = ilen,
        place = place,
    );
    syn::parse_str(&source).unwrap_or_else(|err| panic!("{}: {}", source, err))
}

/// The config of the tests of modules: primitives and the prelude (`Vec`, `Box`, ..) are copied.
fn module_config() -> LeakerConfig {
    let mut config = LeakerConfig::new();
    config.allow_primitive().allow_prelude();
    config
}

fn analyze_module(item: &Item) -> std::result::Result<Referrer, Vec<ErrorItem>> {
    analyze_with(Leaker::from_config(module_config()), item, true)
}

/// `ILEN` imported from outside the module (not declared by it), nested or not.
fn imported(nested: bool) -> &'static str {
    if nested {
        "use super::super::base::BLEN as ILEN;"
    } else {
        "use super::base::BLEN as ILEN;"
    }
}

/// The error for `name` beside `module_item`, a name that the module declares.
fn beside_module_item(name: &str, module_item: &str) -> Shown {
    error(
        name,
        &[&format!(
            "the type names `{}`, which the module declares and the copy declares again, so it \
             is not leaked as a whole",
            module_item
        )],
        &format!(
            "{0}, import `{1}` with an absolute `use` (`use ::my_crate::{1};`), or {2}",
            absolute(name),
            name,
            ALLOW
        ),
    )
}

/// The shape of the report (a trait method in a nested module, with a const generic of the method):
/// the holder of `ILEN` is a path of another crate that names `super::RC` too, which the module
/// declares. The copy declares `RC` again, so the holder is not leaked as a whole, and `ILEN` was
/// leaked alone as a type (`type Type = ILEN;`), which does not compile if it is a const.
#[test]
fn a_bare_identifier_beside_a_name_the_module_declares_is_not_leaked_as_a_type() {
    let place = "pub trait Tr<T> { fn m<const M: usize>(&self, a0: $H); }";
    let holder = "::l::BC<$RC<*mut crate::base::B0, M>, ILEN>";
    for (nested, name) in [(true, "super::RC"), (false, "RC")] {
        let item = module_with(nested, imported(nested), place, holder);
        let err = analyze_module(&item).err().unwrap();
        assert_eq!(
            shown(&err),
            [beside_module_item("ILEN", name)],
            "{}",
            nested
        );
    }
    // The exact text, as the macro shows it.
    let item = module_with(true, imported(true), place, holder);
    let err = analyze_module(&item).err().unwrap();
    assert_eq!(
        to_syn_error(&err).to_string(),
        "`ILEN` may be a const, and the type that holds it cannot be leaked as a whole\n\
         \n\
         \x20 = note: the type names `super::RC`, which the module declares and the copy \
         declares again, so it is not leaked as a whole\n\
         \x20 = help: write `ILEN` with an absolute path (`::my_crate::ILEN`, or \
         `{ ::my_crate::ILEN }` for a const), import `ILEN` with an absolute `use` \
         (`use ::my_crate::ILEN;`), or allow it with `LeakerConfig::allow_path()`\n"
    );
}

/// The identifier is declared by the module, or imported with an absolute path (which declares it),
/// or written so that it is not leaked: it is copied as written (it resolves in the copy), whatever
/// else the holder names.
#[test]
fn a_name_the_module_declares_beside_the_identifier_is_no_cause_if_the_identifier_is_declared() {
    let place = "pub trait Tr<T> { fn m<const M: usize>(&self, a0: $H); }";
    for nested in [true, false] {
        for ilen in [
            // Declared by the module.
            "pub const ILEN: usize = 2;",
            // Imported with an absolute path.
            "use ::l::base::BLEN as ILEN;",
        ] {
            for holder in [
                "::l::BC<$RC<u8, 3>, ILEN>",
                "::l::BC<ILEN, $RC<u8, 3>>",
                "::l::BC<$R0, ILEN>",
                "::l::BC<Vec<$R0>, ILEN>",
                "::l::BC<$RC<[u8; M], M>, ILEN>",
            ] {
                let item = module_with(nested, ilen, place, holder);
                let referrer = analyze_module(&item).unwrap_or_else(|err| {
                    panic!("{}, {}, {}: {}", nested, ilen, holder, messages(&err))
                });
                assert!(
                    entries(&referrer).is_empty(),
                    "{}, {}, {}: {:?}",
                    nested,
                    ilen,
                    holder,
                    entries(&referrer)
                );
            }
        }
        // Written with an absolute path, in braces or not.
        for ilen in ["::l::base::BLEN", "{ ::l::base::BLEN }"] {
            let holder = format!("::l::BC<$RC<u8, 3>, {}>", ilen);
            let item = module_with(nested, imported(nested), place, &holder);
            let referrer = analyze_module(&item).unwrap();
            assert!(entries(&referrer).is_empty(), "{}", ilen);
        }
    }
    // Allowed by the config: left as written, so it is no cause either.
    let item = module_with(true, imported(true), place, "::l::BC<$RC<u8, 3>, ILEN>");
    let mut config = module_config();
    config.allow_path(parse_quote!(ILEN));
    let referrer = analyze_with(Leaker::from_config(config), &item, true).unwrap();
    assert!(entries(&referrer).is_empty());
}

/// The module declares the path of the holder itself (`RC<u8, ILEN>`): the identifier is leaked
/// alone as a type, the soft cause that the docs name (the module's own type takes a type here, as
/// a rule), also from the holders of that holder.
#[test]
fn a_name_the_module_declares_as_the_path_of_the_holder_still_leaks_the_identifier() {
    let place = "pub trait Tr<T> { fn m<const M: usize>(&self, a0: $H); }";
    for nested in [true, false] {
        for holder in [
            "$RC<u8, ILEN>",
            "$RC<[u8; M], ILEN>",
            "Box<$RC<u8, ILEN>>",
            "Vec<$RC<u8, ILEN>>",
            "(u8, $RC<u8, ILEN>)",
            "&'static $RC<u8, ILEN>",
            // The path of the holder is the module's, whatever the holder names beside it.
            "$RC<$R0, ILEN>",
            "::l::BC<u8, $RC<u8, ILEN>>",
            "::l::BC<$R0, $RC<u8, ILEN>>",
            "::l::BC<$RC<u8, ILEN>, M>",
            "Box<dyn $Tr0>",
        ] {
            let item = module_with(nested, imported(nested), place, holder);
            let referrer = analyze_module(&item)
                .unwrap_or_else(|err| panic!("{}, {}: {}", nested, holder, messages(&err)));
            let leaked = entries(&referrer);
            if holder.contains("ILEN") {
                assert_eq!(leaked, ["ILEN"], "{}, {}", nested, holder);
            } else {
                assert!(leaked.is_empty(), "{}, {}: {:?}", nested, holder, leaked);
            }
        }
    }
}

/// The standard library takes types only (but for `array::IntoIter`), so `Result<Local, Error>` is
/// no error: `Error` is imported from outside the module, and it is leaked alone as a type.
#[test]
fn a_standard_path_beside_a_name_the_module_declares_leaks_the_identifier() {
    for (holder, leaked) in [
        ("Result<Local, Error>", true),
        ("Result<Error, Vec<Local>>", true),
        ("::core::result::Result<Local, Error>", true),
        ("::std::collections::HashMap<Local, Error>", true),
        ("::core::array::IntoIter<Local, Error>", false),
        ("::std::array::IntoIter<Local, Error>", false),
        ("::l::Result<Local, Error>", false),
    ] {
        let source = format!(
            "mod m {{
                use super::Error;
                pub struct Local;
                pub struct S(pub {});
            }}",
            holder
        );
        let item: Item = syn::parse_str(&source).unwrap();
        match analyze_module(&item) {
            Ok(referrer) => {
                assert!(leaked, "{}: {:?}", holder, entries(&referrer));
                assert_eq!(entries(&referrer), ["Error"], "{}", holder);
            }
            Err(err) => {
                assert!(!leaked, "{}: {}", holder, messages(&err));
                assert_eq!(
                    shown(&err),
                    [beside_module_item("Error", "Local")],
                    "{}",
                    holder
                );
            }
        }
    }
}

/// Every place of the module with every holder: the identifier is an entry on its own only where
/// the module declares the path of its holder (or the path is the standard library's), an error
/// where the holder names a name of the module beside it, and part of an entry otherwise. Nested
/// or not, for each place.
#[test]
fn in_a_module_the_identifier_is_leaked_alone_only_for_a_path_the_module_declares() {
    #[derive(Clone, Copy, PartialEq, Debug)]
    enum Holder {
        /// The identifier is leaked alone as a type: the module declares the path of the holder, or
        /// it is a path of the standard library, which takes types only.
        Alone,
        /// A path of another crate that names an item of the module.
        Beside,
        /// A path of another crate.
        Outside,
    }
    let holders = [
        ("$RC<u8, ILEN>", Holder::Alone),
        ("Box<$RC<$R0, ILEN>>", Holder::Alone),
        ("Result<$R0, ILEN>", Holder::Alone),
        ("Result<ILEN, $RC<u8, 3>>", Holder::Alone),
        ("Option<ILEN>", Holder::Outside),
        ("::core::result::Result<$R0, ILEN>", Holder::Alone),
        ("::std::collections::HashMap<Vec<$R0>, ILEN>", Holder::Alone),
        ("::core::array::IntoIter<$R0, ILEN>", Holder::Beside),
        ("::core::simd::Simd<$R0, ILEN>", Holder::Beside),
        ("::l::BC<u8, ILEN>", Holder::Outside),
        ("::l::BC<Vec<u8>, ILEN>", Holder::Outside),
        ("::l::BC<(u8, u8), ILEN>", Holder::Outside),
        ("::l::BC<M, ILEN>", Holder::Outside),
        ("::l::BC<$RC<u8, 3>, ILEN>", Holder::Beside),
        ("::l::BC<ILEN, $RC<u8, 3>>", Holder::Beside),
        ("::l::BC<$R0, ILEN>", Holder::Beside),
        ("::l::BC<Vec<$R0>, ILEN>", Holder::Beside),
        ("::l::BC<&'static [$R0], ILEN>", Holder::Beside),
        ("::l::BC<(u8, $R0), ILEN>", Holder::Beside),
        ("::l::BC<fn($R0), ILEN>", Holder::Beside),
        ("::l::BC<[$R0; 2], ILEN>", Holder::Beside),
        ("::l::BC<<$R0 as ::l::Tr>::Out, ILEN>", Holder::Beside),
        ("::l::BC<dyn $Tr0, ILEN>", Holder::Beside),
        ("Vec<::l::BC<$R0, ILEN>>", Holder::Beside),
        ("::l::BC<$R0, ILEN, ILEN>", Holder::Beside),
        ("::l::BC<$RC<u8, ILEN>, ILEN>", Holder::Beside),
        ("::l::BC<crate::base::B0, $R0, ILEN>", Holder::Beside),
    ];
    let places = [
        "pub trait Tr<T> { fn m<const M: usize>(&self, a0: $H); }",
        "pub trait Tr<T> { fn m<const M: usize>(&self) -> $H; }",
        "pub struct S<const M: usize> { pub a: $H }",
        "pub enum E<const M: usize> { A($H), B }",
        "pub union U<const M: usize> { pub a: ::core::mem::ManuallyDrop<$H> }",
        "pub fn f<const M: usize>(x: $H) {}",
        "pub type A<const M: usize> = $H;",
        "pub struct S2<const M: usize>(pub Option<$H>);",
        "pub struct S3<const M: usize>(pub [$H; 2]);",
        "pub struct S4<const M: usize>(pub fn($H) -> u8);",
    ];
    let (mut alone, mut rejected, mut whole) = (0, 0, 0);
    for place in places {
        for (holder, kind) in holders {
            for nested in [true, false] {
                let item = module_with(nested, imported(nested), place, holder);
                let seen = format!("{} in {} (nested = {})", holder, place, nested);
                match (kind, analyze_module(&item)) {
                    (Holder::Beside, Err(err)) => {
                        let items = shown(&err);
                        assert!(!items.is_empty(), "{}", seen);
                        for item in &items {
                            assert_eq!(item.message, headline("ILEN"), "{}", seen);
                            assert!(
                                item.notes[0].starts_with("the type names `"),
                                "{}: {:?}",
                                seen,
                                item.notes
                            );
                        }
                        rejected += 1;
                    }
                    (Holder::Beside, Ok(referrer)) => {
                        panic!("accepted: {}: {:?}", seen, entries(&referrer))
                    }
                    (_, Err(err)) => panic!("{}: {}", seen, messages(&err)),
                    (kind, Ok(referrer)) => {
                        let leaked = entries(&referrer);
                        let is_alone = leaked.iter().any(|entry| entry == "ILEN");
                        assert_eq!(is_alone, kind == Holder::Alone, "{}: {:?}", seen, leaked);
                        if is_alone {
                            alone += 1;
                        } else {
                            whole += 1;
                            assert!(
                                leaked.iter().any(|entry| entry.contains("ILEN")),
                                "{}: {:?}",
                                seen,
                                leaked
                            );
                        }
                    }
                }
            }
        }
    }
    // The test checks something: every kind of outcome happened many times.
    assert!(
        alone > 20 && rejected > 100 && whole > 40,
        "{} {} {}",
        alone,
        rejected,
        whole
    );
}

/// `enter_mod` declares the names itself: the names it lists are declared, and the rest is not.
#[test]
fn a_name_that_enter_mod_declares_beside_the_identifier_is_an_error_too() {
    let item: Item = parse_quote! {
        pub struct S {
            pub a: ::l::BC<Local, ILEN>,
        }
    };
    let analyze_in = |declared: &[&str]| {
        let declared: Vec<Ident> = declared
            .iter()
            .map(|name| Ident::new(name, proc_macro2::Span::call_site()))
            .collect();
        let mut leaker = Leaker::from_config(module_config());
        leaker.enter_mod("m", declared, |view| view.visit_item(&item));
        leaker.result().finish()
    };
    // `Local` is declared: `BC<Local, ILEN>` names it beside the identifier.
    let err = analyze_in(&["Local"]).err().unwrap();
    assert_eq!(shown(&err), [beside_module_item("ILEN", "Local")]);
    // `ILEN` is declared too: it is copied as written.
    let referrer = analyze_in(&["Local", "ILEN"]).unwrap();
    assert!(entries(&referrer).is_empty());
    // `Local` is not declared: it is a relative path, and the path that holds it is an entry.
    let referrer = analyze_in(&[]).unwrap();
    assert_eq!(entries(&referrer), [":: l :: BC < Local , ILEN >"]);
}

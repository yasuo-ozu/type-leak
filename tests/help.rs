//! The help of every kind of error of the Leaker.
//!
//! An error says what is wrong (`ErrorItem::message`), why (`ErrorItem::notes`) and what to write
//! instead (`ErrorItem::help`). The help is the only part of an error that a user can act on, so
//! every kind of error has one, with the tokens of the error in it, and the tests below pin its
//! text:
//!
//! - `every_kind_of_error_has_a_help` is a table with one row per kind of error (the rows are the
//!   kinds of error of `NotInternableError`): the headline and the exact help of each error of the
//!   row;
//! - `a_help_is_rendered_after_the_notes` asserts how the stable layout (`syn::Error`, which
//!   `ToTokens` renders as `compile_error!`) shows it, as `= help: ..` after the `= note: ..`s.
//!
//! The nightly layout (a `help` of the `Diagnostic`, with its own span) is in `tests/diagnostic.rs`.

use syn::*;
use type_leak::*;

/// The config of the rows with `::p` allowed (the crate of the property test).
fn allowing() -> Leaker {
    let mut config = LeakerConfig::new();
    config.allow_primitive().allow_std_prelude();
    config.allowed_paths.push(parse_quote!(::p));
    Leaker::from_config(config)
}

fn id(name: &str) -> Ident {
    Ident::new(name, proc_macro2::Span::call_site())
}

/// One row: the kind of error, how to make it, and the headline and help of each of its errors.
struct Row {
    kind: &'static str,
    error: fn() -> NotInternableError,
    items: &'static [(&'static str, &'static str)],
}

const ROWS: &[Row] = &[
    // ── a path that depends on the context ───────────────────────────────────────────────────
    Row {
        kind: "a relative name of the prelude",
        error: || {
            let item: ItemTrait = parse_quote!(
                trait Tr: Clone {}
            );
            Leaker::new().intern_trait(&item, true).unwrap_err()
        },
        items: &[(
            "use absolute path: 'Clone'",
            "use `::core::clone::Clone` instead of `Clone`",
        )],
    },
    Row {
        kind: "a primitive type in a const expression",
        error: || {
            let item: ItemTrait = parse_quote!(
                trait Tr: ::my_crate::Sup<{ usize::MAX }> {}
            );
            Leaker::new().intern_trait(&item, true).unwrap_err()
        },
        items: &[(
            "use absolute path: 'usize :: MAX'",
            "use `::core::primitive::usize` instead of `usize`",
        )],
    },
    Row {
        kind: "a relative path that is not in the prelude",
        error: || {
            let item: ItemTrait = parse_quote!(
                trait Tr: MyBound {}
            );
            Leaker::new().intern_trait(&item, true).unwrap_err()
        },
        items: &[(
            "use absolute path: 'MyBound'",
            "write `MyBound` with an absolute path (e.g. `::my_crate::MyBound`), or allow it \
             with `LeakerConfig::allowed_paths`",
        )],
    },
    Row {
        kind: "a path of the crate",
        error: || {
            let item: ItemTrait = parse_quote!(
                trait Tr: crate::m::Bound {}
            );
            Leaker::new().intern_trait(&item, true).unwrap_err()
        },
        items: &[(
            "use absolute path: 'crate :: m :: Bound'",
            "write `crate::m::Bound` with an absolute path (e.g. `::my_crate::m::Bound`), or \
             allow it with `LeakerConfig::allowed_paths`",
        )],
    },
    Row {
        kind: "a type with an elided lifetime",
        error: || {
            let item: ItemStruct = parse_quote!(
                struct S {
                    a: MyType<'_>,
                }
            );
            Leaker::new().intern_struct(&item, true).unwrap_err()
        },
        items: &[(
            "use absolute path: 'MyType < '_ >'",
            "write `MyType` with an absolute path (e.g. `::my_crate::MyType`), or allow it with \
             `LeakerConfig::allowed_paths`",
        )],
    },
    Row {
        kind: "the type of a const parameter",
        error: || {
            let item: ItemStruct = parse_quote!(
                struct S<const N: Idx>(u8);
            );
            Leaker::new().intern_struct(&item, true).unwrap_err()
        },
        items: &[(
            "use absolute path: 'Idx'",
            "write `Idx` with an absolute path (e.g. `::my_crate::Idx`), or allow it with \
             `LeakerConfig::allowed_paths`",
        )],
    },
    Row {
        kind: "a trait object that takes its lifetime from the reference around it",
        error: || {
            let item: ItemTrait = parse_quote!(
                trait Tr {
                    fn f(&self) -> &dyn RelTr;
                }
            );
            allowing().intern_trait(&item, true).unwrap_err()
        },
        items: &[(
            "use absolute path: 'RelTr'",
            "write `RelTr` with an absolute path (e.g. `::my_crate::RelTr`), allow it with \
             `LeakerConfig::allowed_paths`, or name the lifetime of the trait object (e.g. \
             `&'a dyn Trait` or `dyn Trait + 'a`)",
        )],
    },
    Row {
        kind: "a type that names `Self`, where `Self` is denied",
        error: || {
            let item: ItemTrait = parse_quote!(
                trait Tr {
                    fn f(&self) -> Rel<Self>;
                }
            );
            allowing().intern_trait(&item, false).unwrap_err()
        },
        items: &[(
            "use absolute path: 'Rel < Self >'",
            "write `Rel` with an absolute path (e.g. `::my_crate::Rel`), allow it with \
             `LeakerConfig::allowed_paths`, or analyze the item with `allow_self = true`",
        )],
    },
    // ── `Self` ───────────────────────────────────────────────────────────────────────────────
    Row {
        kind: "`Self` where no scope declares it",
        error: || {
            allowing()
                .enter_generics(&Generics::default(), |view| {
                    view.visit_type(&parse_quote!(Rel<Self>))
                })
                .unwrap_err()
        },
        items: &[(
            "`Self` has no meaning here",
            "write the type instead of `Self`, or enter a scope that declares `Self` around it \
             (`Leaker::enter_type`, `enter_impl` or `enter_trait`)",
        )],
    },
    Row {
        kind: "`Self::A` in an impl of a trait",
        error: || {
            let item: ItemImpl = parse_quote!(
                impl ::p::Tr for u8 {
                    type A = u8;
                    fn f(&self, x: Wrap<Self::A>) {}
                }
            );
            allowing().intern_impl(&item, true).unwrap_err()
        },
        items: &[(
            "write `<Self as ::p::Tr>::A` instead of `Self::A` in `Wrap<Self::A>`: the copy \
             cannot tell which trait `A` comes from",
            "replace `Self::A` with `<Self as ::p::Tr>::A`",
        )],
    },
    Row {
        kind: "`Self::A` in an impl of a trait written relative",
        error: || {
            let item: ItemImpl = parse_quote!(
                impl Tr for S {
                    type A = u8;
                    fn f(&self, x: Wrap<Self::A>) {}
                }
            );
            allowing().intern_impl(&item, true).unwrap_err()
        },
        items: &[
            (
                "use absolute path: 'Tr'",
                "write `Tr` with an absolute path (e.g. `::my_crate::Tr`), or allow it with \
                 `LeakerConfig::allowed_paths`",
            ),
            (
                "write `<Self as Tr>::A` instead of `Self::A` in `Wrap<Self::A>`: the copy \
                 cannot tell which trait `A` comes from",
                "replace `Self::A` with `<Self as ::my_crate::Tr>::A`, with `Tr` written as an \
                 absolute path",
            ),
        ],
    },
    Row {
        kind: "`Self::A` in a struct",
        error: || {
            let item: ItemStruct = parse_quote!(
                struct S<T>(T, Wrap<Self::A>);
            );
            allowing().intern_struct(&item, true).unwrap_err()
        },
        items: &[(
            "write `<Self as SomeTrait>::A` instead of `Self::A` in `Wrap<Self::A>`: the copy \
             cannot tell which trait `A` comes from",
            "replace `Self::A` with `<Self as SomeTrait>::A`, naming the trait that defines `A` \
             with its absolute path",
        )],
    },
    Row {
        kind: "`Self::A` in an inherent impl",
        error: || {
            let item: ItemImpl = parse_quote!(
                impl S {
                    fn f(&self, x: Wrap<Self::A>) {}
                }
            );
            allowing().intern_impl(&item, true).unwrap_err()
        },
        items: &[(
            "write `<Self as SomeTrait>::A` instead of `Self::A` in `Wrap<Self::A>`: the copy \
             cannot tell which trait `A` comes from",
            "replace `Self::A` with `<Self as SomeTrait>::A`, naming the trait that defines `A` \
             with its absolute path",
        )],
    },
    Row {
        kind: "`Self::A` in a bound of the generics of an entry",
        error: || {
            let item: ItemStruct = parse_quote!(
                struct S<T: ::p::Tr<Self::A>>(Wrap<T>);
            );
            allowing().intern_struct(&item, true).unwrap_err()
        },
        items: &[
            (
                "write `<Self as SomeTrait>::A` instead of `Self::A` in a bound of the generics \
                 of `Wrap<T>`: the copy cannot tell which trait `A` comes from",
                "replace `Self::A` with `<Self as SomeTrait>::A`, naming the trait that defines \
                 `A` with its absolute path",
            ),
            // The bound is an error of its own: rustc cannot prove it while it checks the bounds.
            (
                "`Self :: A` is in a bound or where predicate, and a bound of its generics names \
                 `Self :: A`",
                "write the bound without `Self::A`, or use `Self::A` outside the bounds and \
                 where predicates",
            ),
        ],
    },
    Row {
        kind: "`Self` in the default of a parameter",
        error: || {
            let item: ItemTrait = parse_quote!(
                trait Tr<T: Default = Wrap<Self>> {}
            );
            allowing().intern_trait(&item, true).unwrap_err()
        },
        items: &[(
            "`Wrap < Self >` is in the default of a parameter, and names `Self`",
            "write the default without `Self`: name the type instead of `Self` in `Wrap<Self>`",
        )],
    },
    Row {
        kind: "a bound of an outer trait that names its `Self`",
        error: || {
            allowing()
                .enter_trait(&id("Tr"), &parse_quote!(<T: AsRef<Self>>), true, |view| {
                    view.visit_item(&parse_quote!(
                        impl ::p::W for X {
                            fn f(&self, x: Rel<T>) {}
                        }
                    ))
                })
                .unwrap_err()
        },
        items: &[(
            "the bound of `T` mentions a `Self` which has no name here",
            "write the bound of `T` without `Self`, or write `Rel<T>` with an absolute path, so \
             that it is not leaked",
        )],
    },
    Row {
        kind: "a bound of an outer type that names an associated item of its `Self`",
        error: || {
            allowing()
                .enter_type(&id("S"), &parse_quote!(<T: AsRef<Self::X>>), true, |view| {
                    view.visit_item(&parse_quote!(
                        impl ::p::W for X {
                            fn f(&self, x: Rel<T>) {}
                        }
                    ))
                })
                .unwrap_err()
        },
        items: &[(
            "the bound of `T` mentions `Self :: X`, which has no name here",
            "write the bound of `T` without `Self::X`, or write `Rel<T>` with an absolute path, \
             so that it is not leaked",
        )],
    },
    Row {
        kind: "a generic whose bound mentions a `Self` that is denied",
        error: || {
            let item: ItemTrait = parse_quote!(
                trait Tr {
                    fn f<T: ::core::cmp::PartialEq<Self>>(&self, x: Wrap<T>);
                }
            );
            allowing().intern_trait(&item, false).unwrap_err()
        },
        items: &[(
            "`Wrap < T >` has a generic whose bound mentions `Self`, which cannot be leaked",
            "write the bound of the generics of `Wrap<T>` without `Self`, or analyze the item \
             with `allow_self = true`",
        )],
    },
    Row {
        kind: "the self type of an impl whose bound names `Self`",
        error: || {
            let item: ItemImpl = parse_quote!(
                impl<T> S<T> where T: AsRef<Option<Self>> {}
            );
            allowing().intern_impl(&item, true).unwrap_err()
        },
        items: &[
            (
                "`S < T >` is in the self type of the impl, and a bound of its generics names \
                 `Self`",
                "write `S<T>` with an absolute path, or the bound of its generics without `Self`",
            ),
            (
                "`Self` is in a bound or where predicate, and a bound of its generics names \
                 `Self`, which is a leaked type",
                "write the bound without `Self`, or use `Self` outside the bounds and where \
                 predicates",
            ),
        ],
    },
    // ── bounds that name leaked types ────────────────────────────────────────────────────────
    Row {
        kind: "a leaked type in a predicate whose generic has a leaked bound",
        error: || {
            let item: ItemTrait = parse_quote!(
                trait Tr<T: Fn(Rel)>
                where
                    Wrap<T>: Sized,
                {
                }
            );
            allowing().intern_trait(&item, true).unwrap_err()
        },
        items: &[(
            "`Wrap < T >` is in a bound or where predicate, and a bound of its generics names a \
             leaked type",
            "write the leaked type of the bound with an absolute path, or use `Wrap<T>` outside \
             the bounds and where predicates",
        )],
    },
    Row {
        kind: "a bound that names an associated type of `Self`",
        error: || {
            let item: ItemTrait = parse_quote!(
                trait Tr {
                    type A0;
                    fn m<X>(&self)
                    where
                        X: Fn(&Self::A0) -> u8,
                        Rel<X>: Default;
                }
            );
            allowing().intern_trait(&item, true).unwrap_err()
        },
        items: &[(
            "`Rel < X >` is in a bound or where predicate, and a bound of its generics names \
             `Self :: A0`",
            "write the bound without `Self::A0`, or use `Rel<X>` outside the bounds and where \
             predicates",
        )],
    },
    Row {
        kind: "a type that names `Self` in a predicate of its trait",
        error: || {
            let item: ItemTrait = parse_quote!(
                trait Tr
                where
                    Rel<Self>: Sized,
                {
                }
            );
            allowing().intern_trait(&item, true).unwrap_err()
        },
        items: &[(
            "`Rel < Self >` names `Self` in a bound or where predicate of its trait",
            "write the predicate of the trait without `Rel<Self>`, or use `Rel<Self>` outside \
             the predicates of the trait (in a method, a field or a signature)",
        )],
    },
    // ── lifetimes ────────────────────────────────────────────────────────────────────────────
    Row {
        kind: "a lifetime of a method that the copy would leave unconstrained",
        error: || {
            let item: ItemTrait = parse_quote!(
                trait Tr {
                    fn f<'b>(&self, x: Wrap<'b>) -> &'b u8;
                    fn g<'c>(&self, x: &'c dyn RelTrait) -> &'c u8;
                }
            );
            allowing().intern_trait(&item, true).unwrap_err()
        },
        items: &[
            (
                "`Wrap < 'b >` leaves the lifetime `'b` of the method unconstrained in the copy \
                 (E0195)",
                "keep `'b` in another argument of the method (for example `&'b self`), or write \
                 `Wrap<'b>` with an absolute path",
            ),
            (
                "`RelTrait` leaves the lifetime `'c` of the method unconstrained in the copy \
                 (E0195)",
                "keep `'c` in another argument of the method (for example `&'c self`), write \
                 `RelTrait` with an absolute path, or write the bound of the trait object \
                 (`dyn Trait + 'c`)",
            ),
        ],
    },
    Row {
        kind: "a `for<..>` lifetime of an output that no input keeps",
        error: || {
            let item: ItemTrait = parse_quote!(
                trait Tr {
                    fn f<F: for<'x> Fn(Rel<'x>) -> &'x u8>(&self, f: F);
                }
            );
            allowing().intern_trait(&item, true).unwrap_err()
        },
        items: &[(
            "use absolute path: 'Rel < 'x >'",
            "keep an input that names `'x` as written (for example `&'x u8`), or write \
             `Rel<'x>` with an absolute path",
        )],
    },
    Row {
        kind: "an elided output lifetime of a method with no receiver",
        error: || {
            let item: ItemTrait = parse_quote!(
                trait Tr<'a, T: 'a> {
                    fn g(x: &Rel<T>) -> &u8;
                }
            );
            allowing().intern_trait(&item, true).unwrap_err()
        },
        items: &[(
            "use absolute path: 'Rel < T >'",
            "name the lifetime of the inputs and of the output, give the method a receiver \
             (`&self`), or write `Rel<T>` with an absolute path",
        )],
    },
    Row {
        kind: "an elided output lifetime of `Fn(..)` in a bound",
        error: || {
            let item: ItemTrait = parse_quote!(
                trait Tr<'a, T: 'a, F: Fn(&Rel<T>) -> &u8> {}
            );
            allowing().intern_trait(&item, true).unwrap_err()
        },
        items: &[(
            "use absolute path: 'Rel < T >'",
            "name the lifetime of the inputs and of the output, or write `Rel<T>` with an \
             absolute path",
        )],
    },
    Row {
        kind: "an elided output lifetime that a path hides",
        error: || {
            let item: ItemTrait = parse_quote!(
                trait Tr<'a, T: 'a, F: Fn(::p::WRef<u8>, Rel<T>) -> &u8> {}
            );
            allowing().intern_trait(&item, true).unwrap_err()
        },
        items: &[(
            "use absolute path: 'Rel < T >'",
            "write the lifetime that the path hides (`Ref<'x, u8>`), and name it in the output, \
             or write `Rel<T>` with an absolute path",
        )],
    },
    Row {
        kind: "an elided output lifetime whose holder cannot be leaked",
        error: || {
            let item: ItemTrait = parse_quote!(
                trait Tr<'a, T: 'a> {
                    fn f(&self, x: &dyn Fn(&Rel<T>) -> &u8);
                }
            );
            allowing().intern_trait(&item, true).unwrap_err()
        },
        items: &[(
            "use absolute path: 'Rel < T >'",
            "name the lifetime of the inputs and of the output, or write `Rel<T>` with an \
             absolute path",
        )],
    },
    Row {
        kind: "an elided output lifetime, where `Self` has lifetimes and is denied",
        error: || {
            let item: ItemStruct = parse_quote!(
                pub struct S<'a>(fn(::core::slice::Iter<u8>, Self) -> &u8, &'a u8);
            );
            allowing().intern_struct(&item, false).unwrap_err()
        },
        items: &[(
            "use absolute path: 'Self'",
            "write the lifetime that the path hides (`Ref<'x, u8>`), and name it in the output, \
             or analyze the item with `allow_self = true`",
        )],
    },
];

#[test]
fn every_kind_of_error_has_a_help() {
    for row in ROWS {
        let err = (row.error)();
        let items: Vec<ErrorItem> = err.items().collect();
        let got: Vec<(String, String)> = items
            .iter()
            .map(|item| {
                (
                    item.message.clone(),
                    item.help_text().unwrap_or_default().to_string(),
                )
            })
            .collect();
        let expected: Vec<(String, String)> = row
            .items
            .iter()
            .map(|(message, help)| (message.to_string(), help.to_string()))
            .collect();
        assert_eq!(got, expected, "{}", row.kind);
        for item in &items {
            let (_, help) = item.help.as_ref().expect(row.kind);
            // The help says what to write, the notes why: it is none of them, and not the headline.
            assert!(!help.is_empty(), "{}", row.kind);
            assert_ne!(*help, item.message, "{}", row.kind);
            assert!(
                item.notes.iter().all(|(_, note)| note != help),
                "{}: {}",
                row.kind,
                help
            );
            // It starts like a sentence of rustc's help: in lower case, and with no full stop.
            assert!(
                help.starts_with(|c: char| c.is_lowercase()) && !help.ends_with('.'),
                "{}: {}",
                row.kind,
                help
            );
        }
        // The first error's help is what the accessors of the error tell.
        assert_eq!(err.help().as_deref(), items[0].help_text(), "{}", row.kind);
    }
}

#[test]
fn a_help_is_rendered_after_the_notes() {
    for row in ROWS {
        let err = (row.error)();
        let items: Vec<ErrorItem> = err.items().collect();
        let errors: Vec<syn::Error> = syn::Error::from(err.clone()).into_iter().collect();
        assert_eq!(errors.len(), items.len(), "{}", row.kind);
        for (error, item) in errors.iter().zip(&items) {
            let rendered = error.to_string();
            let help = item.help_text().unwrap();
            // The headline, a blank line, the notes (`= note: ..`), the help (`= help: ..`).
            let mut expected = format!("{}\n", item.message);
            expected += "\n";
            for (_, note) in &item.notes {
                expected += &format!("  = note: {}\n", note);
            }
            expected += &format!("  = help: {}\n", help);
            assert_eq!(rendered, expected, "{}", row.kind);
        }
        // And `ToTokens` renders the same errors, as `compile_error!`s.
        let tokens = {
            use template_quote::quote;
            quote!(#err).to_string()
        };
        assert_eq!(
            tokens.matches("compile_error").count(),
            items.len(),
            "{}: {}",
            row.kind,
            tokens
        );
        for item in &items {
            assert!(
                tokens.contains(&format!("= help: {}", item.help_text().unwrap())),
                "{}: {}",
                row.kind,
                tokens
            );
        }
    }
}

/// The notes keep their words when a help is added: why, not what to write.
#[test]
fn a_note_that_is_not_about_the_fix_has_no_advice() {
    for row in ROWS {
        let err = (row.error)();
        for item in err.items() {
            for (_, note) in &item.notes {
                assert!(
                    !note.contains("Write the")
                        && !note.contains("name the lifetime of the inputs"),
                    "{}: {}",
                    row.kind,
                    note
                );
            }
        }
    }
}

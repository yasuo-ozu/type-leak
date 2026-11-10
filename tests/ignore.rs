//! `LeakerResult::ignore()`: the copy writes the whole part of an error as written, or an entry
//! that the error is about is removed.
//!
//! - A table of the errors of every kind, with what `ignore()` of each leaves: the errors, the
//!   copy (`E<ix>` for an entry), the live entries and the impls. An entry that is used elsewhere stays one there.
//! - The errors that share a part go together, as do the errors inside the part. An error that is
//!   not one of the errors, or that `allow()` dropped, changes nothing.
//! - An entry that is removed keeps its `EntryId`, its Repeater key and its bucket type for the
//!   others; it has no impl, is not in `iter()`, and its type is written as written.
//! - The tokens of the result: `ignore()` before the trip and after it give the same result, and
//!   the parser refuses what is not there.

mod common;

use quote::ToTokens;
use syn::visit_mut::VisitMut;
use syn::*;
use type_leak::*;

/// Takes the trailing comma off the fields and the where predicates, which rustfmt adds to an item
/// in a macro: the copies are compared without it.
struct NoTrailing;

impl VisitMut for NoTrailing {
    fn visit_fields_named_mut(&mut self, fields: &mut FieldsNamed) {
        fields.named.pop_punct();
        syn::visit_mut::visit_fields_named_mut(self, fields);
    }
    fn visit_fields_unnamed_mut(&mut self, fields: &mut FieldsUnnamed) {
        fields.unnamed.pop_punct();
        syn::visit_mut::visit_fields_unnamed_mut(self, fields);
    }
    fn visit_where_clause_mut(&mut self, clause: &mut WhereClause) {
        clause.predicates.pop_punct();
        syn::visit_mut::visit_where_clause_mut(self, clause);
    }
}

/// `item` without the trailing commas that rustfmt may have added.
fn plain(mut item: Item) -> Item {
    NoTrailing.visit_item_mut(&mut item);
    item
}

/// The copy of `item` with `E<ix>` for an entry.
fn copy_of(referrer: &Referrer, top: ContextId, item: &Item) -> Item {
    let mut copy = item.clone();
    common::visitor(referrer, top, |_, ix, _| {
        parse_str(&format!("E{}", ix.index())).unwrap()
    })
    .visit_item_mut(&mut copy);
    plain(copy)
}

fn intern<A: Allow>(leaker: &mut Leaker<A>, item: &Item) -> ContextId {
    match item {
        Item::Struct(item) => leaker.intern_struct(item),
        Item::Trait(item) => leaker.intern_trait(item),
        Item::Impl(item) => leaker.intern_impl(item),
        Item::Mod(item) => leaker.intern_mod(item),
        _ => panic!("no such item"),
    }
}

/// The result of `item` and the context of its scope.
fn analyze(item: &Item) -> (LeakerResult, ContextId) {
    let mut leaker = Leaker::new();
    let top = intern(&mut leaker, item);
    (leaker.result(), top)
}

/// The reason and the cause of each error.
fn errors_of(result: &LeakerResult) -> Vec<(Reason, Cause)> {
    result
        .errors()
        .iter()
        .map(|error| (error.reason, error.cause.clone()))
        .collect()
}

/// The causes of the errors.
fn causes(result: &LeakerResult) -> Vec<Cause> {
    result.errors().iter().map(|e| e.cause.clone()).collect()
}

/// The `EntryId` and the type of each entry that is left.
fn live(result: &LeakerResult) -> Vec<(usize, Type)> {
    let referrer = result.clone().into_referrer();
    let found: Vec<(usize, Type)> = referrer
        .iter()
        .map(|(ix, ty, _)| (ix.index(), ty.clone()))
        .collect();
    found
}

/// The Repeater impls of the result, for the bucket types of the id 1.
fn impls(result: &LeakerResult) -> Vec<RepeaterImpl> {
    let repeater: Path = parse_quote!(Repeater);
    let param: Ident = parse_quote!(__S);
    result
        .repeater_impls(&Projection::new(
            repeater.clone(),
            1,
            ImplSelf::Bucket {
                param: param.clone(),
            },
        ))
        .unwrap()
}

fn copy_after(result: &LeakerResult, top: ContextId, item: &Item) -> Item {
    copy_of(&result.clone().into_referrer(), top, item)
}

fn tokens(result: &LeakerResult) -> String {
    result.to_token_stream().to_string()
}

fn read(tokens: proc_macro2::TokenStream) -> Result<LeakerResult> {
    parse2(tokens)
}

/// The result as the tokens of another macro read it back.
fn trip(result: &LeakerResult) -> LeakerResult {
    read(result.to_token_stream()).unwrap()
}

/// `(reason, a type)`.
fn rt(reason: Reason, ty: Type) -> (Reason, Cause) {
    (reason, Cause::Type(ty))
}

/// `(reason, a trait)`.
fn rtr(reason: Reason, path: Path) -> (Reason, Cause) {
    (reason, Cause::Trait(path))
}

/// What `ignore()` of one error gives.
struct Ignored {
    /// The error, by its index.
    error: usize,
    left: Vec<(Reason, Cause)>,
    /// The entries that are left, by `EntryId`.
    live: Vec<(usize, Type)>,
    /// The number of Repeater impls.
    impls: usize,
    copy: Item,
}

/// An item, the errors of its analysis, and what `ignore()` of them leaves.
struct Row {
    name: &'static str,
    item: Item,
    errors: Vec<(Reason, Cause)>,
    live: Vec<(usize, Type)>,
    /// The copy as it is: `allow()` of every error changes no entry and no part.
    copy: Item,
    ignored: Vec<Ignored>,
}

fn rows() -> Vec<Row> {
    use Reason::*;
    vec![
        // A flat conflict: `MyType` is written as it is, `Inner<'_>` has an error of its own, and
        // `Rel` is an entry that the other field uses too.
        Row {
            name: "a flat conflict",
            item: parse_quote!(
                struct S<'a>(MyType<'_, Inner<'_>, Rel>, Rel, &'a u8);
            ),
            errors: vec![
                rt(RelativePath, parse_quote!(MyType<'_, Inner<'_>, Rel>)),
                rt(RelativePath, parse_quote!(Inner<'_>)),
            ],
            live: vec![(0, parse_quote!(Rel)), (1, parse_quote!(&'a u8))],
            copy: parse_quote!(
                struct S<'a>(MyType<'_, Inner<'_>, E0>, E0, E1);
            ),
            ignored: vec![
                // The whole part is as written, with the error inside it gone, and `Rel` is an
                // entry in the other field.
                Ignored {
                    error: 0,
                    left: vec![],
                    live: vec![(0, parse_quote!(Rel)), (1, parse_quote!(&'a u8))],
                    impls: 2,
                    copy: parse_quote!(
                        struct S<'a>(MyType<'_, Inner<'_>, Rel>, E0, E1);
                    ),
                },
                // The error inside it only: `MyType` stays an error, and its part is as before.
                Ignored {
                    error: 1,
                    left: vec![rt(RelativePath, parse_quote!(MyType<'_, Inner<'_>, Rel>))],
                    live: vec![(0, parse_quote!(Rel)), (1, parse_quote!(&'a u8))],
                    impls: 2,
                    copy: parse_quote!(
                        struct S<'a>(MyType<'_, Inner<'_>, E0>, E0, E1);
                    ),
                },
            ],
        },
        // The part has an anchor: the trait object is written as it is inside `::p::W` only.
        Row {
            name: "a part with an anchor",
            item: parse_quote!(
                struct S(::p::W<'_, dyn RelTr<Rel2>, Rel>, ::p::B<dyn RelTr<Rel2>>);
            ),
            errors: vec![rtr(RelativePath, parse_quote!(RelTr<Rel2>))],
            live: vec![
                (0, parse_quote!(::p::B<dyn RelTr<Rel2>>)),
                (1, parse_quote!(Rel)),
                (2, parse_quote!(Rel2)),
            ],
            copy: parse_quote!(
                struct S(::p::W<'_, dyn RelTr<E2>, E1>, E0);
            ),
            ignored: vec![Ignored {
                error: 0,
                left: vec![],
                live: vec![
                    (0, parse_quote!(::p::B<dyn RelTr<Rel2>>)),
                    (1, parse_quote!(Rel)),
                    (2, parse_quote!(Rel2)),
                ],
                impls: 3,
                copy: parse_quote!(
                    struct S(::p::W<'_, dyn RelTr<Rel2>, E1>, E0);
                ),
            }],
        },
        // A part that is whole already (a trait bound): nothing changes but the error.
        Row {
            name: "a trait bound",
            item: parse_quote!(
                trait Tr<T: RelTr<Rel>> {
                    fn f(&self) -> Rel;
                }
            ),
            errors: vec![rtr(RelativePath, parse_quote!(RelTr<Rel>))],
            live: vec![(0, parse_quote!(Rel))],
            copy: parse_quote!(
                trait Tr<T: RelTr<Rel>> {
                    fn f(&self) -> E0;
                }
            ),
            ignored: vec![Ignored {
                error: 0,
                left: vec![],
                live: vec![(0, parse_quote!(Rel))],
                impls: 1,
                copy: parse_quote!(
                    trait Tr<T: RelTr<Rel>> {
                        fn f(&self) -> E0;
                    }
                ),
            }],
        },
        Row {
            name: "a name that may be a const",
            item: parse_quote!(
                struct S {
                    a: ::p::W<Self::A, LEN>,
                    b: Rel,
                }
            ),
            errors: vec![rt(MaybeConst, parse_quote!(LEN))],
            live: vec![(0, parse_quote!(Rel))],
            copy: parse_quote!(
                struct S {
                    a: ::p::W<Self::A, LEN>,
                    b: E0,
                }
            ),
            ignored: vec![Ignored {
                error: 0,
                left: vec![],
                live: vec![(0, parse_quote!(Rel))],
                impls: 1,
                copy: parse_quote!(
                    struct S {
                        a: ::p::W<Self::A, LEN>,
                        b: E0,
                    }
                ),
            }],
        },
        // A part that a signature made: it is left as written where the copy would leave the
        // lifetime of the method unconstrained, `f`, and `g` has the entry.
        Row {
            name: "a part made by a signature",
            item: parse_quote!(
                trait Tr {
                    fn f<'b>(&self, x: Wrap<'b, Rel>) -> &'b u8;
                    fn g<'b>(&self, x: Wrap<'b, Rel>);
                }
            ),
            errors: vec![rt(UnconstrainedLifetime, parse_quote!(Wrap<'b, Rel>))],
            live: vec![
                (0, parse_quote!(Wrap<'b, Rel>)),
                (1, parse_quote!(Rel)),
                (2, parse_quote!(u8)),
            ],
            copy: parse_quote!(
                trait Tr {
                    fn f<'b>(&self, x: Wrap<'b, E1>) -> &'b E2;
                    fn g<'b>(&self, x: E0);
                }
            ),
            ignored: vec![Ignored {
                error: 0,
                left: vec![],
                live: vec![
                    (0, parse_quote!(Wrap<'b, Rel>)),
                    (1, parse_quote!(Rel)),
                    (2, parse_quote!(u8)),
                ],
                impls: 3,
                copy: parse_quote!(
                    trait Tr {
                        fn f<'b>(&self, x: Wrap<'b, Rel>) -> &'b E2;
                        fn g<'b>(&self, x: E0);
                    }
                ),
            }],
        },
        Row {
            name: "a projection of Self",
            item: parse_quote!(
                impl ::p::Tr for u8 {
                    type A = u8;
                    fn f(&self, x: Wrap<Self::A, Rel>, y: Rel) {}
                }
            ),
            errors: vec![rt(SelfProjection, parse_quote!(Wrap<Self::A, Rel>))],
            live: vec![(0, parse_quote!(Rel)), (1, parse_quote!(u8))],
            copy: parse_quote!(
                impl ::p::Tr for E1 {
                    type A = E1;
                    fn f(&self, x: Wrap<Self::A, E0>, y: E0) {}
                }
            ),
            ignored: vec![Ignored {
                error: 0,
                left: vec![],
                live: vec![(0, parse_quote!(Rel)), (1, parse_quote!(u8))],
                impls: 2,
                copy: parse_quote!(
                    impl ::p::Tr for E1 {
                        type A = E1;
                        fn f(&self, x: Wrap<Self::A, Rel>, y: E0) {}
                    }
                ),
            }],
        },
        Row {
            name: "Self in a default",
            item: parse_quote!(
                trait Tr<T, U = Rel<Self, Other>> {
                    fn f(&self, x: Rel2, y: Other);
                }
            ),
            errors: vec![rt(SelfInDefault, parse_quote!(Rel<Self, Other>))],
            live: vec![(0, parse_quote!(Other)), (1, parse_quote!(Rel2))],
            copy: parse_quote!(
                trait Tr<T, U = Rel<Self, E0>> {
                    fn f(&self, x: E1, y: E0);
                }
            ),
            ignored: vec![Ignored {
                error: 0,
                left: vec![],
                live: vec![(0, parse_quote!(Other)), (1, parse_quote!(Rel2))],
                impls: 2,
                copy: parse_quote!(
                    trait Tr<T, U = Rel<Self, Other>> {
                        fn f(&self, x: E1, y: E0);
                    }
                ),
            }],
        },
        Row {
            name: "Self in a predicate of a trait",
            item: parse_quote!(
                trait Tr: ::p::Sup<::p::m!(Self)> {
                    fn f(&self, x: Rel2);
                }
            ),
            errors: vec![rt(SelfInTraitPredicate, parse_quote!(::p::m!(Self)))],
            live: vec![(0, parse_quote!(Rel2))],
            copy: parse_quote!(
                trait Tr: ::p::Sup<::p::m!(Self)> {
                    fn f(&self, x: E0);
                }
            ),
            ignored: vec![Ignored {
                error: 0,
                left: vec![],
                live: vec![(0, parse_quote!(Rel2))],
                impls: 1,
                copy: parse_quote!(
                    trait Tr: ::p::Sup<::p::m!(Self)> {
                        fn f(&self, x: E0);
                    }
                ),
            }],
        },
        Row {
            name: "an unprovable bound",
            item: parse_quote!(
                trait Tr<T: ::p::BT1<String>> {
                    type A0;
                    fn m<X>(&self, r: Rel)
                    where
                        X: ::core::convert::Into<Self::A0>;
                }
            ),
            errors: vec![rt(UnprovableBound, parse_quote!(Self::A0))],
            live: vec![(0, parse_quote!(Rel)), (1, parse_quote!(String))],
            copy: parse_quote!(
                trait Tr<T: ::p::BT1<E1>> {
                    type A0;
                    fn m<X>(&self, r: E0)
                    where
                        X: ::core::convert::Into<Self::A0>;
                }
            ),
            ignored: vec![Ignored {
                error: 0,
                left: vec![],
                live: vec![(0, parse_quote!(Rel)), (1, parse_quote!(String))],
                impls: 2,
                copy: parse_quote!(
                    trait Tr<T: ::p::BT1<E1>> {
                        type A0;
                        fn m<X>(&self, r: E0)
                        where
                            X: ::core::convert::Into<Self::A0>;
                    }
                ),
            }],
        },
        // The self type of an impl, and the `Self` of the bound: two parts.
        Row {
            name: "the self type of an impl",
            item: parse_quote!(
                impl<T> S<T, Other>
                where
                    T: ::core::convert::AsRef<::core::option::Option<Self>>,
                {
                    fn g(&self, y: Other) {}
                }
            ),
            errors: vec![
                rt(SelfInImplSelfType, parse_quote!(S<T, Other>)),
                rt(UnprovableBound, parse_quote!(Self)),
            ],
            live: vec![(0, parse_quote!(Other))],
            copy: parse_quote!(
                impl<T> S<T, E0>
                where
                    T: ::core::convert::AsRef<::core::option::Option<Self>>,
                {
                    fn g(&self, y: E0) {}
                }
            ),
            ignored: vec![
                Ignored {
                    error: 0,
                    left: vec![rt(UnprovableBound, parse_quote!(Self))],
                    live: vec![(0, parse_quote!(Other))],
                    impls: 1,
                    copy: parse_quote!(
                        impl<T> S<T, Other>
                        where
                            T: ::core::convert::AsRef<::core::option::Option<Self>>,
                        {
                            fn g(&self, y: E0) {}
                        }
                    ),
                },
                Ignored {
                    error: 1,
                    left: vec![rt(SelfInImplSelfType, parse_quote!(S<T, Other>))],
                    live: vec![(0, parse_quote!(Other))],
                    impls: 1,
                    copy: parse_quote!(
                        impl<T> S<T, E0>
                        where
                            T: ::core::convert::AsRef<::core::option::Option<Self>>,
                        {
                            fn g(&self, y: E0) {}
                        }
                    ),
                },
            ],
        },
        // The error is about an entry, found twice: it is removed, and its type is written as
        // written wherever it would have been replaced.
        Row {
            name: "an entry for any type",
            item: parse_quote!(
                mod m {
                    struct N<T: ::core::convert::AsRef<Self>>(
                        Rel2<Self, T>,
                        ::core::option::Option<Rel2<Self, T>>,
                    );
                }
            ),
            errors: vec![
                rt(NestedSelfInBounds, parse_quote!(Rel2<Self, T>)),
                rt(NestedSelfInBounds, parse_quote!(Rel2<Self, T>)),
            ],
            live: vec![(0, parse_quote!(Rel2<Self, T>))],
            copy: parse_quote!(
                mod m {
                    struct N<T: ::core::convert::AsRef<Self>>(E0, ::core::option::Option<E0>);
                }
            ),
            ignored: vec![
                Ignored {
                    error: 0,
                    left: vec![],
                    live: vec![],
                    impls: 0,
                    copy: parse_quote!(
                        mod m {
                            struct N<T: ::core::convert::AsRef<Self>>(
                                Rel2<Self, T>,
                                ::core::option::Option<Rel2<Self, T>>,
                            );
                        }
                    ),
                },
                // Both errors are about the one entry.
                Ignored {
                    error: 1,
                    left: vec![],
                    live: vec![],
                    impls: 0,
                    copy: parse_quote!(
                        mod m {
                            struct N<T: ::core::convert::AsRef<Self>>(
                                Rel2<Self, T>,
                                ::core::option::Option<Rel2<Self, T>>,
                            );
                        }
                    ),
                },
            ],
        },
    ]
}

#[test]
fn ignore_of_each_error_leaves_the_errors_the_copy_the_entries_and_the_impls_of_the_table() {
    for row in rows() {
        let (result, top) = analyze(&row.item);
        assert_eq!(errors_of(&result), row.errors, "{}", row.name);
        assert_eq!(live(&result), row.live, "{}", row.name);
        assert_eq!(
            copy_after(&result, top, &row.item),
            plain(row.copy.clone()),
            "{}",
            row.name
        );
        // `allow()` of every error changes no entry and no part.
        let (allowed, _) = common::allow_every_cause(result.clone());
        assert!(allowed.errors().is_empty(), "{}", row.name);
        assert_eq!(
            copy_after(&allowed, top, &row.item),
            plain(row.copy.clone()),
            "{}",
            row.name
        );
        for ignored in &row.ignored {
            let name = format!("{}: ignore of the error {}", row.name, ignored.error);
            let error = result.errors()[ignored.error].clone();
            let after = result.clone().ignore(&error);
            assert_eq!(errors_of(&after), ignored.left, "{}", name);
            assert_eq!(live(&after), ignored.live, "{}", name);
            assert_eq!(impls(&after).len(), ignored.impls, "{}", name);
            assert_eq!(
                copy_after(&after, top, &row.item),
                plain(ignored.copy.clone()),
                "{}",
                name
            );
        }
    }
}

#[test]
fn ignore_of_every_error_in_turn_leaves_none() {
    for row in rows() {
        let (mut result, top) = analyze(&row.item);
        while let Some(error) = result.errors().first().cloned() {
            let before = result.errors().len();
            result = result.ignore(&error);
            assert!(result.errors().len() < before, "{}", row.name);
        }
        assert!(result.errors().is_empty(), "{}", row.name);
        // The copy of the last error of the table where it is the only one, or the entry that all
        // the errors are about.
        if row.ignored.len() == 1 || row.name == "an entry for any type" {
            let last = row.ignored.last().unwrap();
            assert_eq!(
                copy_after(&result, top, &row.item),
                plain(last.copy.clone()),
                "{}",
                row.name
            );
        }
        // The result is a Referrer that `finish()` gives.
        assert!(result.finish().is_ok(), "{}", row.name);
    }
}

#[test]
fn an_error_that_is_none_of_the_errors_changes_nothing() {
    let (result, _) = analyze(&parse_quote!(
        struct S<'a>(MyType<'_, Rel>, Rel, &'a u8);
    ));
    // An error of another analysis, and a changed one.
    let (other, _) = analyze(&parse_quote!(
        struct S(Other<'_>);
    ));
    let mut changed = result.errors()[0].clone();
    changed.message.push('!');
    for error in [other.errors()[0].clone(), changed] {
        let same = result.clone().ignore(&error);
        assert_eq!(tokens(&same), tokens(&result));
    }
}

#[test]
fn ignore_twice_and_after_allow_changes_nothing() {
    let item: Item = parse_quote!(
        struct S<'a>(MyType<'_, Rel>, Rel, &'a u8);
    );
    let (result, top) = analyze(&item);
    let error = result.errors()[0].clone();
    let once = result.clone().ignore(&error);
    assert!(once.errors().is_empty());
    let twice = once.clone().ignore(&error);
    assert_eq!(tokens(&twice), tokens(&once));
    // `allow()` dropped it: nothing is made whole, nor removed.
    let (allowed, _) = common::allow_every_cause(result.clone());
    let after = allowed.clone().ignore(&error);
    assert_eq!(tokens(&after), tokens(&allowed));
    assert_eq!(
        copy_after(&after, top, &item),
        parse_quote!(
            struct S<'a>(MyType<'_, E0>, E0, E1);
        )
    );
    // And the other way: what `ignore()` dropped is gone for `allow()`.
    let left = once.allow(|_: &Cause, _: Place| true);
    assert!(left.errors().is_empty());
}

#[test]
fn ignore_drops_the_errors_inside_the_part_at_any_depth() {
    let item: Item = parse_quote!(
        struct S(Outer<'_, Middle<'_, Deep<'_>>>, Deep<'_>);
    );
    let (result, top) = analyze(&item);
    let outer: Type = parse_quote!(Outer<'_, Middle<'_, Deep<'_>>>);
    let middle: Type = parse_quote!(Middle<'_, Deep<'_>>);
    let deep: Type = parse_quote!(Deep<'_>);
    let cause = |ty: &Type| Cause::Type(ty.clone());
    assert_eq!(
        causes(&result),
        [cause(&outer), cause(&middle), cause(&deep), cause(&deep)]
    );
    // The outer error drops the two inside it. The `Deep<'_>` of the other field is an error of
    // its own, and its part is the same.
    let after = result.clone().ignore(&result.errors()[0].clone());
    assert_eq!(causes(&after), [cause(&deep)]);
    assert_eq!(
        copy_after(&after, top, &item),
        parse_quote!(
            struct S(Outer<'_, Middle<'_, Deep<'_>>>, Deep<'_>);
        )
    );
    // The middle one: the outer error stays, and the part of `Middle` is whole.
    let after = result.clone().ignore(&result.errors()[1].clone());
    assert_eq!(causes(&after), [cause(&outer), cause(&deep)]);
    // The deep one makes its part whole wherever it is, so both errors about it go together.
    let after = result.clone().ignore(&result.errors()[2].clone());
    assert_eq!(causes(&after), [cause(&outer), cause(&middle)]);
    let after = result.clone().ignore(&result.errors()[3].clone());
    assert_eq!(causes(&after), [cause(&outer), cause(&middle)]);
}

#[test]
fn errors_about_the_same_part_go_together() {
    // The same type is a conflict in the signature and in the body, which `check_values` reads.
    let item: ItemTrait = parse_quote!(
        trait Tr {
            fn f(&self, a: MyType<'_, Rel>) {
                let b: MyType<'_, Rel> = loop {};
            }
        }
    );
    let mut leaker = Leaker::new();
    leaker.check_values = true;
    let top = leaker.intern_trait(&item);
    let result = leaker.result();
    let my_type = Cause::Type(parse_quote!(MyType<'_, Rel>));
    let errors: Vec<&ErrorItem> = result
        .errors()
        .iter()
        .filter(|error| error.cause == my_type)
        .collect();
    assert_eq!(errors.len(), 2);
    assert!(!errors[0].place.in_value && errors[1].place.in_value);
    let after = result.clone().ignore(errors[1]);
    assert!(after.errors().iter().all(|error| error.cause != my_type));
    // The part is whole in the signature and in the body.
    let item = Item::Trait(item);
    let copy = copy_after(&after, top, &item);
    let expected: Item = parse_quote!(
        trait Tr {
            fn f(&self, a: MyType<'_, Rel>) {
                let b: MyType<'_, Rel> = loop {};
            }
        }
    );
    assert_eq!(copy, expected);
}

#[test]
fn independent_errors_are_ignored_in_either_order() {
    let item: Item = parse_quote!(
        impl<T> S<T, Other>
        where
            T: ::core::convert::AsRef<::core::option::Option<Self>>,
        {
            fn g(&self, y: Other) {}
        }
    );
    let (result, top) = analyze(&item);
    let (first, second) = (result.errors()[0].clone(), result.errors()[1].clone());
    let one = result.clone().ignore(&first).ignore(&second);
    let other = result.clone().ignore(&second).ignore(&first);
    assert!(one.errors().is_empty());
    assert_eq!(tokens(&one), tokens(&other));
    assert_eq!(copy_after(&one, top, &item), copy_after(&other, top, &item));
}

#[test]
fn an_error_that_is_about_no_part_is_only_dropped() {
    // `Self` where it has no meaning, a path in a value, and syntax that cannot be read.
    let mut leaker = Leaker::new();
    leaker.check_values = true;
    leaker.enter_generics(&Generics::default(), |view| {
        view.visit_type(&parse_quote!(Rel<Self>));
    });
    let item: ItemTrait = parse_quote!(
        trait Tr {
            fn f(&self) {
                crate::g();
                macro_rules! m {
                    () => {};
                }
            }
        }
    );
    leaker.intern_trait(&item);
    let result = leaker.result();
    assert_eq!(
        result.errors().iter().map(|e| e.reason).collect::<Vec<_>>(),
        [
            Reason::SelfWithoutMeaning,
            Reason::RelativePath,
            Reason::Unreadable
        ]
    );
    // The Referrer is the same whichever is ignored, and the error goes.
    let referrer =
        |result: &LeakerResult| result.clone().into_referrer().to_token_stream().to_string();
    for error in result.errors() {
        let after = result.clone().ignore(error);
        assert_eq!(after.errors().len(), 2);
        assert!(after.errors().iter().all(|e| e != error));
        assert_eq!(referrer(&after), referrer(&result));
    }
}

#[test]
fn an_error_about_an_entry_inside_a_part_stays() {
    // `Rel2<Self, T>` is an entry for any type, inside a part and beside it. The error of the
    // part drops the part's own errors, not the one about the entry, which may be used elsewhere.
    let item: Item = parse_quote!(
        mod m {
            struct N<T: ::core::convert::AsRef<Self>>(MyType<'_, Rel2<Self, T>>, Rel2<Self, T>);
        }
    );
    let (result, top) = analyze(&item);
    let reasons = |result: &LeakerResult| {
        result
            .errors()
            .iter()
            .map(|error| error.reason)
            .collect::<Vec<_>>()
    };
    assert_eq!(
        reasons(&result),
        [
            Reason::RelativePath,
            Reason::NestedSelfInBounds,
            Reason::NestedSelfInBounds
        ]
    );
    let after = result.clone().ignore(&result.errors()[0].clone());
    assert_eq!(
        reasons(&after),
        [Reason::NestedSelfInBounds, Reason::NestedSelfInBounds]
    );
    // The entry is still one: the part is as written, and the other field has the entry.
    let expected: Item = parse_quote!(
        mod m {
            struct N<T: ::core::convert::AsRef<Self>>(MyType<'_, Rel2<Self, T>>, E0);
        }
    );
    assert_eq!(copy_after(&after, top, &item), expected);
    // Ignoring it removes it for both.
    let both = after.clone().ignore(&after.errors()[0].clone());
    assert!(both.errors().is_empty());
    assert!(live(&both).is_empty());
}

#[test]
fn an_error_of_a_scope_ignored_inside_a_part_with_self_without_meaning_goes_with_it() {
    let mut leaker = Leaker::new();
    leaker.enter_generics(&Generics::default(), |view| {
        view.visit_type(&parse_quote!(MyType<'_, Self>));
        view.visit_type(&parse_quote!(Rel<Self>));
    });
    let result = leaker.result();
    assert_eq!(
        result.errors().iter().map(|e| e.reason).collect::<Vec<_>>(),
        [
            Reason::RelativePath,
            Reason::SelfWithoutMeaning,
            Reason::SelfWithoutMeaning
        ]
    );
    // The `Self` inside the part goes with it; the other one stays.
    let after = result.clone().ignore(&result.errors()[0].clone());
    assert_eq!(causes(&after), [Cause::Type(parse_quote!(Self))]);
    // The other mention is another error, with a part of none.
    let after = result.clone().ignore(&result.errors()[2].clone());
    assert_eq!(after.errors().len(), 2);
}

// ── an entry that is removed ─────────────────────────────────────────────────────────────────

/// A nested struct with an entry for any type, `Rel2<Self, T, Other>`, which holds `Other`, and
/// two entries beside it.
fn removable() -> (Item, LeakerResult, ContextId) {
    let item: Item = parse_quote!(
        mod m {
            struct N<T: ::core::convert::AsRef<Self>>(Rel2<Self, T, Other>, Other, Third);
        }
    );
    let (result, top) = analyze(&item);
    (item, result, top)
}

#[test]
fn a_removed_entry_keeps_its_place_and_the_others_keep_theirs() {
    let (item, result, top) = removable();
    assert_eq!(
        live(&result),
        [
            (0, parse_quote!(Rel2<Self, T, Other>)),
            (1, parse_quote!(Other)),
            (2, parse_quote!(Third))
        ]
    );
    let removed = result.clone().ignore(&result.errors()[0].clone());
    assert_eq!(
        live(&removed),
        [(1, parse_quote!(Other)), (2, parse_quote!(Third))]
    );
    let (before, after) = (
        result.clone().into_referrer(),
        removed.clone().into_referrer(),
    );
    // The others keep the primitive type of their impls, the KEY of their impls and their impls.
    let id = 1;
    let bucket = |referrer: &Referrer, n: usize| {
        common::bucket_self(referrer, id, common::entry_id_at(referrer, n))
    };
    assert_eq!(bucket(&before, 1), bucket(&after, 0));
    assert_eq!(bucket(&before, 2), bucket(&after, 1));
    let kept: Vec<String> = impls(&result)[1..]
        .iter()
        .map(|r| r.to_token_stream().to_string())
        .collect();
    let remaining: Vec<String> = impls(&removed)
        .iter()
        .map(|r| r.to_token_stream().to_string())
        .collect();
    assert_eq!(kept, remaining);
    assert_eq!(
        impls(&removed)
            .iter()
            .map(|r| r.entry.index())
            .collect::<Vec<_>>(),
        [1, 2]
    );
    // The removed entry names `Self`: with it, a marker fails, and without it, it does not.
    assert!(before.has_self() && !after.has_self());
    let marker: Type = parse_quote!(Marker);
    let repeater: Path = parse_quote!(Repeater);
    let with = |result: &LeakerResult| {
        result
            .repeater_impls(&Projection::new(
                repeater.clone(),
                1,
                ImplSelf::Marker(marker.clone()),
            ))
            .map(|impls| impls.len())
    };
    assert!(matches!(with(&result), Err(EncodeError::NeedsSelf(_))));
    assert_eq!(with(&removed).ok(), Some(2));
    // The type is written as it is, with `Other` in it as written, though `Other` is an entry.
    let expected: Item = parse_quote!(
        mod m {
            struct N<T: ::core::convert::AsRef<Self>>(Rel2<Self, T, Other>, E1, E2);
        }
    );
    assert_eq!(copy_after(&removed, top, &item), expected);
}

#[test]
fn removing_the_only_entry_leaves_a_referrer_with_none() {
    let (item, result, top) = analyze_entry();
    let referrer = result.clone().into_referrer();
    assert!(!referrer.is_empty() && referrer.iter().count() == 1);
    let after = result.clone().ignore(&result.errors()[0].clone());
    let referrer = after.clone().into_referrer();
    assert!(referrer.is_empty());
    assert_eq!(referrer.iter().count(), 0);
    assert!(impls(&after).is_empty());
    assert_eq!(
        copy_after(&after, top, &item),
        parse_quote!(
            mod m {
                struct N<T: ::core::convert::AsRef<Self>>(Rel2<Self, T>);
            }
        )
    );
    // The index of the removed entry is still one of the Referrer: the bucket helper has none.
    assert_eq!(
        common::bucket_self(
            &referrer,
            1,
            common::entry_id_at(&result.into_referrer(), 0)
        ),
        None
    );
}

fn analyze_entry() -> (Item, LeakerResult, ContextId) {
    let item: Item = parse_quote!(
        mod m {
            struct N<T: ::core::convert::AsRef<Self>>(Rel2<Self, T>);
        }
    );
    let (result, top) = analyze(&item);
    (item, result, top)
}

#[test]
fn ignore_before_and_after_the_impls_are_written_gives_the_same_impls() {
    let (_, result, _) = removable();
    // The impls of the entries that stay, written before the entry is removed.
    let written: Vec<String> = impls(&result)
        .iter()
        .map(|r| r.to_token_stream().to_string())
        .collect();
    let removed = result.clone().ignore(&result.errors()[0].clone());
    let after: Vec<String> = impls(&removed)
        .iter()
        .map(|r| r.to_token_stream().to_string())
        .collect();
    assert_eq!(written[1..], after[..]);
}

// ── the tokens of the result ─────────────────────────────────────────────────────────────────

#[test]
fn ignore_before_the_trip_and_after_it_give_the_same_result() {
    for row in rows() {
        let (result, top) = analyze(&row.item);
        // The errors are the same after the trip, and so are the tokens.
        assert_eq!(tokens(&trip(&result)), tokens(&result), "{}", row.name);
        for (ix, error) in result.errors().iter().enumerate() {
            let name = format!("{}: error {}", row.name, ix);
            let before = result.clone().ignore(error);
            let after = trip(&result);
            let after = after.clone().ignore(&after.errors()[ix]);
            assert_eq!(tokens(&before), tokens(&after), "{}", name);
            // And the same again after the trip of the result of `ignore()`.
            assert_eq!(tokens(&trip(&before)), tokens(&before), "{}", name);
            assert_eq!(
                copy_after(&trip(&before), top, &row.item),
                copy_after(&before, top, &row.item),
                "{}",
                name
            );
            assert_eq!(live(&trip(&before)), live(&before), "{}", name);
            assert_eq!(
                impls(&trip(&before))
                    .iter()
                    .map(|r| r.to_token_stream().to_string())
                    .collect::<Vec<_>>(),
                impls(&before)
                    .iter()
                    .map(|r| r.to_token_stream().to_string())
                    .collect::<Vec<_>>(),
                "{}",
                name
            );
        }
    }
}

#[test]
fn a_removed_entry_is_written_with_a_dash_and_read_back() {
    let (_, result, _) = removable();
    let after = result.clone().ignore(&result.errors()[0].clone());
    let text = tokens(&after);
    // `- Type` in the group of the entry, which keeps its place among the others.
    assert!(text.contains("(- Rel2 < Self , T , Other >)"), "{}", text);
    assert!(text.contains("(Other , Third)"), "{}", text);
    let again = trip(&after);
    assert_eq!(tokens(&again), text);
    assert_eq!(live(&again), live(&after));
    // A hand-written Referrer with a dash among the entries.
    let referrer: Referrer = parse_quote!("type-leak-format-16"(A, -B, C));
    assert_eq!(
        referrer
            .iter()
            .map(|(ix, _, _)| ix.index())
            .collect::<Vec<_>>(),
        [0, 2]
    );
    assert_eq!(
        referrer.to_token_stream().to_string(),
        "\"type-leak-format-16\" (A , - B , C)"
    );
}

#[test]
fn the_parser_refuses_a_link_to_what_is_not_there() {
    let (result, _) = analyze(&parse_quote!(
        struct S<'a>(MyType<'_, Inner<'_>, Rel>, Rel, &'a u8);
    ));
    let text = tokens(&result);
    assert!(
        text.contains("part 0 inner [1]") && text.contains("id 1 part 1"),
        "{}",
        text
    );
    let message = |text: &str| -> String {
        read(text.parse().unwrap())
            .err()
            .unwrap_or_else(|| panic!("read: {}", text))
            .to_string()
    };
    // A part, an entry and an error that are not there.
    assert_eq!(
        message(&text.replacen("part 0 inner [1]", "part 9 inner [1]", 1)),
        "the part 9 of an error is not one of the parts"
    );
    assert_eq!(
        message(&text.replacen("part 0 inner [1]", "part 0 entry 7 inner [1]", 1)),
        "the entry 7 of an error is not one of the entries"
    );
    assert_eq!(
        message(&text.replacen("part 0 inner [1]", "part 0 inner [4]", 1)),
        "an error inside an error has the id 4: no error has it"
    );
    // Two errors have the one id, and a word is written twice.
    assert_eq!(
        message(&text.replacen("part 0 inner [1]", "id 1 part 0 inner [1]", 1)),
        "the id 1 is the id of another error"
    );
    assert_eq!(
        message(&text.replacen("part 0 inner [1]", "part 0 part 1 inner [1]", 1)),
        "`part` is written twice"
    );
    assert_eq!(
        message(&text.replacen("part 0 inner [1]", "part 0 inner [1] inner []", 1)),
        "`inner` is written twice"
    );
    // The same result reads.
    assert!(read(text.parse().unwrap()).is_ok());
}

#[test]
fn the_errors_alone_carry_their_links_too() {
    for row in rows() {
        let (result, _) = analyze(&row.item);
        for error in result.errors() {
            let read: ErrorItem = parse2(error.to_token_stream()).unwrap();
            assert_eq!(&read, error, "{}", row.name);
        }
    }
    // An error is not equal to one that is about another part.
    let (result, _) = analyze(&parse_quote!(
        struct S(Rel<'_>, Rel<'_, u8>);
    ));
    assert_ne!(result.errors()[0], result.errors()[1]);
}

#[test]
fn the_result_after_ignore_is_the_same_every_time() {
    for row in rows() {
        let run = || {
            let (result, _) = analyze(&row.item);
            let all: Vec<ErrorItem> = result.errors().to_vec();
            all.iter()
                .map(|error| tokens(&result.clone().ignore(error)))
                .collect::<Vec<_>>()
        };
        assert_eq!(run(), run(), "{}", row.name);
    }
}

#[test]
fn the_errors_of_several_calls_are_linked_each_to_those_of_its_own_call() {
    let (first, second): (Item, Item) = (
        parse_quote!(
            struct A(MyType<'_, Inner<'_>>);
        ),
        parse_quote!(
            struct B(Other<'_, Deep<'_>>);
        ),
    );
    let mut leaker = Leaker::new();
    let (top_a, top_b) = (intern(&mut leaker, &first), intern(&mut leaker, &second));
    let result = leaker.result();
    let cause = |ty: Type| Cause::Type(ty);
    assert_eq!(
        causes(&result),
        [
            cause(parse_quote!(MyType<'_, Inner<'_>>)),
            cause(parse_quote!(Inner<'_>)),
            cause(parse_quote!(Other<'_, Deep<'_>>)),
            cause(parse_quote!(Deep<'_>))
        ]
    );
    // The ids are those of the whole list: the result is read back, which refuses two of a kind.
    let after = trip(&result).ignore(&result.errors()[2].clone());
    assert_eq!(
        causes(&after),
        [
            cause(parse_quote!(MyType<'_, Inner<'_>>)),
            cause(parse_quote!(Inner<'_>))
        ]
    );
    let after = trip(&after);
    // The other call: `MyType` and `Inner` go, `Other` and `Deep` are as they were.
    let both = after.clone().ignore(&after.errors()[0].clone());
    assert!(both.errors().is_empty());
    assert_eq!(
        copy_after(&both, top_a, &first),
        plain(parse_quote!(
            struct A(MyType<'_, Inner<'_>>);
        ))
    );
    assert_eq!(
        copy_after(&both, top_b, &second),
        plain(parse_quote!(
            struct B(Other<'_, Deep<'_>>);
        ))
    );
}

#[test]
fn ignore_after_allow_dropped_an_error_inside_the_part_still_drops_the_part() {
    let item: Item = parse_quote!(
        struct S(Outer<'_, Inner<'_>, Rel>, Rel);
    );
    let (result, top) = analyze(&item);
    assert_eq!(result.errors().len(), 2);
    // `allow()` drops the inner error, and the outer one no longer holds it.
    let allowed =
        result.allow(|cause: &Cause, _: Place| *cause == Cause::Type(parse_quote!(Inner)));
    assert_eq!(
        causes(&allowed),
        [Cause::Type(parse_quote!(Outer<'_, Inner<'_>, Rel>))]
    );
    let again = trip(&allowed);
    assert_eq!(tokens(&again), tokens(&allowed));
    let after = again.clone().ignore(&again.errors()[0].clone());
    assert!(after.errors().is_empty());
    assert_eq!(
        copy_after(&after, top, &item),
        plain(parse_quote!(
            struct S(Outer<'_, Inner<'_>, Rel>, E0);
        ))
    );
}

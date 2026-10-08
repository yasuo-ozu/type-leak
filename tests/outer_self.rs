//! The bounds of a generic of an outer scope that mention `Self`, used where `Self` is something
//! else. They are written for the `Self` of the scope that declares them, so the context of the
//! type rewrites them for it (`T: Tr<Self>` of `struct S<T: Tr<Self>>` is `T: Tr<S<T>>`), or fails
//! if `Self` has no name (the implementor of a trait). The Referrer computes the same context from
//! the scopes of the copy.

use syn::visit_mut::VisitMut;
use syn::*;
use template_quote::{quote, ToTokens};
use type_leak::{ContextId, Leaker, NotInternableError, Referrer, SelfRequirement};

fn s(x: &impl ToTokens) -> String {
    quote!(#x).to_string()
}

fn ident(name: &str) -> Ident {
    Ident::new(name, proc_macro2::Span::call_site())
}

fn self_of(referrer: &Referrer, context: ContextId) -> String {
    match referrer.context_self(context) {
        None => "-".to_string(),
        Some(SelfRequirement::Bound(path)) => format!("Self: {}", s(path)),
        Some(SelfRequirement::Type(ty)) => format!("Self = {}", s(ty)),
        Some(other) => format!("{:?}", other),
    }
}

/// The required entries as `type | generics | Self`.
fn entries(referrer: &Referrer) -> Vec<String> {
    (0..referrer.iter().count())
        .filter(|&ix| referrer.must(ix))
        .map(|ix| {
            let generics = referrer.generics(ix);
            let where_clause = &generics.where_clause;
            format!(
                "{} | {} | {}",
                s(&referrer.iter().nth(ix).unwrap()),
                s(&quote!(#generics #where_clause)),
                self_of(referrer, referrer.context(ix))
            )
        })
        .collect()
}

fn e(ty: &str, generics: &str, self_ty: &str) -> String {
    format!("{} | {} | {}", ty, generics, self_ty)
}

/// The generics of `struct S<T: ::p::Tr<Self>>`, which mention `Self`.
fn bound() -> Generics {
    parse_quote!(<T: ::p::Tr<Self>>)
}

/// The same bound as a where predicate.
fn predicate() -> Generics {
    let mut generics: Generics = parse_quote!(<T>);
    generics.where_clause = Some(parse_quote!(where T: ::p::Tr<Self>));
    generics
}

// ── the Leaker ───────────────────────────────────────────────────────────────────────────────

#[test]
fn a_bound_is_rewritten_for_the_self_of_the_scope_that_declares_it() {
    // `Self` is `W` where `Rel<T>` is written, but `T: ::p::Tr<Self>` meant `Self = S<T>`.
    let mut leaker = Leaker::new();
    let (outer, (inner, ())) = leaker
        .enter_type(&ident("S"), &bound(), true, |v| {
            v.enter_impl(&Generics::default(), &parse_quote!(W), true, |v| {
                v.visit_type(&parse_quote!(Rel<T>))
            })
            .unwrap()
        })
        .unwrap();
    let referrer = leaker.finish();
    assert_eq!(
        entries(&referrer),
        [e("Rel < T >", "< T : :: p :: Tr < S < T > > >", "-")]
    );
    // The scope that declared the bound is the one the Referrer needs: it records its `Self`
    // although no entry names `Self`. The scope of the impl declares no bound.
    assert_eq!(self_of(&referrer, outer), "Self = S < T >");
    assert_eq!(s(referrer.context_generics(outer)), s(&bound()));
    assert_eq!(self_of(&referrer, inner), "-");
    // `Self` is not part of the entry, so the entry is not that of the scope.
    assert_ne!(referrer.context(0), outer);
}

#[test]
fn a_scope_which_no_entry_depends_on_does_not_record_self() {
    // No type uses `T`: nothing depends on the `Self` of the struct.
    let mut leaker = Leaker::new();
    let (outer, _) = leaker
        .enter_type(&ident("S"), &bound(), true, |v| {
            v.enter_impl(&Generics::default(), &parse_quote!(W), true, |v| {
                v.visit_type(&parse_quote!(Rel))
            })
            .unwrap()
        })
        .unwrap();
    let referrer = leaker.finish();
    assert_eq!(entries(&referrer), [e("Rel", "", "-")]);
    assert_eq!(self_of(&referrer, outer), "-");
}

#[test]
fn the_same_type_in_the_scope_itself_keeps_self() {
    // Where `Self` is the one that declares the bound, nothing changes: `Self` is part of the
    // context, as it was. The two entries are told apart by it.
    let mut leaker = Leaker::new();
    let (outer, ()) = leaker
        .enter_type(&ident("S"), &bound(), true, |v| {
            v.visit_type(&parse_quote!(Rel<T>));
            v.enter_impl(&Generics::default(), &parse_quote!(W), true, |v| {
                v.visit_type(&parse_quote!(Rel<T>))
            })
            .unwrap();
        })
        .unwrap();
    let referrer = leaker.finish();
    assert_eq!(
        entries(&referrer),
        [
            e("Rel < T >", "< T : :: p :: Tr < Self > >", "Self = S < T >"),
            e("Rel < T >", "< T : :: p :: Tr < S < T > > >", "-"),
        ]
    );
    assert_eq!(self_of(&referrer, outer), "Self = S < T >");
    assert_eq!(referrer.context(0), outer);
}

#[test]
fn a_where_predicate_is_rewritten_too() {
    // The same bound as a predicate: the entry is the same. A predicate on `Self` itself, which
    // brings `T` along, is rewritten as well.
    let by_param = {
        let mut leaker = Leaker::new();
        leaker
            .enter_type(&ident("S"), &bound(), true, |v| {
                v.enter_impl(&Generics::default(), &parse_quote!(W), true, |v| {
                    v.visit_type(&parse_quote!(Rel<T>))
                })
                .unwrap()
            })
            .unwrap();
        leaker.finish()
    };
    let by_predicate = {
        let mut leaker = Leaker::new();
        leaker
            .enter_type(&ident("S"), &predicate(), true, |v| {
                v.enter_impl(&Generics::default(), &parse_quote!(W), true, |v| {
                    v.visit_type(&parse_quote!(Rel<T>))
                })
                .unwrap()
            })
            .unwrap();
        leaker.finish()
    };
    assert_eq!(entries(&by_param), entries(&by_predicate));
    let mut leaker = Leaker::new();
    let mut generics: Generics = parse_quote!(<T>);
    generics.where_clause = Some(parse_quote!(where Self: ::p::Tr<T>, Self: Sized));
    leaker
        .enter_type(&ident("S"), &generics, true, |v| {
            v.enter_impl(&Generics::default(), &parse_quote!(W), true, |v| {
                v.visit_type(&parse_quote!(Rel<T>))
            })
            .unwrap()
        })
        .unwrap();
    // `Self: Sized` names no generic, so it is not one of the entry (it bounds the other `Self`).
    assert_eq!(
        entries(&leaker.finish()),
        [e(
            "Rel < T >",
            "< T > where S < T > : :: p :: Tr < T >",
            "-"
        )]
    );
}

#[test]
fn the_self_of_a_trait_has_no_name() {
    // The implementor of the trait cannot be written where `Self` is something else.
    let mut leaker = Leaker::new();
    let (_, inner) = leaker
        .enter_trait(&ident("Tr"), &bound(), true, |v| {
            v.enter_impl(&Generics::default(), &parse_quote!(W), true, |v| {
                v.visit_type(&parse_quote!(Rel<T>))
            })
            .map(|_| ())
        })
        .unwrap();
    let err = inner.unwrap_err();
    assert!(matches!(err, NotInternableError::RelativePath(_)));
    assert_eq!(
        err.to_string(),
        "the bound of `T` mentions a `Self` which has no name here"
    );
    assert_eq!(
        err.note().as_deref(),
        Some(
            "the bound of `T` mentions the `Self` of the trait `Tr < T >`, which has no name \
             here: `Self` is something else in the item around"
        )
    );
    // The mention of `Self` is a cause too.
    let message = syn::Error::from(err).to_string();
    assert!(message.contains("it cannot be leaked as a whole because of `Self`"));
    // The scope around goes on, with nothing recorded.
    assert_eq!(leaker.finish().iter().count(), 0);
    // So is it for the items of the scope, and for a where predicate.
    let bound_headline = "the bound of `T` mentions a `Self` which has no name here";
    let predicate_headline = "the where predicate `T : :: p :: Tr < Self >` mentions a `Self` \
                              which has no name here";
    for (generics, item, headline) in [
        (
            bound(),
            parse_quote!(impl ::p::W { fn f(x: Rel<T>) {} }),
            bound_headline,
        ),
        (
            bound(),
            parse_quote!(
                struct V {
                    a: Rel<T>,
                }
            ),
            bound_headline,
        ),
        (
            predicate(),
            parse_quote!(
                trait Q {
                    fn f(x: Rel<T>);
                }
            ),
            predicate_headline,
        ),
    ] {
        let item: Item = item;
        let err = Leaker::new()
            .enter_trait(&ident("Tr"), &generics, true, |v| v.visit_item(&item))
            .unwrap_err();
        assert_eq!(err.to_string(), headline, "{}", s(&item));
        let note = err.note().unwrap();
        assert!(
            note.contains("the `Self` of the trait `Tr < T >`"),
            "{}",
            note
        );
    }
}

#[test]
fn an_unnameable_self_is_reported_for_the_bound_not_the_path() {
    // The path of `Rel<T>` is no worse than that of `Other<'_>`: it is the bound of `T` that cannot
    // be written, so the headline is about the bound, and the note tells why.
    let mut leaker = Leaker::new();
    let mut checked = None;
    let (_, inner) = leaker
        .enter_trait(&ident("Tr"), &bound(), true, |v| {
            v.enter_impl(&Generics::default(), &parse_quote!(W), true, |v| {
                // `View::check()` gives the headline with the rest of the conflict.
                checked = Some(v.check(&parse_quote!(Rel<T>)));
                v.visit_type(&parse_quote!(Rel<T>));
                // Another conflict in the same scope is reported too, in order.
                v.visit_type(&parse_quote!(Other<'_>));
            })
            .map(|_| ())
        })
        .unwrap();
    let checked = checked.unwrap();
    assert_eq!(
        checked.must.iter().map(s).collect::<Vec<_>>(),
        ["Rel < T >"]
    );
    assert_eq!(
        checked.impossible.iter().map(s).collect::<Vec<_>>(),
        ["Self"]
    );
    assert!(checked.is_own && !checked.optional);
    assert_eq!(
        checked.headline.as_deref(),
        Some("the bound of `T` mentions a `Self` which has no name here")
    );
    assert!(checked.note.unwrap().contains("which has no name here"));
    // A conflict about the path has no headline of its own.
    let mut leaker2 = Leaker::new();
    let (_, relative) = leaker2
        .enter_generics(&Generics::default(), |v| v.check(&parse_quote!(Other<'_>)))
        .unwrap();
    assert!(relative.headline.is_none());

    let err = inner.unwrap_err();
    assert!(matches!(err, NotInternableError::RelativePath(_)));
    assert_eq!(
        err.to_string(),
        "the bound of `T` mentions a `Self` which has no name here\n\
         use absolute path: 'Other < '_ >'"
    );
    // The first error comes first, with the note which tells why, and the cause.
    assert_eq!(
        err.note().as_deref(),
        Some(
            "the bound of `T` mentions the `Self` of the trait `Tr < T >`, which has no name \
             here: `Self` is something else in the item around"
        )
    );
    let message = syn::Error::from(err).to_string();
    assert!(
        message.starts_with(
            "the bound of `T` mentions a `Self` which has no name here\n\n  \
             = note: the bound of `T` mentions the `Self` of the trait `Tr < T >`, which has no \
             name here: `Self` is something else in the item around\n  \
             = note: it cannot be leaked as a whole because of `Self`\n"
        ),
        "{}",
        message
    );
    // Nothing of it is a note of the error about the path.
    assert!(
        !message.contains("use absolute path: 'Rel < T >'"),
        "{}",
        message
    );
}

#[test]
fn an_associated_item_of_self_has_no_name() {
    // `<S<T>>::X` is ambiguous (E0223), so it is no use to write it for the type.
    let generics: Generics = parse_quote!(<T: ::p::Tr<Self::X>>);
    let mut leaker = Leaker::new();
    let (_, inner) = leaker
        .enter_impl(&generics, &parse_quote!(S<T>), true, |v| {
            v.enter_impl(&Generics::default(), &parse_quote!(W), true, |v| {
                v.visit_type(&parse_quote!(Rel<T>))
            })
            .map(|_| ())
        })
        .unwrap();
    let err = inner.unwrap_err();
    assert_eq!(
        err.to_string(),
        "the bound of `T` mentions `Self :: X`, which has no name here"
    );
    assert_eq!(
        err.note().as_deref(),
        Some(
            "the bound of `T` mentions `Self :: X`, an associated item of the `Self` of `S < T >`, \
             which has no name here: `Self` is something else in the item around"
        )
    );
    // `<Self as ::p::Tr>::X` is written in full, and `S<T>` is `Self` there.
    let generics: Generics = parse_quote!(<T: ::p::Tr<<Self as ::p::Q>::X>>);
    let mut leaker = Leaker::new();
    leaker
        .enter_impl(&generics, &parse_quote!(S<T>), true, |v| {
            v.enter_impl(&Generics::default(), &parse_quote!(W), true, |v| {
                v.visit_type(&parse_quote!(Rel<T>))
            })
            .unwrap()
        })
        .unwrap();
    assert_eq!(
        entries(&leaker.finish()),
        [e(
            "Rel < T >",
            "< T : :: p :: Tr < < S < T > as :: p :: Q > :: X > >",
            "-"
        )]
    );
}

// ── what is unchanged: the same `Self` ───────────────────────────────────────────────────────

#[test]
fn the_same_self_changes_nothing() {
    // A method of the trait, a scope that declares no `Self`, and an impl for the same type: each
    // keeps the `Self` of the scope.
    let mut leaker = Leaker::new();
    leaker
        .enter_trait(&ident("Tr"), &bound(), true, |v| {
            v.visit_type(&parse_quote!(Rel<T>));
            v.enter_generics(&parse_quote!(<U>), |v| {
                v.visit_type(&parse_quote!(Rel2<T, U>))
            })
            .unwrap();
            v.visit_item(&parse_quote!(
                fn f(x: Rel3<T>) {}
            ));
        })
        .unwrap();
    assert_eq!(
        entries(&leaker.finish()),
        [
            e("Rel < T >", "< T : :: p :: Tr < Self > >", "Self: Tr < T >"),
            e(
                "Rel3 < T >",
                "< T : :: p :: Tr < Self > >",
                "Self: Tr < T >"
            ),
            e(
                "Rel2 < T , U >",
                "< T : :: p :: Tr < Self > , U >",
                "Self: Tr < T >"
            ),
        ]
    );
    let mut leaker = Leaker::new();
    leaker
        .enter_type(&ident("S"), &bound(), true, |v| {
            v.enter_impl(&Generics::default(), &parse_quote!(S<T>), true, |v| {
                v.visit_type(&parse_quote!(Rel<T>))
            })
            .unwrap();
            v.visit_item(&parse_quote!(
                impl S<T> {
                    fn f(x: Rel2<T>) {}
                }
            ));
        })
        .unwrap();
    assert_eq!(
        entries(&leaker.finish()),
        [
            e("Rel < T >", "< T : :: p :: Tr < Self > >", "Self = S < T >"),
            e(
                "Rel2 < T >",
                "< T : :: p :: Tr < Self > >",
                "Self = S < T >"
            ),
            // The self type of the nested impl is a type of the scope, written for its `Self`.
            e("S < T >", "< T : :: p :: Tr < Self > >", "Self = S < T >"),
        ]
    );
}

#[test]
fn a_bound_of_the_scope_itself_is_not_rewritten() {
    // `U: ::p::Bar<Self>` is declared by the impl, whose `Self` is the one around `Rel<U>`.
    let mut leaker = Leaker::new();
    leaker
        .enter_type(&ident("S"), &bound(), true, |v| {
            v.enter_impl(
                &parse_quote!(<U: ::p::Bar<Self>>),
                &parse_quote!(W<U>),
                true,
                |v| v.visit_type(&parse_quote!(Rel<T, U>)),
            )
            .unwrap();
        })
        .unwrap();
    // `T` is the struct's, so its bound is for `S<T>`, and `U`'s is not.
    assert_eq!(
        entries(&leaker.finish()),
        [e(
            "Rel < T , U >",
            "< T : :: p :: Tr < S < T > > , U : :: p :: Bar < Self > >",
            "Self = W < U >"
        )]
    );
}

// ── nested items ─────────────────────────────────────────────────────────────────────────────

#[test]
fn the_items_nested_in_a_struct_scope_use_its_generic() {
    // Each item has a `Self` of its own, but the function has none.
    let mut leaker = Leaker::new();
    let (outer, ()) = leaker
        .enter_type(&ident("S"), &bound(), true, |v| {
            v.visit_item(&parse_quote!(
                impl ::p::W {
                    fn f(x: Rel<T>) {}
                }
            ));
            v.visit_item(&parse_quote!(
                struct V {
                    a: Rel2<T>,
                }
            ));
            v.visit_item(&parse_quote!(
                trait Q {
                    fn g(x: Rel3<T>);
                }
            ));
            v.visit_item(&parse_quote!(
                fn h(x: Rel4<T>) {}
            ));
            v.visit_item(&parse_quote!(
                enum E {
                    A(Rel5<T>),
                }
            ));
        })
        .unwrap();
    let referrer = leaker.finish();
    let rewritten = "< T : :: p :: Tr < S < T > > >";
    assert_eq!(
        entries(&referrer),
        [
            e("Rel < T >", rewritten, "-"),
            e("Rel2 < T >", rewritten, "-"),
            e("Rel3 < T >", rewritten, "-"),
            e("Rel5 < T >", rewritten, "-"),
            e(
                "Rel4 < T >",
                "< T : :: p :: Tr < Self > >",
                "Self = S < T >"
            ),
        ]
    );
    // The first four share one context, and the scope records its `Self`.
    assert_eq!(referrer.context(0), referrer.context(3));
    assert_eq!(self_of(&referrer, outer), "Self = S < T >");
    assert_eq!(referrer.context(4), outer);
}

#[test]
fn a_module_has_no_self() {
    // In a module `Self` means nothing, so a bound of an outer scope is rewritten there too. The
    // entries are those of the module.
    let mut leaker = Leaker::new();
    leaker
        .enter_type(&ident("S"), &bound(), true, |v| {
            v.visit_item(&parse_quote!(
                mod m {
                    fn f(x: Rel<T>) {}
                    struct V {
                        a: Rel2<T>,
                    }
                }
            ));
        })
        .unwrap();
    assert_eq!(
        entries(&leaker.finish()),
        [
            e("Rel < T >", "< T : :: p :: Tr < S < T > > >", "-"),
            e("Rel2 < T >", "< T : :: p :: Tr < S < T > > >", "-"),
        ]
    );
}

#[test]
fn allow_self_false_is_not_an_obstacle() {
    // The bound does not mention `Self` once it is rewritten, so a Repeater for a marker can have
    // it.
    let mut leaker = Leaker::new();
    leaker
        .enter_type(&ident("S"), &bound(), false, |v| {
            v.enter_impl(&Generics::default(), &parse_quote!(W), false, |v| {
                v.visit_type(&parse_quote!(Rel<T>))
            })
            .unwrap();
            // Where `Self` is the struct, it is denied as it always was.
            assert!(v
                .enter_generics(&Generics::default(), |v| v
                    .visit_type(&parse_quote!(Rel2<T>)))
                .is_err());
        })
        .unwrap();
}

#[test]
fn several_scopes_each_rewrite_their_bounds() {
    // `T: ::p::Tr<Self>` is for `S<T>`, `U: ::p::Bar<Self>` for `W<U>`, and `Rel<T, U>` is written
    // where `Self` is `V`.
    let mut leaker = Leaker::new();
    let (outer, (middle, (inner, ()))) = leaker
        .enter_type(&ident("S"), &bound(), true, |v| {
            v.enter_impl(
                &parse_quote!(<U: ::p::Bar<Self>>),
                &parse_quote!(W<U>),
                true,
                |v| {
                    v.enter_impl(&Generics::default(), &parse_quote!(V), true, |v| {
                        v.visit_type(&parse_quote!(Rel<T, U>))
                    })
                    .unwrap()
                },
            )
            .unwrap()
        })
        .unwrap();
    let referrer = leaker.finish();
    assert_eq!(
        entries(&referrer),
        [e(
            "Rel < T , U >",
            "< T : :: p :: Tr < S < T > > , U : :: p :: Bar < W < U > > >",
            "-"
        )]
    );
    // Each scope that declared a bound records its `Self`, the one of `V` has none.
    assert_eq!(self_of(&referrer, outer), "Self = S < T >");
    assert_eq!(self_of(&referrer, middle), "Self = W < U >");
    assert_eq!(self_of(&referrer, inner), "-");
}

// ── the defaults of a nested scope ───────────────────────────────────────────────────────────

#[test]
fn a_default_in_a_nested_scope_is_rewritten_for_the_outer_self() {
    // `Rel<T>` is the default of `U` of an impl for `W`, nested in `struct S<T: ::p::Tr<Self>>`.
    // The bound of `T` meant `Self = S<T>`, as for any type written there.
    for generics in [bound(), predicate()] {
        let mut leaker = Leaker::new();
        let (outer, (inner, ())) = leaker
            .enter_type(&ident("S"), &generics, true, |v| {
                v.enter_impl(&parse_quote!(<U = Rel<T>>), &parse_quote!(W), true, |_| ())
                    .unwrap()
            })
            .unwrap();
        let referrer = leaker.finish();
        assert_eq!(
            entries(&referrer),
            [e("Rel < T >", "< T : :: p :: Tr < S < T > > >", "-")]
        );
        // The scope that declared the bound records its `Self`, as for any type that depends on it.
        assert_eq!(self_of(&referrer, outer), "Self = S < T >");
        assert_eq!(self_of(&referrer, inner), "-");
    }
    // A default that names the parameters before it keeps them, with the rewritten bound.
    let mut leaker = Leaker::new();
    leaker
        .enter_type(&ident("S"), &bound(), true, |v| {
            v.enter_impl(
                &parse_quote!(<U, V = Rel<T, U>>),
                &parse_quote!(W),
                true,
                |_| (),
            )
            .unwrap()
        })
        .unwrap();
    assert_eq!(
        entries(&leaker.finish()),
        [e(
            "Rel < T , U >",
            "< T : :: p :: Tr < S < T > > , U >",
            "-"
        )]
    );
}

#[test]
fn a_default_has_no_self_so_the_same_self_is_rewritten_too() {
    // Unlike a type in the scope, which keeps `Self` where it is the same (`Rel3<T>`), a default
    // cannot depend on `Self` (the Repeater impl does not know it), so it is rewritten whatever
    // the scope declares.
    let mut leaker = Leaker::new();
    leaker
        .enter_type(&ident("S"), &bound(), true, |v| {
            v.enter_impl(
                &parse_quote!(<U = Rel<T>>),
                &parse_quote!(S<T>),
                true,
                |_| (),
            )
            .unwrap();
            v.enter_generics(&parse_quote!(<U = Rel2<T>>), |_| ())
                .unwrap();
            v.visit_type(&parse_quote!(Rel3<T>));
        })
        .unwrap();
    assert_eq!(
        entries(&leaker.finish()),
        [
            e("Rel < T >", "< T : :: p :: Tr < S < T > > >", "-"),
            e("Rel2 < T >", "< T : :: p :: Tr < S < T > > >", "-"),
            e(
                "Rel3 < T >",
                "< T : :: p :: Tr < Self > >",
                "Self = S < T >"
            ),
        ]
    );
}

#[test]
fn a_default_with_an_outer_bound_on_the_self_of_a_trait_is_an_error() {
    // The implementor of the trait cannot be written in a default, nor where `Self` is `W`.
    let headline = "the bound of `T` mentions a `Self` which has no name here";
    let mut leaker = Leaker::new();
    let (_, inner) = leaker
        .enter_trait(&ident("Tr"), &bound(), true, |v| {
            v.enter_impl(&parse_quote!(<U = Rel<T>>), &parse_quote!(W), true, |_| ())
                .map(|_| ())
        })
        .unwrap();
    let err = inner.unwrap_err();
    assert!(matches!(err, NotInternableError::RelativePath(_)));
    assert_eq!(err.to_string(), headline);
    assert!(
        err.note()
            .unwrap()
            .contains("the `Self` of the trait `Tr < T >`"),
        "{:?}",
        err.note()
    );
    assert_eq!(leaker.finish().iter().count(), 0);
    // A scope that declares no `Self` keeps the one of the trait, but a default has none.
    let mut leaker = Leaker::new();
    let (_, inner) = leaker
        .enter_trait(&ident("Tr"), &bound(), true, |v| {
            v.enter_generics(&parse_quote!(<U = Rel<T>>), |_| ())
                .map(|_| ())
        })
        .unwrap();
    assert_eq!(inner.unwrap_err().to_string(), headline);
}

// ── the Referrer ─────────────────────────────────────────────────────────────────────────────

/// Rewrites each leaked type to `P<index>`.
fn replace(_: Type, ix: usize) -> Type {
    let ix = proc_macro2::Literal::usize_unsuffixed(ix);
    parse_quote!(P<#ix>)
}

#[test]
fn the_referrer_finds_the_entry_of_a_nested_scope() {
    // `Rel<T>` of the scope itself (entry 1, with `Self`) and of the impl nested in it (entry 0,
    // rewritten) are two entries, which the copy tells apart by the scopes it nests.
    let mut leaker = Leaker::new();
    let (outer, ()) = leaker
        .enter_type(&ident("S"), &bound(), true, |v| {
            v.enter_impl(&Generics::default(), &parse_quote!(W), true, |v| {
                v.visit_type(&parse_quote!(Rel<T>))
            })
            .unwrap();
            v.visit_type(&parse_quote!(Rel<T>));
        })
        .unwrap();
    let referrer = leaker.finish();
    assert_eq!(referrer.iter().count(), 2);
    let copy: ItemImpl = parse_quote!(
        impl<T: ::p::Tr<Self>> Foo for S<T> {
            fn m(x: Rel<T>) {
                impl W {
                    fn f(y: Rel<T>) {}
                }
            }
        }
    );
    let expected: ItemImpl = parse_quote!(
        impl<T: ::p::Tr<Self>> Foo for S<T> {
            fn m(x: P<1>) {
                impl W {
                    fn f(y: P<0>) {}
                }
            }
        }
    );
    let mut rewritten = copy.clone();
    referrer
        .clone()
        .visitor(outer, replace)
        .visit_item_impl_mut(&mut rewritten);
    assert_eq!(s(&rewritten), s(&expected));
    // The same after the Referrer was written and read back: the context knows the `Self`.
    let tokens = quote!(#referrer);
    let parsed: Referrer = syn::parse2(tokens).unwrap();
    assert_eq!(parsed, referrer);
    let mut rewritten = copy.clone();
    parsed
        .visitor(outer, replace)
        .visit_item_impl_mut(&mut rewritten);
    assert_eq!(s(&rewritten), s(&expected));
}

#[test]
fn the_referrer_needs_the_nesting_of_the_leaker() {
    // Only the type of the nested impl is an entry. The same type in the scope itself is not, and
    // neither is it in an impl nested at the same `Self`.
    let mut leaker = Leaker::new();
    let (outer, ()) = leaker
        .enter_type(&ident("S"), &bound(), true, |v| {
            v.enter_impl(&Generics::default(), &parse_quote!(W), true, |v| {
                v.visit_type(&parse_quote!(Rel<T>))
            })
            .unwrap();
        })
        .unwrap();
    let referrer = leaker.finish();
    let mut copy: ItemImpl = parse_quote!(
        impl<T: ::p::Tr<Self>> Foo for S<T> {
            fn m(x: Rel<T>) {
                impl S<T> {
                    fn f(y: Rel<T>) {}
                }
                impl W {
                    fn g(z: Rel<T>) {}
                }
                fn h(w: Rel<T>) {}
            }
        }
    );
    referrer
        .visitor(outer, replace)
        .visit_item_impl_mut(&mut copy);
    let expected: ItemImpl = parse_quote!(
        impl<T: ::p::Tr<Self>> Foo for S<T> {
            fn m(x: Rel<T>) {
                impl S<T> {
                    fn f(y: Rel<T>) {}
                }
                impl W {
                    fn g(z: P<0>) {}
                }
                fn h(w: Rel<T>) {}
            }
        }
    );
    assert_eq!(s(&copy), s(&expected));
}

#[test]
fn the_referrer_nests_the_scopes_like_the_leaker() {
    // `Rel<T, U>` in an impl for `V`, in an impl for `W<U>` with its own bound, in `S`.
    let mut leaker = Leaker::new();
    let (outer, ()) = leaker
        .enter_type(&ident("S"), &bound(), true, |v| {
            // The Leaker does not look into the body of a function: the scopes are entered.
            v.enter_impl(
                &parse_quote!(<U: ::p::Bar<Self>>),
                &parse_quote!(::p::W<U>),
                true,
                |v| {
                    v.enter_impl(&Generics::default(), &parse_quote!(V), true, |v| {
                        v.visit_type(&parse_quote!(Rel<T, U>))
                    })
                    .unwrap()
                },
            )
            .unwrap();
        })
        .unwrap();
    let referrer = leaker.finish();
    assert_eq!(
        entries(&referrer),
        [e(
            "Rel < T , U >",
            "< T : :: p :: Tr < S < T > > , U : :: p :: Bar < :: p :: W < U > > >",
            "-"
        )]
    );
    let mut copy: ItemImpl = parse_quote!(
        impl<T: ::p::Tr<Self>> Foo for S<T> {
            fn m() {
                impl<U: ::p::Bar<Self>> ::p::Baz for ::p::W<U> {
                    fn g() {
                        impl V {
                            fn h(x: Rel<T, U>) {}
                        }
                    }
                }
            }
        }
    );
    referrer
        .visitor(outer, replace)
        .visit_item_impl_mut(&mut copy);
    let expected: ItemImpl = parse_quote!(
        impl<T: ::p::Tr<Self>> Foo for S<T> {
            fn m() {
                impl<U: ::p::Bar<Self>> ::p::Baz for ::p::W<U> {
                    fn g() {
                        impl V {
                            fn h(x: P<0>) {}
                        }
                    }
                }
            }
        }
    );
    assert_eq!(s(&copy), s(&expected));
}

#[test]
fn the_referrer_takes_the_self_of_each_nested_item() {
    // A struct, an enum, a union, a trait and an impl each have a `Self` of their own, which the
    // Referrer takes from the item it visits.
    let mut leaker = Leaker::new();
    let (outer, ()) = leaker
        .enter_type(&ident("S"), &bound(), true, |v| {
            for item in [
                parse_quote!(
                    struct V {
                        a: Rel<T>,
                    }
                ),
                parse_quote!(
                    enum E {
                        A(Rel2<T>),
                    }
                ),
                parse_quote!(
                    union U {
                        a: Rel3<T>,
                    }
                ),
                parse_quote!(
                    trait Q {
                        fn g(x: Rel4<T>);
                    }
                ),
                parse_quote!(
                    impl ::p::W {
                        fn f(x: Rel5<T>) {}
                    }
                ),
            ] {
                let item: Item = item;
                v.visit_item(&item);
            }
        })
        .unwrap();
    let referrer = leaker.finish();
    assert_eq!(entries(&referrer).len(), 5);
    let mut copy: ItemImpl = parse_quote!(
        impl<T: ::p::Tr<Self>> Foo for S<T> {
            fn m() {
                struct V {
                    a: Rel<T>,
                }
                enum E {
                    A(Rel2<T>),
                }
                union U {
                    a: Rel3<T>,
                }
                trait Q {
                    fn g(x: Rel4<T>);
                }
                impl ::p::W {
                    fn f(x: Rel5<T>) {}
                }
            }
        }
    );
    referrer
        .visitor(outer, replace)
        .visit_item_impl_mut(&mut copy);
    let expected: ItemImpl = parse_quote!(
        impl<T: ::p::Tr<Self>> Foo for S<T> {
            fn m() {
                struct V {
                    a: P<0>,
                }
                enum E {
                    A(P<1>),
                }
                union U {
                    a: P<2>,
                }
                trait Q {
                    fn g(x: P<3>);
                }
                impl ::p::W {
                    fn f(x: P<4>) {}
                }
            }
        }
    );
    assert_eq!(s(&copy), s(&expected));
}

#[test]
fn the_referrer_finds_the_default_of_a_nested_scope() {
    // The default is analyzed without `Self`: a copy that nests the same scopes finds its entry
    // where the nested scope has another `Self` (`W`), and where it has the same (`S<T>`, or none
    // of its own as the generics of a method).
    let mut leaker = Leaker::new();
    let (outer, ()) = leaker
        .enter_type(&ident("S"), &bound(), true, |v| {
            v.enter_impl(&parse_quote!(<U = Rel<T>>), &parse_quote!(W), true, |_| ())
                .unwrap();
            v.enter_impl(
                &parse_quote!(<U = Rel2<T>>),
                &parse_quote!(S<T>),
                true,
                |_| (),
            )
            .unwrap();
            v.enter_generics(&parse_quote!(<U = Rel3<T>>), |_| ())
                .unwrap();
            // A type of the scope itself is another entry, with its `Self`.
            v.visit_type(&parse_quote!(Rel4<T>));
        })
        .unwrap();
    let referrer = leaker.finish();
    assert_eq!(referrer.iter().count(), 4);
    let copy: ItemImpl = parse_quote!(
        impl<T: ::p::Tr<Self>> Foo for S<T> {
            fn m<U = Rel3<T>>(x: Rel4<T>) {
                impl<U = Rel<T>> W {}
                impl<U = Rel2<T>> S<T> {}
            }
        }
    );
    let expected: ItemImpl = parse_quote!(
        impl<T: ::p::Tr<Self>> Foo for S<T> {
            fn m<U = P<2>>(x: P<3>) {
                impl<U = P<0>> W {}
                impl<U = P<1>> S<T> {}
            }
        }
    );
    let tokens = quote!(#referrer);
    let parsed: Referrer = syn::parse2(tokens).unwrap();
    assert_eq!(parsed, referrer);
    for referrer in [referrer, parsed] {
        let mut rewritten = copy.clone();
        referrer
            .visitor(outer, replace)
            .visit_item_impl_mut(&mut rewritten);
        assert_eq!(s(&rewritten), s(&expected));
    }
}

#[test]
fn the_referrer_finds_a_default_that_uses_a_bound_of_its_own_scope() {
    // The default is analyzed without `Self` in the scope that declares it too: `T: ::p::Bar<Self>`
    // of the nested trait is written as it is, with no `Self` to rewrite it for.
    let mut leaker = Leaker::new();
    let (outer, ()) = leaker
        .enter_generics(&parse_quote!(<X>), |v| {
            v.enter_trait(
                &ident("Tr2"),
                &parse_quote!(<T: ::p::Bar<Self>, U = Rel<T>>),
                true,
                |_| (),
            )
            .unwrap();
        })
        .unwrap();
    let referrer = leaker.finish();
    assert_eq!(
        entries(&referrer),
        [e("Rel < T >", "< T : :: p :: Bar < Self > >", "-")]
    );
    let mut copy: ItemFn = parse_quote!(
        fn f<X>() {
            trait Tr2<T: ::p::Bar<Self>, U = Rel<T>> {}
        }
    );
    referrer
        .visitor(outer, replace)
        .visit_item_fn_mut(&mut copy);
    let expected: ItemFn = parse_quote!(
        fn f<X>() {
            trait Tr2<T: ::p::Bar<Self>, U = P<0>> {}
        }
    );
    assert_eq!(s(&copy), s(&expected));
}

#[test]
fn expand_knows_the_self_of_the_context() {
    // `expand()` has only the scope of the context: the type of the scope itself.
    let mut leaker = Leaker::new();
    let (outer, ()) = leaker
        .enter_type(&ident("S"), &bound(), true, |v| {
            v.visit_type(&parse_quote!(Rel<T>));
            v.enter_impl(&Generics::default(), &parse_quote!(W), true, |v| {
                v.visit_type(&parse_quote!(Rel2<T>))
            })
            .unwrap();
        })
        .unwrap();
    let referrer = leaker.finish();
    assert_eq!(
        s(&referrer.expand(outer, parse_quote!(Rel<T>), replace)),
        "P < 0 >"
    );
    // `Rel2<T>` is that of the nested impl, which `expand()` does not know.
    assert_eq!(
        s(&referrer.expand(outer, parse_quote!(Rel2<T>), replace)),
        "Rel2 < T >"
    );
}

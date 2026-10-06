//! The lifetimes of a `for<..>` binder, and the types leaked with them.
//!
//! A type using the lifetime `'x` of a binder is leaked with `'x` among its generics, so the
//! copy has the encoding `&'x ()` in the trait arguments of a projection
//! (`<S as Repeater<&'x (), ..>>::Type`). rustc does not count a lifetime which occurs only in a
//! projection as constrained by the inputs, so the copy is rejected if the output of the binder
//! (the return type of a function pointer or of `Fn(..) -> R`, or an associated type binding such
//! as `Item = R`) mentions `'x` and no input keeps it as written: E0581 for the return type of a
//! function pointer, and E0582 for the binding of an associated type. So a type using such a
//! lifetime is not leaked on its own: the type holding the binder (a function pointer, a trait
//! object) is leaked as a whole instead if it can be, and it is an error otherwise (also in a bound
//! or a where predicate, where there is none).
//!
//! An input keeps `'x` as written (it is an *anchor*) where it occurs in a position which
//! constrains it (a reference, a tuple, a path, ..., not a projection) and none of the types from
//! the input down to it is leaked: `&'x str` of `Fn(&'x str, Parser<'x>) -> &'x str` is, so
//! `Parser<'x>` is leaked alone, but `&'x str` of `Fn(Rel2<&'x str>) -> &'x str` is not, as
//! `Rel2` is leaked as a whole.

use syn::*;
use template_quote::quote;
use type_leak::{Leaker, LeakerConfig, NotInternableError, Referrer};

fn leaker() -> Leaker {
    let mut config = LeakerConfig::new();
    config.allow_primitive().allow_std_prelude();
    Leaker::from_config(config)
}

fn s(x: &impl template_quote::ToTokens) -> String {
    quote!(#x).to_string()
}

fn analyze(item: &ItemTrait) -> std::result::Result<Referrer, NotInternableError> {
    let mut leaker = leaker();
    leaker.intern_with(&item.generics, |v| v.visit_item_trait(item))?;
    Ok(leaker.finish())
}

/// The leaked types of a trait with the generics each uses.
fn leaked(item: ItemTrait) -> Vec<(String, String)> {
    let referrer = analyze(&item).unwrap_or_else(|e| panic!("{}", syn::Error::from(e)));
    referrer
        .iter_with_generics()
        .map(|(ty, generics)| (s(ty), s(generics)))
        .collect()
}

fn e(ty: &str, generics: &str) -> (String, String) {
    (ty.to_string(), generics.to_string())
}

/// The message and the first note of the error of a trait.
fn error(item: ItemTrait) -> (String, Option<String>) {
    let err = analyze(&item).expect_err("expected an error");
    (err.to_string(), err.note())
}

/// The note of an error in a bound or a predicate, where there is no type holding the binder.
const NOTE: &str = "`'x` is declared by a `for<..>` binder and used in its output (a return type \
                    or an associated type binding), and no input keeps it as written, so a type \
                    using it cannot be leaked: the copy would have `'x` only in a projection, \
                    which does not constrain it (E0581, E0582)";

/// The note of an error in a type holding the binder, which cannot be leaked as a whole.
const NOTE_OWNED: &str = "`'x` is declared by a `for<..>` binder and used in its output (a return \
                          type or an associated type binding), and no input keeps it as written, \
                          so a type using it cannot be leaked alone, and the type declaring the \
                          binder would be leaked as a whole instead: the copy would have `'x` only \
                          in a projection, which does not constrain it (E0581, E0582)";

#[test]
fn a_lifetime_used_only_in_the_inputs_is_leaked_with_the_type() {
    // A function pointer, a bound, a where predicate and a trait object.
    for item in [
        parse_quote! { trait Tr { fn f(&self, f: for<'x> fn(Rel<'x>)); } },
        parse_quote! { trait Tr { fn f<F: for<'x> Fn(Rel<'x>)>(&self, f: F); } },
        parse_quote! { trait Tr { fn f<F>(&self, f: F) where for<'x> F: Fn(Rel<'x>); } },
        parse_quote! { trait Tr { fn f(&self, f: Box<dyn for<'x> Fn(Rel<'x>)>); } },
        parse_quote! { trait Tr: for<'x> Fn(Rel<'x>) {} },
    ] {
        assert_eq!(leaked(item), [e("Rel < 'x >", "< 'x >")]);
    }
}

#[test]
fn a_return_type_which_does_not_mention_the_lifetime() {
    for item in [
        parse_quote! { trait Tr { fn f(&self, f: for<'x> fn(Rel<'x>) -> u8); } },
        parse_quote! { trait Tr { fn f<F: for<'x> Fn(Rel<'x>) -> u8>(&self, f: F); } },
        parse_quote! { trait Tr { fn f<F: for<'x> Fn(Rel<'x>) -> &'static u8>(&self, f: F); } },
        // The lifetime of the binder inside the output is another one.
        parse_quote! { trait Tr { fn f(&self, f: for<'x> fn(Rel<'x>) -> for<'y> fn(&'y u8)); } },
        parse_quote! {
            trait Tr { fn f<F: for<'x> Fn(Rel<'x>) -> Box<dyn for<'y> Fn(&'y u8)>>(&self, f: F); }
        },
    ] {
        assert_eq!(leaked(item), [e("Rel < 'x >", "< 'x >")]);
    }
    // An associated type binding which does not mention it, and a bound on an associated type
    // (a where predicate, whose lifetime is declared for it).
    assert_eq!(
        leaked(parse_quote! {
            trait Tr { fn f<I: for<'x> ::p::Tr2<Rel<'x>, Item = u8>>(&self, i: I); }
        }),
        [e("Rel < 'x >", "< 'x >")]
    );
    assert_eq!(
        leaked(parse_quote! {
            trait Tr { fn f<I>(&self, i: I) where for<'x> Rel<'x>: ::p::Tr2<'x, Item = u8>; }
        }),
        [e("Rel < 'x >", "< 'x >")]
    );
    assert_eq!(
        leaked(parse_quote! {
            trait Tr { fn f<I: for<'x> Iterator<Item: ::p::Tr2<'x, Rel<'x>>>>(&self, i: I); }
        }),
        [e("Rel < 'x >", "< 'x >")]
    );
}

#[test]
fn a_bound_whose_output_mentions_the_lifetime_is_an_error() {
    for item in [
        parse_quote! { trait Tr { fn f<F: for<'x> Fn(Rel<'x>) -> &'x u8>(&self, f: F); } },
        // `Rel<'x>` in the output itself, with no input to keep `'x`.
        parse_quote! { trait Tr { fn f<F: for<'x> Fn() -> Rel<'x>>(&self, f: F); } },
        parse_quote! { trait Tr { fn f<F: for<'x> Fn(u8) -> Rel<'x>>(&self, f: F); } },
        // An associated type binding.
        parse_quote! { trait Tr { fn f<I: for<'x> Iterator<Item = Rel<'x>>>(&self, i: I); } },
        parse_quote! { trait Tr { fn f<I: for<'x> ::p::Tr2<u8, Item = Rel<'x>>>(&self, i: I); } },
        // A where predicate, with the binder on the predicate or on the bound.
        parse_quote! { trait Tr { fn f<I>(&self, i: I) where for<'x> I: Iterator<Item = Rel<'x>>; } },
        parse_quote! {
            trait Tr { fn f<F>(&self, f: F) where F: for<'x> Fn(Rel<'x>) -> &'x u8; }
        },
        parse_quote! {
            trait Tr { fn f<F>(&self, f: F) where for<'x> F: Fn(Rel<'x>) -> &'x u8; }
        },
        parse_quote! {
            trait Tr { fn f<I>(&self, i: I) where for<'x> Rel<'x>: ::p::Tr2<u8, Item = &'x u8>; }
        },
        // A supertrait, and the bound of the trait.
        parse_quote! { trait Tr: for<'x> Fn(Rel<'x>) -> &'x u8 {} },
        parse_quote! { trait Tr<F: for<'x> Fn(Rel<'x>) -> &'x u8> {} },
    ] {
        let item: ItemTrait = item;
        let (message, note) = error(item.clone());
        assert_eq!(message, "use absolute path: 'Rel < 'x >'", "{}", s(&item));
        assert_eq!(note.as_deref(), Some(NOTE), "{}", s(&item));
    }

    // The error is reported with the note and the lifetime as the cause.
    let err = analyze(&parse_quote! {
        trait Tr { fn f<F: for<'x> Fn(Rel<'x>, Rel2) -> &'x u8>(&self, f: F); }
    })
    .err()
    .unwrap();
    assert_eq!(
        syn::Error::from(err).to_string(),
        format!(
            "use absolute path: 'Rel < 'x >'\n\n  = note: {}\n  = note: it cannot be leaked as a \
             whole because of `'x`\n",
            NOTE
        )
    );
    // The same type is not an error where the output does not mention the lifetime.
    assert_eq!(
        leaked(parse_quote! {
            trait Tr { fn f<F: for<'x> Fn(Rel<'x>, Rel2) -> u8>(&self, f: F); }
        }),
        [e("Rel < 'x >", "< 'x >"), e("Rel2", "")]
    );
}

#[test]
fn the_type_holding_the_binder_is_leaked_as_a_whole() {
    // A function pointer, with the lifetime in the return type or in the inputs.
    assert_eq!(
        leaked(parse_quote! { trait Tr { fn f(&self, f: for<'x> fn(Rel<'x>) -> &'x u8); } }),
        [e("for < 'x > fn (Rel < 'x >) -> & 'x u8", "")]
    );
    // `Rel<'x>` is used by the inputs only through itself here (an anchor would keep `'x`).
    assert_eq!(
        leaked(parse_quote! { trait Tr { fn f(&self, f: for<'x> fn(Rel<'x>, u8) -> Rel<'x>); } }),
        [e("for < 'x > fn (Rel < 'x > , u8) -> Rel < 'x >", "")]
    );
    // What else is in it is covered, so it is not leaked on its own.
    assert_eq!(
        leaked(parse_quote! {
            trait Tr { fn f(&self, f: for<'x> fn(Rel<'x>, Rel2) -> &'x u8); }
        }),
        [e("for < 'x > fn (Rel < 'x > , Rel2) -> & 'x u8", "")]
    );
    // The generics it uses are those of the whole.
    assert_eq!(
        leaked(parse_quote! {
            trait Tr<T> { fn f(&self, f: for<'x> fn(Rel<'x>, T) -> &'x u8); }
        }),
        [e("for < 'x > fn (Rel < 'x > , T) -> & 'x u8", "< T >")]
    );
    // A trait object, with the lifetime in the return type or in an associated type binding.
    assert_eq!(
        leaked(parse_quote! {
            trait Tr { fn f(&self, f: Box<dyn for<'x> Fn(Rel<'x>) -> &'x u8>); }
        }),
        [e("dyn for < 'x > Fn (Rel < 'x >) -> & 'x u8", "")]
    );
    assert_eq!(
        leaked(parse_quote! {
            trait Tr { fn f(&self, f: Box<dyn for<'x> Iterator<Item = Rel<'x>>>); }
        }),
        [e("dyn for < 'x > Iterator < Item = Rel < 'x > >", "")]
    );
    // A trait object in a bound is a type like any other.
    assert_eq!(
        leaked(parse_quote! {
            trait Tr { fn f<F: Fn(Box<dyn for<'x> Fn(Rel<'x>) -> &'x u8>)>(&self, f: F); }
        }),
        [e("dyn for < 'x > Fn (Rel < 'x >) -> & 'x u8", "")]
    );
    // The fields of a struct.
    let item: ItemStruct = parse_quote! { struct S { f: for<'x> fn(Rel<'x>) -> &'x u8 } };
    let mut leaker = leaker();
    leaker
        .intern_with(&item.generics, |v| v.visit_item_struct(&item))
        .unwrap();
    let referrer = leaker.finish();
    assert_eq!(
        referrer.iter().map(s).collect::<Vec<_>>(),
        ["for < 'x > fn (Rel < 'x >) -> & 'x u8"]
    );
}

#[test]
fn the_type_holding_the_binder_has_the_generics_it_uses() {
    // The lifetime `'x` is declared by the binder, so it is not among the generics of the whole,
    // whereas the generics of the trait are (with the outlives bound `Rel<'a, 'x, T>` implies).
    assert_eq!(
        leaked(parse_quote! {
            trait Tr<'a, T> { fn f(&self, f: for<'x> fn(Rel<'a, 'x, T>) -> &'x T); }
        }),
        [e(
            "for < 'x > fn (Rel < 'a , 'x , T >) -> & 'x T",
            "< 'a , T : 'a >"
        )]
    );
    // A return type which mentions only a lifetime of the trait leaves `'x` among the generics.
    assert_eq!(
        leaked(parse_quote! {
            trait Tr<'a, T> { fn f(&self, f: for<'x> fn(Rel<'a, 'x, T>) -> &'a T); }
        }),
        [e("Rel < 'a , 'x , T >", "< 'a , 'x , T : 'a >")]
    );
    // One of the lifetimes of the binder is enough.
    assert_eq!(
        leaked(parse_quote! {
            trait Tr { fn f(&self, f: for<'x, 'y> fn(Rel<'x, 'y>) -> &'y u8); }
        }),
        [e("for < 'x , 'y > fn (Rel < 'x , 'y >) -> & 'y u8", "")]
    );
    // A reference to a trait object with a lifetime of the trait is leaked as a whole.
    assert_eq!(
        leaked(parse_quote! {
            trait Tr<'a> { fn f(&self, f: &'a dyn for<'x> Fn(Rel<'x>) -> &'x u8); }
        }),
        [e(
            "& 'a dyn for < 'x > Fn (Rel < 'x >) -> & 'x u8",
            "< 'a >"
        )]
    );
    // A type macro mentioning the lifetime in the output (its tokens are matched by name).
    assert_eq!(
        error(parse_quote! {
            trait Tr { fn f<F: for<'x> Fn(Rel<'x>) -> ::p::m!['x]>(&self, f: F); }
        })
        .1
        .as_deref(),
        Some(NOTE)
    );
}

#[test]
fn the_binders_are_independent() {
    // Only the output of the binder declaring the lifetime counts, also if another one has the
    // same name in turn.
    let mut found = leaked(parse_quote! {
        trait Tr {
            fn f(&self, a: for<'x> fn(Rel<'x>) -> &'x u8, b: for<'x> fn(Rel<'x>));
        }
    });
    found.sort();
    assert_eq!(
        found,
        [
            e("Rel < 'x >", "< 'x >"),
            e("for < 'x > fn (Rel < 'x >) -> & 'x u8", "")
        ]
    );
    // The inner binder declares `'y`, and `'x` is the outer one's, whose output has no lifetime.
    assert_eq!(
        leaked(parse_quote! {
            trait Tr { fn f(&self, f: for<'x> fn(for<'y> fn(Rel<'x, 'y>) -> &'x u8)); }
        }),
        [e("Rel < 'x , 'y >", "< 'x , 'y >")]
    );
    // Rel is not leaked alone as it mentions `'y`, so the inner function pointer is leaked as a
    // whole, with `'x` among its generics.
    assert_eq!(
        leaked(parse_quote! {
            trait Tr { fn f(&self, f: for<'x> fn(for<'y> fn(Rel<'x, 'y>) -> &'y u8)); }
        }),
        [e("for < 'y > fn (Rel < 'x , 'y >) -> & 'y u8", "< 'x >")]
    );
    // And the outer one if its own output mentions `'x`.
    assert_eq!(
        leaked(parse_quote! {
            trait Tr { fn f(&self, f: for<'x> fn(for<'y> fn(Rel<'x, 'y>) -> &'y u8) -> &'x u8); }
        }),
        [e(
            "for < 'x > fn (for < 'y > fn (Rel < 'x , 'y >) -> & 'y u8) -> & 'x u8",
            ""
        )]
    );
}

#[test]
fn the_type_holding_the_binder_which_cannot_be_leaked_is_an_error() {
    // A function pointer with an elided lifetime cannot be leaked as a whole, so `Rel<'x>` has to
    // be, which cannot be leaked alone: it is an error, not a copy rustc rejects.
    assert_eq!(
        error(parse_quote! {
            trait Tr { fn f(&self, f: for<'x> fn(Rel<'x>, &u8) -> &'x u8); }
        }),
        (
            "use absolute path: 'Rel < 'x >'".to_string(),
            Some(NOTE_OWNED.to_string())
        )
    );
    let err = analyze(&parse_quote! {
        trait Tr { fn f(&self, f: for<'x> fn(Rel<'x>, &u8) -> &'x u8); }
    })
    .err()
    .unwrap();
    assert_eq!(
        syn::Error::from(err).to_string(),
        format!(
            "use absolute path: 'Rel < 'x >'\n\n  = note: {}\n  = note: it cannot be leaked as a \
             whole because of `& u8`\n",
            NOTE_OWNED
        )
    );
    // `impl Trait` cannot be leaked at all.
    assert_eq!(
        error(parse_quote! {
            trait Tr { fn f(&self, f: impl for<'x> Fn(Rel<'x>) -> &'x u8); }
        }),
        (
            "use absolute path: 'Rel < 'x >'".to_string(),
            Some(NOTE_OWNED.to_string())
        )
    );
    // If the return type does not mention the lifetime, `Rel<'x>` is leaked on its own, whatever
    // the rest of the function pointer is.
    assert_eq!(
        leaked(parse_quote! {
            trait Tr { fn f(&self, f: for<'x> fn(Rel<'x>, &u8) -> u8); }
        }),
        [e("Rel < 'x >", "< 'x >")]
    );
    assert_eq!(
        leaked(parse_quote! {
            trait Tr { fn f(&self, f: impl for<'x> Fn(Rel<'x>) -> u8); }
        }),
        [e("Rel < 'x >", "< 'x >")]
    );
    // The conflict is that of the outermost binder: the inner function pointer is leaked as a
    // whole in place of `Rel<'x, 'y>`, and the outer one has to be in its turn.
    assert_eq!(
        error(parse_quote! {
            trait Tr {
                fn f(&self, f: for<'x> fn(for<'y> fn(Rel<'x, 'y>, &u8) -> &'y u8) -> &'x u8);
            }
        })
        .0,
        "use absolute path: 'Rel < 'x , 'y >'"
    );
}

#[test]
fn check_tells_the_type_holding_the_binder_is_interned() {
    let leaker = leaker();
    let check = |ty: Type| leaker.check(&ty, &Generics::default());
    let tokens = |tokens: &[proc_macro2::TokenStream]| -> Vec<String> {
        tokens.iter().map(|t| t.to_string()).collect()
    };
    // `Rel<'x>` alone is leaked (as a component), unless the return type mentions `'x`.
    let result = check(parse_quote!(for<'x> fn(Rel<'x>)));
    assert_eq!(tokens(&result.must), ["Rel < 'x >"]);
    assert!(!result.is_own && result.impossible.is_empty());
    let result = check(parse_quote!(for<'x> fn(Rel<'x>) -> &'x u8));
    assert_eq!(tokens(&result.must), ["Rel < 'x >"]);
    assert!(result.is_own && result.impossible.is_empty());
    // Which is a conflict if it cannot be.
    let result = check(parse_quote!(for<'x> fn(Rel<'x>, &u8) -> &'x u8));
    assert!(result.is_own);
    assert_eq!(tokens(&result.impossible), ["& u8"]);
}

#[test]
fn the_copy_is_rewritten_to_the_type_holding_the_binder() {
    let item: ItemTrait = parse_quote! {
        trait Tr {
            fn f(&self, a: for<'x> fn(Rel<'x>) -> &'x u8, b: for<'x> fn(Rel<'x>));
        }
    };
    let referrer = analyze(&item).unwrap();
    // The entries are ordered by context and then by their tokens.
    assert_eq!(
        referrer.iter().map(s).collect::<Vec<_>>(),
        ["for < 'x > fn (Rel < 'x >) -> & 'x u8", "Rel < 'x >"]
    );
    let mut copy = item.clone();
    syn::visit_mut::VisitMut::visit_item_trait_mut(
        &mut referrer.into_visitor(None, |_, ix| parse_quote!(P<#ix>)),
        &mut copy,
    );
    let expected: ItemTrait = parse_quote! {
        trait Tr {
            fn f(&self, a: P<0usize>, b: for<'x> fn(P<1usize>));
        }
    };
    assert_eq!(s(&copy), s(&expected));
}

#[test]
fn an_input_which_keeps_the_lifetime_anchors_it() {
    // `&'x str` stays where it is written, so `'x` is constrained whatever `Parser<'x>` is
    // replaced by: it is leaked alone, with `'x` among its generics.
    for item in [
        parse_quote! {
            trait Tr { fn f<F: for<'x> Fn(&'x str, Parser<'x>) -> &'x str>(&self, f: F); }
        },
        // The anchor may follow the type using the lifetime.
        parse_quote! {
            trait Tr { fn f<F: for<'x> Fn(Parser<'x>, &'x str) -> &'x str>(&self, f: F); }
        },
        parse_quote! {
            trait Tr { fn f<F>(&self, f: F) where F: for<'x> Fn(&'x str, Parser<'x>) -> &'x str; }
        },
        parse_quote! {
            trait Tr { fn f<F>(&self, f: F) where for<'x> F: Fn(&'x str, Parser<'x>) -> &'x str; }
        },
        parse_quote! { trait Tr: for<'x> Fn(&'x str, Parser<'x>) -> &'x str {} },
        parse_quote! {
            trait Tr { fn f(&self, f: for<'x> fn(&'x str, Parser<'x>) -> &'x str); }
        },
        parse_quote! {
            trait Tr { fn f(&self, f: Box<dyn for<'x> Fn(&'x str, Parser<'x>) -> &'x str>); }
        },
        parse_quote! {
            trait Tr { fn f(&self, f: impl for<'x> Fn(&'x str, Parser<'x>) -> &'x str); }
        },
        // The type using the lifetime is in the output.
        parse_quote! { trait Tr { fn f<F: for<'x> Fn(&'x u8) -> Rel<'x>>(&self, f: F); } },
        parse_quote! { trait Tr { fn f(&self, f: for<'x> fn(&'x u8) -> Rel<'x>); } },
        parse_quote! { trait Tr { fn f(&self, f: Box<dyn for<'x> Fn(&'x u8) -> Rel<'x>>); } },
    ] {
        let item: ItemTrait = item;
        let found = leaked(item.clone());
        assert_eq!(found.len(), 1, "{}", s(&item));
        assert_eq!(found[0].1, "< 'x >", "{}", s(&item));
        assert!(found[0].0 == "Parser < 'x >" || found[0].0 == "Rel < 'x >");
    }
    // An associated type binding of a bound, with the trait arguments as the inputs.
    assert_eq!(
        leaked(parse_quote! {
            trait Tr { fn f<I: for<'x> ::p::Tr2<&'x u8, Item = Rel<'x>>>(&self, i: I); }
        }),
        [e("Rel < 'x >", "< 'x >")]
    );
    assert_eq!(
        leaked(parse_quote! {
            trait Tr {
                fn f(&self, i: Box<dyn for<'x> ::core::ops::Index<&'x u8, Output = Rel<'x>>>);
            }
        }),
        [e("Rel < 'x >", "< 'x >")]
    );
    // A lifetime argument of the trait is an input as well, with no type to be leaked in between.
    assert_eq!(
        leaked(parse_quote! {
            trait Tr { fn f<I: for<'x> ::p::Tr3<'x, Item = Rel<'x>>>(&self, i: I); }
        }),
        [e("Rel < 'x >", "< 'x >")]
    );
    // The bounded type of a predicate is an input of its bounds, but of that bound only.
    assert_eq!(
        leaked(parse_quote! {
            trait Tr { fn f<I>(&self, i: I) where for<'x> &'x u8: ::p::Tr2<u8, Item = Rel<'x>>; }
        }),
        [e("Rel < 'x >", "< 'x >")]
    );
    assert_eq!(
        leaked(parse_quote! {
            trait Tr {
                fn f<I>(&self, i: I)
                where
                    for<'x> I: ::p::Tr2<&'x u8, Item = Rel<'x>> + ::p::Tr3<u8>;
            }
        }),
        [e("Rel < 'x >", "< 'x >")]
    );
    assert_eq!(
        leaked(parse_quote! {
            trait Tr { fn f<I>(&self, i: I) where for<'x> Rel<'x>: ::p::Tr3<'x, Item = &'x u8>; }
        }),
        [e("Rel < 'x >", "< 'x >")]
    );
}

#[test]
fn the_positions_which_constrain_the_lifetime_are_anchors() {
    // A reference, a pointer, an array, a slice, a tuple, a path (its lifetime and type arguments),
    // a parenthesis and a trait object (its arguments and its lifetime bound).
    for input in [
        "&'x str",
        "*const &'x str",
        "[&'x str; 3]",
        "&'static [&'x str]",
        "(u8, &'x str)",
        "(&'x str)",
        "::p::Option<&'x str>",
        "::p::Cow<'x, str>",
        "::p::Option<(u8, &'x str)>",
        "Box<dyn ::p::Tr3<'x>>",
        "Box<dyn ::p::Tr3<&'x str>>",
        "Box<dyn ::p::Tr3<(), Item = u8> + 'x>",
        "Box<dyn Fn(&'x str)>",
    ] {
        let input: Type = parse_str(input).unwrap();
        let item: ItemTrait = parse_quote! {
            trait Tr { fn f<F: for<'x> Fn(#input, Parser<'x>) -> &'x str>(&self, f: F); }
        };
        assert_eq!(
            leaked(item.clone()),
            [e("Parser < 'x >", "< 'x >")],
            "{}",
            s(&item)
        );
    }
}

#[test]
fn an_occurrence_which_does_not_constrain_the_lifetime_is_not_an_anchor() {
    // The copy is rejected unless `Parser<'x>` is not leaked alone, which it cannot be in a bound.
    for input in [
        // A projection.
        "<T as ::p::Tr3<'x>>::A",
        "<T as ::p::Tr3>::A<'x>",
        "T::A<'x>",
        "Self::A<'x>",
        "(u8, <T as ::p::Tr3<'x>>::A)",
        // The associated type binding and the output of a trait object are its output.
        "Box<dyn ::p::Tr3<Item = &'x str>>",
        "Box<dyn Fn() -> &'x str>",
        // Not looked into: a function pointer and a trait object with a binder of its own (a type
        // macro and `impl Trait` neither, but they are not allowed there or leaked as a whole).
        "fn(&'x str)",
        "Box<dyn for<'y> Fn(&'x str, &'y u8)>",
        // The length of an array is an unevaluated constant, which constrains nothing.
        "[u8; { ::p::N::<'x> }]",
    ] {
        let input: Type = parse_str(input).unwrap();
        let item: ItemTrait = parse_quote! {
            trait Tr<T: ::p::Tr3> {
                fn f<F: for<'x> Fn(#input, Parser<'x>) -> &'x str>(&self, f: F);
            }
        };
        let (message, note) = error(item.clone());
        assert_eq!(
            message,
            "use absolute path: 'Parser < 'x >'",
            "{}",
            s(&item)
        );
        assert_eq!(note.as_deref(), Some(NOTE), "{}", s(&item));
    }
    // Only an occurrence in the output.
    for item in [
        parse_quote! { trait Tr { fn f<F: for<'x> Fn(Parser<'x>) -> &'x str>(&self, f: F); } },
        parse_quote! { trait Tr { fn f<F: for<'x> Fn(u8) -> Rel<'x>>(&self, f: F); } },
        parse_quote! {
            trait Tr { fn f<F: for<'x> Fn(u8, &'static str) -> (&'x str, Rel<'x>)>(&self, f: F); }
        },
        // In the output of another bound of the predicate.
        parse_quote! {
            trait Tr {
                fn f<I>(&self, i: I)
                where
                    for<'x> I: ::p::Tr2<u8, Item = Rel<'x>> + ::p::Tr3<&'x u8>;
            }
        },
    ] {
        let item: ItemTrait = item;
        assert_eq!(error(item.clone()).1.as_deref(), Some(NOTE), "{}", s(&item));
    }
}

#[test]
fn an_occurrence_in_a_type_leaked_as_a_whole_is_not_an_anchor() {
    // `Rel2` is a relative path, so `Rel2<&'x str>` is leaked as a whole, whatever it holds.
    assert_eq!(
        error(parse_quote! {
            trait Tr { fn f<F: for<'x> Fn(Rel2<&'x str>) -> &'x str>(&self, f: F); }
        }),
        (
            "use absolute path: 'Rel2 < & 'x str >'".to_string(),
            Some(NOTE.to_string())
        )
    );
    // Also deeper, and where the type using the lifetime precedes it.
    for item in [
        parse_quote! {
            trait Tr { fn f<F: for<'x> Fn(::p::Wrap<Rel2<&'x str>>) -> &'x str>(&self, f: F); }
        },
        parse_quote! {
            trait Tr { fn f<F: for<'x> Fn(Parser<'x>, Rel2<&'x str>) -> &'x str>(&self, f: F); }
        },
        parse_quote! {
            trait Tr { fn f<F: for<'x> Fn(Rel2<&'x str>, Parser<'x>) -> &'x str>(&self, f: F); }
        },
        parse_quote! {
            trait Tr {
                fn f<F: for<'x> Fn((u8, ::p::Wrap<Rel2<&'x str>>), Parser<'x>) -> &'x str>(&self, f: F);
            }
        },
    ] {
        let item: ItemTrait = item;
        assert_eq!(error(item.clone()).1.as_deref(), Some(NOTE), "{}", s(&item));
    }
    // A function pointer holding it is leaked as a whole instead.
    assert_eq!(
        leaked(parse_quote! {
            trait Tr { fn f(&self, f: for<'x> fn(Rel2<&'x str>, Parser<'x>) -> &'x str); }
        }),
        [e(
            "for < 'x > fn (Rel2 < & 'x str > , Parser < 'x >) -> & 'x str",
            ""
        )]
    );
    // Another input keeps the lifetime: both are leaked alone.
    let mut found = leaked(parse_quote! {
        trait Tr {
            fn f<F: for<'x> Fn(Parser<'x>, Rel2<&'x str>, &'x u8) -> &'x str>(&self, f: F);
        }
    });
    found.sort();
    assert_eq!(
        found,
        [
            e("Parser < 'x >", "< 'x >"),
            e("Rel2 < & 'x str >", "< 'x >")
        ]
    );
}

#[test]
fn the_types_on_the_way_to_the_anchor_are_never_leaked() {
    // The tuple is used by many methods, so interning it once would cover all the uses of `Rel2`
    // and `Parser<'x>` (as it does where it uses no lifetime of a binder), but it mentions `'x`, so
    // it is never a type to intern: its components are interned instead, and it keeps `&'x str`.
    let mut methods = Vec::new();
    for n in 0..3 {
        let name = Ident::new(&format!("f{}", n), proc_macro2::Span::call_site());
        methods.push(quote! {
            fn #name<F: for<'x> Fn((&'x str, Parser<'x>, Rel2)) -> &'x str>(&self, f: F);
        });
    }
    let item: ItemTrait = parse_quote! { trait Tr { #(#methods)* } };
    let referrer = analyze(&item).unwrap();
    let found: Vec<_> = referrer
        .iter_with_generics()
        .map(|(ty, generics)| (s(ty), s(generics)))
        .collect();
    assert_eq!(
        found,
        [e("Parser < 'x >", "< 'x >"), e("Rel2", "")],
        "{:?}",
        found
    );
    // The same tuple without the lifetime is interned once.
    let mut methods = Vec::new();
    for n in 0..3 {
        let name = Ident::new(&format!("f{}", n), proc_macro2::Span::call_site());
        methods.push(quote! {
            fn #name<F: for<'x> Fn((u8, Parser, Rel2), &'x str) -> &'x str>(&self, f: F);
        });
    }
    let item: ItemTrait = parse_quote! { trait Tr { #(#methods)* } };
    assert_eq!(
        analyze(&item).unwrap().iter().map(s).collect::<Vec<_>>(),
        ["(u8 , Parser , Rel2)"]
    );
    // And the copy keeps `&'x str` where it was.
    let item: ItemTrait = parse_quote! {
        trait Tr {
            fn f<F: for<'x> Fn((&'x str, Parser<'x>, Rel2)) -> &'x str>(&self, f: F);
            fn g<F: for<'x> Fn((&'x str, Parser<'x>, Rel2)) -> &'x str>(&self, f: F);
        }
    };
    let referrer = analyze(&item).unwrap();
    let mut copy = item.clone();
    syn::visit_mut::VisitMut::visit_item_trait_mut(
        &mut referrer.into_visitor(None, |ty, _| {
            let ty = s(&ty);
            parse_quote!(P<#ty>)
        }),
        &mut copy,
    );
    let expected: ItemTrait = parse_quote! {
        trait Tr {
            fn f<F: for<'x> Fn((&'x str, P<"Parser < 'x >">, P<"Rel2">)) -> &'x str>(&self, f: F);
            fn g<F: for<'x> Fn((&'x str, P<"Parser < 'x >">, P<"Rel2">)) -> &'x str>(&self, f: F);
        }
    };
    assert_eq!(s(&copy), s(&expected));
}

#[test]
fn the_anchor_is_checked_for_each_binder() {
    // Nested binders: each takes its own anchors, and a binder in the inputs is not an anchor.
    assert_eq!(
        leaked(parse_quote! {
            trait Tr {
                fn f<F: for<'x> Fn(for<'y> fn(&'y u8, Rel<'y>) -> &'y u8, &'x str, Parser<'x>) -> &'x str>(&self, f: F);
            }
        }),
        [e("Rel < 'y >", "< 'y >"), e("Parser < 'x >", "< 'x >")]
    );
    assert_eq!(
        leaked(parse_quote! {
            trait Tr {
                fn f(&self, f: for<'x> fn(for<'y> fn(&'y u8, Rel<'x, 'y>) -> &'y u8, &'x str) -> &'x str);
            }
        }),
        [e("Rel < 'x , 'y >", "< 'x , 'y >")]
    );
    // The inner function pointer holds `'x` in its inputs, which is not looked into.
    assert_eq!(
        error(parse_quote! {
            trait Tr {
                fn f<F: for<'x> Fn(fn(&'x u8), Parser<'x>) -> &'x str>(&self, f: F);
            }
        })
        .1
        .as_deref(),
        Some(NOTE)
    );
    // A deep nesting takes a time proportional to its size (a binder inside a binder which is
    // looked into is not looked into again).
    let mut ty: Type = parse_quote!(Rel<'a0>);
    for n in (0..60).rev() {
        let lifetime = Lifetime::new(&format!("'a{}", n), proc_macro2::Span::call_site());
        ty = parse_quote!(for<#lifetime> fn(&#lifetime u8, #ty) -> &#lifetime u8);
    }
    let item: ItemTrait = parse_quote! { trait Tr { fn f(&self, f: #ty); } };
    assert_eq!(leaked(item), [e("Rel < 'a0 >", "< 'a0 >")]);
}

#[test]
fn check_tells_the_anchored_type_is_interned_alone() {
    let leaker = leaker();
    let check = |ty: Type| leaker.check(&ty, &Generics::default());
    let tokens = |tokens: &[proc_macro2::TokenStream]| -> Vec<String> {
        tokens.iter().map(|t| t.to_string()).collect()
    };
    // `Rel<'x>` stays a component of the function pointer, which is not interned.
    let result = check(parse_quote!(for<'x> fn(&'x u8) -> Rel<'x>));
    assert_eq!(tokens(&result.must), ["Rel < 'x >"]);
    assert!(!result.is_own && result.impossible.is_empty());
    // The anchor is lost if it is in a type interned itself.
    let result = check(parse_quote!(for<'x> fn(Rel2<&'x u8>) -> Rel<'x>));
    assert!(result.is_own && result.impossible.is_empty());
}

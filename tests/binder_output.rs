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
//! constrains it (a reference, a tuple, a path, an associated type binding or the lifetime bound
//! of a trait object, a function pointer, ..., anything but a projection) and none of the types
//! from the input down to it is leaked: `&'x str` of `Fn(&'x str, Parser<'x>) -> &'x str` is, so
//! `Parser<'x>` is leaked alone, but `&'x str` of `Fn(Rel2<&'x str>) -> &'x str` is not, as
//! `Rel2` is leaked as a whole. Which positions constrain was checked with rustc (stable and
//! nightly, from 1.61 on), for the inputs of function pointers and of `Fn(..)` and other bounds.

mod common;

use common::check_in;
use syn::punctuated::Punctuated;
use syn::*;
use template_quote::quote;
use type_leak::{ContextId, Leaker, LeakerConfig, NotInternableError, Referrer};

/// A Leaker which allows `::p`, standing for an external crate which the Referrer's crate is known
/// to have as the Leaker's: its types may be an anchor. The types of the other external crates may
/// be replaced by the Referrer, so they are not (see `external_crate_types_do_not_anchor`).
fn leaker() -> Leaker {
    leaker_with(&[parse_quote!(::p)])
}

fn leaker_with(allowed: &[Path]) -> Leaker {
    let mut config = LeakerConfig::new();
    config.allow_primitive().allow_std_prelude();
    config.allowed_paths.extend(allowed.iter().cloned());
    Leaker::from_config(config)
}

fn s(x: &impl template_quote::ToTokens) -> String {
    quote!(#x).to_string()
}

fn analyze(item: &ItemTrait) -> std::result::Result<Referrer, NotInternableError> {
    analyze_with(leaker(), item)
}

fn analyze_with(
    leaker: Leaker,
    item: &ItemTrait,
) -> std::result::Result<Referrer, NotInternableError> {
    analyze_in(leaker, item).map(|(referrer, _)| referrer)
}

/// Like `analyze_with()`, with the top-level context of the trait: the scope of its copy.
fn analyze_in(
    mut leaker: Leaker,
    item: &ItemTrait,
) -> std::result::Result<(Referrer, ContextId), NotInternableError> {
    let context = leaker.intern_trait(item, true)?;
    Ok((leaker.finish(), context))
}

/// The leaked types of a trait with the generics each uses.
fn leaked(item: ItemTrait) -> Vec<(String, String)> {
    leaked_by(leaker(), item)
}

/// The leaked types of a trait with the generics each uses, by the given Leaker. An optional entry
/// (see `Referrer::must()`) is preceded by `?`.
fn leaked_by(leaker: Leaker, item: ItemTrait) -> Vec<(String, String)> {
    let referrer =
        analyze_with(leaker, &item).unwrap_or_else(|e| panic!("{}", syn::Error::from(e)));
    referrer
        .iter_with_generics()
        .enumerate()
        .map(|(ix, (ty, generics))| {
            let mark = if referrer.must(ix) { "" } else { "? " };
            (format!("{}{}", mark, s(ty)), s(generics))
        })
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

/// What to write, for the error of `NOTE`: keep an input, or leak nothing.
const HELP: &str = "keep an input that names `'x` as written (for example `&'x u8`), or write \
                    `Rel<'x>` with an absolute path";

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
             whole because of `'x`\n  = help: {}\n",
            NOTE, HELP
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
    leaker.intern_struct(&item, true).unwrap();
    let referrer = leaker.finish();
    assert_eq!(
        referrer.iter().map(s).collect::<Vec<_>>(),
        ["for < 'x > fn (Rel < 'x >) -> & 'x u8"]
    );
}

#[test]
fn the_type_holding_the_binder_has_the_generics_it_uses() {
    // The lifetime `'x` is declared by the binder, so it is not among the generics of the whole,
    // whereas the generics of the trait are (`Rel<'a, 'x, T>` is not assumed to imply `T: 'a`: a
    // path with a lifetime argument records no outlives bound, see `tests/outlives_bounds.rs`).
    assert_eq!(
        leaked(parse_quote! {
            trait Tr<'a, T> { fn f(&self, f: for<'x> fn(Rel<'a, 'x, T>) -> &'x T); }
        }),
        [e(
            "for < 'x > fn (Rel < 'a , 'x , T >) -> & 'x T",
            "< 'a , T >"
        )]
    );
    // A return type which mentions only a lifetime of the trait leaves `'x` among the generics.
    assert_eq!(
        leaked(parse_quote! {
            trait Tr<'a, T> { fn f(&self, f: for<'x> fn(Rel<'a, 'x, T>) -> &'a T); }
        }),
        [e("Rel < 'a , 'x , T >", "< 'a , 'x , T >")]
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
    // A function pointer with a trait object under an elided reference cannot be leaked as a
    // whole, so `Rel<'x>` has to be, which cannot be leaked alone: it is an error, not a copy
    // rustc rejects. (An elided lifetime of its own would not stop it, see
    // `tests/elided_output.rs`.)
    assert_eq!(
        error(parse_quote! {
            trait Tr { fn f(&self, f: for<'x> fn(Rel<'x>, &dyn ::core::fmt::Debug) -> &'x u8); }
        }),
        (
            "use absolute path: 'Rel < 'x >'".to_string(),
            Some(NOTE_OWNED.to_string())
        )
    );
    let err = analyze(&parse_quote! {
        trait Tr { fn f(&self, f: for<'x> fn(Rel<'x>, &dyn ::core::fmt::Debug) -> &'x u8); }
    })
    .err()
    .unwrap();
    assert_eq!(
        syn::Error::from(err).to_string(),
        format!(
            "use absolute path: 'Rel < 'x >'\n\n  = note: {}\n  = note: it cannot be leaked as a \
             whole because of `dyn :: core :: fmt :: Debug`\n  = note: the lifetime of the trait \
             object defaults to that of the enclosing reference or path\n  = help: {}\n",
            NOTE_OWNED, HELP
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
                fn f(
                    &self,
                    f: for<'x> fn(
                        for<'y> fn(Rel<'x, 'y>, &dyn ::core::fmt::Debug) -> &'y u8
                    ) -> &'x u8,
                );
            }
        })
        .0,
        "use absolute path: 'Rel < 'x , 'y >'"
    );
    // With an elided lifetime of its own, the outer function pointer is leaked as a whole.
    assert_eq!(
        leaked(parse_quote! {
            trait Tr {
                fn f(&self, f: for<'x> fn(for<'y> fn(Rel<'x, 'y>, &u8) -> &'y u8) -> &'x u8);
            }
        }),
        [e(
            "for < 'x > fn (for < 'y > fn (Rel < 'x , 'y > , & u8) -> & 'y u8) -> & 'x u8",
            ""
        )]
    );
}

#[test]
fn check_tells_the_type_holding_the_binder_is_interned() {
    let mut leaker = leaker();
    let mut check = |ty: Type| check_in(&mut leaker, &ty, &Generics::default());
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
    // The elided lifetime of the function pointer is its own, so it does not stop it from being
    // leaked as a whole.
    let result = check(parse_quote!(for<'x> fn(Rel<'x>, &u8) -> &'x u8));
    assert_eq!(tokens(&result.must), ["Rel < 'x >"]);
    assert!(result.is_own && result.impossible.is_empty());
    // Which is a conflict if it cannot be: a trait object under an elided reference takes its
    // lifetime from it.
    let result = check(parse_quote!(
        for<'x> fn(Rel<'x>, &dyn ::core::fmt::Debug) -> &'x u8
    ));
    assert!(result.is_own);
    assert_eq!(tokens(&result.impossible), ["dyn :: core :: fmt :: Debug"]);
}

#[test]
fn the_copy_is_rewritten_to_the_type_holding_the_binder() {
    let item: ItemTrait = parse_quote! {
        trait Tr {
            fn f(&self, a: for<'x> fn(Rel<'x>) -> &'x u8, b: for<'x> fn(Rel<'x>));
        }
    };
    let (referrer, context) = analyze_in(leaker(), &item).unwrap();
    // The entries are ordered by context and then by their tokens.
    assert_eq!(
        referrer.iter().map(s).collect::<Vec<_>>(),
        ["for < 'x > fn (Rel < 'x >) -> & 'x u8", "Rel < 'x >"]
    );
    let mut copy = item.clone();
    syn::visit_mut::VisitMut::visit_item_trait_mut(
        &mut referrer.visitor(context, |_, ix| parse_quote!(P<#ix>)),
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
    // a parenthesis, a trait object (its arguments, its associated type bindings, and its lifetime
    // bound), and a function pointer (its inputs and its output). Also the inputs and the output of
    // `Fn(..)`, as written in a trait object (a nested one has no effect on what the outer binder
    // constrains: `'x` is the outer one's).
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
        // An associated type binding, wherever it is (of the trait object, or of one in it).
        "Box<dyn ::p::Tr3<Item = &'x str>>",
        "Box<dyn Iterator<Item = &'x u8>>",
        "Box<dyn Iterator<Item = (u8, &'x u8)> + Send>",
        "Option<Box<dyn Iterator<Item = &'x u8>>>",
        "Box<dyn Iterator<Item = Box<dyn Iterator<Item = &'x u8>>>>",
        "Box<dyn Iterator<Item = fn(&'x u8)>>",
        // The lifetime bound.
        "Box<dyn ::p::Tr3 + 'x>",
        "Box<dyn ::p::Tr3 + Send + 'x>",
        "Box<dyn Iterator<Item = Box<dyn ::p::Tr3 + 'x>>>",
        "&'x dyn ::p::Tr3",
        // The output of `Fn(..)` in a trait object, and what is nested in it.
        "Box<dyn Fn() -> &'x str>",
        "Box<dyn Fn(u8) -> Box<dyn Iterator<Item = &'x u8>>>",
        "Box<dyn Fn(Box<dyn Iterator<Item = &'x u8>>)>",
        "Box<dyn Fn() -> Box<dyn Fn(&'x u8)>>",
        // A function pointer, with an implicit or an explicit binder of its own.
        "fn(&'x str)",
        "fn() -> &'x str",
        "fn(fn(&'x str))",
        "for<'y> fn(&'x str, &'y u8)",
        "for<'y> fn(&'y u8) -> &'x u8",
        // A trait object with a binder of its own, in its arguments, its bindings and its output.
        "Box<dyn for<'y> Fn(&'x str, &'y u8)>",
        "Box<dyn for<'y> Fn(&'y u8) -> &'x str>",
        "Box<dyn for<'y> ::p::Tr3<Item = &'x str>>",
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
        // Also in an associated type binding, the output of `Fn(..)`, or its arguments, a function
        // pointer, and a trait object with a binder of its own (rustc: E0581).
        "Box<dyn ::p::Tr3<Item = <T as ::p::Tr3<'x>>::A>>",
        "Box<dyn Iterator<Item = (u8, T::A<'x>)>>",
        "Box<dyn Fn(<T as ::p::Tr3<'x>>::A)>",
        "Box<dyn Fn() -> <T as ::p::Tr3<'x>>::A>",
        "fn(<T as ::p::Tr3<'x>>::A)",
        "fn() -> <T as ::p::Tr3<'x>>::A",
        "Box<dyn for<'y> Fn(<T as ::p::Tr3<'x>>::A, &'y u8)>",
        "Box<dyn for<'y> Fn(&'y u8) -> <T as ::p::Tr3<'x>>::A>",
        // A binder of its own which declares the same name has another lifetime (rustc rejects it,
        // but a macro may expand to it).
        "for<'x> fn(&'x str)",
        "Box<dyn for<'x> Fn(&'x str)>",
        // The length of an array is an unevaluated constant, which constrains nothing (`impl Trait`
        // and type macros are not looked into either, but they are not allowed there, or are leaked
        // as a whole).
        "[u8; { ::p::N::<'x> }]",
        "Box<dyn Iterator<Item = [u8; { ::p::N::<'x> }]>>",
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
        // The associated type bindings of the bound itself are its output, not an input: rustc
        // rejects `T: for<'x> Tr<u8, Output = &'x u8>` (E0582), whatever the bounded type is.
        parse_quote! {
            trait Tr { fn f<I: for<'x> ::p::Tr2<u8, Item = (&'x u8, Rel<'x>)>>(&self, i: I); }
        },
        parse_quote! {
            trait Tr {
                fn f<I>(&self, i: I) where for<'x> I: ::p::Tr2<u8, Item = (&'x u8, Rel<'x>)>;
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
fn an_associated_type_binding_or_a_lifetime_bound_in_an_input_anchors_the_lifetime() {
    // rustc counts what is in a trait object (an associated type binding, the output of `Fn(..)`,
    // the lifetime bound), a function pointer, and in a trait object with a binder of its own, as
    // constraining the lifetime like a type argument: in a bound, a where predicate, a function
    // pointer, a trait object, and the bounded type of a predicate.
    for anchor in [
        "Box<dyn Iterator<Item = &'x u8>>",
        "Box<dyn ::p::Tr3 + 'x>",
        "&'x dyn ::p::Tr3",
        "Box<dyn Fn() -> &'x u8>",
        "Box<dyn Fn(Box<dyn Iterator<Item = &'x u8>>)>",
        "fn(&'x u8) -> u8",
        "Box<dyn for<'y> Fn(&'x u8, &'y u8)>",
    ] {
        let anchor: Type = parse_str(anchor).unwrap();
        for item in [
            parse_quote! {
                trait Tr { fn f<F: for<'x> Fn(#anchor, Parser<'x>) -> &'x u8>(&self, f: F); }
            },
            parse_quote! {
                trait Tr {
                    fn f<F>(&self, f: F) where F: for<'x> Fn(#anchor, Parser<'x>) -> &'x u8;
                }
            },
            parse_quote! {
                trait Tr {
                    fn f<F>(&self, f: F) where for<'x> F: Fn(#anchor, Parser<'x>) -> &'x u8;
                }
            },
            parse_quote! { trait Tr: for<'x> Fn(#anchor, Parser<'x>) -> &'x u8 {} },
            parse_quote! {
                trait Tr { fn f(&self, f: for<'x> fn(#anchor, Parser<'x>) -> &'x u8); }
            },
            parse_quote! {
                trait Tr {
                    fn f(&self, f: Box<dyn for<'x> Fn(#anchor, Parser<'x>) -> &'x u8>);
                }
            },
            parse_quote! {
                trait Tr {
                    fn f<I>(&self, i: I) where for<'x> #anchor: ::p::Tr2<u8, Item = Parser<'x>>;
                }
            },
            // The type using the lifetime precedes the anchor, or is in the output.
            parse_quote! {
                trait Tr { fn f<F: for<'x> Fn(Parser<'x>, #anchor) -> &'x u8>(&self, f: F); }
            },
            parse_quote! {
                trait Tr { fn f<F: for<'x> Fn(#anchor) -> Parser<'x>>(&self, f: F); }
            },
        ] {
            let item: ItemTrait = item;
            assert_eq!(
                leaked(item.clone()),
                [e("Parser < 'x >", "< 'x >")],
                "{}",
                s(&item)
            );
        }
    }
    // The bound's own associated type bindings are not inputs, whatever the trait arguments are:
    // `T: for<'x> Tr<&'x u8, Output = &'x u8>` is accepted by rustc through `&'x u8`.
    assert_eq!(
        leaked(parse_quote! {
            trait Tr {
                fn f<I: for<'x> ::p::Tr2<Box<dyn Iterator<Item = &'x u8>>, Item = Rel<'x>>>(&self, i: I);
            }
        }),
        [e("Rel < 'x >", "< 'x >")]
    );
}

#[test]
fn a_reference_to_a_trait_object_which_is_leaked_as_a_whole_does_not_anchor_the_lifetime() {
    // `&'x dyn RelTr` is leaked as a whole (the default of its trait object is computed from the
    // trait, which the Leaker cannot see), so `'x` is only in a projection of the copy, like in a
    // path with a lifetime and a trait object: the reference is no anchor.
    for item in [
        parse_quote! { trait Tr { fn f<F: for<'x> Fn(&'x dyn RelTr) -> &'x u8>(&self, f: F); } },
        parse_quote! { trait Tr { fn f<F>(&self, f: F) where for<'x> F: Fn(&'x dyn RelTr) -> &'x u8; } },
        parse_quote! {
            trait Tr { fn f<F: for<'x> Fn(&'x dyn RelTr, Parser<'x>) -> &'x u8>(&self, f: F); }
        },
    ] {
        let item: ItemTrait = item;
        assert_eq!(error(item.clone()).1.as_deref(), Some(NOTE), "{}", s(&item));
    }
    // A function pointer or a trait object holding the binder is leaked instead.
    assert_eq!(
        leaked(parse_quote! {
            trait Tr { fn f(&self, f: for<'x> fn(&'x dyn RelTr) -> &'x u8); }
        }),
        [e("for < 'x > fn (& 'x dyn RelTr) -> & 'x u8", "")]
    );
    assert_eq!(
        leaked(parse_quote! {
            trait Tr { fn f(&self, f: Box<dyn for<'x> Fn(&'x dyn RelTr) -> &'x u8>); }
        }),
        [e("dyn for < 'x > Fn (& 'x dyn RelTr) -> & 'x u8", "")]
    );
    // Another input keeps `'x` as written: the reference is leaked alone, with `'x`.
    for item in [
        parse_quote! { trait Tr { fn f(&self, f: for<'x> fn(&'x dyn RelTr, &'x u8) -> &'x u8); } },
        parse_quote! {
            trait Tr { fn f<F: for<'x> Fn(&'x dyn RelTr, &'x u8) -> &'x u8>(&self, f: F); }
        },
        // Also one which is not leaked (its trait is absolute).
        parse_quote! {
            trait Tr { fn f(&self, f: for<'x> fn(&'x dyn RelTr, &'x dyn ::p::Tr3) -> &'x u8); }
        },
    ] {
        let item: ItemTrait = item;
        assert_eq!(
            leaked(item.clone()),
            [e("& 'x dyn RelTr", "< 'x >")],
            "{}",
            s(&item)
        );
    }
    // The reference is leaked alone if the output does not mention the lifetime.
    for item in [
        parse_quote! { trait Tr { fn f(&self, f: for<'x> fn(&'x dyn RelTr) -> u8); } },
        parse_quote! { trait Tr { fn f<F: for<'x> Fn(&'x dyn RelTr) -> u8>(&self, f: F); } },
        parse_quote! { trait Tr { fn f<F>(&self, f: F) where for<'x> F: Fn(&'x dyn RelTr); } },
    ] {
        let item: ItemTrait = item;
        assert_eq!(
            leaked(item.clone()),
            [e("& 'x dyn RelTr", "< 'x >")],
            "{}",
            s(&item)
        );
    }
    // A reference to a trait object which is not a cause stays in the copy, and anchors `'x`.
    assert_eq!(
        leaked(parse_quote! {
            trait Tr { fn f(&self, f: for<'x> fn(&'x dyn ::p::Tr3, Parser<'x>) -> &'x u8); }
        }),
        [e("Parser < 'x >", "< 'x >")]
    );
}

#[test]
fn a_trait_object_which_is_leaked_as_a_whole_does_not_anchor_the_lifetime() {
    // The occurrence stays in the copy only if none of the types holding it is leaked as a whole
    // (it is replaced by a projection then). A trait object (or a path) which is a cause itself
    // (a relative path, also in a lifetime bound or a binding) is leaked as a whole, so the
    // lifetime is only in a projection, and the type using it is not leaked on its own.
    for input in [
        "Box<dyn RelIter<Item = &'x u8>>",
        "Box<dyn ::p::Tr3<Item = &'x u8> + RelMarker>",
        "Box<dyn RelTr + 'x>",
        "Box<dyn Fn() -> &'x u8 + RelMarker>",
        "Box<dyn for<'y> RelFn<'y, Item = &'x u8>>",
        "Rel2<Box<dyn Iterator<Item = &'x u8>>>",
        "Rel2<fn(&'x u8)>",
    ] {
        let input: Type = parse_str(input).unwrap();
        let item: ItemTrait = parse_quote! {
            trait Tr { fn f<F: for<'x> Fn(#input, Parser<'x>) -> &'x u8>(&self, f: F); }
        };
        assert_eq!(error(item.clone()).1.as_deref(), Some(NOTE), "{}", s(&item));
    }
    // A function pointer holding it is leaked as a whole instead.
    assert_eq!(
        leaked(parse_quote! {
            trait Tr {
                fn f(&self, f: for<'x> fn(Box<dyn RelIter<Item = &'x u8>>, Parser<'x>) -> &'x u8);
            }
        }),
        [e(
            "for < 'x > fn (Box < dyn RelIter < Item = & 'x u8 > > , Parser < 'x >) -> & 'x u8",
            ""
        )]
    );
    // Another trait object with the lifetime keeps it: both are leaked alone, and the trait
    // object which is a cause itself is leaked with the lifetime among its generics.
    let mut found = leaked(parse_quote! {
        trait Tr {
            fn f<F: for<'x> Fn(
                Box<dyn RelIter<Item = &'x u8>>,
                Box<dyn Iterator<Item = &'x u8>>,
                Parser<'x>
            ) -> &'x u8>(&self, f: F);
        }
    });
    found.sort();
    assert_eq!(
        found,
        [
            e("Parser < 'x >", "< 'x >"),
            e("dyn RelIter < Item = & 'x u8 >", "< 'x >")
        ]
    );
    // One which does not mention the lifetime is not in the way.
    let mut found = leaked(parse_quote! {
        trait Tr {
            fn f<F: for<'x> Fn(
                Box<dyn RelIter>,
                Box<dyn Iterator<Item = &'x u8>>,
                Parser<'x>
            ) -> &'x u8>(&self, f: F);
        }
    });
    found.sort();
    assert_eq!(found, [e("Parser < 'x >", "< 'x >"), e("dyn RelIter", "")]);
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
    let (referrer, context) = analyze_in(leaker(), &item).unwrap();
    let mut copy = item.clone();
    syn::visit_mut::VisitMut::visit_item_trait_mut(
        &mut referrer.visitor(context, |ty, _| {
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
fn the_copy_keeps_the_trait_object_holding_the_anchor() {
    // The trait object, the function pointer and the lifetime bound stay as written.
    let item: ItemTrait = parse_quote! {
        trait Tr {
            fn f<F: for<'x> Fn(
                Box<dyn Iterator<Item = &'x u8>>,
                fn(&'x u8),
                Box<dyn ::p::Tr3 + 'x>,
                Parser<'x>,
            ) -> &'x u8>(&self, f: F);
        }
    };
    let (referrer, context) = analyze_in(leaker(), &item).unwrap();
    assert_eq!(
        referrer.iter().map(s).collect::<Vec<_>>(),
        ["Parser < 'x >"]
    );
    let mut copy = item.clone();
    syn::visit_mut::VisitMut::visit_item_trait_mut(
        &mut referrer.visitor(context, |_, ix| parse_quote!(P<#ix>)),
        &mut copy,
    );
    let expected: ItemTrait = parse_quote! {
        trait Tr {
            fn f<F: for<'x> Fn(
                Box<dyn Iterator<Item = &'x u8>>,
                fn(&'x u8),
                Box<dyn ::p::Tr3 + 'x>,
                P<0usize>,
            ) -> &'x u8>(&self, f: F);
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
    // The inner function pointer holds `'x` in its inputs, which constrains it as well.
    assert_eq!(
        leaked(parse_quote! {
            trait Tr {
                fn f<F: for<'x> Fn(fn(&'x u8), Parser<'x>) -> &'x str>(&self, f: F);
            }
        }),
        [e("Parser < 'x >", "< 'x >")]
    );
    // The anchor of `'x` stays in the inner function pointer, which is not leaked as long as no
    // type is leaked in its place: none uses `'y`, or the output does not mention it.
    assert_eq!(
        leaked(parse_quote! {
            trait Tr {
                fn f<F: for<'x> Fn(for<'y> fn(&'x u8, Rel<'y>), Parser<'x>) -> &'x str>(&self, f: F);
            }
        }),
        [e("Rel < 'y >", "< 'y >"), e("Parser < 'x >", "< 'x >")]
    );
    assert_eq!(
        leaked(parse_quote! {
            trait Tr {
                fn f<F: for<'x> Fn(for<'y> fn(&'x u8, &'y u8) -> &'y u8, Parser<'x>) -> &'x str>(&self, f: F);
            }
        }),
        [e("Parser < 'x >", "< 'x >")]
    );
    // The inner binder is not looked into again while the outer one is (see `Binder`), so a type
    // using the lifetime of the inner one is taken to be leaked as the inner function pointer, as
    // it would be if `'y` had no anchor (`&'y u8` is one here): the anchor of `'x` in it is not
    // counted, which is conservative.
    assert_eq!(
        error(parse_quote! {
            trait Tr {
                fn f<F: for<'x> Fn(for<'y> fn(&'x u8, &'y u8, Rel<'y>) -> &'y u8, Parser<'x>) -> &'x str>(&self, f: F);
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
    let mut leaker = leaker();
    let mut check = |ty: Type| check_in(&mut leaker, &ty, &Generics::default());
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
    // An associated type binding in a trait object anchors it, unless the trait object is.
    let result = check(parse_quote!(
        for<'x> fn(Box<dyn Iterator<Item = &'x u8>>) -> Rel<'x>
    ));
    assert_eq!(tokens(&result.must), ["Rel < 'x >"]);
    assert!(!result.is_own && result.impossible.is_empty());
    let result = check(parse_quote!(
        for<'x> fn(Box<dyn RelIter<Item = &'x u8>>) -> Rel<'x>
    ));
    assert!(result.is_own && result.impossible.is_empty());
}

/// The Leaker which allows `::other`, as `::p` is allowed by `leaker()`.
fn leaker_allowing_other() -> Leaker {
    leaker_with(&[parse_quote!(::other)])
}

/// Same, which allows nothing external: `::other` is an optional entry, which the Referrer may
/// replace.
fn leaker_allowing_nothing() -> Leaker {
    leaker_with(&[])
}

#[test]
fn external_crate_types_do_not_anchor_the_lifetime() {
    // The Referrer may replace the type of an external crate (`::other::Wrap<&'x u8>`) as a whole
    // by a projection, so what is in it (`&'x u8`) stays in the copy only if the Referrer does not.
    // Old: it anchored `'x` (a path of an external crate was not leaked), `Parser<'x>` was leaked
    // alone. New: it is no anchor, as the rest of an absolute path is not either, so `Parser<'x>`
    // cannot be leaked alone: it is an error in a bound.
    let item: ItemTrait = parse_quote! {
        trait Tr {
            fn f<F: for<'x> Fn(::other::Wrap<&'x u8>, Parser<'x>) -> &'x u8>(&self, f: F);
        }
    };
    let (message, note) = error_by(leaker_allowing_nothing(), item.clone());
    assert_eq!(message, "use absolute path: 'Parser < 'x >'");
    assert_eq!(note.as_deref(), Some(NOTE));
    // The same in a predicate, a trait object in a type and a function pointer in a bound.
    for item in [
        parse_quote! {
            trait Tr {
                fn f<F>(&self, f: F) where for<'x> F: Fn(::other::Wrap<&'x u8>, Parser<'x>) -> &'x u8;
            }
        },
        parse_quote! {
            trait Tr {
                fn f<I: for<'x> ::p::Tr2<::other::Wrap<&'x u8>, Item = Parser<'x>>>(&self, i: I);
            }
        },
    ] {
        let item: ItemTrait = item;
        let (_, note) = error_by(leaker_allowing_nothing(), item.clone());
        assert_eq!(note.as_deref(), Some(NOTE), "{}", s(&item));
    }
    // It is an anchor where the crate is allowed (an absolute path in `allowed_paths`).
    assert_eq!(
        leaked_by(leaker_allowing_other(), item.clone()),
        [e("Parser < 'x >", "< 'x >")]
    );
    // A function pointer holding the binder is leaked as a whole instead of the type using `'x`:
    // old, `Parser<'x>` alone.
    let item: ItemTrait = parse_quote! {
        trait Tr {
            fn f(&self, f: for<'x> fn(::other::Wrap<&'x u8>, Parser<'x>) -> &'x u8);
        }
    };
    assert_eq!(
        leaked_by(leaker_allowing_nothing(), item.clone()),
        [e(
            "for < 'x > fn (:: other :: Wrap < & 'x u8 > , Parser < 'x >) -> & 'x u8",
            ""
        )]
    );
    assert_eq!(
        leaked_by(leaker_allowing_other(), item),
        [e("Parser < 'x >", "< 'x >")]
    );
    // Whatever it is nested in, but a path of `::core` or an allowed one.
    for input in [
        "::other::Wrap<&'x u8>",
        "::other::Wrap<'x>",
        "::other::Wrap<(u8, &'x u8)>",
        "::other::Wrap<fn(&'x u8)>",
        "::core::option::Option<::other::Wrap<&'x u8>>",
        "(u8, ::other::Wrap<&'x u8>)",
        "[::other::Wrap<&'x u8>; 3]",
        "Option<::other::Wrap<&'x u8>>",
        "Box<dyn ::other::Tr<&'x u8>>",
        "Box<dyn ::other::Tr + 'x>",
        "Box<dyn ::other::Tr<Item = &'x u8>>",
        "Box<dyn ::core::iter::Iterator<Item = ::other::Wrap<&'x u8>>>",
        "fn(::other::Wrap<&'x u8>)",
        "&'static ::other::Wrap<&'x u8>",
        "*const ::other::Wrap<&'x u8>",
        // A reference to a trait object of an external crate (which inherits its lifetime) and a
        // path with a lifetime and such a trait object are leaked as a whole with it, so nothing
        // is kept in them, not even the lifetime of the reference or the path.
        "&'x dyn ::other::Tr",
        "&'x (dyn ::other::Tr)",
        "::core::cell::Ref<'x, dyn ::other::Tr>",
        // The length of an array naming an external crate makes it a type which may be replaced.
        "[&'x u8; ::other::N]",
    ] {
        let input: Type = parse_str(input).unwrap();
        let item: ItemTrait = parse_quote! {
            trait Tr {
                fn f(&self, f: for<'x> fn(#input, Parser<'x>) -> &'x u8);
            }
        };
        let leaked = leaked_by(leaker_allowing_nothing(), item.clone());
        assert_eq!(leaked.len(), 1, "{}", s(&item));
        assert!(
            leaked[0].0.starts_with("for < 'x > fn ("),
            "{}: {:?}",
            s(&item),
            leaked
        );
        assert_eq!(leaked[0].1, "", "{}", s(&item));
        // Where the crate is allowed, `'x` is anchored by it.
        assert_eq!(
            leaked_by(leaker_allowing_other(), item.clone()),
            [e("Parser < 'x >", "< 'x >")],
            "{}",
            s(&item)
        );
    }
}

#[test]
fn a_trait_object_of_an_external_crate_with_a_lifetime_bound_is_leaked_alone() {
    // A lifetime bound written on the trait object makes it a type to leak alone (with its bound),
    // and the reference stays as written, so it keeps `'x` whichever the Referrer does.
    for input in [
        "&'x (dyn ::other::Tr + 'x)",
        "&'x (dyn ::other::Tr + Send + 'x)",
    ] {
        let input: Type = parse_str(input).unwrap();
        let item: ItemTrait = parse_quote! {
            trait Tr { fn f(&self, f: for<'x> fn(#input, Parser<'x>) -> &'x u8); }
        };
        assert_eq!(
            leaked_by(leaker_allowing_nothing(), item.clone()),
            [
                e("Parser < 'x >", "< 'x >"),
                e(
                    &format!(
                        "? {}",
                        s(&parse_str::<Type>(&input_without_ref(&input)).unwrap())
                    ),
                    "< 'x >"
                ),
            ],
            "{}",
            s(&item)
        );
    }
}

/// The referent of a reference type.
fn input_without_ref(ty: &Type) -> String {
    match ty {
        Type::Reference(reference) => match &*reference.elem {
            Type::Paren(paren) => s(&paren.elem),
            elem => s(elem),
        },
        _ => unreachable!(),
    }
}

#[test]
fn the_types_which_stay_as_written_still_anchor_the_lifetime() {
    // `::core` has the same crate in every crate, and a relative path allowed by the Leaker (the
    // prelude) is not leaked: they stay where they are written, with what is in them.
    for input in [
        "&'x str",
        "::core::option::Option<&'x u8>",
        "::core::cell::Ref<'x, u8>",
        "::core::option::Option<(u8, &'x u8)>",
        "::core::primitive::u8, &'x u8",
        "Option<&'x u8>",
        "Box<dyn Iterator<Item = &'x u8>>",
        "Box<dyn ::core::fmt::Debug + 'x>",
        "&'x dyn ::core::fmt::Debug",
        // Also inside a type of `::core`, next to an external one, and behind a reference.
        "::core::option::Option<(::other::Wrap<u8>, &'x u8)>",
        "&'x ::other::Wrap<u8>",
        "(::other::Wrap<&'x u8>, &'x u8)",
    ] {
        // `u8, &'x u8` is two inputs.
        let inputs: Punctuated<Type, Token![,]> =
            parse::Parser::parse_str(Punctuated::parse_terminated, input).unwrap();
        let item: ItemTrait = parse_quote! {
            trait Tr {
                fn f<F: for<'x> Fn(#inputs, Parser<'x>) -> &'x u8>(&self, f: F);
                fn g(&self, f: for<'x> fn(#inputs, Parser<'x>) -> &'x u8);
            }
        };
        let referrer = analyze_with(leaker_allowing_nothing(), &item)
            .unwrap_or_else(|e| panic!("{}: {}", input, syn::Error::from(e)));
        // `Parser<'x>` is leaked alone (twice: in each binder, which are the same entry), and the
        // external types are optional entries with `'x` among their generics, if they use it.
        let required: Vec<String> = referrer
            .iter()
            .enumerate()
            .filter(|(ix, _)| referrer.must(*ix))
            .map(|(_, ty)| s(ty))
            .collect();
        assert_eq!(required, ["Parser < 'x >"], "{}", input);
    }
}

#[test]
fn an_external_crate_type_which_uses_the_lifetime_of_a_binder_is_leaked_with_its_holder() {
    // The type cannot be an optional entry alone (leaked alone it has `'x` only in a projection),
    // so the type holding the binder is, as a whole, if it is a type: no error, as nothing which
    // has to be leaked uses `'x`. A bound or a predicate holds it, which is not a type, so it is
    // copied as it is.
    assert_eq!(
        leaked_by(
            leaker_allowing_nothing(),
            parse_quote! {
                trait Tr { fn f(&self, f: for<'x> fn(::other::Wrap<&'x u8>) -> &'x u8); }
            }
        ),
        [e(
            "? for < 'x > fn (:: other :: Wrap < & 'x u8 >) -> & 'x u8",
            ""
        )]
    );
    assert_eq!(
        leaked_by(
            leaker_allowing_nothing(),
            parse_quote! {
                trait Tr { fn f(&self, f: ::std::boxed::Box<dyn for<'x> Fn(::other::Wrap<&'x u8>) -> &'x u8>); }
            }
        ),
        [e(
            "? :: std :: boxed :: Box < dyn for < 'x > Fn (:: other :: Wrap < & 'x u8 >) -> & 'x u8 >",
            ""
        )]
    );
    for item in [
        parse_quote! {
            trait Tr { fn f<F: for<'x> Fn(::other::Wrap<&'x u8>) -> &'x u8>(&self, f: F); }
        },
        parse_quote! {
            trait Tr { fn f<F>(&self, f: F) where for<'x> F: Fn(::other::Wrap<&'x u8>) -> &'x u8; }
        },
        // The holder has a trait object under an elided reference, so it cannot be leaked either.
        parse_quote! {
            trait Tr {
                fn f(&self, f: for<'x> fn(::other::Wrap<&'x u8>, &dyn ::core::fmt::Debug) -> &'x u8);
            }
        },
    ] {
        let item: ItemTrait = item;
        assert_eq!(
            leaked_by(leaker_allowing_nothing(), item.clone()),
            [],
            "{}",
            s(&item)
        );
    }
    // An elided lifetime of its own does not stop the holder from being leaked.
    assert_eq!(
        leaked_by(
            leaker_allowing_nothing(),
            parse_quote! {
                trait Tr { fn f(&self, f: for<'x> fn(::other::Wrap<&'x u8>, &u8) -> &'x u8); }
            }
        ),
        [e(
            "? for < 'x > fn (:: other :: Wrap < & 'x u8 > , & u8) -> & 'x u8",
            ""
        )]
    );
    // With a cause in it (`Parser`), the required entries are not affected: `Parser` is leaked as
    // usual, and the function pointer, which contains it, is the optional entry.
    assert_eq!(
        leaked_by(
            leaker_allowing_nothing(),
            parse_quote! {
                trait Tr { fn f(&self, f: for<'x> fn(::other::Wrap<&'x u8, (Parser,)>) -> &'x u8); }
            }
        ),
        [
            e("Parser", ""),
            e(
                "? for < 'x > fn (:: other :: Wrap < & 'x u8 , (Parser ,) >) -> & 'x u8",
                ""
            )
        ]
    );
    // One which does not use the lifetime of the output is an optional entry as usual.
    let item: ItemTrait = parse_quote! {
        trait Tr { fn f(&self, f: for<'x> fn(::other::Wrap<u8>) -> &'x u8); }
    };
    assert_eq!(
        leaked_by(leaker_allowing_nothing(), item),
        [e("? :: other :: Wrap < u8 >", "")]
    );
    // And one which another input anchors, with `'x` among its generics.
    let item: ItemTrait = parse_quote! {
        trait Tr { fn f(&self, f: for<'x> fn(::other::Wrap<&'x u8>, &'x u8, Parser<'x>) -> &'x u8); }
    };
    assert_eq!(
        leaked_by(leaker_allowing_nothing(), item),
        [
            e("Parser < 'x >", "< 'x >"),
            e("? :: other :: Wrap < & 'x u8 >", "< 'x >")
        ]
    );
}

#[test]
fn the_anchor_of_an_external_crate_type_is_checked_by_check() {
    let mut leaker = leaker_allowing_nothing();
    let result = check_in(
        &mut leaker,
        &parse_quote!(for<'x> fn(::other::Wrap<&'x u8>) -> Parser<'x>),
        &Generics::default(),
    );
    // The function pointer holding `'x` is interned as a whole instead of `Parser<'x>`.
    assert!(result.is_own && result.impossible.is_empty());
    let result = check_in(
        &mut leaker_allowing_other(),
        &parse_quote!(for<'x> fn(::other::Wrap<&'x u8>) -> Parser<'x>),
        &Generics::default(),
    );
    assert!(!result.is_own && result.impossible.is_empty());
    // A path of `::core` anchors.
    let result = check_in(
        &mut leaker,
        &parse_quote!(for<'x> fn(::core::option::Option<&'x u8>) -> Parser<'x>),
        &Generics::default(),
    );
    assert!(!result.is_own && result.impossible.is_empty());
}

#[test]
fn the_copy_keeps_what_anchors_with_external_crate_types() {
    // The Referrer which leaves the optional entries keeps the type of the external crate where it
    // is written (it is as anchored as the Leaker's was), and the one which replaces it has the
    // function pointer replaced as a whole.
    let item: ItemTrait = parse_quote! {
        trait Tr {
            fn f(&self, f: for<'x> fn(::other::Wrap<&'x u8>, &'x u8, Parser<'x>) -> &'x u8);
        }
    };
    let (referrer, context) = analyze_in(leaker_allowing_nothing(), &item).unwrap();
    let mut kept = item.clone();
    syn::visit_mut::VisitMut::visit_item_trait_mut(
        &mut referrer
            .clone()
            .visitor(context, |_, ix| parse_quote!(P<#ix>)),
        &mut kept,
    );
    let expected: ItemTrait = parse_quote! {
        trait Tr {
            fn f(&self, f: for<'x> fn(::other::Wrap<&'x u8>, &'x u8, P<0usize>) -> &'x u8);
        }
    };
    assert_eq!(s(&kept), s(&expected));
    // The Referrer which replaces them: the type is replaced (with its entry), and `&'x u8`
    // anchors `'x`, so the copy is valid.
    let mut replaced = item.clone();
    syn::visit_mut::VisitMut::visit_item_trait_mut(
        &mut referrer
            .allow_ambiguous_crate_name(false)
            .visitor(context, |_, ix| parse_quote!(P<#ix>)),
        &mut replaced,
    );
    let expected: ItemTrait = parse_quote! {
        trait Tr {
            fn f(&self, f: for<'x> fn(P<1usize>, &'x u8, P<0usize>) -> &'x u8);
        }
    };
    assert_eq!(s(&replaced), s(&expected));
    // The one whose function pointer is leaked as a whole is replaced by it in both.
    let item: ItemTrait = parse_quote! {
        trait Tr {
            fn f(&self, f: for<'x> fn(::other::Wrap<&'x u8>, Parser<'x>) -> &'x u8);
        }
    };
    let (referrer, context) = analyze_in(leaker_allowing_nothing(), &item).unwrap();
    for allow in [true, false] {
        let mut copy = item.clone();
        syn::visit_mut::VisitMut::visit_item_trait_mut(
            &mut referrer
                .clone()
                .allow_ambiguous_crate_name(allow)
                .visitor(context, |_, ix| parse_quote!(P<#ix>)),
            &mut copy,
        );
        let expected: ItemTrait = parse_quote! {
            trait Tr { fn f(&self, f: P<0usize>); }
        };
        assert_eq!(s(&copy), s(&expected));
    }
}

fn error_by(leaker: Leaker, item: ItemTrait) -> (String, Option<String>) {
    let err = analyze_with(leaker, &item).expect_err("expected an error");
    (err.to_string(), err.note())
}

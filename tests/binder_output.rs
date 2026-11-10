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
//!
//! The module `elided` has the same for an output with an elided lifetime (E0106).

mod common;

use common::{check_in, starts_with};
use syn::punctuated::Punctuated;
use syn::*;
use template_quote::quote;
use type_leak::{
    to_syn_error, Allow, Cause, ContextId, ErrorItem, Leaker, Options, Place, Referrer, Rules,
};

/// A Leaker which allows `::p`, standing for an external crate which the Referrer's crate is known
/// to have as the Leaker's: its types may be an anchor. The types of the other external crates may
/// be replaced by the Referrer, so they are not (see `external_crate_types_do_not_anchor`).
fn leaker() -> Leaker<impl Allow> {
    leaker_with(&[parse_quote!(::p)])
}

fn leaker_with(allowed: &[Path]) -> Leaker<impl Allow> {
    let allowed = allowed.to_vec();
    let config = Rules::new()
        .primitive()
        .prelude()
        .or_if(move |cause: &Cause, place: Place| {
            allowed.iter().any(|a| starts_with(a.clone())(cause, place))
        });
    Leaker::from_config(config, Options::new())
}

fn s(x: &impl template_quote::ToTokens) -> String {
    quote!(#x).to_string()
}

/// The messages of the errors, one per line.
fn messages(errors: &[ErrorItem]) -> String {
    errors
        .iter()
        .map(|item| item.message.as_str())
        .collect::<Vec<_>>()
        .join("\n")
}

fn analyze(item: &ItemTrait) -> std::result::Result<Referrer, Vec<ErrorItem>> {
    analyze_with(leaker(), item)
}

fn analyze_with<A: Allow>(
    leaker: Leaker<A>,
    item: &ItemTrait,
) -> std::result::Result<Referrer, Vec<ErrorItem>> {
    analyze_in(leaker, item).map(|(referrer, _)| referrer)
}

/// Like `analyze_with()`, with the top-level context of the trait: the scope of its copy.
fn analyze_in<A: Allow>(
    mut leaker: Leaker<A>,
    item: &ItemTrait,
) -> std::result::Result<(Referrer, ContextId), Vec<ErrorItem>> {
    let context = leaker.intern_trait(item, true);
    Ok((leaker.result().finish()?, context))
}

/// The leaked types of a trait with the generics each uses.
fn leaked(item: ItemTrait) -> Vec<(String, String)> {
    leaked_by(leaker(), item)
}

/// The leaked types of a trait with the generics each uses, by the given Leaker.
fn leaked_by<A: Allow>(leaker: Leaker<A>, item: ItemTrait) -> Vec<(String, String)> {
    let referrer = analyze_with(leaker, &item).unwrap_or_else(|e| panic!("{}", to_syn_error(&e)));
    referrer
        .iter()
        .map(|(_, ty, context)| (ty, &context.generics))
        .map(|(ty, generics)| (s(ty), s(generics)))
        .collect()
}

fn e(ty: &str, generics: &str) -> (String, String) {
    (ty.to_string(), generics.to_string())
}

/// The message and the first note of the error of a trait.
fn error(item: ItemTrait) -> (String, Option<String>) {
    error_by(leaker(), item)
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
        assert_eq!(message, "use absolute path: 'Rel<'x>'", "{}", s(&item));
        assert_eq!(note.as_deref(), Some(NOTE), "{}", s(&item));
    }

    // The error is reported with the note and the lifetime as the cause.
    let err = analyze(&parse_quote! {
        trait Tr { fn f<F: for<'x> Fn(Rel<'x>, Rel2) -> &'x u8>(&self, f: F); }
    })
    .err()
    .unwrap();
    assert_eq!(
        to_syn_error(&err).to_string(),
        format!(
            "use absolute path: 'Rel<'x>'\n\n  = note: {}\n  = note: it cannot be leaked as a \
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
    leaker.intern_struct(&item, true);
    let referrer = leaker.result().finish().unwrap();
    assert_eq!(
        common::types(&referrer).map(s).collect::<Vec<_>>(),
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
    // rustc rejects. (An elided lifetime of its own would not stop it, see the module `elided`.)
    // The one error, then its text: the first note is that of `NOTE_OWNED`.
    let err = analyze(&parse_quote! {
        trait Tr { fn f(&self, f: for<'x> fn(Rel<'x>, &dyn ::core::fmt::Debug) -> &'x u8); }
    })
    .err()
    .unwrap();
    assert_eq!(messages(&err), "use absolute path: 'Rel<'x>'");
    assert_eq!(
        to_syn_error(&err).to_string(),
        format!(
            "use absolute path: 'Rel<'x>'\n\n  = note: {}\n  = note: it cannot be leaked as a \
             whole because of `dyn ::core::fmt::Debug`\n  = note: the lifetime of the trait \
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
            "use absolute path: 'Rel<'x>'".to_string(),
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
        "use absolute path: 'Rel<'x, 'y>'"
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

/// A Leaker which allows the paths `allowed`.
fn leaker_allowing(allowed: &[&str]) -> Leaker<impl Allow> {
    let allowed: Vec<Path> = allowed
        .iter()
        .map(|path| parse_str(path).unwrap())
        .collect();
    leaker_with(&allowed)
}

/// A row of `check_tells_whether_the_holder_is_interned()`: the paths the Leaker allows, the
/// generics, the type, `must` (if it is asserted), `is_own` and `impossible`.
type CheckRow = (
    &'static [&'static str],
    &'static str,
    &'static str,
    Option<&'static [&'static str]>,
    bool,
    &'static [&'static str],
);

/// `View::check()` of a type that holds a binder or a signature tells which causes are leaked,
/// whether the type is interned itself, and which are impossible.
#[test]
fn check_tells_whether_the_holder_is_interned() {
    let rows: Vec<CheckRow> = vec![
        // `Rel<'x>` alone is leaked (as a component), unless the return type mentions `'x`.
        (
            &["::p"],
            "",
            "for<'x> fn(Rel<'x>)",
            Some(&["Rel < 'x >"]),
            false,
            &[],
        ),
        (
            &["::p"],
            "",
            "for<'x> fn(Rel<'x>) -> &'x u8",
            Some(&["Rel < 'x >"]),
            true,
            &[],
        ),
        // The elided lifetime of the function pointer is its own, so it does not stop it from being
        // leaked as a whole.
        (
            &["::p"],
            "",
            "for<'x> fn(Rel<'x>, &u8) -> &'x u8",
            Some(&["Rel < 'x >"]),
            true,
            &[],
        ),
        // Which is a conflict if it cannot be: a trait object under an elided reference takes its
        // lifetime from it.
        (
            &["::p"],
            "",
            "for<'x> fn(Rel<'x>, &dyn ::core::fmt::Debug) -> &'x u8",
            None,
            true,
            &["dyn :: core :: fmt :: Debug"],
        ),
        // `Rel<'x>` stays a component of the function pointer, which is not interned.
        (
            &["::p"],
            "",
            "for<'x> fn(&'x u8) -> Rel<'x>",
            Some(&["Rel < 'x >"]),
            false,
            &[],
        ),
        // The anchor is lost if it is in a type interned itself.
        (
            &["::p"],
            "",
            "for<'x> fn(Rel2<&'x u8>) -> Rel<'x>",
            None,
            true,
            &[],
        ),
        // An associated type binding in a trait object anchors it, unless the trait object is.
        (
            &["::p"],
            "",
            "for<'x> fn(Box<dyn Iterator<Item = &'x u8>>) -> Rel<'x>",
            Some(&["Rel < 'x >"]),
            false,
            &[],
        ),
        (
            &["::p"],
            "",
            "for<'x> fn(Box<dyn RelIter<Item = &'x u8>>) -> Rel<'x>",
            None,
            true,
            &[],
        ),
        // The function pointer holding `'x` is interned as a whole instead of `Parser<'x>`, unless
        // the crate of the type that anchors is allowed; a path of `::core` anchors.
        (
            &[],
            "",
            "for<'x> fn(::other::Wrap<&'x u8>) -> Parser<'x>",
            None,
            true,
            &[],
        ),
        (
            &["::other"],
            "",
            "for<'x> fn(::other::Wrap<&'x u8>) -> Parser<'x>",
            None,
            false,
            &[],
        ),
        (
            &[],
            "",
            "for<'x> fn(::core::option::Option<&'x u8>) -> Parser<'x>",
            None,
            false,
            &[],
        ),
        // The holder of an elided output is the cause of its own, and it has no impossible one: its
        // elided lifetimes are its own. A box holds it: the cause is the trait object, and not the
        // box.
        (
            &[],
            "<'a, T: 'a>",
            "dyn Fn(&Rel<T>) -> &u8",
            Some(&["Rel < T >"]),
            true,
            &[],
        ),
        (
            &[],
            "<'a, T: 'a>",
            "fn(&Rel<T>) -> &u8",
            Some(&["Rel < T >"]),
            true,
            &[],
        ),
        (
            &[],
            "<'a, T: 'a>",
            "Box<dyn Fn(&Rel<T>) -> &u8>",
            Some(&["Rel < T >"]),
            false,
            &[],
        ),
        // Not where the entry has no lifetimes.
        (
            &[],
            "<T>",
            "dyn Fn(&Rel<T>) -> &u8",
            Some(&["Rel < T >"]),
            false,
            &[],
        ),
    ];
    for (allowed, generics, ty, must, is_own, impossible) in rows {
        let generics: Generics = if generics.is_empty() {
            Generics::default()
        } else {
            parse_str(generics).unwrap()
        };
        let ty: Type = parse_str(ty).unwrap();
        let text = s(&ty);
        let tokens = |causes: &[Cause]| -> Vec<String> {
            causes
                .iter()
                .map(|cause| quote!(#cause).to_string())
                .collect()
        };
        let result = check_in(&mut leaker_allowing(allowed), &ty, &generics);
        if let Some(must) = must {
            assert_eq!(tokens(&result.must), must, "{}", text);
        }
        assert_eq!(result.is_own, is_own, "{}", text);
        assert_eq!(tokens(&result.impossible), impossible, "{}", text);
    }
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
        common::types(&referrer).map(s).collect::<Vec<_>>(),
        ["for < 'x > fn (Rel < 'x >) -> & 'x u8", "Rel < 'x >"]
    );
    let mut copy = item.clone();
    syn::visit_mut::VisitMut::visit_item_trait_mut(
        &mut referrer.visitor(context, |_, ix, _| parse_quote!(P<#ix>)),
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
        assert_eq!(message, "use absolute path: 'Parser<'x>'", "{}", s(&item));
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
            "use absolute path: 'Rel2<&'x str>'".to_string(),
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
        .iter()
        .map(|(_, ty, context)| (ty, &context.generics))
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
        analyze(&item)
            .unwrap()
            .iter()
            .map(|(_, ty, _)| s(ty))
            .collect::<Vec<_>>(),
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
        &mut referrer.visitor(context, |ty, _, _| {
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
        common::types(&referrer).map(s).collect::<Vec<_>>(),
        ["Parser < 'x >"]
    );
    let mut copy = item.clone();
    syn::visit_mut::VisitMut::visit_item_trait_mut(
        &mut referrer.visitor(context, |_, ix, _| parse_quote!(P<#ix>)),
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

/// The Leaker which allows `::other`, as `::p` is allowed by `leaker()`.
fn leaker_allowing_other() -> Leaker<impl Allow> {
    leaker_with(&[parse_quote!(::other)])
}

/// Same, which allows nothing external: `::other` is an optional entry, which the Referrer may
/// replace.
fn leaker_allowing_nothing() -> Leaker<impl Allow> {
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
    assert_eq!(message, "use absolute path: 'Parser<'x>'");
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
    // It is an anchor where the crate is allowed (an absolute path given to `Rules::or_if()`).
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
            trait Tr { fn f(&self, f: for<'x> fn(#input) -> &'x u8); }
        };
        assert_eq!(
            leaked_by(leaker_allowing_nothing(), item.clone()),
            [e(
                &s(&parse_str::<Type>(&input_without_ref(&input)).unwrap()),
                "< 'x >"
            )],
            "{}",
            s(&item)
        );
        // With a type to leak beside it, one entry covers both: the function pointer, as a whole.
        let item: ItemTrait = parse_quote! {
            trait Tr { fn f(&self, f: for<'x> fn(#input, Parser<'x>) -> &'x u8); }
        };
        let pointer: Type = parse_quote!(for<'x> fn(#input, Parser<'x>) -> &'x u8);
        let pointer = s(&pointer);
        assert_eq!(
            leaked_by(leaker_allowing_nothing(), item.clone()),
            [e(&pointer, "")],
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
            .unwrap_or_else(|e| panic!("{}: {}", input, to_syn_error(&e)));
        // `Parser<'x>` is leaked alone (twice: in each binder, which are the same entry), and it
        // comes first. The external types are entries with `'x` among their generics, if they use
        // it.
        let leaked: Vec<String> = common::types(&referrer).map(s).collect();
        assert_eq!(leaked[0], "Parser < 'x >", "{}", input);
        assert_eq!(
            leaked.iter().filter(|ty| *ty == "Parser < 'x >").count(),
            1,
            "{}",
            input
        );
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
            "for < 'x > fn (:: other :: Wrap < & 'x u8 >) -> & 'x u8",
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
            ":: std :: boxed :: Box < dyn for < 'x > Fn (:: other :: Wrap < & 'x u8 >) -> & 'x u8 >",
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
            "for < 'x > fn (:: other :: Wrap < & 'x u8 > , & u8) -> & 'x u8",
            ""
        )]
    );
    // With a cause in it (`Parser`), the function pointer which contains it is the only entry, and
    // it is required (`Parser` was an entry of its own, and the function pointer an optional one).
    assert_eq!(
        leaked_by(
            leaker_allowing_nothing(),
            parse_quote! {
                trait Tr { fn f(&self, f: for<'x> fn(::other::Wrap<&'x u8, (Parser,)>) -> &'x u8); }
            }
        ),
        [e(
            "for < 'x > fn (:: other :: Wrap < & 'x u8 , (Parser ,) >) -> & 'x u8",
            ""
        )]
    );
    // One which does not use the lifetime of the output is an optional entry as usual.
    let item: ItemTrait = parse_quote! {
        trait Tr { fn f(&self, f: for<'x> fn(::other::Wrap<u8>) -> &'x u8); }
    };
    assert_eq!(
        leaked_by(leaker_allowing_nothing(), item),
        [e(":: other :: Wrap < u8 >", "")]
    );
    // And one which another input anchors, with `'x` among its generics.
    let item: ItemTrait = parse_quote! {
        trait Tr { fn f(&self, f: for<'x> fn(::other::Wrap<&'x u8>, &'x u8) -> &'x u8); }
    };
    assert_eq!(
        leaked_by(leaker_allowing_nothing(), item),
        [e(":: other :: Wrap < & 'x u8 >", "< 'x >")]
    );
    // With `Parser<'x>` beside it, the function pointer is one entry (it was `Parser<'x>` and the
    // optional `Wrap<&'x u8>`).
    let item: ItemTrait = parse_quote! {
        trait Tr { fn f(&self, f: for<'x> fn(::other::Wrap<&'x u8>, &'x u8, Parser<'x>) -> &'x u8); }
    };
    assert_eq!(
        leaked_by(leaker_allowing_nothing(), item),
        [e(
            "for < 'x > fn (:: other :: Wrap < & 'x u8 > , & 'x u8 , Parser < 'x >) -> & 'x u8",
            ""
        )]
    );
}

#[test]
fn the_copy_keeps_what_anchors_with_external_crate_types() {
    // A macro that keeps the optional entries as written keeps the type of the external crate where
    // it is written (it is as anchored as the Leaker's was), and the one which replaces it has the
    // type replaced, with `&'x u8` anchoring `'x` in both.
    let item: ItemTrait = parse_quote! {
        trait Tr {
            fn f(&self, f: for<'x> fn(::other::Wrap<&'x u8>, &'x u8) -> &'x u8);
        }
    };
    let mut leaker = leaker_allowing_nothing();
    let context = leaker.intern_trait(&item, true);
    let result = leaker.result();
    let referrer = result.finish().unwrap();
    // The macro which replaces them: the type is replaced (with its entry), and `&'x u8`
    // anchors `'x`, so the copy is valid.
    let mut replaced = item.clone();
    syn::visit_mut::VisitMut::visit_item_trait_mut(
        &mut referrer.visitor(context, |_, ix, _| parse_quote!(P<#ix>)),
        &mut replaced,
    );
    let expected: ItemTrait = parse_quote! {
        trait Tr {
            fn f(&self, f: for<'x> fn(P<0usize>, &'x u8) -> &'x u8);
        }
    };
    assert_eq!(s(&replaced), s(&expected));
    // With a type to leak beside it, one entry covers all: the function pointer, replaced as a
    // whole by both (it was `Parser<'x>` and the optional `Wrap<&'x u8>`). The same where the
    // function pointer is leaked as a whole for the other input.
    for item in [
        parse_quote! {
            trait Tr {
                fn f(&self, f: for<'x> fn(::other::Wrap<&'x u8>, &'x u8, Parser<'x>) -> &'x u8);
            }
        },
        parse_quote! {
            trait Tr {
                fn f(&self, f: for<'x> fn(::other::Wrap<&'x u8>, Parser<'x>) -> &'x u8);
            }
        },
    ] {
        let item: ItemTrait = item;
        let mut leaker = leaker_allowing_nothing();
        let context = leaker.intern_trait(&item, true);
        let referrer = leaker.result().finish().unwrap();
        let mut copy = item.clone();
        syn::visit_mut::VisitMut::visit_item_trait_mut(
            &mut referrer.visitor(context, |_, ix, _| parse_quote!(P<#ix>)),
            &mut copy,
        );
        let expected: ItemTrait = parse_quote! {
            trait Tr { fn f(&self, f: P<0usize>); }
        };
        assert_eq!(s(&copy), s(&expected), "{}", s(&item));
    }
}

fn error_by<A: Allow>(leaker: Leaker<A>, item: ItemTrait) -> (String, Option<String>) {
    let err = analyze_with(leaker, &item).expect_err("expected an error");
    (
        messages(&err),
        err[0]
            .notes
            .first()
            .map(|(_, n)| n.as_str())
            .map(str::to_string),
    )
}

mod elided {
    //! The elided lifetime in the output of a signature, next to a leaked type that has lifetimes.
    //!
    //! `Box<dyn Fn(&Rel<T>) -> &u8>` has one lifetime in the inputs of the sugar, the elided one of
    //! `&`, which elision gives to the output. A leaked type that has lifetimes (`Rel<T>` in a trait
    //! with `'a, T: 'a` has `<'a, T: 'a>`) is replaced by a projection that names them in its
    //! arguments (`<S as Repeater<(&'a (), T, &'a T), ..>>::Type`), so the inputs of the copy have
    //! several lifetimes, and rustc cannot resolve the output (E0106). rustc counts the lifetimes that
    //! are distinct in the inputs: each elided one, and each name once. A function pointer or `Fn(..)`
    //! inside an input has its own signature. A method takes the lifetime of its output from a
    //! receiver that is a reference, else it counts the inputs like the others.
    //!
    //! So a type that adds lifetimes to such inputs is not leaked alone: the type holding the signature
    //! (a function pointer, a trait object, or a reference to one that takes its lifetime) is leaked
    //! as a whole instead, which keeps the elision as written in the impl of the Repeater. Where
    //! nothing holds the signature (a bound, `impl Fn(..)`, a method that has no receiver which is a
    //! reference), or it cannot be leaked (a trait object under an elided reference, which takes its
    //! lifetime from it), it is an error.
    //!
    //! The elided lifetimes inside a function pointer or `Fn(..)` are its own (rustc makes them
    //! lifetimes of its binder), so they do not stop it from being leaked as a whole. They stop the
    //! types inside it, which are not the holder.
    //!
    //! The copies compile as the tests of `tests/findings.rs` (F12) and the property test show.

    use super::{e, messages, s};
    use syn::*;
    use type_leak::{to_syn_error, Allow, ErrorItem, Leaker, Options, Referrer, Rules};

    fn leaker() -> Leaker<impl Allow> {
        let config = Rules::new().primitive().prelude();
        Leaker::from_config(config, Options::new())
    }

    fn analyze(item: &ItemTrait) -> std::result::Result<Referrer, Vec<ErrorItem>> {
        let mut leaker = leaker();
        leaker.intern_trait(item, true);
        leaker.result().finish()
    }

    /// The leaked types of a trait with the generics each uses.
    fn leaked(item: ItemTrait) -> Vec<(String, String)> {
        let referrer = analyze(&item).unwrap_or_else(|e| panic!("{}: {}", s(&item), messages(&e)));
        referrer
            .iter()
            .map(|(_, ty, context)| (ty, &context.generics))
            .map(|(ty, generics)| (s(ty), s(generics)))
            .collect()
    }

    /// The message and the first note of the error of a trait.
    fn error(item: ItemTrait) -> (String, String) {
        let err = match analyze(&item) {
            Ok(_) => panic!("{} is accepted", s(&item)),
            Err(err) => err,
        };
        (
            messages(&err),
            err[0]
                .notes
                .first()
                .map(|(_, n)| n.as_str())
                .unwrap()
                .to_string(),
        )
    }

    /// The help of the error of a trait.
    fn help(item: &ItemTrait) -> String {
        match analyze(item) {
            Ok(_) => panic!("{} is accepted", s(item)),
            Err(err) => err[0].help_text().unwrap().to_string(),
        }
    }

    /// The note of the error where nothing holds the signature.
    const NOTE: &str =
        "the output `&u8` has an elided lifetime, which elision takes from the only \
                        lifetime of the inputs, but a leaked type in the inputs has `'a` among its \
                        generics, and the copy names it in the arguments of its projection, so the \
                        inputs have more than one lifetime (E0106)";

    /// What to write for the error of `NOTE` in `Fn(..)` or a function pointer.
    const HELP: &str =
        "name the lifetime of the inputs and of the output, or write `Rel<T>` with an \
                        absolute path";

    /// What to write for the error of `NOTE` in a method: it can have a receiver too.
    const HELP_METHOD: &str =
        "name the lifetime of the inputs and of the output, give the method a \
                               receiver (`&self`), or write `Rel<T>` with an absolute path";

    /// The note of the error where the holder cannot be leaked.
    const NOTE_OWNED: &str = "the output `&u8` has an elided lifetime, which elision takes from the \
                              only lifetime of the inputs, but a leaked type in the inputs has `'a` \
                              among its generics, and the copy names it in the arguments of its \
                              projection, so the inputs have more than one lifetime (E0106), so a \
                              type using them cannot be leaked alone, and the type holding the \
                              signature would be leaked as a whole instead";

    #[test]
    fn the_holder_is_leaked_as_a_whole() {
        // The sugar of `Fn(..)` in a trait object, in a box, behind a reference, with other bounds.
        for (item, holder) in [
            (
                parse_quote! { trait Tr<'a, T: 'a> { fn f(&self, x: Box<dyn Fn(&Rel<T>) -> &u8>); } },
                "dyn Fn (& Rel < T >) -> & u8",
            ),
            (
                parse_quote! {
                    trait Tr<'a, T: 'a> { fn f(&self, x: Box<dyn FnMut(&Rel<T>) -> &u8 + Send>); }
                },
                "dyn FnMut (& Rel < T >) -> & u8 + Send",
            ),
            (
                parse_quote! { trait Tr<'a, T: 'a> { fn f(&self, x: Vec<Box<dyn Fn(&Rel<T>) -> &u8>>); } },
                "dyn Fn (& Rel < T >) -> & u8",
            ),
            // The reference takes the lifetime of the trait object, so it is leaked with it.
            (
                parse_quote! { trait Tr<'a, T: 'a> { fn f(&self, x: &'a dyn Fn(&Rel<T>) -> &u8); } },
                "& 'a dyn Fn (& Rel < T >) -> & u8",
            ),
            (
                parse_quote! {
                    trait Tr<'a, T: 'a> { fn f(&self, x: &'a (dyn Fn(&Rel<T>) -> &u8 + 'a)); }
                },
                "& 'a (dyn Fn (& Rel < T >) -> & u8 + 'a)",
            ),
            // A function pointer.
            (
                parse_quote! { trait Tr<'a, T: 'a> { fn f(&self, x: fn(&Rel<T>) -> &u8); } },
                "fn (& Rel < T >) -> & u8",
            ),
            (
                parse_quote! {
                    trait Tr<'a, T: 'a> { fn f(&self, x: unsafe extern "C" fn(&Rel<T>) -> &u8); }
                },
                "unsafe extern \"C\" fn (& Rel < T >) -> & u8",
            ),
            // Other elided lifetimes in the output, and a lifetime argument in the input.
            (
                parse_quote! {
                    trait Tr<'a, T: 'a> { fn f(&self, x: fn(&Rel<T>) -> ::p::Ref<'_, u8>); }
                },
                "fn (& Rel < T >) -> :: p :: Ref < '_ , u8 >",
            ),
            (
                parse_quote! {
                    trait Tr<'a, T: 'a> {
                        fn f(&self, x: Box<dyn Fn(&Rel<T>) -> Box<dyn Iterator<Item = &u8> + '_>>);
                    }
                },
                "dyn Fn (& Rel < T >) -> Box < dyn Iterator < Item = & u8 > + '_ >",
            ),
            // The input is the only lifetime, whatever it is: the entry adds `'a` to it.
            (
                parse_quote! {
                    trait Tr<'a, T: 'a> { fn f(&self, x: Box<dyn Fn(&'static Rel<T>) -> &u8>); }
                },
                "dyn Fn (& 'static Rel < T >) -> & u8",
            ),
            (
                parse_quote! {
                    trait Tr<'a, T: 'a> { fn f(&self, x: Box<dyn for<'x> Fn(&'x Rel<T>) -> &u8>); }
                },
                "dyn for < 'x > Fn (& 'x Rel < T >) -> & u8",
            ),
        ] {
            let item: ItemTrait = item;
            let found = leaked(item.clone());
            assert_eq!(found.len(), 1, "{}: {:?}", s(&item), found);
            assert_eq!(found[0].0, holder, "{}", s(&item));
            assert!(found[0].1.contains("'a"), "{}: {:?}", s(&item), found);
        }
        // The generics are those of the whole: the lifetimes and the parameters the holder uses.
        assert_eq!(
            leaked(parse_quote! {
                trait Tr<'a, 'b, T: 'a, U: 'b> { fn f(&self, x: Box<dyn Fn(&Rel<T>) -> &u8>, y: &'b U); }
            }),
            [e("dyn Fn (& Rel < T >) -> & u8", "< 'a , T : 'a >")]
        );
        // What else is in it is covered, so it is not leaked on its own.
        assert_eq!(
            leaked(parse_quote! {
                trait Tr<'a, T: 'a> { fn f(&self, x: Box<dyn Fn(&Rel<T>, u8) -> &Rel2>); }
            }),
            [e("dyn Fn (& Rel < T > , u8) -> & Rel2", "< 'a , T : 'a >")]
        );
        // `Self` in an entry: the holder is leaked with it.
        assert_eq!(
            leaked(parse_quote! {
                trait Tr<'a> where Self: Sized + 'a { fn f(&self, x: Box<dyn Fn(&Rel<Self>) -> &u8>); }
            }),
            [e("dyn Fn (& Rel < Self >) -> & u8", "< 'a >")]
        );
        // The fields of a struct.
        let item: ItemStruct = parse_quote! {
            struct S<'a, T: 'a> { f: fn(&Rel<T>) -> &u8, g: Box<dyn Fn(&Rel<T>) -> &u8>, h: &'a Rel<T> }
        };
        let mut leaker = leaker();
        leaker.intern_struct(&item, true);
        let referrer = leaker.result().finish().unwrap();
        let mut found: Vec<String> = crate::common::types(&referrer).map(s).collect();
        found.sort();
        assert_eq!(
            found,
            [
                "& 'a Rel < T >",
                "dyn Fn (& Rel < T >) -> & u8",
                "fn (& Rel < T >) -> & u8"
            ]
        );
    }

    #[test]
    fn a_holder_inside_a_signature_is_the_entry_for_the_one_around() {
        // The inner one is the holder of `Rel<T>`, and it adds `'a` to the inputs of the outer one,
        // which is the entry in turn.
        assert_eq!(
            leaked(parse_quote! {
                trait Tr<'a, T: 'a> {
                    fn f(&self, x: Box<dyn Fn(&Box<dyn Fn(&Rel<T>) -> &u8>) -> &u8>);
                }
            }),
            [e(
                "dyn Fn (& Box < dyn Fn (& Rel < T >) -> & u8 >) -> & u8",
                "< 'a , T : 'a >"
            )]
        );
        // The lifetimes in a signature inside an input are not those of the one around: the outer one
        // has only `&`, and the inner one names its own lifetime, so `Rel<T>` is leaked alone.
        assert_eq!(
            leaked(parse_quote! {
                trait Tr<'a, T: 'a> {
                    fn f(&self, x: Box<dyn Fn(&Box<dyn for<'x> Fn(&'x Rel<T>) -> &'x u8>) -> &u8>);
                }
            }),
            [e("Rel < T >", "< 'a , T : 'a >")]
        );
        // The output of a signature is not in its inputs: an inner one there is its own holder.
        assert_eq!(
            leaked(parse_quote! {
                trait Tr<'a, T: 'a> {
                    fn f(&self, x: Box<dyn Fn(&u8) -> Box<dyn Fn(&Rel<T>) -> &u8>>);
                }
            }),
            [e("dyn Fn (& Rel < T >) -> & u8", "< 'a , T : 'a >")]
        );
        // A method with a receiver that is a reference: its own elision is not at stake.
        assert_eq!(
            leaked(parse_quote! {
                trait Tr<'a, T: 'a> {
                    fn f(&self, x: &Rel<T>, y: Box<dyn Fn(&Rel<T>) -> &u8>) -> &u8;
                }
            }),
            [
                e("Rel < T >", "< 'a , T : 'a >"),
                e("dyn Fn (& Rel < T >) -> & u8", "< 'a , T : 'a >")
            ]
        );
    }

    /// The `Leaker` docs, Elided lifetimes: "a function pointer or `Fn(..)` inside an input has its
    /// own signature". The same holds for one in the OUTPUT of the sugar: its elided lifetimes are its
    /// own and do not count for the outer one, whose only lifetime is that of `&`. So `Rel<T>` is
    /// leaked alone, whatever the signature in the output elides. Audit mutant
    /// `e_output_barefn_not_own` (a bare function pointer in the output counts for the outer
    /// signature, and the whole object is leaked).
    #[test]
    fn an_inner_signature_in_the_output_has_its_own_elision() {
        for output in [
            // A function pointer, an `Fn(..)`, and one with a binder in the inputs.
            "fn(&u8) -> &u8",
            "Box<dyn Fn(&u8) -> &u8>",
            "for<'y> fn(&'y u8) -> &'y u8",
            // In a tuple with another type, and behind a reference.
            "(u8, fn(&u8) -> &u8)",
            "Box<fn(&u8) -> &u8>",
        ] {
            let tr: ItemTrait = parse_str(&format!(
                "trait Tr<'a, T: 'a> {{ fn f(&self, x: Box<dyn Fn(&Rel<T>) -> {}>); }}",
                output
            ))
            .unwrap();
            assert_eq!(
                leaked(tr),
                [e("Rel < T >", "< 'a , T : 'a >")],
                "{}",
                output
            );
        }
        // An inner signature in the inputs that names its lifetimes is its own too: `Rel<T>` is leaked
        // alone. Without the names, the inner one is the holder of `Rel<T>`, and adds `'a` to the
        // inputs of the one around, which is the entry in turn (see the test above).
        assert_eq!(
            leaked(parse_quote! {
                trait Tr<'a, T: 'a> {
                    fn f(&self, x: Box<dyn Fn(for<'y> fn(&'y Rel<T>) -> &'y u8, &u8) -> &u8>);
                }
            }),
            [e("Rel < T >", "< 'a , T : 'a >")]
        );
        assert_eq!(
            leaked(parse_quote! {
                trait Tr<'a, T: 'a> {
                    fn f(&self, x: Box<dyn Fn(fn(&Rel<T>) -> &u8, &u8) -> &u8>);
                }
            }),
            [e(
                "dyn Fn (fn (& Rel < T >) -> & u8 , & u8) -> & u8",
                "< 'a , T : 'a >"
            )]
        );
    }

    #[test]
    fn a_type_with_no_lifetime_is_leaked_alone() {
        // No lifetime among the generics of the entry, so the copy has only the elided one.
        for item in [
            parse_quote! { trait Tr<T> { fn f(&self, x: Box<dyn Fn(&Rel<T>) -> &u8>); } },
            parse_quote! { trait Tr<'a, T> { fn f(&self, x: Box<dyn Fn(&Rel<T>) -> &u8>); } },
            parse_quote! { trait Tr<T> { fn f(&self, x: fn(&Rel<T>) -> &u8); } },
            parse_quote! { trait Tr { fn f(&self, x: Box<dyn Fn(&Rel) -> &u8>); } },
            parse_quote! { trait Tr<T> { fn g(x: &Rel<T>) -> &u8; } },
            parse_quote! { trait Tr<T, F: Fn(&Rel<T>) -> &u8> {} },
        ] {
            let item: ItemTrait = item;
            let found = leaked(item.clone());
            assert_eq!(found.len(), 1, "{}: {:?}", s(&item), found);
            assert!(found[0].0.starts_with("Rel"), "{}: {:?}", s(&item), found);
            assert!(!found[0].1.contains("'a"), "{}: {:?}", s(&item), found);
        }
    }

    #[test]
    fn the_elision_is_not_at_stake_if_the_output_does_not_rely_on_it() {
        let alone = [e("Rel < T >", "< 'a , T : 'a >")];
        for item in [
            // The output has no lifetime, or names it.
            parse_quote! { trait Tr<'a, T: 'a> { fn f(&self, x: Box<dyn Fn(&Rel<T>) -> u8>); } },
            parse_quote! { trait Tr<'a, T: 'a> { fn f(&self, x: Box<dyn Fn(&Rel<T>) -> Rel<T>>); } },
            parse_quote! { trait Tr<'a, T: 'a> { fn f(&self, x: Box<dyn Fn(&Rel<T>) -> &'a u8>); } },
            parse_quote! { trait Tr<'a, T: 'a> { fn f(&self, x: fn(&Rel<T>) -> &'static u8); } },
            parse_quote! { trait Tr<'a, T: 'a> { fn f(&self, x: fn(&Rel<T>)); } },
            // The output of an inner signature is its own.
            parse_quote! {
                trait Tr<'a, T: 'a> { fn f(&self, x: Box<dyn Fn(&Rel<T>) -> Box<dyn Fn(&u8) -> &u8>>); }
            },
            // The output names the lifetime of the input.
            parse_quote! {
                trait Tr<'a, T: 'a> { fn f(&self, x: Box<dyn for<'x> Fn(&'x Rel<T>) -> &'x u8>); }
            },
            parse_quote! { trait Tr<'a, T: 'a> { fn f(&self, x: for<'x> fn(&'x Rel<T>) -> &'x u8); } },
            // The entry is in the output.
            parse_quote! { trait Tr<'a, T: 'a> { fn f(&self, x: Box<dyn Fn(&u8) -> &Rel<T>>); } },
            // More than one lifetime in the inputs, or none: there is no elision to keep.
            parse_quote! { trait Tr<'a, T: 'a> { fn f(&self, x: fn(&u8, &Rel<T>) -> u8); } },
            // A method takes the lifetime of the output from the receiver that is a reference.
            parse_quote! { trait Tr<'a, T: 'a> { fn g(&self, x: &Rel<T>) -> &u8; } },
            parse_quote! { trait Tr<'a, T: 'a> { fn g(&mut self, x: &Rel<T>) -> &u8; } },
            parse_quote! { trait Tr<'a, T: 'a> { fn g(&'a self, x: &Rel<T>) -> &u8; } },
            parse_quote! { trait Tr<'a, T: 'a> { fn g(self: &Self, x: &Rel<T>) -> &u8; } },
            parse_quote! { trait Tr<'a, T: 'a> { fn g(self: &mut Self, x: &Rel<T>) -> &u8; } },
            parse_quote! {
                trait Tr<'a, T: 'a> { fn g(self: ::core::pin::Pin<&mut Self>, x: &Rel<T>) -> &u8; }
            },
            // Or it has no elided lifetime.
            parse_quote! { trait Tr<'a, T: 'a> { fn g(x: &Rel<T>) -> &'a u8; } },
            parse_quote! { trait Tr<'a, T: 'a> { fn g(x: &Rel<T>) -> u8; } },
        ] {
            let item: ItemTrait = item;
            assert_eq!(leaked(item.clone()), alone, "{}", s(&item));
        }
        // The only lifetime in the inputs is the one the entry has: nothing is added.
        assert_eq!(
            leaked(parse_quote! { trait Tr<'a, T: 'a> { fn g(x: &'a Rel<T>) -> &u8; } }),
            [e("& 'a Rel < T >", "< 'a , T : 'a >")]
        );
        assert_eq!(
            leaked(parse_quote! { trait Tr<'a, T: 'a> { fn g(x: Rel2<'a>, y: &'a u8) -> &u8; } }),
            [e("Rel2 < 'a >", "< 'a >")]
        );
    }

    #[test]
    fn a_method_with_no_lifetime_from_its_receiver_is_an_error() {
        // Nothing holds a method, so the type that adds the lifetime is an error.
        for item in [
            parse_quote! { trait Tr<'a, T: 'a> { fn g(x: &Rel<T>) -> &u8; } },
            parse_quote! { trait Tr<'a, T: 'a> { fn g(x: &Rel<T>, y: u8) -> &u8; } },
            // A receiver that is not a reference.
            parse_quote! { trait Tr<'a, T: 'a>: Sized { fn g(self, x: &Rel<T>) -> &u8; } },
            parse_quote! { trait Tr<'a, T: 'a> { fn g(self: Box<Self>, x: &Rel<T>) -> &u8; } },
            parse_quote! { trait Tr<'a, T: 'a> { fn g(self: ::std::rc::Rc<Self>, x: &Rel<T>) -> &u8; } },
            // The type is not at the top of the input.
            parse_quote! { trait Tr<'a, T: 'a> { fn g(x: &(u8, Rel<T>)) -> &u8; } },
            parse_quote! { trait Tr<'a, T: 'a> { fn g(x: &[Rel<T>]) -> &u8; } },
        ] {
            let item: ItemTrait = item;
            let (message, note) = error(item.clone());
            assert_eq!(message, "use absolute path: 'Rel<T>'", "{}", s(&item));
            assert_eq!(note, NOTE, "{}", s(&item));
            assert_eq!(help(&item), HELP_METHOD, "{}", s(&item));
        }
        // Another output with an elided lifetime.
        for (item, output) in [
            (
                parse_quote! { trait Tr<'a, T: 'a> { fn g(x: &Rel<T>) -> ::p::Ref<'_, u8>; } },
                "::p::Ref<'_, u8>",
            ),
            (
                parse_quote! { trait Tr<'a, T: 'a> { fn g(x: &Rel<T>) -> Box<dyn Iterator<Item = &u8>>; } },
                "Box<dyn Iterator<Item = &u8>>",
            ),
        ] {
            let item: ItemTrait = item;
            let (_, note) = error(item.clone());
            assert_eq!(
                note,
                NOTE.replace("`&u8`", &format!("`{}`", output)),
                "{}",
                s(&item)
            );
        }
        // The message, as rustc shows it: the lifetime that is elided is the cause.
        let err = analyze(&parse_quote! { trait Tr<'a, T: 'a> { fn g(x: &Rel<T>) -> &u8; } })
            .err()
            .unwrap();
        assert_eq!(
            to_syn_error(&err).to_string(),
            format!(
                "use absolute path: 'Rel<T>'\n\n  = note: {}\n  = note: it cannot be leaked as a \
                 whole because of `&u8`\n  = help: {}\n",
                NOTE, HELP_METHOD
            )
        );
        // A function item of a module has no receiver either.
        let item: ItemMod =
            parse_quote! { mod m { pub fn g<'a, T: 'a>(x: &Rel<T>) -> &u8 { loop {} } } };
        let mut module_leaker = leaker();
        module_leaker.intern_mod(&item);
        assert_eq!(
            module_leaker.result().errors()[0]
                .notes
                .first()
                .map(|(_, n)| n.as_str()),
            Some(NOTE)
        );
        // And an impl.
        let item: ItemImpl = parse_quote! {
            impl<'a, T: 'a> ::p::Tr<'a, T> for u8 { fn g(x: &Rel<T>) -> &u8 { loop {} } }
        };
        let mut impl_leaker = leaker();
        impl_leaker.intern_impl(&item, true);
        assert_eq!(
            impl_leaker.result().errors()[0]
                .notes
                .first()
                .map(|(_, n)| n.as_str()),
            Some(NOTE)
        );
    }

    #[test]
    fn a_bound_with_the_elision_is_an_error() {
        // Nothing holds a bound: it is not a type.
        for item in [
            parse_quote! { trait Tr<'a, T: 'a, F: Fn(&Rel<T>) -> &u8> {} },
            parse_quote! { trait Tr<'a, T: 'a, F> where F: Fn(&Rel<T>) -> &u8 {} },
            parse_quote! { trait Tr<'a, T: 'a>: Fn(&Rel<T>) -> &u8 {} },
            parse_quote! { trait Tr<'a, T: 'a> { type A: Fn(&Rel<T>) -> &u8; } },
            parse_quote! { trait Tr<'a, T: 'a> { fn f<F: Fn(&Rel<T>) -> &u8>(&self, f: F); } },
            parse_quote! { trait Tr<'a, T: 'a> { fn f<F>(&self, f: F) where F: Fn(&Rel<T>) -> &u8; } },
            parse_quote! { trait Tr<'a, T: 'a> { fn f<F>(&self, f: F) where Box<F>: Fn(&Rel<T>) -> &u8; } },
        ] {
            let item: ItemTrait = item;
            let (message, note) = error(item.clone());
            assert_eq!(message, "use absolute path: 'Rel<T>'", "{}", s(&item));
            assert_eq!(note, NOTE, "{}", s(&item));
            assert_eq!(help(&item), HELP, "{}", s(&item));
        }
        // No lifetime in the entry, or no elision: fine.
        assert_eq!(
            leaked(parse_quote! { trait Tr<T, F: Fn(&Rel<T>) -> &u8> {} }),
            [e("Rel < T >", "< T >")]
        );
        assert_eq!(
            leaked(parse_quote! { trait Tr<'a, T: 'a, F: for<'x> Fn(&'x Rel<T>) -> &'x u8> {} }),
            [e("Rel < T >", "< 'a , T : 'a >")]
        );
    }

    #[test]
    fn a_holder_that_cannot_be_leaked_is_an_error() {
        // `impl Trait` cannot be leaked at all.
        let (message, note) = error(parse_quote! {
            trait Tr<'a, T: 'a> { fn f(&self, x: impl Fn(&Rel<T>) -> &u8); }
        });
        assert_eq!(message, "use absolute path: 'Rel<T>'");
        assert_eq!(note, NOTE_OWNED);
        // A trait object under an elided reference takes its lifetime from it, so it cannot be leaked
        // alone, and the reference has no lifetime to name.
        let err = analyze(&parse_quote! {
            trait Tr<'a, T: 'a> { fn f(&self, x: &dyn Fn(&Rel<T>) -> &u8); }
        })
        .err()
        .unwrap();
        assert_eq!(
            to_syn_error(&err).to_string(),
            format!(
                "use absolute path: 'Rel<T>'\n\n  = note: {}\n  = note: it cannot be leaked as a \
                 whole because of `dyn Fn (&Rel<T>) -> &u8`\n  = note: the lifetime of the trait \
                 object defaults to that of the enclosing reference or path\n  = help: {}\n",
                NOTE_OWNED, HELP
            )
        );
        // Naming it is the fix.
        assert_eq!(
            leaked(parse_quote! {
                trait Tr<'a, T: 'a> { fn f(&self, x: &'a dyn Fn(&Rel<T>) -> &u8); }
            }),
            [e("& 'a dyn Fn (& Rel < T >) -> & u8", "< 'a , T : 'a >")]
        );
    }

    #[test]
    fn an_elided_lifetime_of_the_holder_is_its_own() {
        // `&u8` is a lifetime of the function pointer (rustc makes it one of its binder), so it does
        // not stop it from being leaked as a whole: the impl of the Repeater writes it as it is.
        assert_eq!(
            leaked(parse_quote! {
                trait Tr { fn f(&self, f: for<'x> fn(Rel<'x>, &u8) -> &'x u8); }
            }),
            [e("for < 'x > fn (Rel < 'x > , & u8) -> & 'x u8", "")]
        );
        assert_eq!(
            leaked(parse_quote! {
                trait Tr { fn f(&self, f: Box<dyn for<'x> Fn(Rel<'x>, &u8) -> &'x u8>); }
            }),
            [e("dyn for < 'x > Fn (Rel < 'x > , & u8) -> & 'x u8", "")]
        );
        assert_eq!(
            leaked(parse_quote! {
                trait Tr { fn f(&self, f: for<'x> fn(Rel<'x>, ::p::Ref<'_, u8>) -> &'x u8); }
            }),
            [e(
                "for < 'x > fn (Rel < 'x > , :: p :: Ref < '_ , u8 >) -> & 'x u8",
                ""
            )]
        );
        // The types inside it are not the holder: a lifetime that is elided in them is not theirs, so
        // they cannot be leaked on their own.
        let (message, note) = error(parse_quote! {
            trait Tr { fn f(&self, f: Box<dyn Fn(Rel<&Rel2>) -> u8>); }
        });
        assert_eq!(message, "use absolute path: 'Rel<&Rel2>'");
        assert_eq!(note, "it cannot be leaked as a whole because of `&Rel2`");
        // The holder is not an entry for that reason alone: with no cause in it, it is copied.
        let referrer = analyze(&parse_quote! {
            trait Tr { fn f(&self, f: Box<dyn Fn(&u8) -> &u8>, g: for<'x> fn(&'x u8, &u8)); }
        })
        .unwrap();
        assert_eq!(referrer.iter().count(), 0);
    }

    #[test]
    fn the_copy_is_rewritten_to_the_holder() {
        let item: ItemTrait = parse_quote! {
            trait Tr<'a, T: 'a> {
                fn f(&self, a: Box<dyn Fn(&Rel<T>) -> &u8>, b: fn(&Rel<T>) -> &u8, c: &Rel<T>);
            }
        };
        let mut leaker = leaker();
        let context = leaker.intern_trait(&item, true);
        let referrer = leaker.result().finish().unwrap();
        // The entries are ordered by context and then by their tokens.
        assert_eq!(
            crate::common::types(&referrer).map(s).collect::<Vec<_>>(),
            [
                "Rel < T >",
                "dyn Fn (& Rel < T >) -> & u8",
                "fn (& Rel < T >) -> & u8"
            ]
        );
        let mut copy = item.clone();
        syn::visit_mut::VisitMut::visit_item_trait_mut(
            &mut referrer.visitor(context, |_, ix, _| parse_quote!(P<#ix>)),
            &mut copy,
        );
        let expected: ItemTrait = parse_quote! {
            trait Tr<'a, T: 'a> {
                fn f(&self, a: Box<P<1usize>>, b: P<2usize>, c: &P<0usize>);
            }
        };
        assert_eq!(s(&copy), s(&expected));
    }

    #[test]
    fn a_view_walks_the_signature_like_the_leaker() {
        use syn::visit::Visit;
        // `visit_signature` of the view analyzes the arguments of a method with its elision.
        let sig: Signature = parse_quote!(fn g(x: &Rel<T>) -> &u8);
        let generics: Generics = parse_quote!(<'a, T: 'a>);
        let mut leaker = leaker();
        leaker.enter_generics(&generics, |view| view.visit_signature(&sig));
        assert_eq!(
            leaker.result().errors()[0]
                .notes
                .first()
                .map(|(_, n)| n.as_str()),
            Some(NOTE)
        );
        // And the sugar and the function pointers in a type visited alone.
        let ty: Type = parse_quote!(Box<dyn Fn(&Rel<T>) -> &u8>);
        let mut leaker = self::leaker();
        leaker.enter_generics(&generics, |view| view.visit_type(&ty));
        assert_eq!(
            leaker
                .result()
                .finish()
                .unwrap()
                .iter()
                .map(|(_, ty, _)| s(ty))
                .collect::<Vec<_>>(),
            ["dyn Fn (& Rel < T >) -> & u8"]
        );
    }

    #[test]
    fn an_elided_output_may_take_the_lifetime_of_the_binder() {
        // The output has the only lifetime of the inputs, `'x` of the binder, which `Rel<'x>` leaked
        // alone would have only in a projection: rustc rejects the copy (E0581, E0582) as it does for
        // an output that names `'x` (see the tests of the binder), so the holder is the entry.
        assert_eq!(
            leaked(parse_quote! { trait Tr { fn f(&self, f: for<'x> fn(Rel<'x>) -> &u8); } }),
            [e("for < 'x > fn (Rel < 'x >) -> & u8", "")]
        );
        assert_eq!(
            leaked(
                parse_quote! { trait Tr { fn f(&self, f: Box<dyn for<'x> Fn(Rel<'x>) -> &u8>); } }
            ),
            [e("dyn for < 'x > Fn (Rel < 'x >) -> & u8", "")]
        );
        // An input that keeps `'x` anchors it, whatever the output is.
        assert_eq!(
            leaked(
                parse_quote! { trait Tr { fn f(&self, f: for<'x> fn(&'x u8, Rel<'x>) -> &u8); } }
            ),
            [e("Rel < 'x >", "< 'x >")]
        );
        // A bound or a where predicate cannot be leaked, so it is an error.
        for item in [
            parse_quote! { trait Tr { fn f<F: for<'x> Fn(Rel<'x>) -> &u8>(&self, f: F); } },
            parse_quote! { trait Tr { fn f<F>(&self, f: F) where F: for<'x> Fn(Rel<'x>) -> &u8; } },
            parse_quote! { trait Tr { fn f<F>(&self, f: F) where for<'x> F: Fn(Rel<'x>) -> &u8; } },
        ] {
            let item: ItemTrait = item;
            let (message, _) = error(item.clone());
            assert_eq!(message, "use absolute path: 'Rel<'x>'", "{}", s(&item));
        }
        // The lifetime of the trait is not declared by the binder, so it is not at stake.
        assert_eq!(
            leaked(parse_quote! { trait Tr<'a> { fn f(&self, f: fn(Rel<'a>) -> &u8); } }),
            [e("Rel < 'a >", "< 'a >")]
        );
        assert_eq!(
            leaked(parse_quote! { trait Tr { fn f(&self, f: for<'x> fn(Rel<'x>, u8) -> u8); } }),
            [e("Rel < 'x >", "< 'x >")]
        );
    }
}

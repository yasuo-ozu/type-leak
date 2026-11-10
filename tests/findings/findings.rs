//! Findings of the property test `tests/prop_cross_crate`, as library-level tests.
//!
//! The property test generates crates, builds them, and reports what is wrong in them: a Leaker's
//! crate or a Referrer's crate that does not compile, or an item that should have been rejected.
//! Each finding is minimized here to a test of the API of `type-leak`, with no generated crate and
//! no `cargo` build. A test asserts what the library produces today: what a fix made it produce
//! (`fixed`), or what the docs of `Leaker` document (`limitation`). The doc of each test `fN_..`
//! says the trigger, the cause and the fix of the finding FN.
//!
//! Most fixes are rejections: the Leaker reports an error for an item whose copy rustc cannot
//! compile. The test of such an item asserts the rejection, with the exact headline of the error
//! and its note (`assert_rejected()`, `assert_unstable()`). The controls, the items that compile,
//! are asserted accepted, with their exact entries or copy.
//!
//! A macro replaces every entry through the Repeater: the `Referrer` does not tell the entries that
//! depend on the context from the optional ones, the types of another crate. A type of another
//! crate (`::foo::X`) is an entry, unless the rule of the Leaker allows its path
//! (`Rules::external_crates()`, `or_if(starts_with(..))`): it is no entry then, and the copy
//! writes it as it is.

use crate::common;

use common::{
    arguments, entry_index, messages, parts_of, repeater, s, starts_with, t, with_where, Recovered,
};
use syn::punctuated::Punctuated;
use syn::visit::Visit;
use syn::visit_mut::VisitMut;
use syn::*;
use template_quote::quote;
use type_leak::*;

// ── helpers ──────────────────────────────────────────────────────────────────

/// Accepts the `NestedSelfInBounds` errors of `result` (the risk of an entry for any type that names
/// `Self`), as a macro that takes the risk does: the entries stay, the other errors too.
fn accept_nested_self(result: LeakerResult) -> LeakerResult {
    let risks: Vec<Cause> = result
        .errors()
        .iter()
        .filter(|error| error.reason == Reason::NestedSelfInBounds)
        .map(|error| error.cause.clone())
        .collect();
    result.allow(move |cause: &Cause, _: Place| risks.contains(cause))
}

/// Analyzes `item` with `leaker`, and returns the Referrer with the top-level context, or the
/// errors that remain.
fn analyze_in<A: Allow>(
    leaker: Leaker<A>,
    item: &Item,
) -> std::result::Result<(ContextId, Referrer), LeakerResultError> {
    analyze_result_in(leaker, item).map(|(context, result)| (context, result.finish().unwrap()))
}

/// Like `analyze_in()`, with the result that writes the impls.
fn analyze_result_in<A: Allow>(
    mut leaker: Leaker<A>,
    item: &Item,
) -> std::result::Result<(ContextId, LeakerResult), LeakerResultError> {
    let context = common::intern_item(&mut leaker, item);
    let result = leaker.result();
    result.clone().finish().map(|_| (context, result))
}

/// The errors that remain after the analysis of `leaker`.
fn errors_of<A: Allow>(leaker: Leaker<A>) -> Vec<ErrorItem> {
    leaker.result().errors().to_vec()
}

/// Asserts that the Leaker rejected `item`, with exactly the `messages` (the headline of each error,
/// in order, joined with a line break by `messages()`), and returns the errors.
///
/// A bug input has to be rejected. A loop that skips the rejected items (`let Ok(..) = .. else {
/// continue }`) runs its checks on none of them, and pins nothing: it would be the same if the
/// rejection regressed, and the checks run for the items that are accepted only.
fn assert_rejected<T, E: std::ops::Deref<Target = [ErrorItem]>>(
    item: &impl template_quote::ToTokens,
    result: std::result::Result<T, E>,
    messages: &[&str],
) -> Vec<ErrorItem> {
    let err = match result {
        Ok(_) => panic!("`{}` is accepted, and it does not compile", s(item)),
        Err(err) => err,
    };
    assert_eq!(common::messages(&err), messages.join("\n"), "`{}`", s(item));
    err.to_vec()
}

/// Asserts that the Leaker accepted `item`, and returns the value: the mirror of
/// `assert_rejected()`. The panic has the messages and the notes of the errors.
fn assert_accepted<T, E: std::ops::Deref<Target = [ErrorItem]>>(
    item: &impl template_quote::ToTokens,
    result: std::result::Result<T, E>,
) -> T {
    assert_accepted_as(&format!("`{}`", s(item)), result)
}

/// Like `assert_accepted()`, for a case that `what` names (the item, with its scope or option).
fn assert_accepted_as<T, E: std::ops::Deref<Target = [ErrorItem]>>(
    what: &str,
    result: std::result::Result<T, E>,
) -> T {
    result.unwrap_or_else(|err| {
        panic!(
            "{} is rejected, and it compiles: {}\n{}",
            what,
            messages(&err),
            all_notes(&err).join("\n")
        )
    })
}

/// The notes of all the errors of `err`, in order.
fn all_notes(err: &[ErrorItem]) -> Vec<String> {
    err.iter()
        .flat_map(|item| item.notes.iter().map(|(_, note)| note.clone()))
        .collect()
}

/// The helps of all the errors of `err`, in order.
fn all_helps(err: &[ErrorItem]) -> Vec<String> {
    err.iter()
        .filter_map(|item| item.help.as_ref().map(|(_, help)| help.clone()))
        .collect()
}

/// Tokens as a message writes them: `Wrap < T >` is `Wrap<T>`, `Self :: A` is `Self::A`, and a
/// path after a keyword or a separator keeps its space (`dyn ::p::Tr`, `A, ::p::X`).
fn plain(text: &str) -> String {
    let mut text = text.to_string();
    for before in [
        "as", "dyn", "impl", "for", "in", "mut", "const", "where", ",", ":", "->", "=",
    ] {
        text = text.replace(&format!("{} :: ", before), &format!("{} ::", before));
    }
    [
        (" :: ", "::"),
        (":: ", "::"),
        (" <", "<"),
        ("< ", "<"),
        (" >", ">"),
        ("& ", "&"),
        (" ,", ","),
        (" ;", ";"),
    ]
    .iter()
    .fold(text, |text, (from, to)| text.replace(from, to))
}

/// What the bound that an error of an unstable bound blames names (`Why` of
/// `src/analyze/unstable.rs`).
#[derive(Clone, Copy)]
enum Blames<'a> {
    /// A type that is leaked too.
    Leaked,
    /// `Self`, the self type of an impl.
    SelfType,
    /// `Self::A` (or `<Self as Tr>::A`), as written.
    SelfAssoc(&'a str),
    /// An associated type that rustc normalizes (`T::Item`), as written.
    Projection(&'a str),
    /// A path that may be a type alias of a projection (F64): the generic, and its bound that
    /// names it in the arguments of its path, as written.
    Hidden(&'a str, &'a str),
    /// A type that is leaked too, in a bound of the impl that rustc proves to resolve `Self::A` by
    /// the impl of the trait.
    LeakedInImpl,
}

/// Asserts that the Leaker rejected `item` for the type `ty` (as the error writes it) in a bound or
/// where predicate, because the bound `bound` of its generics is one that the environment of the
/// copy cannot prove. The error has the headline, and the one note, that say what the bound
/// `blames` (`env_cycle()` and `self_env_cycle()` of `src/analyze/predicates.rs`). `ty` is `Self`
/// for the self type of an impl.
fn assert_unstable<T, E: std::ops::Deref<Target = [ErrorItem]>>(
    item: &impl template_quote::ToTokens,
    result: std::result::Result<T, E>,
    ty: &str,
    bound: &str,
    blames: Blames,
) -> Vec<ErrorItem> {
    // The errors write tokens as a help does.
    let (ty, bound) = (plain(ty), plain(bound));
    let (ty, bound) = (ty.as_str(), bound.as_str());
    // What the bound names, after "the bound `..` of its generics".
    let names = |blames: Blames| match blames {
        Blames::SelfType => "names `Self`, which the copy writes as the self type of the impl, a \
                             path through the Repeater that rustc has not normalized yet in the \
                             environment, where it must prove the bound (E0277, E0271)"
            .to_string(),
        Blames::SelfAssoc(path) => format!(
            "names `{}`, which rustc normalizes to the associated type of the host in the impl of \
             the Repeater, but not yet in the environment (E0277, E0271)",
            plain(path)
        ),
        Blames::Projection(path) => format!(
            "names `{}`, which rustc normalizes in the impl of the Repeater (an impl, a predicate \
             of the item that equates it, or a supertrait of a bound of its generic, defines it), \
             but not yet in the environment (E0277, E0271, E0275)",
            plain(path)
        ),
        Blames::Hidden(generic, hiding) => format!(
            "names `{}` in the arguments of the bound `{}`, whose path may be a type alias of a \
             projection that another bound of `{}` equates (an alias is not told from a \
             struct): rustc normalizes it in the impl of the Repeater, but not yet in the \
             environment (E0277)",
            generic,
            plain(hiding),
            generic
        ),
        Blames::Leaked | Blames::LeakedInImpl => unreachable!(),
    };
    let of_self_type = ty == "Self";
    let why = match blames {
        Blames::Leaked if !of_self_type => format!(
            "the bound `{}` of its generics names a type that is leaked too: the copy writes both \
             as paths through the Repeater, and rustc cannot normalize the one in a bound or where \
             predicate while it checks the bounds (E0277, E0275)",
            bound
        ),
        Blames::Leaked => format!(
            "`Self` is the self type of the impl, which the copy writes with its leaked types as \
             paths through the Repeater, and the bound `{}` of their generics names a type that is \
             leaked too, and rustc cannot normalize the one in a bound or where predicate while it \
             checks the bounds (E0277, E0275)",
            bound
        ),
        Blames::LeakedInImpl => format!(
            "the copy for a concrete implementor resolves `{}` by the impl of the trait, which \
             needs the bound `{}` of the impl, where the copy writes the leaked type as a path \
             through the Repeater, and rustc cannot normalize the one in a bound or where \
             predicate while it checks the bounds (E0277, E0275)",
            ty, bound
        ),
        blames if !of_self_type => {
            format!("the bound `{}` of its generics {}", bound, names(blames))
        }
        blames => format!(
            "`Self` is the self type of the impl, which the copy writes with its leaked types as \
             paths through the Repeater, and the bound `{}` of their generics {}",
            bound,
            names(blames)
        ),
    };
    let (what, advice) = match blames {
        Blames::Leaked | Blames::LeakedInImpl => (
            "a leaked type".to_string(),
            "write the leaked type of the bound with an absolute path".to_string(),
        ),
        Blames::SelfType => (
            "`Self`, which is a leaked type".to_string(),
            "write the bound without `Self`".to_string(),
        ),
        Blames::SelfAssoc(path) | Blames::Projection(path) => (
            format!("`{}`", plain(path)),
            format!("write the bound without `{}`", plain(path)),
        ),
        Blames::Hidden(generic, hiding) => (
            format!(
                "`{}` in the arguments of `{}`, which may hide a projection",
                generic,
                plain(hiding)
            ),
            format!(
                "write the bound `{}` without a path that names `{}`",
                plain(hiding),
                generic
            ),
        ),
    };
    let headline = format!(
        "`{}` is in a bound or where predicate, and a bound of its generics names {}",
        ty, what
    );
    let err = assert_rejected(item, result, &[&headline]);
    // The note says why, and the help what to write.
    assert_eq!(all_notes(&err), [why], "`{}`", s(item));
    let help = format!(
        "{}, or use `{}` outside the bounds and where predicates",
        advice,
        plain(ty)
    );
    assert_eq!(all_helps(&err), [help], "`{}`", s(item));
    err
}

/// The copy of `item` (in the scope `context`) that a macro makes with `replaced()`.
fn copied(referrer: &Referrer, context: ContextId, mut item: Item) -> Item {
    common::visitor(referrer, context, common::placeholder).visit_item_mut(&mut item);
    item
}

/// Rewrites a copy of `item` (in the scope `context`), replacing the `ix`-th entry with `P<ix>`.
fn rewrite(referrer: &Referrer, context: ContextId, item: Item) -> String {
    s(&copied(referrer, context, item))
}

/// The entries of `referrer` as text: as the tokens print (`spaced`), or without spaces.
fn entries_text(referrer: &Referrer, spaced: bool) -> Vec<String> {
    common::types(referrer)
        .map(|ty| {
            if spaced {
                s(ty)
            } else {
                s(ty).replace(' ', "")
            }
        })
        .collect()
}

/// Asserts that the entries of `referrer`, the Referrer of `item`, are `expected` (without spaces).
fn assert_entries(item: &impl template_quote::ToTokens, referrer: &Referrer, expected: &[&str]) {
    assert_eq!(entries_text(referrer, false), expected, "`{}`", s(item));
}

/// The generics of the context of the entry `ix`, as the tokens print.
fn context_text(referrer: &Referrer, ix: usize) -> String {
    s(&common::entry(referrer, ix).generics)
}

/// Like `context_text()`, with the where clause.
fn context_text_with_where(referrer: &Referrer, ix: usize) -> String {
    generics_text(&common::entry(referrer, ix).generics)
}

/// The first note of `error`.
fn first_note(error: &ErrorItem) -> &str {
    error.notes.first().map(|(_, n)| n.as_str()).unwrap()
}

/// The generics that the Repeater impl of the entry `ix` declares, for an entry that does not name
/// `Self`.
fn impl_generics(result: &LeakerResult, ix: usize) -> Generics {
    let marker: Type = parse_quote!(crate::Marker);
    let r = result
        .repeater_impls(&Projection::new(
            repeater(),
            7,
            ImplSelf::Marker(marker.clone()),
        ))
        .unwrap()
        .remove(ix);
    r.generics
}

// ── F1 ───────────────────────────────────────────────────────────────────────

/// F1 (fixed): a `?Sized` type parameter is encoded as a bare tuple element.
///
/// Trigger: `struct S<T: ?Sized, U>(Rel<T, U>)`. The Repeater's arguments of the entry `Rel<T, U>`
/// were the tuple `(T, U)`, and the impl that a macro emits did not compile (E0277).
///
/// Cause: a tuple is well-formed only if all its elements but the last are `Sized`.
///
/// Fix: a parameter with a `?Sized` bound is encoded as `*const T`, which is `Sized` whatever `T`
/// is. Both sides encode the same generics of the entry's context, so they agree. The exact
/// encodings are in `an_unsized_parameter_is_encoded_as_a_pointer_to_it` in
/// `tests/referrer/encode.rs`.
#[test]
fn f01_unsized_type_param_encoding() {
    let param: Ident = parse_quote!(__S);
    // The item, and the encoding of the arguments of its entry. The bound is on the parameter,
    // wherever it was written (a where predicate is moved there).
    let cases: Vec<(ItemStruct, &str)> = vec![
        (
            parse_quote!(
                pub struct S<T: ?Sized, U>(Rel<T, U>);
            ),
            "(* const T , U)",
        ),
        (
            parse_quote!(
                pub struct S<A, B: ?Sized, C>(Rel<A, B, C>);
            ),
            "(A , * const B , C)",
        ),
        (
            parse_quote!(
                pub struct S<T, U: ?Sized>(Rel<T, U>);
            ),
            "(T , * const U)",
        ),
        (
            parse_quote!(
                pub struct S<T, U>(Rel<T, U>)
                where
                    T: ?Sized;
            ),
            "(* const T , U)",
        ),
        (
            parse_quote!(
                pub struct S<'a, T: ?Sized + 'a, U>(&'a Rel<T, U>);
            ),
            "(& 'a () , * const T , U , & 'a T , & 'a U)",
        ),
    ];
    for (item, encoded) in cases {
        let mut leaker = common::std_leaker();
        leaker.intern_struct(&item);
        let result = leaker.result();
        let referrer = assert_accepted(&item, result.clone().finish());
        assert_eq!(referrer.iter().count(), 1, "`{}`", s(&item));
        let r = result
            .repeater_impls(&Projection::new(
                repeater(),
                7,
                ImplSelf::Param(param.clone()),
            ))
            .unwrap()
            .remove(0);
        assert_eq!(s(&common::impl_args(&r)), encoded, "`{}`", s(&item));
        // The Referrer passes each parameter by its own name: the same encoding.
        let on_referrer_side =
            common::repeater_args(common::entry(&referrer, 0), common::identity, None).unwrap();
        assert_eq!(s(&on_referrer_side), encoded, "`{}`", s(&item));
    }
}

// ── F2 ───────────────────────────────────────────────────────────────────────
/// F2 (fixed): a bound that names an entry, and a where predicate that names an entry, in one item.
///
/// Trigger: `struct S<T: Fn(Rel)>(Wrap<T>) where Wrap<T>: Sized` (also in a trait, with `T` a
/// generic of the trait or of a method). `Rel` and `Wrap<T>` are entries. The Repeater's impl for
/// `Wrap<T>` has `T: Fn(Rel)`, the copy has `T: Fn(<() as Repeater<..>>::Type)`, and the predicate
/// `<..>::Type: Sized` makes rustc normalize `Wrap<T>` in the environment, which needs that impl:
/// E0277 "expected an `Fn(Rel)` closure, found `T`" (E0275 or E0284 for other bounds).
///
/// Cause: rustc does not normalize a projection in a bound or where predicate while it checks the
/// bounds, so the copy's bound is not the one that the impl needs.
///
/// Fix: the Leaker rejects the item (see "Bounds that name leaked types" in the `Leaker` docs).
/// The control `Tr3` (`where T: Sized`) compiles and stays accepted.
#[test]
fn f02_bound_names_leaked_type_env_cycle() {
    let items: Vec<Item> = vec![
        parse_quote!(
            pub struct S<T: Fn(Rel)>(Wrap<T>)
            where
                Wrap<T>: Sized;
        ),
        parse_quote!(
            pub trait Tr<T: Fn(Rel)> {
                fn f(&self, a: Wrap<T>)
                where
                    Wrap<T>: Sized;
            }
        ),
        parse_quote!(
            pub trait Tr2 {
                fn f<T: Fn(Rel)>(&self, a: Wrap<T>)
                where
                    Wrap<T>: Sized;
            }
        ),
    ];
    // The rejection is the fix: each is rejected, and the copy of none exists.
    for item in &items {
        assert_unstable(
            item,
            analyze_in(common::std_leaker(), item),
            "Wrap < T >",
            "T : Fn (Rel)",
            Blames::Leaked,
        );
    }
    // The control: `T: Sized` names no entry. It is accepted.
    let control: Item = parse_quote!(
        pub trait Tr3 {
            fn f<T: Fn(Rel)>(&self, a: Wrap<T>)
            where
                T: Sized;
        }
    );
    let (context, referrer) = assert_accepted(&control, analyze_in(common::std_leaker(), &control));
    assert_eq!(
        rewrite(&referrer, context, control),
        "pub trait Tr3 { fn f < T : Fn (P < 0usize >) > (& self , a : P < 1usize >) where T : Sized ; }"
    );
}

// ── F3 ───────────────────────────────────────────────────────────────────────

/// F3 (fixed): a where predicate that names a lifetime drags its type parameter into the context.
///
/// Trigger: `struct S<'a, T> where T: Iterator<Item = Bl<'a>> {..}`. The context of the entry
/// `Bl<'a>` became `<'a, T: Iterator<Item = Bl<'a>>>`, whereas with the inline bound it is `<'a>`.
/// The Visitor computed the context without `T`, never found the entry, and left
/// `crate::base::Bl<'a>` as written, which another crate cannot resolve.
///
/// Fix: a where predicate on a generic is a bound of it, as the inline one is, so the context is
/// `<'a>` in both. The test asserts that the entry is rewritten in each form.
#[test]
fn f03_where_predicate_pulls_param_into_context() {
    let items: Vec<Item> = vec![
        // The inline bound.
        parse_quote! {
            pub struct S1<'a, T: Iterator<Item = crate::base::Bl<'a>>> {
                f1: ::core::marker::PhantomData<&'a ()>,
                f2: ::core::marker::PhantomData<*const T>,
            }
        },
        // The same bound as a where predicate.
        parse_quote! {
            pub struct S<'a, T> where T: Iterator<Item = crate::base::Bl<'a>> {
                f1: ::core::marker::PhantomData<&'a ()>,
                f2: ::core::marker::PhantomData<*const T>,
            }
        },
        // With the entry in a field as well.
        parse_quote! {
            pub struct S2<'a, T> where T: Iterator<Item = crate::base::Bl<'a>> {
                f1: crate::base::Bl<'a>,
                f2: ::core::marker::PhantomData<*const T>,
            }
        },
    ];
    for item in items {
        let (context, referrer) = analyze_in(common::std_leaker(), &item).unwrap();
        let copy = rewrite(&referrer, context, item.clone());
        assert!(
            !copy.contains("Bl"),
            "the entry `Bl<'a>` is not rewritten: {}",
            copy
        );
    }
}

// ── F5 ───────────────────────────────────────────────────────────────────────

/// F5 (limitation): no `T: 'a` is recorded for a path with a lifetime; a written bound is.
///
/// Trigger: `impl<'a, T> E<'a, T>` with `struct E<'a, T> { f: &'a mut T }`, where rustc infers
/// `T: 'a`. The impl compiles through the implied bounds of its self type, but the Repeater's impl
/// for the entry `E<'a, T>` has no `T: 'a`: E0309 in the Leaker's crate.
///
/// Cause: the Leaker cannot read the definition of a path, so a lifetime argument records nothing
/// (the `Leaker` docs, "Outlives bounds").
///
/// Limitation: a guessed bound would make the copy stricter than the original for a path that
/// needs none (`struct Unrelated<'a, T>(&'a (), T)`). Write the bound: it is recorded, and encoded
/// in the Repeater's arguments as `&'a T`, which the impl does not declare.
#[test]
fn f05_inferred_outlives_of_self_type() {
    let encode = |result: &LeakerResult, ix: usize| {
        let param: Ident = parse_quote!(__S);
        let r = result
            .repeater_impls(&Projection::new(
                repeater(),
                7,
                ImplSelf::Param(param.clone()),
            ))
            .unwrap()
            .remove(ix);
        common::impl_args(&r)
    };

    // As written: nothing is recorded, so the impl cannot prove `T: 'a`.
    let item: ItemImpl = parse_quote! {
        impl<'a, T> E<'a, T> { pub fn f(&self) -> Rel { loop {} } }
    };
    let mut leaker = common::std_leaker();
    leaker.intern_impl(&item);
    let result = leaker.result();
    let referrer = result.clone().finish().unwrap();
    let ix = entry_index(&referrer, "E < 'a , T >");
    let generics = &common::entry(&referrer, ix).generics;
    assert_eq!(s(generics), "< 'a , T >");
    assert_eq!(encode(&result, ix), parse_quote!((&'a (), T)));

    // With the bound written: recorded, and encoded.
    let item: ItemImpl = parse_quote! {
        impl<'a, T: 'a> E<'a, T> { pub fn f(&self) -> Rel { loop {} } }
    };
    let mut leaker = common::std_leaker();
    leaker.intern_impl(&item);
    let result = leaker.result();
    let referrer = result.clone().finish().unwrap();
    let ix = entry_index(&referrer, "E < 'a , T >");
    assert_eq!(context_text(&referrer, ix), "< 'a , T : 'a >");
    assert_eq!(encode(&result, ix), parse_quote!((&'a (), T, &'a T)));
    // The impl declares the generics without the encoded bound.
    assert_eq!(s(&impl_generics(&result, ix)), "< 'a , T >");
}

// ── F6 ───────────────────────────────────────────────────────────────────────

/// F6 (fixed): `intern_impl` accepts `Self::A` of the implemented trait in a leaked type.
///
/// Trigger: `impl ::leaker::m::Tr for u8 { type A = u8; fn f(&self, x: Rel<Self::A>) {} }`. The
/// entry gives `impl Repeater<..> for u8 { type Type = Rel<Self::A>; }`, where `A` is no
/// associated type of the Repeater (E0220).
///
/// Cause: the context says that `Self` is `u8`, and nothing says which trait `A` belongs to.
///
/// Fix: an error that asks for `<Self as ::leaker::m::Tr>::A`, also where only a bound of the
/// entry's generics names `Self::A`. A type that holds `Self::A` is never leaked whole: without a
/// relative path (`Vec<Self::A>`) it is copied as written, else its parts are the entries. The
/// trait named (`<Self as ::leaker::m::Tr>::A`), and `Self::A` in a trait, are accepted.
#[test]
fn f06_impl_assoc_type_of_self_in_repeater() {
    let item: ItemImpl = parse_quote! {
        impl ::leaker::m::Tr for u8 {
            type A = u8;
            fn f(&self, x: Rel<Self::A>) {}
        }
    };
    let mut leaker = common::std_leaker();
    leaker.intern_impl(&item);
    let result = leaker.result();
    let err = result.errors();
    assert!(
        !err.is_empty(),
        "accepted, with the entries {:?}",
        result
            .clone()
            .recovered()
            .iter()
            .map(|(_, ty, _)| s(ty))
            .collect::<Vec<_>>()
    );
    // The headline says what to write, with the trait of the impl.
    assert_eq!(
        messages(err),
        "write `<Self as ::leaker::m::Tr>::A` instead of `Self::A` in `Rel<Self::A>`: the copy \
         cannot tell which trait `A` comes from"
    );
    assert_eq!(
        err[0].help_text(),
        Some("replace `Self::A` with `<Self as ::leaker::m::Tr>::A`")
    );

    // A bound of the generics of the entry that names it.
    let item: ItemImpl = parse_quote! {
        impl<T: ::p::Tr<Self::A>> ::leaker::m::Tr for W<T> {
            type A = u8;
            fn f(&self, x: Rel<T>) {}
        }
    };
    let mut leaker = common::std_leaker();
    leaker.intern_impl(&item);
    let result = leaker.result();
    let err = result.errors();
    let messages: Vec<&str> = err.iter().map(|item| item.message.as_str()).collect();
    assert_eq!(
        messages[..2],
        [
            "write `<Self as ::leaker::m::Tr>::A` instead of `Self::A` in a bound of the \
             generics of `W<T>`: the copy cannot tell which trait `A` comes from",
            "write `<Self as ::leaker::m::Tr>::A` instead of `Self::A` in a bound of the \
             generics of `Rel<T>`: the copy cannot tell which trait `A` comes from"
        ]
    );

    // Accepted: not in an entry, or the trait is named, or in a trait.
    let impls: Vec<ItemImpl> = vec![
        parse_quote! {
            impl ::leaker::m::Tr for u8 { type A = u8; fn f(&self, x: Vec<Self::A>) {} }
        },
        parse_quote! {
            impl ::leaker::m::Tr for u8 {
                type A = u8;
                fn f(&self, x: Rel<<Self as ::leaker::m::Tr>::A>) {}
            }
        },
    ];
    for item in impls {
        let mut leaker = common::std_leaker();
        leaker.intern_impl(&item);
        leaker.result().finish().unwrap();
    }
    // A type that holds `Self::A` is not leaked whole, however many entries it would cover: the
    // fewest entries are its parts (`Rel` and `Rel2`), not the function pointer.
    let item: ItemImpl = parse_quote! {
        impl ::leaker::m::Tr for u8 {
            type A = u8;
            fn f(&self, x: fn(Rel, Rel2) -> Self::A) {}
        }
    };
    let mut leaker = common::std_leaker();
    leaker.intern_impl(&item);
    let mut entries: Vec<String> = leaker
        .result()
        .finish()
        .unwrap()
        .iter()
        .map(|(_, ty, _)| s(ty))
        .collect();
    entries.sort();
    assert_eq!(entries, ["Rel", "Rel2"]);
    let item: ItemTrait = parse_quote! {
        pub trait Tr { type A; fn f(&self, x: Rel<Self::A>); }
    };
    let mut leaker = common::std_leaker();
    leaker.intern_trait(&item);
    assert_eq!(
        s(leaker.result().finish().unwrap().iter().next().unwrap().1),
        "Rel < Self :: A >"
    );
}

// ── F7 ───────────────────────────────────────────────────────────────────────

/// F7 (fixed): a method lifetime in a leaked path, and in the return type.
///
/// Trigger: `fn f<'x>(&self, a: Wrap<'x>) -> &'x u8` (or `-> Wrap<'x>`) in a trait, with
/// `Wrap<'x>` leaked. `'x` is late-bound in the trait, but the copy names it only in
/// `<Self as Repeater<(&'x (),), ..>>::Type`: E0195 for the impl of the trait.
///
/// Cause: a projection does not constrain a lifetime, so rustc makes `'x` early-bound in the copy.
///
/// Fix: an error with the E0195 note (the `Leaker` docs, "Method lifetimes"). Accepted: another
/// input keeps `'x` as written (`&'x u8`, `&'x self`, `Option<&'x str>`, a function pointer), a
/// bound makes it early-bound in both, no input or no output names it, or it is the trait's.
#[test]
fn f07_method_lifetime_in_leaked_path() {
    let rejected: Vec<ItemTrait> = vec![
        parse_quote!(
            pub trait Tr {
                fn f<'x>(&self, a: Wrap<'x>) -> &'x u8;
            }
        ),
        parse_quote!(
            pub trait Tr1 {
                fn f<'x>(&self, a: Wrap<'x>) -> Wrap<'x>;
            }
        ),
        // The path holds the lifetime in an argument, however deep.
        parse_quote!(
            pub trait Tr3 {
                fn f<'x>(&self, a: Rel<&'x u8>, b: Vec<Wrap<'x>>) -> &'x u8;
            }
        ),
        // Only the second lifetime is kept.
        parse_quote!(
            pub trait Tr9 {
                fn f<'x, 'y>(&self, a: Wrap<'x>, b: &'y u8) -> (&'x u8, &'y u8);
            }
        ),
    ];
    for item in rejected {
        let mut leaker = common::std_leaker();
        leaker.intern_trait(&item);
        let result = leaker.result();
        let err = result.errors();
        assert!(!err.is_empty(), "`{}` is accepted", item.ident);
        let note = first_note(&err[0]);
        assert!(
            note.contains("`'x`") && note.contains("(E0195)"),
            "`{}`: {}",
            item.ident,
            note
        );
    }
    let accepted: Vec<ItemTrait> = vec![
        parse_quote!(
            pub trait Tr2 {
                fn f<'x>(&self, a: &'x u8) -> Wrap<'x>;
            }
        ),
        // Another argument keeps it, `&'x self` too.
        parse_quote!(
            pub trait Tr4 {
                fn f<'x>(&self, a: &'x u8, b: Wrap<'x>) -> &'x u8;
                fn g<'x>(&'x self, a: Wrap<'x>) -> &'x u8;
            }
        ),
        // Kept as written in a type that is not leaked, however deep: a path of the allowed
        // names, a function pointer, a tuple, or the type that holds the path.
        parse_quote!(
            pub trait Tr8 {
                fn f<'x>(&self, a: Option<&'x str>, b: Wrap<'x>) -> &'x str;
                fn g<'x>(&self, a: for<'h> fn(&'h u8, &'x u8), b: Wrap<'x>) -> &'x u8;
                fn h<'x>(&self, a: Box<(&'x u8, Wrap<'x>)>) -> &'x u8;
                fn i<'x>(&self, a: &'x Wrap<'x>) -> &'x u8;
            }
        ),
        // A bound makes it early-bound in both.
        parse_quote!(
            pub trait Tr5 {
                fn f<'x, 'y: 'x>(&self, a: Wrap<'x>, b: &'y u8) -> &'x u8;
                fn g<'x>(&self, a: Wrap<'x>) -> &'x u8
                where
                    Self: 'x;
                fn h<'x, T: 'x>(&self, a: Wrap<'x>, t: T) -> &'x u8;
            }
        ),
        // No input mentions it, or the output does not: nothing to compare.
        parse_quote!(
            pub trait Tr6 {
                fn f<'x>(&self) -> Wrap<'x>;
                fn g<'x>(&self, a: Wrap<'x>);
            }
        ),
        // A lifetime of the trait is not a lifetime of the method.
        parse_quote!(
            pub trait Tr7<'a> {
                fn f(&self, a: Wrap<'a>) -> &'a u8;
            }
        ),
    ];
    for item in accepted {
        let mut leaker = common::std_leaker();
        leaker.intern_trait(&item);
        leaker.result().finish().unwrap();
    }
}

// ── F8 ───────────────────────────────────────────────────────────────────────

/// F8 (fixed): the bounds of a generic are merged and printed with an ambiguous `+`.
///
/// Trigger: `struct S<T: Clone>(Rel<T>) where T: Fn(()) -> &'static dyn Any`. The where predicate
/// is merged into the bounds of `T`, printed `T: Fn(()) -> &'static dyn Any + Clone`, where
/// `+ Clone` joins the trait object (rustc: "ambiguous `+` in a type", which has no code, and
/// E0225 for the non-auto `Clone`).
///
/// Cause: the return type of the `Fn` sugar needs parentheses when another bound follows it.
///
/// Fix: the canonical bounds of a context parenthesize such a return type when another bound
/// follows or it has several bounds itself (`-> (&'static dyn Any) + Clone`), whatever the order
/// and parentheses of the source. The test checks the generics of the Repeater's impl as printed
/// (syn parses what rustc rejects).
#[test]
fn f08_merged_bound_ambiguous_plus() {
    // The bounds of three shapes, sorted: `Fn() -> (..)` is parenthesized before another bound.
    let three =
        "< T : :: core :: ops :: Fn (()) -> (& 'static dyn :: core :: any :: Any) + :: core \
                 :: ops :: FnMut () -> (* const dyn :: core :: any :: Any) + :: core :: ops :: \
                 FnOnce () -> (fn () -> & 'static dyn :: core :: any :: Any) + Clone + Send >";
    // The item, the generics of the context of its entry, and those of the Repeater's impl.
    let items: Vec<(ItemStruct, &str, &str)> = vec![
        (
            parse_quote! {
                pub struct S<T: Clone>(pub Rel<T>)
                where
                    T: ::core::ops::Fn(()) -> &'static dyn ::core::any::Any;
            },
            "< T : :: core :: ops :: Fn (()) -> (& 'static dyn :: core :: any :: Any) + Clone >",
            "< T : :: core :: ops :: Fn (()) -> (& 'static dyn :: core :: any :: Any) + Clone >",
        ),
        // Several bounds of this kind, through a pointer and a function pointer.
        (
            parse_quote! {
                pub struct S<T: Clone + Send>(pub Rel<T>)
                where
                    T: ::core::ops::Fn(()) -> &'static dyn ::core::any::Any,
                    T: ::core::ops::FnMut() -> *const dyn ::core::any::Any,
                    T: ::core::ops::FnOnce() -> fn() -> &'static dyn ::core::any::Any;
            },
            three,
            three,
        ),
        // The same, with the return types parenthesized in the source: the same generics.
        (
            parse_quote! {
                pub struct S<T: Clone + Send>(pub Rel<T>)
                where
                    T: ::core::ops::Fn(()) -> (&'static dyn ::core::any::Any),
                    T: ::core::ops::FnMut() -> (*const dyn ::core::any::Any),
                    T: ::core::ops::FnOnce() -> (fn() -> &'static dyn ::core::any::Any);
            },
            three,
            three,
        ),
        // A return type with several bounds needs them even if the bound is the last.
        (
            parse_quote! {
                pub struct S<T: ::core::ops::Fn(()) -> (dyn ::core::any::Any + Send)>(pub Rel<T>);
            },
            "< T : :: core :: ops :: Fn (()) -> (dyn :: core :: any :: Any + Send) >",
            "< T : :: core :: ops :: Fn (()) -> (dyn :: core :: any :: Any + Send) >",
        ),
        // A bound that the context gets from a lifetime that the entry names: the last one.
        (
            parse_quote! {
                pub struct S<'a, T: ::core::ops::Fn(()) -> &'static dyn ::core::any::Any>(pub Rel<&'a T>);
            },
            "< 'a , T : 'a + :: core :: ops :: Fn (()) -> & 'static dyn :: core :: any :: Any >",
            "< 'a , T : :: core :: ops :: Fn (()) -> & 'static dyn :: core :: any :: Any >",
        ),
    ];
    for (item, context, impl_generics_text) in &items {
        let mut leaker = common::std_leaker();
        leaker.intern_struct(item);
        let result = leaker.result();
        let referrer = result.clone().finish().unwrap();
        assert_eq!(referrer.iter().count(), 1, "`{}`", s(item));
        assert_eq!(context_text(&referrer, 0), *context, "`{}`", s(item));
        assert_eq!(
            s(&impl_generics(&result, 0)),
            *impl_generics_text,
            "`{}`",
            s(item)
        );
    }
}

// ── F9 ───────────────────────────────────────────────────────────────────────
/// F9 (fixed): `Self` replaced by a reference to a function pointer is printed without parentheses.
///
/// Trigger: a trait method with `where Self: 'y`, copied for the host `&'static unsafe fn(i32,
/// R0)` (`with_self()`). The copy has `&'static unsafe fn(i32, R0): 'y`, which rustc does not
/// parse ("return types are denoted using `->`").
///
/// Cause: after a function pointer with no output, rustc reads the `:` as its return type.
///
/// Fix: the replacement of `Self` parenthesizes a bounded type that needs it (`(&fn(..)): 'y`), in
/// the Visitor, in the generics of an entry and where the Leaker rewrites a bound for another
/// `Self`. It does the same for a pointee with several bounds (`&(dyn Any + Send)`).
#[test]
fn f09_fn_pointer_self_predicate_print() {
    let item: ItemTrait = parse_quote! {
        pub trait Tr {
            fn m<'y>(&self, a: u8) where Self: 'y;
            fn n(&self, a: u8) where Self: Clone;
            fn r(&self) -> Rel<Self> where Self: Clone;
        }
    };
    let mut leaker = common::std_leaker();
    let context = leaker.intern_trait(&item);
    let referrer = leaker.result().finish().unwrap();
    // The self type, and the bounded type of the where predicates of the copy.
    let self_tys: Vec<(Type, &str)> = vec![
        (
            parse_quote!(&'static unsafe fn(i32, R0)),
            "(& 'static unsafe fn (i32 , R0))",
        ),
        (parse_quote!(*const fn(i32)), "(* const fn (i32))"),
        (
            parse_quote!(fn(i32) -> &'static fn(u8)),
            "(fn (i32) -> & 'static fn (u8))",
        ),
        (parse_quote!(fn(i32)), "fn (i32)"),
    ];
    for (self_ty, bounded) in self_tys {
        let mut copy = Item::Trait(item.clone());
        common::visitor(&referrer, context, common::placeholder)
            .with_self(self_ty.clone())
            .visit_item_mut(&mut copy);
        assert_eq!(
            s(&copy),
            format!(
                "pub trait Tr {{ fn m < 'y > (& self , a : u8) where {b} : 'y ; fn n (& self , a : \
                 u8) where {b} : Clone ; fn r (& self) -> P < 0usize > where {b} : Clone ; }}",
                b = bounded
            ),
        );
    }

    // The pointee of a reference and of a pointer.
    let item: ItemTrait = parse_quote! {
        pub trait Tr2 {
            fn p(&self, a: &Self, b: *const Self, c: Box<Self>) where &'static Self: Clone;
        }
    };
    let mut leaker = common::std_leaker();
    let context = leaker.intern_trait(&item);
    let referrer = leaker.result().finish().unwrap();
    let self_ty: Type = parse_quote!(dyn ::core::any::Any + Send + 'static);
    let mut copy = Item::Trait(item.clone());
    common::visitor(&referrer, context, common::placeholder)
        .with_self(self_ty)
        .visit_item_mut(&mut copy);
    assert_eq!(
        s(&copy),
        "pub trait Tr2 { fn p (& self , a : & (dyn :: core :: any :: Any + Send + 'static) , b : \
         * const (dyn :: core :: any :: Any + Send + 'static) , c : Box < dyn :: core :: any :: \
         Any + Send + 'static >) where & 'static (dyn :: core :: any :: Any + Send + 'static) : \
         Clone ; }"
    );
}

// ── F11 ──────────────────────────────────────────────────────────────────────

/// F11 (limitation): a copy of an impl whose self type is an entry.
///
/// Trigger: `impl<T> ::p::Tr for Imp<T> {}` with `Imp` a relative path. The copy is `impl<T>
/// ::p::Tr for <() as Repeater<(T,), ..>>::Type {}`: E0207 "the type parameter `T` is not
/// constrained by the impl trait, self type, or predicates" (E0118 or E0116 for an inherent impl).
///
/// Cause: rustc does not count a projection as constraining `T`, and the replacement of an entry
/// is never a type that rustc sees through.
///
/// Limitation: a macro should check such an impl, not copy it. The test pins what the library
/// does: the entry is the whole self type, with `T` as its argument.
#[test]
fn f11_copied_impl_param_constrained_only_by_leaked_type() {
    let item: ItemImpl = parse_quote! {
        impl<T> ::p::Tr for Imp<T> {}
    };
    let mut leaker = common::std_leaker();
    let context = leaker.intern_impl(&item);
    let referrer = leaker.result().finish().unwrap();
    let ix = entry_index(&referrer, "Imp < T >");
    assert_eq!(context_text(&referrer, ix), "< T >");
    let args = common::generic_args(&common::entry(&referrer, ix).generics, common::identity)
        .unwrap()
        .iter()
        .map(s)
        .collect::<Vec<_>>();
    assert_eq!(args, ["T"]);
    assert_eq!(
        rewrite(&referrer, context, Item::Impl(item)),
        s(&quote! {
            impl<T> ::p::Tr for P<0usize> {}
        })
    );
}

// ── F12 ──────────────────────────────────────────────────────────────────────

/// The lifetimes of the inputs of a signature as rustc's elision counts them: each elided one is
/// its own, a named one counts once. A function pointer or `Fn(..)` inside has its own binder.
#[derive(Default)]
struct Lifetimes {
    elided: usize,
    named: Vec<String>,
}

impl Lifetimes {
    fn count(&self) -> usize {
        self.elided + self.named.len()
    }
}

impl<'ast> Visit<'ast> for Lifetimes {
    fn visit_type_reference(&mut self, i: &'ast TypeReference) {
        if i.lifetime.is_none() {
            self.elided += 1;
        }
        syn::visit::visit_type_reference(self, i)
    }
    fn visit_lifetime(&mut self, i: &'ast Lifetime) {
        if i.ident == "_" {
            self.elided += 1;
        } else if !self.named.contains(&i.to_string()) {
            self.named.push(i.to_string());
        }
    }
    fn visit_type_bare_fn(&mut self, _: &'ast TypeBareFn) {}
    fn visit_parenthesized_generic_arguments(&mut self, _: &'ast ParenthesizedGenericArguments) {}
}

/// `true` if the output has an elided lifetime (a nested signature has its own).
#[derive(Default)]
struct ElidedOutput(bool);

impl<'ast> Visit<'ast> for ElidedOutput {
    fn visit_type_reference(&mut self, i: &'ast TypeReference) {
        self.0 |= i.lifetime.is_none();
        syn::visit::visit_type_reference(self, i)
    }
    fn visit_lifetime(&mut self, i: &'ast Lifetime) {
        self.0 |= i.ident == "_";
    }
    fn visit_type_bare_fn(&mut self, _: &'ast TypeBareFn) {}
    fn visit_parenthesized_generic_arguments(&mut self, _: &'ast ParenthesizedGenericArguments) {}
}

/// The signatures that are left in a copy: the sugar of `Fn(..) -> ..` and the function pointers,
/// and those whose elided output lifetime elision cannot resolve (it needs exactly one lifetime in
/// the inputs).
#[derive(Default)]
struct Signatures {
    /// How many there are.
    count: usize,
    /// Those that elision cannot resolve.
    broken: Vec<String>,
}

impl Signatures {
    fn check<'a>(&mut self, inputs: impl IntoIterator<Item = &'a Type>, output: &ReturnType) {
        self.count += 1;
        let mut elided = ElidedOutput::default();
        elided.visit_return_type(output);
        let mut lifetimes = Lifetimes::default();
        for input in inputs {
            lifetimes.visit_type(input);
        }
        if elided.0 && lifetimes.count() != 1 {
            self.broken.push(format!(
                "{} (the inputs have {} lifetimes)",
                s(output),
                lifetimes.count()
            ));
        }
    }
}

/// The signatures in `item`.
fn signatures_of(item: &Item) -> Signatures {
    let mut signatures = Signatures::default();
    signatures.visit_item(item);
    signatures
}

/// The signatures in `item`.
fn signatures_of_trait(item: &ItemTrait) -> Signatures {
    let mut signatures = Signatures::default();
    signatures.visit_item_trait(item);
    signatures
}

impl<'ast> Visit<'ast> for Signatures {
    fn visit_parenthesized_generic_arguments(&mut self, i: &'ast ParenthesizedGenericArguments) {
        self.check(&i.inputs, &i.output);
        syn::visit::visit_parenthesized_generic_arguments(self, i)
    }
    fn visit_type_bare_fn(&mut self, i: &'ast TypeBareFn) {
        self.check(i.inputs.iter().map(|arg| &arg.ty), &i.output);
        syn::visit::visit_type_bare_fn(self, i)
    }
}

/// The entries of `item` and its copy, made as a macro does: each entry is replaced by the
/// projection of the Repeater, with the arguments of the entry encoded by the const slots.
fn f12_copy(item: &ItemTrait) -> std::result::Result<(Vec<String>, ItemTrait), LeakerResultError> {
    f12_copy_with(common::std_leaker(), item)
}

/// Like `f12_copy()`, with the Leaker `leaker`.
fn f12_copy_with<A: Allow>(
    mut leaker: Leaker<A>,
    item: &ItemTrait,
) -> std::result::Result<(Vec<String>, ItemTrait), LeakerResultError> {
    let context = leaker.intern_trait(item);
    let referrer = leaker.result().finish()?;
    let mut copy = item.clone();
    let repeater = common::repeater();
    common::visitor(&referrer, context, |_, ix, entry| {
        let self_ty: Type = parse_quote!(());
        common::projection(repeater.clone(), 0)
            .of(&self_ty, entry, ix)
            .unwrap()
    })
    .visit_item_trait_mut(&mut copy);
    Ok((common::type_texts(&referrer), copy))
}

/// F12 (fixed): an elided output lifetime in the `Fn` sugar or a function pointer, next to an entry
/// that has lifetimes.
///
/// Trigger: `Box<dyn Fn(&Rel<T>) -> &u8>` in `trait Tr<'a, T: 'a>`. The entry `Rel<T>` has the
/// generics `<'a, T: 'a>`, so its replacement names lifetimes (`<() as Repeater<(&'a (), T, &'a
/// T), ..>>::Type`): E0106 "missing lifetime specifier". `Tr1<T>`, without `'a`, works.
///
/// Cause: elision needs exactly one lifetime in the inputs, and the replacement adds more.
///
/// Fix: the whole holder is the entry (`type Type = dyn Fn(&Rel<T>) -> &u8;`), which keeps the
/// elision as written. Where nothing can hold it (a bound, a method without a reference receiver,
/// a trait object under an elided reference) it is an error with the E0106 note.
#[test]
fn f12_elided_lifetime_in_fn_sugar_with_generics() {
    // Where the entry has lifetimes, the holder is the entry, whatever it is.
    for (item, holder) in [
        (
            parse_quote! {
                pub trait Tr<'a, T: 'a> { fn f(&self, x: Box<dyn Fn(&Rel<T>) -> &u8>); }
            },
            "dyn Fn (& Rel < T >) -> & u8",
        ),
        (
            parse_quote! {
                pub trait Tr<'a, T: 'a> { fn f(&self, x: fn(&Rel<T>) -> &u8); }
            },
            "fn (& Rel < T >) -> & u8",
        ),
        (
            parse_quote! {
                pub trait Tr<'a, T: 'a> { fn f(&self, x: &'a dyn Fn(&Rel<T>) -> &u8); }
            },
            "& 'a dyn Fn (& Rel < T >) -> & u8",
        ),
    ] {
        let item: ItemTrait = item;
        let (entries, copy) = f12_copy(&item).unwrap();
        assert_eq!(entries, [holder], "{}", s(&item));
        let signatures = signatures_of_trait(&copy);
        assert_eq!(signatures.count, 0, "nothing of it is left: {}", s(&copy));
    }
    // The control: no lifetime in the arguments of the entry, so the sugar is left as written,
    // with the elided lifetime of its input as the only one.
    for item in [
        parse_quote! {
            pub trait Tr1<T> { fn f(&self, x: Box<dyn Fn(&Rel<T>) -> &u8>); }
        },
        parse_quote! {
            pub trait Tr1<T> { fn f(&self, x: fn(&Rel<T>) -> &u8); }
        },
    ] {
        let item: ItemTrait = item;
        let (entries, copy) = f12_copy(&item).unwrap();
        assert_eq!(entries, ["Rel < T >"], "{}", s(&item));
        let signatures = signatures_of_trait(&copy);
        assert_eq!(signatures.count, 1, "{}", s(&copy));
        assert!(signatures.broken.is_empty(), "{:?}", signatures.broken);
    }
    // Where the elision is not at stake, the entry is left alone: the output names its lifetime,
    // has none, or a method takes it from its receiver.
    for item in [
        parse_quote! {
            pub trait Tr<'a, T: 'a> { fn f(&self, x: Box<dyn Fn(&Rel<T>) -> Rel<T>>); }
        },
        parse_quote! {
            pub trait Tr<'a, T: 'a> {
                fn f(&self, x: Box<dyn for<'x> Fn(&'x Rel<T>) -> &'x u8>);
            }
        },
        parse_quote! {
            pub trait Tr<'a, T: 'a> { fn f(&self, x: Box<dyn Fn(&u8) -> &Rel<T>>); }
        },
        parse_quote! {
            pub trait Tr<'a, T: 'a> { fn f(&self, x: &Rel<T>) -> &u8; }
        },
    ] {
        let item: ItemTrait = item;
        let (entries, copy) = f12_copy(&item).unwrap();
        assert_eq!(entries, ["Rel < T >"], "{}", s(&item));
        let signatures = signatures_of_trait(&copy);
        assert!(signatures.broken.is_empty(), "{:?}", signatures.broken);
    }
    // Where nothing can hold the signature (a bound, a method with no lifetime of its receiver),
    // the item is rejected, not copied wrongly.
    for item in [
        parse_quote! { pub trait Tr<'a, T: 'a> { fn f(x: &Rel<T>) -> &u8; } },
        parse_quote! { pub trait Tr<'a, T: 'a> { fn f(self: Box<Self>, x: &Rel<T>) -> &u8; } },
        parse_quote! { pub trait Tr<'a, T: 'a> { fn f<F: Fn(&Rel<T>) -> &u8>(&self, f: F); } },
        parse_quote! { pub trait Tr<'a, T: 'a, F: Fn(&Rel<T>) -> &u8> {} },
        // Also where the holder cannot be leaked: a trait object under an elided reference.
        parse_quote! { pub trait Tr<'a, T: 'a> { fn f(&self, x: &dyn Fn(&Rel<T>) -> &u8); } },
    ] {
        let item: ItemTrait = item;
        let err = f12_copy(&item)
            .err()
            .unwrap_or_else(|| panic!("{} is accepted", s(&item)));
        let note = first_note(&err[0]);
        assert!(note.contains("E0106"), "{}: {}", s(&item), note);
    }
}

// ── F13 ──────────────────────────────────────────────────────────────────────

/// Whether `generics` name `Self`: in a bound or a where predicate.
fn generics_name_self(generics: &Generics) -> bool {
    struct Find(bool);
    impl<'ast> Visit<'ast> for Find {
        fn visit_path(&mut self, i: &'ast Path) {
            self.0 |= i.leading_colon.is_none()
                && i.segments.first().map_or(false, |s| s.ident == "Self");
            syn::visit::visit_path(self, i)
        }
    }
    let mut find = Find(false);
    find.visit_generics(generics);
    find.0
}

/// Whether `ty` is `needle`, or a type inside it.
fn type_contains(ty: &Type, needle: &str) -> bool {
    struct Find<'a>(&'a str, bool);
    impl<'ast, 'a> Visit<'ast> for Find<'a> {
        fn visit_type(&mut self, i: &'ast Type) {
            self.1 |= s(i) == self.0;
            syn::visit::visit_type(self, i)
        }
    }
    let mut find = Find(needle, false);
    find.visit_type(ty);
    find.1
}

/// The entries that the Referrer cannot replace: the entry is the self type of its own context, or
/// a type in it, and the generics of that context name `Self`. What the copy writes for `Self` is
/// the self type with the entry replaced, `<Self as Repeater<..>>::Type` where `Self` is that same
/// projection.
fn self_referential_entries(referrer: &Referrer) -> Vec<String> {
    common::types(referrer)
        .enumerate()
        .filter(|(ix, ty)| {
            let own_self = matches!(
                common::entry(referrer, *ix).self_ty.as_ref(),
                Some(SelfRequirement::Type(self_ty)) if type_contains(self_ty, &s(*ty))
            );
            own_self && generics_name_self(&common::entry(referrer, *ix).generics)
        })
        .map(|(_, ty)| s(ty))
        .collect()
}

/// F13 (fixed): a bound of the generics of an impl names `Self`, and the self type is an entry.
///
/// Trigger: `impl<T> S<T> where T: AsRef<Option<Self>> { fn f(&self, a: Rel) {} }` (also inline,
/// in an impl of a trait, or a type in the self type: `impl<T: AsRef<Self>> ::p::W<Rel<T>>`). The
/// entry `S<T>` has `Self` = `S<T>` in its context, and the copy replaces `S<T>`: the entry is its
/// own `Self`, so no type can be written for `Self`.
///
/// Cause: rewriting the bound for the self type names the entry in its own bound (E0277), the type
/// for `Self` in the copy is a cycle (E0391), and the self type has no parts to leak instead.
///
/// Fix: the Leaker rejects it ("is in the self type of the impl, and a bound of its generics names
/// `Self`"), also in a nested impl. Accepted: an optional self type (`::ext::S<T>`, then copied as
/// written), a self type that is not an entry, bounds without `Self`, and the bound on a generic
/// that no entry of the self type uses (`impl<T: AsRef<Self>, U> ::p::W<T, Rel<U>>`): rustc
/// normalizes `Self` with the impls of those entries only (F19).
#[test]
fn f13_impl_bound_names_self_of_leaked_self_type() {
    let rejected: Vec<ItemImpl> = vec![
        parse_quote!(
            impl<T> S<T>
            where
                T: AsRef<Option<Self>>,
            {
                fn f(&self, a: Rel) {}
            }
        ),
        parse_quote!(
            impl<T: AsRef<[Self]>, const N: usize> S<T, N> {
                fn f(&self, a: Rel) {}
            }
        ),
        parse_quote!(
            impl<T> ::p::Tr for S<T>
            where
                T: AsRef<Option<Self>>,
            {
                fn f(&self, a: Rel) {}
            }
        ),
        // A type in the self type.
        parse_quote!(
            impl<T: AsRef<Self>> ::p::W<Rel<T>> {
                fn f(&self, a: Rel) {}
            }
        ),
        parse_quote!(
            impl<'a, T: 'a> ::p::Tr<Rel2> for ::p::W<&'a Rel<T>> where T: AsRef<Self> {}
        ),
        // A written `Self: 'a`, which is encoded as `&'a Self`.
        parse_quote!(
            impl<'a> S<'a> where Self: 'a {}
        ),
    ];
    let copied: Vec<ItemImpl> = vec![
        // An absolute path of a crate that the Leaker does not allow: an optional entry.
        parse_quote!(
            impl<T> ::ext::S<T> where T: AsRef<Option<Self>> {}
        ),
        parse_quote!(
            impl<T: AsRef<Self>> ::p::Tr for ::ext::W<T> {}
        ),
        // The type of the self type that depends on `Self` is optional, and the other is not.
        parse_quote!(
            impl<T: AsRef<Self>, U> ::p::Tr<Rel<U>> for ::p::W<::ext::S<T>, ::ext::V<U>> {}
        ),
        // The parts are the optional entries, not the tuple that holds them.
        parse_quote!(
            impl<T: AsRef<Self>> ::p::Tr for ::p::W<(::ext::A<T>, ::ext::B<T>, T)> {}
        ),
        // The same under a reference, which the optional entry under it cannot stand for.
        parse_quote!(
            impl<'a, T: AsRef<Self>, U> ::p::Tr for ::p::W<&'a (::ext::S<T>, ::ext::V<U>)> {}
        ),
    ];
    let controls: Vec<ItemImpl> = vec![
        // The self type is not leaked: it is copied as written.
        parse_quote!(
            impl<T> ::p::S<T>
            where
                T: AsRef<Option<Self>>,
            {
                fn f(&self, a: Rel) {}
            }
        ),
        // The bounds do not name `Self`.
        parse_quote!(
            impl<T> S<T>
            where
                T: AsRef<Option<Rel2>>,
            {
                fn f(&self, a: Rel) {}
            }
        ),
        // `Self` is in a signature, not in a bound.
        parse_quote!(
            impl<T: Clone> S<T> {
                fn f(&self, a: Option<Self>) {}
            }
        ),
        // The bound is on a generic that the entries of the self type do not use.
        parse_quote!(
            impl<T: AsRef<Self>, U> ::p::W<T, Rel<U>> {
                fn f(&self, a: Rel) {}
            }
        ),
        // The same as a where predicate, in an impl of a trait, with the bound inline or not.
        parse_quote!(
            impl<T, U> ::p::W<T, Rel<U>>
            where
                T: AsRef<Self>,
            {
                fn f(&self, a: Rel) {}
            }
        ),
        parse_quote!(
            impl<T: AsRef<Self>, U> ::p::Tr for ::p::W<T, Rel<U>> {}
        ),
        parse_quote!(
            impl<T, U> ::p::Tr for ::p::W<T, Rel<U>> where T: AsRef<Self> {}
        ),
        // Two generics that no entry uses have the bound, and the one that an entry uses has a
        // bound that is stable.
        parse_quote!(
            impl<T: AsRef<Self>, U: Clone, V: AsRef<Self>> ::p::W<T, Rel<U>, V> {}
        ),
        // A generic with a bound that names a leaked type, which no entry in the self type uses.
        parse_quote!(
            impl<T: Fn(Rel2<u8>), U, V: AsRef<Self>> ::p::W<T, Rel<U>, V> {
                fn f(&self, a: Rel) {}
            }
        ),
        // `Self` in a bound of a method, and in a where predicate on the type itself.
        parse_quote!(
            impl<T, U> ::p::W<T, Rel<U>> where Self: Sized {}
        ),
        parse_quote!(
            impl<T, U> ::p::W<T, Rel<U>> {
                fn f<X: AsRef<Self>>(&self, x: X, a: Rel) {}
            }
        ),
        // An entry in the trait's arguments or in a method that depends on `Self` (a position, not
        // a predicate): the Repeater's impl is for the self type, which the copy writes.
        parse_quote!(
            impl<T: AsRef<Self>, U> ::p::Tr<Rel2<T>> for ::p::W<T, Rel<U>> {}
        ),
        parse_quote!(
            impl<T: AsRef<Self>, U> ::p::W<T, Rel<U>> {
                fn f(&self, a: Rel2<T>) {}
            }
        ),
        // An entry in a predicate that does not depend on `Self`.
        parse_quote!(
            impl<T: AsRef<Self>, U> ::p::Tr for ::p::W<T, Rel<U>> where Rel2<U>: Default {}
        ),
        // The parts of the self type are the entries, not the tuple that holds them: it would be
        // cheaper, but depends on `Self`.
        parse_quote!(
            impl<T: AsRef<Self>> ::p::Tr for ::p::W<(Rel2, Rel3, T)> {}
        ),
        // `Self` is only in a method, and in the trait's arguments, which the copy writes with the
        // self type.
        parse_quote!(
            impl<T> ::p::Tr<Rel<T>> for ::p::W<T> {
                fn f(&self)
                where
                    T: AsRef<Self>,
                {
                }
            }
        ),
    ];
    for item in &rejected {
        let mut leaker = common::p_leaker();
        leaker.intern_impl(item);
        let result = leaker.result();
        let err = result.errors();
        assert!(
            !err.is_empty(),
            "`{}` is accepted, and the entries {:?} are their own `Self`: the Referrer has no \
             type to write for it",
            s(item),
            self_referential_entries(&result.clone().recovered())
        );
        let headline = "is in the self type of the impl, and a bound of its generics names `Self`";
        let (shown, note) = (messages(err), first_note(&err[0]));
        assert!(shown.contains(headline), "{}: {}", s(item), shown);
        assert!(
            note.contains("no type can be written for `Self`"),
            "{}: {}",
            s(item),
            note
        );
        // The recovered Referrer has the rest of the item, and the entries that the error is about
        // are not among them.
        let referrer = result.recovered();
        assert!(
            self_referential_entries(&referrer).is_empty(),
            "{}: {:?}",
            s(item),
            self_referential_entries(&referrer)
        );
        // Also as a nested item.
        let mut leaker = common::p_leaker();
        leaker.enter_generics(&Generics::default(), |view| {
            view.visit_item(&Item::Impl(item.clone()))
        });
        assert!(
            messages(&errors_of(leaker)).contains(headline),
            "{}",
            s(item)
        );
    }
    // A reference that holds a type that must be an entry is one itself (its parts cannot stand
    // for it), if it can name the generics: those of the Leaker, not the local ones of a nested
    // item, where only the parts are entries.
    let item: ItemImpl = parse_quote!(
        impl<'a, T: AsRef<Self>> ::p::Tr for ::p::W<&'a (Rel2, T)> {}
    );
    let mut leaker = common::p_leaker();
    leaker.intern_impl(&item);
    assert!(
        messages(&errors_of(leaker)).contains("`&'a (Rel2, T)` is in the self type of the impl")
    );
    for item in copied.iter().chain(&controls) {
        let mut leaker = common::p_leaker();
        let context = leaker.intern_impl(item);
        let result = leaker.result();
        let referrer = assert_accepted(item, result.clone().finish());
        assert!(
            self_referential_entries(&referrer).is_empty(),
            "{}",
            s(item)
        );
        // The copy replaces the optional entries, but not those in the self type.
        let mut copy = Item::Impl(item.clone());
        let referrer = result.finish().unwrap();
        common::visitor(&referrer, context, common::placeholder).visit_item_mut(&mut copy);
        if let Item::Impl(copy) = copy {
            if copied.contains(item) {
                let self_ty = s(&copy.self_ty);
                assert!(self_ty.contains(":: ext :: "), "{}: {}", s(item), self_ty);
            }
        }
    }
    // The optional entry is one where the bound does not name `Self`: the test is not empty.
    let item: ItemImpl = parse_quote!(
        impl<T: Clone> ::ext::S<T> {}
    );
    let mut leaker = common::p_leaker();
    let context = leaker.intern_impl(&item);
    let result = leaker.result();
    let referrer = result.clone().finish().unwrap();
    assert_eq!(common::type_texts(&referrer), [":: ext :: S < T >"]);
    let mut copy = Item::Impl(item);
    let referrer = result.finish().unwrap();
    common::visitor(&referrer, context, common::placeholder).visit_item_mut(&mut copy);
    let copy = match copy {
        Item::Impl(copy) => copy,
        _ => unreachable!(),
    };
    assert_eq!(s(&copy.self_ty), "P < 0usize >");
}

// ── F14 ──────────────────────────────────────────────────────────────────────

/// The copy of `item` that a macro makes (the entries replaced by `P<ix>`), if the Leaker accepts
/// `item`.
fn f14_copy(item: &ItemImpl) -> std::result::Result<ItemImpl, LeakerResultError> {
    let mut leaker = common::p_leaker();
    let context = leaker.intern_impl(item);
    let referrer = leaker.result().finish()?;
    match copied(&referrer, context, Item::Impl(item.clone())) {
        Item::Impl(copy) => Ok(copy),
        _ => unreachable!(),
    }
}

/// How an item of F14 ends: accepted, with its copy, or rejected, for the type and the bound that
/// the error names.
type Outcome<'a> = std::result::Result<&'a str, (&'a str, &'a str)>;

/// F14 (fixed): a where predicate of a method names `Self::A`, in an impl of a trait whose own
/// bound names a leaked type.
///
/// Trigger: `impl<T> ::p::Tr<T> for u8 where T: Fn(Rel) { type A1 = u8; fn m<X>(&self, a: Rel)
/// where X: Into<Self::A1> {} }` (also the bound inline, or the predicate on `u8`). The copy has
/// `T: Fn(<() as Repeater<..>>::Type)`. rustc normalizes `Self::A1` while it builds the environment
/// of `m`, with the impl of `Tr<T>` for `u8`, which needs `T: Fn(Rel)`: E0277 "expected an
/// `Fn(Rel)` closure, found `T`".
///
/// Cause: it is F2, but the contexts of `Rel` and `Self::A1` have no generics, so the check of F2
/// did not see the bounds of the impl that rustc proves.
///
/// Fix: in an impl, the bounds of the impl count for `Self::A` in a predicate. Accepted: a bound
/// that names no leaked type, `Self::A1` outside the predicates, and a bound with a type of another
/// crate (`T: Fn(::ext::Rel)`), which is no entry there and is written as it is.
#[test]
fn f14_impl_predicate_names_self_projection_with_leaked_bound() {
    let items: Vec<(ItemImpl, Outcome)> = vec![
        (
            parse_quote!(
                impl<T> ::p::Tr<T> for u8
                where
                    T: Fn(Rel),
                {
                    type A1 = u8;
                    fn m<X>(&self, a: Rel)
                    where
                        X: Into<Self::A1>,
                    {
                    }
                }
            ),
            Err(("Self :: A1", "T : Fn (Rel)")),
        ),
        // The bound written inline, the predicate on a type that is not a generic.
        (
            parse_quote!(
                impl<T: Fn(Rel)> ::p::Tr<T> for u8 {
                    type A1 = u8;
                    fn m(&self, a: Rel)
                    where
                        u8: Into<Self::A1>,
                    {
                    }
                }
            ),
            Err(("Self :: A1", "T : Fn (Rel)")),
        ),
        // The leaked type of the bound is a type of another crate, an entry that the copy replaces,
        // and which the Leaker accepts all the same: the copy is checked.
        (
            parse_quote!(
                impl<T> ::p::Tr<T> for u8
                where
                    T: Fn(::ext::Rel),
                {
                    type A1 = u8;
                    fn m<X>(&self, a: u8)
                    where
                        X: Into<Self::A1>,
                    {
                    }
                }
            ),
            Ok("impl < T > :: p :: Tr < T > for u8 where T : Fn (:: ext :: Rel) , { type A1 = u8 ; fn m \
                 < X > (& self , a : u8) where X : Into < Self :: A1 > , { } }"),
        ),
        // The controls. The bound names no leaked type.
        (
            parse_quote!(
                impl<T: Clone> ::p::Tr<T> for u8 {
                    type A1 = u8;
                    fn m<X>(&self, a: Rel)
                    where
                        X: Into<Self::A1>,
                    {
                    }
                }
            ),
            Ok("impl < T : Clone > :: p :: Tr < T > for u8 { type A1 = u8 ; fn m < X > (& self , a : P < \
                 0usize >) where X : Into < Self :: A1 > , { } }"),
        ),
        // `Self::A1` is a type of the signature, which rustc normalizes once the environment is built.
        (
            parse_quote!(
                impl<T> ::p::Tr<T> for u8
                where
                    T: Fn(Rel),
                {
                    type A1 = u8;
                    fn m<X>(&self, a: Self::A1, b: Rel)
                    where
                        X: Into<Rel>,
                    {
                    }
                }
            ),
            Ok("impl < T > :: p :: Tr < T > for u8 where T : Fn (P < 0usize >) , { type A1 = u8 ; fn m < X \
                 > (& self , a : Self :: A1 , b : P < 0usize >) where X : Into < P < 0usize > > , \
                 { } }"),
        ),
    ];
    for (item, outcome) in &items {
        // Each is rejected, or accepted with its copy.
        match outcome {
            Err((ty, bound)) => {
                assert_unstable(item, f14_copy(item), ty, bound, Blames::LeakedInImpl);
            }
            Ok(copy) => assert_eq!(s(&assert_accepted(item, f14_copy(item))), *copy),
        }
    }
}

// ── F15 ──────────────────────────────────────────────────────────────────────

/// The copy of a struct, an enum or a union that a macro makes with `Referrer::visitor(..)
/// .with_self(self_ty)`, where `self_ty` is the type as the Leaker's crate names it
/// (`::p::S<'a, T>`), and the entries of the item. Each entry is replaced by `P<ix>`.
fn f15_copy(item: &Item) -> std::result::Result<(Vec<String>, Item), LeakerResultError> {
    let (ident, generics) = match item {
        Item::Struct(i) => (&i.ident, &i.generics),
        Item::Enum(i) => (&i.ident, &i.generics),
        Item::Union(i) => (&i.ident, &i.generics),
        _ => panic!("unsupported item"),
    };
    let mut leaker = common::std_leaker();
    let context = common::intern_item(&mut leaker, item);
    let referrer = leaker.result().finish()?;
    let args: Vec<GenericArgument> = generics.params.iter().map(common::own_name_arg).collect();
    let self_ty: Type = parse_quote!(::p::#ident<#(#args),*>);
    let mut copy = item.clone();
    common::visitor(&referrer, context, common::placeholder)
        .with_self(self_ty)
        .visit_item_mut(&mut copy);
    Ok((common::type_texts(&referrer), copy))
}

/// F15 (fixed): `with_self(ty)` writes `Self` into a signature with an elided output lifetime, and
/// `ty` has lifetimes.
///
/// Trigger: `struct S<'a> { f: fn(&Self) -> &u8, g: &'a u8 }` copied with
/// `with_self(::p::S<'a>)`. The copy `fn(&::p::S<'a>) -> &u8` has two lifetimes in its input:
/// E0106 "missing lifetime specifier". The same for `Fn(&Self) -> &u8` and `&Self::A`.
///
/// Cause: rustc counts no lifetime for `Self`, and the Referrer replaced it without looking at the
/// signature.
///
/// Fix: where the inputs show their one lifetime, the Referrer names it with a fresh `for<'e>`
/// binder (F23). Where a path hides it (`fn(::core::slice::Iter<u8>, Self) -> &u8`, F28), the
/// signature is the entry, or an error with the E0106 note in a bound. An entry in the inputs
/// makes the signature the entry, as in F12. The copy of an impl or a trait keeps `Self`.
#[test]
fn f15_self_rewritten_in_elided_signature() {
    // The inputs show their one lifetime: the Referrer names it (F23), so `Self` leaks nothing.
    // The copy has the lifetime in a `for<..>` binder.
    for item in [
        parse_quote!(
            pub struct S<'a> {
                f: fn(&Self) -> &u8,
                g: &'a u8,
            }
        ),
        parse_quote!(
            pub struct S<'a>(Box<dyn Fn(&Self) -> &u8>, &'a u8);
        ),
        parse_quote!(
            pub enum E<'a, T> {
                A(fn(&(T, Self)) -> &u8),
                B(&'a T),
            }
        ),
        parse_quote!(
            pub union U<'a> {
                f: fn(&Self) -> &u8,
                g: &'a u8,
            }
        ),
        // `Self` deep in the inputs, next to a type that is allowed.
        parse_quote!(
            pub struct S<'a>(Box<dyn Fn(&Vec<Self>) -> &u8>, &'a u8);
        ),
        // The reference takes its lifetime from the trait object.
        parse_quote!(
            pub struct S<'a>(&'a dyn Fn(&Self) -> &u8, &'a u8);
        ),
        // `Self::A`, which the visitor writes `<::p::S<'a>>::A`, has no lifetime of its own either.
        parse_quote!(
            pub struct S<'a>(Box<dyn Fn(&Self::A) -> &u8>, &'a u8);
        ),
    ] {
        let item: Item = item;
        let (entries, copy) = assert_accepted_as(&format!("`{}`", s(&item)), f15_copy(&item));
        assert!(entries.is_empty(), "{}: {:?}", s(&item), entries);
        let signatures = signatures_of(&copy);
        assert_eq!(signatures.count, 1, "{}", s(&copy));
        assert!(
            signatures.broken.is_empty(),
            "{}: {:?}",
            s(&copy),
            signatures.broken
        );
        assert!(s(&copy).contains("for <"), "{}", s(&copy));
    }
    // The holder is the entry where the Referrer cannot name the lifetime: a path hides it (the
    // Leaker takes it as the one that elision gives, F28), so `Self` adds to it.
    for (item, holder) in [
        (
            parse_quote!(
                pub struct S<'a>(fn(::core::slice::Iter<u8>, Self) -> &u8, &'a u8);
            ),
            "fn (:: core :: slice :: Iter < u8 > , Self) -> & u8",
        ),
        (
            parse_quote!(
                pub struct S<'a>(Box<dyn Fn(::core::slice::Iter<u8>, Self) -> &u8>, &'a u8);
            ),
            "dyn Fn (:: core :: slice :: Iter < u8 > , Self) -> & u8",
        ),
    ] {
        let item: Item = item;
        let (entries, copy) = f15_copy(&item).unwrap();
        assert_eq!(entries, [holder], "{}", s(&item));
        let signatures = signatures_of(&copy);
        assert_eq!(signatures.count, 0, "nothing of it is left: {}", s(&copy));
    }
    // An entry that adds lifetimes to the inputs makes the signature the holder, as in F12: here
    // `Self` is its qualified self type. It is an optional entry (an external path), so a Referrer
    // that leaves those as written (the default) gets the lifetime named, as above.
    {
        let item: Item = parse_quote!(
            pub struct S<'a>(Box<dyn Fn(&<Self as ::p::Tr>::A) -> &u8>, &'a u8);
        );
        let (entries, copy) = f15_copy(&item).unwrap();
        assert_eq!(
            entries,
            ["dyn Fn (& < Self as :: p :: Tr > :: A) -> & u8"],
            "{}",
            s(&item)
        );
        let signatures = signatures_of(&copy);
        assert!(
            signatures.broken.is_empty(),
            "{}: {:?}",
            s(&copy),
            signatures.broken
        );
    }
    // Where the elision is not at stake, `Self` is left alone: `Self` has no lifetime, the inputs
    // keep the lifetime that `Self` has, or the output does not rely on elision.
    for item in [
        parse_quote!(
            pub struct S<T> {
                f: fn(&Self) -> &u8,
                g: T,
            }
        ),
        parse_quote!(
            pub struct S<'a> {
                f: for<'e> fn(&'e Self) -> &'e u8,
                g: &'a u8,
            }
        ),
        parse_quote!(
            pub struct S<'a> {
                f: fn(&'a Self) -> &'a u8,
                g: &'a u8,
            }
        ),
        parse_quote!(
            pub struct S<'a> {
                f: fn(&'a Self) -> &u8,
                g: &'a u8,
            }
        ),
        parse_quote!(
            pub struct S<'a> {
                f: fn(&Self) -> Self,
                g: &'a u8,
            }
        ),
        parse_quote!(
            pub struct S<'a> {
                f: fn(&u8) -> &Self,
                g: &'a u8,
            }
        ),
        // Its own signature: the lifetimes of the inner function pointer are not those of the
        // outer one.
        parse_quote!(
            pub struct S<'a> {
                f: fn(&fn(&Self) -> Self) -> &u8,
                g: &'a u8,
            }
        ),
    ] {
        let item: Item = item;
        let (entries, copy) = assert_accepted(&item, f15_copy(&item));
        let signatures = signatures_of(&copy);
        assert!(
            signatures.broken.is_empty(),
            "{}: {:?}",
            s(&copy),
            signatures.broken
        );
        assert!(entries.is_empty(), "{}: {:?}", s(&item), entries);
        // Nothing is named that the original does not name: no binder is added.
        assert_eq!(
            s(&copy).matches("for <").count(),
            s(&item).matches("for <").count(),
            "{}",
            s(&copy)
        );
    }
    // Where nothing can hold the signature and the Referrer cannot name its lifetime (a path hides
    // it, in a bound), the item is rejected, not copied wrongly.
    for item in [
        parse_quote! { pub struct S<'a, F: Fn(::core::slice::Iter<u8>, Self) -> &u8>(&'a F); },
        parse_quote! {
            pub enum E<'a, T> where T: Fn(::core::slice::Iter<u8>, Self) -> &u8 { A(&'a T) }
        },
    ] {
        let item: Item = item;
        let err = f15_copy(&item)
            .err()
            .unwrap_or_else(|| panic!("{} is accepted", s(&item)));
        let note = first_note(&err[0]);
        assert!(note.contains("E0106"), "{}: {}", s(&item), note);
    }
    // The copy of an impl keeps `Self`, and so does that of a trait, whose `Self::A` is written
    // `<Host>::A` (and `<__S>::A` in the Repeater): nothing is leaked.
    for item in [
        parse_quote!(
            impl<'a> ::p::Tr for &'a u8 {
                type A = u8;
                fn f(&self, x: Box<dyn Fn(&Self) -> &u8>);
                fn g(x: &Self) -> &u8;
                fn h(x: fn(&Self::A, Self) -> &u8);
            }
        ),
        parse_quote!(
            pub trait Tr<'a> {
                type A;
                fn f(&self, x: Box<dyn Fn(&Self) -> &u8>, y: fn(&Self::A) -> &u8);
            }
        ),
    ] {
        let item: Item = item;
        let (context, referrer) = analyze_in(common::std_leaker(), &item).unwrap();
        let entries: Vec<String> = common::type_texts(&referrer);
        assert!(entries.is_empty(), "{}: {:?}", s(&item), entries);
        // The same visitor as for the others: `Self` is not replaced in the copy of an impl.
        let mut copy = item.clone();
        let host: Type = parse_quote!(Host);
        let mut visitor = common::visitor(&referrer, context, common::placeholder);
        match &item {
            Item::Trait(_) => visitor.with_self(host).visit_item_mut(&mut copy),
            _ => visitor.visit_item_mut(&mut copy),
        }
        let signatures = signatures_of(&copy);
        assert!(
            signatures.broken.is_empty(),
            "{}: {:?}",
            s(&copy),
            signatures.broken
        );
    }
}

// ── F16 ──────────────────────────────────────────────────────────────────────

/// Whether `generics` say `name: 'lifetime`, where `name` is a type parameter (`T`) or a
/// lifetime parameter (`'a`), as a bound or as a where predicate.
fn generics_say_outlives(generics: &Generics, name: &str, lifetime: &str) -> bool {
    let has = |bounds: &Punctuated<TypeParamBound, Token![+]>| {
        bounds
            .iter()
            .any(|b| matches!(b, TypeParamBound::Lifetime(l) if l.to_string() == lifetime))
    };
    let has_lt =
        |bounds: &Punctuated<Lifetime, Token![+]>| bounds.iter().any(|l| l.to_string() == lifetime);
    generics.params.iter().any(|param| match param {
        GenericParam::Type(t) => t.ident == name && has(&t.bounds),
        GenericParam::Lifetime(l) => l.lifetime.to_string() == name && has_lt(&l.bounds),
        GenericParam::Const(_) => false,
    }) || generics
        .where_clause
        .iter()
        .flat_map(|w| &w.predicates)
        .any(|predicate| match predicate {
            WherePredicate::Type(p) => s(&p.bounded_ty) == name && has(&p.bounds),
            WherePredicate::Lifetime(p) => p.lifetime.to_string() == name && has_lt(&p.bounds),
            _ => false,
        })
}

/// F16 (fixed): the outlives bound that a method signature implies for `&'x Self::A` is recorded in
/// the context of the entry, as the where predicate `Self::A: 'x`.
///
/// Trigger: `trait Tr<'a, T> { type A0; fn m<'x>(&'x self, a: Rel<&'x mut Self::A0>) where Self:
/// 'x; }`. The type of `a` implies `<Self as Tr<'a, T>>::A0: 'x`, but the Repeater's impl of the
/// entry has no signature around it: E0309 "the associated type `<__S as Tr<'a, T>>::A0` may not
/// live long enough" in the Leaker's crate.
///
/// Cause: the Leaker recorded only `Self: 'x` for the reference.
///
/// Fix: `Self::A0: 'x` is recorded as if written, and the impl declares it; the Referrer computes
/// the same context from the signature. The components (`T: 'x`, `'a: 'x`) are not recorded: the
/// copy would be stricter than the original (E0491 in an impl for a host). `Self: 'x` is recorded
/// only where a where clause writes it (F34).
#[test]
fn f16_implied_outlives_of_self_projection_lost() {
    // `Self: 'x` is recorded where a where clause writes it, and not for the projection alone (F34:
    // it is more than the original has, which the receiver implies at best).
    let items: Vec<OutlivesCase> = vec![
        (
            parse_quote!(
                pub trait Tr<'a, T> {
                    type A0;
                    fn m<'x>(&'x self, a: Rel<&'x mut Self::A0>)
                    where
                        Self: 'x;
                }
            ),
            "Rel <",
            &["Self :: A0 : 'x"],
            &["'x"],
            &[],
            "(& 'a () , & 'x () , T , & 'x Self)",
        ),
        // Only the receiver implies `Self: 'x`: it is not recorded for the reference (F34).
        (
            parse_quote!(
                pub trait Tr<T> {
                    type A0;
                    fn m<'x>(&'x self, a: Rel<&'x Self::A0>);
                }
            ),
            "Rel <",
            &["Self :: A0 : 'x"],
            &[],
            &[],
            "(& 'x () , T)",
        ),
        // No generics.
        (
            parse_quote!(
                pub trait Tr {
                    type A0;
                    fn m<'x>(&'x self, a: Rel<&'x Self::A0>);
                }
            ),
            "Rel <",
            &["Self :: A0 : 'x"],
            &[],
            &[],
            "(& 'x () ,)",
        ),
        // Under a function pointer.
        (
            parse_quote!(
                pub trait Tr<'a, T> {
                    type A0;
                    fn m<'x>(&'x self, a: Rel<fn(&u8) -> &&'x Self::A0>)
                    where
                        Self: 'x;
                }
            ),
            "Rel <",
            &["Self :: A0 : 'x"],
            &["'x"],
            &[],
            "(& 'a () , & 'x () , T , & 'x Self)",
        ),
        // Written: recorded, and encoded.
        (
            parse_quote!(
                pub trait Tr<'a, T> {
                    type A0;
                    fn m<'x>(&'x self, a: Rel<&'x mut Self::A0>)
                    where
                        Self: 'x,
                        T: 'x,
                        'a: 'x;
                }
            ),
            "Rel <",
            &["Self :: A0 : 'x"],
            &["'x"],
            &["T", "'a"],
            "(& 'a () , & 'x () , T , & 'x & 'a () , & 'x T , & 'x Self)",
        ),
    ];
    for case in &items {
        check_outlives_case(case);
    }
}

// ── F17 ──────────────────────────────────────────────────────────────────────

/// F17 (fixed): an optional entry in a where predicate that names a lifetime, a predicate of the
/// entry's own context.
///
/// Trigger: `struct S<'a, T: 'a> where T: Fn(&'a ::ext::W<T, ::ext::B0>) -> u8 { f0: u8 }`, with
/// `::ext` not allowed. The reference is an optional entry (it holds `::ext::B0`) whose context
/// has the predicate that names it. The copy has `T: Fn(<() as Repeater<..>>::Type)`, while the
/// Repeater's impl needs it as written: E0277 "expected an `Fn(&W<T, B0>)` closure, found `T`".
///
/// Cause: the rule of F2 (an optional type in such a bound is written as it is) did not cover an
/// optional type leaked whole (a reference with a lifetime) whose own context names it.
///
/// Fix: such a type is no entry, nor are the optional types in it (its parts cannot stand for it).
/// The items and the controls (`::ext::W<T>` alone, or no lifetime around it) are accepted, with
/// their exact entries.
#[test]
fn f17_optional_entry_in_own_context() {
    // The item, and its entries: the optional type that the bound of its own context names is none.
    let items: Vec<(ItemStruct, &[&str])> = vec![
        (
            parse_quote!(
                pub struct S<'a, T: 'a>
                where
                    T: Fn(&'a ::ext::W<T, ::ext::B0>) -> u8,
                {
                    f0: u8,
                }
            ),
            &[],
        ),
        // The bound written inline, a trait object that holds the other optional type.
        (
            parse_quote!(
                pub struct S<'a, T: 'a + Fn(&'a ::ext::W<dyn Fn(T) -> ::ext::B0 + 'a>) -> u8> {
                    f0: &'a T,
                }
            ),
            &[],
        ),
        // The controls. No optional type inside: left as written.
        (
            parse_quote!(
                pub struct S<'a, T: 'a>
                where
                    T: Fn(&'a ::ext::W<T>) -> u8,
                {
                    f0: u8,
                }
            ),
            &[],
        ),
        // No lifetime around: the predicate is not part of the context of `::ext::B0`.
        (
            parse_quote!(
                pub struct S<'a, T: 'a>
                where
                    T: Fn(::ext::W<T, ::ext::B0>) -> u8,
                {
                    f0: &'a T,
                }
            ),
            &["::ext::B0"],
        ),
    ];
    for (item, entries) in &items {
        let mut leaker = common::std_leaker();
        leaker.intern_struct(item);
        let referrer = assert_accepted(item, leaker.result().finish());
        assert_entries(item, &referrer, entries);
    }
}

// ── F18 ──────────────────────────────────────────────────────────────────────

/// F18 (fixed): a type of another crate that holds a method lifetime is copied as written.
///
/// Trigger: `trait Tr { fn m<'x>(&self, a: ::ext::W<&'x u8>) -> Rel<'x>; }`, with `::ext` not
/// allowed. `'x` is late-bound in the trait, but the copy was `fn m<'x>(&self, a: P<1>) -> P<0>`:
/// projections do not constrain `'x`, and the impl in the Referrer's crate was E0195 "lifetime
/// parameters differ in whether they are early- or late-bound".
///
/// Cause: the check of F7 counted the optional type as an input that keeps `'x`, which holds only
/// for a copy that writes it as it is.
///
/// Fix: if the only inputs that keep a method lifetime (outside the types leaked whole) are types
/// of another crate, they are no entries, so every copy keeps them; their crate must resolve in the
/// Referrer's crate. Another input that keeps it (`&'x u8`, `&'x self`), an output without it, a
/// lifetime of the trait, or an inherent impl (F22) leaves them entries. Each case is an item, its
/// entries and the copy (`P<ix>` for an entry).
#[test]
fn f18_the_optional_entries_that_hold_a_method_lifetime_are_copied_as_written() {
    let cases: Vec<(Item, &[&str], &str)> = vec![
        // The type is the only input that holds `'x`: also behind a box, in a tuple, and in the
        // output too (the same type, in the same context).
        (
            parse_quote!(
                trait Tr {
                    fn m<'x>(&self, a: ::ext::W<&'x u8>) -> Rel<'x>;
                }
            ),
            &["Rel<'x>"],
            "trait Tr { fn m < 'x > (& self , a : :: ext :: W < & 'x u8 >) -> P < 0usize > ; }",
        ),
        (
            parse_quote!(
                trait Tr {
                    fn m<'x>(&self, a: ::ext::W<'x>) -> &'x u8;
                }
            ),
            &[],
            "trait Tr { fn m < 'x > (& self , a : :: ext :: W < 'x >) -> & 'x u8 ; }",
        ),
        (
            parse_quote!(
                trait Tr {
                    fn m<'x>(&self, a: Box<::ext::W<'x>>) -> &'x u8;
                }
            ),
            &[],
            "trait Tr { fn m < 'x > (& self , a : Box < :: ext :: W < 'x > >) -> & 'x u8 ; }",
        ),
        (
            parse_quote!(
                trait Tr {
                    fn m<'x>(&self, a: (::ext::W<'x>, u8)) -> &'x u8;
                }
            ),
            &[],
            "trait Tr { fn m < 'x > (& self , a : (:: ext :: W < 'x > , u8)) -> & 'x u8 ; }",
        ),
        (
            parse_quote!(
                trait Tr {
                    fn m<'x>(&self, a: ::ext::W<'x>) -> ::ext::W<'x>;
                }
            ),
            &[],
            "trait Tr { fn m < 'x > (& self , a : :: ext :: W < 'x >) -> :: ext :: W < 'x > ; }",
        ),
        // Two optional types hold it: both are kept. A type beside them that holds no lifetime of
        // the method stays optional.
        (
            parse_quote!(
                trait Tr {
                    fn m<'x>(&self, a: ::ext::W<'x>, b: ::ext2::V<'x>, c: ::ext::Y<u8>) -> &'x u8;
                }
            ),
            &["::ext::Y<u8>"],
            "trait Tr { fn m < 'x > (& self , a : :: ext :: W < 'x > , b : :: ext2 :: V < 'x > , c : P < 0usize >) -> & 'x u8 ; }",
        ),
        // A type leaked whole holds it too, and does not keep it in the copy.
        (
            parse_quote!(
                trait Tr {
                    fn m<'x>(&self, a: ::ext::W<'x>, b: Rel<'x>) -> &'x u8;
                }
            ),
            &["Rel<'x>"],
            "trait Tr { fn m < 'x > (& self , a : :: ext :: W < 'x > , b : P < 0usize >) -> & 'x u8 ; }",
        ),
        // The method of an impl of a trait is compared with the trait's, as a trait's is.
        (
            parse_quote!(
                impl ::p::Tr for ::p::X {
                    fn m<'x>(&self, a: ::ext::W<'x>) -> &'x u8 {
                        loop {}
                    }
                }
            ),
            &["::p::X"],
            "impl :: p :: Tr for P < 0usize > { fn m < 'x > (& self , a : :: ext :: W < 'x >) -> & 'x u8 { loop { } } }",
        ),
        // Controls: another input keeps `'x` as written, the receiver or a reference, so the type
        // is an optional entry as ever.
        (
            parse_quote!(
                trait Tr {
                    fn m<'x>(&self, a: ::ext::W<'x>, b: &'x u8) -> &'x u8;
                }
            ),
            &["::ext::W<'x>"],
            "trait Tr { fn m < 'x > (& self , a : P < 0usize > , b : & 'x u8) -> & 'x u8 ; }",
        ),
        (
            parse_quote!(
                trait Tr {
                    fn m<'x>(&'x self, a: ::ext::W<'x>) -> &'x u8;
                }
            ),
            &["::ext::W<'x>"],
            "trait Tr { fn m < 'x > (& 'x self , a : P < 0usize >) -> & 'x u8 ; }",
        ),
        // The output does not name it, or it is a lifetime of the trait: nothing to keep.
        (
            parse_quote!(
                trait Tr {
                    fn m<'x>(&self, a: ::ext::W<'x>) -> u8;
                }
            ),
            &["::ext::W<'x>"],
            "trait Tr { fn m < 'x > (& self , a : P < 0usize >) -> u8 ; }",
        ),
        (
            parse_quote!(
                trait Tr<'x> {
                    fn m(&self, a: ::ext::W<'x>) -> &'x u8;
                }
            ),
            &["::ext::W<'x>"],
            "trait Tr < 'x > { fn m (& self , a : P < 0usize >) -> & 'x u8 ; }",
        ),
        // A method of an inherent impl has no signature to be compared with (F22).
        (
            parse_quote!(
                impl ::p::X {
                    fn m<'x>(&self, a: ::ext::W<'x>) -> &'x u8 {
                        loop {}
                    }
                }
            ),
            &["::p::X", "::ext::W<'x>"],
            "impl P < 0usize > { fn m < 'x > (& self , a : P < 1usize >) -> & 'x u8 { loop { } } }",
        ),
    ];
    for (item, expected_entries, expected) in &cases {
        let what = s(item);
        let mut leaker = common::std_leaker();
        let context = common::intern_item(&mut leaker, item);
        let result = leaker.result();
        let referrer = assert_accepted(item, result.finish());
        let entries = entries_text(&referrer, false);
        let expected_entries: Vec<String> =
            expected_entries.iter().map(|ty| ty.to_string()).collect();
        assert_eq!(entries, expected_entries, "`{}`", what);
        assert_eq!(
            rewrite(&referrer, context, item.clone()),
            *expected,
            "`{}`",
            what
        );
    }
}

// ── F19 and F20 ──────────────────────────────────────────────────────────────

/// The copy of `item` that a macro makes: the entries replaced by `P<ix>`.
fn f19_copy(item: &Item) -> std::result::Result<(Referrer, Item), LeakerResultError> {
    f19_copy_with(common::std_leaker(), item)
}

/// Like `f19_copy()`, with the Leaker that `common::p_leaker()` makes (it allows `::p`, so that the self
/// type of an impl has parts that are entries and a part that is not).
fn f19_copy_with<A: Allow>(
    leaker: Leaker<A>,
    item: &Item,
) -> std::result::Result<(Referrer, Item), LeakerResultError> {
    f19_result_with(leaker, item).map(|(_, referrer, copy)| (referrer, copy))
}

/// Like `f19_copy_with()`, with the result that writes the impls too.
fn f19_result_with<A: Allow>(
    mut leaker: Leaker<A>,
    item: &Item,
) -> std::result::Result<(LeakerResult, Referrer, Item), LeakerResultError> {
    let context = common::intern_item(&mut leaker, item);
    let result = leaker.result();
    let referrer = result.clone().finish()?;
    let mut copy = item.clone();
    common::visitor(&referrer, context, common::placeholder).visit_item_mut(&mut copy);
    Ok((result, referrer, copy))
}

/// F19 (fixed): in an impl whose self type is leaked, a method has a bound that names `Self` on a
/// generic, and a predicate that names a leaked type that uses the generic.
///
/// Trigger: `impl S { fn m<X>(&self) where X: Fn(Self) -> u8, Rel<X>: Default {} }`, with `S` and
/// `Rel` leaked. The Repeater's impl of `Rel<X>` has `X: Fn(S) -> u8`; the copy has `X: Fn(Self)
/// -> u8` with `Self` the replaced self type, not normalized yet where rustc proves the predicate:
/// E0277 "expected an `Fn(S)` closure, found `X`" (E0271 for a `for<'h>` bound).
///
/// Cause: it is F2 with `Self` as the leaked type, and the check looked only at the generics of the
/// impl, not at those of a method.
///
/// Fix: a bound is unstable if it names a leaked type, `Self` where the copy replaces the self type
/// or a part of it, `Self::A` (F20) or an associated type that rustc normalizes (F21); an entry in
/// a predicate whose context has one is an error. For `Self` only the bounds of the generics that
/// the entries of the self type use count (F13, F24). The controls (the entry in a position, a
/// bound without `Self`) are accepted.
#[test]
fn f19_impl_self_in_bound_of_entry_in_predicate() {
    let items: Vec<Item> = vec![
        parse_quote!(
            impl S {
                fn m<X>(&self)
                where
                    X: Fn(Self) -> u8,
                    Rel<X>: Default,
                {
                }
            }
        ),
        parse_quote!(
            impl S {
                fn m<X>(&self, a0: X)
                where
                    X: for<'h> Fn(&'h Self) -> &'h Result<Self, f64>,
                    Rel<Rel2<X>>: Default,
                {
                }
            }
        ),
        // The bound inline.
        parse_quote!(
            impl S {
                fn m<X: Fn(&Self) -> u8>(&self)
                where
                    Rel<X>: Default,
                {
                }
            }
        ),
    ];
    let controls: Vec<(Item, &[&str])> = vec![
        // The entry is in a position, which rustc normalizes once the environment is built.
        (
            parse_quote!(
                impl S {
                    fn m<X>(&self, a0: Rel<X>)
                    where
                        X: Fn(Self) -> u8,
                    {
                    }
                }
            ),
            &["S", "Rel<X>"],
        ),
        // The bound does not name `Self`.
        (
            parse_quote!(
                impl S {
                    fn m<X>(&self)
                    where
                        X: Fn(u8) -> u8,
                        Rel<X>: Default,
                    {
                    }
                }
            ),
            &["S", "Rel<X>"],
        ),
    ];
    // Each is rejected: no copy of it exists whose predicate names such an entry. The type is copied
    // as written whole, and a type in it that conflicts for the same reason has an error of its own
    // after it (`Rel2<X>` in `Rel<Rel2<X>>`).
    let blamed = [
        ("Rel < X >", "X : Fn (Self) -> u8", Blames::SelfType, vec![]),
        (
            "Rel < Rel2 < X > >",
            "X : for < 'h > Fn (& 'h Self) -> & 'h Result < Self , f64 >",
            Blames::SelfType,
            vec!["Rel2<X>"],
        ),
        (
            "Rel < X >",
            "X : Fn (& Self) -> u8",
            Blames::SelfType,
            vec![],
        ),
    ];
    assert_eq!(items.len(), blamed.len());
    for (item, (ty, bound, blames, inside)) in items.iter().zip(blamed) {
        let result = f19_copy(item).map_err(|errors| {
            let mut errors = errors.into_vec();
            let rest = errors.split_off(1);
            let expected: Vec<String> = inside
                .iter()
                .map(|ty| {
                    format!(
                        "`{}` is in a bound or where predicate, and a bound of its generics names \
                         `Self`, which is a leaked type",
                        ty
                    )
                })
                .collect();
            assert_eq!(messages(&rest), expected.join("\n"));
            errors
        });
        assert_unstable(item, result, ty, bound, blames);
    }
    for (item, entries) in &controls {
        let (referrer, _) = assert_accepted(item, f19_copy(item));
        assert_entries(item, &referrer, entries);
    }
    // The self type is leaked by its parts only (`::p::W` is allowed, `Rel3<u8>` and `Rel<U>` are
    // not): `Self` is replaced in the copy as well, and an entry in a predicate whose context has
    // the bound `T: AsRef<Self>` is the same cycle (E0277 "the trait bound `T:
    // AsRef<W<T, U, Rel3<u8>>>` is not satisfied", checked with rustc stable and nightly: the
    // Repeater's impl for the entry has the bound with the Leaker's self type). The bound on `T`
    // is not a bound of the context of an entry in the self type, so the impl is accepted without
    // the predicate (F13).
    let by_parts: Vec<Item> = vec![
        parse_quote!(
            impl<T: AsRef<Self>, U> ::p::W<T, U, Rel3<u8>> {
                fn f(&self, a: Rel3<u8>)
                where
                    Rel2<T>: Default,
                {
                }
            }
        ),
        parse_quote!(
            impl<T: AsRef<Self>, U> ::p::Tr for ::p::W<T, U, Rel3<u8>> where Rel2<T>: Default {}
        ),
        // The bound of a method that names the type, in the self type with an entry that uses
        // `U` only.
        parse_quote!(
            impl<T: AsRef<Self>, U> ::p::W<T, U, Rel<U>> {
                fn f<X: Into<Rel2<T>>>(&self, x: X, a: Rel3<u8>) {}
            }
        ),
    ];
    for item in &by_parts {
        assert_unstable(
            item,
            f19_copy_with(common::p_leaker(), item),
            "Rel2 < T >",
            "T : AsRef < Self >",
            Blames::SelfType,
        );
    }
    let by_parts_controls: Vec<(Item, &[&str])> = vec![
        // The predicate names an entry that does not use `T`, whose bound names `Self`.
        (
            parse_quote!(
                impl<T: AsRef<Self>, U> ::p::W<T, U, Rel<U>> {
                    fn f(&self, a: Rel3<u8>)
                    where
                        Rel2<U>: Default,
                    {
                    }
                }
            ),
            &["Rel<U>", "Rel2<U>", "Rel3<u8>"],
        ),
        (
            parse_quote!(
                impl<T: AsRef<Self>, U> ::p::Tr<Rel2<U>> for ::p::W<T, U, Rel<U>> where Rel3<U>: Default {}
            ),
            &["Rel<U>", "Rel2<U>", "Rel3<U>"],
        ),
        // The `Item` that a bound equates is normalized when the copy proves the Repeater's impl
        // of `Rel<I>` for `Self` (E0277, "expected an `Fn(u8)` closure, found `I`"), unless the
        // context of the entry leaves out `I: Fn(I::Item)` (F26): it compiles.
        (
            parse_quote!(
                impl<I: Iterator<Item = u8> + Fn(I::Item), V: AsRef<Self>> ::p::W<Rel<I>, I, V> {}
            ),
            &["Rel<I>"],
        ),
    ];
    for (item, entries) in &by_parts_controls {
        let (referrer, _) = assert_accepted(item, f19_copy_with(common::p_leaker(), item));
        assert_entries(item, &referrer, entries);
    }
    // `Self` in a predicate: rustc normalizes the types in the self type that the copy replaces,
    // with the Repeater's impls of their entries, which need the bounds of the generics that they
    // use (E0277 "expected an `Fn(Rel3<u8>)` closure, found `T`", checked with rustc stable and
    // nightly). The bound of a generic that no such type uses does not matter (F13).
    let normalized: Vec<(ItemImpl, &str, Blames)> = vec![
        (
            parse_quote!(
                impl<T: Fn(Rel2<u8>), V: AsRef<Self>> ::p::W<Rel<T>, T, V> {
                    fn f(&self, a: Rel3<u8>) {}
                }
            ),
            "T : Fn (Rel2 < u8 >)",
            Blames::Leaked,
        ),
        (
            parse_quote!(
                impl<T, V> ::p::W<Rel<T>, T, V>
                where
                    T: Fn(Rel2<u8>),
                    V: AsRef<Self>,
                {
                }
            ),
            "T : Fn (Rel2 < u8 >)",
            Blames::Leaked,
        ),
    ];
    for (item, bound, blames) in &normalized {
        let mut leaker = common::p_leaker();
        leaker.intern_impl(item);
        assert_unstable(item, leaker.result().finish(), "Self", bound, *blames);
    }
}

/// F20 (fixed): in a trait, a method has a bound that names `Self::A` on a generic, and a predicate
/// that names a leaked type that uses the generic.
///
/// Trigger: `trait Tr { type A0; fn m<X>(&self) where X: Fn(&Self::A0) -> u8, Rel<X>: Default; }`.
/// The Repeater's impl of `Rel<X>` has `X: Fn(&<__S as Tr>::A0) -> u8`. The copy for a host has
/// `X: Fn(&<Host as Tr>::A0) -> u8`, not normalized yet where rustc proves `<Host as
/// Repeater<(X,), ..>>::Type: Default`: E0277 "expected an `Fn(&u8)` closure, found `X`".
///
/// Cause: the rule for `Self::A` looked at the bounds of the trait, not at those of a method.
///
/// Fix: with F19, a bound that names `Self::A` is unstable, so the entry in a predicate is an
/// error. The controls (the entry in a position, a bound without `Self::A`) are accepted.
#[test]
fn f20_trait_assoc_in_bound_of_entry_in_predicate() {
    let items: Vec<Item> = vec![
        parse_quote!(
            pub trait Tr {
                type A0;
                fn m<X>(&self)
                where
                    X: Fn(&Self::A0) -> u8,
                    Rel<X>: Default;
            }
        ),
        parse_quote!(
            pub trait Tr<'a, T> {
                type A1;
                fn m<X>(&self)
                where
                    T: for<'h> Fn(&'h (X, Self::A1, &'a dyn ::core::any::Any)) -> &'h String,
                    Rel<T, X>: Default;
            }
        ),
    ];
    let controls: Vec<(Item, &[&str])> = vec![
        (
            parse_quote!(
                pub trait Tr {
                    type A0;
                    fn m<X>(&self, a0: Rel<X>)
                    where
                        X: Fn(&Self::A0) -> u8;
                }
            ),
            &["Rel<X>"],
        ),
        (
            parse_quote!(
                pub trait Tr {
                    type A0;
                    fn m<X>(&self)
                    where
                        X: Fn(&u8) -> u8,
                        Rel<X>: Default;
                }
            ),
            &["Rel<X>"],
        ),
    ];
    // Each is rejected: no copy of it exists whose predicate names such an entry.
    let blamed = [
        ("Rel < X >", "X : Fn (& Self :: A0) -> u8", Blames::SelfAssoc("Self :: A0")),
        ("Rel < T , X >", "T : for < 'h > Fn (& 'h (X , Self :: A1 , & 'a dyn :: core :: any :: Any)) -> & 'h String", Blames::SelfAssoc("Self :: A1")),
    ];
    assert_eq!(items.len(), blamed.len());
    for (item, (ty, bound, blames)) in items.iter().zip(blamed) {
        assert_unstable(item, f19_copy(item), ty, bound, blames);
    }
    for (item, entries) in &controls {
        let (referrer, _) = assert_accepted(item, f19_copy(item));
        assert_entries(item, &referrer, entries);
    }
}

// ── F21 ──────────────────────────────────────────────────────────────────────

/// F21 (fixed): a bound names the `Item` of an iterator, and a predicate of an inner item equates
/// the `Item` to a leaked type.
///
/// Trigger: `trait Tr<T: Iterator> where T: Fn(<T as Iterator>::Item) { fn m(&self) where T:
/// Iterator<Item = Rel<T>>; }` (also an impl). The Repeater's impl of `Rel<T>` has `T:
/// Fn(T::Item)`, and the copy has `T: Iterator<Item = <() as Repeater<(T,), ..>>::Type>`: rustc
/// proves that impl while it normalizes `T::Item`, which the entry defines: E0275 "overflow
/// evaluating the requirement" (E0277 for a tuple).
///
/// Cause: the check of F2 did not see it: the bound names no leaked type, only the `Item`, and it
/// is a bound of an enclosing item.
///
/// Fix: with F19, a bound that names the `Item` (or the `Output` of `Fn(..)`, or a projection that
/// an impl defines) is unstable if an equation of it is in scope. The controls (the `Item` in a
/// position, a bound without `Item`) are accepted.
#[test]
fn f21_iterator_item_in_bound_of_entry_in_item_binding() {
    let items: Vec<Item> = vec![
        parse_quote!(
            impl<T: Iterator> S<T>
            where
                T: Fn(T::Item),
            {
                fn m(&self)
                where
                    T: Iterator<Item = Result<T, I0>>,
                {
                }
            }
        ),
        parse_quote!(
            pub trait Tr<T: Iterator>
            where
                T: Fn(<T as Iterator>::Item),
            {
                fn m(&self)
                where
                    T: Iterator<Item = Rel<T>>;
            }
        ),
    ];
    let controls: Vec<(Item, &[&str])> = vec![
        // The `Item` is in a position.
        (
            parse_quote!(
                pub trait Tr<T: Iterator> {
                    fn m(&self, a: T::Item)
                    where
                        T: Iterator<Item = Rel<T>>;
                }
            ),
            &["Rel<T>"],
        ),
        // The bound names no `Item`.
        (
            parse_quote!(
                pub trait Tr<T: Iterator>
                where
                    T: Fn(u8),
                {
                    fn m(&self)
                    where
                        T: Iterator<Item = Rel<T>>;
                }
            ),
            &["Rel<T>"],
        ),
    ];
    // Each is rejected: no copy of it exists whose predicate names such an entry.
    let blamed = [
        (
            "Result < T , I0 >",
            "T : Fn (T :: Item) + Iterator",
            Blames::Projection("T :: Item"),
        ),
        (
            "Rel < T >",
            "T : Fn (< T as Iterator > :: Item) + Iterator",
            Blames::Projection("< T as Iterator > :: Item"),
        ),
    ];
    assert_eq!(items.len(), blamed.len());
    for (item, (ty, bound, blames)) in items.iter().zip(blamed) {
        assert_unstable(item, f19_copy(item), ty, bound, blames);
    }
    for (item, entries) in &controls {
        let (referrer, _) = assert_accepted(item, f19_copy(item));
        assert_entries(item, &referrer, entries);
    }
}

// ── F22 ──────────────────────────────────────────────────────────────────────

/// F22 (fixed): the check of F7 for E0195 also rejected a free function and an inherent method.
///
/// Trigger: `pub fn f<'x>(a: crate::Wrap<'x>) -> &'x u8 { .. }` visited by a `View`, and `impl
/// ::p::W { fn f<'x>(&self, a: Wrap<'x>) -> &'x u8 { .. } }`: rejected with the note of F7,
/// although their copies compile.
///
/// Cause: E0195 is about a method whose lifetime is early-bound where the trait's is late-bound. A
/// free function or an inherent method has no trait signature to differ from.
///
/// Fix: the check runs only for a method of a trait or of an impl of a trait (still rejected, also
/// with `&'x dyn RelTrait`), nested or not. A signature that a `View` visits alone may be a
/// trait's, so it is checked.
#[test]
fn f22_method_lifetime_in_free_function_and_inherent_method() {
    // The controls: the copy of a trait, and of an impl of a trait, has the E0195 of F7.
    let mut rejected: Vec<Item> = vec![
        parse_quote!(
            pub trait Tr {
                fn f<'x>(&self, a: Wrap<'x>) -> &'x u8;
            }
        ),
        parse_quote!(
            impl ::p::Tr for ::p::W {
                fn f<'x>(&self, a: Wrap<'x>) -> &'x u8 {
                    loop {}
                }
            }
        ),
        parse_quote!(
            impl ::p::Tr for ::p::W {
                fn f<'x>(&self, a: crate::Wrap<'x>) -> &'x u8 {
                    loop {}
                }
            }
        ),
    ];
    rejected.push(parse_quote!(
        pub trait Tr {
            fn f<'x>(&self, a: &'x dyn RelTrait) -> &'x u8;
        }
    ));
    for item in &rejected {
        let mut leaker = common::p_leaker();
        common::intern_item(&mut leaker, item);
        let err = errors_of(leaker);
        assert!(!err.is_empty(), "`{}` is accepted", s(item));
        let note = first_note(&err[0]);
        assert!(
            note.contains("(E0195)"),
            "`{}` is rejected, but not for E0195: {}",
            s(item),
            note
        );
    }
    // No signature to differ from.
    let accepted: Vec<Item> = vec![
        parse_quote!(
            pub fn f<'x>(a: crate::Wrap<'x>) -> &'x u8 {
                loop {}
            }
        ),
        parse_quote!(
            impl ::p::W {
                fn f<'x>(&self, a: Wrap<'x>) -> &'x u8 {
                    loop {}
                }
            }
        ),
        // A free function with two inputs that name the lifetime, and an inherent impl of a type.
        parse_quote!(
            pub fn f<'x>(a: crate::Wrap<'x>, b: crate::Wrap<'x>) -> crate::Wrap<'x> {
                loop {}
            }
        ),
        parse_quote!(
            impl ::p::D {
                pub fn g<'x>(&self, a: crate::Wrap<'x>) -> &'x u8 {
                    loop {}
                }
            }
        ),
        // A trait object that takes the lifetime from the reference.
        parse_quote!(
            pub fn f<'x>(a: &'x dyn crate::RelTrait) -> &'x u8 {
                loop {}
            }
        ),
        parse_quote!(
            impl<T> ::p::W<T> {
                fn f<'x>(&self, a: &'x dyn RelTrait, b: Wrap<'x>) -> Wrap<'x> {
                    loop {}
                }
            }
        ),
    ];
    for item in &accepted {
        let mut leaker = common::p_leaker();
        match item {
            // A function item is nested: a `View` visits it.
            Item::Fn(_) => {
                leaker.enter_generics(&Generics::default(), |view| view.visit_item(item));
            }
            _ => {
                common::intern_item(&mut leaker, item);
            }
        }
        assert_accepted(item, leaker.result().finish());
    }
    // The walk of a `View` is the same: the methods of an inherent impl are not checked, those of
    // a trait impl are, and a signature visited alone may be a trait's, so it is.
    let method: Signature = parse_quote!(fn f<'x>(&self, a: Wrap<'x>) -> &'x u8);
    let inherent: ItemImpl = parse_quote!(
        impl ::p::W {
            fn f<'x>(&self, a: Wrap<'x>) -> &'x u8 {
                loop {}
            }
        }
    );
    let of_trait: ItemImpl = parse_quote!(
        impl ::p::Tr for ::p::W {
            fn f<'x>(&self, a: Wrap<'x>) -> &'x u8 {
                loop {}
            }
        }
    );
    let nested = |item: &ItemImpl| {
        let mut leaker = common::p_leaker();
        leaker.enter_generics(&Generics::default(), |view| {
            view.visit_item(&Item::Impl(item.clone()))
        });
        errors_of(leaker)
    };
    assert!(nested(&inherent).is_empty());
    assert!(first_note(&nested(&of_trait)[0]).contains("(E0195)"));
    let mut leaker = common::p_leaker();
    leaker.enter_generics(&Generics::default(), |view| view.visit_signature(&method));
    let result = leaker.result();
    let err = result.errors();
    // `Self` of the receiver has no meaning here, which is told too, in the order of the syntax.
    assert_eq!(err[0].reason, Reason::SelfWithoutMeaning);
    assert!(first_note(&err[1]).contains("(E0195)"));
}

// ── F23 ──────────────────────────────────────────────────────────────────────

/// F23 (fixed): `Self` in an elided signature was rejected where the Referrer could write the
/// lifetime.
///
/// Trigger: `struct S<'a, F: Fn(&Self) -> &u8>(&'a F)` (also a where predicate, or a field `f:
/// fn(&Self) -> &u8`). F15 leaked such a signature whole where `with_self()` adds lifetimes for
/// `Self` (E0106), and rejected it where nothing can hold it, even with no call of `with_self()`.
///
/// Cause: `Self` counts no lifetime for rustc, so the elided output has the lifetime of the only
/// `&`, which the Referrer can name.
///
/// Fix: where `with_self()` writes a type with lifetimes for a `Self` in the inputs of a function
/// pointer or `Fn(..)` whose output is elided and whose inputs show exactly one lifetime, the
/// Visitor names it with a fresh binder (`for<'e> Fn(&'e ::p::S<'a, F>) -> &'e u8`), or writes the
/// inputs' name in the output. The Leaker accepts these items with no entry.
#[test]
fn f23_self_in_elided_signature_where_referrer_can_rewrite() {
    let items: Vec<Item> = vec![
        // A bound, and a where predicate: nothing can hold the signature.
        parse_quote!(
            pub struct S<'a, F: Fn(&Self) -> &u8>(&'a F);
        ),
        parse_quote!(
            pub enum E<'a, T>
            where
                T: Fn(&Self) -> &u8,
            {
                A(&'a T),
            }
        ),
        // A field: the signature could be the entry (F15), but the Referrer names the lifetime.
        parse_quote!(
            pub struct S<'a> {
                f: fn(&Self) -> &u8,
                g: &'a u8,
            }
        ),
        parse_quote!(
            pub struct S<'a>(Box<dyn Fn(&Self) -> &u8>, &'a u8);
        ),
        parse_quote!(
            pub struct S<'a, T>(Box<dyn Fn(&(T, Self)) -> &u8 + 'a>, &'a T);
        ),
        // The inputs name their lifetime: the output takes it.
        parse_quote!(
            pub struct S<'a, 'b> {
                f: for<'x> fn(&'x Self) -> &u8,
                g: &'a u8,
                h: &'b u8,
            }
        ),
        parse_quote!(
            pub struct S<'a, F: for<'x> Fn(&'x Self) -> &u8>(&'a F);
        ),
        // The lifetime is in a path (`'_`).
        parse_quote!(
            pub struct S<'a>(fn(::core::cell::Ref<'_, Self>) -> &u8, &'a u8);
        ),
        // Each of the nested signatures has its binder.
        parse_quote!(
            pub struct S<'a>(fn(&(Self, fn(&Self) -> &u8)) -> &u8, &'a u8);
        ),
        // Several bounds of one item, and a bound of a where predicate.
        parse_quote!(
            pub struct S<'a, F: Fn(&Vec<Self>) -> &u8, G>(F, G, &'a u8)
            where
                G: Fn(&Self) -> Option<&u8>;
        ),
    ];
    for item in &items {
        let (entries, copy) = assert_accepted_as(&format!("`{}`", s(item)), f15_copy(item));
        assert!(entries.is_empty(), "{}: {:?}", s(item), entries);
        let signatures = signatures_of(&copy);
        assert!(
            signatures.broken.is_empty(),
            "{}: {:?}",
            s(&copy),
            signatures.broken
        );
        assert!(signatures.count > 0, "{}", s(&copy));
    }
}

// ── F24 ──────────────────────────────────────────────────────────────────────

/// F24 (fixed): a bound that names a leaked type, on a generic that the entries of the self type do
/// not use, made a bare `Self` in a predicate an error.
///
/// Trigger: `impl<T: Fn(Rel2<u8>), U> ::p::Tr<U> for ::p::W2<T, Rel<U>> where Self: Clone {}`
/// (also `fn m<X: AsRef<Self>>(&self, x: X)`). The entries are `Rel2<u8>` and `Rel<U>`, rustc
/// normalizes `Self` with the impl for `Rel<U>`, whose context has no bound, and the copy compiles.
///
/// Cause: the check looked at every type of the self type that may be chosen as an entry, the
/// whole `W2<T, Rel<U>>` too, whose context has `T: Fn(Rel2<u8>)`; the parts are chosen instead.
///
/// Fix: with a bare `Self` in a predicate, such a type is kept from being an entry when nothing
/// forces it, so its parts are the entries and the check looks at them. It stays an error where
/// an entry that must be one uses `T` (`::p::W2<Rel<T>, U>`).
#[test]
fn f24_self_in_predicate_with_leaked_bound_of_unused_generic() {
    let items: Vec<ItemImpl> = vec![
        parse_quote!(
            impl<T: Fn(Rel2<u8>), U> ::p::Tr<U> for ::p::W2<T, Rel<U>> where Self: Clone {}
        ),
        parse_quote!(
            impl<T: Fn(Rel2<u8>), U> ::p::Tr<U> for ::p::W2<T, Rel<U>> {
                fn m<X: AsRef<Self>>(&self, x: X) {}
            }
        ),
        parse_quote!(
            impl<T: Fn(Rel2<u8>), U> ::p::Tr<U> for ::p::W2<T, Rel<U>>
            where
                Self: Clone,
            {
                fn f(&self, a: u8) {}
                fn m<X: AsRef<Self>>(&self, x: X) {}
            }
        ),
        // The bound as a where predicate.
        parse_quote!(
            impl<T, U> ::p::Tr<U> for ::p::W2<T, Rel<U>>
            where
                T: Fn(Rel2<u8>),
                Self: Clone,
            {
            }
        ),
        // A type between the self type and the entry.
        parse_quote!(
            impl<T: Fn(Rel2<u8>), U> ::p::Tr<U> for ::p::W3<::p::W1<Rel<U>>, T, u8> where Self: Clone {}
        ),
        parse_quote!(
            impl<T: Fn(Rel2<u8>), U> ::p::Tr<U> for ::p::W2<::p::W1<T>, Rel<U>> where Self: Clone {}
        ),
    ];
    for item in &items {
        let (referrer, _) = assert_accepted(
            item,
            f19_copy_with(common::p_leaker(), &Item::Impl(item.clone())),
        );
        let mut entries: Vec<String> = common::type_texts(&referrer);
        entries.sort();
        assert_eq!(entries, ["Rel < U >", "Rel2 < u8 >"], "{}", s(item));
    }
    // An entry that must be one uses `T`, whose bound is unstable: an error, as before.
    let rejected: Vec<ItemImpl> = vec![
        parse_quote!(
            impl<T: Fn(Rel2<u8>), U> ::p::Tr<U> for ::p::W2<Rel<T>, U> where Self: Clone {}
        ),
        parse_quote!(
            impl<T: Fn(Rel2<u8>), V: AsRef<Self>> ::p::W3<Rel<T>, T, V> {}
        ),
        parse_quote!(
            impl<T: Fn(Rel2<u8>), U> ::p::Tr<U> for ::p::W3<::p::W1<Rel<T>>, T, u8> where Self: Clone {}
        ),
    ];
    for item in &rejected {
        assert_unstable(
            item,
            f19_copy_with(common::p_leaker(), &Item::Impl(item.clone())),
            "Self",
            "T : Fn (Rel2 < u8 >)",
            Blames::Leaked,
        );
    }
}

// ── F25 and F26 ──────────────────────────────────────────────────────────────

/// F25 (fixed): a supertrait fixes the `Item` that a bound names, and an entry that uses the
/// generic is in a predicate.
///
/// Trigger: `trait Tr<T: ::p::WIter> where T: Fn(T::Item) { fn m(&self) where Rel<T>: Default; }`,
/// with `trait WIter: Iterator<Item = u8> {}` in the Leaker's crate. The Repeater's impl has `T:
/// Fn(T::Item)`, normalized to `T: Fn(u8)`; the environment of the copy has it not normalized yet
/// where rustc proves the predicate: E0277 "expected an `Fn(u8)` closure, found `T`".
///
/// Cause: it is F21 with the equation in a supertrait, which the Leaker cannot read.
///
/// Fix (F26): the context of `Rel<T>` leaves out `T: Fn(T::Item)`, as another bound of `T` (a
/// trait not of the standard library) may fix an associated type, so the impl has `T: ::p::WIter`
/// only. A bound of another generic (`U: Fn(T::Item)`) stays, and is rejected: a projection of a
/// generic with such a trait bound counts as one that rustc normalizes.
#[test]
fn f25_supertrait_item_in_bound_of_entry_in_predicate() {
    let accepted: Vec<Item> = vec![
        // The controls: rigid `Item`, and the entry in a position.
        parse_quote!(
            pub trait Tr4<T: Iterator>
            where
                T: Fn(T::Item),
            {
                fn m(&self)
                where
                    Rel<T>: Default;
            }
        ),
        parse_quote!(
            pub trait Tr2<T: ::p::WIter> {
                fn m(&self, x: Rel<T>)
                where
                    T: Fn(T::Item);
            }
        ),
        // More controls: every bound of the generic is a trait of the standard library (compiled
        // with rustc too), and a trait with a supertrait is no bound of the generic that the
        // `Item` names.
        parse_quote!(
            pub trait Tr<T: ::core::iter::Iterator>
            where
                T: Fn(T::Item),
            {
                fn m(&self)
                where
                    Rel<T>: Default;
            }
        ),
        parse_quote!(
            pub trait Tr<T: Iterator + Clone>
            where
                T: Fn(<T as Iterator>::Item),
            {
                fn m(&self)
                where
                    Rel<T>: Default;
            }
        ),
        parse_quote!(
            pub trait Tr<T: Iterator, U: ::p::WIter>
            where
                U: Fn(T::Item),
            {
                fn m(&self)
                where
                    Rel<T>: Default;
            }
        ),
        // No projection.
        parse_quote!(
            pub trait Tr<T: ::p::WIter + Fn(u8)> {
                fn m(&self)
                where
                    Rel<T>: Default;
            }
        ),
    ];
    for item in &accepted {
        assert_accepted(item, f19_copy_with(common::p_leaker(), item));
    }
    // What the Leaker rejected before F26 (the price): the supertrait does not fix the `Item`, or
    // the trait has an associated type of its own. The copy compiles, and so it does with the bound
    // left out of the context of the entry.
    let formerly_priced: Vec<(Item, &str)> = vec![
        (
            parse_quote!(
                pub trait Tr<T: ::p::WIter2>
                where
                    T: Fn(T::Item),
                {
                    fn m(&self)
                    where
                        Rel<T>: Default;
                }
            ),
            "< T : :: p :: WIter2 >",
        ),
        (
            parse_quote!(
                pub trait Tr<T: ::p::Cont>
                where
                    T: Fn(T::Out),
                {
                    fn m(&self)
                    where
                        Rel<T>: Default;
                }
            ),
            "< T : :: p :: Cont >",
        ),
    ];
    for (item, generics) in &formerly_priced {
        let (referrer, copy) = assert_accepted(item, f19_copy_with(common::p_leaker(), item));
        assert_eq!(
            context_text_with_where(&referrer, entry_index(&referrer, "Rel < T >")),
            *generics
        );
        assert!(has_replacement_in_predicate(&copy), "{}", s(&copy));
    }
    // Each is accepted: the entry that a predicate of the copy names has the bound `T: ::p::WIter`
    // only, so no bound of its context names an `Item`. (The parameters are compared: the
    // predicate of a method that names a generic of the method is in the context too.)
    let items: Vec<Item> = vec![
        parse_quote!(
            pub trait Tr<T: ::p::WIter>
            where
                T: Fn(T::Item),
            {
                fn m(&self)
                where
                    Rel<T>: Default;
            }
        ),
        // The projection with a qualified path.
        parse_quote!(
            pub trait Tr<T: ::p::WIter>
            where
                T: Fn(<T as Iterator>::Item),
            {
                fn m(&self)
                where
                    Rel<T>: Default;
            }
        ),
        // The bound as a where predicate.
        parse_quote!(
            pub trait Tr<T>
            where
                T: ::p::WIter,
                T: Fn(T::Item),
            {
                fn m(&self)
                where
                    Rel<T>: Default;
            }
        ),
        parse_quote!(
            pub trait Tr {
                fn m<T: ::p::WIter + Fn(T::Item)>(&self)
                where
                    Rel<T>: Default;
            }
        ),
        parse_quote!(
            impl<T: ::p::WIter> ::p::W2<T, u8>
            where
                T: Fn(T::Item),
            {
                fn m(&self)
                where
                    Rel<T>: Default,
                {
                }
            }
        ),
    ];
    for item in &items {
        let (referrer, copy) = assert_accepted(item, f19_copy_with(common::p_leaker(), item));
        assert!(has_replacement_in_predicate(&copy), "{}", s(&copy));
        assert_eq!(
            context_text(&referrer, entry_index(&referrer, "Rel < T >")),
            "< T : :: p :: WIter >",
            "{}",
            s(item)
        );
    }
    // The projection of another generic is not left out: the bound of `U` is not a bound of `T`.
    let other: Item = parse_quote!(
        pub trait Tr<T: ::p::WIter, U: Fn(T::Item)> {
            fn m(&self)
            where
                Rel<U>: Default;
        }
    );
    assert_unstable(
        &other,
        f19_copy_with(common::p_leaker(), &other),
        "Rel < U >",
        "U : Fn (T :: Item)",
        Blames::Projection("T :: Item"),
    );
}

/// The Repeater's impl of the entry `ix`, for the implementor `__S` (what a macro writes), with
/// the key of `id` and the entry.
fn repeater_impl_item(result: &LeakerResult, ix: usize, id: u64) -> ItemImpl {
    let param: Ident = parse_quote!(__S);
    let r = result
        .repeater_impls(&Projection::new(
            repeater(),
            id,
            ImplSelf::Param(param.clone()),
        ))
        .unwrap_or_else(|e| panic!("{}", e))
        .remove(ix);
    parse_quote!(#r)
}

/// The Repeater key of the entry `n` of the Leaker `id`, as `result` numbers its entries.
fn key_of(result: &LeakerResult, id: u64, n: usize) -> u128 {
    common::key(
        id,
        common::entry_id_at(&result.clone().finish().unwrap(), n),
    )
}

/// The copy of `item` (nested in the scope `context`, held by a `mod`) that a macro makes: each
/// entry is the projection of the Repeater with `id`, with `Self` as the implementor (the own
/// `Self` of the nested item, for the entries for any type; the others are left as written, as the
/// macro gives no closure to `with_nested_self()`). The file has the item.
fn macro_copy(referrer: &Referrer, context: ContextId, item: &Item, id: u64) -> File {
    let mut copy = file_of(vec![parse_quote!(mod w { #item })]);
    let projection = common::projection(parse_quote!(Repeater), id);
    referrer
        .visitor(context, &projection)
        .visit_file_mut(&mut copy);
    match copy.items.remove(0) {
        Item::Mod(ItemMod {
            content: Some((_, items)),
            ..
        }) => file_of(items),
        other => panic!("`{}` is no module", s(&other)),
    }
}

/// The generics of an entry with their where clause, as the Repeater's impl has them.
fn generics_text(generics: &Generics) -> String {
    let where_clause = &generics.where_clause;
    s(&quote!(#generics #where_clause))
}

/// F26 (fixed): a type alias hides the projection that a bound names.
///
/// Trigger: `trait Tr<T: Iterator<Item = u8>> where T: Fn(::p::WIt<T>) { fn m(&self) where Rel<T>:
/// Default; }` with `type WIt<T> = <T as Iterator>::Item;`. The Leaker saw no projection, and the
/// copy is E0277 "expected an `Fn(u8)` closure, found `T`", as in F21 and F25.
///
/// Cause: the Repeater's impl of `Rel<T>` carried every bound of `T`. rustc proves its where clause
/// normalized (`T: Fn(u8)`) while it builds the environment of the method, which has the bound as
/// written. The Leaker cannot tell the alias from a struct (`::p::WFoo<T>`, which compiles), so
/// it cannot reject the alias alone.
///
/// Fix: the context of an entry leaves out a bound of `T` that names `T` in the arguments of its
/// path (`T: Fn(X<T>)`) if another bound of `T` fixes an associated type: an equation, or a trait
/// not of the standard library that does not name `T`. An entry that names `Self` or writes out
/// the self type keeps every bound, and is rejected in a predicate where such a bound may hide a
/// projection (F64).
///
/// Limitation: a leaked type that needs the bound left out (`struct Rel<T: Fn(X<T>)>`) is E0277 in
/// the Leaker's crate; an alias in a bound of another generic (`U: Fn(::p::WIt<T>)`) is not seen
/// (E0277 in the copy).
#[test]
fn f26_alias_hides_projection_in_bound_of_entry_in_predicate() {
    let alias: Item = parse_quote!(
        pub trait Tr<T: Iterator<Item = u8>>
        where
            T: Fn(::p::WIt<T>),
        {
            fn m(&self)
            where
                Rel<T>: Default;
        }
    );
    let (result, referrer, copy) = f19_result_with(common::p_leaker(), &alias).unwrap();
    let ix = entry_index(&referrer, "Rel < T >");
    // The context leaves out `T: Fn(::p::WIt<T>)`: the Repeater's impl has `T: Iterator<Item = u8>`
    // only, and the copy is as written, with the entry in the predicate.
    assert_eq!(
        context_text_with_where(&referrer, ix),
        "< T : Iterator < Item = u8 > >"
    );
    assert_eq!(
        s(&repeater_impl_item(&result, ix, 0)),
        "impl < T : Iterator < Item = u8 > , __S : ? :: core :: marker :: Sized > crate :: \
         Repeater < (T ,) , 0u128 > for __S { type Type = Rel < T > ; }"
    );
    assert!(has_replacement_in_predicate(&copy), "{}", s(&copy));
    assert_eq!(
        s(&copy),
        "pub trait Tr < T : Iterator < Item = u8 > > where T : Fn (:: p :: WIt < T >) , { fn m (& \
         self) where P < 0usize > : Default ; }"
    );
    // The same for the projection written (the Leaker rejected it), the equation in a where
    // predicate, a struct (it compiled before), and a method's generic.
    let same: Vec<Item> = vec![
        parse_quote!(
            pub trait Tr<T: Iterator<Item = u8>>
            where
                T: Fn(T::Item),
            {
                fn m(&self)
                where
                    Rel<T>: Default;
            }
        ),
        parse_quote!(
            pub trait Tr<T: Iterator<Item = u8>>
            where
                T: Fn(<T as Iterator>::Item),
            {
                fn m(&self)
                where
                    Rel<T>: Default;
            }
        ),
        parse_quote!(
            pub trait Tr<T>
            where
                T: Iterator<Item = u8>,
                T: Fn(::p::WIt<T>),
            {
                fn m(&self)
                where
                    Rel<T>: Default;
            }
        ),
        parse_quote!(
            pub trait Tr<T: Iterator<Item = u8>>
            where
                T: Fn(::p::WFoo<T>),
            {
                fn m(&self)
                where
                    Rel<T>: Default;
            }
        ),
        // A bound of a trait whose arguments name `T`, and a projection as an argument.
        parse_quote!(
            pub trait Tr<T: Iterator<Item = u8> + PartialEq<::p::WFoo<T>>> {
                fn m(&self)
                where
                    Rel<T>: Default;
            }
        ),
        parse_quote!(
            pub trait Tr<T: Iterator<Item = u8> + Into<<T as Iterator>::Item>> {
                fn m(&self)
                where
                    Rel<T>: Default;
            }
        ),
        // The method's generic.
        parse_quote!(
            pub trait Tr {
                fn m<T: Iterator<Item = u8> + Fn(::p::WIt<T>)>(&self)
                where
                    Rel<T>: Default;
            }
        ),
    ];
    for item in &same {
        let (referrer, copy) = assert_accepted(item, f19_copy_with(common::p_leaker(), item));
        assert!(has_replacement_in_predicate(&copy), "{}", s(&copy));
        assert_eq!(
            context_text(&referrer, entry_index(&referrer, "Rel < T >")),
            "< T : Iterator < Item = u8 > >",
            "{}",
            s(item)
        );
    }
    // The controls: no equation of the associated type, so the bound stays (`T::Item` is rigid,
    // and so is the alias of it, which compiles); a bound that does not name `T`; the bound that
    // is the equation, even if it names `T`; and a bound of a trait that names `T` itself does not
    // fix an associated type for the other bound (it is a bound of the trait that is not one of the
    // standard library, but it names `T`).
    let kept: Vec<(Item, &str)> = vec![
        (
            parse_quote!(
                pub trait Tr<T: Iterator>
                where
                    T: Fn(::p::WIt<T>),
                {
                    fn m(&self)
                    where
                        Rel<T>: Default;
                }
            ),
            "< T : Fn (:: p :: WIt < T >) + Iterator >",
        ),
        (
            parse_quote!(
                pub trait Tr<T: Iterator<Item = u8> + Fn(u8)> {
                    fn m(&self)
                    where
                        Rel<T>: Default;
                }
            ),
            "< T : Fn (u8) + Iterator < Item = u8 > >",
        ),
        (
            parse_quote!(
                pub trait Tr<T: Iterator<Item = Option<T>> + Fn(::p::WIt<T>)> {
                    fn m(&self, a: Rel<T>);
                }
            ),
            "< T : Iterator < Item = Option < T > > >",
        ),
        (
            parse_quote!(
                pub trait Tr<T: ::p::Tr<::p::WFoo<T>> + Fn(::p::WIt<T>)> {
                    fn m(&self)
                    where
                        Rel<T>: Default;
                }
            ),
            "< T : :: p :: Tr < :: p :: WFoo < T > > + Fn (:: p :: WIt < T >) >",
        ),
    ];
    for (item, generics) in &kept {
        let (referrer, _) = assert_accepted(item, f19_copy_with(common::p_leaker(), item));
        assert_eq!(
            context_text(&referrer, entry_index(&referrer, "Rel < T >")),
            *generics,
            "{}",
            s(item)
        );
    }
    // The self type of the item, or the associated type of its trait, is well-formed only with the
    // bounds that the item declares, which the Repeater's impl needs whole: the context of an
    // entry that depends on `Self` as a type or an implementor keeps them (the property test, seed
    // 1: `struct S<T: Iterator<Item = u8> + Fn(T::Item)>` with the entry `Wrap2<S<T>>` was E0277 in
    // the Leaker's crate).
    let self_type: ItemStruct = parse_quote!(
        pub struct S<T: Iterator<Item = u8>>(pub Wrap2<Self>)
        where
            T: Fn(::p::WIt<T>);
    );
    let mut leaker = common::p_leaker();
    leaker.intern_struct(&self_type);
    let referrer = leaker.result().finish().unwrap();
    assert_eq!(
        context_text_with_where(&referrer, entry_index(&referrer, "Wrap2 < Self >")),
        "< T : Fn (:: p :: WIt < T >) + Iterator < Item = u8 > >"
    );
    // The self type of an impl, written out, is declared with the bounds of the item. The top-level
    // context records the self type then, which the Referrer needs to find the entry (the copy
    // below has both entries replaced).
    let impl_self: ItemImpl = parse_quote!(
        impl<T: Iterator<Item = u8>> S<T>
        where
            T: Fn(::p::WIt<T>),
        {
            fn m(&self, x: Rel<T>) {}
        }
    );
    let mut leaker = common::p_leaker();
    let context = leaker.intern_impl(&impl_self);
    let referrer = leaker.result().finish().unwrap();
    assert_eq!(
        context_text_with_where(&referrer, entry_index(&referrer, "S < T >")),
        "< T : Fn (:: p :: WIt < T >) + Iterator < Item = u8 > >"
    );
    assert_eq!(
        context_text_with_where(&referrer, entry_index(&referrer, "Rel < T >")),
        "< T : Iterator < Item = u8 > >"
    );
    // The copy of the impl, in the scope of the top-level context: both entries are found.
    let mut copy = impl_self.clone();
    common::visitor(&referrer, context, common::placeholder).visit_item_impl_mut(&mut copy);
    assert_eq!(
        s(&copy),
        "impl < T : Iterator < Item = u8 > > P < 0usize > where T : Fn (:: p :: WIt < T >) , { fn m (& self , x : P < 1usize >) { } }"
    );
    let assoc: Item = parse_quote!(
        pub trait Tr<T: Iterator<Item = u8>>
        where
            T: Fn(::p::WIt<T>),
        {
            type A;
            fn m(&self, x: Rel<Self::A>);
        }
    );
    let (referrer, _) = f19_copy_with(common::p_leaker(), &assoc).unwrap();
    assert_eq!(
        context_text_with_where(&referrer, entry_index(&referrer, "Rel < Self :: A >")),
        "< T : Fn (:: p :: WIt < T >) + Iterator < Item = u8 > >"
    );
    // The equation of a method is not in the context of the entry (F21), so the bound stays and
    // the Leaker rejects it where it is written, as before.
    let method_equation: Item = parse_quote!(
        pub trait Tr<T: Iterator>
        where
            T: Fn(T::Item),
        {
            fn m(&self)
            where
                T: Iterator<Item = Rel<T>>;
        }
    );
    assert_unstable(
        &method_equation,
        f19_copy_with(common::p_leaker(), &method_equation),
        "Rel < T >",
        "T : Fn (T :: Item) + Iterator",
        Blames::Projection("T :: Item"),
    );
    // The bound of another generic that names the fixed `Item` stays, and the Leaker rejects it.
    let other: Item = parse_quote!(
        pub trait Tr<T: Iterator<Item = u8>, U: Fn(T::Item)> {
            fn m(&self)
            where
                Rel<U>: Default;
        }
    );
    assert_unstable(
        &other,
        f19_copy_with(common::p_leaker(), &other),
        "Rel < U >",
        "U : Fn (T :: Item)",
        Blames::Projection("T :: Item"),
    );
    // `Self::A` in a predicate of a method of a trait whose bound names the `Item` is no impl for
    // rustc itself (see above): the Leaker rejects the projection written, and the alias too (F64:
    // it cannot tell it from a struct, with which the impl compiles).
    let written_self_assoc: Item = parse_quote!(
        pub trait Tr<T: Iterator<Item = u8>>
        where
            T: Fn(T::Item),
        {
            type A: Clone;
            fn m(&self)
            where
                Self::A: Clone;
        }
    );
    let err = assert_rejected(
        &written_self_assoc,
        f19_copy_with(common::p_leaker(), &written_self_assoc),
        &["`Self::A` is in a bound or where predicate, and a bound of its generics names `T::Item`"],
    );
    assert!(
        all_notes(&err)[0].starts_with(
            "the copy for a concrete implementor resolves `Self::A` by the impl of the trait, \
             which needs the bound `T : Fn (T::Item) + Iterator<Item = u8>` of its generics"
        ),
        "{:?}",
        all_notes(&err)
    );
    let aliased_self_assoc: Item = parse_quote!(
        pub trait Tr<T: Iterator<Item = u8>>
        where
            T: Fn(::p::WIt<T>),
        {
            type A: Clone;
            fn m(&self)
            where
                Self::A: Clone;
        }
    );
    let err = assert_rejected(
        &aliased_self_assoc,
        f19_copy_with(common::p_leaker(), &aliased_self_assoc),
        &[
            "`Self::A` is in a bound or where predicate, and a bound of its generics names `T` in \
           the arguments of `Fn (::p::WIt<T>)`, which may hide a projection",
        ],
    );
    assert!(
        all_notes(&err)[0].starts_with(
            "the copy for a concrete implementor resolves `Self::A` by the impl of the trait, \
             which needs the bound `T : Fn (::p::WIt<T>) + Iterator<Item = u8>` of its generics"
        ),
        "{:?}",
        all_notes(&err)
    );
    // A bound that names the leaked type of its own entry: with the equation, it is left out of
    // the context of that entry (F2, and nothing is left to be a cycle); without it, it is the
    // cycle that F2 rejects.
    let leaked_with_equation: Item = parse_quote!(
        pub trait Tr<T: Iterator<Item = u8> + Into<Rel<T>>> {
            fn m(&self);
        }
    );
    let (referrer, _) = f19_copy_with(common::p_leaker(), &leaked_with_equation).unwrap();
    assert_eq!(
        context_text_with_where(&referrer, entry_index(&referrer, "Rel < T >")),
        "< T : Iterator < Item = u8 > >"
    );
    let leaked: Item = parse_quote!(
        pub trait Tr<T: Into<Rel<T>>> {
            fn m(&self);
        }
    );
    assert_unstable(
        &leaked,
        f19_copy_with(common::p_leaker(), &leaked),
        "Rel < T >",
        "T : Into < Rel < T > >",
        Blames::Leaked,
    );
}

/// Whether a bound of a parameter or a where predicate of the copy names a replacement `P<ix>`.
fn has_replacement_in_predicate(item: &Item) -> bool {
    fn names_replacement(node: &impl template_quote::ToTokens) -> bool {
        let text = s(node);
        let tokens: Vec<&str> = text.split(' ').collect();
        tokens.windows(2).any(|pair| pair == ["P", "<"])
    }
    struct Find(bool);
    impl<'ast> Visit<'ast> for Find {
        fn visit_type_param(&mut self, i: &'ast TypeParam) {
            self.0 |= names_replacement(&i.bounds);
            syn::visit::visit_type_param(self, i)
        }
        fn visit_predicate_type(&mut self, i: &'ast PredicateType) {
            self.0 |= names_replacement(i);
            syn::visit::visit_predicate_type(self, i)
        }
    }
    let mut find = Find(false);
    find.visit_item(item);
    find.0
}

// ── F27 ──────────────────────────────────────────────────────────────────────

/// F27 (fixed): the bounds of an argument-position `impl Trait` were not checked as the bounds of a
/// generic.
///
/// Trigger: `trait Tr<T: Fn(Rel0)> { fn m(&self, x: impl Into<Wrap<T>>); }` (also nested, as in
/// `Vec<impl Into<Wrap<T>>>`). The Repeater's impl for `Wrap<T>` needs `T: Fn(Rel0)`, which the
/// copy has only with `Rel0` rewritten: E0277 "expected an `Fn(Rel0)` closure, found `T`".
///
/// Cause: `x: impl Into<Wrap<T>>` is `x: X` with the bound `X: Into<Wrap<T>>`, the cycle of F2,
/// but the check took a bound inside an `impl Trait` for the bound of a trait object or a path.
///
/// Fix: the bounds of an `impl Trait` anywhere in the type of a function argument are bounds of
/// the item, as those of a generic are, so the item is rejected. A return-position `impl Trait`
/// stays accepted: its bound is proved where the body is.
#[test]
fn f27_argument_impl_trait_bound_names_leaked_type_env_cycle() {
    let accepted: Vec<Item> = vec![
        // The bound of the generic names no leaked type.
        parse_quote!(
            pub trait Tr<T: Fn(u8)> {
                fn m(&self, x: impl Into<Wrap<T>>);
            }
        ),
        // The entry does not use the generic.
        parse_quote!(
            pub trait Tr<T: Fn(Rel0)> {
                fn m(&self, x: impl Into<Rel2>);
            }
        ),
        // Return position.
        parse_quote!(
            pub trait Tr<T: Fn(Rel0)> {
                fn m(&self) -> impl Into<Wrap<T>>;
            }
        ),
        // The bound of the `impl Trait` names no entry that uses a generic with a bound that names
        // a leaked type.
        parse_quote!(
            pub trait Tr<T: Clone> {
                fn m(&self, x: impl Iterator<Item = Wrap<T>>, y: &impl Into<Rel2<T>>);
            }
        ),
        parse_quote!(
            pub trait Tr {
                fn m<T: Clone>(&self, x: impl Into<Wrap<T>>);
            }
        ),
    ];
    for item in &accepted {
        assert_accepted(item, f19_copy_with(common::p_leaker(), item));
    }
    // Each is rejected: no copy of it exists that has both a bound of a generic and a bound of an
    // `impl Trait` that mention replacements.
    let rejected: Vec<Item> = vec![
        // A generic written instead of the `impl Trait`.
        parse_quote!(
            pub trait Tr<T: Fn(Rel0)> {
                fn m<X: Into<Wrap<T>>>(&self, x: X);
            }
        ),
        parse_quote!(
            pub trait Tr<T: Fn(Rel0)> {
                fn m(&self, x: impl Into<Wrap<T>>);
            }
        ),
        parse_quote!(
            pub trait Tr<T: Fn(Rel0)> {
                fn m(&self, x: impl Iterator<Item: Into<Wrap<T>>>);
            }
        ),
        parse_quote!(
            pub trait Tr {
                fn m<T: Fn(Rel0)>(&self, x: impl Into<Wrap<T>>);
            }
        ),
        // Nested in the type of the argument, and in an equation of the bound.
        parse_quote!(
            pub trait Tr<T: Fn(Rel0)> {
                fn m(&self, x: Vec<impl Into<Wrap<T>>>);
            }
        ),
        parse_quote!(
            pub trait Tr {
                fn m<T: Fn(Rel0)>(&self, x: &impl Into<Wrap<T>>);
            }
        ),
        parse_quote!(
            pub trait Tr<T: Fn(Rel0)> {
                fn m(&self, x: impl Iterator<Item = Wrap<T>>);
            }
        ),
        parse_quote!(
            pub trait Tr<T: Fn(Rel0)> {
                fn m(&self, x: impl Iterator<Item = impl Into<Wrap<T>>>);
            }
        ),
        // A method of an impl.
        parse_quote!(
            impl<T: Fn(Rel0)> ::p::W2<T, u8> {
                fn m(&self, x: impl Into<Wrap<T>>) {}
            }
        ),
    ];
    for item in &rejected {
        assert_unstable(
            item,
            f19_copy_with(common::p_leaker(), item),
            "Wrap < T >",
            "T : Fn (Rel0)",
            Blames::Leaked,
        );
    }
}

// ── F28 and F29 ──────────────────────────────────────────────────────────────

/// F28 (fixed): a path that hides a lifetime in the inputs of an elided signature, next to an entry
/// that adds lifetimes.
///
/// Trigger: `trait Tr<'a, T: 'a> { fn f(&self, x: fn(::p::WRef<u8>, Rel<T>) -> &u8); }` with
/// `struct WRef<'a, T: ?Sized>(&'a T)`. The output takes the only input lifetime, which `WRef<u8>`
/// hides; the copy of the entry `Rel<T>` names `'a`, so the inputs have two: E0106 "which one of
/// argument 1's 2 lifetimes it is borrowed from".
///
/// Cause: the Leaker did not see the hidden lifetime (F12 is the same for `&`).
///
/// Fix: an elided output lifetime with no input lifetime that the Leaker sees can only be a hidden
/// one, so it counts as the one elision uses. The holder is leaked whole, or, where nothing can
/// hold it (a bound, a function whose receiver is no reference), the item is rejected with a note
/// that a path hides the lifetime. A hidden lifetime in the output (`fn(&Rel<T>) -> WRef<u8>`) is
/// still not seen (E0106).
#[test]
fn f28_hidden_lifetime_in_elided_signature_with_generics() {
    // What compiles, and stays accepted.
    let controls: Vec<(ItemTrait, &str)> = vec![
        // The lifetime is written: the holder is the entry.
        (
            parse_quote!(
                pub trait Tr1<'a, T: 'a> {
                    fn f(&self, x: fn(::p::WRef<'_, u8>, Rel<T>) -> &u8);
                }
            ),
            "fn (:: p :: WRef < '_ , u8 > , Rel < T >) -> & u8",
        ),
        // The entry adds no lifetime.
        (
            parse_quote!(
                pub trait Tr5<T> {
                    fn f(&self, x: fn(::p::WRef<u8>, Rel<T>) -> &u8);
                }
            ),
            "Rel < T >",
        ),
        // The output does not rely on elision.
        (
            parse_quote!(
                pub trait Tr3<'a, T: 'a> {
                    fn f(&self, x: fn(::p::WRef<u8>, Rel<T>) -> Rel<T>);
                }
            ),
            "Rel < T >",
        ),
        // The hidden lifetime is in another signature than the entry.
        (
            parse_quote!(
                pub trait Tr7<'a, T: 'a> {
                    fn f(&self, x: fn(::p::WRef<u8>) -> &u8, y: Rel<T>);
                }
            ),
            "Rel < T >",
        ),
    ];
    for (item, holder) in &controls {
        let (entries, _) = assert_accepted(item, f12_copy_with(common::p_leaker(), item));
        assert_eq!(entries, [*holder], "{}", s(item));
    }
    // The holder is leaked whole: its signature keeps the hidden lifetime where it is written.
    let items: Vec<(ItemTrait, &str)> = vec![
        (
            parse_quote!(
                pub trait Tr<'a, T: 'a> {
                    fn f(&self, x: fn(::p::WRef<u8>, Rel<T>) -> &u8);
                }
            ),
            "fn (:: p :: WRef < u8 > , Rel < T >) -> & u8",
        ),
        (
            parse_quote!(
                pub trait Tr<'a, T: 'a> {
                    fn f(&self, x: Box<dyn Fn(::p::WRef<u8>, Rel<T>) -> &u8>);
                }
            ),
            "dyn Fn (:: p :: WRef < u8 > , Rel < T >) -> & u8",
        ),
        (
            parse_quote!(
                pub trait Tr<'a, T: 'a> {
                    fn f(&self, x: fn(::core::slice::Iter<u8>, Rel<T>) -> &u8);
                }
            ),
            "fn (:: core :: slice :: Iter < u8 > , Rel < T >) -> & u8",
        ),
        (
            parse_quote!(
                pub trait Tr<'a, T: 'a> {
                    fn f(&self, x: fn(::p::WRef<Rel<T>>) -> &u8);
                }
            ),
            "fn (:: p :: WRef < Rel < T > >) -> & u8",
        ),
    ];
    for (item, holder) in &items {
        let (entries, copy) = assert_accepted(item, f12_copy_with(common::p_leaker(), item));
        assert_eq!(entries, [*holder], "{}", s(item));
        let signatures = signatures_of_trait(&copy);
        assert_eq!(
            signatures.count,
            0,
            "`{}`: the entries {:?} leave a signature in the copy, whose elided output has the hidden lifetime and the one of `Rel<T>`: {}",
            s(item),
            entries,
            s(&copy)
        );
    }
    // Where nothing can hold the signature (a bound, a function with no receiver that is a
    // reference), the item is rejected, with a note that says the lifetime is hidden.
    let rejected: Vec<ItemTrait> = vec![
        parse_quote!(
            pub trait Tr<'a, T: 'a, F: Fn(::p::WRef<u8>, Rel<T>) -> &u8> {}
        ),
        parse_quote!(
            pub trait Tr<'a, T: 'a> {
                fn f<F: Fn(::p::WRef<u8>, Rel<T>) -> &u8>(&self, f: F);
            }
        ),
        parse_quote!(
            pub trait Tr<'a, T: 'a> {
                fn f(x: ::p::WRef<u8>, y: Rel<T>) -> &u8;
            }
        ),
        parse_quote!(
            pub trait Tr<'a, T: 'a> {
                fn f(self, x: ::core::slice::Iter<u8>, y: Rel<T>) -> &u8;
            }
        ),
    ];
    for item in &rejected {
        let err = assert_rejected(
            item,
            f12_copy_with(common::p_leaker(), item),
            &["use absolute path: 'Rel<T>'"],
        );
        // The note says that the lifetime is hidden by a path, and the help how to write it.
        assert_eq!(
            all_notes(&err),
            [
                "the output `&u8` has an elided lifetime, which elision takes from the only \
                 lifetime of the inputs, which a path hides (the syntax does not show it), but a \
                 leaked type in the inputs has `'a` among its generics, and the copy names it in \
                 the arguments of its projection, so the inputs have more than one lifetime \
                 (E0106)",
                "it cannot be leaked as a whole because of `&u8`"
            ],
            "{}",
            s(item)
        );
        let helps = all_helps(&err);
        assert_eq!(helps.len(), 1, "{}", s(item));
        assert!(
            helps[0].starts_with(
                "write the lifetime that the path hides (`Ref<'x, u8>`), and name it in the \
                 output, "
            ) && helps[0].ends_with("or write `Rel<T>` with an absolute path"),
            "{}: {}",
            s(item),
            helps[0]
        );
    }
    // A method with a receiver that is a reference takes the lifetime from it: nothing is hidden.
    let accepted: ItemTrait = parse_quote!(
        pub trait Tr<'a, T: 'a> {
            fn f(&self, x: ::p::WRef<u8>, y: Rel<T>) -> &u8;
        }
    );
    let (entries, _) = f12_copy_with(common::p_leaker(), &accepted).unwrap();
    assert_eq!(entries, ["Rel < T >"]);
}

/// F29 (limitation): an entry that hides a lifetime.
///
/// Trigger: `trait Tr0 { fn f(&self, x: fn(RelRef<u8>) -> &u8); }` (or `x: RelRef<u8>`) with
/// `struct RelRef<'a, T>(&'a T)` in the Leaker's crate. The entry `RelRef<u8>` has no generics,
/// and its Repeater's impl (`type Type = RelRef<u8>;`) cannot elide the lifetime: E0106 "missing
/// lifetime specifier" in the Leaker's crate.
///
/// Cause: whether `RelRef` has a lifetime is in its definition, which the Leaker does not see.
///
/// Limitation: the error is not silent. The way out is to name the lifetime (`RelRef<'a, u8>`);
/// `RelRef<'_, u8>` is rejected, as `'_` cannot be leaked whole.
#[test]
fn f29_entry_hides_lifetime() {
    for item in [
        parse_quote!(
            pub trait Tr0 {
                fn f(&self, x: fn(RelRef<u8>) -> &u8);
            }
        ),
        parse_quote!(
            pub trait Tr1 {
                fn f(&self, x: RelRef<u8>);
            }
        ),
    ] {
        let item: ItemTrait = item;
        let mut leaker = common::std_leaker();
        leaker.intern_trait(&item);
        let referrer = leaker.result().finish().unwrap();
        let ix = entry_index(&referrer, "RelRef < u8 >");
        assert!(
            common::entry(&referrer, ix).generics.params.is_empty(),
            "`{}`: the entry has the generics `{}`",
            s(&item),
            context_text(&referrer, ix)
        );
    }
    let mut leaker = common::std_leaker();
    leaker.intern_trait(&parse_quote!(
        pub trait Tr2 {
            fn f(&self, x: fn(RelRef<'_, u8>) -> &u8);
        }
    ));
    let result = leaker.result();
    let err = result.errors();
    assert!(
        first_note(&err[0]).contains("because of `'_`"),
        "{}",
        first_note(&err[0])
    );
}

// ── F30 ──────────────────────────────────────────────────────────────────────

/// F30 (fixed): the outlives bound that a signature implies for `&'x T::Item`, with `T` a generic,
/// is in the context of the entry as the where predicate `T::Item: 'x`, not as `T: 'x`.
///
/// Trigger: `trait Tr<'x, T: Iterator> { fn m(&self, x: Rel<&'x T::Item>); }`. The Leaker recorded
/// `T: 'x`, which the impl of the copy for a host type cannot prove: E0309 "the parameter type `T`
/// may not live long enough".
///
/// Cause: the projection was split into its components, but the signature implies `T::Item: 'x`.
///
/// Fix: the context has the predicate `T::Item: 'x`, which the Repeater's impl declares and the
/// copy's signature implies; `T: 'x` only where `T` is named apart (`&'x (T, T::Item)`). The
/// components stay where no predicate fits: a lifetime of a `for<..>` binder, a projection of a
/// type that is no generic, and one that a bound equates (`Item = U` gives `U: 'x`), as rustc
/// normalizes it and does not decompose the result.
#[test]
fn f30_implied_outlives_of_generic_projection_recorded_as_component() {
    let items: Vec<OutlivesCase> = vec![
        // The control: `T: 'x` written, and recorded.
        (
            parse_quote!(
                pub trait Tr<'x, T: Iterator + 'x> {
                    fn m(&self, x: Rel<&'x T::Item>);
                }
            ),
            "Item",
            &["T :: Item : 'x"],
            &[],
            &["T"],
            "(& 'x () , T , & 'x T)",
        ),
        (
            parse_quote!(
                pub trait Tr<'x, T: Iterator> {
                    fn m(&self, x: Rel<&'x T::Item>);
                }
            ),
            "Item",
            &["T :: Item : 'x"],
            &[],
            &[],
            "(& 'x () , T)",
        ),
        (
            parse_quote!(
                pub trait Tr<'x, T: Iterator> {
                    fn m(&self, x: Rel<&'x <T as Iterator>::Item>);
                }
            ),
            "Item",
            &["< T as Iterator > :: Item : 'x"],
            &[],
            &[],
            "(& 'x () , T)",
        ),
        // Written: the same predicate.
        (
            parse_quote!(
                pub trait Tr<'x, T: Iterator>
                where
                    T::Item: 'x,
                {
                    fn m(&self, x: Rel<&'x T::Item>);
                }
            ),
            "Item",
            &["T :: Item : 'x"],
            &[],
            &[],
            "(& 'x () , T)",
        ),
        // The Leaker does not see the bounds of the associated type (`type A: 'static`).
        (
            parse_quote!(
                pub trait Tr<'x, T: ::p::WTrA> {
                    fn m(&self, x: Rel<&'x <T as ::p::WTrA>::A>);
                }
            ),
            "WTrA > :: A",
            &["< T as :: p :: WTrA > :: A : 'x"],
            &[],
            &[],
            "(& 'x () , T)",
        ),
        // A generic of the method.
        (
            parse_quote!(
                pub trait Tr {
                    fn m<'x, I: Iterator>(&self, x: Rel<&'x I::Item>);
                }
            ),
            "Item",
            &["I :: Item : 'x"],
            &[],
            &[],
            "(& 'x () , I)",
        ),
        // The arguments of the trait, with lifetimes of the trait, are not bounded.
        (
            parse_quote!(
                pub trait Tr<'x, 'b, T: ::p::Tr<'b, U>, U> {
                    fn m(&self, x: Rel<&'x <T as ::p::Tr<'b, U>>::A>);
                }
            ),
            "Tr < 'b , U > > :: A",
            &["< T as :: p :: Tr < 'b , U > > :: A : 'x"],
            &[],
            &[],
            "(& 'b () , & 'x () , T , U)",
        ),
        // A generic associated type, and a projection of a projection.
        (
            parse_quote!(
                pub trait Tr<'x, 'b, T: ::p::Gat> {
                    fn m(&self, x: Rel<&'x T::A<'b>>);
                }
            ),
            "A < 'b >",
            &["T :: A < 'b > : 'x"],
            &[],
            &[],
            "(& 'b () , & 'x () , T)",
        ),
        (
            parse_quote!(
                pub trait Tr<'x, T: ::p::Tr2>
                where
                    T::A: ::p::Tr3,
                {
                    fn m(&self, x: Rel<&'x <T::A as ::p::Tr3>::B>);
                }
            ),
            "Tr3 > :: B",
            &[
                "< T :: A as :: p :: Tr3 > :: B : 'x",
                "T :: A : :: p :: Tr3",
            ],
            &[],
            &[],
            "(& 'x () , T)",
        ),
        // Around the reference: `&'x Rel<T::Item>` is the entry, with `'x` as its own.
        (
            parse_quote!(
                pub trait Tr<'x, T: Iterator> {
                    fn m(&self, x: &'x Rel<T::Item>);
                }
            ),
            "& 'x Rel",
            &["T :: Item : 'x"],
            &[],
            &[],
            "(& 'x () , T)",
        ),
        // `T` is mentioned apart from the projection: `T: 'x` is required, and recorded too.
        (
            parse_quote!(
                pub trait Tr<'x, T: Iterator> {
                    fn m(&self, x: Rel<&'x (T, T::Item)>);
                }
            ),
            "Item",
            &["T :: Item : 'x"],
            &[],
            &["T"],
            "(& 'x () , T , & 'x T)",
        ),
        // The bounds equate the associated type: what it equals is mentioned.
        (
            parse_quote!(
                pub trait Tr<'x, T: Iterator<Item = (::p::B1<T>, u8)>> {
                    fn m(&self, x: Rel<&'x T::Item>);
                }
            ),
            "Item",
            &[],
            &[],
            &["T"],
            "(& 'x () , T , & 'x T)",
        ),
        (
            parse_quote!(
                pub trait Tr<'x, T: Iterator<Item = u8>> {
                    fn m(&self, x: Rel<&'x <T as Iterator>::Item>);
                }
            ),
            "Item",
            &[],
            &[],
            &[],
            "(& 'x () , T)",
        ),
        // The bound equates the associated type to another generic: that one outlives.
        (
            parse_quote!(
                pub trait Tr<'x, T: Iterator<Item = U>, U> {
                    fn m(&self, x: Rel<&'x T::Item>);
                }
            ),
            "Item",
            &[],
            &[],
            &["U"],
            "(& 'x () , T , U , & 'x U)",
        ),
        // Not a projection of a generic: `<Vec<T> as IntoIterator>::Item` is `T`.
        (
            parse_quote!(
                pub trait Tr<'x, T> {
                    fn m(&self, x: Rel<&'x <Vec<T> as IntoIterator>::Item>);
                }
            ),
            "IntoIterator",
            &[],
            &[],
            &["T"],
            "(& 'x () , T , & 'x T)",
        ),
        // The lifetime of a binder is not a generic of the context: no predicate can be written.
        (
            parse_quote!(
                pub trait Tr<'x, T: ::p::Tr<'static, ()>> {
                    fn m(&self, x: Rel<&'x for<'z> fn(<T as ::p::Tr<'z, ()>>::A, &'z u8)>);
                }
            ),
            "fn",
            &[],
            &[],
            &["T"],
            "(& 'x () , T , & 'x T)",
        ),
    ];
    for case in &items {
        check_outlives_case(case);
    }
    // An impl, whose `Self` is a type.
    let item: ItemImpl = parse_quote!(
        impl<'x, T: Iterator> ::p::Tr<'x> for W<T> {
            fn m(&self, x: Rel<&'x T::Item>) {}
        }
    );
    let mut leaker = common::std_leaker();
    leaker.intern_impl(&item);
    let referrer = leaker.result().finish().unwrap();
    let ix = entry_index(&referrer, "Rel < & 'x T :: Item >");
    assert_eq!(
        context_text_with_where(&referrer, ix),
        "< 'x , T : Iterator > where T :: Item : 'x"
    );
}

// ── F31 ──────────────────────────────────────────────────────────────────────

/// F31 (limitation): a function pointer or `Fn(..)` inside an input has its own signature only for
/// rustc 1.64 and later.
///
/// Trigger: `trait Tr<'a, T: 'a> { fn f(&self, x: fn(&for<'x> fn(&'x Rel<T>) -> &'x u8) ->
/// &u8); }` (also with `Box<dyn Fn(..)>`). The copy of the entry `Rel<T>` in the inner signature
/// names `'a`. rustc 1.64 and later do not count it for the outer signature, which has one
/// lifetime (the `&`); earlier versions count two and reject the copy (E0106).
///
/// Cause: the Leaker does not know the toolchain that expands its macros, so it follows the
/// current one (F12's rule): `Rel<T>` alone is the entry.
///
/// Limitation: the copy fails on rustc before 1.64 (the MSRV is 1.56). Leaking the outer holder
/// whole, as the Leaker does where the inner signature has no binder (`fn(&fn(&Rel<T>) -> &u8) ->
/// &u8`), would work on all of them at the cost of more entries.
#[test]
fn f31_nested_signature_has_its_own_lifetimes() {
    for item in [
        parse_quote!(
            pub trait Tr<'a, T: 'a> {
                fn f(&self, x: Box<dyn Fn(&Box<dyn for<'x> Fn(&'x Rel<T>) -> &'x u8>) -> &u8>);
            }
        ),
        parse_quote!(
            pub trait Tr<'a, T: 'a> {
                fn f(&self, x: fn(&for<'x> fn(&'x Rel<T>) -> &'x u8) -> &u8);
            }
        ),
    ] {
        let item: ItemTrait = item;
        let (entries, copy) = f12_copy(&item).unwrap();
        // The entry is `Rel<T>`, not the outer signature: the inner signature has its own binder.
        // Both signatures are left in the copy.
        assert_eq!(entries, ["Rel < T >"], "{}", s(&item));
        let signatures = signatures_of_trait(&copy);
        assert_eq!(signatures.count, 2, "{}", s(&copy));
    }
    // Without a binder, the inner signature elides its own lifetimes, and the outer one is the
    // holder: the same rule, not rustc's of any version.
    let unbound: ItemTrait = parse_quote!(
        pub trait Tr<'a, T: 'a> {
            fn f(&self, x: fn(&fn(&Rel<T>) -> &u8) -> &u8);
        }
    );
    let (entries, _) = f12_copy(&unbound).unwrap();
    assert_eq!(entries, ["fn (& fn (& Rel < T >) -> & u8) -> & u8"]);
}

// ── F32 ──────────────────────────────────────────────────────────────────────

/// F32 (fixed): the encoding parenthesized a trait object that needs no parentheses.
///
/// Trigger: a `T: ?Sized` parameter whose argument is a trait object with one bound: its encoding
/// was `(*const (dyn Tr),)`. `*const dyn Tr` means the same, so rustc's `unused_parens` lint says
/// "unnecessary parentheses around type" where the code is compiled with it on.
///
/// Cause: the encoding of an unsized argument parenthesized every trait object, as two bounds need
/// (`*const (dyn Tr + Send)`, else rustc says "ambiguous `+` in a type").
///
/// Fix: the pointee is parenthesized by the rule for a reference, only for a trait object with
/// several bounds; the Leaker and the Referrer share it, so they print the same tokens. The other
/// parentheses that the library adds are needed ("ambiguous `+` in a type", or "return types are
/// denoted using `->`"):
/// `&'a (dyn A + 'a)`, `Fn() -> (&dyn A) + Send` in the bounds of a context, `(&fn(u8)): 'a` as
/// the bounded type of a predicate. The test asserts the exact outputs.
#[test]
fn f32_parentheses_only_where_rustc_needs_them() {
    // The encoding of the arguments of a `T: ?Sized`, at a use site of the entry `Rel<T>`.
    let unsized_trait: ItemTrait = parse_quote!(
        pub trait Tr<T: ?Sized> {
            fn f(&self) -> Rel<T>;
        }
    );
    let mut leaker = common::std_leaker();
    leaker.intern_trait(&unsized_trait);
    let unsized_result = leaker.result();
    let unsized_referrer = unsized_result.clone().finish().unwrap();
    let encode = |arg: Type| -> Type {
        let args = [GenericArgument::Type(arg)];
        let site = common::answers(&unsized_trait.generics.params, &args);
        common::repeater_args(common::entry(&unsized_referrer, 0), site, None).unwrap()
    };
    // The parentheses that are needed are there, and no other.
    for (arg, expected) in [
        (
            parse_quote!(dyn ::p::A + Send),
            "(* const (dyn :: p :: A + Send) ,)",
        ),
        (
            parse_quote!(dyn ::p::A + 'static),
            "(* const (dyn :: p :: A + 'static) ,)",
        ),
        (parse_quote!(str), "(* const str ,)"),
        // One bound, with the parentheses that rustc does not need, or none.
        (parse_quote!(dyn ::p::A), "(* const dyn :: p :: A ,)"),
        (
            parse_quote!(dyn for<'a> ::p::B<'a>),
            "(* const dyn for < 'a > :: p :: B < 'a > ,)",
        ),
        (
            parse_quote!(dyn Fn(u8) -> &'static dyn ::p::A),
            "(* const dyn Fn (u8) -> & 'static dyn :: p :: A ,)",
        ),
    ] {
        assert_eq!(s(&encode(arg)), expected);
    }
    // The parentheses of an argument are the site's own: they stay (the encoder used to drop them).
    assert_eq!(
        s(&encode(parse_quote!((dyn ::p::A)))),
        "(* const (dyn :: p :: A) ,)"
    );
    let with_lifetime: ItemTrait = parse_quote!(
        pub trait Tr<'a, T: ?Sized + 'a> {
            fn f(&self, x: &'a Rel<T>);
        }
    );
    let mut leaker = common::std_leaker();
    leaker.intern_trait(&with_lifetime);
    let with_lifetime_referrer = leaker.result().finish().unwrap();
    let args: Vec<GenericArgument> = vec![
        parse_quote!('x),
        GenericArgument::Type(parse_quote!(dyn ::p::A + 'x)),
    ];
    let site = common::answers(&with_lifetime.generics.params, &args);
    let encoded =
        common::repeater_args(common::entry(&with_lifetime_referrer, 0), site, None).unwrap();
    assert_eq!(
        s(&encoded),
        "(& 'x () , * const (dyn :: p :: A + 'x) , & 'x (dyn :: p :: A + 'x))"
    );
    // The bounds of a context.
    let item: ItemStruct = parse_quote!(
        pub struct S<T: Send>(Rel<T>)
        where
            T: Fn(()) -> &'static dyn ::core::any::Any;
    );
    let mut leaker = common::std_leaker();
    leaker.intern_struct(&item);
    let result = leaker.result();
    let referrer = result.clone().finish().unwrap();
    let generics = impl_generics(&result, entry_index(&referrer, "Rel < T >"));
    assert_eq!(
        s(&generics),
        "< T : Fn (()) -> (& 'static dyn :: core :: any :: Any) + Send >"
    );
    // `Self` replaced by a trait object, a function pointer.
    let tr: ItemTrait = parse_quote!(
        pub trait Tr {
            fn p(&self, a: &Self, b: *const Self)
            where
                &'static Self: Clone;
        }
    );
    let mut leaker = common::std_leaker();
    let context = leaker.intern_trait(&tr);
    let referrer = leaker.result().finish().unwrap();
    let self_tys: Vec<(Type, &str)> = vec![
        (
            parse_quote!(dyn ::p::A + Send),
            "pub trait Tr { fn p (& self , a : & (dyn :: p :: A + Send) , b : * const (dyn :: p :: A \
             + Send)) where & 'static (dyn :: p :: A + Send) : Clone ; }",
        ),
        (
            parse_quote!(dyn ::p::A),
            "pub trait Tr { fn p (& self , a : & dyn :: p :: A , b : * const dyn :: p :: A) where & \
             'static dyn :: p :: A : Clone ; }",
        ),
        (
            parse_quote!(fn(u8)),
            "pub trait Tr { fn p (& self , a : & fn (u8) , b : * const fn (u8)) where (& 'static fn \
             (u8)) : Clone ; }",
        ),
    ];
    for (self_ty, expected) in self_tys {
        let mut copy = Item::Trait(tr.clone());
        common::visitor(&referrer, context, common::placeholder)
            .with_self(self_ty.clone())
            .visit_item_mut(&mut copy);
        assert_eq!(s(&copy), expected, "Self = `{}`", s(&self_ty));
    }
    // The Leaker's encoding of the parameter is the Referrer's of the same argument.
    let param: Ident = parse_quote!(__S);
    let leaker_impl = unsized_result
        .repeater_impls(&Projection::new(
            repeater(),
            7,
            ImplSelf::Param(param.clone()),
        ))
        .unwrap()
        .remove(0);
    let leaker_side = common::impl_args(&leaker_impl);
    assert_eq!(s(&leaker_side), s(&encode(parse_quote!(T))));
    assert_eq!(s(&leaker_side), "(* const T ,)");
}

// ── F33 ──────────────────────────────────────────────────────────────────────

/// F33 (fixed): a where predicate of the context of an entry names the `Item` of an iterator, and a
/// predicate of an inner item equates the `Item` to a leaked type.
///
/// Trigger: `trait Tr<T: Iterator> where Vec<T::Item>: Default { fn m(&self) where T:
/// Iterator<Item = Rel<T>>; }` (also `u8: Into<T::Item>`, `T::Item: Clone`). The Repeater's impl
/// of `Rel<T>` has the predicate, and rustc proves it in the copy while it normalizes the
/// projection that defines `T::Item`: E0275 "overflow evaluating the requirement `<() as
/// Repeater<(T,), 0, 0>>::Type == <T as Iterator>::Item`".
///
/// Cause: it is the cycle of F21 and F25, but the check read only the bounds of a generic and of a
/// predicate on a generic, and took the rest as proved by an impl, though it names the `Item`.
///
/// Fix: the whole predicate counts if it names a projection that a bound or predicate in scope
/// defines (`Self::A` and a projection of a type that an impl defines do not). Limitation: one
/// that compiles (`T: Iterator<Item = u8>` with `Vec<T::Item>: Default`) is rejected too.
#[test]
fn f33_where_predicate_names_item_that_a_predicate_in_scope_defines() {
    let items: Vec<Item> = vec![
        // The finding of the property test.
        parse_quote!(
            pub trait Tr18_1<T: Clone + ::core::fmt::Debug + Iterator>
            where
                R2<[T::Item; 3], B1<T>>: Default,
            {
                fn m<U>(&self, a: U)
                where
                    T: Fn(T::Item) -> u8,
                    T: Iterator<Item = (J1<T>, u8)>;
            }
        ),
        parse_quote!(
            pub trait Tr<T: Iterator>
            where
                Vec<T::Item>: Default,
            {
                fn m(&self)
                where
                    T: Iterator<Item = Rel<T>>;
            }
        ),
        parse_quote!(
            impl<T: Iterator> ::p::W2<T, u8>
            where
                Vec<T::Item>: Default,
            {
                fn m(&self)
                where
                    T: Iterator<Item = Rel<T>>,
                {
                }
            }
        ),
        // The projection written as a qualified path, a bound of the predicate, a higher-ranked one.
        parse_quote!(
            pub trait Tr<T: Iterator>
            where
                Vec<<T as Iterator>::Item>: Default,
            {
                fn m(&self)
                where
                    T: Iterator<Item = Rel<T>>;
            }
        ),
        parse_quote!(
            pub trait Tr<T: Iterator>
            where
                u8: Into<T::Item>,
            {
                fn m(&self)
                where
                    T: Iterator<Item = Rel<T>>;
            }
        ),
        parse_quote!(
            pub trait Tr<T: Iterator>
            where
                for<'a> &'a [T::Item]: Default,
            {
                fn m(&self)
                where
                    T: Iterator<Item = Rel<T>>;
            }
        ),
        // The projection is the bounded type of a predicate on a generic.
        parse_quote!(
            pub trait Tr<T: Iterator>
            where
                T::Item: Clone,
            {
                fn m(&self)
                where
                    T: Iterator<Item = Rel<T>>;
            }
        ),
        // The equation is another one that an entry does not name (rustc: E0277).
        parse_quote!(
            pub trait Tr<T: Iterator<Item = F>, F>
            where
                Vec<T::Item>: Clone,
            {
                fn m(&self)
                where
                    Rel<T>: Default;
            }
        ),
        parse_quote!(
            pub trait Tr<T: Iterator<Item = u8>>
            where
                T::Item: Into<T>,
            {
                fn m(&self)
                where
                    Rel<T>: Default;
            }
        ),
        // The equation is in a bound of the method, the predicate is in the trait.
        parse_quote!(
            pub trait Tr<T: Iterator>
            where
                Vec<T::Item>: Default,
            {
                fn m<X>(&self, a: X)
                where
                    T: Iterator<Item = (Rel<T>, X)>;
            }
        ),
    ];
    let accepted: Vec<Item> = vec![
        // Both predicates are in the method: the context of `Rel<T>` has the generics of the trait.
        parse_quote!(
            pub trait Tr<T: Iterator> {
                fn m(&self)
                where
                    Vec<T::Item>: Default,
                    T: Iterator<Item = Rel<T>>;
            }
        ),
        // Nothing defines the `Item`.
        parse_quote!(
            pub trait Tr<T: Iterator>
            where
                Rel<[T::Item; 3]>: Default,
            {
                fn m(&self);
            }
        ),
        parse_quote!(
            pub trait Tr<T: Iterator>
            where
                Vec<T::Item>: Default,
            {
                fn m(&self)
                where
                    Rel<T>: Default;
            }
        ),
        // The entry of the equation has no generics, or uses another generic.
        parse_quote!(
            pub trait Tr<T: Iterator>
            where
                Vec<T::Item>: Default,
            {
                fn m(&self)
                where
                    T: Iterator<Item = Rel<u8>>;
            }
        ),
        parse_quote!(
            pub trait Tr<T: Iterator, X>
            where
                Vec<T::Item>: Default,
            {
                fn m(&self)
                where
                    T: Iterator<Item = Rel<X>>;
            }
        ),
        // The equation is of a generic of the method, that the predicate of the trait does not name.
        parse_quote!(
            pub trait Tr<T: Iterator> {
                fn m<U>(&self)
                where
                    U: Iterator<Item = Rel<T>>,
                    Vec<U::Item>: Default;
            }
        ),
        // The projection of a type that an impl defines is proved by that impl, whatever defines the
        // `Item` of `T`.
        parse_quote!(
            pub trait Tr<T: Iterator>
            where
                Vec<(T, <Vec<T> as IntoIterator>::Item)>: Default,
            {
                fn m(&self)
                where
                    T: Iterator<Item = Rel<T>>;
            }
        ),
    ];
    let price: Vec<Item> = vec![
        // `Item = u8` and `Item = F` compile with this predicate, and the Leaker does not tell.
        parse_quote!(
            pub trait Tr<T: Iterator<Item = u8>>
            where
                Vec<T::Item>: Default,
            {
                fn m(&self)
                where
                    Rel<T>: Default;
            }
        ),
        parse_quote!(
            pub trait Tr<T: Iterator<Item = F>, F>
            where
                Vec<T::Item>: Default,
            {
                fn m(&self)
                where
                    Rel<T>: Default;
            }
        ),
        // A supertrait of the bound may define the `Item`.
        parse_quote!(
            pub trait Tr<T: ::p::WIter>
            where
                Vec<T::Item>: Default,
            {
                fn m(&self)
                where
                    Rel<T>: Default;
            }
        ),
    ];
    // Each is rejected: no copy of it exists whose predicate names such an entry.
    let blamed = [
        (
            "J1 < T >",
            "R2 < [T :: Item ; 3] , B1 < T > > : Default",
            Blames::Projection("T :: Item"),
        ),
        (
            "Rel < T >",
            "Vec < T :: Item > : Default",
            Blames::Projection("T :: Item"),
        ),
        (
            "Rel < T >",
            "Vec < T :: Item > : Default",
            Blames::Projection("T :: Item"),
        ),
        (
            "Rel < T >",
            "Vec < < T as Iterator > :: Item > : Default",
            Blames::Projection("< T as Iterator > :: Item"),
        ),
        (
            "Rel < T >",
            "u8 : Into < T :: Item >",
            Blames::Projection("T :: Item"),
        ),
        (
            "Rel < T >",
            "for < 'a > & 'a [T :: Item] : Default",
            Blames::Projection("T :: Item"),
        ),
        (
            "Rel < T >",
            "T :: Item : Clone",
            Blames::Projection("T :: Item"),
        ),
        (
            "Rel < T >",
            "Vec < T :: Item > : Clone",
            Blames::Projection("T :: Item"),
        ),
        (
            "Rel < T >",
            "T :: Item : Into < T >",
            Blames::Projection("T :: Item"),
        ),
        (
            "Rel < T >",
            "Vec < T :: Item > : Default",
            Blames::Projection("T :: Item"),
        ),
    ];
    assert_eq!(items.len(), blamed.len());
    for (item, (ty, bound, blames)) in items.iter().zip(blamed) {
        assert_unstable(
            item,
            f19_copy_with(common::p_leaker(), item),
            ty,
            bound,
            blames,
        );
    }
    for item in &accepted {
        assert_accepted(item, f19_copy_with(common::p_leaker(), item));
    }
    // These are rejected (this is the price of F33): all three for the same predicate.
    for item in &price {
        assert_unstable(
            item,
            f19_copy_with(common::p_leaker(), item),
            "Rel < T >",
            "Vec < T :: Item > : Default",
            Blames::Projection("T :: Item"),
        );
    }
}

// ── F34 ──────────────────────────────────────────────────────────────────────

/// A case of F34 and F35: the trait, a text of the entry, the where predicates of its context, the
/// lifetimes that `Self` outlives, the generics that `: 'x` bounds, the encoded arguments.
type OutlivesCase<'a> = (
    ItemTrait,
    &'a str,
    &'a [&'a str],
    &'a [&'a str],
    &'a [&'a str],
    &'a str,
);

/// Checks the case: the context of the entry has the where predicates listed, the lifetimes that
/// `Self` outlives, the generics that `: 'x` bounds (those listed and no other), and the encoded
/// arguments; the Referrer replaces the entry in a copy of the trait and in a copy where `Self` is
/// the host. Returns the Referrer and the context of the trait.
fn check_outlives_case(case: &OutlivesCase) -> (Referrer, ContextId) {
    let (item, needle, expected, self_expected, outlived, encoded) = case;
    let mut leaker = common::std_leaker();
    let context = leaker.intern_trait(item);
    let referrer = assert_accepted(item, leaker.result().finish());
    let ix = common::types(&referrer)
        .position(|t| s(t).contains(needle))
        .unwrap_or_else(|| panic!("`{}`: no entry with `{}`", s(item), needle));
    let generics = &common::entry(&referrer, ix).generics;
    let mut predicates: Vec<String> = generics
        .where_clause
        .iter()
        .flat_map(|w| &w.predicates)
        .map(s)
        .collect();
    predicates.sort();
    let mut expected: Vec<String> = expected.iter().map(|p| p.to_string()).collect();
    expected.sort();
    assert_eq!(
        predicates,
        expected,
        "`{}`: the where predicates of the context (generics `{}`)",
        s(item),
        s(generics)
    );
    let self_outlives = common::entry(&referrer, ix).self_outlives.as_slice();
    assert_eq!(
        self_outlives
            .iter()
            .map(|l| l.to_string())
            .collect::<Vec<_>>(),
        self_expected.to_vec(),
        "`{}`: `Self: 'x`",
        s(item)
    );
    // The generics that `: 'x` bounds: those listed and no other.
    for param in &generics.params {
        let name = match param {
            GenericParam::Lifetime(l) => l.lifetime.to_string(),
            GenericParam::Type(t) => t.ident.to_string(),
            GenericParam::Const(c) => c.ident.to_string(),
        };
        if name == "'x" {
            continue;
        }
        assert_eq!(
            generics_say_outlives(generics, &name, "'x"),
            outlived.contains(&name.as_str()),
            "`{}`: `{}: 'x` in the context (generics `{}`)",
            s(item),
            name,
            s(generics)
        );
    }
    let arguments =
        common::repeater_args(common::entry(&referrer, ix), common::identity, None).unwrap();
    assert_eq!(s(&arguments), *encoded, "`{}`", s(item));
    // The Referrer finds the entry in the copy, which has the signature and no predicate, and
    // in the copy where `Self` is the host.
    let copy = rewrite(&referrer, context, Item::Trait(item.clone()));
    assert!(
        !copy.contains("Rel <"),
        "`{}`: not replaced in the copy: {}",
        s(item),
        copy
    );
    let mut with_self = Item::Trait(item.clone());
    common::visitor(&referrer, context, common::placeholder)
        .with_self(parse_quote!(Host))
        .visit_item_mut(&mut with_self);
    assert!(
        !s(&with_self).contains("Rel <"),
        "`{}`: not replaced in the copy with `Self` = `Host`: {}",
        s(item),
        s(&with_self)
    );
    (referrer, context)
}

/// F34 (fixed): the outlives bound that a signature implies for `&'x P`, with `P` a projection of
/// `Self` other than the bare `Self::A`, was recorded as the components of `P`.
///
/// Trigger: `trait Tr<'a, T> { type A0; fn m<'x>(&'x self, a: Rel<&'x <Self as Tr<'a, T>>::A0>)
/// where Self: 'x; }` (also a GAT `Self::A<'b>`, `<Self::A0 as Tr2>::B`). The context had `'a:
/// 'x`, `T: 'x` and `Self: 'x`, which the impl of the copy for a host type cannot prove: E0491 "in
/// type `&'x &'a ()`, reference has a longer lifetime than the data it references".
///
/// Cause: only the bare `Self::A` (F16) and a projection of a generic (F30) were kept whole.
///
/// Fix: any projection of `Self` is a unit: the context has the predicate `P: 'x`, no component,
/// and `Self: 'x` only where it is written or `Self` is named apart. Where a bound equates it
/// (`where Self: Iterator<Item = (Vec<T>, u8)>`), what it equals outlives instead (`T: 'x`), as
/// rustc does not decompose the normalized predicate (E0309). A supertrait that equates it is not
/// read (a limitation): E0309 in the Leaker's crate until the trait writes `T: 'x`.
#[test]
fn f34_implied_outlives_of_self_projection_forms_recorded_as_components() {
    let items: Vec<OutlivesCase> = vec![
        // Qualified, with the arguments of the trait. `Self: 'x` is written.
        (
            parse_quote!(
                pub trait Tr<'a, T> {
                    type A0;
                    fn m<'x>(&'x self, a: Rel<&'x <Self as Tr<'a, T>>::A0>)
                    where
                        Self: 'x;
                }
            ),
            "A0",
            &["< Self as Tr < 'a , T > > :: A0 : 'x"],
            &["'x"],
            &[],
            "(& 'a () , & 'x () , T , & 'x Self)",
        ),
        // A lifetime of the trait, and no `Self: 'x`: the projection brings none.
        (
            parse_quote!(
                pub trait Tr<'x, T> {
                    type A0;
                    fn m(&self, a: Rel<&'x <Self as Tr<'x, T>>::A0>);
                }
            ),
            "A0",
            &["< Self as Tr < 'x , T > > :: A0 : 'x"],
            &[],
            &[],
            "(& 'x () , T)",
        ),
        // No arguments: the same predicate, and no `Self: 'x`.
        (
            parse_quote!(
                pub trait Tr {
                    type A0;
                    fn m<'x>(&'x self, a: Rel<&'x <Self as Tr>::A0>);
                }
            ),
            "A0",
            &["< Self as Tr > :: A0 : 'x"],
            &[],
            &[],
            "(& 'x () ,)",
        ),
        // A generic associated type.
        (
            parse_quote!(
                pub trait Tr<'b> {
                    type A<'c>;
                    fn m<'x>(&'x self, a: Rel<&'x Self::A<'b>>)
                    where
                        Self: 'x;
                }
            ),
            "A < 'b >",
            &["Self :: A < 'b > : 'x"],
            &["'x"],
            &[],
            "(& 'b () , & 'x () , & 'x Self)",
        ),
        // The generic argument is part of the path: no `T: 'x`.
        (
            parse_quote!(
                pub trait Tr<'x, T> {
                    type G<U>;
                    fn m(&self, a: Rel<&'x Self::G<T>>);
                }
            ),
            "G < T >",
            &["Self :: G < T > : 'x"],
            &[],
            &[],
            "(& 'x () , T)",
        ),
        // The trait of the path is another one, with lifetimes: the where clause says it is one.
        (
            parse_quote!(
                pub trait Tr<'x, 'b, U>
                where
                    Self: ::p::Tr2<'b, U>,
                {
                    fn m(&self, a: Rel<&'x <Self as ::p::Tr2<'b, U>>::B>);
                }
            ),
            "Tr2 < 'b , U > > :: B",
            &[
                "< Self as :: p :: Tr2 < 'b , U > > :: B : 'x",
                "Self : :: p :: Tr2 < 'b , U >",
            ],
            &[],
            &[],
            "(& 'b () , & 'x () , U)",
        ),
        // A projection of a projection.
        (
            parse_quote!(
                pub trait Tr<'x> {
                    type A0: ::p::Tr3;
                    fn m(&self, a: Rel<&'x <Self::A0 as ::p::Tr3>::B>);
                }
            ),
            "Tr3 > :: B",
            &["< Self :: A0 as :: p :: Tr3 > :: B : 'x"],
            &[],
            &[],
            "(& 'x () ,)",
        ),
        // Around the reference: `&'x Rel<..>` is the entry.
        (
            parse_quote!(
                pub trait Tr<'x, T> {
                    type A0;
                    fn m(&self, a: &'x Rel<<Self as Tr<'x, T>>::A0>);
                }
            ),
            "& 'x Rel",
            &["< Self as Tr < 'x , T > > :: A0 : 'x"],
            &[],
            &[],
            "(& 'x () , T)",
        ),
        // `Self` is mentioned apart from the projection: `Self: 'x` is required, and recorded.
        (
            parse_quote!(
                pub trait Tr<'x, T> {
                    type A0;
                    fn m(&self, a: Rel<&'x (Self, <Self as Tr<'x, T>>::A0)>);
                }
            ),
            "A0",
            &["< Self as Tr < 'x , T > > :: A0 : 'x"],
            &["'x"],
            &[],
            "(& 'x () , T , & 'x Self)",
        ),
        // The bounds equate the associated type: what it equals is mentioned, and no predicate.
        (
            parse_quote!(
                pub trait Tr<'x, T>
                where
                    Self: Iterator<Item = (Vec<T>, u8)>,
                {
                    fn m(&self, a: Rel<&'x Self::Item>);
                }
            ),
            "Item",
            &["Self : Iterator < Item = (Vec < T > , u8) >"],
            &[],
            &["T"],
            "(& 'x () , T , & 'x T)",
        ),
        (
            parse_quote!(
                pub trait Tr<'x, T>
                where
                    Self: Iterator<Item = (Vec<T>, u8)>,
                {
                    fn m(&self, a: Rel<&'x <Self as Iterator>::Item>);
                }
            ),
            "Item",
            &["Self : Iterator < Item = (Vec < T > , u8) >"],
            &[],
            &["T"],
            "(& 'x () , T , & 'x T)",
        ),
        (
            parse_quote!(
                pub trait Tr<'x>
                where
                    Self: Iterator<Item = u8>,
                {
                    fn m(&self, a: Rel<&'x Self::Item>);
                }
            ),
            "Item",
            &["Self : Iterator < Item = u8 >"],
            &[],
            &[],
            "(& 'x () ,)",
        ),
        // It equals a type that names `Self`.
        (
            parse_quote!(
                pub trait Tr<'x>
                where
                    Self: Iterator<Item = Box<Self>>,
                {
                    fn m(&self, a: Rel<&'x Self::Item>);
                }
            ),
            "Item",
            &["Self : Iterator < Item = Box < Self > >"],
            &["'x"],
            &[],
            "(& 'x () , & 'x Self)",
        ),
        // The lifetime of a binder is not a generic of the context: no predicate can be written.
        (
            parse_quote!(
                pub trait Tr<'x> {
                    fn m(&self, a: Rel<&'x for<'z> fn(<Self as ::p::Tr2<'z>>::A, &'z u8)>);
                }
            ),
            "fn",
            &[],
            &["'x"],
            &[],
            "(& 'x () , & 'x Self)",
        ),
        // A supertrait that equates the associated type is not read (a limitation, see above):
        // the predicate, as for the bare path, and the Leaker's crate is E0309 until `T: 'x` is
        // written.
        (
            parse_quote!(
                pub trait Tr<'x, T>: Iterator<Item = (Vec<T>, u8)> {
                    fn m(&self, a: Rel<&'x Self::Item>);
                }
            ),
            "Item",
            &["Self :: Item : 'x"],
            &[],
            &[],
            "(& 'x () , T)",
        ),
    ];
    for case in &items {
        check_outlives_case(case);
    }
    // The control: the bounds written are recorded as before.
    let control: ItemTrait = parse_quote!(
        pub trait Tr<'a, T> {
            type A0;
            fn m<'x>(&'x self, a: Rel<&'x <Self as Tr<'a, T>>::A0>)
            where
                Self: 'x,
                T: 'x,
                'a: 'x;
        }
    );
    let mut leaker = common::std_leaker();
    leaker.intern_trait(&control);
    let referrer = leaker.result().finish().unwrap();
    let ix = entry_index(&referrer, "Rel < & 'x < Self as Tr < 'a , T > > :: A0 >");
    let generics = &common::entry(&referrer, ix).generics;
    assert!(generics_say_outlives(generics, "T", "'x"));
    assert!(generics_say_outlives(generics, "'a", "'x"));
    assert_eq!(
        common::entry(&referrer, ix)
            .self_outlives
            .as_slice()
            .iter()
            .map(|l| l.to_string())
            .collect::<Vec<_>>(),
        vec!["'x".to_string()]
    );
}

// ── F35 ──────────────────────────────────────────────────────────────────────

/// F35 (fixed): a generic associated type whose where clause says `Self: 'b` makes `Self::G<'x>`
/// well-formed only if `Self: 'x`, which F34 stopped recording for the projection.
///
/// Trigger: `trait Tr { type G<'b> where Self: 'b; fn m<'x>(&'x self, a: Rel<&'x Self::G<'x>>) ->
/// u8; }`. The signature implies `Self: 'x` through the where clause of `G`, but the Repeater's
/// impl, with only the predicate `(<__S>::G<'x>): 'x`, is E0309 on `__S`.
///
/// Cause: the Leaker did not read the where clause of `G`.
///
/// Fix: the Leaker reads the where clauses and bounds of the GATs of the trait it analyzes; a
/// projection that names one adds their outlives relations, with the parameters substituted by
/// the projection's arguments (`Self: 'x`, `T: 'x`, `'a: 'x`, `Self::A: 'x`), next to F34's
/// predicate. The Referrer computes the same context, also from a serialized Referrer.
///
/// Limitation: a GAT that the Leaker cannot see (of a supertrait, or of a generic's trait) adds
/// nothing: the method must write the bound (`where Self: 'x`).
#[test]
fn f35_gat_where_clause_of_self_projection_not_recorded() {
    let items: Vec<OutlivesCase> = vec![
        // The trigger.
        (
            parse_quote!(
                pub trait Tr {
                    type G<'b>
                    where
                        Self: 'b;
                    fn m<'x>(&'x self, a: Rel<&'x Self::G<'x>>) -> u8;
                }
            ),
            "G",
            &["Self :: G < 'x > : 'x"],
            &["'x"],
            &[],
            "(& 'x () , & 'x Self)",
        ),
        // Qualified.
        (
            parse_quote!(
                pub trait Tr {
                    type G<'b>
                    where
                        Self: 'b;
                    fn m<'x>(&'x self, a: Rel<&'x <Self as Tr>::G<'x>>) -> u8;
                }
            ),
            "G",
            &["< Self as Tr > :: G < 'x > : 'x"],
            &["'x"],
            &[],
            "(& 'x () , & 'x Self)",
        ),
        // No reference around the projection: no predicate, and the where clause of `G` counts.
        (
            parse_quote!(
                pub trait Tr {
                    type G<'b>
                    where
                        Self: 'b;
                    fn m<'x>(&'x self, a: Rel<Self::G<'x>>) -> u8;
                }
            ),
            "G",
            &[],
            &["'x"],
            &[],
            "(& 'x () , & 'x Self)",
        ),
        // The bound of the parameter of `G` (an inline bound is the same as a where predicate).
        (
            parse_quote!(
                pub trait Tr {
                    type G<'b, U: 'b>;
                    fn m<'x, V>(&self, a: Rel<&'x Self::G<'x, V>>) -> u8;
                }
            ),
            "G",
            &["Self :: G < 'x , V > : 'x"],
            &[],
            &["V"],
            "(& 'x () , V , & 'x V)",
        ),
        // A parameter of the trait, with the where clause of `G`.
        (
            parse_quote!(
                pub trait Tr<T> {
                    type G<'b>
                    where
                        T: 'b,
                        Self: 'b;
                    fn m<'x>(&'x self, a: Rel<&'x Self::G<'x>>) -> u8;
                }
            ),
            "G",
            &["Self :: G < 'x > : 'x"],
            &["'x"],
            &["T"],
            "(& 'x () , T , & 'x T , & 'x Self)",
        ),
        // A lifetime of the trait.
        (
            parse_quote!(
                pub trait Tr<'a> {
                    type G<'b>
                    where
                        'a: 'b,
                        Self: 'b;
                    fn m<'x>(&'x self, a: Rel<&'x Self::G<'x>>) -> u8;
                }
            ),
            "G",
            &["Self :: G < 'x > : 'x"],
            &["'x"],
            &["'a"],
            "(& 'a () , & 'x () , & 'x & 'a () , & 'x Self)",
        ),
        // The arguments of the trait in a qualified path take the place of its parameters.
        (
            parse_quote!(
                pub trait Tr<T, U> {
                    type G<'b>
                    where
                        T: 'b;
                    fn m<'x>(&self, a: Rel<<Self as Tr<U, T>>::G<'x>>) -> u8;
                }
            ),
            "G",
            &[],
            &[],
            &["U"],
            "(& 'x () , T , U , & 'x U)",
        ),
        // A projection of `Self` in the where clause of `G`.
        (
            parse_quote!(
                pub trait Tr {
                    type A;
                    type G<'b>
                    where
                        Self::A: 'b,
                        Self: 'b;
                    fn m<'x>(&'x self, a: Rel<&'x Self::G<'x>>) -> u8;
                }
            ),
            "G",
            &["Self :: A : 'x", "Self :: G < 'x > : 'x"],
            &["'x"],
            &[],
            "(& 'x () , & 'x Self)",
        ),
        // A where clause that is not an outlives brings nothing (the method writes it).
        (
            parse_quote!(
                pub trait Tr {
                    type G<'b>
                    where
                        Self: Clone;
                    fn m<'x>(&self, a: Rel<Self::G<'x>>) -> u8
                    where
                        Self: Clone;
                }
            ),
            "G",
            &["Self : Clone"],
            &[],
            &[],
            "(& 'x () ,)",
        ),
        // A generic associated type with no where clause: nothing more than F34 records.
        (
            parse_quote!(
                pub trait Tr {
                    type G<'b>;
                    fn m<'x>(&'x self, a: Rel<&'x Self::G<'x>>) -> u8;
                }
            ),
            "G",
            &["Self :: G < 'x > : 'x"],
            &[],
            &[],
            "(& 'x () ,)",
        ),
        // The lifetime argument is not a lifetime of the context: no bound can be written.
        (
            parse_quote!(
                pub trait Tr {
                    type G<'b>
                    where
                        Self: 'b;
                    fn m<'x>(&'x self, a: Rel<&'x Self::G<'static>>) -> u8
                    where
                        Self: 'static;
                }
            ),
            "G",
            &["Self :: G < 'static > : 'x", "Self : 'static"],
            &[],
            &[],
            "(& 'x () ,)",
        ),
        // The generic associated type of a supertrait, and of a generic's trait: not seen, so the
        // bound is the one that the method writes.
        (
            parse_quote!(
                pub trait Tr: ::p::Ext {
                    fn m<'x>(&'x self, a: Rel<&'x Self::G<'x>>) -> u8;
                }
            ),
            "G",
            &["Self :: G < 'x > : 'x"],
            &[],
            &[],
            "(& 'x () ,)",
        ),
        (
            parse_quote!(
                pub trait Tr<T: ::p::Ext> {
                    fn m<'x>(&self, a: Rel<&'x T::G<'x>>) -> u8;
                }
            ),
            "G",
            &["T :: G < 'x > : 'x"],
            &[],
            &[],
            "(& 'x () , T)",
        ),
    ];
    for case in &items {
        let (referrer, context) = check_outlives_case(case);
        let item = &case.0;
        // The Referrer finds the entry in the signatures alone, after the Referrer is written as
        // tokens and parsed back (the top-level context carries what the copy needs).
        let tokens = quote!(#referrer);
        let parsed: Referrer = syn::parse2(tokens.clone()).unwrap_or_else(|e| {
            panic!(
                "`{}`: the Referrer `{}` does not parse: {}",
                s(item),
                tokens,
                e
            )
        });
        assert_eq!(parsed, referrer, "`{}`: parsed back", s(item));
        for trait_item in &item.items {
            if let TraitItem::Fn(method) = trait_item {
                let mut sig = method.sig.clone();
                common::visitor(&parsed, context, common::placeholder)
                    .visit_signature_mut(&mut sig);
                assert!(
                    !s(&sig).contains("Rel <"),
                    "`{}`: not replaced in the signature alone: {}",
                    s(item),
                    s(&sig)
                );
            }
        }
    }
}

/// What the closure of `Visitor::with_nested_self()` was called with: the `Self` of the nested
/// item, one line for each call.
type NestedCalls = std::rc::Rc<std::cell::RefCell<Vec<String>>>;

/// A visitor for the copy in the scope `context`, with `with_nested_self()`: each entry becomes
/// `P<ix>`, those that depend on the own `Self` of a nested item too, written for the type
/// `Nested` that the closure returns. The calls of the closure are recorded.
fn nested_visitor(referrer: &Referrer, context: ContextId) -> (impl VisitMut + '_, NestedCalls) {
    let calls = NestedCalls::default();
    let log = calls.clone();
    let visitor = common::visitor(referrer, context, common::placeholder).with_nested_self(
        move |requirement| {
            log.borrow_mut().push(match requirement {
                SelfRequirement::Type(ty) => format!("Self = {}", s(ty)),
                SelfRequirement::Bound(path) => format!("Self: {}", s(path)),
                SelfRequirement::Any => "Self any".to_string(),
                _ => unreachable!(),
            });
            Some(parse_quote!(Nested))
        },
    );
    (visitor, calls)
}

fn file_of(items: Vec<Item>) -> File {
    File {
        shebang: None,
        attrs: Vec::new(),
        items,
    }
}

/// Where the nested item is analyzed, and where its copy is: the item is always visited by
/// `enter_generics(<>, |view| view.visit_item(&item))`.
#[derive(Clone, Copy, Debug)]
enum Nest {
    /// The copy is in the body of a method, where an item nested in a scope is.
    EnterGenerics,
    /// The copy is in a `mod`, which declares the items it holds.
    EnterGenericsInMod,
}

const NESTS: [Nest; 2] = [Nest::EnterGenerics, Nest::EnterGenericsInMod];

/// Analyzes `item` nested as `nest` says, rewrites the copy with `nested_visitor()`, and returns
/// the rewritten item and the entries.
fn nested_copy(nest: Nest, item: &Item) -> (Item, Referrer) {
    nested_copy_in(common::std_leaker(), nest, item)
}

/// Like `nested_copy()`, and the calls of the closure of `with_nested_self()` too.
fn nested_copy_logged(nest: Nest, item: &Item) -> (Item, Referrer, Vec<String>) {
    nested_run(common::std_leaker(), nest, item, (false, false), true)
}

/// Like `nested_copy()`, but the visitor has no `with_nested_self()`: the entries for the own type
/// of the nested item are left as written.
fn nested_copy_unhooked(nest: Nest, item: &Item) -> (Item, Referrer) {
    let (copy, referrer, _) = nested_run(common::std_leaker(), nest, item, (false, false), false);
    (copy, referrer)
}

/// Like `nested_copy()`, analyzed by `leaker`.
fn nested_copy_in<A: Allow>(leaker: Leaker<A>, nest: Nest, item: &Item) -> (Item, Referrer) {
    nested_copy_recovering(leaker, nest, item, false, false)
}

/// Like `nested_copy()`, with the `NestedSelfInBounds` errors accepted
/// ([`accept_nested_self()`]).
fn nested_copy_accepting(nest: Nest, item: &Item) -> (Item, Referrer) {
    nested_copy_recovering(common::std_leaker(), nest, item, false, true)
}

/// Like `nested_copy_in()`; with `recover` the errors of the Leaker do not fail the copy:
/// the part that conflicts is copied as written (every error is allowed first). With `accept`
/// the `NestedSelfInBounds` errors are accepted first ([`accept_nested_self()`]).
fn nested_copy_recovering<A: Allow>(
    leaker: Leaker<A>,
    nest: Nest,
    item: &Item,
    recover: bool,
    accept: bool,
) -> (Item, Referrer) {
    let (copy, referrer, _) = nested_run(leaker, nest, item, (recover, accept), true);
    (copy, referrer)
}

/// The copy of `nested_copy_recovering()`, with the calls of the closure of `with_nested_self()`;
/// with `hooked` false the visitor has none.
fn nested_run<A: Allow>(
    mut leaker: Leaker<A>,
    nest: Nest,
    item: &Item,
    (recover, accept): (bool, bool),
    hooked: bool,
) -> (Item, Referrer, Vec<String>) {
    let context = leaker
        .enter_generics(&Generics::default(), |view| view.visit_item(item))
        .0;
    let mut result = leaker.result();
    if accept {
        result = accept_nested_self(result);
    }
    let referrer = if recover {
        result.recovered()
    } else {
        result.finish().unwrap()
    };
    let (mut visitor, calls): (Box<dyn VisitMut + '_>, NestedCalls) = if hooked {
        let (visitor, calls) = nested_visitor(&referrer, context);
        (Box::new(visitor), calls)
    } else {
        let visitor = common::visitor(&referrer, context, common::placeholder);
        (Box::new(visitor), NestedCalls::default())
    };
    let copy = match nest {
        Nest::EnterGenericsInMod => {
            let mut file = file_of(vec![parse_quote!(mod w { #item })]);
            visitor.visit_file_mut(&mut file);
            match file.items.remove(0) {
                Item::Mod(ItemMod {
                    content: Some((_, mut items)),
                    ..
                }) => items.remove(0),
                other => panic!("`{}` is no module", s(&other)),
            }
        }
        Nest::EnterGenerics => {
            let mut host: ItemImpl = parse_quote!(impl Host { fn m() { #item } });
            visitor.visit_item_impl_mut(&mut host);
            match host.items.remove(0) {
                ImplItem::Fn(ImplItemFn { mut block, .. }) => match block.stmts.remove(0) {
                    Stmt::Item(item) => item,
                    other => panic!("`{}` is no item", s(&other)),
                },
                other => panic!("`{}` is no method", s(&other)),
            }
        }
    };
    drop(visitor);
    let calls = calls.take();
    (copy, referrer, calls)
}

/// F36 (fixed): an entry that names the own `Self` of an item nested in a scope was recorded, but
/// the Visitor never found it in the copy.
///
/// Trigger: `struct S<T> { b: Rel<T>, c: Vec<Rel2<Self>> }` visited by `View::visit_item()` (also
/// an enum, a union, a trait or an impl). The copy kept `Rel2<Self>`, which the Referrer's crate
/// cannot resolve (E0412, E0425, E0433).
///
/// Cause: inside a nested item the Visitor skipped every entry that names `Self`, as meant for the
/// `Self` around, which the item hides; so it skipped the item's own `Self` too.
///
/// Fix: the Visitor tracks the `Self` that the Leaker gave each nested item, so it finds the entry,
/// and `Visitor::with_nested_self()` writes it: its closure gets the item's `SelfRequirement` once
/// and returns the type (only the macro knows the Leaker's type for it). The test runs each kind
/// with the copy in a method body and in a `mod`.
#[test]
fn f36_nested_item_self_entry_not_rewritten() {
    struct Case {
        item: Item,
        /// The `Self` that the nested item has, in the line recorded for the closure.
        nested: &'static str,
        /// The copy, with `P<ix>`: the indices of the entries `Rel < T >` and `Rel2 < Self >`.
        expected: fn(usize, usize) -> Item,
    }
    let cases = vec![
        Case {
            item: parse_quote!(
                struct S<T> {
                    b: Rel<T>,
                    c: Vec<Rel2<Self>>,
                }
            ),
            nested: "Self = S < T >",
            expected: |p, n| {
                parse_quote!(
                    struct S<T> {
                        b: P<#p>,
                        c: Vec<P<#n>>,
                    }
                )
            },
        },
        Case {
            item: parse_quote!(
                enum E<T> {
                    A(Vec<Rel2<Self>>),
                    B { b: Rel<T> },
                }
            ),
            nested: "Self = E < T >",
            expected: |p, n| {
                parse_quote!(
                    enum E<T> {
                        A(Vec<P<#n>>),
                        B { b: P<#p> },
                    }
                )
            },
        },
        Case {
            item: parse_quote!(
                union U<T> {
                    a: ::core::mem::ManuallyDrop<Vec<Rel2<Self>>>,
                    b: ::core::mem::ManuallyDrop<Rel<T>>,
                }
            ),
            nested: "Self = U < T >",
            expected: |p, n| {
                parse_quote!(
                    union U<T> {
                        a: ::core::mem::ManuallyDrop<Vec<P<#n>>>,
                        b: ::core::mem::ManuallyDrop<P<#p>>,
                    }
                )
            },
        },
        Case {
            item: parse_quote!(
                trait Tr<T> {
                    fn f(&self, x: Rel<T>) -> Vec<Rel2<Self>>;
                }
            ),
            nested: "Self: Tr < T >",
            expected: |p, n| {
                parse_quote!(
                    trait Tr<T> {
                        fn f(&self, x: P<#p>) -> Vec<P<#n>>;
                    }
                )
            },
        },
        Case {
            item: parse_quote!(
                impl<T> ::p::W for ::p::S<T> {
                    fn f(x: Rel<T>) -> Vec<Rel2<Self>> {
                        loop {}
                    }
                }
            ),
            nested: "Self = :: p :: S < T >",
            expected: |p, n| {
                parse_quote!(
                    impl<T> ::p::W for ::p::S<T> {
                        fn f(x: P<#p>) -> Vec<P<#n>> {
                            loop {}
                        }
                    }
                )
            },
        },
    ];
    for case in &cases {
        for nest in NESTS {
            // `::p` is allowed: `::p::S<T>`, the self type of the impl, is no entry.
            let (copy, referrer, calls) =
                nested_run(common::p_leaker(), nest, &case.item, (false, false), true);
            let (p, n) = (
                entry_index(&referrer, "Rel < T >"),
                entry_index(&referrer, "Rel2 < Self >"),
            );
            let what = format!("`{}` in {:?}", s(&case.item), nest);
            assert_eq!(s(&copy), s(&(case.expected)(p, n)), "{}", what);
            // The closure is asked for the item that has the entry that names its own `Self`, once.
            assert_eq!(calls, [case.nested], "{}", what);
        }
    }
}

/// F36: without `Visitor::with_nested_self()`, the entries that name the own `Self` of a nested
/// item (`Type`, `Bound`) are left as written, and the others are rewritten: the visitor does not
/// know what to write for a `Self` that is not its own.
#[test]
fn f36_without_with_nested_self_the_entry_of_the_own_self_is_left() {
    let item: Item = parse_quote!(
        struct S<T> {
            b: Rel<T>,
            c: Vec<Rel2<Self>>,
        }
    );
    let (copy, referrer) = nested_copy_unhooked(Nest::EnterGenerics, &item);
    let p = entry_index(&referrer, "Rel < T >");
    let expected: Item = parse_quote!(
        struct S<T> {
            b: P<#p>,
            c: Vec<Rel2<Self>>,
        }
    );
    assert_eq!(s(&copy), s(&expected));
    // The same with a trait, whose `Self` is the implementor (`Bound`).
    let tr: Item = parse_quote!(
        trait Tr<T> {
            fn f(&self, x: Rel<T>) -> Vec<Rel2<Self>>;
        }
    );
    let (copy, referrer) = nested_copy_unhooked(Nest::EnterGenerics, &tr);
    let p = entry_index(&referrer, "Rel < T >");
    let expected: Item = parse_quote!(
        trait Tr<T> {
            fn f(&self, x: P<#p>) -> Vec<Rel2<Self>>;
        }
    );
    assert_eq!(s(&copy), s(&expected));
}

/// F36: a closure of `with_nested_self()` that returns `None` leaves the entries of that item as
/// written, as no closure does, and is asked once for each item that has such an entry, not for an
/// item that has none.
#[test]
fn f36_a_closure_that_returns_none_leaves_the_entry_and_is_asked_once_for_each_item() {
    let outer: ItemMod = parse_quote!(
        mod a {
            struct S<T> {
                a: Rel<T>,
                b: Rel2<Self>,
                c: Vec<Rel2<Self>>,
            }
            struct Plain<T>(Rel<T>);
        }
    );
    let mut leaker = common::std_leaker();
    let (context, ()) = leaker.enter_generics(&Generics::default(), |view| {
        for item in &outer.content.as_ref().unwrap().1 {
            view.visit_item(item);
        }
    });
    let referrer = leaker.result().finish().unwrap();
    let mut file = file_of(vec![Item::Mod(outer.clone())]);
    let asked = std::cell::RefCell::new(Vec::new());
    common::visitor(&referrer, context, common::placeholder)
        .with_nested_self(|requirement| {
            asked.borrow_mut().push(s(&requirement_text(requirement)));
            None
        })
        .visit_file_mut(&mut file);
    // `S` has two entries for its own `Self` (both `Rel2<Self>`: one entry), `Plain` has none.
    assert_eq!(asked.into_inner().len(), 1);
    let p = entry_index(&referrer, "Rel < T >");
    let expected: File = parse_quote!(
        mod a {
            struct S<T> {
                a: P<#p>,
                b: Rel2<Self>,
                c: Vec<Rel2<Self>>,
            }
            struct Plain<T>(P<#p>);
        }
    );
    assert_eq!(s(&file), s(&expected));
}

/// The requirement as tokens, for the lines that the tests of the closure record.
fn requirement_text(requirement: &SelfRequirement) -> proc_macro2::TokenStream {
    match requirement {
        SelfRequirement::Type(ty) => quote!(#ty),
        SelfRequirement::Bound(path) => quote!(#path),
        _ => quote!(),
    }
}

/// The index of the entry `Rel2 < Self >` whose `Self` is the type `self_ty`.
fn rel2_of(referrer: &Referrer, self_ty: &str) -> usize {
    let found: Vec<usize> = (0..referrer.iter().count())
        .filter(|&ix| {
            s(&common::types(referrer).nth(ix).unwrap()) == "Rel2 < Self >"
                && matches!(common::entry(referrer, ix).self_ty.as_ref(),
                    Some(SelfRequirement::Type(ty)) if s(ty) == self_ty)
        })
        .collect();
    assert_eq!(found.len(), 1, "the entries of `Self` = {}", self_ty);
    found[0]
}

/// F36: the `Self` of the nested item hides the one around. The entry of the outer `Self` and the
/// one of the nested item's are two entries, and an item the Leaker did not analyze, which has yet
/// another `Self`, keeps `Rel2<Self>`: no entry has its context.
#[test]
fn f36_the_own_self_hides_the_one_around() {
    let nested: Item = parse_quote!(
        struct V {
            c: Vec<Rel2<Self>>,
        }
    );
    let mut leaker = common::std_leaker();
    let (_, context) = leaker.enter_generics(&Generics::default(), |view| {
        view.enter_impl(&Generics::default(), &parse_quote!(::p::W), None, |view| {
            view.visit_type(&parse_quote!(Rel2<Self>));
            view.visit_item(&nested);
        })
        .0
    });
    let referrer = leaker.result().finish().unwrap();
    // `Rel2<Self>` of the impl (`Self` = `::p::W`) and of `V` (`Self` = `V`): two entries.
    let (outer, inner) = (rel2_of(&referrer, ":: p :: W"), rel2_of(&referrer, "V"));
    let mut copy: ItemImpl = parse_quote!(
        impl ::p::W {
            fn f(x: Rel2<Self>) {
                struct V {
                    c: Vec<Rel2<Self>>,
                }
                struct Other {
                    c: Vec<Rel2<Self>>,
                }
            }
        }
    );
    let (mut visitor, calls) = nested_visitor(&referrer, context);
    visitor.visit_item_impl_mut(&mut copy);
    let expected: ItemImpl = parse_quote!(
        impl ::p::W {
            fn f(x: P<#outer>) {
                struct V {
                    c: Vec<P<#inner>>,
                }
                struct Other {
                    c: Vec<Rel2<Self>>,
                }
            }
        }
    );
    assert_eq!(s(&copy), s(&expected));
    assert_eq!(calls.take(), ["Self = V"]);
}

/// F36: an entry whose context names the own `Self` of a nested item through a bound (`Rel<T>` with
/// `T: ::p::Tr<Self>`), a `Self: 'a` or a generic associated type of a trait is written by the
/// closure of `with_nested_self()`. In a struct, an enum or a union the first is for any type, and
/// the copy passes its own `Self` (F37).
#[test]
fn f36_a_bound_or_an_outlives_that_names_the_own_self() {
    // The item, the copy, and how many times the closure is asked: not for the struct whose bound
    // names `Self`, where the entry is for any type.
    let cases: Vec<(Item, &str, usize)> = vec![
        (
            parse_quote!(
                struct S<T: ::p::Tr<Self>> {
                    b: Rel<T>,
                }
            ),
            "struct S<T: ::p::Tr<Self>> { b: P<0usize>, }",
            0,
        ),
        (
            parse_quote!(
                trait Tr<T: ::p::Tr<Self>> {
                    fn f(&self, x: Rel<T>);
                }
            ),
            "trait Tr<T: ::p::Tr<Self>> { fn f(&self, x: P<0usize>); }",
            1,
        ),
        (
            parse_quote!(
                struct S<'a, T: 'a> {
                    b: &'a Rel<T>,
                    c: &'a Vec<Rel2<Self>>,
                }
            ),
            "struct S<'a, T: 'a> { b: &'a P<0usize>, c: &'a Vec<P<1usize>>, }",
            1,
        ),
        (
            parse_quote!(
                trait Tr<'a, T> {
                    type G<'b>
                    where
                        Self: 'b;
                    fn f(&self, x: &'a Rel<T>, y: Self::G<'a>) -> Vec<Rel2<Self::G<'a>>>;
                }
            ),
            "trait Tr<'a, T> { type G<'b> where Self: 'b; fn f(&self, x: &'a P<0usize>, y: Self::G<'a>) -> Vec<P<1usize>>; }",
            1,
        ),
    ];
    for (item, expected, asked) in cases {
        let (copy, _, calls) = nested_copy_logged(Nest::EnterGenerics, &item);
        let expected: Item = syn::parse_str(expected).unwrap();
        assert_eq!(s(&copy), s(&expected));
        assert_eq!(calls.len(), asked, "`{}`", s(&item));
    }
}

/// Analyzes `item` nested in a scope (`enter_generics(<>, |view| view.visit_item(item))`), and
/// returns the result of the Leaker.
fn nested_analysis(item: &Item) -> LeakerResult {
    let mut leaker = common::std_leaker();
    leaker.enter_generics(&Generics::default(), |view| view.visit_item(item));
    leaker.result()
}

/// Asserts that `result`, the analysis of `item`, has the entry `ty` for any type
/// (`SelfRequirement::Any`), whose impl is for the parameter `__S`, which a marker cannot stand for
/// (`NeedsSelf`), and that the serialized Referrer keeps it. Returns the index of the entry, and
/// the impl.
fn assert_any_entry(item: &Item, result: &LeakerResult, ty: &str) -> (usize, RepeaterImpl) {
    let what = format!("`{}`", s(item));
    let referrer = assert_accepted_as(&what, result.clone().finish());
    let ix = common::types(&referrer)
        .position(|t| s(t) == ty)
        .unwrap_or_else(|| panic!("{}: no entry {}", what, ty));
    let is_any = |referrer: &Referrer| {
        matches!(
            common::entry(referrer, ix).self_ty.as_ref(),
            Some(SelfRequirement::Any)
        )
    };
    assert!(is_any(&referrer), "{}", what);
    let param: Ident = parse_quote!(__S);
    let r = result
        .repeater_impls(&Projection::new(repeater(), 7, ImplSelf::Param(param)))
        .unwrap()
        .remove(ix);
    assert_eq!(r.self_ty, parse_quote!(__S), "{}", what);
    // A marker is not what the copy has the bound for: the call fails, as the entry needs `Self`.
    let marker: Type = parse_quote!(crate::Marker);
    assert!(
        matches!(
            result.repeater_impls(&Projection::new(repeater(), 7, ImplSelf::Marker(marker))),
            Err(EncodeError::NeedsSelf(_))
        ),
        "{}",
        what
    );
    // The serialized form keeps it.
    let again: Referrer = parse2(quote!(#referrer)).unwrap();
    assert_eq!(again, referrer, "{}", what);
    assert!(is_any(&again), "{}", what);
    (ix, r)
}

/// The items of F37 that are supported: a struct, an enum or a union with a generic whose bound
/// names its `Self` (`S<T: AsRef<Self>>`, `where T: AsRef<(Self, u8)>`), and a field that uses
/// the generic in a type that does not name `Self` (`Rel<T>`).
fn f37_supported() -> Vec<(Item, &'static str)> {
    vec![
        (
            parse_quote!(
                pub struct S<T: ::core::convert::AsRef<Self>> {
                    b: Rel<T>,
                }
            ),
            "Rel < T >",
        ),
        (
            parse_quote!(
                pub struct S<T>(Rel<T>)
                where
                    T: ::core::convert::AsRef<(Self, u8)>;
            ),
            "Rel < T >",
        ),
        (
            parse_quote!(
                pub union U<T: ::core::marker::Copy + ::core::convert::AsRef<Self>> {
                    a: ::core::mem::ManuallyDrop<Rel<T>>,
                }
            ),
            "Rel < T >",
        ),
        // The report of the property test.
        (
            parse_quote!(
                pub enum E0_1<'a, T: ::core::marker::Send + 'a, const V: usize = 3>
                where
                    T: ::core::convert::AsRef<
                        ::std::collections::BTreeMap<(Self, Self), self::Rl<'a>>,
                    >,
                {
                    V0(&'a super::base::B2<T, ::core::result::Result<u32, u16>>),
                    V1,
                }
            ),
            "super :: base :: B2 < T , :: core :: result :: Result < u32 , u16 > >",
        ),
    ]
}

/// The items of F37 that stay rejected: the type of the entry names `Self` too.
fn f37_rejected() -> Vec<(Item, &'static str)> {
    vec![
        (
            parse_quote!(
                pub struct S<T: ::core::convert::AsRef<Self>> {
                    b: Rel2<Self, T>,
                }
            ),
            "Rel2 < Self , T >",
        ),
        (
            parse_quote!(
                pub enum E<T: ::core::convert::AsRef<Self>> {
                    A(Rel<Self>),
                    B(T),
                }
            ),
            "Rel < Self >",
        ),
        (
            parse_quote!(
                pub struct S<T>(Vec<Rel2<Self>>, T)
                where
                    T: ::core::convert::AsRef<(Self, u8)>;
            ),
            "Rel2 < Self >",
        ),
    ]
}

/// F37 (fixed): a struct, an enum or a union nested in a scope, whose generic has a bound that
/// names its `Self`, and a field that uses the generic.
///
/// Trigger: `struct S<T: AsRef<Self>> { b: Rel<T> }` (or `where T: AsRef<(Self, u8)>`) visited by
/// `View::visit_item()`. The Referrer's crate failed with E0277 "the trait bound `T: AsRef<..>` is
/// not satisfied", required for the Leaker's `S` to implement the Repeater.
///
/// Cause: the copy of a nested item is a type of its own, but the Repeater's impl was for the
/// Leaker's `S<T>` and needed the bound for that type, which the copy's environment does not have.
///
/// Fix: the type of the entry does not name `Self`, so the entry is for any type that the bound
/// holds for (`SelfRequirement::Any`): `impl<T: AsRef<__S>, __S: ?Sized> Repeater<(T,), KEY> for
/// __S { type Type = Rel<T>; }`, and the copy passes its own `Self`. An `ImplSelf::Marker` fails
/// with `EncodeError::NeedsSelf`. The item analyzed as itself (`intern_*`) is as before.
#[test]
fn f37_nested_type_with_a_bound_that_names_its_own_self_is_supported() {
    for (item, ty) in &f37_supported() {
        let what = format!("`{}`", s(item));
        // The entry is for any type that the bound holds for. Its impl is for a type parameter
        // with no bound of its own (`Self` is a struct, an enum or a union), and the parameter
        // replaces `Self` in the bound. It is `Sized` where a bound needs it (`(Self, u8)`), else
        // `?Sized`: see `f37_the_impl_is_sized_only_where_the_bounds_need_it`.
        let (_, r) = assert_any_entry(item, &nested_analysis(item), ty);
        let last = r.generics.params.last().unwrap();
        let needs_sized = s(item).contains("(Self");
        let expected: GenericParam = if needs_sized {
            parse_quote!(__S)
        } else {
            parse_quote!(__S: ?::core::marker::Sized)
        };
        assert_eq!(last, &expected, "{}", what);
        let generics = s(&r.generics);
        assert!(!generics.contains("Self"), "{}: {}", what, generics);
        assert!(generics.contains("__S"), "{}: {}", what, generics);
        assert_eq!(s(&r.ty), *ty, "{}", what);
    }
}

/// F37: the Repeater's impl of an entry for any type declares `__S: ?Sized` unless a bound needs
/// it `Sized`. With `__S: Sized`, the copy of a struct whose last field is the entry is E0275
/// "overflow evaluating the requirement `S<T>: Sized`" on rustc 1.56 to 1.66.
///
/// `?Sized` is used where every `__S` in the bounds is a whole argument of a std trait or type with
/// a `?Sized` parameter (`AsRef<__S>`, `Box<__S>`), or is behind a reference or a pointer.
/// Limitation: any other use (a tuple, a slice, a trait of the user, an associated type) keeps
/// `Sized`, so on rustc before 1.70 the entry in the last field is still E0275; the ways out are
/// another field, `where Self: Sized` on the item, or an absolute path for the type of the entry.
#[test]
fn f37_the_impl_is_sized_only_where_the_bounds_need_it() {
    // Every `Self` in the bounds is a whole argument of a trait or type with a `?Sized` parameter,
    // or behind a reference or a pointer.
    let unsized_ok: Vec<Item> = vec![
        parse_quote!(
            pub struct S<T: ::core::convert::AsRef<Self>> {
                a: u8,
                b: Rel<T>,
            }
        ),
        parse_quote!(
            pub struct S<T: AsRef<Self>> {
                a: u8,
                b: Rel<T>,
            }
        ),
        parse_quote!(
            pub struct S<T>(u8, Rel<T>)
            where
                T: ::core::convert::AsMut<Self>;
        ),
        parse_quote!(
            pub struct S<T: ::core::borrow::Borrow<Self> + ::core::borrow::BorrowMut<Self>>(
                u8,
                Rel<T>,
            );
        ),
        parse_quote!(
            pub struct S<T: PartialEq<Self> + ::core::cmp::PartialOrd<Self>>(u8, Rel<T>);
        ),
        parse_quote!(
            pub enum S<T: ::core::convert::AsRef<Box<Self>>> {
                A(u8),
                B(Rel<T>),
            }
        ),
        parse_quote!(
            pub enum S<T: ::core::convert::AsRef<::std::rc::Rc<Self>>> {
                A(u8),
                B(Rel<T>),
            }
        ),
        parse_quote!(
            pub struct S<T: ::core::convert::AsRef<::std::sync::Arc<Self>>>(u8, Rel<T>);
        ),
        parse_quote!(
            pub struct S<T: ::core::convert::AsRef<::core::marker::PhantomData<Self>>>(u8, Rel<T>);
        ),
        parse_quote!(
            pub struct S<T: ::core::convert::AsRef<*const Self> + ::core::convert::AsRef<*mut Self>>(
                u8,
                Rel<T>,
            );
        ),
        parse_quote!(
            pub struct S<T: ::core::convert::AsRef<&'static Self>>(u8, Rel<T>);
        ),
        // The bound of another generic, and a predicate on `Self` that is not `Sized`.
        parse_quote!(
            pub struct S<T: ::core::convert::AsRef<Self>, U: ::core::borrow::Borrow<Self>>(
                u8,
                Rel<T>,
                U,
            )
            where
                Self: 'static;
        ),
        parse_quote!(
            pub union S<T: ::core::marker::Copy + ::core::convert::AsRef<Self>> {
                a: ::core::mem::ManuallyDrop<Rel<T>>,
                b: u8,
            }
        ),
    ];
    // Some use of `Self` may need `Sized`: it stays.
    let sized: Vec<Item> = vec![
        parse_quote!(
            pub struct S<T: ::core::convert::AsRef<(Self, u8)>> {
                a: u8,
                b: Rel<T>,
            }
        ),
        parse_quote!(
            pub struct S<T: ::core::convert::AsRef<(u8, Self)>> {
                a: u8,
                b: Rel<T>,
            }
        ),
        parse_quote!(
            pub struct S<T>(u8, Rel<T>)
            where
                T: ::core::convert::AsRef<Vec<Self>>;
        ),
        parse_quote!(
            pub struct S<T: ::core::convert::AsRef<Option<Self>>>(u8, Rel<T>);
        ),
        parse_quote!(
            pub struct S<T: ::core::convert::AsRef<[Self]>>(u8, Rel<T>);
        ),
        parse_quote!(
            pub struct S<T: ::core::convert::AsRef<[Self; 2]>>(u8, Rel<T>);
        ),
        parse_quote!(
            pub struct S<T: ::core::iter::Iterator<Item = Self>>(u8, Rel<T>);
        ),
        parse_quote!(
            pub struct S<T: ::core::ops::Fn(Self)>(u8, Rel<T>);
        ),
        // A trait of the user: its parameter is `Sized`.
        parse_quote!(
            pub struct S<T: ::p::Tr<Self>>(u8, Rel<T>);
        ),
        // A trait of std whose parameter is `Sized`.
        parse_quote!(
            pub struct S<T: ::core::convert::From<Self>>(u8, Rel<T>);
        ),
        // A bound that needs `Sized` next to one that does not.
        parse_quote!(
            pub struct S<T: ::core::convert::AsRef<Self> + ::core::convert::AsRef<(Self, u8)>>(
                u8,
                Rel<T>,
            );
        ),
        // `Self` is `Sized` by a predicate.
        parse_quote!(
            pub struct S<T: ::core::convert::AsRef<Self>>(u8, Rel<T>)
            where
                Self: ::core::marker::Sized;
        ),
    ];
    let param: Ident = parse_quote!(__S);
    let unsized_param: GenericParam = parse_quote!(__S: ?::core::marker::Sized);
    let sized_param: GenericParam = parse_quote!(__S);
    for (items, expected, what) in [
        (&unsized_ok, &unsized_param, "?Sized"),
        (&sized, &sized_param, "Sized"),
    ] {
        for item in items {
            let result = nested_analysis(item);
            let referrer = assert_accepted_as(&format!("`{}`", s(item)), result.clone().finish());
            let ix = common::types(&referrer)
                .position(|t| s(t) == "Rel < T >")
                .unwrap_or_else(|| panic!("`{}`: no entry", s(item)));
            assert!(
                matches!(
                    common::entry(&referrer, ix).self_ty.as_ref(),
                    Some(SelfRequirement::Any)
                ),
                "`{}`",
                s(item)
            );
            let r = result
                .repeater_impls(&Projection::new(
                    repeater(),
                    7,
                    ImplSelf::Param(param.clone()),
                ))
                .unwrap()
                .remove(ix);
            assert_eq!(
                r.generics.params.last().unwrap(),
                expected,
                "{} expected, `{}`: {}",
                what,
                s(item),
                s(&r.generics)
            );
        }
    }
    // The impl as a whole.
    let item: Item = parse_quote!(
        pub struct S<T: ::core::convert::AsRef<Self>> {
            a: u8,
            b: Rel<T>,
        }
    );
    let result = nested_analysis(&item);
    result.clone().finish().unwrap();
    let imp = repeater_impl_item(&result, 0, 7);
    let key = key_of(&result, 7, 0);
    let expected: ItemImpl = parse_quote! {
        impl<T: ::core::convert::AsRef<__S>, __S: ?::core::marker::Sized>
            crate::Repeater<(T,), #key> for __S
        {
            type Type = Rel<T>;
        }
    };
    assert_eq!(imp, expected);
}

/// F37: the copy of such an item. The Visitor finds the entry in the copy of the nested item, for
/// which `Self` is any type (`SelfRequirement::Any`), and writes the path with the copy's own type,
/// `Self`, as the implementor.
#[test]
fn f37_the_copy_passes_its_own_type() {
    let item: Item = parse_quote!(
        pub struct S<T: ::core::convert::AsRef<Self>, U> {
            b: Rel<T>,
            c: Wrap<U>,
            d: Pair<Rel, U>,
        }
    );
    // In every nest: `Rel<T>` depends on the `Self` of `S`, the others do not, and all are found.
    for nest in NESTS {
        let (copy, referrer, calls) = nested_copy_logged(nest, &item);
        // The entry for any type needs no closure.
        assert!(calls.is_empty(), "{:?}: {:?}", nest, calls);
        let (b, c, d) = (
            entry_index(&referrer, "Rel < T >"),
            entry_index(&referrer, "Wrap < U >"),
            entry_index(&referrer, "Pair < Rel , U >"),
        );
        let expected: Item = parse_quote!(
            pub struct S<T: ::core::convert::AsRef<Self>, U> {
                b: P<#b>,
                c: P<#c>,
                d: P<#d>,
            }
        );
        assert_eq!(s(&copy), s(&expected), "{:?}", nest);
    }
    // The entry is for any type: the visitor writes the path through the Repeater with `Self`.
    let mut leaker = common::std_leaker();
    let (context, ()) = leaker.enter_generics(&Generics::default(), |view| view.visit_item(&item));
    let referrer = leaker.result().finish().unwrap();
    let copy = macro_copy(&referrer, context, &item, 7);
    let text = s(&copy.items[0]);
    assert!(
        text.contains(&format!(
            "b : < Self as Repeater < (T ,) , {} > > :: Type",
            s(&common::key(7, common::entry_id_at(&referrer, 0)))
        )),
        "{}",
        text
    );
    let c = entry_index(&referrer, "Wrap < U >");
    assert!(
        text.contains(&format!(
            "c : < Self as Repeater < (U ,) , {} > > :: Type",
            s(&common::key(7, common::entry_id_at(&referrer, c)))
        )),
        "{}",
        text
    );
}

/// F37: a default that names `Self` (`U = Rel2<Self, T>` with `T: AsRef<Self>`) is rejected, also
/// when the risk is accepted: the copy cannot write `Self` in a default, and neither can the
/// original (E0735). A default that does not use the generic (`U = Rel`) is accepted, and its
/// entries need no `Self`.
#[test]
fn f37_a_default_that_names_self_with_a_bound_that_names_it_is_rejected() {
    let item: Item = parse_quote!(
        pub struct S<T: ::core::convert::AsRef<Self>, U = Rel2<Self, T>>(pub T, pub U);
    );
    for accepted in [false, true] {
        let result = if accepted {
            accept_nested_self(nested_analysis(&item))
        } else {
            nested_analysis(&item)
        };
        let err = assert_rejected(
            &item,
            result.finish(),
            &["`Rel2<Self, T>` is in the default of a parameter, and it and a bound of its generics name `Self`"],
        );
        assert!(
            all_notes(&err)[0].contains("cannot write in the default of a parameter (E0735)"),
            "{:?}",
            all_notes(&err)
        );
        assert_eq!(all_helps(&err), ["write the default without `Self`"]);
    }
    // The default does not use the generic: nothing depends on the `Self` of the bound.
    let item: Item = parse_quote!(
        pub struct S<T: ::core::convert::AsRef<Self>, U = Rel>(pub T, pub Wrap<U>);
    );
    let referrer = assert_accepted(&item, nested_analysis(&item).finish());
    assert_eq!(referrer.iter().count(), 2);
    assert!((0..2).all(|ix| common::entry(&referrer, ix).self_ty.is_none()),);
}

/// F37: an entry whose type names `Self` too (`Rel2<Self, T>` with `T: AsRef<Self>`) is for any
/// type, and what it needs of `Self` is not known (F42): it is a `NestedSelfInBounds` error, which
/// `finish()` returns until the macro allows it, and once allowed the Referrer has the entry. Its
/// help names the way out of F38, an absolute path. As the item itself (`intern_*`), `Self` is the
/// Leaker's type, which has the bound: accepted.
#[test]
fn f37_a_type_that_names_self_with_a_bound_that_names_it_is_a_risk_error_with_its_entry() {
    const HEADLINE: &str = "names `Self` and has a generic whose bound mentions `Self`";
    for (item, ty) in &f37_rejected() {
        let result = nested_analysis(item);
        let err = assert_rejected(
            item,
            result.clone().finish(),
            &[&format!("`{}` {}", plain(ty), HEADLINE)],
        );
        assert_eq!(err[0].reason, Reason::NestedSelfInBounds);
        // The recovered Referrer has the entry: the error is the risk of it.
        let referrer = result.recovered();
        let entries: Vec<String> = common::type_texts(&referrer);
        assert!(
            entries.iter().any(|entry| entry == ty),
            "`{}`: {:?}",
            s(item),
            entries
        );
        let notes = all_notes(&err);
        assert_eq!(
            notes[0],
            "it names the `Self` of an item nested in the analyzed syntax, and so do the \
             bounds of its generics: the entry is for any type that the bound holds for, and \
             the copy passes its own type as `Self`. That is right only if the declaration of \
             the type has no trait bound on the parameter that receives `Self` (E0277 in the \
             impl of the Repeater), and, on rustc before 1.70, if the copy's field is not the \
             last field of a struct (E0275)",
            "`{}`",
            s(item)
        );
        let helps = all_helps(&err);
        assert_eq!(helps.len(), 1, "`{}`", s(item));
        // The ways out: the F38 fix (an absolute path) is one.
        assert!(
            helps[0].contains(
                "without `Self`, write the type without `Self`, or write the type with an \
                 absolute path (it is copied as written, and its `Self` is the copy's own)"
            ),
            "{}",
            helps[0]
        );
    }
    // As the item itself, `Self` is written as the Leaker's type, and the Repeater's impl has the
    // bound for it: accepted, with the entry, whose impl is for that type.
    for (item, _) in f37_rejected().iter().chain(&f37_supported()) {
        let mut leaker = common::std_leaker();
        common::intern_item(&mut leaker, item);
        let referrer = assert_accepted(item, leaker.result().finish());
        assert!(referrer.iter().count() > 0, "`{}`", s(item));
        for ix in 0..referrer.iter().count() {
            assert!(
                !matches!(
                    common::entry(&referrer, ix).self_ty.as_ref(),
                    Some(SelfRequirement::Any)
                ),
                "`{}`: the item itself has the Leaker's type for `Self`",
                s(item)
            );
        }
    }
}

// ── F38 ──────────────────────────────────────────────────────────────────────

/// F38 (fixed): an absolute type that names the own `Self` of a struct, an enum or a union nested
/// in a scope was an optional entry, so the copy replaced it with the Leaker's `Self`.
///
/// Trigger: `struct Node { next: ::std::boxed::Box<Self> }` visited by `View::visit_item()`. The
/// copy wrote the Leaker's `Node` for `Self`, so it was not recursive, and `::c::Rel2<Self, T>`
/// with `T: AsRef<Self>` meant two types in one item.
///
/// Cause: the Leaker leaked it like any absolute type, though it resolves the same in the copy and
/// its only link to the item is `Self`, which there is the copy's own type.
///
/// Fix: such a type is no entry, also where the graph would find it cheaper than its parts
/// (`::c::Both<::d::X, ::d::Y, Self>`): it is copied as written. Its parts that must be leaked
/// still are entries (`Rel<T>` in `::c::Rel2<Self, Rel<T>>`), and so are the relative types that
/// name `Self` and a reference with a lifetime. It is the way out that F37's error suggests.
#[test]
fn f38_an_absolute_type_that_names_the_own_self_is_copied_as_written() {
    // The item, its entries, and the copy (`P<ix>` for an entry), or `None` if it is the item.
    let cases: Vec<(Item, &[&str], Option<&str>)> = vec![
        // The report: a recursive type through a box.
        (
            parse_quote!(
                pub struct Node {
                    next: ::std::boxed::Box<Self>,
                }
            ),
            &[],
            None,
        ),
        // `::c::W<T>` does not name `Self`: an optional entry, as before.
        (
            parse_quote!(
                pub struct Node<T> {
                    a: ::c::W<T>,
                    b: ::c::Rel2<Self, T>,
                    c: Vec<::c::Rel2<Self, T>>,
                }
            ),
            &["::c::W<T>"],
            Some(
                "pub struct Node < T > { a : P < 0usize > , b : :: c :: Rel2 < Self , T > , c : Vec < \
                 :: c :: Rel2 < Self , T > > , }",
            ),
        ),
        // The same with a bound that names `Self`, and without it: no entry.
        (
            parse_quote!(
                pub struct S<T: ::core::convert::AsRef<Self>> {
                    b: ::c::Rel2<Self, T>,
                }
            ),
            &[],
            None,
        ),
        (
            parse_quote!(
                pub struct S<T> {
                    b: ::c::Rel2<Self, T>,
                }
            ),
            &[],
            None,
        ),
        // The entries that do not name `Self` are as before: `Rel<T>` is required, and `::c::W<T>`
        // is optional; with the bound, both are for any type that it holds for (`Any`), and the
        // copy passes its own.
        (
            parse_quote!(
                pub struct S<T: ::core::convert::AsRef<Self>> {
                    a: ::c::Rel2<Self, T>,
                    b: Rel<T>,
                    c: ::c::W<T>,
                }
            ),
            &["Rel<T>", "::c::W<T>"],
            Some(
                "pub struct S < T : :: core :: convert :: AsRef < Self > > { a : :: c :: Rel2 < Self \
                 , T > , b : P < 0usize > , c : P < 1usize > , }",
            ),
        ),
        (
            parse_quote!(
                pub struct S<T> {
                    a: ::c::Rel2<Self, T>,
                    b: Rel<T>,
                    c: ::c::W<T>,
                }
            ),
            &["Rel<T>", "::c::W<T>"],
            Some(
                "pub struct S < T > { a : :: c :: Rel2 < Self , T > , b : P < 0usize > , c : P < \
                 1usize > , }",
            ),
        ),
        // A part that must be leaked is an entry, in the type that is copied as written.
        (
            parse_quote!(
                pub struct S<T> {
                    b: ::c::Rel2<Self, Rel<T>>,
                }
            ),
            &["Rel<T>"],
            Some("pub struct S < T > { b : :: c :: Rel2 < Self , P < 0usize > > , }"),
        ),
        // The parts are the entries, not the whole type with the `Self` of the Leaker, even where
        // one entry for the whole would save one: in the optional ones and in the required ones.
        (
            parse_quote!(
                pub struct S {
                    f: ::c::Both<::d::X, ::d::Y, Self>,
                }
            ),
            &["::d::X", "::d::Y"],
            Some("pub struct S { f : :: c :: Both < P < 0usize > , P < 1usize > , Self > , }"),
        ),
        (
            parse_quote!(
                pub struct S {
                    f: ::c::Both<Rel<u8>, Rel3<u8>, Self>,
                }
            ),
            &["Rel<u8>", "Rel3<u8>"],
            Some("pub struct S { f : :: c :: Both < P < 0usize > , P < 1usize > , Self > , }"),
        ),
        // A reference with a lifetime is leaked whole with what it holds, so it is the entry, and
        // `Self` in it is the type that the macro writes.
        (
            parse_quote!(
                pub struct S {
                    f: &'static ::c::Both<Rel<u8>, Rel3<u8>, Self>,
                }
            ),
            &["&'static::c::Both<Rel<u8>,Rel3<u8>,Self>"],
            Some("pub struct S { f : P < 0usize > , }"),
        ),
        // An enum and a union.
        (
            parse_quote!(
                pub enum E<T> {
                    A(::c::W<Self>),
                    B(::c::W<T>),
                }
            ),
            &["::c::W<T>"],
            Some("pub enum E < T > { A (:: c :: W < Self >) , B (P < 0usize >) , }"),
        ),
        (
            parse_quote!(
                pub union U<T> {
                    a: ::core::mem::ManuallyDrop<::c::W<Self>>,
                    b: ::core::mem::ManuallyDrop<::c::W<T>>,
                }
            ),
            &["::c::W<T>"],
            Some(
                "pub union U < T > { a : :: core :: mem :: ManuallyDrop < :: c :: W < Self > > , b : \
                 :: core :: mem :: ManuallyDrop < P < 0usize > > , }",
            ),
        ),
        // The ways out that F37 suggests for the relative types that name `Self` and a bound that
        // names it: the absolute path is accepted, and copied as written.
        (
            parse_quote!(
                pub enum E<T: ::core::convert::AsRef<Self>> {
                    A(::c::Rel<Self>),
                    B(T),
                }
            ),
            &[],
            None,
        ),
        (
            parse_quote!(
                pub struct S<T>(Vec<::c::Rel2<Self>>, T)
                where
                    T: ::core::convert::AsRef<(Self, u8)>;
            ),
            &[],
            None,
        ),
    ];
    for (item, entries, expected) in &cases {
        for nest in NESTS {
            let what = format!("`{}` in {:?}", s(item), nest);
            let (copy, referrer, calls) = nested_copy_logged(nest, item);
            assert_eq!(entries_text(&referrer, false), *entries, "{}", what);
            // With no entry, the closure of `with_nested_self()` is not asked.
            if entries.is_empty() {
                assert!(calls.is_empty(), "{}: {:?}", what, calls);
            }
            assert_eq!(
                s(&copy),
                expected.map_or_else(|| s(item), String::from),
                "{}",
                what
            );
        }
    }
}

/// F38, the controls: the types that must be leaked and name the own `Self` are entries, written
/// by `with_nested_self()`, and the item analyzed as itself (`intern_*`), a nested trait and a
/// nested impl are as before: their optional entries that name `Self` stay entries, and the copy
/// replaces them.
#[test]
fn f38_the_relative_types_and_the_other_items_are_unchanged() {
    // Relative types that name `Self` are entries: the copy holds what the macro writes. The
    // absolute one beside them is not.
    let item: Item = parse_quote!(
        pub struct S<T> {
            a: Rel<Self>,
            b: inner::W<Self>,
            c: crate::W<Self>,
            d: ::c::W<Self>,
            e: ::c::W<(Self, Rel2<Self>)>,
        }
    );
    for nest in NESTS {
        let what = format!("{:?}", nest);
        let (copy, referrer, calls) = nested_copy_logged(nest, &item);
        assert_eq!(calls, ["Self = S < T >"], "{}", what);
        assert_eq!(
            entries_text(&referrer, false),
            [
                "Rel<Self>",
                "Rel2<Self>",
                "crate::W<Self>",
                "inner::W<Self>"
            ],
            "{}",
            what
        );
        for ix in 0..referrer.iter().count() {
            assert!(
                matches!(
                    common::entry(&referrer, ix).self_ty.as_ref(),
                    Some(SelfRequirement::Type(_))
                ),
                "{}",
                what
            );
        }
        let (a, b, c, rel2) = (
            entry_index(&referrer, "Rel < Self >"),
            entry_index(&referrer, "inner :: W < Self >"),
            entry_index(&referrer, "crate :: W < Self >"),
            entry_index(&referrer, "Rel2 < Self >"),
        );
        let expected: Item = parse_quote!(
            pub struct S<T> {
                a: P<#a>,
                b: P<#b>,
                c: P<#c>,
                d: ::c::W<Self>,
                e: ::c::W<(Self, P<#rel2>)>,
            }
        );
        assert_eq!(s(&copy), s(&expected), "{}", what);
    }

    // The item analyzed as itself: `Self` is the Leaker's type in the copy (`with_self()`), so the
    // optional entry that names it is as before.
    let node: ItemStruct = parse_quote!(
        pub struct Node {
            next: ::std::boxed::Box<Self>,
        }
    );
    let mut leaker = common::std_leaker();
    leaker.intern_struct(&node);
    let referrer = leaker.result().finish().unwrap();
    assert_eq!(entries_text(&referrer, false), ["::std::boxed::Box<Self>"]);
    assert!(matches!(
        common::entry(&referrer, 0).self_ty.as_ref(),
        Some(SelfRequirement::Type(_))
    ));

    // A nested trait: `Self` is the implementor, for which the macro writes a type: the optional
    // entry stays an entry, and the copy replaces it.
    let tr: Item = parse_quote!(
        pub trait Tr {
            fn f(&self) -> ::c::W<Self>;
        }
    );
    for nest in NESTS {
        let (copy, referrer, calls) = nested_copy_logged(nest, &tr);
        assert_eq!(calls, ["Self: Tr"], "{:?}", nest);
        assert_eq!(
            entries_text(&referrer, false),
            ["::c::W<Self>"],
            "{:?}",
            nest
        );
        assert!(
            matches!(
                common::entry(&referrer, 0).self_ty.as_ref(),
                Some(SelfRequirement::Bound(_))
            ),
            "{:?}",
            nest
        );
        let expected: Item = parse_quote!(
            pub trait Tr {
                fn f(&self) -> P<0usize>;
            }
        );
        assert_eq!(s(&copy), s(&expected), "{:?}", nest);
    }

    // A nested impl: `Self` is the self type as written, the same in the copy; the entry stays an
    // entry.
    let imp: Item = parse_quote!(
        impl<T> ::p::W for ::p::S<T> {
            fn f() -> ::c::W<Self> {
                loop {}
            }
        }
    );
    for nest in NESTS {
        let (copy, referrer, calls) = nested_copy_logged(nest, &imp);
        assert_eq!(calls, ["Self = :: p :: S < T >"], "{:?}", nest);
        let ix = entry_index(&referrer, ":: c :: W < Self >");
        assert!(
            matches!(
                common::entry(&referrer, ix).self_ty.as_ref(),
                Some(SelfRequirement::Type(_))
            ),
            "{:?}",
            nest
        );
        assert!(!s(&copy).contains("W < Self >"), "{:?}", nest);
    }
}

// ── F39 ──────────────────────────────────────────────────────────────────────

/// F39 (fixed): the default of a parameter of a nested trait, with an entry that needs the trait's
/// `Self`, was left as written by the Visitor.
///
/// Trigger: `trait Tr<T: AsRef<Self>, U = Rel<T>> { fn f(&self, u: U); }` visited by
/// `View::visit_item()`. The copy kept the relative `Rel<T>`, which the Referrer's crate cannot
/// resolve; the same type in a method was replaced.
///
/// Cause: the Leaker analyzed the default with the trait's own `Self`, and the Visitor looked it up
/// without, so the contexts differed.
///
/// Fix: a default that does not name `Self` is analyzed alone, as for a trait the macro entered
/// (F65): its entry has no `Self` requirement and needs no closure. A default that names `Self`
/// (`U = Rel<Self>`) is an entry for `Self: Tr<T, U>`, written by `with_nested_self()`; its
/// Repeater arguments name `U`, which a declaration cannot pass to its own default (E0128), so the
/// macro passes a type that the host implements the trait for.
#[test]
fn f39_the_default_of_a_nested_trait_that_needs_its_self_is_written_through_the_host() {
    let item: Item = parse_quote!(
        pub trait Tr<T: ::core::convert::AsRef<Self>, U = Rel<T>> {
            fn f(&self, u: U);
        }
    );
    // The default does not name `Self`, only the bound of `T` does: it is analyzed alone, as the
    // default of a trait that the macro entered (F65), so its entry is for any type that the bound
    // holds for, with no `Self` requirement.
    for nest in NESTS {
        let what = format!("{:?}", nest);
        let (copy, referrer, calls) = nested_copy_logged(nest, &item);
        // A plain entry: no closure.
        assert!(calls.is_empty(), "{}: {:?}", what, calls);
        assert_eq!(referrer.iter().count(), 1, "{}", what);
        assert_eq!(entries_of(&referrer, "Rel < T >"), [(0, false)], "{}", what);
        assert!(
            common::bucket_self(&referrer, 0, common::entry_id_at(&referrer, 0)).is_none(),
            "{}",
            what
        );
        assert!(common::entry(&referrer, 0).self_ty.is_none(), "{}", what);
        assert_eq!(
            context_text(&referrer, 0),
            "< T : :: core :: convert :: AsRef < Self > >",
            "{}",
            what
        );
        let expected: Item = parse_quote!(
            pub trait Tr<T: ::core::convert::AsRef<Self>, U = P<0usize>> {
                fn f(&self, u: U);
            }
        );
        assert_eq!(s(&copy), s(&expected), "{}", what);
    }

    // As a macro writes it. The Repeater's arguments of the entry do not include `U`: the default
    // can name them.
    let mut leaker = common::std_leaker();
    let (context, ()) = leaker.enter_generics(&Generics::default(), |view| view.visit_item(&item));
    let referrer = leaker.result().finish().unwrap();
    let file = macro_copy(&referrer, context, &item, 7);
    let key = common::key(7, common::entry_id_at(&referrer, 0));
    let expected: Item = parse_quote!(
        pub trait Tr<T: ::core::convert::AsRef<Self>, U = <Self as Repeater<(T,), #key>>::Type> {
            fn f(&self, u: U);
        }
    );
    assert_eq!(s(&file.items[0]), s(&expected));
    let args = common::repeater_args(common::entry(&referrer, 0), common::identity, None).unwrap();
    assert_eq!(s(&args), "(T ,)");

    // A default that names `Self` (`Rhs = Self`) cannot be analyzed alone: its entry is for the
    // implementors, `Self: Tr<T, U>`, which the Visitor writes through the host. The arguments of
    // the Repeater include the defaulted parameter, which a declaration cannot name in its own
    // default (E0128): the macro passes a type for it that the host implements the trait for.
    let named: Item = parse_quote!(
        pub trait Tr<T, U = Rel<Self>> {
            fn f(&self, u: U);
        }
    );
    for nest in NESTS {
        let what = format!("{:?}", nest);
        let (copy, referrer, calls) = nested_copy_logged(nest, &named);
        assert_eq!(calls, ["Self: Tr < T , U >"], "{}", what);
        assert_eq!(referrer.iter().count(), 1, "{}", what);
        match common::entry(&referrer, 0).self_ty.as_ref() {
            Some(SelfRequirement::Bound(path)) => {
                assert_eq!(s(path), "Tr < T , U >", "{}", what)
            }
            other => panic!("{}: {:?}", what, other),
        }
        let expected: Item = parse_quote!(
            pub trait Tr<T, U = P<0usize>> {
                fn f(&self, u: U);
            }
        );
        assert_eq!(s(&copy), s(&expected), "{}", what);
        // Without the closure the default is left as written: it needs the trait's `Self`.
        let (copy, _) = nested_copy_unhooked(nest, &named);
        assert_eq!(s(&copy), s(&named), "{}", what);
    }

    // Without the closure, the default that does not name `Self` is still written (no closure is
    // asked for it above), and the same type in a method is an entry of the trait's `Self`, which
    // is left as written.
    let method: Item = parse_quote!(
        pub trait Tr<T: ::core::convert::AsRef<Self>> {
            fn f(&self, u: Rel<T>);
        }
    );
    for nest in NESTS {
        let (copy, _) = nested_copy_unhooked(nest, &method);
        assert_eq!(s(&copy), s(&method), "{:?}", nest);
    }
}

/// F39, the controls: a nested trait whose default does not need its `Self` is replaced as before,
/// and so is a method that does. Nothing in a trait is for any type (`SelfRequirement::Any`), and
/// an associated type or a constant that uses the generic is replaced like the default.
#[test]
fn f39_the_other_traits_are_unchanged() {
    // The item, and its copy with `P` for each entry.
    let cases: Vec<(Item, Item)> = vec![
        (
            parse_quote!(
                pub trait Tr<T, U = Rel<T>> {
                    fn f(&self, u: U);
                }
            ),
            parse_quote!(
                pub trait Tr<T, U = P<0usize>> {
                    fn f(&self, u: U);
                }
            ),
        ),
        (
            parse_quote!(
                pub trait Tr<T: Clone, U = Rel<T>> {
                    fn f(&self, u: U);
                }
            ),
            parse_quote!(
                pub trait Tr<T: Clone, U = P<0usize>> {
                    fn f(&self, u: U);
                }
            ),
        ),
        // The bound names `Self`, and the default does not use the generic.
        (
            parse_quote!(
                pub trait Tr<T: ::core::convert::AsRef<Self>, U = Rel> {
                    fn f(&self, t: T, u: U);
                }
            ),
            parse_quote!(
                pub trait Tr<T: ::core::convert::AsRef<Self>, U = P<0usize>> {
                    fn f(&self, t: T, u: U);
                }
            ),
        ),
        // The same entry in a method: written for the trait's `Self`.
        (
            parse_quote!(
                pub trait Tr<T: ::core::convert::AsRef<Self>> {
                    fn f(&self, u: Rel<T>);
                }
            ),
            parse_quote!(
                pub trait Tr<T: ::core::convert::AsRef<Self>> {
                    fn f(&self, u: P<0usize>);
                }
            ),
        ),
        // An associated type and a constant keep the `Self` of the trait.
        (
            parse_quote!(
                pub trait Tr<T: ::core::convert::AsRef<Self>> {
                    type A = Rel<T>;
                    const C: Rel2<T>;
                }
            ),
            parse_quote!(
                pub trait Tr<T: ::core::convert::AsRef<Self>> {
                    type A = P<0usize>;
                    const C: P<1usize>;
                }
            ),
        ),
    ];
    for (item, expected) in &cases {
        for nest in NESTS {
            let what = format!("{:?} `{}`", nest, s(item));
            let (copy, referrer) = nested_copy(nest, item);
            assert_eq!(s(&copy), s(expected), "{}", what);
            for ix in 0..referrer.iter().count() {
                assert!(
                    !matches!(
                        common::entry(&referrer, ix).self_ty.as_ref(),
                        Some(SelfRequirement::Any)
                    ),
                    "{}",
                    what
                );
            }
        }
    }
}

/// How a trait analyzed as the item itself is analyzed.
#[derive(Clone, Copy, Debug)]
enum Root {
    /// `intern_trait()`.
    InternTrait,
    /// `enter_trait()` and a walk of the item, as a macro that enters the scope itself does.
    EnterTrait,
    /// `enter_trait()` and `view.visit_item(&item)`: the item is visited again, as a nested item.
    EnterTraitItem,
    /// A scope of a view inside `enter_generics()` (a nested `enter_trait()`) and a walk of the
    /// item.
    NestedEnterTrait,
}

const ROOTS: [Root; 4] = [
    Root::InternTrait,
    Root::EnterTrait,
    Root::EnterTraitItem,
    Root::NestedEnterTrait,
];

fn root_analysis(root: Root, item: &ItemTrait) -> (Referrer, ContextId) {
    let mut leaker = common::std_leaker();
    let context = match root {
        Root::InternTrait => leaker.intern_trait(item),
        Root::EnterTrait => {
            leaker
                .enter_trait(&item.ident, &item.generics, |view| {
                    syn::visit::visit_item_trait(view, item)
                })
                .0
        }
        Root::EnterTraitItem => {
            let nested = Item::Trait(item.clone());
            leaker
                .enter_trait(&item.ident, &item.generics, |view| view.visit_item(&nested))
                .0
        }
        Root::NestedEnterTrait => {
            leaker
                .enter_generics(&Generics::default(), |view| {
                    view.enter_trait(&item.ident, &item.generics, |view| {
                        syn::visit::visit_item_trait(view, item)
                    })
                    .0
                })
                .1
        }
    };
    (leaker.result().finish().unwrap(), context)
}

/// The entries of `referrer` of the type `ty`, with what `Self` means in their context.
fn entries_of(referrer: &Referrer, ty: &str) -> Vec<(usize, bool)> {
    (0..referrer.iter().count())
        .filter(|&ix| s(common::types(referrer).nth(ix).unwrap()) == ty)
        .map(|ix| (ix, common::entry(referrer, ix).self_ty.is_some()))
        .collect()
}

/// F39: the default of a trait analyzed as the item itself (`intern_trait()`, `enter_trait()`) and
/// copied with `with_self(host)` is analyzed alone (without `Self`, with the parameters before it),
/// and the Visitor looks it up the same way, also when another entry (`Rel2<Self>`) gives the scope
/// a `Self` (a lookup with that `Self` missed it, and the copy kept `Rel<T>`). The same type in a
/// method stays an entry for the trait's `Self`.
///
/// The default is recorded once: the walk of the trait must not analyze it again. It did in
/// `View::enter_trait()` and in `view.visit_item()` of the same trait after `enter_trait()`, which
/// gave a second entry for the trait's `Self` and a second Repeater impl. A different trait visited
/// inside is analyzed in full, with entries of its own.
#[test]
fn f39_the_default_of_the_trait_itself_is_looked_up_without_self() {
    let host: Type = parse_quote!(Host);
    let item: ItemTrait = parse_quote!(
        pub trait Tr<T: ::core::convert::AsRef<Self>, U = Rel<T>> {
            fn f(&self, u: U) -> Rel2<Self>;
        }
    );
    for root in ROOTS {
        let what = format!("{:?}", root);
        let (referrer, context) = root_analysis(root, &item);
        // The entry analyzed alone has no `Self`; its generics have the bound.
        let alone: Vec<usize> = entries_of(&referrer, "Rel < T >")
            .into_iter()
            .filter(|(_, has_self)| !has_self)
            .map(|(ix, _)| ix)
            .collect();
        assert_eq!(alone.len(), 1, "{}", what);
        let (default, rel2) = (alone[0], entry_index(&referrer, "Rel2 < Self >"));
        // Each is recorded once.
        assert_eq!(referrer.iter().count(), 2, "{}", what);
        assert_eq!(
            context_text(&referrer, default),
            "< T : :: core :: convert :: AsRef < Self > >",
            "{}",
            what
        );
        let mut copy = item.clone();
        common::visitor(&referrer, context, common::placeholder)
            .with_self(host.clone())
            .visit_item_trait_mut(&mut copy);
        let expected: ItemTrait = parse_quote!(
            pub trait Tr<T: ::core::convert::AsRef<Host>, U = P<#default>> {
                fn f(&self, u: U) -> P<#rel2>;
            }
        );
        assert_eq!(s(&copy), s(&expected), "{}", what);
    }

    // Only the default: the scope has no `Self`. It is recorded once: the entry analyzed alone
    // (no `Self`, and the generics before `U`) is the only one.
    let item: ItemTrait = parse_quote!(
        pub trait Tr<T: ::core::convert::AsRef<Self>, U = Rel<T>> {
            fn f(&self, u: U);
        }
    );
    for root in ROOTS {
        let what = format!("{:?}", root);
        let (referrer, context) = root_analysis(root, &item);
        assert_eq!(referrer.iter().count(), 1, "{}", what);
        assert_eq!(entries_of(&referrer, "Rel < T >"), [(0, false)], "{}", what);
        assert_eq!(
            context_text(&referrer, 0),
            "< T : :: core :: convert :: AsRef < Self > >",
            "{}",
            what
        );
        let default = entries_of(&referrer, "Rel < T >")
            .into_iter()
            .find(|(_, has_self)| !has_self)
            .unwrap_or_else(|| panic!("{}: no entry", what))
            .0;
        let mut copy = item.clone();
        common::visitor(&referrer, context, common::placeholder)
            .with_self(host.clone())
            .visit_item_trait_mut(&mut copy);
        let expected: ItemTrait = parse_quote!(
            pub trait Tr<T: ::core::convert::AsRef<Host>, U = P<#default>> {
                fn f(&self, u: U);
            }
        );
        assert_eq!(s(&copy), s(&expected), "{}", what);
    }

    // The same type in the default and in a method: the method's entry is for the trait's `Self`,
    // the default's is not.
    let item: ItemTrait = parse_quote!(
        pub trait Tr<T: ::core::convert::AsRef<Self>, U = Rel<T>> {
            fn f(&self, u: Rel<T>);
        }
    );
    for root in ROOTS {
        let what = format!("{:?}", root);
        let (referrer, context) = root_analysis(root, &item);
        let entries = entries_of(&referrer, "Rel < T >");
        assert_eq!(referrer.iter().count(), 2, "{}: {:?}", what, entries);
        let default = entries.iter().find(|(_, has_self)| !has_self).unwrap().0;
        let method = entries.iter().find(|(_, has_self)| *has_self).unwrap().0;
        let mut copy = item.clone();
        common::visitor(&referrer, context, common::placeholder)
            .with_self(host.clone())
            .visit_item_trait_mut(&mut copy);
        let expected: ItemTrait = parse_quote!(
            pub trait Tr<T: ::core::convert::AsRef<Host>, U = P<#default>> {
                fn f(&self, u: P<#method>);
            }
        );
        assert_eq!(s(&copy), s(&expected), "{}", what);
    }

    // A default whose type does not depend on `Self`: one entry for the scope and the walk.
    let item: ItemTrait = parse_quote!(
        pub trait Tr<T, U = Rel<T>> {
            fn f(&self, u: U);
        }
    );
    for root in ROOTS {
        let (referrer, _) = root_analysis(root, &item);
        assert_eq!(referrer.iter().count(), 1, "{:?}", root);
    }

    // A trait nested in the scope that is not the one it was entered for is analyzed in full: its
    // default is another type, and an entry of its own.
    let item: ItemTrait = parse_quote!(
        pub trait Tr<T: ::core::convert::AsRef<Self>, U = Rel<T>> {
            fn f(&self, u: U);
        }
    );
    let other: Item = parse_quote!(
        pub trait Tr<T: ::core::convert::AsRef<Self>, U = Rel2<T>> {
            fn f(&self, u: U);
        }
    );
    let mut leaker = common::std_leaker();
    leaker.enter_trait(&item.ident, &item.generics, |view| view.visit_item(&other));
    let referrer = leaker.result().finish().unwrap();
    assert_eq!(referrer.iter().count(), 2);
    let rel = entries_of(&referrer, "Rel < T >");
    assert!(matches!(rel[..], [(_, false)]), "{:?}", rel);
    let rel2 = entries_of(&referrer, "Rel2 < T >");
    assert!(matches!(rel2[..], [(_, false)]), "{:?}", rel2);
}

/// F39, the contrast: the same default in a struct nested in a scope has an entry without `Self`
/// and without the bound that names it (the copy cannot write `Self` in a default, E0735), whereas
/// a trait's keeps the bound (F65), as does the struct analyzed as itself.
#[test]
fn f39_a_nested_struct_has_no_self_in_the_entry_of_the_default() {
    let item: Item = parse_quote!(
        pub struct S<T: ::core::convert::AsRef<Self>, U = Rel<T>>(pub T, pub U);
    );
    let referrer = assert_accepted(&item, nested_analysis(&item).finish());
    assert_eq!(entries_of(&referrer, "Rel < T >"), [(0, false)],);
    // The bound that names `Self` is not in the context.
    assert_eq!(context_text(&referrer, 0), "< T >");
    // The item itself keeps the bound: its default is analyzed alone, with the generics before it.
    if let Item::Struct(item) = &item {
        let mut leaker = common::std_leaker();
        leaker.intern_struct(item);
        let referrer = leaker.result().finish().unwrap();
        assert_eq!(entries_of(&referrer, "Rel < T >"), [(0, false)]);
        assert_eq!(
            context_text(&referrer, 0),
            "< T : :: core :: convert :: AsRef < Self > >"
        );
    }
}

// ── F40 ──────────────────────────────────────────────────────────────────────

/// A Leaker that reads the values and the bodies (`check_values`, `check_bodies`), as the
/// downstream crate of F40 has it: primitive types and the names of the standard prelude are
/// allowed.
fn values_leaker() -> Leaker<impl Allow> {
    let mut leaker = Leaker::with_rule(Rules::new().primitive().prelude());
    leaker.check_values = true;
    leaker.check_bodies = true;
    leaker
}

/// Analyzes the trait `source` with `check_values`, and returns its result and its context.
fn f40_result(source: &str) -> (ItemTrait, LeakerResult, ContextId) {
    let item: ItemTrait = parse_str(source).unwrap_or_else(|e| panic!("{}: {}", source, e));
    let mut leaker = values_leaker();
    let context = leaker.intern_trait(&item);
    (item, leaker.result(), context)
}

/// Analyzes the trait `source` with `check_values`, and returns its Referrer and its context.
fn f40_trait(source: &str) -> (ItemTrait, Referrer, ContextId) {
    let (item, result, context) = f40_result(source);
    let referrer = assert_accepted_as(source, result.finish());
    (item, referrer, context)
}

/// The copy of `item` in the scope `context`, with the `ix`-th entry replaced by `P<ix>`.
fn f40_copy(referrer: &Referrer, context: ContextId, item: &ItemTrait) -> String {
    rewrite(referrer, context, Item::Trait(item.clone()))
}

/// The messages and the notes of the errors that `check_values` gives to the trait `source`.
fn f40_errors(source: &str) -> Vec<(String, String)> {
    let item: ItemTrait = parse_str(source).unwrap_or_else(|e| panic!("{}: {}", source, e));
    let mut leaker = values_leaker();
    leaker.intern_trait(&item);
    let err = errors_of(leaker);
    assert!(!err.is_empty(), "`{}` is accepted", source);
    err.iter()
        .map(|item| {
            let note = item
                .notes
                .first()
                .map(|(_, n)| n.as_str())
                .unwrap_or("")
                .to_string();
            (item.message.clone(), note)
        })
        .collect()
}

/// F40 (fixed): a type in a value is an entry when the Leaker reads the values.
///
/// Trigger: `trait Tr { const N: usize = ::core::mem::size_of::<inner::Local>(); }`. The Leaker
/// found no entry for `inner::Local`, so the copy of the default kept a relative path that does
/// not resolve in the Referrer's crate (E0433). With `check_values` every relative path in a
/// value was rejected, a type too.
///
/// Cause: a value was read only to reject its relative paths; a type in it was never analyzed,
/// though the Visitor replaces a type where it stands in an expression.
///
/// Fix: with `check_values` (and `check_bodies` for a body), a type in a type position of a value
/// is analyzed like any type of the item, with the generics and `Self` in scope, and is an entry.
/// A path that is no type is still rejected. Without `check_values` a value is not read.
#[test]
fn f40_a_type_in_the_default_of_an_associated_const_is_carried() {
    let source = "trait Tr { const N: usize = ::core::mem::size_of::<inner::Local>(); }";
    let item: ItemTrait = parse_str(source).unwrap();

    // Not read without `check_values`: no entry, no error, as before.
    let mut leaker = common::std_leaker();
    leaker.intern_trait(&item);
    assert!(leaker.result().finish().unwrap().is_empty());

    // Read with it: the type is an entry.
    let (item, result, context) = f40_result(source);
    let referrer = assert_accepted_as(source, result.clone().finish());
    assert_eq!(common::type_texts(&referrer), ["inner :: Local"]);
    assert_eq!(context_text(&referrer, 0), "");
    assert!(common::entry(&referrer, 0).self_ty.is_none());
    // The Visitor replaces it where it stands in the default.
    let expected: ItemTrait = parse_quote!(
        trait Tr {
            const N: usize = ::core::mem::size_of::<P<0usize>>();
        }
    );
    assert_eq!(f40_copy(&referrer, context, &item), s(&expected));

    // The Repeater's impl is written like the one of an entry of a signature.
    let r = result
        .repeater_impls(&Projection::new(
            repeater(),
            7,
            ImplSelf::Marker(parse_quote!(())),
        ))
        .unwrap()
        .remove(0);
    assert_eq!(s(&r.ty), "inner :: Local");

    // A type is replaced, a path that is not a type is still rejected.
    assert_eq!(
        f40_errors("trait Tr { const N: usize = inner::X; }"),
        [(
            "use absolute path: 'inner::X'".to_string(),
            "it is in the default value of `N`, which the copy writes as it is".to_string()
        )]
    );
}

/// F40: the default body of a method, and the body of a method of an impl, are read the same way,
/// in the generics of the method.
#[test]
fn f40_a_type_in_a_default_body_is_carried_in_the_generics_of_the_method() {
    let source = "trait Tr<T: ::core::default::Default> {
        fn f<U: ::core::default::Default>(&self) -> usize {
            let a: inner::Local = ::core::unimplemented!();
            let b: inner::Wrap<U> = ::core::unimplemented!();
            let c: inner::Wrap<T> = ::core::unimplemented!();
            let d: inner::Wrap<Self> = ::core::unimplemented!();
            ::core::mem::size_of::<inner::Wrap<U>>()
        }
    }";
    let (item, referrer, context) = f40_trait(source);
    let generics = |ty: &str| context_text(&referrer, entry_index(&referrer, ty));
    // The context of an entry is the generics it uses, and their bounds, as in a signature.
    assert_eq!(generics("inner :: Local"), "");
    assert_eq!(
        generics("inner :: Wrap < U >"),
        "< U : :: core :: default :: Default >"
    );
    assert_eq!(
        generics("inner :: Wrap < T >"),
        "< T : :: core :: default :: Default >"
    );
    // `Self` is the implementor of the trait.
    let with_self = entry_index(&referrer, "inner :: Wrap < Self >");
    assert!(matches!(
        common::entry(&referrer, with_self).self_ty.as_ref(),
        Some(SelfRequirement::Bound(_))
    ));
    assert_eq!(referrer.iter().count(), 4);
    // Every one is replaced where it stands. (`inner::Wrap<U>` twice, by one entry.)
    let ix = |ty: &str| entry_index(&referrer, ty);
    let expected: ItemTrait = parse_str(
        &"trait Tr<T: ::core::default::Default> {
            fn f<U: ::core::default::Default>(&self) -> usize {
                let a: @0 = ::core::unimplemented!();
                let b: @1 = ::core::unimplemented!();
                let c: @2 = ::core::unimplemented!();
                let d: @3 = ::core::unimplemented!();
                ::core::mem::size_of::<@1>()
            }
        }"
        .replace("@0", &format!("P<{}usize>", ix("inner :: Local")))
        .replace("@1", &format!("P<{}usize>", ix("inner :: Wrap < U >")))
        .replace("@2", &format!("P<{}usize>", ix("inner :: Wrap < T >")))
        .replace("@3", &format!("P<{}usize>", ix("inner :: Wrap < Self >"))),
    )
    .unwrap();
    assert_eq!(f40_copy(&referrer, context, &item), s(&expected));

    // The same in an impl: the body of a method, the value of a const, the generics of the impl.
    let item: ItemImpl = parse_quote! {
        impl<T: ::core::default::Default> ::p::Tr<T> for ::p::S<T> {
            const N: usize = ::core::mem::size_of::<inner::Local>();
            fn f(&self) -> usize {
                let w: inner::Wrap<T> = ::core::unimplemented!();
                ::core::mem::size_of::<inner::Wrap<Self>>()
            }
        }
    };
    let mut leaker = values_leaker();
    let context = leaker.intern_impl(&item);
    let referrer = leaker.result().finish().unwrap();
    // The contexts are numbered in the order they are met, and the first is that of the self type
    // `::p::S<T>`, which is the context of `inner::Wrap<T>` too. The types of another crate come
    // after the others.
    let entries: Vec<String> = common::type_texts(&referrer);
    assert_eq!(
        entries,
        [
            "inner :: Wrap < T >",
            "inner :: Local",
            "inner :: Wrap < Self >",
            ":: p :: S < T >"
        ]
    );
    let mut copy = item.clone();
    common::visitor(&referrer, context, common::placeholder).visit_item_impl_mut(&mut copy);
    let expected: ItemImpl = parse_quote! {
        impl<T: ::core::default::Default> ::p::Tr<T> for P<3usize> {
            const N: usize = ::core::mem::size_of::<P<1usize>>();
            fn f(&self) -> usize {
                let w: P<0usize> = ::core::unimplemented!();
                ::core::mem::size_of::<P<2usize>>()
            }
        }
    };
    assert_eq!(s(&copy), s(&expected));
}

/// F40: every type position that the Visitor walks in a value is read and replaced (a cast, a
/// `let`, a closure, the arguments of a path, a method call, a struct literal or a pattern, a
/// qualified path, a block, a branch, an array length).
#[test]
fn f40_every_type_position_of_a_value_is_replaced() {
    // The body, the entries in order of the `@k`, and the body the copy must have.
    let table: &[(&str, &[&str], &str)] = &[
        ("let _ = p as inner::C;", &["inner :: C"], "let _ = p as @0;"),
        (
            "let _: inner::A = ::core::unimplemented!();",
            &["inner :: A"],
            "let _: @0 = ::core::unimplemented!();",
        ),
        (
            "let c = |a: inner::A, b: u8| -> inner::B { ::core::unimplemented!() }; let _ = c;",
            &["inner :: A", "inner :: B"],
            "let c = |a: @0, b: u8| -> @1 { ::core::unimplemented!() }; let _ = c;",
        ),
        (
            "let _ = ::core::mem::size_of::<inner::Local>();",
            &["inner :: Local"],
            "let _ = ::core::mem::size_of::<@0>();",
        ),
        (
            "let _ = Vec::<Local>::new();",
            &["Local"],
            "let _ = Vec::<@0>::new();",
        ),
        (
            "let _ = ::core::iter::empty().collect::<Vec<inner::A>>();",
            &["inner :: A"],
            "let _ = ::core::iter::empty().collect::<Vec<@0>>();",
        ),
        (
            "let _ = <inner::A>::new(); let _ = <inner::B as ::core::default::Default>::default();",
            &["inner :: A", "inner :: B"],
            "let _ = <@0>::new(); let _ = <@1 as ::core::default::Default>::default();",
        ),
        (
            "let _ = <u8 as ::p::Tr<inner::A>>::f();",
            &["inner :: A"],
            "let _ = <u8 as ::p::Tr<@0>>::f();",
        ),
        (
            "let y = ::core::unimplemented!(); let _ = ::p::S::<inner::A> { x: 1 }; let ::p::S::<inner::B> { x } = y;",
            &["inner :: A", "inner :: B"],
            "let y = ::core::unimplemented!(); let _ = ::p::S::<@0> { x: 1 }; let ::p::S::<@1> { x } = y;",
        ),
        (
            "let _: ::core::option::Option<(inner::A, &inner::B)> = ::core::unimplemented!();",
            &["inner :: A", "inner :: B"],
            "let _: ::core::option::Option<(@0, &@1)> = ::core::unimplemented!();",
        ),
        (
            "let _: [inner::A; 2] = ::core::unimplemented!();",
            &["inner :: A"],
            "let _: [@0; 2] = ::core::unimplemented!();",
        ),
        (
            "let _ = [0u8; ::core::mem::size_of::<inner::A>()];",
            &["inner :: A"],
            "let _ = [0u8; ::core::mem::size_of::<@0>()];",
        ),
        (
            "if p > 0 { let _: inner::A = ::core::unimplemented!(); } else { match p { 0 => { let _ = p as inner::B; } _ => {} } }",
            &["inner :: A", "inner :: B"],
            "if p > 0 { let _: @0 = ::core::unimplemented!(); } else { match p { 0 => { let _ = p as @1; } _ => {} } }",
        ),
        (
            "let _: Box<dyn inner::Tr> = ::core::unimplemented!();",
            &["dyn inner :: Tr"],
            "let _: Box<@0> = ::core::unimplemented!();",
        ),
    ];
    for (body, entries, copy) in table {
        let source = format!("trait Tr<T> {{ fn f<U>(&self, p: u8) {{ {} }} }}", body);
        let (item, referrer, context) = f40_trait(&source);
        let found: Vec<String> = common::type_texts(&referrer);
        let mut sorted = found.clone();
        sorted.sort();
        let mut wanted: Vec<String> = entries.iter().map(|e| e.to_string()).collect();
        wanted.sort();
        assert_eq!(sorted, wanted, "{}", body);
        let mut text = copy.to_string();
        for (k, ty) in entries.iter().enumerate() {
            text = text.replace(
                &format!("@{}", k),
                &format!("P<{}usize>", entry_index(&referrer, ty)),
            );
        }
        let expected: ItemTrait = parse_str(&format!(
            "trait Tr<T> {{ fn f<U>(&self, p: u8) {{ {} }} }}",
            text
        ))
        .unwrap();
        assert_eq!(
            f40_copy(&referrer, context, &item),
            s(&expected),
            "{}",
            body
        );
    }
}

/// F40: an absolute type in a value is an entry as in a signature: one that holds a relative type
/// is one required entry, one that holds none an optional entry, which is no entry when the
/// Leaker's rule allows its crate.
#[test]
fn f40_an_absolute_type_of_a_value_is_an_optional_entry() {
    let source =
        "trait Tr { fn f(&self) { let _: ::p::Wrap<inner::Local> = ::core::unimplemented!(); } }";
    let (item, result, context) = f40_result(source);
    let referrer = result.clone().finish().unwrap();
    assert_eq!(
        entries_text(&referrer, true),
        [":: p :: Wrap < inner :: Local >"]
    );
    let expected: ItemTrait = parse_quote!(
        trait Tr {
            fn f(&self) {
                let _: P<0usize> = ::core::unimplemented!();
            }
        }
    );
    let referrer = result.finish().unwrap();
    assert_eq!(f40_copy(&referrer, context, &item), s(&expected));
    // One with no relative type in it is an optional entry.
    let source = "trait Tr { fn f(&self) { let _: ::p::Wrap<()> = ::core::unimplemented!(); } }";
    let (item, result, context) = f40_result(source);
    let referrer = result.clone().finish().unwrap();
    assert_eq!(entries_text(&referrer, true), [":: p :: Wrap < () >"]);
    assert_eq!(f40_copy(&referrer, context, &item), s(&expected));
    // With the crate allowed by the rule of the Leaker it is no entry: the copy writes it as it is.
    let mut leaker = Leaker::with_rule(Rules::new().primitive().prelude().external_crates());
    leaker.check_values = true;
    leaker.check_bodies = true;
    let context = leaker.intern_trait(&item);
    let referrer = leaker.result().finish().unwrap();
    assert!(referrer.is_empty());
    assert_eq!(f40_copy(&referrer, context, &item), s(&item));
}

/// F40, the nested trait: the `Self` of a trait nested in a scope is its own, and an entry of a
/// value that names it is written by `with_nested_self()`, as one of a signature is.
#[test]
fn f40_a_nested_trait_writes_the_entry_that_names_its_self() {
    let item: Item = parse_quote!(
        pub trait Tr {
            const N: usize = ::core::mem::size_of::<inner::Local>();
            fn f(&self) -> usize {
                let w: inner::Wrap<Self> = ::core::unimplemented!();
                ::core::mem::size_of::<inner::Local>()
            }
        }
    );
    for nest in NESTS {
        let (copy, referrer, calls) =
            nested_run(values_leaker(), nest, &item, (false, false), true);
        assert_eq!(calls, ["Self: Tr"], "{:?}", nest);
        let local = entry_index(&referrer, "inner :: Local");
        let wrap = entry_index(&referrer, "inner :: Wrap < Self >");
        let expected: Item = parse_str(
            &"pub trait Tr {
                const N: usize = ::core::mem::size_of::<P<@0>>();
                fn f(&self) -> usize {
                    let w: P<@1> = ::core::unimplemented!();
                    ::core::mem::size_of::<P<@0>>()
                }
            }"
            .replace("@0", &format!("{}usize", local))
            .replace("@1", &format!("{}usize", wrap)),
        )
        .unwrap();
        assert_eq!(s(&copy), s(&expected), "{:?}", nest);
    }
}

/// F40, the controls: a path that is no type, a type that names what the value declares, a type
/// with `_`, the tokens of a macro and an item nested in a body are not leaked whole. `Self` in a
/// trait's body is the implementor; in a function nested in a scope it has no meaning.
#[test]
fn f40_what_is_not_a_type_position_stays_rejected() {
    let rejected = |path: &str| format!("use absolute path: '{}'", plain(path));
    let value = "it is in the default body of `f`, which the copy writes as it is";
    // Not a type: a function, an associated function, a struct literal, a variant, a constant,
    // the trait of a qualified path.
    for (body, path) in [
        ("let _ = inner::f();", "inner :: f"),
        ("let _ = Local::new();", "Local :: new"),
        ("let _ = Local { a: 1 };", "Local"),
        ("let _ = inner::E::V;", "inner :: E :: V"),
        ("let _ = inner::X + 1;", "inner :: X"),
        ("let _ = <u8 as inner::Tr>::f();", "inner :: Tr"),
        (
            "let _ = Wrap2::<inner::A>::new();",
            "Wrap2 :: < inner :: A > :: new",
        ),
    ] {
        let source = format!("trait Tr {{ fn f(&self) {{ {} }} }}", body);
        assert_eq!(
            f40_errors(&source),
            [(rejected(path), value.to_string())],
            "{}",
            body
        );
    }
    // A type that names something the value declares is read, not leaked: the relative path in
    // it is rejected, and the note says why. Without one it is fine, and leaks nothing.
    let declared = format!(
        "{}; the type names `Loc`, which the value declares, so it is not leaked",
        value
    );
    assert_eq!(
        f40_errors("trait Tr { fn f(&self) { struct Loc; let _: ::p::Pair<Loc, inner::A> = ::core::unimplemented!(); } }"),
        [(rejected("inner :: A"), declared.clone())]
    );
    assert_eq!(
        f40_errors("trait Tr { fn f(&self) { use ::p::Loc; let _: ::p::Pair<Loc, inner::A> = ::core::unimplemented!(); } }"),
        [(rejected("inner :: A"), declared)]
    );
    assert_eq!(
        f40_errors("trait Tr { fn f(&self) { let _ = for<'a> |x: &'a inner::A| x; } }"),
        [(
            rejected("inner :: A"),
            format!(
                "{}; the type names `'a`, which the value declares, so it is not leaked",
                value
            )
        )]
    );
    for body in [
        "struct Loc; let _: Loc = ::core::unimplemented!();",
        "struct Loc; let _: ::p::Pair<Loc, u8> = ::core::unimplemented!();",
        "fn id<V>(v: V) -> V { let w: V = v; w } let _ = id(1);",
        "struct Loc<'a>(&'a u8); let _: Loc<'_> = ::core::unimplemented!();",
    ] {
        let source = format!("trait Tr {{ fn f(&self) {{ {} }} }}", body);
        let (_, referrer, _) = f40_trait(&source);
        assert!(referrer.is_empty(), "{}", body);
    }
    // The tokens of a macro are not walked by the Visitor.
    assert_eq!(
        f40_errors("trait Tr { fn f(&self) { ::core::assert_eq!(::core::mem::size_of::<inner::A>(), 4); } }"),
        [(rejected("inner :: A"), value.to_string())]
    );
    // The types of an item that the body declares are the item's, not the body's.
    assert_eq!(
        f40_errors("trait Tr { fn f(&self) { fn g<V>() { let _: inner::Wrap<V> = ::core::unimplemented!(); } } }"),
        [(rejected("inner :: Wrap < V >"), value.to_string())]
    );
    assert_eq!(
        f40_errors("trait Tr { fn f(&self) { struct L(inner::A); } }"),
        [(rejected("inner :: A"), value.to_string())]
    );
    // `_` is inferred where it is written, so no impl of the Repeater can hold it.
    let (_, referrer, _) = f40_trait(
        "trait Tr { fn f(&self, p: u8) { let _: Vec<_> = ::core::unimplemented!(); let _: ::p::Wrap<_> = ::core::unimplemented!(); let _ = p as _; } }",
    );
    assert!(referrer.is_empty());
    let (_, referrer, _) = f40_trait(
        "trait Tr { fn f(&self) { let _: ::p::Wrap<inner::A, _> = ::core::unimplemented!(); } }",
    );
    assert_eq!(common::type_texts(&referrer), ["inner :: A"]);
    let errors =
        f40_errors("trait Tr { fn f(&self) { let _: Rel<_> = ::core::unimplemented!(); } }");
    assert_eq!(errors.len(), 1);
    assert_eq!(errors[0].0, rejected("Rel < _ >"));
    assert_eq!(errors[0].1, "it cannot be leaked as a whole because of `_`");
    // A lifetime that is elided: the type that has it is not leaked whole, as in a signature.
    let errors =
        f40_errors("trait Tr { fn f(&self) { let _: Rel<'_> = ::core::unimplemented!(); } }");
    assert_eq!(errors[0].0, rejected("Rel < '_ >"));
    let (_, referrer, _) =
        f40_trait("trait Tr { fn f(&self) { let _: &inner::A = ::core::unimplemented!(); } }");
    assert_eq!(common::type_texts(&referrer), ["inner :: A"]);
    // `Self` in a trait's body is the implementor, as in a signature.
    let item: ItemTrait = parse_quote!(
        trait Tr {
            fn f(&self) {
                let _: inner::A<Self> = ::core::unimplemented!();
            }
        }
    );
    let mut leaker = values_leaker();
    leaker.intern_trait(&item);
    let result = leaker.result();
    assert!(result.finish().is_ok());
    // Without a trait: `Self` has no meaning in a function nested in a scope.
    let item: Item = parse_quote!(
        pub fn f() {
            let _: inner::A<Self> = ::core::unimplemented!();
        }
    );
    let mut leaker = values_leaker();
    leaker.enter_generics(&Generics::default(), |view| view.visit_item(&item));
    let result = leaker.result();
    let err = result.errors();
    assert!(!err.is_empty());
    assert!(err
        .iter()
        .all(|item| item.reason == Reason::SelfWithoutMeaning));
}

/// F40: a type of a body that names a lifetime of the method is an entry for it. The body is no
/// input of the signature, so leaking the type does not leave the lifetime unconstrained (E0195).
#[test]
fn f40_a_lifetime_of_the_method_in_a_type_of_its_body() {
    let source = "trait Tr {
        fn f<'a>(&self, x: &'a u8) -> &'a u8 {
            let w: inner::Lt<'a> = ::core::unimplemented!();
            let r: &'a inner::A = ::core::unimplemented!();
            x
        }
    }";
    let (item, referrer, context) = f40_trait(source);
    assert_eq!(referrer.iter().count(), 2);
    let (lt, a) = (
        entry_index(&referrer, "inner :: Lt < 'a >"),
        entry_index(&referrer, "inner :: A"),
    );
    // `'a` is a lifetime of the method: the entry is for it.
    assert_eq!(context_text(&referrer, lt), "< 'a >");
    assert_eq!(context_text(&referrer, a), "");
    let expected: ItemTrait = parse_str(
        &"trait Tr {
            fn f<'a>(&self, x: &'a u8) -> &'a u8 {
                let w: P<@0> = ::core::unimplemented!();
                let r: &'a P<@1> = ::core::unimplemented!();
                x
            }
        }"
        .replace("@0", &format!("{}usize", lt))
        .replace("@1", &format!("{}usize", a)),
    )
    .unwrap();
    assert_eq!(f40_copy(&referrer, context, &item), s(&expected));
}

/// F40: a function that a scope visits (`enter_generics()` and `View::visit_item()`) has its body
/// read too.
#[test]
fn f40_a_function_visited_through_a_scope() {
    let item: Item = parse_quote! {
        fn f<T: ::core::default::Default>() -> usize {
            let x: inner::Wrap<T> = ::core::unimplemented!();
            ::core::mem::size_of::<inner::Local>()
        }
    };
    let mut leaker = values_leaker();
    let (context, ()) = leaker.enter_generics(&parse_quote!(<>), |view| view.visit_item(&item));
    let referrer = leaker.result().finish().unwrap();
    let mut found: Vec<String> = common::type_texts(&referrer);
    found.sort();
    assert_eq!(found, ["inner :: Local", "inner :: Wrap < T >"]);
    let wrap = entry_index(&referrer, "inner :: Wrap < T >");
    let local = entry_index(&referrer, "inner :: Local");
    let expected: Item = parse_str(
        &"fn f<T: ::core::default::Default>() -> usize {
            let x: P<@0> = ::core::unimplemented!();
            ::core::mem::size_of::<P<@1>>()
        }"
        .replace("@0", &format!("{}usize", wrap))
        .replace("@1", &format!("{}usize", local)),
    )
    .unwrap();
    // The copy is the function of a body, nested in the scope: its generics are its own.
    assert_eq!(rewrite(&referrer, context, item), s(&expected));
}

// ── F41 ──────────────────────────────────────────────────────────────────────

/// The items of F41, with the name of the item and the generics of the context of the entry in
/// the default: a struct, an enum or a union nested in a scope, whose default (`U = Wrap<T>`) uses a
/// generic with a bound that names the item's `Self`. The bounds that name `Self` are not in the
/// context; the others are (`Clone`).
fn f41_items() -> Vec<(Item, &'static str)> {
    vec![
        (
            parse_quote!(
                pub struct S<T: ::core::convert::AsRef<Self>, U = Wrap<T>>(pub T, pub U);
            ),
            "< T >",
        ),
        (
            parse_quote!(
                pub struct S<T: Clone + ::core::convert::AsRef<Self>, U = Wrap<T>>(pub T, pub U);
            ),
            "< T : Clone >",
        ),
        // The bound is a where predicate, and names `Self` among other types.
        (
            parse_quote!(
                pub struct S<T, U = Wrap<T>>(pub T, pub U)
                where
                    T: Clone + ::core::convert::AsRef<(Self, u8)>;
            ),
            "< T : Clone >",
        ),
        (
            parse_quote!(
                pub enum E<T: Clone + ::core::convert::AsRef<Self>, U = Wrap<T>> {
                    A(T),
                    B(U),
                }
            ),
            "< T : Clone >",
        ),
        (
            parse_quote!(
                pub union Un<T: Copy + ::core::convert::AsRef<Self>, U: Copy = Wrap<T>> {
                    a: ::core::mem::ManuallyDrop<T>,
                    b: ::core::mem::ManuallyDrop<U>,
                }
            ),
            "< T : Copy >",
        ),
        // A lifetime, and `Self: 'a`, which is not in the context either.
        (
            parse_quote!(
                pub struct S<'a, T: ::core::convert::AsRef<Self> + 'a, U = Wrap<&'a T>>(
                    pub &'a T,
                    pub U,
                );
            ),
            "< 'a , T : 'a >",
        ),
        (
            parse_quote!(
                pub struct S<'a, T: ::core::convert::AsRef<Self>, U = Wrap<&'a T>>(
                    pub &'a T,
                    pub U,
                )
                where
                    Self: 'a;
            ),
            "< 'a , T : 'a >",
        ),
    ]
}

/// F41 (fixed): a nested struct, enum or union whose default uses a generic with a bound that
/// names `Self` was rejected, though rustc accepts it.
///
/// Trigger: `pub struct S<T: Clone + AsRef<Self>, U = Wrap<T>>(pub T, pub U);` nested in a scope:
/// "`Wrap < T >` is in the default of a parameter, and a bound of its generics names `Self`".
///
/// Cause: by F37 the entry is for any type that the bound holds for, and the copy passes its own
/// type; but a copy cannot write `Self` in a default (E0735).
///
/// Fix: rustc checks a default where it is used, so its entry needs no `Self`: the context is the
/// generics before it without the bounds that name `Self` (`T: Clone`), the impl is `impl<T:
/// Clone, __S: ?Sized> Repeater<(T,), KEY> for __S`, and the copy writes an ordinary default.
///
/// Limitation: if `Wrap` needs a bound that only a dropped bound implies (`struct Wrap<A:
/// Iterator>` with `T: Iterator<Item = Self>`), the generated impl fails with E0277.
#[test]
fn f41_a_default_that_needs_self_only_through_a_bound_is_accepted() {
    let param: Ident = parse_quote!(__S);
    let marker: Type = parse_quote!(crate::Marker);
    for (item, generics) in &f41_items() {
        let what = format!("`{}`", s(item));
        let result = nested_analysis(item);
        let referrer = assert_accepted_as(&what, result.clone().finish());
        assert_eq!(referrer.iter().count(), 1, "{}", what);
        // The entry has no `Self`, and none of the bounds that name it.
        assert!(common::entry(&referrer, 0).self_ty.is_none(), "{}", what);
        assert_eq!(context_text(&referrer, 0), *generics, "{}", what);
        // Its impl is for any type, with no bound of its own.
        let r = result
            .repeater_impls(&Projection::new(
                repeater(),
                7,
                ImplSelf::Param(param.clone()),
            ))
            .unwrap()
            .remove(0);
        assert_eq!(r.self_ty, parse_quote!(__S), "{}", what);
        assert_eq!(
            r.generics.params.last().unwrap(),
            &parse_quote!(__S: ?::core::marker::Sized),
            "{}",
            what
        );
        assert!(!s(&r.generics).contains("Self"), "{}", what);
        // A marker will do as well.
        result
            .repeater_impls(&Projection::new(
                repeater(),
                7,
                ImplSelf::Marker(marker.clone()),
            ))
            .unwrap_or_else(|e| panic!("{}: {}", what, e))
            .remove(0);
        // The serialized form keeps it.
        let again: Referrer = parse2(quote!(#referrer)).unwrap();
        assert_eq!(again, referrer, "{}", what);
        assert!(common::entry(&again, 0).self_ty.is_none(), "{}", what);
    }
    // The impl as a whole, for the item of the report.
    let item: Item = parse_quote!(
        pub struct S<T: Clone + ::core::convert::AsRef<Self>, U = Wrap<T>>(pub T, pub U);
    );
    let result = nested_analysis(&item);
    result.clone().finish().unwrap();
    let imp = repeater_impl_item(&result, 0, 7);
    let key = key_of(&result, 7, 0);
    let expected: ItemImpl = parse_quote! {
        impl<T: Clone, __S: ?::core::marker::Sized> crate::Repeater<(T,), #key> for __S {
            type Type = Wrap<T>;
        }
    };
    assert_eq!(imp, expected);
}

/// F41: the Visitor finds the entry of the default without the bounds that name `Self`; the copy
/// keeps the bound and writes an ordinary default (`U = <() as Repeater<(T,), KEY>>::Type`).
#[test]
fn f41_the_copy_writes_an_ordinary_default() {
    for (item, _) in &f41_items() {
        for nest in NESTS {
            let what = format!("{:?} `{}`", nest, s(item));
            let (copy, referrer, calls) = nested_copy_logged(nest, item);
            // The entry of a default is a plain one: the closure of `with_nested_self()` is not asked.
            assert!(calls.is_empty(), "{}: {:?}", what, calls);
            let ty = s(common::types(&referrer).next().unwrap());
            assert_eq!(s(&copy), s(item).replace(&ty, "P < 0usize >"), "{}", what);
        }
    }

    // The closure of a macro writes the path through the Repeater with `()` as the implementor.
    let item: Item = parse_quote!(
        pub struct S<T: Clone + ::core::convert::AsRef<Self>, U = Wrap<T>>(pub T, pub U);
    );
    let mut leaker = common::std_leaker();
    let (context, ()) = leaker.enter_generics(&Generics::default(), |view| view.visit_item(&item));
    let referrer = leaker.result().finish().unwrap();
    let mut module: ItemMod = parse_quote!(mod w { #item });
    let repeater: Path = parse_quote!(Repeater);
    common::visitor(&referrer, context, |_, ix, context| {
        common::projection(repeater.clone(), 7)
            .of(&parse_quote!(()), context, ix)
            .unwrap()
    })
    .visit_item_mod_mut(&mut module);
    let copy = module.content.unwrap().1.remove(0);
    // (Parsed from text: a formatter would add a comma after the last generic.)
    let expected: Item = parse_str(&format!(
        "pub struct S<T: Clone + ::core::convert::AsRef<Self>, \
         U = <() as Repeater<(T,), {}>>::Type>(pub T, pub U);",
        s(&common::key(7, common::entry_id_at(&referrer, 0)))
    ))
    .unwrap();
    assert_eq!(s(&copy), s(&expected));
}

/// F41: a default and a field of the same type (`Wrap<T>`) are two entries, one without `Self` and
/// one for any type (`SelfRequirement::Any`); a second default that names the first (`V =
/// Pair<Rel, U>`) has its own entry, with the generic `U` only.
#[test]
fn f41_the_default_and_the_field_have_their_own_entries() {
    let item: Item = parse_quote!(
        pub struct S<T: ::core::convert::AsRef<Self>, U = Wrap<T>, V = Pair<Rel, U>>(
            pub T,
            pub U,
            pub V,
            pub Wrap<T>,
        );
    );
    for nest in NESTS {
        let what = format!("{:?}", nest);
        let (copy, referrer) = nested_copy(nest, &item);
        let wrap = entries_of(&referrer, "Wrap < T >");
        assert_eq!(wrap.len(), 2, "{}: {:?}", what, wrap);
        let default = wrap.iter().find(|(_, has_self)| !has_self).unwrap().0;
        let field = wrap.iter().find(|(_, has_self)| *has_self).unwrap().0;
        let pair = entry_index(&referrer, "Pair < Rel , U >");
        assert!(common::entry(&referrer, pair).self_ty.is_none(), "{}", what);
        assert_eq!(context_text(&referrer, pair), "< U >", "{}", what);
        assert!(
            matches!(
                common::entry(&referrer, field).self_ty.as_ref(),
                Some(SelfRequirement::Any)
            ),
            "{}",
            what
        );
        let expected: Item = parse_quote!(
            pub struct S<T: ::core::convert::AsRef<Self>, U = P<#default>, V = P<#pair>>(
                pub T,
                pub U,
                pub V,
                pub P<#field>,
            );
        );
        assert_eq!(s(&copy), s(&expected), "{}", what);
    }
}

/// F41, the control: a bound that names no `Self` stays in the context of the default. (Allowing
/// `NestedSelfInBounds` changes nothing for the items of F41: `f42_the_controls_are_unchanged`.
/// A nested trait keeps the bound that names `Self` in the context of its default: F39.)
#[test]
fn f41_the_controls_are_unchanged() {
    // No bound names `Self`: the entry has the bound, as ever.
    let item: Item = parse_quote!(
        pub struct S<T: Clone, U = Wrap<T>>(pub T, pub U);
    );
    let referrer = nested_analysis(&item).finish().unwrap();
    assert_eq!(entries_of(&referrer, "Wrap < T >"), [(0, false)]);
    assert_eq!(context_text(&referrer, 0), "< T : Clone >");
}

// ── F42 ──────────────────────────────────────────────────────────────────────

/// The items of F42, with the name of the item, the type of the entry, and its Repeater's impl (with
/// `KEY` for the key of the entry): the items of `f37_rejected()` and the others that F37 rejected.
/// The entry is for any type that the bounds hold for; `Self` is the parameter in the type too, and
/// the generics of the item that the type does not use are not in the impl (`U`).
fn f42_items() -> Vec<(Item, &'static str, &'static str)> {
    vec![
        (
            parse_quote!(
                pub struct S<T: ::core::convert::AsRef<Self>> {
                    pub b: Rel2<Self, T>,
                }
            ),
            "Rel2 < Self , T >",
            "impl<T: ::core::convert::AsRef<__S>, __S> crate::Repeater<(T,), KEY> for __S {
                type Type = Rel2<__S, T>;
            }",
        ),
        (
            parse_quote!(
                pub enum E<T: ::core::convert::AsRef<Self>> {
                    A(Rel<Self>),
                    B(T),
                }
            ),
            "Rel < Self >",
            "impl<__S> crate::Repeater<(), KEY> for __S {
                type Type = Rel<__S>;
            }",
        ),
        (
            parse_quote!(
                pub struct S<T>(Vec<Rel2<Self>>, T)
                where
                    T: ::core::convert::AsRef<(Self, u8)>;
            ),
            "Rel2 < Self >",
            "impl<__S> crate::Repeater<(), KEY> for __S {
                type Type = Rel2<__S>;
            }",
        ),
        (
            parse_quote!(
                pub union Un<T: Copy + ::core::convert::AsRef<Self>> {
                    a: ::core::mem::ManuallyDrop<Rel2<Self, T>>,
                }
            ),
            "Rel2 < Self , T >",
            "impl<T: ::core::convert::AsRef<__S> + Copy, __S> crate::Repeater<(T,), KEY> for __S {
                type Type = Rel2<__S, T>;
            }",
        ),
        // The generic `U`, which the type does not use, is not in the impl.
        (
            parse_quote!(
                pub struct S<T: ::core::convert::AsRef<Self>, U> {
                    pub b: Rel2<Self, T>,
                    pub c: U,
                }
            ),
            "Rel2 < Self , T >",
            "impl<T: ::core::convert::AsRef<__S>, __S> crate::Repeater<(T,), KEY> for __S {
                type Type = Rel2<__S, T>;
            }",
        ),
        // A predicate on `Self` is a predicate on the parameter.
        (
            parse_quote!(
                pub enum E<T: ::core::convert::AsRef<Self>>
                where
                    Self: Sized,
                {
                    A(Rel<Self>),
                    B(T),
                }
            ),
            "Rel < Self >",
            "impl<__S> crate::Repeater<(), KEY> for __S where __S: Sized {
                type Type = Rel<__S>;
            }",
        ),
    ]
}

/// F42 (fixed): a nested struct, enum or union whose bound and type both name its `Self` was
/// rejected.
///
/// Trigger: `struct S<T: AsRef<Self>> { b: Rel2<Self, T> }` nested in a scope: "has a generic
/// whose bound mentions `Self`, which cannot be leaked".
///
/// Cause: an entry for the Leaker's type fails in the copy (E0277): there the bound holds for the
/// copy's own type only.
///
/// Fix: the entry is for any type that the bound holds for (`SelfRequirement::Any`), with `Self`
/// the impl's parameter in the type too (`impl<T: AsRef<__S>, __S> .. for __S { type Type =
/// Rel2<__S, T>; }`); the copy passes its own `Self`. `__S` is `?Sized` only where nothing needs
/// `Sized`. The entry has the risk error `NestedSelfInBounds`, which `allow()` drops. Risks: a
/// bound on the parameter of the leaked type that gets `Self` (`struct Rel2<A: Clone, B>`) fails
/// at the impl (E0277); rustc before 1.70 gives E0275 for the entry in a struct's last field.
#[test]
fn f42_a_type_that_names_self_with_a_bound_that_names_it_is_an_entry_with_a_risk() {
    for (item, ty, expected) in &f42_items() {
        let result = accept_nested_self(nested_analysis(item));
        let (ix, _) = assert_any_entry(item, &result, ty);
        // The impl: `Self` is the parameter in the type and in the bounds. It is `Sized`
        // (not `?Sized`), as the type needs it.
        let imp = repeater_impl_item(&result, ix, 7);
        let key = s(&common::key(
            7,
            common::entry_id_at(&result.clone().finish().unwrap(), ix),
        ));
        let expected: ItemImpl = syn::parse_str(&expected.replace("KEY", &key)).unwrap();
        assert_eq!(s(&imp), s(&expected), "`{}`", s(item));
    }
}

/// F42: the Visitor finds the entry in the copy of the nested item and writes it with the copy's
/// own `Self` as the implementor.
#[test]
fn f42_the_copy_passes_its_own_type() {
    for (item, ty, _) in &f42_items() {
        for nest in NESTS {
            let what = format!("{:?} `{}`", nest, s(item));
            let (copy, referrer, calls) =
                nested_run(common::std_leaker(), nest, item, (false, true), true);
            // The entry is for any type: it needs no closure.
            assert!(calls.is_empty(), "{}: {:?}", what, calls);
            let ix = entry_index(&referrer, ty);
            // The type of the entry becomes `P<ix>`, wherever it is. (`Vec<Rel2<Self>>` keeps
            // the vector.)
            assert_eq!(
                s(&copy),
                s(item).replace(ty, &format!("P < {}usize >", ix)),
                "{}",
                what
            );
        }
    }

    // The visitor writes the path through the Repeater with `Self`: the field is the type of the
    // original, with the copy's own `S` for the first argument.
    let item: Item = parse_quote!(
        pub struct S<T: ::core::convert::AsRef<Self>> {
            pub b: Rel2<Self, T>,
        }
    );
    let mut leaker = common::std_leaker();
    let (context, ()) = leaker.enter_generics(&Generics::default(), |view| view.visit_item(&item));
    let referrer = accept_nested_self(leaker.result()).finish().unwrap();
    let copy = macro_copy(&referrer, context, &item, 7);
    let key = common::key(7, common::entry_id_at(&referrer, 0));
    let expected: Item = parse_quote!(
        pub struct S<T: ::core::convert::AsRef<Self>> {
            pub b: <Self as Repeater<(T,), #key>>::Type,
        }
    );
    assert_eq!(s(&copy.items[0]), s(&expected));
}

/// F42: the impl of a risk entry is `?Sized` only if every `Self` in the type is directly behind a
/// reference or raw pointer (`&'static Self`) or an argument that `std` takes unsized
/// (`Box<Self>`); `Self` in `&Wrap<Self>`, a slice or a macro keeps `Sized`.
#[test]
fn f42_the_impl_of_a_risk_entry_is_sized_as_the_uses_of_self_in_its_type_need() {
    let cases: [(&str, bool); 11] = [
        ("Rel2<Self, T>", true),
        ("Rel2<&'static Self, T>", false),
        ("Rel2<*const Self, T>", false),
        ("Rel2<*mut Self, T>", false),
        ("Rel2<Box<Self>, T>", false),
        ("Rel2<(&'static Self, u8), T>", false),
        ("Rel2<Wrap<*const Self>, T>", false),
        // A reference or a pointer to another type that names `Self`, a slice, a macro.
        ("Rel2<&'static Wrap<Self>, T>", true),
        ("Rel2<*const Wrap<Self>, T>", true),
        ("Rel2<&'static [Self], T>", true),
        ("Rel2<m!(Self), T>", true),
    ];
    let unsized_param: GenericParam = parse_quote!(__S: ?::core::marker::Sized);
    let sized_param: GenericParam = parse_quote!(__S);
    for (ty, needs_sized) in cases {
        let ty: Type = parse_str(ty).unwrap();
        let item: Item = parse_quote!(
            pub struct S<T: ::core::convert::AsRef<Self>>(pub u8, pub #ty);
        );
        let result = accept_nested_self(nested_analysis(&item));
        let referrer = result.clone().finish().unwrap();
        let ix = entry_index(&referrer, &s(&ty));
        assert!(
            matches!(
                common::entry(&referrer, ix).self_ty.as_ref(),
                Some(SelfRequirement::Any)
            ),
            "`{}`",
            s(&ty)
        );
        let imp = repeater_impl_item(&result, ix, 7);
        assert_eq!(
            imp.generics.params.last().unwrap(),
            if needs_sized {
                &sized_param
            } else {
                &unsized_param
            },
            "`{}`: {}",
            s(&ty),
            s(&imp.generics)
        );
    }
}

/// F42, the controls: with the risk allowed, a type that names `Self` under bounds that do not is
/// for the Leaker's type (`SelfRequirement::Type`); what F37 and F41 accepted, and the nested items
/// that name `Self` in no bound that the environment proves, are accepted with entries and give the
/// same Referrer; an absolute type that names `Self` is copied as written; and `Self::A` in a bound
/// is rejected.
#[test]
fn f42_the_controls_are_unchanged() {
    // The bound names no `Self`: the entry is for the Leaker's type, whose impl has the bounds.
    let item: Item = parse_quote!(
        pub struct S<T: Clone, U> {
            pub b: Rel2<Self, T>,
            pub c: U,
        }
    );
    let referrer = accept_nested_self(nested_analysis(&item)).finish().unwrap();
    let ix = entry_index(&referrer, "Rel2 < Self , T >");
    assert!(matches!(
        common::entry(&referrer, ix).self_ty.as_ref(),
        Some(SelfRequirement::Type(_))
    ),);
    let plain = nested_analysis(&item);
    assert_eq!(plain.finish().unwrap(), referrer);

    // Everything that was accepted gives the same Referrer: the entries for any type of F37, the
    // defaults of F41, the nested ones that name `Self` with no bound that does, and the others.
    let accepted: Vec<Item> = f37_supported()
        .into_iter()
        .chain(f41_items())
        .map(|(item, _)| item)
        .chain(vec![
            parse_quote!(
                pub struct S<T: Clone>(Rel<Self>, T);
            ),
            // A bound on `Self`, or on a type that an impl proves.
            parse_quote!(
                pub struct S<T>(Rel<Self>, T)
                where
                    Self: Sized;
            ),
            parse_quote!(
                pub struct S<T>(Rel<T>, Vec<Self>)
                where
                    Vec<Self>: Clone;
            ),
            // The generic whose bound names `Self` is not used by the entry.
            parse_quote!(
                pub struct S<T: ::core::convert::AsRef<Self>, U>(Rel<U>, T);
            ),
            // `Self: 'a` is encoded by the Repeater's impl of the Leaker's type.
            parse_quote!(
                pub struct S<'a, T: 'a>(&'a Rel<Self>, &'a T);
            ),
            // A trait: its `Self` is the implementor, which the copy's impl for a host writes.
            parse_quote!(
                pub trait Tr<T: ::core::convert::AsRef<Self>> {
                    fn f(&self, x: Rel<Self>) -> Rel<T>;
                }
            ),
            parse_quote!(
                pub trait Tr<T: ::core::convert::AsRef<Self>> {
                    fn f(&self) -> Rel<T>;
                }
            ),
        ])
        .collect();
    for item in &accepted {
        let plain = assert_accepted(item, nested_analysis(item).finish());
        assert!(plain.iter().count() > 0, "`{}`", s(item));
        let option = accept_nested_self(nested_analysis(item));
        assert_eq!(plain, option.finish().unwrap(), "`{}`", s(item));
    }

    // The item itself: `Self` is written as the Leaker's type, which has the bound: no risk.
    for (item, _, _) in &f42_items() {
        let mut leaker = common::std_leaker();
        common::intern_item(&mut leaker, item);
        let plain = leaker.result().finish().unwrap();
        assert!(
            (0..plain.iter().count()).all(|ix| !matches!(
                common::entry(&plain, ix).self_ty.as_ref(),
                Some(SelfRequirement::Any)
            )),
            "`{}`",
            s(item)
        );
    }

    // An absolute type that names `Self` is no entry, with the risk accepted too: it is copied as
    // written.
    let item: Item = parse_quote!(
        pub struct S<T: ::core::convert::AsRef<Self>> {
            pub b: ::p::Rel2<Self, T>,
        }
    );
    for nest in NESTS {
        let (copy, referrer) = nested_copy_accepting(nest, &item);
        assert_eq!(referrer.iter().count(), 0, "{:?}", nest);
        assert_eq!(s(&copy), s(&item), "{:?}", nest);
    }

    // Accepting the risk does not accept `Self::A` in a bound, whose impl has no trait to take `A`
    // from.
    let item: Item = parse_quote!(
        pub struct S<T: ::core::iter::Iterator<Item = Self::A>> {
            pub b: Rel2<Self, T>,
        }
    );
    let result = accept_nested_self(nested_analysis(&item));
    assert!(result.finish().is_err());
}

// ── F43 ──────────────────────────────────────────────────────────────────────

/// The Referrer that the serialized `parts` give, with the generics `generics` after the list.
fn read_parts(parts: &str, generics: &str) -> std::result::Result<Referrer, String> {
    let tag = format!("\"type-leak-format-{}\"", Referrer::FORMAT_VERSION);
    parse_str::<Referrer>(&format!("{} [{}] 2 (Rel) {} ; ()", tag, parts, generics))
        .map_err(|e| e.to_string())
}

/// Asserts that each of `parts` is read, read back to the same Referrer after it is written, and
/// written again as it was.
fn assert_parts_read_back(parts: &[&str], generics: &str) {
    for parts in parts {
        let referrer = read_parts(parts, generics).unwrap_or_else(|e| panic!("{}: {}", parts, e));
        let written = parts_of(&referrer);
        let again: Referrer = parse2(quote!(#referrer)).unwrap();
        assert_eq!(again, referrer, "{}", parts);
        assert_eq!(parts_of(&again), written, "{}", parts);
        assert_eq!(
            written.join(" , ").replace(' ', ""),
            parts.replace(' ', ""),
            "{}",
            parts
        );
    }
}

/// Asserts that each of `parts` is refused, with an error that has the message.
fn assert_parts_refused(rows: &[(&str, &str)], generics: &str) {
    for (parts, message) in rows {
        let error = read_parts(parts, generics)
            .err()
            .unwrap_or_else(|| panic!("read: {}", parts));
        assert!(error.contains(message), "{}: {}", parts, error);
    }
}

/// The Referrer without the parts it carries: what the visitor did before the parts were known.
fn without_parts(referrer: &Referrer) -> Referrer {
    use proc_macro2::{Delimiter, TokenTree};
    let tokens: Vec<TokenTree> = template_quote::ToTokens::to_token_stream(referrer)
        .into_iter()
        .filter(
            |token| !matches!(token, TokenTree::Group(g) if g.delimiter() == Delimiter::Bracket),
        )
        .collect();
    parse2(tokens.into_iter().collect()).unwrap()
}

/// The scope of F43: the trait `Tr<T: AsRef<Self>>` (a bound of `T` names the `Self` of the trait),
/// with `direct` visited in it as types, and `nested` an item nested in it
/// (`View::visit_item()`), which has a `Self` of its own.
fn f43_analyze<A: Allow>(
    config: A,
    check_values: bool,
    direct: &[Type],
    nested: &Item,
) -> (LeakerResult, ContextId) {
    let mut leaker = Leaker::with_rule(config);
    leaker.check_values = check_values;
    leaker.check_bodies = check_values;
    let generics: Generics = parse_quote!(<T: AsRef<Self>>);
    let (context, ()) = leaker.enter_trait(&parse_quote!(Tr), &generics, |view| {
        for ty in direct {
            view.visit_type(ty);
        }
        view.visit_item(nested);
    });
    (leaker.result(), context)
}

/// The copy of the scope of F43: the trait, with `direct` as the arguments of a method, and the
/// nested item in the body of another (the copy nests the items as the Leaker did).
fn f43_copy(referrer: &Referrer, context: ContextId, direct: &[Type], nested: &Item) -> String {
    let inputs = direct.iter().enumerate().map(|(i, ty)| {
        let name = Ident::new(&format!("x{}", i), proc_macro2::Span::call_site());
        quote!(#name: #ty)
    });
    let mut copy: ItemTrait = parse_quote!(trait Tr<T: AsRef<Self>> {
        fn direct(&self, #(#inputs),*);
        fn nested() { #nested }
    });
    common::visitor(referrer, context, common::placeholder).visit_item_trait_mut(&mut copy);
    s(&copy)
}

/// F43 (fixed): a type copied as written because its context cannot be written
/// (`Reason::UnnameableSelf`) was unknown to the Referrer.
///
/// Trigger: `enter_trait(Tr<T: AsRef<Self>>)` around `impl ::p::W for X { fn f(&self, x: Rel<T,
/// Other<u8>>) .. }`: the bound of `T` names a `Self` that has no name in the impl, so `Rel<T, ..>`
/// is a conflict. The Visitor looked it up by its tokens and replaced the entries in it. Where the
/// part is whole (`::p::W<'a, T, dyn RelTr>`) the copy broke: the entry `dyn RelTr` is `+
/// 'static` where the path made it `+ 'a`, E0308 "lifetime mismatch".
///
/// Cause: a part was recorded only if its context could be written.
///
/// Fix: the part has no context (`unwritable` in the serialized form), and the Visitor finds it
/// where it cannot write the context either: the same tokens, the same place in a predicate or
/// not, the same anchor. Where the context can be written, the same tokens are an entry.
#[test]
fn f43_a_type_whose_context_cannot_be_written_is_copied_as_written() {
    let nested: Item = parse_quote!(
        impl ::p::W for X {
            fn f(&self, x: Rel<T, Other<u8>>) {}
            fn g(&self, y: Other<u8>) {}
        }
    );
    // The same tokens where the context can be written: in the trait itself.
    let direct: Vec<Type> = vec![parse_quote!(Rel<T, Other<u8>>)];
    let (result, context) = f43_analyze(common::std_config(), false, &direct, &nested);
    assert_eq!(
        messages(result.errors()),
        "the bound of `T` mentions a `Self` which has no name here"
    );
    assert_eq!(result.errors()[0].reason, Reason::UnnameableSelf);
    let referrer = result.recovered();
    // Not whole, and no entry has its tokens and context: a part that the visitor does not look at,
    // which is there because the error is about it.
    assert_eq!(
        parts_of(&referrer),
        ["type (Rel < T , Other < u8 > >) unwritable"]
    );
    let (other, outer) = (
        entry_index(&referrer, "Other < u8 >"),
        entry_index(&referrer, "Rel < T , Other < u8 > >"),
    );
    // The part is left as written, and `Other<u8>` in it is an entry as in `g`; `Rel<T, Other<u8>>`
    // of the trait (a context that can be written) is one as ever.
    let expected = format!(
        "trait Tr < T : AsRef < Self > > {{ fn direct (& self , x0 : P < {outer}usize >) ; fn nested () {{ impl :: p :: W for P < {x}usize > {{ fn f (& self , x : Rel < T , P < {other}usize > >) {{ }} fn g (& self , y : P < {other}usize >) {{ }} }} }} }}",
        outer = outer,
        other = other,
        x = entry_index(&referrer, "X"),
    );
    assert_eq!(f43_copy(&referrer, context, &direct, &nested), expected);
    // The context of the entry of the trait that names `Self` has the `Self` of the scope.
    assert!(common::entry(&referrer, outer).self_ty.is_some());
    // It is written and read back.
    let again: Referrer = parse2(quote!(#referrer)).unwrap();
    assert_eq!(again, referrer);
    assert_eq!(parts_of(&again), parts_of(&referrer));
}

/// F43: the whole part `::p::W<'a, T, dyn RelTr>` is left as written, though `dyn RelTr` is an
/// entry beside it (`PhantomData<dyn RelTr>`, with another lifetime).
#[test]
fn f43_the_entry_inside_the_part_is_not_replaced() {
    let nested: Item = parse_quote!(
        impl<'a> ::p::W for X {
            fn f(&self, x: ::p::W<'a, T, dyn RelTr>) {}
            fn g(&self, y: ::core::marker::PhantomData<dyn RelTr>) {}
        }
    );
    let (result, context) = f43_analyze(common::std_config(), false, &[], &nested);
    assert_eq!(result.errors().len(), 1);
    assert_eq!(result.errors()[0].reason, Reason::UnnameableSelf);
    let referrer = result.recovered();
    assert_eq!(
        parts_of(&referrer),
        ["type (:: p :: W < 'a , T , dyn RelTr >) unwritable whole"]
    );
    let (object, x) = (
        entry_index(&referrer, "dyn RelTr"),
        entry_index(&referrer, "X"),
    );
    let expected = format!(
        "trait Tr < T : AsRef < Self > > {{ fn direct (& self ,) ; fn nested () {{ impl < 'a > :: p :: W for P < {}usize > {{ fn f (& self , x : :: p :: W < 'a , T , dyn RelTr >) {{ }} fn g (& self , y : :: core :: marker :: PhantomData < P < {}usize > >) {{ }} }} }} }}",
        x, object
    );
    assert_eq!(f43_copy(&referrer, context, &[], &nested), expected);
}

/// F43: a conflicting bound (`U: RelTrait<T, Other>`) in the nested impl is a whole `unwritable`
/// part; the path of a value (`inner::make::<T, Other>()`) is no part: it is written as it is, and
/// the types in its arguments are replaced.
#[test]
fn f43_a_bound_and_a_value_whose_context_cannot_be_written_are_written_as_they_are() {
    let nested: Item = parse_quote!(
        impl<U: RelTrait<T, Other>> ::p::W for ::p::Host {
            fn f(&self, y: Other) -> usize {
                inner::make::<T, Other>()
            }
        }
    );
    // The rule of the Leaker allows `::p`: `::p::Host` is no entry, and the copy writes it as it is.
    let config = common::p_config();
    let (result, context) = f43_analyze(config, true, &[], &nested);
    assert!(result
        .errors()
        .iter()
        .all(|error| error.reason == Reason::RelativePath));
    assert_eq!(result.errors().len(), 2);
    let referrer = result.recovered();
    assert_eq!(
        parts_of(&referrer),
        ["trait (RelTrait < T , Other >) unwritable whole predicate"]
    );
    let other = entry_index(&referrer, "Other");
    let expected = format!(
        "trait Tr < T : AsRef < Self > > {{ fn direct (& self ,) ; fn nested () {{ impl < U : RelTrait < T , Other > > :: p :: W for :: p :: Host {{ fn f (& self , y : P < {other}usize >) -> usize {{ inner :: make :: < T , P < {other}usize > > () }} }} }} }}",
        other = other
    );
    assert_eq!(f43_copy(&referrer, context, &[], &nested), expected);
}

/// F43: the serialized part. `unwritable` stands where the number of the context would be, before
/// `predicate` and `in (..)`, and the parser reads what the serializer writes (and only that).
#[test]
fn f43_an_unwritable_part_is_serialized_and_read_back() {
    // What is written is read back to the same Referrer, and written again as it was.
    assert_parts_read_back(
        &[
            "type (R<T>) unwritable",
            "trait (RelTr<T>) unwritable predicate",
            "type (R<T>) unwritable whole in (Wrap<R<T>>) 1",
            "type (R<T>) unwritable in (Wrap<R<T>>) 1",
            "type (R<T>) unwritable predicate in (Wrap<R<T>>) 0",
            "expr (f::<T>) unwritable, type (R) 1, type (R<T>) 0 predicate",
        ],
        "< T >",
    );
    assert_parts_refused(
        &[
            ("type (R)", "expected integer literal"),
            ("type (R) nothing", "expected a context or `unwritable`"),
            (
                "type (R) unwritable nope",
                "expected `whole`, `sig`, `predicate`, `in` or `,`",
            ),
            ("type (R) unwritable in (&R)", "expected integer literal"),
            (
                "type (R) unwritable in (&R) 5",
                "the context 5 of a part is not one of the contexts",
            ),
            (
                "type (R) unwritable unwritable",
                "expected `whole`, `sig`, `predicate`, `in` or `,`",
            ),
        ],
        "< T >",
    );
}

// ── F44 ──────────────────────────────────────────────────────────────────────

/// The copy of the trait, once with `Self`, once with `with_self(Host)`.
fn f44_copies(referrer: &Referrer, context: ContextId, item: &ItemTrait) -> [String; 2] {
    let plain = {
        let mut copy = item.clone();
        common::visitor(referrer, context, common::placeholder).visit_item_trait_mut(&mut copy);
        s(&copy)
    };
    let hosted = {
        let mut copy = item.clone();
        common::visitor(referrer, context, common::placeholder)
            .with_self(parse_quote!(Host))
            .visit_item_trait_mut(&mut copy);
        s(&copy)
    };
    [plain, hosted]
}

/// F44 (fixed): a part whose context names `Self` was not found when no entry named `Self`, so the
/// Visitor replaced the types inside it.
///
/// Trigger: `trait Tr where Rel<Self, Other>: Sized { fn g(&self, y: Other); }`: the part
/// `Rel<Self, Other>` (`SelfInTraitPredicate`) had `Other` replaced. With a position-sensitive
/// entry (`dyn Tr` under a path with a lifetime parameter) the copy is E0308.
///
/// Cause: the top-level context of the scope recorded `Self` only if an entry named it, so the
/// Visitor computed the part's context without `Self`, and it never matched.
///
/// Fix: a part copied as written whose context depends on `Self` counts too, so the top-level
/// context has `Self` and the Visitor finds the part, with `Self` or with a host.
#[test]
fn f44_a_part_that_names_self_is_found_when_no_entry_does() {
    // SelfInTraitPredicate; the entry beside is `Other`, in `g`.
    let item: ItemTrait = parse_quote!(
        trait Tr
        where
            Rel<Self, Other>: Sized,
        {
            fn g(&self, y: Other);
        }
    );
    let mut leaker = Leaker::with_rule(common::p_config());
    let context = leaker.intern_trait(&item);
    let result = leaker.result();
    assert_eq!(result.errors().len(), 1);
    assert_eq!(result.errors()[0].reason, Reason::SelfInTraitPredicate);
    let referrer = result.recovered();
    // In a predicate: whole.
    let parts = parts_of(&referrer);
    assert_eq!(parts.len(), 1);
    assert!(
        parts[0].starts_with("type (Rel < Self , Other >)"),
        "{}",
        parts[0]
    );
    // The part is as written, with `Self` or the host; the entry beside it is replaced.
    let expected =
        "trait Tr where Rel < SELF , Other > : Sized , { fn g (& self , y : P < 0usize >) ; }";
    let [plain, hosted] = f44_copies(&referrer, context, &item);
    assert_eq!(plain, expected.replace("SELF", "Self"));
    assert_eq!(hosted, expected.replace("SELF", "Host"));
    // It is written and read back.
    let again: Referrer = parse2(quote!(#referrer)).unwrap();
    assert_eq!(again, referrer);
}

/// F44: `SelfInImplSelfType` (`impl<T> S<T, Other> where T: AsRef<Option<Self>>`): the bound is a
/// whole part, the self type keeps its head as written, and `Other` is replaced in the self type
/// and in the method.
#[test]
fn f44_the_self_type_of_an_impl_is_found_when_no_entry_names_self() {
    let item: ItemImpl = parse_quote!(
        impl<T> S<T, Other>
        where
            T: AsRef<Option<Self>>,
        {
            fn g(&self, y: Other) {}
        }
    );
    let mut leaker = Leaker::with_rule(common::p_config());
    let context = leaker.intern_impl(&item);
    let result = leaker.result();
    assert_eq!(
        result.errors().iter().map(|e| e.reason).collect::<Vec<_>>(),
        [Reason::SelfInImplSelfType, Reason::UnprovableBound]
    );
    let referrer = result.recovered();
    // The `Self` of the bound, in a predicate, is whole. The self type is not: it is a part that
    // the visitor does not look at, there because its error is about it (the last part).
    let parts = parts_of(&referrer);
    assert_eq!(parts.len(), 2);
    assert!(parts[0].contains("predicate") && !parts[1].contains("whole"));
    let mut copy = item.clone();
    common::visitor(&referrer, context, common::placeholder).visit_item_impl_mut(&mut copy);
    // The self type is as written but for `Other`, which is an entry in it as in the method.
    let expected: ItemImpl = parse_quote!(
        impl<T> S<T, P<0usize>>
        where
            T: AsRef<Option<Self>>,
        {
            fn g(&self, y: P<0usize>) {}
        }
    );
    assert_eq!(s(&copy), s(&expected));
}

// ── F45 ──────────────────────────────────────────────────────────────────────

/// F45 (fixed): a part in an item nested in a scope was not found by a Visitor without
/// `with_nested_self()`, so the types inside it were replaced.
///
/// Trigger: a nested `pub trait Tr where Rel<Self, Other>: Sized { fn g(&self, y: Other); }`
/// (`SelfInTraitPredicate`, copied as written): the copy replaced `Other` in the part.
///
/// Cause: such a Visitor looks nothing up whose context names the `Self` of a nested item (`Self`
/// means something else there), and that rule hid the part too.
///
/// Fix: a part matches whether the Visitor looks up the entries of the nested `Self` or not: a
/// part is protected, not replaced. The test runs with and without the closure.
#[test]
fn f45_a_part_in_a_nested_item_is_left_as_written() {
    let items: Vec<(Item, &str, usize)> = vec![
        (
            parse_quote!(
                pub trait Tr
                where
                    Rel<Self, Other>: Sized,
                {
                    fn g(&self, y: Other);
                }
            ),
            "pub trait Tr where Rel < Self , Other > : Sized , { fn g (& self , y : P < 0usize >) ; }",
            1,
        ),
        // The same for a bound whose trait path conflicts, which is a node and no type.
        (
            parse_quote!(
                pub trait Tr<T: ::core::convert::AsRef<Self>> {
                    fn m<U: RelTrait<T, Other>>(&self, u: U);
                    fn g(&self, y: Other);
                }
            ),
            "pub trait Tr < T : :: core :: convert :: AsRef < Self > > { fn m < U : RelTrait < T , Other > > (& self , u : U) ; fn g (& self , y : P < 0usize >) ; }",
            1,
        ),
    ];
    for (item, expected, parts) in &items {
        for nest in NESTS {
            for hooked in [false, true] {
                let what = format!("{:?} `{}`, hooked {}", nest, s(item), hooked);
                let (copy, referrer, calls) =
                    nested_run(common::std_leaker(), nest, item, (true, false), hooked);
                assert_eq!(s(&copy), *expected, "{}", what);
                // The part is no entry, so the closure of `with_nested_self()` is not asked.
                assert!(calls.is_empty(), "{}: {:?}", what, calls);
                assert_eq!(parts_of(&referrer).len(), *parts, "{}", what);
            }
        }
    }
}

// ── F46 ──────────────────────────────────────────────────────────────────────

/// F46 (fixed): the help for a relative path that starts with `core`, `std` or `alloc` suggested a
/// path in the Leaker's crate.
///
/// Trigger: `trait Tr: core::fmt::Debug {}`: the help said "write `core::fmt::Debug` with an
/// absolute path (e.g. `::my_crate::core::fmt::Debug`)", but the crate itself is meant.
///
/// Fix: the example is the path with leading colons (`::core::fmt::Debug`). Any other relative
/// path, also one with the name in a later segment (`m::core::X`), keeps the Leaker's crate.
#[test]
fn f46_the_help_for_a_relative_std_path_suggests_the_crate_itself() {
    let help_of = |item: Item| -> Vec<String> {
        let mut leaker = Leaker::new();
        common::intern_item(&mut leaker, &item);
        let result = leaker.result();
        all_helps(result.errors())
    };
    let help = |path: &str, example: &str| {
        format!(
            "write `{}` with an absolute path (e.g. `{}`)",
            path, example
        )
    };
    for (item, path, example) in [
        // A bound, and a type; the three crates.
        (
            parse_quote!(
                trait Tr: core::fmt::Debug {}
            ),
            "core::fmt::Debug",
            "::core::fmt::Debug",
        ),
        (
            parse_quote!(
                trait Tr: std::fmt::Display {}
            ),
            "std::fmt::Display",
            "::std::fmt::Display",
        ),
        (
            parse_quote!(
                trait Tr: alloc::fmt::Write {}
            ),
            "alloc::fmt::Write",
            "::alloc::fmt::Write",
        ),
        (
            parse_quote!(
                struct S {
                    a: std::cell::Ref<'_, u8>,
                }
            ),
            "std::cell::Ref",
            "::std::cell::Ref",
        ),
        // The crate of the Leaker: a name later in the path, a path of the crate, and a name
        // that is not one of the three.
        (
            parse_quote!(
                trait Tr: m::core::Bound {}
            ),
            "m::core::Bound",
            "::my_crate::m::core::Bound",
        ),
        (
            parse_quote!(
                trait Tr: crate::core::Bound {}
            ),
            "crate::core::Bound",
            "::my_crate::core::Bound",
        ),
        (
            parse_quote!(
                trait Tr: stdx::Bound {}
            ),
            "stdx::Bound",
            "::my_crate::stdx::Bound",
        ),
    ] {
        assert_eq!(help_of(item), [help(path, example)], "{}", path);
    }
}

// ── F47 ──────────────────────────────────────────────────────────────────────

/// F47 (fixed): `Self: 'x` made the context of an entry keep the bounds that F26 leaves out.
///
/// Trigger: `trait Tr { fn m<'x, Y: 'x + Iterator<Item = u8>>(&self) where Self: 'x, Y:
/// Fn(::p::WIt<Y>) -> u8, Rel<Y>: Default; }` (or the same in an impl): E0277 "expected an
/// `Fn(u8)` closure, found `Y`" in the copy.
///
/// Cause: `Rel<Y>` does not name `Self`, but `Y: 'x` pulls in `Self: 'x`, so its context depended
/// on `Self`, and such a context kept every bound of every generic, `Y: Fn(WIt<Y>)` too (F26).
///
/// Fix: an entry that names `Self` or writes out the self type keeps every bound; one that depends
/// on `Self` only through a bound or an outlives predicate keeps the bounds of the generics that
/// the description of `Self` names (`impl<T> S<T>`, `Tr<T>`) and leaves out the others. The bound
/// that stays is the one that F64 rejects in a predicate.
#[test]
fn f47_self_in_an_outlives_predicate_keeps_the_bounds_of_the_description_of_self_only() {
    let leaked = |item: &Item| f19_copy_with(common::p_leaker(), item).unwrap();
    // `Self: 'x` only: `Y: Fn(WIt<Y>)` is left out, in a trait and in an impl.
    let outlives: Vec<(Item, SelfRequirement)> = vec![
        (
            parse_quote!(
                pub trait Tr {
                    fn m<'x, Y: 'x + Iterator<Item = u8>>(&self)
                    where
                        Self: 'x,
                        Y: Fn(::p::WIt<Y>) -> u8,
                        Rel<Y>: Default;
                }
            ),
            SelfRequirement::Bound(parse_quote!(Tr)),
        ),
        (
            parse_quote!(
                impl<'a> ::p::Tr<'a> for ::p::W {
                    fn m<'x, Y: 'x + Iterator<Item = u8>>(&'x self) -> u8
                    where
                        Self: 'x,
                        Y: Fn(::p::WIt<Y>) -> u8,
                        Rel<Y>: Default,
                    {
                        loop {}
                    }
                }
            ),
            SelfRequirement::Type(parse_quote!(::p::W)),
        ),
    ];
    for (item, self_req) in &outlives {
        let (referrer, copy) = leaked(item);
        let ix = entry_index(&referrer, "Rel < Y >");
        assert_eq!(
            context_text_with_where(&referrer, ix),
            "< 'x , Y : 'x + Iterator < Item = u8 > > where Rel < Y > : Default",
            "{}",
            s(item)
        );
        assert_eq!(
            common::entry(&referrer, ix).self_ty.as_ref(),
            Some(self_req)
        );
        assert!(has_replacement_in_predicate(&copy), "{}", s(&copy));
        // The context is written and read back.
        let again: Referrer = parse2(quote!(#referrer)).unwrap();
        assert_eq!(again, referrer);
        assert_eq!(
            context_text_with_where(&again, ix),
            context_text_with_where(&referrer, ix)
        );
    }
    // The generic that the description of `Self` names keeps every bound (`Tr<T>`, `S<T>`), a
    // generic of the method does not. The entry is in an argument: in a predicate it is an error
    // (F64, below), as the bound of `T` stays.
    let described: Vec<Item> = vec![
        parse_quote!(
            pub trait Tr<T: Iterator<Item = u8>>
            where
                T: Fn(::p::WIt<T>),
            {
                fn m<'x, Y: 'x + Iterator<Item = u8>>(&'x self, a: Rel<T, Y>)
                where
                    Self: 'x,
                    Y: Fn(::p::WIt<Y>) -> u8;
            }
        ),
        parse_quote!(
            impl<T: Iterator<Item = u8> + Fn(::p::WIt<T>)> ::p::S<T> {
                fn m<'x, Y: 'x + Iterator<Item = u8>>(&'x self, a: Rel<T, Y>)
                where
                    Self: 'x,
                    Y: Fn(::p::WIt<Y>) -> u8,
                {
                }
            }
        ),
    ];
    for item in &described {
        let (referrer, copy) = leaked(item);
        assert_eq!(
            context_text(&referrer, entry_index(&referrer, "Rel < T , Y >")),
            "< 'x , T : Fn (:: p :: WIt < T >) + Iterator < Item = u8 > , Y : 'x + Iterator < \
             Item = u8 > >",
            "{}",
            s(item)
        );
        assert!(!has_replacement_in_predicate(&copy), "{}", s(&copy));
        assert!(s(&copy).contains("a : P < "), "{}", s(&copy));
    }
    // An entry that names `Self` keeps every bound, also those of the generics of the method.
    let names_self: Item = parse_quote!(
        pub trait Tr {
            type A;
            fn m<Y: Iterator<Item = u8> + Fn(::p::WIt<Y>)>(&self, x: Rel<Self::A, Y>);
        }
    );
    let (referrer, _) = leaked(&names_self);
    assert_eq!(
        context_text(&referrer, entry_index(&referrer, "Rel < Self :: A , Y >")),
        "< Y : Fn (:: p :: WIt < Y >) + Iterator < Item = u8 > >"
    );
}

// ── F48 ──────────────────────────────────────────────────────────────────────

/// The items of F48 in one Leaker, with the errors that remain and their top-level contexts.
fn f48_leak(items: &[Item]) -> (Referrer, Vec<ContextId>, Vec<Reason>) {
    let mut leaker = common::std_leaker();
    let contexts = items
        .iter()
        .map(|item| common::intern_item(&mut leaker, item))
        .collect();
    let result = leaker.result();
    let reasons = result.errors().iter().map(|e| e.reason).collect();
    (result.recovered(), contexts, reasons)
}

/// The copies of `items`, each with the entries replaced by `P<ix>`.
fn f48_copies(referrer: &Referrer, contexts: &[ContextId], items: &[Item]) -> Vec<String> {
    items
        .iter()
        .zip(contexts)
        .map(|(item, context)| {
            let mut copy = item.clone();
            common::visitor(referrer, *context, common::placeholder).visit_item_mut(&mut copy);
            s(&copy)
        })
        .collect()
}

/// F48 (fixed): a type copied as written for the signature it is in was left so everywhere.
///
/// Trigger: `fn f<'x>(&self, a: Wrap<'x>) -> &'x u8` in a trait. Leaked, `Wrap<'x>` would leave
/// `'x` unconstrained (E0195), so it is an `UnconstrainedLifetime` error and, allowed, copied as
/// written. The part had no anchor, so the copy left every `Wrap<'x>` of the context as written
/// (`fn g<'x>(&self, a: Wrap<'x>)`, a field of `struct S<'x>`), where the Referrer's crate may not
/// name `Wrap` (E0412). The same for E0106 (`fn f(x: &u8, y: Wrap<T>) -> &u8` in `trait Tr<'a, T:
/// 'a>`, where the projection of `Wrap<T>` names `'a`).
///
/// Fix: the part carries a mark: `sig ('x)` (the method lifetimes of the type that the output
/// names and no bound does) or `sig (elided)` (the signature relies on elision and the type adds
/// lifetimes). The Visitor leaves it as written only in the inputs of a signature with the same
/// fault (for `sig ('x)`, where the copy leaves the lifetime unconstrained, F72): not in another
/// item, an output, a bound, or a type visited alone.
#[test]
fn f48_a_type_corrected_for_a_signature_is_left_only_in_a_signature_like_it() {
    // E0195. `Tr::f` and `Tr2::f` output the lifetime, `Tr3::f` does not, and `Tr4::f` names it in
    // the output as an entry too.
    let items: Vec<Item> = vec![
        parse_quote!(
            trait Tr {
                fn f<'x>(&self, a: Wrap<'x>) -> &'x u8;
                fn g<'x>(&self, a: Wrap<'x>);
            }
        ),
        parse_quote!(
            struct S<'x> {
                w: Wrap<'x>,
            }
        ),
        parse_quote!(
            trait Tr2 {
                fn f<'x>(&self, a: Wrap<'x>) -> &'x u8;
            }
        ),
        parse_quote!(
            trait Tr3 {
                fn f<'x>(&self, a: Wrap<'x>);
            }
        ),
        parse_quote!(
            trait Tr4 {
                fn f<'x>(&self, a: Wrap<'x>) -> Wrap<'x>;
            }
        ),
    ];
    let (referrer, contexts, reasons) = f48_leak(&items);
    assert_eq!(reasons, [Reason::UnconstrainedLifetime; 3]);
    assert_eq!(parts_of(&referrer), ["type (Wrap < 'x >) 0 sig ('x)"]);
    assert_eq!(
        f48_copies(&referrer, &contexts, &items),
        [
            "trait Tr { fn f < 'x > (& self , a : Wrap < 'x >) -> & 'x u8 ; fn g < 'x > (& self , a : P < 0usize >) ; }",
            "struct S < 'x > { w : P < 0usize > , }",
            "trait Tr2 { fn f < 'x > (& self , a : Wrap < 'x >) -> & 'x u8 ; }",
            "trait Tr3 { fn f < 'x > (& self , a : P < 0usize >) ; }",
            "trait Tr4 { fn f < 'x > (& self , a : Wrap < 'x >) -> P < 0usize > ; }",
        ]
    );
    // Without the mark, a part that is whole is left everywhere: the field of `S` and `g` too. (A
    // part that is neither whole nor marked nor anchored is not looked at.)
    let unmarked: Referrer = parse_str(&s(&referrer).replace(" sig ('x)", " whole")).unwrap();
    assert_eq!(parts_of(&unmarked), ["type (Wrap < 'x >) 0 whole"]);
    let copies = f48_copies(&unmarked, &contexts, &items);
    assert_eq!(copies[1], "struct S < 'x > { w : Wrap < 'x > , }");
    assert!(copies[0].contains("fn g < 'x > (& self , a : Wrap < 'x >)"));

    // E0106. `f` has no receiver and an elided output, and `Wrap<T>` adds `'a` to its one input
    // lifetime. `g` and `k` have a receiver that gives its lifetime to the output, and `h` has no
    // output.
    let items: Vec<Item> = vec![
        parse_quote!(
            trait Tr<'a, T: 'a> {
                fn f(x: &u8, y: Wrap<T>) -> &u8;
                fn g(&self, y: Wrap<T>);
                fn h(y: Wrap<T>);
                fn k(&self, y: Wrap<T>) -> &u8;
            }
        ),
        parse_quote!(
            struct S<'a, T: 'a>(::core::marker::PhantomData<&'a T>, Wrap<T>);
        ),
    ];
    let (referrer, contexts, reasons) = f48_leak(&items);
    assert_eq!(reasons, [Reason::RelativePath]);
    assert_eq!(parts_of(&referrer), ["type (Wrap < T >) 0 sig (elided)"]);
    assert_eq!(
        f48_copies(&referrer, &contexts, &items),
        [
            "trait Tr < 'a , T : 'a > { fn f (x : & u8 , y : Wrap < T >) -> & u8 ; fn g (& self , y : P < 0usize >) ; fn h (y : P < 0usize >) ; fn k (& self , y : P < 0usize >) -> & u8 ; }",
            "struct S < 'a , T : 'a > (:: core :: marker :: PhantomData < & 'a T > , P < 0usize >) ;",
        ]
    );
}

/// F48: where the visitor leaves a marked part, and where it does not.
#[test]
fn f48_a_marked_part_is_left_in_the_arguments_of_a_signature_only() {
    let item: ItemTrait = parse_quote!(
        trait Tr {
            fn f<'x>(&self, a: Wrap<'x>) -> &'x u8;
        }
    );
    let mut leaker = common::std_leaker();
    let context = leaker.intern_trait(&item);
    // Another item gives the type an entry, so there is something to replace it with.
    let struct_context = leaker.intern_struct(&parse_quote!(
        struct S<'x>(Wrap<'x>);
    ));
    let referrer = leaker.result().recovered();
    let replace = |_: Type, ix: EntryId, _: &Context| -> Type { parse_quote!(P<#ix>) };
    let ty: Type = parse_quote!(Wrap<'x>);
    let sig: Signature = parse_quote!(fn f<'x>(&self, a: Wrap<'x>) -> &'x u8);
    let other: Signature = parse_quote!(fn f<'x>(&self, a: Wrap<'x>));

    // In the arguments of the signature that has the fault: as written.
    let mut copy = sig.clone();
    common::visitor(&referrer, context, replace).visit_signature_mut(&mut copy);
    assert_eq!(s(&copy), s(&sig));
    // In a signature without it, in the output, and in no signature: replaced.
    let mut copy = other.clone();
    common::visitor(&referrer, context, replace).visit_signature_mut(&mut copy);
    assert_eq!(s(&copy), "fn f < 'x > (& self , a : P < 0usize >)");
    let mut copy: Signature = parse_quote!(fn f<'x>(&self, a: Wrap<'x>) -> Wrap<'x>);
    common::visitor(&referrer, context, replace).visit_signature_mut(&mut copy);
    assert_eq!(
        s(&copy),
        "fn f < 'x > (& self , a : Wrap < 'x >) -> P < 0usize >"
    );
    let mut copy = ty.clone();
    common::visitor(&referrer, struct_context, replace).visit_type_mut(&mut copy);
    assert_eq!(s(&copy), "P < 0usize >");
    assert_eq!(
        s(&common::expand(&referrer, struct_context, ty, replace)),
        "P < 0usize >"
    );
}

/// F48: the mark is written after the context, and the parser reads what the serializer writes.
#[test]
fn f48_a_mark_is_serialized_and_read_back() {
    let generics = "< 'x , 'y >";
    assert_parts_read_back(
        &[
            "type (R<'x>) 0 sig ('x)",
            "type (R<'x, 'y>) 1 sig ('x, 'y) predicate",
            "type (R<T>) 0 sig (elided) in (Wrap<R<T>>) 1",
            "type (R<T>) unwritable sig (elided) predicate in (Wrap<R<T>>) 0",
            "type (R<'x>) 0, type (R<'x>) 0 sig ('x), type (R) 1 sig (elided)",
        ],
        generics,
    );
    // A part with a mark is not the part without.
    assert_ne!(
        read_parts("type (R<'x>) 0 sig ('x)", generics).unwrap(),
        read_parts("type (R<'x>) 0", generics).unwrap()
    );
    assert_parts_refused(
        &[
            ("type (R) 0 sig", "expected parentheses"),
            ("type (R) 0 sig ()", "expected lifetime"),
            ("type (R) 0 sig (nope)", "expected `elided` or lifetimes"),
            (
                "type (R) 0 sig (elided, 'x)",
                "expected `elided` or lifetimes",
            ),
            ("type (R) 0 sig ('x,)", "expected lifetime"),
            ("type (R) 0 sig ('x, 'x)", "the lifetime is written twice"),
            ("type (R) 0 sig ('x 'y)", "expected `,`"),
            (
                "type (R) 0 sig ('x) sig ('x)",
                "expected `predicate`, `in` or `,`",
            ),
            ("type (R) 0 predicate sig ('x)", "expected `,`"),
            (
                "type (R) 0 nope",
                "expected `whole`, `sig`, `predicate`, `in` or `,`",
            ),
            (
                "type (R) 0 whole nope",
                "expected `sig`, `predicate`, `in` or `,`",
            ),
            (
                "type (R) 0 sig ('x) whole",
                "expected `predicate`, `in` or `,`",
            ),
        ],
        generics,
    );
}

// ── F49 ──────────────────────────────────────────────────────────────────────

/// F49 (fixed): the E0106 check ignored a lifetime that another input keeps.
///
/// Trigger: `fn f(x: &'a u8, y: Wrap<T>) -> &u8` in `trait Tr<'a, T: 'a>`. The copy of `Wrap<T>`
/// names `'a` in its projection (`T: 'a` is in its context), so two inputs have a lifetime and the
/// elided output fails with E0106.
///
/// Cause: the check left out the lifetimes that the inputs keep, but rustc counts each input
/// apart: an elided output needs exactly one input with any lifetime (named, `'static`, elided or
/// hidden by a path), and that input with one distinct lifetime. A reference receiver gives its
/// lifetime to the output whatever the others have.
///
/// Fix: the Leaker and the Referrer count per input (of a method, a function pointer and
/// `Fn(..)`). A type in an input that does not hold the lifetime the inputs keep adds it: it is
/// copied as written and reported (`sig (elided)`, F48), or the function pointer is leaked whole.
/// A type in the input that names it (`x: &'a Wrap<T>`) adds nothing.
#[test]
fn f49_a_lifetime_that_another_input_keeps_breaks_elision() {
    // `f` and `g` are broken, in either order. The others are controls: `Wrap<T>` is the entry.
    let item: ItemTrait = parse_quote!(
        trait Tr<'a, T: 'a> {
            fn f(x: &'a u8, y: Wrap<T>) -> &u8;
            fn g(y: Wrap<T>, x: &'a u8) -> &u8;
            fn h(x: &'a Wrap<T>) -> &u8;
            fn i(x: (&'a u8, Wrap<T>)) -> &u8;
            fn j(x: &'a u8, y: Wrap<T>) -> &'a u8;
            fn k(&self, x: &'a u8, y: Wrap<T>) -> &u8;
            fn l(x: &'a u8, y: Wrap<T>);
            fn m(x: &'a u8, y: Wrap<u8>) -> &u8;
        }
    );
    let item = Item::Trait(item);
    let (referrer, contexts, reasons) = f48_leak(std::slice::from_ref(&item));
    assert_eq!(reasons, [Reason::RelativePath; 2]);
    assert_eq!(parts_of(&referrer), ["type (Wrap < T >) 0 sig (elided)"]);
    let copies = f48_copies(&referrer, &contexts, std::slice::from_ref(&item));
    assert_eq!(
        copies,
        [
            "trait Tr < 'a , T : 'a > { fn f (x : & 'a u8 , y : Wrap < T >) -> & u8 ; \
          fn g (y : Wrap < T > , x : & 'a u8) -> & u8 ; \
          fn h (x : P < 0usize >) -> & u8 ; \
          fn i (x : (& 'a u8 , P < 1usize >)) -> & u8 ; \
          fn j (x : & 'a u8 , y : P < 1usize >) -> & 'a u8 ; \
          fn k (& self , x : & 'a u8 , y : P < 1usize >) -> & u8 ; \
          fn l (x : & 'a u8 , y : P < 1usize >) ; \
          fn m (x : & 'a u8 , y : P < 2usize >) -> & u8 ; }"
        ]
    );
    // Before the fix there was no error and no part: `f` and `g` had the entry, whose projection
    // names `'a`.
    let before = f48_copies(
        &without_parts(&referrer),
        &contexts,
        std::slice::from_ref(&item),
    );
    assert!(before[0].contains("fn f (x : & 'a u8 , y : P < 1usize >) -> & u8 ;"));
    assert!(before[0].contains("fn g (y : P < 1usize > , x : & 'a u8) -> & u8 ;"));

    // The error says that rustc counts the lifetimes of each input, and what to write.
    let err = assert_rejected(
        &item,
        analyze_in(common::std_leaker(), &item),
        &["use absolute path: 'Wrap<T>'"; 2],
    );
    let note = "the output `&u8` has an elided lifetime, which elision takes from the only \
                lifetime of the inputs, but a leaked type in the inputs has `'a` among its \
                generics, and the copy names it in the arguments of its projection, so the inputs \
                have more than one lifetime, as rustc counts the lifetimes of each input apart \
                (E0106)";
    let whole = "it cannot be leaked as a whole because of `&u8`";
    assert_eq!(all_notes(&err), [note, whole, note, whole]);
    let help = "name the lifetime of the inputs and of the output, give the method a receiver \
                (`&self`), or write `Wrap<T>` with an absolute path";
    assert_eq!(all_helps(&err), [help, help]);

    // The same lifetime is not a second one if the input that has it has the type: that input has
    // one lifetime. Without the type in another input, the entries are as the controls above.
    let controls: ItemTrait = parse_quote!(
        trait Tr<'a, T: 'a> {
            fn h(x: &'a Wrap<T>) -> &u8;
            fn i(x: (&'a u8, Wrap<T>)) -> &u8;
            fn j(x: &'a u8, y: Wrap<T>) -> &'a u8;
            fn k(&self, x: &'a u8, y: Wrap<T>) -> &u8;
            fn l(x: &'a u8, y: Wrap<T>);
            fn m(x: &'a u8, y: Wrap<u8>) -> &u8;
        }
    );
    let (_, referrer) = analyze_in(common::std_leaker(), &Item::Trait(controls)).unwrap();
    assert!(parts_of(&referrer).is_empty());
    let entries: Vec<String> = common::type_texts(&referrer);
    assert_eq!(entries, ["& 'a Wrap < T >", "Wrap < T >", "Wrap < u8 >"]);
}

/// F49: a function pointer and `Fn(..)` count each input too: in a type the whole signature is
/// leaked (`fn(&'a u8, Wrap<T>) -> &u8`, `dyn Fn(..)`), in a bound it is an error.
#[test]
fn f49_a_function_pointer_and_fn_sugar_count_each_input() {
    let items: Vec<Item> = vec![
        parse_quote!(
            struct S<'a, T: 'a> {
                f: fn(&'a u8, Wrap<T>) -> &u8,
                g: fn(&'a Wrap<T>) -> &u8,
                h: fn((&'a u8, Wrap<T>)) -> &u8,
                i: fn(&'a u8, Wrap<T>) -> &'a u8,
            }
        ),
        parse_quote!(
            trait Tr<'a, T: 'a> {
                fn m(&self, f: Box<dyn Fn(&'a u8, Wrap<T>) -> &u8>);
                fn n(&self, f: Box<dyn Fn(&'a u8, Wrap<T>) -> &'a u8>);
            }
        ),
    ];
    let (referrer, contexts, reasons) = f48_leak(&items);
    assert!(reasons.is_empty());
    let entries: Vec<String> = common::type_texts(&referrer);
    assert_eq!(
        entries,
        [
            "& 'a Wrap < T >",
            "Wrap < T >",
            "dyn Fn (& 'a u8 , Wrap < T >) -> & u8",
            "fn (& 'a u8 , Wrap < T >) -> & u8",
        ]
    );
    assert_eq!(
        f48_copies(&referrer, &contexts, &items),
        [
            "struct S < 'a , T : 'a > { f : P < 3usize > , g : fn (P < 0usize >) -> & u8 , \
             h : fn ((& 'a u8 , P < 1usize >)) -> & u8 , i : fn (& 'a u8 , P < 1usize >) -> & 'a u8 , }",
            "trait Tr < 'a , T : 'a > { fn m (& self , f : Box < P < 2usize > >) ; \
             fn n (& self , f : Box < dyn Fn (& 'a u8 , P < 1usize >) -> & 'a u8 >) ; }",
        ]
    );
    // In a bound nothing can hold the signature: the error, with the type as written.
    let bound: ItemTrait = parse_quote!(
        trait Tr<'a, T: 'a> {
            fn m<F: Fn(&'a u8, Wrap<T>) -> &u8>(&self, f: F);
        }
    );
    let err = assert_rejected(
        &bound,
        analyze_in(common::std_leaker(), &Item::Trait(bound.clone())),
        &["use absolute path: 'Wrap<T>'"],
    );
    assert!(
        all_notes(&err)[0].ends_with("as rustc counts the lifetimes of each input apart (E0106)")
    );
}

// ── F50 ──────────────────────────────────────────────────────────────────────

/// The analysis of `item` with `check_values` and the primitive types allowed, and the path rule
/// `rule` (`"std"`, `"crate"` or the prelude) if there is one.
fn f50_analyze(item: &ItemTrait, rule: Option<&str>) -> LeakerResult {
    fn run<A: Allow>(item: &ItemTrait, config: A) -> LeakerResult {
        let mut leaker = Leaker::with_rule(config);
        leaker.check_values = true;
        leaker.check_bodies = true;
        leaker.intern_trait(item);
        leaker.result()
    }
    let config = Rules::new().primitive();
    let config = match rule {
        None => config,
        Some("std") => config.std_crates(),
        Some("crate") => config.crate_paths(),
        Some(_) => config.prelude(),
    };
    run(item, config)
}

/// The paths of the error `at` of `result`, as text.
fn f50_paths(result: &LeakerResult, at: usize) -> Vec<String> {
    result.errors()[at].paths.iter().map(s).collect()
}

/// F50 (fixed): a path rule given to the result dropped the error of a value's path and left the
/// relative types in its arguments as written.
///
/// Trigger: `core::mem::size_of::<*const Rel>()` in a default method, with `check_values` and
/// `check_bodies`. The error's only path was `core::mem::size_of`, so `allow()` with
/// `Rules::std_crates()` dropped it and `Rel` stayed relative in the copy, though a Leaker given
/// the rule leaks it.
///
/// Cause: the whole path of the value was copied as written, its arguments too.
///
/// Fix: only the path of the value is written as it is; the types in its arguments are leaked as
/// in any type position of a value. So the rule drops the error, and the result equals that of a
/// Leaker given the rule. The same for `Vec::<Rel>::new()` with `prelude()` and the trait of a
/// qualified path.
#[test]
fn f50_the_relative_types_in_the_arguments_of_a_value_path_are_leaked() {
    // The value, the rule (`std`, `crate` or `prelude`) given to the result and to the config, the
    // paths of the error, and the entries of the config that has the rule, which are those of the
    // result.
    struct Case {
        item: ItemTrait,
        rule: &'static str,
        paths: &'static [&'static str],
        entries: &'static [&'static str],
    }
    let case = |item, rule, paths, entries| Case {
        item,
        rule,
        paths,
        entries,
    };
    let cases = vec![
        case(
            parse_quote!(
                trait Tr {
                    fn f(&self) {
                        let _ = core::mem::size_of::<*const Rel>();
                    }
                }
            ),
            "std",
            &["core :: mem :: size_of"],
            &["Rel"],
        ),
        case(
            parse_quote!(
                trait Tr {
                    fn f(&self) {
                        crate::base::bg::<*const R0>();
                    }
                }
            ),
            "crate",
            &["crate :: base :: bg"],
            &["R0"],
        ),
        // Deeper: the arguments of the arguments.
        case(
            parse_quote!(
                trait Tr {
                    fn f(&self) {
                        let _ = Vec::<Option<Rel>>::new();
                    }
                }
            ),
            "prelude",
            &["Vec :: new"],
            &["Option < Rel >"],
        ),
        // The default of an associated const.
        case(
            parse_quote!(
                trait Tr {
                    const C: usize = core::mem::size_of::<*const Rel>();
                }
            ),
            "std",
            &["core :: mem :: size_of"],
            &["Rel"],
        ),
        // The trait of a qualified path.
        case(
            parse_quote!(
                trait Tr {
                    fn f(&self) {
                        let _ = <u8 as core::convert::From<Rel>>::from;
                    }
                }
            ),
            "std",
            &["core :: convert :: From"],
            &["Rel"],
        ),
    ];
    for Case {
        item,
        rule,
        paths,
        entries,
    } in cases
    {
        let what = s(&item);
        let result = f50_analyze(&item, None);
        assert_eq!(result.errors().len(), 1, "{}", what);
        assert_eq!(f50_paths(&result, 0), paths, "{}", what);
        // The result has the entries of the config already: the arguments are leaked, not written as
        // they are.
        let found: Vec<String> = result
            .clone()
            .recovered()
            .iter()
            .map(|(_, ty, _)| s(ty))
            .collect();
        assert_eq!(found, entries, "{}", what);
        // The rule drops the error, as the config that has it has none.
        let after = result.clone();
        let after = match rule {
            "std" => after.allow(Rules::new().std_crates()),
            "crate" => after.allow(Rules::new().crate_paths()),
            _ => after.allow(Rules::new().prelude()),
        };
        assert!(after.errors().is_empty(), "{}", what);
        let config = f50_analyze(&item, Some(rule));
        assert!(config.errors().is_empty(), "{}", what);
        assert_eq!(
            config.finish().unwrap(),
            after.finish().unwrap(),
            "{}",
            what
        );
    }

    // Not paths of the error: a generic, and what the value declares.
    for item in [
        parse_quote!(
            trait Tr {
                fn f<T>(&self) {
                    let _ = core::mem::size_of::<*const T>();
                }
            }
        ),
        parse_quote!(
            trait Tr {
                fn f(&self) {
                    struct L;
                    let _ = core::mem::size_of::<*const L>();
                }
            }
        ),
    ] {
        let result = f50_analyze(&item, None);
        assert_eq!(
            f50_paths(&result, 0),
            ["core :: mem :: size_of"],
            "{}",
            s(&item)
        );
    }
}

// ── F51 ──────────────────────────────────────────────────────────────────────

/// F51 (fixed): a nested item with the risk allowed took a bound that names a leaked type that
/// names `Self`, which `intern_struct()` rejects.
///
/// Trigger: `struct S<T: Clone + AsRef<R1<Self>>> { f0: T, f1: R1<(Self, T)> }` nested in a scope,
/// with `NestedSelfInBounds` allowed: the copy failed with E0277 `T: AsRef<R1<S<T>>>`.
///
/// Cause: nested, the entry `R1<Self>` in the bound is for any type, and its context has no
/// generic (`T` is not in the type), so the check of the bounds of its context found none.
///
/// Fix: an entry in a predicate that is for any type because it names `Self` is checked with the
/// context of the type of `Self`, as `intern_struct()` does: `UnprovableBound`, also when no entry
/// uses `T`.
#[test]
fn f51_a_nested_item_with_a_bound_that_names_a_leaked_type_is_rejected() {
    let items: Vec<Item> = vec![
        parse_quote!(
            pub struct S<T: Clone + AsRef<R1<Self>>> {
                pub f0: T,
                pub f1: R1<(Self, T)>,
            }
        ),
        // A where predicate, and no entry that uses `T`.
        parse_quote!(
            pub struct S<T>(pub T)
            where
                T: AsRef<R1<Self>>;
        ),
        parse_quote!(
            pub enum E<T: AsRef<R1<Self>>> {
                A(T, R1<(Self, T)>),
            }
        ),
    ];
    let message = "`R1<Self>` is in a bound or where predicate, and a bound of its generics \
                   names a leaked type";
    for item in &items {
        let what = format!("`{}`", s(item));
        let result = accept_nested_self(nested_analysis(item));
        let errors = result.errors();
        assert_eq!(errors.len(), 1, "{}", what);
        assert_eq!(errors[0].reason, Reason::UnprovableBound, "{}", what);
        assert_eq!(errors[0].message, message, "{}", what);
    }
    // `intern_struct()` reports the same.
    let item: ItemStruct = parse_quote!(
        pub struct S<T: Clone + AsRef<R1<Self>>> {
            pub f0: T,
            pub f1: R1<(Self, T)>,
        }
    );
    let mut leaker = common::std_leaker();
    leaker.intern_struct(&item);
    let direct = accept_nested_self(leaker.result());
    assert_eq!(direct.errors().len(), 1);
    assert_eq!(direct.errors()[0].message, message);

    // Controls: a bound that names `Self` without a leaked type is accepted, as F42 says.
    let item: Item = parse_quote!(
        pub struct S<T: Clone + AsRef<Self>> {
            pub f0: T,
            pub f1: R1<(Self, T)>,
        }
    );
    let referrer = assert_accepted(&item, accept_nested_self(nested_analysis(&item)).finish());
    let entries: Vec<String> = common::type_texts(&referrer);
    assert_eq!(entries, ["R1 < (Self , T) >"]);
}

// ── F52 ──────────────────────────────────────────────────────────────────────

/// F52 (fixed): an unrelated error made the optional entries of a trait's predicates entries.
///
/// Trigger: `trait Tr<T> where T: Fn(&::ext::B2<T, ::ext::B0>) -> u8 { type A0; fn m<Y>(&self)
/// where Y: RelTr<Self::A0>; }` with `::ext` not allowed: the copy replaced `::ext::B0`, and the
/// host's impl failed with E0277 "expected an `Fn(&B2<T, B0>)` closure, found `T`".
///
/// Cause: `Self::A0` in a predicate is resolved by the host's impl with the bounds as written, so
/// `::ext::B0` must stay as written (F17). But `RelTr<..>` is an error copied whole, and the visit
/// that keeps `B0` did not run inside it.
///
/// Fix: the probe inside the part keeps what its `Self` and `Self::A` make not optional.
///
/// Limitation: a part with no relative path to probe for (an error of a lifetime, say) is not
/// probed.
#[test]
fn f52_an_unrelated_error_does_not_make_an_optional_entry_of_the_predicate() {
    let item = |method: &str| -> ItemTrait {
        let method: proc_macro2::TokenStream = method.parse().unwrap();
        parse_quote!(
            pub trait Tr<T>
            where
                T: ::core::ops::Fn(&::ext::B2<T, ::ext::B0>) -> u8,
            {
                type A0;
                #method
            }
        )
    };
    let entries = |method: &str| -> (usize, Vec<String>) {
        let mut leaker = common::std_leaker();
        leaker.intern_trait(&item(method));
        let result = leaker.result();
        (
            result.errors().len(),
            entries_text(&result.recovered(), true),
        )
    };
    // The error is `RelTr<Self::A0>`, and `Self::A0` in it is resolved by the host: no entry.
    assert_eq!(
        entries("fn m<Y>(&self) where Y: RelTr<Self::A0>;"),
        (1, vec![])
    );
    // The same without the error.
    assert_eq!(
        entries("fn m<Y>(&self) where Y: ::core::convert::Into<Self::A0>;"),
        (0, vec![])
    );
    // Controls: nothing resolves `Self::A0`, the host has the predicate as it is in the trait and
    // the optional entry is one.
    assert_eq!(
        entries("fn m<Y>(&self) where Y: RelTr<u8>;"),
        (1, vec![":: ext :: B0".to_string()])
    );
    assert_eq!(entries(""), (0, vec![":: ext :: B0".to_string()]));
    // The path of the part allowed in the config: it is not an error, and `Self::A0` is visited.
    let config = Rules::new()
        .primitive()
        .prelude()
        .or_if(starts_with(parse_quote!(RelTr)));
    let mut leaker = Leaker::with_rule(config);
    leaker.intern_trait(&item("fn m<Y>(&self) where Y: RelTr<Self::A0>;"));
    let result = leaker.result();
    assert!(result.errors().is_empty());
    assert!(entries_text(&result.finish().unwrap(), true).is_empty());
}

// ── F53 ──────────────────────────────────────────────────────────────────────

/// The generics of the Repeater impl of the entry `entry` of `referrer`, and its type, as text.
fn f53_declared(result: &LeakerResult, entry: &str, target: ImplSelf) -> (String, String) {
    let referrer = result.clone().finish().unwrap();
    let ix = common::types(&referrer)
        .position(|ty| s(ty) == entry)
        .unwrap();
    let r = result
        .repeater_impls(&Projection::new(repeater(), 3, target.clone()))
        .unwrap()
        .remove(ix);
    (with_where(&r.generics), s(&r.self_ty))
}

/// F53 (fixed): the parameter for the implementor was declared after the const parameters.
///
/// Trigger: `trait Tr<'a, T: 'a, const N: usize> { fn f(&self, x: &'a Wrap<T, N>); }` with
/// `ImplSelf::Param` (or `Bucket`, for an entry that needs `Self`): the impl was `impl<'a, T,
/// const N: usize, __S: ?Sized> ..`, which rustc before 1.59 rejects ("type parameters must be
/// declared prior to const parameters"), though the library supports rustc 1.56.
///
/// Fix: `repeater_impls()` declares the parameter after the lifetimes and the types, before the
/// first const parameter.
#[test]
fn f53_the_parameter_for_the_implementor_is_declared_before_the_const_parameters() {
    let param: Ident = parse_quote!(__S);
    let bucket = ImplSelf::Bucket {
        param: param.clone(),
    };
    let analyze =
        |item: Item| assert_accepted(&item, analyze_result_in(common::std_leaker(), &item)).1;
    let declared = |generics: Generics| with_where(&generics);
    // A trait: an entry for any type, and one for the implementor (`Self::Out` is bound by it).
    let result = analyze(parse_quote! {
        trait Tr<'a, T: 'a, const N: usize> {
            type Out;
            fn f(&self, x: &'a Wrap<T, N>) -> Rel<Self::Out, N>;
        }
    });
    let any = "& 'a Wrap < T , N >";
    let implementor = "Rel < Self :: Out , N >";
    let (generics, self_ty) = f53_declared(&result, any, ImplSelf::Param(param.clone()));
    assert_eq!(
        generics,
        declared(parse_quote!(<'a, T, __S: ?::core::marker::Sized, const N: usize>))
    );
    assert_eq!(self_ty, "__S");
    // A bucket is a primitive type, and declares nothing.
    let (generics, self_ty) = f53_declared(&result, any, bucket.clone());
    assert_eq!(generics, declared(parse_quote!(<'a, T, const N: usize>)));
    assert_ne!(self_ty, "__S");
    for target in [ImplSelf::Param(param.clone()), bucket.clone()] {
        let (generics, self_ty) = f53_declared(&result, implementor, target);
        assert_eq!(
            generics,
            declared(parse_quote!(<
                'a,
                T,
                __S: ?::core::marker::Sized + Tr<'a, T, N>,
                const N: usize
            >))
        );
        assert_eq!(self_ty, "__S");
    }
    // A struct: an entry for any type, and one for the type, which declares no parameter.
    let result = analyze(parse_quote! {
        struct S<'a, T: 'a, const N: usize> { a: &'a Wrap<T, N>, b: Rel<Self, N> }
    });
    let (generics, _) = f53_declared(&result, any, ImplSelf::Param(param.clone()));
    assert_eq!(
        generics,
        declared(parse_quote!(<'a, T, __S: ?::core::marker::Sized, const N: usize>))
    );
    let (generics, _) = f53_declared(&result, any, bucket.clone());
    assert_eq!(generics, declared(parse_quote!(<'a, T, const N: usize>)));
    let (generics, self_ty) =
        f53_declared(&result, "Rel < Self , N >", ImplSelf::Param(param.clone()));
    assert_eq!(generics, declared(parse_quote!(<'a, T, const N: usize>)));
    assert_eq!(self_ty, "S < 'a , T , N >");
    // A nested struct whose bound names its own `Self`: the impl is for any type that holds it.
    let result = nested_analysis(&parse_quote! {
        pub struct S<T: AsRef<Self>, const N: usize>(T, Wrap<T, N>);
    });
    for target in [ImplSelf::Param(param.clone()), bucket.clone()] {
        let (generics, self_ty) = f53_declared(&result, "Wrap < T , N >", target);
        assert_eq!(
            generics,
            declared(parse_quote!(<T: AsRef<__S>, __S: ?::core::marker::Sized, const N: usize>))
        );
        assert_eq!(self_ty, "__S");
    }
}

// ── F54 ──────────────────────────────────────────────────────────────────────

/// F54 (fixed): serializing a Referrer panicked for an entry headed by `::crate`, `::self`,
/// `::super` or `::Self`.
///
/// Trigger: `struct S { a: ::crate::X }`, which the Leaker accepts (the path is absolute; rustc
/// rejects the item later). The crate `crate` was recorded in the entry's list of crates, and
/// writing it as `r#crate` panics: these names cannot be raw identifiers.
///
/// Fix: the Referrer carries no list of crates any more, so the panic cannot happen; the test
/// keeps the round trip of such an entry.
#[test]
fn f54_an_entry_headed_by_a_path_keyword_serializes() {
    for head in ["crate", "self", "super", "Self"] {
        let head: Ident = Ident::new(head, proc_macro2::Span::call_site());
        let items: Vec<(ItemStruct, &str)> = vec![
            (parse_quote!(struct S { a: ::#head::X }), "a path"),
            // A real crate next to it is still listed.
            (
                parse_quote!(struct S { a: ::#head::X<::p::Y> }),
                "a path with a crate",
            ),
            // The tokens of a macro, in which paths are not parsed.
            (
                parse_quote!(
                    struct S {
                        a: ::q::m!(::#head::X, ::p::Y),
                    }
                ),
                "a macro",
            ),
        ];
        for (item, what) in items {
            let what = format!("`{}` ({})", s(&item), what);
            let mut leaker = common::std_leaker();
            leaker.intern_struct(&item);
            let result = leaker.result();
            let tokens = template_quote::ToTokens::to_token_stream(&result);
            let referrer = result.finish().unwrap();
            assert!(!referrer.is_empty(), "{}", what);
            // The tokens parse back to the same Referrer, both as a Referrer and as a result.
            assert_eq!(
                parse2::<Referrer>(tokens.clone()).unwrap(),
                referrer,
                "{}",
                what
            );
            let read = parse2::<LeakerResult>(tokens).unwrap();
            assert_eq!(read.finish().unwrap(), referrer, "{}", what);
        }
    }
}

// ── F56 ──────────────────────────────────────────────────────────────────────

/// What the Leaker finds in `source`, a struct or an enum, analyzed with `Rules::primitive()`: for
/// each error its cause, `in_value` and `in_default`, and then the entries. `check_values` is on
/// and off, which must not matter (an enum discriminant and the default of a const parameter are
/// read either way).
fn f56_read(source: &str) -> (Vec<(String, bool, bool)>, Vec<String>) {
    let item: Item = parse_str(source).unwrap();
    let mut found = Vec::new();
    for check_values in [false, true] {
        let mut leaker = Leaker::with_rule(Rules::new().primitive());
        leaker.check_values = check_values;
        common::intern_item(&mut leaker, &item);
        let result = leaker.result();
        assert!(
            result
                .errors()
                .iter()
                .all(|error| error.reason == Reason::RelativePath),
            "{}",
            source
        );
        let errors = (result.errors().iter())
            .map(|error| {
                (
                    s(&error.cause),
                    error.place.in_value,
                    error.place.in_default,
                )
            })
            .collect();
        let entries = result.recovered().iter().map(|(_, ty, _)| s(ty)).collect();
        found.push((errors, entries));
    }
    let read = found.pop().unwrap();
    assert_eq!(found.pop().unwrap(), read, "{}", source);
    read
}

/// F56 (fixed): a relative macro or struct literal in an enum discriminant or in the default of a
/// const parameter was no error, and a local binding there was one.
///
/// Trigger: `enum E { B = two!() }` and `struct S<const N: usize = { Rel { a: 1 }.a }>(..)` gave
/// no error, though the copy resolves `two!` and `Rel` in the Referrer's crate; `enum E { B = {
/// let x: Rel = 1; x } }` reported the binding `x` as a relative path.
///
/// Cause: they were read at the item's depth, with no scope: a macro was a cause only in a type, a
/// struct literal had no node, and only the value walk of `check_values` tracked bindings.
///
/// Fix: both are read by that walk, with or without `check_values`: a binding, an item, a generic
/// or `Self` is local; a relative macro, struct literal, function or constant is a `RelativePath`
/// error (not `in_value`; `in_default` for a const parameter); a type in a type position is
/// leaked. The cause of `<::p::S as Rel>::N` is the trait `Rel`, and its self type is an entry.
#[test]
fn f56_a_discriminant_and_the_default_of_a_const_parameter_are_read_like_a_value() {
    // The source, the causes of the errors, the entries.
    let cases: &[(&str, &[&str], &[&str])] = &[
        // The macro and the struct literal are errors.
        ("enum E { B = two!() }", &["two !"], &[]),
        ("enum E { B = 1 + two!() }", &["two !"], &[]),
        ("enum E { A = 1, B = { two!() } }", &["two !"], &[]),
        (
            "struct S<const N: usize = { two!() }>([u8; N]);",
            &["two !"],
            &[],
        ),
        ("enum E { B = Rel { a: 1 }.a }", &["Rel"], &[]),
        (
            "struct S<const N: usize = { Rel { a: 1 }.a }>([u8; N]);",
            &["Rel"],
            &[],
        ),
        // A binding is local: a `let`, a closure parameter, a `match` arm, an item of the block.
        // The types in their type positions are leaked.
        ("enum E { B = { let x: Rel = 1; x } }", &[], &["Rel"]),
        ("enum E { B = (|x: u8| x)(1) as isize }", &[], &[]),
        ("enum E { B = match 2 { y => y } }", &[], &[]),
        ("enum E { B = { let (a, b) = (1, 2); a + b } }", &[], &[]),
        ("enum E { B = { const K: u8 = 1; K as isize } }", &[], &[]),
        (
            "struct S<const N: usize = { let x = 1; x }>([u8; N]);",
            &[],
            &[],
        ),
        (
            "struct S<const N: usize = { (|x: usize| x)(1) }>([u8; N]);",
            &[],
            &[],
        ),
        // Controls: what was right stays.
        ("enum E { B = REL }", &["REL"], &[]),
        ("enum E { B = ::m::two!() }", &[], &[]),
        ("struct S([u8; two!()]);", &["two !"], &[]),
        // A binding in an array length is local too (F110): the array names nothing relative.
        ("struct S([u8; { let x = 1; x }]);", &[], &[]),
        ("enum E { B = 1 as Rel }", &[], &["Rel"]),
        ("enum E { B = <Rel as ::p::Tr>::N }", &[], &["Rel"]),
        // The path is written as it is, and the type in its arguments is leaked, as it is of an
        // absolute path (the next case).
        (
            "enum E { B = rel_fn::<Rel2>() }",
            &["rel_fn :: < Rel2 >"],
            &["Rel2"],
        ),
        ("enum E { B = ::p::f::<Rel2>() }", &[], &["Rel2"]),
        (
            "enum E { B = match 1 { Rel::A => 1, _ => 2 } }",
            &["Rel :: A"],
            &[],
        ),
        // The trait of a qualified path is the cause, and its self type an entry.
        (
            "enum E { B = <::p::S as Rel>::N }",
            &["Rel"],
            &[":: p :: S"],
        ),
    ];
    for (source, causes, entries) in cases {
        let (errors, found) = f56_read(source);
        let found_causes: Vec<&str> = errors.iter().map(|(cause, _, _)| cause.as_str()).collect();
        assert_eq!(&found_causes, causes, "{}", source);
        assert_eq!(&found, entries, "{}", source);
        // The default of a parameter is a default, and neither is a value of `check_values`.
        let in_default = source.contains("const N");
        assert!(
            errors
                .iter()
                .all(|&(_, value, default)| !value && default == in_default),
            "{}",
            source
        );
    }

    // The error is the one of a relative path: its headline and note.
    let item: ItemEnum = parse_quote!(
        enum E {
            B = two!(),
        }
    );
    let mut leaker = Leaker::with_rule(Rules::new().primitive());
    leaker.intern_enum(&item);
    let err = assert_rejected(
        &item,
        leaker.result().finish(),
        &["use absolute path: 'two!'"],
    );
    assert_eq!(
        all_notes(&err),
        ["it is in the discriminant of `B`, which the copy writes as it is"]
    );
    let item: ItemStruct = parse_quote!(
        struct S<const N: usize = { Rel { a: 1 }.a }>([u8; N]);
    );
    let mut leaker = Leaker::with_rule(Rules::new().primitive());
    leaker.intern_struct(&item);
    let err = assert_rejected(
        &item,
        leaker.result().finish(),
        &["use absolute path: 'Rel'"],
    );
    assert_eq!(
        all_notes(&err),
        ["it is in the default of `N`, which the copy writes as it is"]
    );

    // `Rules::values()` accepts what `check_values` reads, which this is not. `Rules::defaults()`
    // accepts the default of a parameter, not a discriminant.
    let analyze = |source: &str| {
        let mut leaker = Leaker::with_rule(Rules::new().primitive());
        leaker.check_values = true;
        common::intern_item(&mut leaker, &parse_str::<Item>(source).unwrap());
        leaker.result()
    };
    let (discriminant, default) = (
        "enum E { B = two!() }",
        "struct S<const N: usize = { two!() }>([u8; N]);",
    );
    // The errors that remain after `Rules::values()` and after `Rules::defaults()`.
    for (source, values, defaults) in [(discriminant, 1, 1), (default, 1, 0)] {
        let result = analyze(source).allow(Rules::new().values());
        assert_eq!(result.errors().len(), values, "{}", source);
        let result = analyze(source).allow(Rules::new().defaults());
        assert_eq!(result.errors().len(), defaults, "{}", source);
    }

    // `Self` is read as in a value of `check_values`: no error, no entry.
    let item = "enum E { B = Self::K }";
    assert_eq!(f56_read(item), (vec![], vec![]));
}

// ── F57 ──────────────────────────────────────────────────────────────────────

/// The primitive names that `tokens` (`Leaker::assertions()`) checks, in order.
fn f57_names(tokens: proc_macro2::TokenStream) -> Vec<String> {
    let mut names = Vec::new();
    for token in tokens {
        match token {
            proc_macro2::TokenTree::Group(group) => names.extend(f57_names(group.stream())),
            proc_macro2::TokenTree::Ident(ident) => {
                let ident = ident.to_string();
                let name = ident.strip_prefix("__type_leak_assert_");
                names.extend(
                    name.and_then(|name| name.strip_suffix("_is_primitive"))
                        .map(str::to_string),
                );
            }
            _ => (),
        }
    }
    names
}

/// The primitive names that the assertions of `source` check with `Rules::primitive()`, which must
/// be those with a rule that drops every error whose cause is no bare path besides (the analysis
/// leaks the same paths: it is asked about bare paths only), and the same again: the text of the
/// assertions does not change.
fn f57_assertions(source: &str, check_values: bool) -> Vec<String> {
    let item: Item = parse_str(source).unwrap();
    let mut texts = Vec::new();
    for _ in 0..2 {
        let mut leaker = Leaker::with_rule(Rules::new().primitive());
        leaker.check_values = check_values;
        leaker.check_bodies = check_values;
        common::intern_item(&mut leaker, &item);
        texts.push(leaker.assertions().to_string());
        let config = Rules::new()
            .primitive()
            .or_if(|cause, _| common::bare_path(cause).is_none());
        let mut leaker = Leaker::with_rule(config);
        leaker.check_values = check_values;
        leaker.check_bodies = check_values;
        common::intern_item(&mut leaker, &item);
        texts.push(leaker.assertions().to_string());
    }
    assert_eq!(texts[0], texts[1], "{}", source);
    assert_eq!(texts[0], texts[2], "{}", source);
    assert_eq!(texts[1], texts[3], "{}", source);
    f57_names(texts[0].parse().unwrap())
}

/// F57 (fixed): `Leaker::assertions()` left out the primitive names in a part copied as written.
///
/// Trigger: `struct A { a: Rel<'_, u8> }`, `struct A { a: &dyn RelTr<u8> }` or `struct A<T:
/// Into<u32>> { a: T }` with `Rules::primitive()`: the assertions were empty. The copy writes `u8`
/// as it is, which is another type if the Leaker's crate shadows `u8`; the assertion is for that.
///
/// Cause: a conflicting type is copied whole, and the primitive names that the visit of its
/// components recorded were dropped with them; a conflicting bound or path of a value
/// (`Into<u32>`, `rel_fn::<u8>`) was not visited at all.
///
/// Fix: a part keeps the primitive names that it is copied with: those of its components and of
/// the types leaked whole in it, or, for a bound or the path of a value, those of its syntax. A
/// type replaced by its entry asserts nothing.
#[test]
fn f57_the_primitive_names_in_a_part_copied_as_written_are_asserted() {
    let cases: &[(&str, &[&str])] = &[
        // The three of the trigger.
        ("struct A { a: Rel<'_, u8> }", &["u8"]),
        ("struct A { a: &dyn RelTr<u8> }", &["u8"]),
        ("struct A<T: Into<u32>> { a: T }", &["u32"]),
        // In order of appearance, at any depth of the part.
        (
            "struct A { a: Rel<'_, ::core::option::Option<(u8, u16)>> }",
            &["u8", "u16"],
        ),
        (
            "struct A<T: Into<(u32, u16)> + From<u8>> { a: T }",
            &["u32", "u16", "u8"],
        ),
        // A path in a where predicate.
        ("struct A<T> where T: Into<u32> { a: T }", &["u32"]),
        // Only the part: the entry beside it, and the types it holds, assert nothing.
        ("struct A { a: Rel<'_, u8>, b: Rel2<u16> }", &["u8"]),
        ("struct A { a: Rel<u8>, b: Wrap<Rel2<u16>> }", &[]),
        ("struct A { a: [u8; size_of::<u16>()] }", &[]),
        // The same without an error: copied, so assumed.
        ("struct A { a: ::core::option::Option<u8> }", &["u8"]),
    ];
    for (source, names) in cases {
        assert_eq!(&f57_assertions(source, false), names, "{}", source);
    }
    // Nothing is assumed where `u8` is a relative path, which the error tells.
    let item: ItemStruct = parse_quote!(
        struct A {
            a: Rel<'_, u8>,
        }
    );
    let mut leaker = Leaker::new();
    leaker.intern_struct(&item);
    assert!(f57_names(leaker.assertions()).is_empty());

    // The path of a value that is a part, with `check_values` or not (a discriminant, the default
    // of a const parameter).
    let value_cases: &[(&str, bool, &[&str])] = &[
        (
            "trait Tr { fn f(&self) { rel_fn::<u8>(); } }",
            true,
            &["u8"],
        ),
        (
            "trait Tr { const C: usize = rel_fn::<u16>(); }",
            true,
            &["usize", "u16"],
        ),
        ("enum E { B = rel_fn::<u8>() }", false, &["u8"]),
        ("enum E { B = rel_fn::<u8>() }", true, &["u8"]),
        (
            "struct S<const N: usize = { rel_fn::<u32>() }>([u8; N]);",
            false,
            &["usize", "u32", "u8"],
        ),
        ("trait Tr { fn f(&self) { rel_fn::<Rel>(); } }", true, &[]),
    ];
    for (source, check_values, names) in value_cases {
        assert_eq!(&f57_assertions(source, *check_values), names, "{}", source);
    }

    // An entry that is cut and copied as written instead keeps its names: the type is a part
    // through `cut_conflicts()` (the self type of an impl whose bound names `Self`), and through
    // `correct_unmeant_self()` (`Self` where it has no meaning).
    let item: ItemImpl = parse_quote!(
        impl<T: Clone> Tr for Rel<u8, T>
        where
            Self: ::core::marker::Sized,
            T: AsRef<Self>,
        {
        }
    );
    let mut leaker = Leaker::with_rule(Rules::new().primitive());
    leaker.intern_impl(&item);
    let assertions = f57_names(leaker.assertions());
    let errors = leaker.result().errors().to_vec();
    assert!(errors
        .iter()
        .any(|error| error.message.contains("is in the self type of the impl")));
    assert_eq!(assertions, ["u8"]);
    let mut leaker = Leaker::with_rule(Rules::new().primitive());
    leaker.enter_generics(&Generics::default(), |view| {
        view.visit_type(&parse_quote!(Rel<u8, Self>))
    });
    assert_eq!(f57_names(leaker.assertions()), ["u8"]);
    // Control: the same type without `Self` is an entry, and asserts nothing.
    let mut leaker = Leaker::with_rule(Rules::new().primitive());
    leaker.enter_generics(&Generics::default(), |view| {
        view.visit_type(&parse_quote!(Rel<u8>))
    });
    assert!(f57_names(leaker.assertions()).is_empty());
}

// ── F58 ──────────────────────────────────────────────────────────────────────

/// What `intern_impl()` and the scope are compared by: the serialized result (the Referrer and
/// each error with its notes and help), and the headlines and helps of the errors, for a failure
/// that can be read.
fn f58_snapshot(result: LeakerResult) -> (String, Vec<String>, Vec<String>) {
    let errors = result.errors();
    let messages = errors.iter().map(|e| e.message.clone()).collect();
    (s(&result), messages, all_helps(errors))
}

/// `item` analyzed by `intern_impl()`.
fn f58_interned(item: &ItemImpl) -> (String, Vec<String>, Vec<String>) {
    let mut leaker = common::std_leaker();
    leaker.intern_impl(item);
    f58_snapshot(leaker.result())
}

/// `item` analyzed by `enter_impl()` with the trait of the impl, and the walk of the whole impl
/// that `View` documents (`syn::visit::visit_item_impl()`).
fn f58_entered(item: &ItemImpl) -> (String, Vec<String>, Vec<String>) {
    let trait_ = item.trait_.as_ref().map(|(_, path, _)| path);
    let mut leaker = common::std_leaker();
    leaker.enter_impl(&item.generics, &item.self_ty, trait_, |view| {
        syn::visit::visit_item_impl(view, item)
    });
    f58_snapshot(leaker.result())
}

/// F58 (fixed): `enter_impl()` could not say whether the impl is inherent or of which trait.
///
/// Trigger: `impl S { fn f<'a>(&self, x: Wrap<'a>) -> &'a u8 { .. } }` walked in `enter_impl()`
/// gave E0195 (`UnconstrainedLifetime`) and no entry for `Wrap<'a>`, while `intern_impl()` accepts
/// it. `impl RelTr for S {}` had no error, and the help for `Self::A` in a trait impl named a
/// placeholder instead of the trait.
///
/// Cause: only `intern_impl()` passed the trait to the scope and marked an impl inherent, which
/// the check of a method's signature reads: it compares a method with the trait's signature unless
/// the impl is inherent.
///
/// Fix: `enter_impl(generics, self_ty, trait_: Option<&Path>, f)`. `None` is an inherent impl,
/// whose methods are not compared; `Some(path)` is the trait, which the scope checks and the help
/// for `Self::A` names. With the walk `syn::visit::visit_item_impl(view, &item)` the scope gives
/// what `intern_impl()` gives, but for the self type of an impl whose bound names `Self` (below).
#[test]
fn f58_enter_impl_is_told_the_trait_of_the_impl() {
    // The inherent impl: no error, two entries (`u8` is allowed).
    let item: ItemImpl = parse_quote!(
        impl S {
            fn f<'a>(&self, x: Wrap<'a>) -> &'a u8 {
                todo!()
            }
        }
    );
    let mut leaker = common::std_leaker();
    leaker.enter_impl(&item.generics, &item.self_ty, None, |view| {
        syn::visit::visit_item_impl(view, &item)
    });
    let result = leaker.result();
    assert!(result.errors().is_empty(), "{}", s(&result));
    let entries: Vec<String> = result
        .finish()
        .unwrap()
        .iter()
        .map(|(_, ty, _)| s(ty))
        .collect();
    assert_eq!(entries, ["S", "Wrap < 'a >"]);
    // The same method in an impl of a trait is compared with the trait's signature: E0195.
    let item: ItemImpl = parse_quote!(
        impl ::p::Tr for S {
            fn f<'a>(&self, x: Wrap<'a>) -> &'a u8 {
                todo!()
            }
        }
    );
    let mut leaker = common::std_leaker();
    let trait_ = item.trait_.as_ref().map(|(_, path, _)| path);
    leaker.enter_impl(&item.generics, &item.self_ty, trait_, |view| {
        syn::visit::visit_item_impl(view, &item)
    });
    let reasons: Vec<Reason> = leaker.result().errors().iter().map(|e| e.reason).collect();
    assert_eq!(reasons, [Reason::UnconstrainedLifetime]);
    // The path of the trait is checked on entering the scope.
    let item: ItemImpl = parse_quote!(impl RelTr for S {});
    let (_, trait_, _) = item.trait_.as_ref().unwrap();
    let mut leaker = common::std_leaker();
    leaker.enter_impl(&item.generics, &item.self_ty, Some(trait_), |_| ());
    let result = leaker.result();
    assert_eq!(messages(result.errors()), "use absolute path: 'RelTr'");
    // The help for `Self::A` names the trait.
    let item: ItemImpl = parse_quote!(
        impl ::p::Tr for S {
            type A = u8;
            fn f(&self) -> Wrap<Self::A> {
                todo!()
            }
        }
    );
    let mut leaker = common::std_leaker();
    let trait_ = item.trait_.as_ref().map(|(_, path, _)| path);
    leaker.enter_impl(&item.generics, &item.self_ty, trait_, |view| {
        syn::visit::visit_item_impl(view, &item)
    });
    let result = leaker.result();
    let helps = all_helps(result.errors());
    assert_eq!(helps.len(), 1, "{}", s(&result));
    assert_eq!(helps[0], "replace `Self::A` with `<Self as ::p::Tr>::A`");
}

/// F58: `enter_impl()` with the trait and the walk of the impl gives what `intern_impl()` gives,
/// for inherent and trait impls (absolute or relative), with a method lifetime or a `Self::A`.
#[test]
fn f58_the_scope_of_an_impl_gives_what_intern_impl_gives() {
    let impls: Vec<ItemImpl> = vec![
        // Inherent.
        parse_quote!(impl S {}),
        parse_quote!(
            impl<T> S<T>
            where
                T: Clone,
            {
                fn g(&self) -> Wrap<T> {
                    todo!()
                }
            }
        ),
        parse_quote!(impl S { fn f<'a>(&self, x: Wrap<'a>) -> &'a u8 { todo!() } }),
        parse_quote!(
            impl<'b> S<'b> {
                fn f<'a>(&self, x: &'a Rel<'b>) -> Rel2<'a> {
                    todo!()
                }
            }
        ),
        parse_quote!(impl S { const C: Rel = todo!(); }),
        parse_quote!(impl ::p::S { fn f(&self, x: Wrap<Self>) {} }),
        parse_quote!(
            impl<T> S<T> {
                fn f(&self) -> Wrap<<Self as ::p::Tr>::A> {
                    todo!()
                }
            }
        ),
        // A trait with an absolute path.
        parse_quote!(impl ::p::Tr for S {}),
        parse_quote!(
            impl<T> ::p::Tr<Rel<T>, Rel2> for S<T> {}
        ),
        parse_quote!(
            impl<T> ::p::Tr<T, Rel<T>> for ::p::S<T> where T: ::p::Bound<Rel2> {}
        ),
        parse_quote!(impl ::p::Tr for S { fn f<'a>(&self, x: Wrap<'a>) -> &'a u8 { todo!() } }),
        parse_quote!(impl ::p::Tr for S { type A = Rel; fn f(&self) -> Wrap<Self::A> { todo!() } }),
        parse_quote!(
            impl<T> ::p::Tr<T> for S<T> {
                type A = u8;
                fn f(&self) -> Wrap<Self::A> {
                    todo!()
                }
            }
        ),
        parse_quote!(impl ::p::Tr for ::p::S { fn f(&self, x: Rel<Self>) { todo!() } }),
        // A trait with a relative path.
        parse_quote!(impl RelTr for S {}),
        parse_quote!(
            impl<T> RelTr<Rel<T>> for S<T> {}
        ),
        parse_quote!(impl m::RelTr<Rel> for S { fn f(&self) -> Wrap<Self::A> { todo!() } }),
        parse_quote!(impl RelTr for ::p::S { fn f<'a>(&self, x: Wrap<'a>) -> &'a u8 { todo!() } }),
        // A bound that names `Self`, where the self type has no entry.
        parse_quote!(
            impl<T> ::p::Tr<T> for u8 where T: AsRef<Self> {}
        ),
        parse_quote!(
            impl<T: ::p::Bound<Rel<Self>>> ::p::Tr<T> for u8 {}
        ),
    ];
    let mut errors = 0;
    for item in &impls {
        let interned = f58_interned(item);
        let entered = f58_entered(item);
        assert_eq!(interned, entered, "`{}`", s(item));
        errors += interned.1.len();
    }
    // The impls with an error are compared with it, not only the clean ones.
    assert!(errors >= 8, "{}", errors);
}

/// F58 (limitation): the walk of the whole impl in `enter_impl()` does not cut the entries of the
/// self type of an impl whose bound names `Self`.
///
/// `impl<T> ::p::S<Rel<T>> where T: AsRef<Self> {}`: `intern_impl()` reports `Rel<T>` in the self
/// type (`SelfInImplSelfType`: the copy would write `Self` as a projection that stands for itself,
/// E0391), and copies an optional self type (`impl<T> ::p::S<T> ..`) as written, as it walks the
/// self type apart. `syn::visit::visit_item_impl(view, &item)` visits the self type like any type:
/// it reports the bound only, and `::p::S<T>` is an optional entry. For such an impl use
/// `intern_impl()` (or `View::visit_item()` in a scope).
#[test]
fn f58_the_walk_in_the_scope_does_not_cut_the_self_type_of_the_impl() {
    let reasons =
        |errors: &[ErrorItem]| -> Vec<Reason> { errors.iter().map(|e| e.reason).collect() };
    let analyze = |item: &ItemImpl, interned: bool| -> LeakerResult {
        let mut leaker = common::std_leaker();
        if interned {
            leaker.intern_impl(item);
        } else {
            leaker.enter_impl(&item.generics, &item.self_ty, None, |view| {
                syn::visit::visit_item_impl(view, item)
            });
        }
        leaker.result()
    };
    let item: ItemImpl = parse_quote!(
        impl<T> ::p::S<Rel<T>> where T: AsRef<Self> {}
    );
    assert_eq!(
        reasons(analyze(&item, true).errors()),
        [Reason::SelfInImplSelfType, Reason::UnprovableBound]
    );
    assert_eq!(
        reasons(analyze(&item, false).errors()),
        [Reason::UnprovableBound]
    );
    let item: ItemImpl = parse_quote!(
        impl<T> ::p::S<T> where T: AsRef<Self> {}
    );
    let entries = |interned: bool| -> Vec<String> {
        entries_text(&analyze(&item, interned).finish().unwrap(), true)
    };
    assert!(entries(true).is_empty());
    assert_eq!(entries(false), [":: p :: S < T >"]);
}

// ── F59 and F60 ──────────────────────────────────────────────────────────────

/// `given` completed for an impl of an item with the parameters `params`, as `complete_args()` does
/// it in a scope where every path is allowed as written, so a default is what it says.
fn complete_args_of(
    params: &Punctuated<GenericParam, Token![,]>,
    given: &[GenericArgument],
) -> std::result::Result<Vec<GenericArgument>, EncodeError> {
    let generics = Generics {
        params: params.clone(),
        ..Default::default()
    };
    common::complete_generics(&generics, given)
}

/// The entries of `trait Tr<'a, T, U = Rel2<T>>`, which use `'a`, `T` and `U` (and `T` alone), and
/// the context of the trait.
fn f59_trait() -> (ItemTrait, Referrer, ContextId) {
    let tr: ItemTrait = parse_quote! {
        trait Tr<'a, T, U = Rel2<T>> {
            type Item;
            fn f(&self, x: &'a T) -> Wrap<'a, T, U>;
            fn g(&self) -> Rel<U>;
            fn h(&self) -> Rel3<T>;
        }
    };
    let mut leaker = common::std_leaker();
    let context = leaker.intern_trait(&tr);
    (tr, leaker.result().finish().unwrap(), context)
}

/// F59 (fixed): the arguments of a use site were taken by position and nothing was checked.
///
/// Trigger: `trait Tr<'a, T, U = Rel2<T>>` and the arguments of `impl Tr<'static, Item = u8, u8>`
/// as written: `Rel<U>` was encoded as `(u8,)`, the argument of `T`, as the third argument was
/// taken for `U`. Too many arguments, or one of another kind, were taken as they came too.
///
/// Fix: `Referrer::complete_args()` fails with `UnexpectedArgument` for more arguments than
/// parameters or one of the wrong kind, and with `MissingArgument` for a missing one without a
/// default; a binding is no argument. The encoder and `substitute_args()` fail with
/// `UnexpectedArgument` for a value that does not fit its place (a lifetime for a type). A name
/// with no answer stays.
#[test]
fn f59_the_arguments_of_a_use_site_are_checked_against_the_parameters() {
    let (tr, referrer, context) = f59_trait();
    let params = &tr.generics.params;
    let g = entry_index(&referrer, "Rel < U >");
    let h = entry_index(&referrer, "Rel3 < T >");
    let f = entry_index(&referrer, "Wrap < 'a , T , U >");
    // The arguments as written: completed (the default is written through the Repeater, and the
    // parameters before it are replaced), then written at the site.
    let written = |ix: usize, args: &[&str]| {
        let completed = common::completed_args(&referrer, context, &tr.generics, &arguments(args))?;
        common::repeater_args(
            common::entry(&referrer, ix),
            common::answers(params, &completed),
            None,
        )
    };
    // The arguments as completed.
    let encode = |ix: usize, args: &[&str]| {
        let args = arguments(args);
        common::repeater_args(
            common::entry(&referrer, ix),
            common::answers(params, &args),
            None,
        )
    };
    let unexpected = |result: std::result::Result<Type, EncodeError>, arg: &str| match result {
        Err(EncodeError::UnexpectedArgument(found)) => assert_eq!(s(&*found), arg),
        other => panic!("{:?}, not an unexpected `{}`", other, arg),
    };

    // The arguments as written, with a binding: it was `(u8,)` for `Rel<U>`. A binding is no
    // argument, so `U` takes its default, `Rel2<u8>` through the Repeater.
    let key = common::key(
        0,
        common::entry_id_at(&referrer, entry_index(&referrer, "Rel2 < T >")),
    );
    assert_eq!(
        written(g, &["'static", "Item = u8", "u8"]).unwrap(),
        parse_quote!((<Self as Repeater<(u8,), #key>>::Type,))
    );
    assert_eq!(
        written(h, &["'static", "Item = u8", "u8"]).unwrap(),
        parse_quote!((u8,))
    );
    // Too many.
    unexpected(written(g, &["'static", "u8", "u16", "u32"]), "u32");
    unexpected(written(g, &["'static", "'static", "u8", "u16"]), "'static");
    unexpected(written(g, &["'static", "u8", "u16", "'x"]), "'x");
    // A parameter with no default and no argument is missing.
    assert!(matches!(
        written(g, &["'static"]),
        Err(EncodeError::MissingArgument(_))
    ));
    assert!(matches!(
        written(g, &[]),
        Err(EncodeError::MissingArgument(_))
    ));

    // The arguments as completed, with a value that does not fit its place: it was taken as it came.
    unexpected(encode(f, &["'static", "Item = u8", "u8"]), "Item = u8");
    unexpected(encode(h, &["'static", "Item = u8", "u8"]), "Item = u8");
    unexpected(encode(f, &["u8", "'static", "u16"]), "u8");
    unexpected(encode(f, &["'static", "3", "u16"]), "3");
    unexpected(encode(f, &["'static", "'static", "u16"]), "'static");
    unexpected(encode(f, &["u8", "u8", "u16"]), "u8");

    // The right arguments, as before.
    assert_eq!(
        encode(g, &["'static", "u8", "u16"]).unwrap(),
        parse_quote!((u16,))
    );
    assert_eq!(
        encode(f, &["'static", "u8", "u16"]).unwrap(),
        parse_quote!((&'static (), u8, u16))
    );
    // They may stop early: an entry that uses what is given is encoded, and a name that has no
    // answer stays.
    assert_eq!(encode(h, &["'static", "u8"]).unwrap(), parse_quote!((u8,)));
    assert_eq!(encode(g, &["'static", "u8"]).unwrap(), parse_quote!((U,)));
    assert_eq!(encode(g, &[]).unwrap(), parse_quote!((U,)));
    // The scope of the analysis has no root parameter.
    assert_eq!(
        common::repeater_args(common::entry(&referrer, g), common::identity, None).unwrap(),
        parse_quote!((U,))
    );

    // The arguments of the generics of an entry are what the encoder is given, and fail alike.
    let args = arguments(&["'static", "Item = u8", "u8"]);
    assert!(matches!(
        common::generic_args(
            &common::entry(&referrer, h).generics,
            common::answers(params, &args)
        ),
        Err(EncodeError::UnexpectedArgument(_))
    ));
    let args = arguments(&["'static", "u8"]);
    let generic_args = common::generic_args(
        &common::entry(&referrer, h).generics,
        common::answers(params, &args),
    )
    .unwrap();
    assert_eq!(generic_args, [parse_quote!(u8)]);
}

/// F59: a const parameter takes a const argument, a block, or a bare identifier (which syn parses
/// as a type); anything else is `UnexpectedArgument`, in the encoder and in `complete_args()`.
#[test]
fn f59_a_const_parameter_takes_a_const_a_block_or_an_identifier() {
    let tr: ItemTrait = parse_quote!(
        trait Tr<'a, const N: usize> {
            fn f(&self) -> Wrap<'a, N>;
        }
    );
    let mut leaker = common::std_leaker();
    leaker.intern_trait(&tr);
    let referrer = leaker.result().finish().unwrap();
    let encode = |args: &[&str]| {
        let args = arguments(args);
        common::repeater_args(
            common::entry(&referrer, 0),
            common::answers(&tr.generics.params, &args),
            None,
        )
        .map(|ty| s(&ty))
    };
    for (written, encoded) in [
        ("3", "(& 'x () , [() ; { 3 }])"),
        ("{ 1 + 1 }", "(& 'x () , [() ; { 1 + 1 }])"),
        ("M", "(& 'x () , [() ; { M }])"),
    ] {
        assert_eq!(encode(&["'x", written]).unwrap(), encoded, "{}", written);
    }
    for written in ["Vec<u8>", "::m::M", "&u8", "'x", "Item=3"] {
        match encode(&["'x", written]) {
            Err(EncodeError::UnexpectedArgument(found)) => {
                assert_eq!(s(&*found).replace(' ', ""), written)
            }
            other => panic!("{}: {:?}", written, other),
        }
    }
    // `complete_args()` refuses the same, and a third argument.
    let complete = |args: &[&str]| common::complete_generics(&tr.generics, &arguments(args));
    for written in ["Vec<u8>", "::m::M", "&u8"] {
        match complete(&["'x", written]) {
            Err(EncodeError::UnexpectedArgument(found)) => {
                assert_eq!(s(&*found).replace(' ', ""), written)
            }
            other => panic!("{}: {:?}", written, other),
        }
    }
    assert!(matches!(
        complete(&["'x", "3", "4"]),
        Err(EncodeError::UnexpectedArgument(_))
    ));
}

/// F60 (fixed): a macro had no way to write the types of a copy that are not entries with the
/// arguments of the use site.
///
/// Trigger: `t: T` in `impl Tr<u8> for ..`: the Visitor leaves it as written, and each macro wrote
/// its own substitution, with gaps (lifetimes, consts, `Self`, a trait object behind a reference).
///
/// Fix: `substitute_args()` writes a type at the site in one pass: each name replaced by what the
/// site answers (a root parameter by its argument, `Self` by the self type), what replaces a name
/// not substituted again, and a trait object with several bounds parenthesized behind a pointer.
#[test]
fn f60_a_type_is_written_at_the_use_site() {
    let params: Punctuated<GenericParam, Token![,]> = parse_quote!('a, T, const N: usize);
    // The site: the arguments of the root parameters (none: the scope of the analysis).
    let site = |args: &[&str]| arguments(args);
    let write = |site: &[GenericArgument], ty: Type| {
        s(&common::substitute_type(&ty, common::answers(&params, site), None).unwrap())
    };
    let write_as = |site: &[GenericArgument], ty: Type, self_ty: &Type| {
        s(&common::substitute_type(&ty, common::answers(&params, site), Some(self_ty)).unwrap())
    };
    let given = site(&["'static", "Vec<u8>", "3"]);

    // Lifetimes, types and consts.
    assert_eq!(
        write(&given, parse_quote!(&'a Wrap<T, N>)),
        t(parse_quote!(&'static Wrap<Vec<u8>, 3>))
    );
    assert_eq!(
        write(&given, parse_quote!([T; N])),
        t(parse_quote!([Vec<u8>; { 3 }]))
    );
    assert_eq!(
        write(&given, parse_quote!(fn(&'a T) -> <T as Tr<'a>>::A)),
        t(parse_quote!(
            fn(&'static Vec<u8>) -> <Vec<u8> as Tr<'static>>::A
        ))
    );
    // A bare identifier is a const argument too.
    let named = site(&["'a", "T", "M"]);
    assert_eq!(
        write(&named, parse_quote!(Wrap<N, [u8; N]>)),
        t(parse_quote!(Wrap<M, [u8; { M }]>))
    );
    // The projection of a parameter is the projection of its argument, with the trait written, or
    // of an identifier, which rustc resolves as a type parameter (F90).
    assert_eq!(
        write(&given, parse_quote!(Option<<T as Iterator>::Item>)),
        t(parse_quote!(Option<<Vec<u8> as Iterator>::Item>))
    );
    assert_eq!(
        write(&named, parse_quote!(Option<T::Item>)),
        t(parse_quote!(Option<<T>::Item>))
    );
    // What the site gives no argument stays, and so does what is not a parameter.
    let early = site(&["'static"]);
    assert_eq!(
        write(&early, parse_quote!(&'a Wrap<T, U>)),
        t(parse_quote!(&'static Wrap<T, U>))
    );
    assert_eq!(
        write(&[], parse_quote!(&'a Wrap<T, Self::A>)),
        t(parse_quote!(&'a Wrap<T, Self::A>))
    );

    // `Self` is the second argument, in a type, a projection and a macro; its own generics are not
    // substituted again.
    let hosted = site(&["'static", "u8", "3"]);
    let host: Type = parse_quote!(Host<T>);
    assert_eq!(
        write_as(&hosted, parse_quote!(&'a (Self, T, <Self as Tr>::B)), &host),
        t(parse_quote!(&'static (Host<T>, u8, <Host<T> as Tr>::B)))
    );
    let ident: Type = parse_quote!(H);
    assert_eq!(
        write_as(&hosted, parse_quote!(Self::A), &ident),
        t(parse_quote!(<H>::A))
    );
    assert_eq!(
        write_as(&hosted, parse_quote!(m!(Self, T)), &host),
        t(parse_quote!(m!(Host<T>, T)))
    );
    // With `None`, `Self` stays.
    assert_eq!(
        write(&hosted, parse_quote!(m!(Self, T))),
        t(parse_quote!(m!(Self, T)))
    );

    // A trait object with several bounds needs parentheses behind a reference or a pointer, one
    // with a single bound does not.
    let object = site(&["'a", "dyn ::core::fmt::Debug + 'a", "3"]);
    assert_eq!(
        write(&object, parse_quote!(&'a T)),
        t(parse_quote!(&'a (dyn ::core::fmt::Debug + 'a)))
    );
    assert_eq!(
        write(&object, parse_quote!(*const T)),
        t(parse_quote!(*const (dyn ::core::fmt::Debug + 'a)))
    );
    assert_eq!(
        write(&object, parse_quote!(Box<T>)),
        t(parse_quote!(Box<dyn ::core::fmt::Debug + 'a>))
    );
    let object = site(&["'a", "dyn ::core::fmt::Debug", "3"]);
    assert_eq!(
        write(&object, parse_quote!(&'a T)),
        t(parse_quote!(&'a dyn ::core::fmt::Debug))
    );
    // `Self` is a trait object too.
    let object: Type = parse_quote!(dyn Send + Sync);
    assert_eq!(
        write_as(&[], parse_quote!(&'a Self), &object),
        t(parse_quote!(&'a (dyn Send + Sync)))
    );
}

// ── F62 ──────────────────────────────────────────────────────────────────────

/// F62 (fixed): macros wrote the Repeater's projection by hand.
///
/// Trigger: `<X as Repeater<#args, #id, #ix>>::Type` in a macro broke in its users' crates when
/// the Repeater's parameters changed (`ID` and `NTH` became one `KEY`).
///
/// Fix: `Projection::of(self_ty, context, entry)` writes `<X as Repeater<args, KEY>>::Type` for
/// the Repeater path it was given, with `KEY = Projection::key(entry)`.
#[test]
fn f62_the_repeater_path_and_the_projection_are_what_a_macro_wrote_by_hand() {
    let tr: ItemTrait = parse_quote! {
        trait Tr<'a, T: 'a, const N: usize> { fn f(&self, x: &'a Wrap<T, N>) -> Rel; }
    };
    let mut leaker = common::std_leaker();
    leaker.intern_trait(&tr);
    let referrer = leaker.result().finish().unwrap();
    let entry = common::entry(&referrer, entry_index(&referrer, "& 'a Wrap < T , N >"));
    let args: Type = parse_quote!((&'a (), T, [(); N], &'a T));
    let host: Type = parse_quote!(Host<T>);
    for repeater in [
        "crate::Repeater",
        "::repo::Repeater",
        "repo::m::Repeater",
        "Repeater",
    ] {
        let repeater: Path = parse_str(repeater).unwrap();
        for (id, ix) in [(7u64, 3usize), (0, 0), (u64::MAX, 1000)] {
            let ix: EntryId = parse_quote!(#ix);
            // As a macro writes it by hand, with the key of `id` and `ix`.
            let key = common::key(id, ix);
            let by_hand = quote!(<#host as #repeater<#args, #key>>::Type);
            assert_eq!(
                common::projection(repeater.clone(), id)
                    .of(&host, entry, ix)
                    .unwrap(),
                parse2::<Type>(by_hand).unwrap()
            );
        }
    }
    // As the tokens a test writes.
    let repeater = common::repeater();
    let three: EntryId = parse_quote!(3);
    let key = common::key(7, three);
    assert_eq!(
        common::projection(repeater.clone(), 7)
            .of(&parse_quote!(Self), entry, three)
            .unwrap(),
        parse_quote!(<Self as crate::Repeater<(&'a (), T, [(); N], &'a T), #key>>::Type)
    );
}

/// F62: the impl that `RepeaterImpl` writes as tokens (with the `KEY`, which `path` lacks) is the
/// impl a macro writes by hand, for `ImplSelf::Param`, `Marker` and `Bucket`.
#[test]
fn f62_the_impl_written_through_the_path_is_the_impl_written_by_hand() {
    let traits: [ItemTrait; 2] = [
        parse_quote! {
            trait Tr<'a, T: 'a, const N: usize> {
                type Out;
                fn f(&self, x: &'a Wrap<T, N>) -> Rel;
                fn g(&self) -> Wrap2<Self::Out>;
            }
        },
        parse_quote! {
            trait Tr<'a, T: 'a, const N: usize> {
                fn f(&self, x: &'a Wrap<T, N>) -> Rel;
            }
        },
    ];
    let repeater = common::repeater();
    let (param, marker): (Ident, Type) = (parse_quote!(__S), parse_quote!(crate::Marker));
    let id = 42u64;
    let mut written = 0;
    for tr in &traits {
        let mut leaker = common::std_leaker();
        leaker.intern_trait(tr);
        let result = leaker.result();
        result.clone().finish().unwrap();
        for target in [
            ImplSelf::Param(param.clone()),
            ImplSelf::Marker(marker.clone()),
            ImplSelf::Bucket {
                param: param.clone(),
            },
        ] {
            let impls =
                match result.repeater_impls(&Projection::new(repeater.clone(), id, target.clone()))
                {
                    Ok(impls) => impls,
                    // The entry of `g` names `Self`.
                    Err(_) => continue,
                };
            for r in impls {
                let entry = r.entry;
                let (impl_generics, _, where_clause) = r.generics.split_for_impl();
                let (args, self_ty, ty) = (common::impl_args(&r), &r.self_ty, &r.ty);
                let key = common::key(id, entry);
                let by_hand: ItemImpl = parse_quote! {
                    impl #impl_generics crate::Repeater<#args, #key> for #self_ty #where_clause {
                        type Type = #ty;
                    }
                };
                // `ToTokens` of the impl writes it too, with the `KEY`, which `path` has not.
                assert_eq!(
                    r.path,
                    parse_quote!(crate::Repeater<#args>),
                    "{} {:?}",
                    entry,
                    target
                );
                let through_tokens: ItemImpl = parse_quote!(#r);
                assert_eq!(through_tokens, by_hand, "{} {:?}", entry, target);
                written += 1;
            }
        }
    }
    assert!(written >= 12, "{}", written);
}

// ── F63 ──────────────────────────────────────────────────────────────────────

/// The copy of the trait `item` with the entries replaced by `P<ix>`, the errors (the reason and
/// the cause of each), and the entries, with the rule `config` given to the Leaker.
fn f63_leak<A: Allow>(config: A, item: &ItemTrait) -> (String, Vec<(Reason, String)>, Vec<String>) {
    let mut leaker = Leaker::with_rule(config);
    let context = leaker.intern_trait(item);
    let result = leaker.result();
    let errors = result
        .errors()
        .iter()
        .map(|error| (error.reason, s(&error.cause)))
        .collect();
    let referrer = result.recovered();
    let mut copy = item.clone();
    common::visitor(&referrer, context, common::placeholder).visit_item_trait_mut(&mut copy);
    (s(&copy), errors, common::type_texts(&referrer))
}

/// F63 (fixed): a method lifetime that the output names and only leaked types keep was reported on
/// the outermost of them only, and the types inside it that keep it too were still leaked.
///
/// Trigger: `fn m<'x>(&mut self, a0: crate::base::B2<Result<RU<255>, I0>, *mut
/// crate::base::BL<'x, i64>>) -> &'x u8` in a trait: `B2<..>` is reported
/// (`UnconstrainedLifetime`) and only its head is written as is, so `BL<'x, i64>` stayed an entry
/// and `'x` was only in a projection: E0195. No error named `BL`, so allowing every cause did not
/// help.
///
/// Fix: after the types that are errors, the check looks again along the ways to the lifetime,
/// past them: while every way still meets a type leaked whole, the next type on each way is an
/// error too, with its own head path (`B2`, then `BL`) and the same signature mark. A way that
/// reaches the lifetime ends it (`B2<A, &'x u8>` is one error). The optional types on the way are
/// no entries either (`::ext::W` in `B2<::ext::W<BL<'x, i64>>>`).
#[test]
fn f63_the_types_that_hold_a_method_lifetime_inside_a_type_written_as_it_is_are_errors_too() {
    let item: ItemTrait = parse_quote!(
        trait Tr {
            fn m<'x>(
                &mut self,
                a0: crate::base::B2<Result<RU<255>, I0>, *mut crate::base::BL<'x, i64>>,
            ) -> &'x u8;
        }
    );
    let unconstrained = Reason::UnconstrainedLifetime;
    let b2 = "crate :: base :: B2 < Result < RU < 255 > , I0 > , * mut crate :: base :: BL < 'x , i64 > >";
    let bl = "crate :: base :: BL < 'x , i64 >";
    let mut leaker = common::std_leaker();
    leaker.intern_trait(&item);
    let result = leaker.result();
    // The outer type first, then the one in it, each with its own path.
    let errors: Vec<(Reason, String, Vec<String>)> = result
        .errors()
        .iter()
        .map(|e| (e.reason, s(&e.cause), e.paths.iter().map(s).collect()))
        .collect();
    assert_eq!(
        errors,
        [
            (
                unconstrained,
                b2.to_string(),
                vec!["crate :: base :: B2".to_string()]
            ),
            (
                unconstrained,
                bl.to_string(),
                vec!["crate :: base :: BL".to_string()]
            ),
        ]
    );
    // `Result<RU<255>, I0>` is the only entry: both heads are written as they are, in this
    // signature only (`sig ('x)`).
    let referrer = result.recovered();
    assert_eq!(
        common::type_texts(&referrer),
        ["Result < RU < 255 > , I0 >"]
    );
    assert_eq!(
        parts_of(&referrer),
        [
            "type (crate :: base :: B2 < Result < RU < 255 > , I0 > , * mut crate :: base :: BL < 'x , i64 > >) 0 sig ('x)",
            "type (crate :: base :: BL < 'x , i64 >) 0 sig ('x)",
        ]
    );
    let (copy, _, _) = f63_leak(common::std_config(), &item);
    assert_eq!(
        copy,
        "trait Tr { fn m < 'x > (& mut self , a0 : crate :: base :: B2 < P < 0usize > , * mut crate :: base :: BL < 'x , i64 > > ,) -> & 'x u8 ; }"
    );

    // A rule that allows the two paths leaves no error, in the analysis or after it, and the copy
    // is the same.
    let base: Path = parse_quote!(crate::base);
    let allow_base = starts_with(base);
    let rule = Rules::new().primitive().prelude().or_if(allow_base.clone());
    let (allowed_copy, errors, entries) = f63_leak(rule, &item);
    assert_eq!(errors, []);
    assert_eq!(entries, ["Result < RU < 255 > , I0 >"]);
    assert_eq!(allowed_copy, copy);
    let mut leaker = common::std_leaker();
    let context = leaker.intern_trait(&item);
    let referrer = leaker
        .result()
        .allow(Rules::new().or_if(allow_base))
        .finish()
        .unwrap();
    let mut after = item.clone();
    common::visitor(&referrer, context, common::placeholder).visit_item_trait_mut(&mut after);
    assert_eq!(s(&after), copy);

    // The shapes: the errors (outer types first, the types of a way in order, each way's in the
    // order of the ways) and the entries that remain.
    let cases: Vec<(ItemTrait, Vec<&str>, Vec<&str>)> = vec![
        // Three types on the way.
        (
            parse_quote!(
                trait Tr {
                    fn m<'x>(&self, a: R2<R3<Rel<'x>>>) -> &'x u8;
                }
            ),
            vec![
                "R2 < R3 < Rel < 'x > > >",
                "R3 < Rel < 'x > >",
                "Rel < 'x >",
            ],
            vec![],
        ),
        // Two ways, each through a type leaked whole: both are errors.
        (
            parse_quote!(
                trait Tr {
                    fn m<'x>(&self, a: R2<Rel<'x>, *mut Rel<'x>>) -> &'x u8;
                }
            ),
            vec![
                "R2 < Rel < 'x > , * mut Rel < 'x > >",
                "Rel < 'x >",
                "Rel < 'x >",
            ],
            vec![],
        ),
        // One way keeps the lifetime as written (`&'x u8`): the first round is the whole error,
        // and `Rel<'x>` is an entry as it was.
        (
            parse_quote!(
                trait Tr {
                    fn m<'x>(&self, a: R2<Rel<'x>, &'x u8>) -> &'x u8;
                }
            ),
            vec!["R2 < Rel < 'x > , & 'x u8 >"],
            vec!["Rel < 'x >"],
        ),
        // The same when the free way is another input.
        (
            parse_quote!(
                trait Tr {
                    fn m<'x>(&self, a: R2<Rel<'x>>, b: &'x u8) -> &'x u8;
                }
            ),
            vec![],
            vec!["R2 < Rel < 'x > >"],
        ),
        // An optional entry on the way (`::ext::W` is not allowed) is not one: it holds the
        // lifetime, so the copy must not replace it.
        (
            parse_quote!(
                trait Tr {
                    fn m<'x>(&self, a: R2<::ext::W<Rel<'x>>>) -> &'x u8;
                }
            ),
            vec!["R2 < :: ext :: W < Rel < 'x > > >", "Rel < 'x >"],
            vec![],
        ),
        // No error, and as before: the lifetime is not named by the output.
        (
            parse_quote!(
                trait Tr {
                    fn m<'x>(&self, a: R2<Rel<'x>>);
                }
            ),
            vec![],
            vec!["R2 < Rel < 'x > >"],
        ),
    ];
    for (item, causes, entries) in cases {
        let (_, errors, found) = f63_leak(common::std_config(), &item);
        let (reasons, found_causes): (Vec<Reason>, Vec<String>) = errors.into_iter().unzip();
        assert_eq!(reasons, vec![unconstrained; causes.len()], "{}", s(&item));
        assert_eq!(found_causes, causes, "{}", s(&item));
        assert_eq!(found, entries, "{}", s(&item));
    }
}

// ── F64 ──────────────────────────────────────────────────────────────────────

/// F64 (fixed): a bound that names its generic through a type alias was not an unstable bound.
///
/// Trigger: `impl<T, U: Iterator<Item = u8>> S<T, U> where U: Fn(WIt<U>) -> u8 { fn m(&self, a:
/// u8) where T: Fn(&Self) -> u8 {} }`, `type WIt<T> = <T as Iterator>::Item;`: E0277 "expected an
/// `Fn(u8)` closure, found `U`" in the Referrer's crate.
///
/// Cause: an entry that names `Self` keeps the bounds that F26 leaves out of other contexts, so
/// its impl proves `U: Fn(u8)` against the method's environment, which has `U: Fn(WIt<U>)` as
/// written (the cycle of F21, F25, F26). A projection there was rejected, an alias was not.
///
/// Fix: a bound that names its type parameter in its path's arguments is unstable (an
/// `UnprovableBound` error) if another bound of it fixes an associated type, or an equation on it
/// is in a scope around the entry. Limitation: a struct there (`U: Fn(WFoo<U>)`) is rejected too,
/// as it cannot be told from an alias; a bound of another generic behind an alias (`T:
/// Fn(WIt<U>)` with the equation on `U`) is not covered.
#[test]
fn f64_an_alias_in_a_bound_that_the_context_keeps_is_rejected_in_a_predicate() {
    let wit = "Fn (:: p :: WIt < U >) -> u8";
    // The item, the type that the error names, the bound it blames, and what the bound names.
    let rejected: Vec<(Item, &str, String, Blames)> = vec![
        // `Self` of an impl: the bug.
        (
            parse_quote!(
                impl<T, U: Iterator<Item = u8>> S<T, U>
                where
                    U: Fn(::p::WIt<U>) -> u8,
                {
                    fn m(&self, a: u8)
                    where
                        T: Fn(&Self) -> u8,
                    {
                    }
                }
            ),
            "Self",
            format!("U : {} + Iterator < Item = u8 >", wit),
            Blames::Hidden("U", wit),
        ),
        // The projection written was rejected before the fix.
        (
            parse_quote!(
                impl<T, U: Iterator<Item = u8>> S<T, U>
                where
                    U: Fn(<U as Iterator>::Item) -> u8,
                {
                    fn m(&self, a: u8)
                    where
                        T: Fn(&Self) -> u8,
                    {
                    }
                }
            ),
            "Self",
            "U : Fn (< U as Iterator > :: Item) -> u8 + Iterator < Item = u8 >".to_string(),
            Blames::Projection("< U as Iterator > :: Item"),
        ),
        // The alias inside a path of the standard library still hides it.
        (
            parse_quote!(
                impl<T, U: Iterator<Item = u8>> S<T, U>
                where
                    U: Into<::core::option::Option<::p::WIt<U>>>,
                {
                    fn m(&self, a: u8)
                    where
                        T: Fn(&Self) -> u8,
                    {
                    }
                }
            ),
            "Self",
            "U : Into < :: core :: option :: Option < :: p :: WIt < U > > > + Iterator < Item = u8 >"
                .to_string(),
            Blames::Hidden(
                "U",
                "Into < :: core :: option :: Option < :: p :: WIt < U > > >",
            ),
        ),
        // A struct instead of the alias is rejected too (the price: the Leaker cannot tell them).
        (
            parse_quote!(
                impl<T, U: Iterator<Item = u8>> S<T, U>
                where
                    U: Fn(::p::WFoo<U>) -> u8,
                {
                    fn m(&self, a: u8)
                    where
                        T: Fn(&Self) -> u8,
                    {
                    }
                }
            ),
            "Self",
            "U : Fn (:: p :: WFoo < U >) -> u8 + Iterator < Item = u8 >".to_string(),
            Blames::Hidden("U", "Fn (:: p :: WFoo < U >) -> u8"),
        ),
        // An entry that names `Self` in a trait, and one that depends on it through `Self: 'x`
        // (F47), in the predicate of a method.
        (
            parse_quote!(
                pub trait Tr<T, U: Iterator<Item = u8>>
                where
                    U: Fn(::p::WIt<U>) -> u8,
                {
                    fn m(&self)
                    where
                        Wrap<Self>: Default;
                }
            ),
            "Wrap < Self >",
            format!("U : {} + Iterator < Item = u8 >", wit),
            Blames::Hidden("U", wit),
        ),
        (
            parse_quote!(
                pub trait Tr<T: Iterator<Item = u8>>
                where
                    T: Fn(::p::WIt<T>),
                {
                    fn m<'x, Y: 'x + Iterator<Item = u8>>(&'x self)
                    where
                        Self: 'x,
                        Y: Fn(::p::WIt<Y>) -> u8,
                        Rel<T, Y>: Default;
                }
            ),
            "Rel < T , Y >",
            "T : Fn (:: p :: WIt < T >) + Iterator < Item = u8 >".to_string(),
            Blames::Hidden("T", "Fn (:: p :: WIt < T >)"),
        ),
        // The equation is in a predicate of the method (F21), not in the scope of the generic: the
        // bound is not left out, and the entry is rejected.
        (
            parse_quote!(
                impl<T: Iterator + Fn(::p::WIt<T>)> S<T> {
                    fn m(&self, a: u8)
                    where
                        T: Iterator<Item = u8>,
                        Rel<T>: Default,
                    {
                    }
                }
            ),
            "Rel < T >",
            "T : Fn (:: p :: WIt < T >) + Iterator".to_string(),
            Blames::Hidden("T", "Fn (:: p :: WIt < T >)"),
        ),
    ];
    for (item, ty, bound, blames) in &rejected {
        assert_unstable(
            item,
            f19_copy_with(common::p_leaker(), item),
            ty,
            bound,
            *blames,
        );
    }
    // The equation as a bound of the method's generic: it is in the scope that declares the
    // generic, so F26 leaves the bound out and the entry is accepted.
    let method_generic: Item = parse_quote!(
        pub trait Tr {
            fn m<T: Iterator<Item = u8> + Fn(::p::WIt<T>)>(&self, a: u8)
            where
                Rel<T>: Default;
        }
    );
    let (referrer, _) = f19_copy_with(common::p_leaker(), &method_generic).unwrap();
    assert_eq!(
        context_text_with_where(&referrer, entry_index(&referrer, "Rel < T >")),
        "< T : Iterator < Item = u8 > > where Rel < T > : Default"
    );
}

/// F64: the controls that compile stay accepted.
#[test]
fn f64_what_compiles_stays_accepted() {
    let accepted: Vec<(&str, Item)> = vec![
        // Nothing fixes the `Item` of `U`: it is rigid, and the bound is as the impl has it.
        (
            "rigid",
            parse_quote!(
                impl<T, U: Iterator> S<T, U>
                where
                    U: Fn(::p::WIt<U>) -> u8,
                {
                    fn m(&self, a: u8)
                    where
                        T: Fn(&Self) -> u8,
                    {
                    }
                }
            ),
        ),
        // `Self` outside a predicate: the Repeater's impl is proved when the type is named.
        (
            "position",
            parse_quote!(
                impl<T, U: Iterator<Item = u8>> S<T, U>
                where
                    U: Fn(::p::WIt<U>) -> u8,
                {
                    fn m(&self, a: Option<Self>) {}
                }
            ),
        ),
        // An entry that does not name `Self` leaves the bound out of its context (F26).
        (
            "left out",
            parse_quote!(
                impl<T, U: Iterator<Item = u8>> S<T, U>
                where
                    U: Fn(::p::WIt<U>) -> u8,
                {
                    fn m(&self, a: u8)
                    where
                        Rel<U>: Default,
                    {
                    }
                }
            ),
        ),
        // The generic itself, a type that syntax builds and a path of the standard library in the
        // arguments are no alias of a projection.
        (
            "generic itself",
            parse_quote!(
                impl<T, U: Iterator<Item = u8> + From<U> + Fn(&U) + Into<(U, [U; 2])>> S<T, U>
                where
                    U: Into<::core::option::Option<U>>,
                {
                    fn m(&self, a: u8)
                    where
                        T: Fn(&Self) -> u8,
                    {
                    }
                }
            ),
        ),
        // A bound that does not name its generic.
        (
            "not named",
            parse_quote!(
                impl<T, U: Iterator<Item = u8>> S<T, U>
                where
                    U: Fn(u8) -> u8,
                {
                    fn m(&self, a: u8)
                    where
                        T: Fn(&Self) -> u8,
                    {
                    }
                }
            ),
        ),
    ];
    for (name, item) in &accepted {
        assert_accepted_as(
            &format!("{}: `{}`", name, s(item)),
            f19_copy_with(common::p_leaker(), item),
        );
    }
    // The limitation of F26: the bound of another generic behind an alias, which the context
    // leaves as it is.
    let other: Item = parse_quote!(
        pub trait Tr<T: Iterator<Item = u8>, U: Fn(::p::WIt<T>)> {
            fn m(&self)
            where
                Rel<U>: Default;
        }
    );
    f19_copy_with(common::p_leaker(), &other).unwrap();
}

// ── F65 ──────────────────────────────────────────────────────────────────────

/// How `f65_the_default_of_a_nested_trait_is_found` analyzes the trait.
#[derive(Clone, Copy, Debug)]
enum F65Shape {
    /// `intern_trait()`: the trait itself.
    InternTrait,
    /// `enter_generics()` and `view.visit_item()`: the trait is nested in a scope.
    EnterGenerics,
}

const F65_SHAPES: [F65Shape; 2] = [F65Shape::InternTrait, F65Shape::EnterGenerics];

/// Analyzes `tr` as `shape` says and returns the result and the context of the visitor that writes
/// the defaults of the trait.
fn f65_analysis(shape: F65Shape, tr: &ItemTrait) -> (Referrer, ContextId) {
    let mut leaker = common::std_leaker();
    let context = match shape {
        F65Shape::InternTrait => leaker.intern_trait(tr),
        F65Shape::EnterGenerics => {
            let item = Item::Trait(tr.clone());
            leaker
                .enter_generics(&Generics::default(), |view| view.visit_item(&item))
                .0
        }
    };
    (leaker.result().finish().unwrap(), context)
}

/// F65 (fixed): the default of a trait nested in a scope was left as written when a bound of an
/// earlier parameter names `Self`.
///
/// Trigger: `pub trait Tr<T: AsRef<Self>, U = R1<T>>` visited by `view.visit_item()`: the default
/// `R1<T>` was not written through the Repeater, so the copy named `R1`, which the Referrer's crate
/// does not have. With `intern_trait()`, or with `T: Clone`, it was written.
///
/// Cause: a nested trait's default was analyzed with the trait's own `Self`, so its entry was for
/// the implementors (`Self: Tr<T, U>`); that names `U`, the parameter the default fills, so it
/// cannot be written before `U` exists. `T: Clone` gives the entry no `Self`.
///
/// Fix: a nested trait analyzes the defaults that do not name `Self` alone, as `intern_trait()`
/// does: the entry has no `Self` requirement, and the Visitor writes it like the default of the
/// trait itself. A default that names `Self` (`U = R1<Self>`) keeps its entry for the trait's
/// `Self`, left as written without `with_nested_self()`.
#[test]
fn f65_the_default_of_a_nested_trait_is_found() {
    for bound in ["AsRef<Self>", "Clone"] {
        let tr: ItemTrait = parse_str(&format!(
            "pub trait Tr<T: {}, U = R1<T>> {{ fn f(&self, u: U); }}",
            bound
        ))
        .unwrap();
        for shape in F65_SHAPES {
            let what = format!("{}, {:?}", bound, shape);
            let (referrer, context) = f65_analysis(shape, &tr);
            // One entry, for any implementor that the bound holds for.
            assert_eq!(referrer.iter().count(), 1, "{}", what);
            let ix = entry_index(&referrer, "R1 < T >");
            assert!(common::entry(&referrer, ix).self_ty.is_none(), "{}", what);
            assert_eq!(
                common::bucket_self(&referrer, 0, common::entry_id_at(&referrer, ix)).is_none(),
                bound != "Clone",
                "{}",
                what
            );

            // The visitor writes the default of the trait that it visits, in each shape.
            let mut calls = Vec::new();
            let visited = common::visited_trait(&referrer, context, &tr, |ty, ix, _| {
                calls.push((s(&ty), ix.index()));
                parse_quote!(<Self as R<#ix>>::Type)
            });
            assert_eq!(calls, [("R1 < T >".to_string(), ix)], "{}", what);
            match &visited.generics.params[1] {
                GenericParam::Type(TypeParam {
                    default: Some(default),
                    ..
                }) => assert_eq!(s(default), "< Self as R < 0usize > > :: Type", "{}", what),
                other => panic!("{}: {}", what, s(other)),
            }
            // The scope of the trait completes the arguments of an impl.
            if let F65Shape::InternTrait = shape {
                let key = common::key(0, common::entry_id_at(&referrer, ix));
                let expected: Vec<GenericArgument> = vec![
                    parse_quote!(u8),
                    parse_quote!(<Self as Repeater<(u8,), #key>>::Type),
                ];
                let completed =
                    common::completed_args(&referrer, context, &tr.generics, &arguments(&["u8"]))
                        .unwrap();
                assert_eq!(completed, expected, "{}", what);
                // A default that is given is not completed.
                let given = arguments(&["u8", "u16"]);
                let completed =
                    common::completed_args(&referrer, context, &tr.generics, &given).unwrap();
                assert_eq!(s(&completed[1]), "u16", "{}", what);
            }
        }
    }

    // A default that names `Self` has the entry for the trait's `Self` only, which cannot be
    // completed before `U` exists: with no closure it is returned as written, as before.
    let tr: ItemTrait = parse_quote!(
        pub trait Tr<T, U = R1<Self>> {
            fn f(&self, u: U);
        }
    );
    let (referrer, context) = f65_analysis(F65Shape::EnterGenerics, &tr);
    let ix = entry_index(&referrer, "R1 < Self >");
    assert!(matches!(
        common::entry(&referrer, ix).self_ty.as_ref(),
        Some(SelfRequirement::Bound(_))
    ));
    let visited = common::visited_trait(&referrer, context, &tr, |_, _, _| unreachable!());
    match &visited.generics.params[1] {
        GenericParam::Type(TypeParam {
            default: Some(default),
            ..
        }) => assert_eq!(s(default), "R1 < Self >"),
        other => panic!("{}", s(other)),
    }
}

// ── F66 ──────────────────────────────────────────────────────────────────────

/// F66 (fixed): `substitute_args()` and `complete_args()` doubled the braces of a braced use of a
/// const parameter.
///
/// Trigger: the site `<const N: usize>` with `[M]`: `[u8; { N }]` became `[u8; { { M } }]`, which
/// rustc rejects for a generic `M` ("generic parameters may not be used in const operations");
/// `Rel<{ N }>` and a default `U = [u8; { N }]` likewise.
///
/// Cause: the substitution walked into the block and replaced the path `N` there by the braced
/// argument, so the argument was braced twice.
///
/// Fix: a block that holds the parameter alone is the parameter, so the argument is braced once.
/// Any other block is walked.
#[test]
fn f66_a_braced_const_parameter_is_braced_once_at_a_use_site() {
    let params: Punctuated<GenericParam, Token![,]> = parse_quote!(const N: usize);
    let site = |arg: &str| arguments(&[arg]);
    let substituted = |site: &[GenericArgument], written: &str| {
        let ty: Type = parse_str(written).unwrap();
        s(&common::substitute_type(&ty, common::answers(&params, site), None).unwrap())
    };
    for (written, arg, expected) in [
        ("[u8; { N }]", "M", "[u8 ; { M }]"),
        ("Rel<{ N }>", "M", "Rel < { M } >"),
        ("[u8; N]", "M", "[u8 ; { M }]"),
        ("Rel<N>", "M", "Rel < M >"),
        ("[u8; { N }]", "3", "[u8 ; { 3 }]"),
        ("[u8; { N }]", "{ M }", "[u8 ; { M }]"),
        ("Rel<{ N }>", "{ 1 + 1 }", "Rel < { 1 + 1 } >"),
        // A block in a block is the parameter braced once too.
        ("[u8; { { N } }]", "M", "[u8 ; { M }]"),
        // A block that does not hold the parameter alone is walked as before.
        ("[u8; { N + 1 }]", "M", "[u8 ; { ({ M }) + 1 }]"),
    ] {
        assert_eq!(
            substituted(&site(arg), written),
            expected,
            "{} with {}",
            written,
            arg
        );
        // The default of a parameter that names `N` so is completed the same way
        // (`complete_args()` and `substitute_args()` replace the parameter alike).
        let default: Type = parse_str(written).unwrap();
        let params: Punctuated<GenericParam, Token![,]> =
            parse_quote!(const N: usize, U = #default);
        let completed = complete_args_of(&params, &arguments(&[arg])).unwrap();
        assert_eq!(s(&completed[1]), expected, "{} with {}", written, arg);
    }
    // The default of a parameter that names the const parameter the same way is completed once braced.
    let tr: ItemTrait = parse_quote!(
        pub trait Tr<const N: usize, U = [u8; { N }]> {
            fn f(&self, u: U);
        }
    );
    let mut leaker = common::std_leaker();
    let context = leaker.intern_trait(&tr);
    let referrer = leaker.result().finish().unwrap();
    let completed =
        common::completed_args(&referrer, context, &tr.generics, &arguments(&["M"])).unwrap();
    let completed: Vec<String> = completed.iter().map(s).collect();
    assert_eq!(completed, ["M", "[u8 ; { M }]"]);
}

// ── F67 ──────────────────────────────────────────────────────────────────────

/// F67 (fixed): of the optional types around a trait object bounded by a method lifetime, only the
/// outermost was kept from being an entry.
///
/// Trigger: `fn m<'x>(a0: ::std::vec::Vec<::std::boxed::Box<dyn ::core::ops::Fn(R0) + 'x>>) ->
/// &'x ()` in a trait, with `::std` not allowed. Only `Vec` was written as is; `Box<dyn Fn(R0) +
/// 'x>` holds `R0`, so it was a required entry, and the copy had `'x` only in a projection: E0195.
///
/// Cause: F18 keeps the optional types between an input and the lifetime from being entries, but
/// its scan stopped at the first such type.
///
/// Fix: the scan goes on inside the types that stopped it, until a way reaches the lifetime or
/// none is left. Every optional type on the way is written as it is, and only `R0` is replaced. A
/// leaked type that keeps the lifetime is still the error of F63 (`Wrap<'x, R0>`).
#[test]
fn f67_the_optional_types_on_the_way_to_a_method_lifetime_are_all_written_as_they_are() {
    let shapes: Vec<ItemTrait> = vec![
        parse_quote!(
            trait Tr {
                fn m<'x>(
                    a0: ::std::vec::Vec<::std::boxed::Box<dyn ::core::ops::Fn(R0) + 'x>>,
                ) -> &'x ();
            }
        ),
        parse_quote!(
            trait Tr {
                fn m<'x>(
                    a0: ::std::vec::Vec<
                        ::std::option::Option<::std::boxed::Box<dyn ::core::ops::Fn(R0) + 'x>>,
                    >,
                ) -> &'x ();
            }
        ),
        parse_quote!(
            trait Tr {
                fn m<'x>(
                    a0: ::std::vec::Vec<
                        ::std::boxed::Box<::std::boxed::Box<dyn ::core::ops::Fn(R0) + 'x>>,
                    >,
                ) -> &'x ();
            }
        ),
        parse_quote!(
            trait Tr {
                fn m<'x>(
                    a0: ::std::vec::Vec<(::std::boxed::Box<dyn ::core::ops::Fn(R0) + 'x>, u8)>,
                ) -> &'x ();
            }
        ),
        // Two inputs, each with the lifetime in a trait object under a crate's type.
        parse_quote!(
            trait Tr {
                fn m<'x>(
                    a0: ::std::vec::Vec<::std::boxed::Box<dyn ::core::ops::Fn(R0) + 'x>>,
                    a1: ::ext::Opt<::std::boxed::Box<dyn ::core::ops::Fn(R0) + 'x>>,
                ) -> &'x ();
            }
        ),
    ];
    for item in &shapes {
        // Only `R0` is an entry, and it is no error; the types around stay as written, with `'x`.
        let (copy, errors, entries) = f63_leak(Rules::new().primitive(), item);
        assert_eq!(errors, [], "{}", s(item));
        assert_eq!(entries, ["R0"], "{}", s(item));
        assert_eq!(
            copy,
            s(item).replace("(R0)", "(P < 0usize >)"),
            "{}",
            s(item)
        );
        // A Leaker that allows every crate writes the same.
        let (trusted, errors, entries) = f63_leak(Rules::new().primitive().external_crates(), item);
        assert_eq!((errors, entries), (vec![], vec!["R0".to_string()]));
        assert_eq!(trusted, copy);
    }

    // With no `R0` in the object, no optional type is an entry either: `'x` stays in the input.
    let (_, errors, entries) = f63_leak(
        Rules::new().primitive(),
        &parse_quote!(
            trait Tr {
                fn m<'x>(a0: ::std::vec::Vec<::std::boxed::Box<dyn ::ext::Tr0 + 'x>>) -> &'x ();
            }
        ),
    );
    assert_eq!((errors, entries), (vec![], vec![]));

    // Controls: where some input keeps the lifetime (`&'x u8`), or the output does not name it, the
    // outermost optional type is the entry, as before.
    let whole =
        ":: std :: vec :: Vec < :: std :: boxed :: Box < dyn :: core :: ops :: Fn (R0) + 'x > >";
    for item in [
        parse_quote!(
            trait Tr {
                fn m<'x>(
                    a0: ::std::vec::Vec<::std::boxed::Box<dyn ::core::ops::Fn(R0) + 'x>>,
                    a1: &'x u8,
                ) -> &'x ();
            }
        ),
        parse_quote!(
            trait Tr {
                fn m<'x>(
                    a0: ::std::vec::Vec<::std::boxed::Box<dyn ::core::ops::Fn(R0) + 'x>>,
                ) -> u8;
            }
        ),
    ] {
        let (_, errors, entries) = f63_leak(Rules::new().primitive(), &item);
        assert_eq!((errors, entries), (vec![], vec![whole.to_string()]));
    }

    // A leaked type that holds the lifetime is still the error of F63, and `R0` the entry.
    let (_, errors, entries) = f63_leak(
        Rules::new().primitive(),
        &parse_quote!(
            trait Tr {
                fn m<'x>(a0: ::std::vec::Vec<::ext::Wrap<'x, R0>>) -> &'x ();
            }
        ),
    );
    assert_eq!(errors, [(Reason::UnconstrainedLifetime, "R0".to_string())]);
    assert_eq!(entries, ["R0"]);
}

// ── F68 ──────────────────────────────────────────────────────────────────────

/// What a Referrer makes of `item` (analyzed with the rule `config`): the errors of the analysis,
/// the entries (as written), and the self type of the copy, with the entries replaced by `P<ix>`.
fn f68_copy<A: Allow>(config: A, item: &ItemImpl) -> (Vec<Reason>, Vec<String>, String) {
    let mut leaker = Leaker::with_rule(config);
    let context = leaker.intern_impl(item);
    let result = leaker.result();
    let reasons = result.errors().iter().map(|e| e.reason).collect();
    let referrer = result.recovered();
    let entries = common::type_texts(&referrer);
    let mut copy = item.clone();
    common::visitor(&referrer, context, common::placeholder).visit_item_impl_mut(&mut copy);
    (reasons, entries, s(&*copy.self_ty))
}

/// F68 (fixed): an optional self type of an impl was an entry when another unstable bound hid the
/// bound that names `Self`.
///
/// Trigger: `impl ::ext::BI<0> { fn m<X>(&self, a0: X) where X: AsRef<(&'static ::ext::B0, bool)>,
/// X: Fn(&Self) -> u8, crate::base::B1<X>: Default { .. } }` with `::ext` not allowed: E0277
/// "expected an `Fn(&BI<0>)` closure, found `X`". The context of `B1<X>` has `X: Fn(&Self)`, and
/// the copy replaced the self type, so `Self` was a projection that rustc does not normalize where
/// it proves the predicate.
///
/// Cause: the check that keeps optional types out of such cycles looked at the first unstable
/// bound of the context only (`X: AsRef<..>`, whatever the order), so only `B0` was blocked.
///
/// Fix: every unstable bound counts, so one bound that names `Self` keeps the optional self type
/// from being an entry, and the copy writes `::ext::BI<0>` as it is. No error.
#[test]
fn f68_every_unstable_bound_keeps_an_optional_self_type_from_being_an_entry() {
    let b1 = "crate :: base :: B1 < X >".to_string();
    let kept = ":: ext :: BI < 0 >";
    // The item, and the self type of its copy. The only entry is `B1<X>`; the self type, and `B0`,
    // are no optional entries, and the copy writes the self type as it is.
    let cases: Vec<(ItemImpl, &str)> = vec![
        (
            parse_quote!(
                impl ::ext::BI<0> {
                    fn m<X>(&self, a0: X)
                    where
                        X: ::core::convert::AsRef<(&'static ::ext::B0, bool)>,
                        X: ::core::ops::Fn(&Self) -> u8,
                        crate::base::B1<X>: ::core::default::Default,
                    {
                        loop {}
                    }
                }
            ),
            kept,
        ),
        // The same with the bounds in the other order, with the bounds in the generics of the
        // method, and with a bound that names a projection of the generic before the one that
        // names `Self` (that was no entry before, a control).
        (
            parse_quote!(
                impl ::ext::BI<0> {
                    fn m<X>(&self, a0: X)
                    where
                        X: ::core::ops::Fn(&Self) -> u8,
                        X: ::core::convert::AsRef<(&'static ::ext::B0, bool)>,
                        crate::base::B1<X>: ::core::default::Default,
                    {
                        loop {}
                    }
                }
            ),
            kept,
        ),
        (
            parse_quote!(
                impl ::ext::BI<0> {
                    fn m<X: ::core::convert::AsRef<(&'static ::ext::B0, bool)> + ::core::ops::Fn(&Self) -> u8>(&self, a0: X)
                    where
                        crate::base::B1<X>: ::core::default::Default,
                    {
                        loop {}
                    }
                }
            ),
            kept,
        ),
        (
            parse_quote!(
                impl ::ext::BI<0> {
                    fn m<X>(&self, a0: X)
                    where
                        X: ::core::iter::Iterator,
                        X: ::core::ops::Fn(<X as ::core::iter::Iterator>::Item) -> u8,
                        X: ::core::ops::Fn(&Self) -> u8,
                        crate::base::B1<X>: ::core::default::Default,
                    {
                        loop {}
                    }
                }
            ),
            kept,
        ),
        // Without the bound that names an optional type, it was so already.
        (
            parse_quote!(
                impl ::ext::BI<0> {
                    fn m<X>(&self, a0: X)
                    where
                        X: ::core::ops::Fn(&Self) -> u8,
                        crate::base::B1<X>: ::core::default::Default,
                    {
                        loop {}
                    }
                }
            ),
            kept,
        ),
        // The self type names two crates (its own, and a trait that its generic is bound by): the
        // same.
        (
            parse_quote!(
                impl<T: ::other::Tr> ::ext::W<T> {
                    fn m<X>(&self, a0: X)
                    where
                        X: ::core::convert::AsRef<(&'static ::ext::B0, bool)>,
                        X: ::core::ops::Fn(&Self) -> u8,
                        crate::base::B1<X>: ::core::default::Default,
                    {
                        loop {}
                    }
                }
            ),
            ":: ext :: W < T >",
        ),
    ];
    for (item, kept) in &cases {
        let (errors, entries, self_ty) = f68_copy(Rules::new().primitive(), item);
        assert_eq!(
            (errors, entries, self_ty.as_str()),
            (vec![], vec![b1.clone()], *kept),
            "{}",
            s(item)
        );
    }

    // Controls: the entry is in a position (`a0: B1<X>`), which rustc normalizes once the
    // environment is built, so nothing is blocked: the self type is an optional entry, which the
    // copy replaces, unless the rule of the Leaker allows `::ext`: then it is no entry.
    let in_position: ItemImpl = parse_quote!(
        impl ::ext::BI<0> {
            fn m<X>(&self, a0: crate::base::B1<X>)
            where
                X: ::core::convert::AsRef<(&'static ::ext::B0, bool)>,
                X: ::core::ops::Fn(&Self) -> u8,
            {
                loop {}
            }
        }
    );
    let (errors, entries, self_replaced) = f68_copy(Rules::new().primitive(), &in_position);
    assert_eq!(errors, []);
    assert!(entries.contains(&kept.to_string()), "{:?}", entries);
    assert_eq!(self_replaced, "P < 2usize >");
    let (errors, entries, written) =
        f68_copy(Rules::new().primitive().external_crates(), &in_position);
    assert_eq!(errors, []);
    assert!(!entries.contains(&kept.to_string()), "{:?}", entries);
    assert_eq!(written, kept);

    // A self type that must be an entry is still the error of F19, whatever the bounds name.
    let relative: ItemImpl = parse_quote!(
        impl crate::base::BI<0> {
            fn m<X>(&self, a0: X)
            where
                X: ::core::convert::AsRef<(&'static ::ext::B0, bool)>,
                X: ::core::ops::Fn(&Self) -> u8,
                crate::base::B1<X>: ::core::default::Default,
            {
                loop {}
            }
        }
    );
    let (errors, _, _) = f68_copy(Rules::new().primitive(), &relative);
    assert_eq!(errors, [Reason::UnprovableBound]);
}

// ── F69 ──────────────────────────────────────────────────────────────────────

/// F69 (fixed): `substitute_args()` and `complete_args()` wrote an argument with lifetimes into a
/// function pointer or `Fn(..)` whose output elides its lifetime: E0106.
///
/// Trigger: the site `<U>` with `[&'static str]`: `fn(&::core::cell::Cell<[U; 1]>) -> &u8` became
/// `fn(&::core::cell::Cell<[&'static str; 1]>) -> &u8`, whose inputs have two lifetimes. The same
/// for `&dyn Fn(&U) -> &u8`, a `Self` with lifetimes, and a completed default `U = fn(&T) -> &u8`.
///
/// Cause: the substitution left the output as written; rustc resolves the original's elision first.
///
/// Fix: where the inputs then show more than one lifetime, the elided one is named with a fresh
/// binder: `for<'__tl_s0> fn(&'__tl_s0 ::core::cell::Cell<[&'static str; 1]>) -> &'__tl_s0 u8`.
/// Inputs that name their lifetime (`fn(&'a U) -> &u8`) give it to the output.
///
/// Limitation: a lifetime that a path hides (`fn(::core::cell::Ref<U>) -> &u8`) cannot be named:
/// the copy is as written, and E0106 for an argument with lifetimes.
#[test]
fn f69_the_elided_lifetime_of_a_signature_is_named_where_a_use_site_adds_lifetimes() {
    let params: Punctuated<GenericParam, Token![,]> = parse_quote!(U);
    let site = |arg: &str| arguments(&[arg]);
    let same_as = |site: &[GenericArgument], self_ty: Option<&Type>, ty: &str, expected: &str| {
        let ty = parse_str::<Type>(ty).unwrap();
        assert_eq!(
            s(&common::substitute_type(&ty, common::answers(&params, site), self_ty).unwrap()),
            s(&parse_str::<Type>(expected).unwrap()),
            "{}",
            s(&ty)
        )
    };
    let same =
        |site: &[GenericArgument], ty: &str, expected: &str| same_as(site, None, ty, expected);

    // The inputs show one lifetime and the argument adds one: named.
    let site_static = site("&'static str");
    same(
        &site_static,
        "fn(&::core::cell::Cell<[U; 1]>) -> &u8",
        "for<'__tl_s0> fn(&'__tl_s0 ::core::cell::Cell<[&'static str; 1]>) -> &'__tl_s0 u8",
    );
    same(
        &site_static,
        "&dyn Fn(&U) -> &u8",
        "&dyn for<'__tl_s0> Fn(&'__tl_s0 &'static str) -> &'__tl_s0 u8",
    );
    same(
        &site_static,
        "Box<dyn Fn(&U) -> &u8>",
        "Box<dyn for<'__tl_s0> Fn(&'__tl_s0 &'static str) -> &'__tl_s0 u8>",
    );
    same(
        &site_static,
        "impl Fn(&U) -> &u8",
        "impl for<'__tl_s0> Fn(&'__tl_s0 &'static str) -> &'__tl_s0 u8",
    );
    same(
        &site_static,
        "fn(&<U as Tr>::A) -> &u8",
        "for<'__tl_s0> fn(&'__tl_s0 <&'static str as Tr>::A) -> &'__tl_s0 u8",
    );
    // A named lifetime too, and a binder that the signature has already.
    let site_wrap = site("Wrap<'a>");
    same(
        &site_wrap,
        "fn(&::core::cell::Cell<[U; 1]>) -> &u8",
        "for<'__tl_s0> fn(&'__tl_s0 ::core::cell::Cell<[Wrap<'a>; 1]>) -> &'__tl_s0 u8",
    );
    same(
        &site_static,
        "for<'x> fn(&U) -> &u8",
        "for<'x, '__tl_s0> fn(&'__tl_s0 &'static str) -> &'__tl_s0 u8",
    );
    // The lifetime of a signature inside another is named apart from the one around.
    same(
        &site_static,
        "fn(&U, fn(&U) -> &u8) -> &u8",
        "for<'__tl_s0> fn(&'__tl_s0 &'static str, for<'__tl_s1> fn(&'__tl_s1 &'static str) -> \
         &'__tl_s1 u8) -> &'__tl_s0 u8",
    );
    // Two signatures side by side may use the same name.
    same(
        &site_static,
        "(fn(&U) -> &u8, fn(&U) -> &u8)",
        "(for<'__tl_s0> fn(&'__tl_s0 &'static str) -> &'__tl_s0 u8, \
         for<'__tl_s0> fn(&'__tl_s0 &'static str) -> &'__tl_s0 u8)",
    );
    // No lifetime of the argument or the type is a fresh name.
    let site_taken = site("&'__tl_s0 str");
    same(
        &site_taken,
        "fn(&U) -> &u8",
        "for<'__tl_s1> fn(&'__tl_s1 &'__tl_s0 str) -> &'__tl_s1 u8",
    );
    same(
        &site_static,
        "fn(&'__tl_s0 u8, fn(&U) -> &u8) -> &'__tl_s0 u8",
        "fn(&'__tl_s0 u8, for<'__tl_s1> fn(&'__tl_s1 &'static str) -> &'__tl_s1 u8) -> \
         &'__tl_s0 u8",
    );

    // The inputs name their lifetime: the output takes it.
    same(
        &site_static,
        "fn(&'a U) -> &u8",
        "fn(&'a &'static str) -> &'a u8",
    );
    same(&site_wrap, "fn(&'a U) -> &u8", "fn(&'a Wrap<'a>) -> &u8");
    // `Self` with lifetimes, as `Visitor::with_self()` does.
    let with_self: Type = parse_quote!(::p::S<'a>);
    same_as(
        &[],
        Some(&with_self),
        "fn(&Self) -> &u8",
        "for<'__tl_s0> fn(&'__tl_s0 ::p::S<'a>) -> &'__tl_s0 u8",
    );
    same_as(
        &[],
        Some(&with_self),
        "dyn Fn(&<Self as Tr>::A) -> &u8",
        "dyn for<'__tl_s0> Fn(&'__tl_s0 <::p::S<'a> as Tr>::A) -> &'__tl_s0 u8",
    );

    // Unchanged: no lifetime added, the lifetime already named, an output that does not elide, no
    // elision to keep (two lifetimes: not valid as written), a lifetime that a path hides.
    let site_u8 = site("u8");
    same(
        &site_u8,
        "fn(&::core::cell::Cell<[U; 1]>) -> &u8",
        "fn(&::core::cell::Cell<[u8; 1]>) -> &u8",
    );
    same(
        &site_static,
        "for<'x> fn(&'x U) -> &'x u8",
        "for<'x> fn(&'x &'static str) -> &'x u8",
    );
    same(&site_static, "fn(&U) -> u8", "fn(&&'static str) -> u8");
    same(
        &site_static,
        "fn(&U) -> &'static u8",
        "fn(&&'static str) -> &'static u8",
    );
    same(&site_static, "fn(U) -> &u8", "fn(&'static str) -> &u8");
    same(
        &site_static,
        "fn(&U, &u8) -> &u8",
        "fn(&&'static str, &u8) -> &u8",
    );
    same(
        &site_static,
        "fn(::core::cell::Ref<U>) -> &u8",
        "fn(::core::cell::Ref<&'static str>) -> &u8",
    );
    same(&[], "fn(&U) -> &u8", "fn(&U) -> &u8");

    // `complete_args()` completes the default that is such a signature.
    let tr: ItemTrait = parse_quote!(
        pub trait Tr<T, U = fn(&T) -> &u8, V = Box<dyn Fn(&T, &T) -> u8>> {
            fn f(&self, t: T) -> U;
        }
    );
    let mut leaker = common::std_leaker();
    let context = leaker.intern_trait(&tr);
    let referrer = leaker.result().finish().unwrap();
    let complete = |arg: &str| -> Vec<GenericArgument> {
        common::completed_args(&referrer, context, &tr.generics, &arguments(&[arg])).unwrap()
    };
    let strings = |args: Vec<GenericArgument>| -> Vec<String> { args.iter().map(s).collect() };
    assert_eq!(
        strings(complete("&'static str")),
        strings(arguments(&[
            "&'static str",
            "for<'__tl_s0> fn(&'__tl_s0 &'static str) -> &'__tl_s0 u8",
            "Box<dyn Fn(&&'static str, &&'static str) -> u8>",
        ]))
    );
    assert_eq!(
        strings(complete("u8")),
        strings(arguments(&[
            "u8",
            "fn(&u8) -> &u8",
            "Box<dyn Fn(&u8, &u8) -> u8>"
        ]))
    );
}

// ── F70 ──────────────────────────────────────────────────────────────────────

/// The copy of `item`, a struct that has the entry `::std::vec::Vec<T>`, that a macro makes at a
/// use site: the visitor writes the path through the Repeater by the generics of the entry (`Self`
/// is written as `self_ty` if given), and one pass writes the arguments `args` of the site into the
/// fields (`self_ty` is a type at the site).
fn f70_copy(item: &ItemStruct, args: &[&str], self_ty: Option<Type>) -> ItemStruct {
    let mut leaker = common::std_leaker();
    let context = leaker.intern_struct(item);
    let referrer = leaker.result().finish().unwrap();
    let entries: Vec<String> = common::type_texts(&referrer);
    assert_eq!(entries, [":: std :: vec :: Vec < T >"], "{}", s(item));
    let given = arguments(args);
    let repeater = common::repeater();
    let mut copy = item.clone();
    let mut visitor = common::visitor(&referrer, context, |_, ix, entry| {
        let self_ty: Type = parse_quote!(Self);
        common::projection(repeater.clone(), 7)
            .of(&self_ty, entry, ix)
            .unwrap()
    });
    if let Some(self_ty) = self_ty {
        visitor = visitor.with_self(self_ty);
    }
    visitor.visit_item_struct_mut(&mut copy);
    // The arguments of the site, in one pass over the types of the copy.
    common::Written::new(common::answers(&item.generics.params, &given)).item(copy)
}

/// F70 (fixed): an entry's projection in the inputs of a function pointer or `Fn(..)` whose output
/// elides its lifetime broke the elision once a site gave it a lifetime (E0106).
///
/// Trigger: `struct S<T> { f: fn(&::std::vec::Vec<T>) -> &u8 }` with `::std` not allowed, at the
/// site `T := &'static str`: `fn(&<Self as Repeater<(&'static str,), KEY>>::Type) -> &u8` has two
/// lifetimes in its inputs. The same for `dyn Fn(&Vec<T>) -> &u8`.
///
/// Cause: the elided lifetime was named only for `Self` (F15).
///
/// Fix: the Visitor reads what the original signature relies on before it folds it, and names the
/// lifetime that the path it writes or `Self` adds (`'__tl_e<n>`); the site's pass
/// (`substitute_args()`, and `complete_args()` for a default) names the one its arguments add.
///
/// Limitation: a lifetime that the inputs name inside an entry (`fn(&'a Vec<T>) -> &u8`, entry
/// `&'a Vec<T>`) or that a path hides (F69) is not named; the copy is as written.
#[test]
fn f70_the_elided_lifetime_of_a_signature_is_named_where_the_visitor_writes_lifetimes_into_its_inputs(
) {
    // The path through the Repeater, at the site `T := &'static str`, and at `T := u8`.
    let entry: EntryId = parse_quote!(0);
    let key = s(&common::key(7, entry));
    let projection = |args: &str| format!("<Self as crate::Repeater<({},), {}>>::Type", args, key);
    let p = &projection("&'static str");
    let p_u8 = &projection("u8");
    let same =
        |generics: &str, field: &str, args: &[&str], self_ty: Option<&str>, expected: &str| {
            let item: ItemStruct =
                parse_str(&format!("pub struct S{} {{ pub f: {} }}", generics, field)).unwrap();
            let copy = f70_copy(&item, args, self_ty.map(|ty| parse_str(ty).unwrap()));
            let expected: ItemStruct = parse_str(&format!(
                "pub struct S{} {{ pub f: {} }}",
                generics, expected
            ))
            .unwrap();
            assert_eq!(s(&copy), s(&expected), "{}", field);
        };
    let t = "<T>";
    let lt = &["&'static str"];

    // The inputs show the elided lifetime and the path adds one: named, in the inputs and the output.
    same(
        t,
        "fn(&::std::vec::Vec<T>) -> &u8",
        lt,
        None,
        &format!("for<'__tl_s0> fn(&'__tl_s0 {}) -> &'__tl_s0 u8", p),
    );
    same(
        t,
        "::core::marker::PhantomData<dyn ::core::ops::Fn(&::std::vec::Vec<T>) -> &u8 + 'static>",
        lt,
        None,
        &format!(
            "::core::marker::PhantomData<dyn for<'__tl_s0> ::core::ops::Fn(&'__tl_s0 {}) -> \
             &'__tl_s0 u8 + 'static>",
            p
        ),
    );
    same(
        t,
        "fn(&::std::vec::Vec<T>, u8) -> &u8",
        lt,
        None,
        &format!("for<'__tl_s0> fn(&'__tl_s0 {}, u8) -> &'__tl_s0 u8", p),
    );
    // The inputs name their lifetime: the output takes it.
    same(
        t,
        "for<'x> fn(&'x ::std::vec::Vec<T>) -> &u8",
        lt,
        None,
        &format!("for<'x> fn(&'x {}) -> &'x u8", p),
    );
    // A signature inside another has its own, named apart; its own elided lifetime stays.
    same(
        t,
        "fn(&::std::vec::Vec<T>, fn(&::std::vec::Vec<T>) -> &u8) -> &u8",
        lt,
        None,
        &format!(
            "for<'__tl_s0> fn(&'__tl_s0 {p}, for<'__tl_s1> fn(&'__tl_s1 {p}) -> &'__tl_s1 u8) -> \
             &'__tl_s0 u8",
            p = p
        ),
    );
    same(
        t,
        "fn(&::std::vec::Vec<T>, fn(&u8) -> &u8) -> &u8",
        lt,
        None,
        &format!(
            "for<'__tl_s0> fn(&'__tl_s0 {}, fn(&u8) -> &u8) -> &'__tl_s0 u8",
            p
        ),
    );
    // `Self` that adds no lifetime, next to the path that does; and `Self` that adds one, which F15
    // named before (one binder, not two).
    same(
        t,
        "fn(&(Self, ::std::vec::Vec<T>)) -> &u8",
        lt,
        Some("::p::S<u8>"),
        &format!(
            "for<'__tl_s0> fn(&'__tl_s0 (::p::S<u8>, {})) -> &'__tl_s0 u8",
            p
        ),
    );
    same(
        "<'a, T>",
        "fn(&(Self, ::std::vec::Vec<T>)) -> &u8",
        &["'static", "&'static str"],
        Some("::p::S<'static, u8>"),
        &format!(
            "for<'__tl_e0> fn(&'__tl_e0 (::p::S<'static, u8>, {})) -> &'__tl_e0 u8",
            p
        ),
    );

    // Unchanged: no lifetime added, the lifetime already named, an output that does not elide.
    same(
        t,
        "fn(&::std::vec::Vec<T>) -> &u8",
        &["u8"],
        None,
        &format!("fn(&{}) -> &u8", p_u8),
    );
    same(
        t,
        "for<'x> fn(&'x ::std::vec::Vec<T>) -> &'x u8",
        lt,
        None,
        &format!("for<'x> fn(&'x {}) -> &'x u8", p),
    );
    same(
        t,
        "fn(&::std::vec::Vec<T>) -> u8",
        lt,
        None,
        &format!("fn(&{}) -> u8", p),
    );
    // (The site's one pass names the elided lifetime of the original even if the argument has an
    // elided lifetime of its own: it is not the Visitor that writes the argument.)
    same(
        t,
        "fn(&::std::vec::Vec<T>) -> &u8",
        &["&str"],
        None,
        &format!(
            "for<'__tl_s0> fn(&'__tl_s0 {}) -> &'__tl_s0 u8",
            projection("&str")
        ),
    );

    // `complete_args()` writes the path for the entry of a default the same way.
    let tr: ItemTrait = parse_quote!(
        pub trait Tr<T, U = fn(&::std::vec::Vec<T>) -> &u8> {
            fn f(&self, t: T) -> U;
        }
    );
    let mut leaker = common::std_leaker();
    let context = leaker.intern_trait(&tr);
    let referrer = leaker.result().finish().unwrap();
    let repeater = common::repeater();
    let complete = |arg: &str| -> Vec<String> {
        common::completed_args_for(
            &referrer,
            context,
            &tr.generics,
            &arguments(&[arg]),
            &common::projection(repeater.clone(), 7),
        )
        .unwrap()
        .iter()
        .map(s)
        .collect()
    };
    let expected = |texts: &[&str]| -> Vec<String> { arguments(texts).iter().map(s).collect() };
    assert_eq!(
        complete("&'static str"),
        expected(&[
            "&'static str",
            &format!("for<'__tl_s0> fn(&'__tl_s0 {}) -> &'__tl_s0 u8", p),
        ])
    );
    assert_eq!(
        complete("u8"),
        expected(&["u8", &format!("fn(&{}) -> &u8", p_u8)])
    );
}

// ── F71 ──────────────────────────────────────────────────────────────────────

/// What a Referrer makes of `item` (analyzed with the rule `config`): the errors of the analysis,
/// the entries (as written), and the copy, with each entry replaced by `P<ix>`.
fn f71_copy<A: Allow>(config: A, item: &Item) -> (Vec<Reason>, Vec<String>, String) {
    let mut leaker = Leaker::with_rule(config);
    let context = common::intern_item(&mut leaker, item);
    let result = leaker.result();
    let reasons = result.errors().iter().map(|e| e.reason).collect();
    let referrer = result.finish().unwrap();
    let entries = common::type_texts(&referrer);
    let mut copy = item.clone();
    common::visitor(&referrer, context, common::placeholder).visit_item_mut(&mut copy);
    (reasons, entries, s(&copy))
}

/// F71 (fixed): an optional type in a where predicate that a context carries is written as it is.
///
/// Trigger: `struct S<T> where ::std::vec::Vec<T>: Clone { cl: Cl<T> }`. `Vec<T>` and `Cl<T>` were
/// entries, each with `where Vec<T>: Clone` on its Repeater impl, which the copy cannot prove, as
/// it has the predicate only through the Repeater: E0277 "the trait bound `T: Clone` is not
/// satisfied".
///
/// Cause: the Leaker took a predicate that is not on a generic as proven by an impl, which
/// `Vec<T>: Clone` is not without `T: Clone`.
///
/// Fix: an optional type in any where predicate of a context is no entry and is written as it is,
/// as in the bound cycles (F17, F18, F68). A type that names no generic (`Vec<u8>: Into<U>`) has an
/// empty context and stays an entry.
///
/// Limitation: a type that must be an entry there (a relative `Pair<T, T>: Clone`) is still
/// replaced: E0277 in the copy, and the Leaker does not reject it.
#[test]
fn f71_an_optional_type_in_a_where_predicate_that_a_context_carries_is_written_as_it_is() {
    // The entries of the item, and the copy: the item with each entry replaced by `P<ix>`.
    let cases: Vec<(Item, Vec<&str>)> = vec![
        // The bounded type, with the entry that has the predicate in its context (the fixture 9005).
        (
            parse_quote!(
                pub struct S<T>
                where
                    ::std::vec::Vec<T>: ::core::clone::Clone,
                {
                    pub cl: Cl<T>,
                }
            ),
            vec!["Cl < T >"],
        ),
        // The type in the bound of a predicate on a type that is no generic, and in one on another.
        (
            parse_quote!(
                pub struct S<T>
                where
                    u8: ::core::convert::Into<::std::vec::Vec<T>>,
                {
                    pub cl: Cl<T>,
                }
            ),
            vec!["Cl < T >"],
        ),
        (
            parse_quote!(
                pub struct S<T, U>
                where
                    ::std::vec::Vec<T>: ::core::convert::Into<U>,
                {
                    pub cl: Cl<U>,
                }
            ),
            vec!["Cl < U >"],
        ),
        // Nested: neither type is an entry.
        (
            parse_quote!(
                pub struct S<T>
                where
                    ::std::vec::Vec<::std::boxed::Box<T>>: ::core::clone::Clone,
                {
                    pub cl: Cl<T>,
                }
            ),
            vec!["Cl < T >"],
        ),
        // Behind a reference with a binder.
        (
            parse_quote!(
                pub struct S<T>
                where
                    for<'a> &'a ::std::vec::Vec<T>: ::core::iter::IntoIterator,
                {
                    pub cl: Cl<T>,
                }
            ),
            vec!["Cl < T >"],
        ),
        // The type is in a field too: the same type in the same context is no entry there either.
        (
            parse_quote!(
                pub struct S<T>
                where
                    ::std::vec::Vec<T>: ::core::clone::Clone,
                {
                    pub held: ::std::vec::Vec<T>,
                    pub cl: Cl<T>,
                }
            ),
            vec!["Cl < T >"],
        ),
        // No other entry: the type in the predicate was the only one, and its own context carries it.
        (
            parse_quote!(
                pub struct S<T>
                where
                    ::std::vec::Vec<T>: ::core::clone::Clone,
                {
                    pub marker: ::core::marker::PhantomData<T>,
                }
            ),
            vec![],
        ),
        // The predicate of a method (on a generic of the method), of a trait and of an impl.
        (
            parse_quote!(
                pub trait Tr {
                    fn m<U>(&self, a: Cl<U>) -> Cl<U>
                    where
                        ::std::vec::Vec<U>: ::core::clone::Clone;
                }
            ),
            vec!["Cl < U >"],
        ),
        (
            parse_quote!(
                pub trait Tr<T>
                where
                    ::std::vec::Vec<T>: ::core::clone::Clone,
                {
                    fn m(&self, a: Cl<T>) -> Cl<T>;
                }
            ),
            vec!["Cl < T >"],
        ),
        (
            parse_quote!(
                impl<T> Holder<T>
                where
                    ::std::vec::Vec<T>: ::core::clone::Clone,
                {
                    pub fn m(&self, a: Cl<T>) -> Cl<T> {
                        a
                    }
                }
            ),
            vec!["Cl < T >", "Holder < T >"],
        ),
        // What an impl would prove (`T: Clone` is a bound of the generic), a lifetime, a const: the
        // Leaker cannot tell, the type is written as it is.
        (
            parse_quote!(
                pub struct S<T: ::core::clone::Clone>
                where
                    ::std::vec::Vec<T>: ::core::clone::Clone,
                {
                    pub cl: Cl<T>,
                }
            ),
            vec!["Cl < T >"],
        ),
        (
            parse_quote!(
                pub struct S<'a, T: 'a>
                where
                    ::std::vec::Vec<&'a T>: ::core::clone::Clone,
                {
                    pub cl: Cl<&'a T>,
                }
            ),
            vec!["Cl < & 'a T >"],
        ),
        (
            parse_quote!(
                pub struct S<const N: usize>
                where
                    ::std::vec::Vec<[u8; N]>: ::core::fmt::Debug,
                {
                    pub cl: Cl<[u8; N]>,
                }
            ),
            vec!["Cl < [u8 ; N] >"],
        ),
    ];
    for (item, entries) in &cases {
        // The types of `::std` are no entries, whether the Leaker allows `::std` or not.
        for trusting in [false, true] {
            let (errors, found, copy) = if trusting {
                f71_copy(Rules::new().primitive().prelude().external_crates(), item)
            } else {
                f71_copy(Rules::new().primitive().prelude(), item)
            };
            assert_eq!(errors, [], "{}", s(item));
            assert_eq!(&found, entries, "{}", s(item));
            // The copy is the item, with the entries replaced and the types of `::std` as written.
            let mut expected = s(item);
            for (ix, entry) in entries.iter().enumerate() {
                expected = expected.replace(entry, &format!("P < {}usize >", ix));
            }
            assert_eq!(copy, expected, "{}", s(item));
        }
    }

    // Controls, what was an entry and is. A type that names no generic: its context is empty (it
    // has no predicate), and the entry that carries the predicate finds it normalized.
    let concrete: Item = parse_quote!(
        pub struct S<U>
        where
            ::std::vec::Vec<u8>: ::core::convert::Into<U>,
        {
            pub cl: Cl<U>,
        }
    );
    let (errors, entries, copy) = f71_copy(Rules::new().primitive().prelude(), &concrete);
    assert_eq!(errors, []);
    assert_eq!(entries, ["Cl < U >", ":: std :: vec :: Vec < u8 >"]);
    assert_eq!(
        copy,
        s(&concrete)
            .replace("Cl < U >", "P < 0usize >")
            .replace(":: std :: vec :: Vec < u8 >", "P < 1usize >")
    );
    // The type of another generic: it has no predicate in its context.
    let other: Item = parse_quote!(
        pub struct S<T, U>
        where
            ::std::vec::Vec<T>: ::core::clone::Clone,
        {
            pub v: ::std::vec::Vec<U>,
            pub cl: Cl<T>,
        }
    );
    let (errors, entries, copy) = f71_copy(Rules::new().primitive().prelude(), &other);
    assert_eq!(errors, []);
    assert_eq!(entries, ["Cl < T >", ":: std :: vec :: Vec < U >"]);
    assert_eq!(
        copy,
        s(&other)
            .replace("Cl < T >", "P < 0usize >")
            .replace(":: std :: vec :: Vec < U >", "P < 1usize >")
    );
    // No predicate: the type is an entry, unless the rule of the Leaker allows `::std`.
    let plain: Item = parse_quote!(
        pub struct S<T> {
            pub v: ::std::vec::Vec<T>,
        }
    );
    let (_, entries, _) = f71_copy(Rules::new().primitive().prelude(), &plain);
    assert_eq!(entries, [":: std :: vec :: Vec < T >"]);
    let (_, entries, _) = f71_copy(Rules::new().primitive().prelude().external_crates(), &plain);
    assert_eq!(entries, Vec::<String>::new());
    // The bound of a generic: no entry before the fix either (F2, F17).
    let bound: Item = parse_quote!(
        pub struct S<T: ::core::convert::Into<::std::vec::Vec<T>>> {
            pub cl: Cl<T>,
        }
    );
    let (_, entries, _) = f71_copy(Rules::new().primitive().prelude(), &bound);
    assert_eq!(entries, ["Cl < T >"]);

    // The limitation: a type that must be an entry is in the predicate, and the copy replaces it.
    let required: Item = parse_quote!(
        pub struct S<T>
        where
            Pair<T, T>: ::core::clone::Clone,
        {
            pub cl: Cl<T>,
        }
    );
    let (errors, entries, _) = f71_copy(Rules::new().primitive().prelude(), &required);
    assert_eq!(errors, []);
    assert_eq!(entries, ["Cl < T >", "Pair < T , T >"]);
    // The same for a primitive type that the rule does not allow, which must be an entry: the map
    // that holds it is one (and the map without it is not).
    let primitive: Item = parse_quote!(
        pub struct S<T>
        where
            ::std::collections::BTreeMap<T, u8>: ::core::clone::Clone,
        {
            pub cl: Cl<T>,
        }
    );
    let (errors, entries, _) = f71_copy(Rules::new().prelude(), &primitive);
    assert_eq!(errors, []);
    assert_eq!(
        entries,
        [":: std :: collections :: BTreeMap < T , u8 >", "Cl < T >"]
    );
    let (_, entries, _) = f71_copy(Rules::new().primitive().prelude(), &primitive);
    assert_eq!(entries, ["Cl < T >"]);
}

// ── F72 ──────────────────────────────────────────────────────────────────────

/// The type of the first typed argument of the method `name` of `copy`, as tokens.
fn f72_first_input(copy: &ItemTrait, name: &str) -> String {
    let sig = copy
        .items
        .iter()
        .find_map(|item| match item {
            TraitItem::Fn(method) if method.sig.ident == name => Some(&method.sig),
            _ => None,
        })
        .unwrap_or_else(|| panic!("no method `{}`", name));
    sig.inputs
        .iter()
        .find_map(|input| match input {
            FnArg::Typed(arg) => Some(s(&*arg.ty)),
            FnArg::Receiver(_) => None,
        })
        .unwrap_or_else(|| panic!("`{}` has no typed input", name))
}

/// F72 (fixed): a type copied as written for a method lifetime is so only where the copy leaves the
/// lifetime unconstrained.
///
/// Trigger: `fn f<'x>(&self, a: Wrap<'x>) -> &'x u8;` and
/// `fn g<'x>(&self, a: Wrap<'x>, b: &'x u8) -> &'x u8;` in one trait. `f` reports `Wrap<'x>`
/// (`UnconstrainedLifetime`) and keeps it as written (F48); in `g`, where `b` keeps `'x`, it is an
/// entry, but the copy of `g` kept it as written too: a relative path of the Leaker's crate in the
/// Referrer's (E0412).
///
/// Cause: the Visitor applied the part to every signature whose output names the lifetime and no
/// bound does, though another input (`&'x u8`, `(u8, &'x u8)`, `&'x self`) may keep it.
///
/// Fix: the Visitor scans the inputs with the entries replaced, with the Leaker's own scan, and
/// applies the part only where the copy leaves the lifetime unconstrained.
///
/// Limitation: the choice is per signature, not per place: an entry inside a culprit
/// (`R2<Rel<'x>, &'x u8>`) is written as it is where another signature gives it a part.
#[test]
fn f72_a_type_copied_as_written_for_a_method_lifetime_is_left_only_where_the_copy_leaves_it_unconstrained(
) {
    let unconstrained = Reason::UnconstrainedLifetime;
    let wrap = "Wrap < 'x >";
    let as_written = wrap.to_string();
    let entry = "P < 0usize >".to_string();

    // The trigger: `f` has the error and keeps the type, `g` has the entry.
    let item: ItemTrait = parse_quote!(
        trait Tr {
            fn f<'x>(&self, a: Wrap<'x>) -> &'x u8;
            fn g<'x>(&self, a: Wrap<'x>, b: &'x u8) -> &'x u8;
        }
    );
    let (copy, errors, entries) = f63_leak(common::std_config(), &item);
    assert_eq!(errors, [(unconstrained, wrap.to_string())]);
    assert_eq!(entries, [wrap]);
    assert_eq!(
        copy,
        "trait Tr { fn f < 'x > (& self , a : Wrap < 'x >) -> & 'x u8 ; \
         fn g < 'x > (& self , a : P < 0usize > , b : & 'x u8) -> & 'x u8 ; }"
    );
    // `g` alone, as it was.
    let item_g: ItemTrait = parse_quote!(
        trait Tr {
            fn g<'x>(&self, a: Wrap<'x>, b: &'x u8) -> &'x u8;
        }
    );
    let (copy_g, errors, _) = f63_leak(common::std_config(), &item_g);
    assert_eq!(errors, []);
    assert_eq!(
        copy_g,
        "trait Tr { fn g < 'x > (& self , a : P < 0usize > , b : & 'x u8) -> & 'x u8 ; }"
    );

    // The same for a macro that writes the methods one by one (`visit_signature_mut()`).
    let mut leaker = common::std_leaker();
    let context = leaker.intern_trait(&item);
    let referrer = leaker.result().recovered();
    let replace = |_: Type, ix: EntryId, _: &Context| -> Type { parse_quote!(P<#ix>) };
    let signatures: [(Signature, &str); 2] = [
        (
            parse_quote!(fn f<'x>(&self, a: Wrap<'x>) -> &'x u8),
            "fn f < 'x > (& self , a : Wrap < 'x >) -> & 'x u8",
        ),
        (
            parse_quote!(fn g<'x>(&self, a: Wrap<'x>, b: &'x u8) -> &'x u8),
            "fn g < 'x > (& self , a : P < 0usize > , b : & 'x u8) -> & 'x u8",
        ),
    ];
    for (sig, expected) in signatures {
        let mut copy = sig.clone();
        common::visitor(&referrer, context, replace).visit_signature_mut(&mut copy);
        assert_eq!(s(&copy), expected, "{}", s(&sig));
    }

    // The input that keeps the lifetime, in other positions: the type is the entry in `g`.
    let keepers = [
        "&'x u8",
        "&'x mut u8",
        "Option<&'x u8>",
        "(u8, &'x u8)",
        "[&'x u8; 2]",
        "Box<&'x u8>",
        "fn(&'x u8)",
        "Box<dyn Fn(&'x u8)>",
        "*const &'x u8",
    ];
    for keeper in keepers {
        let keeper: Type = syn::parse_str(keeper).unwrap();
        let item: ItemTrait = parse_quote!(
            trait Tr {
                fn f<'x>(&self, a: Wrap<'x>) -> &'x u8;
                fn g<'x>(&self, a: Wrap<'x>, b: #keeper) -> &'x u8;
            }
        );
        let (copy, errors, _) = f63_leak(common::std_config(), &item);
        let copy: ItemTrait = syn::parse_str(&copy).unwrap();
        let what = s(&keeper);
        assert_eq!(errors, [(unconstrained, wrap.to_string())], "{}", what);
        assert_eq!(f72_first_input(&copy, "f"), as_written, "{}", what);
        assert_eq!(f72_first_input(&copy, "g"), entry, "{}", what);
    }
    // The receiver keeps it: the type is the entry in `h`.
    let item: ItemTrait = parse_quote!(
        trait Tr {
            fn f<'x>(&self, a: Wrap<'x>) -> &'x u8;
            fn h<'x>(&'x self, a: Wrap<'x>) -> &'x u8;
        }
    );
    let (copy, errors, _) = f63_leak(common::std_config(), &item);
    let copy: ItemTrait = syn::parse_str(&copy).unwrap();
    assert_eq!(errors, [(unconstrained, wrap.to_string())]);
    assert_eq!(f72_first_input(&copy, "f"), as_written);
    assert_eq!(f72_first_input(&copy, "h"), entry);

    // Both signatures are errors, so both keep the type as written: `f2` and `f3` (two ways, each
    // through a `Wrap<'x>`) and `f4` (a lifetime that only a projection of the trait keeps, which
    // constrains nothing); `g` has the entry. The projection is a `Self::A<'x>` of a GAT.
    let item: ItemTrait = parse_quote!(
        trait Tr {
            type A<'a>;
            fn f<'x>(&self, a: Wrap<'x>) -> &'x u8;
            fn f2<'x>(&self, a: Wrap<'x>) -> &'x u8;
            fn f3<'x>(&self, a: Wrap<'x>, b: *mut Wrap<'x>) -> &'x u8;
            fn f4<'x>(&self, a: Wrap<'x>, b: Self::A<'x>) -> &'x u8;
            fn g<'x>(&self, a: Wrap<'x>, b: &'x u8) -> &'x u8;
        }
    );
    let (copy, errors, entries) = f63_leak(common::std_config(), &item);
    let copy: ItemTrait = syn::parse_str(&copy).unwrap();
    assert_eq!(errors, vec![(unconstrained, wrap.to_string()); 5]);
    assert_eq!(entries, [wrap]);
    for name in ["f", "f2", "f3", "f4"] {
        assert_eq!(f72_first_input(&copy, name), as_written, "{}", name);
    }
    assert_eq!(f72_first_input(&copy, "g"), entry);
    let sig = copy.items.iter().find_map(|item| match item {
        TraitItem::Fn(method) if method.sig.ident == "f3" => Some(method.sig.clone()),
        _ => None,
    });
    assert_eq!(
        s(&sig.unwrap()),
        "fn f3 < 'x > (& self , a : Wrap < 'x > , b : * mut Wrap < 'x >) -> & 'x u8"
    );

    // A rule that allows no `Option`: it is leaked too, so `g` has two errors, and both types stay
    // as written there; `k` keeps the lifetime in a reference, and has the entry.
    let item: ItemTrait = parse_quote!(
        trait Tr {
            fn f<'x>(&self, a: Wrap<'x>) -> &'x u8;
            fn g<'x>(&self, a: Wrap<'x>, b: Option<&'x u8>) -> &'x u8;
            fn k<'x>(&self, a: Wrap<'x>, b: &'x u8) -> &'x u8;
        }
    );
    let (copy, errors, entries) = f63_leak(Rules::new().primitive(), &item);
    let causes: Vec<String> = errors.iter().map(|(_, cause)| cause.clone()).collect();
    assert_eq!(causes, [wrap, wrap, "Option < & 'x u8 >"].map(String::from));
    assert_eq!(entries, [wrap]);
    let copy: ItemTrait = syn::parse_str(&copy).unwrap();
    assert_eq!(f72_first_input(&copy, "f"), as_written);
    assert_eq!(f72_first_input(&copy, "g"), as_written);
    assert_eq!(f72_first_input(&copy, "k"), entry);

    // The mark names two lifetimes, and one of them is kept: `f` and `h` have the error for `'x`
    // (`Wrap2<'x, 'y>` is the only input that names it), so the type stays as written; in `g` every
    // input keeps one of them, so it is the entry.
    let item: ItemTrait = parse_quote!(
        trait Tr {
            fn f<'x, 'y>(&self, a: Wrap2<'x, 'y>, b: &'y u8) -> (&'x u8, &'y u8);
            fn g<'x, 'y>(&self, a: Wrap2<'x, 'y>, b: &'x u8, c: &'y u8) -> (&'x u8, &'y u8);
            fn h<'x, 'y>(&self, a: Wrap2<'x, 'y>, b: &'y u8) -> (&'x u8, &'y u8);
        }
    );
    let (copy, errors, entries) = f63_leak(common::std_config(), &item);
    let copy: ItemTrait = syn::parse_str(&copy).unwrap();
    assert_eq!(
        errors,
        vec![(unconstrained, "Wrap2 < 'x , 'y >".to_string()); 2]
    );
    assert_eq!(entries, ["Wrap2 < 'x , 'y >"]);
    assert_eq!(f72_first_input(&copy, "f"), "Wrap2 < 'x , 'y >");
    assert_eq!(f72_first_input(&copy, "g"), entry);
    assert_eq!(f72_first_input(&copy, "h"), "Wrap2 < 'x , 'y >");

    // Another lifetime name is another type: the part of `f` is not the one of `g`.
    let item: ItemTrait = parse_quote!(
        trait Tr {
            fn f<'x>(&self, a: Wrap<'x>) -> &'x u8;
            fn g<'y>(&self, a: Wrap<'y>, b: &'y u8) -> &'y u8;
        }
    );
    let (copy, _, entries) = f63_leak(common::std_config(), &item);
    assert_eq!(entries, ["Wrap < 'y >"]);
    assert_eq!(
        copy,
        "trait Tr { fn f < 'x > (& self , a : Wrap < 'x >) -> & 'x u8 ; \
         fn g < 'y > (& self , a : P < 0usize > , b : & 'y u8) -> & 'y u8 ; }"
    );
}

// ── F73 ──────────────────────────────────────────────────────────────────────

/// F73 (fixed): `Referrer::complete_args()` braced a braced const argument twice, where
/// `substitute_args()` braces it once (F66).
///
/// Trigger: `trait Tr<const N: usize, U = [u8; { N }], V = Rel<{ N }>>` and
/// `impl<const M: usize> Tr<{ M }> for G<M>`: the default became `[u8; { { M } }]`, which rustc
/// rejects for a generic `M` ("generic parameters may not be used in const operations"). Fixture
/// 9246.
///
/// Cause: the completion braced the given block `{ M }` as it was, where `substitute_args()` takes
/// the block off first.
///
/// Fix: both write the value with one helper: a block that holds the value alone is taken off, then
/// the value is braced once. Each completed default equals what `substitute_args()` writes.
#[test]
fn f73_complete_args_braces_a_braced_const_argument_once() {
    let params: Punctuated<GenericParam, Token![,]> = parse_quote!(
        const N: usize,
        U = [u8; { N }],
        V = Rel<{ N }>,
        W = [u8; N],
        X = Rel<N>,
        Y = [u8; { N + 1 }]
    );
    let defaults: Vec<Type> = params
        .iter()
        .filter_map(|param| match param {
            GenericParam::Type(param) => param.default.clone(),
            _ => None,
        })
        .collect();
    let only_n: Punctuated<GenericParam, Token![,]> = parse_quote!(const N: usize);
    for arg in ["{ M }", "M", "3", "{ 3 }", "{ 1 + 1 }", "{ { M } }"] {
        let site = arguments(&[arg]);
        let completed = complete_args_of(&params, &site).unwrap();
        assert_eq!(completed.len(), params.len(), "{}", arg);
        for (default, completed) in defaults.iter().zip(&completed[1..]) {
            let expected =
                common::substitute_type(default, common::answers(&only_n, &site), None).unwrap();
            assert_eq!(s(completed), s(&expected), "{} with {}", s(default), arg);
        }
    }
    // The trigger, with the exact tokens.
    let completed = complete_args_of(&params, &arguments(&["{ M }"])).unwrap();
    let tokens: Vec<String> = completed.iter().map(s).collect();
    assert_eq!(tokens[0], "M");
    assert_eq!(tokens[1], "[u8 ; { M }]");
    assert_eq!(tokens[2], "Rel < { M } >");
    assert_eq!(tokens[3], "[u8 ; { M }]");
    assert_eq!(tokens[4], "Rel < M >");
    let completed = complete_args_of(&params, &arguments(&["{ 1 + 1 }"])).unwrap();
    assert_eq!(s(&completed[1]), "[u8 ; { 1 + 1 }]");
    assert_eq!(s(&completed[2]), "Rel < { 1 + 1 } >");
}

// ── F75 ──────────────────────────────────────────────────────────────────────

/// F75 (fixed): the generics of a `RepeaterImpl` that declared the parameter for the implementor
/// had no angle brackets, so `ItemImpl::from(r)` was not equal to the impl written by hand.
///
/// Trigger: `struct S(Rel)` with `ImplSelf::Param(__S)`: the impl declares `__S` alone. Its tokens
/// were right, but `syn` compares `<` and `>` as `Option`s, so it was not equal to
/// `impl<__S: ?Sized> Repeater<(), KEY> for __S { .. }` parsed by `syn`.
///
/// Cause: the parameter was inserted into the empty generics of the context, whose `<` and `>` are
/// `None`.
///
/// Fix: the impl sets them when it inserts the parameter. The test compares each impl with what
/// `syn` reads from its tokens.
#[test]
fn f75_the_generics_of_a_repeater_impl_have_their_angle_brackets() {
    let item: ItemStruct = parse_quote!(
        struct S(Rel);
    );
    let mut leaker = common::std_leaker();
    leaker.intern_struct(&item);
    let result = leaker.result();
    let repeater = common::repeater();
    let param: Ident = parse_quote!(__S);
    let r = result
        .repeater_impls(&Projection::new(
            repeater.clone(),
            5,
            ImplSelf::Param(param.clone()),
        ))
        .unwrap()
        .remove(0);
    let key = common::key(5, r.entry);
    let expected: ItemImpl = parse_quote!(
        impl<__S: ?::core::marker::Sized> crate::Repeater<(), #key> for __S {
            type Type = Rel;
        }
    );
    assert_eq!(ItemImpl::from(r), expected);

    // The impl is what `syn` reads from its tokens, whether the context has generics or not, and
    // for each way to name the implementor.
    let items: Vec<Item> = vec![
        parse_quote!(
            struct S(Rel);
        ),
        parse_quote!(
            struct S<T>(Rel<T>);
        ),
        parse_quote!(
            struct S<'a, T: 'a, const N: usize>(&'a Rel<T, N>);
        ),
        parse_quote!(
            trait Tr {
                fn f(&self) -> Rel;
            }
        ),
        parse_quote!(
            trait Tr<T> {
                fn f(&self) -> Rel<T, Self>;
            }
        ),
    ];
    let marker: Type = parse_quote!(crate::Marker);
    let mut compared = 0;
    for item in items {
        let mut leaker = common::std_leaker();
        common::intern_item(&mut leaker, &item);
        let result = leaker.result();
        for target in [
            ImplSelf::Param(param.clone()),
            ImplSelf::Marker(marker.clone()),
            ImplSelf::Bucket {
                param: param.clone(),
            },
        ] {
            let Ok(impls) =
                result.repeater_impls(&Projection::new(repeater.clone(), 5, target.clone()))
            else {
                continue;
            };
            for r in impls {
                let what = format!("{}: {} {:?}", s(&item), r.entry, target);
                let read: ItemImpl = parse2(quote!(#r)).unwrap();
                assert_eq!(ItemImpl::from(r), read, "{}", what);
                compared += 1;
            }
        }
    }
    assert!(compared >= 10, "{}", compared);
}

// ── F74 ──────────────────────────────────────────────────────────────────────

/// F74 (fixed): `Referrer::complete_args()` left the associated type of an earlier parameter in a
/// default as written (`U = T::Item`), where `substitute_args()` writes `<value>::Item`.
///
/// Trigger: `trait Tr<T: Iterator, U = T::Item>` and `impl<I: Iterator> Tr<I> for G<I>`: the
/// completed arguments were `[I, T::Item]`, and `T` is not in the impl (E0433). No fixture: stable
/// rustc rejects `T::Item` in a default (E0220).
///
/// Cause: the completion replaced a type that is the name of an earlier parameter, but walked into
/// other paths as they were, so `T::Item` kept its `T`.
///
/// Fix: `T::A..` with `T` an earlier parameter is written `<value>::A..`, as `substitute_args()`
/// writes it; an argument that is no identifier gets the trait of the one bound (F90).
///
/// Limitation: a path in an expression of a default (`[u8; T::LEN]`) is not written so; stable
/// rustc rejects such a default ("generic parameters may not be used in const operations"), so it
/// needs `generic_const_exprs` (nightly).
#[test]
fn f74_complete_args_writes_the_associated_type_of_an_earlier_parameter() {
    // The trigger.
    let params: Punctuated<GenericParam, Token![,]> =
        parse_quote!(T, U = T::Item, V = Vec<T::Item>);
    let completed = complete_args_of(&params, &arguments(&["I"])).unwrap();
    let completed: Vec<String> = completed.iter().map(s).collect();
    assert_eq!(completed, ["I", "< I > :: Item", "Vec < < I > :: Item >"]);
    // An argument that is not an identifier needs the trait (F90): the one bound of `T`.
    let params: Punctuated<GenericParam, Token![,]> = parse_quote!(T: Iterator, U = T::Item);
    let completed = complete_args_of(&params, &arguments(&["IntoIter<u8>"])).unwrap();
    assert_eq!(s(&completed[1]), "< IntoIter < u8 > as Iterator > :: Item");

    // Each default is what `substitute_args()` gives for it, whatever the argument: the same type,
    // or the same error (`T` has no bound here).
    let only_t: Punctuated<GenericParam, Token![,]> = parse_quote!(T);
    for written in [
        "T::Item",
        "Vec<T::Item>",
        "Option<(T, T::Item)>",
        "&'static T::Item",
        "[T::Item; 3]",
        "T::A<T>",
        "T::A<T::B>",
        "T::A::B",
        "<T as Iterator>::Item",
        "Box<dyn Fn(T::Item) -> T::Other>",
        "fn(T::Item) -> u8",
        // Not a parameter: left as written.
        "X::Item",
        "::T::Item",
    ] {
        for arg in [
            "I",
            "IntoIter<u8>",
            "(u8, i8)",
            "&'static str",
            "dyn Tr + Send",
        ] {
            let default: Type = parse_str(written).unwrap();
            let params: Punctuated<GenericParam, Token![,]> = parse_quote!(T, U = #default);
            let site = arguments(&[arg]);
            let completed = complete_args_of(&params, &site).map(|completed| s(&completed[1]));
            let expected = common::substitute_type(&default, common::answers(&only_t, &site), None)
                .map(|expected| s(&expected));
            assert_eq!(
                completed.map_err(|error| error.to_string()),
                expected.map_err(|error| error.to_string()),
                "{} with {}",
                written,
                arg
            );
        }
    }
    // A name that is no parameter before the default is not an argument.
    let params: Punctuated<GenericParam, Token![,]> = parse_quote!(T, U = X::Item);
    let completed = complete_args_of(&params, &arguments(&["I"])).unwrap();
    assert_eq!(s(&completed[1]), "X :: Item");
}

// ── F76 ──────────────────────────────────────────────────────────────────────

/// F76 (fixed): the parameter that an impl declares for `Self` (`ImplSelf::Param`, or the `param`
/// of `ImplSelf::Bucket`) clashed with a generic of the entry that has the same name.
///
/// Trigger: `struct S<__S>(Rel<__S>)` with `ImplSelf::Param(__S)`, or a trait `Tr<__S>` whose
/// method returns `Rel<Self, __S>`: the impl declared `__S` twice (`impl<__S, __S: ?Sized> ..`,
/// E0403 in the Leaker's crate). Fixture 9248.
///
/// Cause: the impl declared the given name whatever the entry, its context and its bound used.
///
/// Fix: the impl takes the given name unless one of them uses it, else the first of `{name}0`,
/// `{name}1`, .. that none uses. A lifetime of that name is no clash.
#[test]
fn f76_the_parameter_for_self_does_not_clash_with_a_generic_of_the_entry() {
    let repeater = common::repeater();
    let param: Ident = parse_quote!(__S);
    let impls_of = |item: Item, target: ImplSelf| -> Vec<ItemImpl> {
        let mut leaker = common::std_leaker();
        common::intern_item(&mut leaker, &item);
        let result = leaker.result();
        assert!(result.errors().is_empty(), "{}", s(&item));
        let impls = result
            .repeater_impls(&Projection::new(repeater.clone(), 5, target.clone()))
            .unwrap();
        impls.into_iter().map(ItemImpl::from).collect()
    };
    let key0 = common::key(5, parse_quote!(0));

    // The trigger: the generic of the entry is `__S`.
    let item: Item = parse_quote!(
        struct S<__S>(Rel<__S>);
    );
    let expected: ItemImpl = parse_quote!(
        impl<__S, __S0: ?::core::marker::Sized> crate::Repeater<(__S,), #key0> for __S0 {
            type Type = Rel<__S>;
        }
    );
    assert_eq!(impls_of(item, ImplSelf::Param(param.clone())), [expected]);
    // The bucket's parameter is for the entries that need `Self`: the trait's `Self` is `__S0`, and
    // the bound names the trait's own `__S`.
    let item_self: Item = parse_quote!(
        trait Tr<__S> {
            fn f(&self) -> Rel<Self, __S>;
        }
    );
    let expected: ItemImpl = parse_quote!(
        impl<__S, __S0: ?::core::marker::Sized + Tr<__S>> crate::Repeater<(__S,), #key0> for __S0 {
            type Type = Rel<__S0, __S>;
        }
    );
    let expected = std::slice::from_ref(&expected);
    assert_eq!(
        impls_of(item_self.clone(), ImplSelf::Param(param.clone())),
        expected
    );
    assert_eq!(
        impls_of(
            item_self,
            ImplSelf::Bucket {
                param: param.clone()
            }
        ),
        expected
    );

    // The next free name: `__S0` is used too. A const parameter keeps its place after the types.
    let item: Item = parse_quote!(
        struct S<__S, __S0, const N: usize>(Rel<__S, __S0, N>);
    );
    let expected: ItemImpl = parse_quote!(
        impl<__S, __S0, __S1: ?::core::marker::Sized, const N: usize>
            crate::Repeater<(__S, __S0, [(); N]), #key0> for __S1
        {
            type Type = Rel<__S, __S0, N>;
        }
    );
    assert_eq!(impls_of(item, ImplSelf::Param(param.clone())), [expected]);

    // A lifetime named like the parameter is no clash, a type of that name that is not declared is.
    let item: Item = parse_quote!(
        struct S<'__S>(Rel<'__S>);
    );
    let expected: ItemImpl = parse_quote!(
        impl<'__S, __S: ?::core::marker::Sized> crate::Repeater<(&'__S (),), #key0> for __S {
            type Type = Rel<'__S>;
        }
    );
    assert_eq!(impls_of(item, ImplSelf::Param(param.clone())), [expected]);
    let item: Item = parse_quote!(
        struct S<T>(Rel<T, __S>);
    );
    let expected: ItemImpl = parse_quote!(
        impl<T, __S0: ?::core::marker::Sized> crate::Repeater<(T,), #key0> for __S0 {
            type Type = Rel<T, __S>;
        }
    );
    assert_eq!(impls_of(item, ImplSelf::Param(param.clone())), [expected]);

    // No clash, no change: the name is the one given. A struct that names `Self` declares none, a
    // marker neither.
    let item: Item = parse_quote!(
        struct S<T>(Rel<T>);
    );
    let expected: ItemImpl = parse_quote!(
        impl<T, __S: ?::core::marker::Sized> crate::Repeater<(T,), #key0> for __S {
            type Type = Rel<T>;
        }
    );
    assert_eq!(impls_of(item, ImplSelf::Param(param.clone())), [expected]);
    let item: Item = parse_quote!(
        struct S<__S>(Rel<Self, __S>);
    );
    let expected: ItemImpl = parse_quote!(
        impl<__S> crate::Repeater<(__S,), #key0> for S<__S> {
            type Type = Rel<Self, __S>;
        }
    );
    assert_eq!(impls_of(item, ImplSelf::Param(param.clone())), [expected]);
    let marker: Type = parse_quote!(crate::Marker);
    let item: Item = parse_quote!(
        struct S<__S>(Rel<__S>);
    );
    let expected: ItemImpl = parse_quote!(
        impl<__S> crate::Repeater<(__S,), #key0> for crate::Marker {
            type Type = Rel<__S>;
        }
    );
    assert_eq!(impls_of(item, ImplSelf::Marker(marker.clone())), [expected]);
}

/// F76: a name that only the tokens of a macro use (`::p::m!(__S)`) is taken too, and the parameter
/// of an entry for any type (`SelfRequirement::Any`) is renamed as well, in the bound that names
/// `Self` too.
#[test]
fn f76_the_name_in_the_tokens_of_a_macro_is_taken_and_so_is_the_one_of_an_any_entry() {
    let repeater = common::repeater();
    let param: Ident = parse_quote!(__S);
    let impls_of = |result: LeakerResult| -> Vec<ItemImpl> {
        assert!(result.errors().is_empty());
        let impls = result
            .repeater_impls(&Projection::new(
                repeater.clone(),
                5,
                ImplSelf::Param(param.clone()),
            ))
            .unwrap();
        impls.into_iter().map(ItemImpl::from).collect()
    };
    let top_level = |item: Item| -> LeakerResult {
        let mut leaker = common::std_leaker();
        common::intern_item(&mut leaker, &item);
        leaker.result()
    };
    let (key0, key1) = (
        common::key(5, parse_quote!(0)),
        common::key(5, parse_quote!(1)),
    );

    // The macro is in the type of the entry.
    let item: Item = parse_quote!(
        struct S<T>(Rel<T, ::p::m!(__S)>);
    );
    let expected: ItemImpl = parse_quote!(
        impl<T, __S0: ?::core::marker::Sized> crate::Repeater<(T,), #key0> for __S0 {
            type Type = Rel<T, ::p::m!(__S)>;
        }
    );
    assert_eq!(impls_of(top_level(item)), [expected]);

    // The macro is in a bound of the generic that the entry uses: the entry `::p::m!(__S)` has no
    // generics and its own tokens, and `Rel<T>` carries the bound. The macro in a predicate that
    // the context of `Rel<T>` leaves out clashes with nothing: its entry is the only one renamed.
    let item: Item = parse_quote!(
        struct S<T: ::p::Tr<::p::m!(__S)>>(Rel<T>);
    );
    let expected: [ItemImpl; 2] = [
        parse_quote!(
            impl<__S0: ?::core::marker::Sized> crate::Repeater<(), #key0> for __S0 {
                type Type = ::p::m!(__S);
            }
        ),
        parse_quote!(
            impl<T: ::p::Tr<::p::m!(__S)>, __S0: ?::core::marker::Sized>
                crate::Repeater<(T,), #key1> for __S0
            {
                type Type = Rel<T>;
            }
        ),
    ];
    assert_eq!(impls_of(top_level(item)), expected);
    let item: Item = parse_quote!(
        struct S<T>(Rel<T>)
        where
            ::p::m!(__S): Clone;
    );
    let expected: [ItemImpl; 2] = [
        parse_quote!(
            impl<__S0: ?::core::marker::Sized> crate::Repeater<(), #key0> for __S0 {
                type Type = ::p::m!(__S);
            }
        ),
        parse_quote!(
            impl<T, __S: ?::core::marker::Sized> crate::Repeater<(T,), #key1> for __S {
                type Type = Rel<T>;
            }
        ),
    ];
    assert_eq!(impls_of(top_level(item)), expected);

    // The implementor of a trait is a parameter too, and the macro is in the type.
    let item: Item = parse_quote!(
        trait Tr<T> {
            fn f(&self) -> Rel<Self, T, ::p::m!(__S)>;
        }
    );
    let expected: ItemImpl = parse_quote!(
        impl<T, __S0: ?::core::marker::Sized + Tr<T>> crate::Repeater<(T,), #key0> for __S0 {
            type Type = Rel<__S0, T, ::p::m!(__S)>;
        }
    );
    assert_eq!(impls_of(top_level(item)), [expected]);

    // An entry for any type: the generic of the nested struct is `__S`, and so is the parameter
    // that the impl would declare for the type.
    let item: Item = parse_quote!(
        pub struct S<__S: ::core::convert::AsRef<Self>>(pub u8, pub Rel<__S>);
    );
    let expected: ItemImpl = parse_quote!(
        impl<__S: ::core::convert::AsRef<__S0>, __S0: ?::core::marker::Sized>
            crate::Repeater<(__S,), #key0> for __S0
        {
            type Type = Rel<__S>;
        }
    );
    let result = nested_analysis(&item);
    let referrer = result.clone().finish().unwrap();
    assert!(matches!(
        common::entry(&referrer, 0).self_ty.as_ref(),
        Some(SelfRequirement::Any)
    ),);
    assert_eq!(impls_of(result), std::slice::from_ref(&expected),);
}

// ── F77 ──────────────────────────────────────────────────────────────────────

/// F77 (fixed): a path of the Repeater that had arguments was not refused, and each place that
/// wrote the trait's arguments put them after those of the path.
///
/// Trigger: `crate::Repeater<u8>` as the path of the Repeater: the impls, the projections and the
/// const slots were written `crate::Repeater<u8><(T,)>`, which `parse_quote!` refused late, with a
/// message about the tokens.
///
/// Cause: three places wrote `parse_quote!(#repeater<..>)` each.
///
/// Fix: one helper appends the arguments to the last segment, and panics, naming the path, if it
/// has any ("the path of the Repeater names the trait without arguments: `..`"). An empty `<>` or
/// `::<>` counts as none. `Projection::new()` checks the path, and so do `repeater_impls()` and
/// `Projection::of()` for a path put into the public field afterwards, before any item.
#[test]
fn f77_the_path_of_the_repeater_is_checked_before_it_is_used() {
    let item: Item = parse_quote!(
        struct S<T, const N: u8, const B: bool, const M: u128, const L: usize>(Rel<T, N, B, M, L>);
    );
    let mut leaker = common::std_leaker();
    common::intern_item(&mut leaker, &item);
    let result = leaker.result();
    let param: Ident = parse_quote!(__S);
    let marker: Type = parse_quote!(crate::Marker);

    // `<>` is the same as no arguments, for the impls, the projection and the slots (the consts of
    // the item have one each: `u8`, `bool`, `u128` and `usize`). The paths
    // with an empty list are parsed from text: rustfmt removes `<>` inside `parse_quote!`.
    let pairs: Vec<(Path, Vec<Path>)> = ["crate::Repeater", "::a::Repeater", "Repeater"]
        .iter()
        .map(|&bare| {
            let empties: Vec<Path> = [format!("{}<>", bare), format!("{}::<>", bare)]
                .iter()
                .map(|text| {
                    let path = parse_str::<Path>(text).unwrap();
                    match &path.segments.last().unwrap().arguments {
                        PathArguments::AngleBracketed(args) => {
                            assert!(args.args.is_empty(), "{}", text)
                        }
                        other => panic!("{} has {:?}", text, other),
                    }
                    path
                })
                .collect();
            let bare = parse_str::<Path>(bare).unwrap();
            assert!(bare.segments.last().unwrap().arguments.is_none());
            (bare, empties)
        })
        .collect();
    let referrer = result.clone().finish().unwrap();
    let (ix, _, context) = referrer.iter().next().unwrap();
    let mut compared = 0;
    for (bare, empties) in &pairs {
        let impls = |repeater: &Path, target: ImplSelf| -> Vec<RepeaterImpl> {
            result
                .repeater_impls(&Projection::new(repeater.clone(), 5, target.clone()))
                .unwrap()
        };
        for empty in empties {
            for target in [
                ImplSelf::Param(param.clone()),
                ImplSelf::Marker(marker.clone()),
                ImplSelf::Bucket {
                    param: param.clone(),
                },
            ] {
                let (without, with) = (impls(bare, target.clone()), impls(empty, target.clone()));
                assert_eq!(with, without, "{} {:?}", s(empty), target);
                // The path of the trait has the one argument of the entry, after no other.
                let mut expected = bare.clone();
                let args = common::impl_args(&with[0]);
                expected.segments.last_mut().unwrap().arguments =
                    PathArguments::AngleBracketed(parse_quote!(<#args>));
                assert_eq!(with[0].path, expected);
                assert_eq!(s(&with[0]), s(&without[0]));
                compared += 1;
            }
            let self_ty: Type = parse_quote!(Self);
            let projection =
                |repeater: &Path| common::projection(repeater.clone(), 5).of(&self_ty, context, ix);
            assert_eq!(
                projection(empty).unwrap(),
                projection(bare).unwrap(),
                "{}",
                s(empty)
            );
        }
    }
    assert_eq!(compared, 3 * 2 * 3);

    // Arguments are refused, with a message that names the path.
    let empty_result = common::std_leaker().result();
    let mut parenthesized: Path = parse_quote!(a::Repeater);
    parenthesized.segments.last_mut().unwrap().arguments =
        PathArguments::Parenthesized(parse_quote!((u8)));
    let bad: [(Path, &str); 4] = [
        (parse_quote!(crate::Repeater<u8>), "crate::Repeater<u8>"),
        (
            parse_quote!(::a::Repeater<(u8, T)>),
            "::a::Repeater<(u8, T)>",
        ),
        (parse_quote!(Repeater<'a>), "Repeater<'a>"),
        (parenthesized, "a::Repeater (u8)"),
    ];
    for (repeater, written) in bad {
        let message = format!(
            "the path of the Repeater names the trait without arguments: `{}`",
            written
        );
        // `Projection::new()` refuses it.
        assert_eq!(
            common::panic_message(|| {
                let _ = common::projection(repeater.clone(), 5);
            }),
            message,
            "Projection::new with {}",
            written
        );
        // The path is a public field: one put there afterwards is refused by each method, before
        // any item, though the item has const parameters, and also with no entry.
        let with_path = |impl_self: ImplSelf| {
            let mut projection = Projection::new(parse_quote!(Repeater), 5, impl_self);
            projection.path = repeater.clone();
            projection
        };
        for (what, result) in [("entries", &result), ("no entry", &empty_result)] {
            for target in [
                ImplSelf::Param(param.clone()),
                ImplSelf::Marker(marker.clone()),
            ] {
                assert_eq!(
                    common::panic_message(|| {
                        let _ = result.repeater_impls(&with_path(target.clone()));
                    }),
                    message,
                    "repeater_impls of {} with {} for {:?}",
                    what,
                    written,
                    target
                );
            }
        }
        assert_eq!(
            common::panic_message(|| {
                let _ =
                    with_path(ImplSelf::Param(param.clone())).of(&parse_quote!(Self), context, ix);
            }),
            message,
            "Projection::of with {}",
            written
        );
    }
}

// ── F78 ──────────────────────────────────────────────────────────────────────

/// F78 (fixed): the panic for a context id that the Referrer does not have did not name it.
///
/// Trigger: `visitor()` with a `ContextId` of another Referrer panicked with `no entry found for
/// key`, which names neither the call nor the id. A macro that reads the ids from the tokens of
/// another macro meets it.
///
/// Cause: the lookup was `self.contexts[&context]`.
///
/// Fix: `visitor()` panics with `the context N is not one of this Referrer`.
#[test]
fn f78_an_id_that_the_referrer_does_not_have_is_named_in_the_panic() {
    let items: Vec<(Item, usize)> = vec![
        (
            parse_quote!(
                struct S;
            ),
            0,
        ),
        (
            parse_quote!(
                struct S(Rel);
            ),
            1,
        ),
        (
            parse_quote!(
                struct S<T>(Rel, Wrap<T>, (Rel, T));
            ),
            2,
        ),
    ];
    for (item, entries) in items {
        let mut leaker = common::std_leaker();
        common::intern_item(&mut leaker, &item);
        let referrer = leaker.result().finish().unwrap();
        assert_eq!(referrer.iter().count(), entries);
        // A context of another Referrer: the ids are small numbers, so one beyond them.
        let unknown: ContextId = parse_quote!(99);
        let message = "the context 99 is not one of this Referrer";
        assert_eq!(
            common::panic_message(|| {
                let _ = common::visitor(&referrer, unknown, |ty, _, _| ty);
            }),
            message
        );
        assert_eq!(
            common::panic_message(|| {
                let _ = common::expand(&referrer, unknown, parse_quote!(Rel), |ty, _, _| ty);
            }),
            message
        );
    }
}

// ── F79 ──────────────────────────────────────────────────────────────────────

/// F79 (fixed, docs only): the docs said that the rule is never asked about a path that starts
/// with `Self`, but a rule on `Self` accepts the error of a `Self` that has no meaning.
///
/// Trigger: `Rel<Self>` in `enter_generics()` of no generics is the error `SelfWithoutMeaning`,
/// with the cause `Self` and no [`paths`](ErrorItem::paths). A rule that allows
/// `Cause::Type(Self)` drops it, given to `Leaker::with_rule()` or to `LeakerResult::allow()`.
///
/// Cause: the sentence was about the walk of the analysis, which does not ask about `Self`; the
/// filter of the errors asks about the cause of an error with no paths.
///
/// Fix: the docs of `Allow` and `Reason::SelfWithoutMeaning` say so. The rule is asked about `Self`
/// once for the error, never as a path of the walk.
#[test]
fn f79_a_rule_on_self_accepts_the_error_of_a_self_without_meaning() {
    let ty: Type = parse_quote!(Rel<Self>);
    let self_ty = Cause::Type(parse_quote!(Self));
    let rel = Cause::Type(parse_quote!(Rel));
    fn scope<A: Allow>(leaker: &mut Leaker<A>, ty: &Type) {
        leaker.enter_generics(&Generics::default(), |view| view.visit_type(ty));
    }

    // Without a rule: one error, for `Self`, with no paths.
    let mut leaker = Leaker::new();
    scope(&mut leaker, &ty);
    let result = leaker.result();
    assert_eq!(result.errors().len(), 1);
    assert_eq!(result.errors()[0].reason, Reason::SelfWithoutMeaning);
    assert_eq!(result.errors()[0].cause, self_ty);
    assert!(result.errors()[0].paths.is_empty());

    // The rule on `Self` drops it at the result.
    let asked = std::cell::RefCell::new(Vec::new());
    let on_self = |cause: &Cause, _: Place| {
        asked.borrow_mut().push(cause.clone());
        *cause == Cause::Type(parse_quote!(Self))
    };
    assert!(result.allow(on_self).finish().is_ok());
    assert_eq!(asked.borrow().as_slice(), std::slice::from_ref(&self_ty));

    // And in the Leaker: it is asked about `Rel` as the analysis walks (more than once), and about
    // `Self` once, last, for the error at the end of the scope, not as a path of the walk.
    asked.borrow_mut().clear();
    let mut leaker = Leaker::with_rule(on_self);
    scope(&mut leaker, &ty);
    assert!(leaker.result().finish().is_ok());
    let asked = asked.borrow();
    assert!(asked.iter().all(|cause| *cause == rel || *cause == self_ty));
    assert_eq!(asked.iter().filter(|cause| **cause == self_ty).count(), 1);
    assert_eq!(asked.last(), Some(&self_ty));
    drop(asked);

    // `Self::Item` is another cause: a rule on `Self` does not accept it.
    let item: Type = parse_quote!(Wrap<Self::Item>);
    let mut leaker = Leaker::with_rule(on_self);
    scope(&mut leaker, &item);
    let errors = leaker.result().errors().to_vec();
    assert_eq!(errors.len(), 1);
    assert_eq!(errors[0].cause, Cause::Type(parse_quote!(Self::Item)));
}

// ── F81 ──────────────────────────────────────────────────────────────────────

/// F81 (fixed): the error of a bound that names a `Self` that has no name (`UnnameableSelf`) named
/// a different generic from run to run.
///
/// Trigger: `enter_trait(Tr, <T: AsRef<Self>, U: Into<Self>, V: From<Self>>)` with, inside,
/// `enter_type(N, <>)` and `visit_type(Rel<T, U, V>)`: the error said "the bound of `T`", `U` or
/// `V` by chance, so the carried `@ [..]` and the `compile_error!` differed between builds. The
/// crate-private list of the outer `Self` that a context depends on varied in order too.
///
/// Cause: the walk over the generics that the type and their bounds name used a stack filled from
/// `HashSet`s, whose order varies, and it returns the first failure it finds.
///
/// Fix: the names go on the stack in declaration order (outer scopes first), so the walk, its
/// first failure and the list are the same every time.
#[test]
fn f81_the_unnameable_self_is_found_in_declaration_order() {
    // The messages and the notes of the errors.
    let unnameable = || {
        let mut leaker = Leaker::new();
        let generics: Generics = parse_quote!(<T: AsRef<Self>, U: Into<Self>, V: From<Self>>);
        leaker.enter_trait(&parse_quote!(Tr), &generics, |view| {
            view.enter_type(&parse_quote!(N), &Generics::default(), |view| {
                view.visit_type(&parse_quote!(Rel<T, U, V>))
            });
        });
        let result = leaker.result();
        (
            result
                .errors()
                .iter()
                .map(|error| error.message.clone())
                .collect::<Vec<_>>(),
            all_notes(result.errors()),
        )
    };
    // The context of the entry, written with `Debug`: its list of the outer `Self` is private.
    let substituted = || {
        let mut leaker = Leaker::new();
        let generics: Generics = parse_quote!(<T: AsRef<Self>>);
        leaker.enter_impl(&generics, &parse_quote!(S<T>), None, |view| {
            view.enter_type(&parse_quote!(N1), &parse_quote!(<U: Into<Self>>), |view| {
                view.enter_type(&parse_quote!(N2), &parse_quote!(<W>), |view| {
                    view.visit_type(&parse_quote!(Rel<T, U, W>))
                });
            });
        });
        let referrer = leaker.result().finish().unwrap();
        referrer
            .iter()
            .map(|(_, ty, context)| (s(ty), format!("{:?}", context)))
            .collect::<Vec<_>>()
    };

    let first = unnameable();
    assert_eq!(
        first.0,
        ["the bound of `T` mentions a `Self` which has no name here"]
    );
    assert_eq!(
        first.1[0],
        "the bound of `T` mentions the `Self` of the trait `Tr<T, U, V>`, which has no name \
         here: `Self` is something else in the item around"
    );
    let outer = substituted();
    assert_eq!(outer.len(), 1);
    assert_eq!(outer[0].0, t(parse_quote!(Rel<T, U, W>)));
    // `S<T>`, then `N1<U>`: the outer scope first.
    let list = &outer[0].1[outer[0].1.find("substituted: ").unwrap()..];
    let (s, n1) = (
        list.find("Ident(S)").unwrap(),
        list.find("Ident(N1)").unwrap(),
    );
    assert!(s < n1, "{}", list);
    // Each `HashSet` has a hasher of its own: a run in one process shows what runs of many do.
    for _ in 0..100 {
        assert_eq!(unnameable(), first);
        assert_eq!(substituted(), outer);
    }
}

// ── F82 ──────────────────────────────────────────────────────────────────────

/// F82 (fixed): a receiver `self: &Box<Self>` (or `&Rc<Self>`, `&Arc<Self>`) was taken to give its
/// lifetime to the elided output, as rustc does only from 1.81 (the MSRV is 1.56).
///
/// Trigger: `impl S { fn m<'a, T: 'a>(self: &Box<Self>, y: &u8, x: Rel<T>) -> &u8 { .. } }` was
/// accepted with the entry `Rel<T>`, whose copy names `'a` in `x`. rustc before 1.81 leaves such a
/// receiver out of the count, so the copy's inputs have two lifetimes: E0106.
///
/// Cause: the Leaker took any reference in the type of the receiver for one that gives its
/// lifetime.
///
/// Fix: only a reference to `Self` counts (`&self`, `self: &mut Self`, `Pin<&Self>`,
/// `Box<&Self>`), or to another type, which may be the impl's (`&S`). Another receiver is left out
/// of the count, so `Rel<T>` is an error, as for `self: Box<Self>`.
///
/// Limitation: `&Box<S>`, the impl's type written out, counts as a reference to another type.
#[test]
fn f82_a_reference_to_a_box_of_self_gives_no_lifetime_to_the_output() {
    let analyze = |receiver: FnArg| -> Vec<String> {
        let item: ItemImpl = parse_quote!(
            impl S {
                fn m<'a, T: 'a>(#receiver, y: &u8, x: Rel<T>) -> &u8 {
                    loop {}
                }
            }
        );
        let mut leaker = Leaker::with_rule(Rules::new().primitive().std_crates());
        leaker.intern_impl(&item);
        let result = leaker.result();
        result
            .errors()
            .iter()
            .map(|error| error.message.clone())
            .collect()
    };

    // The receivers that have a reference to `Self`: it gives its lifetime to the output, so adding
    // `'a` to `x` is fine (rustc 1.56 and later).
    let giving: Vec<FnArg> = vec![
        parse_quote!(&self),
        parse_quote!(&mut self),
        parse_quote!(&'a self),
        parse_quote!(self: &Self),
        parse_quote!(self: &mut Self),
        parse_quote!(self: &'a Self),
        parse_quote!(self: &S),
        parse_quote!(self: ::std::boxed::Box<&Self>),
        parse_quote!(self: ::std::rc::Rc<&Self>),
        parse_quote!(self: ::std::pin::Pin<&Self>),
        parse_quote!(self: ::std::pin::Pin<&mut Self>),
        parse_quote!(self: &::std::pin::Pin<&Self>),
    ];
    for receiver in giving {
        let what = s(&receiver);
        assert_eq!(analyze(receiver), Vec::<String>::new(), "{}", what);
    }

    // A receiver with no reference: the inputs have `y` and, in the copy, `'a`.
    let expected = analyze(parse_quote!(self: ::std::boxed::Box<Self>));
    assert_eq!(expected, ["use absolute path: 'Rel<T>'"]);
    assert_eq!(analyze(parse_quote!(self)), expected);
    // A reference to a type that holds `Self`: rustc before 1.81 leaves it out of the count.
    let leaving: Vec<FnArg> = vec![
        parse_quote!(self: &::std::boxed::Box<Self>),
        parse_quote!(self: &::std::rc::Rc<Self>),
        parse_quote!(self: &::std::sync::Arc<Self>),
        parse_quote!(self: &mut ::std::boxed::Box<Self>),
    ];
    for receiver in leaving {
        let what = s(&receiver);
        assert_eq!(analyze(receiver), expected, "{}", what);
    }
}

// ── F83 ──────────────────────────────────────────────────────────────────────

/// F83 (fixed): the Referrer did not see every name that a block declares, so it replaced a type in
/// the block that names one by the entry of the same tokens outside.
///
/// Trigger: `impl S { pub fn f(_x: b) { use ::p::m::b::{self}; let _y: b = b::X; } }` with
/// `check_values`: `b` of the parameter is an entry, `b` in the body is what the block imports. The
/// copy wrote `let _y: <H as Repeater<..>>::Type = b::X;` (E0308). The same for a block's
/// `const fn n` with `[u8; n()]` (fixture 9249) and a `macro_rules!` named like a type.
///
/// Cause: the Referrer left out `self` of `use a::b::{self}` and the names of a `fn` and a
/// `macro_rules!`; the Leaker kept a list of its own.
///
/// Fix: one list of declared names serves both: types, traits, consts, statics, functions,
/// modules, `macro_rules!`, extern crates, and what a `use` brings in (`self` names what is before
/// the group; `as _` and a glob bring none). Namespaces are not told apart.
#[test]
fn f83_a_block_hides_the_entries_that_name_what_it_declares() {
    let copy = |item: &ItemImpl| -> String {
        let mut leaker = Leaker::with_rule(Rules::new().primitive());
        leaker.check_values = true;
        leaker.check_bodies = true;
        let context = leaker.intern_impl(item);
        let referrer = leaker.result().recovered();
        let mut copy = item.clone();
        let mut visitor = common::visitor(&referrer, context, common::placeholder);
        visitor.visit_item_impl_mut(&mut copy);
        s(&copy)
    };
    // Each item has the entry of its first parameter, `P<1usize>` (`P<0usize>` is `S`), and a type
    // in the block that either is the same tokens or is not.
    let cases: Vec<(&str, ItemImpl, ItemImpl)> = vec![
        (
            "use a::b::{self}",
            parse_quote!(
                impl S {
                    pub fn f(_x: b) {
                        use ::p::m::b::{self};
                        let _y: b = b::X;
                    }
                }
            ),
            parse_quote!(
                impl P<0usize> {
                    pub fn f(_x: P<1usize>) {
                        use ::p::m::b::{self};
                        let _y: b = b::X;
                    }
                }
            ),
        ),
        (
            "use a::{b::{self}, c}",
            parse_quote!(
                impl S {
                    pub fn f(_x: b) {
                        use ::p::m::{b::{self}, c};
                        let _y: b = b::X;
                    }
                }
            ),
            parse_quote!(
                impl P<0usize> {
                    pub fn f(_x: P<1usize>) {
                        use ::p::m::{b::{self}, c};
                        let _y: b = b::X;
                    }
                }
            ),
        ),
        (
            "use a::b (the name was seen)",
            parse_quote!(
                impl S {
                    pub fn f(_x: b) {
                        use ::p::m::b;
                        let _y: b = b::X;
                    }
                }
            ),
            parse_quote!(
                impl P<0usize> {
                    pub fn f(_x: P<1usize>) {
                        use ::p::m::b;
                        let _y: b = b::X;
                    }
                }
            ),
        ),
        (
            "const fn n",
            parse_quote!(
                impl S {
                    pub fn f(_x: [u8; n()]) {
                        const fn n() -> usize {
                            3
                        }
                        let _a: [u8; n()] = [0; 3];
                    }
                }
            ),
            parse_quote!(
                impl P<0usize> {
                    pub fn f(_x: P<1usize>) {
                        const fn n() -> usize {
                            3
                        }
                        let _a: [u8; n()] = [0; 3];
                    }
                }
            ),
        ),
        (
            "macro_rules! m, named like a type",
            parse_quote!(
                impl S {
                    pub fn f(_x: Wrap<m>) {
                        macro_rules! m {
                            () => {};
                        }
                        let _a: Wrap<m> = loop {};
                    }
                }
            ),
            parse_quote!(
                impl P<0usize> {
                    pub fn f(_x: P<1usize>) {
                        macro_rules! m {
                            () => {};
                        }
                        let _a: Wrap<m> = loop {};
                    }
                }
            ),
        ),
        // The names that a `use` does not bring: `self as c` brings `c`, `as _` none.
        (
            "use a::b::{self as c}",
            parse_quote!(
                impl S {
                    pub fn f(_x: b) {
                        use ::p::m::b::{self as c};
                        let _y: b = c::X;
                    }
                }
            ),
            parse_quote!(
                impl P<0usize> {
                    pub fn f(_x: P<1usize>) {
                        use ::p::m::b::{self as c};
                        let _y: P<1usize> = c::X;
                    }
                }
            ),
        ),
        (
            "use a::b as _",
            parse_quote!(
                impl S {
                    pub fn f(_x: b) {
                        use ::p::m::b as _;
                        let _y: b = loop {};
                    }
                }
            ),
            parse_quote!(
                impl P<0usize> {
                    pub fn f(_x: P<1usize>) {
                        use ::p::m::b as _;
                        let _y: P<1usize> = loop {};
                    }
                }
            ),
        ),
    ];
    for (name, item, expected) in cases {
        assert_eq!(copy(&item), s(&expected), "{}", name);
    }
}

// ── F84 ──────────────────────────────────────────────────────────────────────

/// F84 (fixed): a where predicate whose bounded type is a generic in parentheses (`where (T): ..`)
/// or in an invisible group (`$t:ty` of a `macro_rules!`) was not a bound of the generic.
///
/// Trigger: `fn f<'b, T>(&self, t: T) where (T): ::p::Bound<&'b dyn RelTr>;` in a trait was
/// rejected (`UnprovableBound`, no entry), where `T: ::p::Bound<..>` has the entry `&'b dyn RelTr`.
/// The context of an entry also changed when the Referrer was written and read back.
///
/// Cause: the code that tells a bound of the generic from another predicate, and the code that
/// moves it to the generic, matched only a bounded type that is the bare path.
///
/// Fix: both look through parentheses and groups, so the predicate moves to the parameter: the
/// same context, errors and entries as `T: ..`.
#[test]
fn f84_a_predicate_on_a_generic_in_parentheses_or_a_group_is_a_bound_of_it() {
    // Wraps the bounded type of each predicate in an invisible group.
    struct Group;
    impl VisitMut for Group {
        fn visit_predicate_type_mut(&mut self, i: &mut PredicateType) {
            let inner = std::mem::replace(&mut i.bounded_ty, Type::Verbatim(Default::default()));
            i.bounded_ty = Type::Group(TypeGroup {
                group_token: Default::default(),
                elem: Box::new(inner),
            });
        }
    }
    // The errors, the entries with their generics, the serialized Referrer, whether it reads back
    // equal, and the copy with `P<ix>` for each entry.
    let outcome = |item: &Item| {
        let mut leaker = common::std_leaker();
        let context = common::intern_item(&mut leaker, item);
        let result = leaker.result();
        let errors = messages(result.errors());
        let referrer = result.finish().unwrap();
        let entries: Vec<(String, String)> = referrer
            .iter()
            .map(|(_, ty, context)| (s(ty), s(&context.generics)))
            .collect();
        let tokens = s(&referrer);
        let back: Referrer = syn::parse_str(&tokens).unwrap();
        assert!(back == referrer, "{} reads back as another", tokens);
        let mut copy = item.clone();
        common::visitor(&referrer, context, common::placeholder).visit_item_mut(&mut copy);
        (errors, entries, tokens, s(&copy))
    };
    // The same item with the predicate written in each way.
    let forms: Vec<(Item, Item, Item)> = vec![
        (
            parse_quote!(
                trait Tr {
                    fn f<'b, T>(&self, t: T)
                    where
                        T: ::p::Bound<&'b dyn RelTr>;
                }
            ),
            parse_quote!(
                trait Tr {
                    fn f<'b, T>(&self, t: T)
                    where
                        (T): ::p::Bound<&'b dyn RelTr>;
                }
            ),
            parse_quote!(
                trait Tr {
                    fn f<'b, T>(&self, t: T)
                    where
                        T: ::p::Bound<P<0usize>>;
                }
            ),
        ),
        (
            parse_quote!(
                struct S<'a, T>(&'a (), T)
                where
                    T: Iterator<Item = Rel<'a>>;
            ),
            parse_quote!(
                struct S<'a, T>(&'a (), T)
                where
                    (T): Iterator<Item = Rel<'a>>;
            ),
            parse_quote!(
                struct S<'a, T>(&'a (), T)
                where
                    T: Iterator<Item = P<0usize>>;
            ),
        ),
    ];
    for (bare, parenthesized, replaced) in forms {
        let (errors, entries, tokens, copy) = outcome(&bare);
        assert_eq!(errors, "", "{}", s(&bare));
        assert_eq!(entries.len(), 1, "{}", s(&bare));
        assert_eq!(copy, s(&replaced));
        let mut grouped = bare.clone();
        Group.visit_item_mut(&mut grouped);
        for (form, item) in [("parentheses", &parenthesized), ("group", &grouped)] {
            let (errors, got, serialized, copied) = outcome(item);
            assert_eq!(errors, "", "{}: {}", form, s(item));
            assert_eq!(got, entries, "{}: {}", form, s(item));
            // The contexts are the same, but for the generics as declared, which the scope's
            // context keeps as they are written (`(T)`).
            let split = |text: &str| {
                let at = text.find(" declared ").unwrap();
                (text[..at].to_string(), text[at..].replace("(T)", "T"))
            };
            assert_eq!(split(&serialized), split(&tokens), "{}: {}", form, s(item));
            // The predicate stays as it was written, with the type replaced.
            assert_eq!(copied.replace("(T)", "T"), copy, "{}: {}", form, s(item));
        }
    }
    // The entries: the type of the bound of the method, and `Rel<'a>`, each with its lifetime.
    let (_, entries, _, _) = outcome(&forms_of_the_method());
    assert_eq!(
        entries,
        [(t(parse_quote!(&'b dyn RelTr)), "< 'b >".to_string())]
    );
    fn forms_of_the_method() -> Item {
        parse_quote!(
            trait Tr {
                fn f<'b, T>(&self, t: T)
                where
                    (T): ::p::Bound<&'b dyn RelTr>;
            }
        )
    }
}

// ── F86 ──────────────────────────────────────────────────────────────────────

/// Analyzes the type `[u8; block]` as a field and in the default of a type parameter (of a struct,
/// as a second parameter, of an enum, a union and a trait), with `check_values` and without, and
/// asserts that each default gives the entries of the field, and that the generics `<T = [u8;
/// block]>` give no error in `enter_generics()` and `enter_type()`. Returns the entries of the
/// field, with whether their context has `Self`.
fn assert_default_like_field(name: &str, block: &Expr) -> Vec<(String, bool)> {
    let ty: Type = parse_quote!([u8; #block]);
    let entries = |item: &Item, check_values: bool| {
        let mut leaker = common::std_leaker();
        leaker.check_values = check_values;
        common::intern_item(&mut leaker, item);
        let referrer =
            (leaker.result().finish()).unwrap_or_else(|e| panic!("{}: {}", s(item), messages(&e)));
        (referrer.iter())
            .map(|(_, ty, context)| (s(ty), context.self_ty.is_some()))
            .collect::<Vec<_>>()
    };
    let field = entries(&parse_quote!(struct S(#ty);), false);
    for check_values in [false, true] {
        let what = format!("{}, check_values {}", name, check_values);
        assert_eq!(
            entries(&parse_quote!(struct S(#ty);), check_values),
            field,
            "{}",
            what
        );
        for (place, item) in [
            ("struct", parse_quote!(struct S<T = #ty>(T);)),
            (
                "second parameter",
                parse_quote!(struct S<U, T = #ty>(U, T);),
            ),
            ("enum", parse_quote!(enum E<T = #ty> { A(T) })),
            ("union", parse_quote!(union Un<T = #ty> { a: T })),
            ("trait", parse_quote!(trait Tr<T = #ty> {})),
        ] {
            assert_eq!(
                entries(&item, check_values),
                field,
                "{} in a {}",
                what,
                place
            );
        }
        let generics: Generics = parse_quote!(<T = #ty>);
        let mut leaker = common::std_leaker();
        leaker.check_values = check_values;
        leaker.enter_generics(&generics, |_| ());
        assert_eq!(
            messages(leaker.result().errors()),
            "",
            "{} in enter_generics",
            what
        );
        let mut leaker = common::std_leaker();
        leaker.check_values = check_values;
        leaker.enter_type(&parse_quote!(S), &generics, |_| ());
        assert_eq!(
            messages(leaker.result().errors()),
            "",
            "{} in enter_type",
            what
        );
    }
    field
}

/// F86 (fixed): a generic of an item nested in the default of a type parameter was taken for a
/// relative path.
///
/// Trigger: `struct S<T = [u8; { struct Z<U>(U); 3 }]>(T)` made the array an entry (as a field it
/// is none), and `struct Z<U>(U, Option<Self>)` there gave `SelfInDefault`. The same for a `fn`, an
/// enum, a union, a trait or an impl in the block, and in the default of a trait.
///
/// Cause: the defaults of the root item's type parameters are analyzed by a visitor of their own,
/// which started outside every item, so it took the nested item for the outermost one: `U` was a
/// relative path, and its `Self` the one of the scope.
///
/// Fix: that visitor starts inside the scope, so an item in a default is nested, with its own
/// generics and `Self`: a default has the same errors and entries as the same type in a field, with
/// `check_values` or not. The harness also holds F85's input, a value block in the default of a
/// type parameter (`[u8; { const C: usize = rel::N; 3 }]`), which F101 made an entry like a field.
#[test]
fn f86_the_generics_of_an_item_in_the_default_of_a_type_parameter_are_its_own() {
    // The block, and the entries of the type: none where the generics of the item are its own.
    let blocks: Vec<(&str, Expr, &[&str])> = vec![
        (
            "struct",
            parse_quote!({
                struct Z<U>(U);
                3
            }),
            &[],
        ),
        (
            "enum",
            parse_quote!({
                enum Z<U> {
                    A(U),
                }
                3
            }),
            &[],
        ),
        (
            "union",
            parse_quote!({
                union Z<U> {
                    a: U,
                }
                3
            }),
            &[],
        ),
        (
            "fn",
            parse_quote!({
                fn f<U>(_u: U) {}
                3
            }),
            &[],
        ),
        (
            "trait",
            parse_quote!({
                trait Tr<U> {
                    fn f(&self, u: U);
                }
                3
            }),
            &[],
        ),
        (
            "impl",
            parse_quote!({
                impl<U> Z<U> {}
                3
            }),
            &["[u8 ; { impl < U > Z < U > { } 3 }]"],
        ),
        (
            "Self of the item",
            parse_quote!({
                struct Z<U>(U, Option<Self>);
                3
            }),
            &[],
        ),
        (
            "a relative type in the item",
            parse_quote!({
                struct Z<U>(U, Rel);
                3
            }),
            &["[u8 ; { struct Z < U > (U , Rel) ; 3 }]"],
        ),
        (
            "two generics",
            parse_quote!({
                struct Z<U, V: Into<U>>(U, V);
                3
            }),
            &[],
        ),
        // F85: a relative path in a value is part of the array, which is the entry (F101).
        (
            "a relative path in a value",
            parse_quote!({
                const C: usize = rel::N;
                3
            }),
            &["[u8 ; { const C : usize = rel :: N ; 3 }]"],
        ),
        (
            "no relative path in a value",
            parse_quote!({
                const C: usize = 2;
                3
            }),
            &[],
        ),
    ];
    for (name, block, expected) in &blocks {
        let entries = assert_default_like_field(name, block);
        let entries: Vec<&str> = entries.iter().map(|(ty, _)| ty.as_str()).collect();
        assert_eq!(entries, *expected, "{}", name);
    }
}

// ── F88 ──────────────────────────────────────────────────────────────────────

/// F88 (fixed): a `Self` projection in parentheses or an invisible group, in a bound, was reported
/// twice.
///
/// Trigger: `struct S<T: ::p::Bound<(Self::A)>>(T)` gave two `UnprovableBound` (`Self::A` and
/// `(Self::A)`; `((Self::A))` three) and two parts in the recovered Referrer, where
/// `Bound<Self::A>` gives one of each. The same with the group of a `$t:ty`, for `<Self as Tr>::A`,
/// `Self::A::B`, one in the arguments of a path, and the bare `Self` in a bound of an impl.
///
/// Cause: the check of a type in a predicate that names `Self` looks through the wrappers, and it
/// runs for each type the walk decides: the one inside and the wrapper.
///
/// Fix: the check skips the wrapper, so the type inside has the error once, as without it.
#[test]
fn f88_a_self_projection_in_parentheses_or_a_group_has_one_error() {
    // Wraps each type headed by `Self` (`Self`, `Self::A`, `<Self as Tr>::A`) `times` times in
    // parentheses, or in an invisible group.
    struct Wrap {
        group: bool,
        times: usize,
        wrapped: usize,
    }
    impl VisitMut for Wrap {
        fn visit_type_mut(&mut self, i: &mut Type) {
            let headed = match &*i {
                Type::Path(path) => match &path.qself {
                    Some(qself) => matches!(&*qself.ty, Type::Path(p) if p.path.is_ident("Self")),
                    None => path
                        .path
                        .segments
                        .first()
                        .map_or(false, |s| s.ident == "Self"),
                },
                _ => false,
            };
            if !headed {
                return syn::visit_mut::visit_type_mut(self, i);
            }
            for _ in 0..self.times {
                let elem = Box::new(i.clone());
                *i = if self.group {
                    Type::Group(TypeGroup {
                        group_token: Default::default(),
                        elem,
                    })
                } else {
                    Type::Paren(TypeParen {
                        paren_token: Default::default(),
                        elem,
                    })
                };
                self.wrapped += 1;
            }
        }
    }
    // Replaces `Rel` by the entry that the copy writes for it.
    struct Replace;
    impl VisitMut for Replace {
        fn visit_type_mut(&mut self, i: &mut Type) {
            if matches!(&*i, Type::Path(p) if p.path.is_ident("Rel")) {
                *i = parse_quote!(P<0usize>);
            }
            syn::visit_mut::visit_type_mut(self, i);
        }
    }
    // The items, and the reasons of the errors of each.
    let items: Vec<(&str, Item, Vec<Reason>)> = vec![
        (
            "a bound",
            parse_quote!(
                struct S<T: ::p::Bound<Self::A>>(T);
            ),
            vec![Reason::UnprovableBound],
        ),
        (
            "a where predicate",
            parse_quote!(
                struct S<T>(T)
                where
                    T: ::p::Bound<Self::A>;
            ),
            vec![Reason::UnprovableBound],
        ),
        (
            "a qualified path",
            parse_quote!(
                struct S<T: ::p::Bound<<Self as ::p::Tr>::A>>(T);
            ),
            vec![Reason::UnprovableBound],
        ),
        (
            "two segments after Self",
            parse_quote!(
                struct S<T: ::p::Bound<Self::A::B>>(T);
            ),
            vec![Reason::UnprovableBound],
        ),
        (
            "in the arguments of a path",
            parse_quote!(
                struct S<T: ::p::Bound<Vec<Self::A>>>(T);
            ),
            vec![Reason::UnprovableBound],
        ),
        (
            "Self of an impl, and a leaked type",
            parse_quote!(
                impl<T: Into<Self> + AsRef<Rel>> S<T> {}
            ),
            vec![Reason::SelfInImplSelfType, Reason::UnprovableBound],
        ),
    ];
    // The reasons and messages of the errors, and the entries, and the copy of the item.
    let analyze = |item: &Item| -> (Vec<(Reason, String)>, Vec<String>, String) {
        let mut leaker = common::std_leaker();
        let context = common::intern_item(&mut leaker, item);
        let result = leaker.result();
        let errors = (result.errors().iter())
            .map(|error| (error.reason, error.message.clone()))
            .collect();
        let referrer = result.recovered();
        let entries = referrer.iter().map(|(_, ty, _)| s(ty)).collect();
        let mut copy = item.clone();
        common::visitor(&referrer, context, common::placeholder).visit_item_mut(&mut copy);
        (errors, entries, s(&copy))
    };
    for (name, item, reasons) in &items {
        let (errors, entries, _) = analyze(item);
        assert_eq!(
            errors.iter().map(|(reason, _)| *reason).collect::<Vec<_>>(),
            *reasons,
            "{}",
            name
        );
        for group in [false, true] {
            for times in [1, 2] {
                let mut wrapped = item.clone();
                let mut wrap = Wrap {
                    group,
                    times,
                    wrapped: 0,
                };
                wrap.visit_item_mut(&mut wrapped);
                assert!(wrap.wrapped >= times, "{}: nothing to wrap", name);
                let what = format!("{}, group: {}, times: {}", name, group, times);
                let (got_errors, got_entries, got_copy) = analyze(&wrapped);
                // The same errors and entries as without the wrapper.
                assert_eq!(got_errors, errors, "{}", what);
                assert_eq!(got_entries, entries, "{}", what);
                // The copy is the item with the wrapper, and the entry replaced.
                let mut expected = wrapped.clone();
                Replace.visit_item_mut(&mut expected);
                assert_eq!(got_copy, s(&expected), "{}", what);
            }
        }
    }
}

// ── F89 ──────────────────────────────────────────────────────────────────────

/// F89 (fixed): the outlives relations of the where clause of a generic associated type were
/// written for the wrong generics when the argument for its parameter is not a name.
///
/// Trigger: `trait Tr { type G<'b, U: Deref> where U: 'b, U::Target: 'b; fn f<'x, W: Deref>(&self,
/// u: W, a: Rel<&'x Self::G<'x, &'x u8>>); }`: the context of `Rel<..>` had `U::Target: 'x`, with
/// the GAT's own `U`, which the Repeater's impl does not declare (E0412; fixture 9251). The same
/// for a parameter of the trait (`V::Item: 'x` with `<Self as Tr<&'x mut T>>::G<'x>`).
///
/// Cause: `U::Item` with `U` replaced by `Arg` was written as the segments of `Arg` then `Item`,
/// a projection only if `Arg` is a name, and left as `U::Item` where `Arg` is no path (`&'x u8`).
///
/// Fix: a name is replaced as before (`T::Item`); any other argument gives `<Arg>::Item`, whose
/// parts are what the context requires (`T: 'x` for `&'x mut T`). This can be stricter than the
/// original (`T::IntoIter: 'x` for `<T::IntoIter as Iterator>::Item: 'x`), as the context keeps no
/// trait bound of the GAT.
#[test]
fn f89_the_arguments_of_a_generic_associated_type_replace_its_parameters_in_its_where_clause() {
    // The context of the entry `Rel<..>` of the method `f` of the trait: its generics and its
    // predicates.
    let context_of = |item: ItemTrait| -> String {
        let mut leaker = Leaker::with_rule(Rules::new().primitive().prelude().std_crates());
        leaker.intern_trait(&item);
        let result = leaker.result();
        assert!(result.errors().is_empty(), "{}", s(&item));
        let referrer = result.finish().unwrap();
        let entries: Vec<_> = referrer.iter().collect();
        assert_eq!(entries.len(), 1, "{}", s(&item));
        let generics = &entries[0].2.generics;
        let predicates: Vec<String> = (generics.where_clause.iter())
            .flat_map(|where_clause| where_clause.predicates.iter())
            .map(s)
            .collect();
        format!("{} where {}", s(generics), predicates.join(" , "))
    };
    // Each trait declares `type G` with a relation on a projection of a parameter (of `G`, or of
    // the trait), and `f` uses it with the argument for the parameter. The context has the
    // relation of the argument: `X::Item: 'x` for the name `X`, the parts of the argument for a type
    // that is no name, and none for a type with no generic.
    let gat = |params: &str, arg: &str| -> ItemTrait {
        parse_str(&format!(
            "trait Tr {{ type G<'b, U: Iterator> where U::Item: 'b; \
             fn f<'x{}>(&self, a: Rel<&'x Self::G<'x, {}>>); }}",
            params, arg
        ))
        .unwrap()
    };
    let trait_param = |params: &str, arg: &str| -> ItemTrait {
        parse_str(&format!(
            "trait Tr<V: Iterator> {{ type G<'b> where V::Item: 'b; \
             fn f<'x{}>(&self, a: Rel<&'x <Self as Tr<{}>>::G<'x>>); }}",
            params, arg
        ))
        .unwrap()
    };
    let cases: Vec<(&str, ItemTrait, &str)> = vec![
        // A name: the projection of the generic.
        (
            "a generic",
            gat(", T: Iterator", "T"),
            "< 'x , T : Iterator > where Self :: G < 'x , T > : 'x , T :: Item : 'x",
        ),
        (
            "the trait parameter, a generic",
            trait_param(", T: Iterator", "T"),
            "< 'x , T : Iterator , V : Iterator > where < Self as Tr < T > > :: G < 'x > : 'x , T :: Item : 'x",
        ),
        // A type with no generic: nothing to prove.
        (
            "a concrete type",
            gat("", "::std::vec::IntoIter<u8>"),
            "< 'x > where Self :: G < 'x , :: std :: vec :: IntoIter < u8 > > : 'x",
        ),
        (
            "the trait parameter, a concrete type",
            trait_param("", "::std::vec::IntoIter<u8>"),
            "< 'x , V : Iterator > where < Self as Tr < :: std :: vec :: IntoIter < u8 > > > :: G < 'x > : 'x",
        ),
        // A type that is no name: its parts, not the parameter of the declaration.
        (
            "a reference to a generic",
            gat(", T: Iterator", "&'x mut T"),
            "< 'x , T : 'x + Iterator > where Self :: G < 'x , & 'x mut T > : 'x",
        ),
        (
            "the trait parameter, a reference to a generic",
            trait_param(", T: Iterator", "&'x mut T"),
            "< 'x , T : 'x + Iterator , V : Iterator > where < Self as Tr < & 'x mut T > > :: G < 'x > : 'x",
        ),
        // The same name in the method as the parameter of the declaration: not taken for it.
        (
            "a reference to a generic that has the name of the parameter",
            gat(", U: Iterator", "&'x mut U"),
            "< 'x , U : 'x + Iterator > where Self :: G < 'x , & 'x mut U > : 'x",
        ),
        (
            "the trait parameter, a reference to a generic with its name",
            trait_param(", V2: Iterator", "&'x mut V2"),
            "< 'x , V : Iterator , V2 : 'x + Iterator > where < Self as Tr < & 'x mut V2 > > :: G < 'x > : 'x",
        ),
        // A projection of a generic: `<T::IntoIter>::Item` is the projection of `T::IntoIter`, and
        // names no `T::IntoIter::Item`.
        (
            "a projection of a generic",
            gat(", T: IntoIterator", "T::IntoIter"),
            "< 'x , T : IntoIterator > where Self :: G < 'x , T :: IntoIter > : 'x , T :: IntoIter : 'x",
        ),
    ];
    for (name, item, expected) in cases {
        assert_eq!(context_of(item), expected, "{}", name);
    }
    // The case of the fixture: the parameter of `G` is named `U` and no generic of the method is.
    let item: ItemTrait = parse_quote!(
        trait Tr {
            type G<'b, U: ::core::ops::Deref>
            where
                U: 'b,
                U::Target: 'b;
            fn f<'x, W: ::core::ops::Deref>(&self, u: W, a: Rel<&'x Self::G<'x, &'x u8>>);
        }
    );
    assert_eq!(
        context_of(item),
        "< 'x > where Self :: G < 'x , & 'x u8 > : 'x"
    );
}

// ── F90 ──────────────────────────────────────────────────────────────────────

/// F90 (fixed): `substitute_args()` and `Referrer::complete_args()` wrote the associated type of an
/// argument that is not a type parameter as `<arg>::A`, which rustc rejects.
///
/// Trigger: `trait Tr<T: Iterator> { fn f(&self) -> Option<T::Item>; }` and
/// `impl Tr<::std::vec::IntoIter<u8>> for ..`: the copy had
/// `Option<<::std::vec::IntoIter<u8>>::Item>`, E0223 "ambiguous associated type" (fixture 9252).
///
/// Cause: F60 and F74 wrote `<value>::A` for every value; it resolves only for a type parameter.
///
/// Fix: one identifier keeps `<V>::A`. Otherwise `complete_args()` writes `<value as Trait>::A`
/// with the parameter's one trait bound (inline or in a where clause, bindings dropped). With none,
/// several or a `for<'a>` binder, and always in `substitute_args()`, which knows no bounds, the
/// call fails with `EncodeError::AmbiguousAssociatedType` and changes nothing.
///
/// Limitations: the trait must declare the type itself (`F: Fn(u8) -> u8` gives the refused
/// `<V as Fn(u8)>::Output`), and one identifier may name a concrete type.
#[test]
fn f90_the_associated_type_of_an_argument_is_written_through_a_bound_or_refused() {
    let complete = |params: Punctuated<GenericParam, Token![,]>, given: &[&str]| {
        complete_args_of(&params, &arguments(given))
            .map(|completed| s(completed.last().unwrap()))
            .map_err(|error| error.to_string())
    };
    let ambiguous =
        |params: Punctuated<GenericParam, Token![,]>, given: &[&str]| match complete_args_of(
            &params,
            &arguments(given),
        ) {
            Err(EncodeError::AmbiguousAssociatedType(ty)) => s(&*ty),
            other => panic!("{}: {:?}", s(&params), other),
        };
    let into_iter = ["::std::vec::IntoIter<u8>"];

    // The trigger, in a default: the one bound of `T`.
    assert_eq!(
        complete(parse_quote!(T: Iterator, U = Option<T::Item>), &into_iter),
        Ok(t(parse_quote!(
            Option<<::std::vec::IntoIter<u8> as Iterator>::Item>
        )))
    );
    // The bindings of the bound go; the path, the lifetime and `?Sized` are as they are.
    for (bound, trait_path) in [
        ("Iterator<Item = u8>", "Iterator"),
        (
            "::core::iter::Iterator<Item = u8>",
            "::core::iter::Iterator",
        ),
        ("'a + ?Sized + Iterator", "Iterator"),
        ("Iterator<Item: Clone>", "Iterator"),
        ("Tr<'a, u8, Item = u8>", "Tr<'static, u8>"),
    ] {
        let generics: Generics = parse_str(&format!("<'a, T: {}, U = T::Item>", bound)).unwrap();
        let params = generics.params;
        let expected = format!("<::std::vec::IntoIter<u8> as {}>::Item", trait_path);
        assert_eq!(
            complete(params, &["'static", "::std::vec::IntoIter<u8>"]),
            Ok(t(parse_str(&expected).unwrap())),
            "{}",
            bound
        );
    }
    // The earlier parameters are replaced in the arguments of the bound and in the path after `T`
    // (`T::A<U>`), and `<T>::A` is the same path as `T::A`.
    assert_eq!(
        complete(parse_quote!(U, T: Tr<U>, V = T::A<U>), &["u8", "Vec<u8>"]),
        Ok(t(parse_quote!(<Vec<u8> as Tr<u8>>::A<u8>)))
    );
    assert_eq!(
        complete(parse_quote!(T: Iterator, U = <T>::Item), &into_iter),
        Ok(t(parse_quote!(
            <::std::vec::IntoIter<u8> as Iterator>::Item
        )))
    );
    // An argument that is one identifier keeps `<V>::A`, with a bound or without.
    for params in [
        parse_quote!(T: Iterator, U = T::Item),
        parse_quote!(T, U = T::Item),
        parse_quote!(T: Clone + Iterator, U = T::Item),
    ] {
        assert_eq!(complete(params, &["I"]), Ok(t(parse_quote!(<I>::Item))));
    }

    // The bound may be in the where clause, which counts with the bound of the parameter: one
    // bound, with either; two, with both.
    for (generics, expected) in [
        (
            "<T, U = T::Item> where T: Iterator",
            Ok("<::std::vec::IntoIter<u8> as Iterator>::Item"),
        ),
        (
            "<T: ?Sized, U = T::Item> where T: Iterator<Item = u8>",
            Ok("<::std::vec::IntoIter<u8> as Iterator>::Item"),
        ),
        (
            "<T: Iterator, U = T::Item> where T: Clone",
            Err("<::std::vec::IntoIter<u8>>::Item"),
        ),
        (
            "<T, U = T::Item> where T: Iterator, T: Clone",
            Err("<::std::vec::IntoIter<u8>>::Item"),
        ),
        // A predicate on another type is no bound of `T`.
        (
            "<T: Iterator, U = T::Item> where Vec<T>: Clone",
            Ok("<::std::vec::IntoIter<u8> as Iterator>::Item"),
        ),
    ] {
        let item: ItemTrait = parse_str(&format!("trait Tr {} {{}}", generics)).unwrap();
        match (
            common::complete_generics(&item.generics, &arguments(&into_iter)),
            expected,
        ) {
            (Ok(completed), Ok(expected)) => assert_eq!(
                s(&completed[1]),
                t(parse_str(expected).unwrap()),
                "{}",
                generics
            ),
            (Err(EncodeError::AmbiguousAssociatedType(found)), Err(expected)) => {
                assert_eq!(s(&*found), t(parse_str(expected).unwrap()), "{}", generics)
            }
            (other, _) => panic!("{}: {:?}", generics, other),
        }
    }

    // No bound, several, or a binder: the error has the path as it would be written.
    for params in [
        parse_quote!(T, U = T::Item),
        parse_quote!(T: ?Sized, U = T::Item),
        parse_quote!(T: 'static, U = T::Item),
        parse_quote!(T: Clone + Iterator, U = T::Item),
        parse_quote!(T: for<'a> Tr<'a>, U = T::Item),
    ] {
        assert_eq!(
            ambiguous(params, &into_iter),
            t(parse_quote!(<::std::vec::IntoIter<u8>>::Item))
        );
    }
    // The first one in the default is reported.
    assert_eq!(
        ambiguous(parse_quote!(T, U = (T::A, T::B)), &into_iter),
        t(parse_quote!(<::std::vec::IntoIter<u8>>::A))
    );
    // The error converts into a `syn::Error`.
    let params: Punctuated<GenericParam, Token![,]> = parse_quote!(T, U = T::Item);
    let error = complete_args_of(&params, &arguments(&into_iter)).unwrap_err();
    assert_eq!(
        syn::Error::from(error).to_string(),
        "the associated type '< :: std :: vec :: IntoIter < u8 > > :: Item' needs its trait: \
         rustc resolves `<Type>::Item` only for a type parameter; write `<Type as Trait>::Item`"
    );

    // `substitute_args()` has no bounds: a value that is not an identifier is refused, and
    // nothing is replaced.
    let into = |_: &Ident| Some(parse_quote!(::std::vec::IntoIter<u8>));
    for (written, first) in [
        ("T::Item", "<::std::vec::IntoIter<u8>>::Item"),
        ("<T>::Item", "<::std::vec::IntoIter<u8>>::Item"),
        ("(<T>::A, T::B)", "<::std::vec::IntoIter<u8>>::A"),
    ] {
        let mut args: Vec<GenericArgument> = vec![parse_str(written).unwrap(), parse_quote!(T)];
        let before = args.clone();
        match substitute_args(args.iter_mut(), into) {
            Err(EncodeError::AmbiguousAssociatedType(found)) => {
                assert_eq!(s(&*found), t(parse_str(first).unwrap()), "{}", written)
            }
            other => panic!("{}: {:?}", written, other),
        }
        assert_eq!(args, before, "{}", written);
    }
    // An identifier is written, and so is the path that has its trait.
    for (written, expected, value) in [
        ("T::Item", "<I>::Item", "I"),
        ("<T>::Item", "<I>::Item", "I"),
        (
            "Option<<T as Iterator>::Item>",
            "Option<<::std::vec::IntoIter<u8> as Iterator>::Item>",
            "::std::vec::IntoIter<u8>",
        ),
        // `Self::A` in an expression is not a type: it stays `<value>::A`.
        (
            "[u8; Self::LEN]",
            "[u8; <::std::vec::IntoIter<u8>>::LEN]",
            "::std::vec::IntoIter<u8>",
        ),
    ] {
        let value: Type = parse_str(value).unwrap();
        let mut args: Vec<GenericArgument> = vec![parse_str(written).unwrap()];
        substitute_args(args.iter_mut(), |name: &Ident| {
            (name == "T" || name == "Self").then(|| GenericArgument::Type(value.clone()))
        })
        .unwrap();
        let expected: GenericArgument = parse_str(expected).unwrap();
        assert_eq!(s(&args[0]), s(&expected), "{}", written);
    }
    // A value that is not a type is a misfit, as before.
    let mut args: Vec<GenericArgument> = vec![parse_quote!(T::Item)];
    let err = substitute_args(args.iter_mut(), |_: &Ident| Some(parse_quote!('a))).unwrap_err();
    assert!(matches!(err, EncodeError::UnexpectedArgument(_)));
}

// ── F91 ──────────────────────────────────────────────────────────────────────

/// F91 (fixed): a copy wrote `where <Host>::A: Clone` as its first where predicate, which rustc
/// rejects: it reads `where <Host>` as generics of the `where` ("generic parameters on `where`
/// clauses are reserved for future use").
///
/// Trigger: `fn m(&self) -> Rel where Self::A: Clone` copied with `Visitor::with_self(Host)`, which
/// writes `<Host>::A`; or `where T::Item: Clone` of `trait Tr<T: Iterator>` written for
/// `impl<I: Iterator> Tr<I> for ..` by the `Substitute` of `Referrer::complete_args()`, which
/// writes `<I>::Item` (fixture 9254).
///
/// Cause: the Repeater's impl guarded its own where clause, but the Visitor and the `Substitute`
/// did not.
///
/// Fix: both parenthesize the first predicate of a where clause whose bounded type is a qualified
/// path without `as` (`where (<Host>::A): Clone`). Other predicates, one with a binder
/// (`for<'a> <Host>::A: Clone`) and `<Host as Tr>::A` stay, as rustc takes them.
#[test]
fn f91_a_qualified_path_at_the_start_of_a_where_predicate_is_parenthesized() {
    let projection = common::projection(parse_quote!(Repeater), 7);
    let where_of = |imp: &ItemImpl| -> Vec<String> {
        imp.items
            .iter()
            .filter_map(|item| match item {
                ImplItem::Fn(f) => Some(s(&f.sig.generics.where_clause)),
                _ => None,
            })
            .collect()
    };

    // `Visitor::with_self()`: `Self::A` becomes `<Host>::A`.
    for (written, expected) in [
        (
            "where Self::A: ::core::clone::Clone",
            "where (< Host > :: A) : :: core :: clone :: Clone",
        ),
        (
            "where Self::A: ::core::clone::Clone, u8: Copy",
            "where (< Host > :: A) : :: core :: clone :: Clone , u8 : Copy",
        ),
        // Only the first predicate is read as generics of the `where`.
        (
            "where u8: Copy, Self::A: ::core::clone::Clone",
            "where u8 : Copy , < Host > :: A : :: core :: clone :: Clone",
        ),
        (
            "where for<'a> Self::A: Tr2<'a>",
            "where for < 'a > < Host > :: A : Tr2 < 'a >",
        ),
        // A type that is not a path, a path that has its trait and a `Self` alone.
        (
            "where &'static Self::A: Send",
            "where & 'static < Host > :: A : Send",
        ),
        (
            "where <Self as Tr>::A: Send",
            "where < Host as Tr > :: A : Send",
        ),
        ("where Self: Send", "where Host : Send"),
    ] {
        let item: ItemTrait = parse_quote!(
            pub trait Tr {
                type A;
                fn m(&self) -> Rel;
            }
        );
        let mut leaker = common::std_leaker();
        let top = leaker.intern_trait(&item);
        let referrer = leaker.result().finish().unwrap();
        let method: ImplItemFn =
            parse_str(&format!("fn m(&self) -> Rel {} {{ todo!() }}", written)).unwrap();
        let mut imp: ItemImpl = parse_quote!(impl Tr for Host { type A = u8; });
        imp.items.push(ImplItem::Fn(method));
        referrer
            .visitor(top, &projection)
            .with_self(parse_quote!(Host))
            .unwrap()
            .visit_item_impl_mut(&mut imp);
        assert_eq!(where_of(&imp), [expected], "{}", written);
    }

    // `Referrer::complete_args()`: `T::Item` of the trait becomes `<I>::Item`.
    let item: ItemTrait = parse_quote!(
        pub trait Tr<T: ::core::iter::Iterator> {
            fn m(&self) -> Rel;
        }
    );
    let mut leaker = common::std_leaker();
    let top = leaker.intern_trait(&item);
    let referrer = leaker.result().finish().unwrap();
    let given: Vec<GenericArgument> = vec![parse_quote!(I)];
    for (written, expected) in [
        (
            "where T::Item: ::core::clone::Clone",
            "where (< I > :: Item) : :: core :: clone :: Clone",
        ),
        (
            "where T::Item: ::core::clone::Clone, u8: Copy",
            "where (< I > :: Item) : :: core :: clone :: Clone , u8 : Copy",
        ),
        (
            "where u8: Copy, T::Item: ::core::clone::Clone",
            "where u8 : Copy , < I > :: Item : :: core :: clone :: Clone",
        ),
        // `where <T>::Item` does not parse (as rustc, syn reads `<T>` as generics of the `where`).
        (
            "where (<T>::Item): ::core::clone::Clone",
            "where (< I > :: Item) : :: core :: clone :: Clone",
        ),
        (
            "where u8: Copy, <T>::Item: ::core::clone::Clone",
            "where u8 : Copy , < I > :: Item : :: core :: clone :: Clone",
        ),
        (
            "where for<'a> T::Item: Tr2<'a>",
            "where for < 'a > < I > :: Item : Tr2 < 'a >",
        ),
        // Already in parentheses, with its trait, and no associated type.
        (
            "where (T::Item): ::core::clone::Clone",
            "where (< I > :: Item) : :: core :: clone :: Clone",
        ),
        (
            "where <T as ::core::iter::Iterator>::Item: Send",
            "where < I as :: core :: iter :: Iterator > :: Item : Send",
        ),
        ("where T: Send", "where I : Send"),
    ] {
        let mut sub = referrer.complete_args(top, &given, &projection).unwrap();
        let mut method: ImplItemFn =
            parse_str(&format!("fn m(&self) -> Rel {} {{ todo!() }}", written)).unwrap();
        sub.visit_impl_item_fn_mut(&mut method);
        assert_eq!(
            s(&method.sig.generics.where_clause),
            expected,
            "{}",
            written
        );
    }
}

// ── F92 ──────────────────────────────────────────────────────────────────────

/// F92 (fixed): `Referrer::complete_args()` wrote a trait object with several bounds behind a
/// reference or a pointer without parentheses, where `substitute_args()` writes them.
///
/// Trigger: `trait Tr<T: ?Sized, U = &'static T>` with `dyn A + Send` for `T` and `U` left out: the
/// default became `&'static dyn A + Send`, "ambiguous `+` in a type", and the copy had it
/// (fixture 9255).
///
/// Cause: the completion did not parenthesize the pointee, where `substitute_args()` does.
///
/// Fix: the completion does the same; a trait object with one bound needs none.
#[test]
fn f92_complete_args_parenthesizes_a_trait_object_with_several_bounds_behind_a_pointer() {
    let only_t: Punctuated<GenericParam, Token![,]> = parse_quote!(T: ?Sized);
    for (written, argument, expected) in [
        // The trigger.
        ("&'static T", "dyn A + Send", "& 'static (dyn A + Send)"),
        (
            "&'static mut T",
            "dyn A + Send",
            "& 'static mut (dyn A + Send)",
        ),
        ("*const T", "dyn A + Send", "* const (dyn A + Send)"),
        ("*mut T", "dyn A + Send", "* mut (dyn A + Send)"),
        (
            "&'static T",
            "dyn A + 'static",
            "& 'static (dyn A + 'static)",
        ),
        // Inside other types.
        (
            "Box<&'static T>",
            "dyn A + Send",
            "Box < & 'static (dyn A + Send) >",
        ),
        (
            "&'static &'static T",
            "dyn A + Send",
            "& 'static & 'static (dyn A + Send)",
        ),
        (
            "(&'static T, u8)",
            "dyn A + Send",
            "(& 'static (dyn A + Send) , u8)",
        ),
        // One bound needs none, and a value that is not behind a pointer is as it is.
        ("&'static T", "dyn A", "& 'static dyn A"),
        ("Box<T>", "dyn A + Send", "Box < dyn A + Send >"),
        ("T", "dyn A + Send", "dyn A + Send"),
    ] {
        let default: Type = parse_str(written).unwrap();
        let params: Punctuated<GenericParam, Token![,]> = parse_quote!(T: ?Sized, U = #default);
        let site = arguments(&[argument]);
        let completed = complete_args_of(&params, &site).unwrap();
        assert_eq!(s(&completed[1]), expected, "{} with {}", written, argument);
        // The same as `substitute_args()` writes for the default.
        let substituted =
            common::substitute_type(&default, common::answers(&only_t, &site), None).unwrap();
        assert_eq!(
            s(&completed[1]),
            s(&substituted),
            "{} with {}",
            written,
            argument
        );
    }
}

// ── F93 ──────────────────────────────────────────────────────────────────────

/// F93 (fixed): the `Debug` text of `Substitute`, `Referrer`, `LeakerResult` and `Visitor` differed
/// from run to run for the same value.
///
/// Trigger: `format!("{:?}", value)` of a value built twice from the same input (many parameters,
/// entries or errors).
///
/// Cause: some fields are `HashMap`s and `HashSet`s, and the derived `Debug` prints them in the
/// order they iterate, which differs from map to map.
///
/// Fix: those fields are wrapped in a crate-private type whose `Debug` lists the entries sorted by
/// text. Lookups and the public API do not change.
#[test]
fn f93_the_debug_of_the_public_types_is_the_same_in_every_run() {
    // The same value 20 times, from the same input, in this process. The names of the types that
    // print differently are kept, to report them all.
    let mut differ: Vec<&str> = Vec::new();
    let mut same = |name: &'static str, print: &dyn Fn() -> String| {
        let first = print();
        // It has what the test is about: many keys.
        assert!(first.matches("Ident(").count() >= 20, "{}: {}", name, first);
        if (1..20).any(|_| print() != first) {
            differ.push(name);
        }
    };

    // A `Substitute` with many parameters, which have bounds (`bounds`), lifetimes, types and a
    // const.
    let generics: Generics = parse_quote!(
        <'a, 'b, 'c, A: Iterator, B: Clone, C: Send, D: Sync, E: Copy, F: Sized, G, H, I,
            const N: usize>
    );
    let given: Vec<GenericArgument> = arguments(&[
        "'static", "'static", "'static", "IA", "IB", "IC", "ID", "IE", "IF", "IG", "IH", "II", "3",
    ]);
    same("Substitute", &|| {
        format!("{:?}", common::substitute_for(&generics, &given).unwrap())
    });

    // A trait with many entries and many errors (a relative trait in each bound).
    let build = || {
        let item: ItemTrait = parse_quote!(
            trait Tr<'a, T> {
                fn f<
                    A: RelA,
                    B: RelB,
                    C: RelC,
                    D: RelD,
                    E: RelE,
                    F: RelF,
                    G: RelG,
                    H: RelH,
                    I: RelI,
                >(
                    &self,
                    a: A,
                    b: B,
                    c: C,
                    d: D,
                    e: E,
                    f: F,
                    g: G,
                    h: H,
                    i: I,
                ) -> Wrap<T>;
                fn g(
                    &self,
                    a: Rel0,
                    b: Rel1,
                    c: Rel2,
                    d: Rel3,
                    e: Rel4,
                    f: Rel5,
                    g: Rel6,
                    h: Rel7,
                    i: Rel8,
                ) -> Wrap<&'a T>;
            }
        );
        let mut leaker = common::std_leaker();
        let top = leaker.intern_trait(&item);
        (leaker.result(), top)
    };
    same("LeakerResult", &|| {
        let (result, _) = build();
        assert_eq!(result.errors().len(), 9);
        format!("{:?}", result)
    });
    // The Referrer of the errors that are allowed: each one is a part.
    let allowed = || {
        let (result, top) = build();
        let referrer = result.allow(|_: &Cause, _: Place| true).finish().unwrap();
        (referrer, top)
    };
    same("Referrer", &|| format!("{:?}", allowed().0));
    // The Visitor prints its Referrer.
    let projection = common::projection(parse_quote!(Repeater), 7);
    same("Visitor", &|| {
        let (referrer, top) = allowed();
        format!("{:?}", referrer.visitor(top, &projection))
    });

    assert!(
        differ.is_empty(),
        "the Debug text of the same input differs between builds: {:?}",
        differ
    );
}

// ── F95 ──────────────────────────────────────────────────────────────────────

/// F95 (fixed): a conflict that holds a bare identifier (`N`, which may be a const) was copied with
/// its parts leaked, where the same rule given to the Leaker copies it whole (`MaybeConst`).
///
/// Trigger: `struct S(Rel<N, [u8; { let _f = || 0; 3 }]>)` (the closure is a hard cause, `Rel`
/// the type's own) or `Rel<N, Self::A>` in an impl, with `Leaker::new()`, then `allow()` of every
/// cause and `finish()`: `N` was an entry, so the Repeater's impl wrote a const as a type (E0573)
/// and the copy wrote a type where rustc may expect a const (E0747). Fixture 9256.
///
/// Cause: the search for bare identifiers stopped at a type that has a cause of its own, so the
/// conflict was not made whole.
///
/// Fix: such a conflict is copied whole, with the identifier among the paths of its error, so both
/// stages give the same Referrer.
#[test]
fn f95_a_conflict_that_holds_a_bare_identifier_is_copied_whole() {
    let items: Vec<Item> = vec![
        parse_quote!(
            struct S(
                Rel<
                    N,
                    [u8; {
                        let _f = || 0;
                        3
                    }],
                >,
            );
        ),
        parse_quote!(impl Tr for X { type A = u8; fn f(&self, _x: Rel<N, Self::A>) {} }),
    ];
    for item in &items {
        let case = s(item);
        // The rule given to the Leaker: `N` may be a const, so the type is copied whole.
        let mut early = Leaker::with_rule(|cause: &Cause, _: Place| {
            cause
                .path()
                .map_or(false, |path| path.is_ident("Rel") || path.is_ident("Tr"))
        });
        common::intern_item(&mut early, item);
        let early = early.result();
        assert_eq!(
            early.errors().iter().map(|e| e.reason).collect::<Vec<_>>(),
            [Reason::MaybeConst],
            "{}",
            case
        );
        // The rule given to the result: the type has errors of its own, and `N` is among the paths
        // of the one of the part.
        let mut late = Leaker::new();
        common::intern_item(&mut late, item);
        let late = late.result();
        assert!(
            late.errors().iter().any(|e| e
                .paths
                .iter()
                .any(|path| path.path().map_or(false, |path| path.is_ident("N")))),
            "{}: {}",
            case,
            messages(late.errors())
        );
        let (early, late) = (early.recovered(), late.recovered());
        assert!(
            !entries_text(&late, true).contains(&"N".to_string()),
            "{}",
            case
        );
        assert!(
            parts_of(&late).iter().all(|part| part.ends_with("whole")),
            "{}: {:?}",
            case,
            parts_of(&late)
        );
        assert_eq!(late, early, "{}", case);
    }
}

// ── F94 ──────────────────────────────────────────────────────────────────────

/// F94 (fixed): `Visitor::with_self()` refused a host that is a trait object without a lifetime
/// bound (`dyn Shape`) because of an entry that the host is never written for.
///
/// Trigger: one Leaker analyzes `trait Borrowing<'a> where Self: 'a { fn get(&self) -> Ref<'a,
/// Self>; }` and `trait Plain { fn name(&self) -> Name<Self>; }`; `visitor(plain, ..)
/// .with_self(dyn Shape)` was `Err(UnboundedTraitObject)`. The same for an item nested in the
/// visited one (`trait Inner<'a> where Self: 'a`, also in a method body).
///
/// Cause: `with_self()` checked every live entry, but the host is written only for the entries of
/// the visited item's `Self`; another item, or a nested one, has a `Self` of its own.
///
/// Fix: `with_self()` checks only the entries whose `self_ty` is the visited context's; a scope
/// with none (`enter_generics()`) checks no entry. An entry of the visited `Self` with `Self: 'a`
/// still refuses `dyn Shape`, and `dyn Shape + 'static` fits.
#[test]
fn f94_with_self_checks_only_the_entries_of_the_visited_self() {
    let projection = common::projection(parse_quote!(Repeater), 7);
    let bare: Type = parse_quote!(dyn Shape);
    let bounded: Type = parse_quote!(dyn Shape + 'static);
    let refused = |referrer: &Referrer, context: ContextId, host: &Type| match referrer
        .visitor(context, &projection)
        .with_self(host.clone())
    {
        Ok(_) => None,
        Err(EncodeError::UnboundedTraitObject(ty)) => Some(*ty),
        Err(error) => panic!("another error: {:?}", error),
    };
    // The copy of `item` that the visitor writes with the host.
    let written = |referrer: &Referrer, context: ContextId, host: &Type, mut item: Item| {
        referrer
            .visitor(context, &projection)
            .with_self(host.clone())
            .unwrap()
            .visit_item_mut(&mut item);
        s(&item)
    };
    // Some entry with `Self: 'a` in its context is in the Referrer.
    let has_outlives = |referrer: &Referrer, ty: &str| {
        referrer
            .iter()
            .any(|(_, t, context)| s(t) == ty && !context.self_outlives.is_empty())
    };

    // (1) An entry of another top-level item of the same Leaker.
    let borrowing: ItemTrait = parse_quote!(
        pub trait Borrowing<'a>
        where
            Self: 'a,
        {
            fn get(&self) -> Ref<'a, Self>;
        }
    );
    let plain: ItemTrait = parse_quote!(
        pub trait Plain {
            fn name(&self) -> Name<Self>;
        }
    );
    let mut leaker = Leaker::new();
    let borrowing_id = leaker.intern_trait(&borrowing);
    let plain_id = leaker.intern_trait(&plain);
    let referrer = leaker.result().finish().unwrap();
    assert!(has_outlives(&referrer, "Ref < 'a , Self >"));
    let key = projection.key(common::entry_id_at(
        &referrer,
        entry_index(&referrer, "Name < Self >"),
    ));
    for host in [&bare, &bounded] {
        assert_eq!(refused(&referrer, plain_id, host), None, "{}", s(host));
        let expected: ItemTrait = parse_quote!(
            pub trait Plain {
                fn name(&self) -> <#host as Repeater<(), #key>>::Type;
            }
        );
        assert_eq!(
            written(&referrer, plain_id, host, Item::Trait(plain.clone())),
            s(&expected),
            "{}",
            s(host)
        );
    }
    // The entry of the visited item itself still refuses the host without a bound.
    assert_eq!(refused(&referrer, borrowing_id, &bare), Some(bare.clone()));
    assert_eq!(refused(&referrer, borrowing_id, &bounded), None);

    // (2) An entry of an item nested in the visited syntax: it has a `Self` of its own.
    let inner: Item = parse_quote!(
        trait Inner<'a>
        where
            Self: 'a,
        {
            fn f(&self) -> Rel<'a, Self>;
        }
    );
    let mut leaker = Leaker::new();
    let (scope, ()) = leaker.enter_generics(&Generics::default(), |view| view.visit_item(&inner));
    let referrer = leaker.result().finish().unwrap();
    assert!(has_outlives(&referrer, "Rel < 'a , Self >"));
    for host in [&bare, &bounded] {
        assert_eq!(refused(&referrer, scope, host), None, "{}", s(host));
    }
    // Inside an impl, whose context has a `Self` of its own: the nested trait is a nested item, the
    // host is not written in it and its entry stays as written.
    let mut leaker = Leaker::new();
    let (_, context) = leaker.enter_generics(&Generics::default(), |view| {
        view.enter_impl(&Generics::default(), &parse_quote!(::p::W), None, |view| {
            view.visit_item(&inner)
        })
        .0
    });
    let referrer = leaker.result().finish().unwrap();
    assert!(has_outlives(&referrer, "Rel < 'a , Self >"));
    let imp: ItemImpl = parse_quote!(
        impl ::p::W {
            fn m(&self) {
                trait Inner<'a>
                where
                    Self: 'a,
                {
                    fn f(&self) -> Rel<'a, Self>;
                }
            }
        }
    );
    for host in [&bare, &bounded] {
        assert_eq!(refused(&referrer, context, host), None, "{}", s(host));
        assert_eq!(
            written(&referrer, context, host, Item::Impl(imp.clone())),
            s(&imp),
            "{}",
            s(host)
        );
    }

    // The same in a trait: the item nested in the body of a method has a `Self` of its own.
    let mut leaker = Leaker::new();
    let (outer_id, ()) = leaker.enter_trait(&parse_quote!(Outer), &Generics::default(), |view| {
        view.visit_type(&parse_quote!(Name<Self>));
        view.visit_item(&inner);
    });
    let referrer = leaker.result().finish().unwrap();
    assert!(has_outlives(&referrer, "Rel < 'a , Self >"));
    let key = projection.key(common::entry_id_at(
        &referrer,
        entry_index(&referrer, "Name < Self >"),
    ));
    for host in [&bare, &bounded] {
        assert_eq!(refused(&referrer, outer_id, host), None, "{}", s(host));
        let expected: ItemTrait = parse_quote!(
            pub trait Outer {
                fn name(&self) -> <#host as Repeater<(), #key>>::Type;
                fn make(&self) {
                    trait Inner<'a>
                    where
                        Self: 'a,
                    {
                        fn f(&self) -> Rel<'a, Self>;
                    }
                }
            }
        );
        let outer: ItemTrait = parse_quote!(
            pub trait Outer {
                fn name(&self) -> Name<Self>;
                fn make(&self) {
                    trait Inner<'a>
                    where
                        Self: 'a,
                    {
                        fn f(&self) -> Rel<'a, Self>;
                    }
                }
            }
        );
        assert_eq!(
            written(&referrer, outer_id, host, Item::Trait(outer)),
            s(&expected),
            "{}",
            s(host)
        );
    }

    // A scope with no `Self` has no entry of the visited `Self`: always Ok.
    let mut leaker = Leaker::new();
    leaker.intern_trait(&borrowing);
    let (scope, ()) = leaker.enter_generics(&Generics::default(), |view| {
        view.visit_type(&parse_quote!(Rel<u8>))
    });
    let referrer = leaker.result().finish().unwrap();
    assert_eq!(refused(&referrer, scope, &bare), None);
}

// ── F101 ─────────────────────────────────────────────────────────────────────

/// F101 (fixed): an item in a block in an array length is part of the type, whatever
/// `Leaker::check_values` says.
///
/// Trigger: `struct S<T>([Rel<T>; { const X: usize = rel::N; 3 }])`. With `check_values` off the
/// value of `X` was not read, so `Rel<T>` alone was the entry and the copy wrote `rel::N`
/// unresolved (E0425); with it on it was the error `RelativePath`. The same for a `static`, a `fn`,
/// a nested trait's defaults, an enum discriminant and the default of a const parameter.
///
/// Cause: the value of a nested item was read with the walk of a value, which needs `check_values`
/// and rejects relative paths, though a type that holds one cannot be rewritten: it must be an
/// entry whole, as with `{ let _x = rel::N; 3 }`.
///
/// Fix: in a type the value or body is read as the rest of the length is. Outside a type
/// `check_values` keeps its meaning. A closure, an assignment or a macro in a nested body that
/// names a relative path is still an error, as in a `let`.
#[test]
fn f101_a_value_in_a_block_in_an_array_length_is_part_of_the_type() {
    let found = |check_values: bool, array: &Type| {
        let item: Item = parse_quote!(struct S<T>(#array););
        let mut leaker = Leaker::with_rule(Rules::new().primitive());
        leaker.check_values = check_values;
        leaker.check_bodies = check_values;
        common::intern_item(&mut leaker, &item);
        let result = leaker.result();
        let errors: Vec<(Reason, String)> = (result.errors().iter())
            .map(|error| (error.reason, s(&error.cause)))
            .collect();
        (errors, entries_text(&result.recovered(), false))
    };
    let whole = |array: &Type| vec![s(array).replace(' ', "")];
    // The blocks that name `rel::N` in a value: the leaked array is the same as for the `let`.
    let blocks: Vec<(&str, Block)> = vec![
        (
            "let",
            parse_quote!({
                let _x = rel::N;
                3
            }),
        ),
        (
            "const",
            parse_quote!({
                const X: usize = rel::N;
                3
            }),
        ),
        (
            "static",
            parse_quote!({
                static X: usize = rel::N;
                3
            }),
        ),
        (
            "fn",
            parse_quote!({
                fn f() -> usize {
                    rel::N
                }
                3
            }),
        ),
        (
            "fn with a parameter and a generic",
            parse_quote!({
                fn f<U>(a: U) -> usize {
                    rel::g(a)
                }
                3
            }),
        ),
        (
            "default of an associated const and of a method",
            parse_quote!({
                trait Z {
                    const A: usize = rel::N;
                    fn f() -> usize {
                        rel::N
                    }
                }
                3
            }),
        ),
        (
            "inherent impl",
            parse_quote!({
                struct Z;
                impl Z {
                    const A: usize = rel::N;
                }
                3
            }),
        ),
        (
            "discriminant",
            parse_quote!({
                enum E {
                    A = rel::N,
                }
                3
            }),
        ),
        (
            "default of a const parameter",
            parse_quote!({
                struct Z<const N: usize = { rel::N }>;
                3
            }),
        ),
        (
            "cast to a relative type",
            parse_quote!({
                const X: usize = 1u8 as Alias;
                3
            }),
        ),
    ];
    let absolute: Vec<Block> = vec![
        parse_quote!({
            const X: usize = ::core::mem::size_of::<u8>();
            3
        }),
        // The names that the block declares (F110).
        parse_quote!({
            const X: usize = 3;
            const Y: usize = X;
            3
        }),
        parse_quote!({
            fn f() -> usize {
                ::core::mem::size_of::<T>()
            }
            3
        }),
        parse_quote!({
            fn f() -> usize {
                ::core::unimplemented!()
            }
            3
        }),
    ];
    let closures: Vec<Block> = vec![
        parse_quote!({
            let c = |a: u8| a;
            3
        }),
        parse_quote!({
            fn f() -> usize {
                let c = || 1;
                c()
            }
            3
        }),
    ];
    for check_values in [false, true] {
        for (name, block) in &blocks {
            let array: Type = parse_quote!([Rel<T>; #block]);
            assert_eq!(
                found(check_values, &array),
                (vec![], whole(&array)),
                "{}, check_values {}",
                name,
                check_values
            );
        }
        // The values that name nothing relative leave `Rel<T>` alone the entry.
        for block in &absolute {
            let array: Type = parse_quote!([Rel<T>; #block]);
            assert_eq!(
                found(check_values, &array),
                (vec![], vec!["Rel<T>".to_string()]),
                "{}, check_values {}",
                s(block),
                check_values
            );
        }
        // A closure cannot be copied by the entry, but these name nothing relative: the parameter
        // `a` and the local `c` are the block's own (F110), so `Rel<T>` alone is the entry.
        for block in &closures {
            let array: Type = parse_quote!([Rel<T>; #block]);
            assert_eq!(
                found(check_values, &array),
                (vec![], vec!["Rel<T>".to_string()]),
                "{}, check_values {}",
                s(block),
                check_values
            );
        }
    }

    // Outside a type `check_values` keeps its meaning: the default of an associated const and a
    // default method of a trait, and the value and the body of a `const` and a `fn` that a View
    // visits, are read with it (the error is in a value) and not without it.
    let trait_item: Item = parse_quote!(
        trait Tr {
            const A: usize = rel::N;
            fn f(&self) -> usize {
                rel::N
            }
        }
    );
    let items: Vec<(&str, Item, usize)> = vec![
        ("trait", trait_item, 2),
        (
            "const",
            parse_quote!(
                const X: usize = rel::N;
            ),
            1,
        ),
        (
            "static",
            parse_quote!(
                static X: usize = rel::N;
            ),
            1,
        ),
        (
            "fn",
            parse_quote!(
                fn f() -> usize {
                    rel::N
                }
            ),
            1,
        ),
    ];
    for (name, item, count) in &items {
        for check_values in [false, true] {
            let mut leaker = Leaker::with_rule(Rules::new().primitive());
            leaker.check_values = check_values;
            leaker.check_bodies = check_values;
            match item {
                Item::Trait(item) => {
                    leaker.intern_trait(item);
                }
                _ => {
                    leaker.enter_generics(&Generics::default(), |view| view.visit_item(item));
                }
            }
            let result = leaker.result();
            let errors: Vec<(Reason, String, bool)> = (result.errors().iter())
                .map(|error| (error.reason, s(&error.cause), error.place.in_value))
                .collect();
            let expected = if check_values {
                vec![(Reason::RelativePath, "rel :: N".to_string(), true); *count]
            } else {
                vec![]
            };
            assert_eq!(errors, expected, "{}, check_values {}", name, check_values);
            assert_eq!(result.recovered().iter().count(), 0, "{}", name);
        }
    }
}

// ── F96 ──────────────────────────────────────────────────────────────────────

/// F96 (fixed): a const argument that starts a statement or a match arm was written as a block that
/// is the left operand of an operator, a cast or a call (`{ 3 + 0 } * 2`), which rustc reads as a
/// statement that ends at the block (E0308, E0614).
///
/// Trigger: `Referrer::complete_args()` for `trait Tr<const N: usize>` with `{ 3 + 0 }`, then
/// `Substitute` over `impl Tr<{ 3 + 0 }> for Host { fn len(&self) -> usize { N * 2 } }`. The same
/// for `N.pow(2)`, `N as u64`, and `substitute_args()` on `{ N * 2 }`.
///
/// Cause: a const is written as the block `{ value }` wherever its name is an expression, and `syn`
/// before 2.0.67 prints `Binary(Block, *, 2)` without parentheses.
///
/// Fix: a block written as the leftmost operand of an expression that starts a statement or an arm
/// gets parentheses (`({ 3 + 0 }) * 2`). A block that was there stays, and so does a const in
/// another place (`let x = { 3 } * 2;`). An arm visited by itself is not covered.
#[test]
fn f96_a_const_that_starts_a_statement_or_an_arm_is_written_in_parentheses() {
    let generics: Generics = parse_quote!(<const N: usize>);
    let given = arguments(&["{ 3 + 0 }"]);
    let impl_with = |body: &str| -> ItemImpl {
        parse_str(&format!(
            "impl Tr<{{ 3 + 0 }}> for Host {{ fn len(&self) -> usize {{ {} }} }}",
            body
        ))
        .unwrap()
    };
    for (written, expected) in [
        // The trigger, and the other forms that go on from the operand.
        ("N * 2", "({ 3 + 0 }) * 2"),
        ("N.pow(2)", "({ 3 + 0 }).pow(2)"),
        ("N as usize", "({ 3 + 0 }) as usize"),
        ("N + 1 + 2", "({ 3 + 0 }) + 1 + 2"),
        ("N[0]", "({ 3 + 0 })[0]"),
        ("N..5", "({ 3 + 0 })..5"),
        ("N.0.pow(2)", "({ 3 + 0 }).0.pow(2)"),
        ("N * 2; 0", "({ 3 + 0 }) * 2; 0"),
        ("let x = 1; N * x", "let x = 1; ({ 3 + 0 }) * x"),
        // In a match arm, and in the statements of a nested block.
        (
            "match 1 { _ => N * 2, _ => 0 }",
            "match 1 { _ => ({ 3 + 0 }) * 2, _ => 0 }",
        ),
        (
            "if true { N * 2 } else { 0 }",
            "if true { ({ 3 + 0 }) * 2 } else { 0 }",
        ),
        (
            "let f = || { N * 2 }; f()",
            "let f = || { ({ 3 + 0 }) * 2 }; f()",
        ),
        // A const alone, or where an expression goes on from something else, is as before.
        ("N", "{ 3 + 0 }"),
        ("1 + N", "1 + { 3 + 0 }"),
        ("let x = N * 2; x", "let x = { 3 + 0 } * 2; x"),
        ("[0u8; N].len()", "[0u8; { 3 + 0 }].len()"),
        ("(N).pow(2)", "({ 3 + 0 }).pow(2)"),
        ("0 + N * 2", "0 + { 3 + 0 } * 2"),
        // A block of the item stays as it is.
        ("{ 1 }.pow(2)", "{ 1 }.pow(2)"),
    ] {
        let mut substitute = common::substitute_for(&generics, &given).unwrap();
        let mut imp = impl_with(written);
        substitute.visit_item_impl_mut(&mut imp);
        assert_eq!(imp, impl_with(expected), "{}", written);
    }

    // The tree of the trigger: the left operand of the tail is no block.
    let mut substitute = common::substitute_for(&generics, &given).unwrap();
    let mut imp = impl_with("N * 2");
    substitute.visit_item_impl_mut(&mut imp);
    let tail = match &imp.items[0] {
        ImplItem::Fn(method) => method.block.stmts.last().cloned(),
        _ => None,
    };
    match tail {
        Some(Stmt::Expr(Expr::Binary(binary), None)) => {
            assert!(
                matches!(&*binary.left, Expr::Paren(paren) if matches!(&*paren.expr, Expr::Block(_))),
                "{}",
                s(&imp)
            );
        }
        other => panic!("not a binary tail: {:?}", other.as_ref().map(s)),
    }

    // A statement that is visited alone.
    let mut substitute = common::substitute_for(&generics, &given).unwrap();
    let mut stmt: Stmt = parse_quote!(N * 2;);
    substitute.visit_stmt_mut(&mut stmt);
    assert_eq!(stmt, parse_quote!(({ 3 + 0 }) * 2;));

    // `substitute_args()` on a constant expression with a block: its statements.
    for (written, expected) in [
        ("{ N * 2 }", "{ ({ 3 }) * 2 }"),
        ("{ N.pow(2) }", "{ ({ 3 }).pow(2) }"),
        ("{ N as u64 }", "{ ({ 3 }) as u64 }"),
        ("{ 1 + N }", "{ 1 + { 3 } }"),
        ("{ N }", "{ 3 }"),
    ] {
        let mut args = arguments(&[written]);
        substitute_args(args.iter_mut(), |name: &Ident| {
            (name == "N").then(|| parse_quote!(3))
        })
        .unwrap();
        assert_eq!(args, arguments(&[expected]), "{}", written);
    }
}

// ── F97 ──────────────────────────────────────────────────────────────────────

/// F97 (fixed): the head of a path in an expression was substituted only for `Self` and in the
/// qualified form: `T::LEN` and `T::default()` kept the parameter, which the copy does not declare
/// (E0433), where `<T>::LEN` was written.
///
/// Trigger: `Referrer::complete_args()` for `trait Tr<K: Default>` with `u8`, then `Substitute`
/// over `fn make(&self) -> K { let a: K = K::default(); a }` gave `let a: u8 = K::default();`.
/// `substitute_args()` on `{ T::LEN }` kept `T::LEN` the same way.
///
/// Cause: a path in an expression was written only when it started with `Self`; `T::f` is no type,
/// so the substitution of types does not see it.
///
/// Fix: the first segment of a path in an expression whose answer is a type is written
/// `<value>::f`. An answer that is no type leaves the path as written and is no error (`a::f()` is
/// a module path, and `a` may name a lifetime), except for `Self`.
#[test]
fn f97_the_head_of_a_path_in_an_expression_is_substituted() {
    // The trigger.
    let generics: Generics = parse_quote!(<K: ::core::default::Default>);
    let given = arguments(&["u8"]);
    let impl_with = |body: &str| -> ItemImpl {
        parse_str(&format!(
            "impl Tr<u8> for Host {{ fn make(&self) -> u8 {{ {} }} }}",
            body
        ))
        .unwrap()
    };
    for (written, expected) in [
        (
            "let a: K = K::default(); a",
            "let a: u8 = <u8>::default(); a",
        ),
        // `<K>::f` and `Self::f` were written before.
        ("<K>::default()", "<u8>::default()"),
        ("Self::default()", "Self::default()"),
        // The rest of the path, with arguments that name the parameter.
        ("K::default::<K>()", "<u8>::default::<u8>()"),
        ("K::A::B", "<u8>::A::B"),
        ("[K::default(); K::LEN]", "[<u8>::default(); <u8>::LEN]"),
        ("K::default().max(K::MIN)", "<u8>::default().max(<u8>::MIN)"),
        // A path that is not headed by a parameter, or has arguments on its head, stays.
        ("std::mem::drop(1)", "std::mem::drop(1)"),
        ("Vec::<K>::new()", "Vec::<u8>::new()"),
        ("::K::default()", "::K::default()"),
        ("a::f()", "a::f()"),
    ] {
        let mut substitute = common::substitute_for(&generics, &given).unwrap();
        let mut imp = impl_with(written);
        substitute.visit_item_impl_mut(&mut imp);
        assert_eq!(imp, impl_with(expected), "{}", written);
    }

    // The same for `substitute_args()`, with `Self`, and a module that has the name of a lifetime.
    let site = |name: &Ident| -> Option<GenericArgument> {
        match name.to_string().as_str() {
            "T" => Some(parse_quote!(U)),
            "V" => Some(parse_quote!(Vec<u8>)),
            "N" => Some(parse_quote!(3)),
            "Self" => Some(parse_quote!(Host)),
            "a" => Some(parse_quote!('static)),
            _ => None,
        }
    };
    for (written, expected) in [
        ("{ T::LEN }", "{ <U>::LEN }"),
        ("{ <T>::LEN }", "{ <U>::LEN }"),
        ("{ Self::LEN }", "{ <Host>::LEN }"),
        ("[u8; T::LEN]", "[u8; <U>::LEN]"),
        ("{ T::f::<T, N>() }", "{ <U>::f::<U, 3>() }"),
        // A type that is not an identifier is written the same.
        ("{ V::LEN }", "{ <Vec<u8>>::LEN }"),
        ("{ m!(T::LEN) }", "{ m!(T::LEN) }"),
        // A lifetime answer for `a` is no answer for the module `a`; nothing is an error.
        ("{ a::f() }", "{ a::f() }"),
        (
            "{ std::mem::size_of::<T>() }",
            "{ std::mem::size_of::<U>() }",
        ),
        ("{ crate::LEN }", "{ crate::LEN }"),
    ] {
        let mut args = arguments(&[written]);
        substitute_args(args.iter_mut(), site).unwrap();
        assert_eq!(args, arguments(&[expected]), "{}", written);
    }

    // A value that is no type for a head is not an error; for `Self` it is, as it was.
    let mut args = arguments(&["{ T::LEN }"]);
    substitute_args(args.iter_mut(), |name: &Ident| {
        (name == "T").then(|| parse_quote!(3))
    })
    .unwrap();
    assert_eq!(args, arguments(&["{ T::LEN }"]));
    let mut args = arguments(&["{ Self::LEN }"]);
    let err = substitute_args(args.iter_mut(), |name: &Ident| {
        (name == "Self").then(|| parse_quote!(3))
    })
    .unwrap_err();
    assert!(
        matches!(err, EncodeError::UnexpectedArgument(_)),
        "{:?}",
        err
    );
}

// ── F98 ──────────────────────────────────────────────────────────────────────

/// F98 (fixed): the help of a relative path wrote an absolute example that disagreed with other
/// helps: `crate::m!` gave `::my_crate::crate::m!`, `std::line!` gave `::my_crate::std::line!`, and
/// `Self::Output` in `impl core::ops::Add` gave `<Self as ::my_crate::core::ops::Add>::Output`.
///
/// Trigger: `check_values` with `crate::m!(1)`, `super::m!()`, `std::line!()` or `include!("x.rs")`
/// in the value of a const, and
/// `impl<T: core::ops::Add<Output = Self::Output>> core::ops::Add for W<Rel<T>>`.
///
/// Cause: three helps wrote the example, each its own way.
///
/// Fix: one function writes it: `::{path}` for a path of two segments or more under `std`, `core`
/// or `alloc`, else `::my_crate::{path}` without `crate`, `self` and `super`. `include!` and
/// `is_x86_feature_detected!` are in the table of the standard macros.
#[test]
fn f98_the_example_of_an_absolute_path_is_the_same_for_every_help() {
    let help_of = |value: &str| -> Vec<String> {
        let item: ItemImpl =
            parse_str(&format!("impl S {{ const C: usize = {}; }}", value)).unwrap();
        let mut leaker = Leaker::new();
        leaker.check_values = true;
        leaker.intern_impl(&item);
        all_helps(leaker.result().errors())
    };
    for (value, help) in [
        (
            "crate::m!(1)",
            "write `crate::m!` with an absolute path (e.g. `::my_crate::m!`)",
        ),
        (
            "super::m!()",
            "write `super::m!` with an absolute path (e.g. `::my_crate::m!`)",
        ),
        (
            "m::n!()",
            "write `m::n!` with an absolute path (e.g. `::my_crate::m::n!`)",
        ),
        (
            "std::line!()",
            "write `std::line!` with an absolute path (e.g. `::std::line!`)",
        ),
        (
            "core::line!()",
            "write `core::line!` with an absolute path (e.g. `::core::line!`)",
        ),
        (
            "include!(\"x.rs\")",
            "use `::core::include!` instead of `include!`",
        ),
        (
            "is_x86_feature_detected!(\"avx2\")",
            "use `::std::is_x86_feature_detected!` instead of `is_x86_feature_detected!`",
        ),
    ] {
        // The macro is the one error, and an `include!` is no format string.
        assert_eq!(help_of(value), [help], "{}", value);
    }
    // A path of the trait of an impl, which `Self::Output` is replaced through.
    let item: ItemImpl = parse_quote!(
        impl<T: core::ops::Add<Output = Self::Output>> core::ops::Add for W<Rel<T>> {
            type Output = u8;
            fn add(self, other: Self) -> Self::Output {
                0
            }
        }
    );
    let mut leaker = Leaker::new();
    leaker.intern_impl(&item);
    let helps = all_helps(leaker.result().errors());
    assert!(
        helps.contains(
            &"replace `Self::Output` with `<Self as ::core::ops::Add>::Output`, with \
              `core::ops::Add` written as an absolute path"
                .to_string()
        ),
        "{:?}",
        helps
    );
    assert!(
        helps.iter().all(|help| !help.contains("::my_crate::core")),
        "{:?}",
        helps
    );
}

// ── F99 ──────────────────────────────────────────────────────────────────────

/// F99 (fixed): the help and the span of `SelfInImplSelfType` blamed the first `Self` in the where
/// clause of the impl, not the `Self: 'a` that ties the entry to `Self`.
///
/// Trigger: `impl<'a, T, U> Tr<U> for W<'a, Rel<'a, T>> where U: Into<Self::Y>, Self: 'a {}`. The
/// context of `Rel<'a, T>` has no `U`, so `U: Into<Self::Y>` is not what makes its context depend
/// on `Self`: `Self: 'a` is. The help said "or the bound of its generics without `Self::Y`".
///
/// Cause: a `Self: 'a` that the context has moves into `self_outlives`, so the search for the
/// mention of `Self` fell back to the first `Self` anywhere in the where clause.
///
/// Fix: the fallback looks only at the predicates `Self: 'a` whose lifetime is in `self_outlives`.
/// The span (the `Self` of that predicate) is checked in `tests/crates/spans`.
#[test]
fn f99_the_self_that_ties_the_entry_is_the_one_in_the_outlives_predicate() {
    let item: ItemImpl = parse_quote!(
        impl<'a, T, U> Tr<U> for W<'a, Rel<'a, T>>
        where
            U: Into<Self::Y>,
            Self: 'a,
        {
        }
    );
    let mut leaker = Leaker::new();
    leaker.intern_impl(&item);
    let result = leaker.result();
    let errors: Vec<&ErrorItem> = result
        .errors()
        .iter()
        .filter(|e| e.reason == Reason::SelfInImplSelfType)
        .collect();
    assert!(!errors.is_empty(), "{}", messages(result.errors()));
    for error in errors {
        let help = error.help_text().unwrap();
        assert!(
            help.ends_with("or the bound of its generics without `Self`"),
            "{}",
            help
        );
    }
    // Without `Self: 'a` the bound of `U` is not tied to the entry either, so there is no error.
    let item: ItemImpl = parse_quote!(
        impl<'a, T, U> Tr<U> for W<'a, Rel<'a, T>> where U: Into<Self::Y> {}
    );
    let mut leaker = Leaker::new();
    leaker.intern_impl(&item);
    assert!(leaker
        .result()
        .errors()
        .iter()
        .all(|e| e.reason != Reason::SelfInImplSelfType),);
}

// ── F100 ─────────────────────────────────────────────────────────────────────

/// F100 (fixed): the text of `EncodeError::NeedsSelf` said that the entry "names `Self`, or its
/// generics do".
///
/// Trigger: `trait Tr<'a> where Self: 'a { fn f(&self) -> Rel<'a>; }`: neither `Rel<'a>` nor `<'a>`
/// names `Self`; its context has `Self: 'a`, which is why the impl cannot be for a marker.
///
/// Fix: "(it names it, or its context does)", also in the docs of `ImplSelf` and `has_self()`.
#[test]
fn f100_the_error_for_an_entry_that_needs_self_names_its_context() {
    let marker = Projection::new(
        parse_quote!(Repeater),
        7,
        ImplSelf::Marker(parse_quote!(crate::Marker)),
    );
    for (what, item, ty) in [
        (
            "the context has `Self: 'a`",
            parse_quote!(
                trait Tr<'a>
                where
                    Self: 'a,
                {
                    fn f(&self) -> Rel<'a>;
                }
            ),
            "Rel < 'a >",
        ),
        (
            "the entry names `Self`",
            parse_quote!(
                trait Tr {
                    fn f(&self) -> Rel<Self>;
                }
            ),
            "Rel < Self >",
        ),
    ] {
        let mut leaker = Leaker::new();
        leaker.intern_trait(&item);
        let error = match leaker.result().repeater_impls(&marker) {
            Err(error) => error,
            Ok(impls) => panic!("{}: {} impls", what, impls.len()),
        };
        assert!(matches!(&error, EncodeError::NeedsSelf(_)), "{}", what);
        assert_eq!(
            error.to_string(),
            format!(
                "the type '{}' needs `Self` (it names it, or its context does), so its Repeater \
                 impl cannot be for a marker type; use `ImplSelf::Param`",
                ty
            ),
            "{}",
            what
        );
    }
}

// ── F102 ─────────────────────────────────────────────────────────────────────

/// F102 (fixed): the `Self` of an item nested in the default of a type parameter was taken for the
/// `Self` of the scope.
///
/// Trigger: `struct S<T = [u8; { struct Z<U>(U, Rel<Self>); 3 }]>(T)` was rejected with
/// `SelfInDefault` (also for a trait and `enter_type()`), and
/// `enter_generics(<T = [u8; { struct Z(Option<Box<Self>>); 3 }]>)` with `SelfWithoutMeaning`. As
/// a field there was no error, but the entry's context had `Self = S` for nothing.
///
/// Cause: entering the nested item did not leave the default, and the item's mention of `Self`
/// reached the type that owns the block.
///
/// Fix: a nested item clears the default and the mention of `Self` while it is visited (it cannot
/// name the scope's `Self`: E0401). A `Self` outside the item in the default is still an error.
#[test]
fn f102_the_self_of_an_item_in_the_default_of_a_parameter_is_its_own() {
    let blocks: Vec<(&str, Expr)> = vec![
        (
            "struct with an entry",
            parse_quote!({
                struct Z<U>(U, Rel<Self>);
                3
            }),
        ),
        (
            "struct with a box",
            parse_quote!({
                struct Z(Option<Box<Self>>);
                3
            }),
        ),
        (
            "enum",
            parse_quote!({
                enum Z {
                    A(Rel<Self>),
                }
                3
            }),
        ),
        (
            "union",
            parse_quote!({
                union Z {
                    a: std::mem::ManuallyDrop<Rel<Self>>,
                }
                3
            }),
        ),
        (
            "trait",
            parse_quote!({
                trait Tr {
                    fn f(x: Rel<Self>);
                }
                3
            }),
        ),
        (
            "impl",
            parse_quote!({
                impl<U> Z<U> {
                    fn f(x: Rel<Self>) {}
                }
                3
            }),
        ),
    ];
    for (name, block) in &blocks {
        let entries = assert_default_like_field(name, block);
        assert!(
            entries.iter().all(|(_, has_self)| !has_self),
            "{} as a field: {:?}",
            name,
            entries
        );
    }
    // The block alone, in a scope with no `Self`: no error.
    let mut leaker = common::std_leaker();
    leaker.enter_generics(&Generics::default(), |view| {
        view.visit_type(&parse_quote!(
            [u8; {
                struct Z(Option<Box<Self>>);
                3
            }]
        ))
    });
    assert_eq!(messages(leaker.result().errors()), "");
    // A `Self` outside the item, in the default of a scope with no `Self`, is still an error.
    let mut leaker = common::std_leaker();
    leaker.enter_generics(&parse_quote!(<T = Option<Self>>), |_| ());
    let errors = leaker.result().errors().to_vec();
    assert_eq!(errors.len(), 1);
    assert_eq!(errors[0].reason, Reason::SelfWithoutMeaning);
    // And in the default of a struct, beside a block with an item.
    let mut leaker = common::std_leaker();
    leaker.intern_struct(&parse_quote!(
        struct S<
            T = (
                Rel<Self>,
                [u8; {
                    struct Z(Self);
                    3
                }],
            ),
        >(T);
    ));
    let errors = leaker.result().errors().to_vec();
    assert_eq!(errors.len(), 1, "{:?}", errors);
    assert_eq!(errors[0].reason, Reason::SelfInDefault);
}

// ── F103 ─────────────────────────────────────────────────────────────────────

/// F103 (fixed): the default of a const parameter was not read when the type of the parameter is
/// relative.
///
/// Trigger: `struct S<const N: Idx = { rel::X }> { a: [u8; N] }` reported `Idx` only. With a rule
/// that allows `Idx` given to the Leaker, `rel::X` was reported; given to the result after the
/// analysis (`allow()`), `finish()` was `Ok` and the copy wrote `{ rel::X }` unchecked.
///
/// Cause: `visit_const_param()` returned after the error for the type.
///
/// Fix: the default is read after the error too, as the analysis goes on after any error.
#[test]
fn f103_the_default_of_a_const_parameter_is_read_when_its_type_is_relative() {
    let item: ItemStruct = parse_quote!(
        struct S<const N: Idx = { rel::X }> {
            a: [u8; N],
        }
    );
    let allow_idx = || {
        Rules::new()
            .primitive()
            .prelude()
            .or_if(starts_with(parse_quote!(Idx)))
    };
    // Both are reported, in this order.
    let mut leaker = common::std_leaker();
    leaker.intern_struct(&item);
    let result = leaker.result();
    assert_eq!(
        messages(result.errors()),
        "use absolute path: 'Idx'\nuse absolute path: 'rel::X'"
    );
    // The rule given after the analysis leaves the second, as the one given to the Leaker does.
    let unchecked = "the default is accepted unchecked";
    let after = (result.allow(allow_idx()).finish()).expect_err(unchecked);
    let mut leaker = Leaker::with_rule(allow_idx());
    leaker.intern_struct(&item);
    let before = leaker.result().finish().expect_err(unchecked);
    assert_eq!(messages(&after), "use absolute path: 'rel::X'");
    assert_eq!(after.to_vec(), before.to_vec());
}

// ── F104 ─────────────────────────────────────────────────────────────────────

/// F104 (fixed): the `Self` of an item that a value declares was checked against the scope.
///
/// Trigger: with `check_values`, `enter_generics(<>, |view| view.visit_item(..))` of
/// `fn f() { struct L; impl L { fn g() -> Self { L } } }` gave `SelfWithoutMeaning`, and a trait
/// in the block of a const with a default method naming `Self::X` gave it twice. Under
/// `intern_struct()` and the like the scope has a `Self`, so there was no error.
///
/// Cause: the walk of a value took the `Self` of a struct, enum, union, trait or impl in a block
/// for the one of the scope.
///
/// Fix: inside such an item the walk does not check `Self`: it is the item's own, and the copy
/// writes it as it is. A `Self` of the value itself, outside such an item, is still an error.
#[test]
fn f104_the_self_of_an_item_that_a_value_declares_is_its_own() {
    let analyze = |item: &Item| -> Vec<(Reason, String)> {
        let mut leaker = Leaker::with_rule(Rules::new().primitive().prelude());
        leaker.check_values = true;
        leaker.check_bodies = true;
        leaker.enter_generics(&Generics::default(), |view| view.visit_item(item));
        let result = leaker.result();
        (result.errors().iter())
            .map(|error| (error.reason, error.message.clone()))
            .collect()
    };
    for (name, item) in [
        (
            "an impl in a function",
            parse_quote!(
                fn f() {
                    struct L;
                    impl L {
                        fn g() -> Self {
                            L
                        }
                    }
                }
            ),
        ),
        (
            "a trait in a constant",
            parse_quote!(
                const C: usize = {
                    trait Q {
                        fn q(&self) -> usize {
                            Self::X
                        }
                        const X: usize;
                    }
                    0
                };
            ),
        ),
        (
            "a struct with a method in a static",
            parse_quote!(
                static S: usize = {
                    struct L(usize);
                    impl L {
                        fn get(&self) -> usize {
                            let _s: Self = L(self.0);
                            Self::N
                        }
                        const N: usize = 1;
                    }
                    0
                };
            ),
        ),
        (
            "an enum and a union",
            parse_quote!(
                fn f() -> usize {
                    enum E {
                        A,
                    }
                    impl E {
                        fn a() -> Self {
                            Self::A
                        }
                    }
                    union U {
                        a: u8,
                    }
                    impl U {
                        fn new() -> Self {
                            Self { a: 0 }
                        }
                    }
                    0
                }
            ),
        ),
    ] {
        assert_eq!(analyze(&item), vec![], "{}", name);
    }
    // A `Self` of the value itself, where the scope has none, is an error: before the item, after
    // it, and in an item that has no `Self` of its own.
    for (name, item) in [
        (
            "alone",
            parse_quote!(
                const X: usize = Self::N;
            ),
        ),
        (
            "after the item",
            parse_quote!(
                fn f() -> usize {
                    struct L;
                    impl L {
                        fn g() -> Self {
                            L
                        }
                    }
                    Self::N
                }
            ),
        ),
        (
            "before the item",
            parse_quote!(
                fn f() -> usize {
                    let _n = Self::N;
                    struct L;
                    0
                }
            ),
        ),
        (
            "in a function beside the item",
            parse_quote!(
                fn f() {
                    struct L;
                    fn g() -> usize {
                        Self::N
                    }
                }
            ),
        ),
    ] {
        let errors = analyze(&item);
        assert!(
            !errors.is_empty()
                && errors
                    .iter()
                    .all(|(reason, _)| *reason == Reason::SelfWithoutMeaning),
            "{}: {:?}",
            name,
            errors
        );
    }
}

// ── F105 ─────────────────────────────────────────────────────────────────────

/// F105 (fixed): `::alloc::vec![..]` and a relative `std::vec![..]` were read as macros of the
/// user.
///
/// Trigger: with `check_values` and `Rules::std_crates()`, `::alloc::vec!["{x}"]` and
/// `std::vec!["{x}"]` gave a false `RelativePath` for `x`: the string was read as a format string,
/// which a macro of the user may have. `::std::vec!["{x}"]` gave none.
///
/// Cause: the walk knew a library macro only by a bare name, or under `::core::` and `::std::`.
///
/// Fix: `::alloc::` counts too, and so does a relative path to `std`, `core` or `alloc` that the
/// rule allows and the value does not declare. A relative path that the rule does not allow is
/// still rejected, and its string read as a format string.
#[test]
fn f105_a_macro_of_alloc_or_a_relative_std_macro_is_one_of_the_library() {
    fn messages<A: Allow>(rule: A, item: &Item) -> Vec<String> {
        let mut leaker = Leaker::with_rule(rule);
        leaker.check_values = true;
        leaker.check_bodies = true;
        leaker.enter_generics(&Generics::default(), |view| view.visit_item(item));
        let result = leaker.result();
        (result.errors().iter())
            .map(|error| error.message.clone())
            .collect()
    }
    let std_crates = || Rules::new().std_crates();
    for path in [
        "::std::vec",
        "::alloc::vec",
        "::core::format_args",
        "std::vec",
        "alloc::vec",
        "core::format_args",
    ] {
        let source = format!(
            "fn f() {{ let _ = {}![\"{{x}}\"]; let _ = {}![\"{{y}} {{}}\", 1]; }}",
            path, path
        );
        let item: Item = parse_str(&source).unwrap();
        if path.ends_with("format_args") {
            // It formats: `x` and `y` are captured, so they are paths.
            let found = messages(std_crates(), &item);
            assert_eq!(
                found,
                ["use absolute path: 'x'", "use absolute path: 'y'"],
                "{}",
                path
            );
        } else {
            assert_eq!(
                messages(std_crates(), &item),
                Vec::<String>::new(),
                "{}",
                path
            );
        }
    }
    // A relative path that the rule does not allow is rejected, and its string is read as a
    // format string.
    let item: Item = parse_quote!(
        fn f() {
            let _ = std::vec!["{x}"];
        }
    );
    assert_eq!(
        messages(Rules::new(), &item),
        ["use absolute path: 'std::vec!'", "use absolute path: 'x'"]
    );
    // A `std` that the value declares is the user's own.
    let item: Item = parse_quote!(
        fn f() {
            mod std {
                pub struct Z;
            }
            let _ = std::vec!["{x}"];
        }
    );
    assert_eq!(
        messages(std_crates(), &item)
            .iter()
            .filter(|message| message.ends_with("'x'"))
            .count(),
        1
    );
}

// ── F106 ─────────────────────────────────────────────────────────────────────

/// F106 (fixed): a qualified struct literal or pattern was reported as a path of its own.
///
/// Trigger: with `Leaker::check_values`, `<T as Tr2>::Out { a: 1 }` was reported as the expression
/// `Tr2::Out`, whereas `<T as Tr2>::OUT` is the trait `Tr2`: a rule that asks for the kind of the
/// cause, and the paths of the error, differed. The same for the patterns
/// `<T as Tr2>::Out { a }` and `<T as Tr2>::Out(a)`.
///
/// Cause: the walk of a value took the path of a struct literal or pattern, with the trait inside
/// it, as one path, whereas it splits the path of a qualified expression.
///
/// Fix: a struct literal, a struct pattern and a tuple struct pattern go through the same check as
/// an expression path, then their fields are read.
#[test]
fn f106_a_qualified_struct_literal_or_pattern_is_checked_like_a_qualified_path() {
    let causes = |body: &str| -> (Vec<Cause>, Vec<Vec<Cause>>) {
        let source = format!("fn f<T: ::p::Tr0, U>(x: U) {{ {} }}", body);
        let item: Item = parse_str(&source).unwrap();
        let mut leaker = Leaker::with_rule(Rules::new().primitive().prelude());
        leaker.check_values = true;
        leaker.check_bodies = true;
        leaker.enter_generics(&Generics::default(), |view| view.visit_item(&item));
        let result = leaker.result();
        let errors = result.errors();
        (
            errors.iter().map(|error| error.cause.clone()).collect(),
            errors.iter().map(|error| error.paths.clone()).collect(),
        )
    };
    let path = causes("let _v = <T as Tr2>::OUT;");
    assert_eq!(path.0, [Cause::Trait(parse_quote!(Tr2))]);
    assert_eq!(path.1, [vec![Cause::Trait(parse_quote!(Tr2))]]);
    for (what, body) in [
        ("a struct literal", "let _v = <T as Tr2>::Out { a: 1 };"),
        (
            "a struct literal with a base",
            "let _v = <T as Tr2>::Out { a: 1, ..<T as Tr2>::OUT };",
        ),
        ("a struct pattern", "let <T as Tr2>::Out { a, .. } = x;"),
        ("a tuple struct pattern", "let <T as Tr2>::Out(a, ..) = x;"),
    ] {
        let found = causes(body);
        // The base of the literal is the same trait: the error is once per place.
        assert_eq!(found.0.first(), path.0.first(), "{}", what);
        assert_eq!(found.1.first(), path.1.first(), "{}", what);
        assert!(
            found
                .0
                .iter()
                .all(|cause| *cause == Cause::Trait(parse_quote!(Tr2))),
            "{}: {:?}",
            what,
            found.0
        );
    }
    // An unqualified one is a path of an expression, as before.
    let found = causes("let _v = Out { a: 1 };");
    assert_eq!(found.0, [Cause::Expr(parse_quote!(Out))]);
    // The fields are read: a relative path in one is reported.
    let found = causes("let _v = <T as ::p::Tr2>::Out { a: rel::X };");
    assert_eq!(found.0, [Cause::Expr(parse_quote!(rel::X))]);
    // And the qualified self type is leaked, as in a path.
    for body in [
        "let _v = <Rel as ::p::Tr2>::OUT;",
        "let _v = <Rel as ::p::Tr2>::Out { a: 1 };",
        "let <Rel as ::p::Tr2>::Out { a, .. } = x;",
        "let <Rel as ::p::Tr2>::Out(a, ..) = x;",
    ] {
        let source = format!("fn f<U>(x: U) {{ {} }}", body);
        let item: Item = parse_str(&source).unwrap();
        let mut leaker = Leaker::with_rule(Rules::new().primitive().prelude());
        leaker.check_values = true;
        leaker.check_bodies = true;
        leaker.enter_generics(&Generics::default(), |view| view.visit_item(&item));
        let referrer = leaker.result().finish().unwrap();
        let entries: Vec<String> = referrer.iter().map(|(_, ty, _)| s(ty)).collect();
        assert_eq!(entries, ["Rel"], "{}", body);
    }
}

// ── F107 ─────────────────────────────────────────────────────────────────────

/// F107 (fixed): the Visitor did not replace an entry that names a generic associated type through
/// the trait with its arguments (`<Self as Tr<B, A>>::G<'x>`) when the trait's parameters were not
/// declared in the order the context writes them (types by name, consts last).
///
/// Trigger: `trait Tr<B, A> { type G<'b> where B: 'b; fn m<'x>(&self, a: Rel<<Self as Tr<B,
/// A>>::G<'x>>); }`: the copy left `a: Rel<..>` as written; with `Tr<A, B>` and `where A: 'b` it
/// was replaced. The same for `Tr<Item, Error>`, and a const parameter before a type.
///
/// Cause: the written arguments were paired with the scope's parameters by position, and the
/// Visitor's outermost scope had them in the context's order: the bound `B: 'b` became `A: 'x`.
///
/// Fix: the scope of a context keeps the parameters as declared, and pairs the arguments with
/// those.
#[test]
fn f107_the_trait_arguments_of_a_gat_projection_are_paired_with_the_declared_parameters() {
    for (name, generics, outlives, args) in [
        ("types in order", "A, B", "A", "A, B"),
        ("types out of order", "B, A", "B", "B, A"),
        ("names", "Item, Error", "Item", "Item, Error"),
        ("names out of order", "Item, Error", "Error", "Item, Error"),
        ("a const before a type", "const N: usize, T", "T", "N, T"),
        (
            "a const between types",
            "A, const N: usize, B",
            "B",
            "A, N, B",
        ),
    ] {
        let item: ItemTrait = parse_str(&format!(
            "pub trait Tr<{}> {{ \
                 type G<'b> where {}: 'b; \
                 fn m<'x>(&self, a: Rel<<Self as Tr<{}>>::G<'x>>); \
             }}",
            generics, outlives, args
        ))
        .unwrap();
        let mut leaker = Leaker::new();
        let context = leaker.intern_trait(&item);
        let referrer = leaker.result().finish().unwrap();
        assert_eq!(referrer.iter().count(), 1, "{}", name);
        let text = rewrite(&referrer, context, Item::Trait(item));
        assert!(text.contains("a : P < 0usize >"), "{}: {}", name, text);
    }
}

// ── F108 ─────────────────────────────────────────────────────────────────────

/// F108 (fixed): `Self::LEN` in the tokens of a macro became `Vec<u8>::LEN` for the host `Vec<u8>`,
/// which is not an expression (`<Vec<u8>>::LEN` is, and is what the same path gets outside a
/// macro). A host like `dyn A + B` broke every `Self::..` in tokens the same way.
///
/// Trigger: `substitute_args()` on `{ m!(Self::LEN) }`, and `Visitor::with_self()` on a copy that
/// keeps a macro as written.
///
/// Cause: the tokens of a macro are not parsed, so `Self` was replaced by the tokens of the host
/// wherever it stood.
///
/// Fix: a `Self` that a `::` follows is replaced by `<host>`, unless the host is a path with no
/// arguments (`Host`, `a::Host`), which stays as it was (`m!(Host::LEN)`).
#[test]
fn f108_self_with_a_path_after_it_in_macro_tokens_is_written_in_angle_brackets() {
    for (host, written, expected) in [
        // The trigger.
        ("Vec<u8>", "{ m!(Self::LEN) }", "{ m!(<Vec<u8>>::LEN) }"),
        ("Vec<u8>", "{ Self::LEN }", "{ <Vec<u8>>::LEN }"),
        ("dyn A + B", "{ m!(Self::LEN) }", "{ m!(<dyn A + B>::LEN) }"),
        // In groups, and more than one.
        (
            "Vec<u8>",
            "{ m!([Self::A; 2], (Self::B, Self::C::D)) }",
            "{ m!([<Vec<u8>>::A; 2], (<Vec<u8>>::B, <Vec<u8>>::C::D)) }",
        ),
        // A host that is a qualified path, a reference, a tuple.
        (
            "<T as Tr>::X",
            "{ m!(Self::LEN) }",
            "{ m!(<<T as Tr>::X>::LEN) }",
        ),
        ("&'a Host", "{ m!(Self::LEN) }", "{ m!(<&'a Host>::LEN) }"),
        ("(A, B)", "{ m!(Self::LEN) }", "{ m!(<(A, B)>::LEN) }"),
        // A host that is a path with no arguments stays as it was.
        ("Host", "{ m!(Self::LEN) }", "{ m!(Host::LEN) }"),
        ("a::Host", "{ m!(Self::LEN) }", "{ m!(a::Host::LEN) }"),
        ("::a::Host", "{ m!(Self::A::B) }", "{ m!(::a::Host::A::B) }"),
        // `Self` that no `::` follows is replaced as it was: a bound (`Self: ::p::Tr`, where the
        // `::` is the start of a path), a cast, a type.
        (
            "Vec<u8>",
            "{ m!(Self, Self: ::p::Tr, Self as Tr, Self<T>) }",
            "{ m!(Vec<u8>, Vec<u8>: ::p::Tr, Vec<u8> as Tr, Vec<u8><T>) }",
        ),
    ] {
        let host: Type = parse_str(host).unwrap();
        let mut args = arguments(&[written]);
        substitute_args(args.iter_mut(), |name: &Ident| {
            (name == "Self").then(|| GenericArgument::Type(host.clone()))
        })
        .unwrap();
        // Without spaces: the spacing of the tokens that were written differs from the parsed ones.
        assert_eq!(
            s(&args[0]).replace(' ', ""),
            s(&arguments(&[expected])[0]).replace(' ', ""),
            "substitute_args, {}: {}",
            s(&host),
            written
        );
    }

    // `Visitor::with_self()` does the same in a macro that the copy keeps as written (a bound of a
    // generic whose type names `Self` is an error of the Leaker, `UnprovableBound`).
    let tr: ItemTrait = parse_quote!(
        trait Tr<T: ::p::Tr<::p::m!(Self::LEN, [Self::A; 2])>> {
            fn f(&self, t: T);
        }
    );
    let mut leaker = Leaker::with_rule(Rules::new().primitive().prelude());
    let context = leaker.intern_trait(&tr);
    let result = leaker.result();
    assert_eq!(result.errors().len(), 1, "{}", messages(result.errors()));
    assert_eq!(result.errors()[0].reason, Reason::UnprovableBound);
    let referrer = result.recovered();
    assert_eq!(referrer.iter().count(), 0);
    for (host, expected) in [
        (
            "Vec<u8>",
            quote!(::p::m!(<Vec<u8>>::LEN, [<Vec<u8>>::A; 2])),
        ),
        (
            "dyn A + B",
            quote!(::p::m!(<dyn A + B>::LEN, [<dyn A + B>::A; 2])),
        ),
        ("Host", quote!(::p::m!(Host::LEN, [Host::A; 2]))),
    ] {
        let mut copy = tr.clone();
        common::visitor(&referrer, context, common::placeholder)
            .with_self(parse_str(host).unwrap())
            .visit_item_trait_mut(&mut copy);
        let expected: ItemTrait = parse_quote!(
            trait Tr<T: ::p::Tr<#expected>> {
                fn f(&self, t: T);
            }
        );
        assert_eq!(
            s(&copy).replace(' ', ""),
            s(&expected).replace(' ', ""),
            "with_self, {}",
            host
        );
    }
}

// ── F109 ─────────────────────────────────────────────────────────────────────

/// F109 (fixed): `Referrer::complete_args()` took the one bound of `T` for the trait of `T::Out`
/// when a where predicate with its own binder (`where for<'a> T: Other<'a>`) was a second bound.
///
/// Trigger: `trait Tr<T: ::p::Iter, U = T::Out> where for<'a> T: ::p::Other<'a>`, with `Out`
/// declared by `Other`, completed for `::std::vec::IntoIter<u8>`: the default was
/// `<IntoIter<u8> as ::p::Iter>::Out`, which rustc refuses as the wrong trait (E0576, "cannot find
/// associated type `Out` in trait `p::Iter`"). The copy fails either way: only rustc's message was
/// wrong. No fixture: stable rustc rejects `T::Out` in a default (E0220).
///
/// Cause: the search for the single bound read only the predicates without a binder.
///
/// Fix: a trait bound in a predicate with a binder ends the search, as a bound with a binder does:
/// `AmbiguousAssociatedType`, or `<V>::Out` for an argument that is one identifier.
#[test]
fn f109_a_where_predicate_with_a_binder_is_not_a_single_bound() {
    let into_iter = ["::std::vec::IntoIter<u8>"];
    for (generics, binder) in [
        (
            "<T: ::p::Iter, U = T::Out> where for<'a> T: ::p::Other<'a>",
            "on the predicate",
        ),
        (
            "<T: ::p::Iter, U = T::Out> where T: for<'a> ::p::Other<'a>",
            "on the bound",
        ),
        (
            "<T, U = T::Out> where for<'a> T: ::p::Other<'a>",
            "on the predicate, alone",
        ),
    ] {
        let item: ItemTrait = parse_str(&format!("trait Tr {} {{}}", generics)).unwrap();
        match common::complete_generics(&item.generics, &arguments(&into_iter)) {
            Err(EncodeError::AmbiguousAssociatedType(found)) => assert_eq!(
                s(&*found),
                t(parse_quote!(<::std::vec::IntoIter<u8>>::Out)),
                "{}",
                binder
            ),
            other => panic!("{}: {:?}", binder, other),
        }
        let completed = common::complete_generics(&item.generics, &arguments(&["I"])).unwrap();
        assert_eq!(s(&completed[1]), t(parse_quote!(<I>::Out)), "{}", binder);
    }
}

// ── F110 ─────────────────────────────────────────────────────────────────────

/// F110 (fixed): a name that a block in a type declares (a `let` binding, a parameter of a closure
/// or of a nested function, an item of the block) was taken for a relative path.
///
/// Trigger: `struct S<T>([Rel<T>; { let c = |a: u8| a; 3 }])` was the error `RelativePath` at `a`,
/// "it cannot be leaked as a whole because of `| a : u8 | a`", which sends the user to a wrong fix.
/// Without a closure a local (`{ let x = 3; let _y = x; 3 }`) made the whole array the entry, where
/// `Rel<T>` alone does. The same for a match arm, `if let`, `for`, a label, a receiver, and a block
/// in a const argument of a bound.
///
/// Cause: the visitor took every name in an expression of a type for a relative path; a closure,
/// which the Repeater's impl cannot copy, made that an error.
///
/// Fix: the walk of an expression in a type keeps the names its blocks declare, as the walk of a
/// value does: a `let` binds from the next statement, parameters, arms and loops in their bodies,
/// items and labels in the whole block (not `use`, `extern crate` or `macro_rules!`). A nested item
/// does not see the bindings around it.
#[test]
fn f110_a_name_that_a_block_in_a_type_declares_is_no_relative_path() {
    let found = |check_values: bool, block: &Block| {
        let item: Item = parse_quote!(struct S<T>([Rel<T>; #block]););
        let rule = Rules::new()
            .crate_paths()
            .external_crates()
            .primitive()
            .prelude();
        let mut leaker = Leaker::with_rule(rule);
        leaker.check_values = check_values;
        let context = common::intern_item(&mut leaker, &item);
        let result = leaker.result();
        let errors: Vec<(Reason, String)> = (result.errors().iter())
            .map(|error| (error.reason, s(&error.cause)))
            .collect();
        let referrer = result.recovered();
        let copy = s(&copied(&referrer, context, item)).replace(' ', "");
        (errors, entries_text(&referrer, false), copy)
    };
    // The locals: `Rel<T>` alone is the entry, and the copy writes the length as it is.
    let local: Vec<(&str, Block)> = vec![
        (
            "a parameter of a closure",
            parse_quote!({
                let c = |a: u8| a;
                3
            }),
        ),
        (
            "a local of a nested function",
            parse_quote!({
                fn f() -> usize {
                    let c = || 1;
                    c()
                }
                3
            }),
        ),
        (
            "a `let` binding",
            parse_quote!({
                let x = 3;
                let _y = x;
                3
            }),
        ),
        (
            "no local (control)",
            parse_quote!({
                let _c = || 1;
                3
            }),
        ),
        (
            "a parameter of a nested function",
            parse_quote!({
                fn f(a: usize) -> usize {
                    let c = |b: usize| a + b;
                    c(1)
                }
                f(2)
            }),
        ),
        (
            "an item of the block, before it",
            parse_quote!({
                let _c = || X;
                const X: usize = 3;
                X
            }),
        ),
        (
            "a struct of the block and its impl",
            parse_quote!({
                struct Z(usize);
                impl Z {
                    const A: usize = 3;
                }
                let _c = || Z(1);
                let _z: Z = Z(2);
                Z::A
            }),
        ),
        (
            "a trait of the block and its impl",
            parse_quote!({
                trait Tr {
                    const A: usize;
                }
                struct Z;
                impl Tr for Z {
                    const A: usize = 3;
                }
                let _c = || 1;
                <Z as Tr>::A
            }),
        ),
        (
            "the receiver of a method",
            parse_quote!({
                struct Z(usize);
                impl Z {
                    fn get(&self) -> usize {
                        self.0
                    }
                }
                let _c = || 1;
                3
            }),
        ),
        (
            "a module of the block and its items",
            parse_quote!({
                mod m {
                    pub const N: usize = 3;
                    pub const M: usize = N;
                }
                let _c = || 1;
                m::M
            }),
        ),
        (
            "a label",
            parse_quote!({
                let _c = || 1;
                'outer: loop {
                    break 'outer 3;
                }
            }),
        ),
        (
            "a match arm",
            parse_quote!({
                let _c = || 1;
                match 3 {
                    n if n > 0 => n,
                    _ => 0,
                }
            }),
        ),
        (
            "`if let`",
            parse_quote!({
                let _c = || 1;
                if let Some(v) = Some(3) {
                    v
                } else {
                    3
                }
            }),
        ),
        (
            "`while let`",
            parse_quote!({
                let _c = || 1;
                let mut n = 0;
                while let Some(v) = Some(n).filter(|m| *m < 3) {
                    n = v + 1;
                }
                n
            }),
        ),
        (
            "`for`",
            parse_quote!({
                let _c = || 1;
                let mut n = 0;
                for i in 0..3 {
                    n += i;
                }
                n
            }),
        ),
    ];
    // The relative paths: still an error with a closure, the whole array the entry without one.
    let relative: Vec<(&str, Block, &str)> = vec![
        (
            "a relative path in a closure",
            parse_quote!({
                let c = |a: u8| rel::N;
                3
            }),
            "rel :: N",
        ),
        (
            "a relative function in a closure, with its parameter",
            parse_quote!({
                let c = |a: u8| g(a);
                3
            }),
            "g",
        ),
        (
            "a name used before its `let`",
            parse_quote!({
                let _c = || 1;
                let _y = x;
                let x = 3;
                3
            }),
            "x",
        ),
        (
            "a parameter of a closure after it",
            parse_quote!({
                let _c = |a: u8| a;
                let _d = a;
                3
            }),
            "a",
        ),
        (
            "a binding of a match arm after it",
            parse_quote!({
                let _c = || 1;
                match 3 {
                    n => n,
                };
                n
            }),
            "n",
        ),
        (
            "a binding around a nested function",
            parse_quote!({
                let x = 3;
                fn f() -> usize {
                    let _c = || 1;
                    x
                }
                3
            }),
            "x",
        ),
        (
            "a name that a `use` brings in",
            parse_quote!({
                use rel::N;
                let _c = || 1;
                N
            }),
            "N",
        ),
    ];
    let without_closure: Block = parse_quote!({
        let _y = x;
        let x = 3;
        3
    });
    for check_values in [false, true] {
        for (name, block) in &local {
            let item: Item = parse_quote!(struct S<T>([P<0usize>; #block]););
            assert_eq!(
                found(check_values, block),
                (
                    vec![],
                    vec!["Rel<T>".to_string()],
                    s(&item).replace(' ', "")
                ),
                "{}, check_values {}",
                name,
                check_values
            );
        }
        for (name, block, cause) in &relative {
            let (errors, entries, _) = found(check_values, block);
            assert_eq!(
                (errors, entries),
                (
                    vec![(Reason::RelativePath, cause.to_string())],
                    vec!["Rel<T>".to_string()]
                ),
                "{}, check_values {}",
                name,
                check_values
            );
        }
        let array: Type = parse_quote!([Rel<T>; #without_closure]);
        let (errors, entries, copy) = found(check_values, &without_closure);
        assert_eq!(
            (errors, entries, copy),
            (
                vec![],
                vec![s(&array).replace(' ', "")],
                "structS<T>(P<0usize>);".to_string()
            ),
            "a name used before its `let`, without a closure, check_values {}",
            check_values
        );
        // A const argument of a bound is read outside a type, where a relative path is an error
        // with or without a closure: the locals there are local too.
        for (item, cause) in [
            (
                "struct S<T: ::p::Tr<{ let c = |a: usize| a; 3 }>>(Rel<T>);",
                None,
            ),
            (
                "struct S<T>(Rel<T>) where T: ::p::Tr<{ let x = 3; x }>;",
                None,
            ),
            (
                "struct S<T>(Rel<T>) where T: ::p::Tr<{ let x = 3; y }>;",
                Some("y"),
            ),
        ] {
            let item: Item = parse_str(item).unwrap();
            let rule = Rules::new().external_crates().primitive().prelude();
            let mut leaker = Leaker::with_rule(rule);
            leaker.check_values = check_values;
            common::intern_item(&mut leaker, &item);
            let result = leaker.result();
            let errors: Vec<(Reason, String)> = (result.errors().iter())
                .map(|error| (error.reason, s(&error.cause)))
                .collect();
            let expected: Vec<(Reason, String)> = (cause.into_iter())
                .map(|cause| (Reason::RelativePath, cause.to_string()))
                .collect();
            assert_eq!(
                (errors, entries_text(&result.recovered(), false)),
                (expected, vec!["Rel<T>".to_string()]),
                "{}, check_values {}",
                s(&item),
                check_values
            );
        }
    }
}

// ── F111 ─────────────────────────────────────────────────────────────────────

/// F111 (fixed): an array that is an optional entry only through its length was replaced, though it
/// is the only input that keeps a method lifetime.
///
/// Trigger, E0195: `fn m<'x>(&self, a0: [&'x String; ::l::base::BLEN]) -> super::base::Bl<'x>` in
/// a trait, with `::l` not allowed. The length names a const of another crate, so the array was an
/// optional entry and the copy `fn m<'x>(&self, a0: P<1>) -> P<0>`: `'x` only in projections, so
/// early-bound in the copy's impl where the trait has it late-bound. No error.
///
/// Cause: the check of F18 (an optional type that alone keeps a lifetime is written as it is) runs
/// only for a lifetime that no input keeps as written, and that search looked into the array as
/// into a tuple, taking `&'x String` for a keeper, though the copy replaces the whole array.
///
/// Fix: the search does not look into an array whose length names another crate, so the array is
/// no optional entry and is written as it is (its crate must resolve in the Referrer's crate); a
/// relative type in it is the entry (`[&'x Rel; ..]`). Beside such an array, a trait object of
/// this crate behind `&'x` is now the error `UnconstrainedLifetime`.
#[test]
fn f111_an_array_optional_through_its_length_that_keeps_a_method_lifetime_is_written_as_it_is() {
    // An item, the errors, the entries and the copy (`P<ix>` for an entry).
    type Case = (
        ItemTrait,
        Vec<(Reason, String)>,
        Vec<&'static str>,
        &'static str,
    );
    let bl = "super :: base :: Bl < 'x >";
    let cases: Vec<Case> = vec![
        // The array alone keeps `'x`: it is written as it is.
        (
            parse_quote!(
                trait Tr {
                    fn m<'x>(&self, a0: [&'x String; ::l::base::BLEN]) -> super::base::Bl<'x>;
                }
            ),
            vec![],
            vec![bl],
            "trait Tr { fn m < 'x > (& self , a0 : [& 'x String ; :: l :: base :: BLEN]) -> P < 0usize > ; }",
        ),
        // The element is of another crate too: both are written as they are.
        (
            parse_quote!(
                trait Tr {
                    fn m<'x>(
                        &self,
                        a0: [::l::base::B1<&'x u8>; ::l::base::BLEN],
                    ) -> super::base::Bl<'x>;
                }
            ),
            vec![],
            vec![bl],
            "trait Tr { fn m < 'x > (& self , a0 : [:: l :: base :: B1 < & 'x u8 > ; :: l :: base :: BLEN] ,) -> P < 0usize > ; }",
        ),
        // The element holds a relative type: it is the entry, not the array around it.
        (
            parse_quote!(
                trait Tr {
                    fn m<'x>(&self, a0: [&'x Rel; ::l::base::BLEN]) -> super::base::Bl<'x>;
                }
            ),
            vec![],
            vec![bl, "Rel"],
            "trait Tr { fn m < 'x > (& self , a0 : [& 'x P < 1usize > ; :: l :: base :: BLEN]) -> P < 0usize > ; }",
        ),
        // The shape that the property test found: the array and a type of another crate.
        (
            parse_quote!(
                trait Tr {
                    fn m<'x>(
                        &self,
                        a0: [&'x String; ::l::base::BLEN],
                        a1: ::l::base::B1<&'x u8>,
                    ) -> super::base::Bl<'x>;
                }
            ),
            vec![],
            vec![bl],
            "trait Tr { fn m < 'x > (& self , a0 : [& 'x String ; :: l :: base :: BLEN] , a1 : :: l :: base :: B1 < & 'x u8 > ,) -> P < 0usize > ; }",
        ),
        // A relative length: the array is a required entry, reported (the cause is the length)
        // and written as it is.
        (
            parse_quote!(
                trait Tr {
                    fn m<'x>(&self, a0: [&'x String; super::base::BLEN]) -> super::base::Bl<'x>;
                }
            ),
            vec![(
                Reason::UnconstrainedLifetime,
                "super :: base :: BLEN".to_string(),
            )],
            vec![bl],
            "trait Tr { fn m < 'x > (& self , a0 : [& 'x String ; super :: base :: BLEN]) -> P < 0usize > ; }",
        ),
        // Controls: a type of another crate in its place is written as it is, as before (F18);
        // another input keeps `'x` as written, so the array is an optional entry, as before.
        (
            parse_quote!(
                trait Tr {
                    fn m<'x>(&self, a0: ::l::base::B1<&'x u8>) -> super::base::Bl<'x>;
                }
            ),
            vec![],
            vec![bl],
            "trait Tr { fn m < 'x > (& self , a0 : :: l :: base :: B1 < & 'x u8 >) -> P < 0usize > ; }",
        ),
        (
            parse_quote!(
                trait Tr {
                    fn m<'x>(
                        &self,
                        a0: [&'x String; ::l::base::BLEN],
                        a1: &'x u8,
                    ) -> super::base::Bl<'x>;
                }
            ),
            vec![],
            vec![bl, "[& 'x String ; :: l :: base :: BLEN]"],
            "trait Tr { fn m < 'x > (& self , a0 : P < 1usize > , a1 : & 'x u8 ,) -> P < 0usize > ; }",
        ),
        // A trait object of this crate behind `&'x` beside the array takes its lifetime from the
        // reference, which only it keeps then: it is the error, and the array stays an optional
        // entry, as beside a type of another crate.
        (
            parse_quote!(
                trait Tr {
                    fn m<'x>(
                        &self,
                        a0: [&'x String; ::l::base::BLEN],
                        a1: &'x dyn Tr0,
                    ) -> super::base::Bl<'x>;
                }
            ),
            vec![(Reason::UnconstrainedLifetime, "Tr0".to_string())],
            vec![bl, "[& 'x String ; :: l :: base :: BLEN]"],
            "trait Tr { fn m < 'x > (& self , a0 : P < 1usize > , a1 : & 'x dyn Tr0 ,) -> P < 0usize > ; }",
        ),
        (
            parse_quote!(
                trait Tr {
                    fn m<'x>(
                        &self,
                        a0: ::l::base::B1<&'x u8>,
                        a1: &'x dyn Tr0,
                    ) -> super::base::Bl<'x>;
                }
            ),
            vec![(Reason::UnconstrainedLifetime, "Tr0".to_string())],
            vec![bl, ":: l :: base :: B1 < & 'x u8 >"],
            "trait Tr { fn m < 'x > (& self , a0 : P < 1usize > , a1 : & 'x dyn Tr0 ,) -> P < 0usize > ; }",
        ),
    ];
    for (item, errors, entries, copy) in cases {
        let found = f63_leak(common::std_config(), &item);
        let entries: Vec<String> = entries.into_iter().map(String::from).collect();
        assert_eq!(found, (copy.to_string(), errors, entries), "{}", s(&item));
    }
}

//! The errors of the Leaker as data (`ErrorItem`): what each is about (`Cause`) and why (`Reason`),
//! how it is written as tokens and read back, and how it is reported.
//!
//! - One test for each `Reason`: an item that triggers it, the reason of the error, and the kind
//!   and the text of its cause. The other kinds of cause have a test of their own.
//! - `every_error_is_written_and_read_back` is a round trip of every error above: the tokens
//!   that are written again are the same, and so are the fields. `the_serialized_form_is_*` pin
//!   the grammar, and what the parser refuses.
//! - `to_compile_error()`, `emit()`, `to_syn_error()`, and the other ways to take an error as a
//!   whole.
//! - The inputs that cannot be leaked on stable Rust, and how the errors for them read (`limits`).
//! - The help of every kind of error (see the section at the end).
//!
//! The errors are reported at the same places after the round trip: see `tests/diagnostic.rs` (a
//! macro that does it, and rustc, with the nightly layout of a `help`, which has its own span) and
//! `tests/crates/spans` (the lines and the columns).

mod common;

use common::{allow_every_cause, causes_accepting, starts_with, AcceptCause, Rule};
use std::cell::RefCell;
use syn::visit_mut::VisitMut;
use syn::*;
use template_quote::ToTokens;
use type_leak::*;

thread_local! {
    /// The rules that `configured()` adds to the config of a trigger: the config stage of the
    /// tests that compare it with the result stage. Empty in the thread of any other test.
    static RULES: RefCell<Vec<Rule>> = RefCell::new(Vec::new());
}

/// `config`, with the rules of `with_rules()`.
fn configured<A: Allow>(config: A) -> impl Allow {
    let rules = RULES.with(|rules| rules.borrow().clone());
    (config, move |cause: &Cause, place: Place| {
        rules.iter().any(|rule| rule.allow(cause, place))
    })
}

/// The Leaker of the triggers: `Leaker::from_config(config, ..)` with the rules of `with_rules()`.
fn leaker_from<A: Allow>(config: A) -> Leaker<impl Allow> {
    Leaker::from_config(configured(config), Options::new())
}

/// Runs `f` with `rules` added to the config of the triggers.
fn with_rules<R>(rules: &[Rule], f: impl FnOnce() -> R) -> R {
    RULES.with(|set| *set.borrow_mut() = rules.to_vec());
    let result = f();
    RULES.with(|set| set.borrow_mut().clear());
    result
}

/// The config of the triggers with `::p` allowed (the crate of the property test).
fn allowing_config() -> Rules<impl Allow + Clone> {
    Rules::new()
        .primitive()
        .prelude()
        .or_if(starts_with(parse_quote!(::p)))
}

fn allowing() -> Leaker<impl Allow> {
    leaker_from(allowing_config())
}

/// The config of the triggers that read the values of the items too (`check_values`).
fn reading_values_config() -> Rules<impl Allow + Clone> {
    Rules::new().primitive().prelude()
}

/// An analysis of an item that goes wrong.
type Trigger = fn() -> LeakerResult;

/// Runs `analyze` on `leaker` and returns its result.
fn run<A: Allow, R>(
    mut leaker: Leaker<A>,
    analyze: impl FnOnce(&mut Leaker<A>) -> R,
) -> LeakerResult {
    analyze(&mut leaker);
    leaker.result()
}

/// The errors of a trigger.
fn errors(trigger: Trigger) -> Vec<ErrorItem> {
    trigger().errors().to_vec()
}

/// The errors that `leaker` has after `analyze` ran on it.
fn errors_in<A: Allow, R>(
    leaker: Leaker<A>,
    analyze: impl FnOnce(&mut Leaker<A>) -> R,
) -> Vec<ErrorItem> {
    run(leaker, analyze).errors().to_vec()
}

fn id(name: &str) -> Ident {
    Ident::new(name, proc_macro2::Span::call_site())
}

/// The word that names the kind of a cause.
fn kind(cause: &Cause) -> &'static str {
    match cause {
        Cause::Type(_) => "type",
        Cause::Trait(_) => "trait",
        Cause::Expr(_) => "expr",
        Cause::Macro(_) => "macro",
        Cause::Lifetime(_) => "lifetime",
        Cause::Unknown(_) => "unknown",
        Cause::Unreadable(_) => "unreadable",
        other => panic!("{:?}", other),
    }
}

/// The tokens of a cause, as text.
fn text(cause: &Cause) -> String {
    cause.to_token_stream().to_string()
}

/// Asserts the reason, the kind of the cause and its text.
#[track_caller]
fn assert_item(item: &ErrorItem, reason: Reason, cause_kind: &str, cause_text: &str) {
    assert_eq!(item.reason, reason, "{}", item.message);
    assert_eq!(kind(&item.cause), cause_kind, "{}", item.message);
    assert_eq!(text(&item.cause), cause_text, "{}", item.message);
}

/// The notes of an item, as the text of the tokens they are about and the note.
fn notes_of(item: &ErrorItem) -> Vec<(String, String)> {
    item.notes
        .iter()
        .map(|(at, note)| (at.to_string(), note.clone()))
        .collect()
}

/// The text of the notes of an item.
fn note_texts(item: &ErrorItem) -> Vec<&str> {
    item.notes.iter().map(|(_, note)| note.as_str()).collect()
}

/// The help of an item, as the text of the tokens it is about and the help.
fn help_of(item: &ErrorItem) -> Option<(String, String)> {
    item.help
        .as_ref()
        .map(|(at, help)| (at.to_string(), help.clone()))
}

/// The analysis of a trait with these items. `check_values` reads the values too.
fn trait_result(source: &str, check_values: bool) -> LeakerResult {
    let item: ItemTrait = parse_str(source).unwrap();
    if check_values {
        let leaker = Leaker::from_config(
            configured(reading_values_config()),
            Options::new().check_values(),
        );
        run(leaker, |leaker| leaker.intern_trait(&item, true))
    } else {
        run(leaker_from(allowing_config()), |leaker| {
            leaker.intern_trait(&item, true)
        })
    }
}

/// The errors that a trait with these items gives. `check_values` reads the values too.
fn trait_errors(source: &str, check_values: bool) -> Vec<ErrorItem> {
    trait_result(source, check_values).errors().to_vec()
}

// ── the triggers: an item for each reason ────────────────────────────────────────────────────

/// A trait in a bound and a type with an elided lifetime.
fn relative_path() -> LeakerResult {
    let item: ItemTrait = parse_quote!(
        trait Tr: Clone {
            fn f(&self, a: MyType<'_>);
        }
    );
    run(leaker_from(Rules::new()), |leaker| {
        leaker.intern_trait(&item, true)
    })
}

#[test]
fn reason_relative_path() {
    let items = errors(relative_path);
    assert_eq!(items.len(), 2);
    assert_item(&items[0], Reason::RelativePath, "trait", "Clone");
    assert_item(&items[1], Reason::RelativePath, "type", "MyType < '_ >");
    assert_eq!(items[0].message, "use absolute path: 'Clone'");
}

/// `Self` where no scope declares it.
fn self_without_meaning() -> LeakerResult {
    run(leaker_from(allowing_config()), |leaker| {
        leaker.enter_generics(&Generics::default(), |view| {
            view.visit_type(&parse_quote!(Rel<Self>));
            view.visit_type(&parse_quote!(Wrap<Self::Item>));
        })
    })
}

#[test]
fn reason_self_without_meaning() {
    let items = errors(self_without_meaning);
    assert_eq!(items.len(), 2);
    assert_item(&items[0], Reason::SelfWithoutMeaning, "type", "Self");
    assert_item(
        &items[1],
        Reason::SelfWithoutMeaning,
        "type",
        "Self :: Item",
    );
    // The note and the help are about the mention.
    assert_eq!(items[1].notes[0].0.to_string(), "Self :: Item");
    assert_eq!(
        items[1].help.as_ref().unwrap().0.to_string(),
        "Self :: Item"
    );
    for item in &items {
        assert_eq!(item.message, "`Self` has no meaning here");
        assert!(
            item.notes[0]
                .1
                .starts_with("`Self` has no meaning here: it is declared only"),
            "{}",
            item.notes[0].1
        );
        assert_eq!(item.help_text(), Some("write the type instead of `Self`"));
    }
}

/// Syntax that the Leaker cannot read in a body, read with `check_values`.
fn unreadable() -> LeakerResult {
    trait_result(
        "trait Tr { fn f(&self) { macro_rules! m { () => {} } ::p::mac!(a b c); } }",
        true,
    )
}

#[test]
fn reason_unreadable() {
    let items = errors(unreadable);
    assert_eq!(items.len(), 2);
    assert_item(
        &items[0],
        Reason::Unreadable,
        "unreadable",
        "macro_rules ! m { () => { } }",
    );
    assert_item(
        &items[1],
        Reason::Unreadable,
        "unreadable",
        ":: p :: mac ! (a b c)",
    );
    assert_eq!(
        items[0].message,
        "cannot analyze `macro_rules ! m { () => { } }` in the default body of `f`"
    );
}

/// A bare identifier that may be a const, in a path that names `Self`, which is denied.
fn maybe_const() -> LeakerResult {
    let item: ItemStruct = parse_quote!(
        struct S {
            a: ::p::W<[Self], LEN>,
        }
    );
    run(leaker_from(allowing_config()), |leaker| {
        leaker.intern_struct(&item, false)
    })
}

#[test]
fn reason_maybe_const() {
    let items = errors(maybe_const);
    assert_eq!(items.len(), 1);
    // The bare identifier is the cause.
    assert_item(&items[0], Reason::MaybeConst, "type", "LEN");
}

/// A lifetime of a method that the copy would leave unconstrained.
fn unconstrained_lifetime() -> LeakerResult {
    trait_result(
        "trait Tr { fn f<'b>(&self, x: Wrap<'b>) -> &'b u8; }",
        false,
    )
}

#[test]
fn reason_unconstrained_lifetime() {
    let items = errors(unconstrained_lifetime);
    assert_eq!(items.len(), 1);
    assert_item(
        &items[0],
        Reason::UnconstrainedLifetime,
        "type",
        "Wrap < 'b >",
    );
    assert!(
        items[0].message.ends_with("(E0195)"),
        "{}",
        items[0].message
    );
    // The note says why.
    assert!(
        items[0].notes[0].1.starts_with("it names `'b`"),
        "{}",
        items[0].notes[0].1
    );
}

/// `Self::A` in an impl of a trait, where the Repeater's impl cannot write it.
fn self_projection() -> LeakerResult {
    let item: ItemImpl = parse_quote!(
        impl ::p::Tr for u8 {
            type A = u8;
            fn f(&self, x: Wrap<Self::A>) {}
        }
    );
    run(leaker_from(allowing_config()), |leaker| {
        leaker.intern_impl(&item, true)
    })
}

#[test]
fn reason_self_projection() {
    let items = errors(self_projection);
    assert_eq!(items.len(), 1);
    assert_item(
        &items[0],
        Reason::SelfProjection,
        "type",
        "Wrap < Self :: A >",
    );
}

/// A bound of an outer trait that names its `Self`, for a type in an item nested in it.
fn unnameable_self() -> LeakerResult {
    run(leaker_from(allowing_config()), |leaker| {
        leaker.enter_trait(&id("Tr"), &parse_quote!(<T: AsRef<Self>>), true, |view| {
            view.visit_item(&parse_quote!(
                impl ::p::W for X {
                    fn f(&self, x: Rel<T>) {}
                }
            ))
        })
    })
}

#[test]
fn reason_unnameable_self() {
    let items = errors(unnameable_self);
    assert_eq!(items.len(), 1);
    assert_item(&items[0], Reason::UnnameableSelf, "type", "Rel < T >");
}

/// `Self` in the default of a parameter.
fn self_in_default() -> LeakerResult {
    trait_result("trait Tr<T: Default = Wrap<Self>> {}", false)
}

#[test]
fn reason_self_in_default() {
    let items = errors(self_in_default);
    assert_eq!(items.len(), 1);
    // The cause is the whole default, the note is at the `Self` in it.
    assert_item(&items[0], Reason::SelfInDefault, "type", "Wrap < Self >");
    assert_eq!(items[0].notes[0].0.to_string(), "Self");
}

/// A generic whose bound mentions a `Self` that is denied.
fn denied_self_in_bounds() -> LeakerResult {
    let item: ItemTrait = parse_quote!(
        trait Tr {
            fn f<T: ::core::cmp::PartialEq<Self>>(&self, x: Wrap<T>);
        }
    );
    run(leaker_from(allowing_config()), |leaker| {
        leaker.intern_trait(&item, false)
    })
}

#[test]
fn reason_denied_self_in_bounds() {
    let items = errors(denied_self_in_bounds);
    assert_eq!(items.len(), 1);
    assert_item(&items[0], Reason::DeniedSelfInBounds, "type", "Wrap < T >");
    assert_eq!(
        note_texts(&items[0]).first(),
        Some(&"the bounds of its generics mention `Self`, which cannot be leaked")
    );
}

/// A type that names the `Self` of a nested struct, with a generic whose bound names it too.
fn nested_self_in_bounds() -> LeakerResult {
    run(leaker_from(allowing_config()), |leaker| {
        leaker.enter_mod("m", [id("S")], |view| {
            view.visit_item(&parse_quote!(
                struct S<T: ::core::convert::AsRef<Self>>(Rel2<Self, T>);
            ))
        })
    })
}

#[test]
fn reason_nested_self_in_bounds() {
    let items = errors(nested_self_in_bounds);
    assert_eq!(items.len(), 1);
    assert_item(
        &items[0],
        Reason::NestedSelfInBounds,
        "type",
        "Rel2 < Self , T >",
    );
}

/// A leaked type in a predicate whose generic has a leaked bound.
fn unprovable_bound() -> LeakerResult {
    let item: ItemTrait = parse_quote!(
        trait Tr<T: Fn(Rel)>
        where
            Wrap<T>: Sized,
        {
        }
    );
    run(leaker_from(allowing_config()), |leaker| {
        leaker.intern_trait(&item, true)
    })
}

#[test]
fn reason_unprovable_bound() {
    let items = errors(unprovable_bound);
    assert_eq!(items.len(), 1);
    assert_item(&items[0], Reason::UnprovableBound, "type", "Wrap < T >");
}

/// A type that names `Self` in a predicate of its trait.
fn self_in_trait_predicate() -> LeakerResult {
    let item: ItemTrait = parse_quote!(
        trait Tr
        where
            Rel<Self>: Sized,
        {
        }
    );
    run(leaker_from(allowing_config()), |leaker| {
        leaker.intern_trait(&item, true)
    })
}

#[test]
fn reason_self_in_trait_predicate() {
    let items = errors(self_in_trait_predicate);
    assert_eq!(items.len(), 1);
    assert_item(
        &items[0],
        Reason::SelfInTraitPredicate,
        "type",
        "Rel < Self >",
    );
}

/// The self type of an impl whose bound names `Self`.
fn self_in_impl_self_type() -> LeakerResult {
    let item: ItemImpl = parse_quote!(
        impl<T> S<T> where T: AsRef<Option<Self>> {}
    );
    run(leaker_from(allowing_config()), |leaker| {
        leaker.intern_impl(&item, true)
    })
}

#[test]
fn reason_self_in_impl_self_type() {
    let items = errors(self_in_impl_self_type);
    assert_eq!(items.len(), 2);
    assert_item(&items[0], Reason::SelfInImplSelfType, "type", "S < T >");
    // The bound is an error of its own: rustc cannot prove it while it checks the bounds.
    assert_item(&items[1], Reason::UnprovableBound, "type", "Self");
}

// ── the other kinds of cause ─────────────────────────────────────────────────────────────────

/// A path in a const expression (the default of a const parameter) and a path in a value.
fn values() -> LeakerResult {
    trait_result(
        "trait Tr<const N: usize = SIZE> { fn f(&self) { let _ = inner::N; let _ = Rel::<u8>::f(); } }",
        true,
    )
}

#[test]
fn the_cause_of_a_path_in_a_value_is_a_value() {
    let items = errors(values);
    assert_eq!(items.len(), 3);
    assert_item(&items[0], Reason::RelativePath, "expr", "SIZE");
    assert_item(&items[1], Reason::RelativePath, "expr", "inner :: N");
    assert_item(
        &items[2],
        Reason::RelativePath,
        "expr",
        "Rel :: < u8 > :: f",
    );
}

/// The path of a macro in a value, and the trait of a qualified path.
fn macros_and_qualified_traits() -> LeakerResult {
    trait_result(
        "trait Tr { fn f(&self) { mac!(1); let _ = <u8 as Rel>::f(); } }",
        true,
    )
}

#[test]
fn the_cause_of_a_macro_is_its_path_and_that_of_a_qualified_trait_is_a_trait() {
    let items = errors(macros_and_qualified_traits);
    assert_eq!(items.len(), 2);
    assert_item(&items[0], Reason::RelativePath, "macro", "mac !");
    assert!(matches!(&items[0].cause, Cause::Macro(path) if path.is_ident("mac")));
    assert_item(&items[1], Reason::RelativePath, "trait", "Rel");
    assert_eq!(items[0].message, "use absolute path: 'mac!'");
}

/// A lifetime that no scope declares, in the supertrait of a trait where `Self` is denied.
fn lifetime() -> LeakerResult {
    let item: ItemTrait = parse_quote!(
        trait Tr: ::p::Sup<&'a Self> {}
    );
    run(leaker_from(allowing_config()), |leaker| {
        leaker.intern_trait(&item, false)
    })
}

#[test]
fn the_cause_of_a_bound_is_the_trait() {
    let items = trait_errors("trait Tr: MyBound { fn f<T: Other<u8>>(&self); }", false);
    assert_eq!(items.len(), 2);
    assert_item(&items[0], Reason::RelativePath, "trait", "MyBound");
    assert_item(&items[1], Reason::RelativePath, "trait", "Other < u8 >");
    assert!(matches!(&items[1].cause, Cause::Trait(path) if path.segments.len() == 1));
}

// ── the kind of the cause is that of the syntax node which holds the part reported ──────────

/// The errors that a type gives in the generics `<T>`, where `Self` has no meaning.
fn type_errors<A: Allow>(mut leaker: Leaker<A>, ty: Type) -> Vec<ErrorItem> {
    leaker.enter_generics(&parse_quote!(<T>), |view| view.visit_type(&ty));
    leaker.result().errors().to_vec()
}

/// The note of an item at `at`, if it has one there.
fn note_at<'a>(item: &'a ErrorItem, at: &str) -> Option<&'a str> {
    item.notes
        .iter()
        .find(|(tokens, _)| tokens.to_string() == at)
        .map(|(_, note)| note.as_str())
}

#[test]
fn the_trait_of_a_dyn_type_is_a_trait() {
    // The error is reported at the trait, which is the path of a bound of the `dyn` type: the cause
    // is a trait, not a type. The note is at the `dyn` type as a whole.
    let items = type_errors(allowing(), parse_quote!(::p::W<'_, dyn RelTr>));
    assert_eq!(items.len(), 1);
    assert_item(&items[0], Reason::RelativePath, "trait", "RelTr");
    assert!(matches!(&items[0].cause, Cause::Trait(path) if path.is_ident("RelTr")));
    assert_eq!(
        note_at(&items[0], "dyn RelTr"),
        Some("it cannot be leaked as a whole because of `dyn RelTr`")
    );
    // A method whose lifetime the copy would leave unconstrained (E0195).
    let item: ItemTrait = parse_quote!(
        trait Tr {
            fn f<'b>(&self, x: &'b dyn RelTr) -> &'b ();
        }
    );
    let mut leaker = allowing();
    leaker.intern_trait(&item, true);
    let result = leaker.result();
    let items = result.errors();
    assert_eq!(items.len(), 1);
    assert_item(&items[0], Reason::UnconstrainedLifetime, "trait", "RelTr");
    assert!(
        items[0].notes[0]
            .1
            .starts_with("the lifetime of the trait object defaults to `'b`"),
        "{}",
        items[0].notes[0].1
    );
}

#[test]
fn the_trait_of_an_impl_type_is_a_trait() {
    let items = type_errors(allowing(), parse_quote!(impl MyTrait));
    assert_eq!(items.len(), 1);
    assert_item(&items[0], Reason::RelativePath, "trait", "MyTrait");
    assert_eq!(
        note_at(&items[0], "impl MyTrait"),
        Some("it cannot be leaked as a whole because of `impl MyTrait`")
    );
}

#[test]
fn a_path_in_a_closure_in_an_array_length_is_a_value() {
    let items = type_errors(
        allowing(),
        parse_quote!(
            [u8; {
                let c = |a: u8| a;
                3
            }]
        ),
    );
    assert_eq!(items.len(), 1);
    assert_item(&items[0], Reason::RelativePath, "expr", "a");
    assert!(matches!(&items[0].cause, Cause::Expr(Expr::Path(path)) if path.path.is_ident("a")));
}

#[test]
fn a_macro_call_in_an_array_length_is_a_macro() {
    let items = type_errors(allowing(), parse_quote!([u8; two!()]));
    assert_eq!(items.len(), 1);
    // The error is reported at the path, and the cause writes the call with its `!`.
    assert_item(&items[0], Reason::RelativePath, "macro", "two !");
    assert!(matches!(&items[0].cause, Cause::Macro(path) if path.is_ident("two")));
    assert_eq!(items[0].help.as_ref().unwrap().0.to_string(), "two");
    // A macro that is a type is a type, as a whole.
    let item: ItemStruct = parse_quote!(
        struct S {
            f: ::p::Wrap<::p::m!(Self)>,
        }
    );
    let mut leaker = allowing();
    leaker.intern_struct(&item, false);
    let result = leaker.result();
    let items = result.errors();
    assert_eq!(items.len(), 1);
    assert_item(
        &items[0],
        Reason::RelativePath,
        "type",
        ":: p :: m ! (Self)",
    );
}

#[test]
fn a_type_in_a_turbofish_in_an_array_length_is_a_type() {
    let item: ItemTrait = parse_quote!(
        trait Tr {
            fn f(
                &self,
                x: [u8; {
                    let c = |_: ::p::U| ::p::f::<Big>();
                    3
                }],
            );
        }
    );
    let mut leaker = allowing();
    leaker.intern_trait(&item, true);
    let result = leaker.result();
    let items = result.errors();
    assert_eq!(items.len(), 1);
    assert_item(&items[0], Reason::RelativePath, "type", "Big");
}

#[test]
fn a_relative_type_in_an_absolute_holder_that_cannot_be_leaked_is_a_type() {
    // `Rel` is reported, not the holder `dyn ::p::Tr<Rel>` or the reference to it: the holder is
    // not at fault, and has no cause of its own that depends on the context.
    let item: ItemTrait = parse_quote!(
        trait Tr {
            fn f<'b>(&self, x: &'b dyn ::p::Tr<Rel>) -> &'b ();
        }
    );
    let mut leaker = allowing();
    leaker.intern_trait(&item, true);
    let result = leaker.result();
    let items = result.errors();
    assert_eq!(items.len(), 1);
    assert_item(&items[0], Reason::MaybeConst, "type", "Rel");
    // A bare identifier among the arguments of a holder that `Self` stops from being leaked.
    let mut leaker = allowing();
    leaker.enter_type(&id("S"), &Generics::default(), false, |view| {
        view.visit_type(&parse_quote!(::l::W<[Self], ILEN>))
    });
    let result = leaker.result();
    let items = result.errors();
    assert_eq!(items.len(), 1);
    assert_item(&items[0], Reason::MaybeConst, "type", "ILEN");
    assert_eq!(
        note_at(&items[0], "Self"),
        Some("it cannot be leaked as a whole because of `Self`")
    );
}

#[test]
fn the_part_reported_may_be_smaller_than_the_type_that_cannot_be_leaked() {
    // `u8` and `a` depend on the context, and the array type cannot be leaked whole because of the
    // closure. The error is reported at the first, `u8`; the other is a note.
    let item: ItemTrait = parse_quote!(
        trait Tr {
            fn f(
                &self,
                x: [u8; {
                    let c = |a: u8| a;
                    3
                }],
            );
        }
    );
    let mut leaker = Leaker::new();
    leaker.intern_trait(&item, true);
    let result = leaker.result();
    let items = result.errors();
    assert_eq!(items.len(), 1);
    assert_item(&items[0], Reason::RelativePath, "type", "u8");
    assert_eq!(
        note_at(&items[0], "a"),
        Some("`a` also depends on the type context")
    );
    // Where `u8` is allowed, `a` is the first. A closure cannot be written in the impl of the
    // Repeater, so it is an error naming the closure, and the array type is no entry.
    let mut leaker = allowing();
    leaker.intern_trait(&item, true);
    let result = leaker.result();
    let items = result.errors();
    assert_eq!(items.len(), 1);
    assert_item(&items[0], Reason::RelativePath, "expr", "a");
    assert_eq!(note_at(&items[0], "a"), None);
    assert_eq!(items[0].message, "use absolute path: 'a'");
    assert_eq!(
        note_texts(&items[0]),
        ["it cannot be leaked as a whole because of `| a : u8 | a`"]
    );
    // Without the closure the array type is an entry.
    let item: ItemTrait = parse_quote!(
        trait Tr {
            fn f(
                &self,
                x: [u8; {
                    let c = 3;
                    c
                }],
            );
        }
    );
    let mut leaker = allowing();
    leaker.intern_trait(&item, true);
    assert_eq!(leaker.result().finish().unwrap().iter().count(), 1);
}

#[test]
fn a_bound_with_a_binder_is_unreadable() {
    // No node of the syntax is the bound as a whole: a path is the trait without the binder.
    let item: ItemTrait = parse_quote!(
        trait Tr<F>
        where
            F: for<'a> Fn(&'a u8),
        {
        }
    );
    let mut leaker = Leaker::new();
    leaker.intern_trait(&item, true);
    let result = leaker.result();
    let items = result.errors();
    assert_eq!(items.len(), 1);
    assert_item(
        &items[0],
        Reason::RelativePath,
        "unreadable",
        "for < 'a > Fn (& 'a u8)",
    );
}

#[test]
fn a_bound_in_parentheses_is_unreadable() {
    let item: ItemTrait = parse_quote!(
        trait Tr: (Clone) {}
    );
    let mut leaker = Leaker::new();
    leaker.intern_trait(&item, true);
    let result = leaker.result();
    let items = result.errors();
    assert_eq!(items.len(), 1);
    assert_item(&items[0], Reason::RelativePath, "unreadable", "(Clone)");
}

#[test]
fn a_lifetime_is_a_lifetime() {
    // `'a` is not declared by any scope: the error is reported at it, as it always was.
    let items = errors(lifetime);
    assert_eq!(items.len(), 1);
    assert_item(&items[0], Reason::RelativePath, "lifetime", "'a");
    assert!(matches!(&items[0].cause, Cause::Lifetime(lifetime) if lifetime.ident == "a"));
}

#[test]
fn the_path_of_a_qualified_type_is_unreadable() {
    // `Tr<u8>::X` is the trait with the name that follows the `>`: a path that is no node, and not
    // a type (`syn` has the `qself` and the path apart).
    let leak = |ty: Type| {
        let mut leaker = Leaker::new();
        leaker.enter_impl(
            &Generics::default(),
            &parse_quote!(B),
            None,
            false,
            |view| view.visit_type(&ty),
        );
        leaker.result().errors().to_vec()
    };
    let items = leak(parse_quote!(<Self as super::Tr<u8>>::X));
    assert_eq!(items.len(), 1);
    assert_item(
        &items[0],
        Reason::RelativePath,
        "unreadable",
        "super :: Tr < u8 > :: X",
    );
    // `Self` cannot be leaked here, so the type cannot be leaked as a whole, and its trait is
    // relative. `u8` in the arguments of the trait is leaked alone: the notes do not name it.
    assert_eq!(items[0].message, "use absolute path: 'super::Tr<u8>::X'");
    assert_eq!(
        note_texts(&items[0]),
        ["it cannot be leaked as a whole because of `Self`"]
    );
    // The type around it is written as it is, `Rel` with it, and what it holds has an error of its
    // own (`Tr`, with `Self`).
    let items = leak(parse_quote!(Rel<<Self as Tr<u8>>::X>));
    assert_eq!(items.len(), 2);
    assert_eq!(
        items[0].message,
        "use absolute path: 'Rel<<Self as Tr<u8>>::X>'"
    );
    assert_eq!(items[1].message, "use absolute path: 'Tr<u8>::X'");
    for item in &items {
        assert_eq!(
            note_texts(item),
            ["it cannot be leaked as a whole because of `Self`"]
        );
    }
    let items = type_errors(allowing(), parse_quote!(<T as RelTrait<'_>>::Ty));
    assert_eq!(items.len(), 1);
    assert_item(
        &items[0],
        Reason::RelativePath,
        "unreadable",
        "RelTrait < '_ > :: Ty",
    );
}

// ── the serialized form ──────────────────────────────────────────────────────────────────────

/// Every error above, by the name of the trigger, and the errors of the other kinds of cause.
fn every_trigger() -> Vec<(&'static str, Trigger)> {
    vec![
        ("relative_path", relative_path),
        ("self_without_meaning", self_without_meaning),
        ("unreadable", unreadable),
        ("maybe_const", maybe_const),
        ("unconstrained_lifetime", unconstrained_lifetime),
        ("self_projection", self_projection),
        ("unnameable_self", unnameable_self),
        ("self_in_default", self_in_default),
        ("denied_self_in_bounds", denied_self_in_bounds),
        ("nested_self_in_bounds", nested_self_in_bounds),
        ("unprovable_bound", unprovable_bound),
        ("self_in_trait_predicate", self_in_trait_predicate),
        ("self_in_impl_self_type", self_in_impl_self_type),
        ("values", values),
        ("macros", macros_and_qualified_traits),
        ("lifetime", lifetime),
    ]
}

/// The errors of every trigger.
fn every_error() -> Vec<(&'static str, Vec<ErrorItem>)> {
    every_trigger()
        .into_iter()
        .map(|(name, trigger)| (name, errors(trigger)))
        .collect()
}

/// Reads the items that were written one after the other.
fn read_all(tokens: proc_macro2::TokenStream) -> Result<Vec<ErrorItem>> {
    use syn::parse::Parser;
    (|input: parse::ParseStream| -> Result<Vec<ErrorItem>> {
        let mut items = Vec::new();
        while !input.is_empty() {
            items.push(input.parse()?);
        }
        Ok(items)
    })
    .parse2(tokens)
}

/// Asserts that `read` has the fields of `item`.
#[track_caller]
fn assert_same(read: &ErrorItem, item: &ErrorItem) {
    assert_eq!(kind(&read.cause), kind(&item.cause), "{}", item.message);
    assert_eq!(text(&read.cause), text(&item.cause), "{}", item.message);
    assert_eq!(read.reason, item.reason, "{}", item.message);
    assert_eq!(read.message, item.message);
    assert_eq!(notes_of(read), notes_of(item), "{}", item.message);
    assert_eq!(help_of(read), help_of(item), "{}", item.message);
    assert_eq!(read.in_value, item.in_value, "{}", item.message);
    assert_eq!(read.in_default, item.in_default, "{}", item.message);
    assert_eq!(read.paths, item.paths, "{}", item.message);
}

#[test]
fn the_triggers_have_every_reason_and_every_kind_of_cause() {
    let items: Vec<ErrorItem> = every_error()
        .into_iter()
        .flat_map(|(_, items)| items)
        .collect();
    for reason in [
        Reason::RelativePath,
        Reason::SelfWithoutMeaning,
        Reason::Unreadable,
        Reason::MaybeConst,
        Reason::UnconstrainedLifetime,
        Reason::SelfProjection,
        Reason::UnnameableSelf,
        Reason::SelfInDefault,
        Reason::DeniedSelfInBounds,
        Reason::NestedSelfInBounds,
        Reason::UnprovableBound,
        Reason::SelfInTraitPredicate,
        Reason::SelfInImplSelfType,
    ] {
        assert!(
            items.iter().any(|item| item.reason == reason),
            "{:?}",
            reason
        );
    }
    for expected in ["type", "trait", "expr", "macro", "lifetime", "unreadable"] {
        assert!(
            items.iter().any(|item| kind(&item.cause) == expected),
            "{}",
            expected
        );
    }
}

#[test]
fn every_error_is_written_and_read_back() {
    for (name, items) in every_error() {
        assert!(!items.is_empty(), "{}", name);
        for item in &items {
            let tokens = item.to_token_stream();
            let read: ErrorItem =
                parse2(tokens.clone()).unwrap_or_else(|e| panic!("{}: {}: {}", name, tokens, e));
            // The tokens that are written again are the same, and so are the fields.
            assert_eq!(
                read.to_token_stream().to_string(),
                tokens.to_string(),
                "{}",
                name
            );
            assert_same(&read, item);
        }
        // Several items follow one another, and are read in a loop.
        let all: proc_macro2::TokenStream = items.iter().map(ToTokens::to_token_stream).collect();
        let read = read_all(all.clone()).unwrap();
        assert_eq!(read.len(), items.len(), "{}", name);
        for (read, item) in read.iter().zip(&items) {
            assert_same(read, item);
        }
        let again: proc_macro2::TokenStream = read.iter().map(ToTokens::to_token_stream).collect();
        assert_eq!(again.to_string(), all.to_string(), "{}", name);
    }
}

#[test]
fn the_serialized_form_is_one_brace_group_for_each_item() {
    let items = errors(relative_path);
    // The tokens of the cause, of the notes and of the help are written as they are, the message,
    // the notes and the help as string literals.
    assert_eq!(
        items[1].to_token_stream().to_string(),
        "{ type (MyType < '_ >) RelativePath \"use absolute path: 'MyType<'_>'\" \
         [('_) \"it cannot be leaked as a whole because of `'_`\"] \
         (MyType) \"write `MyType` with an absolute path (e.g. `::my_crate::MyType`)\" paths [type (MyType)] }"
    );
    // A serialization is not a report: it is no `compile_error!`.
    let all: proc_macro2::TokenStream = items.iter().map(ToTokens::to_token_stream).collect();
    assert!(!all.to_string().contains("compile_error"));
    assert_eq!(all.into_iter().count(), items.len());
}

/// Parses a hand-written item.
fn parse_item(source: &str) -> Result<ErrorItem> {
    parse_str(source)
}

#[test]
fn the_help_is_optional_and_so_are_the_notes() {
    let item = parse_item(r#"{ type ( A ) RelativePath "m" [] }"#).unwrap();
    assert_item(&item, Reason::RelativePath, "type", "A");
    assert!(item.notes.is_empty() && item.help.is_none());
    let item = parse_item(
        r#"{ type ( A ) RelativePath "m" [ ( B ) "why" , ( C D ) "also" , ] ( E ) "do" }"#,
    )
    .unwrap();
    assert_eq!(
        notes_of(&item),
        [
            ("B".to_string(), "why".to_string()),
            ("C D".to_string(), "also".to_string())
        ]
    );
    assert_eq!(help_of(&item), Some(("E".to_string(), "do".to_string())));
    // And the items follow one another.
    let two = read_all(
        r#"{ type ( A ) RelativePath "m" [] } { trait ( B ) Unreadable "n" [] ( B ) "h" }"#
            .parse()
            .unwrap(),
    )
    .unwrap();
    assert_eq!(two.len(), 2);
    assert_item(&two[1], Reason::Unreadable, "trait", "B");
}

#[test]
fn the_place_and_the_paths_follow_the_help_and_are_written_only_if_set() {
    // Neither is written for an error that has no place and no path.
    let plain = parse_item(r#"{ type ( A ) RelativePath "m" [] ( B ) "h" }"#).unwrap();
    assert!(!plain.in_value && !plain.in_default && plain.paths.is_empty());
    assert_eq!(
        plain.to_token_stream().to_string(),
        r#"{ type (A) RelativePath "m" [] (B) "h" }"#
    );
    // After the help, or after the notes if there is none, in this order.
    let placed = parse_item(
        r#"{ type ( A ) RelativePath "m" [] ( B ) "h" value default paths [ type ( a :: b ) , trait ( :: c ) , expr ( Self ) , macro ( crate :: d ! ) , unknown ( e ) ] }"#,
    )
    .unwrap();
    assert!(placed.in_value && placed.in_default);
    let paths: Vec<String> = placed.paths.iter().map(text_of).collect();
    assert_eq!(paths, ["a :: b", ":: c", "Self", "crate :: d !", "e"]);
    assert!(matches!(
        &placed.paths[..],
        [
            Cause::Type(_),
            Cause::Trait(_),
            Cause::Expr(_),
            Cause::Macro(_),
            Cause::Unknown(_)
        ]
    ));
    assert_eq!(
        placed.to_token_stream().to_string(),
        r#"{ type (A) RelativePath "m" [] (B) "h" value default paths [type (a :: b) , trait (:: c) , expr (Self) , macro (crate :: d !) , unknown (e)] }"#
    );
    let only_default = parse_item(r#"{ type ( A ) RelativePath "m" [] default }"#).unwrap();
    assert!(!only_default.in_value && only_default.in_default && only_default.help.is_none());
    assert_eq!(
        only_default.to_token_stream().to_string(),
        r#"{ type (A) RelativePath "m" [] default }"#
    );
    // Empty paths are the same as none, in any order of the words.
    let empty = parse_item(r#"{ type ( A ) RelativePath "m" [] paths [] default value }"#).unwrap();
    assert!(empty.in_value && empty.in_default && empty.paths.is_empty());
    assert_eq!(
        empty.to_token_stream().to_string(),
        r#"{ type (A) RelativePath "m" [] value default }"#
    );
}

#[test]
fn every_kind_of_cause_is_read_whole() {
    let read = |kind: &str, cause: &str| {
        parse_item(&format!(
            r#"{{ {} ( {} ) RelativePath "m" [] }}"#,
            kind, cause
        ))
    };
    // A type, a trait path (with the sugar of `Fn`), a value, a path as a type has it, a macro.
    assert_item(
        &read("type", "dyn A + B").unwrap(),
        Reason::RelativePath,
        "type",
        "dyn A + B",
    );
    assert_item(
        &read("trait", "Fn(u8) -> u8").unwrap(),
        Reason::RelativePath,
        "trait",
        "Fn (u8) -> u8",
    );
    assert_item(
        &read("trait", ":: a :: B < C >").unwrap(),
        Reason::RelativePath,
        "trait",
        ":: a :: B < C >",
    );
    assert_item(
        &read("expr", "a :: b").unwrap(),
        Reason::RelativePath,
        "expr",
        "a :: b",
    );
    let value = read("expr", "Rel < u8 >").unwrap();
    assert_item(&value, Reason::RelativePath, "expr", "Rel < u8 >");
    assert!(matches!(&value.cause, Cause::Expr(Expr::Path(path)) if path.qself.is_none()));
    assert_item(
        &read("expr", "< u8 as Tr > :: f").unwrap(),
        Reason::RelativePath,
        "expr",
        "< u8 as Tr > :: f",
    );
    // Any expression, a closure for example, and a lifetime.
    let closure = read("expr", "| a : u8 | a").unwrap();
    assert_item(&closure, Reason::RelativePath, "expr", "| a : u8 | a");
    assert!(matches!(&closure.cause, Cause::Expr(Expr::Closure(_))));
    let lifetime = read("lifetime", "'a").unwrap();
    assert_item(&lifetime, Reason::RelativePath, "lifetime", "'a");
    assert!(matches!(&lifetime.cause, Cause::Lifetime(lifetime) if lifetime.ident == "a"));
    // Written again, they are read the same.
    for item in [&closure, &lifetime] {
        let again: ErrorItem = parse2(item.to_token_stream()).unwrap();
        assert_same(&again, item);
    }
    assert_item(
        &read("macro", "a :: mac !").unwrap(),
        Reason::RelativePath,
        "macro",
        "a :: mac !",
    );
    assert_item(
        &read("unknown", "a :: b").unwrap(),
        Reason::RelativePath,
        "unknown",
        "a :: b",
    );
    // Any tokens, groups included.
    assert_item(
        &read("unreadable", "any [ tokens ] { ( here ) } ;").unwrap(),
        Reason::RelativePath,
        "unreadable",
        "any [tokens] { (here) } ;",
    );
}

#[test]
fn a_malformed_item_is_refused() {
    // The source, and a part of the message where the error says what is wrong.
    let refused: &[(&str, Option<&str>)] = &[
        // A cause that is not the whole content of its group.
        (r#"{ type ( A B ) RelativePath "m" [] }"#, None),
        (r#"{ trait ( A < B > C ) RelativePath "m" [] }"#, None),
        (r#"{ macro ( mac ) RelativePath "m" [] }"#, None),
        (r#"{ macro ( mac ! x ) RelativePath "m" [] }"#, None),
        (r#"{ unknown ( a b ) RelativePath "m" [] }"#, None),
        // An unknown kind, reason or field (`nested` was a field of the old serialization).
        (
            r#"{ statement ( A ) RelativePath "m" [] }"#,
            Some("unknown kind `statement`"),
        ),
        (
            r#"{ type ( A ) Nonsense "m" [] }"#,
            Some("unknown reason `Nonsense`"),
        ),
        (
            r#"{ type ( A ) RelativePath "m" [] values }"#,
            Some("unknown field `values`"),
        ),
        (
            r#"{ type ( A ) RelativePath "m" [] nested [ { type ( B ) Unreadable "n" [] } ] }"#,
            Some("unknown field `nested`"),
        ),
        // Something after the help, a message that is no string, no group around the item.
        (r#"{ type ( A ) RelativePath "m" [] ( B ) "h" "x" }"#, None),
        (r#"{ type ( A ) RelativePath m [] }"#, None),
        (r#"type ( A ) RelativePath "m" []"#, None),
        // A path of the list that has arguments, or is no path, or is of a kind with no path.
        (
            r#"{ type ( A ) RelativePath "m" [] paths [ type ( a < b > ) ] }"#,
            None,
        ),
        (
            r#"{ type ( A ) RelativePath "m" [] paths [ unknown ( a < b > ) ] }"#,
            None,
        ),
        (
            r#"{ type ( A ) RelativePath "m" [] paths [ unknown ( & a ) ] }"#,
            None,
        ),
        (r#"{ type ( A ) RelativePath "m" [] paths [ a ] }"#, None),
        (
            r#"{ type ( A ) RelativePath "m" [] paths [ unreadable ( a ) ] }"#,
            None,
        ),
        (
            r#"{ type ( A ) RelativePath "m" [] paths [ lifetime ( 'a ) ] }"#,
            None,
        ),
        (r#"{ type ( A ) RelativePath "m" [] paths a }"#, None),
    ];
    for (source, message) in refused {
        let error = parse_item(source).expect_err(source);
        if let Some(message) = message {
            assert!(error.to_string().contains(message), "{}: {}", source, error);
        }
    }
    // A path of the list that is a path is read.
    for source in [
        r#"{ type ( A ) RelativePath "m" [] paths [ type ( a ) ] }"#,
        r#"{ type ( A ) RelativePath "m" [] paths [ unknown ( a ) ] }"#,
    ] {
        assert!(parse_item(source).is_ok(), "{}", source);
    }
}

// ── the report ───────────────────────────────────────────────────────────────────────────────

#[test]
fn an_item_has_a_message_notes_and_help_and_is_an_error() {
    let items = errors(relative_path);
    let item = &items[1];
    assert_eq!(item.to_string(), item.message);
    assert_eq!(
        item.notes.first().map(|(_, n)| n.as_str()),
        Some("it cannot be leaked as a whole because of `'_`")
    );
    assert_eq!(
        item.help_text(),
        Some("write `MyType` with an absolute path (e.g. `::my_crate::MyType`)")
    );
    // It is a `std::error::Error`, and `From<ErrorItem>` is the `syn::Error` of the layout of
    // rustc: the notes, then the help, after the message.
    let error: Box<dyn std::error::Error> = Box::new(item.clone());
    assert_eq!(error.to_string(), item.message);
    let syn_error: syn::Error = item.clone().into();
    assert_eq!(
        syn_error.to_string(),
        "use absolute path: 'MyType<'_>'\n\n\
         \x20 = note: it cannot be leaked as a whole because of `'_`\n\
         \x20 = help: write `MyType` with an absolute path (e.g. `::my_crate::MyType`)\n"
    );
}

/// `to_compile_error()` is the error of `syn`, and it is pure; `emit()` is the same outside a proc
/// macro, on every toolchain (inside one, nightly's `emit()` reports a `Diagnostic` instead:
/// `tests/diagnostic.rs`).
#[test]
fn to_compile_error_is_the_error_of_syn_and_emit_is_the_same_outside_a_macro() {
    let items = errors(relative_path);
    for item in &items {
        let tokens = item.to_compile_error();
        let expected = syn::Error::from(item.clone()).to_compile_error();
        assert_eq!(tokens.to_string(), expected.to_string());
        assert!(tokens.to_string().contains("compile_error"));
        assert!(
            tokens.to_string().contains(item.help_text().unwrap()),
            "{}",
            tokens
        );
        assert_eq!(item.to_compile_error().to_string(), tokens.to_string());
        assert_eq!(item.emit().to_string(), tokens.to_string());
    }
}

#[test]
fn to_syn_error_combines_the_items_in_order() {
    let items = errors(relative_path);
    let combined = to_syn_error(&items);
    let errors: Vec<String> = combined.into_iter().map(|e| e.to_string()).collect();
    assert_eq!(errors.len(), 2);
    for (error, item) in errors.iter().zip(&items) {
        assert_eq!(*error, syn::Error::from(item.clone()).to_string());
    }
    // Items of several analyses: one error for each.
    let all: Vec<ErrorItem> = every_error()
        .into_iter()
        .flat_map(|(_, items)| items)
        .collect();
    assert_eq!(to_syn_error(&all).into_iter().count(), all.len());
    // Nothing to report is an error at the call site, that tells it.
    let none = to_syn_error(&[]);
    assert_eq!(none.to_string(), "not internable");
    assert_eq!(none.into_iter().count(), 1);
}

#[test]
fn the_span_of_an_item_is_that_of_its_cause() {
    let items = errors(self_without_meaning);
    for item in &items {
        // Outside a proc macro the spans are all the call site: they are equal.
        assert_eq!(
            format!("{:?}", item.span()),
            format!(
                "{:?}",
                item.cause
                    .to_token_stream()
                    .into_iter()
                    .next()
                    .unwrap()
                    .span()
            )
        );
    }
}

// ── accepting the causes ─────────────────────────────────────────────────────────────────────

/// The messages of the errors, for a failure to say what is left.
fn messages(items: &[ErrorItem]) -> Vec<&str> {
    items.iter().map(|item| item.message.as_str()).collect()
}

/// The result of a trigger with these causes allowed.
fn allowing_causes(trigger: Trigger, causes: &[Cause]) -> LeakerResult {
    let mut result = trigger();
    for cause in causes {
        result = result.accept_cause(cause);
    }
    result
}

/// The causes of the errors, to give to `LeakerResult::allow()`.
fn causes_of(errors: &[ErrorItem]) -> Vec<Cause> {
    errors.iter().map(|error| error.cause.clone()).collect()
}

/// Asserts that two Referrers are the same, as values and as tokens.
#[track_caller]
fn assert_same_referrer(recovered: &Referrer, allowed: &Referrer, what: &str) {
    assert_eq!(recovered, allowed, "{}", what);
    assert_eq!(
        recovered.to_token_stream().to_string(),
        allowed.to_token_stream().to_string(),
        "{}",
        what
    );
}

#[test]
fn allowing_the_causes_of_every_trigger_accepts_the_item_and_leaves_the_recovered_referrer() {
    for (name, trigger) in every_trigger() {
        let result = trigger();
        let items = result.errors().to_vec();
        let causes = causes_of(&items);
        assert!(!causes.is_empty(), "{}", name);
        let referrer = result.into_referrer();
        // The same analysis with the causes allowed reports nothing, and has the same entries and
        // parts: allowing drops errors only.
        let (allowed, _) = allow_every_cause(allowing_causes(trigger, &causes));
        if !allowed.errors().is_empty() {
            panic!("{}: {:?} are left", name, messages(allowed.errors()));
        }
        assert_same_referrer(&referrer, &allowed.into_referrer(), name);
        // And it is a Referrer like any other: it round-trips.
        let parsed: Referrer = parse2(referrer.to_token_stream()).unwrap();
        assert_eq!(parsed, referrer, "{}", name);
        // Each cause is needed: without it, its errors, and only they, are reported.
        for cause in &causes {
            let others: Vec<Cause> = causes.iter().filter(|c| *c != cause).cloned().collect();
            let result = allowing_causes(trigger, &others);
            let expected: Vec<&str> = items
                .iter()
                .filter(|item| item.cause == *cause)
                .map(|item| item.message.as_str())
                .collect();
            assert_eq!(
                messages(result.errors()),
                expected,
                "{}: {}",
                name,
                text(cause)
            );
        }
    }
}

// ── the same rules before and after the analysis ────────────────────────────────────────────

/// The errors of a result, as the tokens that serialize them.
fn serialized(result: &LeakerResult) -> Vec<String> {
    result
        .errors()
        .iter()
        .map(|error| error.to_token_stream().to_string())
        .collect()
}

/// Asserts that the rules give the same errors and the same entries given to the config of a
/// trigger and given to its result. The Referrer of the config may have fewer parts: a rule that
/// allows a path (a cause that is one, a place) makes the analysis write it as it is without a
/// conflict, where the result found the conflict, dropped its error and kept the part.
#[track_caller]
fn assert_stages_agree(name: &str, trigger: Trigger, rules: &[Rule]) -> usize {
    let before = with_rules(rules, trigger);
    let mut after = trigger();
    let dropped = after.errors().len();
    for rule in rules {
        after = rule.on_result(after);
    }
    let dropped = dropped - after.errors().len();
    assert_eq!(serialized(&before), serialized(&after), "{}", name);
    let (before, after) = (before.into_referrer(), after.into_referrer());
    let entries = |referrer: &Referrer| -> Vec<String> {
        common::types(referrer)
            .map(|ty| ty.to_token_stream().to_string())
            .collect()
    };
    assert_eq!(entries(&before), entries(&after), "{}", name);
    let length = |referrer: &Referrer| referrer.to_token_stream().to_string().len();
    assert!(length(&before) <= length(&after), "{}", name);
    dropped
}

#[test]
fn the_cause_value_and_default_rules_give_the_same_errors_before_and_after_the_analysis() {
    let (mut by_cause, mut by_value, mut by_default) = (0, 0, 0);
    for (name, trigger) in every_trigger() {
        // An error that has paths is accepted by them, which are path rules in the config too: the
        // result drops the error, and the config may find other errors (`tests/allow_stages.rs`).
        // The cause of an error without paths is the same rule at both stages.
        let errors = trigger().errors().to_vec();
        for error in errors.iter().filter(|error| !error.paths.is_empty()) {
            let left = trigger().accept_cause(&error.cause);
            assert!(
                left.errors().iter().all(|other| other.cause != error.cause),
                "{}: {}",
                name,
                text(&error.cause)
            );
        }
        let causes: Vec<Cause> = errors
            .iter()
            .filter(|error| error.paths.is_empty())
            .map(|error| error.cause.clone())
            .collect();
        for cause in &causes {
            let rule = Rule::Cause(causes_accepting(&errors, cause));
            by_cause += assert_stages_agree(name, trigger, &[rule]);
        }
        let all: Vec<Rule> = causes
            .iter()
            .map(|cause| Rule::Cause(causes_accepting(&errors, cause)))
            .collect();
        by_cause += assert_stages_agree(name, trigger, &all);
        by_value += assert_stages_agree(name, trigger, &[Rule::Values]);
        by_default += assert_stages_agree(name, trigger, &[Rule::Default]);
        let everything = [all, vec![Rule::Values, Rule::Default]].concat();
        assert_stages_agree(name, trigger, &everything);
    }
    // The test drops something with each kind of rule.
    assert!(by_cause > 0 && by_value > 0 && by_default > 0);
}

#[test]
fn a_path_rule_leaves_the_errors_that_have_a_path_it_does_not_allow() {
    // The paths of the errors of the triggers: a rule drops an error only if each is allowed.
    for (name, trigger) in every_trigger() {
        for error in trigger().errors() {
            let paths: Vec<String> = error.paths.iter().map(text_of).collect();
            // A path is relative: no leading `::`, no `Self`, no `$crate`.
            for entry in &error.paths {
                let path = entry.path().expect("a path");
                assert!(path.leading_colon.is_none(), "{}: {:?}", name, paths);
                assert!(path.segments.iter().all(|seg| seg.arguments.is_empty()));
                assert_ne!(text_of(entry), "Self", "{}: {:?}", name, paths);
            }
        }
    }
    // An error of `Self` without meaning or of a lifetime has no path to allow.
    for error in self_without_meaning()
        .errors()
        .iter()
        .chain(lifetime().errors())
    {
        assert!(error.paths.is_empty(), "{}", error.message);
    }
    // Allowing the first path of an error that has several leaves it.
    let result = relative_path();
    let left = result.clone().allow(starts_with(parse_quote!(MyType)));
    assert_eq!(left.errors().len(), 1);
    assert_eq!(left.errors()[0].reason, Reason::RelativePath);
    assert_eq!(text(&left.errors()[0].cause), "Clone");
}

// ── recovery ─────────────────────────────────────────────────────────────────────────────────

#[test]
fn recovery_keeps_the_entries_that_do_not_conflict() {
    let item: ItemStruct = parse_quote!(
        struct S {
            a: Rel,
            b: ::p::W<'_, dyn RelTr>,
        }
    );
    let mut leaker = allowing();
    let id = leaker.intern_struct(&item, true);
    let mut result = leaker.result();
    assert_eq!(result.errors().len(), 1);
    assert_eq!(result.errors()[0].cause, Cause::Trait(parse_quote!(RelTr)));
    let referrer = result.clone().into_referrer();
    // `Rel` is an entry, and `dyn RelTr` is copied as written: a part, not an entry.
    let entries: Vec<String> = common::types(&referrer).map(text_of).collect();
    assert_eq!(entries, ["Rel"]);
    assert!(
        referrer
            .to_token_stream()
            .to_string()
            .contains("[type (dyn RelTr) "),
        "{}",
        referrer.to_token_stream()
    );
    let mut copy = item.clone();
    referrer
        .visitor(id, |_, ix, _| parse_quote!(Entry<#ix>))
        .visit_item_struct_mut(&mut copy);
    let expected: ItemStruct = parse_quote!(
        struct S {
            a: Entry<0usize>,
            b: ::p::W<'_, dyn RelTr>,
        }
    );
    assert_eq!(copy, expected);
    // Allowing the cause drops the error and leaves the Referrer.
    result = result.accept_cause(&Cause::Trait(parse_quote!(RelTr)));
    assert_same_referrer(&referrer, &result.finish().unwrap(), "S");
}

/// The tokens of a type, as text.
fn text_of(ty: &impl ToTokens) -> String {
    ty.to_token_stream().to_string()
}

#[test]
fn the_errors_of_self_are_reported_with_the_others() {
    // `Self` has no meaning in generics: the mention is told with the conflict, in the order in
    // which the syntax has them, and both are recovered from.
    let item: ItemStruct = parse_quote!(
        struct S {
            rel: ::p::W<'_, dyn RelTr>,
            me: Wrap<Self>,
        }
    );
    let analyze = || {
        let mut leaker = allowing();
        leaker.enter_generics(&Generics::default(), |view| {
            syn::visit::visit_item_struct(view, &item)
        });
        leaker.result()
    };
    let result = analyze();
    let errors = result.errors().to_vec();
    assert_eq!(
        errors
            .iter()
            .map(|error| (error.reason, text(&error.cause)))
            .collect::<Vec<_>>(),
        [
            (Reason::RelativePath, "RelTr".to_string()),
            (Reason::SelfWithoutMeaning, "Self".to_string())
        ]
    );
    let referrer = result.clone().into_referrer();
    // The conflict is copied as written, and so is the type that names `Self`: no entry is left.
    assert_eq!(referrer.iter().count(), 0);
    let mut allowed = analyze();
    for cause in causes_of(&errors) {
        allowed = allowed.accept_cause(&cause);
    }
    assert!(allowed.errors().is_empty());
    assert_same_referrer(&referrer, &allowed.into_referrer(), "Self");
    // Allowing the conflict leaves the mention, and the other way round.
    let left = analyze().accept_cause(&errors[0].cause);
    assert_eq!(left.errors().len(), 1);
    assert_eq!(left.errors()[0].reason, Reason::SelfWithoutMeaning);
    let left = analyze().accept_cause(&errors[1].cause);
    assert_eq!(left.errors().len(), 1);
    assert_eq!(left.errors()[0].reason, Reason::RelativePath);
}

// ── inputs that cannot be leaked ────────────────────────────────────────────────────────────

mod limits {
    //! Inputs that cannot be leaked, and how the errors for them read.
    //!
    //! - `limit_*`: inputs which cannot be leaked on stable Rust. `type-leak` must reject them with
    //!   [`ErrorItem`]s.
    //! - `limit_*`, recording behavior: inputs which cannot be handled exactly without name resolution
    //!   (e.g. shadowed primitive names), and how they are detected.

    use super::{errors_in, kind, notes_of};
    use std::collections::BTreeSet;
    use syn::*;
    use template_quote::{quote, ToTokens};
    use type_leak::*;

    fn leaker_config() -> Rules<impl Allow + Clone> {
        Rules::new().primitive().crate_paths()
    }

    /// The messages of the errors, one per line.
    fn messages(errors: &[ErrorItem]) -> String {
        errors
            .iter()
            .map(|item| item.message.as_str())
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// The errors that `leaker` has after `analyze` ran on it, for a test that expects an error.
    fn expect_errors<R>(leaker: Leaker, analyze: impl FnOnce(&mut Leaker) -> R) -> Vec<ErrorItem> {
        let errors = errors_in(leaker, analyze);
        assert!(!errors.is_empty(), "accepted");
        errors
    }

    /// Runs a Leaker on `item_trait`, returning the leaked types or the error message.
    fn leak_trait(item_trait: &ItemTrait) -> std::result::Result<BTreeSet<String>, String> {
        let mut leaker = Leaker::from_config(leaker_config(), Options::new());
        leaker.intern_trait(item_trait, true);
        leaker.reduce_roots();
        let referrer = leaker
            .result()
            .finish()
            .map_err(|errors| messages(&errors))?;
        Ok(crate::common::types(&referrer)
            .map(|t| quote!(#t).to_string())
            .collect())
    }

    // ── limitations ──────────────────────────────────────────────────────────────

    #[test]
    fn limit_const_in_bound() {
        // A relative const in a trait bound cannot be leaked, because a const cannot be recovered
        // from an associated type in a generic context on stable Rust.
        let item: ItemTrait = parse2(quote! {
            pub trait MyTrait: crate::l::Sup<{ SIZE }> {
                fn f(&self);
            }
        })
        .unwrap();
        assert_eq!(
            leak_trait(&item),
            Err("use absolute path: 'SIZE'".to_string())
        );

        // The suggested fix works: an absolute path does not need to be leaked.
        let item: ItemTrait = parse2(quote! {
            pub trait MyTrait: crate::l::Sup<{ ::my_crate::SIZE }> {
                fn f(&self);
            }
        })
        .unwrap();
        assert_eq!(leak_trait(&item), Ok(BTreeSet::new()));
    }

    #[test]
    fn limit_const_param_ty_alias() {
        // The type of a const parameter cannot be a projection, so a relative type alias cannot be
        // leaked there.
        let item: ItemTrait = parse2(quote! {
            pub trait MyTrait<const N: Idx> {
                fn f(&self) -> RelType;
            }
        })
        .unwrap();
        assert_eq!(
            leak_trait(&item),
            Err("use absolute path: 'Idx'".to_string())
        );

        // The suggested fix works: an absolute path to the alias does not need to be leaked.
        let abs_item: ItemTrait = parse2(quote! {
            pub trait MyTrait<const N: ::my_crate::Idx> {
                fn f(&self) -> RelType;
            }
        })
        .unwrap();
        assert_eq!(
            leak_trait(&abs_item),
            Ok(vec!["RelType".to_string()].into_iter().collect())
        );

        // `encode_slot_const()` cannot encode an alias (relative or absolute), because it cannot
        // know which slot to use.
        let repeater: Path = parse2(quote!(crate::Repeater)).unwrap();
        let alias = match &abs_item.generics.params[0] {
            GenericParam::Const(param) => param,
            _ => unreachable!(),
        };
        let err = encode_slot_const(&repeater, &parse_quote!(N), &alias.ty).unwrap_err();
        assert_eq!(
            err.to_string(),
            "unsupported type of const parameter: ':: my_crate :: Idx'"
        );
    }

    // ── holes ────────────────────────────────────────────────────────────────────

    #[test]
    fn bare_identifier_in_bound_is_leaked_as_a_type() {
        // A bare relative identifier in a trait bound (e.g. `SIZE` in `Sup<SIZE>`) is parsed as a
        // type by syn, though it may be a const. There is no type to intern it in (unlike
        // `Wrap<SIZE>`, which is interned as a whole, see `tests/reduce_roots.rs`), so it is leaked as
        // a type whatever its name is. If it is a const, the result does not compile: it should be
        // written as `Sup<{ SIZE }>`, which is an error (see `limit_const_in_bound`), or with an
        // absolute path.
        for item in [
            parse2::<ItemTrait>(quote! {
                pub trait MyTrait: crate::l::Sup<SIZE> {
                    fn f(&self);
                }
            })
            .unwrap(),
            parse2::<ItemTrait>(quote! {
                pub trait MyTrait {
                    fn f<T: crate::l::Sup<SIZE>>(&self);
                }
            })
            .unwrap(),
        ] {
            assert_eq!(
                leak_trait(&item),
                Ok(vec!["SIZE".to_string()].into_iter().collect())
            );
        }

        // So is a relative type.
        let item: ItemTrait = parse2(quote! {
            pub trait MyTrait: crate::l::Sup<MyType> {
                fn f(&self);
            }
        })
        .unwrap();
        assert_eq!(
            leak_trait(&item),
            Ok(vec!["MyType".to_string()].into_iter().collect())
        );

        // The suggested fix works.
        let item: ItemTrait = parse2(quote! {
            pub trait MyTrait: crate::l::Sup<{ ::my_crate::SIZE }> {
                fn f(&self);
            }
        })
        .unwrap();
        assert_eq!(leak_trait(&item), Ok(BTreeSet::new()));
    }

    #[test]
    fn limit_shadowed_primitive() {
        // The type of a const parameter is recognized as a primitive only by its name, so a shadowed
        // primitive name (e.g. `type u8 = i32;` in the Leaker's context) is accepted and copied as it
        // is into the Referrer, where it means the real `u8`. This cannot be known without name
        // resolution, so `Leaker::assertions()` checks it at compile time in the Leaker's context
        // (it fails with "mismatched types" if the name is shadowed).
        let item: ItemTrait = parse2(quote! {
            pub trait MyTrait {
                fn f<const M: u8>(&self);
            }
        })
        .unwrap();
        assert_eq!(leak_trait(&item), Ok(BTreeSet::new()));

        let mut leaker = Leaker::new();
        leaker.intern_trait(&item, true);
        let expected = quote! {
            #[allow(dead_code, non_snake_case)]
            const _: () = {
                fn __type_leak_assert_u8_is_primitive(x: &u8) -> &::core::primitive::u8 { x }
            };
        };
        let (actual, expected): (Item, Item) = (
            parse2(leaker.assertions()).unwrap(),
            parse2(expected).unwrap(),
        );
        assert_eq!(quote!(#actual).to_string(), quote!(#expected).to_string());
        assert!(leaker.result().errors().is_empty());
    }

    #[test]
    fn all_relative_paths_are_reported() {
        // Every relative path which cannot be leaked is reported at once, each as an error.
        let item: ItemTrait = parse2(quote! {
            pub trait MyTrait: Clone + MyBound {
                fn f(&self, x: MyType<'_>);
            }
        })
        .unwrap();
        let err = expect_errors(Leaker::new(), |leaker| leaker.intern_trait(&item, true));
        assert!(err.iter().all(|item| item.reason == Reason::RelativePath));
        // The cause of each is the whole bound or type that is a conflict (a trait is its own path:
        // it needs no note), and what stops the type from being leaked as a whole is a note.
        assert_eq!(
            err.iter()
                .map(|item| (
                    kind(&item.cause),
                    item.cause.to_token_stream().to_string(),
                    notes_of(item)
                ))
                .collect::<Vec<_>>(),
            vec![
                ("trait", "Clone".to_string(), vec![]),
                ("trait", "MyBound".to_string(), vec![]),
                (
                    "type",
                    "MyType < '_ >".to_string(),
                    vec![(
                        "'_".to_string(),
                        "it cannot be leaked as a whole because of `'_`".to_string()
                    )]
                ),
            ]
        );
        assert_eq!(
            messages(&err),
            "use absolute path: 'Clone'\nuse absolute path: 'MyBound'\nuse absolute path: 'MyType<'_>'"
        );
        // The first error and its help.
        assert_eq!(
            err[0].help_text(),
            Some("use `::core::clone::Clone` instead of `Clone`")
        );
        // Each error is a separate `syn::Error`, with its own notes and help.
        let syn_err = to_syn_error(&err);
        assert_eq!(
            syn_err.into_iter().map(|e| e.to_string()).collect::<Vec<_>>(),
            vec![
                "use absolute path: 'Clone'\n\n  = help: use `::core::clone::Clone` instead of `Clone`\n",
                "use absolute path: 'MyBound'\n\n  = help: write `MyBound` with an absolute path (e.g. `::my_crate::MyBound`)\n",
                "use absolute path: 'MyType<'_>'\n\n  = note: it cannot be leaked as a whole because of `'_`\n  = help: write `MyType` with an absolute path (e.g. `::my_crate::MyType`)\n",
            ]
        );
    }

    #[test]
    fn notes_explain_conflicts() {
        // The notes of an error tell why the path cannot be leaked as a whole, and which other paths
        // of the same type depend on the type context (`RelA + RelB<'_>`: the two bounds of a trait
        // object). The types inside have errors of their own. The help tells what to write.
        let item: ItemTrait = parse2(quote! {
            pub trait MyTrait {
                fn f(&self, x: Box<dyn RelA + RelB<'_>>);
            }
        })
        .unwrap();
        let err = expect_errors(Leaker::new(), |leaker| leaker.intern_trait(&item, true));
        // The suggestion of the absolute path is the help.
        assert_eq!(
            err[0].help_text(),
            Some("use `::std::boxed::Box` instead of `Box`")
        );
        assert_eq!(
            err[0].notes.first().map(|(_, n)| n.as_str()),
            Some("it cannot be leaked as a whole because of `'_`")
        );
        assert_eq!(err.len(), 2);
        assert_eq!(
            to_syn_error(&err[..1]).to_string(),
            "use absolute path: 'Box<dyn RelA + RelB<'_>>'\n\n\
             \x20 = note: it cannot be leaked as a whole because of `'_`\n\
             \x20 = help: use `::std::boxed::Box` instead of `Box`\n"
        );
        // The trait object is written as it is too, with its two traits, in an error of its own.
        assert_eq!(err[1].paths.len(), 2);
        assert_eq!(
            to_syn_error(&err[1..]).to_string(),
            "use absolute path: 'RelA'\n\n\
             \x20 = note: it cannot be leaked as a whole because of `'_`\n\
             \x20 = note: `RelB<'_>` also depends on the type context\n\
             \x20 = help: write `RelA` with an absolute path (e.g. `::my_crate::RelA`)\n"
        );
    }
}

// ── the help of every kind of error ──────────────────────────────────────────────────────────

// An error says what is wrong (`ErrorItem::message`), why (`ErrorItem::notes`) and what to write
// instead (`ErrorItem::help`). The help is the only part of an error that a user can act on, so
// every kind of error has one, with the tokens of the error in it:
//
// - `every_kind_of_error_has_a_help` is a table with one row per kind of error (the kinds of
//   `Reason`, and the ways to write a path): the headline and the exact help of each error of the
//   row;
// - `a_help_is_rendered_after_the_notes` asserts how the stable layout (`syn::Error`, which
//   `ErrorItem::to_compile_error()` renders as `compile_error!`) shows it, as `= help: ..` after
//   the `= note: ..`s.

/// One row: the kind of error, how to make it, and the headline and help of each of its errors.
struct Row {
    kind: &'static str,
    error: fn() -> Vec<ErrorItem>,
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
            errors_in(Leaker::new(), |leaker| leaker.intern_trait(&item, true))
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
            errors_in(Leaker::new(), |leaker| leaker.intern_trait(&item, true))
        },
        items: &[(
            "use absolute path: 'usize::MAX'",
            "use `::core::primitive::usize` instead of `usize`",
        )],
    },
    Row {
        kind: "a relative path that is not in the prelude",
        error: || {
            let item: ItemTrait = parse_quote!(
                trait Tr: MyBound {}
            );
            errors_in(Leaker::new(), |leaker| leaker.intern_trait(&item, true))
        },
        items: &[(
            "use absolute path: 'MyBound'",
            "write `MyBound` with an absolute path (e.g. `::my_crate::MyBound`)",
        )],
    },
    Row {
        // `crate::`, `self::` and `super::` name a place in the crate: the example leaves them out.
        kind: "a path of the crate",
        error: || {
            let item: ItemTrait = parse_quote!(
                trait Tr: crate::m::Bound + super::Other {}
            );
            errors_in(Leaker::new(), |leaker| leaker.intern_trait(&item, true))
        },
        items: &[
            (
                "use absolute path: 'crate::m::Bound'",
                "write `crate::m::Bound` with an absolute path (e.g. `::my_crate::m::Bound`)",
            ),
            (
                "use absolute path: 'super::Other'",
                "write `super::Other` with an absolute path (e.g. `::my_crate::Other`)",
            ),
        ],
    },
    Row {
        kind: "a prelude name as the second segment of a relative path",
        error: || {
            let item: ItemTrait = parse_quote!(
                trait Tr: my_mod::Clone {}
            );
            errors_in(Leaker::new(), |leaker| leaker.intern_trait(&item, true))
        },
        items: &[(
            "use absolute path: 'my_mod::Clone'",
            "write `my_mod::Clone` with an absolute path (e.g. `::my_crate::my_mod::Clone`)",
        )],
    },
    Row {
        kind: "a prelude trait in a bound of a method, with a binding",
        error: || {
            let item: ItemTrait = parse_quote!(
                trait Tr {
                    fn f<T: Iterator<Item = u8>>(&self, t: T);
                }
            );
            errors_in(Leaker::new(), |leaker| leaker.intern_trait(&item, true))
        },
        items: &[(
            "use absolute path: 'Iterator<Item = u8>'",
            "use `::core::iter::Iterator` instead of `Iterator`",
        )],
    },
    Row {
        kind: "a prelude trait of a later edition, with an argument",
        error: || {
            let item: ItemTrait = parse_quote!(
                trait Tr: TryFrom<u8> {}
            );
            errors_in(Leaker::new(), |leaker| leaker.intern_trait(&item, true))
        },
        items: &[(
            "use absolute path: 'TryFrom<u8>'",
            "use `::core::convert::TryFrom` instead of `TryFrom`",
        )],
    },
    Row {
        // The relative part of an `impl Trait` is reported, not the whole `impl Trait`.
        kind: "a prelude trait in an `impl Trait`",
        error: || {
            let item: ItemTrait = parse_quote!(
                trait Tr {
                    fn f(&self, x: impl Clone);
                }
            );
            errors_in(Leaker::new(), |leaker| leaker.intern_trait(&item, true))
        },
        items: &[(
            "use absolute path: 'Clone'",
            "use `::core::clone::Clone` instead of `Clone`",
        )],
    },
    Row {
        // The offending tokens are a whole trait bound, not a path.
        kind: "a bound with a binder",
        error: || {
            let item: ItemTrait = parse_quote!(
                trait Tr: for<'a> Fn(&'a u8) {}
            );
            errors_in(Leaker::new(), |leaker| leaker.intern_trait(&item, true))
        },
        items: &[(
            "use absolute path: 'for<'a> Fn (&'a u8)'",
            "use `::core::ops::Fn` instead of `Fn`",
        )],
    },
    Row {
        kind: "a bound in parentheses",
        error: || {
            let item: ItemTrait = parse_quote!(
                trait Tr: (Clone) {}
            );
            errors_in(Leaker::new(), |leaker| leaker.intern_trait(&item, true))
        },
        items: &[(
            "use absolute path: '(Clone)'",
            "use `::core::clone::Clone` instead of `Clone`",
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
            errors_in(Leaker::new(), |leaker| leaker.intern_struct(&item, true))
        },
        items: &[(
            "use absolute path: 'MyType<'_>'",
            "write `MyType` with an absolute path (e.g. `::my_crate::MyType`)",
        )],
    },
    Row {
        kind: "the type of a const parameter",
        error: || {
            let item: ItemStruct = parse_quote!(
                struct S<const N: Idx>(u8);
            );
            errors_in(Leaker::new(), |leaker| leaker.intern_struct(&item, true))
        },
        items: &[(
            "use absolute path: 'Idx'",
            "write `Idx` with an absolute path (e.g. `::my_crate::Idx`)",
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
            errors_in(allowing(), |leaker| leaker.intern_trait(&item, true))
        },
        items: &[(
            "use absolute path: 'RelTr'",
            "write `RelTr` with an absolute path (e.g. `::my_crate::RelTr`), or name the lifetime of the trait object (e.g. \
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
            errors_in(allowing(), |leaker| leaker.intern_trait(&item, false))
        },
        items: &[(
            "use absolute path: 'Rel<Self>'",
            "write `Rel` with an absolute path (e.g. `::my_crate::Rel`)",
        )],
    },
    // ── a bare identifier that may be a const ────────────────────────────────────────────────
    Row {
        kind: "a bare identifier that may be a const, in a path that names `Self`, which is denied",
        error: || errors(maybe_const),
        items: &[(
            "`LEN` may be a const, and the type that holds it cannot be leaked as a whole",
            "write `LEN` with an absolute path (`::my_crate::LEN`, or `{ ::my_crate::LEN }` for a \
             const)",
        )],
    },
    Row {
        kind: "a bare identifier that may be a const, in a path that names `Self::A`",
        error: || {
            let item: ItemStruct = parse_quote!(
                struct S {
                    a: ::p::W<Self::A, LEN>,
                }
            );
            errors_in(allowing(), |leaker| leaker.intern_struct(&item, true))
        },
        items: &[(
            "`LEN` may be a const, and the type that holds it cannot be leaked as a whole",
            "write `LEN` with an absolute path (`::my_crate::LEN`, or `{ ::my_crate::LEN }` for a \
             const), or replace `Self::A` with `<Self as SomeTrait>::A`, naming the trait that \
             defines `A` with its absolute path",
        )],
    },
    Row {
        kind: "a bare identifier that may be a const, in the default of a parameter that names \
               `Self`",
        error: || {
            let item: ItemTrait = parse_quote!(
                trait Tr<T = ::p::W<Self, LEN>> {}
            );
            errors_in(allowing(), |leaker| leaker.intern_trait(&item, true))
        },
        items: &[(
            "`LEN` may be a const, and the type that holds it cannot be leaked as a whole",
            "write `LEN` with an absolute path (`::my_crate::LEN`, or `{ ::my_crate::LEN }` for a \
             const), or name the type instead of `Self` in `::p::W<Self, LEN>`",
        )],
    },
    Row {
        kind: "a bare identifier that may be a const, in a path whose bound names `Self`",
        error: || {
            let item: ItemStruct = parse_quote!(
                struct S<T: ::core::convert::AsRef<Self>> {
                    a: ::p::W<T, LEN>,
                }
            );
            errors_in(allowing(), |leaker| leaker.intern_struct(&item, false))
        },
        items: &[(
            "`LEN` may be a const, and the type that holds it cannot be leaked as a whole",
            "write `LEN` with an absolute path (`::my_crate::LEN`, or `{ ::my_crate::LEN }` for a \
             const), or write the bound of the generics of `::p::W<T, LEN>` without `Self`",
        )],
    },
    Row {
        kind: "a bare identifier that may be a const, in a type that leaves a method lifetime \
               unconstrained",
        error: || {
            let item: ItemTrait = parse_quote!(
                trait Tr {
                    fn f<'b>(&self, y: &'b dyn ::core::convert::AsRef<LEN>) -> &'b u8;
                }
            );
            errors_in(allowing(), |leaker| leaker.intern_trait(&item, true))
        },
        items: &[(
            "`LEN` may be a const, and the type that holds it cannot be leaked as a whole",
            "keep `'b` in another argument of the method (for example `&'b self`), write `LEN` \
             with an absolute path (`::my_crate::LEN`, or `{ ::my_crate::LEN }` for a const), or \
             write the bound of the trait object (`dyn Trait + 'b`)",
        )],
    },
    Row {
        kind: "a bare identifier that may be a const, in a path with an expression that cannot be \
               leaked",
        error: || {
            let item: ItemStruct = parse_quote!(
                struct S {
                    a: ::p::W<{ ::p::m!() }, LEN>,
                }
            );
            errors_in(allowing(), |leaker| leaker.intern_struct(&item, true))
        },
        items: &[(
            "`LEN` may be a const, and the type that holds it cannot be leaked as a whole",
            "write `LEN` with an absolute path (`::my_crate::LEN`, or `{ ::my_crate::LEN }` for a \
             const)",
        )],
    },
    // ── `Self` ───────────────────────────────────────────────────────────────────────────────
    Row {
        kind: "`Self` where no scope declares it",
        error: || errors(self_without_meaning),
        items: &[
            (
                "`Self` has no meaning here",
                "write the type instead of `Self`",
            ),
            (
                "`Self` has no meaning here",
                "write the type instead of `Self`",
            ),
        ],
    },
    Row {
        kind: "`Self::A` in an impl of a trait",
        error: || errors(self_projection),
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
            errors_in(allowing(), |leaker| leaker.intern_impl(&item, true))
        },
        items: &[
            (
                "use absolute path: 'Tr'",
                "write `Tr` with an absolute path (e.g. `::my_crate::Tr`)",
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
            errors_in(allowing(), |leaker| leaker.intern_struct(&item, true))
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
            errors_in(allowing(), |leaker| leaker.intern_impl(&item, true))
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
            errors_in(allowing(), |leaker| leaker.intern_struct(&item, true))
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
                "`Self::A` is in a bound or where predicate, and a bound of its generics names \
                 `Self::A`",
                "write the bound without `Self::A`, or use `Self::A` outside the bounds and \
                 where predicates",
            ),
        ],
    },
    Row {
        kind: "`Self` in the default of a parameter",
        error: || errors(self_in_default),
        items: &[(
            "`Wrap<Self>` is in the default of a parameter, and names `Self`",
            "name the type instead of `Self` in `Wrap<Self>`",
        )],
    },
    Row {
        kind: "a bound of an outer trait that names its `Self`",
        error: || errors(unnameable_self),
        items: &[(
            "the bound of `T` mentions a `Self` which has no name here",
            "write the bound of `T` without `Self`, or write `Rel<T>` with an absolute path, so \
             that it is not leaked",
        )],
    },
    Row {
        kind: "a bound of an outer type that names an associated item of its `Self`",
        error: || {
            errors_in(allowing(), |leaker| {
                leaker.enter_type(&id("S"), &parse_quote!(<T: AsRef<Self::X>>), true, |view| {
                    view.visit_item(&parse_quote!(
                        impl ::p::W for X {
                            fn f(&self, x: Rel<T>) {}
                        }
                    ))
                })
            })
        },
        items: &[(
            "the bound of `T` mentions `Self::X`, which has no name here",
            "write the bound of `T` without `Self::X`, or write `Rel<T>` with an absolute path, \
             so that it is not leaked",
        )],
    },
    Row {
        kind: "a generic whose bound mentions a `Self` that is denied",
        error: || errors(denied_self_in_bounds),
        items: &[(
            "`Wrap<T>` has a generic whose bound mentions `Self`, which cannot be leaked",
            "write the bound of the generics of `Wrap<T>` without `Self`",
        )],
    },
    Row {
        kind: "the self type of an impl whose bound names `Self`",
        error: || errors(self_in_impl_self_type),
        items: &[
            (
                "`S<T>` is in the self type of the impl, and a bound of its generics names \
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
        error: || errors(unprovable_bound),
        items: &[(
            "`Wrap<T>` is in a bound or where predicate, and a bound of its generics names a \
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
            errors_in(allowing(), |leaker| leaker.intern_trait(&item, true))
        },
        items: &[(
            "`Rel<X>` is in a bound or where predicate, and a bound of its generics names \
             `Self::A0`",
            "write the bound without `Self::A0`, or use `Rel<X>` outside the bounds and where \
             predicates",
        )],
    },
    Row {
        kind: "a type that names `Self` in a predicate of its trait",
        error: || errors(self_in_trait_predicate),
        items: &[(
            "`Rel<Self>` names `Self` in a bound or where predicate of its trait",
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
            errors_in(allowing(), |leaker| leaker.intern_trait(&item, true))
        },
        items: &[
            (
                "`Wrap<'b>` leaves the lifetime `'b` of the method unconstrained in the copy \
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
            errors_in(allowing(), |leaker| leaker.intern_trait(&item, true))
        },
        items: &[(
            "use absolute path: 'Rel<'x>'",
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
            errors_in(allowing(), |leaker| leaker.intern_trait(&item, true))
        },
        items: &[(
            "use absolute path: 'Rel<T>'",
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
            errors_in(allowing(), |leaker| leaker.intern_trait(&item, true))
        },
        items: &[(
            "use absolute path: 'Rel<T>'",
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
            errors_in(allowing(), |leaker| leaker.intern_trait(&item, true))
        },
        items: &[(
            "use absolute path: 'Rel<T>'",
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
            errors_in(allowing(), |leaker| leaker.intern_trait(&item, true))
        },
        items: &[(
            "use absolute path: 'Rel<T>'",
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
            errors_in(allowing(), |leaker| leaker.intern_struct(&item, false))
        },
        items: &[(
            "use absolute path: 'Self'",
            "write the lifetime that the path hides (`Ref<'x, u8>`), and name it in the output",
        )],
    },
];

#[test]
fn every_kind_of_error_has_a_help() {
    for row in ROWS {
        let err = (row.error)();
        let items = &err;
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
        for item in items {
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
            // It has no colon of its own outside the code in backticks (a lead clause, a colon and
            // the instruction): a tool that prints "message: help" on one line would show two.
            assert!(
                help.split('`').step_by(2).all(|words| !words.contains(':')),
                "{}: {}",
                row.kind,
                help
            );
        }
        // The first error's help is what the accessors of the error tell.
        assert_eq!(err[0].help_text(), items[0].help_text(), "{}", row.kind);
    }
}

#[test]
fn a_help_is_rendered_after_the_notes() {
    for row in ROWS {
        let err = (row.error)();
        let items = &err;
        let errors: Vec<syn::Error> = to_syn_error(&err).into_iter().collect();
        assert_eq!(errors.len(), items.len(), "{}", row.kind);
        for (error, item) in errors.iter().zip(items) {
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
        // And `to_compile_error()` renders the same errors, as `compile_error!`s.
        let tokens = err
            .iter()
            .map(ErrorItem::to_compile_error)
            .collect::<proc_macro2::TokenStream>()
            .to_string();
        assert_eq!(
            tokens.matches("compile_error").count(),
            items.len(),
            "{}: {}",
            row.kind,
            tokens
        );
        for item in items {
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
        for item in &err {
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

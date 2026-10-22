//! The errors of the Leaker as data (`ErrorItem`): what each is about (`Cause`) and why (`Reason`),
//! how it is written as tokens and read back, and how it is reported.
//!
//! - One test for each `Reason`: an item that triggers it, the reason of the error, and the kind
//!   and the text of its cause. The other kinds of cause have a test of their own.
//! - `every_error_is_written_and_read_back` is a round trip of every error above: the tokens
//!   that are written again are the same, and so are the fields. `the_serialized_form_is_*` pin
//!   the grammar, and what the parser refuses.
//! - `to_compile_error()`, `to_syn_error()`, and the other ways to take an error as a whole.
//!
//! The errors are reported at the same places after the round trip: see `tests/diagnostic.rs` (a
//! macro that does it, and rustc) and `tests/crates/spans` (the lines and the columns).

use syn::*;
use template_quote::ToTokens;
use type_leak::*;

/// The config of the triggers with `::p` allowed (the crate of the property test).
fn allowing_config() -> LeakerConfig {
    let mut config = LeakerConfig::new();
    config.allow_primitive().allow_std_prelude();
    config.allowed_paths.push(parse_quote!(::p));
    config
}

fn allowing() -> Leaker {
    Leaker::from_config(allowing_config())
}

/// The config of the triggers that read the values of the items too.
fn reading_values_config() -> LeakerConfig {
    let mut config = LeakerConfig::new();
    config.allow_primitive().allow_std_prelude().check_values();
    config
}

/// A Leaker of `config` that accepts the conflicts with the causes `allowed`.
fn accepting(mut config: LeakerConfig, allowed: &[Cause]) -> Leaker {
    for cause in allowed {
        config.allow(cause.clone());
    }
    Leaker::from_config(config)
}

/// An analysis of an item that goes wrong, with the causes `allowed` accepted.
type Trigger = fn(&[Cause]) -> std::result::Result<(), Vec<ErrorItem>>;

/// The errors of a trigger that accepts nothing.
fn errors(trigger: Trigger) -> Vec<ErrorItem> {
    trigger(&[]).unwrap_err()
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

/// The help of an item, as the text of the tokens it is about and the help.
fn help_of(item: &ErrorItem) -> Option<(String, String)> {
    item.help
        .as_ref()
        .map(|(at, help)| (at.to_string(), help.clone()))
}

/// The analysis of a trait with these items. `check_values` reads the values too.
fn trait_result(
    source: &str,
    check_values: bool,
    allowed: &[Cause],
) -> std::result::Result<(), Vec<ErrorItem>> {
    let item: ItemTrait = parse_str(source).unwrap();
    let config = if check_values {
        reading_values_config()
    } else {
        allowing_config()
    };
    accepting(config, allowed)
        .intern_trait(&item, true)
        .map(|_| ())
}

/// The errors that a trait with these items gives. `check_values` reads the values too.
fn trait_errors(source: &str, check_values: bool) -> Vec<ErrorItem> {
    trait_result(source, check_values, &[]).unwrap_err()
}

// ── the triggers: an item for each reason ────────────────────────────────────────────────────

/// A trait in a bound and a type with an elided lifetime.
fn relative_path(allowed: &[Cause]) -> std::result::Result<(), Vec<ErrorItem>> {
    let item: ItemTrait = parse_quote!(
        trait Tr: Clone {
            fn f(&self, a: MyType<'_>);
        }
    );
    accepting(LeakerConfig::new(), allowed)
        .intern_trait(&item, true)
        .map(|_| ())
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
fn self_without_meaning(allowed: &[Cause]) -> std::result::Result<(), Vec<ErrorItem>> {
    accepting(allowing_config(), allowed)
        .enter_generics(&Generics::default(), |view| {
            view.visit_type(&parse_quote!(Rel<Self>));
            view.visit_type(&parse_quote!(Wrap<Self::Item>));
        })
        .map(|_| ())
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
    assert_eq!(items[0].message, "`Self` has no meaning here");
    // The note and the help are about the mention.
    assert_eq!(items[1].notes[0].0.to_string(), "Self :: Item");
    assert_eq!(
        items[1].help.as_ref().unwrap().0.to_string(),
        "Self :: Item"
    );
}

/// Syntax that the Leaker cannot read in a body, read with `check_values()`.
fn unreadable(allowed: &[Cause]) -> std::result::Result<(), Vec<ErrorItem>> {
    trait_result(
        "trait Tr { fn f(&self) { macro_rules! m { () => {} } ::p::mac!(a b c); } }",
        true,
        allowed,
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
fn maybe_const(allowed: &[Cause]) -> std::result::Result<(), Vec<ErrorItem>> {
    let item: ItemStruct = parse_quote!(
        struct S {
            a: ::p::W<[Self], LEN>,
        }
    );
    accepting(allowing_config(), allowed)
        .intern_struct(&item, false)
        .map(|_| ())
}

#[test]
fn reason_maybe_const() {
    let items = errors(maybe_const);
    assert_eq!(items.len(), 1);
    // The bare identifier is the cause.
    assert_item(&items[0], Reason::MaybeConst, "type", "LEN");
    assert_eq!(
        items[0].message,
        "`LEN` may be a const, and the type that holds it cannot be leaked as a whole"
    );
}

/// A lifetime of a method that the copy would leave unconstrained.
fn unconstrained_lifetime(allowed: &[Cause]) -> std::result::Result<(), Vec<ErrorItem>> {
    trait_result(
        "trait Tr { fn f<'b>(&self, x: Wrap<'b>) -> &'b u8; }",
        false,
        allowed,
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
}

/// `Self::A` in an impl of a trait, where the Repeater's impl cannot write it.
fn self_projection(allowed: &[Cause]) -> std::result::Result<(), Vec<ErrorItem>> {
    let item: ItemImpl = parse_quote!(
        impl ::p::Tr for u8 {
            type A = u8;
            fn f(&self, x: Wrap<Self::A>) {}
        }
    );
    accepting(allowing_config(), allowed)
        .intern_impl(&item, true)
        .map(|_| ())
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
    assert!(
        items[0]
            .message
            .starts_with("write `<Self as ::p::Tr>::A` instead of `Self::A`"),
        "{}",
        items[0].message
    );
}

/// A bound of an outer trait that names its `Self`, for a type in an item nested in it.
fn unnameable_self(allowed: &[Cause]) -> std::result::Result<(), Vec<ErrorItem>> {
    accepting(allowing_config(), allowed)
        .enter_trait(&id("Tr"), &parse_quote!(<T: AsRef<Self>>), true, |view| {
            view.visit_item(&parse_quote!(
                impl ::p::W for X {
                    fn f(&self, x: Rel<T>) {}
                }
            ))
        })
        .map(|_| ())
}

#[test]
fn reason_unnameable_self() {
    let items = errors(unnameable_self);
    assert_eq!(items.len(), 1);
    assert_item(&items[0], Reason::UnnameableSelf, "type", "Rel < T >");
    assert_eq!(
        items[0].message,
        "the bound of `T` mentions a `Self` which has no name here"
    );
}

/// `Self` in the default of a parameter.
fn self_in_default(allowed: &[Cause]) -> std::result::Result<(), Vec<ErrorItem>> {
    trait_result("trait Tr<T: Default = Wrap<Self>> {}", false, allowed)
}

#[test]
fn reason_self_in_default() {
    let items = errors(self_in_default);
    assert_eq!(items.len(), 1);
    // The cause is the whole default, the note is at the `Self` in it.
    assert_item(&items[0], Reason::SelfInDefault, "type", "Wrap < Self >");
    assert_eq!(items[0].notes[0].0.to_string(), "Self");
    assert_eq!(
        items[0].message,
        "`Wrap < Self >` is in the default of a parameter, and names `Self`"
    );
}

/// A generic whose bound mentions a `Self` that is denied.
fn denied_self_in_bounds(allowed: &[Cause]) -> std::result::Result<(), Vec<ErrorItem>> {
    let item: ItemTrait = parse_quote!(
        trait Tr {
            fn f<T: ::core::cmp::PartialEq<Self>>(&self, x: Wrap<T>);
        }
    );
    accepting(allowing_config(), allowed)
        .intern_trait(&item, false)
        .map(|_| ())
}

#[test]
fn reason_denied_self_in_bounds() {
    let items = errors(denied_self_in_bounds);
    assert_eq!(items.len(), 1);
    assert_item(&items[0], Reason::DeniedSelfInBounds, "type", "Wrap < T >");
}

/// A type that names the `Self` of a nested struct, with a generic whose bound names it too.
fn nested_self_in_bounds(allowed: &[Cause]) -> std::result::Result<(), Vec<ErrorItem>> {
    accepting(allowing_config(), allowed)
        .enter_mod("m", [id("S")], |view| {
            view.visit_item(&parse_quote!(
                struct S<T: ::core::convert::AsRef<Self>>(Rel2<Self, T>);
            ))
        })
        .map(|_| ())
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
fn unprovable_bound(allowed: &[Cause]) -> std::result::Result<(), Vec<ErrorItem>> {
    let item: ItemTrait = parse_quote!(
        trait Tr<T: Fn(Rel)>
        where
            Wrap<T>: Sized,
        {
        }
    );
    accepting(allowing_config(), allowed)
        .intern_trait(&item, true)
        .map(|_| ())
}

#[test]
fn reason_unprovable_bound() {
    let items = errors(unprovable_bound);
    assert_eq!(items.len(), 1);
    assert_item(&items[0], Reason::UnprovableBound, "type", "Wrap < T >");
}

/// A type that names `Self` in a predicate of its trait.
fn self_in_trait_predicate(allowed: &[Cause]) -> std::result::Result<(), Vec<ErrorItem>> {
    let item: ItemTrait = parse_quote!(
        trait Tr
        where
            Rel<Self>: Sized,
        {
        }
    );
    accepting(allowing_config(), allowed)
        .intern_trait(&item, true)
        .map(|_| ())
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
fn self_in_impl_self_type(allowed: &[Cause]) -> std::result::Result<(), Vec<ErrorItem>> {
    let item: ItemImpl = parse_quote!(
        impl<T> S<T> where T: AsRef<Option<Self>> {}
    );
    accepting(allowing_config(), allowed)
        .intern_impl(&item, true)
        .map(|_| ())
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
fn values(allowed: &[Cause]) -> std::result::Result<(), Vec<ErrorItem>> {
    let mut errors = Vec::new();
    errors.extend(
        trait_result("trait Tr<const N: usize = SIZE> {}", false, allowed)
            .err()
            .unwrap_or_default(),
    );
    errors.extend(
        trait_result(
            "trait Tr { fn f(&self) { let _ = inner::N; let _ = Rel::<u8>::f(); } }",
            true,
            allowed,
        )
        .err()
        .unwrap_or_default(),
    );
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
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
fn macros_and_qualified_traits(allowed: &[Cause]) -> std::result::Result<(), Vec<ErrorItem>> {
    trait_result(
        "trait Tr { fn f(&self) { mac!(1); let _ = <u8 as Rel>::f(); } }",
        true,
        allowed,
    )
}

#[test]
fn the_cause_of_a_macro_is_its_path_and_that_of_a_qualified_trait_is_a_trait() {
    let items = errors(macros_and_qualified_traits);
    assert_eq!(items.len(), 2);
    assert_item(&items[0], Reason::RelativePath, "macro", "mac !");
    assert!(matches!(&items[0].cause, Cause::Macro(path) if path.is_ident("mac")));
    assert_item(&items[1], Reason::RelativePath, "trait", "Rel");
    assert_eq!(items[0].message, "use absolute path: 'mac !'");
}

/// A lifetime that no scope declares, in the supertrait of a trait where `Self` is denied.
fn lifetime(allowed: &[Cause]) -> std::result::Result<(), Vec<ErrorItem>> {
    let item: ItemTrait = parse_quote!(
        trait Tr: ::p::Sup<&'a Self> {}
    );
    accepting(allowing_config(), allowed)
        .intern_trait(&item, false)
        .map(|_| ())
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
fn type_errors(leaker: &mut Leaker, ty: Type) -> Vec<ErrorItem> {
    leaker
        .enter_generics(&parse_quote!(<T>), |view| view.visit_type(&ty))
        .unwrap_err()
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
    let items = type_errors(&mut allowing(), parse_quote!(::p::W<'_, dyn RelTr>));
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
    let items = allowing().intern_trait(&item, true).unwrap_err();
    assert_eq!(items.len(), 1);
    assert_item(&items[0], Reason::UnconstrainedLifetime, "trait", "RelTr");
}

#[test]
fn the_trait_of_an_impl_type_is_a_trait() {
    let items = type_errors(&mut allowing(), parse_quote!(impl MyTrait));
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
        &mut allowing(),
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
    let items = type_errors(&mut allowing(), parse_quote!([u8; two!()]));
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
    let items = allowing().intern_struct(&item, false).unwrap_err();
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
    let items = allowing().intern_trait(&item, true).unwrap_err();
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
    let items = allowing().intern_trait(&item, true).unwrap_err();
    assert_eq!(items.len(), 1);
    assert_item(&items[0], Reason::MaybeConst, "type", "Rel");
    // A bare identifier among the arguments of a holder that `Self` stops from being leaked.
    let items = allowing()
        .enter_type(&id("S"), &Generics::default(), false, |view| {
            view.visit_type(&parse_quote!(::l::W<[Self], ILEN>))
        })
        .unwrap_err();
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
    let items = Leaker::new().intern_trait(&item, true).unwrap_err();
    assert_eq!(items.len(), 1);
    assert_item(&items[0], Reason::RelativePath, "type", "u8");
    assert_eq!(
        note_at(&items[0], "a"),
        Some("`a` also depends on the type context")
    );
    // Where `u8` is allowed, `a` is the first.
    let items = allowing().intern_trait(&item, true).unwrap_err();
    assert_eq!(items.len(), 1);
    assert_item(&items[0], Reason::RelativePath, "expr", "a");
    assert_eq!(note_at(&items[0], "a"), None);
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
    let items = Leaker::new().intern_trait(&item, true).unwrap_err();
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
    let items = Leaker::new().intern_trait(&item, true).unwrap_err();
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
    let items = Leaker::new()
        .enter_impl(&Generics::default(), &parse_quote!(B), false, |view| {
            view.visit_type(&parse_quote!(<Self as super::Tr<u8>>::X))
        })
        .unwrap_err();
    assert_eq!(items.len(), 1);
    assert_item(
        &items[0],
        Reason::RelativePath,
        "unreadable",
        "super :: Tr < u8 > :: X",
    );
    let items = type_errors(&mut allowing(), parse_quote!(<T as RelTrait<'_>>::Ty));
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
}

#[test]
fn every_error_is_written_and_read_back() {
    let mut reasons = Vec::new();
    let mut kinds = Vec::new();
    for (name, items) in every_error() {
        assert!(!items.is_empty(), "{}", name);
        for item in &items {
            reasons.push(item.reason);
            kinds.push(kind(&item.cause));
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
    // The triggers have every reason and every kind of cause.
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
        assert!(reasons.contains(&reason), "{:?}", reason);
    }
    for kind in ["type", "trait", "expr", "macro", "lifetime", "unreadable"] {
        assert!(kinds.contains(&kind), "{}", kind);
    }
}

#[test]
fn the_serialized_form_is_one_brace_group_for_each_item() {
    let items = errors(relative_path);
    // The tokens of the cause, of the notes and of the help are written as they are, the message,
    // the notes and the help as string literals.
    assert_eq!(
        items[1].to_token_stream().to_string(),
        "{ type (MyType < '_ >) RelativePath \"use absolute path: 'MyType < '_ >'\" \
         [('_) \"it cannot be leaked as a whole because of `'_`\"] \
         (MyType) \"write `MyType` with an absolute path (e.g. `::my_crate::MyType`), or allow it \
         with `LeakerConfig::allowed_paths`\" }"
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
    // Any tokens, groups included.
    assert_item(
        &read("unreadable", "any [ tokens ] { ( here ) } ;").unwrap(),
        Reason::RelativePath,
        "unreadable",
        "any [tokens] { (here) } ;",
    );
}

#[test]
fn a_cause_that_is_not_the_whole_content_is_refused() {
    for (kind, cause) in [
        ("type", "A B"),
        ("trait", "A < B > C"),
        ("macro", "mac"),
        ("macro", "mac ! x"),
    ] {
        let source = format!(r#"{{ {} ( {} ) RelativePath "m" [] }}"#, kind, cause);
        assert!(parse_item(&source).is_err(), "{}", source);
    }
}

#[test]
fn an_unknown_kind_or_reason_is_named() {
    let error = parse_item(r#"{ statement ( A ) RelativePath "m" [] }"#).unwrap_err();
    assert!(
        error.to_string().contains("unknown kind `statement`"),
        "{}",
        error
    );
    let error = parse_item(r#"{ type ( A ) Nonsense "m" [] }"#).unwrap_err();
    assert!(
        error.to_string().contains("unknown reason `Nonsense`"),
        "{}",
        error
    );
    // Something after the help, or a message that is no string.
    assert!(parse_item(r#"{ type ( A ) RelativePath "m" [] ( B ) "h" "x" }"#).is_err());
    assert!(parse_item(r#"{ type ( A ) RelativePath m [] }"#).is_err());
    assert!(parse_item(r#"type ( A ) RelativePath "m" []"#).is_err());
}

// ── the report ───────────────────────────────────────────────────────────────────────────────

#[test]
fn an_item_has_a_message_notes_and_help_and_is_an_error() {
    let items = errors(relative_path);
    let item = &items[1];
    assert_eq!(item.to_string(), item.message);
    assert_eq!(
        item.note(),
        Some("it cannot be leaked as a whole because of `'_`")
    );
    assert_eq!(
        item.help_text(),
        Some(
            "write `MyType` with an absolute path (e.g. `::my_crate::MyType`), or allow it with \
             `LeakerConfig::allowed_paths`"
        )
    );
    // It is a `std::error::Error`, and `From<ErrorItem>` is the `syn::Error` of the layout of
    // rustc: the notes, then the help, after the message.
    let error: Box<dyn std::error::Error> = Box::new(item.clone());
    assert_eq!(error.to_string(), item.message);
    let syn_error: syn::Error = item.clone().into();
    assert_eq!(
        syn_error.to_string(),
        "use absolute path: 'MyType < '_ >'\n\n\
         \x20 = note: it cannot be leaked as a whole because of `'_`\n\
         \x20 = help: write `MyType` with an absolute path (e.g. `::my_crate::MyType`), or allow it \
         with `LeakerConfig::allowed_paths`\n"
    );
}

#[test]
fn to_compile_error_is_the_error_of_syn() {
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

#[test]
fn allowing_the_causes_of_the_errors_of_each_reason_accepts_the_item() {
    let mut reasons = Vec::new();
    for (name, trigger) in every_trigger() {
        let items = errors(trigger);
        reasons.extend(items.iter().map(|item| item.reason));
        let causes: Vec<Cause> = items.iter().map(|item| item.cause.clone()).collect();
        if let Err(left) = trigger(&causes) {
            panic!("{}: {:?} are left", name, messages(&left));
        }
        // Each cause is needed: without it, its errors, and only they, are reported.
        for cause in &causes {
            let others: Vec<Cause> = causes.iter().filter(|c| *c != cause).cloned().collect();
            let left = trigger(&others).unwrap_err();
            let expected: Vec<&str> = items
                .iter()
                .filter(|item| item.cause == *cause)
                .map(|item| item.message.as_str())
                .collect();
            assert_eq!(messages(&left), expected, "{}: {}", name, text(cause));
        }
    }
    // The triggers have every reason.
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
        assert!(reasons.contains(&reason), "{:?}", reason);
    }
}

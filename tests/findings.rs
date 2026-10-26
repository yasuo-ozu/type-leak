//! Findings of the property test `tests/prop_cross_crate`, as library-level tests.
//!
//! The property test generates crates, builds them, and reports what is wrong in them: a Leaker's
//! crate or a Referrer's crate that does not compile, or an item that should have been rejected.
//! Each finding is minimized here to a test of the API of `type-leak` (`Leaker`, `Referrer`,
//! `RepeaterSlots`), with no generated crate and no `cargo` build. A test asserts what the library
//! produces today: what a fix made it produce (`fixed`), or what the docs of `Leaker` document
//! (`limitation`).
//!
//! | test | finding | kind |
//! |------|---------|------|
//! | `f01_unsized_type_param_encoding` | a `?Sized` parameter is a bare element of the tuple of the trait arguments | fixed |
//! | `f02_bound_names_leaked_type_env_cycle` | a copy with a bound and a where predicate that name entries | fixed |
//! | `f03_where_predicate_pulls_param_into_context` | an entry under a where predicate that names a lifetime is not found | fixed |
//! | `f04_module_crate_relative_path_not_rewritten` | an entry that names a declared item of a module is shared by all modules | fixed |
//! | `f05_inferred_outlives_of_self_type` | no `T: 'a` for a path with a lifetime; written, it is | limitation |
//! | `f06_impl_assoc_type_of_self_in_repeater` | `intern_impl` accepts `Self::A` in an entry | fixed |
//! | `f07_method_lifetime_in_leaked_path` | a method lifetime in a leaked path and the return type | fixed |
//! | `f08_merged_bound_ambiguous_plus` | merged bounds print `-> &dyn Any + Clone` | fixed |
//! | `f09_fn_pointer_self_predicate_print` | `Self` replaced by `&fn(..)` in a where predicate | fixed |
//! | `f10_module_bound_names_declared_item` | a bound of an entry names an item of the module | limitation |
//! | `f11_copied_impl_param_constrained_only_by_leaked_type` | a copied impl whose self type is an entry | limitation |
//! | `f12_elided_lifetime_in_fn_sugar_with_generics` | `Fn(&Rel<T>) -> &u8` with an entry that has lifetimes | fixed |
//! | `f13_impl_bound_names_self_of_leaked_self_type` | a bound of an impl names `Self`, and the self type is an entry: it is its own `Self` | fixed |
//! | `f14_impl_predicate_names_self_projection_with_leaked_bound` | `Self::A` in a predicate of an impl whose bound names a leaked type | fixed |
//! | `f15_self_rewritten_in_elided_signature` | `with_self()` writes `Self`, with lifetimes, into `fn(&Self) -> &u8` | fixed |
//! | `f16_implied_outlives_of_self_projection_lost` | `Self::A0: 'x` of `&'x Self::A0` is not recorded, only `Self: 'x` | fixed |
//! | `f17_optional_entry_in_own_context` | a strict Referrer replaces an optional entry whose own context names it | fixed |
//! | `f18_strict_referrer_keeps_the_optional_type_that_holds_a_method_lifetime` | a strict Referrer replaces the optional type that holds a method lifetime: it is copied as written | fixed |
//! | `f19_impl_self_in_bound_of_entry_in_predicate` | an entry in a predicate has a bound of its context that names `Self`, in an impl whose self type is leaked | fixed |
//! | `f20_trait_assoc_in_bound_of_entry_in_predicate` | an entry in a predicate has a bound of its context that names `Self::A`, in a trait | fixed |
//! | `f21_iterator_item_in_bound_of_entry_in_item_binding` | an entry in `T: Iterator<Item = ..>` has a bound of its context that names the `Item` | fixed |
//! | `f22_method_lifetime_in_free_function_and_inherent_method` | the E0195 check of F7 also rejects a free function and an inherent method | fixed |
//! | `f23_self_in_elided_signature_where_referrer_can_rewrite` | `Self` in an elided signature is rejected in a bound and with `allow_self = false`, where the Referrer could write `for<'e>` | fixed |
//! | `f24_self_in_predicate_with_leaked_bound_of_unused_generic` | a bare `Self` in a predicate is rejected for a leaked bound of a generic that no entry of the self type uses | fixed |
//! | `f25_supertrait_item_in_bound_of_entry_in_predicate` | a supertrait fixes the `Item` that a bound of an entry's generic names | fixed |
//! | `f26_alias_hides_projection_in_bound_of_entry_in_predicate` | a type alias hides the projection that such a bound names: the context of an entry leaves out a bound of a generic that names it, if another bound of it fixes an associated type | fixed |
//! | `f27_argument_impl_trait_bound_names_leaked_type_env_cycle` | the bound of an argument-position `impl Trait` names an entry whose generic has a leaked bound | fixed |
//! | `f28_hidden_lifetime_in_elided_signature_with_generics` | a path hides the one lifetime of an elided signature, next to an entry that adds lifetimes | fixed |
//! | `f29_entry_hides_lifetime` | an entry (a relative path) hides a lifetime: its Repeater impl does not compile | limitation |
//! | `f30_implied_outlives_of_generic_projection_recorded_as_component` | `&'x T::Item` is recorded as `T: 'x`, which the copy cannot prove | fixed |
//! | `f31_nested_signature_has_its_own_lifetimes` | a signature inside an input has its own lifetimes only for rustc 1.64 and later | limitation |
//! | `f32_parentheses_only_where_rustc_needs_them` | a trait object with one bound is parenthesized in the encoding of a `?Sized` argument | fixed |
//! | `f33_where_predicate_names_item_that_a_predicate_in_scope_defines` | a where predicate of an entry's context names an `Item` that a predicate of a method equates to a leaked type | fixed |
//! | `f34_implied_outlives_of_self_projection_forms_recorded_as_components` | `&'x <Self as Tr<'a, T>>::A`, `&'x Self::A<'b>` and a `Self::A` that a bound equates are recorded as components | fixed |
//! | `f35_gat_where_clause_of_self_projection_not_recorded` | `Self::G<'x>` of a generic associated type with `where Self: 'b` needs `Self: 'x`, which F34 stopped recording | fixed |
//! | `f36_nested_item_self_entry_not_rewritten` | an entry that names the own `Self` of a struct, enum, union, trait or impl nested in a module is recorded, and the Visitor never rewrote it: `Visitor::with_nested_self()` | fixed |
//! | `f37_nested_type_with_a_bound_that_names_its_own_self_is_supported` | an entry that uses a generic whose bound names the own `Self` of a struct, enum or union nested in a module: the bound holds for the copy's type, the Repeater's impl needed it for the Leaker's; the impl is for any type now, and the copy passes its own (`SelfRequirement::Any`) | fixed |
//! | `f38_an_absolute_type_that_names_the_own_self_is_copied_as_written` | an absolute type that names the own `Self` of a struct, enum or union nested in a module (`::std::boxed::Box<Self>`) was an optional entry: a strict Referrer replaced it with the Leaker's type, and a trusting one left the copy's own; it is no entry now, and both copy it as written | fixed |
//! | `f39_the_default_of_a_nested_trait_that_needs_its_self_is_written_through_the_host` | the default of a parameter of a trait (`U = Rel<T>` with `T: AsRef<Self>`) has an entry for the trait's `Self`, and the Visitor left it as written: it is looked up with the trait's own `Self` too, and written by `with_nested_self()`; the default of the trait itself is looked up without `Self`, as it is analyzed | fixed |
//! | `f40_a_type_in_the_default_of_an_associated_const_is_carried` | a type in a value (the default of an associated const, a function body) was never carried, so the copy kept a relative path: with `LeakerConfig::check_values()` a type in a type position of a value (a cast, a `let`, a closure, a turbofish, a qualified path) is an entry, which the Visitor replaces; a path that is not a type stays rejected | fixed |
//! | `f41_a_default_that_needs_self_only_through_a_bound_is_accepted` | the default of a parameter of a struct, enum or union nested in a module that uses a generic with a bound that names the own `Self` (`U = Wrap<T>` with `T: Clone + AsRef<Self>`) was rejected: its entry has no `Self` and none of the bounds that name it now, and the copy writes an ordinary default | fixed |
//! | `f42_the_option_accepts_a_type_that_names_self_with_a_bound_that_names_it` | a type that names the own `Self` of such an item and a bound of the generics it uses does too (`Rel2<Self, T>` with `T: AsRef<Self>`) was rejected: with `LeakerConfig::nested_self_entries()` it is an entry for any type, with `Self` as the parameter of the impl, and the copy passes its own type | fixed (opt-in) |
//! | `f43_a_type_whose_context_cannot_be_written_is_copied_as_written` | a type whose context cannot be written (a bound names a `Self` that has no name there) is copied as written, and the Referrer replaced the entries inside it: the part has no context (`unwritable`) | fixed |
//! | `f44_a_part_that_names_self_is_found_when_no_entry_does` | a part whose context names `Self` was never found when no entry named it: the top-level context of the scope left `Self` out | fixed |
//! | `f45_a_part_in_a_nested_item_is_left_without_with_nested_self` | a part in an item nested in a scope was replaced by a visitor without `with_nested_self()` | fixed |
//! | `f46_the_help_for_a_relative_std_path_suggests_the_crate_itself` | the help for a relative path that starts with `core`, `std` or `alloc` suggested `::my_crate::core::fmt::Debug` | fixed |
//! | `f47_self_in_an_outlives_predicate_keeps_the_bounds_of_the_description_of_self_only` | `Self: 'x` made the context of an entry keep the bounds that F26 leaves out: E0277 in the copy | fixed |
//! | `f48_a_type_corrected_for_a_signature_is_left_only_in_a_signature_like_it` | a type copied as written for the signature it is in (E0195, E0106) was left so in every other item and signature of its context | fixed |
//! | `f49_a_lifetime_that_another_input_keeps_breaks_elision` | the E0106 check left out the lifetime that the inputs keep, but rustc counts the lifetimes of each input apart: a leaked type in another input that names it breaks the elision | fixed |
//! | `f50_the_relative_types_in_the_arguments_of_a_value_path_are_paths_of_its_error` | the paths of the error of a value path left out the relative types in its arguments: a path rule after the analysis dropped the error, and the copy wrote them as they were | fixed |
//! | `f51_a_nested_item_with_a_bound_that_names_a_leaked_type_is_rejected` | `enter_mod` with `nested_self_entries()` accepted `S<T: AsRef<Rel<Self>>>` beside an entry that uses `T`, which `intern_struct()` rejects: E0277 in the copy | fixed |
//! | `f52_an_unrelated_error_does_not_make_an_optional_entry_of_the_predicate` | a part copied as written with `Self::A` in a predicate made no optional entry of the types in the bounds of its context a part too, so a strict Referrer replaced them (E0277) | fixed |
//!
//! No finding is a `bug` any more: each is `fixed` (a regression test) or a `limitation` (it pins
//! behaviour that the docs of `Leaker` document, or that follows from what rustc accepts). No test
//! is `#[ignore]`d.
//!
//! Most fixes are rejections: the Leaker reports an error for an item whose copy rustc cannot
//! compile. The test of such an item asserts the rejection, with the exact headline of the error and
//! its note (`assert_rejected()`, `assert_unstable()`), and not an oracle on the copy: an oracle
//! that runs for the accepted items only (`let Ok(..) = .. else { continue }`) runs for none of
//! them, and would pass if the rejection regressed. The oracle (what the copy of an accepted item
//! must not have) is kept for the controls, the items that compile and stay accepted, and it is
//! checked on a copy written by hand (`the_oracles_of_the_findings_find_a_hand_written_cycle`).
//!
//! The test of a finding says how the property test shows it (`TYPE_LEAK_PROP_FEATURES=..`). The
//! features of the fixed findings are on by default in the property test: the Leaker now rejects
//! what they generate (the rejections are counted), or rewrites it. F22 to F24 were rejections of what compiles: a rejection is
//! counted as a success by the property test, which cannot show them (nor F29, F31 and F32, which
//! are about a toolchain, a lint, or an error in the Leaker's crate that the Leaker cannot see).
//! The fix of F22 is checked by the property test the other way: the free functions and inherent
//! methods that it generates with a method lifetime in a leaked path are accepted, and their copies
//! must compile.
//! The tests of F22 to F35 were written from rustc's verdict on a hand-written Leaker's crate (the
//! trait or item, its Repeater's impls) and Referrer's crate (the copy, and for a trait an impl for a
//! host type), on stable (1.98.1) and nightly (1.100.0), which each doc says.
//!
//! F10 and F11 are errors of rustc on a copy of a module or an impl that the test macro makes
//! (E0277, E0207). The tests pin the behaviour of the library that is the cause, not the errors.
//!
//! The feature of F18 (`strict_anchors`) is off in the property test by default: the generator
//! only decides that an item with an external type that holds a method lifetime is not invoked as
//! strict.

use syn::punctuated::Punctuated;
use syn::visit::Visit;
use syn::visit_mut::VisitMut;
use syn::*;
use template_quote::quote;
use type_leak::*;

// ── helpers ──────────────────────────────────────────────────────────────────

/// The config of the property test: primitive types and the names of the standard prelude are
/// allowed.
fn new_leaker() -> Leaker {
    let mut config = LeakerConfig::new();
    config.allow_primitive().allow_prelude();
    Leaker::from_config(config)
}

/// Analyzes `item` with `intern_*`. The errors are read through `Leaker::result()`.
fn intern(leaker: &mut Leaker, item: &Item) -> ContextId {
    match item {
        Item::Struct(item) => leaker.intern_struct(item, true),
        Item::Trait(item) => leaker.intern_trait(item, true),
        Item::Impl(item) => leaker.intern_impl(item, true),
        Item::Mod(item) => leaker.intern_mod(item),
        _ => panic!("unsupported item"),
    }
}

/// Analyzes `item` with `leaker`, and returns the Referrer with the top-level context, or the
/// errors that remain.
fn analyze_in(
    mut leaker: Leaker,
    item: &Item,
) -> std::result::Result<(ContextId, Referrer), Vec<ErrorItem>> {
    let context = intern(&mut leaker, item);
    leaker.result().finish().map(|referrer| (context, referrer))
}

/// The errors that remain after the analysis of `leaker`.
fn errors_of(leaker: Leaker) -> Vec<ErrorItem> {
    leaker.result().errors().to_vec()
}

fn string<T: template_quote::ToTokens>(t: &T) -> String {
    quote!(#t).to_string()
}

/// Asserts that the Leaker rejected `item`, with exactly the `messages` (the headline of each error,
/// in order, joined with a line break by `joined_messages()`), and returns the errors.
///
/// A bug input has to be rejected. A loop that skips the rejected items (`let Ok(..) = .. else {
/// continue }`) runs its oracle on none of them, and pins nothing: it would be the same if the
/// rejection regressed, and the oracle runs for the items that are accepted only.
fn assert_rejected<T>(
    item: &impl template_quote::ToTokens,
    result: std::result::Result<T, Vec<ErrorItem>>,
    messages: &[&str],
) -> Vec<ErrorItem> {
    let err = match result {
        Ok(_) => panic!("`{}` is accepted, and it does not compile", string(item)),
        Err(err) => err,
    };
    assert_eq!(
        joined_messages(&err),
        messages.join("\n"),
        "`{}`",
        string(item)
    );
    err
}

/// The messages of `errors`, one per line.
fn joined_messages(errors: &[ErrorItem]) -> String {
    errors
        .iter()
        .map(|item| item.message.as_str())
        .collect::<Vec<_>>()
        .join("\n")
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

/// Tokens as a help writes them: `Wrap < T >` is `Wrap<T>`, and `Self :: A` is `Self::A`.
fn plain(text: &str) -> String {
    [
        (" :: ", "::"),
        (":: ", "::"),
        (" ::", "::"),
        (" <", "<"),
        ("< ", "<"),
        (" >", ">"),
        ("& ", "&"),
        (" ,", ","),
    ]
    .iter()
    .fold(text.to_string(), |text, (from, to)| text.replace(from, to))
}

/// What the bound that an error of an unstable bound blames names (`Why` of `src/unstable.rs`).
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
    /// A type that is leaked too, in a bound of the impl that rustc proves to resolve `Self::A` by
    /// the impl of the trait.
    LeakedInImpl,
}

/// Asserts that the Leaker rejected `item` for the type `ty` (as the error writes it) in a bound or
/// where predicate, because the bound `bound` of its generics is one that the environment of the
/// copy cannot prove. The error has the headline, and the one note, that say what the bound
/// `blames` (`env_cycle()` and `self_env_cycle()` of `src/analyze.rs`). `ty` is `Self` for the
/// self type of an impl.
fn assert_unstable<T>(
    item: &impl template_quote::ToTokens,
    result: std::result::Result<T, Vec<ErrorItem>>,
    ty: &str,
    bound: &str,
    blames: Blames,
) -> Vec<ErrorItem> {
    // What the bound names, after "the bound `..` of its generics".
    let names = |blames: Blames| match blames {
        Blames::SelfType => "names `Self`, which the copy writes as the self type of the impl, a \
                             path through the Repeater that rustc has not normalized yet in the \
                             environment, where it must prove the bound (E0277, E0271)"
            .to_string(),
        Blames::SelfAssoc(path) => format!(
            "names `{}`, which rustc normalizes to the associated type of the host in the impl of \
             the Repeater, but not yet in the environment (E0277, E0271)",
            path
        ),
        Blames::Projection(path) => format!(
            "names `{}`, which rustc normalizes in the impl of the Repeater (an impl, a predicate \
             of the item that equates it, or a supertrait of a bound of its generic, defines it), \
             but not yet in the environment (E0277, E0271, E0275)",
            path
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
            format!("`{}`", path),
            format!("write the bound without `{}`", plain(path)),
        ),
    };
    let headline = format!(
        "`{}` is in a bound or where predicate, and a bound of its generics names {}",
        ty, what
    );
    let err = assert_rejected(item, result, &[&headline]);
    // The note says why, and the help what to write.
    assert_eq!(all_notes(&err), [why], "`{}`", string(item));
    let help = format!(
        "{}, or use `{}` outside the bounds and where predicates",
        advice,
        plain(ty)
    );
    assert_eq!(all_helps(&err), [help], "`{}`", string(item));
    err
}

/// Rewrites a copy of `item` (in the scope `context`), replacing the `ix`-th entry with `P<ix>`.
fn rewrite(referrer: &Referrer, context: ContextId, mut item: Item) -> String {
    referrer
        .clone()
        .visitor(context, |_, ix| parse_quote!(P<#ix>))
        .visit_item_mut(&mut item);
    string(&item)
}

fn param_arg(param: &GenericParam) -> GenericArgument {
    match param {
        GenericParam::Lifetime(l) => GenericArgument::Lifetime(l.lifetime.clone()),
        GenericParam::Type(t) => {
            let ident = &t.ident;
            GenericArgument::Type(parse_quote!(#ident))
        }
        GenericParam::Const(c) => {
            let ident = &c.ident;
            GenericArgument::Type(parse_quote!(#ident))
        }
    }
}

/// The index of the entry `ty`.
fn entry(referrer: &Referrer, ty: &str) -> usize {
    referrer
        .iter()
        .position(|t| string(t) == ty)
        .unwrap_or_else(|| panic!("no entry {}", ty))
}

// ── F1 ───────────────────────────────────────────────────────────────────────

/// The names of the `?Sized` type parameters of `generics`.
fn unsized_params(generics: &Generics) -> Vec<Ident> {
    let maybe = |bounds: &Punctuated<TypeParamBound, Token![+]>| {
        bounds.iter().any(|b| {
            matches!(
                b,
                TypeParamBound::Trait(TraitBound {
                    modifier: TraitBoundModifier::Maybe(_),
                    ..
                })
            )
        })
    };
    let mut out: Vec<Ident> = generics
        .params
        .iter()
        .filter_map(|param| match param {
            GenericParam::Type(t) if maybe(&t.bounds) => Some(t.ident.clone()),
            _ => None,
        })
        .collect();
    for predicate in generics.where_clause.iter().flat_map(|w| &w.predicates) {
        if let WherePredicate::Type(PredicateType {
            bounded_ty: Type::Path(TypePath { qself: None, path }),
            bounds,
            ..
        }) = predicate
        {
            if let (Some(ident), true) = (path.get_ident(), maybe(bounds)) {
                out.push(ident.clone());
            }
        }
    }
    out
}

/// A `?Sized` parameter of `generics` that `encoded` has as a bare tuple element, which is not the
/// last: a tuple is well-formed only if its last element is unsized.
fn bare_unsized_element(generics: &Generics, encoded: &Type) -> Option<Ident> {
    let unsized_params = unsized_params(generics);
    let Type::Tuple(tuple) = encoded else {
        return None;
    };
    let last = tuple.elems.len().checked_sub(1)?;
    tuple.elems.iter().take(last).find_map(|elem| match elem {
        Type::Path(TypePath { qself: None, path }) => path
            .get_ident()
            .filter(|ident| unsized_params.contains(ident))
            .cloned(),
        _ => None,
    })
}

/// F1: a `?Sized` type parameter is encoded as a bare tuple element.
///
/// Trigger: `struct S<T: ?Sized, U>(Rel<T, U>)`: the entry `Rel<T, U>` uses `<T: ?Sized, U>`, and
/// the Repeater's trait arguments encoded them as the tuple `(T, U)`. A tuple is well-formed only
/// if all its elements but the last are `Sized`, so the impl that a macro emits for the entry did
/// not compile (E0277). Found by the property test, which now generates `?Sized` parameters.
///
/// The fix encodes a parameter with a `?Sized` bound as `*const T`, which is `Sized` whatever `T`
/// is, and the other parameters as before. Both sides agree because both see the bound: the
/// Leaker encodes `Referrer::generics()` of the entry, and the Referrer encodes the arguments with
/// the same generics as parameters.
///
/// The assertions: the encoder succeeds, no `?Sized` parameter is a bare element but the last
/// (the Repeater's argument `T` is `?Sized`, so the last would be fine), and the encoding of the
/// arguments is that of the parameters.
#[test]
fn f01_unsized_type_param_encoding() {
    let slots = RepeaterSlots::new(parse_quote!(crate::Repeater));
    let check = |generics: &Generics, expected: Option<&str>, what: &str| {
        let encoded = slots
            .encode_generics_params_to_ty(&generics.params)
            .unwrap_or_else(|e| panic!("{}: `{}`: {}", what, string(generics), e));
        assert_eq!(
            bare_unsized_element(generics, &encoded),
            None,
            "{}: `{}` encodes `{}` with an unsized parameter in the middle",
            what,
            string(generics),
            string(&encoded)
        );
        if let Some(expected) = expected {
            let expected: Type = parse_str(expected).unwrap();
            assert_eq!(string(&encoded), string(&expected), "{}", what);
        }
        // The Referrer passes each parameter by its own name: the same encoding.
        let args: Vec<GenericArgument> = generics.params.iter().map(param_arg).collect();
        let on_referrer_side = slots
            .encode_generics_args_to_ty(&args, &generics.params)
            .unwrap();
        assert_eq!(string(&on_referrer_side), string(&encoded), "{}", what);
    };

    // The encoder on its own: an unsized parameter first, in the middle and last.
    check(
        &parse_quote!(<T: ?Sized, U>),
        Some("(*const T, U)"),
        "the encoder",
    );
    check(
        &parse_quote!(<T, U: ?Sized, V>),
        Some("(T, *const U, V)"),
        "the encoder",
    );
    check(
        &parse_quote!(<T, U: ?Sized>),
        Some("(T, *const U)"),
        "the encoder",
    );
    check(
        &parse_quote!(<T: ?Sized>),
        Some("(*const T,)"),
        "the encoder",
    );
    // Other bounds stay out of the way, and a lifetime bound is encoded as it is for a sized one.
    check(
        &parse_quote!(<'a, T: Clone + ?Sized + 'a, U>),
        Some("(&'a (), *const T, U, &'a T)"),
        "the encoder",
    );
    // A sized parameter is encoded as before.
    check(&parse_quote!(<T, U: Sized>), Some("(T, U)"), "the encoder");

    // The Referrer's arguments: a pointer to the argument, whatever it is.
    let generics: Generics = parse_quote!(<T: ?Sized, U: ?Sized, V>);
    let args: AngleBracketedGenericArguments =
        parse_quote!(<str, dyn Tr + Send + 'static, [u8; 2]>);
    let encoded = slots
        .encode_generics_args_to_ty(&args.args, &generics.params)
        .unwrap();
    let expected: Type = parse_quote!((*const str, *const (dyn Tr + Send + 'static), [u8; 2]));
    assert_eq!(string(&encoded), string(&expected));

    // The generics of the entry that the Leaker finds: the bound is on the parameter, wherever it
    // was written (a where predicate is moved there).
    let items: Vec<ItemStruct> = vec![
        parse_quote!(
            pub struct S<T: ?Sized, U>(Rel<T, U>);
        ),
        parse_quote!(
            pub struct S<A, B: ?Sized, C>(Rel<A, B, C>);
        ),
        parse_quote!(
            pub struct S<T, U: ?Sized>(Rel<T, U>);
        ),
        parse_quote!(
            pub struct S<T, U>(Rel<T, U>)
            where
                T: ?Sized;
        ),
        parse_quote!(
            pub struct S<'a, T: ?Sized + 'a, U>(&'a Rel<T, U>);
        ),
    ];
    for item in items {
        let mut leaker = new_leaker();
        leaker.intern_struct(&item, true);
        // A rejection is fine.
        let Ok(referrer) = leaker.result().finish() else {
            continue;
        };
        let mut entries = 0;
        for (_, generics) in referrer.iter_with_generics() {
            if generics.params.iter().any(|param| {
                matches!(param, GenericParam::Type(t) if unsized_params(generics).contains(&t.ident))
            }) {
                entries += 1;
                check(generics, None, &string(&item));
            }
        }
        assert!(
            entries > 0,
            "`{}` has no entry with a `?Sized` parameter",
            string(&item)
        );
    }
}

// ── F2 ───────────────────────────────────────────────────────────────────────

/// Finds the replacements `P<..>` of entries in a copy: in the bounds of a type parameter, and in
/// where predicates.
#[derive(Default)]
struct EntryMentions {
    in_bounds: Vec<String>,
    in_where: Vec<String>,
}

/// Whether the tokens of `node` have a replacement `P<..>`.
fn has_replacement(node: &impl template_quote::ToTokens) -> bool {
    struct Find(bool);
    impl<'ast> Visit<'ast> for Find {
        fn visit_path_segment(&mut self, seg: &'ast PathSegment) {
            self.0 |= seg.ident == "P" && !seg.arguments.is_none();
            syn::visit::visit_path_segment(self, seg)
        }
    }
    let mut find = Find(false);
    if let Ok(ty) = parse2::<Type>(quote!(#node)) {
        find.visit_type(&ty);
    }
    find.0
}

impl<'ast> Visit<'ast> for EntryMentions {
    fn visit_type_param(&mut self, i: &'ast TypeParam) {
        for bound in &i.bounds {
            if has_bound_replacement(bound) {
                self.in_bounds.push(string(i));
            }
        }
        syn::visit::visit_type_param(self, i)
    }
    fn visit_predicate_type(&mut self, i: &'ast PredicateType) {
        if has_replacement(&i.bounded_ty) || i.bounds.iter().any(has_bound_replacement) {
            self.in_where.push(string(i));
        }
        syn::visit::visit_predicate_type(self, i)
    }
}

fn has_bound_replacement(bound: &TypeParamBound) -> bool {
    // A bound is not a type: parse the bound as the bound of a trait object.
    parse2::<Type>(quote!(dyn #bound)).map_or(false, |ty| has_replacement(&ty))
}

/// F2: a bound that names an entry, and a where predicate that names an entry, in one item.
///
/// Trigger: `struct S<T: Fn(Rel)>(Wrap<T>) where Wrap<T>: Sized`, and the trait forms (`trait
/// Tr<T: Fn(Rel)> { fn f(&self, a: Wrap<T>) where Wrap<T>: Sized; }`, and the same with `fn f<T:
/// Fn(Rel)>`). `Rel` and `Wrap<T>` are entries. The Repeater's impl for `Wrap<T>` has `T: Fn(Rel)`
/// of the Leaker, whereas the copy has `T: Fn(<() as Repeater<..>>::Type)`, and the predicate
/// `<..>::Type: Sized` of the copy makes rustc normalize `Wrap<T>` in its environment, which needs
/// that impl, and so the bound, which names an entry again: it cannot (E0277 "expected an
/// `Fn(Rel)` closure, found `T`"; the same family: `T: Into<Entry>`, `T: AsRef<..>`, `T:
/// Iterator<Item = ..>`, with overflows E0275 and E0284). Each alone is fine: with `where T: Sized`
/// instead (the control `Tr3`), or without the predicate, the copy compiles. Found by the property
/// test with `TYPE_LEAK_PROP_FEATURES=bound_cycles`.
///
/// The assertion is the least that the current output violates, because the fix is a design
/// decision (reject, or write the bound in the Repeater's impl as the copy has it): either the
/// Leaker rejects the item, or its copy (the entries replaced by `P<ix>`) does not have both a bound
/// and a where predicate that mention a replacement.
///
/// Fixed by rejecting what rustc cannot compile (see "Bounds that name leaked types" in the
/// `Leaker` docs and `tests/bound_cycles.rs`). The control `Tr3` is not rejected: what compiles
/// stays accepted.
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
            analyze_in(new_leaker(), item),
            "Wrap < T >",
            "T : Fn (Rel)",
            Blames::Leaked,
        );
    }
    // The control: `T: Sized` names no entry. It is accepted, and its copy has no bound and where
    // predicate that mention a replacement.
    let control: Item = parse_quote!(
        pub trait Tr3 {
            fn f<T: Fn(Rel)>(&self, a: Wrap<T>)
            where
                T: Sized;
        }
    );
    let (context, referrer) = analyze_in(new_leaker(), &control).unwrap_or_else(|e| {
        panic!(
            "the control is rejected, though it compiles: {}",
            joined_messages(&e)
        )
    });
    let mut copy = control.clone();
    referrer
        .clone()
        .visitor(context, |_, ix| parse_quote!(P<#ix>))
        .visit_item_mut(&mut copy);
    let mut mentions = EntryMentions::default();
    mentions.visit_item(&copy);
    assert!(
        mentions.in_bounds.is_empty() || mentions.in_where.is_empty(),
        "in the copy `{}`, the bounds {:?} and the where predicates {:?} name entries",
        string(&copy),
        mentions.in_bounds,
        mentions.in_where
    );
    // The oracle finds a copy that has both (it is the copy of F2, written by hand).
    let cyclic: Item = parse_quote!(
        pub trait Tr<T: Fn(P<0usize>)> {
            fn f(&self, a: P<1usize>)
            where
                P<1usize>: Sized;
        }
    );
    let mut mentions = EntryMentions::default();
    mentions.visit_item(&cyclic);
    assert_eq!(mentions.in_bounds.len(), 1, "{:?}", mentions.in_bounds);
    assert_eq!(mentions.in_where.len(), 1, "{:?}", mentions.in_where);
}

// ── F3 ───────────────────────────────────────────────────────────────────────

/// F3: a where predicate that names a lifetime drags its type parameter into the context.
///
/// Trigger: `struct S<'a, T> where T: Iterator<Item = Bl<'a>> {..}`, a where predicate (not the
/// inline bound `T: Iterator<Item = Bl<'a>>`) that names the lifetime `'a` in a leaked type
/// `Bl<'a>`. The context of the entry becomes `<'a, T: Iterator<Item = Bl<'a>>>`, whereas with the
/// inline bound it is `<'a>`. The visitor computes the context without `T`, so it never matches,
/// and the type stays as written (in another crate, `crate::base::Bl<'a>` cannot be resolved). Found
/// by the property test with `TYPE_LEAK_PROP_FEATURES=bound_cycles`.
///
/// The assertion: the entry is found, and so rewritten, whether the bound is inline or a where
/// predicate (and in a field). That does not say which of the two contexts is right.
///
/// Fixed: a where predicate on a generic is a bound of it, as the inline one is, so the context of
/// `Bl<'a>` is `<'a>` in both (the Referrer's scope has the bound inline, the Leaker's had it in
/// the where clause, and the predicate that names `'a` pulled `T` into the context).
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
        let (context, referrer) = analyze_in(new_leaker(), &item).unwrap();
        let copy = rewrite(&referrer, context, item.clone());
        assert!(
            !copy.contains("Bl"),
            "the entry `Bl<'a>` is not rewritten: {}",
            copy
        );
    }
}

// ── F4 ───────────────────────────────────────────────────────────────────────

/// F4 (fixed): an entry that names an item of the analyzed module is keyed by the module.
///
/// Trigger: `mod mm { struct R0; struct A(crate::base::B1<R0>); }`, and a constant of another
/// module in a type argument (`RC<{ crate::base::LEN }>`, with `RC` declared in the module). The
/// type is an entry, because `crate::base::B1` cannot be leaked alone. It names a name that the
/// module declares, so it means something only in the module, like a type with a relative path:
/// but it was shared by all modules, since its own path starts with `crate::`. The Repeater impl
/// of such an entry is emitted outside the module, where `R0` is not, and the copy of the module
/// declares `R0` again, so the Referrer hid the entry and left `crate::base` as written, which
/// cannot resolve in another crate (E0433). Found by the property test with
/// `TYPE_LEAK_PROP_FEATURES=module_crate_paths`.
///
/// The fix keys the entry by its module, as `Rel<R0>` is ("Modules": an entry is keyed by its
/// module if it depends on it). Rejecting it would be inconsistent: a relative path in that place
/// is accepted, and `crate::` paths are context-free for the rules of keys otherwise. The
/// assertions: the module is accepted; the entry is keyed by the module (it comes with the path of
/// its module, `[]`, so a macro emits its impl inside it, where `R0` and `crate::base` resolve);
/// the copy of the module has it replaced; and an entry that names no item of the module
/// (`crate::base::B1<u8>`) stays shared by all modules. As for F10, the copy refers to the
/// Leaker's `R0`, not to its own.
#[test]
fn f04_module_crate_relative_path_not_rewritten() {
    let mods: Vec<(ItemMod, &str)> = vec![
        (
            parse_quote! {
                pub mod mm {
                    pub struct R0;
                    pub struct A(pub crate::base::B1<R0>);
                }
            },
            "crate :: base :: B1 < R0 >",
        ),
        (
            parse_quote! {
                pub mod mm1 {
                    pub struct RC<const N: usize>;
                    pub struct S(pub RC<{ crate::base::LEN }>);
                }
            },
            "RC < { crate :: base :: LEN } >",
        ),
    ];
    for (item, expected) in mods {
        let mut leaker = new_leaker();
        let context = leaker.intern_mod(&item);
        let referrer = leaker.result().finish().unwrap();
        assert_eq!(referrer.iter().map(string).collect::<Vec<_>>(), [expected]);
        // Keyed by the module, which is the one analyzed.
        let module = referrer.context_module(referrer.context(0)).unwrap();
        assert_eq!(module, context);
        assert_eq!(referrer.module_path(module), Some(&[][..]));
        // The copy of the module: nothing is left of the path.
        let mut file = File {
            shebang: None,
            attrs: Vec::new(),
            items: item.content.clone().unwrap().1,
        };
        referrer
            .clone()
            .visitor(context, |_, ix| parse_quote!(P<#ix>))
            .visit_file_mut(&mut file);
        let copy = string(&file);
        assert!(copy.contains("P < 0usize >"), "{}", copy);
        assert!(!copy.contains("base"), "{}", copy);
    }

    // The control: a type that names no item of the module is shared by all modules (it is the
    // same entry in `inner`), and one that names an item is keyed by the module it is written in.
    let item: ItemMod = parse_quote! {
        pub mod mm {
            pub struct R0;
            pub struct A(pub crate::base::B1<u8>, pub Vec<crate::base::B1<R0>>);
            pub mod inner { pub struct B(pub crate::base::B1<u8>); }
        }
    };
    let mut leaker = new_leaker();
    leaker.intern_mod(&item);
    let referrer = leaker.result().finish().unwrap();
    let shared = entry(&referrer, "crate :: base :: B1 < u8 >");
    assert_eq!(referrer.context_module(referrer.context(shared)), None);
    let keyed = entry(&referrer, "crate :: base :: B1 < R0 >");
    assert!(referrer.context_module(referrer.context(keyed)).is_some());
    assert_eq!(referrer.iter().count(), 2);
}

// ── F5 ───────────────────────────────────────────────────────────────────────

/// F5 (limitation): no `T: 'a` is recorded for a path with a lifetime; the bound written is.
///
/// The `Leaker` docs, "Outlives bounds": the Leaker cannot read the definition of a path, so a path
/// with a lifetime argument records nothing, even if its definition needs it. Trigger:
/// `impl<'a, T> E<'a, T>` with `struct E<'a, T> { f: &'a mut T }`, where rustc infers `T: 'a`. The
/// impl compiles through the implied bounds of its self type, but the Repeater's impl for the entry
/// `E<'a, T>` has no `T: 'a`: E0309 in the Leaker's crate. Found by the property test with
/// `TYPE_LEAK_PROP_FEATURES=inferred_outlives`.
///
/// It is a limitation and not a bug: a guessed bound would make the copy stricter than the
/// original for a path that needs none (`struct Unrelated<'a, T>(&'a (), T)`). The fix is to write
/// it: then it is recorded, and encoded in the trait arguments as `&'a T`, which the impl does not
/// declare (`impl_generics()`).
#[test]
fn f05_inferred_outlives_of_self_type() {
    let slots = RepeaterSlots::new(parse_quote!(crate::Repeater));
    let encode = |referrer: &Referrer, ix: usize| {
        let (_, generics) = referrer.iter_with_generics().nth(ix).unwrap();
        slots
            .encode_generics_params_to_ty_with_self(
                &generics.params,
                &parse_quote!(Self),
                referrer.context_self_outlives(referrer.context(ix)),
            )
            .unwrap()
    };

    // As written: nothing is recorded, so the impl cannot prove `T: 'a`.
    let item: ItemImpl = parse_quote! {
        impl<'a, T> E<'a, T> { pub fn f(&self) -> Rel { loop {} } }
    };
    let mut leaker = new_leaker();
    leaker.intern_impl(&item, true);
    let referrer = leaker.result().finish().unwrap();
    let ix = entry(&referrer, "E < 'a , T >");
    let generics = referrer.generics(ix);
    assert_eq!(string(generics), "< 'a , T >");
    assert_eq!(encode(&referrer, ix), parse_quote!((&'a (), T)));

    // With the bound written: recorded, and encoded.
    let item: ItemImpl = parse_quote! {
        impl<'a, T: 'a> E<'a, T> { pub fn f(&self) -> Rel { loop {} } }
    };
    let mut leaker = new_leaker();
    leaker.intern_impl(&item, true);
    let referrer = leaker.result().finish().unwrap();
    let ix = entry(&referrer, "E < 'a , T >");
    assert_eq!(string(referrer.generics(ix)), "< 'a , T : 'a >");
    assert_eq!(encode(&referrer, ix), parse_quote!((&'a (), T, &'a T)));
    // The impl declares the generics without the encoded bound.
    let impl_generics = referrer.impl_generics(referrer.context(ix));
    assert_eq!(string(&impl_generics), "< 'a , T >");
}

// ── F6 ───────────────────────────────────────────────────────────────────────

/// F6 (fixed): `intern_impl` accepts `Self::A` of the implemented trait in a leaked type.
///
/// Trigger: `impl ::leaker::m::Tr for u8 { type A = u8; fn f(&self, x: Rel<Self::A>) {} }`.
/// `Rel<Self::A>` is an entry, and the impl that a macro emits is `impl Repeater<..> for u8 {
/// type Type = Rel<Self::A>; }`, where `Self::A` is not an associated type of the Repeater (E0220).
/// The context says that `Self` is `u8`, and nothing says which trait `A` belongs to. Found by the
/// property test with `TYPE_LEAK_PROP_FEATURES=impl_assoc_self`.
///
/// The assertion: the Leaker reports an error for the impl (also if only a bound of the generics
/// of the entry names `Self::A`), and says so in the headline. A type that names
/// `Self::A` is never leaked whole, so where it holds no path that must be leaked
/// (`Vec<Self::A>`) it is copied as written and works, and where it does (`fn(Rel) -> Self::A`)
/// only its parts are entries. The type with the trait named (`<Self as ::leaker::m::Tr>::A`) and
/// `Self::A` in a trait, where the Repeater's impl is for the implementor, are fine.
#[test]
fn f06_impl_assoc_type_of_self_in_repeater() {
    let item: ItemImpl = parse_quote! {
        impl ::leaker::m::Tr for u8 {
            type A = u8;
            fn f(&self, x: Rel<Self::A>) {}
        }
    };
    let mut leaker = new_leaker();
    leaker.intern_impl(&item, true);
    let result = leaker.result();
    let err = result.errors();
    assert!(
        !err.is_empty(),
        "accepted, with the entries {:?}",
        result
            .clone()
            .into_referrer()
            .iter()
            .map(string)
            .collect::<Vec<_>>()
    );
    // The headline says what to write, with the trait of the impl.
    assert_eq!(
        joined_messages(err),
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
    let mut leaker = new_leaker();
    leaker.intern_impl(&item, true);
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
        let mut leaker = new_leaker();
        leaker.intern_impl(&item, true);
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
    let mut leaker = new_leaker();
    leaker.intern_impl(&item, true);
    let mut entries: Vec<String> = leaker
        .result()
        .finish()
        .unwrap()
        .iter()
        .map(string)
        .collect();
    entries.sort();
    assert_eq!(entries, ["Rel", "Rel2"]);
    let item: ItemTrait = parse_quote! {
        pub trait Tr { type A; fn f(&self, x: Rel<Self::A>); }
    };
    let mut leaker = new_leaker();
    leaker.intern_trait(&item, true);
    assert_eq!(
        string(leaker.result().finish().unwrap().iter().next().unwrap()),
        "Rel < Self :: A >"
    );
}

// ── F7 ───────────────────────────────────────────────────────────────────────

/// F7 (fixed): a method lifetime in a leaked path, and in the return type.
///
/// Trigger: `fn f<'x>(&self, a: Wrap<'x>) -> &'x u8` (and `-> Wrap<'x>`), with `Wrap<'x>` leaked.
/// `'x` is late-bound in the trait, but the copy `fn f<'x>(&self, a: <Self as Repeater<(&'x (),)>>
/// ::Type) -> &'x u8` names it only in a projection, which does not constrain it, so rustc makes it
/// early-bound and rejects the method (E0195). the `Leaker` docs, "Method lifetimes": the Leaker
/// reports an error for a trait object leaked as a whole, and the same holds for a leaked path.
/// Found by the property test with `TYPE_LEAK_PROP_FEATURES=method_lifetime_paths`.
///
/// The assertion: the Leaker reports an error for those traits, with the E0195
/// note: a type leaked whole holds `'x` in the copy, and no other input constrains it as written.
/// `a: &'x u8` keeps `'x`, so that trait is accepted (the control), as are the other exemptions: the
/// receiver keeps it, a type that is not leaked keeps it (`Option<&'x str>`, a function pointer),
/// a bound makes it early-bound in both, and no input (or no output) mentions it.
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
        let mut leaker = new_leaker();
        leaker.intern_trait(&item, true);
        let result = leaker.result();
        let err = result.errors();
        assert!(!err.is_empty(), "`{}` is accepted", item.ident);
        let note = err[0].note().unwrap();
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
        let mut leaker = new_leaker();
        leaker.intern_trait(&item, true);
        leaker.result().finish().unwrap();
    }
}

// ── F8 ───────────────────────────────────────────────────────────────────────

/// The number of bounds of the bare trait object that a type ends in, through references and
/// pointers: a `+` after the type joins them (`&dyn Any + Clone`), and a type with several of them
/// is ambiguous even alone (`Fn() -> dyn Any + Send`).
fn bare_object_bounds(ty: &Type) -> Option<usize> {
    match ty {
        Type::Reference(r) => bare_object_bounds(&r.elem),
        Type::Ptr(p) => bare_object_bounds(&p.elem),
        Type::TraitObject(TypeTraitObject { bounds, .. })
        | Type::ImplTrait(TypeImplTrait { bounds, .. }) => Some(bounds.len()),
        Type::BareFn(f) => match &f.output {
            ReturnType::Type(_, t) => bare_object_bounds(t),
            ReturnType::Default => None,
        },
        _ => None,
    }
}

/// The bounds that rustc reads with an ambiguous `+`: a bound of the sugar of `Fn` whose return
/// type ends in a bare trait object, followed by another bound (or with several bounds itself).
fn ambiguous_plus(bounds: &Punctuated<TypeParamBound, Token![+]>) -> Vec<String> {
    let count = bounds.len();
    bounds
        .iter()
        .enumerate()
        .filter(|(ix, bound)| match bound {
            TypeParamBound::Trait(TraitBound { path, .. }) => matches!(
                path.segments.last().map(|seg| &seg.arguments),
                Some(PathArguments::Parenthesized(ParenthesizedGenericArguments {
                    output: ReturnType::Type(_, ty),
                    ..
                })) if matches!(bare_object_bounds(ty), Some(n) if ix + 1 < count || n > 1)
            ),
            _ => false,
        })
        .map(|(_, bound)| string(bound))
        .collect()
}

/// The bounds of the parameters and where predicates of `generics` with an ambiguous `+`.
fn ambiguous_in(generics: &Generics) -> Vec<String> {
    let mut ambiguous = Vec::new();
    for param in &generics.params {
        if let GenericParam::Type(t) = param {
            ambiguous.extend(ambiguous_plus(&t.bounds));
        }
    }
    for predicate in generics.where_clause.iter().flat_map(|w| &w.predicates) {
        if let WherePredicate::Type(p) = predicate {
            ambiguous.extend(ambiguous_plus(&p.bounds));
        }
    }
    ambiguous
}

/// F8: the bounds of a generic are merged and printed with an ambiguous `+`.
///
/// Trigger: `struct S<T: Clone>(Rel<T>) where T: Fn(()) -> &'static dyn Any`. The where predicate
/// is merged into the bounds of `T`, and the generics of the entry are printed `T: Fn(()) ->
/// &'static dyn Any + Clone`, where `+ Clone` joins the trait object (rustc: "ambiguous `+` in a
/// type", E0225). The return type needs parentheses (`-> (&'static dyn Any)`), or the bounds
/// another order. Found by the property test with `TYPE_LEAK_PROP_FEATURES=bound_merge_syntax`.
///
/// The assertion is on the generics that a macro prints for the Repeater's impl
/// (`impl_generics()`): no bound of the sugar of `Fn` with such a return type is followed by
/// another bound, and both bounds are still there. (Parsing the printed generics back with syn
/// does not show it: syn accepts what rustc rejects.)
///
/// Fixed: the canonical bounds of a context parenthesize the return type of a bound that another
/// bound follows (`Fn(()) -> (&'static dyn Any) + Clone`), or that has several bounds itself, and
/// do not depend on the parentheses that the source has. The second part checks other shapes.
#[test]
fn f08_merged_bound_ambiguous_plus() {
    let item: ItemStruct = parse_quote! {
        pub struct S<T: Clone>(pub Rel<T>)
        where
            T: ::core::ops::Fn(()) -> &'static dyn ::core::any::Any;
    };
    let mut leaker = new_leaker();
    leaker.intern_struct(&item, true);
    let referrer = leaker.result().finish().unwrap();
    let ix = entry(&referrer, "Rel < T >");
    let generics = referrer.impl_generics(referrer.context(ix));
    let printed = string(&generics);
    let ambiguous = ambiguous_in(&generics);
    assert!(
        ambiguous.is_empty(),
        "`{}` has an ambiguous `+` after {:?}",
        printed,
        ambiguous
    );
    assert!(
        printed.contains("Clone") && printed.contains("Any"),
        "a bound is lost in `{}`",
        printed
    );

    // Other shapes, and how many bounds each keeps. The bounds are written with and without the
    // parentheses that they need, and in another order: they must give the same generics.
    let items: Vec<ItemStruct> = vec![
        // Several bounds of this kind, through a pointer and a function pointer.
        parse_quote! {
            pub struct S<T: Clone + Send>(pub Rel<T>)
            where
                T: ::core::ops::Fn(()) -> &'static dyn ::core::any::Any,
                T: ::core::ops::FnMut() -> *const dyn ::core::any::Any,
                T: ::core::ops::FnOnce() -> fn() -> &'static dyn ::core::any::Any;
        },
        // The same, with the return types parenthesized in the source.
        parse_quote! {
            pub struct S<T: Clone + Send>(pub Rel<T>)
            where
                T: ::core::ops::Fn(()) -> (&'static dyn ::core::any::Any),
                T: ::core::ops::FnMut() -> (*const dyn ::core::any::Any),
                T: ::core::ops::FnOnce() -> (fn() -> &'static dyn ::core::any::Any);
        },
        // A return type with several bounds needs them even if the bound is the last.
        parse_quote! {
            pub struct S<T: ::core::ops::Fn(()) -> (dyn ::core::any::Any + Send)>(pub Rel<T>);
        },
        // A bound that the context gets from a lifetime that the entry names.
        parse_quote! {
            pub struct S<'a, T: ::core::ops::Fn(()) -> &'static dyn ::core::any::Any>(pub Rel<&'a T>);
        },
    ];
    for item in &items {
        let mut leaker = new_leaker();
        leaker.intern_struct(item, true);
        let referrer = leaker.result().finish().unwrap();
        for (ix, (_, generics)) in referrer.iter_with_generics().enumerate() {
            let ambiguous = ambiguous_in(generics);
            assert!(
                ambiguous.is_empty(),
                "the entry {} of `{}` has `{}`: ambiguous `+` after {:?}",
                ix,
                string(item),
                string(generics),
                ambiguous
            );
            let impl_generics = referrer.impl_generics(referrer.context(ix));
            assert_eq!(
                ambiguous_in(&impl_generics),
                Vec::<String>::new(),
                "`{}`",
                string(&impl_generics)
            );
        }
    }
    // The order and the parentheses of the source do not change the generics of the context.
    let contexts: Vec<String> = items
        .iter()
        .take(2)
        .map(|item| {
            let mut leaker = new_leaker();
            leaker.intern_struct(item, true);
            let referrer = leaker.result().finish().unwrap();
            let ix = entry(&referrer, "Rel < T >");
            string(referrer.context_generics(referrer.context(ix)))
        })
        .collect();
    assert_eq!(contexts[0], contexts[1]);
}

// ── F9 ───────────────────────────────────────────────────────────────────────

/// Whether a type is a function pointer without an output that is reached through a reference or a
/// pointer (also after the output of another function pointer: `fn() -> &fn()`): as the bounded
/// type of a where predicate, rustc reads the `:` after it as the start of its return type.
fn bare_fn_under_pointer(ty: &Type) -> bool {
    let mut under_pointer = false;
    let mut ty = ty;
    loop {
        match ty {
            Type::Reference(r) => {
                under_pointer = true;
                ty = &r.elem;
            }
            Type::Ptr(p) => {
                under_pointer = true;
                ty = &p.elem;
            }
            Type::BareFn(f) => match &f.output {
                ReturnType::Type(_, output) => ty = output,
                ReturnType::Default => return under_pointer,
            },
            _ => return false,
        }
    }
}

/// The bounded types of the where predicates that rustc cannot parse.
struct UnparenthesizedPredicates(Vec<String>);

impl<'ast> Visit<'ast> for UnparenthesizedPredicates {
    fn visit_predicate_type(&mut self, predicate: &'ast PredicateType) {
        if bare_fn_under_pointer(&predicate.bounded_ty) {
            self.0.push(string(predicate));
        }
        syn::visit::visit_predicate_type(self, predicate)
    }
}

/// The references and pointers to a trait object with several bounds, unparenthesized: `&dyn A +
/// B` is ambiguous.
struct UnparenthesizedPointees(Vec<String>);

impl<'ast> Visit<'ast> for UnparenthesizedPointees {
    fn visit_type_reference(&mut self, i: &'ast TypeReference) {
        if matches!(&*i.elem, Type::TraitObject(object) if object.bounds.len() > 1) {
            self.0.push(string(i));
        }
        syn::visit::visit_type_reference(self, i)
    }
    fn visit_type_ptr(&mut self, i: &'ast TypePtr) {
        if matches!(&*i.elem, Type::TraitObject(object) if object.bounds.len() > 1) {
            self.0.push(string(i));
        }
        syn::visit::visit_type_ptr(self, i)
    }
}

/// F9: `Self` replaced by a reference to a function pointer is printed without parentheses.
///
/// Trigger: `impl Tr for &'static unsafe fn(i32, R0)` with `where Self: 'y` on a method. A copy
/// where `Self` is the type has the predicate `&'static unsafe fn(i32, R0): 'y`, which rustc does
/// not parse (`return types are denoted using `->``): the bounded type needs parentheses.
/// `Visitor::with_self()` makes the replacement (`Referrer::iter_with_self()` uses the same one
/// for the generics of an entry). Found by the property test with
/// `TYPE_LEAK_PROP_FEATURES=fn_pointer_self`. Note that what failed to compile there is the
/// witness function of the test macro (`tests/crates/macro/prop.rs`, `SelfSubst`), which makes the
/// same replacement with its own code: it is the replacement in the library that is pinned here.
///
/// The assertion: no where predicate of the copy has a reference or a pointer to a function
/// pointer as the bounded type, unparenthesized (`fn(..): 'y` alone and `&dyn Any: 'y` are fine).
///
/// Fixed: the replacement parenthesizes a bounded type that needs it (`(&fn(..)): 'y`), in the
/// visitor, in `iter_with_self()` and where the Leaker rewrites a bound for another `Self`. The
/// same holds for the pointee of a reference or a pointer that `Self` replaces and that has several
/// bounds (`&(dyn Any + Send)`): the second part pins it.
#[test]
fn f09_fn_pointer_self_predicate_print() {
    let item: ItemTrait = parse_quote! {
        pub trait Tr {
            fn m<'y>(&self, a: u8) where Self: 'y;
            fn n(&self, a: u8) where Self: Clone;
            fn r(&self) -> Rel<Self> where Self: Clone;
        }
    };
    let mut leaker = new_leaker();
    let context = leaker.intern_trait(&item, true);
    let referrer = leaker.result().finish().unwrap();
    let self_tys: Vec<Type> = vec![
        parse_quote!(&'static unsafe fn(i32, R0)),
        parse_quote!(*const fn(i32)),
        parse_quote!(fn(i32) -> &'static fn(u8)),
        parse_quote!(fn(i32)),
    ];
    for self_ty in self_tys {
        let mut copy = Item::Trait(item.clone());
        referrer
            .clone()
            .visitor(context, |_, ix| parse_quote!(P<#ix>))
            .with_self(self_ty.clone())
            .visit_item_mut(&mut copy);
        let mut found = UnparenthesizedPredicates(Vec::new());
        found.visit_item(&copy);
        assert!(
            found.0.is_empty(),
            "Self = `{}`: unparenthesized bounded types, which rustc cannot parse: {:?}",
            string(&self_ty),
            found.0
        );
        // The generics of the entry `Rel<Self>`, which has `where Self: Clone`.
        let mut seen = false;
        for (_, generics) in referrer.iter_with_self(&self_ty) {
            let mut found = UnparenthesizedPredicates(Vec::new());
            found.visit_generics(&generics);
            seen |= generics.where_clause.is_some();
            assert!(
                found.0.is_empty(),
                "Self = `{}`: iter_with_self() has {:?}",
                string(&self_ty),
                found.0
            );
        }
        assert!(seen, "the entry `Rel<Self>` has no where predicate");
    }

    // The pointee of a reference and of a pointer.
    let item: ItemTrait = parse_quote! {
        pub trait Tr2 {
            fn p(&self, a: &Self, b: *const Self, c: Box<Self>) where &'static Self: Clone;
        }
    };
    let mut leaker = new_leaker();
    let context = leaker.intern_trait(&item, true);
    let referrer = leaker.result().finish().unwrap();
    let self_ty: Type = parse_quote!(dyn ::core::any::Any + Send + 'static);
    let mut copy = Item::Trait(item.clone());
    referrer
        .visitor(context, |_, ix| parse_quote!(P<#ix>))
        .with_self(self_ty)
        .visit_item_mut(&mut copy);
    let mut found = UnparenthesizedPointees(Vec::new());
    found.visit_item(&copy);
    assert!(
        found.0.is_empty(),
        "unparenthesized trait objects with several bounds, which are ambiguous: {:?}",
        found.0
    );
}

// ── F10 ──────────────────────────────────────────────────────────────────────

/// F10 (limitation): a bound of an entry names an item that the module declares.
///
/// `Leaker::enter_mod()` and `Referrer::context_module()` docs: the names that the module declares
/// are copied with it, so they are not leaked, and an entry is keyed by its module if a path in a
/// bound of its generics is relative. Trigger:
/// `mod mm { use super::base::B1 as Imp; trait Cv {} struct S<T: Cv>(pub Imp<T>); }`. The entry
/// `Imp<T>` has the context `<T: Cv>` and is keyed by `mm`; the Repeater's impl for it is emitted
/// in the Leaker's module, so it requires the Leaker's `Cv`. The copy of the module declares its
/// own `Cv`, so a projection `<.. as Repeater<(T,), ..>>::Type` in the copy needs
/// `T: leaker::mm::Cv`, which the copy's `T: Cv` is not (E0277). The generator of the property test
/// avoids it: the predicates of a module use no item of the module.
///
/// It is a limitation and not a bug: the entry is keyed by the module correctly, and a copy that
/// declares its own items cannot give them the identity of the Leaker's. The rustc error is a
/// matter of the copy that a macro makes, so this test pins what causes it in the library: the
/// entry keyed by the module, the bound naming the declared item as written, and the copy that
/// keeps the bound and replaces the entry.
#[test]
fn f10_module_bound_names_declared_item() {
    let item: ItemMod = parse_quote! {
        pub mod mm {
            use super::base::B1 as Imp;
            pub trait Cv {}
            pub struct S<T: Cv>(pub Imp<T>);
        }
    };
    let mut leaker = new_leaker();
    let context = leaker.intern_mod(&item);
    let referrer = leaker.result().finish().unwrap();
    let ix = entry(&referrer, "Imp < T >");
    // Keyed by the module, with the bound as written (`Cv` is not an entry: the module declares it).
    let module = referrer.context_module(referrer.context(ix)).unwrap();
    assert_eq!(module, context);
    assert_eq!(referrer.module_path(module), Some(&[][..]));
    assert_eq!(string(referrer.generics(ix)), "< T : Cv >");
    assert_eq!(referrer.iter().count(), 1);
    // The copy of the module keeps the bound, and replaces the entry.
    let mut file = File {
        shebang: None,
        attrs: Vec::new(),
        items: item.content.clone().unwrap().1,
    };
    referrer
        .clone()
        .visitor(context, |_, ix| parse_quote!(P<#ix>))
        .visit_file_mut(&mut file);
    assert_eq!(
        string(&file),
        string(&quote! {
            use super::base::B1 as Imp;
            pub trait Cv {}
            pub struct S<T: Cv>(pub P<0usize>);
        })
    );
}

// ── F11 ──────────────────────────────────────────────────────────────────────

/// F11 (limitation): a copy of an impl whose self type is an entry.
///
/// Trigger: `mod mm { use super::base::B1 as Imp; trait Tr {} impl<T> Tr for Imp<T> {} }`. The
/// entry `Imp<T>` is the self type, so the copy is `impl<T> Tr for P<0>`, where `P<0>` is what a
/// macro replaces the entry with: `<() as Repeater<(T,), ..>>::Type`. rustc does not count a
/// projection as constraining `T` (E0207 "the type parameter `T` is not constrained by the impl
/// trait, self type, or predicates"), and an inherent impl on it gets E0118 or E0116. The
/// generator of the property test avoids it: it copies only the inherent impls of the module's own
/// types, and checks the trait impls.
///
/// It is a limitation and not a bug: the replacement of an entry is never a type that rustc sees
/// through, and the library cannot know what a macro replaces it with. A macro should check such
/// an impl, not copy it. The test pins what the library does with it: the entry is found as the
/// self type, whole, and `T` is passed to the replacement as an argument of the entry.
#[test]
fn f11_copied_impl_param_constrained_only_by_leaked_type() {
    let item: ItemMod = parse_quote! {
        pub mod mm {
            use super::base::B1 as Imp;
            pub trait Tr {}
            impl<T> Tr for Imp<T> {}
        }
    };
    let mut leaker = new_leaker();
    let context = leaker.intern_mod(&item);
    let referrer = leaker.result().finish().unwrap();
    let ix = entry(&referrer, "Imp < T >");
    assert_eq!(string(referrer.generics(ix)), "< T >");
    let args = referrer
        .generic_args(ix, &Punctuated::new(), &[])
        .unwrap()
        .iter()
        .map(string)
        .collect::<Vec<_>>();
    assert_eq!(args, ["T"]);
    let mut file = File {
        shebang: None,
        attrs: Vec::new(),
        items: item.content.clone().unwrap().1,
    };
    referrer
        .clone()
        .visitor(context, |_, ix| parse_quote!(P<#ix>))
        .visit_file_mut(&mut file);
    assert_eq!(
        string(&file),
        string(&quote! {
            use super::base::B1 as Imp;
            pub trait Tr {}
            impl<T> Tr for P<0usize> {}
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
                string(output),
                lifetimes.count()
            ));
        }
    }
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
/// projection of the Repeater, with the arguments of the entry encoded by `RepeaterSlots`.
fn f12_copy(item: &ItemTrait) -> std::result::Result<(Vec<String>, ItemTrait), Vec<ErrorItem>> {
    f12_copy_with(new_leaker(), item)
}

/// Like `f12_copy()`, with the Leaker `leaker`.
fn f12_copy_with(
    mut leaker: Leaker,
    item: &ItemTrait,
) -> std::result::Result<(Vec<String>, ItemTrait), Vec<ErrorItem>> {
    let slots = RepeaterSlots::new(parse_quote!(crate::Repeater));
    let context = leaker.intern_trait(item, true);
    let referrer = leaker.result().finish()?;
    let root_args: Vec<GenericArgument> = item.generics.params.iter().map(param_arg).collect();
    let mut copy = item.clone();
    referrer
        .clone()
        .visitor(context, |_, ix| {
            let args = referrer
                .generic_args(ix, &item.generics.params, &root_args)
                .unwrap();
            let encoded = slots
                .encode_generics_args_to_ty_with_self(
                    &args,
                    &referrer.generics(ix).params,
                    &parse_quote!(()),
                    referrer.context_self_outlives(referrer.context(ix)),
                )
                .unwrap();
            parse_quote!(<() as crate::Repeater<#encoded, 0, #ix>>::Type)
        })
        .visit_item_trait_mut(&mut copy);
    Ok((referrer.iter().map(string).collect(), copy))
}

/// F12: an elided lifetime in the output of the sugar of `Fn` or of a function pointer, next to an
/// entry that has lifetimes.
///
/// Trigger: `Box<dyn Fn(&Rel<T>) -> &u8>` in `trait Tr<'a, T: 'a>`. The sugar has one input
/// lifetime, which elision gives to the output. The entry `Rel<T>` has the generics `<'a, T: 'a>`,
/// so its replacement names lifetimes in its arguments (`<() as Repeater<(&'a (), T, &'a T), ..>>
/// ::Type`), and the inputs have several: elision fails (E0106 "missing lifetime specifier").
/// `Tr1<T>`, without `'a`, works. Found by the property test with
/// `TYPE_LEAK_PROP_FEATURES=elided_fn_generics`, which is on by default now.
///
/// The fix leaks the whole holder, which keeps the elision as written (`type Type = dyn Fn(&Rel<T>)
/// -> &u8;`): the copy has no sugar left. A holder that cannot be leaked (a bound, a method) is an
/// error. The test asserts, without rustc, which entries there are, and that no sugar or function
/// pointer of the copy has an elided output lifetime that its inputs cannot resolve
/// (`Signatures`).
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
        assert_eq!(entries, [holder], "{}", string(&item));
        let mut signatures = Signatures::default();
        signatures.visit_item_trait(&copy);
        assert_eq!(
            signatures.count,
            0,
            "nothing of it is left: {}",
            string(&copy)
        );
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
        assert_eq!(entries, ["Rel < T >"], "{}", string(&item));
        let mut signatures = Signatures::default();
        signatures.visit_item_trait(&copy);
        assert_eq!(signatures.count, 1, "{}", string(&copy));
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
        assert_eq!(entries, ["Rel < T >"], "{}", string(&item));
        let mut signatures = Signatures::default();
        signatures.visit_item_trait(&copy);
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
            .unwrap_or_else(|| panic!("{} is accepted", string(&item)));
        let note = err[0].note().unwrap();
        assert!(note.contains("E0106"), "{}: {}", string(&item), note);
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
            self.1 |= string(i) == self.0;
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
    referrer
        .iter()
        .enumerate()
        .filter(|(ix, ty)| {
            let own_self = matches!(
                referrer.context_self(referrer.context(*ix)),
                Some(SelfRequirement::Type(self_ty)) if type_contains(self_ty, &string(*ty))
            );
            own_self && generics_name_self(referrer.generics(*ix))
        })
        .map(|(_, ty)| string(ty))
        .collect()
}

/// The Leaker of F13: `::p` is allowed.
fn f13_leaker() -> Leaker {
    let mut config = LeakerConfig::new();
    config.allow_primitive().allow_prelude();
    config.allow_path(parse_quote!(::p));
    Leaker::from_config(config)
}

/// F13: a bound of the generics of an impl names `Self`, and the self type of the impl is leaked.
///
/// Trigger: `impl<T> S<T> where T: AsRef<Option<Self>> { .. }` (also the bound written inline,
/// `impl<T: AsRef<[Self]>, const N: usize> S<T, N>`, and an impl of a trait). The self type `S<T>`
/// is an entry, whose context has the generics `<T: AsRef<Option<Self>>>` and `Self` = `S<T>`. The
/// Repeater's impl of the entry is `impl<T: AsRef<Option<Self>>> Repeater<..> for S<T>`, which
/// compiles; but the Referrer writes the projection `<S<T> as Repeater<..>>::Type` with the
/// Leaker's `S<T>` as the implementing type, which in the copy is the projection itself. The
/// entry is its own `Self`: there is no type to write for it (the test macro of the property test
/// says "no type for Self", and a real macro has the same problem: `Referrer::context_self()` is
/// `Type(S<T>)`, the type being replaced). Found by the property test with
/// `TYPE_LEAK_PROP_FEATURES=impl_self_bounds`.
///
/// It is the same for a type inside the self type, `impl<T: AsRef<Self>> ::p::W<Rel<T>>`: the copy
/// writes the self type with `Rel<T>` replaced, and `Rel<T>` depends on `Self`.
///
/// Fixed by rejecting what rustc cannot compile otherwise. The alternatives do not work: rewriting
/// the bound for the self type (`T: AsRef<Option<S<T>>>`, the impl being for `()`) makes the bound
/// name the leaked entry itself (E0277, "Bounds that name leaked types" in the `Leaker` docs); the
/// type for `Self` in the copy is a cycle (E0391); and the self type has no parts to leak instead
/// (`S<T>` is relative as a whole, `Rel<T>` uses the generic). The error is in the style of the
/// other `Self` errors: a headline, and a note on why with the way out (an absolute self type, or
/// bounds without `Self`).
///
/// What is not rejected: an optional entry (the absolute path of a crate that the Leaker does not
/// allow): it is copied as written, so a strict Referrer (`allow_ambiguous_crate_name(false)`) does
/// not replace it either. A bound of an outer scope that names its own `Self` is not the impl's
/// (`the_same_self_changes_nothing` in `outer_self.rs`). The controls are accepted: a self type
/// that is not leaked (an allowed path), and bounds that do not name `Self`.
///
/// The bound that names `Self` is also fine on a generic that no entry in the self type uses
/// (`impl<T: AsRef<Self>, U> ::p::W<T, Rel<U>>`, whatever the way the bound is written, in an impl
/// of a trait too, and for a method). `Self` in the bound is the self type that the copy writes
/// with `Rel<U>` replaced, and rustc normalizes it in the environment with the Repeater's impl of
/// `Rel<U>`, whose context has `U` only, no bound. The F19 rule looked at all the generics of the
/// self type, `T: AsRef<Self>` too, and rejected it: `Self` is unstable only if the bounds of the
/// generics that its entries use are (F19, narrowed). Checked with rustc (stable 1.98 and nightly
/// 1.100) on the Leaker's crate with the impl and the Repeater's impls that the macros of
/// `tests/crates/macro/prop.rs` emit, and on the Referrer's side with the checks that the property
/// test makes (the rewritten type of each position, with the generics and predicates around it, is
/// the original type): they compile for each control below, which the F19 rule rejected (the
/// copy of an impl as a whole is another matter, F11: `U` that only the entry uses is E0207, and
/// where all the generics are constrained, as in `impl<T: AsRef<Self>, U> ::p::Tr for ::p::W<T, U,
/// Rel<U>>`, it compiles as well). The ones that stay rejected fail in rustc, with the entries
/// chosen as the Leaker does (see F19, and `impl<T: Fn(Rel2<u8>), V: AsRef<Self>> ::p::W<Rel<T>,
/// T, V>`: E0277 "expected an `Fn(Rel2<u8>)` closure, found `T`", as `Self` is normalized with the
/// impl for `Rel<T>`, which needs the bound of `T`, and the environment has it with `Rel2<u8>`
/// replaced).
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
        let mut leaker = f13_leaker();
        leaker.intern_impl(item, true);
        let result = leaker.result();
        let err = result.errors();
        assert!(
            !err.is_empty(),
            "`{}` is accepted, and the entries {:?} are their own `Self`: the Referrer has no \
             type to write for it",
            string(item),
            self_referential_entries(&result.clone().into_referrer())
        );
        let headline = "is in the self type of the impl, and a bound of its generics names `Self`";
        let (shown, note) = (joined_messages(err), err[0].note().unwrap());
        assert!(shown.contains(headline), "{}: {}", string(item), shown);
        assert!(
            note.contains("has no type to write for `Self`"),
            "{}: {}",
            string(item),
            note
        );
        // The recovered Referrer has the rest of the item, and the entries that the error is about
        // are not among them.
        let referrer = result.into_referrer();
        assert!(
            self_referential_entries(&referrer).is_empty(),
            "{}: {:?}",
            string(item),
            self_referential_entries(&referrer)
        );
        // Also in a module, and as a nested item.
        let module: ItemMod = parse_quote!(mod m { #item });
        let mut leaker = f13_leaker();
        leaker.intern_mod(&module);
        assert!(
            joined_messages(&errors_of(leaker)).contains(headline),
            "{}",
            string(item)
        );
        let mut leaker = f13_leaker();
        leaker.enter_generics(&Generics::default(), |view| {
            view.visit_item(&Item::Impl(item.clone()))
        });
        assert!(
            joined_messages(&errors_of(leaker)).contains(headline),
            "{}",
            string(item)
        );
    }
    // A reference that holds a type that must be an entry is one itself (its parts cannot stand
    // for it), if it can name the generics: those of the Leaker, not the local ones of a nested
    // item, where only the parts are entries.
    let item: ItemImpl = parse_quote!(
        impl<'a, T: AsRef<Self>> ::p::Tr for ::p::W<&'a (Rel2, T)> {}
    );
    let mut leaker = f13_leaker();
    leaker.intern_impl(&item, true);
    assert!(joined_messages(&errors_of(leaker))
        .contains("`& 'a (Rel2 , T)` is in the self type of the impl"));
    for item in copied.iter().chain(&controls) {
        let mut leaker = f13_leaker();
        let context = leaker.intern_impl(item, true);
        let mut result = leaker.result();
        let referrer = result
            .clone()
            .finish()
            .unwrap_or_else(|e| panic!("`{}` is rejected: {}", string(item), joined_messages(&e)));
        assert!(
            self_referential_entries(&referrer).is_empty(),
            "{}",
            string(item)
        );
        // A strict Referrer replaces the optional entries too, but not those in the self type.
        let mut copy = Item::Impl(item.clone());
        result.allow_ambiguous_crate_name(false);
        result
            .finish()
            .unwrap()
            .visitor(context, |_, ix| parse_quote!(P<#ix>))
            .visit_item_mut(&mut copy);
        if let Item::Impl(copy) = copy {
            if copied.contains(item) {
                let self_ty = string(&copy.self_ty);
                assert!(
                    self_ty.contains(":: ext :: "),
                    "{}: {}",
                    string(item),
                    self_ty
                );
            }
        }
    }
    // The optional entry is one where the bound does not name `Self`: the test is not empty.
    let item: ItemImpl = parse_quote!(
        impl<T: Clone> ::ext::S<T> {}
    );
    let mut leaker = f13_leaker();
    let context = leaker.intern_impl(&item, true);
    let mut result = leaker.result();
    let referrer = result.clone().finish().unwrap();
    assert_eq!(
        referrer.iter().map(|t| string(&t)).collect::<Vec<_>>(),
        [":: ext :: S < T >"]
    );
    let mut copy = Item::Impl(item);
    result.allow_ambiguous_crate_name(false);
    result
        .finish()
        .unwrap()
        .visitor(context, |_, ix| parse_quote!(P<#ix>))
        .visit_item_mut(&mut copy);
    let copy = match copy {
        Item::Impl(copy) => copy,
        _ => unreachable!(),
    };
    assert_eq!(string(&copy.self_ty), "P < 0usize >");
}

// ── F14 ──────────────────────────────────────────────────────────────────────

/// The predicates of an impl and of its methods (the bounds of a parameter and the where
/// predicates) that name `Self::A`.
#[derive(Default)]
struct SelfProjectionsInPredicates(Vec<String>);

/// Whether a node names `Self::A`, an associated type of `Self`.
#[derive(Default)]
struct NamesSelfProjection(bool);

impl<'ast> Visit<'ast> for NamesSelfProjection {
    fn visit_path(&mut self, i: &'ast Path) {
        self.0 |= i.leading_colon.is_none()
            && i.segments.len() > 1
            && i.segments.first().map_or(false, |s| s.ident == "Self");
        syn::visit::visit_path(self, i)
    }
}

impl<'ast> Visit<'ast> for SelfProjectionsInPredicates {
    fn visit_type_param(&mut self, i: &'ast TypeParam) {
        let mut names = NamesSelfProjection::default();
        names.visit_type_param(i);
        if names.0 {
            self.0.push(string(i));
        }
        syn::visit::visit_type_param(self, i)
    }
    fn visit_predicate_type(&mut self, i: &'ast PredicateType) {
        let mut names = NamesSelfProjection::default();
        names.visit_predicate_type(i);
        if names.0 {
            self.0.push(string(i));
        }
        syn::visit::visit_predicate_type(self, i)
    }
}

/// The copy of `item` that a macro makes (the entries replaced by `P<ix>`), with a strict Referrer
/// or not, and whether the Leaker accepts `item`.
fn f14_copy(item: &ItemImpl, strict: bool) -> std::result::Result<ItemImpl, Vec<ErrorItem>> {
    let mut config = LeakerConfig::new();
    config.allow_primitive().allow_prelude();
    config.allow_path(parse_quote!(::p));
    let mut leaker = Leaker::from_config(config);
    let context = leaker.intern_impl(item, true);
    let mut result = leaker.result();
    if strict {
        result.allow_ambiguous_crate_name(false);
    }
    let referrer = result.finish()?;
    let mut copy = Item::Impl(item.clone());
    referrer
        .visitor(context, |_, ix| parse_quote!(P<#ix>))
        .visit_item_mut(&mut copy);
    match copy {
        Item::Impl(copy) => Ok(copy),
        _ => unreachable!(),
    }
}

/// How an item of F14 ends: rejected, for the type and the bound that the error names, or (`None`)
/// accepted.
type Rejected<'a> = Option<(&'a str, &'a str)>;

/// F14: a where predicate of a method names `Self::A`, in an impl of a trait whose own bound names
/// a leaked type.
///
/// Trigger: `impl<T> ::p::Tr<T> for u8 where T: Fn(Rel) { type A1 = u8; fn m<X>(&self, a: Rel) where
/// X: Into<Self::A1> {} }` (the bound can be inline, `impl<T: Fn(Rel)> ..`; and the predicate can
/// be on `T` or on a type that is not a generic, `u8: Into<Self::A1>`). `Rel` is leaked, so the
/// copy has `T: Fn(<() as Repeater<..>>::Type)`. rustc normalizes the predicate `X:
/// Into<Self::A1>` while it builds the environment of the method, which needs the impl of `Tr<T>`
/// for `u8`, and so its bound `T: Fn(Rel)`, which the environment has as `T:
/// Fn(<() as Repeater<..>>::Type)`, not normalized yet: E0277 "expected an `Fn(Rel)` closure,
/// found `T`" (E0271 for `T: for<'h> Fn(&'h Rel) -> &'h T`, "type mismatch resolving"). The same
/// shape as F2, but the check of F2 (a bound of a generic that the leaked type uses names a leaked
/// type) does not see it: the context of `Rel` has no generics, and the context of `Self::A1` of
/// an impl has none either (it is `Self` = `u8`, with no generic written in the path), although
/// the trait impl that rustc proves needs all the generics of the impl. With one of them
/// removed (`where X: Into<Rel>`, `a: Self::A1`, `T: Clone`) the copy compiles. Found by the
/// property test with `TYPE_LEAK_PROP_FEATURES=impl_pred_leaked`.
///
/// The trait case is rejected already
/// (`trait Tr<T: Fn(Rel)> { fn m<X>(&self) where X: Into<Self::A1>; }`, the `Self::A` case of
/// "Bounds that name leaked types" in the `Leaker` docs); the impl needs the same rule, with the
/// generics of the impl instead of those of the context. The assertion: if the Leaker accepts the
/// impl, its copy does not have a replacement in a bound or where predicate of the impl and
/// `Self::A` in a predicate. An optional entry counts if the Referrer is strict (it replaces it).
///
/// The controls are accepted, as they compile: a bound that names no leaked type, and a leaked
/// bound with `Self::A1` outside the predicates.
///
/// Fixed by the rule of F2 for `Self::A` (see "Bounds that name leaked types" in the `Leaker` docs
/// and `tests/bound_cycles.rs`): in an impl, the bounds of the impl count too, not only those of
/// the generics in the context of `Self::A`, as rustc selects the impl of the trait. The optional
/// entries that they name are copied as written for a strict Referrer.
#[test]
fn f14_impl_predicate_names_self_projection_with_leaked_bound() {
    let items: Vec<(ItemImpl, bool, Rejected)> = vec![
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
            false,
            Some(("Self :: A1", "T : Fn (Rel)")),
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
            false,
            Some(("Self :: A1", "T : Fn (Rel)")),
        ),
        // The leaked type of the bound is an optional entry, which a strict Referrer replaces, and
        // which the Leaker accepts all the same: the copy is checked.
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
            true,
            None,
        ),
    ];
    let controls: Vec<(ItemImpl, bool)> = vec![
        // The bound names no leaked type.
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
            false,
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
            false,
        ),
        // An optional entry is left as written by a Referrer that is not strict.
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
            false,
        ),
    ];
    for (item, strict, rejected) in &items {
        // Each is rejected, or its copy is checked (the oracle).
        let copy = match rejected {
            Some((ty, bound)) => {
                assert_unstable(
                    item,
                    f14_copy(item, *strict),
                    ty,
                    bound,
                    Blames::LeakedInImpl,
                );
                continue;
            }
            None => f14_copy(item, *strict).unwrap_or_else(|e| {
                panic!(
                    "`{}` is rejected, and it is not expected to be: {}",
                    string(item),
                    joined_messages(&e)
                )
            }),
        };
        let mut bounds = EntryMentions::default();
        bounds.visit_generics(&copy.generics);
        let mut projections = SelfProjectionsInPredicates::default();
        projections.visit_item_impl(&copy);
        assert!(
            (bounds.in_bounds.is_empty() && bounds.in_where.is_empty()) || projections.0.is_empty(),
            "in the copy `{}`, the bounds {:?} and where predicates {:?} of the impl name entries, \
             and the predicates {:?} name `Self::A`: rustc cannot prove the impl in the environment",
            string(&copy),
            bounds.in_bounds,
            bounds.in_where,
            projections.0
        );
    }
    for (item, strict) in &controls {
        let copy = f14_copy(item, *strict).unwrap_or_else(|e| {
            panic!(
                "`{}` is rejected, and it compiles: {}",
                string(item),
                joined_messages(&e)
            )
        });
        let mut projections = SelfProjectionsInPredicates::default();
        projections.visit_item_impl(&copy);
        let mut bounds = EntryMentions::default();
        bounds.visit_generics(&copy.generics);
        // What is accepted compiles: not both, as above.
        assert!(
            (bounds.in_bounds.is_empty() && bounds.in_where.is_empty()) || projections.0.is_empty(),
            "{}",
            string(&copy)
        );
    }
}

// ── F15 ──────────────────────────────────────────────────────────────────────

/// The copy of a struct, an enum or a union that a macro makes with `Referrer::visitor(..)
/// .with_self(self_ty)`, where `self_ty` is the type as the Leaker's crate names it
/// (`::p::S<'a, T>`), and the entries of the item. Each entry is replaced by `P<ix>`.
fn f15_copy(
    item: &Item,
    allow_self: bool,
) -> std::result::Result<(Vec<String>, Item), Vec<ErrorItem>> {
    let (ident, generics) = match item {
        Item::Struct(i) => (&i.ident, &i.generics),
        Item::Enum(i) => (&i.ident, &i.generics),
        Item::Union(i) => (&i.ident, &i.generics),
        _ => panic!("unsupported item"),
    };
    let mut leaker = new_leaker();
    let context = match item {
        Item::Struct(i) => leaker.intern_struct(i, allow_self),
        Item::Enum(i) => leaker.intern_enum(i, allow_self),
        Item::Union(i) => leaker.intern_union(i, allow_self),
        _ => unreachable!(),
    };
    let referrer = leaker.result().finish()?;
    let args: Vec<GenericArgument> = generics.params.iter().map(param_arg).collect();
    let self_ty: Type = parse_quote!(::p::#ident<#(#args),*>);
    let mut copy = item.clone();
    referrer
        .clone()
        .visitor(context, |_, ix| parse_quote!(P<#ix>))
        .with_self(self_ty)
        .visit_item_mut(&mut copy);
    Ok((referrer.iter().map(string).collect(), copy))
}

/// F15: `Referrer::visitor(..).with_self(ty)` replaces `Self` in a signature with an elided output
/// lifetime, and `ty` has lifetimes.
///
/// Trigger: `struct S<'a> { f: fn(&Self) -> &u8, g: &'a u8 }` copied with `with_self(::p::S<'a>)`,
/// as the macro of a Referrer does for a struct, an enum or a union (so that `Self` is the
/// Leaker's type). The elided lifetime of the output is that of the only lifetime of the input,
/// `&Self`: rustc counts none for `Self`. The copy has `fn(&::p::S<'a>) -> &u8`, whose input has
/// two (the elided one and `'a`): E0106 "missing lifetime specifier ... which one of argument 1's
/// 2 lifetimes it is borrowed from". The same for `Fn(&Self) -> &u8` of a trait object, and for
/// `Self::A`, which the visitor writes as `<::p::S<'a>>::A`. `S<T>` (no lifetime) is fine. The
/// replacement of an entry has the same problem and F12 fixed it, by leaking the whole holder;
/// this is `Self`, which the Leaker does not replace, and the Referrer does, without looking at
/// the signature.
///
/// The first fix was the same as F12's: `Self` (when it is a type with lifetimes that a copy writes,
/// the struct, enum or union itself) in the inputs of such a signature adds lifetimes like an entry
/// does, so the holder (the function pointer, the trait object, the reference that takes its
/// lifetime) is leaked whole, which keeps `Self` and the elision as written in the impl of the
/// Repeater (`impl<'a> Repeater<..> for S<'a> { type Type = fn(&Self) -> &u8; }`, which rustc
/// accepts); with `allow_self == false` it cannot be leaked, and where nothing holds the signature
/// (a bound, a method) it is a conflict, with the note about E0106. F23 refines it: where the
/// inputs show their one lifetime (`&Self`, `Ref<'_, Self>`, a named one) the Referrer names it
/// with a fresh `for<'e>` binder in the inputs and the output, so nothing is leaked and nothing is
/// rejected (`allow_self == false` and a bound too). What the Referrer cannot name is still leaked
/// whole or rejected, as above: a lifetime that a path hides (`fn(::core::slice::Iter<u8>, Self) ->
/// &u8`, which the Leaker takes as the elided one, F28) and a method, which has no binder to
/// write. The copy of an impl keeps `Self`, and so does a trait, whose Repeater impl writes
/// `Self::A` as `<__S>::A` (`Referrer::iter_with_self()`), and `with_self(Host)` the same, so they
/// are not affected (rustc accepts `fn(&<__S>::A) -> &u8`, and rejects `fn(&<__Self as
/// Tr<'a>>::A) -> &u8`).
///
/// The property test shows it with `TYPE_LEAK_PROP_FEATURES=self_elision` (on by default now).
/// There were two causes. The *oracle* of the test macro failed in the Leaker's crate: a witness
/// (`pub fn __tl_w_k<..>() -> PhantomData<Ty>`) and a check write the original type with `Self`
/// as the self type of an impl (`&'static unsafe fn(i32)`) or `Self::A0` as `<__Self as Tr<'a,
/// ..>>::A0` (with the lifetime parameters of the trait), and that breaks the elision of
/// `fn(&Self) -> &Y` (E0106). The oracle does not compare such a position now. The *library*
/// failed in the Referrer's crate, for the `Self` of a struct, an enum or a union with lifetime
/// parameters (the generator names `Self` in such a signature with the feature on).
/// The test asserts which entries there are, and that no sugar of `Fn(..)` and no function pointer
/// of the copy has an elided output lifetime that its inputs cannot resolve (`Signatures`, as in
/// F12).
#[test]
fn f15_self_rewritten_in_elided_signature() {
    // The inputs show their one lifetime: the Referrer names it (F23), so `Self` leaks nothing,
    // and `allow_self == false` is no obstacle. The copy has the lifetime in a `for<..>` binder.
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
        for allow_self in [true, false] {
            let (entries, copy) = f15_copy(&item, allow_self).unwrap_or_else(|e| {
                panic!(
                    "`{}` is rejected (allow_self = {}): {}",
                    string(&item),
                    allow_self,
                    joined_messages(&e)
                )
            });
            assert!(entries.is_empty(), "{}: {:?}", string(&item), entries);
            let mut signatures = Signatures::default();
            signatures.visit_item(&copy);
            assert_eq!(signatures.count, 1, "{}", string(&copy));
            assert!(
                signatures.broken.is_empty(),
                "{}: {:?}",
                string(&copy),
                signatures.broken
            );
            assert!(string(&copy).contains("for <"), "{}", string(&copy));
        }
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
        let (entries, copy) = f15_copy(&item, true).unwrap();
        assert_eq!(entries, [holder], "{}", string(&item));
        let mut signatures = Signatures::default();
        signatures.visit_item(&copy);
        assert_eq!(
            signatures.count,
            0,
            "nothing of it is left: {}",
            string(&copy)
        );
        // With `allow_self == false` the holder cannot be leaked (it names `Self`).
        let err = f15_copy(&item, false)
            .err()
            .unwrap_or_else(|| panic!("{} is accepted", string(&item)));
        let note = err[0].note().unwrap();
        assert!(note.contains("E0106"), "{}: {}", string(&item), note);
    }
    // An entry that adds lifetimes to the inputs makes the signature the holder, as in F12: here
    // `Self` is its qualified self type. It is an optional entry (an external path), so a Referrer
    // that leaves those as written (the default) gets the lifetime named, as above.
    {
        let item: Item = parse_quote!(
            pub struct S<'a>(Box<dyn Fn(&<Self as ::p::Tr>::A) -> &u8>, &'a u8);
        );
        let (entries, copy) = f15_copy(&item, true).unwrap();
        assert_eq!(
            entries,
            ["dyn Fn (& < Self as :: p :: Tr > :: A) -> & u8"],
            "{}",
            string(&item)
        );
        let mut signatures = Signatures::default();
        signatures.visit_item(&copy);
        assert!(
            signatures.broken.is_empty(),
            "{}: {:?}",
            string(&copy),
            signatures.broken
        );
        // With `allow_self == false` the entry that names `Self` is not leaked, so it is left as
        // written, and the Referrer names the lifetime as above.
        let (entries, copy) = f15_copy(&item, false).unwrap();
        assert!(entries.is_empty(), "{}: {:?}", string(&item), entries);
        let mut signatures = Signatures::default();
        signatures.visit_item(&copy);
        assert!(
            signatures.broken.is_empty(),
            "{}: {:?}",
            string(&copy),
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
        let (entries, copy) = f15_copy(&item, true).unwrap_or_else(|e| {
            panic!(
                "`{}` is rejected, and it compiles: {}",
                string(&item),
                joined_messages(&e)
            )
        });
        let mut signatures = Signatures::default();
        signatures.visit_item(&copy);
        assert!(
            signatures.broken.is_empty(),
            "{}: {:?}",
            string(&copy),
            signatures.broken
        );
        assert!(entries.is_empty(), "{}: {:?}", string(&item), entries);
        // Nothing is named that the original does not name: no binder is added.
        assert_eq!(
            string(&copy).matches("for <").count(),
            string(&item).matches("for <").count(),
            "{}",
            string(&copy)
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
        let err = f15_copy(&item, true)
            .err()
            .unwrap_or_else(|| panic!("{} is accepted", string(&item)));
        let note = err[0].note().unwrap();
        assert!(note.contains("E0106"), "{}: {}", string(&item), note);
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
        let (context, referrer) = analyze_in(new_leaker(), &item).unwrap();
        let entries: Vec<String> = referrer.iter().map(string).collect();
        assert!(entries.is_empty(), "{}: {:?}", string(&item), entries);
        // The same visitor as for the others: `Self` is not replaced in the copy of an impl.
        let mut copy = item.clone();
        let host: Type = parse_quote!(Host);
        let mut visitor = referrer.visitor(context, |_, ix| parse_quote!(P<#ix>));
        match &item {
            Item::Trait(_) => visitor.with_self(host).visit_item_mut(&mut copy),
            _ => visitor.visit_item_mut(&mut copy),
        }
        let mut signatures = Signatures::default();
        signatures.visit_item(&copy);
        assert!(
            signatures.broken.is_empty(),
            "{}: {:?}",
            string(&copy),
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
            WherePredicate::Type(p) => string(&p.bounded_ty) == name && has(&p.bounds),
            WherePredicate::Lifetime(p) => p.lifetime.to_string() == name && has_lt(&p.bounds),
            _ => false,
        })
}

/// F16 (fixed): the outlives bound that the signature of a method implies for `&'x Self::A` is in
/// the context of the entry, as the where predicate `Self::A: 'x`.
///
/// Trigger: `trait Tr<'a, T> { type A0; fn m<'x>(&'x self, a: Rel<&'x mut Self::A0>) where Self:
/// 'x; }`. The original relies on the well-formedness of its signature: `&'x mut Self::A0` in the
/// parameter implies `<Self as Tr<'a, T>>::A0: 'x`. The Repeater's impl of the entry
/// `Rel<&'x mut Self::A0>` is `impl<'a, 'x, T, __S: Tr<'a, T>> Repeater<..> for __S { type Type =
/// Rel<&'x mut <__S as Tr<'a, T>>::A0>; }`, with no signature around it, so it needs the bound
/// itself. The Leaker recorded `Self: 'x` for the reference (the context has
/// `Referrer::context_self_outlives()` = `['x]`) and no more: E0309 "the associated type
/// `<__S as Tr<'a, T>>::A0` may not live long enough" in the Leaker's crate. With `where
/// Self::A0: 'x` written it is recorded, and the impl compiles. Found by the property test with
/// `TYPE_LEAK_PROP_FEATURES=implied_outlives` (seeds 1, 4, 5 and 7 of 60 cases: seed 5 has this
/// projection, the others `Self` under a path, below).
///
/// The fix records `Self::A0: 'x` as a where predicate of the context, as if it was written. The
/// Referrer computes the same context for the copy, which has the signature and not the
/// predicate, so it finds the entry. The Repeater's impl declares the predicate (`where <__S as
/// Tr<'a, T>>::A0: 'x`), and in the copy `<Self as Repeater<..>>::Type` normalizes to the original
/// type, whose well-formedness implies it.
///
/// The alternative of recording the components is not equivalent (`Self`, `T: 'x`, `'a: 'x`:
/// rustc proves a projection when each of them outlives `'x`, and the encoded arguments would
/// carry them as `&'x T` and `&'x &'a ()`). The Repeater's impl compiles, but the copy is
/// stricter than the original, which knows `Self::A0: 'x` and not `'a: 'x`: an impl of the trait
/// for a host type with the rewritten signature is E0491 "in type `&'x &'a ()`, reference has a
/// longer lifetime than the data it references" (rustc normalizes the signature to compute its
/// implied bounds, and the arguments that the projection drops are not among them). The property
/// test shows it as E0477 in the Referrer's crate (seed 5). So no `T: 'x` or `'a: 'x` is recorded
/// for the projection (written ones are).
///
/// The same family, which is documented and not a bug (F5, the `Leaker` docs, "Outlives bounds"): a
/// path with a lifetime argument records nothing, so `Rel<Wrap<'x, [Self]>>` with
/// `fn m<'x>(&'x self)`, where `Wrap<'a, T: 'a>` needs `Self: 'x` and the receiver implies it, has
/// no `Self: 'x` in the context (`BL<'x, [Self]>` in the property test). The `Leaker` docs ask to
/// write the bound, and the generator writes it. And a where predicate `X: Into<&'x Self>` of a
/// method relies on the receiver as well, which the witness and the check of the property test
/// (functions with no receiver) do not have: E0309 in the Leaker's crate, from the test macro's own
/// oracle, so the generator writes `Self: 'x` there too.
///
/// F16 kept `Self: 'x` for the reference, besides the predicate. F34 drops it: it is more than the
/// original has (the receiver implies it at best), and a copy for a generic host `Gen<H>` is E0309
/// on `H` for `&'x Gen<H>` where the trait has the lifetime (`trait Tr<'x> { type A0; fn m(&self, a:
/// Rel<&'x Self::A0>); }`). So it is recorded where a where clause writes it (`where Self: 'x`).
///
/// The assertions, for each entry that has `&'x Self::A0`: the context has the predicate
/// `Self::A0: 'x`, and `Self: 'x` where the method writes it, and the arguments of the Repeater
/// are those of the parameters (and `&'x Self` for it), with no outlives of `T` or `'a`; the
/// Referrer replaces the entry in a copy of the trait. The control has the bounds written (`where
/// Self: 'x, T: 'x, 'a: 'x`), which are recorded as before.
#[test]
fn f16_implied_outlives_of_self_projection_lost() {
    let slots = RepeaterSlots::new(parse_quote!(crate::Repeater));
    // (the trait, whether `Self: 'x` is recorded, the encoded arguments). `Self: 'x` is recorded
    // where a where clause writes it, and not for the projection alone (F34: it is more than the
    // original has, which the receiver implies at best).
    let items: Vec<(ItemTrait, bool, &str)> = vec![
        (
            parse_quote!(
                pub trait Tr<'a, T> {
                    type A0;
                    fn m<'x>(&'x self, a: Rel<&'x mut Self::A0>)
                    where
                        Self: 'x;
                }
            ),
            true,
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
            false,
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
            false,
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
            true,
            "(& 'a () , & 'x () , T , & 'x Self)",
        ),
    ];
    for (item, self_recorded, encoded) in &items {
        let mut leaker = new_leaker();
        let context = leaker.intern_trait(item, true);
        let referrer = leaker
            .result()
            .finish()
            .unwrap_or_else(|e| panic!("`{}` is rejected: {}", string(item), joined_messages(&e)));
        let mut checked = 0;
        for ix in 0..referrer.iter().count() {
            // Every entry that has the reference `&'x Self::A0`.
            let ty = string(referrer.iter().nth(ix).unwrap());
            if !(ty.contains("& 'x") && ty.contains("Self :: A0")) {
                continue;
            }
            checked += 1;
            let generics = referrer.generics(ix);
            let predicate_recorded = generics
                .where_clause
                .iter()
                .flat_map(|w| &w.predicates)
                .any(|p| string(p) == "Self :: A0 : 'x");
            assert!(
                predicate_recorded,
                "`{}`: the context of the entry `{}` (generics `{}`) lacks `Self::A0: 'x`",
                string(item),
                ty,
                string(generics)
            );
            let self_outlives = referrer.context_self_outlives(referrer.context(ix));
            assert_eq!(
                self_outlives
                    .iter()
                    .map(|l| l.to_string())
                    .collect::<Vec<_>>(),
                if *self_recorded {
                    vec!["'x".to_string()]
                } else {
                    Vec::new()
                },
                "`{}`",
                string(item)
            );
            // The components are not: the copy cannot prove them.
            for name in ["T", "'a"] {
                assert!(
                    !generics_say_outlives(generics, name, "'x"),
                    "`{}`: `{}: 'x` is recorded for the entry `{}`",
                    string(item),
                    name,
                    ty
                );
            }
            let arguments = slots
                .encode_generics_params_to_ty_with_self(
                    &generics.params,
                    &parse_quote!(Self),
                    self_outlives,
                )
                .unwrap();
            assert_eq!(string(&arguments), *encoded, "`{}`", string(item));
        }
        assert!(checked > 0, "`{}`: no entry", string(item));
        // The Referrer finds the entry in a copy, which has the signature and not the predicate.
        let copy = rewrite(&referrer, context, Item::Trait(item.clone()));
        assert!(
            !copy.contains("Rel < & 'x") && !copy.contains("Rel < fn"),
            "`{}`: not replaced in the copy: {}",
            string(item),
            copy
        );
    }

    // Written: recorded, and encoded.
    let control: ItemTrait = parse_quote!(
        pub trait Tr<'a, T> {
            type A0;
            fn m<'x>(&'x self, a: Rel<&'x mut Self::A0>)
            where
                Self: 'x,
                T: 'x,
                'a: 'x;
        }
    );
    let mut leaker = new_leaker();
    leaker.intern_trait(&control, true);
    let referrer = leaker.result().finish().unwrap();
    let ix = entry(&referrer, "Rel < & 'x mut Self :: A0 >");
    let generics = referrer.generics(ix);
    assert!(generics_say_outlives(generics, "T", "'x"));
    assert!(generics_say_outlives(generics, "'a", "'x"));
    assert_eq!(
        referrer
            .context_self_outlives(referrer.context(ix))
            .iter()
            .map(|l| l.to_string())
            .collect::<Vec<_>>(),
        vec!["'x".to_string()]
    );
}

// ── F17 ──────────────────────────────────────────────────────────────────────

/// Whether `generics` name the type `ty` in a bound or a where predicate.
fn generics_name_type(generics: &Generics, ty: &Type) -> bool {
    struct Find<'a>(&'a Type, bool);
    impl<'ast, 'a> Visit<'ast> for Find<'a> {
        fn visit_type(&mut self, i: &'ast Type) {
            self.1 |= string(i) == string(self.0);
            syn::visit::visit_type(self, i)
        }
    }
    let mut find = Find(ty, false);
    find.visit_generics(generics);
    find.1
}

/// The entries (required or optional: a strict Referrer replaces both) that the bounds or where
/// predicates of their own context name: they need themselves.
fn entries_in_own_context(referrer: &Referrer) -> Vec<String> {
    referrer
        .iter()
        .enumerate()
        .filter(|(ix, ty)| generics_name_type(referrer.generics(*ix), ty))
        .map(|(_, ty)| string(ty))
        .collect()
}

/// F17: an optional entry in a where predicate that names a lifetime, which the predicate is part
/// of the context of.
///
/// Trigger: `struct S<'a, T: 'a> where T: Fn(&'a ::ext::W<T, ::ext::B0>) -> u8 { .. }`, where
/// `::ext` is a crate that the Leaker does not allow, and a strict Referrer
/// (`allow_ambiguous_crate_name(false)`, the Referrer's choice, for a Referrer whose dependencies
/// may differ from the Leaker's) replaces the optional entries. `&'a ::ext::W<T, ::ext::B0>` is an
/// optional entry (it holds `::ext::B0`, another optional type, so it is not left alone as
/// `T: Fn(&'a ::ext::W<T>)` is), and its context has `'a`, `T` and the where predicate that names
/// `'a`, which is the predicate that names the entry. The copy has
/// `T: Fn(<() as Repeater<(&'a (), T, &'a T), ..>>::Type)`, and the Repeater's impl for it has
/// `T: Fn(&'a ::ext::W<T, ::ext::B0>)`, which the environment of the copy does not have in that
/// form: E0277 "expected an `Fn(&W<T, B0>)` closure, found `T`" (standalone with rustc, too). That
/// is F2, whose rule (an optional entry in such a place is copied as written: the `Leaker` docs,
/// "Bounds that name leaked types") does not see this one: the same item with `::ext::W<T>` (no
/// optional type inside), or with a required entry (`Rel`), is not an entry or is rejected. Found
/// by the property test with `TYPE_LEAK_PROP_FEATURES=strict_cycles` (seeds 1 to 8 of 60 cases),
/// which only affects the items that the generator invokes as strict, 25% of them.
///
/// The Leaker should leave that type as written (not an entry), or reject the item. The assertion is
/// that no entry of an accepted item is named by the bounds or where predicates of its own
/// context.
///
/// Fixed: an optional entry that is leaked as a whole (a reference with a lifetime), and whose own
/// context is unstable (its bound names it), is not one, and neither are the optional entries
/// inside it: its parts cannot stand for it. See "Bounds that name leaked types" in the `Leaker`
/// docs.
#[test]
fn f17_optional_entry_in_own_context() {
    let items: Vec<ItemStruct> = vec![
        parse_quote!(
            pub struct S<'a, T: 'a>
            where
                T: Fn(&'a ::ext::W<T, ::ext::B0>) -> u8,
            {
                f0: u8,
            }
        ),
        // The bound written inline, a trait object that holds the other optional type.
        parse_quote!(
            pub struct S<'a, T: 'a + Fn(&'a ::ext::W<dyn Fn(T) -> ::ext::B0 + 'a>) -> u8> {
                f0: &'a T,
            }
        ),
    ];
    let controls: Vec<ItemStruct> = vec![
        // No optional type inside: left as written.
        parse_quote!(
            pub struct S<'a, T: 'a>
            where
                T: Fn(&'a ::ext::W<T>) -> u8,
            {
                f0: u8,
            }
        ),
        // No lifetime around: the predicate is not part of the context of `::ext::B0`.
        parse_quote!(
            pub struct S<'a, T: 'a>
            where
                T: Fn(::ext::W<T, ::ext::B0>) -> u8,
            {
                f0: &'a T,
            }
        ),
    ];
    for item in &items {
        let mut leaker = new_leaker();
        leaker.intern_struct(item, true);
        // Rejected is fine.
        let Ok(referrer) = leaker.result().finish() else {
            continue;
        };
        let entries = entries_in_own_context(&referrer);
        assert!(
            entries.is_empty(),
            "`{}` is accepted, and its entries {:?} are named by the bounds of their own context: a strict Referrer replaces them by a path that needs the bound",
            string(item),
            entries
        );
    }
    for item in &controls {
        let mut leaker = new_leaker();
        leaker.intern_struct(item, true);
        let referrer = leaker
            .result()
            .finish()
            .unwrap_or_else(|e| panic!("`{}` is rejected: {}", string(item), joined_messages(&e)));
        assert!(
            entries_in_own_context(&referrer).is_empty(),
            "{}",
            string(item)
        );
    }
}

// ── F18 ──────────────────────────────────────────────────────────────────────

/// F18 (fixed): a strict Referrer replaced the optional entry that holds a lifetime of a method, so
/// the copy did not have it where the original did.
///
/// `LeakerConfig::allow_path()` docs: an absolute path of a crate that the Leaker does not allow is an
/// optional entry, which a Referrer leaves as written by default and replaces if it is strict
/// (`allow_ambiguous_crate_name(false)`). Trigger:
/// `trait Tr { fn m<'x>(&self, a: ::ext::W<&'x u8>) -> Rel<'x>; }`. In the trait, `'x` is
/// late-bound (the argument holds it, and no bound names it). The Leaker counted the optional type
/// as an anchor, as the default Referrer keeps it. The strict copy was
/// `fn m<'x>(&self, a: P<1>) -> P<0>`: both are projections, which do not constrain `'x`, so it is
/// early-bound, and rustc rejected the impl of the Referrer's crate, where the trait has it
/// late-bound (E0195 "lifetime parameters differ in whether they are early- or late-bound"). Found
/// by the property test with `TYPE_LEAK_PROP_FEATURES=strict_anchors`, for the items that the
/// generator invokes as strict (25% of them): the Leaker's analysis cannot know what the Referrer
/// chooses.
///
/// The fix: the scan of the occurrences that keep a lifetime of a method (the check of F7) looks at
/// the optional entries as the check of a binder does (`careful`). If the only occurrences in the
/// inputs that keep the lifetime, outside the types leaked whole, are inside optional entries, those
/// entries are no optional entries (`not_optional`, as `env_blocked()` does), so every Referrer
/// copies them as written and the lifetime stays constrained. No error: the crate of the type has
/// to resolve in the Referrer's crate, as for any absolute path that is copied as written (which
/// is also what the help of `UnconstrainedLifetime`, "write it with an absolute path", asks for: it
/// now works for a strict Referrer too). Another input that keeps the lifetime (`&'x u8`, `&'x
/// self`) leaves the entry as it was. The cross-crate fixture is 9194 (`tests/cross_crate.rs`), in a
/// Referrer's crate that is invoked as `strict`.
#[test]
fn f18_strict_referrer_keeps_the_optional_type_that_holds_a_method_lifetime() {
    let item: ItemTrait = parse_quote!(
        pub trait Tr {
            fn m<'x>(&self, a: ::ext::W<&'x u8>) -> Rel<'x>;
        }
    );
    // The names of the lifetimes that the arguments of the method of the copy hold.
    let held = |copy: &ItemTrait| -> Vec<String> {
        let Some(TraitItem::Fn(method)) = copy.items.first() else {
            panic!("no method");
        };
        let mut lifetimes = Lifetimes::default();
        for input in &method.sig.inputs {
            if let FnArg::Typed(input) = input {
                lifetimes.visit_type(&input.ty);
            }
        }
        lifetimes.named
    };
    let copy_of = |config: LeakerConfig, strict: bool| {
        let mut leaker = Leaker::from_config(config);
        let context = leaker.intern_trait(&item, true);
        let mut result = leaker.result();
        if strict {
            result.allow_ambiguous_crate_name(false);
        }
        let referrer = result.finish().unwrap();
        let mut copy = item.clone();
        referrer
            .visitor(context, |_, ix| parse_quote!(P<#ix>))
            .visit_item_trait_mut(&mut copy);
        copy
    };
    let mut config = LeakerConfig::new();
    config.allow_primitive().allow_prelude();

    // Every Referrer keeps the optional type, which holds `'x`; `Rel<'x>` is replaced.
    let expected: ItemTrait = parse_quote!(
        pub trait Tr {
            fn m<'x>(&self, a: ::ext::W<&'x u8>) -> P<0usize>;
        }
    );
    for strict in [false, true] {
        let copy = copy_of(config.clone(), strict);
        assert_eq!(held(&copy), ["'x"], "strict {}", strict);
        assert_eq!(string(&copy), string(&expected), "strict {}", strict);
    }
    // With the crate allowed it is not an entry, for either.
    config.allow_path(parse_quote!(::ext));
    assert_eq!(held(&copy_of(config.clone(), false)), ["'x"]);
    assert_eq!(held(&copy_of(config, true)), ["'x"]);
}

/// F18: the shapes. Each is an item, the optional entries it keeps (`?` first), and the copy of a
/// strict Referrer (`P<ix>` for an entry).
#[test]
fn f18_the_optional_entries_that_hold_a_method_lifetime_are_copied_as_written() {
    let cases: Vec<(Item, &[&str], &str)> = vec![
        // The type is the only input that holds `'x`: also behind a box, in a tuple, and in the
        // output too (the same type, in the same context).
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
            &["?::ext::Y<u8>"],
            "trait Tr { fn m < 'x > (& self , a : :: ext :: W < 'x > , b : :: ext2 :: V < 'x > , c : P < 0usize >) -> & 'x u8 ; }",
        ),
        // A type leaked whole holds it too, and does not keep it in the copy.
        (
            parse_quote!(
                trait Tr {
                    fn m<'x>(&self, a: ::ext::W<'x>, b: Rel<'x>) -> &'x u8;
                }
            ),
            &[],
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
            &["?::p::X"],
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
            &["?::ext::W<'x>"],
            "trait Tr { fn m < 'x > (& self , a : P < 0usize > , b : & 'x u8) -> & 'x u8 ; }",
        ),
        (
            parse_quote!(
                trait Tr {
                    fn m<'x>(&'x self, a: ::ext::W<'x>) -> &'x u8;
                }
            ),
            &["?::ext::W<'x>"],
            "trait Tr { fn m < 'x > (& 'x self , a : P < 0usize >) -> & 'x u8 ; }",
        ),
        // The output does not name it, or it is a lifetime of the trait: nothing to keep.
        (
            parse_quote!(
                trait Tr {
                    fn m<'x>(&self, a: ::ext::W<'x>) -> u8;
                }
            ),
            &["?::ext::W<'x>"],
            "trait Tr { fn m < 'x > (& self , a : P < 0usize >) -> u8 ; }",
        ),
        (
            parse_quote!(
                trait Tr<'x> {
                    fn m(&self, a: ::ext::W<'x>) -> &'x u8;
                }
            ),
            &["?::ext::W<'x>"],
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
            &["?::p::X", "?::ext::W<'x>"],
            "impl P < 0usize > { fn m < 'x > (& self , a : P < 1usize >) -> & 'x u8 { loop { } } }",
        ),
    ];
    for (item, optional, expected) in &cases {
        let what = string(item);
        let mut leaker = new_leaker();
        let context = intern(&mut leaker, item);
        let mut result = leaker.result();
        result.allow_ambiguous_crate_name(false);
        let referrer = result
            .finish()
            .unwrap_or_else(|e| panic!("`{}` is rejected: {}", what, joined_messages(&e)));
        let entries: Vec<String> = (0..referrer.iter().count())
            .filter(|&ix| !referrer.must(ix))
            .map(|ix| {
                let ty = string(referrer.iter().nth(ix).unwrap()).replace(' ', "");
                format!("?{}", ty)
            })
            .collect();
        let optional: Vec<String> = optional.iter().map(|ty| ty.to_string()).collect();
        assert_eq!(entries, optional, "`{}`", what);
        assert_eq!(
            rewrite(&referrer, context, item.clone()),
            *expected,
            "`{}`",
            what
        );
    }
}

// ── F19 and F20 ──────────────────────────────────────────────────────────────

/// The indices `ix` of the replacements `P<ix>` in the bounds of a parameter and in the where
/// predicates of an item and of its methods.
#[derive(Default)]
struct ReplacementsInPredicates(Vec<usize>);

/// The indices of the replacements in a node.
struct Indices<'a>(&'a mut Vec<usize>);

impl<'ast> Visit<'ast> for Indices<'_> {
    fn visit_path_segment(&mut self, seg: &'ast PathSegment) {
        if seg.ident == "P" {
            if let PathArguments::AngleBracketed(args) = &seg.arguments {
                if let Some(arg) = args.args.first() {
                    if let Ok(ix) = string(arg).trim_end_matches("usize").trim().parse() {
                        self.0.push(ix);
                    }
                }
            }
        }
        syn::visit::visit_path_segment(self, seg)
    }
}

impl<'ast> Visit<'ast> for ReplacementsInPredicates {
    fn visit_type_param(&mut self, i: &'ast TypeParam) {
        for bound in &i.bounds {
            if let Ok(ty) = parse2::<Type>(quote!(dyn #bound)) {
                Indices(&mut self.0).visit_type(&ty);
            }
        }
        syn::visit::visit_type_param(self, i)
    }
    fn visit_predicate_type(&mut self, i: &'ast PredicateType) {
        Indices(&mut self.0).visit_predicate_type(i);
        syn::visit::visit_predicate_type(self, i)
    }
}

/// The copy of `item` that a macro makes: the entries replaced by `P<ix>`.
fn f19_copy(item: &Item) -> std::result::Result<(Referrer, Item), Vec<ErrorItem>> {
    f19_copy_with(new_leaker(), item)
}

/// Like `f19_copy()`, with the Leaker that `f13_leaker()` makes (it allows `::p`, so that the self
/// type of an impl has parts that are entries and a part that is not).
fn f19_copy_with(
    mut leaker: Leaker,
    item: &Item,
) -> std::result::Result<(Referrer, Item), Vec<ErrorItem>> {
    let context = match item {
        Item::Impl(i) => leaker.intern_impl(i, true),
        Item::Trait(i) => leaker.intern_trait(i, true),
        _ => panic!("unsupported item"),
    };
    let referrer = leaker.result().finish()?;
    let mut copy = item.clone();
    referrer
        .clone()
        .visitor(context, |_, ix| parse_quote!(P<#ix>))
        .visit_item_mut(&mut copy);
    Ok((referrer, copy))
}

/// The entries that a predicate of the copy names, and that have a bound of their context that names
/// `Self`, where `self_is_leaked` says that what `Self` is in the context is replaced in the copy:
/// the Repeater's impl of the entry has the bound with the Leaker's `Self`, and the environment of
/// the copy has it with the copy's, not normalized yet.
fn entries_with_self_bound_in_predicates(
    referrer: &Referrer,
    copy: &Item,
    self_is_leaked: impl Fn(&Referrer, ContextId) -> bool,
) -> Vec<String> {
    let mut used = ReplacementsInPredicates::default();
    used.visit_item(copy);
    let mut found: Vec<String> = used
        .0
        .iter()
        .filter(|&&ix| {
            ix < referrer.iter().count()
                && generics_name_self(referrer.generics(ix))
                && self_is_leaked(referrer, referrer.context(ix))
        })
        .map(|&ix| string(referrer.iter().nth(ix).unwrap()))
        .collect();
    found.dedup();
    found
}

/// F19: in an impl whose self type is leaked, a method has a bound that names `Self` on a generic,
/// and a predicate that names a leaked type that uses the generic.
///
/// Trigger: `impl S { fn m<X>(&self) where X: Fn(Self) -> u8, Rel<X>: Default {} }`, with `S` and
/// `Rel` leaked (also `for<'h> Fn(&'h Self) -> &'h Result<Self, f64>`, which rustc reports as E0271
/// "type mismatch resolving"). The entry `Rel<X>` has the context `<X: Fn(Self) -> u8>` with `Self`
/// = `S`, so the Repeater's impl for `S` has `X: Fn(S) -> u8`. The copy has `X: Fn(Self) -> u8`
/// with `Self` = `<() as Repeater<..>>::Type`, the replaced self type, which rustc has not
/// normalized yet in the environment where it proves the predicate
/// `<S as Repeater<(X,), ..>>::Type: Default`: E0277 "expected an `Fn(S)` closure, found `X`" (the
/// same with rustc alone). That is F2 with `Self` as the leaked type that the bound names, and the
/// check of the `Leaker` docs ("The type is `Self`, in an impl whose self type is leaked, and a
/// bound of the generics of a type in the self type ... is unstable") only looks at the generics of
/// the impl, not at those of a method. With the entry in a position (`a0: Rel<X>`) it compiles.
/// Found by the property test with `TYPE_LEAK_PROP_FEATURES=impl_self_bound_cycle`.
///
/// The assertion: if the Leaker accepts the impl, no entry that a predicate of the copy names has
/// a bound of its context that names `Self` while the self type is an entry.
///
/// Fixed by generalizing the rule of F2: a bound is unstable if it names a leaked type, `Self` (if
/// the self type of an impl is leaked, and for a strict Referrer, if it holds an optional entry,
/// which is copied as written then), `Self::A` or an associated type that rustc normalizes (F20 and
/// F21). An entry in a predicate whose context has such a bound is an error. See "Bounds that name
/// leaked types" in the `Leaker` docs and `tests/bound_cycles.rs`.
///
/// `Self` itself in a predicate (`T: AsRef<Self>`) is a type that the copy writes with its entries
/// replaced, and rustc normalizes it with the Repeater's impls of those entries: it is an error
/// only for the bounds of the generics that they use, not for those of the other generics of the
/// self type (narrowed after F13: `impl<T: AsRef<Self>, U> ::p::W<T, Rel<U>>` compiles). The
/// self type with an entry as a part is also `Self` replaced, so the entry in a predicate whose
/// context has `T: AsRef<Self>` is the same cycle there (the cases of `by_parts` below, which the
/// first oracle did not look at: it asked for the self type as a whole to be an entry).
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
    let controls: Vec<Item> = vec![
        // The entry is in a position, which rustc normalizes once the environment is built.
        parse_quote!(
            impl S {
                fn m<X>(&self, a0: Rel<X>)
                where
                    X: Fn(Self) -> u8,
                {
                }
            }
        ),
        // The bound does not name `Self`.
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
    ];
    // `Self` is replaced in the copy if the self type is an entry, or has one in it.
    let self_is_leaked = |referrer: &Referrer, context: ContextId| {
        matches!(
            referrer.context_self(context),
            Some(SelfRequirement::Type(self_ty))
                if referrer.iter().any(|entry| type_contains(self_ty, &string(entry)))
        )
    };
    // Each is rejected: no copy of it exists whose predicate names such an entry.
    let blamed = [
        ("Rel < X >", "X : Fn (Self) -> u8", Blames::SelfType),
        (
            "Rel < Rel2 < X > >",
            "X : for < 'h > Fn (& 'h Self) -> & 'h Result < Self , f64 >",
            Blames::SelfType,
        ),
        ("Rel < X >", "X : Fn (& Self) -> u8", Blames::SelfType),
    ];
    assert_eq!(items.len(), blamed.len());
    for (item, (ty, bound, blames)) in items.iter().zip(blamed) {
        assert_unstable(item, f19_copy(item), ty, bound, blames);
    }
    for item in &controls {
        let (referrer, copy) = f19_copy(item).unwrap_or_else(|e| {
            panic!(
                "`{}` is rejected, and it compiles: {}",
                string(item),
                joined_messages(&e)
            )
        });
        assert!(
            entries_with_self_bound_in_predicates(&referrer, &copy, self_is_leaked).is_empty(),
            "{}",
            string(&copy)
        );
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
            f19_copy_with(f13_leaker(), item),
            "Rel2 < T >",
            "T : AsRef < Self >",
            Blames::SelfType,
        );
    }
    let by_parts_controls: Vec<Item> = vec![
        // The predicate names an entry that does not use `T`, whose bound names `Self`.
        parse_quote!(
            impl<T: AsRef<Self>, U> ::p::W<T, U, Rel<U>> {
                fn f(&self, a: Rel3<u8>)
                where
                    Rel2<U>: Default,
                {
                }
            }
        ),
        parse_quote!(
            impl<T: AsRef<Self>, U> ::p::Tr<Rel2<U>> for ::p::W<T, U, Rel<U>> where Rel3<U>: Default {}
        ),
        // The `Item` that a bound equates is normalized when the copy proves the Repeater's impl
        // of `Rel<I>` for `Self` (E0277, "expected an `Fn(u8)` closure, found `I`"), unless the
        // context of the entry leaves out `I: Fn(I::Item)` (F26): it compiles.
        parse_quote!(
            impl<I: Iterator<Item = u8> + Fn(I::Item), V: AsRef<Self>> ::p::W<Rel<I>, I, V> {}
        ),
    ];
    for item in &by_parts_controls {
        let (referrer, copy) = f19_copy_with(f13_leaker(), item).unwrap_or_else(|e| {
            panic!(
                "`{}` is rejected, and it compiles: {}",
                string(item),
                joined_messages(&e)
            )
        });
        assert!(
            entries_with_self_bound_in_predicates(&referrer, &copy, self_is_leaked).is_empty(),
            "{}",
            string(&copy)
        );
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
        let mut leaker = f13_leaker();
        leaker.intern_impl(item, true);
        assert_unstable(item, leaker.result().finish(), "Self", bound, *blames);
    }
}

/// F20: in a trait, a method has a bound that names `Self::A` on a generic, and a predicate that
/// names a leaked type that uses the generic.
///
/// Trigger: `trait Tr { type A0; fn m<X>(&self) where X: Fn(&Self::A0) -> u8, Rel<X>: Default; }`
/// (the property test has `T: for<'h> Fn(&'h (X, Self::A1, &'a dyn Any)) -> &'h String`, E0271).
/// The entry `Rel<X>` has the context `<X: Fn(&Self::A0) -> u8>` and `Self` bound by `Tr`, so the
/// Repeater's impl for the implementors has `X: Fn(&<__S as Tr>::A0) -> u8`. The environment of the
/// copy for the host has `X: Fn(&<Host as Tr>::A0) -> u8`, with the projection not normalized yet,
/// where rustc proves `<Host as Repeater<(X,), ..>>::Type: Default`, which needs the impl with
/// `A0` = `u8`: E0277 "expected an `Fn(&u8)` closure, found `X`" (the same with rustc alone).
/// The rule of the `Leaker` docs for `Self::A` ("a bound of the generics of the trait" is unstable)
/// looks at the bounds of the trait, not at those of the method that are in the context of
/// an entry in a predicate. Found by the property test with
/// `TYPE_LEAK_PROP_FEATURES=trait_assoc_bound_cycle`.
///
/// The assertion: if the Leaker accepts the trait, no entry that a predicate of the copy names has
/// a bound of its context that names `Self::A`.
///
/// Fixed with F19: the bound that names `Self::A` is unstable.
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
    let controls: Vec<Item> = vec![
        parse_quote!(
            pub trait Tr {
                type A0;
                fn m<X>(&self, a0: Rel<X>)
                where
                    X: Fn(&Self::A0) -> u8;
            }
        ),
        parse_quote!(
            pub trait Tr {
                type A0;
                fn m<X>(&self)
                where
                    X: Fn(&u8) -> u8,
                    Rel<X>: Default;
            }
        ),
    ];
    let bound = |referrer: &Referrer, context: ContextId| {
        matches!(
            referrer.context_self(context),
            Some(SelfRequirement::Bound(_))
        )
    };
    // Each is rejected: no copy of it exists whose predicate names such an entry.
    let blamed = [
        ("Rel < X >", "X : Fn (& Self :: A0) -> u8", Blames::SelfAssoc("Self :: A0")),
        ("Rel < T , X >", "T : for < 'h > Fn (& 'h (X , Self :: A1 , & 'a dyn :: core :: any :: Any)) -> & 'h String", Blames::SelfAssoc("Self :: A1")),
    ];
    assert_eq!(items.len(), blamed.len());
    for (item, (ty, bound, blames)) in items.iter().zip(blamed) {
        assert_unstable(item, f19_copy(item), ty, bound, blames);
    }
    for item in &controls {
        let (referrer, copy) = f19_copy(item).unwrap_or_else(|e| {
            panic!(
                "`{}` is rejected, and it compiles: {}",
                string(item),
                joined_messages(&e)
            )
        });
        assert!(
            entries_with_self_bound_in_predicates(&referrer, &copy, bound).is_empty(),
            "{}",
            string(&copy)
        );
    }
}

// ── F21 ──────────────────────────────────────────────────────────────────────

/// Whether `generics` name an `Item`: `T::Item`, `<T as Iterator>::Item`.
fn generics_name_item(generics: &Generics) -> bool {
    struct Find(bool);
    impl<'ast> Visit<'ast> for Find {
        fn visit_path(&mut self, i: &'ast Path) {
            self.0 |=
                i.segments.len() > 1 && i.segments.last().map_or(false, |s| s.ident == "Item");
            syn::visit::visit_path(self, i)
        }
    }
    let mut find = Find(false);
    find.visit_generics(generics);
    find.0
}

/// F21: a bound of an iterator names its `Item`, and a predicate of an inner item equates the
/// `Item` to a leaked type.
///
/// Trigger: `impl<T: Iterator> S<T> where T: Fn(T::Item) { fn m(&self) where T: Iterator<Item =
/// Rel<T>> {} }` (and a trait with the bound in its where clause and the equation in a method). The
/// entry `Rel<T>` is in the predicate `T: Iterator<Item = Rel<T>>`, and its context has the
/// generics `<T: Fn(T::Item) + Iterator>`: the Repeater's impl has `T: Fn(T::Item)`. The copy has
/// `T: Iterator<Item = <() as Repeater<(T,), ..>>::Type>`, so rustc proves that impl while it
/// normalizes `T::Item`, which the predicate defines by the entry: E0275 "overflow evaluating the
/// requirement `<() as Repeater<..>>::Type == <T as Iterator>::Item`" (the same with rustc
/// alone; E0277 "expected an `Fn((Rel<T>, u8))` closure" for a tuple). The check of F2 does not see
/// it, as the bound names no leaked type, only the `Item` that the entry defines: the same item
/// with both predicates in one where clause is rejected, but the bound is of an enclosing item here.
/// Found by the property test with `TYPE_LEAK_PROP_FEATURES=iterator_item_cycle`.
///
/// The assertion: if the Leaker accepts the item, no entry that a predicate of the copy names has
/// a bound of its context that names an `Item`.
///
/// Fixed with F19: a bound that names the `Item` (or the `Output` of `Fn(..)`, or a projection that
/// an impl defines) is unstable if an equation of it is in scope. The `Item` of a generic that
/// nothing equates is rigid, and `tests/bound_cycles.rs` has that control.
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
    let controls: Vec<Item> = vec![
        // The `Item` is in a position.
        parse_quote!(
            pub trait Tr<T: Iterator> {
                fn m(&self, a: T::Item)
                where
                    T: Iterator<Item = Rel<T>>;
            }
        ),
        // The bound names no `Item`.
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
    ];
    let check = |item: &Item| {
        let (referrer, copy) = f19_copy(item)?;
        let found = entries_naming_item(&referrer, &copy);
        Ok::<_, Vec<ErrorItem>>((copy, found))
    };
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
        assert_unstable(item, check(item), ty, bound, blames);
    }
    for item in &controls {
        let (copy, found) = check(item).unwrap_or_else(|e| {
            panic!(
                "`{}` is rejected, and it compiles: {}",
                string(item),
                joined_messages(&e)
            )
        });
        assert!(found.is_empty(), "{}", string(&copy));
    }
}

// ── F22 ──────────────────────────────────────────────────────────────────────

/// F22 (fixed): the check of F7 for E0195 also rejected a free function and an inherent method.
///
/// Trigger: `mod m { pub fn f<'x>(a: crate::Wrap<'x>) -> &'x u8 { .. } }` analyzed by
/// `intern_mod()`, and `impl ::p::W { fn f<'x>(&self, a: Wrap<'x>) -> &'x u8 { .. } }`. The note is
/// that of F7: `'x` is late-bound in the original, and the copy has it only in the projection
/// `<() as Repeater<(&'x (),), ..>>::Type`, so it is early-bound, "and rustc rejects the
/// implementation of a method whose lifetime is early-bound where the trait's is late-bound". That
/// is the whole reason (the check is `check_leaked_lifetimes()`, which `with_signature()` ran for
/// every signature that is not the analyzed item): a free function and an inherent method have no
/// trait signature to differ from, so the copy compiles. The same holds for a trait object that
/// takes the lifetime from the reference around it (`&'x dyn RelTrait`): the reference is leaked
/// whole, with the lifetime in the Repeater's arguments, as it is where another argument keeps the
/// lifetime.
///
/// Checked with rustc (stable 1.98.1 and nightly 1.100.0), with a Leaker's crate that has `struct
/// Wrap<'a>` and `impl<'a, __S: ?Sized> Repeater<(&'a (),), 0, 0> for __S { type Type = Wrap<'a>; }`,
/// and a Referrer's crate with the copies
///
/// ```text
/// pub mod m { pub fn f<'x>(a: <() as lk::Repeater<(&'x (),), 0, 0>>::Type) -> &'x u8 { .. } }
/// pub struct W;
/// impl W { pub fn f<'x>(&self, a: <Self as lk::Repeater<(&'x (),), 0, 0>>::Type) -> &'x u8 { .. } }
/// ```
///
/// which compile, with a call of each (`m::f(lk::Wrap(&1))`, `w.f(a)`). The copy of the same method
/// as an impl of a trait of the Leaker's crate, `impl lk::Tr for W { fn f<'x>(&self, a: <Self as
/// lk::Repeater<..>>::Type) -> &'x u8 }` where `trait Tr { fn f<'x>(&self, a: Wrap<'x>) -> &'x u8;
/// }`, is E0195 ("lifetime parameters or bounds on method `f` do not match the trait
/// declaration"). So the controls are rejected still: the trait (F7) and the impl of a trait (the
/// Leaker does not see the signature of the trait, so it assumes the worst).
///
/// What is not a difference: a function that is early-bound in the copy cannot be coerced to
/// `for<'x> fn(Wrap<'x>) -> &'x u8` any more, which the original can be. That is a property of
/// every projection the copy writes (`Self`, `Self::A`, any entry), not what the check is for, and
/// the property test does not make it.
///
/// The assertion: the module, and the inherent impl (also in a module), are accepted. The fix runs
/// the check only for a method of a trait or of an impl of a trait (rustc compares the signature of
/// a method with the trait's in no other case, which was checked with a method of a generic type
/// and an inherent impl in a module too), and not for a function item or a method of an inherent
/// impl. A signature that the walk of a `View` reaches outside an impl (`Visit::visit_signature()`
/// alone) is checked, as the Leaker does not know what it is for.
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
            pub mod m {
                impl ::p::Tr for ::p::W {
                    fn f<'x>(&self, a: crate::Wrap<'x>) -> &'x u8 {
                        loop {}
                    }
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
        let mut leaker = f13_leaker();
        intern(&mut leaker, item);
        let err = errors_of(leaker);
        assert!(!err.is_empty(), "`{}` is accepted", string(item));
        let note = err[0].note().unwrap();
        assert!(
            note.contains("(E0195)"),
            "`{}` is rejected, but not for E0195: {}",
            string(item),
            note
        );
    }
    // No signature to differ from.
    let accepted: Vec<Item> = vec![
        parse_quote!(
            pub mod m {
                pub fn f<'x>(a: crate::Wrap<'x>) -> &'x u8 {
                    loop {}
                }
            }
        ),
        parse_quote!(
            impl ::p::W {
                fn f<'x>(&self, a: Wrap<'x>) -> &'x u8 {
                    loop {}
                }
            }
        ),
        // The same in a module: a free function, and an inherent impl of a type of the module.
        parse_quote!(
            pub mod m {
                pub struct D;
                pub fn f<'x>(a: crate::Wrap<'x>, b: crate::Wrap<'x>) -> crate::Wrap<'x> {
                    loop {}
                }
                impl D {
                    pub fn g<'x>(&self, a: crate::Wrap<'x>) -> &'x u8 {
                        loop {}
                    }
                }
            }
        ),
        // A trait object that takes the lifetime from the reference.
        parse_quote!(
            pub mod m {
                pub fn f<'x>(a: &'x dyn crate::RelTrait) -> &'x u8 {
                    loop {}
                }
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
        let mut leaker = f13_leaker();
        intern(&mut leaker, item);
        if let Err(err) = leaker.result().finish() {
            panic!(
                "`{}` is rejected, and its copy compiles: {}\n{}",
                string(item),
                joined_messages(&err),
                err[0].note().unwrap_or_default()
            );
        }
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
        let mut leaker = f13_leaker();
        leaker.enter_mod("m", [], |view| view.visit_item(&Item::Impl(item.clone())));
        errors_of(leaker)
    };
    assert!(nested(&inherent).is_empty());
    assert!(nested(&of_trait)[0].note().unwrap().contains("(E0195)"));
    let mut leaker = f13_leaker();
    leaker.enter_generics(&Generics::default(), |view| view.visit_signature(&method));
    let result = leaker.result();
    let err = result.errors();
    // `Self` of the receiver has no meaning here, which is told too, in the order of the syntax.
    assert_eq!(err[0].reason, Reason::SelfWithoutMeaning);
    assert!(err[1].note().unwrap().contains("(E0195)"));
}

// ── F23 ──────────────────────────────────────────────────────────────────────

/// F23 (fixed): `Self` in an elided signature was rejected where the Referrer could write the lifetime.
///
/// Trigger: `struct S<'a, F: Fn(&Self) -> &u8>(&'a F)`, `enum E<'a, T> where T: Fn(&Self) -> &u8
/// { A(&'a T) }`, and, with `allow_self = false`, `struct S<'a> { f: fn(&Self) -> &u8, g: &'a u8
/// }`. F15 made the Leaker leak the signature whole where `Self` has lifetimes in the copy
/// (`with_self()` writes `::p::S<'a>`, which adds `'a` to the inputs, and elision fails: E0106), and
/// reject the item where nothing can hold it (a bound, a where predicate) or where `Self` cannot
/// be in the impl of the Repeater (`allow_self = false`). It also rejected when the caller never
/// calls `with_self()`: the Leaker cannot know it, and the copy `struct S<'a, F: Fn(&Self) ->
/// &u8>(&'a F)` is the original.
///
/// A correct behaviour exists: the Referrer writes the lifetime that elision gives, in the
/// signature that `with_self()` changes: `for<'e> Fn(&'e ::p::S<'a, F>) -> &'e u8` (and `for<'e>
/// fn(&'e ::p::S<'a>) -> &'e u8`). `Self` counts no lifetime for rustc, so the elided output has the
/// lifetime of the only `&`, which is what the binder says. Checked with rustc (stable 1.98.1 and
/// nightly 1.100.0), the originals compile in the Leaker's crate (`pub struct S2<'a, F: Fn(&Self) ->
/// &u8>(pub &'a F);`, the enum, `S1<'a> { f: fn(&Self) -> &u8, .. }`, `S4<'a>(Box<dyn Fn(&Self) ->
/// &u8>, ..)` and `S5<'a, T>(Box<dyn Fn(&(T, Self)) -> &u8 + 'a>, ..)`) and these copies in the
/// Referrer's crate compile:
///
/// ```text
/// pub struct S2<'a, F: for<'e> Fn(&'e lk::S2<'a, F>) -> &'e u8>(pub &'a F);
/// pub enum E3<'a, T> where T: for<'e> Fn(&'e lk::E3<'a, T>) -> &'e u8 { A(&'a T) }
/// pub struct S1<'a> { pub f: for<'e> fn(&'e lk::S1<'a>) -> &'e u8, pub g: &'a u8 }
/// pub struct S4<'a>(pub Box<dyn for<'e> Fn(&'e lk::S4<'a>) -> &'e u8>, pub &'a u8);
/// pub struct S5<'a, T>(pub Box<dyn for<'e> Fn(&'e (T, lk::S5<'a, T>)) -> &'e u8 + 'a>, pub &'a T);
/// ```
///
/// while the plain `with_self()` copy `Fn(&lk::S2<'a, F>) -> &u8` is E0106 ("which one of argument 1's
/// 2 lifetimes it is borrowed from"), the failure of F15. Also checked: the same for a named
/// lifetime of the inputs (`for<'x> fn(&'x Self) -> &u8` in `S<'a, 'b>`, whose output must be
/// named `'x`), a `Ref<'_, Self>` input, nested signatures that each have a binder (`fn(&(Self,
/// fn(&Self) -> &u8)) -> &u8`).
///
/// The fix is in the Referrer: where `with_self()` writes a type with lifetimes for a `Self` in the
/// inputs of a function pointer or of `Fn(..)` (in a type, a bound or a where predicate) whose
/// output has an elided lifetime and whose inputs show exactly one lifetime, it names that lifetime
/// with a fresh binder (`for<'__tl_e0>`) in the inputs and in the output, or, if the inputs name
/// it, writes that name in the output. The Leaker does not count `Self` there, so it accepts these
/// items (`allow_self = false` too). It still leaks the signature whole, or rejects, where the
/// Referrer cannot name the lifetime: a path hides it (F28), or the signature is a method's. F15
/// pins that, and the entries that add lifetimes (F12).
///
/// The assertion: the Leaker accepts each item with no entry, and the copy that `with_self()` makes
/// (as the macros of the property test do) has no function pointer or sugar of `Fn` whose elided
/// output the inputs cannot resolve (`Signatures`), the Referrer having named the lifetime. The
/// property test shows it with `TYPE_LEAK_PROP_FEATURES=self_elision` (on by default): it no
/// longer leaks or rejects these items.
#[test]
fn f23_self_in_elided_signature_where_referrer_can_rewrite() {
    let items: Vec<(Item, bool)> = vec![
        // A bound, and a where predicate: nothing can hold the signature.
        (
            parse_quote!(
                pub struct S<'a, F: Fn(&Self) -> &u8>(&'a F);
            ),
            true,
        ),
        (
            parse_quote!(
                pub enum E<'a, T>
                where
                    T: Fn(&Self) -> &u8,
                {
                    A(&'a T),
                }
            ),
            true,
        ),
        // `Self` cannot be in the impl of the Repeater.
        (
            parse_quote!(
                pub struct S<'a> {
                    f: fn(&Self) -> &u8,
                    g: &'a u8,
                }
            ),
            false,
        ),
        (
            parse_quote!(
                pub struct S<'a>(Box<dyn Fn(&Self) -> &u8>, &'a u8);
            ),
            false,
        ),
        (
            parse_quote!(
                pub struct S<'a, T>(Box<dyn Fn(&(T, Self)) -> &u8 + 'a>, &'a T);
            ),
            false,
        ),
        // The inputs name their lifetime: the output takes it.
        (
            parse_quote!(
                pub struct S<'a, 'b> {
                    f: for<'x> fn(&'x Self) -> &u8,
                    g: &'a u8,
                    h: &'b u8,
                }
            ),
            true,
        ),
        (
            parse_quote!(
                pub struct S<'a, F: for<'x> Fn(&'x Self) -> &u8>(&'a F);
            ),
            false,
        ),
        // The lifetime is in a path (`'_`).
        (
            parse_quote!(
                pub struct S<'a>(fn(::core::cell::Ref<'_, Self>) -> &u8, &'a u8);
            ),
            false,
        ),
        // Each of the nested signatures has its binder.
        (
            parse_quote!(
                pub struct S<'a>(fn(&(Self, fn(&Self) -> &u8)) -> &u8, &'a u8);
            ),
            true,
        ),
        // Several bounds of one item, and a bound of a where predicate.
        (
            parse_quote!(
                pub struct S<'a, F: Fn(&Vec<Self>) -> &u8, G>(F, G, &'a u8)
                where
                    G: Fn(&Self) -> Option<&u8>;
            ),
            true,
        ),
    ];
    for (item, allow_self) in &items {
        let (entries, copy) = f15_copy(item, *allow_self).unwrap_or_else(|e| {
            panic!(
                "`{}` is rejected (allow_self = {}), and its copy compiles with `for<'e>`: {}\n{}",
                string(item),
                allow_self,
                joined_messages(&e),
                e[0].note().unwrap_or_default()
            )
        });
        assert!(entries.is_empty(), "{}: {:?}", string(item), entries);
        let mut signatures = Signatures::default();
        signatures.visit_item(&copy);
        assert!(
            signatures.broken.is_empty(),
            "{}: {:?}",
            string(&copy),
            signatures.broken
        );
        assert!(signatures.count > 0, "{}", string(&copy));
    }
}

// ── F24 ──────────────────────────────────────────────────────────────────────

/// F24 (fixed): a bound that names a leaked type on a generic that the entries of the self type do
/// not use made a bare `Self` in a predicate an error.
///
/// Trigger: `impl<T: Fn(Rel2<u8>), U> ::p::Tr<U> for ::p::W2<T, Rel<U>> where Self: Clone {}` (also
/// with a method `fn m<X: AsRef<Self>>(&self, x: X) {}`, and both). The self type has the entry
/// `Rel<U>`, whose context has `U` and no bound, and `T`, whose bound names the leaked `Rel2<u8>`,
/// is no entry's. F13 and F19 narrowed the rule to this
/// ("`impl<T: Fn(Rel2), U, V: AsRef<Self>> ::p::W<T, Rel<U>, V>` is fine", the `Leaker` docs,
/// "Bounds that name leaked types"): rustc normalizes `Self` with the impl of the Repeater for
/// `Rel<U>`, which needs the bounds of `U`, not those of `T`. But with `Self` in a predicate the
/// Leaker rejects it ("`Self` is in a bound or where predicate, and a bound of its generics names a
/// leaked type"): `self_type_entries()` returns the contexts of every type of the self type that
/// the graph MAY choose as an entry, and the whole self type `W2<T, Rel<U>>` is one, with the
/// generics `<T: Fn(Rel2<u8>), U>`, whose bound is unstable. The graph chooses the parts (two
/// entries, `Rel2<u8>` and `Rel<U>`), as it does where the impl has no predicate with `Self`, so
/// the check looks at a context that is not in the result.
///
/// Checked with rustc (stable 1.98.1 and nightly 1.100.0), with the checks of F13: the Leaker's
/// crate with the impl of the trait, the Repeater's impls of the two entries (`impl<__S: ?Sized>
/// Repeater<(), 0, 0> for __S { type Type = Rel2<u8>; }` and `impl<U, __S: ?Sized> Repeater<(U,), 0,
/// 1> for __S { type Type = Rel<U>; }`) and the witnesses compile, and the Referrer's crate with the
/// copy `impl<T: Fn(<() as Repeater<(), 0, 0>>::Type), U> Tr<U> for W2<T, <() as Repeater<(U,), 0,
/// 1>>::Type> where Self: Clone { fn f(&self, a: u8) {} fn m<X: AsRef<Self>>(&self, x: X) {} }`
/// compiles, and so does the comparison of each type position with the original. (The trait is
/// `Tr<X = ()>` with `fn f(&self, _a: u8)` and `fn m<Y: AsRef<Self>>(&self, _y: Y)` in both
/// crates, so that the impl is not an inherent impl of a type of another crate, E0116, and the type
/// parameter `U` is constrained by the trait, E0207 of F11.)
///
/// The assertion: the Leaker accepts the impls, with the parts as entries, as the copy that was
/// checked has them.
///
/// Fixed so that the check and the choice agree: when a bare `Self` is in a bound or where predicate
/// of an impl, a type in the self type that the graph MAY choose, that no cause forces to be an
/// entry, and whose context has an unstable bound is marked `no_entry` (`block_unstable_self_types()`
/// of `src/analyze.rs`), so the graph chooses its parts, and the check looks at the same ones
/// (`self_type_entries()`). It is an error only if a part that must be an entry has an unstable
/// bound in its context. The variants below were checked with rustc as the first: the bound as a
/// where predicate, and a type between the self type and the entry (`W3<W1<Rel<U>>, T, u8>`,
/// `W2<W1<T>, Rel<U>>`). The impls whose entry uses `T` (`W2<Rel<T>, U>`, `W3<Rel<T>, T, V>`)
/// are still errors: `Rel<T>` must be an entry, and its Repeater's impl needs `T: Fn(Rel2<u8>)`.
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
        let (referrer, _) =
            f19_copy_with(f13_leaker(), &Item::Impl(item.clone())).unwrap_or_else(|e| {
                panic!(
                    "`{}` is rejected, and it compiles with the parts as entries: {}\n{}",
                    string(item),
                    joined_messages(&e),
                    e[0].note().unwrap_or_default()
                )
            });
        let mut entries: Vec<String> = referrer.iter().map(string).collect();
        entries.sort();
        assert_eq!(entries, ["Rel < U >", "Rel2 < u8 >"], "{}", string(item));
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
            f19_copy_with(f13_leaker(), &Item::Impl(item.clone())),
            "Self",
            "T : Fn (Rel2 < u8 >)",
            Blames::Leaked,
        );
    }
}

// ── F25 and F26 ──────────────────────────────────────────────────────────────

/// The copy of `item` (an impl or a trait) with the Leaker of F13 (`::p` is allowed), and the
/// entries that a predicate of the copy names whose context has a bound that names an `Item`
/// (the oracle of F21).
fn entries_naming_item_in_predicates(
    item: &Item,
) -> std::result::Result<(Item, Vec<String>), Vec<ErrorItem>> {
    let (referrer, copy) = f19_copy_with(f13_leaker(), item)?;
    let found = entries_naming_item(&referrer, &copy);
    Ok((copy, found))
}

/// The entries that a predicate of `copy` names whose context has a bound that names an `Item`.
fn entries_naming_item(referrer: &Referrer, copy: &Item) -> Vec<String> {
    let mut used = ReplacementsInPredicates::default();
    used.visit_item(copy);
    used.0
        .iter()
        .filter(|&&ix| ix < referrer.iter().count() && generics_name_item(referrer.generics(ix)))
        .map(|&ix| string(referrer.iter().nth(ix).unwrap()))
        .collect()
}

/// F25 (fixed): a supertrait fixes the `Item` that a bound names, and an entry that uses the generic
/// is in a predicate.
///
/// Trigger: `trait Tr<T: ::p::WIter> where T: Fn(T::Item) { fn m(&self) where Rel<T>: Default; }`,
/// with `trait WIter: Iterator<Item = u8> {}` in the Leaker's crate (also the bound on a method, `fn
/// m<T: ::p::WIter + Fn(T::Item)>(&self) where Rel<T>: Default`, and an impl). It is F21 with the
/// equation `Item = u8` in a supertrait instead of a bound or predicate in scope. The entry
/// `Rel<T>` has the generics `<T: ::p::WIter + Fn(T::Item)>`, and the Repeater's impl has `T:
/// Fn(T::Item)` normalized to `T: Fn(u8)`. The environment of the copy has `T: Fn(<T as
/// Iterator>::Item)`, which rustc normalizes only after it has built it, while it proves the
/// predicate `<() as Repeater<(T,), ..>>::Type: Default`: E0277 "expected an `Fn(u8)` closure, found
/// `T`" (checked with rustc, stable 1.98.1 and nightly 1.100.0, by the Leaker's crate with the
/// trait and the Repeater's impl, which compile, and the Referrer's crate with the impl of the trait
/// for a host type, which does not; nightly says `FnOnce`). `Projections` finds the equations of
/// bounds and predicates in scope, and `::p::WIter` is a path to a trait that the Leaker cannot
/// read, so it did not see the equation of its supertrait.
///
/// What compiles: a generic that no equation fixes. `T: Iterator` (rigid `Item`, the control
/// `Tr4`, in `tests/bound_cycles.rs` too) and the entry in a position (`Tr2`: it is normalized once
/// the environment is built). A supertrait that does not fix the `Item` (`trait WIter2: Iterator
/// {}`), and a trait with an associated type of its own (`trait Cont { type Out; }`, `T:
/// Fn(T::Out)`) compile as well (checked with rustc, stable and nightly, as the first), but the
/// Leaker cannot tell them from `WIter`. A supertrait that equates the associated type of its own
/// supertrait (`trait Sub: Cont<Out = u8>`) fails as `WIter` does.
///
/// The property test shows it with `TYPE_LEAK_PROP_FEATURES=supertrait_item` (on by default since
/// the fix; before it, seeds 1, 2 and 3 of 60 cases failed with E0277 "expected an `FnOnce(u8)`
/// closure, found `X`" in the Referrer's crate).
///
/// The assertion is that of F21: if the Leaker accepts the item, no entry that a predicate of the
/// copy names has a bound of its context that names an `Item`.
///
/// First fixed by rejecting: a projection (`T::Item`, `<T as Iterator>::Item`) of a generic that
/// has a bound of a trait that is not one of the standard library (`unstable.rs`, `bindings()`:
/// `T: ::p::WIter`, `T: Tr`, also in a where predicate or an enclosing item) counts as an
/// associated type that rustc normalizes, as one that a bound equates. A bound of the standard
/// library (`Iterator`, `Clone`, `Fn(..)`, a path in `::core`, `::std` or `::alloc`) does not. The
/// price was that the supertrait that does not fix the `Item` and the trait with an associated
/// type of its own were rejected too.
///
/// Fixed for good by F26: the context of the entry leaves out `T: Fn(T::Item)`, because another
/// bound of `T` (`::p::WIter`, a trait that is not one of the standard library) may fix an
/// associated type, so the Repeater's impl has `T: ::p::WIter` only, and the copy compiles (all
/// the items below were built as the fixtures 9175 and 9176 of `tests/cross_crate.rs`, and the
/// price is no price any more). The projection that stays rejected is that of another generic
/// (`U: Fn(T::Item)` next to `T: ::p::WIter`), and the one whose equation is in a predicate of
/// a method (F21), which are not in the context of the entry.
#[test]
fn f25_supertrait_item_in_bound_of_entry_in_predicate() {
    // The controls: rigid `Item`, and the entry in a position.
    let accepted: Vec<Item> = vec![
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
    ];
    for item in &accepted {
        f19_copy_with(f13_leaker(), item).unwrap_or_else(|e| {
            panic!(
                "`{}` is rejected, and it compiles: {}",
                string(item),
                joined_messages(&e)
            )
        });
    }
    // More controls: every bound of the generic is a trait of the standard library (compiled with
    // rustc too), and a trait with a supertrait is no bound of the generic that the `Item` names.
    let std_bounds: Vec<Item> = vec![
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
    for item in &std_bounds {
        f19_copy_with(f13_leaker(), item).unwrap_or_else(|e| {
            panic!(
                "`{}` is rejected, and it compiles: {}",
                string(item),
                joined_messages(&e)
            )
        });
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
        let (referrer, copy) = f19_copy_with(f13_leaker(), item).unwrap_or_else(|e| {
            panic!(
                "`{}` is rejected, and it compiles: {}",
                string(item),
                joined_messages(&e)
            )
        });
        assert_eq!(
            generics_text(referrer.generics(entry(&referrer, "Rel < T >"))),
            *generics
        );
        assert!(has_replacement_in_predicate(&copy), "{}", string(&copy));
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
        let (copy, found) = entries_naming_item_in_predicates(item).unwrap_or_else(|e| {
            panic!(
                "`{}` is rejected, and it compiles: {}",
                string(item),
                joined_messages(&e)
            )
        });
        assert!(found.is_empty(), "{}", string(&copy));
        assert!(has_replacement_in_predicate(&copy), "{}", string(&copy));
        let (referrer, _) = f19_copy_with(f13_leaker(), item).unwrap();
        assert_eq!(
            string(referrer.generics(entry(&referrer, "Rel < T >"))),
            "< T : :: p :: WIter >",
            "{}",
            string(item)
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
        entries_naming_item_in_predicates(&other),
        "Rel < U >",
        "U : Fn (T :: Item)",
        Blames::Projection("T :: Item"),
    );
}

/// The generics of an entry with their where clause, as the Repeater's impl has them.
fn generics_text(generics: &Generics) -> String {
    let where_clause = &generics.where_clause;
    string(&quote!(#generics #where_clause))
}

/// The Repeater's impl of the entry `ix`, for the implementor `__S` (what a macro writes).
fn repeater_impl_text(referrer: &Referrer, ix: usize) -> String {
    let param: Ident = parse_quote!(__S);
    let r = referrer
        .repeater_impl(ix, ImplSelf::Param(&param))
        .unwrap_or_else(|e| panic!("{}", e));
    let repeater: Path = parse_quote!(crate::Repeater);
    let (impl_generics, _, where_clause) = r.generics.split_for_impl();
    let (args, self_ty, ty) = (&r.args, &r.self_ty, &r.ty);
    let impl_: ItemImpl = parse_quote! {
        impl #impl_generics #repeater<#args, 0u64, 0usize> for #self_ty #where_clause {
            type Type = #ty;
        }
    };
    string(&impl_)
}

/// F26 (fixed): a type alias hides the projection that a bound names.
///
/// Trigger: `trait Tr<T: Iterator<Item = u8>> where T: Fn(::p::WIt<T>) { fn m(&self) where Rel<T>:
/// Default; }` with `type WIt<T> = <T as Iterator>::Item;` in the Leaker's crate. It is F21 and F25
/// with the `Item` written through an alias: the Leaker saw no projection and accepted the item,
/// and the copy is E0277 "expected an `Fn(u8)` closure, found `T`" (checked with rustc, stable
/// 1.98.1, nightly 1.100.0 with and without `-Znext-solver`, 1.70 and 1.56, with the Leaker's crate
/// and the Referrer's with the impl of the trait for a host type). With the projection written
/// (`T: Fn(T::Item)`) the Leaker rejected the item.
///
/// The cause is in the impl of the Repeater, not in the alias. The impl of the entry `Rel<T>` has
/// every bound of `T` in its context, `T: Fn(::p::WIt<T>)` too, and rustc has to prove that where
/// clause whenever the copy writes `<() as Repeater<(T,), ..>>::Type: Default` in the where
/// clause of a method, in the environment of the method, which it builds first. It proves `T:
/// Fn(WIt<T>)` there normalized (`T: Fn(u8)`), with the equation `T: Iterator<Item = u8>`, and
/// cannot use the equation to normalize the environment it is building (a limitation of rustc, the
/// same with `-Znext-solver`). The environment has the bound as written. With an impl that does not
/// carry the bound, everything compiles:
///
/// ```text
/// impl<T: Iterator<Item = u8>, __S: ?Sized> Repeater<(T,), 0, 0> for __S { type Type = Rel<T>; }
/// trait C<T: Iterator<Item = u8>> where T: Fn(::p::WIt<T>) {
///     fn m(&self) where <() as ::p::Repeater<(T,), 0, 0>>::Type: Default;
/// }
/// ```
///
/// (the same with `T: Fn(<T as Iterator>::Item)`, F25's written projection, and with `T:
/// Fn(::p::WFoo<T>)`, a real struct, which compiled before). With the bound in the impl, the alias
/// and the written projection fail, and the struct compiles. The Leaker cannot tell an alias from a
/// struct (`::p::WIt<T>` is a path with `T` as its argument, as `::p::WFoo<T>` is), so it could
/// not reject the alias without rejecting the struct.
///
/// Fixed in the context of the entry (`Context::of()`, and so in the Referrer too): it leaves out a
/// bound of a generic `T` that names `T` in the arguments of its path (`T: Fn(X<T>)`, `T:
/// Tr<<T as Iterator>::Item>`, `T: PartialEq<Wrap<T>>`, as a bound or as a where predicate `T: ..`
/// of the scope that declares `T`), if another bound of `T` fixes an associated type: an equation
/// (`T: Iterator<Item = u8>`, `T: Tr<A = X>`), or a trait that is not one of the standard library
/// and does not name `T` itself (`T: ::p::WIter`; the Leaker does not read its supertraits, see
/// F25). A bound that is an equation stays, even if it names `T`, and so does every other bound.
///
/// The rule is about `T` alone. A bound of another generic that names `T`'s `Item` (`U:
/// Fn(T::Item)`) stays, and the Leaker rejects it where it is written as a projection (F21, F25).
/// Behind an alias (`U: Fn(::p::WIt<T>)`) the Leaker does not see it, and it fails the same way
/// (checked with rustc: `impl<U: Iterator<Item = u8>, T: Fn(WIt<U>), __S: ?Sized> Repeater<(T, U),
/// ..>` and the copy `trait C<U: Iterator<Item = u8>, T: Fn(WIt<U>)> { fn m(&self) where <() as
/// Repeater<(T, U), ..>>::Type: Default; }` are E0277). That stays a limitation: the property test
/// does not make it (its alias bounds the iterator itself), and widening the rule to any generic
/// that a bound names (`T: Into<U>` with `U: Iterator<Item = u8>`) would leave out bounds that have
/// nothing to do with a projection. The bound of an inner scope (a where clause of a method) is not
/// in the context of an entry of the generic of the trait at all.
///
/// Not a finding of the Leaker: the same limitation of rustc makes a trait with such a bound
/// unimplementable by a generic impl, whatever `type-leak` does. `trait Tr<U: Iterator<Item = u8>>
/// where U: Fn(WIt<U>) -> u8 { type A: Clone; fn m(&self) where Self::A: Clone; }` and `impl<U:
/// Iterator<Item = u8>> Tr<U> for Host where U: Fn(WIt<U>) -> u8` is E0277 "expected an `Fn(u8)`
/// closure, found `U`" in the impl (checked with rustc alone, stable 1.98.1 and 1.56): the method's
/// predicate names `Self::A`, rustc normalizes it with the impl, and proves the impl's where clause
/// `U: Fn(u8)` against the environment's `U: Fn(WIt<U>)`. The same with `X: Into<Self::A>`, not with
/// `Self::A` in a position, nor with `WFoo<U>` (a struct) for the alias. The Leaker rejects the
/// projection written (`Self::A` in a predicate, and a bound of the generics that names `T::Item`:
/// `impl_unstable_bound()`), and cannot see the alias; the property test (seed 24 of 24 cases, case
/// 9) found it through the impl for a host type that its macro writes, so the generator makes no
/// method predicate that names `Self::A` in a trait that has such a bound.
///
/// The risk: a leaked type that needs the bound that is left out (`struct Rel<T: Fn(X<T>)>`, or a
/// type that names `<T as Tr<Wrap<T>>>::A`) does not compile in the Repeater's impl: E0277 in the
/// Leaker's crate, which the Leaker cannot see. The `Leaker` docs say so. It is not a risk where
/// the Leaker knows the declaration: the entry that names `Self` (the self type of the item,
/// `Wrap2<Self>` in `struct S<T: Iterator<Item = u8> + Fn(T::Item)>`, or `Self::A` of a trait), and
/// the entry that writes out the self type of the scope (`impl<T: Iterator<Item = u8> +
/// Fn(T::Item)> S<T>` with the entry `S<T>`, whose top-level context then records `Self` for the
/// Referrer, as it does for a bound rewritten for it), keep every bound, because the type is
/// well-formed only with the bounds that its item declares (the property test showed it: seeds 1
/// and 2 were E0277 in the Leaker's crate for a struct, an enum, a union, an impl and a trait, with
/// the fixtures 9182 and 9183 of `tests/cross_crate.rs`). An entry that depends on `Self` only
/// through a bound or an outlives predicate (`Self: 'x` of a method) keeps whole the bounds of the
/// generics that the description of `Self` names (`impl<T: ..> S<T>`, `Tr<T>`), and trims the
/// others: see F47.
///
/// The property test shows it with `TYPE_LEAK_PROP_FEATURES=alias_item` (on by default since the
/// fix; before it, seeds 1, 2 and 3 of 60 cases failed with the same E0277 as F25).
///
/// What this lifts: the shapes of F25 (the price is no price any more), the written projection
/// (`T: Iterator<Item = u8> + Fn(T::Item)`), and a bound that names a leaked type next to an
/// equation (`T: Iterator<Item = u8> + Into<Wrap<T>>`, F2, whose only cause was the bound in the
/// impl of its own entry). What stays rejected: the equation in a predicate of a method (F21, `T:
/// Iterator` in the context, so `T: Fn(T::Item)` is kept), `U: Fn(T::Item)` with the equation on
/// `T`, a bound that names a leaked type with no equation, and every bound that names `Self`,
/// `Self::A` or a projection that an impl defines.
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
    let (referrer, copy) = f19_copy_with(f13_leaker(), &alias).unwrap();
    let ix = entry(&referrer, "Rel < T >");
    // The context leaves out `T: Fn(::p::WIt<T>)`: the Repeater's impl has `T: Iterator<Item = u8>`
    // only, and the copy is as written, with the entry in the predicate.
    assert_eq!(
        generics_text(referrer.generics(ix)),
        "< T : Iterator < Item = u8 > >"
    );
    assert_eq!(
        repeater_impl_text(&referrer, ix),
        "impl < T : Iterator < Item = u8 > , __S : ? :: core :: marker :: Sized > crate :: \
         Repeater < (T ,) , 0u64 , 0usize > for __S { type Type = Rel < T > ; }"
    );
    assert!(has_replacement_in_predicate(&copy), "{}", string(&copy));
    assert_eq!(
        string(&copy),
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
        let (referrer, copy) = f19_copy_with(f13_leaker(), item).unwrap_or_else(|e| {
            panic!(
                "`{}` is rejected, and it compiles: {}",
                string(item),
                joined_messages(&e)
            )
        });
        assert!(has_replacement_in_predicate(&copy), "{}", string(&copy));
        assert_eq!(
            string(referrer.generics(entry(&referrer, "Rel < T >"))),
            "< T : Iterator < Item = u8 > >",
            "{}",
            string(item)
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
        let (referrer, _) = f19_copy_with(f13_leaker(), item).unwrap_or_else(|e| {
            panic!(
                "`{}` is rejected, and it compiles: {}",
                string(item),
                joined_messages(&e)
            )
        });
        assert_eq!(
            string(referrer.generics(entry(&referrer, "Rel < T >"))),
            *generics,
            "{}",
            string(item)
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
    let mut leaker = f13_leaker();
    leaker.intern_struct(&self_type, true);
    let referrer = leaker.result().finish().unwrap();
    assert_eq!(
        generics_text(referrer.generics(entry(&referrer, "Wrap2 < Self >"))),
        "< T : Fn (:: p :: WIt < T >) + Iterator < Item = u8 > >"
    );
    // The self type of an impl, written out, is declared with the bounds of the item. The top-level
    // context records the self type then, which the Referrer needs to find the entry.
    let impl_self: ItemImpl = parse_quote!(
        impl<T: Iterator<Item = u8>> S<T>
        where
            T: Fn(::p::WIt<T>),
        {
            fn m(&self, x: Rel<T>) {}
        }
    );
    let mut leaker = f13_leaker();
    let context = leaker.intern_impl(&impl_self, true);
    let referrer = leaker.result().finish().unwrap();
    assert_eq!(
        generics_text(referrer.generics(entry(&referrer, "S < T >"))),
        "< T : Fn (:: p :: WIt < T >) + Iterator < Item = u8 > >"
    );
    assert_eq!(
        generics_text(referrer.generics(entry(&referrer, "Rel < T >"))),
        "< T : Iterator < Item = u8 > >"
    );
    assert_eq!(
        referrer.context_self(context),
        Some(&SelfRequirement::Type(parse_quote!(S<T>)))
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
    let (referrer, _) = f19_copy_with(f13_leaker(), &assoc).unwrap();
    assert_eq!(
        generics_text(referrer.generics(entry(&referrer, "Rel < Self :: A >"))),
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
        entries_naming_item_in_predicates(&method_equation),
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
        entries_naming_item_in_predicates(&other),
        "Rel < U >",
        "U : Fn (T :: Item)",
        Blames::Projection("T :: Item"),
    );
    // `Self::A` in a predicate of a method of a trait whose bound names the `Item` is no impl for
    // rustc itself (see above): the Leaker rejects the projection written, and does not see the
    // alias (it takes it for a struct, with which the impl compiles).
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
        f19_copy_with(f13_leaker(), &written_self_assoc),
        &["`Self :: A` is in a bound or where predicate, and a bound of its generics names `T :: Item`"],
    );
    assert!(
        all_notes(&err)[0].starts_with(
            "the copy for a concrete implementor resolves `Self :: A` by the impl of the trait, \
             which needs the bound `T : Fn (T :: Item) + Iterator < Item = u8 >` of its generics"
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
    f19_copy_with(f13_leaker(), &aliased_self_assoc).unwrap();
    // A bound that names the leaked type of its own entry: with the equation, it is left out of
    // the context of that entry (F2, and nothing is left to be a cycle); without it, it is the
    // cycle that F2 rejects.
    let leaked_with_equation: Item = parse_quote!(
        pub trait Tr<T: Iterator<Item = u8> + Into<Rel<T>>> {
            fn m(&self);
        }
    );
    let (referrer, _) = f19_copy_with(f13_leaker(), &leaked_with_equation).unwrap();
    assert_eq!(
        generics_text(referrer.generics(entry(&referrer, "Rel < T >"))),
        "< T : Iterator < Item = u8 > >"
    );
    let leaked: Item = parse_quote!(
        pub trait Tr<T: Into<Rel<T>>> {
            fn m(&self);
        }
    );
    assert_unstable(
        &leaked,
        f19_copy_with(f13_leaker(), &leaked),
        "Rel < T >",
        "T : Into < Rel < T > >",
        Blames::Leaked,
    );
}

/// Whether a where predicate of the copy names a replacement `P<ix>`.
fn has_replacement_in_predicate(item: &Item) -> bool {
    let mut used = ReplacementsInPredicates::default();
    used.visit_item(item);
    !used.0.is_empty()
}

// ── F27 ──────────────────────────────────────────────────────────────────────

/// The replacements `P<ix>` in the bounds of the `impl Trait` types of a copy: an argument `x:
/// impl Into<P<1>>` is a generic of the function with that bound.
#[derive(Default)]
struct ImplTraitMentions(Vec<String>);

impl<'ast> Visit<'ast> for ImplTraitMentions {
    fn visit_type_impl_trait(&mut self, i: &'ast TypeImplTrait) {
        if i.bounds.iter().any(has_bound_replacement) {
            self.0.push(string(i));
        }
        syn::visit::visit_type_impl_trait(self, i)
    }
}

/// F27 (fixed): the bounds of an argument-position `impl Trait` were not checked as the bounds of a
/// generic.
///
/// Trigger: `trait Tr<T: Fn(Rel0)> { fn m(&self, x: impl Into<Wrap<T>>); }` (also `impl
/// Iterator<Item: Into<Wrap<T>>>`, and `T` a generic of the method). `x: impl Into<Wrap<T>>` is `x:
/// X` with `X: Into<Wrap<T>>`, a bound, so the entry `Wrap<T>` in it is F2: the context of `Wrap<T>`
/// has `T: Fn(Rel0)`, which names a leaked type, so the copy has `T: Fn(<() as Repeater<..>>::Type)`
/// and the impl of the Repeater for `Wrap<T>` needs `T: Fn(Rel0)`: E0277 "expected an `Fn(Rel0)`
/// closure, found `T`" (checked with rustc, stable 1.98.1 and nightly 1.100.0, with the Leaker's crate
/// with the trait and the Repeater's impls, and the Referrer's with the impl of the trait for a host
/// type; nightly says `FnOnce`). The Leaker rejects `fn m<X: Into<Wrap<T>>>(&self, x: X)`, the same
/// with a generic written (its bound is visited outside of a type, `env_depth`); a bound inside the
/// `impl Trait` of a type was not, since the check took it for the bound of a trait object or of a
/// path. The same for `Vec<impl Into<Wrap<T>>>` and `&impl Into<Wrap<T>>`, which a copy with `T`
/// a generic of the method fails too (checked with rustc, stable and nightly).
///
/// The property test shows it with `TYPE_LEAK_PROP_FEATURES=impl_trait_args` (on by default since
/// the fix; before it, seeds 1, 2 and 3 of 60 cases failed with E0277 "expected an `FnOnce(B0)`
/// closure, found `X`", and once with E0271). The test macro does not compare a position that has
/// `impl Trait` (a witness cannot return it): `has_no_witness()` of `tests/crates/macro/prop.rs`.
///
/// The assertion is that of F2, for the bounds of `impl Trait`: either the Leaker rejects the item,
/// or its copy does not have both a bound of a generic and a bound of an `impl Trait` that
/// mention replacements. A return-position `impl Trait` is not at stake: `fn m(&self) -> impl
/// Into<Wrap<T>>` compiles (checked, the same copy), as its bound is proved where the body is.
///
/// Fixed by extending the rule of F2 to them: the types in the bounds of an `impl Trait` in an
/// argument of a function or method (`in_fn_arg` of `src/analyze.rs`, anywhere in the type of the
/// argument, also nested) are in the environment of the item (`env_depth`), as the bounds of a
/// generic are. The fix rejects each of the items; the test pins that.
#[test]
fn f27_argument_impl_trait_bound_names_leaked_type_env_cycle() {
    let controls: Vec<Item> = vec![
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
    ];
    for item in &controls {
        f19_copy_with(f13_leaker(), item).unwrap_or_else(|e| {
            panic!(
                "`{}` is rejected, and it compiles: {}",
                string(item),
                joined_messages(&e)
            )
        });
    }
    // The same with a generic written, which the Leaker rejects.
    let written: Item = parse_quote!(
        pub trait Tr<T: Fn(Rel0)> {
            fn m<X: Into<Wrap<T>>>(&self, x: X);
        }
    );
    assert_unstable(
        &written,
        f19_copy_with(f13_leaker(), &written),
        "Wrap < T >",
        "T : Fn (Rel0)",
        Blames::Leaked,
    );
    // More controls: where the bound of the `impl Trait` names no entry that uses a generic with a
    // bound that names a leaked type.
    let more: Vec<Item> = vec![
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
    for item in &more {
        f19_copy_with(f13_leaker(), item).unwrap_or_else(|e| {
            panic!(
                "`{}` is rejected, and it compiles: {}",
                string(item),
                joined_messages(&e)
            )
        });
    }
    let items: Vec<Item> = vec![
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
    // Each is rejected: no copy of it exists that has both a bound of a generic and a bound of an
    // `impl Trait` that mention replacements.
    let blamed = [
        ("Wrap < T >", "T : Fn (Rel0)", Blames::Leaked),
        ("Wrap < T >", "T : Fn (Rel0)", Blames::Leaked),
        ("Wrap < T >", "T : Fn (Rel0)", Blames::Leaked),
        ("Wrap < T >", "T : Fn (Rel0)", Blames::Leaked),
        ("Wrap < T >", "T : Fn (Rel0)", Blames::Leaked),
        ("Wrap < T >", "T : Fn (Rel0)", Blames::Leaked),
        ("Wrap < T >", "T : Fn (Rel0)", Blames::Leaked),
        ("Wrap < T >", "T : Fn (Rel0)", Blames::Leaked),
    ];
    assert_eq!(items.len(), blamed.len());
    for (item, (ty, bound, blames)) in items.iter().zip(blamed) {
        assert_unstable(item, f19_copy_with(f13_leaker(), item), ty, bound, blames);
    }
}

// ── F28 and F29 ──────────────────────────────────────────────────────────────

/// F28 (fixed): a path that hides a lifetime in the inputs of an elided signature, next to an entry
/// that adds lifetimes.
///
/// Trigger: `trait Tr<'a, T: 'a> { fn f(&self, x: fn(::p::WRef<u8>, Rel<T>) -> &u8); }` with
/// `struct WRef<'a, T: ?Sized>(&'a T)` (also `Box<dyn Fn(::p::WRef<u8>, Rel<T>) -> &u8>`,
/// `fn(::core::slice::Iter<u8>, Rel<T>) -> &u8` with a type of the standard library, and the entry
/// inside the path, `fn(::p::WRef<Rel<T>>) -> &u8`). The inputs have exactly one lifetime, which
/// the path hides (`WRef<'_, u8>`), and elision gives it to the output. The entry `Rel<T>` has the
/// generics `<'a, T: 'a>`, so the copy has `<() as Repeater<(&'a (), T, &'a T), ..>>::Type`, which
/// names `'a`: two lifetimes, E0106 "which one of argument 1's 2 lifetimes it is borrowed from"
/// (checked with rustc, stable 1.98.1 and nightly 1.100.0, with the impl of the trait for a host
/// type; it is what F12 fixed for `&`). The Leaker did not see the hidden lifetime (the `Leaker`
/// docs, Elided lifetimes: "A lifetime that a path hides (`Ref<u8>` for `Ref<'a, u8>`) is not
/// seen", so write `Ref<'_, u8>`), and `Rel<T>` alone was the entry.
///
/// The property test showed it with `TYPE_LEAK_PROP_FEATURES=hidden_lifetimes` (E0106 in the
/// Referrer's crate for `fn(::core::str::Chars, A) -> &R` and `fn(::core::slice::Iter<u8>, A) ->
/// &R`: seeds 2, 5, 6, 7, 8, 9 and 11 of 60 cases failed, 1, 3, 4, 10 and 12 passed, as the entry
/// must add lifetimes); it is on by default now.
///
/// The docs told the user to write `WRef<'_, u8>`, which the Leaker leaks whole with the holder
/// (the control, `Tr1`: compiles, checked). A correct behaviour that needs no advice exists, since
/// an elided output lifetime with no lifetime in the inputs that the Leaker sees can only be the
/// hidden one (the original compiles): the fix counts it as the one that elision uses, as `&` does
/// (`Elision::hidden`), so any lifetime that the copy adds to the inputs breaks it, and the holder
/// is leaked whole, or the item is rejected where nothing can hold it. The same holds for a method
/// with no receiver that is a reference. A hidden lifetime in the output is not seen, so the copy
/// of `fn(&Rel<T>) -> WRef<u8>` is still E0106: the limitation that the `Leaker` docs document.
///
/// The assertion: the Leaker rejects the item with the note of F12 (E0106), or the holder is the
/// entry, so that nothing of it is left in the copy (`Signatures`). The controls are accepted
/// as before: the lifetime written, an entry that adds none (`Tr5<T>`), an output that does not
/// rely on elision, and a hidden lifetime in a signature that has no entry. The items that are
/// rejected name the hidden lifetime in the note, and they are the bound and the method.
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
        let (entries, _) = f12_copy_with(f13_leaker(), item).unwrap_or_else(|e| {
            panic!(
                "`{}` is rejected, and it compiles: {}",
                string(item),
                joined_messages(&e)
            )
        });
        assert_eq!(entries, [*holder], "{}", string(item));
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
        let (entries, copy) = f12_copy_with(f13_leaker(), item)
            .unwrap_or_else(|e| panic!("`{}` is rejected: {}", string(item), joined_messages(&e)));
        assert_eq!(entries, [*holder], "{}", string(item));
        let mut signatures = Signatures::default();
        signatures.visit_item_trait(&copy);
        assert_eq!(
            signatures.count,
            0,
            "`{}`: the entries {:?} leave a signature in the copy, whose elided output has the hidden lifetime and the one of `Rel<T>`: {}",
            string(item),
            entries,
            string(&copy)
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
            f12_copy_with(f13_leaker(), item),
            &["use absolute path: 'Rel < T >'"],
        );
        // The note says that the lifetime is hidden by a path, and the help how to write it.
        assert_eq!(
            all_notes(&err),
            [
                "the output `& u8` has an elided lifetime, which elision takes from the only \
                 lifetime of the inputs, which a path hides (the Leaker does not see it), but a \
                 leaked type in the inputs has `'a` among its generics, and the copy names it in \
                 the arguments of its projection, so the inputs have more than one lifetime \
                 (E0106)",
                "it cannot be leaked as a whole because of `& u8`"
            ],
            "{}",
            string(item)
        );
        let helps = all_helps(&err);
        assert_eq!(helps.len(), 1, "{}", string(item));
        assert!(
            helps[0].starts_with(
                "write the lifetime that the path hides (`Ref<'x, u8>`), and name it in the \
                 output, "
            ) && helps[0].ends_with("or write `Rel<T>` with an absolute path"),
            "{}: {}",
            string(item),
            helps[0]
        );
    }
    // A method with a receiver that is a reference takes the lifetime from it: nothing is hidden.
    let accepted: ItemTrait = parse_quote!(
        pub trait Tr<'a, T: 'a> {
            fn f(&self, x: ::p::WRef<u8>, y: Rel<T>) -> &u8;
        }
    );
    let (entries, _) = f12_copy_with(f13_leaker(), &accepted).unwrap();
    assert_eq!(entries, ["Rel < T >"]);
}

/// F29 (limitation): an entry that hides a lifetime.
///
/// Trigger: `trait Tr0 { fn f(&self, x: fn(RelRef<u8>) -> &u8); }` with `struct RelRef<'a, T>(&'a
/// T)` in the Leaker's crate: `RelRef<u8>` is a relative path, so it is an entry with no
/// generics, and the Repeater's impl is `impl<__S: ?Sized> Repeater<(), 0, 0> for __S { type Type =
/// RelRef<u8>; }`, in which rustc cannot elide the lifetime: E0106 "missing lifetime specifier" in
/// the Leaker's crate, at the macro that emits the impl (checked with rustc, stable 1.98.1 and
/// nightly 1.100.0; the same for a plain argument `x: RelRef<u8>`). It is the same limitation as F28's,
/// where the Leaker does not see what a path hides. There is no correct behaviour: whether `RelRef`
/// has a lifetime is in its definition, which is in another crate, and the error is not silent (the
/// Leaker's crate does not compile). The way out is the lifetime named, `RelRef<'a, u8>` with a
/// lifetime of the Leaker, or of the method for a path that is no entry (`'_` cannot be leaked as a
/// whole). The test pins the acceptance, with no generics, and the rejection of `'_`.
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
        let mut leaker = new_leaker();
        leaker.intern_trait(&item, true);
        let referrer = leaker.result().finish().unwrap();
        let ix = entry(&referrer, "RelRef < u8 >");
        assert!(
            referrer.generics(ix).params.is_empty(),
            "`{}`: the entry has the generics `{}`",
            string(&item),
            string(referrer.generics(ix))
        );
    }
    let mut leaker = new_leaker();
    leaker.intern_trait(
        &parse_quote!(
            pub trait Tr2 {
                fn f(&self, x: fn(RelRef<'_, u8>) -> &u8);
            }
        ),
        true,
    );
    let result = leaker.result();
    let err = result.errors();
    assert!(
        err[0].note().unwrap().contains("because of `'_`"),
        "{}",
        err[0].note().unwrap()
    );
}

// ── F30 ──────────────────────────────────────────────────────────────────────

/// F30 (fixed): the outlives bound that the signature implies for `&'x T::Item`, with `T` a
/// generic, is in the context of the entry as the where predicate `T::Item: 'x`, and not as the
/// components `T: 'x`.
///
/// Trigger: `trait Tr<'x, T: Iterator> { fn m(&self, x: Rel<&'x T::Item>); }` (also `<T as
/// WTrA>::A`, a trait with a bound `type A: 'static`, and `where T::Item: 'x` written). It is F16
/// for a generic: the original relies on the well-formedness of its signature, which implies
/// `<T as Iterator>::Item: 'x` and not `T: 'x`. The Leaker recorded `T: 'x` (the context had `<'x,
/// T: Iterator + 'x>`, encoded `(&'x (), T, &'x T)`), which is more than the original has: the
/// impl of the trait for a host type, with the signature `x: <() as Repeater<(&'x (), T, &'x T),
/// ..>>::Type`, needs `T: 'x` where it is well formed, and its environment has `<T as
/// Iterator>::Item: 'x` at best: E0309 "the parameter type `T` may not live long enough" (checked
/// with rustc, stable 1.98.1 and nightly 1.100.0: the Leaker's crate, with the trait and the
/// Repeater's impl, compiles; so does the copy of a method whose lifetime is a lifetime of the
/// method, where the signature gives `T: 'x`, and the copy where `T: 'x` is written). F16
/// chose, for `Self::A`, to record the projection `Self::A: 'x` and no component, "the copy is
/// stricter than the original" otherwise; the same holds for a generic.
///
/// The property test shows it with `TYPE_LEAK_PROP_FEATURES=projection_outlives` (E0309 in the
/// Referrer's crate before the fix, seeds 1, 2 and 3 of 60 cases failed).
///
/// The fix records the predicate `T::Item: 'x` (`<T as Iterator>::Item: 'x` as written, and so for
/// `<T as Tr<'b, U>>::A`, a GAT `T::A<'b>`, and `<T::A as Tr2>::B`) as a where predicate of the
/// context, as F16 does for `Self::A`: the Repeater's impl declares it (`impl<'x, T: Iterator>
/// Repeater<(&'x (), T), ..> for () where T::Item: 'x`), and in the copy it follows from the
/// normalized signature. Checked with rustc (stable 1.98.1 and nightly 1.100.0): the Repeater's
/// impls, the copy of the trait with the signature rewritten, and impls of the trait and of the
/// copy for a host type, with the lifetime of the trait or of the method, all compile. A written
/// `where T::Item: 'x` is the same predicate (one, not two). Where `T` is also mentioned apart from
/// the projection (`&'x (T, T::Item)`), `T: 'x` is recorded as well, as the original requires it.
/// A projection that cannot be written as a where predicate of the context (it names the lifetime
/// of a `for<..>` binder), and an associated type of a type that is no generic (`<Vec<T> as
/// IntoIterator>::Item`, which rustc normalizes to `T`) are mentioned by their parts, as before.
/// So is a projection that a bound of the context equates (`T: Iterator<Item = X>`): rustc
/// normalizes it to `X`, in the signature (which implies what `X` outlives, `T: 'x` for `Item =
/// (Wrap<T>, u8)`) and in the environment of the Repeater's impl, where the predicate `T::Item:
/// 'x` becomes `(Wrap<T>, u8): 'x`, which rustc does not decompose (E0309 in the Leaker's crate,
/// checked, and found by the property test, case 28 of seed 1 with the feature on). So what `X`
/// mentions is recorded (`T: 'x` there, `U: 'x` for `Item = U`, which the encoded `&'x U` gives
/// to the impl of the Repeater and to the host impl alike, and nothing for `Item = u8`), and no
/// predicate.
///
/// What it costs: a struct that holds the copy (`struct S<'x, T: Iterator>(Rel<&'x T::Item>)`) had
/// `T: 'x` inferred through the encoded arguments (stricter than the original), and now writes
/// `where T::Item: 'x`, which rustc infers for the original and not through the projection of the
/// Repeater (E0309 otherwise, checked with rustc).
///
/// The assertions: the context of the entry has the predicate and not `T: 'x`, the encoded
/// arguments have no `&'x T`, and the Referrer replaces the entry in a copy of the trait, which has
/// the signature and not the predicate. The control has `T: 'x` written, which is recorded as
/// before.
#[test]
fn f30_implied_outlives_of_generic_projection_recorded_as_component() {
    let slots = RepeaterSlots::new(parse_quote!(crate::Repeater));
    // The control: `T: 'x` written, and recorded.
    let control: ItemTrait = parse_quote!(
        pub trait Tr<'x, T: Iterator + 'x> {
            fn m(&self, x: Rel<&'x T::Item>);
        }
    );
    let mut leaker = new_leaker();
    leaker.intern_trait(&control, true);
    let referrer = leaker.result().finish().unwrap();
    let ix = entry(&referrer, "Rel < & 'x T :: Item >");
    assert!(generics_say_outlives(referrer.generics(ix), "T", "'x"));
    // The where predicates, and the generics, of the context of the entry that has `needle`.
    let context_of = |referrer: &Referrer, needle: &str| -> (Vec<String>, Generics) {
        let ix = referrer
            .iter()
            .position(|t| string(t).contains(needle))
            .unwrap_or_else(|| panic!("no entry with `{}`", needle));
        let generics = referrer.generics(ix).clone();
        let predicates = generics
            .where_clause
            .iter()
            .flat_map(|w| &w.predicates)
            .map(string)
            .collect();
        (predicates, generics)
    };
    // (the trait, a text of the entry, the where predicates of its context, whether it has `T:
    // 'x`, the encoded arguments).
    let items: Vec<(ItemTrait, &str, &[&str], bool, &str)> = vec![
        (
            parse_quote!(
                pub trait Tr<'x, T: Iterator> {
                    fn m(&self, x: Rel<&'x T::Item>);
                }
            ),
            "Item",
            &["T :: Item : 'x"],
            false,
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
            false,
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
            false,
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
            false,
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
            false,
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
            false,
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
            false,
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
            false,
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
            false,
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
            true,
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
            true,
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
            false,
            "(& 'x () , T)",
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
            true,
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
            true,
            "(& 'x () , T , & 'x T)",
        ),
    ];
    for (item, needle, expected, outlives_t, encoded) in &items {
        let mut leaker = new_leaker();
        let context = leaker.intern_trait(item, true);
        let referrer = leaker
            .result()
            .finish()
            .unwrap_or_else(|e| panic!("`{}` is rejected: {}", string(item), joined_messages(&e)));
        let (predicates, generics) = context_of(&referrer, needle);
        assert_eq!(
            predicates,
            *expected,
            "`{}`: the where predicates of the context (generics `{}`)",
            string(item),
            string(&generics)
        );
        let named = if string(item).contains("I : Iterator") {
            "I"
        } else {
            "T"
        };
        assert_eq!(
            generics_say_outlives(&generics, named, "'x"),
            *outlives_t,
            "`{}`: `{}: 'x` in the context (generics `{}`)",
            string(item),
            named,
            string(&generics)
        );
        let arguments = slots
            .encode_generics_params_to_ty_with_self(&generics.params, &parse_quote!(Self), &[])
            .unwrap();
        assert_eq!(string(&arguments), *encoded, "`{}`", string(item));
        // The Referrer finds the entry in the copy, which has the signature and no predicate.
        let copy = rewrite(&referrer, context, Item::Trait(item.clone()));
        assert!(
            !copy.contains("Rel < &") && !copy.contains("& 'x Rel"),
            "`{}`: not replaced in the copy: {}",
            string(item),
            copy
        );
    }
    // The bound equates the associated type to another generic: that one outlives.
    let item: ItemTrait = parse_quote!(
        pub trait Tr<'x, T: Iterator<Item = U>, U> {
            fn m(&self, x: Rel<&'x T::Item>);
        }
    );
    let mut leaker = new_leaker();
    leaker.intern_trait(&item, true);
    let referrer = leaker.result().finish().unwrap();
    let (predicates, generics) = context_of(&referrer, "Item");
    assert!(predicates.is_empty(), "{:?}", predicates);
    assert!(generics_say_outlives(&generics, "U", "'x"));
    assert!(!generics_say_outlives(&generics, "T", "'x"));
    // An impl, whose `Self` is a type.
    let item: ItemImpl = parse_quote!(
        impl<'x, T: Iterator> ::p::Tr<'x> for W<T> {
            fn m(&self, x: Rel<&'x T::Item>) {}
        }
    );
    let mut leaker = new_leaker();
    leaker.intern_impl(&item, true);
    let referrer = leaker.result().finish().unwrap();
    let (predicates, generics) = context_of(&referrer, "Rel < & 'x T :: Item >");
    assert_eq!(string(&generics), "< 'x , T : Iterator >");
    assert_eq!(predicates, ["T :: Item : 'x"]);
}

// ── F31 ──────────────────────────────────────────────────────────────────────

/// F31 (limitation): a function pointer or `Fn(..)` inside an input has its own signature only for
/// rustc 1.64 and later.
///
/// Trigger:
/// `trait Tr<'a, T: 'a> { fn f(&self, x: Box<dyn Fn(&Box<dyn for<'x> Fn(&'x Rel<T>) -> &'x u8>) -> &u8>); }`
/// (and `fn(&for<'x> fn(&'x Rel<T>) -> &'x u8) -> &u8`). The outer sugar has one lifetime, the `&`;
/// the inner one has its own binder `'x`, and the entry `Rel<T>` in it names `'a` in the copy.
/// F12's rule follows rustc: the lifetimes of an inner signature do not count for the outer one,
/// and the entry is `Rel<T>` alone (the `Leaker` docs, "Elided lifetimes": "That is how rustc 1.64
/// and later count"). Checked with rustc and the impl of the trait for a host type: 1.56, 1.61,
/// 1.62 and 1.63 count the lifetimes that the inner signature does not bind, so they see two for
/// the outer one (`&` and `'a`) and reject the copy (E0106, and on 1.56 "`impl` item signature
/// doesn't match `trait` item signature"); 1.64, 1.65, 1.70, stable (1.98.1) and nightly accept it.
/// So the change is in 1.64 (the docs said 1.65 before F31 measured it, and say 1.64 now; the
/// crate's MSRV is 1.56 and the property test runs on the current toolchain only). The Leaker does
/// not know the toolchain of the crate that expands its macros, so it follows the current one, and
/// an item with a nested signature in an input is rejected or broken on an older rustc. Leaking the
/// outer holder whole (as the Leaker does where the inner signature has no binder,
/// `fn(&fn(&Rel<T>) -> &u8) -> &u8`) would be right on all, at the cost of more entries, and is a
/// decision for the maintainers: it follows no rustc, so this test pins the rule that follows the
/// current ones, and does not run old toolchains.
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
        assert_eq!(entries, ["Rel < T >"], "{}", string(&item));
        let mut signatures = Signatures::default();
        signatures.visit_item_trait(&copy);
        assert_eq!(signatures.count, 2, "{}", string(&copy));
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

/// The parentheses of a type that rustc does not need. rustc needs one around
///
/// - a trait object or `impl Trait` with several bounds that is the pointee of a reference or of a
///   pointer (`&(dyn A + B)`, `*const (dyn A + B)`);
/// - the return type of the sugar of `Fn` that ends in a bare trait object, if another bound follows
///   it, or it has several bounds itself (`Fn() -> (&dyn A) + Clone`, `Fn() -> (dyn A + B)`);
/// - the bounded type of a where predicate that is a reference or a pointer to a function pointer
///   with no output (`(&fn(u8)): 'a`).
///
/// (Checked with rustc: without them it reads "ambiguous `+` in a type" (E0178), or "return types
/// are denoted using `->`".)
#[derive(Default)]
struct NeedlessParens {
    needed: Vec<*const TypeParen>,
    needless: Vec<String>,
}

impl NeedlessParens {
    fn pointee(&mut self, elem: &Type) {
        if let Type::Paren(paren) = elem {
            if matches!(
                &*paren.elem,
                Type::TraitObject(TypeTraitObject { bounds, .. })
                    | Type::ImplTrait(TypeImplTrait { bounds, .. })
                    if bounds.len() > 1
            ) {
                self.needed.push(paren);
            }
        }
    }

    fn bounds(&mut self, bounds: &Punctuated<TypeParamBound, Token![+]>) {
        let count = bounds.len();
        for (ix, bound) in bounds.iter().enumerate() {
            let TypeParamBound::Trait(TraitBound { path, .. }) = bound else {
                continue;
            };
            let Some(PathArguments::Parenthesized(ParenthesizedGenericArguments {
                output: ReturnType::Type(_, output),
                ..
            })) = path.segments.last().map(|seg| &seg.arguments)
            else {
                continue;
            };
            if let Type::Paren(paren) = &**output {
                if matches!(bare_object_bounds(&paren.elem), Some(n) if ix + 1 < count || n > 1) {
                    self.needed.push(paren);
                }
            }
        }
    }
}

impl<'ast> Visit<'ast> for NeedlessParens {
    fn visit_type_reference(&mut self, i: &'ast TypeReference) {
        self.pointee(&i.elem);
        syn::visit::visit_type_reference(self, i)
    }
    fn visit_type_ptr(&mut self, i: &'ast TypePtr) {
        self.pointee(&i.elem);
        syn::visit::visit_type_ptr(self, i)
    }
    fn visit_type_param(&mut self, i: &'ast TypeParam) {
        self.bounds(&i.bounds);
        syn::visit::visit_type_param(self, i)
    }
    fn visit_type_trait_object(&mut self, i: &'ast TypeTraitObject) {
        self.bounds(&i.bounds);
        syn::visit::visit_type_trait_object(self, i)
    }
    fn visit_predicate_type(&mut self, i: &'ast PredicateType) {
        self.bounds(&i.bounds);
        if let Type::Paren(paren) = &i.bounded_ty {
            if bare_fn_under_pointer(&paren.elem) {
                self.needed.push(paren);
            }
        }
        syn::visit::visit_predicate_type(self, i)
    }
    fn visit_type_paren(&mut self, i: &'ast TypeParen) {
        if !self.needed.contains(&(i as *const TypeParen)) {
            self.needless.push(string(i));
        }
        syn::visit::visit_type_paren(self, i)
    }
}

/// The parentheses of `ty` that rustc does not need.
fn needless_parens(ty: &Type) -> Vec<String> {
    let mut parens = NeedlessParens::default();
    parens.visit_type(ty);
    parens.needless
}

/// The parentheses of `generics` that rustc does not need.
fn needless_parens_in(generics: &Generics) -> Vec<String> {
    let mut parens = NeedlessParens::default();
    parens.visit_generics(generics);
    parens.needless
}

/// F32 (fixed): the encoding parenthesized a trait object that needs no parentheses.
///
/// Trigger: the encoding of a `?Sized` parameter whose argument is a trait object with one bound:
/// `encode_generics_args_to_ty()` gave `(*const (dyn Tr),)` for `T: ?Sized` and `dyn Tr`
/// (`unsized_element()`), as it gives `*const (dyn Tr + Send)` for two bounds, where without the
/// parentheses it is E0178 ("ambiguous `+` in a type"). With one bound `*const dyn Tr` parses and
/// means the same, so those parentheses are for nothing, and rustc's `unused_parens` says "unnecessary
/// parentheses around type" for them (checked with `#![deny(unused_parens)]` on stable 1.98.1 and
/// nightly 1.100.0: `fn f(_: *const (dyn A)) {}` is reported, and compiles without them). It shows
/// wherever the code is compiled with the lint on and is not the output of an external macro (code
/// that a build script prints, for instance); I did not check what a proc macro's own tokens do.
///
/// The other parentheses that the library adds are the ones rustc needs, which are the controls:
/// `&'a (dyn A + 'a)` (the encoding of an outlives bound; a lifetime bound is required, so the
/// object has two), `*const (dyn A + Send)`, `&(dyn A + Send)` for `Self` replaced by a trait object,
/// `Fn() -> (&dyn A) + Send` and `Fn() -> (dyn A + Send)` in the bounds of a context, and `(&fn(u8)):
/// 'a` as the bounded type of a predicate. Without them rustc rejects the type (E0178, or "return
/// types are denoted using `->`", checked on stable and nightly). rustc's lint reports most of them
/// all the same (a false positive: `(&dyn A) + Clone`, `(dyn A + Send)` as the output of `Fn`,
/// `(&'a fn(u8)): 'a` and `*const (dyn A + Send)` are reported, `&(dyn A + Send)` is not), and no
/// other spelling avoids them. The bounds of a context are sorted by their text, so a bound of
/// `Fn(..)` that ends in a trait object is the last one, and needs no parentheses, unless another
/// bound sorts after it (`Clone + Fn() -> &dyn A`, but `Fn() -> (&dyn A) + Send`).
///
/// The fix: `unsized_element()` parenthesizes the pointee by the rule of `parenthesize_pointee()`,
/// the one for a reference, which is for a trait object with several bounds. The Leaker and the
/// Referrer both call it, so they print the same tokens (the Leaker's argument is the parameter
/// `T`, a path, which has none). The encoding is not serialized, so `Referrer::FORMAT_VERSION` stays.
///
/// The assertion: each parenthesis of the output is one that rustc needs (`NeedlessParens`).
#[test]
fn f32_parentheses_only_where_rustc_needs_them() {
    let slots = RepeaterSlots::new(parse_quote!(crate::Repeater));
    // The encoding of the arguments of a `T: ?Sized`.
    let params: Punctuated<GenericParam, Token![,]> = parse_quote!(T: ?Sized);
    let encode = |arg: Type| -> Type {
        slots
            .encode_generics_args_to_ty(&[GenericArgument::Type(arg)], &params)
            .unwrap()
    };
    // The parentheses that are needed are there, and are not reported.
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
        (parse_quote!((dyn ::p::A)), "(* const dyn :: p :: A ,)"),
        (
            parse_quote!(dyn for<'a> ::p::B<'a>),
            "(* const dyn for < 'a > :: p :: B < 'a > ,)",
        ),
        (
            parse_quote!(dyn Fn(u8) -> &'static dyn ::p::A),
            "(* const dyn Fn (u8) -> & 'static dyn :: p :: A ,)",
        ),
    ] {
        let encoded = encode(arg);
        assert_eq!(string(&encoded), expected);
        assert_eq!(needless_parens(&encoded), Vec::<String>::new());
    }
    let with_lifetime: Punctuated<GenericParam, Token![,]> = parse_quote!('a, T: ?Sized + 'a);
    let args: Vec<GenericArgument> = vec![
        parse_quote!('x),
        GenericArgument::Type(parse_quote!(dyn ::p::A + 'x)),
    ];
    let encoded = slots
        .encode_generics_args_to_ty(&args, &with_lifetime)
        .unwrap();
    assert!(
        string(&encoded).contains("& 'x (dyn :: p :: A + 'x)"),
        "{}",
        string(&encoded)
    );
    assert_eq!(needless_parens(&encoded), Vec::<String>::new());
    // The bounds of a context.
    let item: ItemStruct = parse_quote!(
        pub struct S<T: Send>(Rel<T>)
        where
            T: Fn(()) -> &'static dyn ::core::any::Any;
    );
    let mut leaker = new_leaker();
    leaker.intern_struct(&item, true);
    let referrer = leaker.result().finish().unwrap();
    let generics = referrer.impl_generics(referrer.context(entry(&referrer, "Rel < T >")));
    assert!(string(&generics).contains("-> (& 'static dyn :: core :: any :: Any) + Send"));
    assert_eq!(needless_parens_in(&generics), Vec::<String>::new());
    // `Self` replaced by a trait object, a function pointer.
    let tr: ItemTrait = parse_quote!(
        pub trait Tr {
            fn p(&self, a: &Self, b: *const Self)
            where
                &'static Self: Clone;
        }
    );
    let mut leaker = new_leaker();
    let context = leaker.intern_trait(&tr, true);
    let referrer = leaker.result().finish().unwrap();
    for self_ty in [
        parse_quote!(dyn ::p::A + Send),
        parse_quote!(dyn ::p::A),
        parse_quote!(fn(u8)),
    ] {
        let self_ty: Type = self_ty;
        let mut copy = Item::Trait(tr.clone());
        referrer
            .clone()
            .visitor(context, |_, ix| parse_quote!(P<#ix>))
            .with_self(self_ty.clone())
            .visit_item_mut(&mut copy);
        let mut parens = NeedlessParens::default();
        parens.visit_item(&copy);
        assert_eq!(
            parens.needless,
            Vec::<String>::new(),
            "Self = `{}`: {}",
            string(&self_ty),
            string(&copy)
        );
    }
    // The Leaker's encoding of the parameter is the Referrer's of the same argument.
    let leaker_side = slots.encode_generics_params_to_ty(&params).unwrap();
    assert_eq!(string(&leaker_side), string(&encode(parse_quote!(T))));
    assert_eq!(string(&leaker_side), "(* const T ,)");
    // A trait object with one bound, and with a lifetime bound (two).
    for arg in [parse_quote!(dyn ::p::A), parse_quote!(dyn ::p::A + 'static)] {
        let arg: Type = arg;
        let encoded = encode(arg.clone());
        let needless = needless_parens(&encoded);
        // `dyn A + 'static` has two bounds, and needs the parentheses.
        let bounds = match &arg {
            Type::TraitObject(object) => object.bounds.len(),
            _ => unreachable!(),
        };
        assert_eq!(
            needless.len(),
            0,
            "`{}` encodes `{}` with parentheses that rustc does not need ({} bound(s))",
            string(&arg),
            string(&encoded),
            bounds
        );
    }
}

// ── F33 ──────────────────────────────────────────────────────────────────────

/// F33: a where predicate of the context of an entry names the `Item` of an iterator, and a
/// predicate of an inner item equates the `Item` to a leaked type.
///
/// Trigger: `trait Tr<T: Iterator> where Vec<T::Item>: Default { fn m(&self) where T: Iterator<Item
/// = Rel<T>>; }` (also `impl<T: Iterator> ::p::W2<T, u8> where Vec<T::Item>: Default { fn m(&self)
/// where T: Iterator<Item = Rel<T>> {} }`). The property test found it with `trait Tr18_1<T: Clone +
/// Debug + Iterator> where R2<[T::Item; 3], B1<T>>: Default { fn m<U>(&self, a: U) where T:
/// Fn(T::Item) -> u8, T: Iterator<Item = (J1<T>, u8)>; }` (seed 12, 60 cases, case 15, item 18, in a
/// run whose random numbers a draw of the generator shifted: the shape is a rare one). The entry
/// `Rel<T>` of the equation has the context `<T: Iterator> where Vec<T::Item>: Default`: a where
/// predicate that names `T` goes with it, and the Repeater's impl is `impl<T: Iterator, __S: ?Sized>
/// Repeater<(T,), ..> for __S where Vec<T::Item>: Default`. In the copy rustc proves that predicate
/// when it normalizes the path `<() as Repeater<(T,), ..>>::Type` of the equation, so it
/// normalizes `<T as Iterator>::Item`, which the equation defines by the path itself: E0275
/// "overflow evaluating the requirement `<() as Repeater<(T,), 0, 0>>::Type == <T as
/// Iterator>::Item`" (nightly says `<T as Iterator>::Item == _`). That is the cycle of F21 and F25,
/// where the Repeater's impl needs a bound that names the `Item` (`T: Fn(T::Item)`); the check
/// looked only at the bounds of a generic and of a predicate on a generic (`T: Fn(T::Item)`, `T:
/// AsRef<T::Item>`). A predicate that is not on a generic (`Vec<T::Item>: Default`, `u8:
/// Into<T::Item>`) is proved by an impl, which does not need the environment, so it was left out;
/// and so was the bounded type of one on a generic (`T::Item: Clone`), which "is proved by an
/// impl" too. Both name the `Item`, and rustc normalizes it with the predicate in scope.
///
/// Checked with rustc, stable 1.98.1 and nightly 1.100.0, with a Leaker's crate (`struct Rel<U>`,
/// the Repeater's impl above, the original item) and a Referrer's crate (the copy: the impl for
/// a type of the crate, the trait for a host type): the trait, the impl and the original item
/// (`Tr18_1`) all give E0275 in the copy, and the Leaker's crate compiles. With `Item = u8`
/// (nothing leaked) rustc compiles the same predicates (`Vec<T::Item>: Default`, `T::Item:
/// Clone`), but not all of them: `Item = F` with `Vec<T::Item>: Clone` is E0277 "the trait bound
/// `F: Clone` is not satisfied", and `Item = u8` with `T::Item: Into<T>` is E0277 "the trait bound
/// `T: From<u8>` is not satisfied", as the environment has the predicate with the projection that
/// rustc has not normalized yet, and the impl needs it normalized (F25). So a predicate that
/// names an `Item` that is defined is unstable whatever the equation says, as a bound is.
///
/// What compiles: a rigid `Item` (nothing in scope defines it: `Rel<[T::Item; 3]>: Default`), a
/// predicate and an equation that are both in the same where clause of a method (the context of
/// `Rel<T>` has the generics of the item, not the predicates of the method), an equation to a type
/// whose entry has no generics (`T: Iterator<Item = Rel<u8>>`), and one that names an entry of
/// another generic (`T: Iterator<Item = Rel<X>>`, whose context is `<X>`). The assertions hold
/// those accepted.
///
/// The price: the Leaker cannot tell that a predicate compiles. `T: Iterator<Item = u8>` with
/// `Vec<T::Item>: Default`, `T: Iterator<Item = F>` with `Vec<T::Item>: Default`, and a generic
/// with a bound of a trait that the Leaker cannot read (`T: ::p::WIter`, a supertrait may define the
/// `Item`) with it compile, and are rejected, as `T: Fn(T::Item)` is in the same place (F21, F25).
///
/// The property test shows it with `TYPE_LEAK_PROP_FEATURES=item_predicate_cycle` (on by default
/// since the fix; before it, three cases of seed 1 with 60 cases failed with E0275 in the Referrer's
/// crate).
///
/// The assertion: if the Leaker accepts the item, no entry that a predicate of the copy names has
/// a bound or predicate of its context that names an `Item`.
///
/// Fixed by looking at the whole predicate (`Projections::equated()`): the bounded type, and the
/// bounds of a predicate that is not on a generic, count if they name a projection that a bound or
/// predicate in scope defines (`T: Iterator<Item = ..>`, `T: ::p::Tr`, `F: Fn(..) -> ..`). `Self::A`
/// and the projection of a type that an impl defines are not counted there: an impl proves those.
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
            entries_naming_item_in_predicates(item),
            ty,
            bound,
            blames,
        );
    }
    for item in &accepted {
        f19_copy_with(f13_leaker(), item).unwrap_or_else(|e| {
            panic!(
                "`{}` is rejected, and it compiles: {}",
                string(item),
                joined_messages(&e)
            )
        });
    }
    // These are rejected (this is the price of F33): all three for the same predicate.
    for item in &price {
        assert_unstable(
            item,
            f19_copy_with(f13_leaker(), item),
            "Rel < T >",
            "Vec < T :: Item > : Default",
            Blames::Projection("T :: Item"),
        );
    }
}

// ── the oracles ──────────────────────────────────────────────────────────────

/// The oracles of F2, F19 to F21, F25, F27 and F33 find the cycle that they look for.
///
/// The Leaker rejects every item of those findings, so no copy of one exists for the oracle to look
/// at, and the oracle runs only for the controls (which it must find nothing in). The copy of a
/// cycle is written by hand here, with the Referrer of an item that the Leaker accepts and that has
/// the entry in a position: the oracle must find the entry in the predicate of the copy, and in no
/// other place. If it found nothing in any copy, the controls would pass whatever the Leaker did.
#[test]
fn the_oracles_of_the_findings_find_a_hand_written_cycle() {
    // F19: `Self` is in the bound of the entry, and the self type is leaked.
    let control: Item = parse_quote!(
        impl S {
            fn m<X>(&self, a0: Rel<X>)
            where
                X: Fn(Self) -> u8,
            {
            }
        }
    );
    let (referrer, _) = f19_copy(&control).unwrap();
    let rel = referrer
        .iter()
        .position(|ty| string(ty) == "Rel < X >")
        .unwrap();
    let self_is_leaked = |referrer: &Referrer, context: ContextId| {
        matches!(
            referrer.context_self(context),
            Some(SelfRequirement::Type(self_ty))
                if referrer.iter().any(|entry| type_contains(self_ty, &string(entry)))
        )
    };
    let copy: Item = parse_quote!(
        impl S {
            fn m<X>(&self)
            where
                X: Fn(Self) -> u8,
                P<#rel>: Default,
            {
            }
        }
    );
    assert_eq!(
        entries_with_self_bound_in_predicates(&referrer, &copy, self_is_leaked),
        ["Rel < X >"]
    );
    // In a position, the same entry is not in a predicate.
    let copy: Item = parse_quote!(
        impl S {
            fn m<X>(&self, a0: P<#rel>)
            where
                X: Fn(Self) -> u8,
            {
            }
        }
    );
    assert!(entries_with_self_bound_in_predicates(&referrer, &copy, self_is_leaked).is_empty());

    // F20: `Self::A` is in the bound of the entry, in a trait.
    let control: Item = parse_quote!(
        pub trait Tr {
            type A0;
            fn m<X>(&self, a0: Rel<X>)
            where
                X: Fn(&Self::A0) -> u8;
        }
    );
    let (referrer, _) = f19_copy(&control).unwrap();
    let rel = referrer
        .iter()
        .position(|ty| string(ty) == "Rel < X >")
        .unwrap();
    let bound = |referrer: &Referrer, context: ContextId| {
        matches!(
            referrer.context_self(context),
            Some(SelfRequirement::Bound(_))
        )
    };
    let copy: Item = parse_quote!(
        pub trait Tr {
            type A0;
            fn m<X>(&self)
            where
                X: Fn(&Self::A0) -> u8,
                P<#rel>: Default;
        }
    );
    assert_eq!(
        entries_with_self_bound_in_predicates(&referrer, &copy, bound),
        ["Rel < X >"]
    );

    // F21, F25 and F33: the bound of the entry names an `Item`. (Not with `T: ::p::WIter`: since F26
    // the context of the entry leaves out `T: Fn(T::Item)` then.)
    let control: Item = parse_quote!(
        pub trait Tr<T: Iterator>
        where
            T: Fn(T::Item),
        {
            fn m(&self, x: Rel<T>);
        }
    );
    let (referrer, _) = f19_copy_with(f13_leaker(), &control).unwrap();
    let rel = referrer
        .iter()
        .position(|ty| string(ty) == "Rel < T >")
        .unwrap();
    let copy: Item = parse_quote!(
        pub trait Tr<T: Iterator>
        where
            T: Fn(T::Item),
        {
            fn m(&self)
            where
                P<#rel>: Default;
        }
    );
    assert_eq!(entries_naming_item(&referrer, &copy), ["Rel < T >"]);
    let copy: Item = parse_quote!(
        pub trait Tr<T: Iterator>
        where
            T: Fn(T::Item),
        {
            fn m(&self, x: P<#rel>);
        }
    );
    assert!(entries_naming_item(&referrer, &copy).is_empty());

    // F27: a bound of a generic and a bound of an `impl Trait` mention replacements.
    let cyclic: Item = parse_quote!(
        pub trait Tr<T: Fn(P<0usize>)> {
            fn m(&self, x: impl Into<P<1usize>>);
        }
    );
    let mut generics = EntryMentions::default();
    generics.visit_item(&cyclic);
    let mut impl_trait = ImplTraitMentions::default();
    impl_trait.visit_item(&cyclic);
    assert_eq!(generics.in_bounds.len(), 1, "{:?}", generics.in_bounds);
    assert_eq!(impl_trait.0.len(), 1, "{:?}", impl_trait.0);
    let control: Item = parse_quote!(
        pub trait Tr<T: Fn(P<0usize>)> {
            fn m(&self, x: Vec<P<1usize>>);
        }
    );
    let mut impl_trait = ImplTraitMentions::default();
    impl_trait.visit_item(&control);
    assert!(impl_trait.0.is_empty(), "{:?}", impl_trait.0);
}

// ── F34 ──────────────────────────────────────────────────────────────────────

/// F34 (fixed): the outlives bound that the signature implies for `&'x P`, with `P` an associated
/// type path of `Self` that is not the bare `Self::A`, is recorded as the components of `P`, which
/// the copy cannot prove.
///
/// Trigger: `trait Tr<'a, T> { type A0; fn m<'x>(&'x self, a: Rel<&'x <Self as Tr<'a, T>>::A0>)
/// where Self: 'x; }`. It is F16 for a path that F16 does not take for `Self::A0`: F16 records the
/// where predicate `Self::A0: 'x` for the bare `Self::A0` (no qualified path, no arguments), and
/// F30 `T::A: 'x` for a generic `T`. These shapes still went to the components of the path:
///
/// - `<Self as Tr<'a, T>>::A0`, qualified, with the arguments of the trait: the context had `'a:
///   'x`, `T: 'x` and `Self: 'x` (encoded `(&'a (), &'x (), T, &'x &'a (), &'x T, &'x Self)`),
/// - `<Self as Tr>::A0`: `Self: 'x`, which is no more than the predicate when the trait has no
///   arguments (it compiles, see below), and
/// - `Self::A<'b>`, a generic associated type: `'b: 'x` and `Self: 'x`,
///
/// and a projection of a projection (`<Self::A0 as Tr2>::B`) or a path with a type argument
/// (`Self::G<T>`, `T: 'x`).
///
/// Checked with rustc (stable 1.98.1 and nightly 1.100.0): a hand-written Leaker's crate (the trait
/// and the Repeater's impl of the entry, with the context that the Leaker recorded) and Referrer's
/// crate (an impl of the trait for a host type, with the signature rewritten).
///
/// - `<Self as Tr<'a, T>>::A0` with `where Self: 'x` on the method (so `'x` is early-bound; with a
///   late-bound method lifetime rustc accepts the copy): the Leaker's crate compiles, and the impl
///   for `Host` is E0491 "in type `&'x &'a ()`, reference has a longer lifetime than the data it
///   references" (stable), E0477 "the type `&'a ()` does not fulfill the required lifetime"
///   (nightly). Each component alone is an error: `&'x T` is E0309 (the parameter type `T` may not
///   live long enough), `&'x &'a ()` the one above; and the Repeater's impl without them is E0309
///   in the Leaker's crate, which is why the Leaker recorded them. The original compiles, also
///   for a generic host (`impl<'a, T, H> Tr<'a, T> for Gen<H>`). With the predicate `where <__S as
///   Tr<'a, T>>::A0: 'x` in the Repeater's impl and no component, both crates compile.
/// - `Self::A<'b>` with `where Self: 'x`: the same (E0491 / E0477 for `&'x &'b ()`), and it
///   compiles with `where (<__S>::A<'b>): 'x`.
/// - `<Self as Tr>::A0`, no arguments: both crates compile as recorded (`Self: 'x` is the whole of
///   the projection's components), and with the predicate too. No failure here.
/// - A trait lifetime (`trait Tr<'x, T> { fn m(&self, a: Rel<&'x <Self as Tr<'x, T>>::A0>); }`) and
///   a generic host (`impl<'x, T, H: Iterator> Tr<'x, T> for Gen<H> { type A0 = H::Item; .. }`):
///   `Self: 'x` is itself too much, as `T: 'x` is: the copy needs `&'x Gen<H>` and is E0309 (`H` may
///   not live long enough), where the original needs only `H::Item: 'x` from its signature. With
///   the predicate and no `&'x Self` in the arguments it compiles. This holds for the bare
///   `Self::A0` that F16 records with `Self: 'x` as well (checked: `trait Tr<'x> { type A0; fn
///   m(&self, a: Rel<&'x Self::A0>); }` for `Gen<H>`).
///
/// Where a bound of the context equates the associated type (F30's `Binding`, here for `Self`):
/// `trait Tr<'x, T> where Self: Iterator<Item = (Vec<T>, u8)> { fn m(&self, a: Rel<&'x Self::Item>); }`.
/// rustc normalizes `Self::Item` to `(Vec<T>, u8)` in the environment of the Repeater's impl, where
/// the predicate `Self::Item: 'x` becomes `(Vec<T>, u8): 'x`, which it does not decompose: E0309
/// for `T` in the Leaker's crate (both toolchains; the recorded `Self::Item: 'x` alone, as F16 did
/// for the bare path). What the value mentions outlives instead (`T: 'x`, encoded `&'x T`), and no
/// predicate: both crates compile (with a host `impl<T> Iterator for HostT<T>`), also with
/// `&'x Self` in the arguments. A supertrait that equates the associated type
/// (`trait Tr<'x, T>: Iterator<Item = (Vec<T>, u8)>`) is not read: the Referrer's copy of an impl
/// has no supertraits, so the context could not be the same. It is E0309 in the Leaker's crate
/// (checked, for `Self::Item` and `<Self as Iterator>::Item`), and writing `T: 'x` on the trait
/// makes it compile (checked): a limitation, pinned below, which the `Leaker` docs, "Outlives
/// bounds", say.
///
/// The fix treats an associated type path of `Self` (the implementor of a trait, `Bound`) like one
/// of a generic (F30, `is_generic_projection()`): the path is a unit, and what `&'x P` requires is
/// `P: 'x`, a where predicate of the context. The projection brings no `Self: 'x` either (F16 kept
/// it for the bare `Self::A0`: where `Self` is mentioned elsewhere in the type, it is). The
/// Referrer computes the same context for the copy (`Context::of()` is the one function), where
/// `Self` is still `Self`: `Referrer::visitor().with_self(..)` replaces it only in what it hands
/// to the closure, so the entry is found with or without it. The Repeater's impl is for `__S`, and
/// `iter_with_self()` writes the predicate for it (`where <__S as Tr<'a, T>>::A0: 'x`,
/// `(<__S>::A<'b>): 'x`).
///
/// The property test shows it with `TYPE_LEAK_PROP_FEATURES=self_projection_outlives`: with the
/// library as it was, the seeds 1 to 4 of 60 cases failed with E0491 (`&'b &'a ()`, `&'a &'c ()`)
/// and E0309 (`T`, `U`) in the Referrer's crate. The generator makes `Rel<&'a Self::A0>`,
/// `Rel<&'a <Self as Tr<'a, T>>::A0>` with a lifetime of the trait, and a method that names a
/// generic associated type `type G<'g>;` as `&'a Self::G<'c>` (the lifetime `'c` is the trait's
/// and no other type names it, since rustc requires bounds on the associated type for each outlives
/// relation that the signature or the trait has with the lifetime it is given). It does not make
/// the where clause that equates the associated type: that one is for this test.
///
/// The fix bumped `Referrer::FORMAT_VERSION` to 7 (the contexts of these entries differ); it is 8 now.
///
/// The assertions, for each trait: the context of the entry has the predicate (and the ones of the
/// context), the outlives of the components that are listed and no other, `Self: 'x` where listed,
/// and the encoded arguments; and the Referrer replaces the entry in a copy, which has the
/// signature and not the predicate, also with `with_self()`.
#[test]
fn f34_implied_outlives_of_self_projection_forms_recorded_as_components() {
    let slots = RepeaterSlots::new(parse_quote!(crate::Repeater));
    // (the trait, a text of the entry, the where predicates of its context, the lifetimes that
    // `Self` outlives, the generics that `: 'x` bounds, the encoded arguments).
    type Case<'a> = (
        ItemTrait,
        &'a str,
        &'a [&'a str],
        &'a [&'a str],
        &'a [&'a str],
        &'a str,
    );
    let items: Vec<Case> = vec![
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
    for (item, needle, expected, self_expected, outlived, encoded) in &items {
        let mut leaker = new_leaker();
        let context = leaker.intern_trait(item, true);
        let referrer = leaker
            .result()
            .finish()
            .unwrap_or_else(|e| panic!("`{}` is rejected: {}", string(item), joined_messages(&e)));
        let ix = referrer
            .iter()
            .position(|t| string(t).contains(needle))
            .unwrap_or_else(|| panic!("`{}`: no entry with `{}`", string(item), needle));
        let generics = referrer.generics(ix);
        let mut predicates: Vec<String> = generics
            .where_clause
            .iter()
            .flat_map(|w| &w.predicates)
            .map(string)
            .collect();
        predicates.sort();
        let mut expected: Vec<String> = expected.iter().map(|p| p.to_string()).collect();
        expected.sort();
        assert_eq!(
            predicates,
            expected,
            "`{}`: the where predicates of the context (generics `{}`)",
            string(item),
            string(generics)
        );
        let self_outlives = referrer.context_self_outlives(referrer.context(ix));
        assert_eq!(
            self_outlives
                .iter()
                .map(|l| l.to_string())
                .collect::<Vec<_>>(),
            self_expected.to_vec(),
            "`{}`: `Self: 'x`",
            string(item)
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
                string(item),
                name,
                string(generics)
            );
        }
        let arguments = slots
            .encode_generics_params_to_ty_with_self(
                &generics.params,
                &parse_quote!(Self),
                self_outlives,
            )
            .unwrap();
        assert_eq!(string(&arguments), *encoded, "`{}`", string(item));
        // The Referrer finds the entry in the copy, which has the signature and no predicate, and
        // in the copy where `Self` is the host.
        let copy = rewrite(&referrer, context, Item::Trait(item.clone()));
        assert!(
            !copy.contains("Rel <"),
            "`{}`: not replaced in the copy: {}",
            string(item),
            copy
        );
        let mut with_self = Item::Trait(item.clone());
        referrer
            .clone()
            .visitor(context, |_, ix| parse_quote!(P<#ix>))
            .with_self(parse_quote!(Host))
            .visit_item_mut(&mut with_self);
        assert!(
            !string(&with_self).contains("Rel <"),
            "`{}`: not replaced in the copy with `Self` = `Host`: {}",
            string(item),
            string(&with_self)
        );
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
    let mut leaker = new_leaker();
    leaker.intern_trait(&control, true);
    let referrer = leaker.result().finish().unwrap();
    let ix = entry(&referrer, "Rel < & 'x < Self as Tr < 'a , T > > :: A0 >");
    let generics = referrer.generics(ix);
    assert!(generics_say_outlives(generics, "T", "'x"));
    assert!(generics_say_outlives(generics, "'a", "'x"));
    assert_eq!(
        referrer
            .context_self_outlives(referrer.context(ix))
            .iter()
            .map(|l| l.to_string())
            .collect::<Vec<_>>(),
        vec!["'x".to_string()]
    );
}

// ── F35 ──────────────────────────────────────────────────────────────────────

/// F35 (fixed): a generic associated type of the trait whose where clause says `Self: 'b` makes
/// `Self::G<'x>` well-formed only if `Self: 'x`, which F34 stopped recording for the projection
/// (it records the predicate `Self::G<'x>: 'x`).
///
/// Trigger (found by the audit of `newer-type`, probe c30): `trait Tr { type G<'b> where Self:
/// 'b; fn m<'x>(&'x self, a: Rel<&'x Self::G<'x>>) -> u8; }`. The original compiles: the
/// signature implies `Self: 'x` through the where clause of `G`. The Repeater's impl of the entry
/// `Rel<&'x Self::G<'x>>`, with the predicate `where (<__S>::G<'x>): 'x` and no `Self: 'x`, is
/// E0309 on `__S`: the well-formedness of `<__S>::G<'x>` in its type needs `__S: 'x`. Before F34
/// the arguments had `&'x __S`, and it compiled.
///
/// Checked with rustc (stable 1.98.1 and nightly 1.100.0), a hand-written Leaker's crate (the trait
/// and the Repeater's impl of the entry) and Referrer's crate (an impl of the trait for a host type
/// `Host`, and for a generic host `Gen<H>`, with the signature rewritten):
///
/// - the original, `Host` and `Gen<H>`: compiles. The impl of the Repeater with the predicate only
///   (`Rp<(&'x (),), 0, 0>`): E0309 on `__S`, twice. With `&'x __S` in the arguments and no
///   predicate (before F34): compiles. With the predicate and `&'x __S` (the fix): both crates
///   compile, for `Host` and for `Gen<H>`.
/// - `<Self as Tr>::G<'x>`: the same. No reference around it (`Rel<Self::G<'x>>`): the impl is
///   E0309 without `&'x __S`, and compiles with it (so the where clause of `G` counts for any
///   projection that names it, not only under a reference).
/// - `type G<'b, U> where U: 'b;` and `fn m<'x, V>(&self, a: Rel<&'x Self::G<'x, V>>)`: the impl is
///   E0309 on `V` without `&'x V`, and compiles with it, for `Host`. The same for the where
///   predicate `Self::A: 'b` (the impl needs the predicate `Self::A: 'x`, E0309 on `<__S as
///   Tr>::A`), which a bound `Self::A: 'b` of `G` brings.
/// - A predicate that is not an outlives (`where Self: Clone`) is not implied: the original is
///   E0277 unless the method writes it too, and the Leaker records it from the method as before.
/// - The copy has to prove what the where clause says: `where T: 'b` (`T` a parameter of the trait,
///   `impl<T> Tr<T> for Host`) and `where 'a: 'b` are E0309 and E0478 in the impl for `Host` where the
///   method does not write `where T: 'x` or `where 'a: 'x` (it compiles where it does), and `where
///   Self: 'b` is E0309 for `Gen<H>` where the receiver is not `&'x self` (it compiles with `where
///   Self: 'x` written). The original compiles in each case, as it normalizes `G<'x>` to a type
///   that does not name them. The Repeater's impl is E0309 without them in any case, so recording
///   what the where clause says is the best the copy can have: the bound written on the method is
///   what makes it compile for another host.
/// - A generic associated type that the Leaker cannot see: that of a supertrait (`trait Tr: Ext {
///   fn m<'x>(&'x self, a: Rel<&'x Self::G<'x>>); }`) or of a generic's trait (`T::G<'x>` with `T:
///   Ext`, `Ext` external): the original compiles, the impl with the predicate only is E0309 on
///   `__S` or on `T`, and it compiles with the bound written on the method (`where Self: 'x`, `where
///   T: 'x`: F34's and F30's rules record it). A guessed `T: 'x` is no better: for `T::G<'x>` with
///   `trait Ext { type G<'b>; }` (no where clause) the original compiles, and the impl for `Host`
///   with `&'x T` in the arguments is E0309 on `T` (stable and nightly), as the predicate alone
///   compiles. So nothing is recorded for them, and the `Leaker` docs say to write the bound.
///
/// The fix: the Leaker reads the where clauses and the bounds of the generic associated types of
/// the trait that it analyzes with `intern_trait()`, and a projection that names one (`Self::G<..>`,
/// `<Self as Tr<..>>::G<..>`) requires the outlives relations of its where clause, with its
/// parameters (and those of the trait) substituted by the arguments of the projection: they are
/// recorded as an outlives of an entry (`Self: 'x`, `T: 'x`, `'a: 'x`, `Self::A: 'x`), as if the
/// method had written them, in addition to the predicate that F34 records. The Referrer computes
/// the same context, so the top-level context of the trait carries those generic associated types
/// (`Referrer::visitor()` visits the signatures of a trait with no trait around them, as
/// `newer-type` does), and `Referrer::FORMAT_VERSION` is 8.
///
/// The property test shows it with `TYPE_LEAK_PROP_FEATURES=gat_where_clause`: a trait with `type
/// G<'g> where Self: 'g;` and a method that names it (the test macro keeps the where clause in the
/// impl for the host).
///
/// The assertions, for each trait: the context of the entry has the where predicates listed, the
/// lifetimes that `Self` outlives, the generics that `: 'x` bounds (those listed and no other), and
/// the encoded arguments; the Referrer replaces the entry in a copy of the trait and in the
/// signatures of its methods alone, also after the Referrer is written as tokens and parsed back.
#[test]
fn f35_gat_where_clause_of_self_projection_not_recorded() {
    let slots = RepeaterSlots::new(parse_quote!(crate::Repeater));
    // (the trait, a text of the entry, the where predicates of its context, the lifetimes that
    // `Self` outlives, the generics that `: 'x` bounds, the encoded arguments).
    type Case<'a> = (
        ItemTrait,
        &'a str,
        &'a [&'a str],
        &'a [&'a str],
        &'a [&'a str],
        &'a str,
    );
    let items: Vec<Case> = vec![
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
    for (item, needle, expected, self_expected, outlived, encoded) in &items {
        let mut leaker = new_leaker();
        let context = leaker.intern_trait(item, true);
        let referrer = leaker
            .result()
            .finish()
            .unwrap_or_else(|e| panic!("`{}` is rejected: {}", string(item), joined_messages(&e)));
        let ix = referrer
            .iter()
            .position(|t| string(t).contains(needle))
            .unwrap_or_else(|| panic!("`{}`: no entry with `{}`", string(item), needle));
        let generics = referrer.generics(ix);
        let mut predicates: Vec<String> = generics
            .where_clause
            .iter()
            .flat_map(|w| &w.predicates)
            .map(string)
            .collect();
        predicates.sort();
        let mut expected: Vec<String> = expected.iter().map(|p| p.to_string()).collect();
        expected.sort();
        assert_eq!(
            predicates,
            expected,
            "`{}`: the where predicates of the context (generics `{}`)",
            string(item),
            string(generics)
        );
        let self_outlives = referrer.context_self_outlives(referrer.context(ix));
        assert_eq!(
            self_outlives
                .iter()
                .map(|l| l.to_string())
                .collect::<Vec<_>>(),
            self_expected.to_vec(),
            "`{}`: `Self: 'x`",
            string(item)
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
                string(item),
                name,
                string(generics)
            );
        }
        let arguments = slots
            .encode_generics_params_to_ty_with_self(
                &generics.params,
                &parse_quote!(Self),
                self_outlives,
            )
            .unwrap();
        assert_eq!(string(&arguments), *encoded, "`{}`", string(item));
        // The Referrer finds the entry in the copy, which has the signature and no predicate, also
        // in the copy where `Self` is the host, and in the signatures alone, after the Referrer is
        // written as tokens and parsed back (the top-level context carries what the copy needs).
        let copy = rewrite(&referrer, context, Item::Trait(item.clone()));
        assert!(
            !copy.contains("Rel <"),
            "`{}`: not replaced in the copy: {}",
            string(item),
            copy
        );
        let mut with_self = Item::Trait(item.clone());
        referrer
            .clone()
            .visitor(context, |_, ix| parse_quote!(P<#ix>))
            .with_self(parse_quote!(Host))
            .visit_item_mut(&mut with_self);
        assert!(
            !string(&with_self).contains("Rel <"),
            "`{}`: not replaced in the copy with `Self` = `Host`: {}",
            string(item),
            string(&with_self)
        );
        let tokens = quote!(#referrer);
        let parsed: Referrer = syn::parse2(tokens.clone()).unwrap_or_else(|e| {
            panic!(
                "`{}`: the Referrer `{}` does not parse: {}",
                string(item),
                tokens,
                e
            )
        });
        assert_eq!(parsed, referrer, "`{}`: parsed back", string(item));
        for trait_item in &item.items {
            if let TraitItem::Fn(method) = trait_item {
                let mut sig = method.sig.clone();
                parsed
                    .clone()
                    .visitor(context, |_, ix| parse_quote!(P<#ix>))
                    .visit_signature_mut(&mut sig);
                assert!(
                    !string(&sig).contains("Rel <"),
                    "`{}`: not replaced in the signature alone: {}",
                    string(item),
                    string(&sig)
                );
            }
        }
    }
}

/// What `Visitor::with_nested_self()` was called with: the `Self` of the nested item, its name and
/// the `mod` items around it, one line for each call.
type NestedCalls = std::rc::Rc<std::cell::RefCell<Vec<String>>>;

/// A visitor for the copy in the scope `context`, with `with_nested_self()`: the ordinary entries
/// become `P<ix>`, and those that depend on the own `Self` of a nested item `N<ix>`. The calls of
/// the closure are recorded.
fn nested_visitor(referrer: &Referrer, context: ContextId) -> (impl VisitMut, NestedCalls) {
    let calls = NestedCalls::default();
    let log = calls.clone();
    let visitor = referrer
        .clone()
        .visitor(context, |_, ix| parse_quote!(P<#ix>))
        .with_nested_self(move |nested, _, ix| {
            let requirement = match &nested.requirement {
                SelfRequirement::Type(ty) => format!("Self = {}", string(ty)),
                SelfRequirement::Bound(path) => format!("Self: {}", string(path)),
                _ => unreachable!(),
            };
            let ident = nested
                .ident
                .as_ref()
                .map_or("-".to_string(), |ident| ident.to_string());
            let modules: Vec<String> = nested.modules.iter().map(|m| m.to_string()).collect();
            log.borrow_mut().push(format!(
                "{} / {} / {}",
                requirement,
                ident,
                modules.join("::")
            ));
            parse_quote!(N<#ix>)
        });
    (visitor, calls)
}

fn file_of(items: Vec<Item>) -> File {
    File {
        shebang: None,
        attrs: Vec::new(),
        items,
    }
}

fn names(declared: &[&str]) -> Vec<Ident> {
    declared
        .iter()
        .map(|name| Ident::new(name, proc_macro2::Span::call_site()))
        .collect()
}

/// Where the nested item is analyzed, and where its copy is.
#[derive(Clone, Copy, Debug)]
enum Nest {
    /// `enter_mod("m", [name], |view| view.visit_item(&item))`, the copy is a `File`.
    EnterMod,
    /// `intern_mod(mod m { item })`, the copy is a `File` of the items of `m`.
    InternMod,
    /// `intern_mod(mod m { mod n { item } })`, the copy is a `File` with `mod n { item }`.
    InternModNested,
    /// `enter_generics(<>, |view| view.visit_item(&item))`, which is not in a module: the copy is
    /// in the body of a method, where an item nested in a scope that is no module is.
    EnterGenerics,
}

const NESTS: [Nest; 4] = [
    Nest::EnterMod,
    Nest::InternMod,
    Nest::InternModNested,
    Nest::EnterGenerics,
];

/// Analyzes `item` nested as `nest` says, rewrites the copy with `nested_visitor()`, and returns
/// the rewritten item, the entries and the calls of `with_nested_self()`.
fn nested_copy(nest: Nest, item: &Item, declared: &[&str]) -> (Item, Referrer, Vec<String>) {
    nested_copy_as(nest, item, declared, false)
}

/// Like `nested_copy()`, for a Referrer that replaces the optional entries too if `strict`
/// (`allow_ambiguous_crate_name(false)`).
fn nested_copy_as(
    nest: Nest,
    item: &Item,
    declared: &[&str],
    strict: bool,
) -> (Item, Referrer, Vec<String>) {
    nested_copy_with(nest, item, declared, strict, true)
}

/// Like `nested_copy_as()`; with `nested` false the visitor has no `with_nested_self()` (the macro
/// gives no host), and every entry goes to the ordinary closure.
fn nested_copy_with(
    nest: Nest,
    item: &Item,
    declared: &[&str],
    strict: bool,
    nested: bool,
) -> (Item, Referrer, Vec<String>) {
    nested_copy_in(new_leaker(), nest, item, declared, strict, nested)
}

/// Like `nested_copy_with()`, analyzed by `leaker`.
fn nested_copy_in(
    leaker: Leaker,
    nest: Nest,
    item: &Item,
    declared: &[&str],
    strict: bool,
    nested: bool,
) -> (Item, Referrer, Vec<String>) {
    nested_copy_recovering(leaker, nest, item, declared, strict, nested, false)
}

/// Like `nested_copy_in()`; with `recover` the errors of the Leaker do not fail the copy:
/// the part that conflicts is copied as written (`LeakerResult::into_referrer()`).
fn nested_copy_recovering(
    mut leaker: Leaker,
    nest: Nest,
    item: &Item,
    declared: &[&str],
    strict: bool,
    nested: bool,
    recover: bool,
) -> (Item, Referrer, Vec<String>) {
    let context = match nest {
        Nest::EnterMod => {
            leaker
                .enter_mod("m", names(declared), |view| view.visit_item(item))
                .0
        }
        Nest::InternMod => leaker.intern_mod(&parse_quote!(mod m { #item })),
        Nest::InternModNested => leaker.intern_mod(&parse_quote!(mod m { mod n { #item } })),
        Nest::EnterGenerics => {
            leaker
                .enter_generics(&Generics::default(), |view| view.visit_item(item))
                .0
        }
    };
    let mut result = leaker.result();
    result.allow_ambiguous_crate_name(!strict);
    let referrer = if recover {
        result.into_referrer()
    } else {
        result.finish().unwrap()
    };
    let (mut visitor, calls): (Box<dyn VisitMut>, NestedCalls) = if nested {
        let (visitor, calls) = nested_visitor(&referrer, context);
        (Box::new(visitor), calls)
    } else {
        let visitor = referrer
            .clone()
            .visitor(context, |_, ix| parse_quote!(P<#ix>));
        (Box::new(visitor), NestedCalls::default())
    };
    let copy = match nest {
        Nest::EnterMod | Nest::InternMod => {
            let mut file = file_of(vec![item.clone()]);
            visitor.visit_file_mut(&mut file);
            file.items.remove(0)
        }
        Nest::InternModNested => {
            let mut file = file_of(vec![parse_quote!(mod n { #item })]);
            visitor.visit_file_mut(&mut file);
            match file.items.remove(0) {
                Item::Mod(ItemMod {
                    content: Some((_, mut items)),
                    ..
                }) => items.remove(0),
                other => panic!("`{}` is no module", string(&other)),
            }
        }
        Nest::EnterGenerics => {
            let mut host: ItemImpl = parse_quote!(impl Host { fn m() { #item } });
            visitor.visit_item_impl_mut(&mut host);
            match host.items.remove(0) {
                ImplItem::Fn(ImplItemFn { mut block, .. }) => match block.stmts.remove(0) {
                    Stmt::Item(item) => item,
                    other => panic!("`{}` is no item", string(&other)),
                },
                other => panic!("`{}` is no method", string(&other)),
            }
        }
    };
    let calls = calls.take();
    (copy, referrer, calls)
}

/// F36: an entry that names the own `Self` of an item nested in a scope (a struct, an enum, a union,
/// a trait or an impl that `View::visit_item()` or `intern_mod()` visits) is recorded, but the
/// Visitor never found it in the copy.
///
/// The Leaker gives the nested item a `Self` of its own (`Rel2<Self>` is the entry of the struct's
/// context, `SelfRequirement::Type(S<T>)`), whatever `allow_self` the scope around has. The Visitor
/// skipped every entry that names `Self` while it was in a nested item (`no_self` of
/// `Referrer::find()`, meant for the `Self` of the scope around, which the nested item hides), so
/// the entry of the item's own `Self` was skipped as well: the copy kept `Rel2<Self>`, and the
/// Referrer's crate could not resolve `Rel2` (E0412, E0425, E0433). The entry that does not name
/// `Self` (`Rel<T>`) was rewritten. The copy cannot write the replacement by hand either, as the
/// Repeater's impl for the nested struct is for the Leaker's `S<T>`, which only the macro knows the
/// path of (`with_self()` does not replace the `Self` of a nested item).
///
/// The fix: the Visitor keeps a layer for each nested struct, enum, union, trait and impl, with the
/// `Self` the Leaker gave it, so it computes the same contexts and finds the entry.
/// `Visitor::with_nested_self(|nested, leaked, ix| ..)` writes its replacement, as it is the macro
/// that knows the path of the nested item in the Leaker's crate: the closure gets the item
/// (`NestedSelf`: what `Self` is, the name and the modules around). Without it the entries are left
/// as before (the next test).
///
/// The test runs a struct, an enum, a union, a trait and an impl through `enter_mod`, `intern_mod`
/// (also a module nested in the module), and `enter_generics` (the item is in a method body of the
/// copy). The other tests of the finding: an outer `Self` that the nested item hides, `allow_self =
/// false` around, and an item inside an item.
///
/// The entries are the types that must be leaked (relative paths). An absolute type that names the
/// `Self` of a struct, an enum or a union is none, and keeps the copy's `Self` (F38).
///
/// The property test shows it with `TYPE_LEAK_PROP_FEATURES=nested_self`, which is on by default
/// (before the fix, seed 1 with 24 cases failed nine times in the Referrer's crate: `cannot find
/// type `R1` in this scope`, `cannot find trait `Tr0_1``, and E0308 where a position of the copy
/// names the copy's own `Self`).
#[test]
fn f36_nested_item_self_entry_not_rewritten() {
    struct Case {
        item: Item,
        declared: &'static [&'static str],
        /// The `Self` the nested item has and its name, in the line recorded for the closure.
        nested: &'static str,
        /// The copy, with `P<ix>` and `N<ix>`: the indices of the entries `Rel < T >` and
        /// `Rel2 < Self >`.
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
            declared: &["S"],
            nested: "Self = S < T > / S",
            expected: |p, n| {
                parse_quote!(
                    struct S<T> {
                        b: P<#p>,
                        c: Vec<N<#n>>,
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
            declared: &["E"],
            nested: "Self = E < T > / E",
            expected: |p, n| {
                parse_quote!(
                    enum E<T> {
                        A(Vec<N<#n>>),
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
            declared: &["U"],
            nested: "Self = U < T > / U",
            expected: |p, n| {
                parse_quote!(
                    union U<T> {
                        a: ::core::mem::ManuallyDrop<Vec<N<#n>>>,
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
            declared: &["Tr"],
            nested: "Self: Tr < T > / Tr",
            expected: |p, n| {
                parse_quote!(
                    trait Tr<T> {
                        fn f(&self, x: P<#p>) -> Vec<N<#n>>;
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
            declared: &[],
            nested: "Self = :: p :: S < T > / -",
            expected: |p, n| {
                parse_quote!(
                    impl<T> ::p::W for ::p::S<T> {
                        fn f(x: P<#p>) -> Vec<N<#n>> {
                            loop {}
                        }
                    }
                )
            },
        },
    ];
    for case in &cases {
        for nest in NESTS {
            let (copy, referrer, calls) = nested_copy(nest, &case.item, case.declared);
            let (p, n) = (
                entry(&referrer, "Rel < T >"),
                entry(&referrer, "Rel2 < Self >"),
            );
            let what = format!("`{}` in {:?}", string(&case.item), nest);
            assert_eq!(string(&copy), string(&(case.expected)(p, n)), "{}", what);
            // The closure is called for the entry that names the own `Self`, and only for it.
            let modules = match nest {
                Nest::InternModNested => "n",
                _ => "",
            };
            assert_eq!(
                calls,
                [format!("{} / {}", case.nested, modules)],
                "{}",
                what
            );
        }
    }
}

/// F36: without `Visitor::with_nested_self()`, the entries that name the own `Self` of a nested item
/// are left as written, and the others are rewritten, as they were before the fix: the visitor does
/// not know what to write for a `Self` that is not its own.
#[test]
fn f36_without_with_nested_self_the_entry_of_the_own_self_is_left() {
    let item: Item = parse_quote!(
        struct S<T> {
            b: Rel<T>,
            c: Vec<Rel2<Self>>,
        }
    );
    let mut leaker = new_leaker();
    let (context, ()) = leaker.enter_mod("m", names(&["S"]), |view| view.visit_item(&item));
    let referrer = leaker.result().finish().unwrap();
    let p = entry(&referrer, "Rel < T >");
    let mut file = file_of(vec![item]);
    referrer
        .visitor(context, |_, ix| parse_quote!(P<#ix>))
        .visit_file_mut(&mut file);
    let expected: Item = parse_quote!(
        struct S<T> {
            b: P<#p>,
            c: Vec<Rel2<Self>>,
        }
    );
    assert_eq!(string(&file.items[0]), string(&expected));
}

/// The index of the entry `Rel2 < Self >` whose `Self` is the type `self_ty`.
fn rel2_of(referrer: &Referrer, self_ty: &str) -> usize {
    let found: Vec<usize> = (0..referrer.iter().count())
        .filter(|&ix| {
            string(&referrer.iter().nth(ix).unwrap()) == "Rel2 < Self >"
                && matches!(referrer.context_self(referrer.context(ix)),
                    Some(SelfRequirement::Type(ty)) if string(ty) == self_ty)
        })
        .collect();
    assert_eq!(found.len(), 1, "the entries of `Self` = {}", self_ty);
    found[0]
}

/// F36: the `Self` of the nested item hides the one around. The entry of the outer `Self` is
/// written by `type_fn`, the one of the nested item's by `nested_fn`, and an item the Leaker did not
/// analyze, which has yet another `Self`, keeps `Rel2<Self>`: no entry has its context.
#[test]
fn f36_the_own_self_hides_the_one_around() {
    let nested: Item = parse_quote!(
        struct V {
            c: Vec<Rel2<Self>>,
        }
    );
    let mut leaker = new_leaker();
    let (_, context) = leaker.enter_mod("m", names(&[]), |view| {
        view.enter_impl(&Generics::default(), &parse_quote!(::p::W), true, |view| {
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
                    c: Vec<N<#inner>>,
                }
                struct Other {
                    c: Vec<Rel2<Self>>,
                }
            }
        }
    );
    assert_eq!(string(&copy), string(&expected));
    assert_eq!(calls.take(), ["Self = V / V / "]);
}

/// F36: a nested item has its own `Self` whatever `allow_self` the scope around has. With
/// `allow_self = false` the scope itself has no `Self` to leak (`Rel2<Self>` is rejected, and an
/// error rolls the scope back), so no entry names it and the Visitor finds none; the item nested
/// in it has one, and its entry is rewritten.
#[test]
fn f36_allow_self_false_around_a_nested_item() {
    let generics: Generics = parse_quote!(<T>);
    let ident = Ident::new("S", proc_macro2::Span::call_site());
    let nested: Item = parse_quote!(
        struct V {
            b: Rel<T>,
            c: Vec<Rel2<Self>>,
        }
    );
    // The scope's own `Self` is denied.
    let mut leaker = new_leaker();
    leaker.enter_type(&ident, &generics, false, |view| {
        view.visit_type(&parse_quote!(Rel3<Self>))
    });
    let result = leaker.result();
    let err = result.errors();
    assert_eq!(joined_messages(err), "use absolute path: 'Rel3 < Self >'");
    // The item nested in it is not.
    let mut leaker = new_leaker();
    let (context, ()) = leaker.enter_type(&ident, &generics, false, |view| {
        view.visit_type(&parse_quote!(Vec<Self>));
        view.visit_item(&nested);
    });
    let referrer = leaker.result().finish().unwrap();
    // `Vec<Self>` is copied as written (the copy's `Self`): no entry names the scope's `Self`.
    assert!(referrer
        .iter()
        .all(|ty| string(ty) != "Self" && string(ty) != "Vec < Self >"));
    let (p, n) = (entry(&referrer, "Rel < T >"), rel2_of(&referrer, "V"));
    let mut copy: ItemImpl = parse_quote!(
        impl<T> Foo for S<T> {
            fn m(x: Vec<Self>) {
                struct V {
                    b: Rel<T>,
                    c: Vec<Rel2<Self>>,
                }
            }
        }
    );
    let (mut visitor, calls) = nested_visitor(&referrer, context);
    visitor.visit_item_impl_mut(&mut copy);
    let expected: ItemImpl = parse_quote!(
        impl<T> Foo for S<T> {
            fn m(x: Vec<Self>) {
                struct V {
                    b: P<#p>,
                    c: Vec<N<#n>>,
                }
            }
        }
    );
    assert_eq!(string(&copy), string(&expected));
    assert_eq!(calls.take(), ["Self = V / V / "]);
}

/// F36: an item inside an item. Two structs with the same name, `S<T>`, one in a module and one in
/// a module of it, and a trait in the inner one. They have the same `Self`, in two modules: the
/// entries are keyed by the module, so each copy gets its own, and the layers do not leak into each
/// other (the module found by name has no `Self`, and the item in it has its own).
#[test]
fn f36_an_item_inside_an_item() {
    let outer: ItemMod = parse_quote!(
        mod a {
            struct S<T> {
                a: Rel<T>,
                b: Vec<Rel2<Self>>,
            }
            mod b {
                struct S<T> {
                    a: Rel<T>,
                    b: Vec<Rel2<Self>>,
                }
                trait Tr {
                    fn f(&self) -> Vec<Rel2<Self>>;
                }
            }
        }
    );
    let mut leaker = new_leaker();
    let context = leaker.intern_mod(&outer);
    let referrer = leaker.result().finish().unwrap();
    // `Rel<T>` in each module, `Rel2<Self>` for each of the three items: five entries.
    assert_eq!(referrer.iter().count(), 5);
    let mut file = file_of(outer.content.clone().unwrap().1);
    let (mut visitor, calls) = nested_visitor(&referrer, context);
    visitor.visit_file_mut(&mut file);
    let text = string(&file);
    assert!(!text.contains("Rel"), "{}", text);
    assert_eq!(text.matches("N <").count(), 3, "{}", text);
    assert_eq!(
        calls.take(),
        [
            "Self = S < T > / S / ",
            "Self = S < T > / S / b",
            "Self: Tr / Tr / b"
        ]
    );
    // The two `Rel2<Self>` of the structs are two entries (one for each module).
    let mut found: Vec<usize> = Vec::new();
    for item in file.items.iter().chain(match &file.items[1] {
        Item::Mod(ItemMod {
            content: Some((_, items)),
            ..
        }) => items.iter(),
        other => panic!("`{}` is no module", string(other)),
    }) {
        if let Item::Struct(ItemStruct { fields, .. }) = item {
            let text = string(fields);
            let ix: usize = text
                .split("N < ")
                .nth(1)
                .and_then(|rest| rest.split("usize").next())
                .and_then(|n| n.trim().parse().ok())
                .unwrap_or_else(|| panic!("no N in {}", text));
            found.push(ix);
        }
    }
    assert_eq!(found.len(), 2);
    assert_ne!(found[0], found[1]);
}

/// F36: an entry that does not name `Self` but has a bound of its generics that does (`Rel<T>` with
/// `T: ::p::Tr<Self>`), one that needs `Self: 'a`, and one in a trait with a generic associated
/// type, have the own `Self` of the nested item in their context too: `nested_fn` writes them.
/// (In a struct, an enum or a union the first is for any type, and the copy passes its own: F37.)
#[test]
fn f36_a_bound_or_an_outlives_that_names_the_own_self() {
    let cases: Vec<(Item, &str, &str)> = vec![
        (
            parse_quote!(
                struct S<T: ::p::Tr<Self>> {
                    b: Rel<T>,
                }
            ),
            "S",
            "struct S<T: ::p::Tr<Self>> { b: N<0usize>, }",
        ),
        (
            parse_quote!(
                trait Tr<T: ::p::Tr<Self>> {
                    fn f(&self, x: Rel<T>);
                }
            ),
            "Tr",
            "trait Tr<T: ::p::Tr<Self>> { fn f(&self, x: N<0usize>); }",
        ),
        (
            parse_quote!(
                struct S<'a, T: 'a> {
                    b: &'a Rel<T>,
                    c: &'a Vec<Rel2<Self>>,
                }
            ),
            "S",
            "struct S<'a, T: 'a> { b: &'a P<0usize>, c: &'a Vec<N<1usize>>, }",
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
            "Tr",
            "trait Tr<'a, T> { type G<'b> where Self: 'b; fn f(&self, x: &'a P<0usize>, y: Self::G<'a>) -> Vec<N<1usize>>; }",
        ),
    ];
    for (item, name, expected) in cases {
        let mut leaker = new_leaker();
        let (context, ()) = leaker.enter_mod("m", names(&[name]), |view| view.visit_item(&item));
        let referrer = leaker.result().finish().unwrap();
        let (mut visitor, calls) = nested_visitor(&referrer, context);
        let mut file = file_of(vec![item.clone()]);
        visitor.visit_file_mut(&mut file);
        let expected: Item = syn::parse_str(expected).unwrap();
        assert_eq!(string(&file.items[0]), string(&expected));
        assert_eq!(calls.take().len(), 1, "`{}`", string(&item));
    }
}

/// Analyzes `item` nested as `nest` says, and returns the result of the Leaker.
fn nested_analysis(nest: Nest, item: &Item, declared: &[&str]) -> LeakerResult {
    nested_analysis_in(new_leaker(), nest, item, declared)
}

/// Like `nested_analysis()`, in `leaker`.
fn nested_analysis_in(
    mut leaker: Leaker,
    nest: Nest,
    item: &Item,
    declared: &[&str],
) -> LeakerResult {
    match nest {
        Nest::EnterMod => {
            leaker.enter_mod("m", names(declared), |view| view.visit_item(item));
        }
        Nest::InternMod => {
            leaker.intern_mod(&parse_quote!(mod m { #item }));
        }
        Nest::InternModNested => {
            leaker.intern_mod(&parse_quote!(mod m { mod n { #item } }));
        }
        Nest::EnterGenerics => {
            leaker.enter_generics(&Generics::default(), |view| view.visit_item(item));
        }
    }
    leaker.result()
}

/// The items of F37 that are supported: a struct, an enum or a union with a generic whose bound
/// names its `Self` (`S<T: AsRef<Self>>`, `where T: AsRef<(Self, u8)>`), and a field that uses
/// the generic in a type that does not name `Self` (`Rel<T>`).
fn f37_supported() -> Vec<(Item, &'static str, &'static str)> {
    vec![
        (
            parse_quote!(
                pub struct S<T: ::core::convert::AsRef<Self>> {
                    b: Rel<T>,
                }
            ),
            "S",
            "Rel < T >",
        ),
        (
            parse_quote!(
                pub struct S<T>(Rel<T>)
                where
                    T: ::core::convert::AsRef<(Self, u8)>;
            ),
            "S",
            "Rel < T >",
        ),
        (
            parse_quote!(
                pub union U<T: ::core::marker::Copy + ::core::convert::AsRef<Self>> {
                    a: ::core::mem::ManuallyDrop<Rel<T>>,
                }
            ),
            "U",
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
            "E0_1",
            "super :: base :: B2 < T , :: core :: result :: Result < u32 , u16 > >",
        ),
    ]
}

/// The items of F37 that stay rejected: the type of the entry names `Self` too.
fn f37_rejected() -> Vec<(Item, &'static str, &'static str)> {
    vec![
        (
            parse_quote!(
                pub struct S<T: ::core::convert::AsRef<Self>> {
                    b: Rel2<Self, T>,
                }
            ),
            "S",
            "Rel2 < Self , T >",
        ),
        (
            parse_quote!(
                pub enum E<T: ::core::convert::AsRef<Self>> {
                    A(Rel<Self>),
                    B(T),
                }
            ),
            "E",
            "Rel < Self >",
        ),
        (
            parse_quote!(
                pub struct S<T>(Vec<Rel2<Self>>, T)
                where
                    T: ::core::convert::AsRef<(Self, u8)>;
            ),
            "S",
            "Rel2 < Self >",
        ),
    ]
}

/// F37 (fixed): a struct, an enum or a union nested in a scope, whose generic has a bound that
/// names its `Self` (`S<T: AsRef<Self>>`, `where T: AsRef<(Self, Rl)>`), and a field that uses the
/// generic (`Rel<T>`), was accepted, and the Referrer's crate did not compile (E0277: `the trait
/// bound `T: AsRef<tlp_l_075::m0::E0_1<'_, T, V>>` is not satisfied`, required for the Leaker's
/// `E0_1` to implement `Repeater<..>`). It was then rejected (`333a4d6`), and is now supported.
///
/// The copy of a nested item is a type of its own (`Self` is the copy's `S<T>`), whereas the
/// Repeater's impl of the entry was for the Leaker's `S<T>` and needed the bound for that type,
/// which the environment of the copy (`T: AsRef<Self>`, for the copy's) did not have. The type that
/// the field uses is the cause, not the path of the bound: `Rl<'a>`, `self::Rl<'a>`, `crate::Rl<'a>`
/// and the others make no difference (`tests/shadowed_names.rs` has the spellings). The item
/// analyzed as itself (`intern_*`, `enter_type`..) was fine: the Visitor writes its `Self` as the
/// Leaker's type (`with_self()`), which has the bound.
///
/// The fix: the `Self` in the bounds of the generics is the `Self` of the item that declares them.
/// The type of the entry does not name it, so it does not depend on which type it is, and the entry
/// is for any type that the bound holds for (`SelfRequirement::Any`): the Repeater's impl is `impl<T:
/// AsRef<__S>, __S: ?Sized> Repeater<(T,), ID, 0> for __S { type Type = Rel<T>; }`, and the copy
/// passes its own type (`<Self as Repeater<(T,), ID, 0>>::Type`), for which it has the bound. The leaked
/// type `Rel<T>` is the Leaker's. The item analyzed as itself (`intern_*`) is as before: its impl
/// is for the Leaker's type.
///
/// The property test shows it with `TYPE_LEAK_PROP_FEATURES=nested_self` (on by default): seed 4,
/// 120 cases, with `path_named_like_generic`, case 75, item 0 (`scope = "mod"`).
#[test]
fn f37_nested_type_with_a_bound_that_names_its_own_self_is_supported() {
    let param: Ident = parse_quote!(__S);
    for (item, name, ty) in &f37_supported() {
        for nest in NESTS {
            let what = format!("{:?} `{}`", nest, string(item));
            let referrer = nested_analysis(nest, item, &[name])
                .finish()
                .unwrap_or_else(|err| panic!("{} is rejected: {}", what, joined_messages(&err)));
            // The entry is for any type that the bound holds for.
            let ix = referrer
                .iter()
                .position(|t| string(t) == *ty)
                .unwrap_or_else(|| panic!("{}: no entry {}", what, ty));
            assert!(referrer.must(ix), "{}", what);
            assert!(
                matches!(
                    referrer.context_self(referrer.context(ix)),
                    Some(SelfRequirement::Any)
                ),
                "{}",
                what
            );
            // Its impl is for a type parameter with no bound of its own (`Self` is a struct, an
            // enum or a union), and the parameter replaces `Self` in the bound. It is `Sized`
            // where a bound needs it (`(Self, u8)`), else `?Sized`: see
            // `f37_the_impl_is_sized_only_where_the_bounds_need_it`.
            let r = referrer.repeater_impl(ix, ImplSelf::Param(&param)).unwrap();
            assert_eq!(r.self_ty, parse_quote!(__S), "{}", what);
            let last = r.generics.params.last().unwrap();
            let needs_sized = string(item).contains("(Self");
            let expected: GenericParam = if needs_sized {
                parse_quote!(__S)
            } else {
                parse_quote!(__S: ?::core::marker::Sized)
            };
            assert_eq!(last, &expected, "{}", what);
            let generics = string(&r.generics);
            assert!(!generics.contains("Self"), "{}: {}", what, generics);
            assert!(generics.contains("__S"), "{}: {}", what, generics);
            assert_eq!(string(&r.ty), *ty, "{}", what);
            // A marker is not what the copy has the bound for.
            let marker: Type = parse_quote!(crate::Marker);
            assert!(
                matches!(
                    referrer.repeater_impl(ix, ImplSelf::Marker(&marker)),
                    Err(EncodeError::NeedsSelf(_))
                ),
                "{}",
                what
            );
            // The serialized form keeps it.
            let again: Referrer = parse2(quote!(#referrer)).unwrap();
            assert_eq!(again, referrer, "{}", what);
            assert!(
                matches!(
                    again.context_self(again.context(ix)),
                    Some(SelfRequirement::Any)
                ),
                "{}",
                what
            );
        }
    }
}

/// F37: the Repeater's impl of an entry that is for any type (`SelfRequirement::Any`) declares its
/// parameter `__S` with `?Sized` unless a bound needs it `Sized`. The copy of a struct whose last
/// field is the path through the impl (`struct S<T: AsRef<Self>> { a: u8, b: <Self as
/// Repeater<..>>::Type }`) is E0275 on rustc 1.56 to 1.66 ("overflow evaluating the requirement
/// `S<T>: Sized`") if the impl needs `__S: Sized`: the type needs the impl, which needs the type to
/// be `Sized`. (In an enum too on 1.56.) The impl is well-formed without `Sized` if every `__S` of
/// the bounds is a whole type argument of a std trait or type with a `?Sized` parameter
/// (`AsRef<__S>`, `Box<__S>`), or behind a reference or a pointer. Any other use may need `Sized`
/// (a tuple element, a slice, a type or trait of the user, an associated type), and keeps it: on
/// rustc before 1.70, an item with such a bound and the entry in its last field is still E0275, and
/// the ways out are the entry in another field, `where Self: Sized` on the item, or an absolute
/// path for the type of the entry (copied as written, by a Referrer that trusts the names of the
/// crates). The cross-crate test has these shapes (9111, 9125 to 9128); the bug itself shows
/// on a rustc before 1.70 only.
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
            for nest in NESTS {
                let referrer = nested_analysis(nest, item, &["S"])
                    .finish()
                    .unwrap_or_else(|err| {
                        panic!(
                            "{:?} `{}` is rejected: {}",
                            nest,
                            string(item),
                            joined_messages(&err)
                        )
                    });
                let ix = referrer
                    .iter()
                    .position(|t| string(t) == "Rel < T >")
                    .unwrap_or_else(|| panic!("{:?} `{}`: no entry", nest, string(item)));
                assert!(
                    matches!(
                        referrer.context_self(referrer.context(ix)),
                        Some(SelfRequirement::Any)
                    ),
                    "{:?} `{}`",
                    nest,
                    string(item)
                );
                let r = referrer.repeater_impl(ix, ImplSelf::Param(&param)).unwrap();
                assert_eq!(
                    r.generics.params.last().unwrap(),
                    expected,
                    "{} expected, {:?} `{}`: {}",
                    what,
                    nest,
                    string(item),
                    string(&r.generics)
                );
            }
        }
    }
    // The impl as a whole.
    let item: Item = parse_quote!(
        pub struct S<T: ::core::convert::AsRef<Self>> {
            a: u8,
            b: Rel<T>,
        }
    );
    let referrer = nested_analysis(Nest::EnterMod, &item, &["S"])
        .finish()
        .unwrap();
    let r = referrer.repeater_impl(0, ImplSelf::Param(&param)).unwrap();
    let (impl_generics, _, where_clause) = r.generics.split_for_impl();
    let (args, self_ty, ty) = (&r.args, &r.self_ty, &r.ty);
    let imp: ItemImpl = parse_quote! {
        impl #impl_generics crate::Repeater<#args, 7u64, 0usize> for #self_ty #where_clause {
            type Type = #ty;
        }
    };
    let expected: ItemImpl = parse_quote! {
        impl<T: ::core::convert::AsRef<__S>, __S: ?::core::marker::Sized>
            crate::Repeater<(T,), 7u64, 0usize> for __S
        {
            type Type = Rel<T>;
        }
    };
    assert_eq!(imp, expected);
}

/// F37: the copy of such an item. The Visitor finds the entry in the copy of the nested item and
/// hands it to `with_nested_self()`, whose closure tells by `context_self()` that `Self` is any
/// type (`SelfRequirement::Any`) and writes the path with the copy's own type, `Self`, as the
/// implementor. The entries that use no generic with such a bound go to the ordinary closure.
#[test]
fn f37_the_copy_passes_its_own_type() {
    let item: Item = parse_quote!(
        pub struct S<T: ::core::convert::AsRef<Self>, U> {
            b: Rel<T>,
            c: Wrap<U>,
            d: Pair<Rel, U>,
        }
    );
    // In every nest: `Rel<T>` depends on the `Self` of `S`, the others do not.
    for nest in NESTS {
        let (copy, referrer, calls) = nested_copy(nest, &item, &["S"]);
        let (b, c, d) = (
            entry(&referrer, "Rel < T >"),
            entry(&referrer, "Wrap < U >"),
            entry(&referrer, "Pair < Rel , U >"),
        );
        let expected: Item = parse_quote!(
            pub struct S<T: ::core::convert::AsRef<Self>, U> {
                b: N<#b>,
                c: P<#c>,
                d: P<#d>,
            }
        );
        assert_eq!(string(&copy), string(&expected), "{:?}", nest);
        assert_eq!(calls.len(), 1, "{:?}: {:?}", nest, calls);
    }
    // The closure of a macro writes the path through the Repeater, with `Self`.
    let mut leaker = new_leaker();
    let (context, ()) = leaker.enter_mod("m", names(&["S"]), |view| view.visit_item(&item));
    let referrer = leaker.result().finish().unwrap();
    let mut copy = file_of(vec![item.clone()]);
    referrer
        .clone()
        .visitor(
            context,
            |_, ix| parse_quote!(<() as Repeater<(), 7u64, #ix>>::Type),
        )
        .with_nested_self(|_, _, ix| {
            let self_ty: Type = match referrer.context_self(referrer.context(ix)) {
                Some(SelfRequirement::Any) => parse_quote!(Self),
                other => panic!("{:?}", other),
            };
            let args = referrer
                .entry_args(ix, &UseSite::identity().with_self(self_ty.clone()))
                .unwrap();
            parse_quote!(<#self_ty as Repeater<#args, 7u64, #ix>>::Type)
        })
        .visit_file_mut(&mut copy);
    let text = string(&copy.items[0]);
    assert!(
        text.contains("b : < Self as Repeater < (T ,) , 7u64 , 0usize > > :: Type"),
        "{}",
        text
    );
    let c = entry(&referrer, "Wrap < U >");
    assert!(
        text.contains(&format!("c : < () as Repeater < () , 7u64 , {}usize", c)),
        "{}",
        text
    );
}

/// F37: the copy cannot write `Self` in the default of a parameter (E0735). A default whose type
/// does not name `Self` is fine whatever the bounds of the generics it uses say (F41: the entry has
/// no `Self` and none of those bounds). One that names `Self` too, with a bound that does, is
/// rejected, with the reason: the original does not compile either (E0735), so the type cannot be in
/// a default, also with `LeakerConfig::nested_self_entries()`. A default that does not use the
/// generic is as ever.
#[test]
fn f37_a_default_that_names_self_with_a_bound_that_names_it_is_rejected() {
    let item: Item = parse_quote!(
        pub struct S<T: ::core::convert::AsRef<Self>, U = Rel2<Self, T>>(pub T, pub U);
    );
    let mut config = LeakerConfig::new();
    config.allow_primitive().allow_prelude();
    let mut with_option = config.clone();
    with_option.nested_self_entries();
    for config in [config, with_option] {
        for nest in NESTS {
            let result =
                nested_analysis_in(Leaker::from_config(config.clone()), nest, &item, &["S"]);
            let err = assert_rejected(
                &item,
                result.finish(),
                &["`Rel2 < Self , T >` is in the default of a parameter, and it and a bound of its generics name `Self`"],
            );
            assert!(
                all_notes(&err)[0].contains("cannot write in the default of a parameter (E0735)"),
                "{:?}",
                all_notes(&err)
            );
            assert_eq!(all_helps(&err), ["write the default without `Self`"]);
        }
    }
    // The default does not use the generic: nothing depends on the `Self` of the bound.
    let item: Item = parse_quote!(
        pub struct S<T: ::core::convert::AsRef<Self>, U = Rel>(pub T, pub Wrap<U>);
    );
    for nest in NESTS {
        let referrer = nested_analysis(nest, &item, &["S"])
            .finish()
            .unwrap_or_else(|e| panic!("{:?}: {}", nest, joined_messages(&e)));
        assert_eq!(referrer.iter().count(), 2, "{:?}", nest);
        assert!(
            (0..2).all(|ix| referrer.context_self(referrer.context(ix)).is_none()),
            "{:?}",
            nest
        );
    }
}

/// F37: an entry whose type names `Self` too stays rejected, with the reason. Its impl cannot be for
/// any type: it is for the Leaker's (what `Self` is in the type is what the Leaker saw, and what
/// the type needs of it to be well-formed is not known), and the bound that the copy proves is for
/// its own type. As the item itself (`intern_*`), `Self` is written as the Leaker's type, which
/// has the bound: accepted. Written with an absolute path (the help says so) the type is no
/// entry, and the copy has it as written, with its own `Self`: accepted (F38). The help names the
/// way out that takes the risk of the impl for any type: `LeakerConfig::nested_self_entries()`
/// (F42).
#[test]
fn f37_a_type_that_names_self_with_a_bound_that_names_it_is_rejected() {
    const HEADLINE: &str = "has a generic whose bound mentions `Self`, which cannot be leaked";
    for (item, name, ty) in &f37_rejected() {
        for nest in NESTS {
            let result = nested_analysis(nest, item, &[name]).finish();
            let err = assert_rejected(item, result, &[&format!("`{}` {}", ty, HEADLINE)]);
            let notes = all_notes(&err);
            assert_eq!(
                notes[0],
                "it names the `Self` of an item nested in the analyzed syntax, and so do the \
                 bounds of its generics: the Repeater's impl of this type is for the Leaker's \
                 type, but the copy declares a type of its own, which has the bound for itself \
                 (E0277)",
                "{:?} `{}`",
                nest,
                string(item)
            );
            let helps = all_helps(&err);
            assert_eq!(helps.len(), 1, "{:?} `{}`", nest, string(item));
            // The ways out: the F38 fix (an absolute path) is one.
            assert!(
                helps[0].contains(
                    "without `Self`, write the type without `Self`, write the type with an \
                     absolute path (it is copied as written, and its `Self` is the copy's own), \
                     analyze the item with its own `intern_*` call, which writes its `Self` as \
                     the Leaker's type, or accept the entry for any type with \
                     `LeakerConfig::nested_self_entries()`"
                ),
                "{}",
                helps[0]
            );
        }
    }
    // As the item itself, `Self` is written as the Leaker's type, and the Repeater's impl has the
    // bound for it: accepted, with the entry, whose impl is for that type.
    for (item, _, _) in f37_rejected().iter().chain(&f37_supported()) {
        let mut leaker = new_leaker();
        match item {
            Item::Struct(item) => leaker.intern_struct(item, true),
            Item::Enum(item) => leaker.intern_enum(item, true),
            Item::Union(item) => leaker.intern_union(item, true),
            _ => unreachable!(),
        };
        let referrer = leaker
            .result()
            .finish()
            .unwrap_or_else(|e| panic!("`{}` is rejected: {}", string(item), joined_messages(&e)));
        assert!(referrer.required().count() > 0, "`{}`", string(item));
        for ix in 0..referrer.iter().count() {
            assert!(
                !matches!(
                    referrer.context_self(referrer.context(ix)),
                    Some(SelfRequirement::Any)
                ),
                "`{}`: the item itself has the Leaker's type for `Self`",
                string(item)
            );
        }
    }
}

/// F37, the controls: a nested struct, enum or union that names `Self` in no bound that the
/// environment proves, and the other nested items, are accepted in every nest.
#[test]
fn f37_nested_items_that_compile_stay_accepted() {
    let accepted: Vec<(Item, &str)> = vec![
        // `Self` in a field, no bound that names it.
        (
            parse_quote!(
                pub struct S<T: Clone>(Rel<Self>, T);
            ),
            "S",
        ),
        // A bound on `Self`, or on a type that an impl proves.
        (
            parse_quote!(
                pub struct S<T>(Rel<Self>, T)
                where
                    Self: Sized;
            ),
            "S",
        ),
        (
            parse_quote!(
                pub struct S<T>(Rel<T>, Vec<Self>)
                where
                    Vec<Self>: Clone;
            ),
            "S",
        ),
        // The generic whose bound names `Self` is not used by the entry.
        (
            parse_quote!(
                pub struct S<T: ::core::convert::AsRef<Self>, U>(Rel<U>, T);
            ),
            "S",
        ),
        // `Self: 'a` is encoded by the Repeater's impl of the Leaker's type.
        (
            parse_quote!(
                pub struct S<'a, T: 'a>(&'a Rel<Self>, &'a T);
            ),
            "S",
        ),
        // A trait: its `Self` is the implementor, which the copy's impl for a host writes.
        (
            parse_quote!(
                pub trait Tr<T: ::core::convert::AsRef<Self>> {
                    fn f(&self) -> Rel<T>;
                }
            ),
            "Tr",
        ),
    ];
    for (item, name) in &accepted {
        for nest in NESTS {
            let referrer = nested_analysis(nest, item, &[name])
                .finish()
                .unwrap_or_else(|err| {
                    panic!(
                        "{:?} `{}` is rejected: {}",
                        nest,
                        string(item),
                        joined_messages(&err)
                    )
                });
            assert!(referrer.iter().count() > 0, "`{}`", string(item));
        }
    }
}

// ── F38 ──────────────────────────────────────────────────────────────────────

/// The entries of `referrer` as text without spaces, `?` first for an optional one.
fn f38_entries(referrer: &Referrer) -> Vec<String> {
    (0..referrer.iter().count())
        .map(|ix| {
            let ty = string(referrer.iter().nth(ix).unwrap()).replace(' ', "");
            format!("{}{}", if referrer.must(ix) { "" } else { "?" }, ty)
        })
        .collect()
}

/// F38 (fixed): an absolute type that names the own `Self` of a struct, an enum or a union nested in
/// a scope (`View::visit_item()`, the items of `intern_mod()`) was recorded as an optional entry
/// (`SelfRequirement::Type`, `must` is `false`), and the copy of a Referrer that replaces the
/// optional entries (`allow_ambiguous_crate_name(false)`, a strict one) was not the copy of one
/// that does not.
///
/// The type is `::std::boxed::Box<Self>`, `::p::Rel2<Self, T>`: it resolves the same in the copy
/// (`::std` and `::p` are crates, which only the choice of the Referrer makes optional), and its
/// only link to the item is `Self`. The copy declares a type of its own, so `Self` in a type
/// copied as written is that type. A trusting Referrer left the type as written (the copy's
/// `Self`), and a strict one replaced it with `<Leaker's S as Repeater<..>>::Type`, which holds the
/// Leaker's `S` for `Self`: `struct Node { next: ::std::boxed::Box<Self> }` was not recursive in the
/// strict copy (`Box<leaker::Node>`), and `::p::Rel2<Self, T>` meant two different types in the
/// same item according to a bound that names `Self` (with `T: AsRef<Self>` the Leaker records no
/// entry, as it cannot make an impl that is for the Leaker's type: the bound holds for the copy's).
///
/// The fix: such a type is no entry. It is copied as written by both, with `Self` the copy's own,
/// with or without the bound. Only the types that must be leaked use the type that the macro
/// writes for `Self` (`with_nested_self()`, the Leaker's by convention): the relative ones
/// (`Rel<Self>`, `inner::W<Self>`, `crate::W<Self>`). What in an absolute type must be leaked on
/// its own still is (`Rel<T>` in `::p::Rel2<Self, Rel<T>>`), and so is an absolute type that does
/// not name `Self` (`::c::W<T>`, an optional entry). The result is also the way out that the
/// error of F37 suggests for `Rel2<Self, T>` with a bound that names `Self`: write the path
/// absolute.
///
/// The type is no entry as a whole either where the graph finds one entry for it cheaper than
/// one for each of its parts (`::c::Both<::d::X, ::d::Y, Self>`, `::c::Both<Rel<u8>, Rel3<u8>,
/// Self>`): the graph used to pick it, with the Leaker's `Self`, depending on the number of the
/// parts. A reference with a lifetime is leaked whole with what it holds, as ever.
///
/// The property test shows it with `TYPE_LEAK_PROP_FEATURES=nested_self` (on by default), for the
/// items invoked as `strict`: the macro checks that no optional entry of an item nested in a
/// module names its own `Self` (case 11 of seed 4 with 60 cases, case 40 of seed 16, ... had
/// `Result<BTreeMap<fn(i32) -> Self, ..>, Box<dyn Fn(..)>>` as the optional entry, with two
/// parts that are optional entries too).
#[test]
fn f38_an_absolute_type_that_names_the_own_self_is_copied_as_written() {
    struct Case {
        item: Item,
        name: &'static str,
        /// The entries: `?` for an optional one.
        entries: &'static [&'static str],
        /// The copy, as a trusting and as a strict Referrer write it (`P<ix>` for an entry that is
        /// found by `type_fn`, `N<ix>` by `with_nested_self()`).
        expected: fn(&Referrer, bool) -> Item,
    }
    let cases = vec![
        // The report: a recursive type through a box.
        Case {
            item: parse_quote!(
                pub struct Node {
                    next: ::std::boxed::Box<Self>,
                }
            ),
            name: "Node",
            entries: &[],
            expected: |_, _| {
                parse_quote!(
                    pub struct Node {
                        next: ::std::boxed::Box<Self>,
                    }
                )
            },
        },
        // `::c::W<T>` does not name `Self`: an optional entry, as before.
        Case {
            item: parse_quote!(
                pub struct Node<T> {
                    a: ::c::W<T>,
                    b: ::c::Rel2<Self, T>,
                    c: Vec<::c::Rel2<Self, T>>,
                }
            ),
            name: "Node",
            entries: &["?::c::W<T>"],
            expected: |referrer, strict| {
                let a = entry(referrer, ":: c :: W < T >");
                if strict {
                    parse_quote!(
                        pub struct Node<T> {
                            a: P<#a>,
                            b: ::c::Rel2<Self, T>,
                            c: Vec<::c::Rel2<Self, T>>,
                        }
                    )
                } else {
                    parse_quote!(
                        pub struct Node<T> {
                            a: ::c::W<T>,
                            b: ::c::Rel2<Self, T>,
                            c: Vec<::c::Rel2<Self, T>>,
                        }
                    )
                }
            },
        },
        // The same with a bound that names `Self`: as before, no entry (and no change).
        Case {
            item: parse_quote!(
                pub struct S<T: ::core::convert::AsRef<Self>> {
                    b: ::c::Rel2<Self, T>,
                }
            ),
            name: "S",
            entries: &[],
            expected: |_, _| {
                parse_quote!(
                    pub struct S<T: ::core::convert::AsRef<Self>> {
                        b: ::c::Rel2<Self, T>,
                    }
                )
            },
        },
        // Without it.
        Case {
            item: parse_quote!(
                pub struct S<T> {
                    b: ::c::Rel2<Self, T>,
                }
            ),
            name: "S",
            entries: &[],
            expected: |_, _| {
                parse_quote!(
                    pub struct S<T> {
                        b: ::c::Rel2<Self, T>,
                    }
                )
            },
        },
        // The entries that do not name `Self` are as before: `Rel<T>` is required, and `::c::W<T>`
        // is optional; with the bound, both are for any type that it holds for (`Any`), and the
        // copy passes its own.
        Case {
            item: parse_quote!(
                pub struct S<T: ::core::convert::AsRef<Self>> {
                    a: ::c::Rel2<Self, T>,
                    b: Rel<T>,
                    c: ::c::W<T>,
                }
            ),
            name: "S",
            entries: &["Rel<T>", "?::c::W<T>"],
            expected: |referrer, strict| {
                let (b, c) = (
                    entry(referrer, "Rel < T >"),
                    entry(referrer, ":: c :: W < T >"),
                );
                if strict {
                    parse_quote!(
                        pub struct S<T: ::core::convert::AsRef<Self>> {
                            a: ::c::Rel2<Self, T>,
                            b: N<#b>,
                            c: N<#c>,
                        }
                    )
                } else {
                    parse_quote!(
                        pub struct S<T: ::core::convert::AsRef<Self>> {
                            a: ::c::Rel2<Self, T>,
                            b: N<#b>,
                            c: ::c::W<T>,
                        }
                    )
                }
            },
        },
        Case {
            item: parse_quote!(
                pub struct S<T> {
                    a: ::c::Rel2<Self, T>,
                    b: Rel<T>,
                    c: ::c::W<T>,
                }
            ),
            name: "S",
            entries: &["Rel<T>", "?::c::W<T>"],
            expected: |referrer, strict| {
                let (b, c) = (
                    entry(referrer, "Rel < T >"),
                    entry(referrer, ":: c :: W < T >"),
                );
                if strict {
                    parse_quote!(
                        pub struct S<T> {
                            a: ::c::Rel2<Self, T>,
                            b: P<#b>,
                            c: P<#c>,
                        }
                    )
                } else {
                    parse_quote!(
                        pub struct S<T> {
                            a: ::c::Rel2<Self, T>,
                            b: P<#b>,
                            c: ::c::W<T>,
                        }
                    )
                }
            },
        },
        // A part that must be leaked is an entry, in the type that is copied as written.
        Case {
            item: parse_quote!(
                pub struct S<T> {
                    b: ::c::Rel2<Self, Rel<T>>,
                }
            ),
            name: "S",
            entries: &["Rel<T>"],
            expected: |referrer, _| {
                let b = entry(referrer, "Rel < T >");
                parse_quote!(
                    pub struct S<T> {
                        b: ::c::Rel2<Self, P<#b>>,
                    }
                )
            },
        },
        // The graph used to pick the whole type as the entry when it saves entries (one for it, two
        // for its parts), and with it `Self` of the Leaker. The parts are the entries now, in the
        // optional ones and in the required ones.
        Case {
            item: parse_quote!(
                pub struct S {
                    f: ::c::Both<::d::X, ::d::Y, Self>,
                }
            ),
            name: "S",
            entries: &["?::d::X", "?::d::Y"],
            expected: |referrer, strict| {
                let (x, y) = (entry(referrer, ":: d :: X"), entry(referrer, ":: d :: Y"));
                if strict {
                    parse_quote!(
                        pub struct S {
                            f: ::c::Both<P<#x>, P<#y>, Self>,
                        }
                    )
                } else {
                    parse_quote!(
                        pub struct S {
                            f: ::c::Both<::d::X, ::d::Y, Self>,
                        }
                    )
                }
            },
        },
        Case {
            item: parse_quote!(
                pub struct S {
                    f: ::c::Both<Rel<u8>, Rel3<u8>, Self>,
                }
            ),
            name: "S",
            entries: &["Rel<u8>", "Rel3<u8>"],
            expected: |referrer, _| {
                let (rel, rel3) = (
                    entry(referrer, "Rel < u8 >"),
                    entry(referrer, "Rel3 < u8 >"),
                );
                parse_quote!(
                    pub struct S {
                        f: ::c::Both<P<#rel>, P<#rel3>, Self>,
                    }
                )
            },
        },
        // A reference with a lifetime is leaked whole with what it holds, so it is the entry, and
        // `Self` in it is the type that the macro writes, strict or not.
        Case {
            item: parse_quote!(
                pub struct S {
                    f: &'static ::c::Both<Rel<u8>, Rel3<u8>, Self>,
                }
            ),
            name: "S",
            entries: &["&'static::c::Both<Rel<u8>,Rel3<u8>,Self>"],
            expected: |referrer, _| {
                let f = entry(
                    referrer,
                    "& 'static :: c :: Both < Rel < u8 > , Rel3 < u8 > , Self >",
                );
                parse_quote!(
                    pub struct S {
                        f: N<#f>,
                    }
                )
            },
        },
        // An enum and a union.
        Case {
            item: parse_quote!(
                pub enum E<T> {
                    A(::c::W<Self>),
                    B(::c::W<T>),
                }
            ),
            name: "E",
            entries: &["?::c::W<T>"],
            expected: |referrer, strict| {
                let b = entry(referrer, ":: c :: W < T >");
                if strict {
                    parse_quote!(
                        pub enum E<T> {
                            A(::c::W<Self>),
                            B(P<#b>),
                        }
                    )
                } else {
                    parse_quote!(
                        pub enum E<T> {
                            A(::c::W<Self>),
                            B(::c::W<T>),
                        }
                    )
                }
            },
        },
        Case {
            item: parse_quote!(
                pub union U<T> {
                    a: ::core::mem::ManuallyDrop<::c::W<Self>>,
                    b: ::core::mem::ManuallyDrop<::c::W<T>>,
                }
            ),
            name: "U",
            entries: &["?::c::W<T>"],
            expected: |referrer, strict| {
                let b = entry(referrer, ":: c :: W < T >");
                if strict {
                    parse_quote!(
                        pub union U<T> {
                            a: ::core::mem::ManuallyDrop<::c::W<Self>>,
                            b: ::core::mem::ManuallyDrop<P<#b>>,
                        }
                    )
                } else {
                    parse_quote!(
                        pub union U<T> {
                            a: ::core::mem::ManuallyDrop<::c::W<Self>>,
                            b: ::core::mem::ManuallyDrop<::c::W<T>>,
                        }
                    )
                }
            },
        },
        // The ways out that F37 suggests for the relative types that name `Self` and a bound that
        // names it: the absolute path is accepted, and copied as written.
        Case {
            item: parse_quote!(
                pub enum E<T: ::core::convert::AsRef<Self>> {
                    A(::c::Rel<Self>),
                    B(T),
                }
            ),
            name: "E",
            entries: &[],
            expected: |_, _| {
                parse_quote!(
                    pub enum E<T: ::core::convert::AsRef<Self>> {
                        A(::c::Rel<Self>),
                        B(T),
                    }
                )
            },
        },
        Case {
            item: parse_quote!(
                pub struct S<T>(Vec<::c::Rel2<Self>>, T)
                where
                    T: ::core::convert::AsRef<(Self, u8)>;
            ),
            name: "S",
            entries: &[],
            expected: |_, _| {
                parse_quote!(
                    pub struct S<T>(Vec<::c::Rel2<Self>>, T)
                    where
                        T: ::core::convert::AsRef<(Self, u8)>;
                )
            },
        },
    ];
    for case in &cases {
        for nest in NESTS {
            let what = format!("`{}` in {:?}", string(&case.item), nest);
            let (trusting, referrer, _) = nested_copy_as(nest, &case.item, &[case.name], false);
            let (strict, _, _) = nested_copy_as(nest, &case.item, &[case.name], true);
            let expected: Vec<String> = case.entries.iter().map(|e| e.replace(' ', "")).collect();
            assert_eq!(f38_entries(&referrer), expected, "{}", what);
            assert_eq!(
                string(&trusting),
                string(&(case.expected)(&referrer, false)),
                "trusting: {}",
                what
            );
            assert_eq!(
                string(&strict),
                string(&(case.expected)(&referrer, true)),
                "strict: {}",
                what
            );
        }
    }
}

/// F38: `::p::Rel2<Self, T>` is the same with and without a bound that names `Self`: the Leaker
/// records nothing for it, and the copy is the item as written, strict or not. The relative type
/// with the bound stays rejected (F37), and its help says that the absolute path is a way out.
#[test]
fn f38_the_bound_that_names_self_makes_no_difference() {
    let with_bound: Item = parse_quote!(
        pub struct S<T: ::core::convert::AsRef<Self>> {
            b: ::p::Rel2<Self, T>,
        }
    );
    let without: Item = parse_quote!(
        pub struct S<T> {
            b: ::p::Rel2<Self, T>,
        }
    );
    for nest in NESTS {
        for strict in [false, true] {
            let what = format!("{:?}, strict {}", nest, strict);
            let (copy_with, referrer_with, calls_with) =
                nested_copy_as(nest, &with_bound, &["S"], strict);
            let (copy_without, referrer_without, calls_without) =
                nested_copy_as(nest, &without, &["S"], strict);
            assert_eq!(referrer_with.iter().count(), 0, "{}", what);
            assert_eq!(referrer_without.iter().count(), 0, "{}", what);
            assert!(
                calls_with.is_empty() && calls_without.is_empty(),
                "{}",
                what
            );
            assert_eq!(string(&copy_with), string(&with_bound), "{}", what);
            assert_eq!(string(&copy_without), string(&without), "{}", what);
        }
    }
    let relative: Item = parse_quote!(
        pub struct S<T: ::core::convert::AsRef<Self>> {
            b: Rel2<Self, T>,
        }
    );
    for nest in NESTS {
        let result = nested_analysis(nest, &relative, &["S"]).finish();
        let err = assert_rejected(
            &relative,
            result,
            &["`Rel2 < Self , T >` has a generic whose bound mentions `Self`, which cannot be leaked"],
        );
        assert!(
            all_helps(&err)[0].contains("write the type with an absolute path"),
            "{:?}",
            all_helps(&err)
        );
    }
}

/// F38, the controls: the types that must be leaked and name the own `Self` are entries, written
/// by `with_nested_self()` for a trusting and for a strict Referrer, and the item analyzed as
/// itself (`intern_*`), a nested trait and a nested impl are as before: their optional entries
/// that name `Self` stay, and a strict Referrer replaces them.
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
        for strict in [false, true] {
            let what = format!("{:?}, strict {}", nest, strict);
            let (copy, referrer, calls) = nested_copy_as(nest, &item, &["S"], strict);
            assert_eq!(
                f38_entries(&referrer),
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
                        referrer.context_self(referrer.context(ix)),
                        Some(SelfRequirement::Type(_))
                    ),
                    "{}",
                    what
                );
            }
            let (a, b, c, rel2) = (
                entry(&referrer, "Rel < Self >"),
                entry(&referrer, "inner :: W < Self >"),
                entry(&referrer, "crate :: W < Self >"),
                entry(&referrer, "Rel2 < Self >"),
            );
            let expected: Item = parse_quote!(
                pub struct S<T> {
                    a: N<#a>,
                    b: N<#b>,
                    c: N<#c>,
                    d: ::c::W<Self>,
                    e: ::c::W<(Self, N<#rel2>)>,
                }
            );
            assert_eq!(string(&copy), string(&expected), "{}", what);
            assert_eq!(calls.len(), 4, "{}", what);
        }
    }

    // The item analyzed as itself: `Self` is the Leaker's type in the copy (`with_self()`), so the
    // optional entry that names it is as before.
    let node: ItemStruct = parse_quote!(
        pub struct Node {
            next: ::std::boxed::Box<Self>,
        }
    );
    let mut leaker = new_leaker();
    leaker.intern_struct(&node, true);
    let referrer = leaker.result().finish().unwrap();
    assert_eq!(f38_entries(&referrer), ["?::std::boxed::Box<Self>"]);
    assert!(matches!(
        referrer.context_self(referrer.context(0)),
        Some(SelfRequirement::Type(_))
    ));

    // A nested trait: `Self` is the implementor, for which the macro writes a type: the optional
    // entry stays, and a strict Referrer replaces it.
    let tr: Item = parse_quote!(
        pub trait Tr {
            fn f(&self) -> ::c::W<Self>;
        }
    );
    for nest in NESTS {
        let (trusting, referrer, calls) = nested_copy_as(nest, &tr, &["Tr"], false);
        let (strict, _, calls_strict) = nested_copy_as(nest, &tr, &["Tr"], true);
        assert_eq!(f38_entries(&referrer), ["?::c::W<Self>"], "{:?}", nest);
        assert!(
            matches!(
                referrer.context_self(referrer.context(0)),
                Some(SelfRequirement::Bound(_))
            ),
            "{:?}",
            nest
        );
        assert_eq!(string(&trusting), string(&tr), "{:?}", nest);
        assert!(calls.is_empty(), "{:?}", nest);
        let expected: Item = parse_quote!(
            pub trait Tr {
                fn f(&self) -> N<0usize>;
            }
        );
        assert_eq!(string(&strict), string(&expected), "{:?}", nest);
        assert_eq!(calls_strict.len(), 1, "{:?}", nest);
    }

    // A nested impl: `Self` is the self type as written, the same in the copy; the entry stays.
    let imp: Item = parse_quote!(
        impl<T> ::p::W for ::p::S<T> {
            fn f() -> ::c::W<Self> {
                loop {}
            }
        }
    );
    for nest in NESTS {
        let (_, referrer, _) = nested_copy_as(nest, &imp, &[], false);
        let ix = entry(&referrer, ":: c :: W < Self >");
        assert!(!referrer.must(ix), "{:?}", nest);
        assert!(
            matches!(
                referrer.context_self(referrer.context(ix)),
                Some(SelfRequirement::Type(_))
            ),
            "{:?}",
            nest
        );
        let (strict, _, calls) = nested_copy_as(nest, &imp, &[], true);
        assert!(!string(&strict).contains("W < Self >"), "{:?}", nest);
        assert_eq!(calls.len(), 1, "{:?}", nest);
    }
}

// ── F39 ──────────────────────────────────────────────────────────────────────

/// F39 (fixed): the default of a parameter of a trait, when an entry in it needs the trait's `Self`
/// (`trait Tr<T: AsRef<Self>, U = Rel<T>>`: `Rel<T>` is for the implementors of the trait, because
/// the bound of `T` names `Self`), was left as written by the Visitor, whereas the Leaker recorded
/// it as a required entry (`SelfRequirement::Bound`). The copy kept the relative `Rel<T>`, which
/// the Referrer's crate cannot resolve. The same type in a method (`fn f(&self, u: Rel<T>)`) was
/// replaced, by `Visitor::with_nested_self()`.
///
/// The cause: the Visitor looked up the default of a parameter without `Self`, as the Leaker
/// analyzes the default of the generics of a scope that it enters (`enter_*`, `intern_*`:
/// `AnalyzeVisitor::analyze_default_inline()`), and as the default of a struct, an enum or a union
/// has to be (a default cannot name `Self` there: E0735). The Leaker analyzes the default of a trait
/// that `View::visit_item()` (and so `intern_mod()`) visits with the trait's own `Self`: a trait may
/// name `Self` in a default (`Rhs = Self`). So the contexts were not the same, and the entry was not
/// found.
///
/// The fix: the default of a trait nested in the visited syntax is looked up without `Self`
/// first (the entry of a scope that the macro entered with `Leaker::enter_trait()`), then with the
/// trait's own, and written by `Visitor::with_nested_self()`, which tells the entry by
/// `Referrer::context_self()`. Without a host (`with_nested_self()` not called) the entry is left
/// as written, as the same entry in a method is. A struct, an enum and a union nested in the syntax
/// have another rule: the copy cannot write `Self` in their default at all (E0735), so the entry
/// in it has none (F41).
///
/// The Repeater's arguments of such an entry include the defaulted parameter, as the bound
/// `Self: Tr<T, U>` of the implementor names it: a declaration cannot pass it to its own default
/// (E0128), so the macro passes a type for it that the host implements the trait for (the fixtures
/// 9131 and 9132 of `tests/cross_crate.rs`, which compile in the Referrer's crate).
///
/// The property test generates the shape with the feature `trait_self_defaults` (on by default):
/// a fifth of the traits analyzed one by one have a parameter with a bound that names `Self` and
/// trailing parameters whose defaults use it, and the control where the bound names none; the
/// nested ones (45%) are often declared again in the Referrer's crate (`redeclare`), as the
/// cross-crate fixtures 9131 and 9132 are. Before the feature it hardly appeared (a default is
/// generated for a trailing parameter that no bound or predicate names, and the earlier parameters
/// are bounded by predicates that name `Self` only now and then: in the runs of this fix, seeds 1
/// to 24 of 60 cases and seed 4 of 120, 2 of the 2170 traits with a default named `Self` in the
/// header, and neither had a type default that uses a parameter with such a bound). Against the
/// library of the commit before the fix, the generated Referrer's crates do not compile (24
/// cases of seed 1: 10 of them, "cannot find type `R1`", E0425 and E0433: the default is left as
/// written).
#[test]
fn f39_the_default_of_a_nested_trait_that_needs_its_self_is_written_through_the_host() {
    let item: Item = parse_quote!(
        pub trait Tr<T: ::core::convert::AsRef<Self>, U = Rel<T>> {
            fn f(&self, u: U);
        }
    );
    for nest in NESTS {
        for strict in [false, true] {
            let what = format!("{:?}, strict {}", nest, strict);
            let (copy, referrer, calls) = nested_copy_as(nest, &item, &["Tr"], strict);
            // The Leaker: one required entry, for the implementors of the trait. It is not a
            // `Self` of a struct (`Any`), and it names the parameter that it is the default of.
            assert_eq!(referrer.iter().count(), 1, "{}", what);
            assert!(referrer.must(0), "{}", what);
            match referrer.context_self(referrer.context(0)) {
                Some(SelfRequirement::Bound(path)) => {
                    assert_eq!(string(path), "Tr < T , U >", "{}", what)
                }
                other => panic!("{}: {:?}", what, other),
            }
            assert_eq!(
                string(referrer.context_generics(referrer.context(0))),
                "< T : :: core :: convert :: AsRef < Self > , U >",
                "{}",
                what
            );
            // The copy: the default is written for the trait's own `Self`.
            let expected: Item = parse_quote!(
                pub trait Tr<T: ::core::convert::AsRef<Self>, U = N<0usize>> {
                    fn f(&self, u: U);
                }
            );
            assert_eq!(string(&copy), string(&expected), "{}", what);
            assert_eq!(calls.len(), 1, "{}: {:?}", what, calls);
            assert!(
                calls[0].starts_with("Self: Tr < T , U > / Tr / "),
                "{}: {:?}",
                what,
                calls
            );
        }
    }

    // As reported: `enter_mod`, a closure for the host and one for the others.
    let mut leaker = new_leaker();
    let (context, ()) = leaker.enter_mod("m", names(&["Tr"]), |view| view.visit_item(&item));
    let referrer = leaker.result().finish().unwrap();
    let mut file = file_of(vec![item.clone()]);
    referrer
        .clone()
        .visitor(context, |_, ix| parse_quote!(<() as Repeater<#ix>>::Type))
        .with_nested_self(|_, _, ix| parse_quote!(<Host as Repeater<#ix>>::Type))
        .visit_file_mut(&mut file);
    let expected: Item = parse_quote!(
        pub trait Tr<T: ::core::convert::AsRef<Self>, U = <Host as Repeater<0usize>>::Type> {
            fn f(&self, u: U);
        }
    );
    assert_eq!(string(&file.items[0]), string(&expected));
    // The Repeater's arguments of the entry include `U` (the argument of `Self: Tr<T, U>`), which
    // a declaration cannot pass to the default of `U`.
    let args = referrer.entry_args(0, &UseSite::identity()).unwrap();
    assert_eq!(string(&args), "(T , U)");

    // Without a host (no `with_nested_self()`), the entry is left as written, as the same entry
    // is in a method signature of the same context.
    let method: Item = parse_quote!(
        pub trait Tr<T: ::core::convert::AsRef<Self>> {
            fn f(&self, u: Rel<T>);
        }
    );
    for nest in NESTS {
        let (copy, referrer, calls) = nested_copy_with(nest, &item, &["Tr"], false, false);
        assert_eq!(referrer.iter().count(), 1, "{:?}", nest);
        assert_eq!(string(&copy), string(&item), "{:?}", nest);
        assert!(calls.is_empty(), "{:?}", nest);
        let (copy, _, _) = nested_copy_with(nest, &method, &["Tr"], false, false);
        assert_eq!(string(&copy), string(&method), "{:?}", nest);
    }
}

/// F39, the controls: a nested trait whose default does not need its `Self` is replaced by the
/// ordinary closure as before, and a method that does by `with_nested_self()` as before. Nothing in
/// a trait is for any type (`SelfRequirement::Any` is for a struct, an enum or a union), and an
/// associated type or a constant that uses the generic is replaced like the default.
#[test]
fn f39_the_other_traits_are_unchanged() {
    // The item, and its copy with `P` for the ordinary closure and `N` for the nested one.
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
                    fn f(&self, u: N<0usize>);
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
                    type A = N<0usize>;
                    const C: N<1usize>;
                }
            ),
        ),
    ];
    for (item, expected) in &cases {
        for nest in NESTS {
            let what = format!("{:?} `{}`", nest, string(item));
            let (copy, referrer, _) = nested_copy(nest, item, &["Tr"]);
            assert_eq!(string(&copy), string(expected), "{}", what);
            for ix in 0..referrer.iter().count() {
                assert!(referrer.must(ix), "{}", what);
                assert!(
                    !matches!(
                        referrer.context_self(referrer.context(ix)),
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
    let mut leaker = new_leaker();
    let context = match root {
        Root::InternTrait => leaker.intern_trait(item, true),
        Root::EnterTrait => {
            leaker
                .enter_trait(&item.ident, &item.generics, true, |view| {
                    syn::visit::visit_item_trait(view, item)
                })
                .0
        }
        Root::EnterTraitItem => {
            let nested = Item::Trait(item.clone());
            leaker
                .enter_trait(&item.ident, &item.generics, true, |view| {
                    view.visit_item(&nested)
                })
                .0
        }
        Root::NestedEnterTrait => {
            leaker
                .enter_generics(&Generics::default(), |view| {
                    view.enter_trait(&item.ident, &item.generics, true, |view| {
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
        .filter(|&ix| string(referrer.iter().nth(ix).unwrap()) == ty)
        .map(|ix| (ix, referrer.context_self(referrer.context(ix)).is_some()))
        .collect()
}

/// F39: a trait analyzed as the item itself (`intern_trait()`, or `enter_trait()` and a walk of the
/// item) and copied with `with_self(host)`. The default is analyzed alone (without `Self`, with the
/// parameters before it): its entry is for any type that the bound holds for (`Rel<T>` with `T:
/// AsRef<Self>`, where `Self` in the bound is the type of the impl), so the macro picks the
/// implementor, and the Visitor hands the entry to the ordinary closure, whose result is not
/// substituted again. It used to look the default up with the `Self` of the scope as well, and
/// skipped the entry when another entry of the trait (`Rel2<Self>`) gave the scope a `Self`, so
/// the copy kept `Rel<T>`. The item is also analyzed as `enter_trait()` and `view.visit_item()`,
/// and with a scope of a view (`View::enter_trait()`), for which the Leaker used to record the
/// default twice (`f39_the_default_of_a_trait_is_recorded_once`): the roots here are the same.
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
        let (default, rel2) = (alone[0], entry(&referrer, "Rel2 < Self >"));
        assert!(referrer.must(default), "{}", what);
        assert_eq!(
            string(referrer.context_generics(referrer.context(default))),
            "< T : :: core :: convert :: AsRef < Self > >",
            "{}",
            what
        );
        let mut copy = item.clone();
        referrer
            .clone()
            .visitor(context, |_, ix| parse_quote!(P<#ix>))
            .with_self(host.clone())
            .visit_item_trait_mut(&mut copy);
        let expected: ItemTrait = parse_quote!(
            pub trait Tr<T: ::core::convert::AsRef<Host>, U = P<#default>> {
                fn f(&self, u: U) -> P<#rel2>;
            }
        );
        assert_eq!(string(&copy), string(&expected), "{}", what);
    }

    // Only the default: the scope has no `Self`.
    let item: ItemTrait = parse_quote!(
        pub trait Tr<T: ::core::convert::AsRef<Self>, U = Rel<T>> {
            fn f(&self, u: U);
        }
    );
    for root in ROOTS {
        let what = format!("{:?}", root);
        let (referrer, context) = root_analysis(root, &item);
        let default = entries_of(&referrer, "Rel < T >")
            .into_iter()
            .find(|(_, has_self)| !has_self)
            .unwrap_or_else(|| panic!("{}: no entry", what))
            .0;
        let mut copy = item.clone();
        referrer
            .clone()
            .visitor(context, |_, ix| parse_quote!(P<#ix>))
            .with_self(host.clone())
            .visit_item_trait_mut(&mut copy);
        let expected: ItemTrait = parse_quote!(
            pub trait Tr<T: ::core::convert::AsRef<Host>, U = P<#default>> {
                fn f(&self, u: U);
            }
        );
        assert_eq!(string(&copy), string(&expected), "{}", what);
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
        let default = entries.iter().find(|(_, has_self)| !has_self).unwrap().0;
        let method = entries.iter().find(|(_, has_self)| *has_self).unwrap().0;
        let mut copy = item.clone();
        referrer
            .clone()
            .visitor(context, |_, ix| parse_quote!(P<#ix>))
            .with_self(host.clone())
            .visit_item_trait_mut(&mut copy);
        let expected: ItemTrait = parse_quote!(
            pub trait Tr<T: ::core::convert::AsRef<Host>, U = P<#default>> {
                fn f(&self, u: P<#method>);
            }
        );
        assert_eq!(string(&copy), string(&expected), "{}", what);
    }
}

/// F39: the default of a parameter of a trait is recorded once. A scope that declares generics
/// analyzes the default of each parameter alone (without `Self`, with the parameters before it),
/// as `intern_trait()` does for the trait itself, and the walk of the trait must not analyze it
/// again. It did when the walk was not in the outermost scope, where the parameters were skipped
/// (`root_defaulted`): in `View::enter_trait()`, and in `view.visit_item(&item)` after
/// `enter_trait()` for the same trait, a nested item that declares its generics again. The default
/// was then also an entry of the nested trait, for its own `Self` (`Self: Tr<T, U>`, with `U` in
/// the generics of the context). It was harmless, as the Visitor prefers the entry analyzed alone
/// (`f39_the_default_of_the_trait_itself_is_looked_up_without_self`), but it was a second entry in
/// the Referrer and a second impl of the Repeater. (A trait that `enter_mod()` and `visit_item()`
/// or `intern_mod()` visit is analyzed once, as a nested item: its default is the entry for the
/// trait's `Self`, see
/// `f39_the_default_of_a_nested_trait_that_needs_its_self_is_written_through_the_host`.)
#[test]
fn f39_the_default_of_a_trait_is_recorded_once() {
    let item: ItemTrait = parse_quote!(
        pub trait Tr<T: ::core::convert::AsRef<Self>, U = Rel<T>> {
            fn f(&self, u: U);
        }
    );
    for root in ROOTS {
        let (referrer, _) = root_analysis(root, &item);
        assert_eq!(referrer.iter().count(), 1, "{:?}", root);
        // The entry is the one analyzed alone: no `Self`, and the generics before `U`.
        assert_eq!(
            entries_of(&referrer, "Rel < T >"),
            [(0, false)],
            "{:?}",
            root
        );
        assert_eq!(
            string(referrer.context_generics(referrer.context(0))),
            "< T : :: core :: convert :: AsRef < Self > >",
            "{:?}",
            root
        );
    }

    // The other entries of the trait are recorded once too. The same type in a method is an entry
    // of the trait's `Self`, next to the default's (the default is analyzed alone).
    let item: ItemTrait = parse_quote!(
        pub trait Tr<T: ::core::convert::AsRef<Self>, U = Rel<T>> {
            fn f(&self, u: U, v: Rel<T>) -> Rel2<Self>;
        }
    );
    for root in ROOTS {
        let (referrer, _) = root_analysis(root, &item);
        let what = format!("{:?}", root);
        let rel = entries_of(&referrer, "Rel < T >");
        assert_eq!(rel.len(), 2, "{}: {:?}", what, rel);
        assert_eq!(
            rel.iter().filter(|(_, has_self)| !has_self).count(),
            1,
            "{}: {:?}",
            what,
            rel
        );
        assert_eq!(entries_of(&referrer, "Rel2 < Self >").len(), 1, "{}", what);
        assert_eq!(referrer.iter().count(), 3, "{}", what);
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
    let mut leaker = new_leaker();
    leaker.enter_trait(&item.ident, &item.generics, true, |view| {
        view.visit_item(&other)
    });
    let referrer = leaker.result().finish().unwrap();
    assert_eq!(referrer.iter().count(), 2);
    let rel = entries_of(&referrer, "Rel < T >");
    assert!(matches!(rel[..], [(_, false)]), "{:?}", rel);
    let rel2 = entries_of(&referrer, "Rel2 < T >");
    assert!(matches!(rel2[..], [(_, true)]), "{:?}", rel2);

    // As a nested item of a module (`enter_mod()`, `intern_mod()`), the default is the entry for
    // the trait's `Self`, once.
    let mut leaker = new_leaker();
    let nested = Item::Trait(item.clone());
    leaker.enter_mod("m", names(&["Tr"]), |view| view.visit_item(&nested));
    let referrer = leaker.result().finish().unwrap();
    assert_eq!(referrer.iter().count(), 1);
    assert_eq!(entries_of(&referrer, "Rel < T >"), [(0, true)]);
    let module: ItemMod = parse_quote!(
        mod m {
            pub trait Tr<T: ::core::convert::AsRef<Self>, U = Rel<T>> {
                fn f(&self, u: U);
            }
        }
    );
    let mut leaker = new_leaker();
    leaker.intern_mod(&module);
    let referrer = leaker.result().finish().unwrap();
    assert_eq!(referrer.iter().count(), 1);
    assert_eq!(entries_of(&referrer, "Rel < T >"), [(0, true)]);
}

/// F39, the contrast: the same default in a struct nested in a scope has an entry without `Self`
/// (the copy cannot write `Self` in a default, E0735), whereas the trait's is for the trait's `Self`
/// (`SelfRequirement::Bound`, the host). It was rejected until F41.
#[test]
fn f39_a_nested_struct_has_no_self_in_the_entry_of_the_default() {
    let item: Item = parse_quote!(
        pub struct S<T: ::core::convert::AsRef<Self>, U = Rel<T>>(pub T, pub U);
    );
    for nest in NESTS {
        let referrer = nested_analysis(nest, &item, &["S"])
            .finish()
            .unwrap_or_else(|e| panic!("{:?}: {}", nest, joined_messages(&e)));
        assert_eq!(
            entries_of(&referrer, "Rel < T >"),
            [(0, false)],
            "{:?}",
            nest
        );
        // The bound that names `Self` is not in the context.
        assert_eq!(
            string(referrer.context_generics(referrer.context(0))),
            "< T >",
            "{:?}",
            nest
        );
    }
    // The item itself keeps the bound: its default is analyzed alone, with the generics before it.
    if let Item::Struct(item) = &item {
        let mut leaker = new_leaker();
        leaker.intern_struct(item, true);
        let referrer = leaker.result().finish().unwrap();
        assert_eq!(entries_of(&referrer, "Rel < T >"), [(0, false)]);
        assert_eq!(
            string(referrer.context_generics(referrer.context(0))),
            "< T : :: core :: convert :: AsRef < Self > >"
        );
    }
}

// ── F40 ──────────────────────────────────────────────────────────────────────

/// A Leaker that reads the values (`LeakerConfig::check_values()`), as the downstream crate of F40
/// has it: primitive types and the names of the standard prelude are allowed.
fn values_config() -> LeakerConfig {
    let mut config = LeakerConfig::new();
    config.allow_primitive().allow_prelude().check_values();
    config
}

fn values_leaker() -> Leaker {
    Leaker::from_config(values_config())
}

/// Analyzes the trait `source` with the option, and returns its result and its context.
fn f40_result(source: &str) -> (ItemTrait, LeakerResult, ContextId) {
    let item: ItemTrait = parse_str(source).unwrap_or_else(|e| panic!("{}: {}", source, e));
    let mut leaker = values_leaker();
    let context = leaker.intern_trait(&item, true);
    (item, leaker.result(), context)
}

/// Analyzes the trait `source` with the option, and returns its Referrer and its context.
fn f40_trait(source: &str) -> (ItemTrait, Referrer, ContextId) {
    let (item, result, context) = f40_result(source);
    let referrer = result
        .finish()
        .unwrap_or_else(|e| panic!("{}: {}", source, joined_messages(&e)));
    (item, referrer, context)
}

/// The copy of `item` in the scope `context`, with the `ix`-th entry replaced by `P<ix>`.
fn f40_copy(referrer: &Referrer, context: ContextId, item: &ItemTrait) -> String {
    rewrite(referrer, context, Item::Trait(item.clone()))
}

/// The messages and the notes of the errors the option gives to the trait `source`.
fn f40_errors(source: &str) -> Vec<(String, String)> {
    let item: ItemTrait = parse_str(source).unwrap_or_else(|e| panic!("{}: {}", source, e));
    let mut leaker = values_leaker();
    leaker.intern_trait(&item, true);
    let err = errors_of(leaker);
    assert!(!err.is_empty(), "`{}` is accepted", source);
    err.iter()
        .map(|item| {
            let note = item.note().unwrap_or("").to_string();
            (item.message.clone(), note)
        })
        .collect()
}

/// F40 (fixed): a type in a value was never carried. In
/// `trait Tr { const N: usize = ::core::mem::size_of::<inner::Local>(); }` the Leaker found no
/// entry for `inner::Local`, so the copy of the default, which the Visitor writes as it is, kept a
/// relative path that does not resolve in the Referrer's crate (E0433). Reported by the downstream
/// crate `announce` (its finding 14, `types_in_associated_const_defaults_are_not_read`).
///
/// The values are not read unless `LeakerConfig::check_values()` is on, which is unchanged (the
/// first part of the test): a patch release does not reject what it accepted. With the option the
/// check rejected every relative path in a value, a type as well (`use absolute path:
/// 'inner :: Local'`). The copy cannot write a path of an expression through the Repeater, but it
/// can write a type: the Visitor replaces a type where it stands, in an expression as in a
/// signature (it does so for the type in an enum discriminant and an array length, which are
/// always read).
///
/// The fix: with the option, a type in a *type position* of a value is analyzed like any type of
/// the item, with the generics and `Self` in scope, and becomes an entry. The Visitor needed no
/// change for it: the default `VisitMut` walk reaches the same positions, in the scopes of the
/// signature that the Leaker analyzes them in. The cross-crate fixtures 9141 and 9142 of
/// `tests/cross_crate.rs` compile the copy.
/// `FORMAT_VERSION` is unchanged: the serialized form and the meaning of an entry are the same,
/// and an entry of a value is found like any other.
///
/// The property test does not use the option, so it does not show F40.
#[test]
fn f40_a_type_in_the_default_of_an_associated_const_is_carried() {
    let source = "trait Tr { const N: usize = ::core::mem::size_of::<inner::Local>(); }";
    let item: ItemTrait = parse_str(source).unwrap();

    // Not read without the option: no entry, no error, as before.
    let mut leaker = new_leaker();
    leaker.intern_trait(&item, true);
    assert!(leaker.result().finish().unwrap().is_empty());

    // Read with it: the type is a required entry.
    let (item, referrer, context) = f40_trait(source);
    assert_eq!(
        referrer.iter().map(string).collect::<Vec<_>>(),
        ["inner :: Local"]
    );
    assert!(referrer.must(0));
    assert_eq!(referrer.required().count(), 1);
    assert_eq!(string(referrer.context_generics(referrer.context(0))), "");
    assert!(referrer.context_self(referrer.context(0)).is_none());
    // The Visitor replaces it where it stands in the default.
    let expected: ItemTrait = parse_quote!(
        trait Tr {
            const N: usize = ::core::mem::size_of::<P<0usize>>();
        }
    );
    assert_eq!(f40_copy(&referrer, context, &item), string(&expected));

    // The Repeater's impl is written like the one of an entry of a signature.
    let r = referrer
        .repeater_impl(0, ImplSelf::Marker(&parse_quote!(())))
        .unwrap();
    assert_eq!(string(&r.ty), "inner :: Local");

    // A type is replaced, a path that is not a type is still rejected.
    assert_eq!(
        f40_errors("trait Tr { const N: usize = inner::X; }"),
        [(
            "use absolute path: 'inner :: X'".to_string(),
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
    let generics =
        |ty: &str| string(referrer.context_generics(referrer.context(entry(&referrer, ty))));
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
    let with_self = referrer.context(entry(&referrer, "inner :: Wrap < Self >"));
    assert!(matches!(
        referrer.context_self(with_self),
        Some(SelfRequirement::Bound(_))
    ));
    assert_eq!(referrer.iter().count(), 4);
    assert!((0..4).all(|ix| referrer.must(ix)));
    // Every one is replaced where it stands. (`inner::Wrap<U>` twice, by one entry.)
    let ix = |ty: &str| entry(&referrer, ty);
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
    assert_eq!(f40_copy(&referrer, context, &item), string(&expected));

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
    let context = leaker.intern_impl(&item, true);
    let referrer = leaker.result().finish().unwrap();
    let required: Vec<String> = referrer.required().map(|(_, ty)| string(ty)).collect();
    assert_eq!(
        required,
        [
            "inner :: Local",
            "inner :: Wrap < T >",
            "inner :: Wrap < Self >"
        ]
    );
    let mut copy = item.clone();
    referrer
        .visitor(context, |_, ix| parse_quote!(P<#ix>))
        .visit_item_impl_mut(&mut copy);
    let expected: ItemImpl = parse_quote! {
        impl<T: ::core::default::Default> ::p::Tr<T> for ::p::S<T> {
            const N: usize = ::core::mem::size_of::<P<0usize>>();
            fn f(&self) -> usize {
                let w: P<1usize> = ::core::unimplemented!();
                ::core::mem::size_of::<P<2usize>>()
            }
        }
    };
    assert_eq!(string(&copy), string(&expected));
}

/// F40: each type position that the Visitor walks is read: a cast, a `let`, a closure, the
/// arguments of a path, of a method call, of a struct literal and of a pattern, the self type and
/// the trait arguments of a qualified path, and a type in a type, a block, a branch, an array
/// length.
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
        let found: Vec<String> = referrer.iter().map(string).collect();
        let mut sorted = found.clone();
        sorted.sort();
        let mut wanted: Vec<String> = entries.iter().map(|e| e.to_string()).collect();
        wanted.sort();
        assert_eq!(sorted, wanted, "{}", body);
        assert!(referrer.required().count() == found.len(), "{}", body);
        let mut text = copy.to_string();
        for (k, ty) in entries.iter().enumerate() {
            text = text.replace(
                &format!("@{}", k),
                &format!("P<{}usize>", entry(&referrer, ty)),
            );
        }
        let expected: ItemTrait = parse_str(&format!(
            "trait Tr<T> {{ fn f<U>(&self, p: u8) {{ {} }} }}",
            text
        ))
        .unwrap();
        assert_eq!(
            f40_copy(&referrer, context, &item),
            string(&expected),
            "{}",
            body
        );
    }
}

/// F40, the optional entries: an absolute type of another crate that holds the relative one. A
/// Referrer that trusts the names of the crates replaces the relative type only; a strict one
/// replaces the whole type. Both as for a type of a signature.
#[test]
fn f40_an_absolute_type_of_a_value_is_an_optional_entry() {
    let source =
        "trait Tr { fn f(&self) { let _: ::p::Wrap<inner::Local> = ::core::unimplemented!(); } }";
    let (item, result, context) = f40_result(source);
    let referrer = result.clone().finish().unwrap();
    assert_eq!(
        referrer
            .iter()
            .enumerate()
            .map(|(ix, ty)| format!("{}{}", if referrer.must(ix) { "" } else { "?" }, string(ty)))
            .collect::<Vec<_>>(),
        ["inner :: Local", "?:: p :: Wrap < inner :: Local >"]
    );
    assert_eq!(referrer.external_crates_require(1), ["p"]);
    let trusting = f40_copy(&referrer, context, &item);
    let expected: ItemTrait = parse_quote!(
        trait Tr {
            fn f(&self) {
                let _: ::p::Wrap<P<0usize>> = ::core::unimplemented!();
            }
        }
    );
    assert_eq!(trusting, string(&expected));
    let mut strict = result;
    strict.allow_ambiguous_crate_name(false);
    let strict = strict.finish().unwrap();
    let expected: ItemTrait = parse_quote!(
        trait Tr {
            fn f(&self) {
                let _: P<1usize> = ::core::unimplemented!();
            }
        }
    );
    assert_eq!(f40_copy(&strict, context, &item), string(&expected));
}

/// F40, the module: the const, the static and the functions of a module are read with the names
/// the module declares. What the module declares is copied with it, not leaked.
#[test]
fn f40_the_values_of_a_module() {
    let item: ItemMod = parse_quote! {
        mod m {
            pub struct Declared(pub u8);
            pub const SIZE: usize = ::core::mem::size_of::<Declared>() + ::core::mem::size_of::<inner::Local>();
            pub static OTHER: usize = ::core::mem::size_of::<inner::Other>();
            pub fn f() -> usize {
                let x: inner::Local = ::core::unimplemented!();
                let y: Declared = ::core::unimplemented!();
                ::core::mem::size_of::<::p::W<inner::Local>>()
            }
        }
    };
    let mut leaker = values_leaker();
    let module = leaker.intern_mod(&item);
    let referrer = leaker.result().finish().unwrap();
    assert_eq!(
        referrer
            .iter()
            .enumerate()
            .map(|(ix, ty)| format!("{}{}", if referrer.must(ix) { "" } else { "?" }, string(ty)))
            .collect::<Vec<_>>(),
        [
            "inner :: Local",
            "inner :: Other",
            "?:: p :: W < inner :: Local >"
        ]
    );
    // The entries are those of the module, which the copy is in.
    for ix in 0..3 {
        assert_eq!(referrer.context_module(referrer.context(ix)), Some(module));
    }
    let mut file = file_of(item.content.clone().unwrap().1);
    referrer
        .visitor(module, |_, ix| parse_quote!(P<#ix>))
        .visit_file_mut(&mut file);
    let expected: File = parse_quote! {
        pub struct Declared(pub u8);
        pub const SIZE: usize = ::core::mem::size_of::<Declared>() + ::core::mem::size_of::<P<0usize>>();
        pub static OTHER: usize = ::core::mem::size_of::<P<1usize>>();
        pub fn f() -> usize {
            let x: P<0usize> = ::core::unimplemented!();
            let y: Declared = ::core::unimplemented!();
            ::core::mem::size_of::<::p::W<P<0usize>>>()
        }
    };
    assert_eq!(string(&file), string(&expected));
}

/// F40, the nested trait: the `Self` of a trait nested in a scope is its own, and an entry of a
/// value that names it is written by `Visitor::with_nested_self()`, as one of a signature is.
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
    for nest in [Nest::EnterMod, Nest::InternMod] {
        let mut leaker = values_leaker();
        let context = match nest {
            Nest::EnterMod => {
                leaker
                    .enter_mod("m", names(&["Tr"]), |view| view.visit_item(&item))
                    .0
            }
            _ => leaker.intern_mod(&parse_quote!(mod m { #item })),
        };
        let referrer = leaker.result().finish().unwrap();
        let (mut visitor, calls) = nested_visitor(&referrer, context);
        let mut file = file_of(vec![item.clone()]);
        visitor.visit_file_mut(&mut file);
        let local = entry(&referrer, "inner :: Local");
        let wrap = entry(&referrer, "inner :: Wrap < Self >");
        let expected: Item = parse_str(
            &"pub trait Tr {
                const N: usize = ::core::mem::size_of::<P<@0>>();
                fn f(&self) -> usize {
                    let w: N<@1> = ::core::unimplemented!();
                    ::core::mem::size_of::<P<@0>>()
                }
            }"
            .replace("@0", &format!("{}usize", local))
            .replace("@1", &format!("{}usize", wrap)),
        )
        .unwrap();
        assert_eq!(string(&file.items[0]), string(&expected), "{:?}", nest);
        assert_eq!(calls.take(), ["Self: Tr / Tr / "], "{:?}", nest);
    }
}

/// F40, the controls: what the Leaker still rejects, and why. A path that is not a type, a type
/// that names what the value declares, the tokens of a macro and an item nested in a body are
/// not leaked. The type of a signature and the types that need no entry are as ever.
#[test]
fn f40_what_is_not_a_type_position_stays_rejected() {
    let rejected = |path: &str| format!("use absolute path: '{}'", path);
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
    assert_eq!(
        referrer.iter().map(string).collect::<Vec<_>>(),
        ["inner :: A"]
    );
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
    assert_eq!(
        referrer.iter().map(string).collect::<Vec<_>>(),
        ["inner :: A"]
    );
    // `Self` where it is denied: the same rules as elsewhere.
    let item: ItemTrait = parse_quote!(
        trait Tr {
            fn f(&self) {
                let _: inner::A<Self> = ::core::unimplemented!();
            }
        }
    );
    let mut leaker = values_leaker();
    leaker.intern_trait(&item, true);
    let result = leaker.result();
    assert!(result.finish().is_ok());
    let mut leaker = values_leaker();
    leaker.intern_trait(&item, false);
    let result = leaker.result();
    let err = result.errors();
    assert_eq!(err[0].message, rejected("inner :: A < Self >"));
    // Without a trait: `Self` has no meaning in a function of a module.
    let item: ItemMod = parse_quote!(
        mod m {
            pub fn f() {
                let _: inner::A<Self> = ::core::unimplemented!();
            }
        }
    );
    let mut leaker = values_leaker();
    leaker.intern_mod(&item);
    let result = leaker.result();
    let err = result.errors();
    assert!(!err.is_empty());
    assert!(err
        .iter()
        .all(|item| item.reason == Reason::SelfWithoutMeaning));
}

/// F40: the lifetime of a method in a type of its body. The body is no input of the signature, so
/// a type that names the lifetime does not make the copy leave it unconstrained (E0195, which
/// `Leaker::intern_trait` checks for the types of the signature), and the entry is found in the
/// body as in the signature.
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
        entry(&referrer, "inner :: Lt < 'a >"),
        entry(&referrer, "inner :: A"),
    );
    // `'a` is a lifetime of the method: the entry is for it.
    assert_eq!(
        string(referrer.context_generics(referrer.context(lt))),
        "< 'a >"
    );
    assert_eq!(string(referrer.context_generics(referrer.context(a))), "");
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
    assert_eq!(f40_copy(&referrer, context, &item), string(&expected));
}

/// F40: the scopes of the macro that analyzes a function itself (`enter_generics()` and
/// `View::visit_item()`) read its body too, with the generics of the scope.
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
    let mut found: Vec<String> = referrer.iter().map(string).collect();
    found.sort();
    assert_eq!(found, ["inner :: Local", "inner :: Wrap < T >"]);
    let wrap = entry(&referrer, "inner :: Wrap < T >");
    let local = entry(&referrer, "inner :: Local");
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
    assert_eq!(rewrite(&referrer, context, item), string(&expected));
}

// ── F41 ──────────────────────────────────────────────────────────────────────

/// The items of F41, with the name of the item and the generics of the context of the entry in
/// the default: a struct, an enum or a union nested in a scope, whose default (`U = Wrap<T>`) uses a
/// generic with a bound that names the item's `Self`. The bounds that name `Self` are not in the
/// context; the others are (`Clone`).
fn f41_items() -> Vec<(Item, &'static str, &'static str)> {
    vec![
        (
            parse_quote!(
                pub struct S<T: ::core::convert::AsRef<Self>, U = Wrap<T>>(pub T, pub U);
            ),
            "S",
            "< T >",
        ),
        (
            parse_quote!(
                pub struct S<T: Clone + ::core::convert::AsRef<Self>, U = Wrap<T>>(pub T, pub U);
            ),
            "S",
            "< T : Clone >",
        ),
        // The bound is a where predicate, and names `Self` among other types.
        (
            parse_quote!(
                pub struct S<T, U = Wrap<T>>(pub T, pub U)
                where
                    T: Clone + ::core::convert::AsRef<(Self, u8)>;
            ),
            "S",
            "< T : Clone >",
        ),
        (
            parse_quote!(
                pub enum E<T: Clone + ::core::convert::AsRef<Self>, U = Wrap<T>> {
                    A(T),
                    B(U),
                }
            ),
            "E",
            "< T : Clone >",
        ),
        (
            parse_quote!(
                pub union Un<T: Copy + ::core::convert::AsRef<Self>, U: Copy = Wrap<T>> {
                    a: ::core::mem::ManuallyDrop<T>,
                    b: ::core::mem::ManuallyDrop<U>,
                }
            ),
            "Un",
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
            "S",
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
            "S",
            "< 'a , T : 'a >",
        ),
    ]
}

/// F41 (fixed): the default of a parameter of a struct, an enum or a union nested in a scope, when it
/// uses a generic with a bound that names the item's `Self`, was rejected ("`Wrap < T >` is in the
/// default of a parameter, and a bound of its generics names `Self`"), whereas rustc accepts the item:
///
/// ```ignore
/// pub struct S<T: Clone + AsRef<Self>, U = Wrap<T>>(pub T, pub U);   // Wrap is relative
/// ```
///
/// The cause was the F37 scheme: the entry of a type that does not name `Self` but has a bound that
/// does is for any type that the bound holds for, and the copy passes its own type. A copy cannot
/// write `Self` in a default (E0735), so it could not pass it there.
///
/// The fix: the entry in a default does not depend on `Self` at all. rustc checks a default that
/// uses a generic where the default is used (not in the declaration), so the declaration of the copy
/// needs nothing but the type, and the bounds that name `Self` go with the bounds that the original
/// proves for its own type. The context of the entry is the generics before the default, without the
/// bounds and predicates that name `Self` (`T: Clone + AsRef<Self>` is `T: Clone`), and has no
/// `Self` (`Referrer::context_self()` is `None`). The Repeater's impl is for any type, with the
/// other bounds: `impl<T: Clone, __S: ?Sized> Repeater<(T,), ..> for __S { type Type = Wrap<T>; }`,
/// and the copy declares an ordinary default, `U = <() as Repeater<(T,), ..>>::Type`, which the
/// ordinary closure writes (`Visitor::with_nested_self()` is for the entries that depend on the own
/// `Self`). The Visitor computes the same context for the default of a nested struct, enum or union
/// to find it. The cross-crate fixtures 9151 to 9154 check that the default of the copy is the type
/// of the original where it is used (`tests/cross_crate.rs`).
///
/// A limit: if the declaration of `Wrap` needs a bound that follows only from a bound that is left
/// out (`struct Wrap<A: Iterator>` with `T: Iterator<Item = Self>`, which implies `T: Iterator`),
/// the Leaker's crate fails at the generated impl (E0277): the original compiles only because rustc
/// checks the default where it is used, and no scheme can do better. The item itself
/// (`intern_struct()`..) is not changed: its default is analyzed alone, with the generics before it
/// as they are written.
///
/// The property test generates the shape with `TYPE_LEAK_PROP_FEATURES=self_bound_adts` (on by
/// default): a struct, an enum or a union nested in `enter_mod` with such a bound and a default
/// that uses the generic.
#[test]
fn f41_a_default_that_needs_self_only_through_a_bound_is_accepted() {
    let param: Ident = parse_quote!(__S);
    let marker: Type = parse_quote!(crate::Marker);
    for (item, name, generics) in &f41_items() {
        for nest in NESTS {
            let what = format!("{:?} `{}`", nest, string(item));
            let referrer = nested_analysis(nest, item, &[name])
                .finish()
                .unwrap_or_else(|e| panic!("{} is rejected: {}", what, joined_messages(&e)));
            assert_eq!(referrer.iter().count(), 1, "{}", what);
            assert!(referrer.must(0), "{}", what);
            // The entry has no `Self`, and none of the bounds that name it.
            assert!(
                referrer.context_self(referrer.context(0)).is_none(),
                "{}",
                what
            );
            assert_eq!(
                string(referrer.context_generics(referrer.context(0))),
                *generics,
                "{}",
                what
            );
            // Its impl is for any type, with no bound of its own.
            let r = referrer.repeater_impl(0, ImplSelf::Param(&param)).unwrap();
            assert_eq!(r.self_ty, parse_quote!(__S), "{}", what);
            assert_eq!(
                r.generics.params.last().unwrap(),
                &parse_quote!(__S: ?::core::marker::Sized),
                "{}",
                what
            );
            assert!(!string(&r.generics).contains("Self"), "{}", what);
            // A marker will do as well.
            referrer
                .repeater_impl(0, ImplSelf::Marker(&marker))
                .unwrap_or_else(|e| panic!("{}: {}", what, e));
            // The serialized form keeps it.
            let again: Referrer = parse2(quote!(#referrer)).unwrap();
            assert_eq!(again, referrer, "{}", what);
            assert!(again.context_self(again.context(0)).is_none(), "{}", what);
        }
    }
    // The impl as a whole, for the item of the report.
    let item: Item = parse_quote!(
        pub struct S<T: Clone + ::core::convert::AsRef<Self>, U = Wrap<T>>(pub T, pub U);
    );
    let referrer = nested_analysis(Nest::EnterMod, &item, &["S"])
        .finish()
        .unwrap();
    let r = referrer.repeater_impl(0, ImplSelf::Param(&param)).unwrap();
    let (impl_generics, _, where_clause) = r.generics.split_for_impl();
    let (args, self_ty, ty) = (&r.args, &r.self_ty, &r.ty);
    let imp: ItemImpl = parse_quote! {
        impl #impl_generics crate::Repeater<#args, 7u64, 0usize> for #self_ty #where_clause {
            type Type = #ty;
        }
    };
    let expected: ItemImpl = parse_quote! {
        impl<T: Clone, __S: ?::core::marker::Sized> crate::Repeater<(T,), 7u64, 0usize> for __S {
            type Type = Wrap<T>;
        }
    };
    assert_eq!(imp, expected);
}

/// F41: the copy of such an item. The Visitor finds the entry of the default (it computes its context
/// without the bounds that name `Self`, as the Leaker does) and hands it to the ordinary closure, not to
/// `with_nested_self()`. The copy keeps the bound (`T: AsRef<Self>`, for its own type), and writes an
/// ordinary default.
#[test]
fn f41_the_copy_writes_an_ordinary_default() {
    for (item, name, _) in &f41_items() {
        for nest in NESTS {
            for strict in [false, true] {
                let what = format!("{:?} `{}`, strict {}", nest, string(item), strict);
                let (copy, referrer, calls) = nested_copy_as(nest, item, &[name], strict);
                let ty = string(referrer.iter().next().unwrap());
                assert_eq!(
                    string(&copy),
                    string(item).replace(&ty, "P < 0usize >"),
                    "{}",
                    what
                );
                assert!(calls.is_empty(), "{}: {:?}", what, calls);
            }
        }
    }

    // The closure of a macro writes the path through the Repeater with `()` as the implementor.
    let item: Item = parse_quote!(
        pub struct S<T: Clone + ::core::convert::AsRef<Self>, U = Wrap<T>>(pub T, pub U);
    );
    let mut leaker = new_leaker();
    let (context, ()) = leaker.enter_mod("m", names(&["S"]), |view| view.visit_item(&item));
    let referrer = leaker.result().finish().unwrap();
    let mut file = file_of(vec![item.clone()]);
    let args = referrer.entry_args(0, &UseSite::identity()).unwrap();
    referrer
        .clone()
        .visitor(
            context,
            |_, ix| parse_quote!(<() as Repeater<#args, 7u64, #ix>>::Type),
        )
        .visit_file_mut(&mut file);
    // (Parsed from text: a formatter would add a comma after the last generic.)
    let expected: Item = parse_str(
        "pub struct S<T: Clone + ::core::convert::AsRef<Self>, \
         U = <() as Repeater<(T,), 7u64, 0usize>>::Type>(pub T, pub U);",
    )
    .unwrap();
    assert_eq!(string(&file.items[0]), string(&expected));
}

/// F41: the default and a field have the same type (`Wrap<T>`), and a second default names the first.
/// The default is an entry without `Self`, the field one for any type (`SelfRequirement::Any`: the
/// copy passes its own, F37), and the other default has its own entry, whose generics are the
/// parameters it names only.
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
        let (copy, referrer, calls) = nested_copy(nest, &item, &["S"]);
        let wrap = entries_of(&referrer, "Wrap < T >");
        assert_eq!(wrap.len(), 2, "{}: {:?}", what, wrap);
        let default = wrap.iter().find(|(_, has_self)| !has_self).unwrap().0;
        let field = wrap.iter().find(|(_, has_self)| *has_self).unwrap().0;
        let pair = entry(&referrer, "Pair < Rel , U >");
        assert!(
            referrer.context_self(referrer.context(pair)).is_none(),
            "{}",
            what
        );
        assert_eq!(
            string(referrer.context_generics(referrer.context(pair))),
            "< U >",
            "{}",
            what
        );
        assert!(
            matches!(
                referrer.context_self(referrer.context(field)),
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
                pub N<#field>,
            );
        );
        assert_eq!(string(&copy), string(&expected), "{}", what);
        // Only the field goes to the closure of `with_nested_self()`.
        assert_eq!(calls.len(), 1, "{}: {:?}", what, calls);
    }
}

/// F41, the controls: a bound that names no `Self` gives the context it always gave; the option
/// `nested_self_entries()` changes nothing for the default; the trait (F39) and the item itself keep
/// the bound in the context of the default; and the default is not rewritten by a Visitor that has no
/// layer for the nested item (the entry of the default is found without one).
#[test]
fn f41_the_controls_are_unchanged() {
    // No bound names `Self`: the entry has the bound, as ever.
    let item: Item = parse_quote!(
        pub struct S<T: Clone, U = Wrap<T>>(pub T, pub U);
    );
    for nest in NESTS {
        let referrer = nested_analysis(nest, &item, &["S"]).finish().unwrap();
        assert_eq!(entries_of(&referrer, "Wrap < T >"), [(0, false)]);
        assert_eq!(
            string(referrer.context_generics(referrer.context(0))),
            "< T : Clone >"
        );
    }

    // The option changes nothing for these items: the same Referrer.
    for (item, name, _) in &f41_items() {
        for nest in NESTS {
            let plain = nested_analysis(nest, item, &[name]);
            let mut with_option = LeakerConfig::new();
            with_option
                .allow_primitive()
                .allow_prelude()
                .nested_self_entries();
            let option = nested_analysis_in(Leaker::from_config(with_option), nest, item, &[name]);
            assert_eq!(
                plain.finish().unwrap(),
                option.finish().unwrap(),
                "{:?}",
                nest
            );
        }
    }

    // The trait keeps the bound: its default is an entry for the implementors (F39).
    let item: Item = parse_quote!(
        pub trait Tr<T: ::core::convert::AsRef<Self>, U = Rel<T>> {
            fn f(&self, u: U);
        }
    );
    for nest in NESTS {
        let referrer = nested_analysis(nest, &item, &["Tr"]).finish().unwrap();
        assert_eq!(entries_of(&referrer, "Rel < T >"), [(0, true)]);
        assert_eq!(
            string(referrer.context_generics(referrer.context(0))),
            "< T : :: core :: convert :: AsRef < Self > , U >"
        );
    }
}

// ── F42 ──────────────────────────────────────────────────────────────────────

/// A Leaker that accepts the entries of F42 (`LeakerConfig::nested_self_entries()`).
fn nested_self_leaker() -> Leaker {
    let mut config = LeakerConfig::new();
    config
        .allow_primitive()
        .allow_prelude()
        .nested_self_entries();
    Leaker::from_config(config)
}

/// The items of F42, with the name of the item, the type of the entry, and its Repeater's impl (with
/// `IX` for the index of the entry): the items of `f37_rejected()` and the others that F37 rejected.
/// The entry is for any type that the bounds hold for; `Self` is the parameter in the type too, and
/// the generics of the item that the type does not use are not in the impl (`U`).
fn f42_items() -> Vec<(Item, &'static str, &'static str, &'static str)> {
    vec![
        (
            parse_quote!(
                pub struct S<T: ::core::convert::AsRef<Self>> {
                    pub b: Rel2<Self, T>,
                }
            ),
            "S",
            "Rel2 < Self , T >",
            "impl<T: ::core::convert::AsRef<__S>, __S> crate::Repeater<(T,), 7u64, IX> for __S {
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
            "E",
            "Rel < Self >",
            "impl<__S> crate::Repeater<(), 7u64, IX> for __S {
                type Type = Rel<__S>;
            }",
        ),
        (
            parse_quote!(
                pub struct S<T>(Vec<Rel2<Self>>, T)
                where
                    T: ::core::convert::AsRef<(Self, u8)>;
            ),
            "S",
            "Rel2 < Self >",
            "impl<__S> crate::Repeater<(), 7u64, IX> for __S {
                type Type = Rel2<__S>;
            }",
        ),
        (
            parse_quote!(
                pub union Un<T: Copy + ::core::convert::AsRef<Self>> {
                    a: ::core::mem::ManuallyDrop<Rel2<Self, T>>,
                }
            ),
            "Un",
            "Rel2 < Self , T >",
            "impl<T: ::core::convert::AsRef<__S> + Copy, __S> crate::Repeater<(T,), 7u64, IX> for __S {
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
            "S",
            "Rel2 < Self , T >",
            "impl<T: ::core::convert::AsRef<__S>, __S> crate::Repeater<(T,), 7u64, IX> for __S {
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
            "E",
            "Rel < Self >",
            "impl<__S> crate::Repeater<(), 7u64, IX> for __S where __S: Sized {
                type Type = Rel<__S>;
            }",
        ),
    ]
}

/// F42: a struct, an enum or a union nested in a scope, with a generic whose bound names its `Self` and
/// a type that names it too (`Rel2<Self, T>` in `struct S<T: AsRef<Self>>`), was rejected (F37: the
/// headline "has a generic whose bound mentions `Self`, which cannot be leaked"): the entry is for
/// the Leaker's type, which the bound does not hold for in the copy, whose own type has it (E0277).
/// With `LeakerConfig::nested_self_entries()` it is accepted as an entry for any type that the
/// bound holds for (`SelfRequirement::Any`), like the type that does not name `Self` (F37), and
/// `Self` is the parameter of the impl in the type too:
///
/// ```ignore
/// impl<T: AsRef<__S>, __S> Repeater<(T,), ID, 0> for __S { type Type = Rel2<__S, T>; }
/// ```
///
/// The copy passes its own type as the implementor, and writes the field as `<Self as
/// Repeater<(T,), ID, 0>>::Type`, which is `Rel2<S2<T>, T>` with the copy's `S2`: the type of the
/// original, which no other scheme gives (the Leaker's `S<T>` has not the bound in the copy). The
/// parameter `__S` stays `Sized`, as the type needs it (`Rel2<A, B>` has `A: Sized`): it is
/// `?Sized` only where the bounds and the type need no `Sized` (F37).
///
/// The risks are documented on the option: a bound on the parameter of the leaked type that
/// receives `Self` (`struct Rel2<A: Clone, B>`) fails in the Leaker's crate (E0277 at the generated
/// impl, `__S: Clone`), and rustc before 1.70 fails with E0275 for the entry in the last field of a
/// struct. Without the option the items are rejected as ever (`f37_a_type_that_names_self_with_a_
/// bound_that_names_it_is_rejected`, whose help mentions the option).
///
/// The property test generates the shapes with `TYPE_LEAK_PROP_FEATURES=nested_self_entries`, which
/// is on by default: the macro asks for the option for every item nested in `enter_mod`, and the
/// structs, enums and unions of `self_bound_adts` have a field that names `Self` in a relative
/// type. (Seeds 1 to 8 at 60 cases and 17 to 20 at 24 pass; the Leaker's and the Referrer's crates
/// compile.) The cross-crate fixtures 9161 to 9164 check the field of the copy against the original
/// (`tests/cross_crate.rs`); on rustc 1.56 and 1.66 the entry in the last field of a struct is
/// E0275, which `where Self: Sized` avoids: the fixtures keep it out of the last field.
#[test]
fn f42_the_option_accepts_a_type_that_names_self_with_a_bound_that_names_it() {
    let param: Ident = parse_quote!(__S);
    let marker: Type = parse_quote!(crate::Marker);
    for (item, name, ty, expected) in &f42_items() {
        for nest in NESTS {
            let what = format!("{:?} `{}`", nest, string(item));
            let referrer = nested_analysis_in(nested_self_leaker(), nest, item, &[name])
                .finish()
                .unwrap_or_else(|e| panic!("{} is rejected: {}", what, joined_messages(&e)));
            let ix = referrer
                .iter()
                .position(|t| string(t) == *ty)
                .unwrap_or_else(|| panic!("{}: no entry {}", what, ty));
            assert!(referrer.must(ix), "{}", what);
            assert!(
                matches!(
                    referrer.context_self(referrer.context(ix)),
                    Some(SelfRequirement::Any)
                ),
                "{}",
                what
            );
            // The impl: `Self` is the parameter in the type and in the bounds. It is `Sized`
            // (not `?Sized`), as the type needs it.
            let r = referrer.repeater_impl(ix, ImplSelf::Param(&param)).unwrap();
            assert_eq!(r.self_ty, parse_quote!(__S), "{}", what);
            let (impl_generics, _, where_clause) = r.generics.split_for_impl();
            let (args, self_ty, r_ty) = (&r.args, &r.self_ty, &r.ty);
            let ix_lit = proc_macro2::Literal::usize_suffixed(ix);
            let imp: ItemImpl = parse_quote! {
                impl #impl_generics crate::Repeater<#args, 7u64, #ix_lit> for #self_ty #where_clause {
                    type Type = #r_ty;
                }
            };
            let expected: ItemImpl =
                syn::parse_str(&expected.replace("IX", &format!("{}usize", ix))).unwrap();
            assert_eq!(string(&imp), string(&expected), "{}", what);
            // A marker is not what the copy has the bound for.
            assert!(
                matches!(
                    referrer.repeater_impl(ix, ImplSelf::Marker(&marker)),
                    Err(EncodeError::NeedsSelf(_))
                ),
                "{}",
                what
            );
            // The serialized form keeps it.
            let again: Referrer = parse2(quote!(#referrer)).unwrap();
            assert_eq!(again, referrer, "{}", what);
            assert!(
                matches!(
                    again.context_self(again.context(ix)),
                    Some(SelfRequirement::Any)
                ),
                "{}",
                what
            );
        }
    }
}

/// F42: the copy. The Visitor finds the entry in the copy of the nested item (it computes the context
/// for any type where the entry is, since the type names `Self`), and hands it to `with_nested_self()`,
/// whose closure tells by `context_self()` that `Self` is any type and writes the path with the copy's
/// own type as the implementor. A Referrer that replaces the optional entries does the same.
#[test]
fn f42_the_copy_passes_its_own_type() {
    for (item, name, ty, _) in &f42_items() {
        for nest in NESTS {
            for strict in [false, true] {
                let what = format!("{:?} `{}`, strict {}", nest, string(item), strict);
                let (copy, referrer, calls) =
                    nested_copy_in(nested_self_leaker(), nest, item, &[name], strict, true);
                let ix = entry(&referrer, ty);
                // The type of the entry becomes `N<ix>`, wherever it is. (`Vec<Rel2<Self>>` keeps
                // the vector.)
                assert_eq!(
                    string(&copy),
                    string(item).replace(ty, &format!("N < {}usize >", ix)),
                    "{}",
                    what
                );
                // The closure is called for the entry that names `Self`, and for no other.
                assert_eq!(calls.len(), 1, "{}: {:?}", what, calls);
            }
        }
    }

    // The closure of a macro writes the path through the Repeater with `Self`: the field is the type
    // of the original, with the copy's own `S` for the first argument.
    let item: Item = parse_quote!(
        pub struct S<T: ::core::convert::AsRef<Self>> {
            pub b: Rel2<Self, T>,
        }
    );
    let mut leaker = nested_self_leaker();
    let (context, ()) = leaker.enter_mod("m", names(&["S"]), |view| view.visit_item(&item));
    let referrer = leaker.result().finish().unwrap();
    let mut copy = file_of(vec![item.clone()]);
    referrer
        .clone()
        .visitor(
            context,
            |_, ix| parse_quote!(<() as Repeater<(), 7u64, #ix>>::Type),
        )
        .with_nested_self(|_, _, ix| {
            let self_ty: Type = match referrer.context_self(referrer.context(ix)) {
                Some(SelfRequirement::Any) => parse_quote!(Self),
                other => panic!("{:?}", other),
            };
            let args = referrer
                .entry_args(ix, &UseSite::identity().with_self(self_ty.clone()))
                .unwrap();
            parse_quote!(<#self_ty as Repeater<#args, 7u64, #ix>>::Type)
        })
        .visit_file_mut(&mut copy);
    let expected: Item = parse_quote!(
        pub struct S<T: ::core::convert::AsRef<Self>> {
            pub b: <Self as Repeater<(T,), 7u64, 0usize>>::Type,
        }
    );
    assert_eq!(string(&copy.items[0]), string(&expected));
}

/// F42, the controls: only what F37 rejected changes. With the option, a type that names `Self`
/// with no bound that does is for the Leaker's type as before (`SelfRequirement::Type`: it accepts a
/// leaked type with a bound on the parameter that receives `Self`); everything F37 accepted gives
/// the same Referrer; the item itself and the trait are as before; an absolute type that names
/// `Self` is still copied as written; and the default that names `Self` is still rejected (F37).
#[test]
fn f42_the_controls_are_unchanged() {
    // The bound names no `Self`: the entry is for the Leaker's type, whose impl has the bounds.
    let item: Item = parse_quote!(
        pub struct S<T: Clone, U> {
            pub b: Rel2<Self, T>,
            pub c: U,
        }
    );
    for nest in NESTS {
        let referrer = nested_analysis_in(nested_self_leaker(), nest, &item, &["S"])
            .finish()
            .unwrap();
        let ix = entry(&referrer, "Rel2 < Self , T >");
        assert!(
            matches!(
                referrer.context_self(referrer.context(ix)),
                Some(SelfRequirement::Type(_))
            ),
            "{:?}",
            nest
        );
        let plain = nested_analysis(nest, &item, &["S"]);
        assert_eq!(plain.finish().unwrap(), referrer, "{:?}", nest);
    }

    // Everything that was accepted gives the same Referrer: the entries for any type of F37, the
    // nested ones that name `Self` with no bound that does, and the others.
    let accepted: Vec<(Item, &str)> = f37_supported()
        .into_iter()
        .map(|(item, name, _)| (item, name))
        .chain(vec![
            (
                parse_quote!(
                    pub struct S<T: Clone>(Rel<Self>, T);
                ),
                "S",
            ),
            (
                parse_quote!(
                    pub struct S<T>(Rel<Self>, T)
                    where
                        Self: Sized;
                ),
                "S",
            ),
            (
                parse_quote!(
                    pub struct S<T: ::core::convert::AsRef<Self>, U>(Rel<U>, T);
                ),
                "S",
            ),
            (
                parse_quote!(
                    pub trait Tr<T: ::core::convert::AsRef<Self>> {
                        fn f(&self, x: Rel<Self>) -> Rel<T>;
                    }
                ),
                "Tr",
            ),
        ])
        .collect();
    for (item, name) in &accepted {
        for nest in NESTS {
            let plain = nested_analysis(nest, item, &[name]);
            let option = nested_analysis_in(nested_self_leaker(), nest, item, &[name]);
            assert_eq!(
                plain.finish().unwrap(),
                option.finish().unwrap(),
                "{:?} `{}`",
                nest,
                string(item)
            );
        }
    }

    // The item itself: `Self` is written as the Leaker's type, which has the bound, with or without.
    for (item, _, _, _) in &f42_items() {
        let analyze = |mut leaker: Leaker| {
            match item {
                Item::Struct(item) => leaker.intern_struct(item, true),
                Item::Enum(item) => leaker.intern_enum(item, true),
                Item::Union(item) => leaker.intern_union(item, true),
                _ => unreachable!(),
            };
            leaker.result().finish().unwrap()
        };
        let (plain, option) = (analyze(new_leaker()), analyze(nested_self_leaker()));
        assert_eq!(plain, option, "`{}`", string(item));
        assert!(
            (0..plain.iter().count()).all(|ix| !matches!(
                plain.context_self(plain.context(ix)),
                Some(SelfRequirement::Any)
            )),
            "`{}`",
            string(item)
        );
    }

    // An absolute type that names `Self` is no entry, with the option too: it is copied as written.
    let item: Item = parse_quote!(
        pub struct S<T: ::core::convert::AsRef<Self>> {
            pub b: ::p::Rel2<Self, T>,
        }
    );
    for nest in NESTS {
        let (copy, referrer, calls) =
            nested_copy_in(nested_self_leaker(), nest, &item, &["S"], true, true);
        assert_eq!(referrer.iter().count(), 0, "{:?}", nest);
        assert_eq!(string(&copy), string(&item), "{:?}", nest);
        assert!(calls.is_empty(), "{:?}", nest);
    }

    // The option does not accept `Self::A` in a bound, whose impl has no trait to take `A` from.
    let item: Item = parse_quote!(
        pub struct S<T: ::core::iter::Iterator<Item = Self::A>> {
            pub b: Rel2<Self, T>,
        }
    );
    for nest in NESTS {
        let result = nested_analysis_in(nested_self_leaker(), nest, &item, &["S"]);
        assert!(result.finish().is_err(), "{:?}", nest);
    }
}

// ── F43 ──────────────────────────────────────────────────────────────────────

/// The parts that the Referrer carries, as it serializes them after the format tag: one string for
/// each (`KIND (CAUSE) CONTEXT [predicate] [in (TYPE) CONTEXT]`).
fn parts_of(referrer: &Referrer) -> Vec<String> {
    use proc_macro2::{Delimiter, TokenStream, TokenTree};
    let mut tokens = template_quote::ToTokens::to_token_stream(referrer)
        .into_iter()
        .skip(1);
    let list = match tokens.next() {
        Some(TokenTree::Group(group)) if group.delimiter() == Delimiter::Bracket => group.stream(),
        _ => return Vec::new(),
    };
    let (mut parts, mut current) = (Vec::new(), TokenStream::new());
    for token in list {
        match &token {
            TokenTree::Punct(punct) if punct.as_char() == ',' => {
                parts.push(std::mem::take(&mut current).to_string());
            }
            _ => current.extend([token]),
        }
    }
    if !current.is_empty() {
        parts.push(current.to_string());
    }
    parts
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

/// The config of `new_leaker()`.
fn std_config() -> LeakerConfig {
    let mut config = LeakerConfig::new();
    config.allow_primitive().allow_prelude();
    config
}

/// The scope of F43: the trait `Tr<T: AsRef<Self>>` (a bound of `T` names the `Self` of the trait),
/// with `direct` visited in it as types, and `nested` an item nested in it
/// (`View::visit_item()`), which has a `Self` of its own.
fn f43_analyze(config: LeakerConfig, direct: &[Type], nested: &Item) -> (LeakerResult, ContextId) {
    let mut leaker = Leaker::from_config(config);
    let generics: Generics = parse_quote!(<T: AsRef<Self>>);
    let (context, ()) = leaker.enter_trait(&parse_quote!(Tr), &generics, true, |view| {
        for ty in direct {
            view.visit_type(ty);
        }
        view.visit_item(nested);
    });
    (leaker.result(), context)
}

/// The copy of the scope of F43: the trait, with `direct` as the arguments of a method, and the
/// nested item in the body of another (the copy nests the items as the Leaker did).
fn f43_copy(
    referrer: &Referrer,
    context: ContextId,
    direct: &[Type],
    nested: &Item,
    with_nested_self: bool,
) -> String {
    let inputs = direct.iter().enumerate().map(|(i, ty)| {
        let name = Ident::new(&format!("x{}", i), proc_macro2::Span::call_site());
        quote!(#name: #ty)
    });
    let mut copy: ItemTrait = parse_quote!(trait Tr<T: AsRef<Self>> {
        fn direct(&self, #(#inputs),*);
        fn nested() { #nested }
    });
    let visitor = referrer
        .clone()
        .visitor(context, |_, ix| parse_quote!(P<#ix>));
    if with_nested_self {
        visitor
            .with_nested_self(|_, _, ix| parse_quote!(N<#ix>))
            .visit_item_trait_mut(&mut copy);
    } else {
        let mut visitor = visitor;
        visitor.visit_item_trait_mut(&mut copy);
    }
    string(&copy)
}

/// F43 (fixed): the Leaker copies as written a type whose context cannot be written
/// ([`Reason::UnnameableSelf`]), and the Referrer did not know it, so the visitor replaced the
/// types inside it.
///
/// Trigger: a scope that declares a `Self` (`enter_trait(Tr<T: AsRef<Self>>)`) and an item nested in
/// it that has a `Self` of its own (`view.visit_item(impl ::p::W for X { fn f(&self, x: Rel<T,
/// Other>) {} .. })`). The bound of `T` names the `Self` of the trait, which has no name in the
/// impl, so `Rel<T, Other>` has no context: it is a conflict, copied as written, whole. The part
/// was recorded only if its context could be written (`AnalyzeVisitor::node_part()` and the
/// conflicts of `decide()` dropped it), so it was left out. The comment of `decide()` said that
/// such a type "is found nowhere", which is false: the visitor finds it by its tokens, and
/// replaced the entries inside it that are entries elsewhere (`Other` of `g`). That breaks the
/// copy where the entry is position-sensitive: `::p::W<'a, T, dyn RelTr>` has `dyn RelTr + 'a`
/// (the default of the trait object comes from the path around it), and the entry `dyn RelTr` of
/// `Bare<dyn RelTr>` is `+ 'static` (E0308 "lifetime mismatch" in the Referrer's crate).
///
/// The fix: the part has no context (`PartItem::context` is `None`, serialized as `unwritable`
/// where the number of the context would be, before `predicate` and `in (..)`), and the visitor
/// finds it where it cannot write the context either: the same tokens, the same place in a
/// predicate or not, and the same anchor. The same tokens where the context can be written are an
/// entry as before (`Rel<T, Other>` of the method of the trait itself). The top-level context
/// records the `Self` of the scope, or the visitor would write the bound of `T` there too.
///
/// The cross-crate fixture is 9199 (`tests/cross_crate.rs`).
#[test]
fn f43_a_type_whose_context_cannot_be_written_is_copied_as_written() {
    let nested: Item = parse_quote!(
        impl ::p::W for X {
            fn f(&self, x: Rel<T, Other>) {}
            fn g(&self, y: Other) {}
        }
    );
    // The same tokens where the context can be written: in the trait itself.
    let direct: Vec<Type> = vec![parse_quote!(Rel<T, Other>)];
    let (result, context) = f43_analyze(std_config(), &direct, &nested);
    assert_eq!(
        joined_messages(result.errors()),
        "the bound of `T` mentions a `Self` which has no name here"
    );
    assert_eq!(result.errors()[0].reason, Reason::UnnameableSelf);
    let referrer = result.into_referrer();
    assert_eq!(parts_of(&referrer), ["type (Rel < T , Other >) unwritable"]);
    let (other, outer) = (
        entry(&referrer, "Other"),
        entry(&referrer, "Rel < T , Other >"),
    );
    // The part is left as written; `Other` is an entry in `g`, and `Rel<T, Other>` of the trait
    // (a context that can be written) is one as ever.
    let expected = format!(
        "trait Tr < T : AsRef < Self > > {{ fn direct (& self , x0 : P < {outer}usize >) ; fn nested () {{ impl :: p :: W for P < {x}usize > {{ fn f (& self , x : Rel < T , Other >) {{ }} fn g (& self , y : P < {other}usize >) {{ }} }} }} }}",
        outer = outer,
        other = other,
        x = entry(&referrer, "X"),
    );
    assert_eq!(
        f43_copy(&referrer, context, &direct, &nested, false),
        expected
    );
    // The top-level context has the `Self` of the scope: that is how the visitor can tell that
    // the context of the nested type has none.
    assert!(referrer.context_self(context).is_some());
    // Without the part, the visitor replaced `Other` inside it.
    let before = without_parts(&referrer);
    assert_eq!(
        f43_copy(&before, context, &direct, &nested, false),
        expected.replace(
            "x : Rel < T , Other >",
            &format!("x : Rel < T , P < {}usize > >", other)
        )
    );
    // It is written and read back.
    let again: Referrer = parse2(quote!(#referrer)).unwrap();
    assert_eq!(again, referrer);
    assert_eq!(parts_of(&again), parts_of(&referrer));
}

/// F43: a trait object in a path that gives it a lifetime. The part is `::p::W<'a, T, dyn RelTr>`;
/// `dyn RelTr` is an entry beside it, in `::p::Bare<dyn RelTr>`, with another lifetime, and a
/// strict Referrer replaces that path too, as an optional entry.
#[test]
fn f43_the_entry_inside_the_part_is_not_replaced() {
    let nested: Item = parse_quote!(
        impl<'a> ::p::W for X {
            fn f(&self, x: ::p::W<'a, T, dyn RelTr>) {}
            fn g(&self, y: ::p::Bare<dyn RelTr>) {}
        }
    );
    let (result, context) = f43_analyze(std_config(), &[], &nested);
    assert_eq!(result.errors().len(), 1);
    assert_eq!(result.errors()[0].reason, Reason::UnnameableSelf);
    let referrer = result.clone().into_referrer();
    assert_eq!(
        parts_of(&referrer),
        ["type (:: p :: W < 'a , T , dyn RelTr >) unwritable"]
    );
    let (object, x) = (entry(&referrer, "dyn RelTr"), entry(&referrer, "X"));
    let bare = entry(&referrer, ":: p :: Bare < dyn RelTr >");
    assert!(referrer.must(object) && !referrer.must(bare));
    for (strict, g) in [
        (false, format!(":: p :: Bare < P < {}usize > >", object)),
        (true, format!("P < {}usize >", bare)),
    ] {
        let mut result = result.clone();
        result.allow_ambiguous_crate_name(!strict);
        let referrer = result.into_referrer();
        let expected = format!(
            "trait Tr < T : AsRef < Self > > {{ fn direct (& self ,) ; fn nested () {{ impl < 'a > :: p :: W for P < {}usize > {{ fn f (& self , x : :: p :: W < 'a , T , dyn RelTr >) {{ }} fn g (& self , y : {}) {{ }} }} }} }}",
            x, g
        );
        assert_eq!(
            f43_copy(&referrer, context, &[], &nested, false),
            expected,
            "strict {}",
            strict
        );
    }
}

/// F43: the same for a bound whose trait path conflicts (`Rel<T, Other>` in `U: RelTrait<T,
/// Other>`, a node that is no type) and for the path of a value (`inner::make::<T, Other>()`, with
/// `check_values()`): their contexts cannot be written either.
#[test]
fn f43_a_bound_and_a_value_whose_context_cannot_be_written_are_copied_as_written() {
    let nested: Item = parse_quote!(
        impl<U: RelTrait<T, Other>> ::p::W for ::p::Host {
            fn f(&self, y: Other) -> usize {
                inner::make::<T, Other>()
            }
        }
    );
    let mut config = std_config();
    config.check_values();
    let (result, context) = f43_analyze(config, &[], &nested);
    assert!(result
        .errors()
        .iter()
        .all(|error| error.reason == Reason::RelativePath));
    assert_eq!(result.errors().len(), 2);
    let referrer = result.into_referrer();
    assert_eq!(
        parts_of(&referrer),
        [
            "trait (RelTrait < T , Other >) unwritable predicate",
            "expr (inner :: make :: < T , Other >) unwritable"
        ]
    );
    let other = entry(&referrer, "Other");
    let expected = format!(
        "trait Tr < T : AsRef < Self > > {{ fn direct (& self ,) ; fn nested () {{ impl < U : RelTrait < T , Other > > :: p :: W for :: p :: Host {{ fn f (& self , y : P < {}usize >) -> usize {{ inner :: make :: < T , Other > () }} }} }} }}",
        other
    );
    assert_eq!(f43_copy(&referrer, context, &[], &nested, false), expected);
    // Without the parts the bound and the path were rewritten.
    let before = f43_copy(&without_parts(&referrer), context, &[], &nested, false);
    let entry = format!("P < {}usize >", other);
    assert_eq!(
        before,
        expected
            .replace(
                "RelTrait < T , Other >",
                &format!("RelTrait < T , {} >", entry)
            )
            .replace(
                "make :: < T , Other >",
                &format!("make :: < T , {} >", entry)
            )
    );
}

/// F43: the serialized part. `unwritable` stands where the number of the context would be, before
/// `predicate` and `in (..)`, and the parser reads what the serializer writes (and only that).
#[test]
fn f43_an_unwritable_part_is_serialized_and_read_back() {
    let tag = format!("\"type-leak-format-{}\"", Referrer::FORMAT_VERSION);
    let read = |parts: &str| -> std::result::Result<Referrer, String> {
        parse_str::<Referrer>(&format!("{} [{}] 2 (Rel) < T > ; ()", tag, parts))
            .map_err(|e| e.to_string())
    };
    for parts in [
        "type (R<T>) unwritable",
        "trait (RelTr<T>) unwritable predicate",
        "type (R<T>) unwritable in (Wrap<R<T>>) 1",
        "type (R<T>) unwritable predicate in (Wrap<R<T>>) 0",
        "expr (f::<T>) unwritable, type (R) 1, type (R<T>) 0 predicate",
    ] {
        let referrer = read(parts).unwrap_or_else(|e| panic!("{}: {}", parts, e));
        let written = parts_of(&referrer);
        // What is written is read back to the same Referrer, and written again as it was.
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
    for (parts, message) in [
        ("type (R)", "expected integer literal"),
        ("type (R) nothing", "expected a context or `unwritable`"),
        (
            "type (R) unwritable nope",
            "expected `sig`, `predicate`, `in` or `,`",
        ),
        ("type (R) unwritable in (&R)", "expected integer literal"),
        (
            "type (R) unwritable in (&R) 5",
            "the context 5 of a part is not one of the contexts",
        ),
        (
            "type (R) unwritable unwritable",
            "expected `sig`, `predicate`, `in` or `,`",
        ),
    ] {
        let error = read(parts)
            .err()
            .unwrap_or_else(|| panic!("read: {}", parts));
        assert!(error.contains(message), "{}: {}", parts, error);
    }
}

// ── F44 ──────────────────────────────────────────────────────────────────────

/// The Leaker of F44: `::p` is allowed (the types of the items are relative, `::core::..` is
/// absolute), and the prelude and the primitives.
fn f44_config() -> LeakerConfig {
    let mut config = std_config();
    config.allow_path(parse_quote!(::p));
    config
}

/// The copy of the trait, once with `Self`, once with `with_self(Host)`.
fn f44_copies(referrer: &Referrer, context: ContextId, item: &ItemTrait) -> [String; 2] {
    let plain = {
        let mut copy = item.clone();
        referrer
            .clone()
            .visitor(context, |_, ix| parse_quote!(P<#ix>))
            .visit_item_trait_mut(&mut copy);
        string(&copy)
    };
    let hosted = {
        let mut copy = item.clone();
        referrer
            .clone()
            .visitor(context, |_, ix| parse_quote!(P<#ix>))
            .with_self(parse_quote!(Host))
            .visit_item_trait_mut(&mut copy);
        string(&copy)
    };
    [plain, hosted]
}

/// F44 (fixed): a part whose context names `Self` was never found when the top-level context of
/// the scope left `Self` out, so the visitor replaced the types inside it.
///
/// Trigger: a conflict about `Self` that the Leaker copies as written, in a scope that declares
/// `Self`: `DeniedSelfInBounds` (`allow_self = false`, `fn f<T: PartialEq<Self>>(&self, x: Wrap<T,
/// Other>)`), `SelfInTraitPredicate` (`trait Tr where Rel<Self, Other>: Sized`), and
/// `SelfInImplSelfType` with the `UnprovableBound` part that goes with it (`impl<T> S<T, Other>
/// where T: AsRef<Option<Self>>`). No entry names `Self`, so the top-level context of the scope
/// had none (`names_self()` counted the types that may be entries only), and the visitor, whose
/// scope then had none, computed the context of the part without `Self`: it never was the part's.
/// The part was not left as written, and the types in it that are entries elsewhere were replaced
/// (`Other`). That breaks the copy where the entry is position-sensitive (`dyn Tr` under a path
/// with a lifetime parameter: E0308 in the Referrer's crate).
///
/// The fix: `names_self()` counts a part copied as written whose context depends on `Self` too, so
/// the top-level context has it (`Referrer::context_self()`), and the visitor finds the part.
/// Cross-crate fixtures 9195 to 9197 (`tests/cross_crate.rs`).
#[test]
fn f44_a_part_that_names_self_is_found_when_no_entry_does() {
    // DeniedSelfInBounds, SelfInTraitPredicate; the entry beside is `Other`, in `g`.
    let items: [(ItemTrait, bool, &str, Reason, &str); 2] = [
        (
            parse_quote!(
                trait Tr {
                    fn f<T: ::core::cmp::PartialEq<Self>>(&self, x: Wrap<T, Other>);
                    fn g(&self, y: Other);
                }
            ),
            false,
            "type (Wrap < T , Other >)",
            Reason::DeniedSelfInBounds,
            "trait Tr { fn f < T : :: core :: cmp :: PartialEq < SELF > > (& self , x : Wrap < T , Other >) ; fn g (& self , y : P < 0usize >) ; }",
        ),
        (
            parse_quote!(
                trait Tr
                where
                    Rel<Self, Other>: Sized,
                {
                    fn g(&self, y: Other);
                }
            ),
            true,
            "type (Rel < Self , Other >)",
            Reason::SelfInTraitPredicate,
            "trait Tr where Rel < SELF , Other > : Sized , { fn g (& self , y : P < 0usize >) ; }",
        ),
    ];
    for (item, allow_self, part, reason, expected) in &items {
        let what = string(item);
        let mut leaker = Leaker::from_config(f44_config());
        let context = leaker.intern_trait(item, *allow_self);
        let result = leaker.result();
        assert_eq!(result.errors().len(), 1, "{}", what);
        assert_eq!(result.errors()[0].reason, *reason, "{}", what);
        let referrer = result.into_referrer();
        // The top-level context has the `Self` of the trait.
        assert!(
            matches!(
                referrer.context_self(context),
                Some(SelfRequirement::Bound(_))
            ),
            "{}",
            what
        );
        let parts = parts_of(&referrer);
        assert_eq!(parts.len(), 1, "{}", what);
        assert!(parts[0].starts_with(part), "{}: {}", what, parts[0]);
        // The part is as written, with `Self` or the host; the entry beside it is replaced.
        let [plain, hosted] = f44_copies(&referrer, context, item);
        assert_eq!(plain, expected.replace("SELF", "Self"), "{}", what);
        assert_eq!(hosted, expected.replace("SELF", "Host"), "{}", what);
        // It is written and read back.
        let again: Referrer = parse2(quote!(#referrer)).unwrap();
        assert_eq!(again, referrer, "{}", what);
    }
}

/// F44: `SelfInImplSelfType`. The self type of the impl is a part, and so is the `Self` of its
/// bound (`UnprovableBound`, in a predicate): both are left as written, and the entry in a
/// method is replaced.
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
    let mut leaker = Leaker::from_config(f44_config());
    let context = leaker.intern_impl(&item, true);
    let result = leaker.result();
    assert_eq!(
        result.errors().iter().map(|e| e.reason).collect::<Vec<_>>(),
        [Reason::SelfInImplSelfType, Reason::UnprovableBound]
    );
    let referrer = result.into_referrer();
    assert!(matches!(
        referrer.context_self(context),
        Some(SelfRequirement::Type(_))
    ));
    assert_eq!(parts_of(&referrer).len(), 2);
    let mut copy = item.clone();
    referrer
        .clone()
        .visitor(context, |_, ix| parse_quote!(P<#ix>))
        .visit_item_impl_mut(&mut copy);
    let expected: ItemImpl = parse_quote!(
        impl<T> S<T, Other>
        where
            T: AsRef<Option<Self>>,
        {
            fn g(&self, y: P<0usize>) {}
        }
    );
    assert_eq!(string(&copy), string(&expected));
}

// ── F45 ──────────────────────────────────────────────────────────────────────

/// F45 (fixed): a part in an item nested in a scope was not found by a visitor without
/// `Visitor::with_nested_self()`, so the types inside it were replaced.
///
/// Trigger: a conflict about the `Self` of a nested item that the Leaker copies as written:
/// `NestedSelfInBounds` (`struct S<T: AsRef<Self>> { a: Rel2<Self, T, Other> }`) and
/// `SelfInTraitPredicate` (`trait Tr where Rel<Self, Other>: Sized`), with an entry beside it that
/// is an entry in the part too (`Other`). The part's context has `Self`, and a visitor that is not given
/// `with_nested_self()` looks nothing up that names the `Self` of a nested item (`no_self`: `Self`
/// means something else there than in the entries, which only that closure can write). The part
/// is not an entry to replace, but a copy to protect, and the rule hid it: `fits()` was false
/// whenever `no_self` was set and the context of the part had `Self`, so `Other` in it was replaced
/// (`Rel2<Self, T, P<0>>`), where a visitor with the closure left it. A part matches whatever
/// `no_self` says, in `Folder::lookup()` and `Visitor::is_part_node()`. (The default of a
/// parameter of a nested struct that names `Self`, `SelfInDefault`, is looked up without `Self`, so
/// its part is not found with the closure either: not this finding.)
///
/// The cross-crate fixture is 9198 (`tests/cross_crate.rs`).
#[test]
fn f45_a_part_in_a_nested_item_is_left_without_with_nested_self() {
    let items: Vec<(Item, &str, &str)> = vec![
        (
            parse_quote!(
                pub struct S<T: ::core::convert::AsRef<Self>> {
                    pub a: Rel2<Self, T, Other>,
                    pub b: Other,
                }
            ),
            "S",
            "pub struct S < T : :: core :: convert :: AsRef < Self > > { pub a : Rel2 < Self , T , Other > , pub b : P < 0usize > , }",
        ),
        (
            parse_quote!(
                pub trait Tr
                where
                    Rel<Self, Other>: Sized,
                {
                    fn g(&self, y: Other);
                }
            ),
            "Tr",
            "pub trait Tr where Rel < Self , Other > : Sized , { fn g (& self , y : P < 0usize >) ; }",
        ),
        // The same for a bound whose trait path conflicts, which is a node and no type.
        (
            parse_quote!(
                pub trait Tr<T: ::core::convert::AsRef<Self>> {
                    fn m<U: RelTrait<T, Other>>(&self, u: U);
                    fn g(&self, y: Other);
                }
            ),
            "Tr",
            "pub trait Tr < T : :: core :: convert :: AsRef < Self > > { fn m < U : RelTrait < T , Other > > (& self , u : U) ; fn g (& self , y : P < 0usize >) ; }",
        ),
    ];
    for (item, name, expected) in &items {
        for nest in NESTS {
            for nested in [false, true] {
                let what = format!("{:?} `{}`, with_nested_self {}", nest, string(item), nested);
                let (copy, referrer, calls) =
                    nested_copy_recovering(new_leaker(), nest, item, &[name], false, nested, true);
                assert_eq!(string(&copy), *expected, "{}", what);
                // The part is no entry, so the closure of `with_nested_self()` is not called.
                assert!(calls.is_empty(), "{}: {:?}", what, calls);
                assert_eq!(parts_of(&referrer).len(), 1, "{}", what);
            }
        }
    }
}

// ── F46 ──────────────────────────────────────────────────────────────────────

/// F46 (fixed): the help for a relative path that starts with `core`, `std` or `alloc` (`core::fmt::
/// Debug` without the leading `::`) said "write `core::fmt::Debug` with an absolute path (e.g.
/// `::my_crate::core::fmt::Debug`)": it is the crate itself that is meant, so the example is the
/// path with the colons (`::core::fmt::Debug`). Any other relative path keeps the example in the
/// crate of the Leaker, also one that has the name in a later segment (`m::core::X`).
///
/// The cross-crate fixture is the item `Denied` (9193) of `tests/crates/diagnostic_result`, whose
/// error `tests/diagnostic.rs` checks.
#[test]
fn f46_the_help_for_a_relative_std_path_suggests_the_crate_itself() {
    let help_of = |item: Item| -> Vec<String> {
        let mut leaker = Leaker::new();
        match &item {
            Item::Trait(item) => leaker.intern_trait(item, true),
            Item::Struct(item) => leaker.intern_struct(item, true),
            _ => unreachable!(),
        };
        let result = leaker.result();
        all_helps(result.errors())
    };
    let help = |path: &str, example: &str| {
        format!(
            "write `{}` with an absolute path (e.g. `{}`), or allow it with \
             `LeakerConfig::allow_path()`",
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
/// Fn(::p::WIt<Y>) -> u8, Rel<Y>: Default; }`. The entry `Rel<Y>` does not name `Self`, but `Y: 'x`
/// pulls in `Self: 'x`, so its context depended on `Self`, and `Context::compute()` kept every
/// bound of every generic, `Y: Fn(WIt<Y>)` too: E0277 "expected an `Fn(u8)` closure, found `Y`" in
/// the copy (F26). The same for an impl (`Self` is its self type).
///
/// Now an entry that names `Self` (`Rel<Self::A, Y>`), or writes out the self type, keeps every
/// bound, and one that depends on `Self` only through a bound or an outlives predicate keeps whole
/// the bounds of the generics that the description of `Self` names (`impl<T: ..> S<T>`, `Tr<T>`)
/// and leaves out the others. The serialized contexts of such items changed (format 16).
///
/// The cross-crate fixtures are 9200 (trait) and 9201 (impl of a trait) in `tests/cross_crate.rs`.
#[test]
fn f47_self_in_an_outlives_predicate_keeps_the_bounds_of_the_description_of_self_only() {
    let leaked = |item: &Item| -> (Referrer, Item) {
        let mut leaker = f13_leaker();
        let context = match item {
            Item::Impl(i) => leaker.intern_impl(i, true),
            Item::Trait(i) => leaker.intern_trait(i, true),
            _ => panic!("unsupported item"),
        };
        let referrer = leaker.result().finish().unwrap();
        let mut copy = item.clone();
        referrer
            .clone()
            .visitor(context, |_, ix| parse_quote!(P<#ix>))
            .visit_item_mut(&mut copy);
        (referrer, copy)
    };
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
        let ix = entry(&referrer, "Rel < Y >");
        assert_eq!(
            generics_text(referrer.generics(ix)),
            "< 'x , Y : 'x + Iterator < Item = u8 > > where Rel < Y > : Default",
            "{}",
            string(item)
        );
        assert_eq!(referrer.context_self(referrer.context(ix)), Some(self_req));
        assert!(has_replacement_in_predicate(&copy), "{}", string(&copy));
        // The context is written and read back.
        let again: Referrer = parse2(quote!(#referrer)).unwrap();
        assert_eq!(again, referrer);
        assert_eq!(
            generics_text(again.generics(ix)),
            generics_text(referrer.generics(ix))
        );
    }
    // The generic that the description of `Self` names keeps every bound (`Tr<T>`, `S<T>`), a
    // generic of the method does not.
    let described: Vec<Item> = vec![
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
        parse_quote!(
            impl<T: Iterator<Item = u8> + Fn(::p::WIt<T>)> ::p::S<T> {
                fn m<'x, Y: 'x + Iterator<Item = u8>>(&'x self)
                where
                    Self: 'x,
                    Y: Fn(::p::WIt<Y>) -> u8,
                    Rel<T, Y>: Default,
                {
                }
            }
        ),
    ];
    for item in &described {
        let (referrer, copy) = leaked(item);
        assert_eq!(
            string(referrer.generics(entry(&referrer, "Rel < T , Y >"))),
            "< 'x , T : Fn (:: p :: WIt < T >) + Iterator < Item = u8 > , Y : 'x + Iterator < \
             Item = u8 > >",
            "{}",
            string(item)
        );
        assert!(has_replacement_in_predicate(&copy), "{}", string(&copy));
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
        string(referrer.generics(entry(&referrer, "Rel < Self :: A , Y >"))),
        "< Y : Fn (:: p :: WIt < Y >) + Iterator < Item = u8 > >"
    );
}

// ── F48 ──────────────────────────────────────────────────────────────────────

/// The items of F48 in one Leaker, with the errors that remain and their top-level contexts.
fn f48_leak(items: &[Item]) -> (Referrer, Vec<ContextId>, Vec<Reason>) {
    let mut leaker = Leaker::from_config(std_config());
    let contexts = items.iter().map(|item| intern(&mut leaker, item)).collect();
    let result = leaker.result();
    let reasons = result.errors().iter().map(|e| e.reason).collect();
    (result.into_referrer(), contexts, reasons)
}

/// The copies of `items`, each with the entries replaced by `P<ix>`.
fn f48_copies(referrer: &Referrer, contexts: &[ContextId], items: &[Item]) -> Vec<String> {
    items
        .iter()
        .zip(contexts)
        .map(|(item, context)| {
            let mut copy = item.clone();
            referrer
                .clone()
                .visitor(*context, |_, ix| parse_quote!(P<#ix>))
                .visit_item_mut(&mut copy);
            string(&copy)
        })
        .collect()
}

/// F48 (fixed): a type copied as written for the signature it is in was left so everywhere.
///
/// Trigger, E0195: `fn f<'x>(&self, a: Wrap<'x>) -> &'x u8` in a trait: `Wrap<'x>` is the only
/// input that names `'x`, and leaked it would leave it unconstrained, so the Leaker reports it
/// (`UnconstrainedLifetime`) and, once `allow()`ed, copies it as written. The part had no anchor:
/// the Referrer left every `Wrap<'x>` of that context as written, `g<'x>(&self, a: Wrap<'x>)` and
/// the field of `struct S<'x>` in the same Leaker too, where the Referrer's crate may not name
/// `Wrap` (E0412). E0106, the same: `fn f(x: &u8, y: Wrap<T>) -> &u8` in `trait Tr<'a, T: 'a>`,
/// where the projection of `Wrap<T>` names `'a` and the elided output sees two lifetimes.
///
/// The part carries a mark that the Referrer computes at each signature (`SignatureFacts`):
/// `sig ('x)` (the method lifetimes of the type that the output names and no bound does) and `sig
/// (elided)` (the signature relies on elision, and the type adds lifetimes to it). It is left as
/// written only in the arguments of a signature that has the same fault: not in another item, in
/// the output, in a bound, or where no signature is visited (`Referrer::expand()`). The entries,
/// their contexts and the other parts are as before (format 16).
///
/// A limit: another input that keeps the lifetime as written (`&'x self`) makes the Leaker keep
/// the entry, and the Referrer cannot tell: the signature has the same mark there.
///
/// The cross-crate fixtures are 9202 and 9203 (`tests/cross_crate.rs`).
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
    // Without the mark, the part is left everywhere: the field of `S` and `g` too.
    let unmarked: Referrer = parse_str(&string(&referrer).replace(" sig ('x)", "")).unwrap();
    assert_eq!(parts_of(&unmarked), ["type (Wrap < 'x >) 0"]);
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
    let mut leaker = Leaker::from_config(std_config());
    let context = leaker.intern_trait(&item, true);
    // Another item gives the type an entry, so there is something to replace it with.
    let struct_context = leaker.intern_struct(
        &parse_quote!(
            struct S<'x>(Wrap<'x>);
        ),
        true,
    );
    let referrer = leaker.result().into_referrer();
    let replace = |_: Type, ix: usize| -> Type { parse_quote!(P<#ix>) };
    let ty: Type = parse_quote!(Wrap<'x>);
    let sig: Signature = parse_quote!(fn f<'x>(&self, a: Wrap<'x>) -> &'x u8);
    let other: Signature = parse_quote!(fn f<'x>(&self, a: Wrap<'x>));

    // In the arguments of the signature that has the fault: as written.
    let mut copy = sig.clone();
    referrer
        .clone()
        .visitor(context, replace)
        .visit_signature_mut(&mut copy);
    assert_eq!(string(&copy), string(&sig));
    // In a signature without it, in the output, and in no signature: replaced.
    let mut copy = other.clone();
    referrer
        .clone()
        .visitor(context, replace)
        .visit_signature_mut(&mut copy);
    assert_eq!(string(&copy), "fn f < 'x > (& self , a : P < 0usize >)");
    let mut copy: Signature = parse_quote!(fn f<'x>(&self, a: Wrap<'x>) -> Wrap<'x>);
    referrer
        .clone()
        .visitor(context, replace)
        .visit_signature_mut(&mut copy);
    assert_eq!(
        string(&copy),
        "fn f < 'x > (& self , a : Wrap < 'x >) -> P < 0usize >"
    );
    let mut copy = ty.clone();
    referrer
        .clone()
        .visitor(struct_context, replace)
        .visit_type_mut(&mut copy);
    assert_eq!(string(&copy), "P < 0usize >");
    assert_eq!(
        string(&referrer.expand(struct_context, ty, replace)),
        "P < 0usize >"
    );
}

/// F48: the mark is written after the context, and the parser reads what the serializer writes.
#[test]
fn f48_a_mark_is_serialized_and_read_back() {
    let tag = format!("\"type-leak-format-{}\"", Referrer::FORMAT_VERSION);
    let read = |parts: &str| -> std::result::Result<Referrer, String> {
        parse_str::<Referrer>(&format!("{} [{}] 2 (Rel) < 'x , 'y > ; ()", tag, parts))
            .map_err(|e| e.to_string())
    };
    for parts in [
        "type (R<'x>) 0 sig ('x)",
        "type (R<'x, 'y>) 1 sig ('x, 'y) predicate",
        "type (R<T>) 0 sig (elided) in (Wrap<R<T>>) 1",
        "type (R<T>) unwritable sig (elided) predicate in (Wrap<R<T>>) 0",
        "type (R<'x>) 0, type (R<'x>) 0 sig ('x), type (R) 1 sig (elided)",
    ] {
        let referrer = read(parts).unwrap_or_else(|e| panic!("{}: {}", parts, e));
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
    // A part with a mark is not the part without.
    assert_ne!(
        read("type (R<'x>) 0 sig ('x)").unwrap(),
        read("type (R<'x>) 0").unwrap()
    );
    for (parts, message) in [
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
            "expected `sig`, `predicate`, `in` or `,`",
        ),
    ] {
        let error = read(parts)
            .err()
            .unwrap_or_else(|| panic!("read: {}", parts));
        assert!(error.contains(message), "{}: {}", parts, error);
    }
}

// ── F49 ──────────────────────────────────────────────────────────────────────

/// F49 (fixed): the E0106 check ignored a lifetime that another input keeps.
///
/// Trigger: `fn m(x: &'a u8, y: Wrap<T>) -> &u8` in `trait Tr<'a, T: 'a>`. The one lifetime of the
/// inputs is `'a`, and the copy of `Wrap<T>` names `'a` in the arguments of its projection (`T:
/// 'a` is in its context). `Elision::added()` left out the lifetimes that the inputs keep, but
/// rustc counts the lifetimes of each input apart: with `'a` in `x` and in the projection in `y`,
/// two inputs have a lifetime and the copy fails with E0106.
///
/// What rustc does (stable 1.99.0 and nightly 1.101.0, the same on 1.56.1, 1.70.0 and 1.85.0 for the
/// cases below): the elided output is valid only if exactly one input has any lifetime (named,
/// `'static`, elided, or hidden by a path), and that input has exactly one distinct lifetime.
/// `x: &'a u8, y: Foo<'a>`, `x: Foo<'a>, y: Foo<'a>`, `x: &'a u8, y: Foo<'static>` and `x:
/// Foo2<'a, 'b>` fail; `x: Foo2<'a, 'a>` and `x: &'a Foo<'a>` do not. The same holds for a
/// projection `<X as R<(&'a (),)>>::Type`. A receiver that is a reference gives its lifetime to
/// the output whatever the others have (`self: Box<Self>` is an input like another), and `Fn(..)`
/// and a function pointer count their own inputs.
///
/// Fix: the type in an input that does not name the lifetime the inputs keep adds it too
/// (`Elision::for_input()`, per input of a method, a function pointer and `Fn(..)`; the Referrer
/// counts per argument as well: `SignatureFacts::for_input()`). The type is copied as written and
/// reported, as for F48 (`sig (elided)`), or the function pointer is leaked whole. The type in the
/// input that names it (`x: &'a Wrap<T>`, `x: (&'a u8, Wrap<T>)`) adds nothing.
///
/// The cross-crate fixture is 9204 (`tests/cross_crate.rs`).
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
        analyze_in(new_leaker(), &item),
        &["use absolute path: 'Wrap < T >'"; 2],
    );
    let note = "the output `& u8` has an elided lifetime, which elision takes from the only \
                lifetime of the inputs, but a leaked type in the inputs has `'a` among its \
                generics, and the copy names it in the arguments of its projection, so the inputs \
                have more than one lifetime, as rustc counts the lifetimes of each input apart \
                (E0106)";
    let whole = "it cannot be leaked as a whole because of `& u8`";
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
    let (_, referrer) = analyze_in(new_leaker(), &Item::Trait(controls)).unwrap();
    assert!(parts_of(&referrer).is_empty());
    let entries: Vec<String> = referrer.iter().map(string).collect();
    assert_eq!(entries, ["& 'a Wrap < T >", "Wrap < T >", "Wrap < u8 >"]);
}

/// F49: a function pointer and `Fn(..)` count the lifetimes of each input as well. A function
/// pointer in a field is leaked whole, so are the arguments of `Fn(..)` in `dyn Fn(..)`, and a
/// bound is an error.
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
    let entries: Vec<String> = referrer.iter().map(string).collect();
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
        analyze_in(new_leaker(), &Item::Trait(bound.clone())),
        &["use absolute path: 'Wrap < T >'"],
    );
    assert!(
        all_notes(&err)[0].ends_with("as rustc counts the lifetimes of each input apart (E0106)")
    );
}

// ── F50 ──────────────────────────────────────────────────────────────────────

/// The analysis of `item` with `check_values()` and the primitive types allowed.
fn f50_analyze(item: &ItemTrait, configure: impl Fn(&mut LeakerConfig)) -> LeakerResult {
    let mut config = LeakerConfig::new();
    config.check_values().allow_primitive();
    configure(&mut config);
    let mut leaker = Leaker::from_config(config);
    leaker.intern_trait(item, true);
    leaker.result()
}

/// The paths of the error `at` of `result`, as text.
fn f50_paths(result: &LeakerResult, at: usize) -> Vec<String> {
    result.errors()[at].paths.iter().map(string).collect()
}

/// F50 (fixed): the paths of the error of a path of a value left out the relative types in its
/// arguments.
///
/// Trigger: `core::mem::size_of::<*const Rel>()` in a default method, with `check_values()`.
/// Cause: the error of a relative path of a value (`ValueWalk::relative_part()`) had its own path
/// alone (`core::mem::size_of`), where the error of a type or a bound has all the relative paths in
/// it. The whole path is copied as written, `Rel` too, but a path rule after the analysis
/// (`allow_std_crates()`) dropped the error: `Rel` stayed relative in a copy that no crate can
/// resolve, whereas the config that allows the crates leaks it (the entry `Rel`). The same for
/// `Vec::<Rel>::new()` with `allow_prelude()` and `crate::base::bg::<*const R0>()` with
/// `allow_crate()`.
///
/// Fix: the paths of the error have the relative paths of the arguments, at any depth (not a
/// generic, `Self`, or a name the value declares). The probe for the nested errors allows them
/// with the path, so the arguments are nothing wrong inside it (`tests/nested_errors.rs`).
///
/// The cross-crate fixture is 9205 (`stages = true`: the copy after the rule is made in a crate that
/// does not name `Rel` only if the rule left no error), checked in `tests/cross_crate.rs`.
#[test]
fn f50_the_relative_types_in_the_arguments_of_a_value_path_are_paths_of_its_error() {
    // The value, the rule (`std`, `crate` or `prelude`) given to the result and to the config, the
    // paths of the error, the paths the rule leaves, and the entries of the config that has the rule.
    struct Case {
        item: ItemTrait,
        rule: &'static str,
        paths: &'static [&'static str],
        left: &'static [&'static str],
        entries: &'static [&'static str],
    }
    let case = |item, rule, paths, left, entries| Case {
        item,
        rule,
        paths,
        left,
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
            &["core :: mem :: size_of", "Rel"],
            &["Rel"],
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
            &["crate :: base :: bg", "R0"],
            &["R0"],
            &["R0"],
        ),
        // Deeper: the paths of the arguments of the arguments.
        case(
            parse_quote!(
                trait Tr {
                    fn f(&self) {
                        let _ = Vec::<Option<Rel>>::new();
                    }
                }
            ),
            "prelude",
            &["Vec :: new", "Option", "Rel"],
            &["Rel"],
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
            &["core :: mem :: size_of", "Rel"],
            &["Rel"],
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
            &["core :: convert :: From", "Rel"],
            &["Rel"],
            &["Rel"],
        ),
    ];
    for Case {
        item,
        rule,
        paths,
        left,
        entries,
    } in cases
    {
        let what = string(&item);
        let result = f50_analyze(&item, |_| ());
        assert_eq!(result.errors().len(), 1, "{}", what);
        assert_eq!(f50_paths(&result, 0), paths, "{}", what);
        // The rule leaves the error, with the paths it does not allow.
        let mut after = result.clone();
        match rule {
            "std" => after.allow_std_crates(),
            "crate" => after.allow_crate(),
            _ => after.allow_prelude(),
        };
        assert_eq!(after.errors().len(), 1, "{}", what);
        assert_eq!(f50_paths(&after, 0), left, "{}", what);
        // The config that has the rule leaks the types in the arguments: no error, and the entries.
        let config = f50_analyze(&item, |config| {
            match rule {
                "std" => config.allow_std_crates(),
                "crate" => config.allow_crate(),
                _ => config.allow_prelude(),
            };
        });
        assert!(config.errors().is_empty(), "{}", what);
        let referrer = config.into_referrer();
        let found: Vec<String> = referrer.iter().map(string).collect();
        assert_eq!(found, entries, "{}", what);
        // Allowing them too drops the error, and nothing is left inside the part.
        let mut all = after.clone();
        for path in left {
            all.allow_path(syn::parse_str(path).unwrap());
        }
        assert!(all.errors().is_empty(), "{}", what);
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
        let result = f50_analyze(&item, |_| ());
        assert_eq!(
            f50_paths(&result, 0),
            ["core :: mem :: size_of"],
            "{}",
            string(&item)
        );
    }
}

// ── F51 ──────────────────────────────────────────────────────────────────────

/// F51 (fixed): `enter_mod` with `nested_self_entries()` accepted a bound that names a leaked type
/// that names `Self`, which `intern_struct()` rejects.
///
/// Trigger: `struct S<T: Clone + AsRef<R1<Self>>> { f0: T, f1: R1<(Self, T)> }`. The entry
/// `R1<Self>` is in a bound, and `intern_struct()` reports it (`UnprovableBound`): the context of
/// the entry `R1<(Self, T)>` has the bound, and the copy has only its rewritten form in the
/// environment. Nested, the entry `R1<Self>` is for any type and its context has no generic (`T`
/// is not in the type), so the check of the bounds of its context found none; the copy failed (E0277
/// `T: AsRef<R1<S<T>>>`, for the entry in the last field).
///
/// Fix: an entry in a predicate that is for any type because it names `Self` is checked with the
/// context of the type of `Self`, as `intern_*` has it (`nested_self_cycle()`). Like
/// `intern_struct()`, it rejects the item also when no entry uses `T`.
///
/// The cross-crate fixture is 9206 (`tests/cross_crate.rs`).
#[test]
fn f51_a_nested_item_with_a_bound_that_names_a_leaked_type_is_rejected() {
    let items: Vec<(Item, &str)> = vec![
        (
            parse_quote!(
                pub struct S<T: Clone + AsRef<R1<Self>>> {
                    pub f0: T,
                    pub f1: R1<(Self, T)>,
                }
            ),
            "S",
        ),
        // A where predicate, and no entry that uses `T`.
        (
            parse_quote!(
                pub struct S<T>(pub T)
                where
                    T: AsRef<R1<Self>>;
            ),
            "S",
        ),
        (
            parse_quote!(
                pub enum E<T: AsRef<R1<Self>>> {
                    A(T, R1<(Self, T)>),
                }
            ),
            "E",
        ),
    ];
    let message = "`R1 < Self >` is in a bound or where predicate, and a bound of its generics \
                   names a leaked type";
    for (item, name) in &items {
        for nest in NESTS {
            let what = format!("{:?} `{}`", nest, string(item));
            let result = nested_analysis_in(nested_self_leaker(), nest, item, &[name]);
            let errors = result.errors();
            assert_eq!(errors.len(), 1, "{}", what);
            assert_eq!(errors[0].reason, Reason::UnprovableBound, "{}", what);
            assert_eq!(errors[0].message, message, "{}", what);
        }
    }
    // `intern_struct()` reports the same.
    let item: ItemStruct = parse_quote!(
        pub struct S<T: Clone + AsRef<R1<Self>>> {
            pub f0: T,
            pub f1: R1<(Self, T)>,
        }
    );
    let mut leaker = nested_self_leaker();
    leaker.intern_struct(&item, true);
    let direct = leaker.result();
    assert_eq!(direct.errors().len(), 1);
    assert_eq!(direct.errors()[0].message, message);

    // Controls: a bound that names `Self` without a leaked type is accepted, as F42 says.
    let item: Item = parse_quote!(
        pub struct S<T: Clone + AsRef<Self>> {
            pub f0: T,
            pub f1: R1<(Self, T)>,
        }
    );
    for nest in NESTS {
        let referrer = nested_analysis_in(nested_self_leaker(), nest, &item, &["S"])
            .finish()
            .unwrap_or_else(|e| panic!("{:?}: {}", nest, joined_messages(&e)));
        let entries: Vec<String> = referrer.iter().map(string).collect();
        assert_eq!(entries, ["R1 < (Self , T) >"], "{:?}", nest);
    }
}

// ── F52 ──────────────────────────────────────────────────────────────────────

/// The entries of `referrer` as text, `?` before an optional one.
fn f52_entries(referrer: &Referrer) -> Vec<String> {
    referrer
        .iter()
        .enumerate()
        .map(|(ix, ty)| format!("{}{}", if referrer.must(ix) { "" } else { "?" }, string(ty)))
        .collect()
}

/// F52 (fixed): an unrelated error made the optional entries of the predicates of a trait entries.
///
/// Trigger: `trait Tr<T> where T: Fn(&::l::B2<T, ::l::B0>) -> u8 { type A0; fn m<Y>(&self) where
/// Y: RelTr<Self::A0>; }`, with a strict Referrer. `::l::B0` is an optional entry (a crate that the
/// Leaker does not allow). `Self::A0` in a predicate is resolved by the impl of the host with the
/// bounds of the generics as they are, so the optional entries in them stay as written
/// (`not_optional_in_bounds()`, F2, F17): beside `Y: Into<Self::A0>` there is no entry. But the
/// bound of `RelTr<Self::A0>` is an error (`RelTr` is relative) and is copied as written whole, and
/// the visit that made the entries stay did not run in it: the strict Referrer replaced `::l::B0`
/// and the host impl failed (E0277 "expected an `Fn(&B2<T, B0>)` closure, found `T`").
///
/// Fix: the probe that looks inside the part keeps what the `Self` and `Self::A` in it make not
/// optional (`AnalyzeVisitor::probe_kept`). Limit: a part with no relative path to probe for (an
/// error of a lifetime, say) does not.
///
/// The cross-crate fixture is 9207 (`strict`, `tests/cross_crate.rs`).
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
        let mut leaker = new_leaker();
        leaker.intern_trait(&item(method), true);
        let result = leaker.result();
        (result.errors().len(), f52_entries(&result.into_referrer()))
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
        (1, vec!["?:: ext :: B0".to_string()])
    );
    assert_eq!(entries(""), (0, vec!["?:: ext :: B0".to_string()]));
    // The path of the part allowed in the config: it is not an error, and `Self::A0` is visited.
    let mut config = LeakerConfig::new();
    config
        .allow_primitive()
        .allow_prelude()
        .allow_path(parse_quote!(RelTr));
    let mut leaker = Leaker::from_config(config);
    leaker.intern_trait(&item("fn m<Y>(&self) where Y: RelTr<Self::A0>;"), true);
    let result = leaker.result();
    assert!(result.errors().is_empty());
    assert!(f52_entries(&result.into_referrer()).is_empty());
}

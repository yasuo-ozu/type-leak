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
//! | `f18_strict_referrer_replaces_anchor_of_method_lifetime` | a strict Referrer replaces the optional type that holds a method lifetime | limitation |
//! | `f19_impl_self_in_bound_of_entry_in_predicate` | an entry in a predicate has a bound of its context that names `Self`, in an impl whose self type is leaked | fixed |
//! | `f20_trait_assoc_in_bound_of_entry_in_predicate` | an entry in a predicate has a bound of its context that names `Self::A`, in a trait | fixed |
//! | `f21_iterator_item_in_bound_of_entry_in_item_binding` | an entry in `T: Iterator<Item = ..>` has a bound of its context that names the `Item` | fixed |
//! | `f22_method_lifetime_in_free_function_and_inherent_method` | the E0195 check of F7 also rejects a free function and an inherent method | fixed |
//! | `f23_self_in_elided_signature_where_referrer_can_rewrite` | `Self` in an elided signature is rejected in a bound and with `allow_self = false`, where the Referrer could write `for<'e>` | fixed |
//! | `f24_self_in_predicate_with_leaked_bound_of_unused_generic` | a bare `Self` in a predicate is rejected for a leaked bound of a generic that no entry of the self type uses | fixed |
//! | `f25_supertrait_item_in_bound_of_entry_in_predicate` | a supertrait fixes the `Item` that a bound of an entry's generic names | fixed |
//! | `f26_alias_hides_projection_in_bound_of_entry_in_predicate` | a type alias hides the projection that such a bound names | limitation |
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
//!
//! No finding is a `bug` any more: each is `fixed` (a regression test) or a `limitation` (it pins
//! behaviour that the docs of `Leaker` document, or that follows from what rustc accepts). No test
//! is `#[ignore]`d.
//!
//! Most fixes are rejections: the Leaker returns an error for an item whose copy rustc cannot
//! compile. The test of such an item asserts the rejection, with the exact headline of the error and
//! its note (`assert_rejected()`, `assert_unstable()`), and not an oracle on the copy: an oracle
//! that runs for the accepted items only (`let Ok(..) = .. else { continue }`) runs for none of
//! them, and would pass if the rejection regressed. The oracle (what the copy of an accepted item
//! must not have) is kept for the controls, the items that compile and stay accepted, and it is
//! checked on a copy written by hand (`the_oracles_of_the_findings_find_a_hand_written_cycle`).
//!
//! The test of a finding says how the property test shows it (`TYPE_LEAK_PROP_FEATURES=..`). The
//! features of the fixed findings are on by default in the property test: the Leaker now rejects
//! what they generate (the rejections are counted), or rewrites it. The feature of F26 is off, as
//! for the other limitations. F22 to F24 were rejections of what compiles: a rejection is
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
//! The feature of the limitation F18 (`strict_anchors`) is off in
//! the property test: the generator does not make what it fails on (it only decides that
//! an item is not invoked as strict).

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
    config.allow_primitive().allow_std_prelude();
    Leaker::from_config(config)
}

/// Analyzes `item` with `intern_*`.
fn intern(leaker: &mut Leaker, item: &Item) -> std::result::Result<ContextId, NotInternableError> {
    match item {
        Item::Struct(item) => leaker.intern_struct(item, true),
        Item::Trait(item) => leaker.intern_trait(item, true),
        Item::Impl(item) => leaker.intern_impl(item, true),
        Item::Mod(item) => leaker.intern_mod(item),
        _ => panic!("unsupported item"),
    }
}

fn string<T: template_quote::ToTokens>(t: &T) -> String {
    quote!(#t).to_string()
}

/// Asserts that the Leaker rejected `item`, with exactly the `messages` (the headline of each error,
/// in order: the `Display` of the error is these, one per line), and returns the error.
///
/// A bug input has to be rejected. A loop that skips the rejected items (`let Ok(..) = .. else {
/// continue }`) runs its oracle on none of them, and pins nothing: it would be the same if the
/// rejection regressed, and the oracle runs for the items that are accepted only.
fn assert_rejected<T>(
    item: &impl template_quote::ToTokens,
    result: std::result::Result<T, NotInternableError>,
    messages: &[&str],
) -> NotInternableError {
    let err = match result {
        Ok(_) => panic!("`{}` is accepted, and it does not compile", string(item)),
        Err(err) => err,
    };
    assert_eq!(err.to_string(), messages.join("\n"), "`{}`", string(item));
    err
}

/// The notes of all the errors of `err`, in order.
fn all_notes(err: &NotInternableError) -> Vec<String> {
    err.items()
        .flat_map(|item| item.notes.into_iter().map(|(_, note)| note))
        .collect()
}

/// The helps of all the errors of `err`, in order.
fn all_helps(err: &NotInternableError) -> Vec<String> {
    err.items()
        .filter_map(|item| item.help.map(|(_, help)| help))
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
    result: std::result::Result<T, NotInternableError>,
    ty: &str,
    bound: &str,
    blames: Blames,
) -> NotInternableError {
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
        // A rejection is fine.
        if leaker.intern_struct(&item, true).is_err() {
            continue;
        }
        let referrer = leaker.finish();
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
            intern(&mut new_leaker(), item),
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
    let mut leaker = new_leaker();
    let context = intern(&mut leaker, &control)
        .unwrap_or_else(|e| panic!("the control is rejected, though it compiles: {}", e));
    let referrer = leaker.finish();
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
        let mut leaker = new_leaker();
        let context = intern(&mut leaker, &item).unwrap();
        let referrer = leaker.finish();
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
        let context = leaker.intern_mod(&item).unwrap();
        let referrer = leaker.finish();
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
    leaker.intern_mod(&item).unwrap();
    let referrer = leaker.finish();
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
    leaker.intern_impl(&item, true).unwrap();
    let referrer = leaker.finish();
    let ix = entry(&referrer, "E < 'a , T >");
    let generics = referrer.generics(ix);
    assert_eq!(string(generics), "< 'a , T >");
    assert_eq!(encode(&referrer, ix), parse_quote!((&'a (), T)));

    // With the bound written: recorded, and encoded.
    let item: ItemImpl = parse_quote! {
        impl<'a, T: 'a> E<'a, T> { pub fn f(&self) -> Rel { loop {} } }
    };
    let mut leaker = new_leaker();
    leaker.intern_impl(&item, true).unwrap();
    let referrer = leaker.finish();
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
/// The assertion: the Leaker rejects the impl with a `NotInternableError` (also if only a bound of
/// the generics of the entry names `Self::A`), and says so in the headline. A type that names
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
    let result = leaker.intern_impl(&item, true);
    let err = match result {
        Ok(_) => panic!(
            "accepted, with the entries {:?}",
            leaker.finish().iter().map(string).collect::<Vec<_>>()
        ),
        Err(err) => err,
    };
    // The headline says what to write, with the trait of the impl.
    assert_eq!(
        err.to_string(),
        "write `<Self as ::leaker::m::Tr>::A` instead of `Self::A` in `Rel<Self::A>`: the copy \
         cannot tell which trait `A` comes from"
    );
    assert_eq!(
        err.help().as_deref(),
        Some("replace `Self::A` with `<Self as ::leaker::m::Tr>::A`")
    );

    // A bound of the generics of the entry that names it.
    let item: ItemImpl = parse_quote! {
        impl<T: ::p::Tr<Self::A>> ::leaker::m::Tr for W<T> {
            type A = u8;
            fn f(&self, x: Rel<T>) {}
        }
    };
    let err = new_leaker().intern_impl(&item, true).unwrap_err();
    let messages: Vec<String> = err.items().map(|item| item.message).collect();
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
        new_leaker().intern_impl(&item, true).unwrap();
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
    leaker.intern_impl(&item, true).unwrap();
    let mut entries: Vec<String> = leaker.finish().iter().map(string).collect();
    entries.sort();
    assert_eq!(entries, ["Rel", "Rel2"]);
    let item: ItemTrait = parse_quote! {
        pub trait Tr { type A; fn f(&self, x: Rel<Self::A>); }
    };
    let mut leaker = new_leaker();
    leaker.intern_trait(&item, true).unwrap();
    assert_eq!(
        string(leaker.finish().iter().next().unwrap()),
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
/// The assertion: the Leaker rejects those traits with a `NotInternableError` that has the E0195
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
        let err = new_leaker().intern_trait(&item, true).unwrap_err();
        let note = err.note().unwrap();
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
        new_leaker().intern_trait(&item, true).unwrap();
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
    leaker.intern_struct(&item, true).unwrap();
    let referrer = leaker.finish();
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
        leaker.intern_struct(item, true).unwrap();
        let referrer = leaker.finish();
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
            leaker.intern_struct(item, true).unwrap();
            let referrer = leaker.finish();
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
    let context = leaker.intern_trait(&item, true).unwrap();
    let referrer = leaker.finish();
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
    let context = leaker.intern_trait(&item, true).unwrap();
    let referrer = leaker.finish();
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
    let context = leaker.intern_mod(&item).unwrap();
    let referrer = leaker.finish();
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
    let context = leaker.intern_mod(&item).unwrap();
    let referrer = leaker.finish();
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
fn f12_copy(item: &ItemTrait) -> std::result::Result<(Vec<String>, ItemTrait), NotInternableError> {
    f12_copy_with(new_leaker(), item)
}

/// Like `f12_copy()`, with the Leaker `leaker`.
fn f12_copy_with(
    mut leaker: Leaker,
    item: &ItemTrait,
) -> std::result::Result<(Vec<String>, ItemTrait), NotInternableError> {
    let slots = RepeaterSlots::new(parse_quote!(crate::Repeater));
    let context = leaker.intern_trait(item, true)?;
    let referrer = leaker.finish();
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
        let note = err.note().unwrap();
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
    config.allow_primitive().allow_std_prelude();
    config.allowed_paths.push(parse_quote!(::p));
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
        let err = match leaker.intern_impl(item, true) {
            Ok(_) => panic!(
                "`{}` is accepted, and the entries {:?} are their own `Self`: the Referrer has no \
                 type to write for it",
                string(item),
                self_referential_entries(&leaker.finish())
            ),
            Err(err) => err,
        };
        let headline = "is in the self type of the impl, and a bound of its generics names `Self`";
        let (shown, note) = (err.to_string(), err.note().unwrap());
        assert!(shown.contains(headline), "{}: {}", string(item), shown);
        assert!(
            note.contains("has no type to write for `Self`"),
            "{}: {}",
            string(item),
            note
        );
        // A rejected scope leaves nothing behind.
        assert_eq!(leaker.finish().iter().count(), 0, "{}", string(item));
        // Also in a module, and as a nested item.
        let module: ItemMod = parse_quote!(mod m { #item });
        let err = f13_leaker().intern_mod(&module).err();
        assert!(
            err.map_or(false, |err| err.to_string().contains(headline)),
            "{}",
            string(item)
        );
        let mut leaker = f13_leaker();
        let outcome = leaker.enter_generics(&Generics::default(), |view| {
            view.visit_item(&Item::Impl(item.clone()))
        });
        assert!(
            outcome
                .err()
                .map_or(false, |err| err.to_string().contains(headline)),
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
    let err = f13_leaker().intern_impl(&item, true).err();
    assert!(err.map_or(false, |err| err
        .to_string()
        .contains("`& 'a (Rel2 , T)` is in the self type of the impl")));
    for item in copied.iter().chain(&controls) {
        let mut leaker = f13_leaker();
        let context = leaker
            .intern_impl(item, true)
            .unwrap_or_else(|e| panic!("`{}` is rejected: {}", string(item), e));
        let referrer = leaker.finish();
        assert!(
            self_referential_entries(&referrer).is_empty(),
            "{}",
            string(item)
        );
        // A strict Referrer replaces the optional entries too, but not those in the self type.
        let mut copy = Item::Impl(item.clone());
        referrer
            .allow_ambiguous_crate_name(false)
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
    let context = leaker.intern_impl(&item, true).unwrap();
    let referrer = leaker.finish();
    assert_eq!(
        referrer.iter().map(|t| string(&t)).collect::<Vec<_>>(),
        [":: ext :: S < T >"]
    );
    let mut copy = Item::Impl(item);
    referrer
        .allow_ambiguous_crate_name(false)
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
fn f14_copy(item: &ItemImpl, strict: bool) -> std::result::Result<ItemImpl, NotInternableError> {
    let mut config = LeakerConfig::new();
    config.allow_primitive().allow_std_prelude();
    config.allowed_paths.push(parse_quote!(::p));
    let mut leaker = Leaker::from_config(config);
    let context = leaker.intern_impl(item, true)?;
    let mut referrer = leaker.finish();
    if strict {
        referrer = referrer.allow_ambiguous_crate_name(false);
    }
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
                    e
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
        let copy = f14_copy(item, *strict)
            .unwrap_or_else(|e| panic!("`{}` is rejected, and it compiles: {}", string(item), e));
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
) -> std::result::Result<(Vec<String>, Item), NotInternableError> {
    let (ident, generics) = match item {
        Item::Struct(i) => (&i.ident, &i.generics),
        Item::Enum(i) => (&i.ident, &i.generics),
        Item::Union(i) => (&i.ident, &i.generics),
        _ => panic!("unsupported item"),
    };
    let mut leaker = new_leaker();
    let context = match item {
        Item::Struct(i) => leaker.intern_struct(i, allow_self)?,
        Item::Enum(i) => leaker.intern_enum(i, allow_self)?,
        Item::Union(i) => leaker.intern_union(i, allow_self)?,
        _ => unreachable!(),
    };
    let referrer = leaker.finish();
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
                    e
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
        let note = err.note().unwrap();
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
        let (entries, copy) = f15_copy(&item, true)
            .unwrap_or_else(|e| panic!("`{}` is rejected, and it compiles: {}", string(&item), e));
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
        let note = err.note().unwrap();
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
        let mut leaker = new_leaker();
        let context = intern(&mut leaker, &item).unwrap();
        let referrer = leaker.finish();
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
        let context = leaker
            .intern_trait(item, true)
            .unwrap_or_else(|e| panic!("`{}` is rejected: {}", string(item), e));
        let referrer = leaker.finish();
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
    leaker.intern_trait(&control, true).unwrap();
    let referrer = leaker.finish();
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
        // Rejected is fine.
        if leaker.intern_struct(item, true).is_err() {
            continue;
        }
        let referrer = leaker.finish();
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
        leaker
            .intern_struct(item, true)
            .unwrap_or_else(|e| panic!("`{}` is rejected: {}", string(item), e));
        let referrer = leaker.finish();
        assert!(
            entries_in_own_context(&referrer).is_empty(),
            "{}",
            string(item)
        );
    }
}

// ── F18 ──────────────────────────────────────────────────────────────────────

/// F18 (limitation): a strict Referrer replaces the optional entry that holds a lifetime of a
/// method, so the copy does not have it where the original did.
///
/// `Leaker::allowed_paths` docs: an absolute path of a crate that the Leaker does not allow is an
/// optional entry, which a Referrer leaves as written by default and replaces if it is strict
/// (`allow_ambiguous_crate_name(false)`); "replacing a type can lose an anchor or a method
/// lifetime" (`Referrer::must()`). Trigger:
/// `trait Tr { fn m<'x>(&self, a: ::ext::W<&'x u8>) -> Rel<'x>; }`. In the trait, `'x` is
/// late-bound (the argument holds it, and no bound names it). The Leaker counts the optional type
/// as an anchor, as the default Referrer keeps it. The strict copy is
/// `fn m<'x>(&self, a: P<1>) -> P<0>`: both are projections, which do not constrain `'x`, so it is
/// early-bound, and rustc rejects the impl of the Referrer's crate, where the trait has it
/// late-bound (E0195 "lifetime parameters differ in whether they are early- or late-bound"). Found
/// by the property test with `TYPE_LEAK_PROP_FEATURES=strict_anchors`, for the items that the
/// generator invokes as strict (25% of them): the Leaker's analysis cannot know what the Referrer
/// chooses.
///
/// It is a limitation and not a bug: counting the optional type as an anchor is what lets the
/// default Referrer (and a Referrer of the same dependency graph) copy the item; the other choice
/// would reject the item for every Referrer. Instead, allow the crate (`LeakerConfig::allowed_paths`),
/// so that the type is not an entry for any Referrer, or keep the anchor in a required type, or use
/// a Referrer that is not strict. The test pins the two copies.
#[test]
fn f18_strict_referrer_replaces_anchor_of_method_lifetime() {
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
        let context = leaker.intern_trait(&item, true).unwrap();
        let mut referrer = leaker.finish();
        if strict {
            referrer = referrer.allow_ambiguous_crate_name(false);
        }
        let mut copy = item.clone();
        referrer
            .visitor(context, |_, ix| parse_quote!(P<#ix>))
            .visit_item_trait_mut(&mut copy);
        copy
    };
    let mut config = LeakerConfig::new();
    config.allow_primitive().allow_std_prelude();

    // The default Referrer keeps the optional type, which holds `'x`.
    assert_eq!(held(&copy_of(config.clone(), false)), ["'x"]);
    // The strict one replaces it: nothing of the arguments holds `'x`.
    assert_eq!(held(&copy_of(config.clone(), true)), Vec::<String>::new());
    // With the crate allowed it is not an entry, for either.
    config.allowed_paths.push(parse_quote!(::ext));
    assert_eq!(held(&copy_of(config.clone(), false)), ["'x"]);
    assert_eq!(held(&copy_of(config, true)), ["'x"]);
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
fn f19_copy(item: &Item) -> std::result::Result<(Referrer, Item), NotInternableError> {
    f19_copy_with(new_leaker(), item)
}

/// Like `f19_copy()`, with the Leaker that `f13_leaker()` makes (it allows `::p`, so that the self
/// type of an impl has parts that are entries and a part that is not).
fn f19_copy_with(
    mut leaker: Leaker,
    item: &Item,
) -> std::result::Result<(Referrer, Item), NotInternableError> {
    let context = match item {
        Item::Impl(i) => leaker.intern_impl(i, true)?,
        Item::Trait(i) => leaker.intern_trait(i, true)?,
        _ => panic!("unsupported item"),
    };
    let referrer = leaker.finish();
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
        let (referrer, copy) = f19_copy(item)
            .unwrap_or_else(|e| panic!("`{}` is rejected, and it compiles: {}", string(item), e));
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
    ];
    for item in &by_parts_controls {
        let (referrer, copy) = f19_copy_with(f13_leaker(), item)
            .unwrap_or_else(|e| panic!("`{}` is rejected, and it compiles: {}", string(item), e));
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
        // The `Item` that a predicate equates is normalized too (E0277, "expected an `Fn(u8)`
        // closure, found `I`").
        (
            parse_quote!(
                impl<I: Iterator<Item = u8> + Fn(I::Item), V: AsRef<Self>> ::p::W<Rel<I>, I, V> {}
            ),
            "I : Fn (I :: Item) + Iterator < Item = u8 >",
            Blames::Projection("I :: Item"),
        ),
    ];
    for (item, bound, blames) in &normalized {
        assert_unstable(
            item,
            f13_leaker().intern_impl(item, true),
            "Self",
            bound,
            *blames,
        );
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
        let (referrer, copy) = f19_copy(item)
            .unwrap_or_else(|e| panic!("`{}` is rejected, and it compiles: {}", string(item), e));
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
        Ok::<_, NotInternableError>((copy, found))
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
        let (copy, found) = check(item)
            .unwrap_or_else(|e| panic!("`{}` is rejected, and it compiles: {}", string(item), e));
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
        let err = intern(&mut f13_leaker(), item).unwrap_err();
        let note = err.note().unwrap();
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
        if let Err(err) = intern(&mut f13_leaker(), item) {
            panic!(
                "`{}` is rejected, and its copy compiles: {}\n{}",
                string(item),
                err,
                err.note().unwrap_or_default()
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
        f13_leaker().enter_mod("m", [], |view| view.visit_item(&Item::Impl(item.clone())))
    };
    nested(&inherent).unwrap();
    assert!(nested(&of_trait)
        .unwrap_err()
        .note()
        .unwrap()
        .contains("(E0195)"));
    let err = f13_leaker()
        .enter_generics(&Generics::default(), |view| view.visit_signature(&method))
        .unwrap_err();
    assert!(err.note().unwrap().contains("(E0195)"));
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
                e,
                e.note().unwrap_or_default()
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
                    e,
                    e.note().unwrap_or_default()
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
) -> std::result::Result<(Item, Vec<String>), NotInternableError> {
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
/// Leaker cannot tell them from `WIter`: rejecting them is the price. A supertrait that equates
/// the associated type of its own supertrait (`trait Sub: Cont<Out = u8>`) fails as `WIter` does.
///
/// The property test shows it with `TYPE_LEAK_PROP_FEATURES=supertrait_item` (on by default since
/// the fix; before it, seeds 1, 2 and 3 of 60 cases failed with E0277 "expected an `FnOnce(u8)`
/// closure, found `X`" in the Referrer's crate).
///
/// The assertion is that of F21: if the Leaker accepts the item, no entry that a predicate of the
/// copy names has a bound of its context that names an `Item`.
///
/// Fixed by the first alternative of the issue, rejecting: a projection (`T::Item`, `<T as
/// Iterator>::Item`) of a generic that has a bound of a trait that is not one of the standard library
/// (`unstable.rs`, `bindings()`: `T: ::p::WIter`, `T: Tr`, also in a where predicate or an
/// enclosing item) counts as an associated type that rustc normalizes, as one that a bound equates.
/// A bound of the standard library (`Iterator`, `Clone`, `Fn(..)`, a path in `::core`, `::std` or
/// `::alloc`) does not.
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
        f19_copy_with(f13_leaker(), item)
            .unwrap_or_else(|e| panic!("`{}` is rejected, and it compiles: {}", string(item), e));
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
        f19_copy_with(f13_leaker(), item)
            .unwrap_or_else(|e| panic!("`{}` is rejected, and it compiles: {}", string(item), e));
    }
    // The price: the supertrait does not fix the `Item`, or the trait has an associated type of its
    // own, and the copy compiles as well, but the Leaker cannot tell.
    let price: Vec<Item> = vec![
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
    ];
    let priced = [
        (
            "Rel < T >",
            "T : :: p :: WIter2 + Fn (T :: Item)",
            "T :: Item",
        ),
        ("Rel < T >", "T : :: p :: Cont + Fn (T :: Out)", "T :: Out"),
    ];
    assert_eq!(price.len(), priced.len());
    for (item, (ty, bound, path)) in price.iter().zip(priced) {
        assert_unstable(
            item,
            f19_copy_with(f13_leaker(), item),
            ty,
            bound,
            Blames::Projection(path),
        );
    }
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
    // Each is rejected: no copy of it exists whose predicate names such an entry.
    let blamed = [
        (
            "Rel < T >",
            "T : :: p :: WIter + Fn (T :: Item)",
            Blames::Projection("T :: Item"),
        ),
        (
            "Rel < T >",
            "T : :: p :: WIter + Fn (< T as Iterator > :: Item)",
            Blames::Projection("< T as Iterator > :: Item"),
        ),
        (
            "Rel < T >",
            "T : :: p :: WIter + Fn (T :: Item)",
            Blames::Projection("T :: Item"),
        ),
        (
            "Rel < T >",
            "T : :: p :: WIter + Fn (T :: Item)",
            Blames::Projection("T :: Item"),
        ),
        (
            "Rel < T >",
            "T : :: p :: WIter + Fn (T :: Item)",
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
}

/// F26 (limitation): a type alias hides the projection that a bound names.
///
/// Trigger: `trait Tr<T: Iterator<Item = u8>> where T: Fn(::p::WIt<T>) { fn m(&self) where Rel<T>:
/// Default; }` with `type WIt<T> = <T as Iterator>::Item;` in the Leaker's crate. It is F21 and F25
/// with the `Item` written through an alias. With the projection written (`T: Fn(T::Item)`) the
/// Leaker rejects the item; with the alias it accepts it, and the copy is E0277 "expected an `Fn(u8)`
/// closure, found `T`" as in F25 (checked with rustc, stable 1.98.1 and nightly 1.100.0, with the
/// Leaker's crate and the Referrer's with the impl of the trait for a host type). `T: Iterator` with
/// no equation compiles with the alias, as it does with the path (the alias is the projection of a
/// rigid type).
///
/// The property test shows it with `TYPE_LEAK_PROP_FEATURES=alias_item` (off by default: it is a
/// limitation; seeds 1, 2 and 3 of 60 cases fail with the same E0277 as F25).
///
/// No correct behaviour is possible: the Leaker sees `::p::WIt<T>` as a path with `T` as its
/// argument, which a struct `::p::WFoo<T>` is as well, and
/// `T: Iterator<Item = u8> + Fn(::p::WFoo<T>)` with `Rel<T>: Default` compiles (checked: the same
/// impl copy). Rejecting every path that names `T` in the bounds of a generic that an equation
/// fixes, where an entry that uses it is in a predicate, would reject what compiles. The `Leaker`
/// docs say that a `::core` or allowed path is assumed to be a struct, an enum or a union (an alias
/// that drops a lifetime, or hides a projection, is not seen), so allow only crates without such
/// aliases, or write the projection. The test pins the acceptance, and that the written projection
/// is the one that is found.
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
    let generics = string(referrer.generics(ix));
    assert!(
        generics.contains("Fn (:: p :: WIt < T >)"),
        "the context of the entry has the alias as written: {}",
        generics
    );
    assert!(
        !generics_name_item(referrer.generics(ix)),
        "the Leaker finds no `Item` in {}",
        generics
    );
    assert!(has_replacement_in_predicate(&copy), "{}", string(&copy));
    // The projection written, which it finds.
    let written: Item = parse_quote!(
        pub trait Tr<T: Iterator<Item = u8>>
        where
            T: Fn(T::Item),
        {
            fn m(&self)
            where
                Rel<T>: Default;
        }
    );
    assert_unstable(
        &written,
        f19_copy_with(f13_leaker(), &written),
        "Rel < T >",
        "T : Fn (T :: Item) + Iterator < Item = u8 >",
        Blames::Projection("T :: Item"),
    );
    // A struct with the same shape compiles, and cannot be told from the alias.
    let structure: Item = parse_quote!(
        pub trait Tr<T: Iterator<Item = u8>>
        where
            T: Fn(::p::WFoo<T>),
        {
            fn m(&self)
            where
                Rel<T>: Default;
        }
    );
    f19_copy_with(f13_leaker(), &structure).unwrap();
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
        f19_copy_with(f13_leaker(), item)
            .unwrap_or_else(|e| panic!("`{}` is rejected, and it compiles: {}", string(item), e));
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
        f19_copy_with(f13_leaker(), item)
            .unwrap_or_else(|e| panic!("`{}` is rejected, and it compiles: {}", string(item), e));
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
        let (entries, _) = f12_copy_with(f13_leaker(), item)
            .unwrap_or_else(|e| panic!("`{}` is rejected, and it compiles: {}", string(item), e));
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
            .unwrap_or_else(|e| panic!("`{}` is rejected: {}", string(item), e));
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
        leaker.intern_trait(&item, true).unwrap();
        let referrer = leaker.finish();
        let ix = entry(&referrer, "RelRef < u8 >");
        assert!(
            referrer.generics(ix).params.is_empty(),
            "`{}`: the entry has the generics `{}`",
            string(&item),
            string(referrer.generics(ix))
        );
    }
    let err = new_leaker()
        .intern_trait(
            &parse_quote!(
                pub trait Tr2 {
                    fn f(&self, x: fn(RelRef<'_, u8>) -> &u8);
                }
            ),
            true,
        )
        .unwrap_err();
    assert!(
        err.note().unwrap().contains("because of `'_`"),
        "{}",
        err.note().unwrap()
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
    leaker.intern_trait(&control, true).unwrap();
    let referrer = leaker.finish();
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
        let context = leaker
            .intern_trait(item, true)
            .unwrap_or_else(|e| panic!("`{}` is rejected: {}", string(item), e));
        let referrer = leaker.finish();
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
    leaker.intern_trait(&item, true).unwrap();
    let referrer = leaker.finish();
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
    leaker.intern_impl(&item, true).unwrap();
    let referrer = leaker.finish();
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
    leaker.intern_struct(&item, true).unwrap();
    let referrer = leaker.finish();
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
    let context = leaker.intern_trait(&tr, true).unwrap();
    let referrer = leaker.finish();
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
        f19_copy_with(f13_leaker(), item)
            .unwrap_or_else(|e| panic!("`{}` is rejected, and it compiles: {}", string(item), e));
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

    // F21, F25 and F33: the bound of the entry names an `Item`.
    let control: Item = parse_quote!(
        pub trait Tr<T: ::p::WIter>
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
        pub trait Tr<T: ::p::WIter>
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
        pub trait Tr<T: ::p::WIter>
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
        let context = leaker
            .intern_trait(item, true)
            .unwrap_or_else(|e| panic!("`{}` is rejected: {}", string(item), e));
        let referrer = leaker.finish();
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
    leaker.intern_trait(&control, true).unwrap();
    let referrer = leaker.finish();
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
        let context = leaker
            .intern_trait(item, true)
            .unwrap_or_else(|e| panic!("`{}` is rejected: {}", string(item), e));
        let referrer = leaker.finish();
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
    let mut leaker = new_leaker();
    let context = match nest {
        Nest::EnterMod => {
            leaker
                .enter_mod("m", names(declared), |view| view.visit_item(item))
                .unwrap()
                .0
        }
        Nest::InternMod => leaker.intern_mod(&parse_quote!(mod m { #item })).unwrap(),
        Nest::InternModNested => leaker
            .intern_mod(&parse_quote!(mod m { mod n { #item } }))
            .unwrap(),
        Nest::EnterGenerics => {
            leaker
                .enter_generics(&Generics::default(), |view| view.visit_item(item))
                .unwrap()
                .0
        }
    };
    let referrer = leaker.finish();
    let (mut visitor, calls) = nested_visitor(&referrer, context);
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
    let (context, ()) = leaker
        .enter_mod("m", names(&["S"]), |view| view.visit_item(&item))
        .unwrap();
    let referrer = leaker.finish();
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
    let (_, context) = leaker
        .enter_mod("m", names(&[]), |view| {
            view.enter_impl(&Generics::default(), &parse_quote!(::p::W), true, |view| {
                view.visit_type(&parse_quote!(Rel2<Self>));
                view.visit_item(&nested);
            })
            .unwrap()
            .0
        })
        .unwrap();
    let referrer = leaker.finish();
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
    let err = leaker
        .enter_type(&ident, &generics, false, |view| {
            view.visit_type(&parse_quote!(Rel3<Self>))
        })
        .unwrap_err();
    assert_eq!(err.to_string(), "use absolute path: 'Rel3 < Self >'");
    // The item nested in it is not.
    let mut leaker = new_leaker();
    let (context, ()) = leaker
        .enter_type(&ident, &generics, false, |view| {
            view.visit_type(&parse_quote!(Vec<Self>));
            view.visit_item(&nested);
        })
        .unwrap();
    let referrer = leaker.finish();
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
    let context = leaker.intern_mod(&outer).unwrap();
    let referrer = leaker.finish();
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
        let (context, ()) = leaker
            .enter_mod("m", names(&[name]), |view| view.visit_item(&item))
            .unwrap();
        let referrer = leaker.finish();
        let (mut visitor, calls) = nested_visitor(&referrer, context);
        let mut file = file_of(vec![item.clone()]);
        visitor.visit_file_mut(&mut file);
        let expected: Item = syn::parse_str(expected).unwrap();
        assert_eq!(string(&file.items[0]), string(&expected));
        assert_eq!(calls.take().len(), 1, "`{}`", string(&item));
    }
}

/// Analyzes `item` nested as `nest` says, and returns the Leaker with the result.
fn nested_analysis(
    nest: Nest,
    item: &Item,
    declared: &[&str],
) -> (Leaker, std::result::Result<ContextId, NotInternableError>) {
    let mut leaker = new_leaker();
    let result = match nest {
        Nest::EnterMod => leaker
            .enter_mod("m", names(declared), |view| view.visit_item(item))
            .map(|(context, ())| context),
        Nest::InternMod => leaker.intern_mod(&parse_quote!(mod m { #item })),
        Nest::InternModNested => leaker.intern_mod(&parse_quote!(mod m { mod n { #item } })),
        Nest::EnterGenerics => leaker
            .enter_generics(&Generics::default(), |view| view.visit_item(item))
            .map(|(context, ())| context),
    };
    (leaker, result)
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
/// AsRef<__S>, __S> Repeater<(T,), ID, 0> for __S { type Type = Rel<T>; }`, and the copy passes
/// its own type (`<Self as Repeater<(T,), ID, 0>>::Type`), for which it has the bound. The leaked
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
            let (leaker, result) = nested_analysis(nest, item, &[name]);
            if let Err(err) = result {
                panic!("{} is rejected: {}", what, err);
            }
            let referrer = leaker.finish();
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
            // enum or a union), and the parameter replaces `Self` in the bound.
            let r = referrer.repeater_impl(ix, ImplSelf::Param(&param)).unwrap();
            assert_eq!(r.self_ty, parse_quote!(__S), "{}", what);
            let last = r.generics.params.last().unwrap();
            assert_eq!(last, &parse_quote!(__S), "{}", what);
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
    let (context, ()) = leaker
        .enter_mod("m", names(&["S"]), |view| view.visit_item(&item))
        .unwrap();
    let referrer = leaker.finish();
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

/// F37: the copy cannot write `Self` in the default of a parameter (E0735), so it cannot pass its
/// own type there: a type in a default that uses a generic with a bound that names the `Self` of a
/// nested struct, enum or union is rejected, with the reason. A default that does not is fine, and
/// is rewritten. As the item itself (`intern_*`), the default is analyzed without `Self`, as it
/// always was.
#[test]
fn f37_a_default_cannot_pass_the_own_type() {
    let item: Item = parse_quote!(
        pub struct S<T: ::core::convert::AsRef<Self>, U = Wrap<T>>(pub T, pub U);
    );
    for nest in NESTS {
        let (_, result) = nested_analysis(nest, &item, &["S"]);
        let err = assert_rejected(
            &item,
            result,
            &["`Wrap < T >` is in the default of a parameter, and a bound of its generics names `Self`"],
        );
        assert!(
            all_notes(&err)[0].contains("cannot write `Self` in a default (E0735)"),
            "{:?}",
            all_notes(&err)
        );
        assert!(
            all_helps(&err)[0].contains("or write the default with an absolute path"),
            "{:?}",
            all_helps(&err)
        );
    }
    if let Item::Struct(item) = &item {
        let mut leaker = new_leaker();
        leaker.intern_struct(item, true).unwrap();
    }
    // The default does not use the generic: nothing depends on the `Self` of the bound.
    let item: Item = parse_quote!(
        pub struct S<T: ::core::convert::AsRef<Self>, U = Rel>(pub T, pub Wrap<U>);
    );
    for nest in NESTS {
        let (leaker, result) = nested_analysis(nest, &item, &["S"]);
        result.unwrap_or_else(|e| panic!("{:?}: {}", nest, e));
        let referrer = leaker.finish();
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
/// has the bound: accepted.
#[test]
fn f37_a_type_that_names_self_with_a_bound_that_names_it_is_rejected() {
    const HEADLINE: &str = "has a generic whose bound mentions `Self`, which cannot be leaked";
    for (item, name, ty) in &f37_rejected() {
        for nest in NESTS {
            let (_, result) = nested_analysis(nest, item, &[name]);
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
            assert!(
                helps[0].contains(
                    "without `Self`, write the type without `Self`, or analyze the item with \
                     its own `intern_*` call"
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
            Item::Struct(item) => leaker.intern_struct(item, true).map(|_| ()),
            Item::Enum(item) => leaker.intern_enum(item, true).map(|_| ()),
            Item::Union(item) => leaker.intern_union(item, true).map(|_| ()),
            _ => unreachable!(),
        }
        .unwrap_or_else(|e| panic!("`{}` is rejected: {}", string(item), e));
        let referrer = leaker.finish();
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
            let (leaker, result) = nested_analysis(nest, item, &[name]);
            if let Err(err) = result {
                panic!("{:?} `{}` is rejected: {}", nest, string(item), err);
            }
            assert!(leaker.finish().iter().count() > 0, "`{}`", string(item));
        }
    }
}

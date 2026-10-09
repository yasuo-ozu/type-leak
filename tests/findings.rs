//! Findings of the property test `tests/prop_cross_crate`, as library-level tests.
//!
//! The property test generates crates, builds them, and reports what is wrong in them: a Leaker's
//! crate or a Referrer's crate that does not compile, or an item that should have been rejected.
//! Each finding is minimized here to a test of the API of `type-leak` (`Leaker`, `Referrer`,
//! `RepeaterSlots`), with no generated crate and no `cargo` build. A test asserts what the library
//! SHOULD produce, as a property that the current output violates.
//!
//! | test | finding | kind |
//! |------|---------|------|
//! | `f01_unsized_type_param_encoding` | a `?Sized` parameter is a bare element of the tuple of the trait arguments | bug |
//! | `f02_bound_names_leaked_type_env_cycle` | a copy with a bound and a where predicate that name entries | bug |
//! | `f03_where_predicate_pulls_param_into_context` | an entry under a where predicate that names a lifetime is not found | bug |
//! | `f04_module_crate_relative_path_not_rewritten` | `intern_mod` accepts a `crate::` path next to a declared name | bug |
//! | `f05_inferred_outlives_of_self_type` | no `T: 'a` for a path with a lifetime; written, it is | limitation |
//! | `f06_impl_assoc_type_of_self_in_repeater` | `intern_impl` accepts `Self::A` | bug |
//! | `f07_method_lifetime_in_leaked_path` | a method lifetime in a leaked path and the return type | bug |
//! | `f08_merged_bound_ambiguous_plus` | merged bounds print `-> &dyn Any + Clone` | bug |
//! | `f09_fn_pointer_self_predicate_print` | `Self` replaced by `&fn(..)` in a where predicate | bug |
//! | `f10_module_bound_names_declared_item` | a bound of an entry names an item of the module | limitation |
//! | `f11_copied_impl_param_constrained_only_by_leaked_type` | a copied impl whose self type is an entry | limitation |
//! | `f12_elided_lifetime_in_fn_sugar_with_generics` | `Fn(&Rel<T>) -> &u8` with an entry that has lifetimes | bug |
//!
//! The bugs are `#[ignore]`d, as they fail today: run them with
//!
//! ```text
//! cargo test --test findings -- --ignored
//! ```
//!
//! and remove the `#[ignore]` of the one that a fix makes pass. The limitations are not bugs: they
//! pin behaviour that the README documents, or that follows from what rustc accepts, and they pass.
//! The test of a finding says how the property test shows it (`TYPE_LEAK_PROP_FEATURES=..`), and
//! the one-line reason of each failure is in its `#[ignore]` message.
//!
//! F10 and F11 are errors of rustc on a copy of a module or an impl that the test macro makes
//! (E0277, E0207). The tests pin the behaviour of the library that is the cause, not the errors.

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
/// the Repeater's trait arguments encode them as the tuple `(T, U)`. A tuple is well-formed only if
/// its last element is unsized, so the impl that a macro emits for the entry does not compile
/// (E0277). Found by the property test with `TYPE_LEAK_PROP_FEATURES=unsized_params`.
///
/// The assertion: either the Leaker rejects the item, or the encoder fails, or the encoding does
/// not have a `?Sized` parameter as a bare element but the last. That leaves the fix open
/// (`*const T`, `PhantomData<*const T>`, an error). The Repeater's argument `T` is `?Sized`, so a
/// last unsized element is fine.
#[test]
#[ignore = "F1: a ?Sized parameter is a bare, not last, element of the encoded tuple (bug)"]
fn f01_unsized_type_param_encoding() {
    let slots = RepeaterSlots::new(parse_quote!(crate::Repeater));
    let check = |generics: &Generics, what: &str| {
        // An error is a rejection, which is fine.
        if let Ok(encoded) = slots.encode_generics_params_to_ty(&generics.params) {
            assert_eq!(
                bare_unsized_element(generics, &encoded),
                None,
                "{}: `{}` encodes `{}` with an unsized parameter in the middle",
                what,
                string(generics),
                string(&encoded)
            );
        }
    };

    // The encoder on its own.
    check(&parse_quote!(<T: ?Sized, U>), "the encoder");
    check(&parse_quote!(<T, U: ?Sized, V>), "the encoder");

    // The generics of the entry that the Leaker finds.
    let item: ItemStruct = parse_quote! {
        pub struct S<T: ?Sized, U>(Rel<T, U>);
    };
    let mut leaker = new_leaker();
    if leaker.intern_struct(&item, true).is_err() {
        return;
    }
    let referrer = leaker.finish();
    for (_, generics) in referrer.iter_with_generics() {
        check(generics, "the entry");
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
#[test]
#[ignore = "F2: the copy has a bound and a where predicate that both name entries, which rustc cannot normalize (bug)"]
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
        // The control: `T: Sized` names no entry.
        parse_quote!(
            pub trait Tr3 {
                fn f<T: Fn(Rel)>(&self, a: Wrap<T>)
                where
                    T: Sized;
            }
        ),
    ];
    for item in items {
        let mut leaker = new_leaker();
        let Ok(context) = intern(&mut leaker, &item) else {
            continue;
        };
        let referrer = leaker.finish();
        let mut copy = item.clone();
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
    }
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
#[test]
#[ignore = "F3: an entry under a where predicate that names a lifetime is not found by the visitor (bug)"]
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

/// F4: `intern_mod` accepts a path relative to the crate next to a name that the module declares.
///
/// Trigger: `mod mm { struct R0; struct A(crate::base::B1<R0>); }`, and a constant of another
/// module in a type argument (`RC<{ crate::base::LEN }>`). The type is an entry, but the copy of
/// the module declares its own `R0`, so the Referrer hides the entry, and `crate::base` cannot
/// resolve in another crate (E0433). Found by the property test with
/// `TYPE_LEAK_PROP_FEATURES=module_crate_paths`.
///
/// The assertion: the Leaker rejects the module with a `NotInternableError`.
#[test]
#[ignore = "F4: intern_mod accepts a `crate::` path whose entry the copy of the module hides (bug)"]
fn f04_module_crate_relative_path_not_rewritten() {
    let mods: Vec<ItemMod> = vec![
        parse_quote! {
            pub mod mm {
                pub struct R0;
                pub struct A(pub crate::base::B1<R0>);
            }
        },
        parse_quote! {
            pub mod mm1 {
                pub struct RC<const N: usize>;
                pub struct S(pub RC<{ crate::base::LEN }>);
            }
        },
    ];
    for item in mods {
        let mut leaker = new_leaker();
        let result = leaker.intern_mod(&item);
        assert!(
            result.is_err(),
            "`{}` is accepted, with the entries {:?}",
            item.ident,
            leaker.finish().iter().map(string).collect::<Vec<_>>()
        );
    }
}

// ── F5 ───────────────────────────────────────────────────────────────────────

/// F5 (limitation): no `T: 'a` is recorded for a path with a lifetime; the bound written is.
///
/// README, "Outlives bounds": the Leaker cannot read the definition of a path, so a path with a
/// lifetime argument records nothing, even if its definition needs it. Trigger: `impl<'a, T>
/// E<'a, T>` with `struct E<'a, T> { f: &'a mut T }`, where rustc infers `T: 'a`. The impl
/// compiles through the implied bounds of its self type, but the Repeater's impl for the entry
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

/// F6: `intern_impl` accepts `Self::A` of the implemented trait in a leaked type.
///
/// Trigger: `impl ::leaker::m::Tr for u8 { type A = u8; fn f(&self, x: Rel<Self::A>) {} }`.
/// `Rel<Self::A>` is an entry, and the impl that a macro emits is `impl Repeater<..> for u8 {
/// type Type = Rel<Self::A>; }`, where `Self::A` is not an associated type of the Repeater (E0220).
/// The context says that `Self` is `u8`, and nothing says which trait `A` belongs to. Found by the
/// property test with `TYPE_LEAK_PROP_FEATURES=impl_assoc_self`.
///
/// The assertion: the Leaker rejects the impl with a `NotInternableError`.
#[test]
#[ignore = "F6: intern_impl accepts `Self::A`, which the Repeater's impl cannot write (bug)"]
fn f06_impl_assoc_type_of_self_in_repeater() {
    let item: ItemImpl = parse_quote! {
        impl ::leaker::m::Tr for u8 {
            type A = u8;
            fn f(&self, x: Rel<Self::A>) {}
        }
    };
    let mut leaker = new_leaker();
    let result = leaker.intern_impl(&item, true);
    assert!(
        result.is_err(),
        "accepted, with the entries {:?}",
        leaker.finish().iter().map(string).collect::<Vec<_>>()
    );
}

// ── F7 ───────────────────────────────────────────────────────────────────────

/// F7: a method lifetime in a leaked path, and in the return type.
///
/// Trigger: `fn f<'x>(&self, a: Wrap<'x>) -> &'x u8` (and `-> Wrap<'x>`), with `Wrap<'x>` leaked.
/// `'x` is late-bound in the trait, but the copy `fn f<'x>(&self, a: <Self as Repeater<(&'x (),)>>
/// ::Type) -> &'x u8` names it only in a projection, which does not constrain it, so rustc makes it
/// early-bound and rejects the method (E0195). README, "Method lifetimes": the Leaker reports an
/// error for a trait object leaked as a whole, and the same holds for a leaked path. Found by the
/// property test with `TYPE_LEAK_PROP_FEATURES=method_lifetime_paths`.
///
/// The assertion: the Leaker rejects those two traits with a `NotInternableError`. `a: &'x u8`
/// keeps `'x` as written, so that trait is accepted (the control).
#[test]
#[ignore = "F7: a method lifetime named by a leaked path and the return type is accepted (bug)"]
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
    ];
    let accepted: ItemTrait = parse_quote!(
        pub trait Tr2 {
            fn f<'x>(&self, a: &'x u8) -> Wrap<'x>;
        }
    );
    assert!(new_leaker().intern_trait(&accepted, true).is_ok());
    for item in rejected {
        let result = new_leaker().intern_trait(&item, true);
        assert!(result.is_err(), "`{}` is accepted", item.ident);
    }
}

// ── F8 ───────────────────────────────────────────────────────────────────────

/// Whether a type ends in a bare trait object, through references and pointers: a `+` after it
/// joins the object (`&dyn Any + Clone`).
fn ends_in_bare_object(ty: &Type) -> bool {
    match ty {
        Type::Reference(r) => ends_in_bare_object(&r.elem),
        Type::Ptr(p) => ends_in_bare_object(&p.elem),
        Type::TraitObject(_) | Type::ImplTrait(_) => true,
        Type::BareFn(f) => matches!(&f.output, ReturnType::Type(_, t) if ends_in_bare_object(t)),
        _ => false,
    }
}

/// The bounds that rustc reads with an ambiguous `+`: a bound of the sugar of `Fn` whose return
/// type ends in a bare trait object, followed by another bound.
fn ambiguous_plus(bounds: &Punctuated<TypeParamBound, Token![+]>) -> Vec<String> {
    let count = bounds.len();
    bounds
        .iter()
        .take(count.saturating_sub(1))
        .filter(|bound| match bound {
            TypeParamBound::Trait(TraitBound { path, .. }) => matches!(
                path.segments.last().map(|seg| &seg.arguments),
                Some(PathArguments::Parenthesized(ParenthesizedGenericArguments {
                    output: ReturnType::Type(_, ty),
                    ..
                })) if ends_in_bare_object(ty)
            ),
            _ => false,
        })
        .map(string)
        .collect()
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
#[test]
#[ignore = "F8: the merged bounds print `Fn(()) -> &'static dyn Any + Clone`, ambiguous (bug)"]
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
}

// ── F9 ───────────────────────────────────────────────────────────────────────

/// Whether a type is a reference or a pointer to a function pointer: as the bounded type of a where
/// predicate, rustc reads the `:` after `&fn(..)` as the start of a return type.
fn bare_fn_under_pointer(ty: &Type) -> bool {
    fn tail_is_bare_fn(ty: &Type) -> bool {
        match ty {
            Type::Reference(r) => tail_is_bare_fn(&r.elem),
            Type::Ptr(p) => tail_is_bare_fn(&p.elem),
            Type::BareFn(_) => true,
            _ => false,
        }
    }
    match ty {
        Type::Reference(r) => tail_is_bare_fn(&r.elem),
        Type::Ptr(p) => tail_is_bare_fn(&p.elem),
        _ => false,
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
#[test]
#[ignore = "F9: Self replaced by `&fn(..)` is the bounded type of a where predicate without parentheses (bug)"]
fn f09_fn_pointer_self_predicate_print() {
    let item: ItemTrait = parse_quote! {
        pub trait Tr {
            fn m<'y>(&self, a: u8) where Self: 'y;
            fn n(&self, a: u8) where Self: Clone;
        }
    };
    let mut leaker = new_leaker();
    let context = leaker.intern_trait(&item, true).unwrap();
    let referrer = leaker.finish();
    let self_ty: Type = parse_quote!(&'static unsafe fn(i32, R0));

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
        "unparenthesized bounded types, which rustc cannot parse: {:?}",
        found.0
    );
}

// ── F10 ──────────────────────────────────────────────────────────────────────

/// F10 (limitation): a bound of an entry names an item that the module declares.
///
/// README, "Modules": the names that the module declares are copied with it, so they are not
/// leaked, and an entry is keyed by its module if a path in a bound of its generics is relative.
/// Trigger: `mod mm { use super::base::B1 as Imp; trait Cv {} struct S<T: Cv>(pub Imp<T>); }`. The
/// entry `Imp<T>` has the context `<T: Cv>` and is keyed by `mm`; the Repeater's impl for it is
/// emitted in the Leaker's module, so it requires the Leaker's `Cv`. The copy of the module declares
/// its own `Cv`, so a projection `<.. as Repeater<(T,), ..>>::Type` in the copy needs `T:
/// leaker::mm::Cv`, which the copy's `T: Cv` is not (E0277). The generator of the property test
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

/// The lifetimes of a type that elision counts: references (named or elided), and lifetime
/// arguments. A function pointer or `Fn(..)` inside has its own binder.
#[derive(Default)]
struct Lifetimes {
    count: usize,
    elided: usize,
}

impl<'ast> Visit<'ast> for Lifetimes {
    fn visit_type_reference(&mut self, i: &'ast TypeReference) {
        self.count += 1;
        if i.lifetime.as_ref().map_or(true, |l| l.ident == "_") {
            self.elided += 1;
        }
        self.visit_type(&i.elem)
    }
    fn visit_lifetime(&mut self, i: &'ast Lifetime) {
        self.count += 1;
        if i.ident == "_" {
            self.elided += 1;
        }
    }
    fn visit_type_bare_fn(&mut self, _: &'ast TypeBareFn) {}
    fn visit_parenthesized_generic_arguments(&mut self, _: &'ast ParenthesizedGenericArguments) {}
}

fn lifetimes_of<'a>(types: impl IntoIterator<Item = &'a Type>) -> Lifetimes {
    let mut lifetimes = Lifetimes::default();
    for ty in types {
        lifetimes.visit_type(ty);
    }
    lifetimes
}

/// The sugar of `Fn(..) -> ..` whose elided output lifetime elision cannot resolve: it needs
/// exactly one lifetime in the inputs.
struct Elision(Vec<String>);

impl<'ast> Visit<'ast> for Elision {
    fn visit_parenthesized_generic_arguments(&mut self, i: &'ast ParenthesizedGenericArguments) {
        if let ReturnType::Type(_, output) = &i.output {
            let inputs = lifetimes_of(&i.inputs);
            if lifetimes_of([&**output]).elided > 0 && inputs.count != 1 {
                self.0.push(format!(
                    "{} (the inputs have {} lifetimes)",
                    string(i),
                    inputs.count
                ));
            }
        }
        syn::visit::visit_parenthesized_generic_arguments(self, i)
    }
}

/// F12: an elided lifetime in the sugar of `Fn` next to an entry that has lifetimes.
///
/// Trigger: `Box<dyn Fn(&Rel<T>) -> &u8>` in `trait Tr<'a, T: 'a>`. The sugar has one input
/// lifetime, which elision gives to the output. The entry `Rel<T>` has the generics `<'a, T: 'a>`,
/// so its replacement names lifetimes in its arguments (`<() as Repeater<(&'a (), T, &'a T), ..>>
/// ::Type`), and the inputs have several: elision fails (E0106 "missing lifetime specifier").
/// `Tr1<T>`, without `'a`, works. Found by the property test with
/// `TYPE_LEAK_PROP_FEATURES=elided_fn_generics`.
///
/// The test makes the replacement as a macro does (the projection of the Repeater, with the
/// arguments of the entry encoded by `RepeaterSlots`), rewrites the copy, and asserts the minimal
/// property that the copy violates: the sugar of `Fn` with an elided output lifetime has exactly
/// one lifetime in its inputs. A fix is free to name the lifetime or to reject the item.
#[test]
#[ignore = "F12: the replacement of an entry in `Fn(&Rel<T>) -> &u8` names lifetimes, and elision fails (bug)"]
fn f12_elided_lifetime_in_fn_sugar_with_generics() {
    let traits: Vec<ItemTrait> = vec![
        parse_quote! {
            pub trait Tr<'a, T: 'a> { fn f(&self, x: Box<dyn Fn(&Rel<T>) -> &u8>); }
        },
        // The control: no lifetime in the arguments of the entry.
        parse_quote! {
            pub trait Tr1<T> { fn f(&self, x: Box<dyn Fn(&Rel<T>) -> &u8>); }
        },
    ];
    let slots = RepeaterSlots::new(parse_quote!(crate::Repeater));
    for item in traits {
        let mut leaker = new_leaker();
        let Ok(context) = leaker.intern_trait(&item, true) else {
            continue;
        };
        let referrer = leaker.finish();
        let root_args: Vec<GenericArgument> = item.generics.params.iter().map(param_arg).collect();
        let mut copy = Item::Trait(item.clone());
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
            .visit_item_mut(&mut copy);
        let mut elision = Elision(Vec::new());
        elision.visit_item(&copy);
        assert!(
            elision.0.is_empty(),
            "`{}`: elision cannot resolve the output of {:?}\nin {}",
            item.ident,
            elision.0,
            string(&copy)
        );
    }
}

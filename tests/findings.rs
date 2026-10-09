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
//!
//! The bugs that are not fixed are `#[ignore]`d, as they fail today (a fixed one has no
//! `#[ignore]`, and `fixed` in the table): run them with
//!
//! ```text
//! cargo test --test findings -- --ignored
//! ```
//!
//! and remove the `#[ignore]` of the one that a fix makes pass (the table then says `fixed`, and the
//! test stays as a regression test). The limitations are not bugs: they pin behaviour that the
//! README documents, or that follows from what rustc accepts, and they pass.
//! The test of a finding says how the property test shows it (`TYPE_LEAK_PROP_FEATURES=..`), and
//! the one-line reason of each failure is in its `#[ignore]` message. The features of the fixed
//! findings are on by default in the property test: the Leaker now rejects what they generate (the
//! rejections are counted), or rewrites it.
//!
//! F10 and F11 are errors of rustc on a copy of a module or an impl that the test macro makes
//! (E0277, E0207). The tests pin the behaviour of the library that is the cause, not the errors.
//!
//! The feature of the limitation F18 (`strict_anchors`) is off in
//! the property test: the generator does not make what it fails on (it only decides that
//! an item is not invoked as strict). Each test of a bug asserts that the Leaker either rejects
//! the item or gives what the Referrer needs, and says which of the two is up to the fix.

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
/// Fixed by rejecting what rustc cannot compile (see "Bounds that name leaked types" of the README
/// and `tests/bound_cycles.rs`). The control `Tr3` is not rejected: what compiles stays accepted.
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
        // The control: `T: Sized` names no entry.
        parse_quote!(
            pub trait Tr3 {
                fn f<T: Fn(Rel)>(&self, a: Wrap<T>)
                where
                    T: Sized;
            }
        ),
    ];
    let control = items.len() - 1;
    for (ix, item) in items.into_iter().enumerate() {
        let mut leaker = new_leaker();
        let Ok(context) = intern(&mut leaker, &item) else {
            assert_ne!(ix, control, "the control is rejected, though it compiles");
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
    assert_eq!(
        err.to_string(),
        "`Self :: A` cannot be written in the impl of the Repeater"
    );

    // A bound of the generics of the entry that names it.
    let item: ItemImpl = parse_quote! {
        impl<T: ::p::Tr<Self::A>> ::leaker::m::Tr for W<T> {
            type A = u8;
            fn f(&self, x: Rel<T>) {}
        }
    };
    assert!(new_leaker().intern_impl(&item, true).is_err());

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
/// early-bound and rejects the method (E0195). README, "Method lifetimes": the Leaker reports an
/// error for a trait object leaked as a whole, and the same holds for a leaked path. Found by the
/// property test with `TYPE_LEAK_PROP_FEATURES=method_lifetime_paths`.
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
    let slots = RepeaterSlots::new(parse_quote!(crate::Repeater));
    let mut leaker = new_leaker();
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
/// name the leaked entry itself (E0277, "Bounds that name leaked types" of the README); the type
/// for `Self` in the copy is a cycle (E0391); and the self type has no parts to leak instead
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
/// The trait case is rejected already (`trait Tr<T: Fn(Rel)> { fn m<X>(&self) where X:
/// Into<Self::A1>; }`, "It is `Self::A`" of the README); the impl needs the same rule, with the
/// generics of the impl instead of those of the context. The assertion: if the Leaker accepts
/// the impl, its copy does not have a replacement in a bound or where predicate of the impl and
/// `Self::A` in a predicate. An optional entry counts if the Referrer is strict (it replaces it).
///
/// The controls are accepted, as they compile: a bound that names no leaked type, and a leaked
/// bound with `Self::A1` outside the predicates.
///
/// Fixed by the rule of F2 for `Self::A` (see "Bounds that name leaked types" of the README and
/// `tests/bound_cycles.rs`): in an impl, the bounds of the impl count too, not only those of the
/// generics in the context of `Self::A`, as rustc selects the impl of the trait. The optional
/// entries that they name are copied as written for a strict Referrer.
#[test]
fn f14_impl_predicate_names_self_projection_with_leaked_bound() {
    let items: Vec<(ItemImpl, bool)> = vec![
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
        ),
        // The leaked type of the bound is an optional entry, which a strict Referrer replaces.
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
    for (item, strict) in &items {
        // Rejected is fine.
        let Ok(copy) = f14_copy(item, *strict) else {
            continue;
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
/// The fix is the same as F12's: `Self` (when it is a type with lifetimes that a copy writes, the
/// struct, enum or union itself) in the inputs of such a signature adds lifetimes like an entry
/// does. The holder (the function pointer, the trait object, the reference that takes its
/// lifetime) is leaked whole, which keeps `Self` and the elision as written in the impl of the
/// Repeater (`impl<'a> Repeater<..> for S<'a> { type Type = fn(&Self) -> &u8; }`, which rustc
/// accepts). With `allow_self == false` it cannot be leaked, and where nothing holds the signature
/// (a bound, a method) it is a conflict, with the note about E0106. The copy of an impl keeps
/// `Self`, and so does a trait, whose Repeater impl writes `Self::A` as `<__S>::A`
/// (`Referrer::iter_with_self()`), and `with_self(Host)` the same, so they are not affected
/// (rustc accepts `fn(&<__S>::A) -> &u8`, and rejects `fn(&<__Self as Tr<'a>>::A) -> &u8`).
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
    // The holder is the entry: the whole signature is leaked, with `Self` as written.
    for (item, holder) in [
        (
            parse_quote!(
                pub struct S<'a> {
                    f: fn(&Self) -> &u8,
                    g: &'a u8,
                }
            ),
            "fn (& Self) -> & u8",
        ),
        (
            parse_quote!(
                pub struct S<'a>(Box<dyn Fn(&Self) -> &u8>, &'a u8);
            ),
            "dyn Fn (& Self) -> & u8",
        ),
        (
            parse_quote!(
                pub enum E<'a, T> {
                    A(fn(&(T, Self)) -> &u8),
                    B(&'a T),
                }
            ),
            "fn (& (T , Self)) -> & u8",
        ),
        (
            parse_quote!(
                pub union U<'a> {
                    f: fn(&Self) -> &u8,
                    g: &'a u8,
                }
            ),
            "fn (& Self) -> & u8",
        ),
        // `Self` deep in the inputs, next to a type that is allowed.
        (
            parse_quote!(
                pub struct S<'a>(Box<dyn Fn(&Vec<Self>) -> &u8>, &'a u8);
            ),
            "dyn Fn (& Vec < Self >) -> & u8",
        ),
        // The reference takes its lifetime from the trait object, so it is the holder.
        (
            parse_quote!(
                pub struct S<'a>(&'a dyn Fn(&Self) -> &u8, &'a u8);
            ),
            "& 'a dyn Fn (& Self) -> & u8",
        ),
        // `Self` as the qualified self type has none of its own either.
        (
            parse_quote!(
                pub struct S<'a>(Box<dyn Fn(&<Self as ::p::Tr>::A) -> &u8>, &'a u8);
            ),
            "dyn Fn (& < Self as :: p :: Tr > :: A) -> & u8",
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
    }
    // Where nothing can hold the signature (a bound, an elided reference), the item is rejected,
    // not copied wrongly.
    for item in [
        parse_quote! { pub struct S<'a, F: Fn(&Self) -> &u8>(&'a F); },
        parse_quote! { pub struct S<'a>(&'a u8, &dyn Fn(&Self) -> &u8); },
        parse_quote! { pub enum E<'a, T> where T: Fn(&Self) -> &u8 { A(&'a T) } },
        // `Self::A` cannot be written in the Repeater's impl for a type, so the holder cannot be
        // leaked either.
        parse_quote! { pub struct S<'a>(Box<dyn Fn(&Self::A) -> &u8>, &'a u8); },
    ] {
        let item: Item = item;
        let err = f15_copy(&item, true)
            .err()
            .unwrap_or_else(|| panic!("{} is accepted", string(&item)));
        let note = err.note().unwrap();
        assert!(
            note.contains("E0106") || note.contains("Self :: A"),
            "{}: {}",
            string(&item),
            note
        );
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
/// The same family, which is documented and not a bug (F5, README "Outlives bounds"): a path with
/// a lifetime argument records nothing, so `Rel<Wrap<'x, [Self]>>` with `fn m<'x>(&'x self)`,
/// where `Wrap<'a, T: 'a>` needs `Self: 'x` and the receiver implies it, has no `Self: 'x` in
/// the context (`BL<'x, [Self]>` in the property test). The README asks to write the bound, and
/// the generator writes it. And a where predicate `X: Into<&'x Self>` of a method relies on the
/// receiver as well, which the witness and the check of the property test (functions with no
/// receiver) do not have: E0309 in the Leaker's crate, from the test macro's own oracle, so the
/// generator writes `Self: 'x` there too.
///
/// The assertions, for each entry that has `&'x Self::A0`: the context has the predicate
/// `Self::A0: 'x` and `Self: 'x`, and the arguments of the Repeater are those of the parameters
/// and `&'x Self`, with no outlives of `T` or `'a`; the Referrer replaces the entry in a copy of
/// the trait. The control has the bounds written (`where Self: 'x, T: 'x, 'a: 'x`), which are
/// recorded as before.
#[test]
fn f16_implied_outlives_of_self_projection_lost() {
    let slots = RepeaterSlots::new(parse_quote!(crate::Repeater));
    let items: Vec<(ItemTrait, &str)> = vec![
        (
            parse_quote!(
                pub trait Tr<'a, T> {
                    type A0;
                    fn m<'x>(&'x self, a: Rel<&'x mut Self::A0>)
                    where
                        Self: 'x;
                }
            ),
            "(& 'a () , & 'x () , T , & 'x Self)",
        ),
        // Only the receiver implies `Self: 'x`: it is recorded for the reference all the same.
        (
            parse_quote!(
                pub trait Tr<T> {
                    type A0;
                    fn m<'x>(&'x self, a: Rel<&'x Self::A0>);
                }
            ),
            "(& 'x () , T , & 'x Self)",
        ),
        // No generics: `Self: 'x` is all the components.
        (
            parse_quote!(
                pub trait Tr {
                    type A0;
                    fn m<'x>(&'x self, a: Rel<&'x Self::A0>);
                }
            ),
            "(& 'x () , & 'x Self)",
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
            "(& 'a () , & 'x () , T , & 'x Self)",
        ),
    ];
    for (item, encoded) in &items {
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
                vec!["'x".to_string()],
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
/// optional entry (it holds `::ext::B0`, another optional type, so it is not left alone as `T:
/// Fn(&'a ::ext::W<T>)` is), and its context has `'a`, `T` and the where predicate that names
/// `'a`, which is the predicate that names the entry. The copy has `T: Fn(<() as Repeater<(&'a (),
/// T, &'a T), ..>>::Type)`, and the Repeater's impl for it has `T: Fn(&'a ::ext::W<T,
/// ::ext::B0>)`, which the environment of the copy does not have in that form: E0277 "expected an
/// `Fn(&W<T, B0>)` closure, found `T`" (standalone with rustc, too). That is F2, whose rule
/// ("optional entry in such a place is copied as written", README "Bounds that name leaked
/// types") does not see this one: the same item with `::ext::W<T>` (no optional type inside), or
/// with a required entry (`Rel`), is not an entry or is rejected. Found by the property test
/// with `TYPE_LEAK_PROP_FEATURES=strict_cycles` (seeds 1 to 8 of 60 cases), which only affects the
/// items that the generator invokes as strict, 25% of them.
///
/// The Leaker should leave that type as written (not an entry), or reject the item. The assertion is
/// that no entry of an accepted item is named by the bounds or where predicates of its own
/// context.
///
/// Fixed: an optional entry that is leaked as a whole (a reference with a lifetime), and whose own
/// context is unstable (its bound names it), is not one, and neither are the optional entries
/// inside it: its parts cannot stand for it. See "Bounds that name leaked types" of the README.
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
/// README, "External crates": an absolute path of a crate that the Leaker does not allow is an
/// optional entry, which a Referrer leaves as written by default and replaces if it is strict
/// (`allow_ambiguous_crate_name(false)`); "replacing a type can lose an anchor or a method
/// lifetime". Trigger: `trait Tr { fn m<'x>(&self, a: ::ext::W<&'x u8>) -> Rel<'x>; }`. In the
/// trait, `'x` is late-bound (the argument holds it, and no bound names it). The Leaker counts the
/// optional type as an anchor, as the default Referrer keeps it. The strict copy is `fn m<'x>(&self,
/// a: P<1>) -> P<0>`: both are projections, which do not constrain `'x`, so it is early-bound, and
/// rustc rejects the impl of the Referrer's crate, where the trait has it late-bound (E0195
/// "lifetime parameters differ in whether they are early- or late-bound"). Found by the property
/// test with `TYPE_LEAK_PROP_FEATURES=strict_anchors`, for the items that the generator invokes as
/// strict (25% of them): the Leaker's analysis cannot know what the Referrer chooses.
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
/// `Rel` leaked (also `for<'h> Fn(&'h Self) -> &'h Result<Self, f64>`, which rustc reports as
/// E0271 "type mismatch resolving"). The entry `Rel<X>` has the context `<X: Fn(Self) -> u8>`
/// with `Self` = `S`, so the Repeater's impl for `S` has `X: Fn(S) -> u8`. The copy has `X:
/// Fn(Self) -> u8` with `Self` = `<() as Repeater<..>>::Type`, the replaced self type, which rustc
/// has not normalized yet in the environment where it proves the predicate `<S as
/// Repeater<(X,), ..>>::Type: Default`: E0277 "expected an `Fn(S)` closure, found `X`" (the same
/// with rustc alone). That is F2 with `Self` as the leaked type that the bound names, and the
/// check of the README ("It is `Self`, in an impl whose self type is leaked, and a bound of the
/// generics of the impl names a leaked type") only looks at the generics of the impl, not at
/// those of a method. With the entry in a position (`a0: Rel<X>`) it compiles. Found by the
/// property test with `TYPE_LEAK_PROP_FEATURES=impl_self_bound_cycle`.
///
/// The assertion: if the Leaker accepts the impl, no entry that a predicate of the copy names has
/// a bound of its context that names `Self` while the self type is an entry.
///
/// Fixed by generalizing the rule of F2: a bound is unstable if it names a leaked type, `Self` (if
/// the self type of an impl is leaked, and for a strict Referrer, if it holds an optional entry,
/// which is copied as written then), `Self::A` or an associated type that rustc normalizes (F20 and
/// F21). An entry in a predicate whose context has such a bound is an error. See "Bounds that name
/// leaked types" of the README and `tests/bound_cycles.rs`.
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
    for item in &items {
        // Rejected is fine.
        let Ok((referrer, copy)) = f19_copy(item) else {
            continue;
        };
        let found = entries_with_self_bound_in_predicates(&referrer, &copy, self_is_leaked);
        assert!(
            found.is_empty(),
            "in the copy `{}`, a predicate names the entries {:?}, which have a bound that names `Self`, the replaced self type",
            string(&copy),
            found
        );
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
        let err = f19_copy_with(f13_leaker(), item)
            .err()
            .unwrap_or_else(|| panic!("`{}` is accepted, and it does not compile", string(item)));
        assert!(
            err.to_string()
                .contains("a bound of its generics names `Self`, which is a leaked type"),
            "{}: {}",
            string(item),
            err
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
    let normalized: Vec<(ItemImpl, &str)> = vec![
        (
            parse_quote!(
                impl<T: Fn(Rel2<u8>), V: AsRef<Self>> ::p::W<Rel<T>, T, V> {
                    fn f(&self, a: Rel3<u8>) {}
                }
            ),
            "`Self` is in a bound or where predicate, and a bound of its generics names a leaked type",
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
            "`Self` is in a bound or where predicate, and a bound of its generics names a leaked type",
        ),
        // The `Item` that a predicate equates is normalized too (E0277, "expected an `Fn(u8)`
        // closure, found `I`").
        (
            parse_quote!(
                impl<I: Iterator<Item = u8> + Fn(I::Item), V: AsRef<Self>> ::p::W<Rel<I>, I, V> {}
            ),
            "`Self` is in a bound or where predicate, and a bound of its generics names `I :: Item`",
        ),
    ];
    for (item, headline) in &normalized {
        let err = f13_leaker()
            .intern_impl(item, true)
            .err()
            .unwrap_or_else(|| panic!("`{}` is accepted, and it does not compile", string(item)));
        assert!(
            err.to_string().contains(headline),
            "{}: {}",
            string(item),
            err
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
/// The README rule for `Self::A` ("a bound of the generics of the trait names a leaked type")
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
    for item in &items {
        // Rejected is fine.
        let Ok((referrer, copy)) = f19_copy(item) else {
            continue;
        };
        let found = entries_with_self_bound_in_predicates(&referrer, &copy, bound);
        assert!(
            found.is_empty(),
            "in the copy `{}`, a predicate names the entries {:?}, which have a bound that names `Self::A`",
            string(&copy),
            found
        );
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
        let mut used = ReplacementsInPredicates::default();
        used.visit_item(&copy);
        let found: Vec<String> = used
            .0
            .iter()
            .filter(|&&ix| {
                ix < referrer.iter().count() && generics_name_item(referrer.generics(ix))
            })
            .map(|&ix| string(referrer.iter().nth(ix).unwrap()))
            .collect();
        Ok::<_, NotInternableError>((copy, found))
    };
    for item in &items {
        // Rejected is fine.
        let Ok((copy, found)) = check(item) else {
            continue;
        };
        assert!(
            found.is_empty(),
            "in the copy `{}`, a predicate names the entries {:?}, which have a bound that names an `Item`",
            string(&copy),
            found
        );
    }
    for item in &controls {
        let (copy, found) = check(item)
            .unwrap_or_else(|e| panic!("`{}` is rejected, and it compiles: {}", string(item), e));
        assert!(found.is_empty(), "{}", string(&copy));
    }
}

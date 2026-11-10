//! Tests for the helpers for macro authors: `Referrer::required()` and `Referrer::is_plain()`,
//! the description of the Repeater impl of an entry (`Referrer::repeater_impl()`), the arguments of
//! an entry at a use site (`Referrer::entry_args()`), and the items of the errors of the analysis
//! (`Vec<ErrorItem>`).
//!
//! The impl and the arguments are compared with what a macro wrote by hand before the helpers
//! (the replacement of `Self`, `without_encoded_bounds()` and the encoders, and the arguments of a
//! context with the encoders), over all the entries of a set of items, in `src/encode/tests.rs`.

use syn::*;
use template_quote::{quote, ToTokens};
use type_leak::*;

fn s<T: ToTokens>(t: &T) -> String {
    quote!(#t).to_string()
}

fn t(ty: Type) -> String {
    s(&ty)
}

/// The messages of the errors, one per line.
fn messages(errors: &[ErrorItem]) -> String {
    errors
        .iter()
        .map(|item| item.message.as_str())
        .collect::<Vec<_>>()
        .join("\n")
}

fn slots() -> RepeaterSlots {
    RepeaterSlots::new(parse_quote!(crate::Repeater))
}

fn config() -> Allow<impl Fn(&Path) -> bool> {
    Allow::new().allow_primitive().allow_prelude()
}

/// Analyzes a whole item the way a downstream macro does.
fn analyze(item: &Item) -> (Referrer, ContextId) {
    let mut leaker = Leaker::from_config(config(), false, false);
    let context = match item {
        Item::Struct(i) => leaker.intern_struct(i, true),
        Item::Enum(i) => leaker.intern_enum(i, true),
        Item::Union(i) => leaker.intern_union(i, true),
        Item::Trait(i) => leaker.intern_trait(i, true),
        Item::Impl(i) => leaker.intern_impl(i, true),
        Item::Mod(i) => leaker.intern_mod(i),
        _ => panic!("unsupported item"),
    };
    let referrer = leaker
        .result()
        .finish()
        .unwrap_or_else(|e| panic!("{}: {}", s(item), messages(&e)));
    (referrer, context)
}

/// Items with lifetimes and their bounds, consts, `?Sized`, defaults, the three meanings of `Self`
/// (none, the implementor of a trait, a type), `Self: 'a`, `Self::A` under a reference, a module and
/// an optional entry.
fn samples() -> Vec<Item> {
    vec![
        parse_quote! {
            trait Tr<'a, T: 'a, const N: usize> {
                fn f(&self, x: &'a Wrap<T, N>) -> Rel;
                fn g<U: ?Sized>(&self, u: &U) -> Rel2<U>;
            }
        },
        parse_quote! {
            trait Tr<'a> { fn f(&self, x: &'a Rel<Self>); }
        },
        parse_quote! {
            trait Tr {
                type A;
                fn f(&self) -> Wrap<Self::A>;
                fn g<'a>(&self, x: &'a Rel<Self::A>);
            }
        },
        parse_quote! {
            trait Tr<T, U = Rel<T>> { fn f(&self, t: T) -> Rel<T>; }
        },
        parse_quote! {
            trait Tr<const B: bool> { fn f(&self) -> Wrap<B>; }
        },
        parse_quote! {
            trait Tr<T: ::p::Bound<Self>>: ::core::fmt::Debug {
                fn f<U: ::core::cmp::PartialEq<Self>>(&self) -> Wrap2<U>;
                fn g(&self) -> Rel4<m!(Self, [u8; 1])>;
                fn h(&self, t: Rel5<T>);
            }
        },
        parse_quote! {
            struct S<'a, T: 'a + Clone> { f: &'a Rel<Self>, g: Wrap<T>, h: ::foo::X }
        },
        parse_quote! {
            enum E<T> { A(Wrap<Self>), B(Rel<T>) }
        },
        parse_quote! {
            union U<T: Copy> { a: ::core::mem::ManuallyDrop<Wrap<T>> }
        },
        parse_quote! {
            impl<T: Clone> ::p::Tr for W<T> { fn f(&self) -> Wrap<Self> { todo!() } }
        },
        parse_quote! {
            mod m {
                pub struct S<T> { x: Rel<T> }
                pub trait Tr {
                    fn f(&self) -> Rel2;
                    fn g<const B: bool>(&self) -> Wrap<B>;
                }
                pub type A = Rel3;
            }
        },
    ]
}

// ── required entries ─────────────────────────────────────────────────────────────────────────

#[test]
fn required_entries_come_first_and_skip_the_optional_ones() {
    let item: ItemStruct = parse_quote!(
        struct S {
            a: ::foo::X,
            b: Rel,
            c: Wrap<u8>,
            d: ::bar::Y,
        }
    );
    let mut leaker = Leaker::from_config(config(), false, false);
    leaker.intern_struct(&item, true);
    let result = leaker.result();
    let referrer = result.clone().finish().unwrap();
    let all: Vec<String> = referrer.iter().map(s).collect();
    assert_eq!(all.len(), 4, "{:?}", all);
    assert!(!referrer.is_empty());
    let required: Vec<(usize, String)> = referrer.required().map(|(ix, ty)| (ix, s(ty))).collect();
    // The required entries have the smaller indices, and `required()` is `must()` of `iter()`.
    assert_eq!(required.len(), 2, "{:?}", required);
    assert_eq!(
        required.iter().map(|(ix, _)| *ix).collect::<Vec<_>>(),
        [0, 1]
    );
    let mut names: Vec<&str> = required.iter().map(|(_, ty)| ty.as_str()).collect();
    names.sort();
    assert_eq!(names, ["Rel", "Wrap < u8 >"]);
    for (ix, ty) in referrer.iter().enumerate() {
        assert_eq!(
            referrer.required().any(|(found, _)| found == ix),
            referrer.must(ix),
            "{}",
            s(ty)
        );
    }
    assert!(referrer.required().next().is_some());
    // Not affected by how the Referrer treats the optional entries.
    let mut strict = result;
    strict.deny_ambiguous_crate_name();
    let strict = strict.finish().unwrap();
    assert_eq!(
        strict.required().map(|(ix, _)| ix).collect::<Vec<_>>(),
        [0, 1]
    );
    // The serialized Referrer has them too.
    let parsed: Referrer = parse2(quote!(#referrer)).unwrap();
    assert_eq!(
        parsed
            .required()
            .map(|(ix, ty)| (ix, s(ty)))
            .collect::<Vec<_>>(),
        required
    );
}

#[test]
fn a_referrer_with_only_optional_entries_has_no_required_entry() {
    let item: ItemStruct = parse_quote!(
        struct S {
            a: ::foo::X,
            b: u8,
        }
    );
    let mut leaker = Leaker::from_config(config(), false, false);
    leaker.intern_struct(&item, true);
    let referrer = leaker.result().finish().unwrap();
    assert!(!referrer.is_empty());
    assert_eq!(referrer.iter().count(), 1);
    assert!(!referrer.must(0));
    assert!(!referrer.required().next().is_some());
    assert_eq!(referrer.required().count(), 0);
    // Nothing at all.
    let mut leaker = Leaker::from_config(config(), false, false);
    leaker.intern_struct(
        &parse_quote!(
            struct S {
                a: u8,
            }
        ),
        true,
    );
    let referrer = leaker.result().finish().unwrap();
    assert!(referrer.is_empty());
    assert!(!referrer.required().next().is_some());
    assert_eq!(referrer.required().count(), 0);
}

// ── is_plain ─────────────────────────────────────────────────────────────────────────────────

#[test]
fn plain_entries_need_no_generics_self_or_module() {
    let (referrer, _) = analyze(&parse_quote! {
        trait Tr<T> {
            type A;
            fn f(&self, a: Rel, b: Wrap<T>, c: Rel2<Self::A>) -> Rel3<Self>;
        }
    });
    let plain: Vec<String> = referrer
        .iter()
        .enumerate()
        .filter(|(ix, _)| referrer.is_plain(*ix))
        .map(|(_, ty)| s(ty))
        .collect();
    assert_eq!(plain, ["Rel"]);
    // A module keys an entry that has a relative path.
    let (referrer, _) = analyze(&parse_quote! { mod m { pub fn f() -> Rel { todo!() } } });
    assert_eq!(referrer.iter().count(), 1);
    assert!(referrer.context_module(referrer.context(0)).is_some());
    assert!(!referrer.is_plain(0));
    // An entry that is shared by all the modules is plain: `Vec<u8>` has no relative path.
    let (referrer, _) = analyze(&parse_quote! {
        mod m { pub fn f() -> ::p::W<Rel> { todo!() } pub fn g() -> ::q::X<u8> { todo!() } }
    });
    for ix in 0..referrer.iter().count() {
        let module = referrer.context_module(referrer.context(ix));
        assert_eq!(
            referrer.is_plain(ix),
            module.is_none(),
            "{}",
            s(referrer.iter().nth(ix).unwrap())
        );
    }
}

#[test]
fn is_plain_is_what_an_impl_for_a_marker_needs_nothing_for() {
    // The invariant that the test macros check for every entry they emit, over the samples.
    let marker: Type = parse_quote!(__Marker);
    for item in samples() {
        let (referrer, _) = analyze(&item);
        for ix in 0..referrer.iter().count() {
            let needs_nothing =
                match referrer.repeater_impl(ix, ImplSelf::Marker(&marker), |value, param| {
                    slots().encode_const(value, &param.ty)
                }) {
                    Ok(r) => {
                        r.generics.params.is_empty()
                            && r.generics
                                .where_clause
                                .as_ref()
                                .map_or(true, |clause| clause.predicates.is_empty())
                            && s(&r.args) == "()"
                            && r.module.is_none()
                    }
                    Err(EncodeError::NeedsSelf(_)) => false,
                    Err(e) => panic!("{}", e),
                };
            assert_eq!(referrer.is_plain(ix), needs_nothing, "{}: {}", s(&item), ix);
        }
    }
}

/// The sumtype shape: `Leaker::new()` (no allowed path) and `intern_trait(.., false)` on a trait
/// whose only entries are the absolute types of external crates, which are optional. `iter()`
/// counts them, `is_plain()` is true for each, and an impl of the Repeater for a marker is the
/// plain one (no generics, no `Self`, no module), with the entry as the type. A relative type or
/// `Self` makes the call fail and the macro throws the Leaker away.
#[test]
fn a_trait_with_only_optional_external_types_has_plain_entries_for_a_marker() {
    let item: ItemTrait = parse_quote! {
        pub trait Reader {
            fn read(&mut self, buf: &mut [::core::primitive::u8])
                -> ::std::io::Result<::core::primitive::usize>;
            fn cause(&self) -> ::core::option::Option<&(dyn ::std::error::Error + 'static)>;
            fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result;
        }
    };
    let mut leaker = Leaker::new();
    leaker.intern_trait(&item, false);
    let referrer = leaker.result().finish().unwrap();
    // `iter()` counts the optional entries, `required()` does not.
    assert!(!referrer.is_empty());
    assert!(!referrer.required().next().is_some());
    assert_eq!(referrer.required().count(), 0);
    assert_eq!(
        referrer.iter().map(s).collect::<Vec<_>>(),
        [
            ":: std :: io :: Result < :: core :: primitive :: usize >",
            "dyn :: std :: error :: Error + 'static"
        ]
    );
    let marker: Type = parse_quote!(__Marker);
    for (ix, ty) in referrer.iter().enumerate() {
        assert!(!referrer.must(ix));
        assert!(referrer.is_plain(ix), "{}", s(ty));
        assert!(referrer.context_module(referrer.context(ix)).is_none());
        assert!(referrer.context_self(referrer.context(ix)).is_none());
        assert!(referrer
            .context_generics(referrer.context(ix))
            .params
            .is_empty());
        // The impl for the marker, which is what the macro emits (`type Type = #ty`).
        let r = referrer
            .repeater_impl(ix, ImplSelf::Marker(&marker), |value, param| {
                slots().encode_const(value, &param.ty)
            })
            .unwrap();
        assert!(r.generics.params.is_empty());
        assert_eq!(s(&r.args), "()");
        assert_eq!(s(&r.self_ty), "__Marker");
        assert_eq!(s(&r.ty), s(ty));
    }
    // A relative type is a required entry, which the macro does not emit; a type it cannot leak
    // is an error, and the macro throws the Leaker away then.
    let item: ItemTrait = parse_quote! { pub trait Tr { fn f(&self) -> Rel; } };
    let mut leaker = Leaker::new();
    leaker.intern_trait(&item, false);
    let referrer = leaker.result().finish().unwrap();
    assert!(referrer.required().next().is_some() && referrer.must(0));
    let item: ItemTrait = parse_quote! { pub trait Tr { fn f(&self) -> Rel<'_>; } };
    let mut leaker = Leaker::new();
    leaker.intern_trait(&item, false);
    let result = leaker.result();
    assert!(result.finish().is_err());
    // `Self` is not leaked with `allow_self == false`: an absolute type that names it is left as
    // written, and is no entry for the macro to skip.
    let item: ItemTrait = parse_quote! {
        pub trait Tr { fn f(&self) -> ::std::vec::Vec<Self>; }
    };
    let mut leaker = Leaker::new();
    leaker.intern_trait(&item, false);
    assert!(leaker.result().finish().unwrap().is_empty());
}

// ── repeater_impl ────────────────────────────────────────────────────────────────────────────

/// `(<S>::A): Clone` as the first predicate is `<S>::A: Clone`.
fn unparenthesized(mut generics: Generics) -> Generics {
    if let Some(WherePredicate::Type(predicate)) = generics
        .where_clause
        .as_mut()
        .and_then(|clause| clause.predicates.first_mut())
    {
        if let Type::Paren(TypeParen { elem, .. }) = &predicate.bounded_ty {
            if matches!(elem.as_ref(), Type::Path(TypePath { qself: Some(_), .. })) {
                predicate.bounded_ty = (**elem).clone();
            }
        }
    }
    generics
}

fn with_where(generics: &Generics) -> String {
    format!("{} {}", s(generics), s(&generics.where_clause))
}

#[test]
fn repeater_impl_for_an_entry_that_does_not_name_self() {
    let (referrer, _) = analyze(&parse_quote! {
        trait Tr<'a, T: 'a + Clone, U> where U: Into<T> {
            fn f(&self, x: &'a Wrap<T, U>) -> Rel;
        }
    });
    let param: Ident = parse_quote!(__S);
    let wrap = referrer
        .iter()
        .position(|ty| s(ty) == "& 'a Wrap < T , U >")
        .unwrap();
    // Any type implements it: the parameter is declared last, and `T: 'a` is encoded, not
    // declared.
    let r = referrer
        .repeater_impl(wrap, ImplSelf::Param(&param), array_const)
        .unwrap();
    assert_eq!(
        with_where(&r.generics),
        with_where(&parse_quote!(
            <'a, T: Clone, U: Into<T>, __S: ?::core::marker::Sized>
        ))
    );
    assert_eq!(s(&r.self_ty), "__S");
    assert_eq!(s(&r.args), t(parse_quote!((&'a (), T, U, &'a T, &'a U))));
    assert_eq!(s(&r.ty), "& 'a Wrap < T , U >");
    assert_eq!(r.module, None);
    // For a marker, nothing else is declared.
    let marker: Type = parse_quote!(crate::Marker);
    let r = referrer
        .repeater_impl(wrap, ImplSelf::Marker(&marker), array_const)
        .unwrap();
    assert_eq!(
        with_where(&r.generics),
        with_where(&parse_quote!(<'a, T: Clone, U: Into<T>>))
    );
    assert_eq!(s(&r.self_ty), "crate :: Marker");
    assert_eq!(s(&r.args), t(parse_quote!((&'a (), T, U, &'a T, &'a U))));
    // A plain entry: no generics, `()`.
    let rel = referrer.iter().position(|ty| s(ty) == "Rel").unwrap();
    assert!(referrer.is_plain(rel));
    let r = referrer
        .repeater_impl(rel, ImplSelf::Marker(&marker), array_const)
        .unwrap();
    assert!(r.generics.params.is_empty());
    assert_eq!(s(&r.args), "()");
    assert_eq!(s(&r.ty), "Rel");
}

#[test]
fn repeater_impl_for_the_implementor_of_a_trait() {
    let (referrer, _) = analyze(&parse_quote! {
        trait Tr<'a, T> {
            type A;
            fn f(&self, x: &'a Rel<Self::A>) -> Wrap<T, Self>;
        }
    });
    let param: Ident = parse_quote!(__Impl);
    let wrap = referrer
        .iter()
        .position(|ty| s(ty) == "Wrap < T , Self >")
        .unwrap();
    // The parameter has the bound of the trait with its arguments, and replaces `Self`.
    let r = referrer
        .repeater_impl(wrap, ImplSelf::Param(&param), array_const)
        .unwrap();
    assert_eq!(
        with_where(&r.generics),
        with_where(&parse_quote!(<'a, T, __Impl: ?::core::marker::Sized + Tr<'a, T>>))
    );
    assert_eq!(s(&r.self_ty), "__Impl");
    assert_eq!(s(&r.ty), "Wrap < T , __Impl >");
    // There is no `Self` to replace in a marker's impl.
    let marker: Type = parse_quote!(crate::Marker);
    match referrer.repeater_impl(wrap, ImplSelf::Marker(&marker), array_const) {
        Err(EncodeError::NeedsSelf(ty)) => assert_eq!(s(&ty), "Wrap < T , Self >"),
        other => panic!("{:?}", other),
    }
    // `&'a Rel<Self::A>` is under a reference: the predicate of `Self::A` is in its where clause,
    // and as the first predicate it is parenthesized, or rustc takes `where <__Impl>` for generics.
    let rel = referrer
        .iter()
        .position(|ty| s(ty) == "& 'a Rel < Self :: A >")
        .unwrap();
    let r = referrer
        .repeater_impl(rel, ImplSelf::Param(&param), array_const)
        .unwrap();
    let clause = r.generics.where_clause.as_ref().unwrap();
    assert_eq!(
        s(&clause.predicates.first().unwrap()),
        "(< __Impl > :: A) : 'a"
    );
    assert_eq!(s(&r.ty), "& 'a Rel < < __Impl > :: A >");
    // Another predicate first is not touched.
    let original = unparenthesized(r.generics.clone());
    assert_eq!(s(&original.where_clause), "where < __Impl > :: A : 'a");
}

#[test]
fn repeater_impl_for_a_type_is_for_the_type() {
    let (referrer, _) = analyze(&parse_quote! {
        struct S<'a, T: 'a> { f: &'a Rel<Self>, g: Wrap<T> }
    });
    let param: Ident = parse_quote!(__S);
    let rel = referrer
        .iter()
        .position(|ty| s(ty) == "& 'a Rel < Self >")
        .unwrap();
    // The impl is for the struct, with `Self` as written and as the argument of `Self: 'a`.
    let r = referrer
        .repeater_impl(rel, ImplSelf::Param(&param), array_const)
        .unwrap();
    assert_eq!(s(&r.self_ty), "S < 'a , T >");
    assert_eq!(with_where(&r.generics), with_where(&parse_quote!(<'a, T>)));
    assert_eq!(s(&r.args), t(parse_quote!((&'a (), T, &'a T, &'a Self))));
    assert_eq!(s(&r.ty), "& 'a Rel < Self >");
    // It cannot be for a marker.
    let marker: Type = parse_quote!(crate::Marker);
    assert!(matches!(
        referrer.repeater_impl(rel, ImplSelf::Marker(&marker), array_const),
        Err(EncodeError::NeedsSelf(_))
    ));
    // What does not name `Self` is for any type.
    let wrap = referrer
        .iter()
        .position(|ty| s(ty) == "Wrap < T >")
        .unwrap();
    let r = referrer
        .repeater_impl(wrap, ImplSelf::Param(&param), array_const)
        .unwrap();
    assert_eq!(s(&r.self_ty), "__S");
}

#[test]
fn repeater_impl_in_a_module_tells_the_module() {
    let (referrer, context) = analyze(&parse_quote! {
        mod m { pub struct S(pub Rel); pub mod n { pub struct T(pub Rel); } }
    });
    let param: Ident = parse_quote!(__S);
    let modules: Vec<Option<ContextId>> = (0..referrer.iter().count())
        .map(|ix| {
            referrer
                .repeater_impl(ix, ImplSelf::Param(&param), array_const)
                .unwrap()
                .module
        })
        .collect();
    assert_eq!(modules.len(), 2);
    assert!(modules.iter().all(Option::is_some));
    assert_ne!(modules[0], modules[1]);
    let paths: Vec<Vec<String>> = modules
        .iter()
        .map(|module| {
            referrer
                .module_path(module.unwrap())
                .unwrap()
                .iter()
                .map(|ident| ident.to_string())
                .collect()
        })
        .collect();
    assert!(paths.contains(&vec![]), "{:?}", paths);
    assert!(paths.contains(&vec!["n".to_string()]), "{:?}", paths);
    let _ = context;
}

#[test]
fn const_parameters_are_encoded_by_the_closure_or_as_array_lengths() {
    let (referrer, _) = analyze(&parse_quote! {
        trait Tr<const N: usize> { fn f(&self) -> Wrap<N>; }
    });
    let param: Ident = parse_quote!(__S);
    let r = referrer
        .repeater_impl(0, ImplSelf::Param(&param), array_const)
        .unwrap();
    assert_eq!(s(&r.args), t(parse_quote!(([(); N],))));
    let site = UseSite::substituting(&parse_quote!(const N: usize), &[parse_quote!(3)]);
    // The same closure encodes the parameter and the argument.
    let calls = std::cell::RefCell::new(Vec::new());
    let mut encode = |value: &Expr, param: &ConstParam| {
        calls.borrow_mut().push((s(value), s(&param.ty)));
        Ok(parse_quote!(Const<#value>))
    };
    let r = referrer
        .repeater_impl(0, ImplSelf::Param(&param), &mut encode)
        .unwrap();
    assert_eq!(s(&r.args), t(parse_quote!((Const<N>,))));
    let args = referrer.entry_args(0, &site, &mut encode).unwrap();
    assert_eq!(s(&args), t(parse_quote!((Const<3>,))));
    assert_eq!(
        *calls.borrow(),
        [
            ("N".to_string(), "usize".to_string()),
            ("3".to_string(), "usize".to_string())
        ]
    );
    // Without a closure, only `usize` has an encoding.
    let (referrer, _) = analyze(&parse_quote! {
        trait Tr<const B: bool> { fn f(&self) -> Wrap<B>; }
    });
    let err = referrer
        .repeater_impl(0, ImplSelf::Param(&param), array_const)
        .unwrap_err();
    assert!(
        matches!(err, EncodeError::UnsupportedConstType(_)),
        "{}",
        err
    );
    let site = UseSite::substituting(&parse_quote!(const B: bool), &[parse_quote!(true)]);
    let err = referrer.entry_args(0, &site, array_const).unwrap_err();
    assert!(
        matches!(err, EncodeError::UnsupportedConstType(_)),
        "{}",
        err
    );
    // The slots take any primitive, on both sides, with the same closure.
    let slots = slots();
    let encode = |value: &Expr, param: &ConstParam| slots.encode_const(value, &param.ty);
    let r = referrer
        .repeater_impl(0, ImplSelf::Param(&param), encode)
        .unwrap();
    let args = referrer
        .entry_args(0, &UseSite::identity(), encode)
        .unwrap();
    assert_eq!(s(&r.args), s(&args));
    assert_eq!(
        s(&args),
        t(parse_quote!((
            *const dyn crate::Repeater<(), 0, 0, B, Type = ()>,
        )))
    );
}

/// Asserts that `generics` are declared in the order that every rustc accepts: the lifetimes, then
/// the types, then the consts (before 1.59 a type after a const is an error).
fn assert_declared_in_order(generics: &Generics, what: &str) {
    let rank = |param: &GenericParam| match param {
        GenericParam::Lifetime(_) => 0,
        GenericParam::Type(_) => 1,
        GenericParam::Const(_) => 2,
    };
    let ranks: Vec<_> = generics.params.iter().map(rank).collect();
    assert!(
        ranks.windows(2).all(|w| w[0] <= w[1]),
        "{}: {}",
        what,
        s(generics)
    );
}

/// Every entry of every sample, in every mode, is declared in the order of every rustc (F53 in
/// `tests/findings.rs`: the parameter for the implementor came after the const parameters).
#[test]
fn the_generics_of_every_impl_are_declared_in_the_order_of_every_rustc() {
    let param: Ident = parse_quote!(__S);
    let marker: Type = parse_quote!(crate::Marker);
    let mut with_consts = 0;
    for item in bucket_samples() {
        let (referrer, _) = analyze(&item);
        for ix in 0..referrer.iter().count() {
            let encode = |value: &Expr, param: &ConstParam| slots().encode_const(value, &param.ty);
            for target in [
                ImplSelf::Param(&param),
                ImplSelf::Marker(&marker),
                ImplSelf::Bucket {
                    id: 5,
                    param: &param,
                },
            ] {
                if let Ok(r) = referrer.repeater_impl(ix, target, encode) {
                    let what = format!("{}: {} {:?}", s(&item), ix, target);
                    assert_declared_in_order(&r.generics, &what);
                    with_consts += r
                        .generics
                        .params
                        .iter()
                        .any(|param| matches!(param, GenericParam::Const(_)))
                        as usize;
                }
            }
        }
    }
    assert!(with_consts > 5, "{}", with_consts);
}

// ── entry_args ───────────────────────────────────────────────────────────────────────────────

#[test]
fn entry_args_substitutes_the_generics_of_the_trait_and_not_those_of_a_method() {
    let tr: ItemTrait = parse_quote! {
        trait Tr<'a, T: 'a> {
            fn f<'b, U: 'b>(&self, x: &'a Wrap<T>, y: &'b Wrap2<T, U>);
        }
    };
    let (referrer, _) = analyze(&Item::Trait(tr.clone()));
    let wrap2 = referrer
        .iter()
        .position(|ty| s(ty) == "Wrap2 < T , U >")
        .unwrap();
    let args: Vec<GenericArgument> = vec![parse_quote!('static), parse_quote!(Vec<u8>)];
    let site = UseSite::substituting(&tr.generics.params, &args);
    // `'b` and `U` are the method's, passed by their own names.
    assert_eq!(
        s(&referrer.entry_args(wrap2, &site, array_const).unwrap()),
        t(parse_quote!((
            &'static (),
            &'b (),
            Vec<u8>,
            U,
            &'static Vec<u8>,
            &'b Vec<u8>,
            &'b U
        )))
    );
    // Too few arguments.
    let site = UseSite::substituting(&tr.generics.params, &args[..1]);
    assert!(matches!(
        referrer.entry_args(wrap2, &site, array_const),
        Err(EncodeError::MissingArgument(_))
    ));
}

#[test]
fn entry_args_with_the_defaults_completed() {
    // The arguments written at the site are completed first; an entry may be a default, in which
    // case the parameters before it are the ones to substitute.
    let tr: ItemTrait = parse_quote! {
        trait Tr<T, U = Rel<T>> { fn f(&self, t: T) -> Rel<T>; }
    };
    let (referrer, _) = analyze(&Item::Trait(tr.clone()));
    let written: AngleBracketedGenericArguments = parse_quote!(<u8>);
    let completed = referrer
        .complete_args(&written.args, &tr.generics.params, |_, ix, completed| {
            // The parameters of the trait can all be given: the entry of a default uses only the
            // ones before it.
            let site = UseSite::substituting(&tr.generics.params, completed);
            let args = referrer.entry_args(ix, &site, array_const).unwrap();
            parse_quote!(<Self as crate::Repeater<#args, 1, #ix>>::Type)
        })
        .unwrap();
    assert_eq!(
        completed.iter().map(s).collect::<Vec<_>>(),
        [
            "u8".to_string(),
            t(parse_quote!(
                <Self as crate::Repeater<(u8,), 1, 0usize>>::Type
            ))
        ]
    );
}

// ── The items of the errors ──────────────────────────────────────────────────────────────────

#[test]
fn items_has_every_offending_path() {
    let item: ItemTrait = parse_quote! {
        pub trait MyTrait: Clone + MyBound {
            fn f(&self, x: MyType<'_>);
        }
    };
    let mut leaker = Leaker::new();
    leaker.intern_trait(&item, true);
    let result = leaker.result();
    let err = result.errors();
    let items = &err;
    assert_eq!(items.len(), 3);
    assert_eq!(
        items
            .iter()
            .map(|item| item.message.as_str())
            .collect::<Vec<_>>(),
        [
            "use absolute path: 'Clone'",
            "use absolute path: 'MyBound'",
            "use absolute path: 'MyType < '_ >'"
        ]
    );
    assert_eq!(
        items
            .iter()
            .map(|item| item.cause.to_token_stream().to_string())
            .collect::<Vec<_>>(),
        ["Clone", "MyBound", "MyType < '_ >"]
    );
    // The help says what to write, the notes why.
    assert_eq!(
        items[0].help_text(),
        Some("use `::core::clone::Clone` instead of `Clone`")
    );
    assert_eq!(
        items[1].help_text(),
        Some(
            "write `MyBound` with an absolute path (e.g. `::my_crate::MyBound`), or allow it \
             with `Allow::allow_path_if()`"
        )
    );
    assert!(items[2]
        .help_text()
        .unwrap()
        .starts_with("write `MyType` with an absolute path"));
    assert!(items.iter().all(|item| item.help.is_some()));
    assert_eq!(items[0].notes.first().map(|(_, n)| n.as_str()), None);
    assert_eq!(items[1].notes.first().map(|(_, n)| n.as_str()), None);
    assert!(items[2]
        .notes
        .first()
        .map(|(_, n)| n.as_str())
        .unwrap()
        .contains("because of `'_`"));
    assert_eq!(items[2].notes.len(), 1);
    // The first is what the first-error accessors tell.
    assert_eq!(err[0].notes.first().map(|(_, n)| n.as_str()), None);
    assert_eq!(err[0].help_text(), items[0].help_text());
    let _ = err[0].span();
    // The messages are what the headlines say, and `syn::Error` has one error for each.
    assert_eq!(
        messages(err),
        items
            .iter()
            .map(|item| item.message.clone())
            .collect::<Vec<_>>()
            .join("\n")
    );
    assert_eq!(to_syn_error(err).into_iter().count(), items.len());
}

#[test]
fn items_of_self_without_meaning() {
    let mut leaker = Leaker::new();
    leaker.enter_generics(&Generics::default(), |view| {
        view.visit_type(&parse_quote!(Rel<Self>));
        view.visit_type(&parse_quote!(Wrap<Self::Item>));
    });
    let result = leaker.result();
    let err = result.errors();
    let items = err;
    assert_eq!(
        items
            .iter()
            .map(|item| item.cause.to_token_stream().to_string())
            .collect::<Vec<_>>(),
        ["Self", "Self :: Item"]
    );
    for item in items {
        assert_eq!(item.message, "`Self` has no meaning here");
        assert!(item
            .notes
            .first()
            .map(|(_, n)| n.as_str())
            .unwrap()
            .starts_with("`Self` has no meaning here: it is declared only"));
        assert!(item
            .help_text()
            .unwrap()
            .starts_with("write the type instead of `Self`, or enter a scope that declares"));
    }
}

#[test]
fn display_uses_the_headline_of_the_cause() {
    // A cause that is not that a path is relative has its own headline, in the messages, in the
    // first error, in `syn::Error` and in the first line of the tokens.
    fn check(err: &[ErrorItem], headline: &str) {
        assert_eq!(messages(err), headline);
        assert_eq!(err[0].message, headline);
        assert!(to_syn_error(err).to_string().starts_with(headline));
    }
    // A default that names `Self`: it was "use absolute path: 'Self'". The cause is the whole type
    // of the default, and the note is at the `Self` in it.
    let mut leaker = Leaker::new();
    leaker.intern_trait(
        &parse_quote!(
            trait Tr<T, U = Rel<Self>> {}
        ),
        true,
    );
    let result = leaker.result();
    let err = result.errors();
    assert_eq!(err[0].cause.to_token_stream().to_string(), "Rel < Self >");
    assert_eq!(err[0].notes[0].0.to_string(), "Self");
    check(
        err,
        "`Rel<Self>` is in the default of a parameter, and names `Self`",
    );
    // `Self` where a generic has a bound that names a `Self` which has no name here.
    let mut leaker = Leaker::new();
    leaker.enter_trait(
        &parse_quote!(Tr),
        &parse_quote!(<T: ::p::Tr<Self>>),
        true,
        |view| {
            view.enter_type(&parse_quote!(S), &Generics::default(), true, |view| {
                view.visit_type(&parse_quote!(Rel<T>))
            })
        },
    );
    let result = leaker.result();
    let err = result.errors();
    check(
        err,
        "the bound of `T` mentions a `Self` which has no name here",
    );
    // `Self` in a module.
    let mut leaker = Leaker::new();
    leaker.enter_mod("m", [], |view| view.visit_type(&parse_quote!(Self)));
    let result = leaker.result();
    let err = result.errors();
    check(err, "`Self` has no meaning here");
    // A relative path keeps "use absolute path", with the reason in the notes.
    let mut leaker = Leaker::new();
    leaker.intern_trait(
        &parse_quote!(
            trait Tr: Rel {}
        ),
        true,
    );
    let result = leaker.result();
    let err = result.errors();
    check(err, "use absolute path: 'Rel'");
    let mut leaker = Leaker::new();
    leaker.intern_trait(
        &parse_quote!(
            trait Tr<'a, T: 'a> {
                fn g(x: &Rel<T>) -> &u8;
            }
        ),
        true,
    );
    let result = leaker.result();
    let err = result.errors();
    check(err, "use absolute path: 'Rel < T >'");
}

// ── bucket_self and ImplSelf::Bucket ─────────────────────────────────────────────────────────

/// The samples, and the shapes that bucketing tells apart: a trait whose generics name `Self` and
/// whose default does not, a struct whose bound names its own `Self` (`SelfRequirement::Any`), an
/// impl and a module with a nested trait.
fn bucket_samples() -> Vec<Item> {
    let mut items = samples();
    items.extend([
        parse_quote! {
            trait Tr<T: AsRef<Self>, U = Rel<T>> { fn f(&self, t: T) -> U; fn g(&self) -> Wrap<Self>; }
        },
        parse_quote! {
            struct S<T: AsRef<Self>> { a: Wrap<T>, b: Rel, c: ::p::W<Self> }
        },
        parse_quote! {
            impl<T: Clone> ::p::Tr<Rel2<T>> for W<T> { fn f(&self) -> Rel3 { todo!() } }
        },
        parse_quote! {
            mod m {
                pub trait Tr<T: AsRef<Self>, U = Rel<T>> { fn f(&self) -> Wrap<Self::A>; }
                pub struct S<'a, T: 'a> { x: &'a Rel<T> }
            }
        },
    ]);
    items
}

/// The last segment of the primitive type of a bucket: `f32`.
fn bucket_name(ty: &Type) -> String {
    let text = t(ty.clone());
    let name = text.strip_prefix(":: core :: primitive :: ");
    name.unwrap_or_else(|| panic!("not a primitive: {}", text))
        .to_string()
}

fn names_self(generics: &Generics) -> bool {
    fn has_self(tokens: proc_macro2::TokenStream) -> bool {
        tokens.into_iter().any(|tree| match tree {
            proc_macro2::TokenTree::Ident(ident) => ident == "Self",
            proc_macro2::TokenTree::Group(group) => has_self(group.stream()),
            _ => false,
        })
    }
    let where_clause = &generics.where_clause;
    has_self(quote!(#generics #where_clause))
}

/// `Referrer::bucket_self()`, `Referrer::needs_self()` and `repeater_impl(.., ImplSelf::Bucket)`
/// agree for every entry: the impl of an entry that has a bucket is the one for it as for a marker,
/// any other is the one for a parameter, an entry has a bucket exactly if neither its type nor its
/// generics name `Self` (`needs_self()` is false), and `ImplSelf::Marker` fails exactly where
/// `needs_self()` is true.
#[test]
fn bucket_self_agrees_with_the_impl_of_every_entry() {
    let param: Ident = parse_quote!(__S);
    let (mut bucketed, mut other, mut generic_self) = (0, 0, 0);
    for item in bucket_samples() {
        let (referrer, _) = analyze(&item);
        for ix in 0..referrer.iter().count() {
            for id in [0, 1, 7, 12345, u64::MAX] {
                let what = format!("{}: {} ({})", s(&item), ix, id);
                let context = referrer.context(ix);
                let bucket = referrer.bucket_self(ix, id);
                let encode =
                    |value: &Expr, param: &ConstParam| slots().encode_const(value, &param.ty);
                let r = referrer
                    .repeater_impl(ix, ImplSelf::Bucket { id, param: &param }, encode)
                    .unwrap_or_else(|e| panic!("{}: {}", what, e));
                let own_self = referrer.context_self(context).is_some();
                let generics_self = names_self(referrer.context_generics(context));
                assert_eq!(bucket.is_some(), !own_self && !generics_self, "{}", what);
                let needs_self = referrer.needs_self(ix);
                assert_eq!(needs_self, own_self || generics_self, "{}", what);
                assert_eq!(needs_self, bucket.is_none(), "{}", what);
                let marker: Type = parse_quote!(crate::Marker);
                let as_marker = referrer.repeater_impl(ix, ImplSelf::Marker(&marker), encode);
                assert_eq!(
                    matches!(as_marker, Err(EncodeError::NeedsSelf(_))),
                    needs_self,
                    "{}",
                    what
                );
                assert!(as_marker.is_ok() || needs_self, "{}", what);
                match bucket {
                    Some(ty) => {
                        bucketed += 1;
                        bucket_name(&ty);
                        assert_eq!(r.self_ty, ty, "{}", what);
                        let expected = referrer
                            .repeater_impl(ix, ImplSelf::Marker(&ty), encode)
                            .unwrap();
                        assert_eq!(r, expected, "{}", what);
                        // No `__S`.
                        assert!(!with_where(&r.generics).contains("__S"), "{}", what);
                        // It does not depend on how often it is asked.
                        assert_eq!(referrer.bucket_self(ix, id), Some(ty), "{}", what);
                    }
                    None => {
                        other += 1;
                        generic_self += (!own_self) as usize;
                        let expected = referrer
                            .repeater_impl(ix, ImplSelf::Param(&param), encode)
                            .unwrap();
                        assert_eq!(r, expected, "{}", what);
                    }
                }
            }
        }
    }
    assert!(bucketed > 50 && other > 50, "{} {}", bucketed, other);
    // The entries of a default whose generics name `Self` while it does not: they were looked for.
    assert!(generic_self > 0);
}

/// `trait Tr<T: AsRef<Self>, U = Rel<T>>`: the default `Rel<T>` does not name `Self`, so
/// `context_self()` is `None`, but the generics of its context do. With a concrete type that
/// `Self` would be that type, so there is no bucket: the impl is for a parameter, which is the
/// implementor of the trait in the copy.
#[test]
fn a_default_whose_generics_name_self_has_no_bucket() {
    let (referrer, _) = analyze(&parse_quote! {
        trait Tr<T: AsRef<Self>, U = Rel<T>> { fn f(&self, t: T) -> U; }
    });
    let rel = referrer.iter().position(|ty| s(ty) == "Rel < T >").unwrap();
    let context = referrer.context(rel);
    assert!(referrer.context_self(context).is_none());
    assert!(names_self(referrer.context_generics(context)));
    assert_eq!(referrer.bucket_self(rel, 7), None);
    let param: Ident = parse_quote!(__S);
    let r = referrer
        .repeater_impl(
            rel,
            ImplSelf::Bucket {
                id: 7,
                param: &param,
            },
            array_const,
        )
        .unwrap();
    assert_eq!(s(&r.self_ty), "__S");
    assert_eq!(
        r,
        referrer
            .repeater_impl(rel, ImplSelf::Param(&param), array_const)
            .unwrap()
    );
    assert_eq!(
        with_where(&r.generics),
        with_where(&parse_quote!(<T: AsRef<Self>, __S: ?::core::marker::Sized>))
    );
}

/// The entries of one item do not all share a bucket, and the same entry of other ids does not
/// either: the type is `splitmix64(id + ix * 0x9E3779B97F4A7C15)` modulo 17 of the list.
#[test]
fn bucket_types_follow_the_documented_hash() {
    const BUCKETS: [&str; 17] = [
        "bool", "char", "u8", "u16", "u32", "u64", "u128", "usize", "i8", "i16", "i32", "i64",
        "i128", "isize", "f32", "f64", "str",
    ];
    fn splitmix64(x: u64) -> u64 {
        let mut z = x.wrapping_add(0x9E3779B97F4A7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    let (referrer, _) = analyze(&parse_quote! {
        struct S { a: Rel, b: Rel2, c: Rel3, d: Rel4 }
    });
    for id in [0u64, 1, 7, 255, 1 << 40, u64::MAX] {
        for ix in 0..referrer.iter().count() {
            let h = splitmix64(id.wrapping_add((ix as u64).wrapping_mul(0x9E3779B97F4A7C15)));
            let expected = BUCKETS[(h % 17) as usize];
            assert_eq!(
                bucket_name(&referrer.bucket_self(ix, id).unwrap()),
                expected,
                "{} {}",
                id,
                ix
            );
        }
    }
    // The mapping is part of the format: these values must not change without a bump of
    // `Referrer::FORMAT_VERSION`.
    for (id, ix, expected) in [
        (0, 0, "i128"),
        (1, 0, "i32"),
        (7, 0, "bool"),
        (7, 1, "usize"),
        (0, 1, "i128"),
        (0, 2, "i16"),
        (u64::MAX, 0, "f32"),
        (u64::MAX, 3, "f64"),
    ] {
        assert_eq!(
            bucket_name(&referrer.bucket_self(ix, id).unwrap()),
            expected,
            "{} {}",
            id,
            ix
        );
    }
}

/// How many entries each of the 17 buckets gets: every one is used, and none gets more than twice
/// its share.
fn assert_spread(names: impl Iterator<Item = String>, what: &str) {
    let mut counts = std::collections::BTreeMap::new();
    let mut total = 0;
    for name in names {
        *counts.entry(name).or_insert(0usize) += 1;
        total += 1;
    }
    assert_eq!(counts.len(), 17, "{}: {:?}", what, counts);
    let share = total as f64 / 17.0;
    for (name, count) in &counts {
        assert!(
            (*count as f64) <= 2.0 * share,
            "{}: {} has {} of {}: {:?}",
            what,
            name,
            count,
            total,
            counts
        );
    }
}

#[test]
fn bucket_types_spread_over_ids_and_entries() {
    let (referrer, _) = analyze(&parse_quote! { struct S { a: Rel } });
    let bucket = |id: u64| bucket_name(&referrer.bucket_self(0, id).unwrap());
    assert_spread((0..1000u64).map(bucket), "ids 0..1000");
    assert_spread((0..1000u64).map(|k| k << 8).map(bucket), "k << 8");
    assert_spread((0..1000u64).map(|k| k << 16).map(bucket), "k << 16");
    assert_spread((0..1000u64).map(|k| k << 32).map(bucket), "k << 32");
    assert_spread((0..1000u64).map(|k| k * 17).map(bucket), "k * 17");
    // The entries of one item, with one id.
    let fields: Vec<_> = (0..600)
        .map(|n| {
            let (name, ty) = (
                format_ident(&format!("f{}", n)),
                format_ident(&format!("R{}", n)),
            );
            quote!(#name: #ty,)
        })
        .collect();
    let (referrer, _) = analyze(&parse_quote! { struct Big { #(#fields)* } });
    assert_eq!(referrer.iter().count(), 600);
    for id in [0u64, 7, u64::MAX] {
        assert_spread(
            (0..600).map(|ix| bucket_name(&referrer.bucket_self(ix, id).unwrap())),
            "the entries of one item",
        );
    }
}

fn format_ident(name: &str) -> Ident {
    Ident::new(name, proc_macro2::Span::call_site())
}

/// `ImplSelf` stays `Clone`, `Copy` and `Debug`.
#[test]
fn impl_self_bucket_is_copy_and_debug() {
    let param: Ident = parse_quote!(__S);
    let target = ImplSelf::Bucket {
        id: 3,
        param: &param,
    };
    let copy = target;
    assert_eq!(format!("{:?}", target), format!("{:?}", copy));
    assert!(format!("{:?}", target).contains("Bucket"));
    fn is_clone<T: Clone>() {}
    is_clone::<ImplSelf<'_>>();
}

// ── ImplSelf::Marker for an entry whose generics name Self ───────────────────────────────────

/// `Marker` for the entry of `trait Tr<T: AsRef<Self>, U = Rel<T>>`: `context_self()` is `None`, but
/// the generics of its context name `Self` (`needs_self()`), which in an impl for a marker would be
/// the marker: `impl<T: AsRef<Self>> Repeater<..> for Marker` bounds `T: AsRef<Marker>`, where the
/// copy has `T: AsRef<Host>`, so an impl of the copy that instantiates the default (`impl
/// Tr<String> for Host`) fails with E0277 (`String: AsRef<Marker>`). So it is `NeedsSelf`, as for an
/// entry that names `Self`; `ImplSelf::Param` (or `Bucket`, which falls back to it) is the choice.
#[test]
fn marker_is_refused_for_the_entry_of_a_default_whose_generics_name_self() {
    let (referrer, _) = analyze(&parse_quote! {
        trait Tr<T: AsRef<Self>, U = Rel<T>> { fn f(&self, t: T) -> U; }
    });
    let rel = referrer.iter().position(|ty| s(ty) == "Rel < T >").unwrap();
    assert!(referrer.context_self(referrer.context(rel)).is_none());
    assert!(referrer.needs_self(rel));
    let marker: Type = parse_quote!(crate::Marker);
    match referrer.repeater_impl(rel, ImplSelf::Marker(&marker), array_const) {
        Err(EncodeError::NeedsSelf(ty)) => assert_eq!(s(&*ty), "Rel < T >"),
        other => panic!("{:?}", other.map(|r| r.self_ty)),
    }
    // For a parameter, `Self` of the generics is the parameter.
    let param: Ident = parse_quote!(__S);
    let r = referrer
        .repeater_impl(rel, ImplSelf::Param(&param), array_const)
        .unwrap();
    assert_eq!(s(&r.self_ty), "__S");
    assert_eq!(
        with_where(&r.generics),
        with_where(&parse_quote!(<T: AsRef<Self>, __S: ?::core::marker::Sized>))
    );
}

/// `needs_self()` is true for an entry that names `Self` and for one whose generics do, not for the
/// others.
#[test]
fn needs_self_for_an_entry_that_names_self_or_whose_generics_do() {
    let (referrer, _) = analyze(&parse_quote! {
        trait Tr<T: AsRef<Self>, U = Rel<T>> { fn f(&self, t: T) -> U; fn g(&self) -> Wrap<Self>; }
    });
    let wrap = referrer
        .iter()
        .position(|ty| s(ty) == "Wrap < Self >")
        .unwrap();
    let rel = referrer.iter().position(|ty| s(ty) == "Rel < T >").unwrap();
    assert!(referrer.needs_self(wrap) && referrer.needs_self(rel));
    let (referrer, _) = analyze(&parse_quote! {
        trait Tr<T, U = Rel<T>> { fn f(&self, t: T) -> U; fn g(&self) -> Wrap<T>; }
    });
    assert!(referrer.iter().count() > 0);
    assert!((0..referrer.iter().count()).all(|ix| !referrer.needs_self(ix)));
}

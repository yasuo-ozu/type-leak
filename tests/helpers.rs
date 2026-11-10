//! Tests for the helpers for macro authors: the order of the entries, the description of the Repeater
//! impl of an entry (`LeakerResult::repeater_impls()`), the arguments of an entry at a use site
//! (`encode_generics_args_with_params()`), and the items of the errors of the analysis (`Vec<ErrorItem>`).
//!
//! The impl and the arguments are compared with what a macro wrote by hand before the helpers
//! (the replacement of `Self`, `without_encoded_bounds()` and the encoders, and the arguments of a
//! context with the encoders), over all the entries of a set of items, in `src/encode/tests.rs`.

mod common;

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

fn repeater() -> Path {
    parse_quote!(crate::Repeater)
}

fn config() -> Rules<impl Allow + Clone> {
    Rules::new().primitive().prelude()
}

/// Analyzes a whole item the way a downstream macro does.
fn analyze(item: &Item) -> (Referrer, ContextId) {
    let (result, context) = analyze_result(item);
    (result.into_referrer(), context)
}

/// The same, with the result that writes the impls.
fn analyze_result(item: &Item) -> (LeakerResult, ContextId) {
    let mut leaker = Leaker::from_config(config(), Options::new());
    let context = match item {
        Item::Struct(i) => leaker.intern_struct(i, true),
        Item::Enum(i) => leaker.intern_enum(i, true),
        Item::Union(i) => leaker.intern_union(i, true),
        Item::Trait(i) => leaker.intern_trait(i, true),
        Item::Impl(i) => leaker.intern_impl(i, true),
        Item::Mod(i) => leaker.intern_mod(i),
        _ => panic!("unsupported item"),
    };
    let result = leaker.result();
    assert!(
        result.errors().is_empty(),
        "{}: {}",
        s(item),
        messages(result.errors())
    );
    (result, context)
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

// ── entries ──────────────────────────────────────────────────────────────────────────────────

#[test]
fn the_entries_that_depend_on_the_context_come_first() {
    let item: ItemStruct = parse_quote!(
        struct S {
            a: ::foo::X,
            b: Rel,
            c: Wrap<u8>,
            d: ::bar::Y,
        }
    );
    let mut leaker = Leaker::from_config(config(), Options::new());
    leaker.intern_struct(&item, true);
    let result = leaker.result();
    let referrer = result.clone().finish().unwrap();
    let all: Vec<String> = common::types(&referrer).map(s).collect();
    assert!(!referrer.is_empty());
    // `Rel` and `Wrap<u8>` depend on the context: they have the smaller indices, each sorted.
    assert_eq!(
        all,
        ["Rel", "Wrap < u8 >", ":: bar :: Y", ":: foo :: X"],
        "{:?}",
        all
    );
    // Not affected by a rule that allows the crates of the types of other crates.
    let allowed = result.allow(Rules::new().external_crates());
    let allowed = allowed.finish().unwrap();
    assert_eq!(
        allowed.iter().map(|(_, ty, _)| s(ty)).collect::<Vec<_>>(),
        all
    );
    // The serialized Referrer has them in the same order.
    let parsed: Referrer = parse2(quote!(#referrer)).unwrap();
    assert_eq!(common::types(&parsed).map(s).collect::<Vec<_>>(), all);
}

#[test]
fn a_referrer_with_no_entry_is_empty() {
    let item: ItemStruct = parse_quote!(
        struct S {
            a: ::foo::X,
            b: u8,
        }
    );
    // A Leaker that allows the crates finds no entry.
    let rules = config().external_crates();
    let mut leaker = Leaker::from_config(rules, Options::new());
    leaker.intern_struct(&item, true);
    let referrer = leaker.result().finish().unwrap();
    assert!(referrer.is_empty());
    assert_eq!(referrer.iter().count(), 0);
    // The default one finds the type of the other crate.
    let mut leaker = Leaker::from_config(config(), Options::new());
    leaker.intern_struct(&item, true);
    let referrer = leaker.result().finish().unwrap();
    assert!(!referrer.is_empty());
    assert_eq!(referrer.iter().count(), 1);
}

/// The impl of an entry for a marker declares nothing: no generics, where clause or module, and
/// `()` as its arguments.
fn assert_declares_nothing(r: &RepeaterImpl) {
    assert!(r.generics.params.is_empty());
    assert!(r
        .generics
        .where_clause
        .as_ref()
        .map_or(true, |clause| clause.predicates.is_empty()));
    assert_eq!(s(&r.args), "()");
    assert!(r.module.is_none());
}

/// The sumtype shape: `Leaker::new()` (no allowed path) and `intern_trait(.., false)` on a trait
/// whose only entries are the absolute types of external crates. `iter()` counts them, and an impl
/// of the Repeater for a marker declares nothing (no generics, no `Self`, no module), with the
/// entry as the type. A relative type or `Self` makes the call fail and the macro throws the
/// Leaker away.
#[test]
fn a_trait_with_only_external_types_has_entries_that_declare_nothing_for_a_marker() {
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
    let result = leaker.result();
    let referrer = result.clone().finish().unwrap();
    assert!(!referrer.is_empty());
    assert_eq!(
        common::types(&referrer).map(s).collect::<Vec<_>>(),
        [
            ":: std :: io :: Result < :: core :: primitive :: usize >",
            "dyn :: std :: error :: Error + 'static"
        ]
    );
    let marker: Type = parse_quote!(__Marker);
    for (ix, ty) in common::types(&referrer).enumerate() {
        assert!(common::in_module(common::entry(&referrer, ix)).is_none());
        assert!(common::entry(&referrer, ix).self_ty.is_none());
        assert!(common::entry(&referrer, ix).generics.params.is_empty());
        // The impl for the marker, which is what the macro emits (`type Type = #ty`).
        let r = result
            .repeater_impls(
                &repeater(),
                7,
                ImplSelf::Marker(&marker),
                slot_const(&repeater()),
            )
            .unwrap()
            .nth(ix)
            .unwrap();
        assert_declares_nothing(&r);
        assert_eq!(s(&r.self_ty), "__Marker");
        assert_eq!(s(&r.ty), s(ty));
    }
    // A relative type is an entry that depends on the context, which the macro does not emit; a type
    // it cannot leak is an error, and the macro throws the Leaker away then.
    let item: ItemTrait = parse_quote! { pub trait Tr { fn f(&self) -> Rel; } };
    let mut leaker = Leaker::new();
    leaker.intern_trait(&item, false);
    let referrer = leaker.result().finish().unwrap();
    assert_eq!(referrer.iter().count(), 1);
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

// ── repeater_impls ───────────────────────────────────────────────────────────────────────────

fn with_where(generics: &Generics) -> String {
    format!("{} {}", s(generics), s(&generics.where_clause))
}

#[test]
fn repeater_impl_for_an_entry_that_does_not_name_self() {
    let (result, _) = analyze_result(&parse_quote! {
        trait Tr<'a, T: 'a + Clone, U> where U: Into<T> {
            fn f(&self, x: &'a Wrap<T, U>) -> Rel;
        }
    });
    let referrer = result.clone().into_referrer();
    let param: Ident = parse_quote!(__S);
    let wrap = common::types(&referrer)
        .position(|ty| s(ty) == "& 'a Wrap < T , U >")
        .unwrap();
    // Any type implements it: the parameter is declared last, and `T: 'a` is encoded, not
    // declared.
    let r = result
        .repeater_impls(
            &parse_quote!(crate::Repeater),
            7,
            ImplSelf::Param(&param),
            array_const,
        )
        .unwrap()
        .nth(wrap)
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
    let r = result
        .repeater_impls(
            &parse_quote!(crate::Repeater),
            7,
            ImplSelf::Marker(&marker),
            array_const,
        )
        .unwrap()
        .nth(wrap)
        .unwrap();
    assert_eq!(
        with_where(&r.generics),
        with_where(&parse_quote!(<'a, T: Clone, U: Into<T>>))
    );
    assert_eq!(s(&r.self_ty), "crate :: Marker");
    assert_eq!(s(&r.args), t(parse_quote!((&'a (), T, U, &'a T, &'a U))));
    // An entry of a context with no generics: nothing is declared.
    let rel = common::types(&referrer)
        .position(|ty| s(ty) == "Rel")
        .unwrap();
    let r = result
        .repeater_impls(
            &parse_quote!(crate::Repeater),
            7,
            ImplSelf::Marker(&marker),
            array_const,
        )
        .unwrap()
        .nth(rel)
        .unwrap();
    assert_declares_nothing(&r);
    assert_eq!(s(&r.ty), "Rel");
}

#[test]
fn repeater_impl_for_the_implementor_of_a_trait() {
    let (result, _) = analyze_result(&parse_quote! {
        trait Tr<'a, T> {
            type A;
            fn f(&self, x: &'a Rel<Self::A>) -> Wrap<T, Self>;
        }
    });
    let referrer = result.clone().into_referrer();
    let param: Ident = parse_quote!(__Impl);
    let wrap = common::types(&referrer)
        .position(|ty| s(ty) == "Wrap < T , Self >")
        .unwrap();
    // The parameter has the bound of the trait with its arguments, and replaces `Self`.
    let r = result
        .repeater_impls(
            &parse_quote!(crate::Repeater),
            7,
            ImplSelf::Param(&param),
            array_const,
        )
        .unwrap()
        .nth(wrap)
        .unwrap();
    assert_eq!(
        with_where(&r.generics),
        with_where(&parse_quote!(<'a, T, __Impl: ?::core::marker::Sized + Tr<'a, T>>))
    );
    assert_eq!(s(&r.self_ty), "__Impl");
    assert_eq!(s(&r.ty), "Wrap < T , __Impl >");
    // There is no `Self` to replace in a marker's impl: the call fails, for the first of the entries
    // that name `Self` (`Wrap<T, Self>` and `&'a Rel<Self::A>`).
    let rel = common::types(&referrer)
        .position(|ty| s(ty) == "& 'a Rel < Self :: A >")
        .unwrap();
    let marker: Type = parse_quote!(crate::Marker);
    match result.repeater_impls(
        &parse_quote!(crate::Repeater),
        7,
        ImplSelf::Marker(&marker),
        array_const,
    ) {
        Err(EncodeError::NeedsSelf(ty)) => {
            let first = common::types(&referrer).nth(wrap.min(rel)).unwrap();
            assert_eq!(s(&*ty), s(first));
        }
        Err(other) => panic!("{:?}", other),
        Ok(_) => panic!("no error"),
    }
    // `&'a Rel<Self::A>` is under a reference: the predicate of `Self::A` is in its where clause,
    // and as the first predicate it is parenthesized, or rustc takes `where <__Impl>` for generics.
    let r = result
        .repeater_impls(
            &parse_quote!(crate::Repeater),
            7,
            ImplSelf::Param(&param),
            array_const,
        )
        .unwrap()
        .nth(rel)
        .unwrap();
    let clause = r.generics.where_clause.as_ref().unwrap();
    assert_eq!(
        s(&clause.predicates.first().unwrap()),
        "(< __Impl > :: A) : 'a"
    );
    assert_eq!(s(&r.ty), "& 'a Rel < < __Impl > :: A >");
}

#[test]
fn repeater_impl_for_a_type_is_for_the_type() {
    let (result, _) = analyze_result(&parse_quote! {
        struct S<'a, T: 'a> { f: &'a Rel<Self>, g: Wrap<T> }
    });
    let referrer = result.clone().into_referrer();
    let param: Ident = parse_quote!(__S);
    let rel = common::types(&referrer)
        .position(|ty| s(ty) == "& 'a Rel < Self >")
        .unwrap();
    // The impl is for the struct, with `Self` as written and as the argument of `Self: 'a`.
    let r = result
        .repeater_impls(
            &parse_quote!(crate::Repeater),
            7,
            ImplSelf::Param(&param),
            array_const,
        )
        .unwrap()
        .nth(rel)
        .unwrap();
    assert_eq!(s(&r.self_ty), "S < 'a , T >");
    assert_eq!(with_where(&r.generics), with_where(&parse_quote!(<'a, T>)));
    assert_eq!(s(&r.args), t(parse_quote!((&'a (), T, &'a T, &'a Self))));
    assert_eq!(s(&r.ty), "& 'a Rel < Self >");
    // It cannot be for a marker: the call fails for `rel`, the only entry that names `Self`.
    let marker: Type = parse_quote!(crate::Marker);
    let marked = result
        .repeater_impls(
            &parse_quote!(crate::Repeater),
            7,
            ImplSelf::Marker(&marker),
            array_const,
        )
        .err();
    assert!(matches!(
        &marked,
        Some(EncodeError::NeedsSelf(ty)) if s(&**ty) == "& 'a Rel < Self >"
    ));
    // What does not name `Self` is for any type.
    let wrap = common::types(&referrer)
        .position(|ty| s(ty) == "Wrap < T >")
        .unwrap();
    let r = result
        .repeater_impls(
            &parse_quote!(crate::Repeater),
            7,
            ImplSelf::Param(&param),
            array_const,
        )
        .unwrap()
        .nth(wrap)
        .unwrap();
    assert_eq!(s(&r.self_ty), "__S");
}

#[test]
fn repeater_impl_in_a_module_tells_the_module() {
    let (result, context) = analyze_result(&parse_quote! {
        mod m { pub struct S(pub Rel); pub mod n { pub struct T(pub Rel); } }
    });
    let param: Ident = parse_quote!(__S);
    let modules: Vec<Option<ModuleScope>> = result
        .repeater_impls(
            &parse_quote!(crate::Repeater),
            7,
            ImplSelf::Param(&param),
            array_const,
        )
        .unwrap()
        .map(|r| r.module)
        .collect();
    assert_eq!(modules.len(), 2);
    assert!(modules.iter().all(Option::is_some));
    assert_ne!(modules[0], modules[1]);
    // The module's own context: `Itself`, with the path from the root.
    let paths: Vec<Vec<String>> = modules
        .iter()
        .map(|module| match module {
            Some(ModuleScope::Itself {
                path: Some(path), ..
            }) => path.iter().map(|ident| ident.to_string()).collect(),
            other => panic!("not a module of intern_mod(): {:?}", other),
        })
        .collect();
    assert!(paths.contains(&vec![]), "{:?}", paths);
    assert!(paths.contains(&vec!["n".to_string()]), "{:?}", paths);
    let _ = context;
}

#[test]
fn const_parameters_are_encoded_by_the_closure_or_as_array_lengths() {
    let (result, _) = analyze_result(&parse_quote! {
        trait Tr<const N: usize> { fn f(&self) -> Wrap<N>; }
    });
    let referrer = result.clone().into_referrer();
    let param: Ident = parse_quote!(__S);
    let r = result
        .repeater_impls(
            &parse_quote!(crate::Repeater),
            7,
            ImplSelf::Param(&param),
            array_const,
        )
        .unwrap()
        .next()
        .unwrap();
    assert_eq!(s(&r.args), t(parse_quote!(([(); N],))));
    let site = UseSite::substituting(&parse_quote!(const N: usize), &[parse_quote!(3)]);
    // The same closure encodes the parameter and the argument.
    let calls = std::cell::RefCell::new(Vec::new());
    let mut encode = |value: &Expr, param: &ConstParam| {
        calls.borrow_mut().push((s(value), s(&param.ty)));
        Ok(parse_quote!(Const<#value>))
    };
    let r = result
        .repeater_impls(
            &parse_quote!(crate::Repeater),
            7,
            ImplSelf::Param(&param),
            &mut encode,
        )
        .unwrap()
        .next()
        .unwrap();
    assert_eq!(s(&r.args), t(parse_quote!((Const<N>,))));
    let first = common::entry_id_at(&referrer, 0);
    let args = common::repeater_args(
        common::entry(&referrer, first.index()),
        &site,
        None,
        &mut encode,
    )
    .unwrap();
    assert_eq!(s(&args), t(parse_quote!((Const<3>,))));
    // The call checks the parameter before it iterates, and the item encodes it again.
    assert_eq!(
        *calls.borrow(),
        [
            ("N".to_string(), "usize".to_string()),
            ("N".to_string(), "usize".to_string()),
            ("3".to_string(), "usize".to_string())
        ]
    );
    // Without a closure, only `usize` has an encoding.
    let (result, _) = analyze_result(&parse_quote! {
        trait Tr<const B: bool> { fn f(&self) -> Wrap<B>; }
    });
    let referrer = result.clone().into_referrer();
    let err = result
        .repeater_impls(
            &parse_quote!(crate::Repeater),
            7,
            ImplSelf::Param(&param),
            array_const,
        )
        .err()
        .unwrap();
    assert!(
        matches!(err, EncodeError::UnsupportedConstType(_)),
        "{}",
        err
    );
    let site = UseSite::substituting(&parse_quote!(const B: bool), &[parse_quote!(true)]);
    let err = common::repeater_args(
        common::entry(&referrer, first.index()),
        &site,
        None,
        array_const,
    )
    .unwrap_err();
    assert!(
        matches!(err, EncodeError::UnsupportedConstType(_)),
        "{}",
        err
    );
    // The slots take any primitive, on both sides, with the same closure.
    let repeater = repeater();
    let r = result
        .repeater_impls(&repeater, 7, ImplSelf::Param(&param), slot_const(&repeater))
        .unwrap()
        .next()
        .unwrap();
    let args = common::repeater_args(
        common::entry(&referrer, 0),
        &UseSite::identity(),
        None,
        slot_const(&repeater),
    )
    .unwrap();
    assert_eq!(s(&r.args), s(&args));
    assert_eq!(
        s(&args),
        t(parse_quote!((
            *const dyn crate::Repeater<(), 0, B, Type = ()>,
        )))
    );
}

/// The call of `repeater_impls()` makes every check before it returns, and iterates lazily: it
/// calls `const_fn` once for each const parameter of each entry, in order, and stops at the first
/// error. The items call `const_fn` again, as you take them.
#[test]
fn repeater_impls_checks_the_const_parameters_before_it_iterates() {
    let (result, _) = analyze_result(&parse_quote! {
        trait Tr<const N: usize, const M: usize> {
            fn f(&self) -> Wrap<N>;
            fn g(&self) -> Wrap2<M>;
        }
    });
    let param: Ident = parse_quote!(__S);
    let repeater = repeater();
    let calls = std::cell::RefCell::new(Vec::new());
    let encode = |fail: Option<&'static str>| {
        let calls = &calls;
        move |value: &Expr, param: &ConstParam| {
            calls.borrow_mut().push(s(value));
            if fail == Some(param.ident.to_string().as_str()) {
                return Err(EncodeError::UnsupportedConstType(Box::new(
                    param.ty.clone(),
                )));
            }
            Ok(parse_quote!([(); 0]))
        }
    };
    // The first error is that of `M`, the parameter of the second entry: `N` was called before it.
    let result_for_m =
        result.repeater_impls(&repeater, 7, ImplSelf::Param(&param), encode(Some("M")));
    assert!(matches!(
        result_for_m,
        Err(EncodeError::UnsupportedConstType(_))
    ));
    assert_eq!(*calls.borrow(), ["N", "M"]);
    // No error: all the parameters are checked, and no item is made until it is taken.
    calls.borrow_mut().clear();
    let mut impls = result
        .repeater_impls(&repeater, 7, ImplSelf::Param(&param), encode(None))
        .unwrap();
    assert_eq!(*calls.borrow(), ["N", "M"]);
    let first = impls.next().unwrap();
    assert_eq!(s(&first.ty), "Wrap < N >");
    assert_eq!(*calls.borrow(), ["N", "M", "N"]);
    let second = impls.next().unwrap();
    assert_eq!(s(&second.ty), "Wrap2 < M >");
    assert_eq!(*calls.borrow(), ["N", "M", "N", "M"]);
    assert!(impls.next().is_none());
}

/// The error for a marker comes first: the call makes no call of `const_fn` then.
#[test]
fn repeater_impls_checks_a_marker_before_the_const_parameters() {
    let (result, _) = analyze_result(&parse_quote! {
        trait Tr<const N: usize> {
            fn f(&self) -> Wrap<N>;
            fn g(&self) -> Wrap2<Self>;
        }
    });
    let marker: Type = parse_quote!(crate::Marker);
    let repeater = repeater();
    let calls = std::cell::Cell::new(0);
    let error = result
        .repeater_impls(
            &repeater,
            3,
            ImplSelf::Marker(&marker),
            |_: &Expr, param: &ConstParam| {
                calls.set(calls.get() + 1);
                Err(EncodeError::UnsupportedConstType(Box::new(
                    param.ty.clone(),
                )))
            },
        )
        .err()
        .unwrap();
    assert!(matches!(&error, EncodeError::NeedsSelf(ty) if s(&**ty) == "Wrap2 < Self >"));
    assert_eq!(calls.get(), 0);
    // A parameter or a bucket does not fail for `Self`, only for `const_fn`.
    let param: Ident = parse_quote!(__S);
    for target in [ImplSelf::Param(&param), ImplSelf::Bucket { param: &param }] {
        let error = result
            .repeater_impls(&repeater, 3, target, |_: &Expr, param: &ConstParam| {
                calls.set(calls.get() + 1);
                Err(EncodeError::UnsupportedConstType(Box::new(
                    param.ty.clone(),
                )))
            })
            .err()
            .unwrap();
        assert!(matches!(error, EncodeError::UnsupportedConstType(_)));
    }
    assert_eq!(calls.get(), 2);
}

/// `const_fn` must give the same answer every time: the items call it again, and panic if it fails
/// where it did not.
#[test]
#[should_panic(expected = "`const_fn` of `repeater_impls()` failed for the entry 0")]
fn repeater_impls_panics_if_const_fn_changes_its_answer() {
    let (result, _) = analyze_result(&parse_quote! {
        trait Tr<const N: usize> { fn f(&self) -> Wrap<N>; }
    });
    let param: Ident = parse_quote!(__S);
    let repeater = repeater();
    let mut calls = 0;
    let impls = result
        .repeater_impls(
            &repeater,
            7,
            ImplSelf::Param(&param),
            move |_: &Expr, param: &ConstParam| {
                calls += 1;
                if calls == 1 {
                    Ok(parse_quote!([(); 0]))
                } else {
                    Err(EncodeError::UnsupportedConstType(Box::new(
                        param.ty.clone(),
                    )))
                }
            },
        )
        .unwrap();
    for _ in impls {}
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
        let (result, _) = analyze_result(&item);
        let repeater = repeater();
        for target in [
            ImplSelf::Param(&param),
            ImplSelf::Marker(&marker),
            ImplSelf::Bucket { param: &param },
        ] {
            // The call for a marker fails if an entry names `Self`: only the others are compared.
            if let Ok(impls) = result.repeater_impls(&repeater, 5, target, slot_const(&repeater)) {
                for r in impls {
                    let what = format!("{}: {} {:?}", s(&item), r.entry, target);
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

/// `ToTokens` (and `ItemImpl::from(r)`, which it writes) is the impl written by hand from the
/// pieces, with the path of `repeater_trait_path()`, token for token, for every target.
#[test]
fn the_tokens_of_a_repeater_impl_are_the_impl_written_by_hand() {
    let param: Ident = parse_quote!(__S);
    let marker: Type = parse_quote!(crate::Marker);
    let mut compared = 0;
    for item in samples().into_iter().chain(bucket_samples()) {
        let (result, _) = analyze_result(&item);
        let repeater = repeater();
        for target in [
            ImplSelf::Param(&param),
            ImplSelf::Marker(&marker),
            ImplSelf::Bucket { param: &param },
        ] {
            if let Ok(impls) = result.repeater_impls(&repeater, 5, target, slot_const(&repeater)) {
                for r in impls {
                    let what = format!("{}: {} {:?}", s(&item), r.entry, target);
                    let (impl_generics, _, where_clause) = r.generics.split_for_impl();
                    let path = repeater_trait_path(&repeater, &r.args, 5, r.entry);
                    let (self_ty, ty) = (&r.self_ty, &r.ty);
                    let by_hand = quote! {
                        impl #impl_generics #path for #self_ty #where_clause {
                            type Type = #ty;
                        }
                    };
                    assert_eq!(
                        r.to_token_stream().to_string(),
                        by_hand.to_string(),
                        "{}",
                        what
                    );
                    compared += 1;
                }
            }
        }
    }
    assert!(compared > 50, "{}", compared);
}

// ── The arguments of the Repeater ────────────────────────────────────────────────────────────

#[test]
fn the_encoder_substitutes_the_generics_of_the_trait_and_not_those_of_a_method() {
    let tr: ItemTrait = parse_quote! {
        trait Tr<'a, T: 'a> {
            fn f<'b, U: 'b>(&self, x: &'a Wrap<T>, y: &'b Wrap2<T, U>);
        }
    };
    let (referrer, _) = analyze(&Item::Trait(tr.clone()));
    let wrap2 = common::types(&referrer)
        .position(|ty| s(ty) == "Wrap2 < T , U >")
        .unwrap();
    let args: Vec<GenericArgument> = vec![parse_quote!('static), parse_quote!(Vec<u8>)];
    let site = UseSite::substituting(&tr.generics.params, &args);
    // `'b` and `U` are the method's, passed by their own names.
    assert_eq!(
        s(
            &common::repeater_args(common::entry(&referrer, wrap2), &site, None, array_const)
                .unwrap()
        ),
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
        common::repeater_args(common::entry(&referrer, wrap2), &site, None, array_const),
        Err(EncodeError::MissingArgument(_))
    ));
}

#[test]
fn the_encoder_with_the_defaults_completed() {
    // The arguments written at the site are completed first; an entry may be a default, in which
    // case the parameters before it are the ones to substitute.
    let tr: ItemTrait = parse_quote! {
        trait Tr<T, U = Rel<T>> { fn f(&self, t: T) -> Rel<T>; }
    };
    let (referrer, context) = analyze(&Item::Trait(tr.clone()));
    let written: AngleBracketedGenericArguments = parse_quote!(<u8>);
    let completed = referrer
        .complete_args(
            context,
            &tr.generics.params,
            &written.args,
            |_, ix, context, completed| {
                // The parameters of the trait can all be given: the entry of a default uses only the
                // ones before it.
                let site = UseSite::substituting(&tr.generics.params, completed);
                let args = common::repeater_args(context, &site, None, array_const).unwrap();
                repeater_projection(&repeater(), &parse_quote!(Self), &args, 1, ix)
            },
        )
        .unwrap();
    let key = repeater_key(1, common::entry_id_at(&referrer, 0));
    assert_eq!(
        completed.iter().map(s).collect::<Vec<_>>(),
        [
            "u8".to_string(),
            t(parse_quote!(
                <Self as crate::Repeater<(u8,), #key>>::Type
            ))
        ]
    );
}

// ── The items of the errors ──────────────────────────────────────────────────────────────────

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
    check(err, "use absolute path: 'Rel<T>'");
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

/// `Referrer::bucket_self()` and `repeater_impls(ImplSelf::Bucket)` agree for every entry: the
/// impl of an entry that has a bucket is the one for it as for a marker, any other is the one for a
/// parameter, an entry has a bucket exactly if neither its type nor its generics name `Self`, and
/// the call for `ImplSelf::Marker` fails exactly if some entry has none (for the first of them).
///
/// The impl for a marker is made only where no entry names `Self`: where one does, the impl of an
/// entry that has a bucket is compared with the impl for the parameter, without the parameter and
/// for the bucket.
#[test]
fn bucket_self_agrees_with_the_impl_of_every_entry() {
    let param: Ident = parse_quote!(__S);
    let (mut bucketed, mut other, mut generic_self) = (0, 0, 0);
    let (mut marked_ok, mut marked_err) = (0, 0);
    for item in bucket_samples() {
        let (result, _) = analyze_result(&item);
        let referrer = result.clone().into_referrer();
        for id in [0, 1, 7, 12345, u64::MAX] {
            let what = format!("{}: ({})", s(&item), id);
            // The call for a marker fails exactly if some entry has no bucket.
            let repeater = repeater();
            let marker: Type = parse_quote!(crate::Marker);
            let first_without = referrer
                .iter()
                .find(|(entry, _, _)| referrer.bucket_self(*entry, id).is_none());
            let as_marker = result.repeater_impls(
                &repeater,
                id,
                ImplSelf::Marker(&marker),
                slot_const(&repeater),
            );
            match (&as_marker, first_without) {
                (Err(EncodeError::NeedsSelf(ty)), Some((_, first, _))) => {
                    assert_eq!(s(&**ty), s(first), "{}", what);
                    marked_err += 1;
                }
                (Ok(_), None) => marked_ok += 1,
                _ => panic!(
                    "{}: the call for a marker, and the entries without a bucket",
                    what
                ),
            }
            for ix in 0..referrer.iter().count() {
                let what = format!("{}: {} ({})", s(&item), ix, id);
                let entry = common::entry_id_at(&referrer, ix);
                let bucket = referrer.bucket_self(entry, id);
                let r = result
                    .repeater_impls(
                        &repeater,
                        id,
                        ImplSelf::Bucket { param: &param },
                        slot_const(&repeater),
                    )
                    .unwrap_or_else(|e| panic!("{}: {}", what, e))
                    .nth(ix)
                    .unwrap();
                let own_self = common::entry(&referrer, ix).self_ty.is_some();
                let generics_self = names_self(&common::entry(&referrer, ix).generics);
                assert_eq!(bucket.is_some(), !own_self && !generics_self, "{}", what);
                let for_param = result
                    .repeater_impls(
                        &repeater,
                        id,
                        ImplSelf::Param(&param),
                        slot_const(&repeater),
                    )
                    .unwrap()
                    .nth(ix)
                    .unwrap();
                match bucket {
                    Some(ty) => {
                        bucketed += 1;
                        bucket_name(&ty);
                        assert_eq!(r.self_ty, ty, "{}", what);
                        // The impl for a marker, where there is one.
                        if let Ok(mut impls) = result.repeater_impls(
                            &repeater,
                            id,
                            ImplSelf::Marker(&ty),
                            slot_const(&repeater),
                        ) {
                            assert_eq!(r, impls.nth(ix).unwrap(), "{}", what);
                        }
                        // The impl for the parameter, for the type and without the parameter.
                        let mut expected = for_param;
                        expected.self_ty = ty.clone();
                        expected.generics.params = expected
                            .generics
                            .params
                            .iter()
                            .filter(|p| !matches!(p, GenericParam::Type(t) if t.ident == param))
                            .cloned()
                            .collect();
                        assert_eq!(r, expected, "{}", what);
                        // No `__S`.
                        assert!(!with_where(&r.generics).contains("__S"), "{}", what);
                        // It does not depend on how often it is asked.
                        assert_eq!(referrer.bucket_self(entry, id), Some(ty), "{}", what);
                    }
                    None => {
                        other += 1;
                        generic_self += (!own_self) as usize;
                        assert_eq!(r, for_param, "{}", what);
                    }
                }
            }
        }
    }
    assert!(bucketed > 50 && other > 50, "{} {}", bucketed, other);
    // Both outcomes of the call for a marker were met.
    assert!(
        marked_ok > 5 && marked_err > 5,
        "{} {}",
        marked_ok,
        marked_err
    );
    // The entries of a default whose generics name `Self` while it does not: they were looked for.
    assert!(generic_self > 0);
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
                bucket_name(
                    &referrer
                        .bucket_self(common::entry_id_at(&referrer, ix), id)
                        .unwrap()
                ),
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
            bucket_name(
                &referrer
                    .bucket_self(common::entry_id_at(&referrer, ix), id)
                    .unwrap()
            ),
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
    let first = common::entry_id_at(&referrer, 0);
    let bucket = |id: u64| bucket_name(&referrer.bucket_self(first, id).unwrap());
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
            referrer
                .iter()
                .map(|(ix, _, _)| bucket_name(&referrer.bucket_self(ix, id).unwrap())),
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
    let target = ImplSelf::Bucket { param: &param };
    let copy = target;
    assert_eq!(format!("{:?}", target), format!("{:?}", copy));
    assert!(format!("{:?}", target).contains("Bucket"));
    fn is_clone<T: Clone>() {}
    is_clone::<ImplSelf<'_>>();
}

// ── An entry whose generics name Self ────────────────────────────────────────────────────────

/// The entry of `trait Tr<T: AsRef<Self>, U = Rel<T>>`: the default `Rel<T>` does not name `Self`,
/// so `context_self()` is `None`, but the generics of its context do (like the entry
/// `Wrap<Self>`). With a concrete type that `Self` would be that type, so there is no bucket
/// and `Marker` is refused: `impl<T: AsRef<Self>> Repeater<..> for Marker` bounds `T: AsRef<Marker>`,
/// where the copy has `T: AsRef<Host>`, so an impl of the copy that instantiates the default
/// (`impl Tr<String> for Host`) fails with E0277 (`String: AsRef<Marker>`). The impl is for a
/// parameter, which is the implementor of the trait in the copy.
#[test]
fn the_entry_of_a_default_whose_generics_name_self_is_for_a_parameter() {
    let (result, _) = analyze_result(&parse_quote! {
        trait Tr<T: AsRef<Self>, U = Rel<T>> { fn f(&self, t: T) -> U; fn g(&self) -> Wrap<Self>; }
    });
    let referrer = result.clone().into_referrer();
    let rel = common::types(&referrer)
        .position(|ty| s(ty) == "Rel < T >")
        .unwrap();
    let wrap = common::types(&referrer)
        .position(|ty| s(ty) == "Wrap < Self >")
        .unwrap();
    assert!(common::entry(&referrer, rel).self_ty.is_none());
    assert!(names_self(&common::entry(&referrer, rel).generics));
    let (wrap_id, rel_id) = (
        common::entry_id_at(&referrer, wrap),
        common::entry_id_at(&referrer, rel),
    );
    assert_eq!(referrer.bucket_self(wrap_id, 7), None);
    assert_eq!(referrer.bucket_self(rel_id, 7), None);
    // A bucket falls back to a parameter, for which `Self` of the generics is the parameter.
    let param: Ident = parse_quote!(__S);
    let r = result
        .repeater_impls(
            &parse_quote!(crate::Repeater),
            7,
            ImplSelf::Bucket { param: &param },
            array_const,
        )
        .unwrap()
        .nth(rel)
        .unwrap();
    assert_eq!(s(&r.self_ty), "__S");
    assert_eq!(
        r,
        result
            .repeater_impls(
                &parse_quote!(crate::Repeater),
                7,
                ImplSelf::Param(&param),
                array_const
            )
            .unwrap()
            .nth(rel)
            .unwrap()
    );
    assert_eq!(
        with_where(&r.generics),
        with_where(&parse_quote!(<T: AsRef<Self>, __S: ?::core::marker::Sized>))
    );
    // A marker is refused: the call fails for the first entry that needs `Self`, `Wrap<Self>` or
    // `Rel<T>`, whichever is first.
    let marker: Type = parse_quote!(crate::Marker);
    match result.repeater_impls(
        &parse_quote!(crate::Repeater),
        7,
        ImplSelf::Marker(&marker),
        array_const,
    ) {
        Err(EncodeError::NeedsSelf(ty)) => {
            let first = common::types(&referrer).nth(wrap.min(rel)).unwrap();
            assert_eq!(s(&*ty), s(first));
        }
        Err(other) => panic!("{:?}", other),
        Ok(_) => panic!("no error"),
    }
    // Without `Wrap<Self>`, `Rel<T>` is the only entry that needs `Self`, and the call fails for it.
    let (result, _) = analyze_result(&parse_quote! {
        trait Tr<T: AsRef<Self>, U = Rel<T>> { fn f(&self, t: T) -> U; }
    });
    match result.repeater_impls(
        &parse_quote!(crate::Repeater),
        7,
        ImplSelf::Marker(&marker),
        array_const,
    ) {
        Err(EncodeError::NeedsSelf(ty)) => assert_eq!(s(&*ty), "Rel < T >"),
        Err(other) => panic!("{:?}", other),
        Ok(_) => panic!("no error"),
    }
    // No entry needs `Self` if neither it nor its generics name it.
    let (referrer, _) = analyze(&parse_quote! {
        trait Tr<T, U = Rel<T>> { fn f(&self, t: T) -> U; fn g(&self) -> Wrap<T>; }
    });
    assert!(referrer.iter().count() > 0);
    assert!(referrer
        .iter()
        .all(|(ix, _, _)| referrer.bucket_self(ix, 7).is_some()));
}

/// The call gives `const_fn` the const parameters that the items give it, also for an entry whose
/// impl replaces `Self` by the parameter for the implementor. (The default of a const parameter,
/// which names `Self` here, is not in the context, so the replacement changes no const parameter.)
#[test]
fn repeater_impls_checks_the_const_parameters_as_the_items_see_them() {
    let (result, _) = analyze_result(&parse_quote! {
        trait Tr<const N: usize = { Self::LEN }> {
            const LEN: usize;
            fn f(&self) -> Wrap<N, Self>;
        }
    });
    let param: Ident = parse_quote!(__S);
    let repeater = repeater();
    let seen = std::cell::RefCell::new(Vec::new());
    let impls = result
        .repeater_impls(
            &repeater,
            7,
            ImplSelf::Param(&param),
            |_: &Expr, param: &ConstParam| {
                seen.borrow_mut().push(s(param));
                Ok(parse_quote!([(); 0]))
            },
        )
        .unwrap();
    let in_the_call = seen.borrow().clone();
    for _ in impls {}
    let all = seen.borrow().clone();
    assert_eq!(in_the_call, ["const N : usize"]);
    assert_eq!(all.len(), 2 * in_the_call.len());
    assert_eq!(all[in_the_call.len()..], in_the_call[..]);
}

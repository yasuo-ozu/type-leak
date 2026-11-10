//! Tests for the Repeater definition and its const slots, for the errors
//! of the encoders and of the helpers that encode (`EncodeError`): their text, the conversion to
//! `syn::Error`, and the `std::error::Error` impls, and for the encoding of the generics of an
//! entry: `Projection::of()`, and `Referrer::complete_args()` and `substitute_args()` for a site.
//! The last sections are about `complete_args()` and the generics as declared that the top-level
//! context of a scope keeps for it (where they are kept, how they tell two scopes apart, and how
//! they are written as tokens and read back), `Projection::unused()`, and the defaults of type
//! parameters, analyzed with the parameters before them. The spans of the errors are checked in
//! `tests/crates/spans`, which `tests/diagnostic.rs` runs: they need a feature of `proc-macro2`
//! that the tests of this package cannot have.

use crate::common;

use common::{arguments, panic_message, repeater, s, t, Recovered};

use syn::parse::Parser;
use syn::visit_mut::VisitMut;
use syn::*;
use template_quote::{quote, ToTokens};
use type_leak::*;

/// `repeater_definition()` docs: the Repeater has `T` (which may be unsized), `KEY`, and a
/// defaulted const slot per other primitive type, in the documented order and with the documented names;
/// its type may be unsized. Audit mutants `def_unsized_param` (`T: ?Sized` dropped),
/// `enc_slot_order` and `enc_slot_name`; the visibility and the unsized `Type` are also pinned by
/// the fixture crates compiling.
#[test]
fn the_repeater_definition_has_the_documented_shape() {
    let expected: ItemTrait = parse_quote! {
        pub trait Repeater<
            T: ?::core::marker::Sized,
            const KEY: ::core::primitive::u128,
            const BOOL: ::core::primitive::bool = false,
            const CHAR: ::core::primitive::char = '\0',
            const U8: ::core::primitive::u8 = 0,
            const U16: ::core::primitive::u16 = 0,
            const U32: ::core::primitive::u32 = 0,
            const U64: ::core::primitive::u64 = 0,
            const I8: ::core::primitive::i8 = 0,
            const I16: ::core::primitive::i16 = 0,
            const I32: ::core::primitive::i32 = 0,
            const I64: ::core::primitive::i64 = 0,
            const I128: ::core::primitive::i128 = 0,
            const ISIZE: ::core::primitive::isize = 0
        > {
            type Type: ?::core::marker::Sized;
        }
    };
    assert_eq!(
        repeater_definition(&parse_quote!(pub), &parse_quote!(Repeater)),
        expected
    );
    // The visibility and the name are the caller's.
    let mut private = expected;
    private.vis = Visibility::Inherited;
    private.ident = parse_quote!(Other);
    assert_eq!(
        repeater_definition(&Visibility::Inherited, &parse_quote!(Other)),
        private
    );

    // Each primitive fills its own slot, after the defaults of the slots before it.
    let repeater: Path = parse_quote!(R);
    let defaults = [
        "false", "'\\0'", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0",
    ];
    for (position, ty) in [
        "bool", "char", "u8", "u16", "u32", "u64", "i8", "i16", "i32", "i64", "i128", "isize",
    ]
    .iter()
    .enumerate()
    {
        let ty: Type = parse_str(ty).unwrap();
        let before = defaults[..position].join(", ");
        let value = if position == 0 {
            "true"
        } else if position == 1 {
            "'x'"
        } else {
            "5"
        };
        let before = if before.is_empty() {
            before
        } else {
            format!("{}, ", before)
        };
        let expected: Type = parse_str(&format!(
            "*const dyn R<(), 0, {}{}, Type = ()>",
            before, value
        ))
        .unwrap();
        let value: Expr = parse_str(value).unwrap();
        assert_eq!(
            common::slot(&repeater, &value, &ty).unwrap(),
            expected,
            "{}",
            quote!(#ty)
        );
    }
    // `u128` uses `KEY`, `usize` the length of an array, and the others are not supported (see
    // `the_text_of_every_variant`).
    assert_eq!(
        common::slot(&repeater, &parse_quote!(5), &parse_quote!(u128)).unwrap(),
        parse_quote!(*const dyn R<(), 5, Type = ()>)
    );
    assert_eq!(
        common::slot(&repeater, &parse_quote!(5), &parse_quote!(usize)).unwrap(),
        parse_quote!([(); { 5 }])
    );
}

/// The value that a site gives for a const goes into its slot braced unless it is a literal-like
/// token (`FOO`, a block of one identifier is unwrapped) so that it parses as a const argument. A
/// `usize` const is an array length, with the braces that a value a site gives needs.
#[test]
fn a_const_a_site_gives_is_braced_in_its_slot_when_needed() {
    let to_string = |ty: &Type| quote!(#ty).to_string();
    let repeater = common::repeater();
    // The `i32` slot comes after those of `bool`, `char` and the six other integers.
    let before = quote!(false, '\0', 0, 0, 0, 0, 0, 0);
    let i32_slot = |value: proc_macro2::TokenStream| -> proc_macro2::TokenStream {
        quote!(*const dyn crate::Repeater<(), 0, #before, #value, Type = ()>)
    };

    // A multi-segment path (written with or without braces) must be braced to be parsed as a
    // const argument; a literal or a single identifier is not.
    for (ty, value, expected) in [
        ("i32", quote!(i32::MIN), i32_slot(quote!({ i32::MIN }))),
        ("i32", quote!({ i32::MIN }), i32_slot(quote!({ i32::MIN }))),
        ("i32", quote!({ m::FOO }), i32_slot(quote!({ m::FOO }))),
        ("i32", quote!({ FOO }), i32_slot(quote!(FOO))),
        ("i32", quote!(-3), i32_slot(quote!({ -3 }))),
        (
            "u128",
            quote!({ m::FOO }),
            quote!(*const dyn crate::Repeater<(), { m::FOO }, Type = ()>),
        ),
        ("usize", quote!(3), quote!([(); { 3 }])),
        ("usize", quote!(N), quote!([(); { N }])),
        ("usize", quote!(m::FOO), quote!([(); { m::FOO }])),
        ("usize", quote!({ FOO }), quote!([(); { FOO }])),
    ] {
        let ty: Type = parse_str(ty).unwrap();
        let found = common::slot(&repeater, &parse2(value).unwrap(), &ty).unwrap();
        let expected: Type = parse2(expected).unwrap();
        assert_eq!(
            to_string(&found),
            to_string(&expected),
            "{}",
            to_string(&ty)
        );
    }
}

/// `Projection::key()` puts the id in the high half and the index in the low half, so two entries
/// never share a key, and it is part of the format (the impls and the projections must agree).
#[test]
fn the_repeater_key_has_the_id_above_the_index() {
    use std::collections::HashSet;
    // The `EntryId` with the number `n`, which only `Parse` makes.
    let entry = |n: usize| -> EntryId { parse_quote!(#n) };
    assert_eq!(common::key(0, entry(0)), 0);
    assert_eq!(common::key(0, entry(5)), 5);
    assert_eq!(common::key(1, entry(0)), 1 << 64);
    assert_eq!(common::key(7, entry(3)), (7 << 64) | 3);
    assert_eq!(common::key(u64::MAX, entry(0)), u128::from(u64::MAX) << 64);
    assert_eq!(
        common::key(0, entry(u64::MAX as usize)),
        u128::from(u64::MAX)
    );
    assert_eq!(common::key(u64::MAX, entry(u64::MAX as usize)), u128::MAX);
    let ids = [0, 1, 2, 255, 1 << 32, u64::MAX - 1, u64::MAX];
    let indices = [0, 1, 2, 255, 1 << 31, u32::MAX as usize];
    let keys: HashSet<u128> = ids
        .iter()
        .flat_map(|&id| {
            indices
                .iter()
                .map(move |&index| common::key(id, entry(index)))
        })
        .collect();
    assert_eq!(keys.len(), ids.len() * indices.len());
}

// ── the errors ───────────────────────────────────────────────────────────────────────────────

/// The error of completing the arguments in `args` for the parameters in `generics`, then writing
/// them at a use site of an entry that uses every generic.
fn args_error(generics: &str, args: &str) -> EncodeError {
    let generics: Generics = parse_str(generics).unwrap();
    let args: AngleBracketedGenericArguments = parse_str(args).unwrap();
    let args: Vec<GenericArgument> = args.args.into_iter().collect();
    let completed = match common::complete_generics(&generics, &args) {
        Ok(completed) => completed,
        Err(error) => return error,
    };
    let used = generics.params.iter().map(|param| match param {
        GenericParam::Lifetime(param) => param.lifetime.to_token_stream(),
        GenericParam::Type(param) => param.ident.to_token_stream(),
        GenericParam::Const(param) => param.ident.to_token_stream(),
    });
    let item: ItemTrait = parse_quote!(trait Tr #generics { fn f(&self) -> Wrap<#(#used),*>; });
    let mut leaker = Leaker::new();
    leaker.intern_trait(&item);
    let referrer = leaker.result().finish().unwrap();
    common::repeater_args(
        common::entry(&referrer, 0),
        common::answers(&generics.params, &completed),
        None,
    )
    .unwrap_err()
}

/// The error of writing the path through the Repeater for an entry that needs `Self: 'a`, for
/// `self_ty` a trait object without a lifetime bound.
fn unbounded_self_error(self_ty: &str) -> EncodeError {
    let item: ItemTrait = parse_quote!(
        trait Tr<'a> {
            fn f(&self, x: &'a Wrap<Self>);
        }
    );
    let mut leaker = Leaker::new();
    leaker.intern_trait(&item);
    let referrer = leaker.result().finish().unwrap();
    let self_ty: Type = parse_str(self_ty).unwrap();
    let entry = referrer.iter().next().unwrap();
    common::projection(repeater(), 7)
        .of(&self_ty, entry.2, entry.0)
        .unwrap_err()
}

/// The error of encoding `generics` itself: the const parameter of the first one.
fn params_error(generics: &str) -> EncodeError {
    let generics: Generics = parse_str(generics).unwrap();
    let param = generics
        .params
        .iter()
        .find_map(|param| match param {
            GenericParam::Const(param) => Some(param),
            _ => None,
        })
        .unwrap();
    common::slot(&repeater(), &parse_quote!(N), &param.ty).unwrap_err()
}

/// The error of `repeater_impls()` for a marker, which cannot name `Self`: the entry `Wrap<T,
/// Self>`. The source says where the entry is written.
fn needs_self_error(source: &str) -> EncodeError {
    let item: ItemTrait = parse_str(source).unwrap();
    let mut leaker = Leaker::new();
    leaker.intern_trait(&item);
    let result = leaker.result();
    let marker: Type = parse_quote!(crate::Marker);
    // The call fails for the first entry that names `Self`, which is `Wrap<T, Self>` here.
    let error = result
        .repeater_impls(&Projection::new(
            repeater(),
            7,
            ImplSelf::Marker(marker.clone()),
        ))
        .err()
        .unwrap();
    assert!(matches!(&error, EncodeError::NeedsSelf(ty)
        if template_quote::quote!(#ty).to_string() == "Wrap < T , Self >"));
    error
}

const NEEDS_SELF_SOURCE: &str = "trait Tr<'a, T> {\n    fn f(&self) -> Wrap<T, Self>;\n}";

/// Every variant.
fn errors() -> Vec<EncodeError> {
    vec![
        params_error("<const N: MyIdx>"),
        args_error("<T>", "<u8, u16>"),
        args_error("<'a, T>", "<u8>"),
        unbounded_self_error("dyn Tr"),
        needs_self_error(NEEDS_SELF_SOURCE),
        // The default `T::Item` of a parameter whose argument is not an identifier, with no bound.
        args_error("<T, U = T::Item>", "<::std::vec::IntoIter<u8>>"),
    ]
}

/// `complete_args()` matches the arguments to the parameters by kind: an argument of the other
/// kind is no argument of its kind, so the parameter is missing one, and what is left over does not
/// fit (the docs of `complete_args()`).
#[test]
fn complete_args_matches_the_arguments_by_kind() {
    let outcome = |params: &str, given: &[&str]| -> String {
        let generics: Generics = parse_str(&format!("<{}>", params)).unwrap();
        match common::complete_generics(&generics, &arguments(given)) {
            Ok(completed) => format!("ok {}", strings(&completed).join(" ")),
            Err(EncodeError::MissingArgument(param)) => format!("missing {}", s(&*param)),
            Err(EncodeError::UnexpectedArgument(arg)) => format!("unexpected {}", s(&*arg)),
            Err(other) => panic!("{}", other),
        }
    };
    for (params, given, expected) in [
        // A type for a lifetime parameter, a lifetime for a type parameter.
        ("'a", &["[u8]"][..], "missing 'a"),
        ("T", &["'a"], "missing T"),
        // The type parameter has a default: the lifetime is left over.
        ("T = u8", &["'a"], "unexpected 'a"),
        // A const for a type parameter, a type that is not a name for a const parameter.
        ("T", &["{ 3 }"], "unexpected { 3 }"),
        ("const N: usize", &["Vec<u8>"], "unexpected Vec < u8 >"),
        // Too many.
        ("T", &["u8", "u8"], "unexpected u8"),
        ("'a", &["'a", "'b"], "unexpected 'b"),
        // The kinds are matched apart, so their order does not matter.
        ("'a, T", &["u8", "'a"], "ok 'a u8"),
        ("T, 'a", &["'a", "u8"], "ok u8 'a"),
        // With several of each, the lifetimes go to the lifetime parameters in order, the types
        // after them, and the omitted default is written with the types.
        (
            "'a, 'b, T, U = Vec<T>",
            &["u8", "'static", "'x"],
            "ok 'static 'x u8 Vec<u8>",
        ),
        (
            "'a, 'b, T, U",
            &["'static", "u8", "'x", "u16"],
            "ok 'static 'x u8 u16",
        ),
        // A lifetime missing, and one too many, with the types in between.
        ("'a, 'b, T", &["u8", "'static"], "missing 'b"),
        ("'a, 'b, T", &["'a", "u8", "'b", "'c"], "unexpected 'c"),
    ] {
        assert_eq!(outcome(params, given), expected, "<{}> {:?}", params, given);
    }
}

#[test]
fn the_text_of_every_variant() {
    let all = errors();
    let texts: Vec<String> = all.iter().map(|err| err.to_string()).collect();
    assert_eq!(
        texts,
        [
            "unsupported type of const parameter: 'MyIdx'",
            "unexpected generic argument: 'u16'",
            "missing generic argument for ''a'",
            "the trait object 'dyn Tr' has no lifetime bound, which defaults to different \
             lifetimes in the encoding of a type with a lifetime bound; write it (e.g. `+ \
             'static`)",
            "the type 'Wrap < T , Self >' needs `Self` (it names it, or its context does), so its \
             Repeater impl cannot be for a marker type; use `ImplSelf::Param`",
            "the associated type '< :: std :: vec :: IntoIter < u8 > > :: Item' needs its trait: \
             rustc resolves `<Type>::Item` only for a type parameter; write `<Type as Trait>::Item`",
        ]
    );
    assert!(matches!(all[0], EncodeError::UnsupportedConstType(_)));
    assert!(matches!(all[1], EncodeError::UnexpectedArgument(_)));
    assert!(matches!(all[2], EncodeError::MissingArgument(_)));
    assert!(matches!(all[3], EncodeError::UnboundedTraitObject(_)));
    assert!(matches!(all[4], EncodeError::NeedsSelf(_)));
    assert!(matches!(all[5], EncodeError::AmbiguousAssociatedType(_)));
}

#[test]
fn the_text_of_an_unbounded_trait_object_is_that_of_its_type() {
    // The type is shown as it is written: parenthesized, or with other bounds.
    for (self_ty, shown) in [
        ("dyn Tr", "dyn Tr"),
        ("(dyn Tr)", "(dyn Tr)"),
        ("dyn Tr + Send", "dyn Tr + Send"),
    ] {
        let err = unbounded_self_error(self_ty);
        assert_eq!(
            err.to_string(),
            format!(
                "the trait object '{}' has no lifetime bound, which defaults to different \
                 lifetimes in the encoding of a type with a lifetime bound; write it (e.g. `+ \
                 'static`)",
                shown
            )
        );
    }
}

#[test]
fn an_encode_error_is_a_syn_error_with_its_text() {
    for err in errors() {
        let text = err.to_string();
        let converted: syn::Error = err.into();
        assert_eq!(converted.to_string(), text);
        // One error, and it renders as `compile_error!`.
        assert_eq!(converted.clone().into_iter().count(), 1);
        let tokens = converted.to_compile_error().to_string();
        assert!(tokens.contains("compile_error"), "{}", tokens);
        assert!(tokens.contains(&format!("{:?}", text)), "{}", tokens);
    }
}

fn is_error<E: std::error::Error>(_: &E) {}

#[test]
fn both_errors_are_std_errors() {
    // (`ErrorItem` is checked in `tests/errors/error_items.rs`.)
    // `EncodeError`: it can be boxed, `?`-converted into a `Box<dyn Error>`, and has no source.
    for err in errors() {
        is_error(&err);
        let text = err.to_string();
        let debug = format!("{:?}", err);
        let boxed: Box<dyn std::error::Error> = Box::new(err.clone());
        assert_eq!(boxed.to_string(), text);
        assert!(boxed.source().is_none());
        // The derived `Debug` names the variant.
        assert!(debug.starts_with(&text_variant(&err)), "{}", debug);
        let result: std::result::Result<(), Box<dyn std::error::Error>> = (|| Err(err.clone())?)();
        assert_eq!(result.unwrap_err().to_string(), text);
    }
}

/// The name of the variant of `err`.
fn text_variant(err: &EncodeError) -> String {
    match err {
        EncodeError::UnsupportedConstType(_) => "UnsupportedConstType(",
        EncodeError::UnexpectedArgument(_) => "UnexpectedArgument(",
        EncodeError::MissingArgument(_) => "MissingArgument(",
        EncodeError::UnboundedTraitObject(_) => "UnboundedTraitObject(",
        EncodeError::NeedsSelf(_) => "NeedsSelf(",
        EncodeError::AmbiguousAssociatedType(_) => "AmbiguousAssociatedType(",
        _ => unreachable!(),
    }
    .to_string()
}

// ── substitute_args() ───────────────────────────────────────────────────

fn strings(args: &[GenericArgument]) -> Vec<String> {
    args.iter()
        .map(|arg| quote!(#arg).to_string().replace(' ', ""))
        .collect()
}

/// What `f` is asked, in order, for `args`; it answers nothing.
fn asked(args: &[&str]) -> Vec<String> {
    let mut args = arguments(args);
    let mut asked = Vec::new();
    substitute_args(args.iter_mut(), |name: &Ident| {
        asked.push(name.to_string());
        None
    })
    .unwrap();
    asked
}

#[test]
fn substitute_args_asks_for_each_name_once_in_order() {
    // A type (`u8` too), the first segment of a projection, an expression, a lifetime without its
    // apostrophe, and `Self` in a type and in a macro.
    assert_eq!(
        asked(&[
            "Vec<T>",
            "&'a (U, T)",
            "V::A<W>",
            "[u8; N]",
            "Rel<{ M }>",
            "Self::B",
            "m!(Self, X)",
            "'b",
            "'a",
        ]),
        ["T", "a", "U", "V", "W", "u8", "N", "M", "Self", "b"]
    );
    // Not `'static`, `'_`, a name behind a leading colon, with arguments or a qualified self, nor
    // a longer path; `Self` alone in an expression is a value.
    assert_eq!(
        asked(&[
            "&'static T",
            "Rel<'_>",
            "::p::X",
            "p::X",
            "<X as Tr>::A",
            "Wrap<[u8; Self]>",
        ]),
        ["T", "p", "X", "u8"]
    );
    // The first segment of a path in an expression is asked, once, as the name of a type.
    assert_eq!(
        asked(&["[u8; a::B]", "{ a::f(T::LEN, ::p::q::r()) }", "&'a u8"]),
        ["u8", "a", "T"]
    );
    // The tokens of a macro are asked for `Self` only if they name it.
    assert_eq!(asked(&["m!(T, 'a)"]), Vec::<String>::new());
    assert_eq!(asked(&["m!((Self))"]), ["Self"]);
}

/// The binder `'__tl_s<n>` that the library writes for an elided lifetime is its own name: `f` is
/// not asked about it, in `substitute_args()` and in `Substitute`. A lifetime of the copy that is
/// named like it is asked.
#[test]
fn the_fresh_binder_of_a_substitution_is_not_asked_for() {
    // `U` adds a lifetime to the inputs of the function pointer, which gets the binder.
    let mut args = arguments(&["fn(&U) -> &u8"]);
    let mut asked = Vec::new();
    substitute_args(args.iter_mut(), |name: &Ident| {
        asked.push(name.to_string());
        (name == "U").then(|| parse_quote!(&'static str))
    })
    .unwrap();
    assert_eq!(
        args,
        [parse_quote!(
            for<'__tl_s0> fn(&'__tl_s0 &'static str) -> &'__tl_s0 u8
        )]
    );
    assert_eq!(asked, ["U", "u8"]);

    // A catch-all answer, which would be a misfit for a lifetime, is not applied to the binder.
    let mut args = arguments(&["dyn Fn(&U) -> &u8"]);
    substitute_args(args.iter_mut(), |name: &Ident| {
        (name != "u8").then(|| parse_quote!(&'static str))
    })
    .unwrap();
    assert_eq!(
        args,
        [parse_quote!(
            dyn for<'__tl_s0> Fn(&'__tl_s0 &'static str) -> &'__tl_s0 u8
        )]
    );

    // The same in a node. A lifetime of the copy that is named `'__tl_s0` is kept, and the binder
    // takes the next name.
    let mut item: ImplItemFn = parse_quote! {
        fn f(&self, g: Wrap<'__tl_s0, fn(&U) -> &u8>, h: for<'x> fn(&'x fn(&U) -> &u8)) {}
    };
    at("<U>", &["&'static str"]).visit_impl_item_fn_mut(&mut item);
    let expected: ImplItemFn = parse_quote! {
        fn f(
            &self,
            g: Wrap<'__tl_s0, for<'__tl_s1> fn(&'__tl_s1 &'static str) -> &'__tl_s1 u8>,
            h: for<'x> fn(&'x for<'__tl_s0> fn(&'__tl_s0 &'static str) -> &'__tl_s0 u8)
        ) {}
    };
    assert_eq!(s(&item), s(&expected));
}

#[test]
fn substitute_args_replaces_each_name_by_the_kind_that_fits_its_place() {
    let site = |name: &Ident| -> Option<GenericArgument> {
        match name.to_string().as_str() {
            "T" => Some(parse_quote!(Vec<u8>)),
            "N" => Some(parse_quote!(3)),
            "M" => Some(parse_quote!(K)),
            "a" => Some(parse_quote!('static)),
            "Self" => Some(parse_quote!(Host)),
            _ => None,
        }
    };
    for (written, expected) in [
        // A whole argument takes a type or a const.
        ("T", "Vec<u8>"),
        ("N", "3"),
        ("Rel<T, N>", "Rel<Vec<u8>, 3>"),
        // A const in an expression is braced once, and a type that is one identifier is one.
        ("[u8; N]", "[u8; { 3 }]"),
        ("[u8; { N }]", "[u8; { 3 }]"),
        ("[u8; M]", "[u8; { K }]"),
        ("Rel<{ M }>", "Rel<{ K }>"),
        // A lifetime, `Self`, a projection.
        ("&'a T", "&'static Vec<u8>"),
        ("&'b T", "&'b Vec<u8>"),
        (
            "(Self, Self::A, <Self as Tr>::B)",
            "(Host, <Host>::A, <Host as Tr>::B)",
        ),
        ("<T as Tr>::Item<N>", "<Vec<u8> as Tr>::Item<3>"),
        // `Self` in a macro; `Self::A` in an expression; `Self` alone in one is a value.
        ("m!(Self, T)", "m!(Host, T)"),
        ("[u8; Self::LEN]", "[u8; <Host>::LEN]"),
        ("[u8; Self]", "[u8; Self]"),
        // One pass: a value is not substituted again.
        ("Rel<M, K>", "Rel<K, K>"),
    ] {
        let mut args = arguments(&[written]);
        substitute_args(args.iter_mut(), site).unwrap();
        assert_eq!(
            strings(&args),
            strings(&arguments(&[expected])),
            "{}",
            written
        );
    }
}

#[test]
fn substitute_args_refuses_a_value_that_does_not_fit_and_changes_nothing() {
    // (the arguments, the name, its value)
    for (written, name, value) in [
        // A whole argument is a type or a const, not a lifetime or a binding.
        (&["T"][..], "T", "'x"),
        (&["Rel<T>"][..], "T", "'x"),
        (&["Rel<T>"][..], "T", "Item = u8"),
        // Part of a type is a type.
        (&["&T"][..], "T", "3"),
        (&["(T,)"][..], "T", "'x"),
        (&["fn(T)"][..], "T", "3"),
        (&["T::A"][..], "T", "3"),
        (&["Rel<Item = T>"][..], "T", "3"),
        // `Self` in a macro is a type.
        (&["m!(Self)"][..], "Self", "3"),
        // An expression takes a const, or a type that is one identifier.
        (&["[u8; N]"][..], "N", "Vec<u8>"),
        (&["[u8; { N }]"][..], "N", "'x"),
        (&["[u8; N]"][..], "N", "::p::M"),
        // A lifetime takes a lifetime.
        (&["&'a u8"][..], "a", "u8"),
        (&["Rel<'a>"][..], "a", "3"),
        // The first of several arguments is replaced, and then the call fails: no change.
        (&["Rel<U>", "&'a T", "T"][..], "a", "u8"),
    ] {
        let mut args = arguments(written);
        let before = args.clone();
        let value: GenericArgument = parse_str(value).unwrap();
        let err = substitute_args(args.iter_mut(), |asked: &Ident| {
            if asked == name {
                Some(value.clone())
            } else if asked == "U" {
                Some(parse_quote!(u8))
            } else {
                None
            }
        })
        .unwrap_err();
        assert!(
            matches!(&err, EncodeError::UnexpectedArgument(found) if **found == value),
            "{:?} with {}: {:?}",
            written,
            name,
            err
        );
        assert_eq!(args, before, "{:?} with {}", written, name);
    }
}

#[test]
fn substitute_args_changes_nothing_for_an_empty_list_or_no_answer() {
    substitute_args(std::iter::empty(), |_: &Ident| -> Option<GenericArgument> {
        unreachable!()
    })
    .unwrap();
    let mut args = arguments(&["Rel<T, 'a, N>", "&'a Self"]);
    let before = args.clone();
    substitute_args(args.iter_mut(), |_: &Ident| None).unwrap();
    assert_eq!(args, before);
}

// ── Substitute ───────────────────────────────────────────────────────────────────────────────

/// The `Substitute` for an impl of the trait with `generics` that gives it `given`.
fn at(generics: &str, given: &[&str]) -> Substitute {
    // The generics may have a where clause, which `Generics` does not parse.
    let item: ItemTrait = parse_str(&format!("trait Tr {} {{}}", generics)).unwrap();
    common::substitute_for(&item.generics, &arguments(given)).unwrap()
}

#[test]
fn substitute_writes_the_lifetimes_outside_the_types_too() {
    // The receiver, a lifetime bound of a generic and of a predicate, an outlives predicate, a
    // lifetime argument of a bound, the types around them.
    let mut item: ImplItemFn = parse_quote! {
        fn keep<'b: 'a, V: 'a + Tr<'a, K>>(&'a mut self, key: &'a K, v: V) -> &'a K
        where
            Self: 'a,
            K: 'a,
            'a: 'b,
        {
            todo!()
        }
    };
    let mut visitor = at("<'a, K>", &["'x", "u8"]);
    visitor.visit_impl_item_fn_mut(&mut item);
    let expected: ImplItemFn = parse_quote! {
        fn keep<'b: 'x, V: 'x + Tr<'x, u8>>(&'x mut self, key: &'x u8, v: V) -> &'x u8
        where
            Self: 'x,
            u8: 'x,
            'x: 'b,
        {
            todo!()
        }
    };
    assert_eq!(s(&item), s(&expected));

    // The other forms of the receiver, and a signature without a body, as the trait has it.
    for (written, expected) in [
        ("fn f(&'a self)", "fn f(&'x self)"),
        ("fn f(self: &'a Self)", "fn f(self: &'x Self)"),
        (
            "fn f(self: Box<Wrap<'a, K>>)",
            "fn f(self: Box<Wrap<'x, u8>>)",
        ),
        (
            "fn f<T: Tr<'a>>() -> impl Tr<'a>",
            "fn f<T: Tr<'x>>() -> impl Tr<'x>",
        ),
        (
            "fn f() where for<'c> &'c K: 'a",
            "fn f() where for<'c> &'c u8: 'x",
        ),
    ] {
        let mut sig: Signature = parse_str(written).unwrap();
        at("<'a, K>", &["'x", "u8"]).visit_signature_mut(&mut sig);
        let expected: Signature = parse_str(expected).unwrap();
        assert_eq!(s(&sig), s(&expected), "{}", written);
    }
    // `'static` and `'_` are no names: they stay.
    let mut sig: Signature = parse_quote!(fn f(&'static self, x: &'_ u8) where Self: 'static);
    let before = sig.clone();
    at("<'a>", &["'x"]).visit_signature_mut(&mut sig);
    assert_eq!(s(&sig), s(&before));
}

/// The audit's case: `trait Store<'a, K: 'a, D = Page<K>>` at `impl Store<'static, u8> for ..`
/// needs `'a` in `keep(&'a self) -> &'a K`, outside the types a type-only pass reaches.
#[test]
fn substitute_writes_the_arguments_of_a_trait_with_a_lifetime_parameter() {
    let tr: ItemTrait = parse_quote! {
        trait Store<'a, K: 'a, D = Page<K>> {
            fn get(&self, key: &'a K) -> D;
            fn keep(&'a self) -> &'a K;
        }
    };
    let given = arguments(&["'static", "u8"]);
    let mut items: Vec<TraitItem> = tr.items.clone();
    let mut visitor = common::substitute_for(&tr.generics, &given).unwrap();
    for item in &mut items {
        visitor.visit_trait_item_mut(item);
    }
    let expected: Vec<TraitItem> = vec![
        parse_quote!(
            fn get(&self, key: &'static u8) -> Page<u8>;
        ),
        parse_quote!(
            fn keep(&'static self) -> &'static u8;
        ),
    ];
    assert_eq!(items, expected);
}

/// A whole impl or trait is visited by its items: the header declares the generics itself (the
/// generics, the trait path, the self type, the supertraits), and an item in a body has a scope of
/// its own (E0401).
#[test]
fn substitute_leaves_the_header_and_the_items_in_a_body_alone() {
    let imp: ItemImpl = parse_quote! {
        impl<T: Clone> Tr<Vec<T>> for Foo<T> where T: Copy {
            type Out = Option<T>;
            fn f(&self, t: T) -> Vec<T> {
                fn inner(x: T) -> T { x }
                struct S(T);
                let y: T = t;
                todo!()
            }
        }
    };
    let expected: ItemImpl = parse_quote! {
        impl<T: Clone> Tr<Vec<T>> for Foo<T> where T: Copy {
            type Out = Option<Vec<T>>;
            fn f(&self, t: Vec<T>) -> Vec<Vec<T>> {
                fn inner(x: T) -> T { x }
                struct S(T);
                let y: Vec<T> = t;
                todo!()
            }
        }
    };
    // `T` is given `Vec<T>`: the type of the impl itself.
    let mut whole = imp.clone();
    at("<T>", &["Vec<T>"]).visit_item_impl_mut(&mut whole);
    assert_eq!(s(&whole), s(&expected));
    // The items one by one give the same.
    let mut items = imp.clone();
    let mut visitor = at("<T>", &["Vec<T>"]);
    for item in &mut items.items {
        visitor.visit_impl_item_mut(item);
    }
    assert_eq!(s(&items), s(&expected));
    // A trait: the supertraits and the generics stay too.
    let mut tr: ItemTrait = parse_quote! {
        trait Tr<T>: Into<T> where T: Copy {
            fn f(&self, t: T) -> Vec<T>;
        }
    };
    at("<T>", &["u8"]).visit_item_trait_mut(&mut tr);
    let expected: ItemTrait = parse_quote! {
        trait Tr<T>: Into<T> where T: Copy {
            fn f(&self, t: u8) -> Vec<u8>;
        }
    };
    assert_eq!(s(&tr), s(&expected));
}

#[test]
fn substitute_keeps_the_generics_a_method_declares_if_the_site_gives_no_answer() {
    let mut item: ImplItemFn = parse_quote! {
        fn get<'b, V: Into<K>>(&'b self, v: V, key: &'a K) -> &'b K {
            todo!()
        }
    };
    at("<'a, K>", &["'x", "u8"]).visit_impl_item_fn_mut(&mut item);
    let expected: ImplItemFn = parse_quote! {
        fn get<'b, V: Into<u8>>(&'b self, v: V, key: &'x u8) -> &'b u8 {
            todo!()
        }
    };
    assert_eq!(s(&item), s(&expected));
}

#[test]
fn substitute_writes_a_type_as_substitute_args_does() {
    // The visited type is one unit: the braces of a const, the parentheses of an object behind a
    // pointer, and what is written is not substituted again.
    let mut sig: Signature = parse_quote!(fn f(a: [u8; N], b: &T, c: Wrap<{ N }, T>) -> Wrap<U, T>);
    at("<T, const N: usize, U>", &["dyn A + B", "M", "T"]).visit_signature_mut(&mut sig);
    let expected: Signature = parse_quote!(
        fn f(a: [u8; { M }], b: &(dyn A + B), c: Wrap<{ M }, dyn A + B>) -> Wrap<T, dyn A + B>
    );
    assert_eq!(s(&sig), s(&expected));
}

#[test]
fn substitute_writes_a_const_outside_a_type() {
    // A bare identifier in the arguments of a bound is parsed as a type; an expression of an item.
    let mut item: ImplItemFn = parse_quote! {
        fn f<const M: usize>(&self) where T: Tr<N, { N }> {
            let _ = N;
        }
    };
    at("<T, const N: usize>", &["T", "3"]).visit_impl_item_fn_mut(&mut item);
    let expected: ImplItemFn = parse_quote! {
        fn f<const M: usize>(&self) where T: Tr<3, { 3 }> {
            let _ = 3;
        }
    };
    assert_eq!(s(&item), s(&expected));
    // A const for a name that is a type alone does not fit: it stays as written, for rustc to
    // report.
    let mut sig: Signature = parse_quote!(fn f(x: N));
    let before = sig.clone();
    at("<const N: usize>", &["3"]).visit_signature_mut(&mut sig);
    assert_eq!(s(&sig), s(&before));
}

#[test]
fn substitute_leaves_self_alone_and_parenthesizes_the_bounded_type_of_a_predicate() {
    // `Self` is no parameter: it stays in a predicate and in the tokens of a macro.
    let mut item: ImplItemFn = parse_quote! {
        fn f(&self) where Self: 'a {
            m!(Self, T);
        }
    };
    at("<'a, T>", &["'static", "u8"]).visit_impl_item_fn_mut(&mut item);
    let expected: ImplItemFn = parse_quote! {
        fn f(&self) where Self: 'static {
            m!(Self, T);
        }
    };
    assert_eq!(s(&item), s(&expected));
    // What replaces the bounded type of a predicate may need parentheses there.
    let mut sig: Signature = parse_quote!(fn f() where T: Clone);
    at("<T>", &["&fn(u8)"]).visit_signature_mut(&mut sig);
    let expected: Signature = parse_quote!(fn f() where (&fn(u8)): Clone);
    assert_eq!(s(&sig), s(&expected));
}

#[test]
fn substitute_names_the_elided_lifetime_of_a_signature_inside_a_node() {
    // A function pointer in a type, `Fn(..)` in a type, and `Fn(..)` as a bound outside a type:
    // the value adds a lifetime to the inputs, and the output keeps the elided one by its name.
    let mut item: ImplItemFn = parse_quote! {
        fn f<F: Fn(&U) -> &u8>(&self, g: fn(&U) -> &u8, h: Box<dyn Fn(&U) -> &u8>, f: F)
        where
            F: Fn(&U) -> &u8,
        {
            todo!()
        }
    };
    at("<U>", &["&'static str"]).visit_impl_item_fn_mut(&mut item);
    let expected: ImplItemFn = parse_quote! {
        fn f<F: for<'__tl_s0> Fn(&'__tl_s0 &'static str) -> &'__tl_s0 u8>(
            &self,
            g: for<'__tl_s0> fn(&'__tl_s0 &'static str) -> &'__tl_s0 u8,
            h: Box<dyn for<'__tl_s0> Fn(&'__tl_s0 &'static str) -> &'__tl_s0 u8>,
            f: F
        )
        where
            F: for<'__tl_s0> Fn(&'__tl_s0 &'static str) -> &'__tl_s0 u8,
        {
            todo!()
        }
    };
    assert_eq!(s(&item), s(&expected));
}

/// An associated type `T::Item` of an argument that is not a name is written through the one
/// trait bound of `T`, inline or in the where clause, else `<arg>::Item` and rustc reports it
/// (E0223). The arguments of the bound are written too.
#[test]
fn substitute_writes_an_associated_type_through_the_bound_of_the_parameter() {
    // (case, the generics of the trait, the arguments given, `T::Item` as written, the copy)
    for (case, generics, given, expected) in [
        (
            "an inline bound",
            "<T: Iterator>",
            "::std::vec::IntoIter<u8>",
            "<::std::vec::IntoIter<u8> as Iterator>::Item",
        ),
        (
            "an inline bound with arguments, the associated type it fixes dropped",
            "<U, T: Tr<U, Item = u8>>",
            "::std::vec::IntoIter<u8>",
            "<::std::vec::IntoIter<u8> as Tr<u16>>::Item",
        ),
        (
            "a bound in the where clause",
            "<T> where T: Iterator",
            "::std::vec::IntoIter<u8>",
            "<::std::vec::IntoIter<u8> as Iterator>::Item",
        ),
        (
            "a bound in the where clause next to a lifetime bound and `?Sized`",
            "<T: ?Sized + 'static> where T: Iterator",
            "::std::vec::IntoIter<u8>",
            "<::std::vec::IntoIter<u8> as Iterator>::Item",
        ),
        (
            "no bound",
            "<T>",
            "::std::vec::IntoIter<u8>",
            "<::std::vec::IntoIter<u8>>::Item",
        ),
        (
            "two bounds",
            "<T: Iterator + Clone>",
            "::std::vec::IntoIter<u8>",
            "<::std::vec::IntoIter<u8>>::Item",
        ),
        (
            "an argument that is a name",
            "<T: Iterator>",
            "I",
            "<I>::Item",
        ),
    ] {
        let mut sig: Signature = parse_quote!(fn f(a: T::Item, b: <T as Tr>::A));
        let given: Vec<&str> = if generics.starts_with("<U,") {
            vec!["u16", given]
        } else {
            vec![given]
        };
        at(generics, &given).visit_signature_mut(&mut sig);
        let expected: Type = parse_str(expected).unwrap();
        let item = match &sig.inputs[0] {
            FnArg::Typed(arg) => (*arg.ty).clone(),
            other => panic!("{}", s(other)),
        };
        assert_eq!(s(&item), s(&expected), "{}", case);
    }
}

// ── Referrer::complete_args() ────────────────────────────────────────────────────────────────

/// The signature `written`, copied at the site that gives `given` to the trait with `generics`.
fn at_params(generics: &str, given: &[&str], written: &str) -> Signature {
    let mut sig: Signature = parse_str(written).unwrap();
    at(generics, given).visit_signature_mut(&mut sig);
    sig
}

#[test]
fn complete_args_answers_each_parameter_by_its_name_and_kind() {
    // (case, the generics of the trait, the arguments given, the signature written, the copy)
    for (case, generics, given, written, expected) in [
        (
            "a lifetime and a type of one name",
            "<'T, T>",
            &["'x", "u8"][..],
            "fn f(&'T self, t: T) -> &'T T",
            "fn f(&'x self, t: u8) -> &'x u8",
        ),
        (
            "the type comes first",
            "<T, 'T>",
            &["'x", "u8"][..],
            "fn f(&'T self, t: T) -> &'T T",
            "fn f(&'x self, t: u8) -> &'x u8",
        ),
        (
            "declaration order, not the sorted one",
            "<B, A>",
            &["u8", "u16"][..],
            "fn f(a: A, b: B)",
            "fn f(a: u16, b: u8)",
        ),
        (
            "a lifetime and a const of one name",
            "<'N, const N: usize>",
            &["'x", "3"][..],
            "fn f(&'N self, a: [u8; N]) -> &'N u8",
            "fn f(&'x self, a: [u8; { 3 }]) -> &'x u8",
        ),
        (
            "a lifetime without a type of its name",
            "<'a, T>",
            &["'x", "u8"][..],
            "fn f(&'a self, a: a, t: T)",
            "fn f(&'x self, a: a, t: u8)",
        ),
        (
            "a type without a lifetime of its name",
            "<'a, T>",
            &["'x", "u8"][..],
            "fn f(&'T self, t: T, b: &'a T)",
            "fn f(&'T self, t: u8, b: &'x u8)",
        ),
        (
            "a default that uses an earlier parameter",
            "<'a, T, U = Vec<T>>",
            &["'x", "u8"][..],
            "fn f(&'a self, t: T, u: U) -> Option<U>",
            "fn f(&'x self, t: u8, u: Vec<u8>) -> Option<Vec<u8>>",
        ),
        (
            "a default that names a lifetime and a const",
            "<'a, const N: usize, T = &'a [u8; N]>",
            &["'x", "2"][..],
            "fn f(&self, t: T)",
            "fn f(&self, t: &'x [u8; { 2 }])",
        ),
        (
            "the default is given",
            "<T, U = Vec<T>>",
            &["u8", "String"][..],
            "fn f(t: T, u: U)",
            "fn f(t: u8, u: String)",
        ),
        (
            "a binding is skipped",
            "<T, U = Vec<T>>",
            &["u8", "Item = u16"][..],
            "fn f(t: T, u: U)",
            "fn f(t: u8, u: Vec<u8>)",
        ),
        (
            "a name that is not a parameter",
            "<T>",
            &["u8"][..],
            "fn f<'b, V: 'b>(&'b self, t: T, v: V) -> Self where V: Clone",
            "fn f<'b, V: 'b>(&'b self, t: u8, v: V) -> Self where V: Clone",
        ),
        (
            "no parameters",
            "<>",
            &[][..],
            "fn f<'b, V>(&'b self, v: V) -> Self",
            "fn f<'b, V>(&'b self, v: V) -> Self",
        ),
        (
            "an associated type of an earlier parameter in a default, through its where clause",
            "<T, U = Option<T::Item>> where T: Iterator",
            &["::std::vec::IntoIter<u8>"][..],
            "fn f(u: U)",
            "fn f(u: Option<<::std::vec::IntoIter<u8> as Iterator>::Item>)",
        ),
    ] {
        let sig = at_params(generics, given, written);
        let expected: Signature = parse_str(expected).unwrap();
        assert_eq!(s(&sig), s(&expected), "{}", case);
    }
}

#[test]
fn complete_args_fails_for_each_error() {
    // (case, the generics of the trait, the arguments given, the error)
    for (case, generics, given, expected) in [
        ("a parameter with no argument", "<T>", &[][..], "missing T"),
        (
            "a lifetime with no argument",
            "<'a, T>",
            &["u8"][..],
            "missing 'a",
        ),
        ("a type for a lifetime", "<'a>", &["u8"][..], "missing 'a"),
        ("a lifetime for a type", "<T>", &["'a"][..], "missing T"),
        (
            "too many arguments",
            "<T>",
            &["u8", "u16"][..],
            "unexpected u16",
        ),
        (
            "a lifetime too many",
            "<T>",
            &["u8", "'a"][..],
            "unexpected 'a",
        ),
        (
            "a type for a const",
            "<const N: usize>",
            &["Vec<u8>"][..],
            "unexpected Vec < u8 >",
        ),
        (
            "an associated type of an argument that is not a name, no bound",
            "<T, U = Option<T::Item>>",
            &["IntoIter<u8>"][..],
            "ambiguous < IntoIter < u8 > > :: Item",
        ),
        (
            "an associated type of an argument that is not a name, two bounds",
            "<T: Iterator + Clone, U = Option<T::Item>>",
            &["IntoIter<u8>"][..],
            "ambiguous < IntoIter < u8 > > :: Item",
        ),
    ] {
        let generics: Generics = parse_str(generics).unwrap();
        let found = match common::substitute_for(&generics, &arguments(given)) {
            Ok(_) => panic!("{}: no error", case),
            Err(EncodeError::MissingArgument(param)) => format!("missing {}", s(&*param)),
            Err(EncodeError::UnexpectedArgument(arg)) => format!("unexpected {}", s(&*arg)),
            Err(EncodeError::AmbiguousAssociatedType(ty)) => format!("ambiguous {}", s(&*ty)),
            Err(other) => panic!("{}: {}", case, other),
        };
        assert_eq!(found, expected, "{}", case);
    }
}

#[test]
#[should_panic(expected = "is not one of this Referrer")]
fn complete_args_panics_for_a_context_that_is_not_the_referrers() {
    let mut leaker = Leaker::new();
    let context = leaker.intern_struct(&parse_quote!(
        struct S<T>(Rel<T>);
    ));
    let referrer = Leaker::new().result().finish().unwrap();
    let repeater = repeater();
    let _ = referrer.complete_args(
        context,
        &[],
        &Projection::new(repeater.clone(), 0, ImplSelf::Param(parse_quote!(__S))),
    );
}

#[test]
fn the_substitute_is_clone_and_debug() {
    fn is_clone_and_debug<T: Clone + std::fmt::Debug>() {}
    is_clone_and_debug::<Substitute>();
}

// ── Projection::of() ─────────────────────────────────────────────────────────────────────────

#[test]
fn projection_of_writes_the_generics_of_the_entry_by_their_names() {
    let tr: ItemTrait = parse_quote! {
        trait Tr<'a, T: ?Sized + 'a, const N: usize, const B: bool> {
            fn f(&self, x: &'a Wrap<T, N, B>);
        }
    };
    let mut leaker = Leaker::new();
    leaker.intern_trait(&tr);
    let referrer = leaker.result().finish().unwrap();
    let (ix, _, context) = referrer.iter().next().unwrap();
    let repeater = repeater();
    let key = common::key(7, ix);
    let host: Type = parse_quote!(Host);
    // Lifetimes, an unsized type, the consts (by name) in the slots, and the bounds of the
    // generics. A const that has no slot is the error of the call (`the_text_of_every_variant`).
    let projection = common::projection(repeater, 7)
        .of(&host, context, ix)
        .unwrap();
    let expected: Type = parse_quote!(<Host as crate::Repeater<(
        &'a (),
        *const T,
        *const dyn crate::Repeater<(), 0, B, Type = ()>,
        [(); N],
        &'a T
    ), #key>>::Type);
    assert_eq!(projection, expected);
}

// ── the encoding of the generics of an entry ─────────────────────────────────────────────────────
//
// The Leaker analyzes `trait Tr<generics> { fn f(&self) -> All<..>; }`, whose one entry names every
// generic, so the `Context` of the entry has them (`using()`), and `Projection::of()` encodes
// them by their own names. A site writes the arguments it gives with `complete_args()`, which
// takes the defaults, and `substitute_args()` (`at_site()`).

/// The generics without their defaults, which the context of an entry has none of.
fn without_defaults(generics: &Generics) -> Generics {
    let mut generics = generics.clone();
    for param in generics.params.iter_mut() {
        match param {
            GenericParam::Type(param) => {
                param.eq_token = None;
                param.default = None;
            }
            GenericParam::Const(param) => {
                param.eq_token = None;
                param.default = None;
            }
            GenericParam::Lifetime(_) => (),
        }
    }
    generics
}

/// The Referrer of `trait Tr<generics> { fn f(&self) -> All<..>; }`, whose only entry, `All<..>`,
/// names every generic: its context has all of them, without defaults, sorted (lifetimes, types,
/// consts, each by name), and the lifetime bounds as written, whatever they bound. If `outlives`
/// is not empty the trait is `where Self: 'a + ..` and the entry names `Self`: `Self: 'a` is in
/// `Context::self_outlives`, sorted.
fn using(generics: &Generics, outlives: &[&str]) -> Referrer {
    let mut names: Vec<String> = generics
        .params
        .iter()
        .map(|param| match param {
            GenericParam::Lifetime(param) => s(&param.lifetime),
            GenericParam::Type(param) => s(&param.ident),
            GenericParam::Const(param) => s(&param.ident),
        })
        .collect();
    let mut where_clause = String::new();
    if !outlives.is_empty() {
        where_clause = format!("where Self: {}", outlives.join(" + "));
        names.push("Self".to_string());
    }
    let entry = if names.is_empty() {
        "All".to_string()
    } else {
        format!("All<{}>", names.join(", "))
    };
    let item: ItemTrait = parse_str(&format!(
        "trait Tr {} {} {{ fn f(&self) -> {}; }}",
        s(&without_defaults(generics)),
        where_clause,
        entry
    ))
    .unwrap();
    let mut leaker = Leaker::new();
    leaker.intern_trait(&item);
    let result = leaker.result();
    assert!(result.errors().is_empty(), "{}", s(&item));
    result.finish().unwrap()
}

/// The first argument of the Repeater that `Projection::of()` writes for the only entry of
/// `referrer`, `ARGS` of `<self_ty as Repeater<ARGS, KEY>>::Type`: the generics of its context, by
/// their own names.
fn args_of(referrer: &Referrer, self_ty: &Type) -> std::result::Result<Type, EncodeError> {
    let (ix, _, context) = referrer.iter().next().unwrap();
    let projection = common::projection(repeater(), 7).of(self_ty, context, ix)?;
    Ok(common::projection_args(&projection))
}

/// The arguments written as `<..>` at a site, with the bindings.
fn given(args: &str) -> Vec<GenericArgument> {
    let args: AngleBracketedGenericArguments = parse_str(args).unwrap();
    args.args.into_iter().collect()
}

/// The same at a site that gives `args` for the parameters `written` of the item: the arguments are
/// completed (`complete_args()`), and take the place of the names (`substitute_args()`).
/// `self_ty` is what `Self: 'a` names (`None`: `Self`).
fn at_site(
    referrer: &Referrer,
    written: &Generics,
    args: &[GenericArgument],
    self_ty: Option<&Type>,
) -> std::result::Result<Type, EncodeError> {
    let completed = common::complete_generics(written, args)?;
    common::repeater_args_at(
        &repeater(),
        common::entry(referrer, 0),
        common::answers(&written.params, &completed),
        self_ty,
    )
}

/// The `Context` of the only entry of `referrer`, with `Self: 'a` for the lifetimes `outlives`, in
/// this order. The analysis sorts them, and names only the lifetimes of the generics, so this is the
/// one input that no item gives (`Context::self_outlives` is a public field of a value that can be
/// cloned).
fn with_outlives(referrer: &Referrer, outlives: &[&str]) -> (EntryId, Context) {
    let (ix, _, context) = referrer.iter().next().unwrap();
    let mut context = context.clone();
    context.self_outlives = outlives.iter().map(|l| parse_str(l).unwrap()).collect();
    (ix, context)
}

/// Each const type has its slot of the Repeater, for the generics of an entry and for the arguments
/// of a site. The context sorts the consts by name (`B`, `I`, `M`, `N`, `X`), and a site writes a
/// const that it gives in braces (`[(); { 3 }]`), as `substitute_args()` does.
#[test]
fn the_slots_encode_the_consts_of_the_generics_of_an_entry_and_of_a_site() {
    // The trait as written: `M` has the default `N`.
    let written: Generics = parse_quote! {
        <'a, T, const N: usize, const X: u128, const B: bool, const I: i32, const M: usize = N>
    };
    let referrer = using(&written, &[]);
    let encoded = args_of(&referrer, &parse_quote!(Self)).unwrap();
    let expected: Type = parse_quote!((
        &'a (),
        T,
        *const dyn crate::Repeater<(), 0, B, Type = ()>,
        *const dyn crate::Repeater<(), 0, false, '\0', 0, 0, 0, 0, 0, 0, I, Type = ()>,
        [(); M],
        [(); N],
        *const dyn crate::Repeater<(), X, Type = ()>
    ));
    assert_eq!(s(&encoded), s(&expected));

    // The arguments of a site: a bare const `B` (the slot keeps it), a negative literal, a skipped
    // binding, and an omitted default that names a preceding parameter (`M = N` is `3`).
    let args = given("<'static, u8, 3, 7, B, -5, Item = u8>");
    let ty = at_site(&referrer, &written, &args, None).unwrap();
    let expected: Type = parse_quote!((
        &'static (),
        u8,
        *const dyn crate::Repeater<(), 0, B, Type = ()>,
        *const dyn crate::Repeater<(), 0, false, '\0', 0, 0, 0, 0, 0, 0, -5, Type = ()>,
        [(); { 3 }],
        [(); { 3 }],
        *const dyn crate::Repeater<(), 7, Type = ()>
    ));
    assert_eq!(s(&ty), s(&expected));

    // A block is a const argument, and a `bool` and a `0` are written as they are.
    let args = given("<'static, u8, { 1 + 2 }, 7, true, 0>");
    let ty = at_site(&referrer, &written, &args, None).unwrap();
    let expected: Type = parse_quote!((
        &'static (),
        u8,
        *const dyn crate::Repeater<(), 0, true, Type = ()>,
        *const dyn crate::Repeater<(), 0, false, '\0', 0, 0, 0, 0, 0, 0, 0, Type = ()>,
        [(); { 1 + 2 }],
        [(); { 1 + 2 }],
        *const dyn crate::Repeater<(), 7, Type = ()>
    ));
    assert_eq!(s(&ty), s(&expected));

    // Errors of the arguments: a missing lifetime, an extra lifetime, a missing argument that has no
    // default.
    for (args, message) in [
        ("<u8, 3, 7, B, -5>", "missing generic argument for ''a'"),
        (
            "<'static, 'static, u8, 3, 7, B, -5>",
            "unexpected generic argument: ''static'",
        ),
        (
            "<'static, u8>",
            "missing generic argument for 'const N : usize'",
        ),
    ] {
        let err = at_site(&referrer, &written, &given(args), None);
        assert_eq!(err.unwrap_err().to_string(), message, "{}", args);
    }
    // A `bool` has its slot, and an alias has none (`the_text_of_every_variant`).
    let flag: Generics = parse_quote!(<const B: bool>);
    let flag = using(&flag, &[]);
    let ty = args_of(&flag, &parse_quote!(Self)).unwrap();
    assert_eq!(
        s(&ty),
        t(parse_quote!((
            *const dyn crate::Repeater<(), 0, B, Type = ()>,
        )))
    );

    // No generics, and a single generic (a one-element tuple).
    let none = using(&Generics::default(), &[]);
    let ty = args_of(&none, &parse_quote!(Self)).unwrap();
    assert_eq!(s(&ty), t(parse_quote!(())));
    let single: Generics = parse_quote!(<T>);
    let ty = args_of(&using(&single, &[]), &parse_quote!(Self)).unwrap();
    assert_eq!(s(&ty), t(parse_quote!((T,))));
}

/// The const parameters of the impl and of the use site are encoded into the same slots.
#[test]
fn repeater_slots_encode_the_consts_of_the_impl_and_of_the_use_site() {
    let tr: ItemTrait = parse_quote! {
        trait Tr<const B: bool, const C: char = 'z'> { fn f(&self) -> Wrap<B, C>; }
    };
    let mut leaker = Leaker::new();
    leaker.intern_trait(&tr);
    let result = leaker.result();
    let referrer = result.clone().finish().unwrap();
    let repeater = common::repeater();

    // The impl encodes the parameters themselves...
    let param: Ident = parse_quote!(__S);
    let r = result
        .repeater_impls(&Projection::new(
            repeater.clone(),
            7,
            ImplSelf::Param(param.clone()),
        ))
        .unwrap()
        .remove(0);
    assert_eq!(
        common::impl_args(&r),
        parse_quote!((
            *const dyn crate::Repeater<(), 0, B, Type = ()>,
            *const dyn crate::Repeater<(), 0, false, C, Type = ()>
        ))
    );

    // ...and a copy in `impl Tr<true, 'x'> for ..` the arguments.
    let trait_args: Vec<GenericArgument> = vec![parse_quote!(true), parse_quote!('x')];
    let args = common::repeater_args_at(
        &repeater,
        common::entry(&referrer, 0),
        common::answers(&tr.generics.params, &trait_args),
        None,
    )
    .unwrap();
    assert_eq!(
        args,
        parse_quote!((
            *const dyn crate::Repeater<(), 0, true, Type = ()>,
            *const dyn crate::Repeater<(), 0, false, 'x', Type = ()>
        ))
    );
}

/// A default that names a const as a type argument (`IntoIter<u8, N>`: syn parses the `N` as a type)
/// takes the value of the const when it is completed.
#[test]
fn a_default_that_names_a_const_as_a_type_argument_takes_its_value() {
    let written: Generics = parse_quote!(<const N: usize, U = ::core::array::IntoIter<u8, N>>);
    let args = given("<3>");
    let completed = common::complete_generics(&written, &args).unwrap();
    let expected: Vec<GenericArgument> = vec![
        parse_quote!(3),
        parse_quote!(::core::array::IntoIter<u8, 3>),
    ];
    assert_eq!(completed, expected);
    // The encoding at the site: the types, then the const, as the context has them.
    let ty = at_site(&using(&written, &[]), &written, &args, None).unwrap();
    assert_eq!(
        s(&ty),
        t(parse_quote!((::core::array::IntoIter<u8, 3>, [(); { 3 }])))
    );
}

/// The arguments that a site gives are completed for the generics as written (the defaults are
/// there), and written for the names in the encoding of the generics of the entry. A bare `N` is a
/// const, bindings are skipped, and too many or too few arguments are the errors of
/// `complete_args()`. The context sorts the generics, so the arguments go by name.
#[test]
fn the_arguments_of_a_site_are_completed_for_the_generics_as_written_and_placed_by_name() {
    type Row = (
        Generics,
        &'static str,
        std::result::Result<Type, &'static str>,
    );
    let rows: Vec<Row> = vec![
        (
            parse_quote!(<'a, T, const N: usize>),
            "<'static, u8, N>",
            Ok(parse_quote!((&'static (), u8, [(); { N }]))),
        ),
        (
            parse_quote!(<'a, T, const N: usize>),
            "<'static, u8, { 3 }>",
            Ok(parse_quote!((&'static (), u8, [(); { 3 }]))),
        ),
        (
            parse_quote!(<'a, T, const N: usize, U = Vec<T>>),
            "<'static, u8, N, Item = u8>",
            Ok(parse_quote!((&'static (), u8, Vec<u8>, [(); { N }]))),
        ),
        (
            parse_quote!(<'a, T, const N: usize, U = Vec<T>>),
            "<'static, u8, 3, u16, u32>",
            Err("unexpected generic argument: 'u32'"),
        ),
        (
            parse_quote!(<'a, T, const N: usize, U = Vec<T>>),
            "<'static, u8>",
            Err("missing generic argument for 'const N : usize'"),
        ),
        (
            parse_quote!(<T>),
            "<u8, Item = u8>",
            Ok(parse_quote!((u8,))),
        ),
        (
            parse_quote!(<T, const N: usize, const M: usize = 3>),
            "<u8, N>",
            Ok(parse_quote!((u8, [(); { 3 }], [(); { N }]))),
        ),
        // The bounds of the generics are encoded, so the arguments are given for the generics as
        // written, and the arguments of a context's own generics (defaults dropped) are not enough.
        (
            parse_quote!(<'a, T: 'a>),
            "<'static, u8>",
            Ok(parse_quote!((&'static (), u8, &'static u8))),
        ),
        (
            parse_quote!(<T, U>),
            "<u8>",
            Err("missing generic argument for 'U'"),
        ),
        // The documented examples.
        (
            parse_quote!(<'a, T: 'a, const N: usize>),
            "<'static, u8, 3>",
            Ok(parse_quote!((&'static (), u8, [(); { 3 }], &'static u8))),
        ),
        (
            parse_quote!(<T: ?Sized, U>),
            "<str, u8>",
            Ok(parse_quote!((*const str, u8))),
        ),
    ];
    for (written, args, expected) in rows {
        let case = format!("{} with {}", s(&written), args);
        let referrer = using(&written, &[]);
        let encoded = at_site(&referrer, &written, &given(args), None);
        match (encoded, expected) {
            (Ok(ty), Ok(expected)) => assert_eq!(s(&ty), s(&expected), "{}", case),
            (Err(err), Err(expected)) => assert_eq!(err.to_string(), expected, "{}", case),
            (encoded, expected) => panic!("{}: {:?} against {:?}", case, encoded, expected),
        }
    }
}

/// A lifetime bound `T: 'a` is `&'a T` and `'b: 'a` is `&'a &'b ()`, after the elements of the
/// parameters, in the order of the parameters and of their bounds. A bound by `'static`, and one by
/// a lifetime that is not a parameter, is not encoded. The const parameters are encoded into the
/// slots, which get the parameters only.
#[test]
fn the_lifetime_bounds_of_the_generics_are_encoded_after_the_parameters() {
    let rows: Vec<(Generics, Type)> = vec![
        (
            parse_quote!(<
                'a, 'b: 'a, T: 'a + ::core::clone::Clone + 'b, U: 'static, V: 'c, const N: usize
            >),
            parse_quote!((&'a (), &'b (), T, U, V, [(); N], &'a &'b (), &'a T, &'b T)),
        ),
        (
            parse_quote!(<'a, T: 'a, const N: usize>),
            parse_quote!((&'a (), T, [(); N], &'a T)),
        ),
        // Without the bound, nothing follows the parameters.
        (
            parse_quote!(<'a, T, const N: usize>),
            parse_quote!((&'a (), T, [(); N])),
        ),
        (
            parse_quote!(<'a, 'b: 'a, T: 'a + ::core::clone::Clone, const N: usize>),
            parse_quote!((&'a (), &'b (), T, [(); N], &'a &'b (), &'a T)),
        ),
        // A single parameter and its bound; no bound, no element.
        (parse_quote!(<'a: 'a>), parse_quote!((&'a (), &'a &'a ()))),
        (parse_quote!(<T: 'static>), parse_quote!((T,))),
    ];
    for (generics, expected) in rows {
        let referrer = using(&generics, &[]);
        let ty = args_of(&referrer, &parse_quote!(Self)).unwrap();
        assert_eq!(s(&ty), s(&expected), "{}", s(&generics));
    }

    // The slots are used for the parameters only: not for the bounds.
    let generics: Generics = parse_quote!(<'a, T: 'a, const B: bool>);
    let referrer = using(&generics, &[]);
    let ty = args_of(&referrer, &parse_quote!(Self)).unwrap();
    let expected: Type = parse_quote!((
        &'a (),
        T,
        *const dyn crate::Repeater<(), 0, B, Type = ()>,
        &'a T
    ));
    assert_eq!(s(&ty), s(&expected));
}

/// A site's lifetimes and types replace the names in the encoded bounds as well as in the elements
/// of the parameters, whatever they are (the lifetimes of the Referrer's generics, a type of another
/// crate), and a site that gives each generic by its own name changes nothing but the braces of a
/// const.
#[test]
fn the_arguments_of_a_site_replace_the_names_in_the_encoded_bounds() {
    let written: Generics = parse_quote!(<'a, 'b: 'a, T: 'a + 'b, const N: usize, U = Vec<T>>);
    let referrer = using(&written, &[]);
    let expected: Type = parse_quote!((
        &'static (),
        &'x (),
        u8,
        Vec<u8>,
        [(); { 3 }],
        &'static &'x (),
        &'static u8,
        &'x u8
    ));
    let args = given("<'static, 'x, u8, 3>");
    let ty = at_site(&referrer, &written, &args, None).unwrap();
    assert_eq!(s(&ty), s(&expected));

    // The arguments may be any types, and the lifetimes of the Referrer's generics.
    let args = given("<'p, 'q, ::p::W<&'p u8>, N, ()>");
    let ty = at_site(&referrer, &written, &args, None).unwrap();
    let expected: Type = parse_quote!((
        &'p (),
        &'q (),
        ::p::W<&'p u8>,
        (),
        [(); { N }],
        &'p &'q (),
        &'p ::p::W<&'p u8>,
        &'q ::p::W<&'p u8>
    ));
    assert_eq!(s(&ty), s(&expected));

    // The encoding of the arguments is that of the parameters they replace: the same type with the
    // parameters as the arguments (a const is `{ N }` where it is substituted).
    for generics in [
        parse_quote!(<'a, 'b: 'a, T: 'a + 'b, const N: usize>),
        parse_quote!(<'a, T: 'a, U: 'a + ::core::clone::Clone, V: 'static>),
        parse_quote!(<'a: 'a, 'b: 'a>),
        parse_quote!(<T>),
        parse_quote!(<>),
    ] {
        let generics: Generics = generics;
        let referrer = using(&generics, &[]);
        let own: Vec<GenericArgument> = generics.params.iter().map(common::own_name_arg).collect();
        let params = args_of(&referrer, &parse_quote!(Self)).unwrap();
        let replaced = at_site(&referrer, &generics, &own, None).unwrap();
        assert_eq!(
            s(&replaced).replace("{ N }", "N"),
            s(&params),
            "{}",
            s(&generics)
        );
    }
}

/// A trait object that a site gives for a `?Sized` parameter is a pointer to it in the elements of
/// the parameters, and the one with several bounds is parenthesized (`&'a dyn A + 'b` is ambiguous).
/// A parameter that has no bound on a lifetime takes any argument.
///
/// A `dyn Tr` with no lifetime bound is written as it is (`&'x dyn Tr` for `&'a T`) by
/// `substitute_args()`, which has no `UnboundedTraitObject`: that error is the one for the type that
/// `Self: 'a` names (`unbounded_self_error()`).
#[test]
fn a_trait_object_that_a_site_gives_for_an_unsized_parameter_is_a_pointer_to_it() {
    let written: Generics = parse_quote!(<'a, T: ?Sized + 'a>);
    let referrer = using(&written, &[]);
    for object in [
        parse_quote!(dyn Tr + 'static),
        parse_quote!((dyn Tr + 'y)),
        parse_quote!(dyn Tr + Send + 'static),
    ] {
        let object: Type = object;
        let args = vec![parse_quote!('x), GenericArgument::Type(object.clone())];
        let ty = at_site(&referrer, &written, &args, None).unwrap();
        // A parameter that may be unsized is a pointer, so as to be `Sized`.
        let pointee = match &object {
            Type::Paren(TypeParen { elem, .. }) => elem.as_ref().clone(),
            object => object.clone(),
        };
        let expected: Type = parse_quote!((&'x (), *const (#pointee), &'x (#pointee)));
        assert_eq!(s(&ty), s(&expected), "{}", s(&object));
    }
    // Without a bound on the parameter, any argument is fine.
    let written: Generics = parse_quote!(<'a, T: ?Sized>);
    let args = given("<'x, dyn Tr>");
    let ty = at_site(&using(&written, &[]), &written, &args, None).unwrap();
    assert_eq!(s(&ty), t(parse_quote!((&'x (), *const dyn Tr))));
}

/// `Self: 'a` follows the lifetime bounds, once for each lifetime of `Context::self_outlives`, with
/// the type that stands for `Self`; a lifetime that is not a parameter of the generics is skipped.
/// The arguments of a site replace the lifetimes, `Self` included, and a trait object without a
/// lifetime bound cannot stand for `Self` there.
#[test]
fn self_outlives_is_encoded_after_the_lifetime_bounds_with_the_type_for_self() {
    let generics: Generics = parse_quote!(<'a, 'b, T: 'a, const N: usize>);
    let with_self = using(&generics, &["'b", "'a"]);
    // The analysis sorts the lifetimes.
    let outlives = |context: &Context| context.self_outlives.iter().map(s).collect::<Vec<_>>();
    assert_eq!(outlives(common::entry(&with_self, 0)), ["'a", "'b"]);
    let plain = using(&generics, &[]);
    assert!(outlives(common::entry(&plain, 0)).is_empty());
    let self_ty: Type = parse_quote!(__S);
    let ty = args_of(&with_self, &self_ty).unwrap();
    let expected: Type = parse_quote!((&'a (), &'b (), T, [(); N], &'a T, &'a __S, &'b __S));
    assert_eq!(s(&ty), s(&expected));

    // Without the lifetimes it is the encoding of the generics alone.
    let alone: Type = parse_quote!((&'a (), &'b (), T, [(); N], &'a T));
    let ty = args_of(&plain, &self_ty).unwrap();
    assert_eq!(s(&ty), s(&alone));

    // The order of the context's list is the order of the elements, and a lifetime which is not a
    // parameter is skipped.
    let project = |outlives: &[&str]| {
        let (ix, context) = with_outlives(&with_self, outlives);
        let projection = common::projection(repeater(), 7)
            .of(&self_ty, &context, ix)
            .unwrap();
        s(&common::projection_args(&projection))
    };
    let reversed: Type = parse_quote!((&'a (), &'b (), T, [(); N], &'a T, &'b __S, &'a __S));
    assert_eq!(project(&["'b", "'a"]), s(&reversed));
    assert_eq!(project(&["'c"]), s(&alone));
    assert_eq!(
        project(&["'a", "'c"]),
        t(parse_quote!((&'a (), &'b (), T, [(); N], &'a T, &'a __S)))
    );
    assert_eq!(project(&[]), s(&alone));

    // At a site the arguments replace the parameters in the encoding, `Self: 'a` included, and
    // `Self` stays a name.
    let args = given("<'p, 'q, ::p::W, N>");
    let ty = at_site(&with_self, &generics, &args, None).unwrap();
    let expected: Type = parse_quote!((
        &'p (),
        &'q (),
        ::p::W,
        [(); { N }],
        &'p ::p::W,
        &'p Self,
        &'q Self
    ));
    assert_eq!(s(&ty), s(&expected));

    // A trait object without a lifetime bound is rejected as `Self`, if there is a lifetime to
    // outlive (`&'a dyn Tr` would default to `&'a (dyn Tr + 'a)`), by the encoding of the entry and
    // by the one at a site.
    let generics: Generics = parse_quote!(<'a>);
    let with_self = using(&generics, &["'a"]);
    let plain = using(&generics, &[]);
    let dyn_self: Type = parse_quote!(dyn Tr);
    let err = args_of(&with_self, &dyn_self).unwrap_err();
    assert!(
        matches!(err, EncodeError::UnboundedTraitObject(_)),
        "{}",
        err
    );
    let args = given("<'x>");
    let err = at_site(&with_self, &generics, &args, Some(&dyn_self)).unwrap_err();
    assert!(
        matches!(err, EncodeError::UnboundedTraitObject(_)),
        "{}",
        err
    );
    // but not if there is none, and it is fine with the bound written.
    let ty = args_of(&plain, &dyn_self).unwrap();
    assert_eq!(s(&ty), t(parse_quote!((&'a (),))));
    let bounded: Type = parse_quote!(dyn Tr + 'static);
    let ty = args_of(&with_self, &bounded).unwrap();
    assert_eq!(s(&ty), t(parse_quote!((&'a (), &'a (dyn Tr + 'static)))));
}

/// A `?Sized` type parameter is a pointer to it, not a bare element, which is `Sized` whatever the
/// parameter is: for the generics of an entry wherever it is (first, in the middle, last, the only
/// one, with other bounds), and for the arguments of a site, which are any type (F1: see
/// `f01_unsized_type_param_encoding` in `tests/findings/findings.rs` for the generics in the entries of items).
#[test]
fn an_unsized_parameter_is_encoded_as_a_pointer_to_it() {
    let rows: Vec<(Generics, Type)> = vec![
        (parse_quote!(<T: ?Sized, U>), parse_quote!((*const T, U))),
        (
            parse_quote!(<T, U: ?Sized, V>),
            parse_quote!((T, *const U, V)),
        ),
        (parse_quote!(<T, U: ?Sized>), parse_quote!((T, *const U))),
        (parse_quote!(<T: ?Sized>), parse_quote!((*const T,))),
        // Other bounds stay out of the way, and a lifetime bound is encoded as it is for a sized one.
        (
            parse_quote!(<'a, T: ::core::clone::Clone + ?Sized + 'a, U>),
            parse_quote!((&'a (), *const T, U, &'a T)),
        ),
        // A sized parameter is encoded as it is.
        (
            parse_quote!(<T, U: ::core::marker::Sized>),
            parse_quote!((T, U)),
        ),
    ];
    for (generics, expected) in rows {
        let referrer = using(&generics, &[]);
        let ty = args_of(&referrer, &parse_quote!(Self)).unwrap();
        assert_eq!(s(&ty), s(&expected), "{}", s(&generics));
        // The Referrer passes each parameter by its own name: the same encoding.
        let own: Vec<GenericArgument> = generics.params.iter().map(common::own_name_arg).collect();
        let by_name = at_site(&referrer, &generics, &own, None).unwrap();
        assert_eq!(s(&by_name), s(&ty), "{}", s(&generics));
    }

    // The arguments of a site: a pointer to the argument, whatever it is.
    let generics: Generics = parse_quote!(<T: ?Sized, U: ?Sized, V>);
    let referrer = using(&generics, &[]);
    let args = given("<str, dyn Tr + Send + 'static, [u8; 2]>");
    let ty = at_site(&referrer, &generics, &args, None).unwrap();
    let expected: Type = parse_quote!((*const str, *const (dyn Tr + Send + 'static), [u8; 2]));
    assert_eq!(s(&ty), s(&expected));
}

// ── `Referrer::complete_args()` and the generics as declared ─────────────────────────────────

/// The arguments of an impl of the scope `context` of `generics`, completed for the Repeater
/// `crate::Repeater` and the id 9, as text.
fn completed(
    referrer: &Referrer,
    context: ContextId,
    generics: &Generics,
    given: &[&str],
) -> std::result::Result<Vec<String>, EncodeError> {
    common::completed_args_for(
        referrer,
        context,
        generics,
        &arguments(given),
        &common::projection(repeater(), 9),
    )
    .map(|completed| completed.iter().map(s).collect())
}

/// The entry `Rel<T>`-like type that the `s` of the text of its tokens names.
fn entry_of(referrer: &Referrer, text: &str) -> EntryId {
    common::entry_id_at(referrer, common::entry_index(referrer, text))
}

/// `<Self as crate::Repeater<args, KEY>>::Type` of the entry with the text `entry`.
fn projection(referrer: &Referrer, entry: &str, args: &str) -> String {
    let key = common::key(9, entry_of(referrer, entry));
    let args: Type = parse_str(args).unwrap();
    let ty: Type = parse_quote!(<Self as crate::Repeater<#args, #key>>::Type);
    s(&ty)
}

#[test]
fn complete_args_completes_the_arguments_of_an_analyzed_trait() {
    let tr: ItemTrait = parse_quote! {
        trait Tr<'a, T: 'a, const N: usize = 3, U = Rel<T>, V = Vec<U>> {
            fn f(&'a self, t: T) -> RelOut;
        }
    };
    let mut leaker = Leaker::with_rule(Rules::new().prelude());
    let context = leaker.intern_trait(&tr);
    let referrer = leaker.result().finish().unwrap();
    let rel = |t: &str| {
        // The entry `Rel<T>` has the generics `T` (its bound `T: 'a` is encoded) and the lifetime.
        projection(
            &referrer,
            "Rel < T >",
            &format!("(&'static (), {t}, &'static {t})", t = t),
        )
    };
    // (case, the arguments given, the arguments completed)
    for (case, given, expected) in [
        (
            "the defaults are left out",
            &["'static", "u8"][..],
            vec![
                "'static".to_string(),
                "u8".to_string(),
                "3".to_string(),
                rel("u8"),
                format!("Vec < {} >", rel("u8")),
            ],
        ),
        (
            "a default is given, so the next one uses it",
            &["'static", "u8", "5", "i32"][..],
            vec![
                "'static".to_string(),
                "u8".to_string(),
                "5".to_string(),
                "i32".to_string(),
                "Vec < i32 >".to_string(),
            ],
        ),
        (
            "every argument is given",
            &["'x", "u8", "5", "i32", "i64"][..],
            ["'x", "u8", "5", "i32", "i64"].map(String::from).to_vec(),
        ),
        (
            "a binding is skipped, and the kinds go apart",
            &["u8", "'static", "Item = u16"][..],
            vec![
                "'static".to_string(),
                "u8".to_string(),
                "3".to_string(),
                rel("u8"),
                format!("Vec < {} >", rel("u8")),
            ],
        ),
        (
            "a type named like a parameter is not substituted inside a projection",
            &["'static", "U"][..],
            vec![
                "'static".to_string(),
                "U".to_string(),
                "3".to_string(),
                rel("U"),
                format!("Vec < {} >", rel("U")),
            ],
        ),
    ] {
        assert_eq!(
            completed(&referrer, context, &tr.generics, given).unwrap(),
            expected,
            "{}",
            case
        );
    }
    // The errors are `EncodeError`s, from the declaration.
    for (case, given) in [
        ("a type for the lifetime", &["u8", "u8"][..]),
        ("nothing for `T`", &["'static"][..]),
        (
            "one too many",
            &["'static", "u8", "5", "i32", "i64", "u8"][..],
        ),
    ] {
        assert!(
            completed(&referrer, context, &tr.generics, given).is_err(),
            "{}",
            case
        );
    }
}

/// The declared order of the parameters is kept: the generics of the context are sorted by name
/// (lifetimes, types, consts), which is not the order that the arguments of an impl are written in.
#[test]
fn complete_args_takes_the_arguments_in_the_declared_order() {
    let tr: ItemTrait = parse_quote! {
        trait Tr<B, const N: usize, 'a, A: 'a> { fn f(&'a self, a: A, b: B) -> Rel<A, B>; }
    };
    // A lifetime parameter after a type is not valid Rust, but the declaration is kept as written:
    // the kinds are matched apart, the order inside a kind is the declared one.
    let tr: ItemTrait = ItemTrait {
        generics: parse_quote!(<'a, B, A: 'a, const N: usize>),
        ..tr
    };
    let mut leaker = Leaker::new();
    let context = leaker.intern_trait(&tr);
    let referrer = leaker.result().finish().unwrap();
    // The generics of the context are `<'a, A: 'a, B>` (`N` is not used by the entry).
    let entry = common::entry(&referrer, 0);
    let names: Vec<String> = entry.generics.params.iter().map(s).collect();
    assert_eq!(names, ["'a", "A : 'a", "B"]);
    let completed = completed(&referrer, context, &tr.generics, &["'x", "u8", "u16", "3"]).unwrap();
    assert_eq!(completed, ["'x", "u8", "u16", "3"]);
    // `B` is the first type, `A` the second: not the order of the generics of the context.
    let mut sig: Signature = parse_quote!(fn f(&'a self, a: A, b: B));
    let mut substitute = referrer
        .complete_args(
            context,
            &arguments(&["'x", "u8", "u16", "3"]),
            &Projection::new(repeater(), 9, ImplSelf::Param(parse_quote!(__S))),
        )
        .unwrap();
    substitute.visit_signature_mut(&mut sig);
    let expected: Signature = parse_quote!(fn f(&'x self, a: u16, b: u8));
    assert_eq!(s(&sig), s(&expected));
}

/// Two traits whose generics are equal once canonical, and whose declarations differ, are two
/// contexts, so each keeps its own declaration. The same declaration is one context. The contexts
/// of the entries keep their ids: the entries of the traits that have the same context share it.
#[test]
fn two_traits_with_equal_canonical_generics_and_other_declarations_are_two_contexts() {
    let traits: Vec<ItemTrait> = vec![
        parse_quote!(
            trait A<T, U = u8> {
                fn f(&self) -> Rel<T>;
            }
        ),
        parse_quote!(
            trait B<T, U = u16> {
                fn f(&self) -> Rel<T>;
            }
        ),
        parse_quote!(
            trait C<B, A> {
                fn f(&self) -> Rel<B>;
            }
        ),
        parse_quote!(
            trait D<A, B> {
                fn f(&self) -> Rel<B>;
            }
        ),
        // The same declaration as the first one, in another trait.
        parse_quote!(
            trait E<T, U = u8> {
                fn g(&self) -> Rel<T>;
            }
        ),
    ];
    let mut leaker = Leaker::with_rule(Rules::new().primitive());
    let contexts: Vec<ContextId> = traits.iter().map(|tr| leaker.intern_trait(tr)).collect();
    // The same trait again gives the same context.
    assert_eq!(leaker.intern_trait(&traits[1]), contexts[1]);
    let referrer = leaker.result().finish().unwrap();
    let [a, b, c, d, e] = [0, 1, 2, 3, 4].map(|i| contexts[i]);
    assert_ne!(a, b);
    assert_ne!(c, d);
    assert_eq!(a, e);
    for (x, y) in [(a, c), (a, d), (b, c), (b, d)] {
        assert_ne!(x, y);
    }
    // `Rel<T>` of the first two, and `Rel<B>`, `Rel<B>` of the next ones: the entries use the
    // generics `<T>` and `<B>`, which no top level is.
    let entry_contexts = common::context_ids(&referrer);
    for top_level in [a, b, c, d] {
        assert!(!common::serialized_ids(&referrer)
            .entries
            .contains(&top_level));
        assert!(entry_contexts.contains(&top_level));
    }
    // Each completes its own defaults, before and after the tokens.
    let tokens = quote!(#referrer);
    let parsed: Referrer = parse2(tokens).unwrap();
    assert_eq!(parsed, referrer);
    for referrer in [&referrer, &parsed] {
        assert_eq!(
            completed(referrer, a, &traits[0].generics, &["u32"]).unwrap(),
            ["u32", "u8"]
        );
        assert_eq!(
            completed(referrer, b, &traits[1].generics, &["u32"]).unwrap(),
            ["u32", "u16"]
        );
        assert_eq!(
            completed(referrer, e, &traits[4].generics, &["u32"]).unwrap(),
            ["u32", "u8"]
        );
        // `<B, A>` and `<A, B>`: the first argument is for the first declared parameter.
        let written: Signature = parse_quote!(fn f(a: A, b: B));
        for (context, expected) in [(c, "fn f(a: u16, b: u8)"), (d, "fn f(a: u8, b: u16)")] {
            let mut sig = written.clone();
            referrer
                .complete_args(
                    context,
                    &arguments(&["u8", "u16"]),
                    &Projection::new(repeater(), 9, ImplSelf::Param(parse_quote!(__S))),
                )
                .unwrap()
                .visit_signature_mut(&mut sig);
            assert_eq!(s(&sig), s(&parse_str::<Signature>(expected).unwrap()));
        }
    }
}

/// Every scope of generics has its top-level context declared: a struct, an enum, a union, an impl,
/// `enter_generics()`, and a scope inside another one.
#[test]
fn every_scope_of_generics_completes_arguments() {
    let given = arguments(&["u8"]);
    let defaulted: Generics = parse_quote!(<T, U = Vec<T>>);
    let complete = |referrer: &Referrer, context: ContextId| -> Vec<String> {
        common::completed_args_for(
            referrer,
            context,
            &defaulted,
            &given,
            &common::projection(repeater(), 9),
        )
        .unwrap()
        .iter()
        .map(s)
        .collect()
    };
    let mut leaker = Leaker::with_rule(Rules::new().prelude());
    let contexts = vec![
        (
            "struct",
            leaker.intern_struct(&parse_quote!(
                struct S<T, U = Vec<T>>(Rel<T>, U);
            )),
        ),
        (
            "enum",
            leaker.intern_enum(&parse_quote!(
                enum E<T, U = Vec<T>> {
                    A(Rel<T>, U),
                }
            )),
        ),
        (
            "union",
            leaker.intern_union(
                &parse_quote!(union U<T: Copy, U: Copy = Vec<T>> { a: Rel<T>, b: U }),
            ),
        ),
        ("generics", leaker.enter_generics(&defaulted, |_| ()).0),
        (
            "type",
            leaker.enter_type(&parse_quote!(S2), &defaulted, |_| ()).0,
        ),
        (
            "trait",
            leaker.enter_trait(&parse_quote!(Tr), &defaulted, |_| ()).0,
        ),
        (
            "impl",
            leaker
                .enter_impl(&defaulted, &parse_quote!(Host<T, U>), None, |_| ())
                .0,
        ),
        (
            "a trait in a scope",
            leaker
                .enter_generics(&Generics::default(), |view| {
                    view.enter_trait(&parse_quote!(Tr), &defaulted, |_| ()).0
                })
                .1,
        ),
    ];
    let referrer = leaker.result().finish().unwrap();
    for (case, context) in contexts {
        // `Vec` is allowed, so the default is as written, with `T` replaced.
        assert_eq!(
            complete(&referrer, context),
            ["u8", "Vec < u8 >"],
            "{}",
            case
        );
    }
}

/// A default that names `Self` (`U = Rel<Self>`) is a known limit: the Leaker reports it, and the
/// completion leaves it as written, with no panic.
#[test]
fn a_default_that_names_self_is_left_as_written() {
    let tr: ItemTrait = parse_quote!(
        trait Tr<T, U = Rel<Self>> {
            fn f(&self, u: U);
        }
    );
    let mut leaker = Leaker::new();
    let context = leaker.intern_trait(&tr);
    let result = leaker.result();
    let err = result.errors();
    assert_eq!(err.len(), 1);
    assert_eq!(err[0].reason, Reason::SelfInDefault);
    assert_eq!(
        err[0].message,
        "`Rel<Self>` is in the default of a parameter, and names `Self`"
    );
    assert_eq!(
        err[0].notes.first().map(|(_, n)| n.as_str()),
        Some("a default of a parameter cannot depend on `Self`")
    );
    // Nothing is leaked in `::p::W<Self>` or `Self`, which are copied, and in `(Rel, Self)` only
    // `Rel` is.
    for (default, leaked) in [
        (quote!(::p::W<Self>), 0),
        (quote!(Self), 0),
        (quote!((Rel, Self)), 1),
    ] {
        let tr: ItemTrait = parse_quote!(trait Tr<T, U = #default> {});
        let referrer = referrer_of(&tr);
        assert_eq!(referrer.iter().count(), leaked, "{}", s(&default));
        assert!(!referrer.has_self());
    }
    let referrer = result.recovered();
    assert_eq!(
        completed(&referrer, context, &tr.generics, &["u8"]).unwrap(),
        ["u8", "Rel < Self >"]
    );
    // A given argument is as given.
    assert_eq!(
        completed(&referrer, context, &tr.generics, &["u8", "u16"]).unwrap(),
        ["u8", "u16"]
    );
}

fn declared_tr() -> ItemTrait {
    parse_quote! {
        pub trait Tr<'a, K: 'a + ::core::clone::Clone, const N: usize = 3, D = Wrap<K>>
        where
            K: ::core::fmt::Debug,
        {
            fn get(&'a self, key: &'a K) -> D;
        }
    }
}

/// The generics as declared are written in place of the canonical ones, which the parser computes:
/// the Referrer is read back equal, and completes the same.
#[test]
fn the_declaration_is_written_as_tokens_and_read_back() {
    let tr = declared_tr();
    let mut leaker = Leaker::new();
    let context = leaker.intern_trait(&tr);
    let referrer = leaker.result().finish().unwrap();
    let text = s(&referrer);
    // The top-level context is the second group: only the declaration (as written), then no
    // entry. The canonical generics (sorted, with the where predicate moved to the bound, no
    // defaults) are not written, as they are for the context of the entry in the first group.
    assert!(
        text.ends_with(
            "(Wrap < K >) \
             declared < 'a , K : 'a + :: core :: clone :: Clone , const N : usize = 3 , D = Wrap < K > > \
             where K : :: core :: fmt :: Debug , ; ()"
        ),
        "{}",
        text
    );
    assert!(!text.contains("< 'a , D ,"), "{}", text);
    let parsed: Referrer = parse2(quote!(#referrer)).unwrap();
    assert_eq!(parsed, referrer);
    for referrer in [&referrer, &parsed] {
        let completed = completed(referrer, context, &tr.generics, &["'static", "u8"]).unwrap();
        assert_eq!(completed[..3], ["'static", "u8", "3"]);
        assert!(completed[3].contains("Repeater"), "{}", completed[3]);
    }
    // The context id is written as an integer, and the Referrer follows the one that the Leaker
    // returned.
    let tokens = quote!(#referrer, #context);
    let (parsed, id): (Referrer, ContextId) = (|input: syn::parse::ParseStream| {
        let referrer = input.parse()?;
        input.parse::<Token![,]>()?;
        Ok((referrer, input.parse()?))
    })
    .parse2(tokens)
    .unwrap();
    assert_eq!(parsed, referrer);
    assert_eq!(id, context);

    // A trait with no generics has the declaration `declared < > ;`: it completes nothing, and
    // it is not a module.
    let mut leaker = Leaker::new();
    let context = leaker.intern_trait(&parse_quote!(
        trait Plain {
            fn f(&self) -> Rel;
        }
    ));
    let referrer = leaker.result().finish().unwrap();
    assert!(s(&referrer).contains("declared < > ;"), "{}", s(&referrer));
    let parsed: Referrer = parse2(quote!(#referrer)).unwrap();
    assert_eq!(parsed, referrer);
    assert_eq!(
        completed(&parsed, context, &Generics::default(), &[]).unwrap(),
        Vec::<String>::new()
    );
}

/// A `LeakerResult` keeps the declaration of a scope that has errors, with its errors.
#[test]
fn the_declaration_of_a_scope_with_errors_is_carried_by_the_result() {
    let tr: ItemTrait = parse_quote! {
        trait Tr<T, D = Wrap<T>> { fn f(&self, t: T, w: Bad<'_>) -> D; }
    };
    let mut leaker = Leaker::new();
    let context = leaker.intern_trait(&tr);
    let result = leaker.result();
    assert_eq!(result.errors().len(), 1);
    let parsed: LeakerResult = parse2(quote!(#result)).unwrap();
    assert_eq!(parsed.errors().len(), 1);
    let referrer = result.recovered();
    let read = parsed.recovered();
    assert_eq!(read, referrer);
    assert_eq!(
        completed(&read, context, &tr.generics, &["u8"]).unwrap(),
        completed(&referrer, context, &tr.generics, &["u8"]).unwrap()
    );
    // A `Referrer` that was serialized alone reads as a `LeakerResult` with no error, and keeps
    // the declaration too.
    let referrer = Leaker::new().result().finish().unwrap();
    let parsed: LeakerResult = parse2(quote!(#referrer)).unwrap();
    assert!(parsed.errors().is_empty());
}

/// The parser refuses a declaration that is not well formed.
#[test]
fn a_malformed_declaration_is_refused() {
    let tag = format!("\"type-leak-format-{}\"", Referrer::FORMAT_VERSION);
    // (case, the group, part of the message)
    for (case, group, message) in [
        (
            "no angle bracket",
            "declared ;  ()",
            "expected `<` after `declared`",
        ),
        (
            "the generics and the same generics as declared",
            "< T > ; declared < T > ; ()",
            "has both `< generics >` and `declared < .. >`",
        ),
        (
            "the generics and other generics as declared",
            "< T > ; declared < U > ; ()",
            "has both `< generics >` and `declared < .. >`",
        ),
        (
            "a where clause and the generics as declared",
            "where T : Clone ; declared < T > ; ()",
            "has both `< generics >` and `declared < .. >`",
        ),
        (
            "the generics after the generics as declared",
            "declared < T > ; < T > ; ()",
            "expected",
        ),
        (
            "a declaration twice",
            "declared < > ; declared < > ; ()",
            "expected",
        ),
        (
            "a declaration after `Self`",
            "Self ; declared < > ; ()",
            "expected",
        ),
        (
            "the declaration in a second group of a declared context",
            "2 declared < > ; ( ) # 0 declared < > ; ( )",
            "declared already",
        ),
    ] {
        let tokens: proc_macro2::TokenStream = format!("{} {}", tag, group).parse().unwrap();
        let error = parse2::<Referrer>(tokens).expect_err(case);
        assert!(
            error.to_string().contains(message),
            "{}: `{}` has no `{}`",
            case,
            error,
            message
        );
    }
    // The same groups, well formed, are read.
    for group in [
        "declared < > ; ()",
        "< T > ; ()",
        "declared < T = u8 > ; ()",
        "declared < U , T : Clone > ; ()",
        "declared < U , T > where T : Clone ; ()",
    ] {
        let tokens: proc_macro2::TokenStream = format!("{} {}", tag, group).parse().unwrap();
        parse2::<Referrer>(tokens).unwrap_or_else(|error| panic!("{}: {}", group, error));
    }
    // The generics of the context are computed from the generics as declared: sorted, the where
    // predicate moved to the bound, no default.
    let tokens: proc_macro2::TokenStream = format!(
        "{} declared < U = u8 , T > where T : Clone ; ( Wrap < T > )",
        tag
    )
    .parse()
    .unwrap();
    let referrer: Referrer = parse2(tokens).unwrap();
    assert_eq!(
        s(&common::entry(&referrer, 0).generics),
        "< T : Clone , U >"
    );
}

/// The default that names an entry (the shape of `trait Store<'a, K: 'a, D = Page<K>>` of
/// `examples/two-macros`) is written for the self type of the impl that `repeater_impls()` wrote
/// for that entry: `Self` for a parameter, the marker for a marker, and the primitive type of the
/// bucket for a bucket. The impl and the written type have one self type.
#[test]
fn a_default_is_written_for_the_self_type_of_the_impl_of_its_entry() {
    let tr: ItemTrait = parse_quote! {
        trait Store<'a, K: 'a, D = Page<K>> {
            fn get(&self, key: &'a K) -> D;
            fn keep(&'a self) -> &'a K;
        }
    };
    let mut leaker = Leaker::new();
    let context = leaker.intern_trait(&tr);
    let result = leaker.result();
    let referrer = result.clone().finish().unwrap();
    let page = entry_of(&referrer, "Page < K >");
    let given: Vec<GenericArgument> = vec![parse_quote!('static), parse_quote!(u8)];
    let param: Ident = parse_quote!(__S);
    let marker: Type = parse_quote!(crate::Marker);
    let bucket = common::bucket_self(&referrer, 7, page).unwrap();
    for (name, impl_self, expected_self) in [
        ("param", ImplSelf::Param(param.clone()), parse_quote!(Self)),
        ("marker", ImplSelf::Marker(marker.clone()), marker.clone()),
        (
            "bucket",
            ImplSelf::Bucket {
                param: param.clone(),
            },
            bucket,
        ),
    ] {
        let projection = Projection::new(repeater(), 7, impl_self);
        // The impl of `Page<K>`, in crate A.
        let impls = result.repeater_impls(&projection).unwrap();
        let imp = impls.iter().find(|r| r.entry == page).unwrap();
        // The omitted `D` of `impl Store<'static, u8> for ..`, in crate B.
        let completed =
            common::completed_args_for(&referrer, context, &tr.generics, &given, &projection)
                .unwrap();
        let written = match &completed[2] {
            GenericArgument::Type(ty) => ty.clone(),
            other => panic!("{}: not a type: {}", name, s(other)),
        };
        let key = projection.key(page);
        let expected: Type = parse_quote! {
            <#expected_self as crate::Repeater<(&'static (), u8, &'static u8), #key>>::Type
        };
        assert_eq!(s(&written), s(&expected), "{}", name);
        // The impl is for the type that the copy writes. A parameter is every type, `Self` too.
        match name {
            "param" => assert_eq!(s(&imp.self_ty), s(&param), "{}", name),
            _ => assert_eq!(
                s(&imp.self_ty),
                s(&common::projection_self(&written)),
                "{}",
                name
            ),
        }
    }

    // A bucket writes `Self` for an entry that needs it: the impl is for the parameter, though
    // `Rel<T>` names no `Self` (its generics do, `T: AsRef<Self>`).
    let tr: ItemTrait = parse_quote! {
        trait Tr<T: AsRef<Self>, U = Rel<T>> { fn f(&self, t: T) -> U; }
    };
    let mut leaker = Leaker::with_rule(Rules::new().prelude());
    let context = leaker.intern_trait(&tr);
    let result = leaker.result();
    let referrer = result.clone().finish().unwrap();
    let rel = entry_of(&referrer, "Rel < T >");
    assert_eq!(common::bucket_self(&referrer, 7, rel), None);
    let projection = Projection::new(
        repeater(),
        7,
        ImplSelf::Bucket {
            param: param.clone(),
        },
    );
    let impls = result.repeater_impls(&projection).unwrap();
    let completed = common::completed_args_for(
        &referrer,
        context,
        &tr.generics,
        &[parse_quote!(String)],
        &projection,
    )
    .unwrap();
    let key = projection.key(rel);
    let expected: Type = parse_quote!(<Self as crate::Repeater<(String,), #key>>::Type);
    assert_eq!(s(&completed[1]), s(&expected));
    assert_eq!(s(&impls[0].self_ty), s(&param));
}

// ── Projection::unused() ─────────────────────────────────────────────────────────────────────

/// A trait with no entry (`u8` is a primitive type), its context and its result.
fn without_entries() -> (ItemTrait, ContextId, LeakerResult) {
    let tr: ItemTrait = parse_quote! {
        trait Tr<'a, K: 'a, const N: usize = 3, D = u8> { fn get(&self, key: &'a K) -> [D; N]; }
    };
    let mut leaker = Leaker::with_rule(Rules::new().primitive());
    let context = leaker.intern_trait(&tr);
    let result = leaker.result();
    assert!(result.clone().finish().unwrap().is_empty());
    (tr, context, result)
}

/// A trait with an entry, its context and its result.
fn with_entry() -> (ItemTrait, ContextId, LeakerResult) {
    let tr: ItemTrait = parse_quote! {
        trait Tr<K, D = Wrap<K>> { fn get(&self, key: K) -> D; }
    };
    let mut leaker = Leaker::new();
    let context = leaker.intern_trait(&tr);
    let result = leaker.result();
    assert!(!result.clone().finish().unwrap().is_empty());
    (tr, context, result)
}

/// Issue 10 of announce: the Referrer of an item with no entry needs no Repeater, so a visit with
/// an unused projection leaves the copy as it is, as one with a real projection does.
#[test]
fn an_unused_projection_visits_a_copy_of_a_referrer_without_entries() {
    let (_, context, result) = without_entries();
    let referrer = result.finish().unwrap();
    let mut imp: ItemImpl = parse_quote! {
        impl<'a> Tr<'a, u8> for W {
            fn get(&self, key: &'a u8) -> [u8; 3] { let _ = Vec::<u8>::new(); todo!() }
        }
    };
    let before = imp.clone();
    referrer
        .visitor(context, &Projection::unused())
        .visit_item_impl_mut(&mut imp);
    assert_eq!(s(&imp), s(&before));
    let mut real = before.clone();
    referrer
        .visitor(context, &common::projection(repeater(), 3))
        .visit_item_impl_mut(&mut real);
    assert_eq!(s(&imp), s(&real));
}

/// Issue 10 of announce: `complete_args()` takes the arguments and fills in the defaults with an
/// unused projection as with a real one, and fails for the same arguments.
#[test]
fn an_unused_projection_completes_the_arguments_of_a_use_site() {
    let (tr, context, result) = without_entries();
    let referrer = result.finish().unwrap();
    let real = common::projection(repeater(), 3);
    for (case, given) in [
        ("the defaults", &["'x", "Foo"][..]),
        ("a default replaced", &["'x", "Foo", "7"][..]),
        ("every argument", &["'x", "Foo", "7", "char"][..]),
    ] {
        let given = arguments(given);
        let unused = common::completed_args_for(
            &referrer,
            context,
            &tr.generics,
            &given,
            &Projection::unused(),
        )
        .unwrap();
        let expected =
            common::completed_args_for(&referrer, context, &tr.generics, &given, &real).unwrap();
        assert_eq!(
            unused.iter().map(s).collect::<Vec<_>>(),
            expected.iter().map(s).collect::<Vec<_>>(),
            "{}",
            case
        );
    }
    // The defaults `3` and `u8` are in the signature.
    let given = arguments(&["'x", "Foo"]);
    let mut substitute = referrer
        .complete_args(context, &given, &Projection::unused())
        .unwrap();
    let mut sig: Signature = parse_quote!(fn get(&self, key: &'a K) -> [D; N]);
    substitute.visit_signature_mut(&mut sig);
    let expected: Signature = parse_quote!(fn get(&self, key: &'x Foo) -> [u8; { 3 }]);
    assert_eq!(s(&sig), s(&expected));
    // The errors are those of the arguments.
    let err = referrer
        .complete_args(context, &arguments(&["'x"]), &Projection::unused())
        .unwrap_err();
    assert!(matches!(err, EncodeError::MissingArgument(_)), "{}", err);
}

/// Issue 10 of announce: a result with no entry has no Repeater impl to write.
#[test]
fn an_unused_projection_writes_no_impl_for_a_result_without_entries() {
    let (_, _, result) = without_entries();
    let impls = result.repeater_impls(&Projection::unused()).unwrap();
    assert!(impls.is_empty());
}

/// Issue 10 of announce: the three methods that take a projection refuse an unused one for a
/// Referrer with an entry, before they look at anything else.
#[test]
fn an_unused_projection_is_refused_for_a_referrer_with_entries() {
    let (_, context, result) = with_entry();
    let referrer = result.clone().finish().unwrap();
    let unused = Projection::unused();
    let expected = "the Referrer has entries: pass the Projection of crate A, not \
                    Projection::unused()";
    let (_, other, _) = without_entries();
    for (case, message) in [
        (
            "visitor",
            panic_message(|| {
                let _ = referrer.visitor(context, &unused);
            }),
        ),
        (
            "complete_args",
            panic_message(|| {
                let _ = referrer.complete_args(context, &arguments(&["u8"]), &unused);
            }),
        ),
        (
            "repeater_impls",
            panic_message(|| {
                let _ = result.repeater_impls(&unused);
            }),
        ),
        (
            "complete_args, a context of another Referrer",
            panic_message(|| {
                let _ = referrer.complete_args(other, &[], &unused);
            }),
        ),
    ] {
        assert_eq!(message, expected, "{}", case);
    }
}

/// Issue 10 of announce: `of()` and `key()` write an entry, which an unused projection has no
/// Repeater for. It has the placeholders in its fields, and `new()` makes a used one.
#[test]
fn an_unused_projection_writes_no_entry() {
    let (_, _, result) = with_entry();
    let referrer = result.finish().unwrap();
    let (entry, _, context) = referrer.iter().next().unwrap();
    let unused = Projection::unused();
    let expected = "Projection::unused() has no Repeater to write an entry with: pass the \
                    Projection of crate A";
    let message = panic_message(|| {
        let _ = unused.of(&parse_quote!(Self), context, entry);
    });
    assert_eq!(message, expected, "of");
    let message = panic_message(|| {
        let _ = unused.key(entry);
    });
    assert_eq!(message, expected, "key");
    let placeholders = unused.clone();
    assert_eq!(s(&placeholders.path), "Repeater");
    assert_eq!(placeholders.id, 0);
    assert!(matches!(placeholders.impl_self, ImplSelf::Marker(_)));
    assert!(format!("{:?}", unused).starts_with("Projection"));
    // A used one has a key for it.
    let used = common::projection(parse_quote!(Repeater), 0);
    let _ = used.key(entry);
}

// ── the defaults of the parameters ───────────────────────────────────────────────────────────

fn referrer_of(tr: &ItemTrait) -> Referrer {
    let mut leaker = Leaker::new();
    leaker.intern_trait(tr);
    leaker.result().finish().unwrap()
}

/// The context of the trait that `referrer_of()` analyzed: the top-level one, numbered after the
/// contexts of the entries.
fn top_level(referrer: &Referrer) -> ContextId {
    common::context_ids(referrer).last().copied().unwrap()
}

fn parse_generics(tokens: proc_macro2::TokenStream) -> Generics {
    parse2(tokens).unwrap()
}

#[test]
fn test_default_requirements() {
    let tr: ItemTrait = parse_quote! {
        trait Tr<'a, T: ::core::clone::Clone, const N: usize = 3, U = Rel<T>, V = (RelB, U)>
        where
            T: ::p::Conv<RelW>,
            U: ::p::Late,
        {
            fn f(&self) -> RelC;
            fn g(&self) -> All<'a, T, U, V, N>;
        }
    };
    let mut leaker = Leaker::new();
    let top_level = leaker.intern_trait(&tr);
    let referrer = leaker.result().finish().unwrap();
    // The contexts are the requirements of the types, numbered in the order of their first
    // appearance: nothing (`RelB`, `RelW` and `RelC`), the generics of the trait, which `All`
    // uses exactly, and `Rel<T>` (the default of `U`, which uses the parameters before it); the
    // top-level context, which has the generics as declared, is the last, and none of the others.
    // The default of `V`, `(RelB, U)`, is not interned, as `RelB` is enough (it is rebuilt around
    // it), so its context (`<.., U: ::p::Late>`, the number 3) is not in the Referrer: the
    // top-level context is the number 4. The default of `N` is a const.
    assert_eq!(common::context_ids(&referrer).len(), 4);
    assert_eq!(top_level.index(), 4);
    let leaked: Vec<_> = common::types(&referrer)
        .enumerate()
        .map(|(ix, ty)| (common::entry_id(&referrer, ix).index(), s(ty)))
        .collect();
    assert_eq!(
        leaked,
        [
            (0, "RelB".to_string()),
            (0, "RelC".to_string()),
            (0, "RelW".to_string()),
            (1, "All < 'a , T , U , V , N >".to_string()),
            (2, "Rel < T >".to_string()),
        ]
    );
    // The generics of the context of the entry `ix`.
    let generics = |ix: usize| {
        let generics = &common::entry(&referrer, ix).generics;
        s(&quote!(#generics #{&generics.where_clause}))
    };
    assert_eq!(generics(0), "");
    // The bounds of `T` (also in the where clause) and the predicates mentioning it are used.
    assert_eq!(
        generics(4),
        s(&parse_generics(
            quote!(<T: ::core::clone::Clone + ::p::Conv<RelW>>)
        ))
    );
    // The defaults are removed, and the bounds are moved to the parameters.
    assert_ne!(common::entry_id(&referrer, 3), top_level);
    assert_eq!(common::entry_id(&referrer, 3).index(), 1);
    assert_eq!(
        generics(3),
        s(&parse_generics(quote!(
            <'a, T: ::core::clone::Clone + ::p::Conv<RelW>, U: ::p::Late, V, const N: usize>
        )))
    );
    assert!(referrer
        .iter()
        .all(|(_, _, context)| context.self_ty.is_none()));

    // The round trip keeps the requirements.
    let tokens = quote!(#referrer);
    let parsed: Referrer = parse2(tokens).unwrap();
    assert_eq!(parsed, referrer);
    for ((_, _, read), (_, _, context)) in parsed.iter().zip(referrer.iter()) {
        assert_eq!(s(&read.generics), s(&context.generics));
        assert_eq!(
            s(&read.generics.where_clause),
            s(&context.generics.where_clause)
        );
    }
}

/// `complete_args()`: a default is analyzed with the parameters before it, and cannot
/// name a later one. A where predicate that names the later parameter `U` (`T: Conv<U>`) is not
/// in the context of the default `Rel<T>`, which has `T` alone. Audit mutant
/// `default_root_all_params` (the default is analyzed with all the parameters as its root).
#[test]
fn test_default_is_analyzed_with_the_parameters_before_it() {
    let tr: ItemTrait = parse_quote! {
        trait Tr<T, U = Rel<T>> where T: ::p::Conv<U> { fn f(&self); }
    };
    let referrer = referrer_of(&tr);
    assert_eq!(
        referrer
            .iter()
            .map(|(_, ty, context)| (ty, &context.generics))
            .map(|(ty, generics)| (s(ty), s(&quote!(#generics #{&generics.where_clause}))))
            .collect::<Vec<_>>(),
        [("Rel < T >".to_string(), "< T >".to_string())]
    );
    // The predicate is on `U`, a parameter the default of `V` has, and names `V` itself, which it
    // has not: it is left out of the context of the default.
    let tr: ItemTrait = parse_quote! {
        trait Tr<T, U = Rel<T>, V = Rel2<U>> where U: ::p::Conv<V> { fn f(&self); }
    };
    let referrer = referrer_of(&tr);
    assert_eq!(
        referrer
            .iter()
            .map(|(_, ty, context)| (ty, &context.generics))
            .map(|(ty, generics)| (s(ty), s(&quote!(#generics #{&generics.where_clause}))))
            .collect::<Vec<_>>(),
        [
            ("Rel < T >".to_string(), "< T >".to_string()),
            ("Rel2 < U >".to_string(), "< U >".to_string()),
        ]
    );
}

#[test]
fn test_default_is_shared_with_the_items() {
    // `Rel<T>` in the default of `U` and in a method is the same type with the same requirements.
    let tr: ItemTrait = parse_quote!(
        trait Tr<T, U = Rel<T>> {
            fn f(&self, t: T) -> Rel<T>;
            fn g(&self, t: T, u: U) -> Both<T, U>;
        }
    );
    let mut leaker = Leaker::new();
    let top_level = leaker.intern_trait(&tr);
    let referrer = leaker.result().finish().unwrap();
    assert_eq!(referrer.iter().count(), 2);
    let at = |text: &str| {
        common::types(&referrer)
            .position(|ty| s(ty) == text)
            .unwrap()
    };
    let (rel, both) = (at("Rel < T >"), at("Both < T , U >"));
    // `Rel<T>` uses `T` only, whereas the generics of the trait are `<T, U>`, which `Both` uses:
    // two contexts of entries, and the top-level context after them, which is neither.
    assert_eq!(common::context_ids(&referrer).len(), 3);
    assert_ne!(common::entry_id(&referrer, rel), top_level);
    assert_ne!(common::entry_id(&referrer, both), top_level);
    assert_ne!(
        common::entry_id(&referrer, both),
        common::entry_id(&referrer, rel)
    );
    assert_eq!(common::entry(&referrer, both).generics.params.len(), 2);
    assert_eq!(common::entry(&referrer, rel).generics.params.len(), 1);
}

#[test]
fn test_defaults_of_several_calls_are_shared() {
    let tr: ItemTrait = parse_quote!(
        trait Tr<T, U = ::std::vec::Vec<T>> {
            fn f(&self) -> Rel;
        }
    );
    let plain: ItemStruct = parse_quote!(
        struct S<T> {
            a: RelA,
        }
    );
    let mut leaker = Leaker::new();
    let top_plain = leaker.intern_struct(&plain);
    let top_tr = leaker.intern_trait(&tr);
    // A failed call adds no entry (its error is the only one).
    let bad: ItemTrait = parse_quote!(
        trait Tr<T, U = ::std::vec::Vec<T>> {
            fn f(&self) -> Rel<'_>;
        }
    );
    leaker.intern_trait(&bad);
    let result = leaker.result();
    assert_eq!(result.errors().len(), 1);
    let referrer = result.recovered();
    // The types use no generics, so they are in one context, whichever call found them. The
    // default of `U`, the type of an external crate, is an optional entry (after the required
    // ones), which uses `T`: it is in the context `<T>` that the generics of `S<T>` have too. The
    // top-level contexts, which have the generics as declared, are two others: that of `S<T>`,
    // and that of the trait, which the call that failed has too.
    assert_eq!(referrer.iter().count(), 3);
    assert_eq!(common::context_ids(&referrer).len(), 4);
    assert_eq!(common::entry(&referrer, 0).generics.params.len(), 0);
    assert_eq!(common::entry(&referrer, 1).generics.params.len(), 0);
    assert_eq!(
        s(common::types(&referrer).nth(2).unwrap()),
        ":: std :: vec :: Vec < T >"
    );
    assert_ne!(common::entry_id(&referrer, 2), top_plain);
    assert_eq!(common::entry(&referrer, 2).generics.params.len(), 1);
    assert_ne!(top_tr, top_plain);
    // The default is replaced by the entry, with the argument for `T` in its own generics (an
    // optional entry is replaced too).
    let completed =
        common::completed_args(&referrer, top_tr, &tr.generics, &arguments(&["u8"])).unwrap();
    let key = common::key(0, common::entry_id_at(&referrer, 2));
    let projection: Type = parse_quote!(<Self as Repeater<(u8,), #key>>::Type);
    assert_eq!(
        completed.iter().map(s).collect::<Vec<_>>(),
        ["u8".to_string(), s(&projection)]
    );
}

#[test]
fn test_default_const_is_kept() {
    // A relative path in the default of a const parameter is reported, as before.
    let tr: ItemTrait = parse_quote!(
        trait Tr<T, const N: usize = MY_N> {}
    );
    let mut leaker = Leaker::new();
    leaker.intern_trait(&tr);
    let result = leaker.result();
    assert!(result.finish().is_err());
    let tr: ItemTrait = parse_quote!(
        trait Tr<T, const N: usize = 3, const M: usize = { 1 + 2 }> {}
    );
    let referrer = referrer_of(&tr);
    let completed = completed(&referrer, top_level(&referrer), &tr.generics, &["u8"]).unwrap();
    assert_eq!(completed, ["u8", "3", "{ 1 + 2 }"]);
}

#[test]
fn test_complete_args_substitutes_the_copied_part_only() {
    // A leaked type inside a type which is copied (it cannot be leaked as a whole), twice.
    let tr: ItemTrait = parse_quote!(
        trait Tr<T, U = ::p::W<Rel<T>, &'_ T, Rel<T>>> {}
    );
    let referrer = referrer_of(&tr);
    let key = common::key(0, common::entry_id_at(&referrer, 0));
    let projection = |t: Type| {
        let ty: Type = parse_quote!(<Self as Repeater<(#t,), #key>>::Type);
        s(&ty)
    };
    let completed = common::completed_args(
        &referrer,
        top_level(&referrer),
        &tr.generics,
        &arguments(&["Rel<u8>"]),
    )
    .unwrap();
    // `T` is substituted in the copied part and in the projections alike.
    let projected = projection(parse_quote!(Rel<u8>));
    assert_eq!(
        completed.iter().map(s).collect::<Vec<_>>(),
        [
            "Rel < u8 >".to_string(),
            format!(
                ":: p :: W < {} , & '_ Rel < u8 > , {} >",
                projected, projected
            )
        ]
    );
    let completed = common::completed_args(
        &referrer,
        top_level(&referrer),
        &tr.generics,
        &arguments(&["u8"]),
    )
    .unwrap();
    let projected = projection(parse_quote!(u8));
    assert_eq!(
        s(&completed[1]),
        format!(":: p :: W < {} , & '_ u8 , {} >", projected, projected)
    );
}

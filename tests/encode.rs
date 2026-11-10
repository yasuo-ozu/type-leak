//! Tests for the Repeater definition and its const slots, for the errors
//! of the encoders and of the helpers that encode (`EncodeError`): their text, the conversion to
//! `syn::Error`, and the `std::error::Error` impls, and for the encoding of the generics of an
//! entry (the last section): `Projection::of()`, and `Referrer::complete_args()` and
//! `substitute_args()` for a site. The spans of the errors are checked in `tests/crates/spans`,
//! which `tests/diagnostic.rs` runs: they need a feature of `proc-macro2` that the tests of this
//! package cannot have.

mod common;

use common::s;

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
    let private: ItemTrait = parse_quote! {
        trait Other<
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

/// The slots refuse a const parameter of an alias type, and encode the consts of an entry for
/// `Projection::of()` and `LeakerResult::repeater_impls()`.
#[test]
fn the_slots_refuse_an_alias_and_encode_the_consts_of_an_entry() {
    assert!(matches!(
        common::slot(&repeater(), &parse_quote!(A), &parse_quote!(Alias)),
        Err(EncodeError::UnsupportedConstType(_))
    ));

    // In the helpers of the Referrer.
    let tr: ItemTrait = parse_quote!(
        trait Tr<const N: usize, const B: bool> {
            fn f(&self) -> Wrap<N, B>;
        }
    );
    let mut leaker = Leaker::new();
    leaker.intern_trait(&tr);
    let referrer = leaker.result().finish().unwrap();
    let ix = common::entry_id_at(&referrer, 0);
    let args = common::repeater_args_at(
        &repeater(),
        common::entry(&referrer, ix.index()),
        common::identity,
        None,
    )
    .unwrap();
    assert_eq!(
        args,
        parse_quote!((*const dyn crate::Repeater<(), 0, B, Type = ()>, [(); N]))
    );
}

/// The value that a site gives for a const goes into its slot braced unless it is a literal-like
/// token (`FOO`, a block of one identifier is unwrapped) so that it parses as a const argument.
#[test]
fn a_const_a_site_gives_is_braced_in_its_slot_when_needed() {
    let to_string = |ty: &Type| quote!(#ty).to_string();
    let repeater: Path = parse_quote!(crate::Repeater);

    // A multi-segment path (written with or without braces) must be braced to be parsed as a
    // const argument; a literal or a single identifier is not.
    let i32_ty: Type = parse_quote!(i32);
    for (value, expected) in [
        (quote!(i32::MIN), quote!({ i32::MIN })),
        (quote!({ i32::MIN }), quote!({ i32::MIN })),
        (quote!({ m::FOO }), quote!({ m::FOO })),
        (quote!({ FOO }), quote!(FOO)),
        (quote!(-3), quote!({ -3 })),
    ] {
        let ty = common::slot(&repeater, &parse2(value).unwrap(), &i32_ty).unwrap();
        let expected: Type = parse_quote! {
            *const dyn crate::Repeater<(), 0, false, '\0', 0, 0, 0, 0, 0, 0, #expected, Type = ()>
        };
        assert_eq!(to_string(&ty), to_string(&expected));
    }
    let u128_ty: Type = parse_quote!(u128);
    let ty = common::slot(&repeater, &parse_quote!({ m::FOO }), &u128_ty).unwrap();
    let expected: Type = parse_quote!(*const dyn crate::Repeater<(), { m::FOO }, Type = ()>);
    assert_eq!(to_string(&ty), to_string(&expected));
}

/// A `usize` const is an array length, with the braces that a value a site gives needs.
#[test]
fn a_usize_const_a_site_gives_is_an_array_length() {
    let to_string = |ty: &Type| quote!(#ty).to_string();
    let repeater: Path = parse_quote!(crate::Repeater);
    let usize_ty: Type = parse_quote!(usize);
    for (value, expected) in [
        (quote!(3), quote!([(); { 3 }])),
        (quote!(N), quote!([(); { N }])),
        (quote!(m::FOO), quote!([(); { m::FOO }])),
        (quote!({ FOO }), quote!([(); { FOO }])),
    ] {
        let ty = common::slot(&repeater, &parse2(value).unwrap(), &usize_ty).unwrap();
        assert_eq!(to_string(&ty), expected.to_string());
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

fn repeater() -> Path {
    parse_quote!(crate::Repeater)
}

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
            "the type 'Wrap < T , Self >' needs `Self` (it names it, or its generics do), so its \
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
    // `ErrorItem`, one of the errors of the Leaker (it returns a `Vec` of them).
    let item: ItemTrait = parse_quote!(
        trait Tr: Clone {}
    );
    let mut leaker = Leaker::new();
    leaker.intern_trait(&item);
    let result = leaker.result();
    let err = result.errors();
    let err = err[0].clone();
    is_error(&err);
    let boxed: Box<dyn std::error::Error> = Box::new(err.clone());
    assert_eq!(boxed.to_string(), "use absolute path: 'Clone'");
    assert!(boxed.source().is_none());
    let result: std::result::Result<(), Box<dyn std::error::Error>> = (|| Err(err.clone())?)();
    assert_eq!(
        result.unwrap_err().to_string(),
        "use absolute path: 'Clone'"
    );
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

/// The arguments of `texts`, parsed.
fn arguments(texts: &[&str]) -> Vec<GenericArgument> {
    texts.iter().map(|text| parse_str(text).unwrap()).collect()
}

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
            "[u8; a::B]",
        ]),
        ["T", "p", "X", "u8"]
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

/// The arguments and the items written by the `Substitute` of `complete_args()` are what rustc
/// takes for the omitted default, and a leaked type in it is written through the Repeater.
#[test]
fn complete_args_writes_a_leaked_type_of_a_default_through_the_repeater() {
    let tr: ItemTrait = parse_quote! {
        trait Store<'a, K: 'a, const N: usize, D = Page<K, N>> {
            fn get(&'a self, key: &'a K, n: [u8; N]) -> D;
        }
    };
    let mut leaker = Leaker::new();
    let context = leaker.intern_trait(&tr);
    let referrer = leaker.result().finish().unwrap();
    let repeater = repeater();
    let given = arguments(&["'static", "u8", "4"]);
    let mut visitor = referrer
        .complete_args(
            context,
            &given,
            &Projection::new(repeater.clone(), 5, ImplSelf::Param(parse_quote!(__S))),
        )
        .unwrap();
    let mut sig: Signature = parse_quote!(fn get(&'a self, key: &'a K, n: [u8; N]) -> D);
    visitor.visit_signature_mut(&mut sig);
    let page = common::types(&referrer)
        .position(|ty| s(ty) == "Page < K , N >")
        .unwrap();
    let key = common::key(5, common::entry_id_at(&referrer, page));
    let expected: Signature = parse_quote!(
        fn get(&'static self, key: &'static u8, n: [u8; { 4 }])
            -> <Self as crate::Repeater<(&'static (), u8, [(); { 4 }], &'static u8), #key>>::Type
    );
    assert_eq!(s(&sig), s(&expected));
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
#[should_panic(expected = "names the trait without arguments")]
fn complete_args_panics_for_a_repeater_with_arguments() {
    let mut leaker = Leaker::new();
    let context = leaker.intern_struct(&parse_quote!(
        struct S<T>(Rel<T>);
    ));
    let referrer = leaker.result().finish().unwrap();
    let repeater: Path = parse_quote!(crate::Repeater<u8>);
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
    // generics.
    let projection = common::projection(repeater.clone(), 7)
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
    // A const that has no slot is the error of the call.
    let alias: ItemTrait = parse_quote! {
        trait Tr<const A: ::my_crate::Idx> { fn f(&self) -> Wrap<A>; }
    };
    let mut leaker = Leaker::new();
    leaker.intern_trait(&alias);
    let referrer = leaker.result().finish().unwrap();
    let (ix, _, context) = referrer.iter().next().unwrap();
    let err = common::projection(repeater, 7)
        .of(&host, context, ix)
        .unwrap_err();
    assert!(
        matches!(err, EncodeError::UnsupportedConstType(_)),
        "{}",
        err
    );
}

// ── the encoding of the generics of an entry ─────────────────────────────────────────────────────
//
// The Leaker analyzes `trait Tr<generics> { fn f(&self) -> All<..>; }`, whose one entry names every
// generic, so the `Context` of the entry has them (`using()`), and `Projection::of()` encodes
// them by their own names. A site writes the arguments it gives with `complete_args()`, which
// takes the defaults, and `substitute_args()` (`at_site()`).

/// A type written as it is printed, to compare with an encoding.
fn t(ty: Type) -> String {
    s(&ty)
}

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

/// The slot gets the const of the entry by its name, and the argument of a site takes its place in
/// it: braced where it needs braces, as it is where it does not.
#[test]
fn the_consts_a_site_gives_are_braced_in_the_slots() {
    let written: Generics = parse_quote!(<const A: i32, const B: i32>);
    let referrer = using(&written, &[]);
    let slot = |value: proc_macro2::TokenStream| -> Type {
        parse_quote!(*const dyn crate::Repeater<(), 0, false, '\0', 0, 0, 0, 0, 0, 0, #value, Type = ()>)
    };
    let own = args_of(&referrer, &parse_quote!(Self)).unwrap();
    let (a, b) = (slot(quote!(A)), slot(quote!(B)));
    assert_eq!(s(&own), t(parse_quote!((#a, #b))));
    let args = given("<{ i32::MIN }, { N }>");
    let ty = at_site(&referrer, &written, &args, None).unwrap();
    let (a, b) = (slot(quote!({ i32::MIN })), slot(quote!(N)));
    assert_eq!(s(&ty), t(parse_quote!((#a, #b))));
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

/// The Repeater impl of the implementor of a trait: the parameters, the bounds and then `Self: 'a`
/// as `&'a __S`. The encoding of the entry for its own `Self`, for the arguments of a site, and for
/// an entry that does not name `Self`.
#[test]
fn self_outlives_of_a_method_is_encoded_after_the_bounds_the_method_declares() {
    let item: ItemTrait = parse_quote! {
        trait Tr { fn f<'a, T>(&self, x: &'a Rel<(Self, T)>); }
    };
    let mut leaker = Leaker::new();
    leaker.intern_trait(&item);
    let referrer = leaker.result().finish().unwrap();
    let context = common::entry(&referrer, 0);
    let generics = &context.generics;
    assert_eq!(s(generics), "< 'a , T : 'a >");
    assert_eq!(
        context.self_outlives.iter().map(s).collect::<Vec<_>>(),
        ["'a"]
    );
    // The implementor `__S`: the parameters, the bounds and then `&'a __S`.
    let marker: Type = parse_quote!(__S);
    let ty = args_of(&referrer, &marker).unwrap();
    assert_eq!(s(&ty), t(parse_quote!((&'a (), T, &'a T, &'a __S))));
    // Without it, the encoding is that of the generics alone: an entry with the same generics that
    // does not name `Self`, or no lifetime to outlive.
    let alone = t(parse_quote!((&'a (), T, &'a T)));
    let ty = args_of(&using(generics, &[]), &marker).unwrap();
    assert_eq!(s(&ty), alone);
    let (ix, context) = with_outlives(&referrer, &[]);
    let ty = common::projection(repeater(), 7)
        .of(&marker, &context, ix)
        .unwrap();
    assert_eq!(s(&common::projection_args(&ty)), alone);
    // A lifetime which is not a parameter is skipped, like a bound by one.
    let (ix, context) = with_outlives(&referrer, &["'z"]);
    let ty = common::projection(repeater(), 7)
        .of(&marker, &context, ix)
        .unwrap();
    assert_eq!(s(&common::projection_args(&ty)), alone);
    // At the use site, with the arguments substituted, `Self` being the implementor.
    for (args, expected) in [
        ("<'q, u8>", t(parse_quote!((&'q (), u8, &'q u8, &'q Self)))),
        (
            "<'static, u8>",
            t(parse_quote!((&'static (), u8, &'static u8, &'static Self))),
        ),
    ] {
        let ty = at_site(&referrer, generics, &given(args), None).unwrap();
        assert_eq!(s(&ty), expected, "{}", args);
    }
    // The type that `Self: 'a` names is the one given: the host of a site.
    let host: Type = parse_quote!(Host);
    let ty = at_site(&referrer, generics, &given("<'static, u8>"), Some(&host)).unwrap();
    assert_eq!(
        s(&ty),
        t(parse_quote!((&'static (), u8, &'static u8, &'static Host)))
    );
}

/// The arguments of a site with `usize` consts only: `usize` is an array length, the elements after the parameters are the bounds, a `?Sized` parameter is a pointer, and
/// the arguments are those of the parameters when they are the parameters themselves.
#[test]
fn the_arguments_of_a_site_are_encoded_as_the_parameters_for_usize_consts() {
    let written: Generics =
        parse_quote!(<'a, 'b: 'a, T: 'a + ::core::clone::Clone, const N: usize, U: ?Sized>);
    let referrer = using(&written, &[]);
    let args = given("<'x, 'y, u8, { 1 + 2 }, str>");
    let ty = at_site(&referrer, &written, &args, None).unwrap();
    assert_eq!(
        s(&ty),
        t(parse_quote!((
            &'x (),
            &'y (),
            u8,
            *const str,
            [(); { 1 + 2 }],
            &'x &'y (),
            &'x u8
        )))
    );
    // For the parameters themselves, it is the encoding of the entry.
    let own: Vec<GenericArgument> = written.params.iter().map(common::own_name_arg).collect();
    let by_name = at_site(&referrer, &written, &own, None).unwrap();
    let params = args_of(&referrer, &parse_quote!(Self)).unwrap();
    assert_eq!(s(&by_name).replace("{ N }", "N"), s(&params));
    // Too few and too many arguments.
    assert!(matches!(
        at_site(&referrer, &written, &args[..4], None),
        Err(EncodeError::MissingArgument(_))
    ));
    let too_many: Vec<GenericArgument> = args.iter().chain(&args[..1]).cloned().collect();
    assert!(matches!(
        at_site(&referrer, &written, &too_many, None),
        Err(EncodeError::UnexpectedArgument(_))
    ));
}

/// A `?Sized` type parameter is a pointer to it, not a bare element, which is `Sized` whatever the
/// parameter is: for the generics of an entry wherever it is (first, in the middle, last, the only
/// one, with other bounds), and for the arguments of a site, which are any type (F1: see
/// `f01_unsized_type_param_encoding` in `tests/findings.rs` for the generics in the entries of items).
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

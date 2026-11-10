//! Tests for the Repeater definition, `slot_const()` and `array_const()`, for the errors
//! of the encoders and of the helpers that encode (`EncodeError`): their text, the conversion to
//! `syn::Error`, and the `std::error::Error` impls, and for the encoding of the generics of an
//! entry (the last section): `repeater_projection()`, and `complete_params()` and
//! `substitute_args()` for a site. The spans of the errors are checked in `tests/crates/spans`,
//! which `tests/diagnostic.rs` runs: they need a feature of `proc-macro2` that the tests of this
//! package cannot have.

mod common;

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
        parse_quote!([(); 5])
    );
}

/// `slot_const()` refuses a const parameter of an alias type, and it is the `const_fn` of the
/// encoders, as `repeater_projection()` and `LeakerResult::repeater_impls()` take it.
#[test]
fn slot_const_refuses_an_alias_and_encodes_the_consts_of_an_entry() {
    let repeater: Path = parse_quote!(crate::Repeater);
    let mut const_fn = slot_const(&repeater);
    let alias: ConstParam = parse_quote!(const A: Alias);
    assert!(matches!(
        const_fn(&parse_quote!(A), &alias),
        Err(EncodeError::UnsupportedConstType(_))
    ));

    // In the helpers of the Referrer, as the closure of `array_const` is.
    let tr: ItemTrait = parse_quote!(
        trait Tr<const N: usize, const B: bool> {
            fn f(&self) -> Wrap<N, B>;
        }
    );
    let mut leaker = Leaker::new();
    leaker.intern_trait(&tr);
    let referrer = leaker.result().finish().unwrap();
    let ix = common::entry_id_at(&referrer, 0);
    let args = common::repeater_args(
        common::entry(&referrer, ix.index()),
        common::identity,
        None,
        slot_const(&repeater),
    )
    .unwrap();
    assert_eq!(
        args,
        parse_quote!((*const dyn crate::Repeater<(), 0, B, Type = ()>, [(); N]))
    );
}

#[test]
fn test_const_args_are_braced_when_needed() {
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
        (quote!(-3), quote!(-3)),
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

/// `array_const()` encodes a `usize` const as an array length, with the braces that a value needs,
/// and any other type is an error (`slot_const()` encodes it).
#[test]
fn array_const_encodes_usize_consts_only() {
    let to_string = |ty: &Type| quote!(#ty).to_string();
    let usize_param: ConstParam = parse_quote!(const N: usize);
    for (value, expected) in [
        (quote!(3), quote!([(); 3])),
        (quote!(N), quote!([(); N])),
        (quote!(m::FOO), quote!([(); { m::FOO }])),
        (quote!({ FOO }), quote!([(); FOO])),
    ] {
        let ty = array_const(&parse2(value).unwrap(), &usize_param).unwrap();
        assert_eq!(to_string(&ty), expected.to_string());
    }
    let bool_param: ConstParam = parse_quote!(const B: bool);
    match array_const(&parse_quote!(true), &bool_param) {
        Err(EncodeError::UnsupportedConstType(ty)) => assert_eq!(*ty, parse_quote!(bool)),
        other => panic!("{:?}", other),
    }
    // The slots encode it.
    let repeater: Path = parse_quote!(crate::Repeater);
    assert!(common::slot(&repeater, &parse_quote!(true), &bool_param.ty).is_ok());
}

/// `repeater_key()` puts the id in the high half and the index in the low half, so two entries
/// never share a key, and it is part of the format (the impls and the projections must agree).
#[test]
fn the_repeater_key_has_the_id_above_the_index() {
    use std::collections::HashSet;
    use type_leak::repeater_key;
    // The `EntryId` with the number `n`, which only `Parse` makes.
    let entry = |n: usize| -> EntryId { parse_quote!(#n) };
    assert_eq!(repeater_key(0, entry(0)), 0);
    assert_eq!(repeater_key(0, entry(5)), 5);
    assert_eq!(repeater_key(1, entry(0)), 1 << 64);
    assert_eq!(repeater_key(7, entry(3)), (7 << 64) | 3);
    assert_eq!(repeater_key(u64::MAX, entry(0)), u128::from(u64::MAX) << 64);
    assert_eq!(
        repeater_key(0, entry(u64::MAX as usize)),
        u128::from(u64::MAX)
    );
    assert_eq!(repeater_key(u64::MAX, entry(u64::MAX as usize)), u128::MAX);
    let ids = [0, 1, 2, 255, 1 << 32, u64::MAX - 1, u64::MAX];
    let nths = [0, 1, 2, 255, 1 << 31, u32::MAX as usize];
    let keys: HashSet<u128> = ids
        .iter()
        .flat_map(|&id| nths.iter().map(move |&nth| repeater_key(id, entry(nth))))
        .collect();
    assert_eq!(keys.len(), ids.len() * nths.len());
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
    let completed = match complete_params(&generics.params, &args.args) {
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
        slot_const(&repeater()),
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
    repeater_projection(
        &repeater(),
        &self_ty,
        entry.2,
        7,
        entry.0,
        slot_const(&repeater()),
    )
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
        .repeater_impls(
            &parse_quote!(crate::Repeater),
            7,
            ImplSelf::Marker(&marker),
            array_const,
        )
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
    ]
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
        ]
    );
    assert!(matches!(all[0], EncodeError::UnsupportedConstType(_)));
    assert!(matches!(all[1], EncodeError::UnexpectedArgument(_)));
    assert!(matches!(all[2], EncodeError::MissingArgument(_)));
    assert!(matches!(all[3], EncodeError::UnboundedTraitObject(_)));
    assert!(matches!(all[4], EncodeError::NeedsSelf(_)));
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
        ("T::Item<N>", "<Vec<u8>>::Item<3>"),
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

// ── repeater_projection() ────────────────────────────────────────────────────────────────────

#[test]
fn repeater_projection_writes_the_generics_of_the_entry_by_their_names() {
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
    let key = repeater_key(7, ix);
    let host: Type = parse_quote!(Host);
    // Lifetimes, an unsized type, the consts (by name) in the slots, and the bounds of the
    // generics.
    let projection =
        repeater_projection(&repeater, &host, context, 7, ix, slot_const(&repeater)).unwrap();
    let expected: Type = parse_quote!(<Host as crate::Repeater<(
        &'a (),
        *const T,
        *const dyn crate::Repeater<(), 0, B, Type = ()>,
        [(); N],
        &'a T
    ), #key>>::Type);
    assert_eq!(projection, expected);
    // The error of `const_fn` is the error of the call: `array_const` has no `bool`.
    let err = repeater_projection(&repeater, &host, context, 7, ix, array_const).unwrap_err();
    assert!(
        matches!(err, EncodeError::UnsupportedConstType(_)),
        "{}",
        err
    );
}

// ── the encoding of the generics of an entry ─────────────────────────────────────────────────────
//
// The Leaker analyzes `trait Tr<generics> { fn f(&self) -> All<..>; }`, whose one entry names every
// generic, so the `Context` of the entry has them (`using()`), and `repeater_projection()` encodes
// them by their own names. A site writes the arguments it gives with `complete_params()`, which
// takes the defaults, and `substitute_args()` (`at_site()`).

fn s<T: ToTokens>(t: &T) -> String {
    quote!(#t).to_string()
}

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

/// The first argument of the Repeater that `repeater_projection()` writes for the only entry of
/// `referrer`, `ARGS` of `<self_ty as Repeater<ARGS, KEY>>::Type`: the generics of its context, by
/// their own names.
fn args_of(
    referrer: &Referrer,
    self_ty: &Type,
    const_fn: impl FnMut(&Expr, &ConstParam) -> std::result::Result<Type, EncodeError>,
) -> std::result::Result<Type, EncodeError> {
    let (ix, _, context) = referrer.iter().next().unwrap();
    let projection = repeater_projection(&repeater(), self_ty, context, 7, ix, const_fn)?;
    Ok(common::projection_args(&projection))
}

/// The arguments written as `<..>` at a site, with the bindings.
fn given(args: &str) -> Vec<GenericArgument> {
    let args: AngleBracketedGenericArguments = parse_str(args).unwrap();
    args.args.into_iter().collect()
}

/// The same at a site that gives `args` for the parameters `written` of the item: the arguments are
/// completed (`complete_params()`), and take the place of the names (`substitute_args()`).
/// `self_ty` is what `Self: 'a` names (`None`: `Self`).
fn at_site(
    referrer: &Referrer,
    written: &Generics,
    args: &[GenericArgument],
    self_ty: Option<&Type>,
    const_fn: impl FnMut(&Expr, &ConstParam) -> std::result::Result<Type, EncodeError>,
) -> std::result::Result<Type, EncodeError> {
    let completed = complete_params(&written.params, args)?;
    common::repeater_args(
        common::entry(referrer, 0),
        common::answers(&written.params, &completed),
        self_ty,
        const_fn,
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
/// const that it gives in braces (`[(); { 3 }]`), as `substitute_args()` does. `array_const` is no
/// encoding of a `bool`: the slots are.
#[test]
fn the_slots_encode_the_consts_of_the_generics_of_an_entry_and_of_a_site() {
    let repeater = repeater();
    // The trait as written: `M` has the default `N`.
    let written: Generics = parse_quote! {
        <'a, T, const N: usize, const X: u128, const B: bool, const I: i32, const M: usize = N>
    };
    let referrer = using(&written, &[]);
    let encoded = args_of(&referrer, &parse_quote!(Self), slot_const(&repeater)).unwrap();
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
    let ty = at_site(&referrer, &written, &args, None, slot_const(&repeater)).unwrap();
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
    let ty = at_site(&referrer, &written, &args, None, slot_const(&repeater)).unwrap();
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
        let err = at_site(
            &referrer,
            &written,
            &given(args),
            None,
            slot_const(&repeater),
        );
        assert_eq!(err.unwrap_err().to_string(), message, "{}", args);
    }
    // `array_const` and a `bool`: only the slots encode it (an alias has none:
    // `the_text_of_every_variant`).
    let flag: Generics = parse_quote!(<const B: bool>);
    let flag = using(&flag, &[]);
    let err = args_of(&flag, &parse_quote!(Self), array_const).unwrap_err();
    assert!(
        matches!(err, EncodeError::UnsupportedConstType(_)),
        "{}",
        err
    );
    let ty = args_of(&flag, &parse_quote!(Self), slot_const(&repeater)).unwrap();
    assert_eq!(
        s(&ty),
        t(parse_quote!((
            *const dyn crate::Repeater<(), 0, B, Type = ()>,
        )))
    );

    // No generics, and a single generic (a one-element tuple).
    let none = using(&Generics::default(), &[]);
    let ty = args_of(&none, &parse_quote!(Self), slot_const(&repeater)).unwrap();
    assert_eq!(s(&ty), t(parse_quote!(())));
    let single: Generics = parse_quote!(<T>);
    let ty = args_of(
        &using(&single, &[]),
        &parse_quote!(Self),
        slot_const(&repeater),
    )
    .unwrap();
    assert_eq!(s(&ty), t(parse_quote!((T,))));
}

/// A default that names a const as a type argument (`IntoIter<u8, N>`: syn parses the `N` as a type)
/// takes the value of the const when it is completed.
#[test]
fn a_default_that_names_a_const_as_a_type_argument_takes_its_value() {
    let written: Generics = parse_quote!(<const N: usize, U = ::core::array::IntoIter<u8, N>>);
    let args = given("<3>");
    let completed = complete_params(&written.params, &args).unwrap();
    let expected: Vec<GenericArgument> = vec![
        parse_quote!(3),
        parse_quote!(::core::array::IntoIter<u8, 3>),
    ];
    assert_eq!(completed, expected);
    // The encoding at the site: the types, then the const, as the context has them.
    let ty = at_site(&using(&written, &[]), &written, &args, None, array_const).unwrap();
    assert_eq!(
        s(&ty),
        t(parse_quote!((::core::array::IntoIter<u8, 3>, [(); { 3 }])))
    );
}

/// The arguments that a site gives are completed for the generics as written (the defaults are
/// there), and written for the names in the encoding of the generics of the entry. A bare `N` is a
/// const, bindings are skipped, and too many or too few arguments are the errors of
/// `complete_params()`. The context sorts the generics, so the arguments go by name.
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
        let encoded = at_site(&referrer, &written, &given(args), None, array_const);
        match (encoded, expected) {
            (Ok(ty), Ok(expected)) => assert_eq!(s(&ty), s(&expected), "{}", case),
            (Err(err), Err(expected)) => assert_eq!(err.to_string(), expected, "{}", case),
            (encoded, expected) => panic!("{}: {:?} against {:?}", case, encoded, expected),
        }
    }
}

/// A lifetime bound `T: 'a` is `&'a T` and `'b: 'a` is `&'a &'b ()`, after the elements of the
/// parameters, in the order of the parameters and of their bounds. A bound by `'static`, and one by
/// a lifetime that is not a parameter, is not encoded. The const parameters are encoded by the
/// closure, which gets the parameters only.
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
    let repeater = repeater();
    for (generics, expected) in rows {
        let referrer = using(&generics, &[]);
        let ty = args_of(&referrer, &parse_quote!(Self), array_const).unwrap();
        assert_eq!(s(&ty), s(&expected), "{}", s(&generics));
        // The slots encode the same, as `usize` is `[(); N]` for both.
        let ty = args_of(&referrer, &parse_quote!(Self), slot_const(&repeater)).unwrap();
        assert_eq!(s(&ty), s(&expected), "{}", s(&generics));
    }

    // The closure for the consts is used for the parameters only: not for the bounds.
    let generics: Generics = parse_quote!(<'a, T: 'a, const B: bool>);
    let referrer = using(&generics, &[]);
    let ty = args_of(&referrer, &parse_quote!(Self), |value, _| {
        Ok(parse_quote!(::my_crate::ConstBool<#value>))
    })
    .unwrap();
    let expected: Type = parse_quote!((&'a (), T, ::my_crate::ConstBool<B>, &'a T));
    assert_eq!(s(&ty), s(&expected));
    let ty = args_of(&referrer, &parse_quote!(Self), slot_const(&repeater)).unwrap();
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
    let repeater = repeater();
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
    let ty = at_site(&referrer, &written, &args, None, array_const).unwrap();
    assert_eq!(s(&ty), s(&expected));
    let ty = at_site(&referrer, &written, &args, None, slot_const(&repeater)).unwrap();
    assert_eq!(s(&ty), s(&expected));

    // The arguments may be any types, and the lifetimes of the Referrer's generics.
    let args = given("<'p, 'q, ::p::W<&'p u8>, N, ()>");
    let ty = at_site(&referrer, &written, &args, None, array_const).unwrap();
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
        let params = args_of(&referrer, &parse_quote!(Self), array_const).unwrap();
        let replaced = at_site(&referrer, &generics, &own, None, array_const).unwrap();
        assert_eq!(
            s(&replaced).replace("{ N }", "N"),
            s(&params),
            "{}",
            s(&generics)
        );
    }
}

/// The closure for the consts gets the const of the entry by its name, and the argument of a site
/// takes its place in what it wrote: braced where it needs braces, as it is where it does not.
#[test]
fn the_consts_a_site_gives_are_braced_in_what_the_closure_wrote() {
    let written: Generics = parse_quote!(<const A: i32, const B: i32>);
    let referrer = using(&written, &[]);
    let flag = |value: &Expr, _: &ConstParam| Ok(parse_quote!(::my_crate::ConstI32<#value>));
    let own = args_of(&referrer, &parse_quote!(Self), flag).unwrap();
    assert_eq!(
        s(&own),
        t(parse_quote!((
            ::my_crate::ConstI32<A>,
            ::my_crate::ConstI32<B>
        )))
    );
    let args = given("<{ i32::MIN }, { N }>");
    let ty = at_site(&referrer, &written, &args, None, flag).unwrap();
    let expected: Type =
        parse_quote!((::my_crate::ConstI32<{ i32::MIN }>, ::my_crate::ConstI32<N>));
    assert_eq!(s(&ty), s(&expected));
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
        let ty = at_site(&referrer, &written, &args, None, array_const).unwrap();
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
    let ty = at_site(&using(&written, &[]), &written, &args, None, array_const).unwrap();
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
    let ty = args_of(&with_self, &self_ty, array_const).unwrap();
    let expected: Type = parse_quote!((&'a (), &'b (), T, [(); N], &'a T, &'a __S, &'b __S));
    assert_eq!(s(&ty), s(&expected));

    // Without the lifetimes it is the encoding of the generics alone.
    let alone: Type = parse_quote!((&'a (), &'b (), T, [(); N], &'a T));
    let ty = args_of(&plain, &self_ty, array_const).unwrap();
    assert_eq!(s(&ty), s(&alone));

    // The order of the context's list is the order of the elements, and a lifetime which is not a
    // parameter is skipped.
    let project = |outlives: &[&str]| {
        let (ix, context) = with_outlives(&with_self, outlives);
        let projection =
            repeater_projection(&repeater(), &self_ty, &context, 7, ix, array_const).unwrap();
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
    let ty = at_site(&with_self, &generics, &args, None, array_const).unwrap();
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

    // Every const encoding agrees, as `usize` is `[(); N]` for the closure and for the slots,
    // whatever stands for `Self`.
    let repeater = repeater();
    let rows: Vec<(Generics, &[&str])> = vec![
        (
            parse_quote!(<'a, 'b: 'a, T: 'a + 'b, const N: usize>),
            &["'a", "'b"],
        ),
        (parse_quote!(<'a, T: 'a, U: 'static>), &["'a"]),
        (parse_quote!(<'a: 'a, 'b: 'a>), &["'b"]),
        (parse_quote!(<T, const N: usize>), &[]),
        (parse_quote!(<>), &[]),
    ];
    for (generics, outlives) in rows {
        let referrer = using(&generics, outlives);
        for self_ty in [
            parse_quote!(__S),
            parse_quote!(Self),
            parse_quote!(::p::S<T>),
        ] {
            let self_ty: Type = self_ty;
            let by_array = args_of(&referrer, &self_ty, array_const).unwrap();
            let by_slots = args_of(&referrer, &self_ty, slot_const(&repeater)).unwrap();
            assert_eq!(s(&by_array), s(&by_slots), "{}", s(&generics));
        }
    }

    // A trait object without a lifetime bound is rejected as `Self`, if there is a lifetime to
    // outlive (`&'a dyn Tr` would default to `&'a (dyn Tr + 'a)`), by the encoding of the entry and
    // by the one at a site.
    let generics: Generics = parse_quote!(<'a>);
    let with_self = using(&generics, &["'a"]);
    let plain = using(&generics, &[]);
    let dyn_self: Type = parse_quote!(dyn Tr);
    let err = args_of(&with_self, &dyn_self, array_const).unwrap_err();
    assert!(
        matches!(err, EncodeError::UnboundedTraitObject(_)),
        "{}",
        err
    );
    let args = given("<'x>");
    let err = at_site(&with_self, &generics, &args, Some(&dyn_self), array_const).unwrap_err();
    assert!(
        matches!(err, EncodeError::UnboundedTraitObject(_)),
        "{}",
        err
    );
    // but not if there is none, and it is fine with the bound written.
    let ty = args_of(&plain, &dyn_self, array_const).unwrap();
    assert_eq!(s(&ty), t(parse_quote!((&'a (),))));
    let bounded: Type = parse_quote!(dyn Tr + 'static);
    let ty = args_of(&with_self, &bounded, array_const).unwrap();
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
    let ty = args_of(&referrer, &marker, array_const).unwrap();
    assert_eq!(s(&ty), t(parse_quote!((&'a (), T, &'a T, &'a __S))));
    // Without it, the encoding is that of the generics alone: an entry with the same generics that
    // does not name `Self`, or no lifetime to outlive.
    let alone = t(parse_quote!((&'a (), T, &'a T)));
    let ty = args_of(&using(generics, &[]), &marker, array_const).unwrap();
    assert_eq!(s(&ty), alone);
    let (ix, context) = with_outlives(&referrer, &[]);
    let ty = repeater_projection(&repeater(), &marker, &context, 7, ix, array_const).unwrap();
    assert_eq!(s(&common::projection_args(&ty)), alone);
    // A lifetime which is not a parameter is skipped, like a bound by one.
    let (ix, context) = with_outlives(&referrer, &["'z"]);
    let ty = repeater_projection(&repeater(), &marker, &context, 7, ix, array_const).unwrap();
    assert_eq!(s(&common::projection_args(&ty)), alone);
    // At the use site, with the arguments substituted, `Self` being the implementor.
    for (args, expected) in [
        ("<'q, u8>", t(parse_quote!((&'q (), u8, &'q u8, &'q Self)))),
        (
            "<'static, u8>",
            t(parse_quote!((&'static (), u8, &'static u8, &'static Self))),
        ),
    ] {
        let ty = at_site(&referrer, generics, &given(args), None, array_const).unwrap();
        assert_eq!(s(&ty), expected, "{}", args);
    }
    // The type that `Self: 'a` names is the one given: the host of a site.
    let host: Type = parse_quote!(Host);
    let ty = at_site(
        &referrer,
        generics,
        &given("<'static, u8>"),
        Some(&host),
        array_const,
    )
    .unwrap();
    assert_eq!(
        s(&ty),
        t(parse_quote!((&'static (), u8, &'static u8, &'static Host)))
    );
}

/// The arguments of a site with no closure for the consts (`array_const`): `usize` is an array
/// length, the elements after the parameters are the bounds, a `?Sized` parameter is a pointer, and
/// the arguments are those of the parameters when they are the parameters themselves.
#[test]
fn the_arguments_of_a_site_are_encoded_as_the_parameters_with_array_const() {
    let written: Generics =
        parse_quote!(<'a, 'b: 'a, T: 'a + ::core::clone::Clone, const N: usize, U: ?Sized>);
    let referrer = using(&written, &[]);
    let args = given("<'x, 'y, u8, { 1 + 2 }, str>");
    let ty = at_site(&referrer, &written, &args, None, array_const).unwrap();
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
    let by_name = at_site(&referrer, &written, &own, None, array_const).unwrap();
    let params = args_of(&referrer, &parse_quote!(Self), array_const).unwrap();
    assert_eq!(s(&by_name).replace("{ N }", "N"), s(&params));
    // Too few and too many arguments.
    assert!(matches!(
        at_site(&referrer, &written, &args[..4], None, array_const),
        Err(EncodeError::MissingArgument(_))
    ));
    let too_many: Vec<GenericArgument> = args.iter().chain(&args[..1]).cloned().collect();
    assert!(matches!(
        at_site(&referrer, &written, &too_many, None, array_const),
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
        let ty = args_of(&referrer, &parse_quote!(Self), array_const).unwrap();
        assert_eq!(s(&ty), s(&expected), "{}", s(&generics));
        // The Referrer passes each parameter by its own name: the same encoding.
        let own: Vec<GenericArgument> = generics.params.iter().map(common::own_name_arg).collect();
        let by_name = at_site(&referrer, &generics, &own, None, array_const).unwrap();
        assert_eq!(s(&by_name), s(&ty), "{}", s(&generics));
    }

    // The arguments of a site: a pointer to the argument, whatever it is.
    let generics: Generics = parse_quote!(<T: ?Sized, U: ?Sized, V>);
    let referrer = using(&generics, &[]);
    let args = given("<str, dyn Tr + Send + 'static, [u8; 2]>");
    let ty = at_site(&referrer, &generics, &args, None, array_const).unwrap();
    let expected: Type = parse_quote!((*const str, *const (dyn Tr + Send + 'static), [u8; 2]));
    assert_eq!(s(&ty), s(&expected));
}

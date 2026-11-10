//! Tests for the Repeater definition, `RepeaterSlots` and `array_const()`. The encoders themselves
//! are tested in `src/encode/tests.rs`.

use syn::*;
use template_quote::quote;

/// `repeater_definition()` docs: the Repeater has `T` (which may be unsized), `KEY`, and a
/// defaulted const slot per other primitive type, in the documented order and with the documented names;
/// its type may be unsized. Audit mutants `def_unsized_param` (`T: ?Sized` dropped),
/// `enc_slot_order` and `enc_slot_name`; the visibility and the unsized `Type` are also pinned by
/// the fixture crates compiling.
#[test]
fn the_repeater_definition_has_the_documented_shape() {
    use type_leak::*;
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
    let slots = RepeaterSlots::new(parse_quote!(R));
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
            slots.encode_const(&value, &ty).unwrap(),
            expected,
            "{}",
            quote!(#ty)
        );
    }
    // `u128` uses `KEY`, `usize` the length of an array, and the others are not supported.
    assert_eq!(
        slots
            .encode_const(&parse_quote!(5), &parse_quote!(u128))
            .unwrap(),
        parse_quote!(*const dyn R<(), 5, Type = ()>)
    );
    assert_eq!(
        slots
            .encode_const(&parse_quote!(5), &parse_quote!(usize))
            .unwrap(),
        parse_quote!([(); 5])
    );
    assert!(matches!(
        slots.encode_const(&parse_quote!(5), &parse_quote!(String)),
        Err(EncodeError::UnsupportedConstType(_))
    ));
}

/// `RepeaterSlots::repeater()` is the path it was made with, and `const_fn()` is `encode_const()`
/// on the type of the parameter: as `Referrer::entry_args()` and `repeater_impl()` take it, for
/// every const type, and with the error of an unsupported one.
#[test]
fn the_slots_give_their_path_and_the_const_fn() {
    use type_leak::*;
    let path: Path = parse_quote!(crate::Repeater);
    let slots = RepeaterSlots::new(path.clone());
    assert_eq!(slots.repeater(), &path);

    let mut const_fn = slots.const_fn();
    for (param, value) in [
        (parse_quote!(const B: bool), "true"),
        (parse_quote!(const C: char), "'x'"),
        (parse_quote!(const N: usize), "3"),
        (parse_quote!(const K: u64), "9"),
        (parse_quote!(const L: u128), "7"),
        (parse_quote!(const I: i128), "-1"),
    ] {
        let param: ConstParam = param;
        let value: Expr = parse_str(value).unwrap();
        assert_eq!(
            const_fn(&value, &param).unwrap(),
            slots.encode_const(&value, &param.ty).unwrap(),
            "{}",
            quote!(#param)
        );
    }
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
    leaker.intern_trait(&tr, true);
    let referrer = leaker.result().finish().unwrap();
    let args = referrer
        .entry_args(0, &UseSite::identity(), slots.const_fn())
        .unwrap();
    assert_eq!(
        args,
        parse_quote!((*const dyn crate::Repeater<(), 0, B, Type = ()>, [(); N]))
    );
}

#[test]
fn test_const_args_are_braced_when_needed() {
    use type_leak::*;
    let to_string = |ty: &Type| quote!(#ty).to_string();
    let slots = RepeaterSlots::new(parse_quote!(crate::Repeater));

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
        let ty = slots
            .encode_const(&parse2(value).unwrap(), &i32_ty)
            .unwrap();
        let expected: Type = parse_quote! {
            *const dyn crate::Repeater<(), 0, false, '\0', 0, 0, 0, 0, 0, 0, #expected, Type = ()>
        };
        assert_eq!(to_string(&ty), to_string(&expected));
    }
    let u128_ty: Type = parse_quote!(u128);
    let ty = slots
        .encode_const(&parse_quote!({ m::FOO }), &u128_ty)
        .unwrap();
    let expected: Type = parse_quote!(*const dyn crate::Repeater<(), { m::FOO }, Type = ()>);
    assert_eq!(to_string(&ty), to_string(&expected));
}

/// `array_const()` encodes a `usize` const as an array length, with the braces that a value needs,
/// and any other type is an error (`RepeaterSlots` encodes it).
#[test]
fn array_const_encodes_usize_consts_only() {
    use type_leak::*;
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
    let slots = RepeaterSlots::new(parse_quote!(crate::Repeater));
    assert!(slots
        .encode_const(&parse_quote!(true), &bool_param.ty)
        .is_ok());
}

/// `repeater_key()` puts the id in the high half and the index in the low half, so two entries
/// never share a key, and it is part of the format (the impls and the projections must agree).
#[test]
fn the_repeater_key_has_the_id_above_the_index() {
    use std::collections::HashSet;
    use type_leak::repeater_key;
    assert_eq!(repeater_key(0, 0), 0);
    assert_eq!(repeater_key(0, 5), 5);
    assert_eq!(repeater_key(1, 0), 1 << 64);
    assert_eq!(repeater_key(7, 3), (7 << 64) | 3);
    assert_eq!(repeater_key(u64::MAX, 0), u128::from(u64::MAX) << 64);
    assert_eq!(repeater_key(0, u64::MAX as usize), u128::from(u64::MAX));
    assert_eq!(repeater_key(u64::MAX, u64::MAX as usize), u128::MAX);
    let ids = [0, 1, 2, 255, 1 << 32, u64::MAX - 1, u64::MAX];
    let nths = [0, 1, 2, 255, 1 << 31, u32::MAX as usize];
    let keys: HashSet<u128> = ids
        .iter()
        .flat_map(|&id| nths.iter().map(move |&nth| repeater_key(id, nth)))
        .collect();
    assert_eq!(keys.len(), ids.len() * nths.len());
}

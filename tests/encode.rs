//! Tests for the Repeater definition, `RepeaterSlots` and `array_const()`. The encoders themselves
//! are tested in `src/encode/tests.rs`.

use syn::*;
use template_quote::quote;

/// `repeater_definition()` docs: the Repeater has `T` (which may be unsized), `ID` and `NTH`, and a
/// defaulted const slot per primitive type, in the documented order and with the documented names;
/// its type may be unsized. Audit mutants `def_unsized_param` (`T: ?Sized` dropped),
/// `enc_slot_order` and `enc_slot_name`; the visibility and the unsized `Type` are also pinned by
/// the fixture crates compiling.
#[test]
fn the_repeater_definition_has_the_documented_shape() {
    use type_leak::*;
    let expected: ItemTrait = parse_quote! {
        pub trait Repeater<
            T: ?::core::marker::Sized,
            const ID: ::core::primitive::u64,
            const NTH: ::core::primitive::usize,
            const BOOL: ::core::primitive::bool = false,
            const CHAR: ::core::primitive::char = '\0',
            const U8: ::core::primitive::u8 = 0,
            const U16: ::core::primitive::u16 = 0,
            const U32: ::core::primitive::u32 = 0,
            const U128: ::core::primitive::u128 = 0,
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
            const ID: ::core::primitive::u64,
            const NTH: ::core::primitive::usize,
            const BOOL: ::core::primitive::bool = false,
            const CHAR: ::core::primitive::char = '\0',
            const U8: ::core::primitive::u8 = 0,
            const U16: ::core::primitive::u16 = 0,
            const U32: ::core::primitive::u32 = 0,
            const U128: ::core::primitive::u128 = 0,
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
        "bool", "char", "u8", "u16", "u32", "u128", "i8", "i16", "i32", "i64", "i128", "isize",
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
            "*const dyn R<(), 0, 0, {}{}, Type = ()>",
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
    // `u64` uses `ID`, `usize` the length of an array, and the others are not supported.
    assert_eq!(
        slots
            .encode_const(&parse_quote!(5), &parse_quote!(u64))
            .unwrap(),
        parse_quote!(*const dyn R<(), 5, 0, Type = ()>)
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
            *const dyn crate::Repeater<(), 0, 0, false, '\0', 0, 0, 0, 0, 0, 0, #expected, Type = ()>
        };
        assert_eq!(to_string(&ty), to_string(&expected));
    }
    let u64_ty: Type = parse_quote!(u64);
    let ty = slots
        .encode_const(&parse_quote!({ m::FOO }), &u64_ty)
        .unwrap();
    let expected: Type = parse_quote!(*const dyn crate::Repeater<(), { m::FOO }, 0, Type = ()>);
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

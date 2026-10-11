//! Tests for encoding generics into types (`encode_generics_*`, `RepeaterSlots`).

use syn::*;
use template_quote::quote;

#[test]
fn test_encode_generics_const() {
    use type_leak::*;
    let generics: Generics = parse_quote!(<'a, T, const N: usize>);
    let ty = encode_generics_params_to_ty(&generics.params);
    assert_eq!(
        quote!(#ty).to_string(),
        quote!((&'a (), T, [(); N])).to_string()
    );

    // `N` is parsed as a type argument; `params` disambiguates it.
    let args: AngleBracketedGenericArguments = parse_quote!(<'static, u8, N>);
    let ty = encode_generics_args_to_ty_with(
        &args.args,
        &generics.params,
        |expr, _| parse_quote!([(); #expr]),
    )
    .unwrap();
    assert_eq!(
        quote!(#ty).to_string(),
        quote!((&'static (), u8, [(); N])).to_string()
    );

    let args: AngleBracketedGenericArguments = parse_quote!(<'static, u8, { 3 }>);
    let ty = encode_generics_args_to_ty_with(
        &args.args,
        &generics.params,
        |expr, _| parse_quote!([(); #expr]),
    )
    .unwrap();
    assert_eq!(
        quote!(#ty).to_string(),
        quote!((&'static (), u8, [(); 3])).to_string()
    );
}

/// rules.md, Const generics: the Repeater has `T` (which may be unsized), `ID` and `NTH`, and a
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
fn test_repeater_slots() {
    use type_leak::*;
    let to_string = |ty: &Type| quote!(#ty).to_string();

    // The Repeater's definition: T, ID, NTH and 12 defaulted const slots.
    let def = repeater_definition(&parse_quote!(pub), &parse_quote!(Repeater));
    let params: Vec<String> = def
        .generics
        .params
        .iter()
        .map(|p| quote!(#p).to_string())
        .collect();
    assert_eq!(params.len(), 15);
    assert_eq!(
        params[3],
        quote!(const BOOL: ::core::primitive::bool = false).to_string()
    );
    assert_eq!(
        params[14],
        quote!(const ISIZE: ::core::primitive::isize = 0).to_string()
    );

    let slots = RepeaterSlots::new(parse_quote!(crate::Repeater));
    let generics: Generics = parse_quote! {
        <'a, T, const N: usize, const X: u64, const B: bool, const I: i32, const M: usize = N>
    };
    let ty = slots
        .encode_generics_params_to_ty(&generics.params)
        .unwrap();
    let expected: Type = parse_quote!((
        &'a (),
        T,
        [(); N],
        *const dyn crate::Repeater<(), X, 0, Type = ()>,
        *const dyn crate::Repeater<(), 0, 0, B, Type = ()>,
        *const dyn crate::Repeater<(), 0, 0, false, '\0', 0, 0, 0, 0, 0, 0, I, Type = ()>,
        [(); M]
    ));
    assert_eq!(to_string(&ty), to_string(&expected));

    // Arguments: a bare const `B`, a negative literal, a skipped binding, and an omitted default
    // referring to a preceding parameter (redundant braces are removed).
    let args: AngleBracketedGenericArguments = parse_quote!(<'static, u8, 3, 7, B, -5, Item = u8>);
    let ty = slots
        .encode_generics_args_to_ty(&args.args, &generics.params)
        .unwrap();
    let expected: Type = parse_quote!((
        &'static (),
        u8,
        [(); 3],
        *const dyn crate::Repeater<(), 7, 0, Type = ()>,
        *const dyn crate::Repeater<(), 0, 0, B, Type = ()>,
        *const dyn crate::Repeater<(), 0, 0, false, '\0', 0, 0, 0, 0, 0, 0, -5, Type = ()>,
        [(); 3]
    ));
    assert_eq!(to_string(&ty), to_string(&expected));

    let args: AngleBracketedGenericArguments = parse_quote!(<'static, u8, { 1 + 2 }, 7, true, 0>);
    let ty = slots
        .encode_generics_args_to_ty(&args.args, &generics.params)
        .unwrap();
    let expected: Type = parse_quote!((
        &'static (),
        u8,
        [(); { 1 + 2 }],
        *const dyn crate::Repeater<(), 7, 0, Type = ()>,
        *const dyn crate::Repeater<(), 0, 0, true, Type = ()>,
        *const dyn crate::Repeater<(), 0, 0, false, '\0', 0, 0, 0, 0, 0, 0, 0, Type = ()>,
        [(); { 1 + 2 }]
    ));
    assert_eq!(to_string(&ty), to_string(&expected));

    // Errors: a missing lifetime, an extra lifetime, a missing argument without a default.
    let args: AngleBracketedGenericArguments = parse_quote!(<u8, 3, 7, B, -5>);
    let err = slots
        .encode_generics_args_to_ty(&args.args, &generics.params)
        .unwrap_err();
    assert_eq!(err.to_string(), "missing generic argument for ''a'");
    let args: AngleBracketedGenericArguments = parse_quote!(<'static, 'static, u8, 3, 7, B, -5>);
    let err = slots
        .encode_generics_args_to_ty(&args.args, &generics.params)
        .unwrap_err();
    assert_eq!(err.to_string(), "unexpected generic argument: ''static'");
    let args: AngleBracketedGenericArguments = parse_quote!(<'static, u8>);
    let err = slots
        .encode_generics_args_to_ty(&args.args, &generics.params)
        .unwrap_err();
    assert_eq!(
        err.to_string(),
        "missing generic argument for 'const N : usize'"
    );
    let generics: Generics = parse_quote!(<const N: MyIdx>);
    let err = slots
        .encode_generics_params_to_ty(&generics.params)
        .unwrap_err();
    assert_eq!(
        err.to_string(),
        "unsupported type of const parameter: 'MyIdx'"
    );

    // No generics, and a single generic (a one-element tuple).
    let ty = slots
        .encode_generics_params_to_ty(&Generics::default().params)
        .unwrap();
    assert_eq!(to_string(&ty), to_string(&parse_quote!(())));
    let generics: Generics = parse_quote!(<T>);
    let ty = slots
        .encode_generics_params_to_ty(&generics.params)
        .unwrap();
    assert_eq!(to_string(&ty), to_string(&parse_quote!((T,))));
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

    // The braces are also kept for the callback of `encode_generics_args_to_ty_with()`.
    let generics: Generics = parse_quote!(<const A: i32, const B: i32>);
    let args: AngleBracketedGenericArguments = parse_quote!(<{ i32::MIN }, { N }>);
    let ty = encode_generics_args_to_ty_with(
        &args.args,
        &generics.params,
        |expr, _| parse_quote!(::my_crate::ConstI32<#expr>),
    )
    .unwrap();
    let expected: Type =
        parse_quote!((::my_crate::ConstI32<{ i32::MIN }>, ::my_crate::ConstI32<N>));
    assert_eq!(to_string(&ty), to_string(&expected));
}

#[test]
fn test_defaults_substitute_const_type_arguments() {
    use type_leak::*;
    let to_string = |ty: &Type| quote!(#ty).to_string();
    let slots = RepeaterSlots::new(parse_quote!(crate::Repeater));
    // `N` in `IntoIter<u8, N>` is parsed as a type argument.
    let generics: Generics = parse_quote!(<const N: usize, U = ::core::array::IntoIter<u8, N>>);
    let args: AngleBracketedGenericArguments = parse_quote!(<3>);
    let ty = slots
        .encode_generics_args_to_ty(&args.args, &generics.params)
        .unwrap();
    let expected: Type = parse_quote!(([(); 3], ::core::array::IntoIter<u8, 3>));
    assert_eq!(to_string(&ty), to_string(&expected));
}

#[test]
fn test_encode_generics_params_to_ty_with() {
    use type_leak::*;
    let generics: Generics = parse_quote!(<'a, T, const B: bool>);
    let ty = encode_generics_params_to_ty_with(
        &generics.params,
        |ConstParam { ident, .. }| parse_quote!(::my_crate::ConstBool<#ident>),
    );
    let expected: Type = parse_quote!((&'a (), T, ::my_crate::ConstBool<B>));
    assert_eq!(quote!(#ty).to_string(), quote!(#expected).to_string());
}

#[test]
fn test_plain_args_encoder_with_params() {
    use type_leak::*;
    let to_string = |ty: &Type| quote!(#ty).to_string();
    let encode = |args: &AngleBracketedGenericArguments, generics: &Generics| {
        encode_generics_args_to_ty_with(
            &args.args,
            &generics.params,
            |expr, _| parse_quote!([(); #expr]),
        )
    };

    // Omitted defaults are filled, a bare `N` is a const, and bindings are skipped.
    let generics: Generics = parse_quote!(<'a, T, const N: usize, U = Vec<T>>);
    let args: AngleBracketedGenericArguments = parse_quote!(<'static, u8, N, Item = u8>);
    let expected: Type = parse_quote!((&'static (), u8, [(); N], Vec<u8>));
    assert_eq!(
        to_string(&encode(&args, &generics).unwrap()),
        to_string(&expected)
    );

    // Too many or missing arguments are reported.
    let args: AngleBracketedGenericArguments = parse_quote!(<'static, u8, 3, u16, u32>);
    assert_eq!(
        encode(&args, &generics).unwrap_err().to_string(),
        "unexpected generic argument: 'u32'"
    );
    let args: AngleBracketedGenericArguments = parse_quote!(<'static, u8>);
    assert_eq!(
        encode(&args, &generics).unwrap_err().to_string(),
        "missing generic argument for 'const N : usize'"
    );

    // Bindings are skipped (they used to panic).
    let generics: Generics = parse_quote!(<T>);
    let args: AngleBracketedGenericArguments = parse_quote!(<u8, Item = u8>);
    assert_eq!(
        to_string(&encode(&args, &generics).unwrap()),
        to_string(&parse_quote!((u8,)))
    );
}

#[test]
fn test_lifetime_bounds_are_encoded_after_the_parameters() {
    use type_leak::*;
    let to_string = |ty: &Type| quote!(#ty).to_string();
    // `T: 'a` is `&'a T` and `'b: 'a` is `&'a &'b ()`, in the order of the parameters and of
    // their bounds, after the elements of the parameters. A bound by `'static` is not encoded,
    // nor one by a lifetime which is not a parameter.
    let generics: Generics = parse_quote!(<
        'a, 'b: 'a, T: 'a + ::core::clone::Clone + 'b, U: 'static, V: 'c, const N: usize
    >);
    let expected: Type = parse_quote!((&'a (), &'b (), T, U, V, [(); N], &'a &'b (), &'a T, &'b T));
    let ty = encode_generics_params_to_ty(&generics.params);
    assert_eq!(to_string(&ty), to_string(&expected));
    let slots = RepeaterSlots::new(parse_quote!(crate::Repeater));
    let ty = slots
        .encode_generics_params_to_ty(&generics.params)
        .unwrap();
    assert_eq!(to_string(&ty), to_string(&expected));

    // The callback for const parameters is used for the parameters only.
    let generics: Generics = parse_quote!(<'a, T: 'a, const B: bool>);
    let ty = encode_generics_params_to_ty_with(
        &generics.params,
        |ConstParam { ident, .. }| parse_quote!(::my_crate::ConstBool<#ident>),
    );
    let expected: Type = parse_quote!((&'a (), T, ::my_crate::ConstBool<B>, &'a T));
    assert_eq!(to_string(&ty), to_string(&expected));
    let ty = slots
        .encode_generics_params_to_ty(&generics.params)
        .unwrap();
    let expected: Type = parse_quote!((
        &'a (),
        T,
        *const dyn crate::Repeater<(), 0, 0, B, Type = ()>,
        &'a T
    ));
    assert_eq!(to_string(&ty), to_string(&expected));

    // A single parameter and its bound; no bound, no element.
    let generics: Generics = parse_quote!(<'a: 'a>);
    let ty = encode_generics_params_to_ty(&generics.params);
    assert_eq!(
        to_string(&ty),
        to_string(&parse_quote!((&'a (), &'a &'a ())))
    );
    let generics: Generics = parse_quote!(<T: 'static>);
    let ty = encode_generics_params_to_ty(&generics.params);
    assert_eq!(to_string(&ty), to_string(&parse_quote!((T,))));
}

#[test]
fn test_lifetime_bounds_are_substituted_by_the_arguments() {
    use type_leak::*;
    let to_string = |ty: &Type| quote!(#ty).to_string();
    let slots = RepeaterSlots::new(parse_quote!(crate::Repeater));
    let generics: Generics = parse_quote!(<'a, 'b: 'a, T: 'a + 'b, const N: usize, U = Vec<T>>);
    let args: AngleBracketedGenericArguments = parse_quote!(<'static, 'x, u8, 3>);
    let expected: Type = parse_quote!((
        &'static (),
        &'x (),
        u8,
        [(); 3],
        Vec<u8>,
        &'static &'x (),
        &'static u8,
        &'x u8
    ));
    let ty = slots
        .encode_generics_args_to_ty(&args.args, &generics.params)
        .unwrap();
    assert_eq!(to_string(&ty), to_string(&expected));
    let ty = encode_generics_args_to_ty_with(
        &args.args,
        &generics.params,
        |expr, _| parse_quote!([(); #expr]),
    )
    .unwrap();
    assert_eq!(to_string(&ty), to_string(&expected));

    // The arguments may be any types, and the lifetimes of the Referrer's generics.
    let args: AngleBracketedGenericArguments = parse_quote!(<'p, 'q, ::p::W<&'p u8>, N, ()>);
    let ty = slots
        .encode_generics_args_to_ty(&args.args, &generics.params)
        .unwrap();
    let expected: Type = parse_quote!((
        &'p (),
        &'q (),
        ::p::W<&'p u8>,
        [(); N],
        (),
        &'p &'q (),
        &'p ::p::W<&'p u8>,
        &'q ::p::W<&'p u8>
    ));
    assert_eq!(to_string(&ty), to_string(&expected));

    // The encoding of the arguments is that of the parameters they replace: the same type with
    // the parameters as the arguments.
    for generics in [
        parse_quote!(<'a, 'b: 'a, T: 'a + 'b, const N: usize>),
        parse_quote!(<'a, T: 'a, U: 'a + ::core::clone::Clone, V: 'static>),
        parse_quote!(<'a: 'a, 'b: 'a>),
        parse_quote!(<T>),
        parse_quote!(<>),
    ] {
        let generics: Generics = generics;
        let args = args_of(&generics);
        let params = slots
            .encode_generics_params_to_ty(&generics.params)
            .unwrap();
        let replaced = slots
            .encode_generics_args_to_ty(&args, &generics.params)
            .unwrap();
        assert_eq!(to_string(&params), to_string(&replaced));
        let plain_params = encode_generics_params_to_ty(&generics.params);
        let plain_args = encode_generics_args_to_ty_with(
            &args,
            &generics.params,
            |expr, _| parse_quote!([(); #expr]),
        )
        .unwrap();
        assert_eq!(to_string(&plain_params), to_string(&plain_args));
    }
}

/// The parameters as the arguments of the same names.
fn args_of(generics: &Generics) -> Vec<GenericArgument> {
    generics
        .params
        .iter()
        .map(|param| match param {
            GenericParam::Lifetime(param) => GenericArgument::Lifetime(param.lifetime.clone()),
            GenericParam::Type(param) => {
                let ident = &param.ident;
                GenericArgument::Type(parse_quote!(#ident))
            }
            GenericParam::Const(param) => {
                let ident = &param.ident;
                GenericArgument::Type(parse_quote!(#ident))
            }
        })
        .collect()
}

#[test]
fn test_trait_object_arguments_of_a_type_with_a_lifetime_bound() {
    use type_leak::*;
    let to_string = |ty: &Type| quote!(#ty).to_string();
    let slots = RepeaterSlots::new(parse_quote!(crate::Repeater));
    let generics: Generics = parse_quote!(<'a, T: ?Sized + 'a>);
    // `&'a dyn Tr` is `&'a (dyn Tr + 'a)`, whereas it is `dyn Tr + 'static` in the other
    // elements, so the encoding would not be that of the generics.
    for arg in [
        parse_quote!(dyn Tr),
        parse_quote!((dyn Tr)),
        parse_quote!(dyn Tr + Send),
    ] {
        let arg: Type = arg;
        let args: AngleBracketedGenericArguments = parse_quote!(<'x, #arg>);
        let err = slots
            .encode_generics_args_to_ty(&args.args, &generics.params)
            .unwrap_err();
        assert!(
            matches!(err, EncodeError::UnboundedTraitObject(_)),
            "{}",
            err
        );
        assert!(err.to_string().contains("no lifetime bound"), "{}", err);
        let err =
            encode_generics_args_to_ty_with(&args.args, &generics.params, |_, _| unreachable!())
                .unwrap_err();
        assert!(
            matches!(err, EncodeError::UnboundedTraitObject(_)),
            "{}",
            err
        );
    }
    // With the bound written, the trait object is parenthesized (`&'a dyn A + 'b` is ambiguous).
    for arg in [parse_quote!(dyn Tr + 'static), parse_quote!((dyn Tr + 'y))] {
        let arg: Type = arg;
        let args: AngleBracketedGenericArguments = parse_quote!(<'x, #arg>);
        let ty = slots
            .encode_generics_args_to_ty(&args.args, &generics.params)
            .unwrap();
        let object = match &arg {
            Type::Paren(TypeParen { elem, .. }) => elem.as_ref().clone(),
            arg => arg.clone(),
        };
        // A `?Sized` parameter is a pointer (a trait object with several bounds in parentheses), so
        // as to be `Sized`.
        let expected: Type = parse_quote!((&'x (), *const (#object), &'x (#object)));
        assert_eq!(to_string(&ty), to_string(&expected));
    }
    // Without a bound on the parameter, any argument is fine.
    let generics: Generics = parse_quote!(<'a, T: ?Sized>);
    let args: AngleBracketedGenericArguments = parse_quote!(<'x, dyn Tr>);
    let ty = slots
        .encode_generics_args_to_ty(&args.args, &generics.params)
        .unwrap();
    assert_eq!(
        to_string(&ty),
        to_string(&parse_quote!((&'x (), *const dyn Tr)))
    );
}

#[test]
fn test_free_encoders_with_self() {
    use type_leak::*;
    let to_string = |ty: &Type| quote!(#ty).to_string();
    let usize_const =
        |ConstParam { ident, .. }: &ConstParam| -> Type { parse_quote!([(); #ident]) };
    let slots = RepeaterSlots::new(parse_quote!(crate::Repeater));
    let a: Lifetime = parse_quote!('a);
    let b: Lifetime = parse_quote!('b);

    // `Self: 'a` follows the lifetime bounds, in the order of the lifetimes.
    let generics: Generics = parse_quote!(<'a, 'b, T: 'a, const N: usize>);
    let expected: Type = parse_quote!((&'a (), &'b (), T, [(); N], &'a T, &'b __S, &'a __S));
    let ty = encode_generics_params_to_ty_with_self(
        &generics.params,
        &parse_quote!(__S),
        &[b.clone(), a.clone()],
        usize_const,
    )
    .unwrap();
    assert_eq!(to_string(&ty), to_string(&expected));

    // Without the lifetimes it is `encode_generics_params_to_ty_with()`, and a lifetime which is
    // not a parameter is skipped.
    for lifetimes in [&[][..], &[parse_quote!('c)][..]] {
        let ty = encode_generics_params_to_ty_with_self(
            &generics.params,
            &parse_quote!(__S),
            lifetimes,
            usize_const,
        )
        .unwrap();
        let plain = encode_generics_params_to_ty(&generics.params);
        assert_eq!(to_string(&ty), to_string(&plain));
    }

    // The arguments replace the parameters in the encoding of the parameters, `Self` included.
    let args: AngleBracketedGenericArguments = parse_quote!(<'p, 'q, ::p::W, N>);
    let ty = encode_generics_args_to_ty_with_self(
        &args.args,
        &generics.params,
        &parse_quote!(Self),
        &[b.clone(), a.clone()],
        |expr, _| parse_quote!([(); #expr]),
    )
    .unwrap();
    let expected: Type = parse_quote!((
        &'p (),
        &'q (),
        ::p::W,
        [(); N],
        &'p ::p::W,
        &'q Self,
        &'p Self
    ));
    assert_eq!(to_string(&ty), to_string(&expected));

    // The free functions and `RepeaterSlots` are the same encoder: they agree wherever the
    // const parameters are encoded in the same way (`usize` is `[(); N]` for both).
    for generics in [
        parse_quote!(<'a, 'b: 'a, T: 'a + 'b, const N: usize>),
        parse_quote!(<'a, T: 'a, U: 'static>),
        parse_quote!(<'a: 'a, 'b: 'a>),
        parse_quote!(<T, const N: usize>),
        parse_quote!(<>),
    ] {
        let generics: Generics = generics;
        let args = args_of(&generics);
        let lifetimes = [a.clone(), b.clone()];
        for self_ty in [
            parse_quote!(__S),
            parse_quote!(Self),
            parse_quote!(::p::S<T>),
        ] {
            let self_ty: Type = self_ty;
            let free = encode_generics_params_to_ty_with_self(
                &generics.params,
                &self_ty,
                &lifetimes,
                usize_const,
            )
            .unwrap();
            let slotted = slots
                .encode_generics_params_to_ty_with_self(&generics.params, &self_ty, &lifetimes)
                .unwrap();
            assert_eq!(to_string(&free), to_string(&slotted));
            let free = encode_generics_args_to_ty_with_self(
                &args,
                &generics.params,
                &self_ty,
                &lifetimes,
                |expr, _| parse_quote!([(); #expr]),
            )
            .unwrap();
            let slotted = slots
                .encode_generics_args_to_ty_with_self(&args, &generics.params, &self_ty, &lifetimes)
                .unwrap();
            assert_eq!(to_string(&free), to_string(&slotted));
            // and the arguments of the parameters are the parameters
            let params = encode_generics_params_to_ty_with_self(
                &generics.params,
                &self_ty,
                &lifetimes,
                usize_const,
            )
            .unwrap();
            assert_eq!(to_string(&free), to_string(&params));
        }
    }

    // A trait object without a lifetime bound is rejected as `Self`, if there is a lifetime
    // to outlive, with either function.
    let generics: Generics = parse_quote!(<'a>);
    let dyn_self: Type = parse_quote!(dyn Tr);
    let err = encode_generics_params_to_ty_with_self(
        &generics.params,
        &dyn_self,
        std::slice::from_ref(&a),
        usize_const,
    )
    .unwrap_err();
    assert!(
        matches!(err, EncodeError::UnboundedTraitObject(_)),
        "{}",
        err
    );
    let args: AngleBracketedGenericArguments = parse_quote!(<'x>);
    let err = encode_generics_args_to_ty_with_self(
        &args.args,
        &generics.params,
        &dyn_self,
        std::slice::from_ref(&a),
        |_, _| unreachable!(),
    )
    .unwrap_err();
    assert!(
        matches!(err, EncodeError::UnboundedTraitObject(_)),
        "{}",
        err
    );
    // but not if there is none, and it is fine with the bound written.
    let ty = encode_generics_params_to_ty_with_self(&generics.params, &dyn_self, &[], usize_const)
        .unwrap();
    assert_eq!(to_string(&ty), to_string(&parse_quote!((&'a (),))));
    let bounded: Type = parse_quote!(dyn Tr + 'static);
    let ty = encode_generics_params_to_ty_with_self(&generics.params, &bounded, &[a], usize_const)
        .unwrap();
    assert_eq!(
        to_string(&ty),
        to_string(&parse_quote!((&'a (), &'a (dyn Tr + 'static))))
    );
}

#[test]
fn test_args_encoder_requires_the_params_to_encode_the_bounds() {
    use type_leak::*;
    let to_string = |ty: &Type| quote!(#ty).to_string();
    // The encoding of the arguments has the encoded bounds of the parameters (`&'static u8` for
    // `T: 'a`), which the arguments alone cannot tell, so the parameters are required: the
    // result is that of the parameters.
    let generics: Generics = parse_quote!(<'a, T: 'a>);
    let args: AngleBracketedGenericArguments = parse_quote!(<'static, u8>);
    let ty = encode_generics_args_to_ty_with(&args.args, &generics.params, |_, _| unreachable!())
        .unwrap();
    assert_eq!(
        to_string(&ty),
        to_string(&parse_quote!((&'static (), u8, &'static u8)))
    );
    // A context whose generics have no defaults (such as the top-level context) needs the
    // arguments complete.
    let generics: Generics = parse_quote!(<T, U>);
    let args: AngleBracketedGenericArguments = parse_quote!(<u8>);
    let err = encode_generics_args_to_ty_with(&args.args, &generics.params, |_, _| unreachable!())
        .unwrap_err();
    assert_eq!(err.to_string(), "missing generic argument for 'U'");
}

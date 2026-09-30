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
        Some(&generics.params),
        |expr, _| parse_quote!([(); #expr]),
    );
    assert_eq!(
        quote!(#ty).to_string(),
        quote!((&'static (), u8, [(); N])).to_string()
    );

    let args: AngleBracketedGenericArguments = parse_quote!(<'static, u8, { 3 }>);
    let ty = encode_generics_args_to_ty(&args.args);
    assert_eq!(
        quote!(#ty).to_string(),
        quote!((&'static (), u8, [(); 3])).to_string()
    );
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

    // Arguments: a bare const `B`, a negative literal, an expression, an omitted lifetime, a
    // skipped binding, and an omitted default referring to a preceding parameter (redundant
    // braces are removed).
    let args: AngleBracketedGenericArguments = parse_quote!(<u8, 3, 7, B, -5, Item = u8>);
    let ty = slots
        .encode_generics_args_to_ty(&args.args, &generics.params)
        .unwrap();
    let expected: Type = parse_quote!((
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

    // Errors.
    let args: AngleBracketedGenericArguments = parse_quote!(<u8>);
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

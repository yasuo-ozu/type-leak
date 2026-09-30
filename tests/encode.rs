//! Tests for encoding generics into types (`encode_generics_*`, `GenericsCarrier`).

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
fn test_generics_carrier() {
    use type_leak::*;
    let to_string = |ty: &Type| quote!(#ty).to_string();

    // Const parameters of any type (including aliases) and defaults are kept; bounds and the
    // where clause are dropped.
    let mut generics: Generics =
        parse_quote!(<'a, T: ::core::clone::Clone, const B: bool, const N: Idx = 3>);
    generics.where_clause = Some(parse_quote!(where T: 'a));
    let def = GenericsCarrier::definition(&parse_quote!(pub), &parse_quote!(Carrier), &generics);
    let expected: ItemStruct = parse_quote! {
        #[doc(hidden)]
        pub struct Carrier<'a, T: ?::core::marker::Sized, const B: bool, const N: Idx = 3>(
            ::core::marker::PhantomData<(&'a (), *const T,)>
        );
    };
    assert_eq!(quote!(#def).to_string(), quote!(#expected).to_string());

    // Only const parameters: a unit struct.
    let generics: Generics = parse_quote!(<const M: bool>);
    let def = GenericsCarrier::definition(&parse_quote!(pub), &parse_quote!(Local), &generics);
    let expected: ItemStruct = parse_quote! {
        #[doc(hidden)]
        pub struct Local<const M: bool>;
    };
    assert_eq!(quote!(#def).to_string(), quote!(#expected).to_string());

    let carrier = GenericsCarrier::new(parse_quote!(crate::m::Carrier));
    let generics: Generics = parse_quote!(<'a, T, const B: bool, const N: Idx = 3>);
    let ty = carrier.encode_generics_params_to_ty(&generics.params);
    assert_eq!(
        to_string(&ty),
        to_string(&parse_quote!(crate::m::Carrier<'a, T, B, N>))
    );

    // Arguments are used as they are (a bare const `B`, an omitted default), and associated
    // bindings are skipped.
    let args: AngleBracketedGenericArguments = parse_quote!(<'static, u8, B, Item = u8>);
    let ty = carrier.encode_generics_args_to_ty(&args.args);
    assert_eq!(
        to_string(&ty),
        to_string(&parse_quote!(crate::m::Carrier<'static, u8, B>))
    );

    // No generics.
    let ty = carrier.encode_generics_params_to_ty(&Generics::default().params);
    assert_eq!(to_string(&ty), to_string(&parse_quote!(crate::m::Carrier)));
}

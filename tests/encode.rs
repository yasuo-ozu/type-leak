//! Tests for encoding generics into types (`encode_generics_*`, `ConstMarkers`).

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
fn test_const_markers() {
    use type_leak::*;
    let markers = ConstMarkers::new(parse_quote!(::m));
    let generics: Generics = parse_quote!(<'a, T, const B: bool, const X: i32, const N: usize>);
    let ty = markers
        .encode_generics_params_to_ty(&generics.params)
        .unwrap();
    let expected: Type = parse_quote!((&'a (), T, ::m::ConstBool<B>, ::m::ConstI32<X>, [(); N]));
    assert_eq!(quote!(#ty).to_string(), quote!(#expected).to_string());

    // Omitted lifetime, bare const ident, negative value and braced expression.
    let args: AngleBracketedGenericArguments = parse_quote!(<u8, B, -3, { 1 + 2 }>);
    let ty = markers
        .encode_generics_args_to_ty(&args.args, &generics.params)
        .unwrap();
    let expected: Type = parse_quote!((u8, ::m::ConstBool<B>, ::m::ConstI32<-3>, [(); 1 + 2]));
    assert_eq!(quote!(#ty).to_string(), quote!(#expected).to_string());

    // `::core::primitive::*` paths are accepted; type aliases are not.
    let ty: Type = parse_quote!(::core::primitive::char);
    assert!(markers.encode_const(&parse_quote!('x'), &ty).is_ok());
    let ty: Type = parse_quote!(MyIdx);
    assert!(markers.encode_const(&parse_quote!(0), &ty).is_err());
}

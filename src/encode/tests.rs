//! Tests of the encoders on generics written by hand, and of the Repeater impl and the arguments of
//! an entry against the pieces they are made of. The functions below are the wrappers of the
//! encoders and of the Referrer that tests used to call, written over the encoders of this crate.

use super::*;
use crate::{repeater_definition, RepeaterSlots};

fn encode_generics_params_to_ty<'a>(iter: impl IntoIterator<Item = &'a GenericParam>) -> Type {
    encode_generics_params_to_ty_with(iter, |ConstParam { ident, .. }| parse_quote!([(); #ident]))
}

fn encode_generics_params_to_ty_with<'a>(
    iter: impl IntoIterator<Item = &'a GenericParam>,
    mut const_fn: impl FnMut(&'a ConstParam) -> Type,
) -> Type {
    encode_generics_params_with(iter, None, |cp| Ok(const_fn(cp)))
        .expect("the parameters are not trait objects")
}

fn encode_generics_params_to_ty_with_self<'a>(
    iter: impl IntoIterator<Item = &'a GenericParam>,
    self_ty: &Type,
    self_outlives: &[Lifetime],
    mut const_fn: impl FnMut(&'a ConstParam) -> Type,
) -> std::result::Result<Type, EncodeError> {
    encode_generics_params_with(iter, Some((self_ty, self_outlives)), |cp| Ok(const_fn(cp)))
}

fn encode_generics_args_to_ty<'a>(
    iter: impl IntoIterator<Item = &'a GenericArgument>,
    params: &Punctuated<GenericParam, Token![,]>,
) -> std::result::Result<Type, EncodeError> {
    encode_generics_args_with_params(iter, params, None, |value, _| {
        let value = normalize_const_arg(value);
        Ok(parse_quote!([(); #value]))
    })
}

fn encode_generics_args_to_ty_with<'a, 'b>(
    iter: impl IntoIterator<Item = &'a GenericArgument>,
    params: &'b Punctuated<GenericParam, Token![,]>,
    mut const_fn: impl FnMut(&Expr, &'b ConstParam) -> Type,
) -> std::result::Result<Type, EncodeError> {
    encode_generics_args_with_params(iter, params, None, |value, cp| Ok(const_fn(value, cp)))
}

fn encode_generics_args_to_ty_with_self<'a, 'b>(
    iter: impl IntoIterator<Item = &'a GenericArgument>,
    params: &'b Punctuated<GenericParam, Token![,]>,
    self_ty: &Type,
    self_outlives: &[Lifetime],
    mut const_fn: impl FnMut(&Expr, &'b ConstParam) -> Type,
) -> std::result::Result<Type, EncodeError> {
    encode_generics_args_with_params(iter, params, Some((self_ty, self_outlives)), |value, cp| {
        Ok(const_fn(value, cp))
    })
}

/// The encoders with the const parameters encoded by the slots.
trait Slotted {
    fn encode_generics_params_to_ty<'a>(
        &self,
        iter: impl IntoIterator<Item = &'a GenericParam>,
    ) -> std::result::Result<Type, EncodeError>;
    fn encode_generics_params_to_ty_with_self<'a>(
        &self,
        iter: impl IntoIterator<Item = &'a GenericParam>,
        self_ty: &Type,
        self_outlives: &[Lifetime],
    ) -> std::result::Result<Type, EncodeError>;
    fn encode_generics_args_to_ty<'a>(
        &self,
        iter: impl IntoIterator<Item = &'a GenericArgument>,
        params: &Punctuated<GenericParam, Token![,]>,
    ) -> std::result::Result<Type, EncodeError>;
    fn encode_generics_args_to_ty_with_self<'a>(
        &self,
        iter: impl IntoIterator<Item = &'a GenericArgument>,
        params: &Punctuated<GenericParam, Token![,]>,
        self_ty: &Type,
        self_outlives: &[Lifetime],
    ) -> std::result::Result<Type, EncodeError>;
}

impl Slotted for RepeaterSlots {
    fn encode_generics_params_to_ty<'a>(
        &self,
        iter: impl IntoIterator<Item = &'a GenericParam>,
    ) -> std::result::Result<Type, EncodeError> {
        self.params(iter, None)
    }
    fn encode_generics_params_to_ty_with_self<'a>(
        &self,
        iter: impl IntoIterator<Item = &'a GenericParam>,
        self_ty: &Type,
        self_outlives: &[Lifetime],
    ) -> std::result::Result<Type, EncodeError> {
        self.params(iter, Some((self_ty, self_outlives)))
    }
    fn encode_generics_args_to_ty<'a>(
        &self,
        iter: impl IntoIterator<Item = &'a GenericArgument>,
        params: &Punctuated<GenericParam, Token![,]>,
    ) -> std::result::Result<Type, EncodeError> {
        encode_generics_args_with_params(iter, params, None, |value, cp| {
            self.encode_const(value, &cp.ty)
        })
    }
    fn encode_generics_args_to_ty_with_self<'a>(
        &self,
        iter: impl IntoIterator<Item = &'a GenericArgument>,
        params: &Punctuated<GenericParam, Token![,]>,
        self_ty: &Type,
        self_outlives: &[Lifetime],
    ) -> std::result::Result<Type, EncodeError> {
        encode_generics_args_with_params(
            iter,
            params,
            Some((self_ty, self_outlives)),
            |value, cp| self.encode_const(value, &cp.ty),
        )
    }
}

impl RepeaterSlots {
    fn params<'a>(
        &self,
        iter: impl IntoIterator<Item = &'a GenericParam>,
        self_outlives: Option<(&Type, &[Lifetime])>,
    ) -> std::result::Result<Type, EncodeError> {
        encode_generics_params_with(iter, self_outlives, |ConstParam { ident, ty, .. }| {
            self.encode_const(&parse_quote!(#ident), ty)
        })
    }
}

#[test]
fn test_encode_generics_const() {
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

#[test]
fn test_repeater_slots() {
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
fn test_defaults_substitute_const_type_arguments() {
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

#[test]
fn test_const_args_are_braced_for_the_callback() {
    let to_string = |ty: &Type| quote!(#ty).to_string();

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

/// The examples that the documentation of the encoders had.
#[test]
fn test_the_documented_examples() {
    let generics: Generics = parse_quote!(<'a, T: 'a, const N: usize>);
    let args: Vec<GenericArgument> = vec![parse_quote!('static), parse_quote!(u8), parse_quote!(3)];
    let ty = encode_generics_args_to_ty(&args, &generics.params).unwrap();
    assert_eq!(ty, parse_quote!((&'static (), u8, [(); 3], &'static u8)));
    let ty = encode_generics_params_to_ty(&generics.params);
    assert_eq!(ty, parse_quote!((&'a (), T, [(); N], &'a T)));

    // The bounds follow the parameters.
    let generics: Generics = parse_quote!(<'a, 'b: 'a, T: 'a + Clone, const N: usize>);
    let ty = encode_generics_params_to_ty(&generics.params);
    assert_eq!(
        ty,
        parse_quote!((&'a (), &'b (), T, [(); N], &'a &'b (), &'a T))
    );

    // A type parameter that may be unsized is a pointer to it, which is always `Sized`.
    let generics: Generics = parse_quote!(<T: ?Sized, U>);
    let ty = encode_generics_params_to_ty(&generics.params);
    assert_eq!(ty, parse_quote!((*const T, U)));

    // A const encoded as a type of your own.
    let generics: Generics = parse_quote!(<'a, T, const B: bool>);
    let ty = encode_generics_params_to_ty_with(&generics.params, |param| {
        let ident = &param.ident;
        parse_quote!(Flag<#ident>)
    });
    assert_eq!(ty, parse_quote!((&'a (), T, Flag<B>)));
    let generics: Generics = parse_quote!(<T, const B: bool>);
    let args: Vec<GenericArgument> = vec![parse_quote!(u8), parse_quote!(true)];
    let ty = encode_generics_args_to_ty_with(
        &args,
        &generics.params,
        |value, _param| parse_quote!(Flag<#value>),
    )
    .unwrap();
    assert_eq!(ty, parse_quote!((u8, Flag<true>)));

    // `Self: 'a` after the bounds.
    let generics: Generics = parse_quote!(<'a, T: 'a>);
    let self_outlives: Vec<Lifetime> = vec![parse_quote!('a)];
    let ty = encode_generics_params_to_ty_with_self(
        &generics.params,
        &parse_quote!(__S),
        &self_outlives,
        |_param| unreachable!("no const parameter"),
    )
    .unwrap();
    assert_eq!(ty, parse_quote!((&'a (), T, &'a T, &'a __S)));
    let args: Vec<GenericArgument> = vec![parse_quote!('static), parse_quote!(u8)];
    let ty = encode_generics_args_to_ty_with_self(
        &args,
        &generics.params,
        &parse_quote!(Host),
        &self_outlives,
        |_value, _param| unreachable!("no const parameter"),
    )
    .unwrap();
    assert_eq!(
        ty,
        parse_quote!((&'static (), u8, &'static u8, &'static Host))
    );
}

#[test]
fn test_without_encoded_bounds() {
    let generics: Generics = parse_quote!(<'a, 'b: 'a, T: 'a + Clone, U: 'static, V: 'c>);
    let declared = without_encoded_bounds(&generics);
    assert_eq!(
        declared,
        parse_quote!(<'a, 'b, T: Clone, U: 'static, V: 'c>)
    );
}

fn ty_string(ty: &Type) -> String {
    quote!(#ty).to_string()
}

/// Parses `tokens` as a type and prints it in the same way as [`ty_string()`].
fn expected_ty(tokens: proc_macro2::TokenStream) -> String {
    ty_string(&parse2(tokens).unwrap())
}

#[test]
fn limit_plain_encoders_usize_only() {
    // `encode_generics_params_to_ty()` and `encode_generics_args_to_ty_with()` with a `[(); N]` callback encode const generics as
    // array lengths, which must be `usize`. For other types they produce `[(); B]` or
    // `[(); true]`, which rustc rejects; `RepeaterSlots` must be used instead.
    let generics: Generics = parse2(quote!(<const B: bool>)).unwrap();
    let ty = encode_generics_params_to_ty(&generics.params);
    assert_eq!(ty_string(&ty), expected_ty(quote!(([(); B],))));

    let args: AngleBracketedGenericArguments = parse2(quote!(<true>)).unwrap();
    let ty = encode_generics_args_to_ty_with(&args.args, &generics.params, |expr, _| {
        parse2(quote!([(); #expr])).unwrap()
    })
    .unwrap();
    assert_eq!(ty_string(&ty), expected_ty(quote!(([(); true],))));

    let slots = RepeaterSlots::new(parse2(quote!(crate::Repeater)).unwrap());
    let ty = slots
        .encode_generics_params_to_ty(&generics.params)
        .unwrap();
    assert_eq!(
        ty_string(&ty),
        expected_ty(quote!(
            (*const dyn crate::Repeater<(), 0, 0, B, Type = ()>,)
        ))
    );
    let ty = slots
        .encode_generics_args_to_ty(&args.args, &generics.params)
        .unwrap();
    assert_eq!(
        ty_string(&ty),
        expected_ty(quote!((
            *const dyn crate::Repeater<(), 0, 0, true, Type = ()>,
        )))
    );
}

#[test]
fn limit_plain_args_encoder_needs_params() {
    // A bare `N` (parsed by syn as a type argument) is a const, and omitted arguments take the
    // parameters' defaults, which the arguments alone cannot tell, so the parameters are required
    // and the result is that of `encode_generics_params_to_ty()`.
    let generics: Generics = parse2(quote!(<T, const N: usize, const M: usize = 3>)).unwrap();
    let args: AngleBracketedGenericArguments = parse2(quote!(<u8, N>)).unwrap();

    let ty = encode_generics_args_to_ty_with(&args.args, &generics.params, |expr, _| {
        parse2(quote!([(); #expr])).unwrap()
    })
    .unwrap();
    assert_eq!(ty_string(&ty), expected_ty(quote!((u8, [(); N], [(); 3]))));
}

// ── Referrers ────────────────────────────────────────────────────────────────

use crate::{ContextId, Leaker, Referrer};
use proc_macro2::TokenStream;

/// The helpers of the Referrer that tests used to call.
trait ReferrerExt {
    /// The generics that a Repeater impl declares for the entries of `context`.
    fn impl_generics(&self, context: crate::ContextId) -> Generics;
    /// The entries with the generics they use, with `Self` replaced by `self_ty`.
    fn iter_with_self<'a>(
        &'a self,
        self_ty: &'a Type,
    ) -> Box<dyn Iterator<Item = (Type, Generics)> + 'a>;
}

impl ReferrerExt for Referrer {
    fn impl_generics(&self, context: crate::ContextId) -> Generics {
        without_encoded_bounds(self.context_generics(context))
    }
    fn iter_with_self<'a>(
        &'a self,
        self_ty: &'a Type,
    ) -> Box<dyn Iterator<Item = (Type, Generics)> + 'a> {
        use syn::fold::Fold;
        Box::new(self.iter().enumerate().map(move |(ix, ty)| {
            let mut folder = crate::util::ReplaceSelf { self_ty };
            let generics = self.context_generics(self.context(ix));
            (
                folder.fold_type(ty.clone()),
                folder.fold_generics(generics.clone()),
            )
        }))
    }
}

fn s(x: &impl template_quote::ToTokens) -> String {
    quote!(#x).to_string()
}

fn of_trait(item: ItemTrait) -> Referrer {
    let mut leaker = Leaker::new();
    leaker.intern_trait(&item, true);
    leaker.result().finish().unwrap()
}

fn lifetimes(list: &[&str]) -> Vec<Lifetime> {
    list.iter().map(|l| parse_str(l).unwrap()).collect()
}

fn slots() -> RepeaterSlots {
    RepeaterSlots::new(parse_quote!(crate::Repeater))
}

#[test]
fn the_encoders_encode_it_after_the_lifetime_bounds() {
    let referrer = of_trait(parse_quote! {
        trait Tr { fn f<'a, T>(&self, x: &'a Rel<(Self, T)>); }
    });
    let context = referrer.context(0);
    let generics = referrer.context_generics(context);
    assert_eq!(s(generics), "< 'a , T : 'a >");
    let outlives = referrer.context_self_outlives(context);
    // The Repeater's implementation for the implementor `__S`: the parameters, the bounds and then
    // `&'a __S`.
    let marker: Type = parse_quote!(__S);
    let encoded = slots()
        .encode_generics_params_to_ty_with_self(&generics.params, &marker, outlives)
        .unwrap();
    let expected: Type = parse_quote!((&'a (), T, &'a T, &'a __S));
    assert_eq!(s(&encoded), s(&expected));
    // Without it, the encoding is that of the generics alone.
    let encoded = slots()
        .encode_generics_params_to_ty(&generics.params)
        .unwrap();
    let expected: Type = parse_quote!((&'a (), T, &'a T));
    assert_eq!(s(&encoded), s(&expected));
    let encoded = slots()
        .encode_generics_params_to_ty_with_self(&generics.params, &marker, &[])
        .unwrap();
    assert_eq!(s(&encoded), s(&expected));
    // At the use site, with the arguments substituted, and `Self` being the implementor.
    let args: Vec<GenericArgument> = vec![parse_quote!('q), parse_quote!(u8)];
    let self_ty: Type = parse_quote!(Self);
    let encoded = slots()
        .encode_generics_args_to_ty_with_self(&args, &generics.params, &self_ty, outlives)
        .unwrap();
    let expected: Type = parse_quote!((&'q (), u8, &'q u8, &'q Self));
    assert_eq!(s(&encoded), s(&expected));
    let args: Vec<GenericArgument> = vec![parse_quote!('static), parse_quote!(u8)];
    let encoded = slots()
        .encode_generics_args_to_ty_with_self(&args, &generics.params, &self_ty, outlives)
        .unwrap();
    let expected: Type = parse_quote!((&'static (), u8, &'static u8, &'static Self));
    assert_eq!(s(&encoded), s(&expected));
    // A lifetime which is not a parameter is skipped, like a bound by one.
    let encoded = slots()
        .encode_generics_params_to_ty_with_self(&generics.params, &marker, &lifetimes(&["'z"]))
        .unwrap();
    let expected: Type = parse_quote!((&'a (), T, &'a T));
    assert_eq!(s(&encoded), s(&expected));
}

#[test]
fn a_trait_object_without_a_lifetime_bound_cannot_be_encoded() {
    let generics: Generics = parse_quote!(<'a>);
    let outlives = lifetimes(&["'a"]);
    // `&'a dyn Tr` would default to `&'a (dyn Tr + 'a)`.
    let unbounded: Type = parse_quote!(dyn Tr);
    assert!(matches!(
        slots().encode_generics_params_to_ty_with_self(&generics.params, &unbounded, &outlives),
        Err(EncodeError::UnboundedTraitObject(_))
    ));
    // Nothing to encode: no error.
    assert!(slots()
        .encode_generics_params_to_ty_with_self(&generics.params, &unbounded, &[])
        .is_ok());
    let bounded: Type = parse_quote!(dyn Tr + 'static);
    let encoded = slots()
        .encode_generics_params_to_ty_with_self(&generics.params, &bounded, &outlives)
        .unwrap();
    let expected: Type = parse_quote!((&'a (), &'a (dyn Tr + 'static)));
    assert_eq!(s(&encoded), s(&expected));
}

#[test]
fn the_implementation_does_not_declare_it() {
    let referrer = of_trait(parse_quote! {
        trait Tr { fn f<'a, T>(&self, x: &'a Rel<(Self, T)>) where Self: ::core::marker::Sized; }
    });
    let context = referrer.context(0);
    // The generics of the context do not mention it (`Self` is not a generic, and the predicate
    // `Self: Sized` is the one written), and neither do those the implementation declares.
    let with_where = |g: &Generics| s(&quote!(#g #{&g.where_clause}));
    assert_eq!(
        with_where(referrer.context_generics(context)),
        "< 'a , T : 'a > where Self : :: core :: marker :: Sized"
    );
    assert_eq!(
        with_where(&referrer.impl_generics(context)),
        "< 'a , T > where Self : :: core :: marker :: Sized"
    );
    // For the generics of `iter_with_self()` as well.
    let marker: Type = parse_quote!(__S);
    let (_, replaced) = referrer.iter_with_self(&marker).next().unwrap();
    assert_eq!(
        with_where(&without_encoded_bounds(&replaced)),
        "< 'a , T > where __S : :: core :: marker :: Sized"
    );
    // A bound `Self: 'a` which is written is the same as one which is implied: it is moved to
    // `context_self_outlives()`, and the other bounds of the predicate are kept (see
    // `a_written_bound_is_the_same_as_an_implied_one`).
    let referrer = of_trait(parse_quote! {
        trait Tr { fn f<'a>(&self, x: Rel<'a, Self>) where Self: ::core::marker::Sized + 'a; }
    });
    let context = referrer.context(0);
    assert_eq!(
        with_where(referrer.context_generics(context)),
        "< 'a > where Self : :: core :: marker :: Sized"
    );
    assert_eq!(
        with_where(&referrer.impl_generics(context)),
        "< 'a > where Self : :: core :: marker :: Sized"
    );
    assert_eq!(referrer.context_self_outlives(context), lifetimes(&["'a"]));
}

#[test]
fn a_written_bound_is_encoded_and_never_declared() {
    let referrer = of_trait(parse_quote! {
        trait Tr {
            fn f<'a, T: 'a>(&self, x: Rel<'a, (Self, T)>) where Self: ::core::marker::Sized + 'a;
        }
    });
    let context = referrer.context(0);
    let outlives = referrer.context_self_outlives(context);
    assert_eq!(outlives, lifetimes(&["'a"]));
    // Neither the generics of the context nor those the implementation declares have it, and
    // `Sized` of the same predicate is kept.
    let with_where = |g: &Generics| s(&quote!(#g #{&g.where_clause}));
    assert_eq!(
        with_where(referrer.context_generics(context)),
        "< 'a , T : 'a > where Self : :: core :: marker :: Sized"
    );
    assert_eq!(
        with_where(&referrer.impl_generics(context)),
        "< 'a , T > where Self : :: core :: marker :: Sized"
    );
    let marker: Type = parse_quote!(__S);
    let (_, replaced) = referrer.iter_with_self(&marker).next().unwrap();
    assert_eq!(
        with_where(&without_encoded_bounds(&replaced)),
        "< 'a , T > where __S : :: core :: marker :: Sized"
    );
    // It is encoded as `&'a __S`, as the implied one is.
    let encoded = slots()
        .encode_generics_params_to_ty_with_self(
            &referrer.context_generics(context).params,
            &marker,
            outlives,
        )
        .unwrap();
    let expected: Type = parse_quote!((&'a (), T, &'a T, &'a __S));
    assert_eq!(s(&encoded), s(&expected));
}

#[test]
fn without_encoded_bounds_removes_what_the_encoders_encode() {
    let mut with_where: Generics = parse_quote!(<'a, T>);
    with_where.where_clause = Some(parse_quote!(where T: 'a, Vec<T>: Clone));
    let cases: Vec<(Generics, &str)> = vec![
        (
            parse_quote!(<'a, 'b: 'a, T: 'a + Clone, U: 'static, V: 'c, W: ?Sized + 'b, const N: usize>),
            "< 'a , 'b , T : Clone , U : 'static , V : 'c , W : ? Sized , const N : usize >",
        ),
        // Only the lifetime parameters of the list count.
        (parse_quote!(<'b: 'b>), "< 'b >"),
        (parse_quote!(<T: 'a>), "< T : 'a >"),
        // The where clause is the encoders' blind spot, so it is kept.
        (with_where, "< 'a , T > where T : 'a , Vec < T > : Clone"),
    ];
    for (generics, expected) in cases {
        let declared = without_encoded_bounds(&generics);
        assert_eq!(s(&quote!(#declared #{&declared.where_clause})), expected);
    }
    // The encoders encode exactly the bounds which are removed: the elements after the
    // parameters are as many as the bounds dropped.
    let slots = RepeaterSlots::new(parse_quote!(crate::Repeater));
    let count = |generics: &Generics| -> usize {
        generics
            .params
            .iter()
            .map(|param| match param {
                GenericParam::Lifetime(param) => param.bounds.len(),
                GenericParam::Type(param) => param.bounds.len(),
                GenericParam::Const(_) => 0,
            })
            .sum()
    };
    for generics in [
        parse_quote!(<'a, 'b: 'a, T: 'a + 'b + Clone, U: 'static, V: 'c, const N: usize>),
        parse_quote!(<'a, 'b: 'a + 'c, T: 'a>),
        parse_quote!(<'a, T: Clone + 'a, U: 'a + 'static>),
        parse_quote!(<'a, T>),
        parse_quote!(<>),
    ] {
        let generics: Generics = generics;
        let declared = without_encoded_bounds(&generics);
        let elements = |ty: Type| match ty {
            Type::Tuple(tuple) => tuple.elems.len(),
            _ => unreachable!(),
        };
        let extra = |generics: &Generics| -> usize {
            elements(encode_generics_params_to_ty(&generics.params)) - generics.params.len()
        };
        assert_eq!(
            extra(&generics),
            count(&generics) - count(&declared),
            "{}",
            s(&generics)
        );
        // Nothing is left to encode once they are removed.
        assert_eq!(extra(&declared), 0, "{}", s(&declared));
        let slotted = slots
            .encode_generics_params_to_ty(&generics.params)
            .unwrap();
        assert_eq!(
            elements(slotted) - generics.params.len(),
            count(&generics) - count(&declared)
        );
    }
}

/// The generics of the top-level context are the call's (no bound is implied there).
#[test]
fn the_generics_of_a_top_level_context_are_declared_as_they_are() {
    let tr: ItemTrait = parse_quote! { trait Tr<'a, T> { fn f(&self, x: &'a Rel<T>); } };
    let mut leaker = Leaker::new();
    let top_level = leaker.intern_trait(&tr, true);
    let referrer = leaker.result().finish().unwrap();
    assert_eq!(s(&referrer.impl_generics(top_level)), "< 'a , T >");
}

/// Parses `Referrer, ContextId`, as the Referrer's macro receives them from the Leaker's.
fn parse_referrer_and_top_level(tokens: TokenStream) -> (Referrer, ContextId) {
    use syn::parse::Parser;
    (|input: syn::parse::ParseStream| -> Result<(Referrer, ContextId)> {
        let referrer = input.parse()?;
        input.parse::<Token![,]>()?;
        let top_level = input.parse()?;
        input.parse::<syn::parse::Nothing>()?;
        Ok((referrer, top_level))
    })
    .parse2(tokens)
    .unwrap()
}

#[test]
fn the_top_level_context_reaches_the_referrer_side_through_tokens() {
    let usize_const =
        |ConstParam { ident, .. }: &ConstParam| -> Type { parse_quote!([(); #ident]) };

    // The Referrer cannot tell the top-level context (it is not serialized, as it may also be
    // the context of an entry), but the id returned by the Leaker is the same in the Referrer
    // parsed from tokens, and it is written and parsed as an integer.
    let tr: ItemTrait = parse_quote! {
        trait Tr<'a, const N: usize, T: 'a, U = ::core::primitive::u8> {
            fn f(&self, x: Rel<'a, T>) -> Wrap<N>;
        }
    };
    let mut leaker = Leaker::new();
    let top_level = leaker.intern_trait(&tr, true);
    let referrer = leaker.result().finish().unwrap();
    // The generics are in the canonical form: sorted by kind and name, without defaults.
    assert_eq!(
        s(referrer.context_generics(top_level)),
        "< 'a , T : 'a , U , const N : usize >"
    );
    let leaker_params = &referrer.context_generics(top_level).params;
    let on_leaker_side = encode_generics_params_to_ty_with(leaker_params, usize_const);
    let expected: Type = parse_quote!((&'a (), T, U, [(); N], &'a T));
    assert_eq!(s(&on_leaker_side), s(&expected));

    let tokens = quote!(#referrer, #top_level);
    let (parsed, parsed_top_level) = parse_referrer_and_top_level(tokens);
    assert_eq!(parsed, referrer);
    assert_eq!(parsed_top_level, top_level);
    let params = &parsed.context_generics(parsed_top_level).params;
    assert_eq!(s(params), s(leaker_params));

    // The arguments of the trait are completed (the generics of a context have no defaults), and
    // put in the order of the generics (that of the trait differs): the encoding is that of the
    // generics with the arguments substituted. Matching them by position would be wrong.
    let args: AngleBracketedGenericArguments = parse_quote!(<'q, 3, u16>);
    let err =
        encode_generics_args_to_ty_with(&args.args, params, |_, _| unreachable!()).unwrap_err();
    assert!(matches!(err, EncodeError::UnexpectedArgument(_)), "{}", err);
    let completed = parsed
        .complete_args(&args.args, &tr.generics.params, |_, _, _| unreachable!())
        .unwrap();
    let args = parsed
        .context_generic_args(parsed_top_level, &tr.generics.params, &completed)
        .unwrap();
    assert_eq!(
        args.iter().map(s).collect::<Vec<_>>(),
        ["'q", "u16", ":: core :: primitive :: u8", "3"]
    );
    let on_referrer_side =
        encode_generics_args_to_ty_with(&args, params, |expr, _| parse_quote!([(); #expr]))
            .unwrap();
    let expected: Type = parse_quote!((&'q (), u16, ::core::primitive::u8, [(); 3], &'q u16));
    assert_eq!(s(&on_referrer_side), s(&expected));

    // It is `generic_args()` of an entry for the context of the entry.
    for ix in 0..parsed.iter().count() {
        assert_eq!(
            parsed
                .context_generic_args(parsed.context(ix), &tr.generics.params, &completed)
                .unwrap()
                .iter()
                .map(s)
                .collect::<Vec<_>>(),
            parsed
                .context_generic_args(parsed.context(ix), &tr.generics.params, &completed)
                .unwrap()
                .iter()
                .map(s)
                .collect::<Vec<_>>()
        );
    }
}

// ── The helpers against what a macro wrote by hand before them ───────────────────────────────

use crate::{ErrorItem, ImplSelf, LeakerConfig, SelfRequirement, UseSite};

fn t(ty: Type) -> String {
    s(&ty)
}

fn config() -> LeakerConfig {
    let mut config = LeakerConfig::new();
    config.allow_primitive().allow_prelude();
    config
}

/// Analyzes a whole item the way a downstream macro does.
fn analyze(item: &Item) -> (Referrer, ContextId) {
    let mut leaker = Leaker::from_config(config());
    let context = match item {
        Item::Struct(i) => leaker.intern_struct(i, true),
        Item::Enum(i) => leaker.intern_enum(i, true),
        Item::Union(i) => leaker.intern_union(i, true),
        Item::Trait(i) => leaker.intern_trait(i, true),
        Item::Impl(i) => leaker.intern_impl(i, true),
        Item::Mod(i) => leaker.intern_mod(i),
        _ => panic!("unsupported item"),
    };
    let referrer = leaker
        .result()
        .finish()
        .unwrap_or_else(|e| panic!("{}: {}", s(item), messages(&e)));
    (referrer, context)
}

fn generics_of(item: &Item) -> Generics {
    match item {
        Item::Struct(i) => i.generics.clone(),
        Item::Enum(i) => i.generics.clone(),
        Item::Union(i) => i.generics.clone(),
        Item::Trait(i) => i.generics.clone(),
        Item::Impl(i) => i.generics.clone(),
        _ => Generics::default(),
    }
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

/// The messages of the errors, one per line.
fn messages(errors: &[ErrorItem]) -> String {
    errors
        .iter()
        .map(|item| item.message.as_str())
        .collect::<Vec<_>>()
        .join("\n")
}

fn with_where(generics: &Generics) -> String {
    format!("{} {}", s(generics), s(&generics.where_clause))
}

/// What a macro wrote by hand: the generics of the impl with the parameter for `Self`, the type
/// it is for, its trait argument and the type. For `ImplSelf::Param`.
fn by_hand(referrer: &Referrer, ix: usize, self_param: &Ident) -> (Generics, Type, Type, Type) {
    let marker: Type = parse_quote!(#self_param);
    let replaced: Vec<(Type, Generics)> = referrer.iter_with_self(&marker).collect();
    let (ty, generics) = referrer
        .iter()
        .enumerate()
        .map(|(ix, ty)| (ty, referrer.context_generics(referrer.context(ix))))
        .nth(ix)
        .unwrap();
    let context = referrer.context(ix);
    let (ty, generics) = match referrer.context_self(context) {
        Some(SelfRequirement::Bound(_)) => (&replaced[ix].0, &replaced[ix].1),
        _ => (ty, generics),
    };
    let self_in_impl: Type = match referrer.context_self(context) {
        Some(SelfRequirement::Type(_)) => parse_quote!(Self),
        _ => marker.clone(),
    };
    let encoded = slots()
        .encode_generics_params_to_ty_with_self(
            &generics.params,
            &self_in_impl,
            referrer.context_self_outlives(context),
        )
        .unwrap();
    let mut declared = match referrer.context_self(context) {
        Some(SelfRequirement::Bound(_)) => without_encoded_bounds(generics),
        _ => referrer.impl_generics(context),
    };
    let self_ty = match referrer.context_self(context) {
        Some(SelfRequirement::Bound(path)) => {
            declared
                .params
                .push(parse_quote!(#self_param: ?::core::marker::Sized + #path));
            marker
        }
        Some(SelfRequirement::Type(self_ty)) => self_ty.clone(),
        _ => {
            declared
                .params
                .push(parse_quote!(#self_param: ?::core::marker::Sized));
            marker
        }
    };
    (declared, self_ty, encoded, ty.clone())
}

/// `(<S>::A): Clone` as the first predicate is `<S>::A: Clone`.
fn unparenthesized(mut generics: Generics) -> Generics {
    if let Some(WherePredicate::Type(predicate)) = generics
        .where_clause
        .as_mut()
        .and_then(|clause| clause.predicates.first_mut())
    {
        if let Type::Paren(TypeParen { elem, .. }) = &predicate.bounded_ty {
            if matches!(elem.as_ref(), Type::Path(TypePath { qself: Some(_), .. })) {
                predicate.bounded_ty = (**elem).clone();
            }
        }
    }
    generics
}

/// What a macro wrote by hand at a use site.
fn args_by_hand(
    referrer: &Referrer,
    ix: usize,
    root_params: &Punctuated<GenericParam, Token![,]>,
    root_args: &[GenericArgument],
    self_ty: &Type,
) -> Type {
    let args = referrer
        .context_generic_args(referrer.context(ix), root_params, root_args)
        .unwrap();
    slots()
        .encode_generics_args_to_ty_with_self(
            &args,
            &referrer.context_generics(referrer.context(ix)).params,
            self_ty,
            referrer.context_self_outlives(referrer.context(ix)),
        )
        .unwrap()
}

/// One argument for each parameter.
fn arguments_for(generics: &Generics) -> Vec<GenericArgument> {
    generics
        .params
        .iter()
        .map(|param| match param {
            GenericParam::Lifetime(_) => parse_quote!('static),
            GenericParam::Type(_) => parse_quote!(::core::primitive::u8),
            GenericParam::Const(ConstParam { ty, .. }) if s(ty) == "bool" => parse_quote!(true),
            GenericParam::Const(_) => parse_quote!(3),
        })
        .collect()
}

#[test]
fn repeater_impl_is_what_a_macro_writes_by_hand() {
    let self_param: Ident = parse_quote!(__S);
    let mut checked = 0;
    for item in samples() {
        let (referrer, _) = analyze(&item);
        for ix in 0..referrer.iter().count() {
            let r = referrer
                .repeater_impl(ix, ImplSelf::Param(&self_param), |value, param| {
                    slots().encode_const(value, &param.ty)
                })
                .unwrap_or_else(|e| panic!("{}: {}", s(&item), e));
            let (generics, self_ty, args, ty) = by_hand(&referrer, ix, &self_param);
            let what = format!("{}: {}", s(&item), s(&referrer.iter().nth(ix)));
            assert_eq!(
                with_where(&unparenthesized(r.generics.clone())),
                with_where(&generics),
                "{}",
                what
            );
            assert_eq!(s(&r.self_ty), s(&self_ty), "{}", what);
            assert_eq!(s(&r.args), s(&args), "{}", what);
            assert_eq!(s(&r.ty), s(&ty), "{}", what);
            assert_eq!(
                r.module,
                referrer.context_module(referrer.context(ix)),
                "{}",
                what
            );
            checked += 1;
        }
    }
    assert!(checked > 20, "{}", checked);
}

#[test]
fn entry_args_is_what_a_macro_writes_by_hand() {
    let host: Type = parse_quote!(Host<u8>);
    let mut checked = 0;
    for item in samples() {
        let (referrer, _) = analyze(&item);
        let generics = generics_of(&item);
        let root_args = arguments_for(&generics);
        for ix in 0..referrer.iter().count() {
            let what = format!("{}: {}", s(&item), s(&referrer.iter().nth(ix)));
            let encode = |value: &Expr, param: &ConstParam| slots().encode_const(value, &param.ty);
            // Substituted, with a `Self`.
            let site = UseSite::substituting(&generics.params, &root_args).with_self(host.clone());
            let got = referrer.entry_args(ix, &site, encode).unwrap();
            let expected = args_by_hand(&referrer, ix, &generics.params, &root_args, &host);
            assert_eq!(s(&got), s(&expected), "{}", what);
            // In the scope of the analysis (nothing substituted), with the default `Self`.
            let got = referrer
                .entry_args(ix, &UseSite::identity(), encode)
                .unwrap();
            let expected =
                args_by_hand(&referrer, ix, &Punctuated::new(), &[], &parse_quote!(Self));
            assert_eq!(s(&got), s(&expected), "{}", what);
            // The same as the impl, for the generics by their own names.
            let param: Ident = parse_quote!(__S);
            let r = referrer
                .repeater_impl(ix, ImplSelf::Param(&param), encode)
                .unwrap();
            let self_in_impl = match referrer.context_self(referrer.context(ix)) {
                Some(SelfRequirement::Type(_)) => parse_quote!(Self),
                _ => parse_quote!(#param),
            };
            let got = referrer
                .entry_args(ix, &UseSite::identity().with_self(self_in_impl), encode)
                .unwrap();
            assert_eq!(s(&got), s(&r.args), "{}", what);
            checked += 1;
        }
    }
    assert!(checked > 20, "{}", checked);
}

#[test]
fn entry_args_to_ty_has_no_closure() {
    // `encode_generics_args_to_ty()` is `encode_generics_params_to_ty()` for arguments.
    let generics: Generics = parse_quote!(<'a, 'b: 'a, T: 'a + Clone, const N: usize, U: ?Sized>);
    let args: Vec<GenericArgument> = vec![
        parse_quote!('x),
        parse_quote!('y),
        parse_quote!(u8),
        parse_quote!({ 1 + 2 }),
        parse_quote!(str),
    ];
    let ty = encode_generics_args_to_ty(&args, &generics.params).unwrap();
    assert_eq!(
        s(&ty),
        t(parse_quote!((
            &'x (),
            &'y (),
            u8,
            [(); { 1 + 2 }],
            *const str,
            &'x &'y (),
            &'x u8
        )))
    );
    // For the parameters themselves, it is `encode_generics_params_to_ty()`.
    let own = generics
        .params
        .iter()
        .map(|param| match param {
            GenericParam::Lifetime(LifetimeParam { lifetime, .. }) => {
                GenericArgument::Lifetime(lifetime.clone())
            }
            GenericParam::Type(TypeParam { ident, .. }) => {
                GenericArgument::Type(parse_quote!(#ident))
            }
            GenericParam::Const(ConstParam { ident, .. }) => {
                GenericArgument::Const(parse_quote!(#ident))
            }
        })
        .collect::<Vec<_>>();
    assert_eq!(
        s(&encode_generics_args_to_ty(&own, &generics.params).unwrap()),
        s(&encode_generics_params_to_ty(&generics.params))
    );
    // Too few and too many arguments.
    assert!(matches!(
        encode_generics_args_to_ty(&args[..4], &generics.params),
        Err(EncodeError::MissingArgument(_))
    ));
    assert!(matches!(
        encode_generics_args_to_ty(args.iter().chain(&args[..1]), &generics.params),
        Err(EncodeError::UnexpectedArgument(_))
    ));
}

// ── Findings ─────────────────────────────────────────────────────────────────────────────────

fn string<T: template_quote::ToTokens>(t: &T) -> String {
    quote!(#t).to_string()
}

/// The names of the `?Sized` type parameters of `generics`.
fn unsized_params(generics: &Generics) -> Vec<Ident> {
    let maybe = |bounds: &Punctuated<TypeParamBound, Token![+]>| {
        bounds.iter().any(|b| {
            matches!(
                b,
                TypeParamBound::Trait(TraitBound {
                    modifier: TraitBoundModifier::Maybe(_),
                    ..
                })
            )
        })
    };
    let mut out: Vec<Ident> = generics
        .params
        .iter()
        .filter_map(|param| match param {
            GenericParam::Type(t) if maybe(&t.bounds) => Some(t.ident.clone()),
            _ => None,
        })
        .collect();
    for predicate in generics.where_clause.iter().flat_map(|w| &w.predicates) {
        if let WherePredicate::Type(PredicateType {
            bounded_ty: Type::Path(TypePath { qself: None, path }),
            bounds,
            ..
        }) = predicate
        {
            if let (Some(ident), true) = (path.get_ident(), maybe(bounds)) {
                out.push(ident.clone());
            }
        }
    }
    out
}

/// A `?Sized` parameter of `generics` that `encoded` has as a bare tuple element, which is not the
/// last: a tuple is well-formed only if its last element is unsized.
fn bare_unsized_element(generics: &Generics, encoded: &Type) -> Option<Ident> {
    let unsized_params = unsized_params(generics);
    let Type::Tuple(tuple) = encoded else {
        return None;
    };
    let last = tuple.elems.len().checked_sub(1)?;
    tuple.elems.iter().take(last).find_map(|elem| match elem {
        Type::Path(TypePath { qself: None, path }) => path
            .get_ident()
            .filter(|ident| unsized_params.contains(ident))
            .cloned(),
        _ => None,
    })
}

fn param_arg(param: &GenericParam) -> GenericArgument {
    match param {
        GenericParam::Lifetime(l) => GenericArgument::Lifetime(l.lifetime.clone()),
        GenericParam::Type(t) => {
            let ident = &t.ident;
            GenericArgument::Type(parse_quote!(#ident))
        }
        GenericParam::Const(c) => {
            let ident = &c.ident;
            GenericArgument::Type(parse_quote!(#ident))
        }
    }
}

/// F1 (`tests/findings.rs`): a `?Sized` type parameter is encoded as a pointer to it, not as a bare
/// tuple element, which the encoder does for the generics it is given.
#[test]
fn f01_unsized_type_param_encoding_of_written_generics() {
    let slots = RepeaterSlots::new(parse_quote!(crate::Repeater));
    let check = |generics: &Generics, expected: Option<&str>, what: &str| {
        let encoded = slots
            .encode_generics_params_to_ty(&generics.params)
            .unwrap_or_else(|e| panic!("{}: `{}`: {}", what, string(generics), e));
        assert_eq!(
            bare_unsized_element(generics, &encoded),
            None,
            "{}: `{}` encodes `{}` with an unsized parameter in the middle",
            what,
            string(generics),
            string(&encoded)
        );
        if let Some(expected) = expected {
            let expected: Type = parse_str(expected).unwrap();
            assert_eq!(string(&encoded), string(&expected), "{}", what);
        }
        // The Referrer passes each parameter by its own name: the same encoding.
        let args: Vec<GenericArgument> = generics.params.iter().map(param_arg).collect();
        let on_referrer_side = slots
            .encode_generics_args_to_ty(&args, &generics.params)
            .unwrap();
        assert_eq!(string(&on_referrer_side), string(&encoded), "{}", what);
    };

    // The encoder on its own: an unsized parameter first, in the middle and last.
    check(
        &parse_quote!(<T: ?Sized, U>),
        Some("(*const T, U)"),
        "the encoder",
    );
    check(
        &parse_quote!(<T, U: ?Sized, V>),
        Some("(T, *const U, V)"),
        "the encoder",
    );
    check(
        &parse_quote!(<T, U: ?Sized>),
        Some("(T, *const U)"),
        "the encoder",
    );
    check(
        &parse_quote!(<T: ?Sized>),
        Some("(*const T,)"),
        "the encoder",
    );
    // Other bounds stay out of the way, and a lifetime bound is encoded as it is for a sized one.
    check(
        &parse_quote!(<'a, T: Clone + ?Sized + 'a, U>),
        Some("(&'a (), *const T, U, &'a T)"),
        "the encoder",
    );
    // A sized parameter is encoded as before.
    check(&parse_quote!(<T, U: Sized>), Some("(T, U)"), "the encoder");

    // The Referrer's arguments: a pointer to the argument, whatever it is.
    let generics: Generics = parse_quote!(<T: ?Sized, U: ?Sized, V>);
    let args: AngleBracketedGenericArguments =
        parse_quote!(<str, dyn Tr + Send + 'static, [u8; 2]>);
    let encoded = slots
        .encode_generics_args_to_ty(&args.args, &generics.params)
        .unwrap();
    let expected: Type = parse_quote!((*const str, *const (dyn Tr + Send + 'static), [u8; 2]));
    assert_eq!(string(&encoded), string(&expected));
}

/// The bounded types of the where predicates that rustc cannot parse.
struct UnparenthesizedPredicates(Vec<String>);

impl<'ast> syn::visit::Visit<'ast> for UnparenthesizedPredicates {
    fn visit_predicate_type(&mut self, predicate: &'ast PredicateType) {
        if bare_fn_under_pointer(&predicate.bounded_ty) {
            self.0.push(string(predicate));
        }
        syn::visit::visit_predicate_type(self, predicate)
    }
}

/// Whether a type is a function pointer without an output that is reached through a reference or a
/// pointer (also after the output of another function pointer: `fn() -> &fn()`): as the bounded
/// type of a where predicate, rustc reads the `:` after it as the start of its return type.
fn bare_fn_under_pointer(ty: &Type) -> bool {
    let mut under_pointer = false;
    let mut ty = ty;
    loop {
        match ty {
            Type::Reference(r) => {
                under_pointer = true;
                ty = &r.elem;
            }
            Type::Ptr(p) => {
                under_pointer = true;
                ty = &p.elem;
            }
            Type::BareFn(f) => match &f.output {
                ReturnType::Type(_, output) => ty = output,
                ReturnType::Default => return under_pointer,
            },
            _ => return false,
        }
    }
}

/// F9 (`tests/findings.rs`): the generics of an entry with `Self` replaced by a type parenthesize
/// a bounded type that is a reference or a pointer to a function pointer.
#[test]
fn f09_fn_pointer_self_predicate_print_in_the_generics_of_an_entry() {
    use syn::visit::Visit;
    let item: ItemTrait = parse_quote! {
        pub trait Tr {
            fn m<'y>(&self, a: u8) where Self: 'y;
            fn n(&self, a: u8) where Self: Clone;
            fn r(&self) -> Rel<Self> where Self: Clone;
        }
    };
    let mut leaker = Leaker::from_config(config());
    leaker.intern_trait(&item, true);
    let referrer = leaker.result().finish().unwrap();
    let self_tys: Vec<Type> = vec![
        parse_quote!(&'static unsafe fn(i32, R0)),
        parse_quote!(*const fn(i32)),
        parse_quote!(fn(i32) -> &'static fn(u8)),
        parse_quote!(fn(i32)),
    ];
    for self_ty in self_tys {
        // The generics of the entry `Rel<Self>`, which has `where Self: Clone`.
        let mut seen = false;
        for (_, generics) in referrer.iter_with_self(&self_ty) {
            let mut found = UnparenthesizedPredicates(Vec::new());
            found.visit_generics(&generics);
            seen |= generics.where_clause.is_some();
            assert!(
                found.0.is_empty(),
                "Self = `{}`: iter_with_self() has {:?}",
                string(&self_ty),
                found.0
            );
        }
        assert!(seen, "the entry `Rel<Self>` has no where predicate");
    }
}

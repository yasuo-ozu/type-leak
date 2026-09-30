//! Tests for how `type-leak` reports inputs it cannot handle.
//!
//! - `limit_*`: inputs which cannot be leaked on stable Rust. `type-leak` must reject them with
//!   [`NotInternableError`].
//! - `hole_*`: inputs which `type-leak` currently mishandles (a panic, or an output which does not
//!   compile). These tests record the current behavior; update them when the hole is fixed.

use syn::*;
use template_quote::quote;
use type_leak::*;

fn leaker_config() -> LeakerConfig {
    let mut config = LeakerConfig::new();
    config.allow_primitive().allow_crate();
    config
}

/// Runs a Leaker on `item_trait`, returning the leaked types or the error message.
fn leak_trait(item_trait: &ItemTrait) -> std::result::Result<Vec<String>, String> {
    let mut leaker = Leaker::from_config(leaker_config());
    leaker
        .intern_with(&item_trait.generics, |v| v.visit_item_trait(item_trait))
        .map_err(|e| e.to_string())?;
    leaker.reduce_roots();
    Ok(leaker
        .finish()
        .iter()
        .map(|t| quote!(#t).to_string())
        .collect())
}

fn ty_string(ty: &Type) -> String {
    quote!(#ty).to_string()
}

/// Parses `tokens` as a type and prints it in the same way as [`ty_string()`].
fn expected_ty(tokens: proc_macro2::TokenStream) -> String {
    ty_string(&parse2(tokens).unwrap())
}

// ── limitations ──────────────────────────────────────────────────────────────

#[test]
fn limit_const_in_bound() {
    // A relative const in a trait bound cannot be leaked, because a const cannot be recovered
    // from an associated type in a generic context on stable Rust.
    let item: ItemTrait = parse2(quote! {
        pub trait MyTrait: crate::l::Sup<{ SIZE }> {
            fn f(&self);
        }
    })
    .unwrap();
    assert_eq!(
        leak_trait(&item),
        Err("use absolute path: 'SIZE'".to_string())
    );

    // The suggested fix works: an absolute path does not need to be leaked.
    let item: ItemTrait = parse2(quote! {
        pub trait MyTrait: crate::l::Sup<{ ::my_crate::SIZE }> {
            fn f(&self);
        }
    })
    .unwrap();
    assert_eq!(leak_trait(&item), Ok(vec![]));
}

#[test]
fn limit_conflicting_local_generics() {
    // A type interned together with local generics must depend on the same local generics
    // wherever it appears, since the Referrer finds it by the type alone.
    let item: ItemTrait = parse2(quote! {
        pub trait MyTrait {
            fn f<const B: bool>(&self) -> Wrap<B>;
            fn g<const B: u8>(&self) -> Wrap<B>;
        }
    })
    .unwrap();
    assert_eq!(
        leak_trait(&item),
        Err("use absolute path: 'Wrap < B >'".to_string())
    );

    // The suggested fix works: an absolute path does not need to be leaked.
    let item: ItemTrait = parse2(quote! {
        pub trait MyTrait {
            fn f<const B: bool>(&self) -> ::my_crate::Wrap<B>;
            fn g<const B: u8>(&self) -> ::my_crate::Wrap<B>;
        }
    })
    .unwrap();
    assert_eq!(leak_trait(&item), Ok(vec![]));

    // The same local generics in several methods are fine.
    let item: ItemTrait = parse2(quote! {
        pub trait MyTrait {
            fn f<const B: bool>(&self) -> Wrap<B>;
            fn g<const B: bool>(&self, x: Wrap<B>);
        }
    })
    .unwrap();
    assert_eq!(leak_trait(&item), Ok(vec!["Wrap < B >".to_string()]));
}

#[test]
fn limit_const_param_ty_alias() {
    // The type of a const parameter cannot be a projection, so a relative type alias cannot be
    // leaked there.
    let item: ItemTrait = parse2(quote! {
        pub trait MyTrait<const N: Idx> {
            fn f(&self) -> RelType;
        }
    })
    .unwrap();
    assert_eq!(
        leak_trait(&item),
        Err("use absolute path: 'Idx'".to_string())
    );

    // The suggested fix works: an absolute path to the alias does not need to be leaked.
    let abs_item: ItemTrait = parse2(quote! {
        pub trait MyTrait<const N: ::my_crate::Idx> {
            fn f(&self) -> RelType;
        }
    })
    .unwrap();
    assert_eq!(leak_trait(&abs_item), Ok(vec!["RelType".to_string()]));

    // `ConstMarkers` cannot encode an alias (relative or absolute), because it cannot know the
    // underlying primitive type.
    let markers = ConstMarkers::new(parse2(quote!(crate::markers)).unwrap());
    let ty: Type = parse2(quote!(Idx)).unwrap();
    let expected =
        Err("unsupported type of const parameter: 'Idx' (use a primitive type)".to_string());
    let result = markers.encode_const(&parse2(quote!(0)).unwrap(), &ty);
    assert_eq!(
        result.map(|ty| ty_string(&ty)).map_err(|e| e.to_string()),
        expected
    );
    let result = markers.encode_generics_params_to_ty(&item.generics.params);
    assert_eq!(
        result.map(|ty| ty_string(&ty)).map_err(|e| e.to_string()),
        expected
    );
}

#[test]
fn limit_plain_encoders_usize_only() {
    // `encode_generics_params_to_ty()` and `encode_generics_args_to_ty()` encode const generics as
    // array lengths, which must be `usize`. For other types they produce `[(); B]` or
    // `[(); true]`, which rustc rejects; `ConstMarkers` must be used instead.
    let generics: Generics = parse2(quote!(<const B: bool>)).unwrap();
    let ty = encode_generics_params_to_ty(&generics.params);
    assert_eq!(ty_string(&ty), expected_ty(quote!(([(); B],))));

    let args: AngleBracketedGenericArguments = parse2(quote!(<true>)).unwrap();
    let ty = encode_generics_args_to_ty(&args.args);
    assert_eq!(ty_string(&ty), expected_ty(quote!(([(); true],))));

    let markers = ConstMarkers::new(parse2(quote!(crate::markers)).unwrap());
    let ty = markers
        .encode_generics_params_to_ty(&generics.params)
        .unwrap();
    assert_eq!(
        ty_string(&ty),
        expected_ty(quote!((crate::markers::ConstBool<B>,)))
    );
    let ty = markers
        .encode_generics_args_to_ty(&args.args, &generics.params)
        .unwrap();
    assert_eq!(
        ty_string(&ty),
        expected_ty(quote!((crate::markers::ConstBool<true>,)))
    );
}

// ── holes ────────────────────────────────────────────────────────────────────

#[test]
fn hole_omitted_default_arg() {
    // HOLE: a defaulted const parameter omitted in the Referrer's arguments.
    //
    // The Leaker encodes all parameters, but the Referrer encodes only the given arguments, so
    // the two encodings differ and the Repeater's implementation is not found.
    // Expected after fix: missing arguments are filled from the parameters' defaults, giving
    // `(u8, [(); 3])`.
    let generics: Generics = parse2(quote!(<T, const N: usize = 3>)).unwrap();
    let args: AngleBracketedGenericArguments = parse2(quote!(<u8>)).unwrap();
    let markers = ConstMarkers::new(parse2(quote!(crate::markers)).unwrap());

    let params_ty = markers
        .encode_generics_params_to_ty(&generics.params)
        .unwrap();
    let args_ty = markers
        .encode_generics_args_to_ty(&args.args, &generics.params)
        .unwrap();
    assert_eq!(ty_string(&params_ty), expected_ty(quote!((T, [(); N]))));
    assert_eq!(ty_string(&args_ty), expected_ty(quote!((u8,))));
}

#[test]
fn hole_assoc_binding_arg() {
    // HOLE: an associated binding in the arguments cannot be encoded. `ConstMarkers` returns
    // `EncodeError::UnsupportedArgument`, and the plain `encode_generics_args_to_ty()` panics.
    //
    // Expected after fix: bindings are skipped when encoding the arguments, giving `(u8,)`.
    let generics: Generics = parse2(quote!(<T>)).unwrap();
    let args: AngleBracketedGenericArguments = parse2(quote!(<u8, Item = u8>)).unwrap();
    let markers = ConstMarkers::new(parse2(quote!(crate::markers)).unwrap());

    let result = markers.encode_generics_args_to_ty(&args.args, &generics.params);
    assert_eq!(
        result.map(|ty| ty_string(&ty)).map_err(|e| e.to_string()),
        Err("unsupported generic argument: 'Item = u8'".to_string())
    );

    let result = std::panic::catch_unwind(|| encode_generics_args_to_ty(&args.args));
    let payload = result.unwrap_err();
    assert_eq!(
        payload.downcast_ref::<String>().map(String::as_str),
        Some("unsupported generic argument: 'Item = u8'")
    );
}

#[test]
fn hole_bare_const_in_bound() {
    // HOLE: a bare relative const in a trait bound is parsed as a type, and leaked as a type
    // (generating `type Type = SIZE;`, which does not compile).
    //
    // syn cannot distinguish `SIZE` (a const) from a type without name resolution.
    // Expected after fix: rejected with "use absolute path" like `{ SIZE }`
    // (see `limit_const_in_bound`).
    let item: ItemTrait = parse2(quote! {
        pub trait MyTrait: crate::l::Sup<SIZE> {
            fn f(&self);
        }
    })
    .unwrap();
    assert_eq!(leak_trait(&item), Ok(vec!["SIZE".to_string()]));

    // The same applies to bounds of method-level generics.
    let item: ItemTrait = parse2(quote! {
        pub trait MyTrait {
            fn f<T: crate::l::Sup<SIZE>>(&self);
        }
    })
    .unwrap();
    assert_eq!(leak_trait(&item), Ok(vec!["SIZE".to_string()]));
}

#[test]
fn hole_shadowed_primitive() {
    // HOLE: the type of a const parameter is recognized only by its name, so a shadowed primitive
    // name selects the wrong marker (`ConstU8` for `const X: u8` where `type u8 = i32;`).
    //
    // Not fixable without name resolution; the generated code fails to compile instead of
    // silently resolving to another type.
    let item: ItemTrait = parse2(quote! {
        pub trait MyTrait<const X: u8> {
            fn f(&self) -> Wrap<X>;
        }
    })
    .unwrap();
    assert_eq!(leak_trait(&item), Ok(vec!["Wrap < X >".to_string()]));

    let markers = ConstMarkers::new(parse2(quote!(crate::markers)).unwrap());
    let ty = markers
        .encode_generics_params_to_ty(&item.generics.params)
        .unwrap();
    assert_eq!(
        ty_string(&ty),
        expected_ty(quote!((crate::markers::ConstU8<X>,)))
    );
}

#[test]
fn hole_plain_args_encoder_bare_const() {
    // HOLE: without the parameters, `encode_generics_args_to_ty()` cannot tell that a bare `N`
    // (parsed by syn as a type argument) is a const, and encodes it as the type `N` instead of
    // `[(); N]`, so it does not match `encode_generics_params_to_ty()`.
    //
    // Not fixable without the parameters; `encode_generics_args_to_ty_with()` (or
    // `ConstMarkers::encode_generics_args_to_ty()`) with the parameters handles it.
    let generics: Generics = parse2(quote!(<T, const N: usize>)).unwrap();
    let args: AngleBracketedGenericArguments = parse2(quote!(<u8, N>)).unwrap();

    let ty = encode_generics_args_to_ty(&args.args);
    assert_eq!(ty_string(&ty), expected_ty(quote!((u8, N))));

    let ty = encode_generics_args_to_ty_with(&args.args, Some(&generics.params), |expr, _| {
        parse2(quote!([(); #expr])).unwrap()
    });
    assert_eq!(ty_string(&ty), expected_ty(quote!((u8, [(); N]))));
}

#[test]
fn hole_args_not_checked_against_params() {
    // HOLE: arguments are paired with parameters by position without checking their number, so
    // an extra argument is encoded as is, and the encoding silently differs from the parameters'
    // encoding.
    //
    // Expected after fix: a mismatch is reported (e.g. by a panic or a `Result`).
    let generics: Generics = parse2(quote!(<const B: bool>)).unwrap();
    let args: AngleBracketedGenericArguments = parse2(quote!(<true, u8>)).unwrap();
    let markers = ConstMarkers::new(parse2(quote!(crate::markers)).unwrap());

    let ty = markers
        .encode_generics_args_to_ty(&args.args, &generics.params)
        .unwrap();
    assert_eq!(
        ty_string(&ty),
        expected_ty(quote!((crate::markers::ConstBool<true>, u8)))
    );
}

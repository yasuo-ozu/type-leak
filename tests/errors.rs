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
        Err(
            "'Wrap < B >' is used with different method-local generics, which cannot be leaked"
                .to_string()
        )
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

    // The same type with and without local generics (here `B` is a type in `g`).
    for item in [
        parse2::<ItemTrait>(quote! {
            pub trait MyTrait {
                fn f<const B: bool>(&self) -> Wrap<B>;
                fn g(&self) -> Wrap<B>;
            }
        })
        .unwrap(),
        parse2::<ItemTrait>(quote! {
            pub trait MyTrait {
                fn g(&self) -> Wrap<B>;
                fn f<const B: bool>(&self) -> Wrap<B>;
            }
        })
        .unwrap(),
    ] {
        assert_eq!(
            leak_trait(&item),
            Err(
                "'Wrap < B >' is used with different method-local generics, which cannot be leaked"
                    .to_string()
            )
        );
    }

    // The note suggests how to fix it.
    let item: ItemTrait = parse2(quote! {
        pub trait MyTrait {
            fn f<const B: bool>(&self) -> Wrap<B>;
            fn g<const B: u8>(&self) -> Wrap<B>;
        }
    })
    .unwrap();
    let mut leaker = Leaker::from_config(leaker_config());
    let err = leaker
        .intern_with(&item.generics, |v| v.visit_item_trait(&item))
        .err()
        .unwrap();
    assert_eq!(err.1, NotInternableReason::ConflictingLocalGenerics);
    assert_eq!(
        err.note().as_deref(),
        Some("use an absolute path for it, or the same method-local generics wherever it is used")
    );

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

    // `RepeaterSlots` cannot encode an alias (relative or absolute), because it cannot know which
    // slot to use.
    let slots = RepeaterSlots::new(parse2(quote!(crate::Repeater)).unwrap());
    let err = slots
        .encode_generics_params_to_ty(&abs_item.generics.params)
        .unwrap_err();
    assert_eq!(
        err.to_string(),
        "unsupported type of const parameter: ':: my_crate :: Idx'"
    );
}

#[test]
fn limit_plain_encoders_usize_only() {
    // `encode_generics_params_to_ty()` and `encode_generics_args_to_ty()` encode const generics as
    // array lengths, which must be `usize`. For other types they produce `[(); B]` or
    // `[(); true]`, which rustc rejects; `RepeaterSlots` must be used instead.
    let generics: Generics = parse2(quote!(<const B: bool>)).unwrap();
    let ty = encode_generics_params_to_ty(&generics.params);
    assert_eq!(ty_string(&ty), expected_ty(quote!(([(); B],))));

    let args: AngleBracketedGenericArguments = parse2(quote!(<true>)).unwrap();
    let ty = encode_generics_args_to_ty(&args.args);
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
fn limit_plain_args_encoder_without_params() {
    // Without the parameters, `encode_generics_args_to_ty()` encodes the arguments as they are:
    // a bare `N` (parsed by syn as a type argument) is encoded as a type instead of `[(); N]`, and
    // omitted arguments are not filled with the parameters' defaults, so the result may differ
    // from `encode_generics_params_to_ty()`. `encode_generics_args_to_ty_with()` with the
    // parameters handles both.
    let generics: Generics = parse2(quote!(<T, const N: usize, const M: usize = 3>)).unwrap();
    let args: AngleBracketedGenericArguments = parse2(quote!(<u8, N>)).unwrap();

    let ty = encode_generics_args_to_ty(&args.args);
    assert_eq!(ty_string(&ty), expected_ty(quote!((u8, N))));

    let ty = encode_generics_args_to_ty_with(&args.args, Some(&generics.params), |expr, _| {
        parse2(quote!([(); #expr])).unwrap()
    })
    .unwrap();
    assert_eq!(ty_string(&ty), expected_ty(quote!((u8, [(); N], [(); 3]))));
}

// ── holes ────────────────────────────────────────────────────────────────────

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
    // HOLE: the type of a const parameter is recognized as a primitive only by its name, so a
    // shadowed primitive name (e.g. `type u8 = i32;` in the Leaker's context) is accepted as
    // context-free and copied as it is into the Referrer, where it means the real `u8`.
    //
    // Not fixable without name resolution; the Referrer's copy then has another type
    // (`const M: u8` instead of `i32`), which fails to compile rather than resolving silently.
    let item: ItemTrait = parse2(quote! {
        pub trait MyTrait {
            fn f<const M: u8>(&self);
        }
    })
    .unwrap();
    assert_eq!(leak_trait(&item), Ok(vec![]));
}

// ── error messages ───────────────────────────────────────────────────────────

/// Runs a Leaker with the default configuration (no allowed paths), returning the error message
/// and note.
fn leak_error_default(item_trait: &ItemTrait) -> Option<(String, Option<String>)> {
    let mut leaker = Leaker::new();
    leaker
        .intern_with(&item_trait.generics, |v| v.visit_item_trait(item_trait))
        .err()
        .map(|e| (e.to_string(), e.note()))
}

fn error(message: &str, note: Option<&str>) -> Option<(String, Option<String>)> {
    Some((message.to_string(), note.map(str::to_string)))
}

#[test]
fn note_suggests_absolute_prelude_path() {
    // A prelude trait in a bound, without `allow_std_prelude()`.
    let item: ItemTrait = parse2(quote! {
        pub trait MyTrait {
            fn f<T: Iterator<Item = u8>>(&self, t: T);
        }
    })
    .unwrap();
    assert_eq!(
        leak_error_default(&item),
        error(
            "use absolute path: 'Iterator < Item = u8 >'",
            Some("use `::core::iter::Iterator` instead of `Iterator`")
        )
    );

    // A prelude item from `alloc`, in a supertrait.
    let item: ItemTrait = parse2(quote! {
        pub trait MyTrait: ToString {}
    })
    .unwrap();
    assert_eq!(
        leak_error_default(&item),
        error(
            "use absolute path: 'ToString'",
            Some("use `::std::string::ToString` instead of `ToString`")
        )
    );

    // The relative part of an `impl Trait` is reported, rather than the whole `impl Trait`.
    let item: ItemTrait = parse2(quote! {
        pub trait MyTrait {
            fn f(&self, x: impl Clone);
        }
    })
    .unwrap();
    assert_eq!(
        leak_error_default(&item),
        error(
            "use absolute path: 'Clone'",
            Some("use `::core::clone::Clone` instead of `Clone`")
        )
    );

    // A 2021-edition prelude item.
    let item: ItemTrait = parse2(quote! {
        pub trait MyTrait: TryFrom<u8> {}
    })
    .unwrap();
    assert_eq!(
        leak_error_default(&item),
        error(
            "use absolute path: 'TryFrom < u8 >'",
            Some("use `::core::convert::TryFrom` instead of `TryFrom`")
        )
    );
}

#[test]
fn note_suggests_absolute_primitive_path() {
    // A primitive in a const expression of a bound, without `allow_primitive()`.
    let item: ItemTrait = parse2(quote! {
        pub trait MyTrait: ::my_crate::Sup<{ usize::MAX }> {}
    })
    .unwrap();
    assert_eq!(
        leak_error_default(&item),
        error(
            "use absolute path: 'usize :: MAX'",
            Some("use `::core::primitive::usize` instead of `usize`")
        )
    );
}

#[test]
fn no_note() {
    // Not a prelude item or a primitive.
    let item: ItemTrait = parse2(quote! {
        pub trait MyTrait: MyBound {}
    })
    .unwrap();
    assert_eq!(
        leak_error_default(&item),
        error("use absolute path: 'MyBound'", None)
    );

    // A prelude name used as the second segment of a relative path.
    let item: ItemTrait = parse2(quote! {
        pub trait MyTrait: my_mod::Clone {}
    })
    .unwrap();
    assert_eq!(
        leak_error_default(&item),
        error("use absolute path: 'my_mod :: Clone'", None)
    );
}

#[test]
fn note_in_syn_error() {
    // The note is rendered as a `note` of the diagnostic; without a note, the message is kept.
    let item: ItemTrait = parse2(quote! {
        pub trait MyTrait: Clone {}
    })
    .unwrap();
    let mut leaker = Leaker::new();
    let err: syn::Error = leaker
        .intern_with(&item.generics, |v| v.visit_item_trait(&item))
        .err()
        .unwrap()
        .into();
    assert_eq!(
        err.to_string(),
        "use absolute path: 'Clone'\n\n  = note: use `::core::clone::Clone` instead of `Clone`\n"
    );

    let item: ItemTrait = parse2(quote! {
        pub trait MyTrait: MyBound {}
    })
    .unwrap();
    let mut leaker = Leaker::new();
    let err: syn::Error = leaker
        .intern_with(&item.generics, |v| v.visit_item_trait(&item))
        .err()
        .unwrap()
        .into();
    assert_eq!(err.to_string(), "use absolute path: 'MyBound'");
}

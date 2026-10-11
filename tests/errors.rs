//! Tests for how `type-leak` reports inputs it cannot handle.
//!
//! - `limit_*`: inputs which cannot be leaked on stable Rust. `type-leak` must reject them with
//!   [`NotInternableError`].
//! - `limit_*`, recording behavior: inputs which cannot be handled exactly without name resolution
//!   (e.g. shadowed primitive names), and how they are detected.

use std::collections::BTreeSet;
use syn::*;
use template_quote::quote;
use type_leak::*;

fn leaker_config() -> LeakerConfig {
    let mut config = LeakerConfig::new();
    config.allow_primitive().allow_crate();
    config
}

/// Runs a Leaker on `item_trait`, returning the leaked types or the error message.
fn leak_trait(item_trait: &ItemTrait) -> std::result::Result<BTreeSet<String>, String> {
    let mut leaker = Leaker::from_config(leaker_config());
    leaker
        .intern_trait(item_trait, true)
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
    assert_eq!(leak_trait(&item), Ok(BTreeSet::new()));
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
    assert_eq!(
        leak_trait(&abs_item),
        Ok(vec!["RelType".to_string()].into_iter().collect())
    );

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

// ── holes ────────────────────────────────────────────────────────────────────

#[test]
fn bare_identifier_in_bound_is_leaked_as_a_type() {
    // A bare relative identifier in a trait bound (e.g. `SIZE` in `Sup<SIZE>`) is parsed as a
    // type by syn, though it may be a const. There is no type to intern it in (unlike
    // `Wrap<SIZE>`, which is interned as a whole, see `tests/reduce_roots.rs`), so it is leaked as
    // a type whatever its name is. If it is a const, the result does not compile: it should be
    // written as `Sup<{ SIZE }>`, which is an error (see `limit_const_in_bound`), or with an
    // absolute path.
    for item in [
        parse2::<ItemTrait>(quote! {
            pub trait MyTrait: crate::l::Sup<SIZE> {
                fn f(&self);
            }
        })
        .unwrap(),
        parse2::<ItemTrait>(quote! {
            pub trait MyTrait {
                fn f<T: crate::l::Sup<SIZE>>(&self);
            }
        })
        .unwrap(),
    ] {
        assert_eq!(
            leak_trait(&item),
            Ok(vec!["SIZE".to_string()].into_iter().collect())
        );
    }

    // So is a relative type.
    let item: ItemTrait = parse2(quote! {
        pub trait MyTrait: crate::l::Sup<MyType> {
            fn f(&self);
        }
    })
    .unwrap();
    assert_eq!(
        leak_trait(&item),
        Ok(vec!["MyType".to_string()].into_iter().collect())
    );

    // The suggested fix works.
    let item: ItemTrait = parse2(quote! {
        pub trait MyTrait: crate::l::Sup<{ ::my_crate::SIZE }> {
            fn f(&self);
        }
    })
    .unwrap();
    assert_eq!(leak_trait(&item), Ok(BTreeSet::new()));
}

#[test]
fn limit_shadowed_primitive() {
    // The type of a const parameter is recognized as a primitive only by its name, so a shadowed
    // primitive name (e.g. `type u8 = i32;` in the Leaker's context) is accepted and copied as it
    // is into the Referrer, where it means the real `u8`. This cannot be known without name
    // resolution, so `Leaker::assertions()` checks it at compile time in the Leaker's context
    // (it fails with "mismatched types" if the name is shadowed).
    let item: ItemTrait = parse2(quote! {
        pub trait MyTrait {
            fn f<const M: u8>(&self);
        }
    })
    .unwrap();
    assert_eq!(leak_trait(&item), Ok(BTreeSet::new()));

    let mut leaker = Leaker::new();
    leaker.intern_trait(&item, true).unwrap();
    let expected = quote! {
        #[allow(dead_code, non_snake_case)]
        const _: () = {
            fn __type_leak_assert_u8_is_primitive(x: &u8) -> &::core::primitive::u8 { x }
        };
    };
    let (actual, expected): (Item, Item) = (
        parse2(leaker.assertions()).unwrap(),
        parse2(expected).unwrap(),
    );
    assert_eq!(quote!(#actual).to_string(), quote!(#expected).to_string());
}

// ── error messages ───────────────────────────────────────────────────────────

/// Runs a Leaker with the default configuration (no allowed paths), returning the error message
/// and note.
fn leak_error_default(item_trait: &ItemTrait) -> Option<(String, Option<String>)> {
    let mut leaker = Leaker::new();
    leaker
        .intern_trait(item_trait, true)
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
    let err: syn::Error = leaker.intern_trait(&item, true).err().unwrap().into();
    assert_eq!(
        err.to_string(),
        "use absolute path: 'Clone'\n\n  = note: use `::core::clone::Clone` instead of `Clone`\n"
    );

    let item: ItemTrait = parse2(quote! {
        pub trait MyTrait: MyBound {}
    })
    .unwrap();
    let mut leaker = Leaker::new();
    let err: syn::Error = leaker.intern_trait(&item, true).err().unwrap().into();
    assert_eq!(err.to_string(), "use absolute path: 'MyBound'");
}

#[test]
fn error_to_tokens() {
    // The error is emitted as `compile_error!` with the note, as `syn::Error` is.
    let item: ItemTrait = parse2(quote! {
        pub trait MyTrait: Clone {}
    })
    .unwrap();
    let mut leaker = Leaker::new();
    let err = leaker.intern_trait(&item, true).err().unwrap();
    let tokens = quote!(#err).to_string();
    assert!(tokens.contains("compile_error"), "{}", tokens);
    assert!(
        tokens.contains("= note: use `::core::clone::Clone` instead of `Clone`"),
        "{}",
        tokens
    );
    assert_eq!(tokens, syn::Error::from(err).to_compile_error().to_string());
}

#[test]
fn note_for_trait_bounds() {
    // The offending tokens may be a whole trait bound rather than a path.
    let item: ItemTrait = parse2(quote! {
        pub trait MyTrait: for<'a> Fn(&'a u8) {}
    })
    .unwrap();
    assert_eq!(
        leak_error_default(&item),
        error(
            "use absolute path: 'for < 'a > Fn (& 'a u8)'",
            Some("use `::core::ops::Fn` instead of `Fn`")
        )
    );
    let item: ItemTrait = parse2(quote! {
        pub trait MyTrait: (Clone) {}
    })
    .unwrap();
    assert_eq!(
        leak_error_default(&item),
        error(
            "use absolute path: '(Clone)'",
            Some("use `::core::clone::Clone` instead of `Clone`")
        )
    );
}

#[test]
fn all_relative_paths_are_reported() {
    // Every relative path which cannot be leaked is reported at once, each as a conflict.
    let item: ItemTrait = parse2(quote! {
        pub trait MyTrait: Clone + MyBound {
            fn f(&self, x: MyType<'_>);
        }
    })
    .unwrap();
    let mut leaker = Leaker::new();
    let err = leaker.intern_trait(&item, true).err().unwrap();
    let conflicts = match &err {
        NotInternableError::RelativePath(conflicts) => conflicts,
        _ => panic!("{:?}", err),
    };
    let strings = |tokens: &Vec<proc_macro2::TokenStream>| -> Vec<String> {
        tokens.iter().map(|t| t.to_string()).collect()
    };
    assert_eq!(
        conflicts
            .iter()
            .map(|c| (strings(&c.must), strings(&c.impossible), c.is_own))
            .collect::<Vec<_>>(),
        vec![
            (vec!["Clone".to_string()], vec!["Clone".to_string()], true),
            (
                vec!["MyBound".to_string()],
                vec!["MyBound".to_string()],
                true
            ),
            (
                vec!["MyType < '_ >".to_string()],
                vec!["'_".to_string()],
                true
            ),
        ]
    );
    assert_eq!(
        err.to_string(),
        "use absolute path: 'Clone'\nuse absolute path: 'MyBound'\nuse absolute path: 'MyType < '_ >'"
    );
    // The first error and its note.
    assert_eq!(
        err.note().as_deref(),
        Some("use `::core::clone::Clone` instead of `Clone`")
    );
    // Each error is a separate `syn::Error`, with its own note.
    let syn_err: syn::Error = err.into();
    assert_eq!(
        syn_err.into_iter().map(|e| e.to_string()).collect::<Vec<_>>(),
        vec![
            "use absolute path: 'Clone'\n\n  = note: use `::core::clone::Clone` instead of `Clone`\n",
            "use absolute path: 'MyBound'",
            "use absolute path: 'MyType < '_ >'\n\n  = note: it cannot be leaked as a whole because of `'_`\n",
        ]
    );
}

#[test]
fn notes_explain_conflicts() {
    // The notes of an error tell why the path cannot be leaked as a whole, and which other paths
    // inside it depend on the type context.
    let item: ItemTrait = parse2(quote! {
        pub trait MyTrait {
            fn f(&self, x: Box<dyn RelA + RelB<'_>>);
        }
    })
    .unwrap();
    let err = Leaker::new().intern_trait(&item, true).err().unwrap();
    // The suggestion of the absolute path comes first.
    assert_eq!(
        err.note().as_deref(),
        Some("use `::std::boxed::Box` instead of `Box`")
    );
    assert_eq!(
        syn::Error::from(err).to_string(),
        "use absolute path: 'Box < dyn RelA + RelB < '_ > >'\n\n\
         \x20 = note: use `::std::boxed::Box` instead of `Box`\n\
         \x20 = note: it cannot be leaked as a whole because of `'_`\n\
         \x20 = note: `RelA` also depends on the type context\n\
         \x20 = note: `RelB < '_ >` also depends on the type context\n"
    );
}

#[test]
fn a_conflict_that_is_not_about_the_path_has_a_headline_of_its_own() {
    // E0195: the fault is the lifetime that the copy would leave unconstrained, which an absolute
    // path is only one way to fix (the README has the others).
    let item: ItemTrait = parse2(quote! {
        pub trait MyTrait {
            fn f<'b>(&self, x: Wrap<'b>) -> &'b u8;
            fn g<'c>(&self, x: &'c dyn RelTrait) -> &'c u8;
        }
    })
    .unwrap();
    let err = Leaker::new().intern_trait(&item, true).err().unwrap();
    assert_eq!(
        err.to_string(),
        "`Wrap < 'b >` leaves the lifetime `'b` of the method unconstrained in the copy (E0195)\n\
         `RelTrait` leaves the lifetime `'c` of the method unconstrained in the copy (E0195)"
    );
    // The note says why.
    let items: Vec<_> = err.items().collect();
    assert!(items[0].note().unwrap().starts_with("it names `'b`"));
    assert!(items[1]
        .note()
        .unwrap()
        .starts_with("the lifetime of the trait object defaults to `'c`"));

    // A type that uses a generic whose bound mentions `Self`, where `Self` cannot be leaked: the
    // fault is the bound.
    let item: ItemTrait = parse2(quote! {
        pub trait MyTrait {
            fn f<T: ::core::cmp::PartialEq<Self>>(&self, x: Wrap<T>);
        }
    })
    .unwrap();
    let err = Leaker::new().intern_trait(&item, false).err().unwrap();
    assert_eq!(
        err.to_string(),
        "`Wrap < T >` has a generic whose bound mentions `Self`, which cannot be leaked"
    );
    assert_eq!(
        err.note().as_deref(),
        Some("the bounds of its generics mention `Self`, which cannot be leaked")
    );

    // A relative path that the type is leaked as a whole for keeps the headline of the path.
    let item: ItemTrait = parse2(quote! {
        pub trait MyTrait {
            fn f(&self, x: MyType<'_>);
        }
    })
    .unwrap();
    let err = Leaker::new().intern_trait(&item, true).err().unwrap();
    assert_eq!(err.to_string(), "use absolute path: 'MyType < '_ >'");
}

#[test]
fn a_type_that_is_leaked_alone_is_no_cause_of_the_conflict_around_it() {
    // `Self` cannot be leaked here, so `<Self as Tr<u8>>::X` cannot be leaked as a whole, and its
    // trait is relative (`Tr`). `u8` in the arguments of the trait is leaked alone, whatever
    // happens to the type that holds it: it is no error once `Tr` is absolute, so the notes do not
    // name it.
    let leak = |ty: Type| {
        Leaker::new()
            .enter_impl(&Generics::default(), &parse_quote!(B), false, |view| {
                view.visit_type(&ty)
            })
            .err()
            .unwrap()
    };
    let err = leak(parse_quote!(<Self as super::Tr<u8>>::X));
    let items: Vec<_> = err.items().collect();
    assert_eq!(items.len(), 1);
    assert_eq!(
        items[0].message,
        "use absolute path: 'super :: Tr < u8 > :: X'"
    );
    assert_eq!(
        items[0]
            .notes
            .iter()
            .map(|(_, note)| note.as_str())
            .collect::<Vec<_>>(),
        ["it cannot be leaked as a whole because of `Self`"]
    );
    // The paths that are no entry on their own are still listed: the type around `u8` is leaked
    // whole (`Rel`, relative), and what it holds cannot be (`Tr`, with `Self`).
    let err = leak(parse_quote!(Rel<<Self as Tr<u8>>::X>));
    let items: Vec<_> = err.items().collect();
    assert_eq!(items.len(), 1);
    assert_eq!(
        items[0]
            .notes
            .iter()
            .map(|(_, note)| note.as_str())
            .collect::<Vec<_>>(),
        [
            "it cannot be leaked as a whole because of `Self`",
            "`Tr < u8 > :: X` also depends on the type context"
        ]
    );
}

/// README, Bare identifiers and consts: a type with an expression that cannot be leaked as a whole
/// (a closure in an array length) is an error naming the closure, not an entry: a closure cannot
/// be written in the impl of the Repeater. Audit mutant `c_expr_closure_impossible` (the closure
/// is no obstacle, and the array type is an entry).
#[test]
fn a_closure_in_an_array_length_cannot_be_leaked_as_a_whole() {
    let item: ItemTrait = parse_quote! {
        pub trait Tr { fn f(&self, x: [u8; { let c = |a: u8| a; 3 }]); }
    };
    let mut leaker = Leaker::from_config(leaker_config());
    let err = leaker.intern_trait(&item, true).unwrap_err();
    let items: Vec<_> = err.items().collect();
    assert_eq!(items.len(), 1);
    assert_eq!(items[0].message, "use absolute path: 'a'");
    assert_eq!(
        items[0]
            .notes
            .iter()
            .map(|(_, note)| note.as_str())
            .collect::<Vec<_>>(),
        ["it cannot be leaked as a whole because of `| a : u8 | a`"]
    );
    // Without the closure the array type is an entry.
    let item: ItemTrait = parse_quote! {
        pub trait Tr { fn f(&self, x: [u8; { let c = 3; c }]); }
    };
    let mut leaker = Leaker::from_config(leaker_config());
    leaker.intern_trait(&item, true).unwrap();
    assert_eq!(leaker.finish().iter().count(), 1);
}

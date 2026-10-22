//! Regression tests for the analysis of the `Leaker`: the requirements of the leaked types, the
//! module keys, the causes found in trait objects, impls, macros and qualified paths, and the
//! assumptions about primitive types.

mod common;

use common::check_in;
use syn::*;
use template_quote::quote;
use type_leak::{to_syn_error, Leaker, LeakerConfig, Referrer, SelfRequirement};

fn s(x: &impl template_quote::ToTokens) -> String {
    quote!(#x).to_string()
}

fn config() -> LeakerConfig {
    LeakerConfig::new()
}

fn leak_trait_with(config: LeakerConfig, item: &ItemTrait) -> std::result::Result<Leaker, String> {
    let mut leaker = Leaker::from_config(config);
    leaker
        .intern_trait(item, true)
        .map_err(|e| to_syn_error(&e).to_string())?;
    Ok(leaker)
}

fn leak_trait(item: ItemTrait) -> Referrer {
    leak_trait_with(config(), &item).unwrap().finish()
}

fn leak_module_with(config: LeakerConfig, item: &ItemMod) -> std::result::Result<Leaker, String> {
    let mut leaker = Leaker::from_config(config);
    leaker
        .intern_mod(item)
        .map_err(|e| to_syn_error(&e).to_string())?;
    Ok(leaker)
}

/// The required leaked types with their generics (and where clause). The optional entries (the
/// types with the path of an external crate, see `Referrer::must()`) are left out: these tests are
/// about the required ones, and the optional ones are tested by `optional_entries()`.
fn entries(referrer: &Referrer) -> Vec<(String, String)> {
    referrer
        .iter_with_generics()
        .enumerate()
        .filter(|(ix, _)| referrer.must(*ix))
        .map(|(_, (ty, generics))| (s(ty), s(&quote!(#generics #{&generics.where_clause}))))
        .collect()
}

/// The optional leaked types with their generics (and where clause).
fn optional_entries(referrer: &Referrer) -> Vec<(String, String)> {
    referrer
        .iter_with_generics()
        .enumerate()
        .filter(|(ix, _)| !referrer.must(*ix))
        .map(|(_, (ty, generics))| (s(ty), s(&quote!(#generics #{&generics.where_clause}))))
        .collect()
}

/// The path of the module of each entry (see `Referrer::module_path()`).
fn modules(referrer: &Referrer) -> Vec<Option<Vec<String>>> {
    (0..referrer.iter().count())
        .map(|ix| {
            referrer.context_module(referrer.context(ix)).map(|m| {
                referrer
                    .module_path(m)
                    .unwrap()
                    .iter()
                    .map(|i| i.to_string())
                    .collect()
            })
        })
        .collect()
}

fn e(ty: &str, generics: &str) -> (String, String) {
    (ty.to_string(), generics.to_string())
}

#[test]
fn self_predicates_are_requirements() {
    // `where Self: Sized` of a method.
    let referrer = leak_trait(parse_quote! {
        trait Tr {
            fn f(&self) -> Wrap<Self> where Self: ::core::marker::Sized;
        }
    });
    assert_eq!(
        entries(&referrer),
        [e(
            "Wrap < Self >",
            "where Self : :: core :: marker :: Sized"
        )]
    );
    assert!(matches!(
        referrer.context_self(referrer.context(0)),
        Some(SelfRequirement::Bound(_))
    ));

    // The root predicates and those bounding `Self` with a lifetime too: `Self: 'a` is not a
    // predicate of the generics (it is in `context_self_outlives()`, encoded and never declared).
    let referrer = leak_trait(parse_quote! {
        trait Tr<T> where Self: ::core::marker::Send, T: ::core::marker::Sync {
            fn f<'a>(&'a self) -> Wrap<'a, Self> where Self: 'a;
        }
    });
    assert_eq!(
        entries(&referrer),
        [e(
            "Wrap < 'a , Self >",
            "< 'a , T : :: core :: marker :: Sync > where Self : :: core :: marker :: Send"
        )]
    );
    let lifetime: Lifetime = parse_quote!('a);
    assert_eq!(
        referrer.context_self_outlives(referrer.context(0)),
        [lifetime]
    );

    // A type which does not mention `Self` does not get them.
    let referrer = leak_trait(parse_quote! {
        trait Tr {
            fn f(&self) -> Wrap where Self: ::core::marker::Sized;
        }
    });
    assert_eq!(entries(&referrer), [e("Wrap", "")]);
}

#[test]
fn dyn_lifetime_bounds_are_visited() {
    // A generic lifetime is used by the type.
    let referrer = leak_trait(parse_quote! {
        trait Tr {
            fn f<'a>(&self) -> ::std::boxed::Box<dyn RelTr + 'a>;
        }
    });
    assert_eq!(entries(&referrer), [e("dyn RelTr + 'a", "< 'a >")]);
    // The `Box` is an optional entry, with the lifetime among its generics.
    assert_eq!(
        optional_entries(&referrer),
        [e(":: std :: boxed :: Box < dyn RelTr + 'a >", "< 'a >")]
    );
    let mut leaker = Leaker::new();
    let generics = Generics::default();
    // `'_` is impossible.
    let ty: Type = parse_quote!(::p::Wrap<dyn ::p::Tr + '_>);
    let result = check_in(&mut leaker, &ty, &generics);
    assert_eq!(result.impossible.iter().map(s).collect::<Vec<_>>(), ["'_"]);
    // A lifetime which is not a generic is a cause.
    let ty: Type = parse_quote!(::p::Wrap<dyn ::p::Tr + 'b>);
    let result = check_in(&mut leaker, &ty, &generics);
    assert_eq!(result.must.iter().map(s).collect::<Vec<_>>(), ["'b"]);
    assert!(result.impossible.is_empty());
    // A generic one is not.
    let generics: Generics = parse_quote!(<'b>);
    let result = check_in(&mut leaker, &ty, &generics);
    assert!(result.must.is_empty());
}

#[test]
fn reference_to_dyn_defaults_to_its_lifetime() {
    // With an elided lifetime, the trait object cannot be leaked alone.
    let err = leak_trait_with(
        config(),
        &parse_quote! { trait Tr { fn f(&self) -> &dyn RelTr; } },
    )
    .err()
    .unwrap();
    assert!(err.contains("use absolute path: 'RelTr'"), "{}", err);
    assert!(
        err.contains(
            "= note: the lifetime of the trait object defaults to that of the enclosing \
             reference or path\n"
        ),
        "{}",
        err
    );
    // The help says how to name it, among the other fixes.
    assert!(
        err.contains(
            "= help: write `RelTr` with an absolute path (e.g. `::my_crate::RelTr`), allow it \
             with `LeakerConfig::allowed_paths`, or name the lifetime of the trait object (e.g. \
             `&'a dyn Trait` or `dyn Trait + 'a`)\n"
        ),
        "{}",
        err
    );
    for ty in ["&mut dyn RelTr", "&(dyn RelTr)", "&'_ dyn RelTr"] {
        let ty: Type = syn::parse_str(ty).unwrap();
        let item: ItemTrait = parse_quote! { trait Tr { fn f(&self) -> #ty; } };
        assert!(leak_trait_with(config(), &item).is_err(), "{}", s(&ty));
    }
    // Its components can be leaked.
    let referrer = leak_trait(parse_quote! {
        trait Tr { fn f(&self) -> &dyn ::p::Tr<Rel>; }
    });
    assert_eq!(entries(&referrer), [e("Rel", "")]);
    // A named lifetime is inherited by the trait object (or its trait has a bound of its own, which
    // rustc knows and the Leaker does not), so the reference is interned whole.
    let referrer = leak_trait(parse_quote! {
        trait Tr<'a> { fn f(&self) -> &'a dyn RelTr; }
    });
    assert_eq!(entries(&referrer), [e("& 'a dyn RelTr", "< 'a >")]);
    // So is the reference of a lifetime of the method, which the receiver keeps constrained in the
    // copy.
    let referrer = leak_trait(parse_quote! {
        trait Tr { fn f<'a>(&'a self) -> &'a dyn RelTr; }
    });
    assert_eq!(entries(&referrer), [e("& 'a dyn RelTr", "< 'a >")]);
    // Unless the return type mentions it and nothing else keeps it as written: the copy would have
    // it only in a projection, which rustc rejects (E0195).
    let err = leak_trait_with(
        config(),
        &parse_quote! { trait Tr { fn f<'a>(&self, x: &'a dyn RelTr) -> &'a dyn RelTr; } },
    )
    .err()
    .unwrap();
    assert!(
        err.contains(
            "`RelTr` leaves the lifetime `'a` of the method unconstrained in the copy (E0195)"
        ),
        "{}",
        err
    );
    let referrer = leak_trait(parse_quote! {
        trait Tr { fn f<'a>(&'a self) -> &'a (dyn RelTr + 'a); }
    });
    assert_eq!(entries(&referrer), [e("dyn RelTr + 'a", "< 'a >")]);
    // No argument mentions the lifetime (it is in the return type only: early-bound in the
    // original and in the copy alike, which has it only in a projection, so rustc accepts the
    // implementation): old, an error, as the E0195 above; new, the reference or the path is leaked.
    let referrer = leak_trait(parse_quote! {
        trait Tr { fn f<'a>(&self) -> &'a dyn RelTr; }
    });
    assert_eq!(entries(&referrer), [e("& 'a dyn RelTr", "< 'a >")]);
    let referrer = leak_trait(parse_quote! {
        trait Tr { fn f<'a>(&self) -> ::core::cell::Ref<'a, dyn RelTr>; }
    });
    assert_eq!(
        entries(&referrer),
        [e(":: core :: cell :: Ref < 'a , dyn RelTr >", "< 'a >")]
    );
    // An argument which mentions it without keeping it as written does not exempt it.
    let err = leak_trait_with(
        config(),
        &parse_quote! { trait Tr { fn f<'a>(&self, x: ::std::vec::Vec<&'a u8>) -> &'a dyn RelTr; } },
    )
    .err()
    .unwrap();
    assert!(err.contains("E0195"), "{}", err);
    // An absolute trait object of `::core` (which is not leaked as a whole) keeps it as written,
    // whatever is inside it, which is leaked alone: old, an error; new, the other reference is
    // leaked.
    let referrer = leak_trait(parse_quote! {
        trait Tr {
            fn f<'a>(&self, x: &'a dyn RelTr, y: &'a dyn ::core::fmt::Debug) -> &'a ();
            fn g<'a>(&self, x: &'a dyn RelTr, y: &'a dyn ::core::iter::Iterator<Item = Rel>) -> &'a ();
        }
    });
    assert_eq!(
        entries(&referrer),
        [e("& 'a dyn RelTr", "< 'a >"), e("Rel", "")]
    );
    // One which may be a cause itself does not (conservative): a relative trait, a bare
    // identifier which may be a const, and an external crate which is not allowed (old: only with
    // `allow_external_crates` unset): the Referrer may replace the reference to it, which then
    // keeps nothing as written.
    for y in [
        "&'a dyn RelTr2",
        "&'a dyn ::core::convert::AsRef<N>",
        "&'a dyn ::p::Tr",
        "&'a dyn ::p::Tr<Item = Rel>",
    ] {
        let y: Type = syn::parse_str(y).unwrap();
        let err = leak_trait_with(
            config(),
            &parse_quote! { trait Tr { fn f<'a>(&self, x: &'a dyn RelTr, y: #y) -> &'a u8; } },
        )
        .err()
        .unwrap();
        assert!(err.contains("E0195"), "{}: {}", s(&y), err);
    }
    // Unless it is allowed, as `::core` is.
    let mut allowed = config();
    allowed.allowed_paths.push(parse_quote!(::p));
    let referrer = leak_trait_with(
        allowed,
        &parse_quote! { trait Tr { fn f<'a>(&self, x: &'a dyn RelTr, y: &'a dyn ::p::Tr) -> &'a (); } },
    )
    .unwrap()
    .finish();
    assert_eq!(entries(&referrer), [e("& 'a dyn RelTr", "< 'a >")]);
    // The default of a `Box` is `'static`, so the trait object is interned alone.
    let referrer = leak_trait(parse_quote! {
        trait Tr { fn f(&self) -> ::std::boxed::Box<dyn RelTr>; }
    });
    assert_eq!(entries(&referrer), [e("dyn RelTr", "")]);
    // `check()` agrees.
    let result = check_in(
        &mut Leaker::new(),
        &parse_quote!(&dyn RelTr),
        &Generics::default(),
    );
    assert_eq!(
        result.impossible.iter().map(s).collect::<Vec<_>>(),
        ["& dyn RelTr", "dyn RelTr"]
    );
}

#[test]
fn references_imply_outlives_bounds() {
    let referrer = leak_trait(parse_quote! {
        trait Tr {
            fn f<'a, T>(&self, t: &'a T) -> Wrap<&'a T>;
        }
    });
    assert_eq!(
        entries(&referrer),
        [e("Wrap < & 'a T >", "< 'a , T : 'a >")]
    );
    let referrer = leak_trait(parse_quote! {
        trait Tr<'a, T> {
            fn f(&self) -> &'a mut Rel<T>;
        }
    });
    // The bound is on the parameter (`T: 'a`), where it is encoded, and not on the referent.
    assert_eq!(
        entries(&referrer),
        [e("& 'a mut Rel < T >", "< 'a , T : 'a >")]
    );
    // Not for a referent which does not mention a type parameter.
    let referrer = leak_trait(parse_quote! {
        trait Tr {
            fn f<'a>(&self) -> Wrap<&'a Rel>;
        }
    });
    assert_eq!(entries(&referrer), [e("Wrap < & 'a Rel >", "< 'a >")]);
}

#[test]
fn module_key_follows_module_relative_requirements() {
    // The meaning of `Self` is module-relative, although the type itself is not.
    let m: ItemMod = parse_quote! {
        mod m {
            pub mod a { pub struct S(::p::W<'x, Self>); }
            pub mod b { pub struct S(::p::W<'x, Self>); }
        }
    };
    let referrer = leak_module_with(config(), &m).unwrap().finish();
    assert_eq!(referrer.iter().count(), 2);
    assert_eq!(
        modules(&referrer),
        [Some(vec!["a".to_string()]), Some(vec!["b".to_string()])]
    );
    // The bounds of the generics are.
    let m: ItemMod = parse_quote! {
        mod m {
            pub mod a { pub trait L {} pub struct S<T: L>(::p::W<'x, T>); }
            pub mod b { pub trait L {} pub struct S<T: L>(::p::W<'x, T>); }
        }
    };
    let referrer = leak_module_with(config(), &m).unwrap().finish();
    assert_eq!(
        entries(&referrer),
        [
            e(":: p :: W < 'x , T >", "< T : L >"),
            e(":: p :: W < 'x , T >", "< T : L >")
        ]
    );
    assert_eq!(
        modules(&referrer),
        [Some(vec!["a".to_string()]), Some(vec!["b".to_string()])]
    );
    // A type which depends on no module is not keyed.
    let m: ItemMod = parse_quote! {
        mod m {
            pub mod a { pub struct S<T: ::p::L>(::p::W<'x, T>); }
            pub mod b { pub struct S<T: ::p::L>(::p::W<'x, T>); }
        }
    };
    let referrer = leak_module_with(config(), &m).unwrap().finish();
    assert_eq!(referrer.iter().count(), 1);
    assert_eq!(modules(&referrer), [None]);
}

#[test]
fn impl_trait_path_is_checked_like_a_bound() {
    let m: ItemMod = parse_quote! {
        mod m { pub struct S; impl Tr for S {} }
    };
    let err = leak_module_with(config(), &m).err().unwrap();
    assert!(err.contains("use absolute path: 'Tr'"), "{}", err);
    let m: ItemMod = parse_quote! {
        mod m { pub struct S; impl Clone for S {} }
    };
    let err = leak_module_with(config(), &m).err().unwrap();
    assert!(
        err.contains("use `::core::clone::Clone` instead of `Clone`"),
        "{}",
        err
    );
    // A trait of the module, or an absolute path, is fine, and its arguments are analyzed.
    let m: ItemMod = parse_quote! {
        mod m {
            pub struct S;
            pub trait Tr<T> {}
            impl Tr<Rel> for S {}
            impl ::p::Tr<Rel2> for S {}
        }
    };
    let referrer = leak_module_with(config(), &m).unwrap().finish();
    assert_eq!(entries(&referrer), [e("Rel", ""), e("Rel2", "")]);
}

#[test]
fn relative_use_is_not_portable_in_nested_modules() {
    let m: ItemMod = parse_quote! {
        mod m {
            pub mod inner {
                pub use super::super::Rel;
                pub use ::p::Abs;
                pub struct Item;
                pub mod deeper { pub struct Deep; }
            }
            pub struct S(
                inner::Rel, inner::Abs, inner::Item, inner::deeper::Deep, inner::Missing,
                self::inner::Item,
            );
        }
    };
    let referrer = leak_module_with(config(), &m).unwrap().finish();
    assert_eq!(
        entries(&referrer),
        [e("inner :: Missing", ""), e("inner :: Rel", "")]
    );
    assert_eq!(modules(&referrer), [Some(vec![]), Some(vec![])]);
}

#[test]
fn type_macros() {
    // `Self` in the tokens cannot be leaked unless `Self` can be interned.
    let item: ItemTrait = parse_quote! {
        trait Tr { fn f(&self) -> ::p::Wrap<::p::m!(Self)>; }
    };
    assert!(leak_trait_with(config(), &item).is_ok());
    let item: ItemStruct = parse_quote! {
        struct S { f: ::p::Wrap<::p::m!(Self)> }
    };
    assert!(Leaker::from_config(config())
        .intern_struct(&item, true)
        .is_ok());
    let err = Leaker::from_config(config())
        .intern_struct(&item, false)
        .err()
        .map(|e| to_syn_error(&e).to_string())
        .unwrap();
    assert!(err.contains("because of `Self`"), "{}", err);
    // In a module, a macro is expanded where it is written, whatever its path is.
    let m: ItemMod = parse_quote! {
        mod m {
            pub mod a { pub struct S(::p::m!(x)); }
            pub mod b { pub struct S(::p::m!(x)); }
        }
    };
    let referrer = leak_module_with(config(), &m).unwrap().finish();
    assert_eq!(
        modules(&referrer),
        [Some(vec!["a".to_string()]), Some(vec!["b".to_string()])]
    );
}

#[test]
fn primitives_are_assumed_when_the_type_is_copied() {
    let mut config = config();
    config.allow_primitive();
    let leaker = leak_trait_with(
        config.clone(),
        &parse_quote! { trait Tr { fn f(&self) -> &u8; } },
    )
    .unwrap();
    assert!(leaker
        .assertions()
        .to_string()
        .contains("__type_leak_assert_u8_is_primitive"));
    let leaker = leak_trait_with(
        config.clone(),
        &parse_quote! { trait Tr { fn f(&self) -> impl ::p::Into<u16>; } },
    )
    .unwrap();
    assert!(leaker
        .assertions()
        .to_string()
        .contains("__type_leak_assert_u16_is_primitive"));
    // Nothing is recorded on an error.
    let mut leaker = Leaker::from_config(config);
    let item: ItemTrait = parse_quote! { trait Tr: Clone { fn f(&self) -> u8; } };
    assert!(leaker.intern_trait(&item, true).is_err());
    assert!(leaker.assertions().is_empty());
}

#[test]
fn raw_identifiers_are_compared_by_name() {
    let referrer = leak_trait(parse_quote! {
        trait Tr<r#T> { fn f(&self) -> Rel<T>; }
    });
    assert_eq!(entries(&referrer), [e("Rel < T >", "< r#T >")]);
    let referrer = leak_trait(parse_quote! {
        trait Tr<T> { fn f(&self) -> Rel<r#T>; }
    });
    assert_eq!(entries(&referrer), [e("Rel < r#T >", "< T >")]);
    let m: ItemMod = parse_quote! {
        mod m { pub struct S<r#T>(Rel<T>); }
    };
    let referrer = leak_module_with(config(), &m).unwrap().finish();
    assert_eq!(entries(&referrer), [e("Rel < T >", "< r#T >")]);
}

#[test]
fn qualified_self_through_generics() {
    let referrer = leak_trait(parse_quote! {
        trait Tr<T> { fn f(&self) -> Wrap<<T>::X>; }
    });
    assert_eq!(entries(&referrer), [e("Wrap < < T > :: X >", "< T >")]);
    // `check()` in the trait agrees, where `Self` is the implementor.
    let generics: Generics = parse_quote!(<T>);
    let mut leaker = Leaker::new();
    let (_, results) = leaker
        .enter_trait(&parse_quote!(Tr), &generics, true, |view| {
            let mut check = |ty: Type| view.check(&ty);
            [
                check(parse_quote!(<T>::X)),
                check(parse_quote!(<Self>::X)),
                check(parse_quote!(<Rel>::X)),
                check(parse_quote!(<U>::X)),
            ]
        })
        .unwrap();
    let [through_param, through_self, relative, undeclared] = results;
    assert!(through_param.must.is_empty() && through_param.impossible.is_empty());
    assert!(through_self.must.is_empty() && through_self.impossible.is_empty());
    // Otherwise it is impossible.
    assert_eq!(
        relative.impossible.iter().map(s).collect::<Vec<_>>(),
        ["Rel"]
    );
    assert_eq!(
        undeclared.impossible.iter().map(s).collect::<Vec<_>>(),
        ["U"]
    );
}

#[test]
fn parameters_are_ordered_by_kind() {
    let referrer = leak_trait(parse_quote! {
        trait Tr<'b, 'a, const N: usize, U, T> { fn f(&self) -> Wrap<'b, 'a, N, U, T>; }
    });
    assert_eq!(
        entries(&referrer),
        [e(
            "Wrap < 'b , 'a , N , U , T >",
            "< 'a , 'b , T , U , const N : usize >"
        )]
    );
}

#[test]
fn notes_of_defaults_and_impl_trait() {
    let err = leak_trait_with(config(), &parse_quote! { trait Tr<T = Rel<Self>> {} })
        .err()
        .unwrap();
    assert!(
        err.contains("a default of a parameter cannot depend on `Self`"),
        "{}",
        err
    );
    let err = leak_trait_with(
        config(),
        &parse_quote! { trait Tr { fn f(&self) -> Wrap<impl Into<u8>>; } },
    )
    .err()
    .unwrap();
    assert!(
        err.contains("use absolute path: 'Wrap < impl Into < u8 > >'"),
        "{}",
        err
    );
    assert!(
        err.contains("it cannot be leaked as a whole because of `impl Into < u8 >`"),
        "{}",
        err
    );
}

//! Tests for `Leaker::allow_external_crates` and the external crates required by leaked types.

use syn::visit_mut::VisitMut;
use syn::*;
use template_quote::quote;
use type_leak::*;

fn leaker(allow_external_crates: bool) -> Leaker {
    let mut config = LeakerConfig::new();
    config
        .allow_primitive()
        .allow_external_crates(allow_external_crates);
    Leaker::from_config(config)
}

fn leak_trait(
    leaker: &mut Leaker,
    item: &ItemTrait,
) -> std::result::Result<(), NotInternableError> {
    leaker
        .intern_with(&item.generics, |v| v.visit_item_trait(item))
        .map(|_| ())
}

/// Prints tokens without whitespace, to compare them with the source.
fn squash(tokens: impl ToString) -> String {
    tokens.to_string().replace(' ', "")
}

/// Returns the leaked types with `must` and the required crates.
fn entries(referrer: &Referrer) -> Vec<(String, bool, Vec<String>)> {
    referrer
        .iter()
        .enumerate()
        .map(|(ix, ty)| {
            (
                squash(quote!(#ty)),
                referrer.must(ix),
                referrer.external_crates_require(ix).to_vec(),
            )
        })
        .collect()
}

fn entries_of(allow: bool, item: ItemTrait) -> Vec<(String, bool, Vec<String>)> {
    let mut leaker = leaker(allow);
    leak_trait(&mut leaker, &item).unwrap();
    entries(&leaker.finish())
}

fn crates(names: &[&str]) -> Vec<String> {
    names.iter().map(|s| s.to_string()).collect()
}

#[test]
fn default_allows_external_crates() {
    assert!(Leaker::new().allow_external_crates);
    assert!(LeakerConfig::new().allow_external_crates);
    let item: ItemTrait = parse_quote! {
        trait T<const N: ::foo::Idx> where u8: ::foo::Tr {
            fn f(&self) -> ::foo::X;
            fn g(&self, a: [u8; ::foo::N]);
        }
    };
    assert_eq!(entries_of(true, item), vec![]);
}

#[test]
fn disallowed_leaks_external_path() {
    let item: ItemTrait = parse_quote!(
        trait T {
            fn f(&self) -> ::foo::X;
        }
    );
    assert_eq!(
        entries_of(false, item),
        vec![("::foo::X".to_string(), false, crates(&["foo"]))]
    );
}

#[test]
fn disallowed_mixed_with_relative_is_must() {
    let item: ItemTrait = parse_quote!(
        trait T {
            fn f(&self) -> ::foo::W<Rel>;
        }
    );
    assert_eq!(
        entries_of(false, item),
        vec![("::foo::W<Rel>".to_string(), true, crates(&["foo"]))]
    );
}

#[test]
fn disallowed_core_is_context_free() {
    let leaker = leaker(false);
    let ty: Type = parse_quote!(::core::option::Option<Rel>);
    let result = leaker.check(&ty, &Generics::default());
    assert_eq!(
        result.must.iter().map(squash).collect::<Vec<_>>(),
        vec!["Rel"]
    );
    let item: ItemTrait = parse_quote!(
        trait T {
            fn f(&self) -> ::core::option::Option<Rel>;
        }
    );
    let entries = entries_of(false, item);
    assert_eq!(entries.len(), 1);
    assert!(entries[0].1 && entries[0].2.is_empty());
}

#[test]
fn disallowed_std_is_external() {
    let item: ItemTrait = parse_quote!(
        trait T {
            fn f(&self) -> ::std::vec::Vec<u8>;
        }
    );
    assert_eq!(
        entries_of(false, item),
        vec![("::std::vec::Vec<u8>".to_string(), false, crates(&["std"]))]
    );
}

#[test]
fn disallowed_crates_in_local_generics() {
    let item: ItemTrait = parse_quote! {
        trait T {
            fn f<U: ::bar::B>(&self) -> ::foo::W<U>;
        }
    };
    // The bound of a local generic must be context-free (here allowed explicitly).
    let mut config = LeakerConfig::new();
    config.allow_external_crates(false);
    config.allowed_paths.push(parse_quote!(::bar));
    let mut leaker = Leaker::from_config(config);
    leak_trait(&mut leaker, &item).unwrap();
    let entries = entries(&leaker.finish());
    assert_eq!(entries.len(), 1);
    assert!(!entries[0].1);
    assert_eq!(entries[0].2, crates(&["bar", "foo"]));
}

#[test]
fn disallowed_crates_are_sorted_and_deduplicated() {
    let item: ItemTrait = parse_quote!(
        trait T {
            fn f(&self) -> (::z::A, ::a::B, ::z::C<::a::D>);
        }
    );
    let entries = entries_of(false, item);
    assert!(entries.iter().all(|(_, _, c)| !c.is_empty()));
    let all: std::collections::BTreeSet<_> = entries.iter().flat_map(|e| e.2.clone()).collect();
    assert_eq!(all.into_iter().collect::<Vec<_>>(), crates(&["a", "z"]));
}

#[test]
fn disallowed_macro_crates() {
    let item: ItemTrait = parse_quote!(
        trait T {
            fn f(&self) -> m!(::foo::X, ::core::Y);
        }
    );
    let entries = entries_of(false, item);
    assert_eq!(entries.len(), 1);
    assert!(entries[0].1);
    assert_eq!(entries[0].2, crates(&["foo"]));
}

#[test]
fn allowed_path_is_context_free() {
    let mut config = LeakerConfig::new();
    config.allow_external_crates(false);
    config.allowed_paths.push(parse_quote!(::foo));
    let mut leaker = Leaker::from_config(config);
    let item: ItemTrait = parse_quote! {
        trait T: ::foo::Tr {
            fn f(&self, a: ::foo::X) -> ::bar::Y;
        }
    };
    leak_trait(&mut leaker, &item).unwrap();
    assert_eq!(
        entries(&leaker.finish()),
        vec![("::bar::Y".to_string(), false, crates(&["bar"]))]
    );
}

fn error_of(allow: bool, item: ItemTrait) -> NotInternableError {
    let mut leaker = leaker(allow);
    leak_trait(&mut leaker, &item).err().unwrap()
}

#[test]
fn not_portable_external_bound() {
    let err = error_of(
        false,
        parse_quote!(
            trait T: ::foo::Tr {}
        ),
    );
    assert!(matches!(err, NotInternableError::NotPortable(_)));
    assert_eq!(
        err.to_string(),
        "use a path starting with `$crate`: ':: foo :: Tr'"
    );
    assert_eq!(
        err.note().as_deref(),
        Some(
            "re-export it in this crate (e.g. `#[doc(hidden)] pub use ::foo;`) and refer to it \
             with a path starting with `$crate`"
        )
    );
}

#[test]
fn not_portable_prelude_bound() {
    let err = error_of(
        false,
        parse_quote!(
            trait T: Clone {}
        ),
    );
    assert!(matches!(err, NotInternableError::NotPortable(_)));
    assert_eq!(
        err.to_string(),
        "use a path starting with `$crate`: 'Clone'"
    );
    assert_eq!(
        err.note().as_deref(),
        Some("use `::core::clone::Clone` instead of `Clone`")
    );
}

#[test]
fn not_portable_value() {
    let err = error_of(
        false,
        parse_quote!(
            trait T<const N: usize = { Vec::<u8>::new().len() }> {}
        ),
    );
    assert!(matches!(err, NotInternableError::NotPortable(_)));
    let note = err.note().unwrap();
    assert!(note.contains("pub use ::std::vec::Vec;"), "{}", note);
    assert!(note.contains("`$crate`"), "{}", note);
    let syn_err: syn::Error = err.into();
    assert!(syn_err.to_string().contains("= note: re-export it"));
}

#[test]
fn relative_path_message_with_flag_true() {
    let err = error_of(
        true,
        parse_quote!(
            trait T: Clone {}
        ),
    );
    assert!(matches!(err, NotInternableError::RelativePath(_)));
    assert_eq!(err.to_string(), "use absolute path: 'Clone'");
}

#[test]
fn not_portable_external_const_param_type() {
    let err = error_of(
        false,
        parse_quote!(
            trait T<const N: ::foo::Idx> {}
        ),
    );
    assert!(matches!(err, NotInternableError::NotPortable(_)));
}

fn referrer_with_foo() -> Referrer {
    let item: ItemTrait = parse_quote! {
        trait T {
            fn f(&self, a: ::foo::X) -> Rel;
        }
    };
    let mut leaker = leaker(false);
    leak_trait(&mut leaker, &item).unwrap();
    leaker.finish()
}

fn expand(referrer: &Referrer, ty: Type) -> String {
    let ty = referrer.expand(ty, |_, ix| parse_quote!(P<#ix>));
    quote!(#ty).to_string()
}

#[test]
fn referrer_replaces_optional_entry_without_available_crates() {
    let referrer = referrer_with_foo();
    let ix_foo = referrer
        .iter()
        .position(|t| squash(quote!(#t)) == "::foo::X")
        .unwrap();
    assert!(!referrer.must(ix_foo));
    let ty: Type = parse_quote!(::foo::X);
    assert_eq!(expand(&referrer, ty), format!("P < {}usize >", ix_foo));
}

#[test]
fn referrer_keeps_optional_entry_with_available_crate() {
    let referrer = referrer_with_foo().with_available_external_crates(["foo"]);
    assert_eq!(expand(&referrer, parse_quote!(::foo::X)), ":: foo :: X");
    // A required entry is always replaced.
    let ix_rel = referrer
        .iter()
        .position(|t| squash(quote!(#t)) == "Rel")
        .unwrap();
    assert!(referrer.must(ix_rel));
    assert_eq!(
        expand(&referrer, parse_quote!(Rel)),
        format!("P < {}usize >", ix_rel)
    );
}

#[test]
fn referrer_replaces_optional_entry_with_missing_crate() {
    let referrer = referrer_with_foo().with_available_external_crates(["bar"]);
    assert_ne!(expand(&referrer, parse_quote!(::foo::X)), ":: foo :: X");
}

#[test]
fn referrer_visitor_respects_available_crates() {
    let referrer = referrer_with_foo().with_available_external_crates(["foo"]);
    let mut item: ItemTrait = parse_quote!(
        trait T {
            fn f(&self, a: ::foo::X) -> Rel;
        }
    );
    referrer
        .into_visitor(|_, ix| parse_quote!(P<#ix>))
        .visit_item_trait_mut(&mut item);
    let s = quote!(#item).to_string();
    assert!(s.contains(":: foo :: X"), "{}", s);
    assert!(s.contains("P <"), "{}", s);
}

#[test]
fn referrer_serialization_round_trip() {
    let item: ItemTrait = parse_quote! {
        trait T {
            fn f<U: ::bar::B>(&self, u: U, x: ::foo::X, r: Rel) -> ::foo::W<U>;
        }
    };
    let mut config = LeakerConfig::new();
    config.allow_external_crates(false);
    config.allowed_paths.push(parse_quote!(::bar));
    let mut leaker = Leaker::from_config(config);
    leak_trait(&mut leaker, &item).unwrap();
    let referrer = leaker.finish();
    let before = entries(&referrer);
    assert!(before.iter().any(|e| !e.1 && !e.2.is_empty()));
    assert!(before.iter().any(|e| e.1));
    let tokens = quote!(#referrer);
    let s = tokens.to_string();
    assert!(s.contains("? @ [foo] :: foo :: X"), "{}", s);
    let parsed: Referrer = parse2(tokens).unwrap();
    assert_eq!(parsed, referrer);
    assert_eq!(entries(&parsed), before);
}

#[test]
fn referrer_plain_entry_syntax() {
    let version = env!("CARGO_PKG_VERSION");
    let referrer: Referrer =
        parse2(quote!(#version ([u8; 3], ? @[foo, bar] X<[u8]>, @[a] Y))).unwrap();
    assert_eq!(
        entries(&referrer),
        vec![
            ("[u8;3]".to_string(), true, vec![]),
            ("X<[u8]>".to_string(), false, crates(&["foo", "bar"])),
            ("Y".to_string(), true, crates(&["a"])),
        ]
    );
}

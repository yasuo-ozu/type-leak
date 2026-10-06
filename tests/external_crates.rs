//! Tests for the types with the absolute path of an external crate (`::foo::X`), which are optional
//! entries of the Referrer: the Leaker's `allowed_paths`, `Referrer::must()`,
//! `Referrer::external_crates_require()` and `Referrer::allow_ambiguous_crate_name()`.

use syn::visit_mut::VisitMut;
use syn::*;
use template_quote::quote;
use type_leak::*;

/// The tag of the format of a serialized Referrer, which it starts with.
fn tag() -> String {
    format!("type-leak-format-{}", type_leak::Referrer::FORMAT_VERSION)
}

fn leaker() -> Leaker {
    let mut config = LeakerConfig::new();
    config.allow_primitive();
    Leaker::from_config(config)
}

/// A Leaker which allows the external crates `allowed` (`::name`).
fn leaker_allowing(allowed: &[&str]) -> Leaker {
    let mut config = LeakerConfig::new();
    config.allow_primitive();
    for name in allowed {
        let ident: Ident = parse_str(name).unwrap();
        config.allowed_paths.push(parse_quote!(::#ident));
    }
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

fn entries_of(item: ItemTrait) -> Vec<(String, bool, Vec<String>)> {
    let mut leaker = leaker();
    leak_trait(&mut leaker, &item).unwrap();
    entries(&leaker.finish())
}

fn crates(names: &[&str]) -> Vec<String> {
    names.iter().map(|s| s.to_string()).collect()
}

fn entry(ty: &str, must: bool, names: &[&str]) -> (String, bool, Vec<String>) {
    (ty.to_string(), must, crates(names))
}

#[test]
fn absolute_types_are_optional_entries_by_default() {
    // Old: nothing was leaked by default (`allow_external_crates` was `true`), and the type was
    // an optional entry only with it unset. New: always an optional entry, as the Referrer's
    // crate may not mean the same crate by the name `foo`.
    let item: ItemTrait = parse_quote!(
        trait T {
            fn f(&self) -> ::foo::X;
        }
    );
    assert_eq!(entries_of(item), vec![entry("::foo::X", false, &["foo"])]);
    // The types in a tuple (which one entry covers), behind a pointer, in a path of `::core`.
    let item: ItemTrait = parse_quote!(
        trait T {
            fn f(
                &self,
                a: (::a::A, &'static ::b::B),
                b: ::core::option::Option<::c::C>,
                c: *const ::d::D,
            );
        }
    );
    assert_eq!(
        entries_of(item),
        vec![
            entry("(::a::A,&'static::b::B)", false, &["a", "b"]),
            entry("::c::C", false, &["c"]),
            entry("::d::D", false, &["d"]),
        ]
    );
}

#[test]
fn absolute_paths_which_are_not_types_are_copied() {
    // A trait bound, a value and the type of a const parameter cannot be leaked, so they are
    // copied as they are (old: with `allow_external_crates` unset, `NotPortable`; by default the
    // same, with nothing leaked). The types in the other positions are optional entries.
    let item: ItemTrait = parse_quote! {
        trait T<const N: ::foo::Idx, U: ::foo::Tr = ::foo::D> : ::foo::Sup where u8: ::foo::Tr {
            fn f(&self, a: [u8; ::foo::N]) -> ::foo::X;
            fn g<V: ::foo::Tr<::foo::Y>>(&self, v: V) where V: ::foo::Tr2;
        }
    };
    assert_eq!(
        entries_of(item),
        vec![
            entry("::foo::D", false, &["foo"]),
            entry("::foo::X", false, &["foo"]),
            entry("::foo::Y", false, &["foo"]),
            entry("[u8;::foo::N]", false, &["foo"]),
        ]
    );
}

#[test]
fn check_does_not_report_absolute_types() {
    // `check()` tells the causes which make a type be interned: an absolute path of an external
    // crate is not one, as the Referrer may not replace it (as before).
    let leaker = leaker();
    let result = leaker.check(&parse_quote!(::foo::W<(Rel,)>), &Generics::default());
    assert_eq!(
        result.must.iter().map(squash).collect::<Vec<_>>(),
        vec!["Rel"]
    );
    assert!(!result.is_own);
    let result = leaker.check(&parse_quote!(::foo::X), &Generics::default());
    assert!(result.must.is_empty() && result.impossible.is_empty());
}

#[test]
fn core_is_never_a_node() {
    let item: ItemTrait = parse_quote!(
        trait T {
            fn f(&self) -> ::core::option::Option<(Rel,)>;
            fn g(&self, a: ::core::primitive::u8, b: ::core::cell::Cell<::core::primitive::u8>);
        }
    );
    let entries = entries_of(item);
    assert_eq!(entries, vec![entry("Rel", true, &[])]);
    // `check()` agrees.
    let leaker = leaker();
    let ty: Type = parse_quote!(::core::option::Option<(Rel,)>);
    let result = leaker.check(&ty, &Generics::default());
    assert_eq!(
        result.must.iter().map(squash).collect::<Vec<_>>(),
        vec!["Rel"]
    );
    // Also with `core` as a crate with another name than the one a path may start with.
    let item: ItemTrait = parse_quote!(
        trait T {
            fn f(&self) -> ::corelike::X;
        }
    );
    assert_eq!(
        entries_of(item),
        vec![entry("::corelike::X", false, &["corelike"])]
    );
}

#[test]
fn std_and_alloc_are_nodes() {
    // Old: `::std::..` was external only with `allow_external_crates` unset. New: like any crate.
    let item: ItemTrait = parse_quote!(
        trait T {
            fn f(&self) -> ::std::vec::Vec<u8>;
            fn g(&self) -> ::alloc::string::String;
        }
    );
    assert_eq!(
        entries_of(item),
        vec![
            entry("::alloc::string::String", false, &["alloc"]),
            entry("::std::vec::Vec<u8>", false, &["std"]),
        ]
    );
    // Unless they are allowed.
    let item: ItemTrait = parse_quote!(
        trait T {
            fn f(&self) -> ::std::vec::Vec<(u8, Rel)>;
            fn g(&self) -> ::alloc::string::String;
        }
    );
    let mut leaker = leaker_allowing(&["std", "alloc"]);
    leak_trait(&mut leaker, &item).unwrap();
    assert_eq!(entries(&leaker.finish()), vec![entry("Rel", true, &[])]);
}

#[test]
fn allowed_absolute_path_is_never_a_node() {
    // `::foo` allows the paths which start with it, by their segments (not `::foobar`), and not
    // the relative paths (`foo::X`, which are leaked as they depend on the type context).
    let mut leaker = leaker_allowing(&["foo"]);
    let item: ItemTrait = parse_quote! {
        trait T {
            fn f(&self, a: ::foo::X, b: ::foo::inner::W<(Rel,)>, c: ::foobar::Y, d: foo::Z);
        }
    };
    leak_trait(&mut leaker, &item).unwrap();
    assert_eq!(
        entries(&leaker.finish()),
        vec![
            entry("Rel", true, &[]),
            entry("foo::Z", true, &[]),
            entry("::foobar::Y", false, &["foobar"]),
        ]
    );
    // An allowed relative path allows the relative paths only: an absolute one is a node.
    let mut config = LeakerConfig::new();
    config.allowed_paths.push(parse_quote!(my));
    let mut leaker = Leaker::from_config(config);
    let item: ItemTrait = parse_quote! {
        trait T {
            fn f(&self, a: my::X, b: ::my::Y);
        }
    };
    leak_trait(&mut leaker, &item).unwrap();
    assert_eq!(
        entries(&leaker.finish()),
        vec![entry("::my::Y", false, &["my"])]
    );
    // The more specific path may be allowed alone.
    let mut config = LeakerConfig::new();
    config.allowed_paths.push(parse_quote!(::foo::inner));
    let mut leaker = Leaker::from_config(config);
    let item: ItemTrait = parse_quote! {
        trait T {
            fn f(&self, a: ::foo::inner::X, b: ::foo::Y);
        }
    };
    leak_trait(&mut leaker, &item).unwrap();
    assert_eq!(
        entries(&leaker.finish()),
        vec![entry("::foo::Y", false, &["foo"])]
    );
    // `check()` of the same.
    let ty: Type = parse_quote!(::foo::W<(Rel,)>);
    let mut config = LeakerConfig::new();
    config.allowed_paths.push(parse_quote!(::foo));
    let result = Leaker::from_config(config).check(&ty, &Generics::default());
    assert_eq!(
        result.must.iter().map(squash).collect::<Vec<_>>(),
        vec!["Rel"]
    );
}

#[test]
fn the_context_of_an_optional_entry_is_that_of_its_generics() {
    // The bound of a local generic is copied with the type using it, so it requires its crates.
    let item: ItemTrait = parse_quote! {
        trait T {
            fn f<U: ::core::convert::Into<::bar::B>>(&self) -> ::foo::W<U>;
        }
    };
    let entries = entries_of(item);
    assert_eq!(entries.len(), 2);
    assert!(entries.iter().all(|e| !e.1));
    let find = |ty: &str| entries.iter().find(|e| e.0 == ty).unwrap().2.clone();
    assert_eq!(find("::bar::B"), crates(&["bar"]));
    assert_eq!(find("::foo::W<U>"), crates(&["bar", "foo"]));

    // A bound with a path to an external crate is not an error (old, with
    // `allow_external_crates` unset: `NotPortable`), it is copied as it is.
    let item: ItemTrait = parse_quote! {
        trait T {
            fn f<U: ::bar::B>(&self) -> ::foo::W<U>;
        }
    };
    assert_eq!(
        entries_of(item),
        vec![entry("::foo::W<U>", false, &["bar", "foo"])]
    );
}

#[test]
fn crates_are_sorted_and_deduplicated() {
    let item: ItemTrait = parse_quote!(
        trait T {
            fn f(&self) -> (::z::A, ::a::B, ::z::C<::a::D>);
        }
    );
    let entries = entries_of(item);
    assert!(entries.iter().all(|(_, _, c)| !c.is_empty()));
    let all: std::collections::BTreeSet<_> = entries.iter().flat_map(|e| e.2.clone()).collect();
    assert_eq!(all.into_iter().collect::<Vec<_>>(), crates(&["a", "z"]));
    // One cut covers the tuple (all the three are in it).
    let item: ItemTrait = parse_quote!(
        trait T {
            fn f(&self) -> ::z::C<::a::D, ::z::A>;
        }
    );
    assert_eq!(
        entries_of(item),
        vec![entry("::z::C<::a::D,::z::A>", false, &["a", "z"])]
    );
}

#[test]
fn type_macros_are_required() {
    // A type macro is a cause itself, so it is required, which names crates.
    let item: ItemTrait = parse_quote!(
        trait T {
            fn f(&self) -> m!(::foo::X, ::core::Y);
        }
    );
    let entries = entries_of(item);
    assert_eq!(entries.len(), 1);
    assert!(entries[0].1);
    assert_eq!(entries[0].2, crates(&["foo"]));
}

#[test]
fn an_absolute_type_with_a_relative_type_in_it() {
    // Old, with `allow_external_crates` unset: `::foo::W<(Rel,)>` was required (a cause in it is
    // relative) and the only entry, replacing `Rel` in it; by default (set) `Rel` only. New: both
    // are entries, `Rel` is required and the type containing it is an optional entry, which the
    // Referrer replaces as a whole if it does not trust the name of the crate (and leaves, replacing
    // `Rel` in it, otherwise).
    let item: ItemTrait = parse_quote!(
        trait T {
            fn f(&self) -> ::foo::W<(Rel,)>;
        }
    );
    assert_eq!(
        entries_of(item),
        vec![
            entry("Rel", true, &[]),
            entry("::foo::W<(Rel,)>", false, &["foo"]),
        ]
    );
    // A bare identifier in it may be a const, which cannot be leaked alone, so the type is leaked
    // as a whole (as it is without the crate, and in either case it is required).
    let item: ItemTrait = parse_quote!(
        trait T {
            fn f(&self) -> ::foo::W<Rel>;
        }
    );
    assert_eq!(
        entries_of(item),
        vec![entry("::foo::W<Rel>", true, &["foo"])]
    );
    // A required entry with an absolute path in it is the only one, which names the crate.
    let item: ItemTrait = parse_quote!(
        trait T {
            fn f(&self) -> Rel<::foo::X>;
        }
    );
    assert_eq!(
        entries_of(item),
        vec![entry("Rel<::foo::X>", true, &["foo"])]
    );
    // An absolute type which is covered by a required entry is not an entry.
    let item: ItemTrait = parse_quote!(
        trait T {
            fn f(&self) -> (Rel, ::foo::X);
            fn g(&self) -> (Rel, ::foo::X);
        }
    );
    assert_eq!(
        entries_of(item),
        vec![entry("Rel", true, &[]), entry("::foo::X", false, &["foo"])]
    );
}

#[test]
fn a_type_which_cannot_be_leaked_is_not_an_error() {
    // An elided lifetime and `impl Trait` cannot be leaked, which is not an error for a type
    // which is only an optional entry: it is copied as it is (the Referrer may not replace it).
    let item: ItemTrait = parse_quote!(
        trait T {
            fn f(&self) -> ::foo::W<'_>;
            fn g(&self) -> impl ::foo::Tr;
            fn h(&self, x: &::foo::X) -> ::foo::V<'_, ::foo::Y>;
        }
    );
    assert_eq!(
        entries_of(item),
        vec![
            entry("::foo::X", false, &["foo"]),
            entry("::foo::Y", false, &["foo"]),
        ]
    );
    // The relative one is, as before.
    let item: ItemTrait = parse_quote!(
        trait T {
            fn f(&self) -> Rel<'_>;
        }
    );
    let err = leak_trait(&mut leaker(), &item).err().unwrap();
    assert_eq!(err.to_string(), "use absolute path: 'Rel < '_ >'");
}

/// Entries of the same item with all the crates allowed (a Leaker with no external crates), as
/// the required entries of the default Leaker must be.
fn required_entries_match_allowed_leaker(item: ItemTrait, allowed: &[&str]) {
    let mut leaker = leaker();
    leak_trait(&mut leaker, &item).unwrap();
    let default = leaker.finish();
    let mut leaker = leaker_allowing(allowed);
    leak_trait(&mut leaker, &item).unwrap();
    let reference = leaker.finish();
    // The required entries come first, and are exactly those which allowing the crates leaves, in
    // the same order, contexts included.
    let required: Vec<usize> = (0..default.iter().count())
        .filter(|ix| default.must(*ix))
        .collect();
    assert_eq!(
        required,
        (0..reference.iter().count()).collect::<Vec<_>>(),
        "{}",
        quote!(#item)
    );
    for ix in required {
        assert_eq!(
            squash(quote!(#{default.iter().nth(ix).unwrap()})),
            squash(quote!(#{reference.iter().nth(ix).unwrap()}))
        );
        assert_eq!(default.generics(ix), reference.generics(ix));
        assert_eq!(default.context(ix), reference.context(ix));
    }
    // And the optional entries come after.
    let mut seen_optional = false;
    for ix in 0..default.iter().count() {
        assert!(!(seen_optional && default.must(ix)), "{}", quote!(#item));
        seen_optional |= !default.must(ix);
    }
}

#[test]
fn optional_entries_do_not_change_the_required_ones() {
    let items: Vec<ItemTrait> = vec![
        // A cut which covers several types at once is the same with or without the crate.
        parse_quote!(
            trait T {
                fn f(&self) -> ::foo::W<Rel, Rel2>;
                fn g(&self) -> ::foo::W<Rel, Rel2>;
            }
        ),
        parse_quote!(
            trait T {
                fn f(&self) -> (Rel, Rel2, ::foo::X);
                fn g(&self) -> (Rel, Rel2, ::foo::X);
            }
        ),
        parse_quote!(
            trait T {
                fn f(&self) -> Vec<(::foo::X, Rel)>;
                fn g(&self) -> ::foo::V<(Rel, ::foo::X)>;
            }
        ),
        parse_quote!(
            trait T<'a, A> {
                fn f(&self, a: &'a ::foo::W<Rel<A>>, b: ::foo::X<&'a Rel>) -> ::foo::Y<A>;
            }
        ),
        parse_quote!(
            trait T {
                fn f<U>(&self, u: U) -> ::foo::W<U>;
                fn g<U>(&self, u: ::foo::V<U, Rel>) -> Rel2<U>;
            }
        ),
        parse_quote!(
            trait T {
                fn f(&self) -> ::foo::W<::foo::W<Rel>>;
                fn g(&self) -> ::bar::V<::foo::W<Rel>>;
            }
        ),
        parse_quote!(
            trait T {
                fn f(&self, a: [u8; ::foo::N], b: Wrap<[u8; ::foo::N]>) -> Rel;
            }
        ),
        parse_quote!(
            trait T {
                fn f(&self, a: ::foo::Tr<Item = Rel>, b: Box<dyn ::foo::Tr<Item = Rel>>) -> Rel;
            }
        ),
    ];
    for item in items {
        required_entries_match_allowed_leaker(item, &["foo", "bar"]);
    }
}

fn referrer_with_foo() -> Referrer {
    let item: ItemTrait = parse_quote! {
        trait T {
            fn f(&self, a: ::foo::X) -> Rel;
        }
    };
    let mut leaker = leaker();
    leak_trait(&mut leaker, &item).unwrap();
    leaker.finish()
}

fn expand(referrer: &Referrer, ty: Type) -> String {
    let ty = referrer.expand(ty, |_, ix| parse_quote!(P<#ix>));
    quote!(#ty).to_string()
}

#[test]
fn referrer_leaves_optional_entries_by_default() {
    // Old: an optional entry was replaced unless the crates were `with_available_external_crates`;
    // and with `allow_external_crates` set (the default) there was no such entry. New: the default
    // leaves it as written, as the default did.
    let referrer = referrer_with_foo();
    let ix_foo = referrer
        .iter()
        .position(|t| squash(quote!(#t)) == "::foo::X")
        .unwrap();
    assert!(!referrer.must(ix_foo));
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
    // Also with the flag set to `true` explicitly.
    let referrer = referrer.allow_ambiguous_crate_name(true);
    assert_eq!(expand(&referrer, parse_quote!(::foo::X)), ":: foo :: X");
}

#[test]
fn referrer_replaces_optional_entries_without_ambiguous_crate_names() {
    let referrer = referrer_with_foo().allow_ambiguous_crate_name(false);
    let ix_foo = referrer
        .iter()
        .position(|t| squash(quote!(#t)) == "::foo::X")
        .unwrap();
    assert!(!referrer.must(ix_foo));
    assert_eq!(
        expand(&referrer, parse_quote!(::foo::X)),
        format!("P < {}usize >", ix_foo)
    );
    // The types inside an optional entry which is left as written are looked up as usual.
    let item: ItemTrait = parse_quote!(
        trait T {
            fn f(&self) -> ::foo::W<(Rel,)>;
        }
    );
    let mut leaker = leaker();
    leak_trait(&mut leaker, &item).unwrap();
    let referrer = leaker.finish();
    let ix_rel = referrer
        .iter()
        .position(|t| squash(quote!(#t)) == "Rel")
        .unwrap();
    let ix_w = referrer
        .iter()
        .position(|t| squash(quote!(#t)) == "::foo::W<(Rel,)>")
        .unwrap();
    assert_eq!(
        expand(&referrer, parse_quote!(::foo::W<(Rel,)>)),
        format!(":: foo :: W < (P < {}usize > ,) >", ix_rel)
    );
    // And replaced as a whole otherwise.
    let referrer = referrer.allow_ambiguous_crate_name(false);
    assert_eq!(
        expand(&referrer, parse_quote!(::foo::W<(Rel,)>)),
        format!("P < {}usize >", ix_w)
    );
    // Rel alone is replaced either way.
    assert_eq!(
        expand(&referrer, parse_quote!(Rel)),
        format!("P < {}usize >", ix_rel)
    );
}

#[test]
fn referrer_visitor_respects_the_flag() {
    let mut item: ItemTrait = parse_quote!(
        trait T {
            fn f(&self, a: ::foo::X) -> Rel;
        }
    );
    let mut replaced = item.clone();
    referrer_with_foo()
        .into_visitor(None, |_, ix| parse_quote!(P<#ix>))
        .visit_item_trait_mut(&mut item);
    let s = quote!(#item).to_string();
    assert!(s.contains(":: foo :: X"), "{}", s);
    assert!(s.contains("P <"), "{}", s);
    referrer_with_foo()
        .allow_ambiguous_crate_name(false)
        .into_visitor(None, |_, ix| parse_quote!(P<#ix>))
        .visit_item_trait_mut(&mut replaced);
    let s = quote!(#replaced).to_string();
    assert!(!s.contains(":: foo"), "{}", s);
    assert_eq!(s.matches("P <").count(), 2, "{}", s);
}

#[test]
fn referrer_serialization_round_trip() {
    let item: ItemTrait = parse_quote! {
        trait T {
            fn f<U: ::core::convert::Into<::bar::B>>(&self, u: U, x: ::foo::X, r: Rel) -> ::foo::W<U>;
        }
    };
    let mut leaker = leaker();
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
    // The flag is the choice of the Referrer's side, so it is not serialized.
    let strict = referrer.clone().allow_ambiguous_crate_name(false);
    assert_ne!(strict, referrer);
    assert_eq!(quote!(#strict).to_string(), s);
    let parsed: Referrer = parse2(quote!(#strict)).unwrap();
    assert_eq!(parsed, referrer);
    assert_eq!(
        expand(&parsed, parse_quote!(::foo::X)),
        ":: foo :: X",
        "a parsed Referrer has the default"
    );
    let reset = parsed.allow_ambiguous_crate_name(false);
    assert_eq!(reset, strict);
    assert_ne!(expand(&reset, parse_quote!(::foo::X)), ":: foo :: X");
}

#[test]
fn referrer_round_trip_keeps_the_indices() {
    // The required entries come first (so their indices do not depend on the optional ones), and
    // the optional ones follow, also with several contexts: they are numbered the same by the
    // Referrer which parses the serialized one.
    let item: ItemTrait = parse_quote! {
        trait T {
            fn a(&self) -> ::z::A;
            fn f<U>(&self, u: ::foo::V<U>) -> Rel<U>;
            fn g(&self) -> Rel2;
            fn h<const B: bool>(&self) -> (::foo::W<B>, Wrap<B>);
        }
    };
    let mut leaker = leaker();
    leak_trait(&mut leaker, &item).unwrap();
    let referrer = leaker.finish();
    let before = entries(&referrer);
    let musts: Vec<bool> = before.iter().map(|e| e.1).collect();
    assert!(musts.iter().any(|m| !m));
    assert!(musts.windows(2).all(|w| w[0] || !w[1]), "{:?}", before);
    assert!(referrer.contexts().count() > 2);
    let parsed: Referrer = parse2(quote!(#referrer)).unwrap();
    assert_eq!(entries(&parsed), before);
    assert_eq!(parsed, referrer);
    for ix in 0..before.len() {
        assert_eq!(parsed.context(ix), referrer.context(ix));
        assert_eq!(parsed.generics(ix), referrer.generics(ix));
    }
}

#[test]
fn referrer_plain_entry_syntax() {
    let version = tag();
    // The required entries come first, whatever order they are written in.
    let referrer: Referrer =
        parse2(quote!(#version ([u8; 3], ? @[foo, bar] X<[u8]>, @[a] Y))).unwrap();
    assert_eq!(
        entries(&referrer),
        vec![
            ("[u8;3]".to_string(), true, vec![]),
            ("Y".to_string(), true, crates(&["a"])),
            ("X<[u8]>".to_string(), false, crates(&["foo", "bar"])),
        ]
    );
}

#[test]
fn raw_identifier_crates_are_stored_unraw_and_serialized_back() {
    use template_quote::ToTokens;
    let item: ItemTrait = parse_quote! {
        trait T {
            fn f(&self, a: ::r#async::X, b: ::r#fn::Y, c: ::plain::Z);
        }
    };
    let mut leaker = leaker();
    leak_trait(&mut leaker, &item).unwrap();
    let referrer = leaker.finish();
    assert_eq!(
        entries(&referrer),
        [
            entry("::plain::Z", false, &["plain"]),
            entry("::r#async::X", false, &["async"]),
            entry("::r#fn::Y", false, &["fn"]),
        ]
    );
    // It is serialized with the raw identifiers, and parsed back.
    let tokens = referrer.to_token_stream();
    assert!(tokens.to_string().contains("@ [r#async]"));
    assert_eq!(parse2::<Referrer>(tokens).unwrap(), referrer);
    // Allowed by the Leaker, the crate is not a node, by its name (without `r#`).
    let mut leaker = leaker_allowing(&["r#async"]);
    leak_trait(&mut leaker, &item).unwrap();
    assert_eq!(
        entries(&leaker.finish()),
        [
            entry("::plain::Z", false, &["plain"]),
            entry("::r#fn::Y", false, &["fn"]),
        ]
    );
}

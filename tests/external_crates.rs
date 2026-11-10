//! Tests for the types with the absolute path of an external crate (`::foo::X`), which are optional
//! entries of the Referrer: `Rules::or_if()`, `Referrer::must()`,
//! `Referrer::external_crates_require()` and the trust of `LeakerResult::allow()`.

mod common;

use common::{check_in, starts_with};
use syn::parse::Parser;
use syn::visit_mut::VisitMut;
use syn::*;
use template_quote::quote;
use type_leak::*;

/// The tag of the format of a serialized Referrer, which it starts with.
fn tag() -> String {
    format!("type-leak-format-{}", type_leak::Referrer::FORMAT_VERSION)
}

fn leaker() -> Leaker<impl Allow> {
    let config = Rules::new().primitive();
    Leaker::from_config(config, Options::new())
}

/// A Leaker which allows the external crates `allowed` (`::name`).
fn leaker_allowing(allowed: &[&str]) -> Leaker<impl Allow> {
    let allowed: Vec<Path> = allowed
        .iter()
        .map(|name| {
            let ident: Ident = parse_str(name).unwrap();
            parse_quote!(::#ident)
        })
        .collect();
    let config = Rules::new()
        .primitive()
        .or_if(move |cause: &Cause, place: Place| {
            allowed.iter().any(|a| starts_with(a.clone())(cause, place))
        });
    Leaker::from_config(config, Options::new())
}

/// Prints tokens without whitespace, to compare them with the source.
fn squash(tokens: impl template_quote::ToTokens) -> String {
    tokens.to_token_stream().to_string().replace(' ', "")
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
    leaker.intern_trait(&item, true);
    entries(&leaker.result().finish().unwrap())
}

fn crates(names: &[&str]) -> Vec<String> {
    names.iter().map(|s| s.to_string()).collect()
}

fn entry(ty: &str, must: bool, names: &[&str]) -> (String, bool, Vec<String>) {
    (ty.to_string(), must, crates(names))
}

/// `Referrer::external_crates_require()`: an entry that names `Self` where the self type is an
/// external type (`::ext::W`) requires that crate: the entry `Rel<Self>` is replaced by a Referrer
/// that has `::ext` too, as `Self` is `::ext::W` there. Audit mutant `x_crates_self_off` (the
/// crates of the self type are not counted for an entry that names `Self`).
#[test]
fn an_entry_that_names_self_requires_the_crates_of_the_self_type() {
    let item: ItemImpl = parse_quote!(
        impl ::p::Tr for ::ext::W {
            fn f(&self, x: Rel<Self>) {}
        }
    );
    let mut analyzer = leaker();
    analyzer.intern_impl(&item, true);
    let referrer = analyzer.result().finish().unwrap();
    assert_eq!(
        entries(&referrer),
        vec![
            entry("Rel<Self>", true, &["ext"]),
            entry("::ext::W", false, &["ext"])
        ]
    );
    // Without `Self` in the entry, no crate is needed.
    let item: ItemImpl = parse_quote!(
        impl ::p::Tr for ::ext::W {
            fn f(&self, x: Rel) {}
        }
    );
    let mut analyzer = leaker();
    analyzer.intern_impl(&item, true);
    assert_eq!(
        entries(&analyzer.result().finish().unwrap()),
        vec![entry("Rel", true, &[]), entry("::ext::W", false, &["ext"])]
    );
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
    // crate is not one, as the Referrer may not replace it (as before). Such a type is an optional
    // entry, which `check()` tells too.
    let mut leaker = leaker();
    let generics = Generics::default();
    let result = check_in(&mut leaker, &parse_quote!(::foo::W<(Rel,)>), &generics);
    assert_eq!(
        result.must.iter().map(squash).collect::<Vec<_>>(),
        vec!["Rel"]
    );
    assert!(!result.is_own);
    assert!(result.optional);
    let result = check_in(&mut leaker, &parse_quote!(::foo::X), &generics);
    assert!(result.must.is_empty() && result.impossible.is_empty());
    assert!(result.optional);
    // Not so if it has a cause of its own, or cannot be leaked.
    let result = check_in(&mut leaker, &parse_quote!(::foo::W<Rel>), &generics);
    assert!(result.is_own && !result.optional);
    let result = check_in(&mut leaker, &parse_quote!(::foo::W<'_>), &generics);
    assert!(!result.impossible.is_empty() && !result.optional);
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
    let mut leaker = leaker();
    let ty: Type = parse_quote!(::core::option::Option<(Rel,)>);
    let result = check_in(&mut leaker, &ty, &Generics::default());
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
    leaker.intern_trait(&item, true);
    assert_eq!(
        entries(&leaker.result().finish().unwrap()),
        vec![entry("Rel", true, &[])]
    );
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
    leaker.intern_trait(&item, true);
    assert_eq!(
        entries(&leaker.result().finish().unwrap()),
        vec![
            entry("Rel", true, &[]),
            entry("foo::Z", true, &[]),
            entry("::foobar::Y", false, &["foobar"]),
        ]
    );
    // An allowed relative path allows the relative paths only: an absolute one is a node.
    let config = Rules::new().or_if(starts_with(parse_quote!(my)));
    let mut leaker = Leaker::from_config(config, Options::new());
    let item: ItemTrait = parse_quote! {
        trait T {
            fn f(&self, a: my::X, b: ::my::Y);
        }
    };
    leaker.intern_trait(&item, true);
    assert_eq!(
        entries(&leaker.result().finish().unwrap()),
        vec![entry("::my::Y", false, &["my"])]
    );
    // The more specific path may be allowed alone.
    let config = Rules::new().or_if(starts_with(parse_quote!(::foo::inner)));
    let mut leaker = Leaker::from_config(config, Options::new());
    let item: ItemTrait = parse_quote! {
        trait T {
            fn f(&self, a: ::foo::inner::X, b: ::foo::Y);
        }
    };
    leaker.intern_trait(&item, true);
    assert_eq!(
        entries(&leaker.result().finish().unwrap()),
        vec![entry("::foo::Y", false, &["foo"])]
    );
    // `check()` of the same.
    let ty: Type = parse_quote!(::foo::W<(Rel,)>);
    let config = Rules::new().or_if(starts_with(parse_quote!(::foo)));
    let result = check_in(
        &mut Leaker::from_config(config, Options::new()),
        &ty,
        &Generics::default(),
    );
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
    // One cut chooses the entries: `::foo::W<(Rel,)>` has `Rel` in it, so it is the only entry
    // and it is required, whether the Referrer trusts the name of the crate or not. (It was `Rel`
    // and the optional entry `::foo::W<(Rel,)>`, which a Referrer that trusts the crate left,
    // replacing `Rel` in it.)
    let item: ItemTrait = parse_quote!(
        trait T {
            fn f(&self) -> ::foo::W<(Rel,)>;
        }
    );
    assert_eq!(
        entries_of(item),
        vec![entry("::foo::W<(Rel,)>", true, &["foo"])]
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
    // An absolute type which is covered by a required entry is not an entry, and a type that holds
    // both is one entry: the tuple is cheaper than its parts.
    let item: ItemTrait = parse_quote!(
        trait T {
            fn f(&self) -> (Rel, ::foo::X);
            fn g(&self) -> (Rel, ::foo::X);
        }
    );
    assert_eq!(
        entries_of(item),
        vec![entry("(Rel,::foo::X)", true, &["foo"])]
    );
    // Beside each other, they keep their kinds.
    let item: ItemTrait = parse_quote!(
        trait T {
            fn f(&self, a: Rel, b: ::foo::X);
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
    let mut leaker = leaker();
    leaker.intern_trait(&item, true);
    let result = leaker.result();
    let err = result.errors();
    assert_eq!(err.len(), 1);
    assert_eq!(err[0].message, "use absolute path: 'Rel<'_>'");
}

/// Entries of the same item with all the crates allowed (a Leaker with no external crates), and
/// those of the default Leaker: one cut chooses the entries, so the types of the crates only merge
/// the entries of the Leaker that allows them (an entry with a type of a crate in it takes the
/// place of the entries in it), and add the optional ones.
fn optional_entries_only_merge_the_required_ones(item: ItemTrait, allowed: &[&str]) {
    let mut leaker = leaker();
    leaker.intern_trait(&item, true);
    let default = leaker.result().finish().unwrap();
    let mut leaker = leaker_allowing(allowed);
    leaker.intern_trait(&item, true);
    let reference = leaker.result().finish().unwrap();
    let text = |referrer: &Referrer, ix: usize| squash(quote!(#{referrer.iter().nth(ix).unwrap()}));
    let default_entries: Vec<String> = (0..default.iter().count())
        .map(|ix| text(&default, ix))
        .collect();
    // Every entry of the Leaker that allows the crates is an entry of the default one, or is in
    // one (an entry that holds it is cheaper, and the tuple, the path or the reference around it).
    for ix in 0..reference.iter().count() {
        let wanted = text(&reference, ix);
        assert!(
            default_entries.iter().any(|entry| entry.contains(&wanted)),
            "`{}` of {}: {:?}",
            wanted,
            quote!(#item),
            default_entries
        );
    }
    // An entry of the default Leaker that the other has not names a crate, and the entries which
    // are required and have no crate are the same.
    for (ix, entry) in default_entries.iter().enumerate() {
        let found = reference.iter().any(|ty| squash(quote!(#ty)) == *entry);
        assert!(
            found || !default.external_crates_require(ix).is_empty(),
            "`{}` of {}",
            entry,
            quote!(#item)
        );
        if default.external_crates_require(ix).is_empty() {
            assert!(default.must(ix), "{}", quote!(#item));
        }
    }
    // The required entries come first, then the optional ones.
    let mut seen_optional = false;
    for ix in 0..default.iter().count() {
        assert!(!(seen_optional && default.must(ix)), "{}", quote!(#item));
        seen_optional |= !default.must(ix);
    }
}

#[test]
fn optional_entries_only_merge_the_required_ones_of_the_leaker_that_allows_the_crates() {
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
        optional_entries_only_merge_the_required_ones(item, &["foo", "bar"]);
    }
}

/// The result and the top-level context of the trait.
fn result_with_foo() -> (LeakerResult, ContextId) {
    let item: ItemTrait = parse_quote! {
        trait T {
            fn f(&self, a: ::foo::X) -> Rel;
        }
    };
    let mut leaker = leaker();
    let context = leaker.intern_trait(&item, true);
    (leaker.result(), context)
}

fn expand(referrer: &Referrer, context: ContextId, ty: Type) -> String {
    let ty = referrer.expand(context, ty, |_, ix| parse_quote!(P<#ix>));
    quote!(#ty).to_string()
}

/// The rule that allows the absolute paths that start with one of the crates `names`.
fn trusting(names: &[&str]) -> impl Fn(&Cause, Place) -> bool + Clone {
    let prefixes: Vec<Path> = names
        .iter()
        .map(|name| {
            let ident: Ident = parse_str(name).unwrap();
            parse_quote!(::#ident)
        })
        .collect();
    move |cause: &Cause, _: Place| {
        prefixes
            .iter()
            .any(|prefix| starts_with(prefix.clone())(cause, Place::default()))
    }
}

/// The result and the top-level context of a trait with the optional entries `::a::X` and `::b::Y`
/// in the signature of one method each (and so in the context of no generic).
fn result_with_two_crates() -> (LeakerResult, ContextId) {
    let item: ItemTrait = parse_quote! {
        trait T {
            fn f(&self, a: ::a::X) -> Rel;
            fn g(&self, b: ::b::Y);
        }
    };
    let mut leaker = leaker();
    let context = leaker.intern_trait(&item, true);
    (leaker.result(), context)
}

/// The index of the entry `ty`.
fn index_of(referrer: &Referrer, ty: &str) -> usize {
    referrer
        .iter()
        .position(|t| squash(quote!(#t)) == ty)
        .unwrap_or_else(|| panic!("no entry `{}`", ty))
}

/// Whether the Referrer replaces `ty` (an entry) by `P<ix>`, or leaves it as written.
fn replaces(referrer: &Referrer, context: ContextId, ty: &str) -> bool {
    let written: Type = parse_str(ty).unwrap();
    let expanded = expand(referrer, context, written.clone());
    let ix = index_of(referrer, &squash(quote!(#written)));
    if expanded == format!("P < {}usize >", ix) {
        true
    } else {
        assert_eq!(expanded, quote!(#written).to_string());
        false
    }
}

/// A Referrer trusts no optional entry unless a rule of `allow()` allows its paths: it replaces
/// them, as it does the required entries.
#[test]
fn referrer_replaces_optional_entries_by_default() {
    let (result, context) = result_with_foo();
    let referrer = result.clone().finish().unwrap();
    let ix_foo = index_of(&referrer, "::foo::X");
    assert!(!referrer.must(ix_foo));
    assert_eq!(
        expand(&referrer, context, parse_quote!(::foo::X)),
        format!("P < {}usize >", ix_foo)
    );
    // A required entry is replaced either way.
    let ix_rel = index_of(&referrer, "Rel");
    assert!(referrer.must(ix_rel));
    assert_eq!(
        expand(&referrer, context, parse_quote!(Rel)),
        format!("P < {}usize >", ix_rel)
    );
    // `into_referrer()` is the same Referrer.
    let referrer = result.into_referrer();
    assert_eq!(
        expand(&referrer, context, parse_quote!(::foo::X)),
        format!("P < {}usize >", ix_foo)
    );
}

#[test]
fn a_trusted_optional_entry_is_left_as_written() {
    let (result, context) = result_with_foo();
    let referrer = result
        .allow(Rules::new().external_crates())
        .finish()
        .unwrap();
    let ix_foo = index_of(&referrer, "::foo::X");
    assert!(!referrer.must(ix_foo));
    assert_eq!(
        expand(&referrer, context, parse_quote!(::foo::X)),
        ":: foo :: X"
    );
    // A required entry is not affected.
    let ix_rel = index_of(&referrer, "Rel");
    assert_eq!(
        expand(&referrer, context, parse_quote!(Rel)),
        format!("P < {}usize >", ix_rel)
    );
}

#[test]
fn a_type_of_a_crate_with_a_required_type_in_it_is_replaced_trusted_or_not() {
    // `::foo::W<(Rel,)>` is one required entry: the Referrer replaces it as a whole, also if it
    // trusts the name of the crate (it was an optional entry that it left, replacing `Rel` in it).
    let item: ItemTrait = parse_quote!(
        trait T {
            fn f(&self) -> ::foo::W<(Rel,)>;
        }
    );
    let mut analyzer = leaker();
    let context = analyzer.intern_trait(&item, true);
    let result = analyzer.result();
    let trusting = result.clone().allow(Rules::new().external_crates());
    let referrer = trusting.finish().unwrap();
    assert_eq!(referrer.iter().count(), 1);
    assert!(referrer.must(0));
    assert_eq!(
        expand(&referrer, context, parse_quote!(::foo::W<(Rel,)>)),
        "P < 0usize >"
    );
    let referrer = result.finish().unwrap();
    assert_eq!(
        expand(&referrer, context, parse_quote!(::foo::W<(Rel,)>)),
        "P < 0usize >"
    );
    // `Rel` alone is no entry: it is in the other one.
    assert_eq!(
        expand(&referrer, context, parse_quote!(Rel)),
        quote!(Rel).to_string()
    );
    // An optional entry has no type in it that depends on the context, so a trusting Referrer
    // leaves it as it is, and replaces it as a whole otherwise.
    let item: ItemTrait = parse_quote!(
        trait T {
            fn f(&self) -> ::foo::W<::bar::X>;
        }
    );
    let mut analyzer = leaker();
    let context = analyzer.intern_trait(&item, true);
    let result = analyzer.result();
    let trusting = result.clone().allow(Rules::new().external_crates());
    let referrer = trusting.finish().unwrap();
    assert!(!referrer.must(0));
    assert_eq!(
        expand(&referrer, context, parse_quote!(::foo::W<::bar::X>)),
        ":: foo :: W < :: bar :: X >"
    );
    let referrer = result.finish().unwrap();
    assert_eq!(
        expand(&referrer, context, parse_quote!(::foo::W<::bar::X>)),
        "P < 0usize >"
    );
}

/// The type of a crate that has a relative type in it is a required entry, so the Referrer finds it
/// in the scope of the item, and must know what `Self` is there when its context needs `Self`
/// (`X: Into<Self::A1>`, `Self: 'y`): the top-level context of the trait records it, as it does for
/// any required entry. (The entry was optional, which a trusting Referrer does not look up.) The
/// type holds only a type that is leaked alone (`Rel<u8>`), so it is no candidate of the required
/// entries by its own cause, but an optional type is a candidate of either kind.
#[test]
fn a_type_of_a_crate_that_needs_self_is_replaced_by_the_referrer() {
    let item: ItemTrait = parse_quote!(
        trait T {
            type A1;
            fn m<'x, X: 'x>(&self, a: ::foo::W<(X, u8), Rel<u8>>)
            where
                X: ::core::convert::Into<Self::A1>;
            fn n<'y>(&self, b: ::foo::V<&'y u8, Rel2<u8>>)
            where
                Self: 'y;
        }
    );
    let mut analyzer = leaker();
    let context = analyzer.intern_trait(&item, true);
    let result = analyzer.result();
    for trusting in [true, false] {
        let result = result.clone();
        let result = if trusting {
            result.allow(Rules::new().external_crates())
        } else {
            result
        };
        let referrer = result.finish().unwrap();
        assert!(referrer.required().count() == 2 && referrer.iter().count() == 2);
        assert!(referrer.context_self(context).is_some());
        let mut copy = item.clone();
        referrer
            .visitor(context, |_, ix| parse_quote!(P<#ix>))
            .visit_item_trait_mut(&mut copy);
        let expected: ItemTrait = parse_quote!(
            trait T {
                type A1;
                fn m<'x, X: 'x>(&self, a: P<0usize>)
                where
                    X: ::core::convert::Into<Self::A1>;
                fn n<'y>(&self, b: P<1usize>)
                where
                    Self: 'y;
            }
        );
        assert_eq!(squash(&copy), squash(&expected), "trusting {}", trusting);
    }
}

#[test]
fn referrer_visitor_respects_the_trust() {
    let mut item: ItemTrait = parse_quote!(
        trait T {
            fn f(&self, a: ::foo::X) -> Rel;
        }
    );
    let mut replaced = item.clone();
    let (result, context) = result_with_foo();
    result
        .clone()
        .allow(Rules::new().external_crates())
        .finish()
        .unwrap()
        .visitor(context, |_, ix| parse_quote!(P<#ix>))
        .visit_item_trait_mut(&mut item);
    let s = quote!(#item).to_string();
    assert!(s.contains(":: foo :: X"), "{}", s);
    assert!(s.contains("P <"), "{}", s);
    result
        .finish()
        .unwrap()
        .visitor(context, |_, ix| parse_quote!(P<#ix>))
        .visit_item_trait_mut(&mut replaced);
    let s = quote!(#replaced).to_string();
    assert!(!s.contains(":: foo"), "{}", s);
    assert_eq!(s.matches("P <").count(), 2, "{}", s);
}

#[test]
fn the_trust_of_the_result_reaches_the_referrer_of_finish_and_into_referrer() {
    let (result, context) = result_with_foo();
    // The default replaces the optional entry; `allow()` of its crate trusts it, and chains with
    // the other rules.
    for referrer in [
        result.clone().finish().unwrap(),
        result.clone().into_referrer(),
    ] {
        let ix = index_of(&referrer, "::foo::X");
        assert_eq!(
            expand(&referrer, context, parse_quote!(::foo::X)),
            format!("P < {}usize >", ix)
        );
    }
    let trusting = result
        .allow(Rules::new().primitive())
        .allow(Rules::new().external_crates());
    for referrer in [trusting.clone().finish().unwrap(), trusting.into_referrer()] {
        assert_eq!(
            expand(&referrer, context, parse_quote!(::foo::X)),
            ":: foo :: X"
        );
        // A required entry is replaced either way.
        let ix = index_of(&referrer, "Rel");
        assert_eq!(
            expand(&referrer, context, parse_quote!(Rel)),
            format!("P < {}usize >", ix)
        );
    }
}

#[test]
fn a_serialized_referrer_parses_as_a_result_without_errors_and_takes_the_trust() {
    let (result, context) = result_with_foo();
    let referrer = result.finish().unwrap();
    let parsed: LeakerResult = parse2(quote!(#referrer)).unwrap();
    assert!(parsed.errors().is_empty());
    assert_eq!(parsed.clone().into_referrer(), referrer);
    let ix = index_of(&referrer, "::foo::X");
    let replaced = format!("P < {}usize >", ix);
    assert_eq!(
        expand(
            &parsed.clone().finish().unwrap(),
            context,
            parse_quote!(::foo::X)
        ),
        replaced
    );
    let trusting = parsed.allow(Rules::new().external_crates());
    let trusting = trusting.finish().unwrap();
    // `==` ignores the trust, but the Referrers behave differently.
    assert_eq!(trusting, referrer);
    assert_eq!(
        expand(&trusting, context, parse_quote!(::foo::X)),
        ":: foo :: X"
    );
    assert_eq!(expand(&referrer, context, parse_quote!(::foo::X)), replaced);
    // The tokens after the Referrer are left, as the macros pass `referrer , context`.
    let (read, _) = (|input: parse::ParseStream| -> Result<(LeakerResult, Token![,])> {
        Ok((input.parse()?, input.parse()?))
    })
    .parse2(quote!(#referrer ,))
    .unwrap();
    assert!(read.errors().is_empty());
}

/// The trust is not serialized, so `==` ignores it: a trusting Referrer equals its parsed copy
/// (which trusts nothing), and keeps its trust.
#[test]
fn a_trusting_referrer_equals_its_parsed_copy_and_stays_trusting() {
    let (result, context) = result_with_foo();
    let trusting = result.clone().allow(Rules::new().external_crates());
    for trusting in [trusting.clone().finish().unwrap(), trusting.into_referrer()] {
        let parsed: Referrer = parse2(quote!(#trusting)).unwrap();
        assert_eq!(trusting, parsed);
        assert_eq!(parsed, trusting);
        assert_eq!(trusting, trusting.clone());
        // Through a result too.
        let again: LeakerResult = parse2(quote!(#trusting)).unwrap();
        assert_eq!(again.into_referrer(), trusting);
        // Equal, but only the original leaves the optional entry as written.
        let ix = index_of(&trusting, "::foo::X");
        let replaced = format!("P < {}usize >", ix);
        assert_eq!(
            expand(&trusting, context, parse_quote!(::foo::X)),
            ":: foo :: X"
        );
        assert_eq!(expand(&parsed, context, parse_quote!(::foo::X)), replaced);
        // Comparing and cloning changed nothing.
        assert_eq!(
            expand(&trusting.clone(), context, parse_quote!(::foo::X)),
            ":: foo :: X"
        );
    }
    // Anything that the serialization keeps is still compared.
    let mut leaker = leaker();
    leaker.intern_trait(
        &parse_quote!(
            trait T {
                fn f(&self, a: ::foo::Y) -> Rel;
            }
        ),
        true,
    );
    assert_ne!(result.finish().unwrap(), leaker.result().finish().unwrap());
}

#[test]
fn the_trust_of_the_result_is_not_serialized() {
    let (result, context) = result_with_foo();
    let trusting = result.clone().allow(Rules::new().external_crates());
    let tokens = quote!(#trusting);
    assert_eq!(tokens.to_string(), quote!(#result).to_string());
    let parsed: LeakerResult = parse2(tokens).unwrap();
    assert_eq!(
        parsed.clone().finish().unwrap(),
        result.clone().finish().unwrap()
    );
    for referrer in [parsed.clone().finish().unwrap(), parsed.into_referrer()] {
        let ix = index_of(&referrer, "::foo::X");
        assert_eq!(
            expand(&referrer, context, parse_quote!(::foo::X)),
            format!("P < {}usize >", ix),
            "a parsed result trusts nothing"
        );
    }
}

#[test]
fn referrer_serialization_round_trip() {
    let item: ItemTrait = parse_quote! {
        trait T {
            fn f<U: ::core::convert::Into<::bar::B>>(&self, u: U, x: ::foo::X, r: Rel) -> ::foo::W<U>;
        }
    };
    let mut leaker = leaker();
    let context = leaker.intern_trait(&item, true);
    let result = leaker.result();
    let referrer = result.clone().finish().unwrap();
    let before = entries(&referrer);
    assert!(before.iter().any(|e| !e.1 && !e.2.is_empty()));
    assert!(before.iter().any(|e| e.1));
    let tokens = quote!(#referrer);
    let s = tokens.to_string();
    assert!(s.contains("? @ [foo] :: foo :: X"), "{}", s);
    let parsed: Referrer = parse2(tokens).unwrap();
    assert_eq!(parsed, referrer);
    assert_eq!(entries(&parsed), before);
    // The trust is the choice of the Referrer's side, so it is not serialized.
    let trusting = result.allow(Rules::new().external_crates());
    let trusting = trusting.finish().unwrap();
    assert_eq!(trusting, referrer);
    assert_eq!(quote!(#trusting).to_string(), s);
    assert_eq!(
        expand(&trusting, context, parse_quote!(::foo::X)),
        ":: foo :: X"
    );
    let parsed: Referrer = parse2(quote!(#trusting)).unwrap();
    assert_eq!(parsed, referrer);
    let ix = index_of(&parsed, "::foo::X");
    assert_eq!(
        expand(&parsed, context, parse_quote!(::foo::X)),
        format!("P < {}usize >", ix),
        "a parsed Referrer trusts nothing"
    );
    // A parsed result takes a rule again.
    let reset: LeakerResult = parse2(quote!(#trusting)).unwrap();
    let reset = reset
        .allow(Rules::new().external_crates())
        .finish()
        .unwrap();
    assert_eq!(reset, trusting);
    assert_eq!(
        expand(&reset, context, parse_quote!(::foo::X)),
        ":: foo :: X"
    );
}

/// A rule trusts the optional entries that name the crates it allows, and not the others.
#[test]
fn a_rule_trusts_the_entries_of_the_crates_it_allows() {
    let (result, context) = result_with_two_crates();
    let all = result.clone().finish().unwrap();
    assert_eq!(
        entries(&all)
            .iter()
            .map(|e| (e.0.as_str(), e.1))
            .collect::<Vec<_>>(),
        [("Rel", true), ("::a::X", false), ("::b::Y", false)]
    );
    // Nothing is allowed: both are replaced.
    assert!(replaces(&all, context, "::a::X") && replaces(&all, context, "::b::Y"));
    let only_a = result.clone().allow(trusting(&["a"])).finish().unwrap();
    assert!(!replaces(&only_a, context, "::a::X"));
    assert!(replaces(&only_a, context, "::b::Y"));
    let only_b = result.clone().allow(trusting(&["b"])).finish().unwrap();
    assert!(replaces(&only_b, context, "::a::X"));
    assert!(!replaces(&only_b, context, "::b::Y"));
    let both = result.allow(trusting(&["a", "b"])).finish().unwrap();
    assert!(!replaces(&both, context, "::a::X") && !replaces(&both, context, "::b::Y"));
    // A required entry is never left, whatever the rule allows.
    assert!(replaces(&both, context, "Rel"));
}

/// A rule is asked about the paths of the entry on its own: a path of a crate that it allows does not
/// make the entry trusted if another is not.
#[test]
fn an_entry_that_names_two_crates_is_trusted_only_if_both_are_allowed() {
    let item: ItemTrait = parse_quote!(
        trait T {
            fn f(&self, a: (::a::X, ::b::Y));
            fn g(&self, c: ::c::W);
        }
    );
    let mut leaker = leaker();
    let context = leaker.intern_trait(&item, true);
    let result = leaker.result();
    let referrer = result.clone().finish().unwrap();
    // One entry covers the tuple, which names two crates.
    assert_eq!(
        entries(&referrer),
        vec![
            entry("(::a::X,::b::Y)", false, &["a", "b"]),
            entry("::c::W", false, &["c"]),
        ]
    );
    let tuple = "(::a::X,::b::Y)";
    for (allowed, trusted) in [
        (&["a"][..], false),
        (&["b"][..], false),
        (&["c"][..], false),
        (&["a", "b"][..], true),
        (&["b", "a"][..], true),
        (&["a", "b", "c"][..], true),
    ] {
        let referrer = result.clone().allow(trusting(allowed)).finish().unwrap();
        assert_eq!(
            !replaces(&referrer, context, tuple),
            trusted,
            "allowed {:?}",
            allowed
        );
        assert_eq!(
            !replaces(&referrer, context, "::c::W"),
            allowed.contains(&"c"),
            "allowed {:?}",
            allowed
        );
    }
}

/// The paths of an optional entry that a rule is asked about, with their kinds.
fn asked(result: &LeakerResult) -> Vec<Cause> {
    let seen = std::cell::RefCell::new(Vec::new());
    let _ = result.clone().allow(|cause: &Cause, place: Place| {
        assert_eq!(place, Place::default());
        seen.borrow_mut().push(cause.clone());
        true
    });
    seen.into_inner()
}

/// The cause of the path of a type.
fn type_cause(path: Path) -> Cause {
    Cause::Type(Type::Path(TypePath { qself: None, path }))
}

/// The trait `item` with the entries replaced as the Referrer does.
fn trait_copy(referrer: &Referrer, context: ContextId, item: &ItemTrait) -> String {
    let mut copy = item.clone();
    referrer
        .visitor(context, |_, ix| parse_quote!(P<#ix>))
        .visit_item_trait_mut(&mut copy);
    quote!(#copy).to_string()
}

/// The bounds of the context of an entry count: its crate is asked as the trait of a bound, and an
/// entry whose bound is not allowed is not trusted.
#[test]
fn a_bound_in_the_context_of_an_entry_is_asked_as_a_trait() {
    let item: ItemTrait = parse_quote! {
        trait T {
            fn f<U: ::bar::B>(&self, u: U) -> ::foo::W<U>;
        }
    };
    let mut leaker = leaker();
    let context = leaker.intern_trait(&item, true);
    let result = leaker.result();
    let referrer = result.clone().finish().unwrap();
    assert_eq!(
        entries(&referrer),
        vec![entry("::foo::W<U>", false, &["bar", "foo"])]
    );
    // The path of the type and the trait of the bound, bare, once each.
    let causes = asked(&result);
    assert_eq!(
        causes,
        [
            type_cause(parse_quote!(::foo::W)),
            Cause::Trait(parse_quote!(::bar::B)),
        ]
    );
    // Allowing `::foo` is not enough: the bound names `::bar`. Nor is the trait alone.
    let replaced = |names: &[&str]| {
        let referrer = result.clone().allow(trusting(names)).finish().unwrap();
        !trait_copy(&referrer, context, &item).contains(":: foo")
    };
    assert!(replaced(&["foo"]));
    assert!(replaced(&["bar"]));
    assert!(!replaced(&["foo", "bar"]));
    // A rule on the kind: the trait of the bound is a `Cause::Trait`, not the path of a type.
    let kinds = result.clone().allow(|cause: &Cause, _: Place| match cause {
        Cause::Trait(path) => path == &parse_quote!(::bar::B),
        Cause::Type(_) => cause.starts_with(&parse_quote!(::foo)),
        _ => false,
    });
    let referrer = kinds.finish().unwrap();
    assert!(!trait_copy(&referrer, context, &item).contains("P <"));
}

/// A path in the length of an array is of unknown kind. (The kinds of the other places are those
/// of `crates::tests`.)
#[test]
fn the_path_in_an_expression_is_asked_as_unknown() {
    let item: ItemTrait = parse_quote! {
        trait T {
            fn g(&self, b: [u8; ::len::N]);
        }
    };
    let mut leaker = leaker();
    leaker.intern_trait(&item, true);
    let result = leaker.result();
    assert_eq!(
        entries(&result.clone().finish().unwrap()),
        vec![entry("[u8;::len::N]", false, &["len"])]
    );
    assert_eq!(asked(&result), [Cause::Unknown(parse_quote!(::len::N))]);
}

/// `::core` and `$crate` paths are never asked: they are always fine.
#[test]
fn core_paths_are_never_asked() {
    let item: ItemTrait = parse_quote! {
        trait T {
            fn f(&self, a: ::foo::W<::core::option::Option<u8>>, b: ::core::result::Result<::foo::X, u8>);
        }
    };
    let mut leaker = leaker();
    leaker.intern_trait(&item, true);
    let result = leaker.result();
    for cause in asked(&result) {
        assert!(!cause.starts_with(&parse_quote!(::core)), "{:?}", cause);
    }
    // A rule that allows `::foo` alone trusts the entries.
    let referrer = result.allow(trusting(&["foo"])).finish().unwrap();
    assert!(!referrer.is_empty());
}

/// Marks accumulate across the calls of `allow()`: a later rule can trust more, never less.
#[test]
fn the_marks_accumulate_across_calls_of_allow() {
    let (result, context) = result_with_two_crates();
    let first = result.clone().allow(trusting(&["a"]));
    let referrer = first.clone().finish().unwrap();
    assert!(!replaces(&referrer, context, "::a::X"));
    assert!(replaces(&referrer, context, "::b::Y"));
    // A later rule trusts `::b::Y` too.
    let second = first.clone().allow(trusting(&["b"])).finish().unwrap();
    assert!(!replaces(&second, context, "::a::X") && !replaces(&second, context, "::b::Y"));
    // A rule that allows nothing, or others, does not take the trust of `::a::X` back.
    let nothing = first.clone().allow(()).finish().unwrap();
    assert!(!replaces(&nothing, context, "::a::X"));
    assert!(replaces(&nothing, context, "::b::Y"));
    let other = first.allow(trusting(&["c"])).finish().unwrap();
    assert!(!replaces(&other, context, "::a::X"));
    // The order does not matter.
    let reverse = result
        .allow(trusting(&["b"]))
        .allow(trusting(&["a"]))
        .finish()
        .unwrap();
    assert!(!replaces(&reverse, context, "::a::X") && !replaces(&reverse, context, "::b::Y"));
}

/// An entry is trusted by the paths it has, and not by the union of what the rules allow: an entry
/// that names two crates, each allowed by another rule, is not trusted by the first nor by the
/// second, but it is by the rule that allows both.
#[test]
fn the_rules_of_two_calls_do_not_add_up_for_an_entry() {
    let item: ItemTrait = parse_quote!(
        trait T {
            fn f(&self, a: (::a::X, ::b::Y));
        }
    );
    let mut leaker = leaker();
    let context = leaker.intern_trait(&item, true);
    let result = leaker.result();
    let tuple = "(::a::X,::b::Y)";
    let chained = result
        .clone()
        .allow(trusting(&["a"]))
        .allow(trusting(&["b"]))
        .finish()
        .unwrap();
    assert!(replaces(&chained, context, tuple));
    let one = result.allow(trusting(&["a", "b"])).finish().unwrap();
    assert!(!replaces(&one, context, tuple));
}

/// A required entry is not affected, whatever the rule allows.
#[test]
fn required_entries_are_not_affected_by_the_rule() {
    let (result, context) = result_with_two_crates();
    let trusting = result.allow(Rules::new().external_crates());
    let referrer = trusting.finish().unwrap();
    assert!(replaces(&referrer, context, "Rel"));
    assert_eq!(referrer.required().count(), 1);
}

/// `Rules::std_crates()` allows the absolute paths of `std` and `alloc` too: the Leaker makes no
/// optional entry of them, and a result trusts them.
#[test]
fn std_crates_allows_the_absolute_paths_of_std_and_alloc() {
    let item: ItemTrait = parse_quote!(
        trait T {
            fn f(&self, a: ::std::vec::Vec<u8>, b: ::alloc::string::String, c: ::foo::X) -> Rel;
        }
    );
    // As a result: the entries of `::std` and `::alloc` are trusted, and `::foo` is not.
    let mut analyzer = leaker();
    let context = analyzer.intern_trait(&item, true);
    let result = analyzer.result();
    assert_eq!(
        entries(&result.clone().finish().unwrap())
            .iter()
            .map(|e| (e.0.clone(), e.1))
            .collect::<Vec<_>>(),
        [
            ("Rel".to_string(), true),
            ("::alloc::string::String".to_string(), false),
            ("::foo::X".to_string(), false),
            ("::std::vec::Vec<u8>".to_string(), false),
        ]
    );
    let referrer = result
        .clone()
        .allow(Rules::new().std_crates())
        .finish()
        .unwrap();
    assert!(!replaces(&referrer, context, "::std::vec::Vec<u8>"));
    assert!(!replaces(&referrer, context, "::alloc::string::String"));
    assert!(replaces(&referrer, context, "::foo::X"));
    // The default replaces them all.
    let referrer = result.finish().unwrap();
    assert!(replaces(&referrer, context, "::std::vec::Vec<u8>"));
    assert!(replaces(&referrer, context, "::alloc::string::String"));
    // In the Leaker: they are no entries, and the others are.
    let mut analyzer = Leaker::from_config(Rules::new().primitive().std_crates(), Options::new());
    analyzer.intern_trait(&item, true);
    assert_eq!(
        entries(&analyzer.result().finish().unwrap())
            .iter()
            .map(|e| (e.0.clone(), e.1))
            .collect::<Vec<_>>(),
        [("Rel".to_string(), true), ("::foo::X".to_string(), false)]
    );
}

/// `Rules::external_crates()` allows every absolute path: the Leaker makes no optional entry, and
/// a result trusts all.
#[test]
fn external_crates_makes_no_optional_entry_in_the_leaker() {
    let item: ItemTrait = parse_quote!(
        trait T {
            fn f<U: ::bar::B>(&self, a: ::foo::X, b: ::foo::W<(Rel, U)>, c: [u8; ::n::N]) -> Rel;
        }
    );
    let mut analyzer =
        Leaker::from_config(Rules::new().primitive().external_crates(), Options::new());
    analyzer.intern_trait(&item, true);
    let referrer = analyzer.result().finish().unwrap();
    // The types in the paths are entries, and no absolute path is.
    assert!(referrer
        .iter()
        .all(|ty| !squash(quote!(#ty)).starts_with("::")));
    assert_eq!(referrer.required().count(), referrer.iter().count());
    assert!(entries(&referrer).iter().any(|e| e.0 == "Rel"));
    // Relative paths are not allowed by it.
    let item: ItemTrait = parse_quote!(
        trait T {
            fn f(&self) -> Rel;
        }
    );
    let mut analyzer =
        Leaker::from_config(Rules::new().primitive().external_crates(), Options::new());
    analyzer.intern_trait(&item, true);
    assert_eq!(analyzer.result().finish().unwrap().iter().count(), 1);
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
    leaker.intern_trait(&item, true);
    let referrer = leaker.result().finish().unwrap();
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
        assert_eq!(
            parsed.context_generics(parsed.context(ix)),
            referrer.context_generics(referrer.context(ix))
        );
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
    leaker.intern_trait(&item, true);
    let referrer = leaker.result().finish().unwrap();
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
    leaker.intern_trait(&item, true);
    assert_eq!(
        entries(&leaker.result().finish().unwrap()),
        [
            entry("::plain::Z", false, &["plain"]),
            entry("::r#fn::Y", false, &["fn"]),
        ]
    );
}

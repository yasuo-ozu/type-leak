//! Tests for the types with the absolute path of an external crate (`::foo::X`), which are entries of
//! the Referrer unless the Leaker's rule allows them: `Rules::or_if()`, `Rules::std_crates()` and
//! `Rules::external_crates()`.

use crate::common;

use common::{check_in, starts_with};
use syn::parse::Parser;
use syn::visit_mut::VisitMut;
use syn::*;
use template_quote::quote;
use type_leak::*;

fn leaker() -> Leaker<impl Allow> {
    let config = Rules::new().primitive();
    Leaker::with_rule(config)
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
    Leaker::with_rule(config)
}

/// Prints tokens without whitespace, to compare them with the source.
fn squash(tokens: impl template_quote::ToTokens) -> String {
    tokens.to_token_stream().to_string().replace(' ', "")
}

/// Returns the leaked types.
fn entries(referrer: &Referrer) -> Vec<String> {
    common::types(referrer)
        .map(|ty| squash(quote!(#ty)))
        .collect()
}

fn entries_of(item: ItemTrait) -> Vec<String> {
    let mut leaker = leaker();
    leaker.intern_trait(&item);
    entries(&leaker.result().finish().unwrap())
}

#[test]
fn absolute_types_are_entries_by_default() {
    // The Referrer's crate may not mean the same crate by the name `foo`.
    let item: ItemTrait = parse_quote!(
        trait T {
            fn f(&self) -> ::foo::X;
        }
    );
    assert_eq!(entries_of(item), vec!["::foo::X"]);
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
        vec!["(::a::A,&'static::b::B)", "::c::C", "::d::D",]
    );
}

#[test]
fn absolute_paths_which_are_not_types_are_copied() {
    // A trait bound, a value and the type of a const parameter cannot be leaked, so they are
    // copied as they are (old: with `allow_external_crates` unset, `NotPortable`; by default the
    // same, with nothing leaked). The types in the other positions are entries, also a path in
    // the length of an array (`[u8; ::foo::N]`).
    let item: ItemTrait = parse_quote! {
        trait T<const N: ::foo::Idx, U: ::foo::Tr = ::foo::D> : ::foo::Sup where u8: ::foo::Tr {
            fn f(&self, a: [u8; ::foo::N]) -> ::foo::X;
            fn g<V: ::foo::Tr<::foo::Y>>(&self, v: V) where V: ::foo::Tr2;
        }
    };
    assert_eq!(
        entries_of(item),
        vec!["::foo::D", "::foo::X", "::foo::Y", "[u8;::foo::N]",]
    );
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
    assert_eq!(entries, vec!["Rel"]);
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
    assert_eq!(entries_of(item), vec!["::corelike::X"]);
}

#[test]
fn std_and_alloc_are_nodes() {
    // Like any crate (see `std_crates_allows_the_absolute_paths_of_std_and_alloc()`), unless they
    // are allowed.
    let item: ItemTrait = parse_quote!(
        trait T {
            fn f(&self) -> ::std::vec::Vec<(u8, Rel)>;
            fn g(&self) -> ::alloc::string::String;
        }
    );
    let mut leaker = leaker_allowing(&["std", "alloc"]);
    leaker.intern_trait(&item);
    assert_eq!(entries(&leaker.result().finish().unwrap()), vec!["Rel"]);
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
    leaker.intern_trait(&item);
    assert_eq!(
        entries(&leaker.result().finish().unwrap()),
        vec!["Rel", "foo::Z", "::foobar::Y",]
    );
    // An allowed relative path allows the relative paths only: an absolute one is a node.
    let config = Rules::new().or_if(starts_with(parse_quote!(my)));
    let mut leaker = Leaker::with_rule(config);
    let item: ItemTrait = parse_quote! {
        trait T {
            fn f(&self, a: my::X, b: ::my::Y);
        }
    };
    leaker.intern_trait(&item);
    assert_eq!(entries(&leaker.result().finish().unwrap()), vec!["::my::Y"]);
    // The more specific path may be allowed alone.
    let config = Rules::new().or_if(starts_with(parse_quote!(::foo::inner)));
    let mut leaker = Leaker::with_rule(config);
    let item: ItemTrait = parse_quote! {
        trait T {
            fn f(&self, a: ::foo::inner::X, b: ::foo::Y);
        }
    };
    leaker.intern_trait(&item);
    assert_eq!(
        entries(&leaker.result().finish().unwrap()),
        vec!["::foo::Y"]
    );
    // `check()` of the same.
    let ty: Type = parse_quote!(::foo::W<(Rel,)>);
    let config = Rules::new().or_if(starts_with(parse_quote!(::foo)));
    let result = check_in(&mut Leaker::with_rule(config), &ty, &Generics::default());
    assert_eq!(
        result.must.iter().map(squash).collect::<Vec<_>>(),
        vec!["Rel"]
    );
}

#[test]
fn absolute_paths_in_the_bounds_of_a_generic_are_no_error() {
    // The bound of a local generic is copied with the type using it.
    let item: ItemTrait = parse_quote! {
        trait T {
            fn f<U: ::core::convert::Into<::bar::B>>(&self) -> ::foo::W<U>;
        }
    };
    assert_eq!(entries_of(item), vec!["::bar::B", "::foo::W<U>"]);

    // A bound with a path to an external crate is not an error (old, with
    // `allow_external_crates` unset: `NotPortable`), it is copied as it is.
    let item: ItemTrait = parse_quote! {
        trait T {
            fn f<U: ::bar::B>(&self) -> ::foo::W<U>;
        }
    };
    assert_eq!(entries_of(item), vec!["::foo::W<U>"]);
}

#[test]
fn one_cut_covers_the_absolute_types_in_an_absolute_type() {
    let item: ItemTrait = parse_quote!(
        trait T {
            fn f(&self) -> ::z::C<::a::D, ::z::A>;
        }
    );
    assert_eq!(entries_of(item), vec!["::z::C<::a::D,::z::A>"]);
}

#[test]
fn a_type_macro_is_one_entry() {
    // A type macro is a cause itself, so it is an entry as a whole.
    let item: ItemTrait = parse_quote!(
        trait T {
            fn f(&self) -> m!(::foo::X, ::core::Y);
        }
    );
    assert_eq!(entries_of(item), vec!["m!(::foo::X,::core::Y)"]);
}

#[test]
fn an_absolute_type_with_a_relative_type_in_it() {
    // One cut chooses the entries: `::foo::W<(Rel,)>` has `Rel` in it, so it is the only entry.
    // (It was `Rel` and the optional entry `::foo::W<(Rel,)>`.)
    let item: ItemTrait = parse_quote!(
        trait T {
            fn f(&self) -> ::foo::W<(Rel,)>;
        }
    );
    assert_eq!(entries_of(item), vec!["::foo::W<(Rel,)>"]);
    // A bare identifier in it may be a const, which cannot be leaked alone, so the type is leaked
    // as a whole (as it is without the crate).
    let item: ItemTrait = parse_quote!(
        trait T {
            fn f(&self) -> ::foo::W<Rel>;
        }
    );
    assert_eq!(entries_of(item), vec!["::foo::W<Rel>"]);
    // A required entry with an absolute path in it is the only one.
    let item: ItemTrait = parse_quote!(
        trait T {
            fn f(&self) -> Rel<::foo::X>;
        }
    );
    assert_eq!(entries_of(item), vec!["Rel<::foo::X>"]);
    // An absolute type which is covered by a required entry is not an entry, and a type that holds
    // both is one entry: the tuple is cheaper than its parts.
    let item: ItemTrait = parse_quote!(
        trait T {
            fn f(&self) -> (Rel, ::foo::X);
            fn g(&self) -> (Rel, ::foo::X);
        }
    );
    assert_eq!(entries_of(item), vec!["(Rel,::foo::X)"]);
    // Beside each other, they are two entries, the one that depends on the context first.
    let item: ItemTrait = parse_quote!(
        trait T {
            fn f(&self, a: Rel, b: ::foo::X);
        }
    );
    assert_eq!(entries_of(item), vec!["Rel", "::foo::X"]);
}

#[test]
fn a_type_which_cannot_be_leaked_is_not_an_error() {
    // An elided lifetime and `impl Trait` cannot be leaked, which is not an error for a type
    // which is only a type of another crate: it is copied as it is.
    let item: ItemTrait = parse_quote!(
        trait T {
            fn f(&self) -> ::foo::W<'_>;
            fn g(&self) -> impl ::foo::Tr;
            fn h(&self, x: &::foo::X) -> ::foo::V<'_, ::foo::Y>;
        }
    );
    assert_eq!(entries_of(item), vec!["::foo::X", "::foo::Y",]);
    // The relative one is, as before.
    let item: ItemTrait = parse_quote!(
        trait T {
            fn f(&self) -> Rel<'_>;
        }
    );
    let mut leaker = leaker();
    leaker.intern_trait(&item);
    let result = leaker.result();
    let err = result.errors();
    assert_eq!(err.len(), 1);
    assert_eq!(err[0].message, "use absolute path: 'Rel<'_>'");
}

/// Entries of the same item with all the crates allowed (a Leaker with no external crates), and
/// those of the default Leaker: one cut chooses the entries, so the types of the crates only merge
/// the entries of the Leaker that allows them (an entry with a type of a crate in it takes the
/// place of the entries in it), and add the types of the crates.
fn the_default_leaker_merges_the_entries_of_the_one_that_allows_the_crates(
    item: ItemTrait,
    allowed: &[&str],
) {
    // Whether an entry names one of the crates by an absolute path.
    let names_crate = |entry: &str| {
        allowed
            .iter()
            .any(|name| entry.contains(&format!("::{}", name)))
    };
    let mut leaker = leaker();
    leaker.intern_trait(&item);
    let default = leaker.result().finish().unwrap();
    let mut leaker = leaker_allowing(allowed);
    leaker.intern_trait(&item);
    let reference = leaker.result().finish().unwrap();
    let text = |referrer: &Referrer, ix: usize| {
        squash(quote!(#{common::types(referrer).nth(ix).unwrap()}))
    };
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
    // An entry of the default Leaker that the other has not names a crate.
    for entry in &default_entries {
        let found = common::types(&reference).any(|ty| squash(quote!(#ty)) == *entry);
        assert!(
            found || names_crate(entry),
            "`{}` of {}",
            entry,
            quote!(#item)
        );
    }
}

#[test]
fn the_types_of_crates_only_merge_the_entries_of_the_leaker_that_allows_the_crates() {
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
        the_default_leaker_merges_the_entries_of_the_one_that_allows_the_crates(
            item,
            &["foo", "bar"],
        );
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
    let context = leaker.intern_trait(&item);
    (leaker.result(), context)
}

/// `ty` as the Referrer expands it, with every entry replaced by `P<ix>`.
fn expand(referrer: &Referrer, context: ContextId, ty: Type) -> String {
    let ty = common::expand(referrer, context, ty, common::placeholder);
    quote!(#ty).to_string()
}

/// The id of the entry `ty`.
fn index_of(referrer: &Referrer, ty: &str) -> EntryId {
    let n = common::types(referrer)
        .position(|t| squash(quote!(#t)) == ty)
        .unwrap_or_else(|| panic!("no entry `{}`", ty));
    common::entry_id_at(referrer, n)
}

/// `allow()` drops errors only: `finish()` gives the Referrer of the result for any rule, also one
/// that allows the crate of a type of another crate, chained, or given to a result read back from
/// tokens; nothing serialized changes. The copy replaces every entry, the type of the crate too.
#[test]
fn allow_changes_no_entry_in_the_referrer_of_finish() {
    let (result, context) = result_with_foo();
    let plain = result.clone().finish().unwrap();
    let ix_foo = index_of(&plain, "::foo::X");
    let ix_rel = index_of(&plain, "Rel");
    let allowed = result.clone().allow(Rules::new().external_crates());
    assert_eq!(quote!(#allowed).to_string(), quote!(#result).to_string());
    // The entries are written without a mark of their kind.
    assert!(!quote!(#plain).to_string().contains('?'));
    // A serialized Referrer, or result, parses as a result without errors that has the Referrer.
    let read = |tokens: proc_macro2::TokenStream| -> LeakerResult {
        let parsed: LeakerResult = parse2(tokens).unwrap();
        assert!(parsed.errors().is_empty());
        parsed
    };
    let results = [
        result.clone(),
        allowed.clone(),
        result
            .clone()
            .allow(Rules::new().primitive())
            .allow(Rules::new().external_crates()),
        read(quote!(#plain)),
        read(quote!(#allowed)).allow(Rules::new().external_crates()),
    ];
    for result in results {
        let referrer = result.finish().unwrap();
        assert_eq!(referrer, plain);
        assert_eq!(entries(&referrer), entries(&plain));
        assert_eq!(parse2::<Referrer>(quote!(#referrer)).unwrap(), plain);
        assert_eq!(
            expand(&referrer, context, parse_quote!(::foo::X)),
            format!("P < {}usize >", ix_foo)
        );
        // So is the other entry.
        assert_eq!(
            expand(&referrer, context, parse_quote!(Rel)),
            format!("P < {}usize >", ix_rel)
        );
    }
    // The tokens after the Referrer are left, as the macros pass `referrer , context`.
    let (read, _) = (|input: parse::ParseStream| -> Result<(LeakerResult, Token![,])> {
        Ok((input.parse()?, input.parse()?))
    })
    .parse2(quote!(#plain ,))
    .unwrap();
    assert!(read.errors().is_empty());
    // Anything that the serialization keeps is still compared.
    let mut leaker = leaker();
    leaker.intern_trait(&parse_quote!(
        trait T {
            fn f(&self, a: ::foo::Y) -> Rel;
        }
    ));
    assert_ne!(plain, leaker.result().finish().unwrap());
}

#[test]
fn a_type_of_a_crate_with_a_relative_type_in_it_is_replaced_as_a_whole() {
    // `::foo::W<(Rel,)>` is one entry: a macro replaces it as a whole.
    let item: ItemTrait = parse_quote!(
        trait T {
            fn f(&self) -> ::foo::W<(Rel,)>;
        }
    );
    let mut analyzer = leaker();
    let context = analyzer.intern_trait(&item);
    let referrer = analyzer.result().finish().unwrap();
    assert_eq!(referrer.iter().count(), 1);
    assert_eq!(
        expand(&referrer, context, parse_quote!(::foo::W<(Rel,)>)),
        "P < 0usize >"
    );
    // `Rel` alone is no entry: it is in the other one.
    assert_eq!(
        expand(&referrer, context, parse_quote!(Rel)),
        quote!(Rel).to_string()
    );
    // A type with no type in it that depends on the context is an entry as a whole too. A Leaker
    // that allows the crates finds none, so the copy writes it as it is.
    let item: ItemTrait = parse_quote!(
        trait T {
            fn f(&self) -> ::foo::W<::bar::X>;
        }
    );
    let mut analyzer = leaker();
    let context = analyzer.intern_trait(&item);
    let referrer = analyzer.result().finish().unwrap();
    assert_eq!(
        expand(&referrer, context, parse_quote!(::foo::W<::bar::X>)),
        "P < 0usize >"
    );
    let mut analyzer = leaker_allowing(&["foo", "bar"]);
    let context = analyzer.intern_trait(&item);
    let referrer = analyzer.result().finish().unwrap();
    assert!(referrer.is_empty());
    assert_eq!(
        expand(&referrer, context, parse_quote!(::foo::W<::bar::X>)),
        ":: foo :: W < :: bar :: X >"
    );
}

/// The type of a crate that has a relative type in it is an entry, so the Referrer finds it in the
/// scope of the item, and must know what `Self` is there when its context needs `Self` (`X:
/// Into<Self::A1>`, `Self: 'y`): the top-level context of the trait records it, as it does for any
/// entry. The type holds only a type that is leaked alone (`Rel<u8>`), so it is no candidate by its
/// own cause, but a type of another crate is a candidate of either kind.
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
    let context = analyzer.intern_trait(&item);
    let referrer = analyzer.result().finish().unwrap();
    assert_eq!(referrer.iter().count(), 2);
    assert!(referrer
        .iter()
        .all(|(_, _, context)| context.self_ty.is_some()));
    let mut copy = item.clone();
    common::visitor(&referrer, context, common::placeholder).visit_item_trait_mut(&mut copy);
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
    assert_eq!(squash(&copy), squash(&expected));
}

/// `Rules::std_crates()` allows the absolute paths of `std` and `alloc` too: the Leaker makes no
/// entry of them. On a result it changes no entry.
#[test]
fn std_crates_allows_the_absolute_paths_of_std_and_alloc() {
    let item: ItemTrait = parse_quote!(
        trait T {
            fn f(&self, a: ::std::vec::Vec<u8>, b: ::alloc::string::String, c: ::foo::X) -> Rel;
        }
    );
    // As a result: `::std`, `::alloc` and `::foo` all make entries.
    let mut analyzer = leaker();
    analyzer.intern_trait(&item);
    let result = analyzer.result();
    let plain = result.clone().finish().unwrap();
    assert_eq!(
        entries(&plain),
        [
            "Rel",
            "::alloc::string::String",
            "::foo::X",
            "::std::vec::Vec<u8>",
        ]
    );
    // The rule changes no entry.
    let allowed = result.allow(Rules::new().std_crates()).finish().unwrap();
    assert_eq!(allowed, plain);
    // In the Leaker: they are no entries, and the others are.
    let mut analyzer = Leaker::with_rule(Rules::new().primitive().std_crates());
    analyzer.intern_trait(&item);
    assert_eq!(
        entries(&analyzer.result().finish().unwrap()),
        ["Rel", "::foo::X"]
    );
}

/// `Rules::external_crates()` allows every absolute path: the Leaker makes no entry of them. On a
/// result it changes no entry.
#[test]
fn external_crates_makes_no_entry_of_a_crate_in_the_leaker() {
    let item: ItemTrait = parse_quote!(
        trait T {
            fn f<U: ::bar::B>(&self, a: ::foo::X, b: ::foo::W<(Rel, U)>, c: [u8; ::n::N]) -> Rel;
        }
    );
    let mut analyzer = Leaker::with_rule(Rules::new().primitive().external_crates());
    analyzer.intern_trait(&item);
    let referrer = analyzer.result().finish().unwrap();
    // The types in the paths are entries, and no absolute path is.
    assert!(common::types(&referrer).all(|ty| !squash(quote!(#ty)).starts_with("::")));
    assert!(entries(&referrer).iter().any(|e| e == "Rel"));
    // Relative paths are not allowed by it.
    let item: ItemTrait = parse_quote!(
        trait T {
            fn f(&self) -> Rel;
        }
    );
    let mut analyzer = Leaker::with_rule(Rules::new().primitive().external_crates());
    analyzer.intern_trait(&item);
    assert_eq!(analyzer.result().finish().unwrap().iter().count(), 1);
}

#[test]
fn referrer_round_trip_keeps_the_indices() {
    // The entries that depend on the context come first (so their indices do not depend on the
    // types of other crates), and the others follow, also with several contexts: they are numbered
    // the same by the Referrer which parses the serialized one.
    let item: ItemTrait = parse_quote! {
        trait T {
            fn a(&self) -> ::z::A;
            fn f<U>(&self, u: ::foo::V<U>) -> Rel<U>;
            fn g(&self) -> Rel2;
            fn h<const B: bool>(&self) -> (::foo::W<B>, Wrap<B>);
        }
    };
    let mut leaker = leaker();
    leaker.intern_trait(&item);
    let referrer = leaker.result().finish().unwrap();
    let before = entries(&referrer);
    assert!(common::context_ids(&referrer).len() > 2);
    // The contexts of the entries are not in ascending order: the later entries are in a context
    // that an earlier one has.
    let contexts: Vec<ContextId> = (0..before.len())
        .map(|ix| common::entry_id(&referrer, ix))
        .collect();
    assert!(contexts.windows(2).any(|w| w[0] > w[1]), "{:?}", contexts);
    let parsed: Referrer = parse2(quote!(#referrer)).unwrap();
    assert_eq!(entries(&parsed), before);
    assert_eq!(parsed, referrer);
    for ix in 0..before.len() {
        assert_eq!(
            common::entry_id(&parsed, ix),
            common::entry_id(&referrer, ix)
        );
        assert_eq!(
            &common::entry(&parsed, ix).generics,
            &common::entry(&referrer, ix).generics
        );
    }
}

#[test]
fn referrer_plain_entry_syntax() {
    let version = common::format_tag();
    // The entries keep the order they are written in.
    let referrer: Referrer = parse2(quote!(#version ([u8; 3], X<[u8]>, Y))).unwrap();
    assert_eq!(entries(&referrer), vec!["[u8;3]", "X<[u8]>", "Y"]);
}

#[test]
fn raw_identifier_crates_are_serialized_back_and_allowed_by_name() {
    use template_quote::ToTokens;
    let item: ItemTrait = parse_quote! {
        trait T {
            fn f(&self, a: ::r#async::X, b: ::r#fn::Y, c: ::plain::Z);
        }
    };
    let mut leaker = leaker();
    leaker.intern_trait(&item);
    let referrer = leaker.result().finish().unwrap();
    assert_eq!(
        entries(&referrer),
        ["::plain::Z", "::r#async::X", "::r#fn::Y",]
    );
    // It is serialized with the raw identifiers, and parsed back.
    let tokens = referrer.to_token_stream();
    assert!(tokens.to_string().contains("r#async"));
    assert_eq!(parse2::<Referrer>(tokens).unwrap(), referrer);
    // Allowed by the Leaker, the crate is not a node, by its name (without `r#`).
    let mut leaker = leaker_allowing(&["r#async"]);
    leaker.intern_trait(&item);
    assert_eq!(
        entries(&leaker.result().finish().unwrap()),
        ["::plain::Z", "::r#fn::Y"]
    );
}

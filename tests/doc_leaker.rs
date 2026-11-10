//! Examples moved out of the docs of the Leaker, the LeakerResult and `Allow`: each test
//! is the longer example of a public item.

mod common;

use common::Recovered;

use syn::*;
use type_leak::*;

/// `Leaker::assertions()` checks the primitive names that were copied as written.
#[test]
fn assertions_are_empty_unless_a_primitive_name_was_copied() {
    let mut leaker = Leaker::new();
    leaker.intern_struct(&parse_quote! { struct S<const N: u8> { a: Rel } });
    assert!(!leaker.assertions().is_empty()); // it checks that `u8` is the primitive type
}

/// `intern_struct()` leaks `Rel<Self>` too.
#[test]
fn intern_struct_leaks_a_type_that_names_self() {
    let item: ItemStruct = parse_quote! { struct S<T> { a: Wrap<T>, b: Rel<Self> } };
    let mut leaker = Leaker::new();
    leaker.intern_struct(&item);
    assert_eq!(leaker.result().finish().unwrap().iter().count(), 2);
}

/// `intern_enum()` leaks the types of the fields of every variant.
#[test]
fn intern_enum_leaks_the_types_of_the_variants() {
    let item: ItemEnum = parse_quote! { enum E<T> { A(Wrap<T>), B { next: Rel2<Self> } } };
    let mut leaker = Leaker::new();
    leaker.intern_enum(&item);
    assert_eq!(leaker.result().finish().unwrap().iter().count(), 2);
}

/// `intern_union()` leaks the types of the fields.
#[test]
fn intern_union_leaks_the_types_of_the_fields() {
    let item: ItemUnion = parse_quote! { union U<T: ::core::marker::Copy> { a: Wrap<T>, b: Rel } };
    let mut leaker = Leaker::new();
    leaker.intern_union(&item);
    assert_eq!(leaker.result().finish().unwrap().iter().count(), 2);
}

/// `intern_impl()` leaks the trait arguments and the signature, not the body.
#[test]
fn intern_impl_skips_the_body() {
    let item: ItemImpl = parse_quote! {
        impl<T> ::path::MyTrait<Rel<T>> for ::path::Host<T> {
            fn f(&self) -> Rel2<Self> { Skipped::new() } // the body is skipped
        }
    };
    let mut leaker = Leaker::new();
    leaker.intern_impl(&item);
    // `Rel<T>` and `Rel2<Self>`, and the self type, which names another crate.
    assert_eq!(leaker.result().finish().unwrap().iter().count(), 3);
}

/// `enter_generics()` records the error of an inner scope, and the outer one goes on.
#[test]
fn enter_generics_goes_on_after_an_error_of_an_inner_scope() {
    let mut leaker = Leaker::new();
    leaker.enter_generics(&parse_quote!(<T>), |view| {
        view.visit_type(&parse_quote!(Rel<T>));
        // A relative trait cannot be leaked: an error of the inner scope, the outer one goes on.
        view.enter_generics(&parse_quote!(<U: Relative>), |view| {
            view.visit_type_param_bound(&parse_quote!(Relative));
        });
    });
    let result = leaker.result();
    assert_eq!(result.errors().len(), 1);
    assert_eq!(result.recovered().iter().count(), 1);
}

/// `enter_type()` leaks only what the closure visits.
#[test]
fn enter_type_leaks_only_the_visited_types() {
    let item: ItemStruct = parse_quote! { struct S<T> { a: Wrap<T>, b: Skipped } };
    let mut leaker = Leaker::new();
    leaker.enter_type(&item.ident, &item.generics, |view| {
        view.visit_type(&item.fields.iter().next().unwrap().ty); // only the first field
    });
    assert_eq!(leaker.result().finish().unwrap().iter().count(), 1); // `Wrap<T>`
}

/// `enter_impl()` leaks only what the closure visits.
#[test]
fn enter_impl_leaks_only_the_visited_trait_arguments() {
    let item: ItemImpl = parse_quote! { impl<T> ::path::MyTrait<Rel<T>> for MyType<T, Rel2> {} };
    let (_, trait_path, _) = item.trait_.as_ref().unwrap();
    let mut leaker = Leaker::new();
    leaker.enter_impl(&item.generics, &item.self_ty, Some(trait_path), |view| {
        for segment in &trait_path.segments {
            view.visit_path_arguments(&segment.arguments);
        }
    });
    assert_eq!(leaker.result().finish().unwrap().iter().count(), 1); // `Rel<T>`, not `Rel2`
}

/// `enter_trait()` leaks only what the closure visits.
#[test]
fn enter_trait_leaks_only_the_visited_signature() {
    use syn::visit::Visit;
    let item: ItemTrait = parse_quote! {
        trait Tr<T> { fn f(&self, t: T) -> Wrap<T>; fn g(&self); }
    };
    let TraitItem::Fn(method) = &item.items[0] else {
        unreachable!()
    };
    let mut leaker = Leaker::new();
    leaker.enter_trait(&item.ident, &item.generics, |view| {
        view.visit_signature(&method.sig)
    });
    assert_eq!(leaker.result().finish().unwrap().iter().count(), 1); // `Wrap<T>`
}

/// `result()` puts the entries that depend on the context first, then the types of other crates.
#[test]
fn result_lists_the_types_of_other_crates_last() {
    let item: ItemTrait = parse_quote! {
        trait Tr { fn f(&self) -> ::foo::X; fn g(&self) -> Rel; }
    };
    let mut leaker = Leaker::new();
    leaker.intern_trait(&item);
    let referrer = leaker.result().finish().unwrap();
    // `Rel` depends on the context. `::foo::X` names another crate: it comes after.
    let found: Vec<&Type> = referrer.iter().map(|(_, ty, _)| ty).collect();
    assert_eq!(found, [&parse_quote!(Rel), &parse_quote!(::foo::X)]);
}

/// `LeakerResult::allow()` drops the error, and the part is copied as written either way.
#[test]
fn allow_a_cause_copies_that_part_as_written() {
    let item: ItemImpl = parse_quote! {
        impl ::p::S {
            fn f(_: ::p::W<'_, dyn RelTr>) {}  // an elided lifetime, and `RelTr` is relative
            fn g(_: ::p::X<dyn RelTr>) {}      // fine: `dyn RelTr` is leaked
        }
    };
    let mut leaker = Leaker::new();
    leaker.intern_impl(&item);
    let mut result = leaker.result();
    // The error is about the trait, which is smaller than the type `W<'_, dyn RelTr>` it is in.
    assert_eq!(result.errors()[0].cause, Cause::Trait(parse_quote!(RelTr)));

    // `f` is copied as written either way, and only `g` leaks `dyn RelTr`: the entries are that
    // type in `g` and the self type.
    result = result.allow(|c: &Cause, _: Place| *c == Cause::Trait(parse_quote!(RelTr)));
    assert_eq!(result.finish().unwrap().iter().count(), 2);
}

/// `allow()` changes no entry: a rule that allows the crate of a type of another crate leaves it an
/// entry, which the copy replaces. Given to the Leaker, the rule makes it no entry.
#[test]
fn allow_changes_no_entry() {
    let mut leaker = Leaker::new();
    let context = leaker.intern_struct(&parse_quote! { struct S { a: ::foo::X } });
    let result = leaker.result();
    let ty: Type = parse_quote!(::foo::X);
    let replaced: Type = parse_quote!(Replaced);
    let referrer = result.clone().finish().unwrap();
    assert_eq!(
        common::expand(&referrer, context, ty.clone(), |_, _, _| replaced.clone()),
        replaced
    );
    let result = result.allow(Rules::new().external_crates());
    let referrer = result.finish().unwrap();
    assert_eq!(
        common::expand(&referrer, context, ty.clone(), |_, _, _| replaced.clone()),
        replaced
    );
    // A Leaker that has the rule finds no entry, so the copy writes the type as it is.
    let mut leaker = Leaker::with_rule(Rules::new().external_crates());
    let context = leaker.intern_struct(&parse_quote! { struct S { a: ::foo::X } });
    let referrer = leaker.result().finish().unwrap();
    assert!(referrer.is_empty());
    assert_eq!(
        common::expand(&referrer, context, ty.clone(), |_, _, _| replaced.clone()),
        ty
    );
}

/// A `LeakerResult` is written as tokens and read back with the errors that remain.
#[test]
fn leaker_result_with_its_errors_is_read_back_from_tokens() {
    use template_quote::ToTokens;
    let item: ItemTrait = parse_quote! { trait Tr: core::fmt::Debug { fn c(&self) -> Rel; } };
    let mut leaker = Leaker::new();
    leaker.intern_trait(&item);
    let tokens = leaker.result().to_token_stream();

    let result: LeakerResult = parse2(tokens).unwrap();
    assert_eq!(result.errors().len(), 1);
    assert!(result.allow(Rules::new().std_crates()).errors().is_empty());
}

/// `check_values` reads a value: it leaks the types in it and rejects a path that is not one.
#[test]
fn check_values_leaks_types_in_values_and_rejects_other_relative_paths() {
    use syn::visit_mut::VisitMut;
    let item: ItemTrait = parse_quote! {
        trait Tr { const N: usize = ::core::mem::size_of::<inner::Local>(); }
    };
    // Without the option the value is not read: the copy breaks without an error.
    let config = Rules::new().primitive();
    let mut leaker = Leaker::with_rule(config);
    leaker.intern_trait(&item);
    assert!(leaker.result().finish().unwrap().is_empty());

    // With it the type is an entry, which the visitor replaces in the copy of the value.
    let config = || Rules::new().primitive();
    let mut leaker = Leaker::with_rule(config());
    leaker.check_values = true;
    let context = leaker.intern_trait(&item);
    let referrer = leaker.result().finish().unwrap();
    assert_eq!(referrer.iter().count(), 1);
    let mut copy = item.clone();
    let projection = common::projection(parse_quote!(Repeater), 7);
    referrer
        .visitor(context, &projection)
        .visit_item_trait_mut(&mut copy);
    let key = common::key(7, common::entry_id_at(&referrer, 0));
    let expected: ItemTrait = parse_quote! {
        trait Tr {
            const N: usize = ::core::mem::size_of::<<Self as Repeater<(), #key>>::Type>();
        }
    };
    assert_eq!(copy, expected);

    // A path that is not a type cannot be replaced.
    let item: ItemTrait = parse_quote! { trait Tr { const N: usize = inner::N; } };
    let mut leaker = Leaker::with_rule(config());
    leaker.check_values = true;
    leaker.intern_trait(&item);
    let result = leaker.result();
    let error = &result.errors()[0];
    assert_eq!(error.message, "use absolute path: 'inner::N'");
    assert_eq!(
        error.notes.first().map(|(_, note)| note.as_str()),
        Some("it is in the default value of `N`, which the copy writes as it is")
    );
}

/// `Rel2<Self, T>` is an entry for any `Self` with the bound, with the error `NestedSelfInBounds`
/// that `allow()` drops.
#[test]
fn a_nested_type_that_names_self_with_a_self_bound_is_an_entry_with_an_error() {
    let item: Item = parse_quote! {
        pub struct S<T: ::core::convert::AsRef<Self>> { pub b: Rel2<Self, T> }
    };
    let mut leaker = Leaker::with_rule(Rules::new());
    leaker.enter_generics(&Generics::default(), |view| view.visit_item(&item));
    let result = leaker.result();
    // The risk is an error, with no paths, and the type is an entry all the same.
    assert_eq!(result.errors().len(), 1);
    assert_eq!(result.errors()[0].reason, Reason::NestedSelfInBounds);
    assert!(result.errors()[0].paths.is_empty());
    assert!(result.clone().finish().is_err());
    assert_eq!(result.clone().recovered().iter().count(), 1);

    // A rule that allows the cause of the error accepts the risk.
    let cause = result.errors()[0].cause.clone();
    let accepted = cause.clone();
    let result = result.allow(move |c: &Cause, _: Place| *c == accepted);
    let referrer = result.clone().finish().unwrap();
    // The entry is for any type that the bound holds for, with `Self` as its implementor.
    let (_, _, context) = referrer.iter().next().unwrap();
    assert!(matches!(context.self_ty, Some(SelfRequirement::Any)));
    let repeater: Path = parse_quote!(Repeater);
    let param: Ident = parse_quote!(__S);
    let r = result
        .repeater_impls(&Projection::new(
            repeater.clone(),
            7,
            ImplSelf::Param(param.clone()),
        ))
        .unwrap()
        .remove(0);
    // `r.path` has no `KEY`, and `ToTokens` writes the one of the id 7.
    let key = common::key(7, r.entry);
    let imp: ItemImpl = parse_quote!(#r);
    let expected: ItemImpl = parse_quote! {
        impl<T: ::core::convert::AsRef<__S>, __S> Repeater<(T,), #key> for __S {
            type Type = Rel2<__S, T>;
        }
    };
    assert_eq!(imp, expected);

    // The rule of the Leaker drops the error as well, at the end of the call: the entry is the same.
    let mut leaker = Leaker::with_rule(move |c: &Cause, _: Place| *c == cause);
    leaker.enter_generics(&Generics::default(), |view| view.visit_item(&item));
    let result = leaker.result();
    assert!(result.errors().is_empty());
    assert_eq!(result.finish().unwrap(), referrer);
}

/// `?Sized` is left as written, and `Self: Sized` is a relative trait.
#[test]
fn allow_prelude_is_needed_for_sized_but_not_for_relaxed_sized() {
    let relaxed: ItemTrait = parse_quote! { trait Tr<T: ?Sized> { fn f(&self, t: &T); } };
    let mut leaker = Leaker::new();
    leaker.intern_trait(&relaxed);
    assert!(leaker.result().finish().is_ok());
    let required: ItemTrait = parse_quote! { trait Tr { fn f(&self) where Self: Sized; } };
    let mut leaker = Leaker::new();
    leaker.intern_trait(&required);
    assert_eq!(
        leaker.result().errors()[0].message,
        "use absolute path: 'Sized'"
    );
}

/// `Rules::prelude()` leaves `Clone` and `Vec` as written, so `Vec<Rel>` is the only entry.
#[test]
fn allow_prelude_leaves_the_names_of_the_prelude_as_written() {
    let item: ItemTrait = parse_quote! { trait Tr: Clone { fn f(&self) -> Vec<Rel>; } };
    let mut leaker = Leaker::new();
    leaker.intern_trait(&item);
    assert!(leaker.result().finish().is_err());

    let config = Rules::new().prelude();
    let mut leaker = Leaker::with_rule(config);
    leaker.intern_trait(&item);
    // `Clone` and `Vec` are left as written. The only entry is `Vec<Rel>`: a bare `Rel` may be
    // a const, so the type that holds it is leaked whole.
    assert_eq!(leaker.result().finish().unwrap().iter().count(), 1);
}

/// `Rules::crate_paths()` leaves `crate::Foo` as written, which is leaked by default.
#[test]
fn allow_crate_leaves_crate_paths_as_written() {
    let item: ItemStruct = parse_quote! { struct S { a: crate::Foo } };
    // `crate::Foo` is leaked by default...
    let mut leaker = Leaker::new();
    leaker.intern_struct(&item);
    assert_eq!(leaker.result().finish().unwrap().iter().count(), 1);
    // ...and left as written if the copy is in the same crate.
    let config = Rules::new().crate_paths();
    let mut leaker = Leaker::with_rule(config);
    leaker.intern_struct(&item);
    assert!(leaker.result().finish().unwrap().is_empty());
}

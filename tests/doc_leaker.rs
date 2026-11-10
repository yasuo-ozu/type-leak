//! Examples moved out of the docs of the Leaker, the LeakerResult and `Allow`: each test
//! is the longer example of a public item.

use syn::*;
use type_leak::*;

/// `Leaker::assertions()` checks the primitive names that were copied as written.
#[test]
fn assertions_are_empty_unless_a_primitive_name_was_copied() {
    let mut leaker = Leaker::new();
    leaker.intern_struct(&parse_quote! { struct S<const N: u8> { a: Rel } }, true);
    assert!(!leaker.assertions().is_empty()); // it checks that `u8` is the primitive type
}

/// `intern_struct()` with `allow_self` leaks `Rel<Self>`, and without it is an error.
#[test]
fn intern_struct_allow_self_leaks_or_denies_a_type_that_names_self() {
    let item: ItemStruct = parse_quote! { struct S<T> { a: Wrap<T>, b: Rel<Self> } };
    let mut leaker = Leaker::new();
    leaker.intern_struct(&item, true);
    assert_eq!(leaker.result().finish().unwrap().iter().count(), 2);
    let mut leaker = Leaker::new();
    leaker.intern_struct(&item, false);
    assert!(leaker.result().finish().is_err());
}

/// `intern_enum()` leaks the types of the fields of every variant.
#[test]
fn intern_enum_leaks_the_types_of_the_variants() {
    let item: ItemEnum = parse_quote! { enum E<T> { A(Wrap<T>), B { next: Rel2<Self> } } };
    let mut leaker = Leaker::new();
    leaker.intern_enum(&item, true);
    assert_eq!(leaker.result().finish().unwrap().iter().count(), 2);
}

/// `intern_union()` leaks the types of the fields.
#[test]
fn intern_union_leaks_the_types_of_the_fields() {
    let item: ItemUnion = parse_quote! { union U<T: ::core::marker::Copy> { a: Wrap<T>, b: Rel } };
    let mut leaker = Leaker::new();
    leaker.intern_union(&item, true);
    assert_eq!(leaker.result().finish().unwrap().iter().count(), 2);
}

/// `intern_impl()` leaks the trait arguments and the signature, not the body.
#[test]
fn intern_impl_skips_the_body_and_makes_another_crate_optional() {
    let item: ItemImpl = parse_quote! {
        impl<T> ::path::MyTrait<Rel<T>> for ::path::Host<T> {
            fn f(&self) -> Rel2<Self> { Skipped::new() } // the body is skipped
        }
    };
    let mut leaker = Leaker::new();
    leaker.intern_impl(&item, true);
    // `Rel<T>` and `Rel2<Self>` (`::path::Host<T>` names another crate: it is optional).
    assert_eq!(leaker.result().finish().unwrap().required().count(), 2);
}

/// `intern_mod()` copies the declared names, and numbers the scope of a nested module.
#[test]
fn intern_mod_does_not_leak_the_names_the_module_declares() {
    let item: ItemMod = parse_quote! {
        mod m {
            struct Local;
            struct S { a: Local, b: Outside }
            mod inner { struct T(Outside2); }
        }
    };
    let mut leaker = Leaker::new();
    leaker.intern_mod(&item);
    let referrer = leaker.result().finish().unwrap();
    // `Local` is declared in `m`: it is not leaked.
    assert_eq!(referrer.iter().count(), 2);
    // The Repeater impl of `Outside2` is written in `inner`.
    let module = referrer.context_module(referrer.context(1)).unwrap();
    assert_eq!(referrer.module_path(module).unwrap()[0], "inner");
}

/// `enter_mod()` copies the names given as declared, and leaks the others.
#[test]
fn enter_mod_does_not_leak_the_declared_names() {
    let item: ItemStruct = parse_quote! { struct S { a: Local, b: Outside } };
    let mut leaker = Leaker::new();
    let declared = [item.ident.clone(), parse_quote!(Local)]; // the module declares them
    leaker.enter_mod("my_module", declared, |view| {
        view.visit_item(&Item::Struct(item.clone()))
    });
    assert_eq!(leaker.result().finish().unwrap().iter().count(), 1); // only `Outside`
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
    assert_eq!(result.into_referrer().iter().count(), 1);
}

/// `enter_type()` leaks only what the closure visits.
#[test]
fn enter_type_leaks_only_the_visited_types() {
    let item: ItemStruct = parse_quote! { struct S<T> { a: Wrap<T>, b: Skipped } };
    let mut leaker = Leaker::new();
    leaker.enter_type(&item.ident, &item.generics, true, |view| {
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
    leaker.enter_impl(&item.generics, &item.self_ty, true, |view| {
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
    leaker.enter_trait(&item.ident, &item.generics, true, |view| {
        view.visit_signature(&method.sig)
    });
    assert_eq!(leaker.result().finish().unwrap().iter().count(), 1); // `Wrap<T>`
}

/// `result()` puts the required entries first, then the optional ones.
#[test]
fn result_lists_required_entries_before_optional_ones() {
    let item: ItemTrait = parse_quote! {
        trait Tr { fn f(&self) -> ::foo::X; fn g(&self) -> Rel; }
    };
    let mut leaker = Leaker::new();
    leaker.intern_trait(&item, true);
    let referrer = leaker.result().finish().unwrap();
    // `Rel` is required. `::foo::X` is optional: it names another crate.
    assert_eq!(referrer.iter().count(), 2);
    assert!(referrer.must(0) && !referrer.must(1));
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
    leaker.intern_impl(&item, true);
    let mut result = leaker.result();
    // The error is about the trait, which is smaller than the type `W<'_, dyn RelTr>` it is in.
    assert_eq!(result.errors()[0].cause, Cause::Trait(parse_quote!(RelTr)));

    // `f` is copied as written either way, and only `g` leaks `dyn RelTr`.
    result.allow(|e| e.cause == Cause::Trait(parse_quote!(RelTr)));
    assert_eq!(result.finish().unwrap().required().count(), 1);
}

/// `deny_ambiguous_crate_name()` makes the Referrer replace an optional entry.
#[test]
fn deny_ambiguous_crate_name_replaces_optional_entries() {
    let mut leaker = Leaker::new();
    let context = leaker.intern_struct(&parse_quote! { struct S { a: ::foo::X } }, true);
    let mut result = leaker.result();
    let ty: Type = parse_quote!(::foo::X);
    let replaced: Type = parse_quote!(Replaced);
    let referrer = result.clone().finish().unwrap();
    assert_eq!(
        referrer.expand(context, ty.clone(), |_, _| replaced.clone()),
        ty
    );
    let referrer = result.deny_ambiguous_crate_name().clone().finish().unwrap();
    assert_eq!(
        referrer.expand(context, ty.clone(), |_, _| replaced.clone()),
        replaced
    );
}

/// A `LeakerResult` is written as tokens and read back with the errors that remain.
#[test]
fn leaker_result_with_its_errors_is_read_back_from_tokens() {
    use template_quote::ToTokens;
    let item: ItemTrait = parse_quote! { trait Tr: core::fmt::Debug { fn c(&self) -> Rel; } };
    let mut leaker = Leaker::new();
    leaker.intern_trait(&item, true);
    let tokens = leaker.result().to_token_stream();

    let mut result: LeakerResult = parse2(tokens).unwrap();
    assert_eq!(result.errors().len(), 1);
    assert!(result.allow_std_crates().errors().is_empty());
}

/// `check_values` reads a value: it leaks the types in it and rejects a path that is not one.
#[test]
fn check_values_leaks_types_in_values_and_rejects_other_relative_paths() {
    use syn::visit_mut::VisitMut;
    let item: ItemTrait = parse_quote! {
        trait Tr { const N: usize = ::core::mem::size_of::<inner::Local>(); }
    };
    // Without the option the value is not read: the copy breaks without an error.
    let config = Allow::new().allow_primitive();
    let mut leaker = Leaker::from_config(config, false, false);
    leaker.intern_trait(&item, true);
    assert!(leaker.result().finish().unwrap().is_empty());

    // With it the type is an entry, which the visitor replaces in the copy of the value.
    let config = || Allow::new().allow_primitive();
    let mut leaker = Leaker::from_config(config(), true, false);
    let context = leaker.intern_trait(&item, true);
    let referrer = leaker.result().finish().unwrap();
    assert_eq!(referrer.required().count(), 1);
    let mut copy = item.clone();
    referrer
        .visitor(context, |_, ix| parse_quote!(<() as Repeater<#ix>>::Type))
        .visit_item_trait_mut(&mut copy);
    let expected: ItemTrait = parse_quote! {
        trait Tr {
            const N: usize = ::core::mem::size_of::<<() as Repeater<0usize>>::Type>();
        }
    };
    assert_eq!(copy, expected);

    // A path that is not a type cannot be replaced.
    let item: ItemTrait = parse_quote! { trait Tr { const N: usize = inner::N; } };
    let mut leaker = Leaker::from_config(config(), true, false);
    leaker.intern_trait(&item, true);
    let result = leaker.result();
    let error = &result.errors()[0];
    assert_eq!(error.message, "use absolute path: 'inner :: N'");
    assert_eq!(
        error.notes.first().map(|(_, note)| note.as_str()),
        Some("it is in the default value of `N`, which the copy writes as it is")
    );
}

/// `nested_self_entries` makes the entry `Rel2<Self, T>` one for any `Self` with the bound.
#[test]
fn nested_self_entries_accepts_self_in_a_nested_type_with_a_self_bound() {
    let item: Item = parse_quote! {
        pub struct S<T: ::core::convert::AsRef<Self>> { pub b: Rel2<Self, T> }
    };
    let nested = |leaker: &mut Leaker| {
        leaker.enter_mod("m", [parse_quote!(S)], |view| view.visit_item(&item));
    };
    // Without the option it is an error: the bound and the type name `Self`.
    let mut leaker = Leaker::new();
    nested(&mut leaker);
    assert_eq!(leaker.result().errors().len(), 1);

    let mut leaker = Leaker::from_config(Allow::new(), false, true);
    nested(&mut leaker);
    let referrer = leaker.result().finish().unwrap();
    // The entry is for any type that the bound holds for, with `Self` as its implementor.
    assert!(matches!(
        referrer.context_self(referrer.context(0)),
        Some(SelfRequirement::Any)
    ));
    let param: Ident = parse_quote!(__S);
    let r = referrer
        .repeater_impl(0, ImplSelf::Param(&param), array_const)
        .unwrap();
    let (generics, args, ty) = (&r.generics, &r.args, &r.ty);
    let imp: ItemImpl = parse_quote! {
        impl #generics Repeater<#args, 7u64, 0usize> for __S { type Type = #ty; }
    };
    let expected: ItemImpl = parse_quote! {
        impl<T: ::core::convert::AsRef<__S>, __S> Repeater<(T,), 7u64, 0usize> for __S {
            type Type = Rel2<__S, T>;
        }
    };
    assert_eq!(imp, expected);
}

/// `?Sized` is left as written, and `Self: Sized` is a relative trait.
#[test]
fn allow_prelude_is_needed_for_sized_but_not_for_relaxed_sized() {
    let relaxed: ItemTrait = parse_quote! { trait Tr<T: ?Sized> { fn f(&self, t: &T); } };
    let mut leaker = Leaker::new();
    leaker.intern_trait(&relaxed, true);
    assert!(leaker.result().finish().is_ok());
    let required: ItemTrait = parse_quote! { trait Tr { fn f(&self) where Self: Sized; } };
    let mut leaker = Leaker::new();
    leaker.intern_trait(&required, true);
    assert_eq!(
        leaker.result().errors()[0].message,
        "use absolute path: 'Sized'"
    );
}

/// `allow_prelude()` leaves `Clone` and `Vec` as written, so `Vec<Rel>` is the only entry.
#[test]
fn allow_prelude_leaves_the_names_of_the_prelude_as_written() {
    let item: ItemTrait = parse_quote! { trait Tr: Clone { fn f(&self) -> Vec<Rel>; } };
    let mut leaker = Leaker::new();
    leaker.intern_trait(&item, true);
    assert!(leaker.result().finish().is_err());

    let config = Allow::new().allow_prelude();
    let mut leaker = Leaker::from_config(config, false, false);
    leaker.intern_trait(&item, true);
    // `Clone` and `Vec` are left as written. The only entry is `Vec<Rel>`: a bare `Rel` may be
    // a const, so the type that holds it is leaked whole.
    assert_eq!(leaker.result().finish().unwrap().iter().count(), 1);
}

/// `allow_crate()` leaves `crate::Foo` as written, which is leaked by default.
#[test]
fn allow_crate_leaves_crate_paths_as_written() {
    let item: ItemStruct = parse_quote! { struct S { a: crate::Foo } };
    // `crate::Foo` is leaked by default...
    let mut leaker = Leaker::new();
    leaker.intern_struct(&item, true);
    assert_eq!(leaker.result().finish().unwrap().iter().count(), 1);
    // ...and left as written if the copy is in the same crate.
    let config = Allow::new().allow_crate();
    let mut leaker = Leaker::from_config(config, false, false);
    leaker.intern_struct(&item, true);
    assert!(leaker.result().finish().unwrap().is_empty());
}

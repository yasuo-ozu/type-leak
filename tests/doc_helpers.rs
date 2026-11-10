//! Examples moved out of the API docs: each test is the longer example of a public item.

mod common;

use syn::*;
use template_quote::ToTokens;
use type_leak::*;

/// `repeater_impls()` returns the pieces of an impl, and the module in which to write it.
#[test]
fn repeater_impl_pieces_make_the_impl_and_name_its_module() {
    let item: ItemMod = parse_quote!(
        mod m {
            mod inner {
                struct S {
                    a: Outside,
                }
            }
        }
    );
    let mut leaker = Leaker::new();
    leaker.intern_mod(&item);
    let result = leaker.result();

    let repeater: Path = parse_quote!(crate::Repeater);
    let param: Ident = parse_quote!(__S);
    let r = result
        .repeater_impls(&repeater, 7, ImplSelf::Param(&param), slot_const(&repeater))
        .unwrap()
        .next()
        .unwrap();
    // `r.path` has no `KEY`, and `ToTokens` writes the one of the id 7.
    let key = repeater_key(7, r.entry);
    let imp: ItemImpl = parse_quote!(#r);
    let expected: ItemImpl = parse_quote! {
        impl<__S: ?::core::marker::Sized> crate::Repeater<(), #key> for __S {
            type Type = Outside;
        }
    };
    assert_eq!(imp, expected);

    // `Outside` is a relative path: it means something in `m::inner` only. Write the impl there.
    let path = match &r.module {
        Some(ModuleScope::Itself {
            path: Some(path), ..
        }) => path,
        _ => panic!("not a module of intern_mod()"),
    };
    assert_eq!(path.len(), 1);
    assert_eq!(path[0], "inner");
}

/// The encoder passes the generics by name at an identity site, and substitutes them at another.
#[test]
fn use_site_identity_and_substituting_give_the_arguments() {
    let tr: ItemTrait = parse_quote!(
        trait Tr<'a, T: 'a> {
            fn f(&self, x: &'a Wrap<T>);
        }
    );
    let mut leaker = Leaker::new();
    leaker.intern_trait(&tr, true);
    let referrer = leaker.result().finish().unwrap();

    // `impl<'a, T: 'a> Tr<'a, T> for ..` declares the generics: each is passed by its own name.
    let context = common::entry(&referrer, 0);
    let args = common::repeater_args(context, &UseSite::identity(), None, array_const).unwrap();
    assert_eq!(args, parse_quote!((&'a (), T, &'a T)));

    // `impl Tr<'static, u8> for ..` gives them arguments.
    let trait_args: Vec<GenericArgument> = vec![parse_quote!('static), parse_quote!(u8)];
    let site = UseSite::substituting(&tr.generics.params, &trait_args);
    let args = common::repeater_args(context, &site, None, array_const).unwrap();
    assert_eq!(args, parse_quote!((&'static (), u8, &'static u8)));
}

/// The self type that the encoder gets names the type for the `Self: 'a` that an entry needs.
#[test]
fn the_self_type_of_the_encoder_names_the_type_in_self_outlives() {
    let tr: ItemTrait = parse_quote!(
        trait Tr<'a> {
            fn f(&self, x: &'a Wrap<Self>);
        }
    );
    let mut leaker = Leaker::new();
    leaker.intern_trait(&tr, true);
    let referrer = leaker.result().finish().unwrap();
    // The entry `&'a Wrap<Self>` needs `Self: 'a`, which the arguments encode as `&'a Self`.
    let context = common::entry(&referrer, 0);
    let args = common::repeater_args(context, &UseSite::identity(), None, array_const).unwrap();
    assert_eq!(args, parse_quote!((&'a (), &'a Self)));
    // At a site where `Self` is not available, name the type instead.
    let host: Type = parse_quote!(Host);
    let args =
        common::repeater_args(context, &UseSite::identity(), Some(&host), array_const).unwrap();
    assert_eq!(args, parse_quote!((&'a (), &'a Host)));
}

/// `UseSite::generic_args()` takes the trait's argument for `T`, and passes `U` by its name.
#[test]
fn use_site_generic_args_passes_method_generics_by_name() {
    let tr: ItemTrait = parse_quote!(
        trait Tr<T> {
            fn f<U>(&self, t: T, u: U) -> Wrap<T, U>;
        }
    );
    let mut leaker = Leaker::new();
    leaker.intern_trait(&tr, true);
    let referrer = leaker.result().finish().unwrap();
    // `impl Tr<u8> for ..`: `T` is `u8`. `U` is a generic of the method, passed by its name.
    let trait_args: Vec<GenericArgument> = vec![parse_quote!(u8)];
    let (_, _, context) = referrer.iter().next().unwrap();
    let args = UseSite::substituting(&tr.generics.params, &trait_args)
        .generic_args(&context.generics)
        .unwrap();
    let expected: Vec<GenericArgument> = vec![parse_quote!(u8), parse_quote!(U)];
    assert_eq!(args, expected);
}

/// `bucket_self()` is `None` for an entry that names `Self` or whose context does, a type otherwise.
#[test]
fn bucket_self_is_none_for_self_in_the_type_or_in_its_context() {
    let tr: ItemTrait = parse_quote! {
        trait Tr<T: ::core::convert::AsRef<Self>, U = Rel<T>> {
            fn f(&self, t: T) -> U;
            fn g(&self) -> Wrap<Self>;
        }
    };
    let mut leaker = Leaker::new();
    leaker.intern_trait(&tr, true);
    let referrer = leaker.result().finish().unwrap();
    // `Wrap<Self>` names `Self`; the context of `Rel<T>` has it in the bound of `T`.
    let wrap = referrer
        .iter()
        .position(|(_, ty, _)| *ty == parse_quote!(Wrap<Self>))
        .unwrap();
    let rel = referrer
        .iter()
        .position(|(_, ty, _)| *ty == parse_quote!(Rel<T>))
        .unwrap();
    let (wrap, rel) = (
        common::entry_id_at(&referrer, wrap),
        common::entry_id_at(&referrer, rel),
    );
    assert!(referrer.bucket_self(wrap, 7).is_none() && referrer.bucket_self(rel, 7).is_none());

    let mut leaker = Leaker::new();
    leaker.intern_trait(
        &parse_quote!(
            trait Tr<T> {
                fn f(&self) -> Wrap<T>;
            }
        ),
        true,
    );
    let referrer = leaker.result().finish().unwrap();
    assert!(referrer
        .bucket_self(common::entry_id_at(&referrer, 0), 7)
        .is_some());
}

/// `bucket_self()` is a primitive type for an entry without `Self`, and `None` for one with it.
#[test]
fn bucket_self_is_a_primitive_unless_the_entry_needs_self() {
    let tr: ItemTrait = parse_quote! {
        trait Tr<T> { fn f(&self) -> Wrap<T>; fn g(&self) -> Wrap2<Self>; }
    };
    let mut leaker = Leaker::new();
    leaker.intern_trait(&tr, true);
    let result = leaker.result();
    let referrer = result.clone().finish().unwrap();
    let wrap = referrer
        .iter()
        .position(|(_, ty, _)| *ty == parse_quote!(Wrap<T>))
        .unwrap();
    let wrap2 = referrer
        .iter()
        .position(|(_, ty, _)| *ty == parse_quote!(Wrap2<Self>))
        .unwrap();
    let param: Ident = parse_quote!(__S);
    let target = ImplSelf::Bucket { param: &param };

    // `Wrap<T>` does not name `Self`: a primitive type, and no `__S`.
    let bucket = referrer
        .bucket_self(common::entry_id_at(&referrer, wrap), 7)
        .unwrap();
    let r = result
        .repeater_impls(&parse_quote!(crate::Repeater), 7, target, array_const)
        .unwrap()
        .nth(wrap)
        .unwrap();
    assert_eq!(r.self_ty, bucket);
    assert_eq!(r.generics, parse_quote!(<T>));

    // `Wrap2<Self>` does: the implementors of the trait, as with `ImplSelf::Param`.
    assert_eq!(
        referrer.bucket_self(common::entry_id_at(&referrer, wrap2), 7),
        None
    );
    let r = result
        .repeater_impls(&parse_quote!(crate::Repeater), 7, target, array_const)
        .unwrap()
        .nth(wrap2)
        .unwrap();
    assert_eq!(r.self_ty, parse_quote!(__S));
}

/// Each `visit_*` method of the `View` analyzes one kind of syntax with the scope's generics.
#[test]
fn view_visit_methods_analyze_each_kind_of_syntax() {
    let mut generics: Generics = parse_quote!(<T: ::p::Bound<Rel1>>);
    generics.where_clause = Some(parse_quote!(where T: ::p::Other<Rel2>));
    let mut leaker = Leaker::new();
    leaker.enter_generics(&generics, |view| {
        view.visit_type(&parse_quote!(Wrap<T>)); // a type
        let arguments = PathArguments::AngleBracketed(parse_quote!(<Rel3>));
        view.visit_path_arguments(&arguments); // the arguments of a path resolved elsewhere
        if let Some(GenericParam::Type(param)) = generics.params.first() {
            for bound in &param.bounds {
                view.visit_type_param_bound(bound); // a bound
            }
        }
        view.visit_where_clause(generics.where_clause.as_ref().unwrap()); // a where clause
    });
    // `Wrap<T>`, `Rel3`, `Rel1` and `Rel2`.
    assert_eq!(leaker.result().finish().unwrap().iter().count(), 4);
}

/// `View::check()` reports the causes of a type and leaves no trace in the Leaker.
#[test]
fn view_check_reports_the_causes_and_records_nothing() {
    let generics: Generics = parse_quote!(<T>);
    let mut leaker = Leaker::new();
    let (_, result) = leaker.enter_generics(&generics, |view| {
        view.check(&parse_quote!(::foo::W<T, (Rel,)>))
    });
    assert_eq!(result.must.len(), 1); // `Rel`
    assert!(!result.is_own && result.optional);
    assert_eq!(leaker.result().finish().unwrap().iter().count(), 0);
}

/// `View::visit_item()` gives a nested struct its own generics next to those around it.
#[test]
fn view_visit_item_keeps_the_generics_around_and_adds_the_local_ones() {
    let mut leaker = Leaker::new();
    leaker.enter_generics(&parse_quote!(<T>), |view| {
        // `T` is a generic of the scope, and `U` is local to the struct. Each type is an
        // entry with the generics it uses.
        view.visit_item(&parse_quote!(
            struct Inner<U> {
                a: Rel<U>,
                b: Wrap<T>,
            }
        ));
    });
    let referrer = leaker.result().finish().unwrap();
    assert_eq!(referrer.iter().count(), 2);
    for (_, _, context) in referrer.iter() {
        assert_eq!(context.generics.params.len(), 1);
    }
}

/// `View::visit_type()` leaves out the paths of the standard library and leaks the others.
#[test]
fn view_visit_type_skips_the_paths_of_the_standard_library() {
    let mut leaker = Leaker::new();
    leaker.enter_generics(&parse_quote!(<T>), |view| {
        view.visit_type(&parse_quote!(::core::option::Option<Wrap<T>>));
    });
    // `Option<..>` is a path of the standard library, which is the same in every crate.
    assert_eq!(
        leaker
            .result()
            .finish()
            .unwrap()
            .iter()
            .map(|(_, ty, _)| ty)
            .collect::<Vec<_>>(),
        [&parse_quote!(Wrap<T>)]
    );
}

/// `slot_const()` encodes const parameters of the impl and of the use site into the same slots.
#[test]
fn repeater_slots_encode_the_consts_of_the_impl_and_of_the_use_site() {
    let tr: ItemTrait = parse_quote! {
        trait Tr<const B: bool, const C: char = 'z'> { fn f(&self) -> Wrap<B, C>; }
    };
    let mut leaker = Leaker::new();
    leaker.intern_trait(&tr, true);
    let result = leaker.result();
    let referrer = result.clone().finish().unwrap();
    let repeater: Path = parse_quote!(crate::Repeater);

    // The impl encodes the parameters themselves...
    let param: Ident = parse_quote!(__S);
    let r = result
        .repeater_impls(&repeater, 7, ImplSelf::Param(&param), slot_const(&repeater))
        .unwrap()
        .next()
        .unwrap();
    assert_eq!(
        r.args,
        parse_quote!((
            *const dyn crate::Repeater<(), 0, B, Type = ()>,
            *const dyn crate::Repeater<(), 0, false, C, Type = ()>
        ))
    );

    // ...and a copy in `impl Tr<true, 'x'> for ..` the arguments.
    let trait_args: Vec<GenericArgument> = vec![parse_quote!(true), parse_quote!('x')];
    let args = common::repeater_args(
        common::entry(&referrer, 0),
        &UseSite::substituting(&tr.generics.params, &trait_args),
        None,
        slot_const(&repeater),
    )
    .unwrap();
    assert_eq!(
        args,
        parse_quote!((
            *const dyn crate::Repeater<(), 0, true, Type = ()>,
            *const dyn crate::Repeater<(), 0, false, 'x', Type = ()>
        ))
    );
}

/// A context holds the generics that its entry uses, with those that their bounds name.
#[test]
fn context_holds_the_generics_an_entry_uses() {
    let item: ItemTrait = parse_quote! {
        trait Tr<T, U> where U: ::core::convert::Into<T> {
            fn f(&self) -> Rel;
            fn g(&self) -> Wrap<T>;
            fn h(&self) -> Wrap<U>;
        }
    };
    let mut leaker = Leaker::new();
    leaker.intern_trait(&item, true);
    let referrer = leaker.result().finish().unwrap();
    // `Rel` uses no generic. `Wrap<T>` uses `T`. `Wrap<U>` uses `U` and, through the bound
    // `U: Into<T>`, `T`.
    let used: Vec<usize> = referrer
        .iter()
        .map(|(_, _, context)| context.generics.params.len())
        .collect();
    assert_eq!(used, [0, 1, 2]);
}

/// A marker type serves a trait whose entries do not name `Self`; if one does, the call fails, and
/// the entry needs the parameter.
#[test]
fn impl_self_marker_fails_the_call_if_an_entry_names_self() {
    let tr: ItemTrait = parse_quote! {
        trait Tr<T> { fn f(&self) -> Wrap<T>; fn g(&self) -> Wrap2<Self>; }
    };
    let mut leaker = Leaker::new();
    leaker.intern_trait(&tr, true);
    let result = leaker.result();
    let referrer = result.clone().finish().unwrap();
    let wrap2 = referrer
        .iter()
        .position(|(_, ty, _)| *ty == parse_quote!(Wrap2<Self>))
        .unwrap();

    // `Wrap<T>` does not name `Self`: the impl may be for a marker type of your crate.
    let only_wrap: ItemTrait = parse_quote! {
        trait Tr<T> { fn f(&self) -> Wrap<T>; }
    };
    let mut leaker = Leaker::new();
    leaker.intern_trait(&only_wrap, true);
    let marker: Type = parse_quote!(crate::Marker);
    let r = leaker
        .result()
        .repeater_impls(
            &parse_quote!(crate::Repeater),
            7,
            ImplSelf::Marker(&marker),
            array_const,
        )
        .unwrap()
        .next()
        .unwrap();
    assert_eq!(r.self_ty, marker);
    // `Wrap2<Self>` does: it is for the implementors of the trait, so a marker will not do, and the
    // call fails for it.
    let err = result
        .repeater_impls(
            &parse_quote!(crate::Repeater),
            7,
            ImplSelf::Marker(&marker),
            array_const,
        )
        .err()
        .unwrap();
    assert!(matches!(err, EncodeError::NeedsSelf(ty) if *ty == parse_quote!(Wrap2<Self>)));
    let param: Ident = parse_quote!(__S);
    let r = result
        .repeater_impls(
            &parse_quote!(crate::Repeater),
            7,
            ImplSelf::Param(&param),
            array_const,
        )
        .unwrap()
        .nth(wrap2)
        .unwrap();
    assert_eq!(
        r.generics.params.last().unwrap(),
        &parse_quote!(__S: ?::core::marker::Sized + Tr<T>)
    );
}

/// `context_self()` is `Type` in a struct and `Bound` in a trait.
#[test]
fn self_requirement_is_the_type_in_a_struct_and_the_bound_in_a_trait() {
    let mut leaker = Leaker::new();
    leaker.intern_struct(
        &parse_quote!(
            struct S<T> {
                a: Rel<Self>,
            }
        ),
        true,
    );
    leaker.intern_trait(
        &parse_quote!(
            trait Tr<T> {
                fn f(&self) -> Rel<Self>;
            }
        ),
        true,
    );
    let referrer = leaker.result().finish().unwrap();
    // Each has an entry `Rel<Self>`, with its own context.
    assert_eq!(referrer.iter().count(), 2);
    let self_of = |is_struct: bool| {
        referrer
            .iter()
            .find_map(|(_, _, context)| match &context.self_ty {
                Some(requirement @ SelfRequirement::Type(_)) if is_struct => Some(requirement),
                Some(requirement @ SelfRequirement::Bound(_)) if !is_struct => Some(requirement),
                _ => None,
            })
    };
    // In a struct, `Self` is the struct with its own generics as arguments.
    match self_of(true) {
        Some(SelfRequirement::Type(ty)) => assert_eq!(*ty, parse_quote!(S<T>)),
        _ => unreachable!(),
    }
    // In a trait, `Self` is any type that implements the trait.
    match self_of(false) {
        Some(SelfRequirement::Bound(path)) => assert_eq!(*path, parse_quote!(Tr<T>)),
        _ => unreachable!(),
    }
}

/// The top-level context differs from that of an entry with fewer generics, and the id is an
/// integer literal.
#[test]
fn context_id_is_the_top_level_context_and_an_integer_literal() {
    let item: ItemStruct = parse_quote!(
        struct S<T> {
            a: Wrap<T>,
            b: Rel,
        }
    );
    let mut leaker = Leaker::new();
    let top_level = leaker.intern_struct(&item, true);
    let referrer = leaker.result().finish().unwrap();
    // `Wrap<T>` uses the generics of the struct, as the top-level context has them.
    // `Rel` uses no generic, so its context is another one.
    let wrap = referrer
        .iter()
        .position(|(_, ty, _)| *ty == parse_quote!(Wrap<T>))
        .unwrap();
    let rel = referrer
        .iter()
        .position(|(_, ty, _)| *ty == parse_quote!(Rel))
        .unwrap();
    assert_eq!(common::entry_id(&referrer, wrap), top_level);
    assert_ne!(common::entry_id(&referrer, rel), top_level);

    // It is written as an integer literal, so it travels with the Referrer in a macro call.
    let id: ContextId = parse_str("3usize").unwrap();
    assert_eq!(id.index(), 3);
    assert_eq!(id.to_token_stream().to_string(), "3");
}

/// `repeater_impls()` for every entry of a trait: `Rel` uses no generics, so any type implements
/// it; `Wrap2<Self::Out>` names `Self`, so the implementor is bound by the trait and replaces it.
#[test]
fn repeater_impl_for_each_entry_of_a_trait() {
    let tr: ItemTrait = parse_quote! {
        trait Tr<'a, T: 'a> {
            type Out;
            fn f(&self, x: &'a Wrap<T>) -> Rel;
            fn g(&self) -> Wrap2<Self::Out>;
        }
    };
    let mut leaker = Leaker::new();
    leaker.intern_trait(&tr, true);
    let result = leaker.result();
    let referrer = result.clone().finish().unwrap();

    let self_param: Ident = parse_quote!(__S);
    let repeater: Path = parse_quote!(crate::Repeater);
    let impls: Vec<ItemImpl> = result
        .repeater_impls(
            &repeater,
            7,
            ImplSelf::Param(&self_param),
            slot_const(&repeater),
        )
        .unwrap()
        .map(|r| parse_quote!(#r))
        .collect();
    let (rel_ix, wrap2_ix) = (
        common::entry_id_at(&referrer, 1),
        common::entry_id_at(&referrer, 2),
    );
    let (key1, key2) = (repeater_key(7, rel_ix), repeater_key(7, wrap2_ix));
    let rel: ItemImpl = parse_quote! {
        impl<__S: ?::core::marker::Sized> crate::Repeater<(), #key1> for __S {
            type Type = Rel;
        }
    };
    let wrap2: ItemImpl = parse_quote! {
        impl<'a, T, __S: ?::core::marker::Sized + Tr<'a, T>>
            crate::Repeater<(&'a (), T, &'a T), #key2> for __S
        {
            type Type = Wrap2<<__S>::Out>;
        }
    };
    assert!(impls.contains(&rel));
    assert!(impls.contains(&wrap2));
}

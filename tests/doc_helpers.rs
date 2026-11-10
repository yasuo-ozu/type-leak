//! Examples moved out of the API docs: each test is the longer example of a public item.

use syn::*;
use template_quote::ToTokens;
use type_leak::*;

/// `repeater_impl()` returns the pieces of an impl, and the module in which to write it.
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
    let referrer = leaker.result().finish().unwrap();

    let param: Ident = parse_quote!(__S);
    let r = referrer
        .repeater_impl(0, ImplSelf::Param(&param), array_const)
        .unwrap();
    let (impl_generics, _, where_clause) = r.generics.split_for_impl();
    let (args, self_ty, ty) = (&r.args, &r.self_ty, &r.ty);
    let imp: ItemImpl = parse_quote! {
        impl #impl_generics crate::Repeater<#args, 7u64, 0usize> for #self_ty #where_clause {
            type Type = #ty;
        }
    };
    let expected: ItemImpl = parse_quote! {
        impl<__S: ?::core::marker::Sized> crate::Repeater<(), 7u64, 0usize> for __S {
            type Type = Outside;
        }
    };
    assert_eq!(imp, expected);

    // `Outside` is a relative path: it means something in `m::inner` only. Write the impl there.
    let path = referrer.module_path(r.module.unwrap()).unwrap();
    assert_eq!(path.len(), 1);
    assert_eq!(path[0], "inner");
}

/// `entry_args()` passes the generics by name at an identity site, and substitutes them at
/// another.
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
    let args = referrer
        .entry_args(0, &UseSite::identity(), array_const)
        .unwrap();
    assert_eq!(args, parse_quote!((&'a (), T, &'a T)));

    // `impl Tr<'static, u8> for ..` gives them arguments.
    let trait_args: Vec<GenericArgument> = vec![parse_quote!('static), parse_quote!(u8)];
    let site = UseSite::substituting(&tr.generics.params, &trait_args);
    let args = referrer.entry_args(0, &site, array_const).unwrap();
    assert_eq!(args, parse_quote!((&'static (), u8, &'static u8)));
}

/// `UseSite::with_self()` names the type for the `Self: 'a` that an entry needs.
#[test]
fn use_site_with_self_names_the_type_in_self_outlives() {
    let tr: ItemTrait = parse_quote!(
        trait Tr<'a> {
            fn f(&self, x: &'a Wrap<Self>);
        }
    );
    let mut leaker = Leaker::new();
    leaker.intern_trait(&tr, true);
    let referrer = leaker.result().finish().unwrap();
    // The entry `&'a Wrap<Self>` needs `Self: 'a`, which the arguments encode as `&'a Self`.
    let args = referrer
        .entry_args(0, &UseSite::identity(), array_const)
        .unwrap();
    assert_eq!(args, parse_quote!((&'a (), &'a Self)));
    // At a site where `Self` is not available, name the type instead.
    let site = UseSite::identity().with_self(parse_quote!(Host));
    let args = referrer.entry_args(0, &site, array_const).unwrap();
    assert_eq!(args, parse_quote!((&'a (), &'a Host)));
}

/// `needs_self()` is true for an entry that names `Self` or whose context does, false otherwise.
#[test]
fn needs_self_holds_for_self_in_the_type_or_in_its_context() {
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
        .position(|ty| *ty == parse_quote!(Wrap<Self>))
        .unwrap();
    let rel = referrer
        .iter()
        .position(|ty| *ty == parse_quote!(Rel<T>))
        .unwrap();
    assert!(referrer.needs_self(wrap) && referrer.needs_self(rel));

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
    assert!(!referrer.needs_self(0));
}

/// `bucket_self()` is a primitive type for an entry without `Self`, and `None` for one with it.
#[test]
fn bucket_self_is_a_primitive_unless_the_entry_needs_self() {
    let tr: ItemTrait = parse_quote! {
        trait Tr<T> { fn f(&self) -> Wrap<T>; fn g(&self) -> Wrap2<Self>; }
    };
    let mut leaker = Leaker::new();
    leaker.intern_trait(&tr, true);
    let referrer = leaker.result().finish().unwrap();
    let wrap = referrer
        .iter()
        .position(|ty| *ty == parse_quote!(Wrap<T>))
        .unwrap();
    let wrap2 = referrer
        .iter()
        .position(|ty| *ty == parse_quote!(Wrap2<Self>))
        .unwrap();
    let param: Ident = parse_quote!(__S);
    let target = ImplSelf::Bucket {
        id: 7,
        param: &param,
    };

    // `Wrap<T>` does not name `Self`: a primitive type, and no `__S`.
    let bucket = referrer.bucket_self(wrap, 7).unwrap();
    let r = referrer.repeater_impl(wrap, target, array_const).unwrap();
    assert_eq!(r.self_ty, bucket);
    assert_eq!(r.generics, parse_quote!(<T>));

    // `Wrap2<Self>` does: the implementors of the trait, as with `ImplSelf::Param`.
    assert_eq!(referrer.bucket_self(wrap2, 7), None);
    let r = referrer.repeater_impl(wrap2, target, array_const).unwrap();
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
    assert_eq!(
        referrer.context_generics(referrer.context(0)).params.len(),
        1
    );
    assert_eq!(
        referrer.context_generics(referrer.context(1)).params.len(),
        1
    );
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
        leaker.result().finish().unwrap().iter().collect::<Vec<_>>(),
        [&parse_quote!(Wrap<T>)]
    );
}

/// `RepeaterSlots` encodes const parameters of the impl and of the use site into the same slots.
#[test]
fn repeater_slots_encode_the_consts_of_the_impl_and_of_the_use_site() {
    let tr: ItemTrait = parse_quote! {
        trait Tr<const B: bool, const C: char = 'z'> { fn f(&self) -> Wrap<B, C>; }
    };
    let mut leaker = Leaker::new();
    leaker.intern_trait(&tr, true);
    let referrer = leaker.result().finish().unwrap();
    let slots = RepeaterSlots::new(parse_quote!(crate::Repeater));

    // The impl encodes the parameters themselves...
    let param: Ident = parse_quote!(__S);
    let r = referrer
        .repeater_impl(0, ImplSelf::Param(&param), |value, param| {
            slots.encode_const(value, &param.ty)
        })
        .unwrap();
    assert_eq!(
        r.args,
        parse_quote!((
            *const dyn crate::Repeater<(), 0, 0, B, Type = ()>,
            *const dyn crate::Repeater<(), 0, 0, false, C, Type = ()>
        ))
    );

    // ...and a copy in `impl Tr<true, 'x'> for ..` the arguments.
    let trait_args: Vec<GenericArgument> = vec![parse_quote!(true), parse_quote!('x')];
    let args = referrer
        .entry_args(
            0,
            &UseSite::substituting(&tr.generics.params, &trait_args),
            |value, param| slots.encode_const(value, &param.ty),
        )
        .unwrap();
    assert_eq!(
        args,
        parse_quote!((
            *const dyn crate::Repeater<(), 0, 0, true, Type = ()>,
            *const dyn crate::Repeater<(), 0, 0, false, 'x', Type = ()>
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
    let used: Vec<usize> = (0..3)
        .map(|ix| referrer.context_generics(referrer.context(ix)).params.len())
        .collect();
    assert_eq!(used, [0, 1, 2]);
}

/// A marker type serves an entry without `Self`; an entry that names it needs the parameter.
#[test]
fn impl_self_marker_serves_an_entry_without_self_only() {
    let tr: ItemTrait = parse_quote! {
        trait Tr<T> { fn f(&self) -> Wrap<T>; fn g(&self) -> Wrap2<Self>; }
    };
    let mut leaker = Leaker::new();
    leaker.intern_trait(&tr, true);
    let referrer = leaker.result().finish().unwrap();
    let wrap = referrer
        .iter()
        .position(|ty| *ty == parse_quote!(Wrap<T>))
        .unwrap();
    let wrap2 = referrer
        .iter()
        .position(|ty| *ty == parse_quote!(Wrap2<Self>))
        .unwrap();

    // `Wrap<T>` does not name `Self`: the impl may be for a marker type of your crate.
    let marker: Type = parse_quote!(crate::Marker);
    let r = referrer
        .repeater_impl(wrap, ImplSelf::Marker(&marker), array_const)
        .unwrap();
    assert_eq!(r.self_ty, marker);
    // `Wrap2<Self>` does: it is for the implementors of the trait, so a marker will not do.
    let err = referrer
        .repeater_impl(wrap2, ImplSelf::Marker(&marker), array_const)
        .unwrap_err();
    assert!(matches!(err, EncodeError::NeedsSelf(_)));
    let param: Ident = parse_quote!(__S);
    let r = referrer
        .repeater_impl(wrap2, ImplSelf::Param(&param), array_const)
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
    let in_struct = leaker.intern_struct(
        &parse_quote!(
            struct S<T> {
                a: Rel<Self>,
            }
        ),
        true,
    );
    let in_trait = leaker.intern_trait(
        &parse_quote!(
            trait Tr<T> {
                fn f(&self) -> Rel<Self>;
            }
        ),
        true,
    );
    let referrer = leaker.result().finish().unwrap();
    // In a struct, `Self` is the struct with its own generics as arguments.
    match referrer.context_self(in_struct) {
        Some(SelfRequirement::Type(ty)) => assert_eq!(*ty, parse_quote!(S<T>)),
        _ => unreachable!(),
    }
    // In a trait, `Self` is any type that implements the trait.
    match referrer.context_self(in_trait) {
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
        .position(|ty| *ty == parse_quote!(Wrap<T>))
        .unwrap();
    let rel = referrer
        .iter()
        .position(|ty| *ty == parse_quote!(Rel))
        .unwrap();
    assert_eq!(referrer.context(wrap), top_level);
    assert_ne!(referrer.context(rel), top_level);

    // It is written as an integer literal, so it travels with the Referrer in a macro call.
    let id: ContextId = parse_str("3usize").unwrap();
    assert_eq!(id.index(), 3);
    assert_eq!(id.to_token_stream().to_string(), "3");
}

/// `repeater_impl()` for every entry of a trait: `Rel` uses no generics, so any type implements it;
/// `Wrap2<Self::Out>` names `Self`, so the implementor is bound by the trait and replaces it.
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
    let referrer = leaker.result().finish().unwrap();

    let self_param: Ident = parse_quote!(__S);
    let repeater: Path = parse_quote!(crate::Repeater);
    let impls: Vec<ItemImpl> = (0..referrer.iter().count())
        .map(|ix| {
            let r = referrer
                .repeater_impl(ix, ImplSelf::Param(&self_param), array_const)
                .unwrap();
            let (impl_generics, _, where_clause) = r.generics.split_for_impl();
            let (args, self_ty, ty) = (&r.args, &r.self_ty, &r.ty);
            parse_quote! {
                impl #impl_generics #repeater<#args, 7u64, #ix> for #self_ty #where_clause {
                    type Type = #ty;
                }
            }
        })
        .collect();
    let rel: ItemImpl = parse_quote! {
        impl<__S: ?::core::marker::Sized> crate::Repeater<(), 7u64, 1usize> for __S {
            type Type = Rel;
        }
    };
    let wrap2: ItemImpl = parse_quote! {
        impl<'a, T, __S: ?::core::marker::Sized + Tr<'a, T>>
            crate::Repeater<(&'a (), T, &'a T), 7u64, 2usize> for __S
        {
            type Type = Wrap2<<__S>::Out>;
        }
    };
    assert!(impls.contains(&rel));
    assert!(impls.contains(&wrap2));
}

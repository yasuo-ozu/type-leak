//! Tests for `Referrer::visitor(context, ..)` and `Referrer::expand(context, ..)`: the context is
//! the scope of the copy, and the visited syntax is inside it. The visitor takes from the context
//! the generics in scope, what `Self` means and the module.

use syn::visit_mut::VisitMut;
use syn::*;
use template_quote::{quote, ToTokens};
use type_leak::{ContextId, Leaker, Referrer, SelfRequirement};

fn s(x: &impl ToTokens) -> String {
    quote!(#x).to_string()
}

fn id(n: usize) -> ContextId {
    parse_str(&n.to_string()).unwrap()
}

/// Rewrites each leaked type to `P<index>`.
fn replace(_: Type, ix: usize) -> Type {
    let ix = proc_macro2::Literal::usize_unsuffixed(ix);
    parse_quote!(P<#ix>)
}

/// The items of `m`, which stay as they are.
fn items_of(m: &ItemMod) -> Vec<Item> {
    m.content.as_ref().unwrap().1.clone()
}

/// The module of the entry `ix`.
fn module_of(referrer: &Referrer, ix: usize) -> ContextId {
    referrer
        .context_module(referrer.context(ix))
        .expect("the entry is in a module")
}

// ── modules: the visited syntax is inside the module of the context ──────────────────────────

fn two_rels() -> ItemMod {
    parse_quote!(
        mod m {
            struct D {
                a: Rel,
            }
            mod inner {
                use crate::x::Rel;
                struct C {
                    a: Rel,
                }
            }
        }
    )
}

/// The Referrer of `two_rels()` with the context of `m` and that of `inner`. `Rel` is two entries,
/// one in each module: `inner` imports its own.
fn two_rels_referrer() -> (Referrer, ContextId, ContextId) {
    let mut leaker = Leaker::new();
    let m = leaker.intern_mod(&two_rels());
    let referrer = leaker.result().finish().unwrap();
    assert_eq!(referrer.iter().count(), 2);
    let inner = (0..2)
        .map(|ix| module_of(&referrer, ix))
        .find(|module| *module != m)
        .unwrap();
    assert_eq!(
        referrer.module_path(inner),
        Some(&[parse_quote!(inner)][..])
    );
    (referrer, m, inner)
}

fn entry_in(referrer: &Referrer, module: ContextId) -> usize {
    (0..referrer.iter().count())
        .find(|&ix| module_of(referrer, ix) == module)
        .unwrap()
}

#[test]
fn a_mod_item_visited_alone_is_nested_in_the_module_of_the_context() {
    let (referrer, m, inner) = two_rels_referrer();
    let (in_m, in_inner) = (entry_in(&referrer, m), entry_in(&referrer, inner));
    // `inner` visited on its own, inside `m`, is `m::inner`: found by its name, so it gets its own
    // entry and not the one of `m` (it was left as written, as the equal entries of two modules
    // were ambiguous for a visitor that did not know its module).
    let mut item: ItemMod = items_of(&two_rels())
        .into_iter()
        .find_map(|item| match item {
            Item::Mod(item) => Some(item),
            _ => None,
        })
        .unwrap();
    referrer
        .clone()
        .visitor(m, replace)
        .visit_item_mod_mut(&mut item);
    let expected: ItemMod = {
        let ix = proc_macro2::Literal::usize_unsuffixed(in_inner);
        parse_quote!(
            mod inner {
                use crate::x::Rel;
                struct C {
                    a: P<#ix>,
                }
            }
        )
    };
    assert_eq!(s(&item), s(&expected));
    assert_ne!(in_m, in_inner);
}

#[test]
fn the_items_of_a_module_visited_one_by_one_are_in_it() {
    let (referrer, m, inner) = two_rels_referrer();
    let (in_m, in_inner) = (entry_in(&referrer, m), entry_in(&referrer, inner));
    // Each item of `m`, with the context of `m`: the struct is in `m`, the `mod` is `m::inner`.
    let mut items = items_of(&two_rels());
    for item in &mut items {
        referrer.clone().visitor(m, replace).visit_item_mut(item);
    }
    let (ix_m, ix_inner) = (
        proc_macro2::Literal::usize_unsuffixed(in_m),
        proc_macro2::Literal::usize_unsuffixed(in_inner),
    );
    let expected: Vec<Item> = vec![
        parse_quote!(
            struct D {
                a: P<#ix_m>,
            }
        ),
        parse_quote!(
            mod inner {
                use crate::x::Rel;
                struct C {
                    a: P<#ix_inner>,
                }
            }
        ),
    ];
    assert_eq!(
        items.iter().map(s).collect::<Vec<_>>(),
        expected.iter().map(s).collect::<Vec<_>>()
    );
    // The items of `inner` visited one by one are those of `inner` with its own context.
    let inner_items = match &items_of(&two_rels())[1] {
        Item::Mod(item) => items_of(item),
        _ => unreachable!(),
    };
    let mut copy = inner_items.clone();
    for item in &mut copy {
        referrer
            .clone()
            .visitor(inner, replace)
            .visit_item_mut(item);
    }
    assert_eq!(
        s(&copy[1]),
        s(&parse_quote!(
            struct C {
                a: P<#ix_inner>,
            }
        ) as &Item)
    );
    // And with the context of `m` they are `m::inner`'s items in `m`: they find no entry of
    // `inner`, but the one of `m` (as the entries of no module would be found too).
    let mut copy = inner_items;
    for item in &mut copy {
        referrer.clone().visitor(m, replace).visit_item_mut(item);
    }
    assert_eq!(
        s(&copy[1]),
        s(&parse_quote!(
            struct C {
                a: P<#ix_m>,
            }
        ) as &Item)
    );
}

#[test]
fn a_file_of_the_items_of_a_module_is_in_it() {
    let (referrer, m, inner) = two_rels_referrer();
    let (in_m, in_inner) = (entry_in(&referrer, m), entry_in(&referrer, inner));
    let (ix_m, ix_inner) = (
        proc_macro2::Literal::usize_unsuffixed(in_m),
        proc_macro2::Literal::usize_unsuffixed(in_inner),
    );
    let mut file = File {
        shebang: None,
        attrs: Vec::new(),
        items: items_of(&two_rels()),
    };
    referrer
        .clone()
        .visitor(m, replace)
        .visit_file_mut(&mut file);
    let expected: File = parse_quote!(
        struct D {
            a: P<#ix_m>,
        }
        mod inner {
            use crate::x::Rel;
            struct C {
                a: P<#ix_inner>,
            }
        }
    );
    assert_eq!(s(&file), s(&expected));
    // The names that the items of the file declare hide the entries of no module (as in a module),
    // and the entries of a module are kept.
    let mut leaker = Leaker::new();
    let (scope, ()) = leaker.enter_generics(&Generics::default(), |view| {
        view.visit_type(&parse_quote!(Rel));
    });
    let referrer = leaker.result().finish().unwrap();
    let mut file: File = parse_quote!(
        struct Rel;
        struct S(Rel);
    );
    referrer
        .clone()
        .visitor(scope, replace)
        .visit_file_mut(&mut file);
    assert_eq!(
        s(&file),
        s(&parse_quote!(
            struct Rel;
            struct S(Rel);
        ) as &File)
    );
    // Without the declaration, the entry is found.
    let mut file: File = parse_quote!(
        struct S(Rel);
    );
    referrer.visitor(scope, replace).visit_file_mut(&mut file);
    assert_eq!(
        s(&file),
        s(&parse_quote!(
            struct S(P<0>);
        ) as &File)
    );
}

#[test]
fn a_scope_in_a_module_is_in_that_module() {
    // The top-level context of a scope entered in a module records the module, so the visitor of
    // a copy in the scope looks up the entries of the module.
    let mut leaker = Leaker::new();
    let (module, scope) = leaker.enter_mod("m", [], |view| {
        view.enter_generics(&parse_quote!(<T>), |view| {
            view.visit_type(&parse_quote!(Wrap<T>));
            view.visit_type(&parse_quote!(Rel));
        })
        .0
    });
    let referrer = leaker.result().finish().unwrap();
    assert_ne!(module, scope);
    assert_eq!(referrer.context_module(scope), Some(module));
    assert_eq!(referrer.context_module(module), None);
    let mut sig: Signature = parse_quote!(fn f(x: Wrap<T>, y: Rel));
    referrer
        .clone()
        .visitor(scope, replace)
        .visit_signature_mut(&mut sig);
    let rewritten = s(&sig);
    assert_eq!(rewritten.matches("P <").count(), 2, "{}", rewritten);
    // An unrelated scope of the Leaker is in no module: the entries of `m` are not found there.
    let mut leaker = Leaker::new();
    let (module, ()) = leaker.enter_mod("m", [], |view| view.visit_type(&parse_quote!(Rel)));
    let (outside, ()) =
        leaker.enter_generics(&parse_quote!(<T>), |view| view.visit_type(&parse_quote!(T)));
    let referrer = leaker.result().finish().unwrap();
    assert_eq!(referrer.context_module(outside), None);
    let mut ty: Type = parse_quote!(Rel);
    ty = referrer.expand(outside, ty, replace);
    assert_eq!(s(&ty), "Rel");
    ty = referrer.expand(module, parse_quote!(Rel), replace);
    assert_eq!(s(&ty), "P < 0 >");
}

// ── `Self` comes from the context ────────────────────────────────────────────────────────────

#[test]
fn self_is_taken_from_the_context() {
    // The same leaked type in two traits with the same generics: `Rel<Self>` is an entry of each,
    // for the implementors of its own trait. The context says which.
    let tr1: ItemTrait = parse_quote!(
        trait Tr1<T> {
            fn f(&self, t: T) -> Rel<Self>;
        }
    );
    let tr2: ItemTrait = parse_quote!(
        trait Tr2<T> {
            fn f(&self, t: T) -> Rel<Self>;
        }
    );
    // And a struct, where `Self` is the type.
    let st: ItemStruct = parse_quote!(
        struct St<T>(Rel<Self>, T);
    );
    let mut leaker = Leaker::new();
    let c1 = leaker.intern_trait(&tr1, true);
    let c2 = leaker.intern_trait(&tr2, true);
    let c3 = leaker.intern_struct(&st, true);
    let referrer = leaker.result().finish().unwrap();
    assert_eq!(referrer.iter().count(), 3);
    // The scopes differ by what `Self` is, as an entry of each names it.
    assert!(c1 != c2 && c2 != c3 && c1 != c3);
    assert!(matches!(
        referrer.context_self(c1),
        Some(SelfRequirement::Bound(_))
    ));
    assert!(matches!(
        referrer.context_self(c3),
        Some(SelfRequirement::Type(_))
    ));
    let ix_of = |context: ContextId| {
        (0..3)
            .find(|&ix| referrer.context(ix) == context)
            .expect("the top-level context is the context of the entry")
    };
    let expected = |context: ContextId| {
        let ix = proc_macro2::Literal::usize_unsuffixed(ix_of(context));
        s(&parse_quote!(fn f(&self, t: T) -> P<#ix>) as &Signature)
    };
    for (context, name) in [(c1, "Tr1"), (c2, "Tr2")] {
        let mut sig: Signature = parse_quote!(fn f(&self, t: T) -> Rel<Self>);
        referrer
            .clone()
            .visitor(context, replace)
            .visit_signature_mut(&mut sig);
        assert_eq!(s(&sig), expected(context), "{}", name);
    }
    // The copy of the struct is rewritten with its own entry, whatever the order of the entries.
    let mut copy = st.clone();
    referrer
        .clone()
        .visitor(c3, replace)
        .visit_item_struct_mut(&mut copy);
    let ix = proc_macro2::Literal::usize_unsuffixed(ix_of(c3));
    let expected: ItemStruct = parse_quote!(struct St<T>(P<#ix>, T););
    assert_eq!(s(&copy), s(&expected));
    // `expand()` takes `Self` from the context too.
    for context in [c1, c2, c3] {
        let ty = referrer.expand(context, parse_quote!(Rel<Self>), replace);
        let ix = proc_macro2::Literal::usize_unsuffixed(ix_of(context));
        assert_eq!(s(&ty), s(&parse_quote!(P<#ix>) as &Type));
    }
    // A scope that declares no `Self` finds no entry that names it.
    let mut leaker = Leaker::new();
    leaker.intern_trait(&tr1, true);
    let (plain, ()) =
        leaker.enter_generics(&parse_quote!(<T>), |view| view.visit_type(&parse_quote!(T)));
    let referrer = leaker.result().finish().unwrap();
    let ty = referrer.expand(plain, parse_quote!(Rel<Self>), replace);
    assert_eq!(s(&ty), "Rel < Self >");
}

#[test]
fn a_nested_item_declares_its_own_self() {
    // `Self` of an impl nested in the copy is the impl's: the entry of the trait's `Self` is not
    // found there, but the one of the trait's generics is.
    let tr: ItemTrait = parse_quote!(
        trait Tr<T> {
            fn f(&self, t: T) -> (Rel<Self>, Wrap<T>);
        }
    );
    let mut leaker = Leaker::new();
    let context = leaker.intern_trait(&tr, true);
    let referrer = leaker.result().finish().unwrap();
    let mut item: ItemFn = parse_quote!(
        fn g<T>() {
            struct Local(Rel<Self>);
        }
    );
    referrer
        .clone()
        .visitor(context, replace)
        .visit_item_fn_mut(&mut item);
    assert!(s(&item).contains("Rel < Self >"), "{}", s(&item));
}

/// What `Self` of the entry that is the type `ty` (as `to_string()` has it) means.
fn self_of_entry(referrer: &Referrer, ty: &str) -> String {
    let ix = referrer
        .iter()
        .position(|entry| s(entry) == ty)
        .unwrap_or_else(|| panic!("no entry {}", ty));
    match referrer.context_self(referrer.context(ix)) {
        None => "-".to_string(),
        Some(SelfRequirement::Bound(path)) => format!("Self: {}", s(path)),
        Some(SelfRequirement::Type(ty)) => format!("Self = {}", s(ty)),
        Some(other) => format!("{:?}", other),
    }
}

#[test]
fn a_nested_function_or_type_alias_keeps_the_self_around() {
    // The Leaker has the `Self` of the scope in a function or a type alias nested in it: `Self` is
    // E0401 there, so the one around is its only meaning. Only a struct, an enum, a union, a trait
    // and an impl have a `Self` of their own, and the Referrer must agree.
    let mut leaker = Leaker::new();
    let (context, ()) = leaker.enter_impl(&Generics::default(), &parse_quote!(W), true, |v| {
        for item in [
            parse_quote!(
                fn f(x: Rel<Self>) {}
            ),
            parse_quote!(
                type A = Rel2<Self>;
            ),
            parse_quote!(
                struct V {
                    a: Rel3<Self>,
                }
            ),
        ] {
            let item: Item = item;
            v.visit_item(&item);
        }
    });
    let referrer = leaker.result().finish().unwrap();
    assert_eq!(referrer.iter().count(), 3);
    assert_eq!(self_of_entry(&referrer, "Rel < Self >"), "Self = W");
    assert_eq!(self_of_entry(&referrer, "Rel2 < Self >"), "Self = W");
    // The struct has its own `Self`, which the entry names.
    assert_eq!(self_of_entry(&referrer, "Rel3 < Self >"), "Self = V");
    let index = |ty: &str| referrer.iter().position(|entry| s(entry) == ty).unwrap();
    let (rel, rel2) = (index("Rel < Self >"), index("Rel2 < Self >"));
    let tokens = quote!(#referrer);
    let read_back: Referrer = syn::parse2(tokens).unwrap();
    assert_eq!(read_back, referrer);
    for referrer in [referrer.clone(), read_back] {
        // In a method of an impl for `W`, the nested function and type alias are found. The struct
        // nested in the impl has `Self` of its own: the same `Rel<Self>` there is not the entry.
        let mut copy: ItemImpl = parse_quote!(
            impl Foo for W {
                fn m() {
                    fn f(x: Rel<Self>) {}
                    type A = Rel2<Self>;
                    struct V2 {
                        a: Rel<Self>,
                    }
                    struct V {
                        a: Rel3<Self>,
                    }
                }
            }
        );
        referrer
            .clone()
            .visitor(context, replace)
            .visit_item_impl_mut(&mut copy);
        let (rel, rel2) = (
            proc_macro2::Literal::usize_unsuffixed(rel),
            proc_macro2::Literal::usize_unsuffixed(rel2),
        );
        let expected: ItemImpl = parse_quote!(
            impl Foo for W {
                fn m() {
                    fn f(x: P<#rel>) {}
                    type A = P<#rel2>;
                    struct V2 {
                        a: Rel<Self>,
                    }
                    struct V {
                        a: Rel3<Self>,
                    }
                }
            }
        );
        assert_eq!(s(&copy), s(&expected));
        // Not in a method: a function that is not one, which has them nested.
        let mut copy: ItemFn = parse_quote!(
            fn outer() {
                fn f(x: Rel<Self>) {}
                type A = Rel2<Self>;
                fn g() {
                    type B = Rel2<Self>;
                }
            }
        );
        referrer
            .clone()
            .visitor(context, replace)
            .visit_item_fn_mut(&mut copy);
        let expected: ItemFn = parse_quote!(
            fn outer() {
                fn f(x: P<#rel>) {}
                type A = P<#rel2>;
                fn g() {
                    type B = P<#rel2>;
                }
            }
        );
        assert_eq!(s(&copy), s(&expected));
    }
}

#[test]
fn a_nested_type_alias_in_a_trait_keeps_the_self_of_the_trait() {
    // The implementors of the trait are `Self` in a nested function or type alias too.
    let mut leaker = Leaker::new();
    let (context, ()) = leaker.enter_trait(&parse_quote!(Tr), &Generics::default(), true, |v| {
        for item in [
            parse_quote!(
                type A = Rel<Self>;
            ),
            parse_quote!(
                fn f(x: Rel2<Self>) {}
            ),
            // The trait alias has no `Self` of its own either.
            parse_quote!(
                trait Al = ::p::Bar<Rel3<Self>>;
            ),
        ] {
            let item: Item = item;
            v.visit_item(&item);
        }
    });
    let referrer = leaker.result().finish().unwrap();
    assert_eq!(referrer.iter().count(), 3);
    assert_eq!(self_of_entry(&referrer, "Rel < Self >"), "Self: Tr");
    let mut copy: ItemTrait = parse_quote!(
        trait Tr {
            fn m() {
                type A = Rel<Self>;
                fn f(x: Rel2<Self>) {}
                trait Al = ::p::Bar<Rel3<Self>>;
                trait Inner {
                    type B = Rel<Self>;
                }
            }
        }
    );
    let ix = |ty: &str| {
        proc_macro2::Literal::usize_unsuffixed(
            referrer.iter().position(|entry| s(entry) == ty).unwrap(),
        )
    };
    let (rel, rel2, rel3) = (ix("Rel < Self >"), ix("Rel2 < Self >"), ix("Rel3 < Self >"));
    referrer
        .visitor(context, replace)
        .visit_item_trait_mut(&mut copy);
    // The nested trait has its own `Self`.
    let expected: ItemTrait = parse_quote!(
        trait Tr {
            fn m() {
                type A = P<#rel>;
                fn f(x: P<#rel2>) {}
                trait Al = ::p::Bar<P<#rel3>>;
                trait Inner {
                    type B = Rel<Self>;
                }
            }
        }
    );
    assert_eq!(s(&copy), s(&expected));
}

// ── the generics come from the context ───────────────────────────────────────────────────────

#[test]
fn the_generics_are_taken_from_the_context() {
    // `Wrap<T>` is an entry of each trait, which declares `T` differently. The signature of the
    // copy does not declare `T`: the context does, so it is looked up with the right declaration.
    // (It took the exact lookup `into_visitor_with_root()` with the generics of the trait.)
    let tr1: ItemTrait = parse_quote!(
        trait Tr1<T: ::core::clone::Clone> {
            fn f(&self, x: Wrap<T>);
        }
    );
    let tr2: ItemTrait = parse_quote!(
        trait Tr2<T: ::core::marker::Copy> {
            fn f(&self, x: Wrap<T>);
        }
    );
    let mut leaker = Leaker::new();
    let c1 = leaker.intern_trait(&tr1, true);
    let c2 = leaker.intern_trait(&tr2, true);
    assert_ne!(c1, c2);
    let referrer = leaker.result().finish().unwrap();
    assert_eq!(referrer.iter().count(), 2);
    let entry = |context: ContextId| {
        let ix = (0..2).find(|&ix| referrer.context(ix) == context).unwrap();
        proc_macro2::Literal::usize_unsuffixed(ix)
    };
    for context in [c1, c2] {
        let mut sig: Signature = parse_quote!(fn f(&self, x: Wrap<T>));
        referrer
            .clone()
            .visitor(context, replace)
            .visit_signature_mut(&mut sig);
        let ix = entry(context);
        let expected: Signature = parse_quote!(fn f(&self, x: P<#ix>));
        assert_eq!(s(&sig), s(&expected));
    }
    // A generic that neither the context nor the visited syntax declares is not the Leaker's
    // (every undeclared generic was taken to be, when the visitor did not know the scope).
    let mut sig: Signature = parse_quote!(fn f(&self, x: Wrap<U>));
    referrer
        .clone()
        .visitor(c1, replace)
        .visit_signature_mut(&mut sig);
    assert_eq!(
        s(&sig),
        s(&parse_quote!(fn f(&self, x: Wrap<U>)) as &Signature)
    );
    // The copy may declare the generics again, as the entry has them: the item that declares
    // exactly those of the context is the scope itself.
    let mut item = tr2.clone();
    referrer
        .clone()
        .visitor(c2, replace)
        .visit_item_trait_mut(&mut item);
    let ix = entry(c2);
    let expected: ItemTrait = parse_quote!(
        trait Tr2<T: ::core::marker::Copy> { fn f(&self, x: P<#ix>); }
    );
    assert_eq!(s(&item), s(&expected));
    // With other bounds, they are declared by the copy and are another `T` (the entry is not
    // found), unless the outermost generics are the context's.
    let other: ItemImpl = parse_quote!(
        impl<T: ::core::marker::Copy + ::core::fmt::Debug> Tr2<T> for W {
            fn f(&self, x: Wrap<T>) {}
        }
    );
    let mut declared = other.clone();
    referrer
        .clone()
        .visitor(c2, replace)
        .visit_item_impl_mut(&mut declared);
    assert_eq!(s(&declared), s(&other));
    let mut leakers = other.clone();
    referrer
        .clone()
        .visitor(c2, replace)
        .with_outermost_generics_as_leakers()
        .visit_item_impl_mut(&mut leakers);
    assert!(
        s(&leakers).contains(&format!("P < {} >", entry(c2))),
        "{}",
        s(&leakers)
    );
}

#[test]
fn a_local_generic_is_declared_by_the_visited_syntax() {
    // The method declares `B`, which the entry uses: it is found where the signature declares it as
    // the entry has it, and not where `B` is undeclared or declared otherwise.
    let tr: ItemTrait = parse_quote!(
        trait Tr {
            fn f<const B: bool>(&self) -> Wrap<B>;
        }
    );
    let mut leaker = Leaker::new();
    let context = leaker.intern_trait(&tr, true);
    let referrer = leaker.result().finish().unwrap();
    let rewrite = |sig: Signature| {
        let mut sig = sig;
        referrer
            .clone()
            .visitor(context, replace)
            .visit_signature_mut(&mut sig);
        s(&sig)
    };
    assert_eq!(
        rewrite(parse_quote!(fn f<const B: bool>(&self) -> Wrap<B>)),
        s(&parse_quote!(fn f<const B: bool>(&self) -> P<0>) as &Signature)
    );
    for same in [
        parse_quote!(fn f(&self) -> Wrap<B>),
        parse_quote!(fn f<const B: u8>(&self) -> Wrap<B>),
    ] {
        let same: Signature = same;
        assert_eq!(rewrite(same.clone()), s(&same));
    }
}

#[test]
fn a_bound_on_self_alone_need_not_be_written_by_the_copy() {
    // The entry of `&'a Rel2<Self>` has the `where Self: Sized` of its method. A copy where
    // `Self` is a type parameter (`with_self()`) cannot write it, so the entry is found without
    // it: the only difference is a bound that names no generic. An entry with exactly the
    // context of the copy is preferred.
    let only: ItemTrait = parse_quote!(
        trait Tr {
            fn f<'a>(&self, x: &'a Rel2<Self>)
            where
                Self: ::core::marker::Sized;
        }
    );
    let both: ItemTrait = parse_quote!(
        trait Tr {
            fn f<'a>(&self, x: &'a Rel2<Self>)
            where
                Self: ::core::marker::Sized;
            fn g<'a>(&self, x: &'a Rel2<Self>);
        }
    );
    let copy = |tr: &ItemTrait, item: ItemFn| {
        let mut leaker = Leaker::new();
        let context = leaker.intern_trait(tr, true);
        let referrer = leaker.result().finish().unwrap();
        let predicate = |ix: usize| referrer.generics(ix).where_clause.is_some();
        let mut item = item;
        referrer
            .clone()
            .visitor(context, replace)
            .with_self(parse_quote!(S))
            .visit_item_fn_mut(&mut item);
        let text = s(&item);
        // The index of the entry which replaced the type, and whether it has the predicate.
        let ix = (0..referrer.iter().count()).find(|ix| text.contains(&format!("P < {} >", ix)));
        (ix.map(predicate), text)
    };
    let without: ItemFn = parse_quote!(
        fn free<'a, S: Tr + Sized>(x: &'a Rel2<Self>) {}
    );
    let with: ItemFn = parse_quote!(
        fn free<'a, S: Tr + Sized>(x: &'a Rel2<Self>)
        where
            Self: ::core::marker::Sized,
        {
        }
    );
    // Found with or without the predicate when it is the only entry.
    assert_eq!(copy(&only, without.clone()).0, Some(true));
    assert_eq!(copy(&only, with.clone()).0, Some(true));
    // Both entries: the exact one is the one found.
    assert_eq!(copy(&both, without.clone()).0, Some(false));
    assert_eq!(copy(&both, with.clone()).0, Some(true));
    // The entry is not found if the copy writes another bound on `Self`, or if the extra bound
    // names a generic.
    let other: ItemFn = parse_quote!(
        fn free<'a, S: Tr + Sized>(x: &'a Rel2<Self>)
        where
            Self: ::core::fmt::Debug,
        {
        }
    );
    assert_eq!(copy(&only, other).0, None);
    let named: ItemTrait = parse_quote!(
        trait Tr<T> {
            fn f<'a>(&self, x: &'a Rel2<Self>)
            where
                Self: ::core::convert::Into<T>;
        }
    );
    let item: ItemFn = parse_quote!(
        fn free<'a, S: Tr<T> + Sized>(x: &'a Rel2<Self>) {}
    );
    assert_eq!(copy(&named, item).0, None);
    let item: ItemFn = parse_quote!(
        fn free<'a, S: Tr<T> + Sized>(x: &'a Rel2<Self>)
        where
            Self: ::core::convert::Into<T>,
        {
        }
    );
    assert_eq!(copy(&named, item).0, Some(true));
}

// ── an unknown id ────────────────────────────────────────────────────────────────────────────

#[test]
#[should_panic(expected = "the context 7 is not one of the contexts of this Referrer")]
fn visitor_panics_for_an_unknown_context() {
    let mut leaker = Leaker::new();
    leaker.intern_struct(
        &parse_quote!(
            struct S(Rel);
        ),
        true,
    );
    let _ = leaker.result().finish().unwrap().visitor(id(7), replace);
}

#[test]
#[should_panic(expected = "the context 7 is not one of the contexts of this Referrer")]
fn expand_panics_for_an_unknown_context() {
    let mut leaker = Leaker::new();
    leaker.intern_struct(
        &parse_quote!(
            struct S(Rel);
        ),
        true,
    );
    let _ = leaker
        .result()
        .finish()
        .unwrap()
        .expand(id(7), parse_quote!(Rel), replace);
}

#[test]
fn the_id_of_a_dropped_context_is_unknown_too() {
    // The Referrer keeps the ids of the Leaker, so one that it does not have panics even if it is
    // below the largest.
    let mut leaker = Leaker::new();
    let top_level = leaker.intern_struct(
        &parse_quote!(
            struct S<T, U> {
                a: (Rel, T),
            }
        ),
        true,
    );
    let referrer = leaker.result().finish().unwrap();
    let missing: Vec<ContextId> = (0..top_level.index())
        .map(id)
        .filter(|candidate| !referrer.contexts().any(|known| known == *candidate))
        .collect();
    assert!(!missing.is_empty());
    for context in missing {
        let result = std::panic::catch_unwind(|| {
            referrer.clone().visitor(context, replace);
        });
        assert!(result.is_err(), "{}", context);
    }
}

#[test]
fn an_entry_context_is_a_scope_too() {
    // Any context of the Referrer is accepted: the context of an entry has the generics the entry
    // uses (and what `Self` is, if it names it).
    let tr: ItemTrait = parse_quote!(
        trait Tr<T> {
            fn f(&self, t: T) -> (Wrap<T>, Rel);
        }
    );
    let mut leaker = Leaker::new();
    leaker.intern_trait(&tr, true);
    let referrer = leaker.result().finish().unwrap();
    for ix in 0..referrer.iter().count() {
        let ty = referrer.iter().nth(ix).unwrap().clone();
        let rewritten = referrer.expand(referrer.context(ix), ty, replace);
        let ix = proc_macro2::Literal::usize_unsuffixed(ix);
        assert_eq!(s(&rewritten), s(&parse_quote!(P<#ix>) as &Type));
    }
}

// ── the shapes a downstream macro visits ─────────────────────────────────────────────────────

/// A trait with supertraits, associated types with bounds and a default, and a method, all naming
/// leaked types.
fn trait_with_bounds() -> ItemTrait {
    parse_quote! {
        pub trait Tr<T>: ::p::Sup<Rel> + ::p::Sup2<Rel2<T>> {
            type A: ::p::Bound<Rel3>;
            type B<'x>: ::p::Bound<Rel4<T>> = Rel5;
            fn f(&self, x: Rel6) -> Rel7<T>;
        }
    }
}

/// The docs of `Referrer::visitor()`: a macro rewrites the trait it leaked (`visit_item_trait_mut`
/// visits the generics, the supertraits and the items), or goes through the parts itself (as
/// newer-type does: the bounds of the supertraits, the associated types, the signatures of the
/// methods), and gets the same copy. Audit mutants `D3` (the bounds of an associated type are not
/// visited) and `D4` (the supertraits of a visited `ItemTrait` are not).
#[test]
fn a_trait_is_rewritten_whole_and_part_by_part() {
    let tr = trait_with_bounds();
    let mut leaker = Leaker::new();
    let context = leaker.intern_trait(&tr, true);
    let referrer = leaker.result().finish().unwrap();
    // The entries, by context and tokens: those without generics, then those with `<T>`.
    assert_eq!(
        referrer.iter().map(s).collect::<Vec<_>>(),
        [
            "Rel",
            "Rel3",
            "Rel5",
            "Rel6",
            "Rel2 < T >",
            "Rel4 < T >",
            "Rel7 < T >"
        ]
    );
    let expected: ItemTrait = parse_quote! {
        pub trait Tr<T>: ::p::Sup<P<0>> + ::p::Sup2<P<4>> {
            type A: ::p::Bound<P<1>>;
            type B<'x>: ::p::Bound<P<5>> = P<2>;
            fn f(&self, x: P<3>) -> P<6>;
        }
    };
    let mut whole = tr.clone();
    referrer
        .clone()
        .visitor(context, replace)
        .visit_item_trait_mut(&mut whole);
    assert_eq!(s(&whole), s(&expected));

    // Part by part: the supertraits, the associated types and the signatures, as a macro that
    // keeps the trait and replaces what it chooses to does.
    let mut parts = tr.clone();
    let mut visitor = referrer.clone().visitor(context, replace);
    for bound in parts.supertraits.iter_mut() {
        visitor.visit_type_param_bound_mut(bound);
    }
    for item in parts.items.iter_mut() {
        match item {
            TraitItem::Fn(item) => visitor.visit_signature_mut(&mut item.sig),
            TraitItem::Type(item) => visitor.visit_trait_item_type_mut(item),
            _ => unreachable!(),
        }
    }
    assert_eq!(s(&parts), s(&expected));
}

/// The docs of `Referrer::visitor()`: a signature visited by itself is a nested item: its generics
/// are its own, unless `with_outermost_generics_as_leakers()` says they are the context's. So the
/// method `f<T: Clone>` has a `T` of its own, and `Wrap<T>` is not the entry of the trait's `T`,
/// while with the flag it is. Audit mutant `D5` (`visit_signature_mut` treated as no item).
#[test]
fn a_signature_visited_by_itself_is_a_nested_item() {
    let tr: ItemTrait = parse_quote! {
        trait Tr<T> {
            fn f(&self) -> Wrap<T>;
            fn g<const B: bool>(&self) -> Wrap2<B>;
        }
    };
    let mut leaker = Leaker::new();
    let context = leaker.intern_trait(&tr, true);
    let referrer = leaker.result().finish().unwrap();
    for (sig, expected, as_leakers) in [
        // A `T` of the method with a bound is not the trait's `T`.
        (
            "fn f<T: ::core::clone::Clone>(&self) -> Wrap<T>",
            "fn f<T: ::core::clone::Clone>(&self) -> Wrap<T>",
            false,
        ),
        // Unless the visitor takes the generics of the outermost item as the context's.
        (
            "fn f<T: ::core::clone::Clone>(&self) -> Wrap<T>",
            "fn f<T: ::core::clone::Clone>(&self) -> P<0>",
            true,
        ),
        // The generics of a method that the trait does not have are declared.
        (
            "fn g<const B: bool>(&self) -> Wrap2<B>",
            "fn g<const B: bool>(&self) -> P<1>",
            false,
        ),
    ] {
        let mut sig: Signature = parse_str(sig).unwrap();
        let expected: Signature = parse_str(expected).unwrap();
        let mut visitor = referrer.clone().visitor(context, replace);
        if as_leakers {
            visitor = visitor.with_outermost_generics_as_leakers();
        }
        visitor.visit_signature_mut(&mut sig);
        assert_eq!(s(&sig), s(&expected));
    }
}

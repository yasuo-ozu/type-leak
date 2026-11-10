//! The scopes of the `Leaker` (`enter_*()`) and the `View` they give to their closure: what each
//! scope declares, nested scopes, `Self`, and what a scope which fails keeps. One section pins
//! that the scopes of the items are a layer around `enter_generics()`, and the last one that the
//! `View` forwards exactly the `Visit` methods that the analysis overrides.

mod common;

use common::{enter_function, messages, s, to_syn_error, visit_function, AcceptCause};
use syn::visit::Visit;
use syn::*;
use template_quote::ToTokens;
use type_leak::{
    Allow, Cause, ContextId, ErrorItem, Leaker, LeakerResult, Reason, Referrer, Rules,
    SelfRequirement, View,
};

/// The result of the calls `f` makes on a new Leaker.
fn recorded(f: impl FnOnce(&mut Leaker)) -> LeakerResult {
    let mut leaker = Leaker::new();
    f(&mut leaker);
    leaker.result()
}

/// Checks that `errors` has the conflicts of paths, not only those of `Self` without a meaning.
fn assert_relative_path(errors: &[ErrorItem]) {
    assert!(!errors.is_empty());
    assert!(
        errors
            .iter()
            .all(|item| item.reason != Reason::SelfWithoutMeaning),
        "{:?}",
        errors
    );
}

/// Checks that `errors` has only the mentions of `Self` where it has no meaning.
fn assert_self_without_meaning(errors: &[ErrorItem]) {
    assert!(!errors.is_empty());
    assert!(
        errors
            .iter()
            .all(|item| item.reason == Reason::SelfWithoutMeaning),
        "{:?}",
        errors
    );
}

fn id(n: usize) -> ContextId {
    parse_str(&n.to_string()).unwrap()
}

fn ident(name: &str) -> Ident {
    Ident::new(name, proc_macro2::Span::call_site())
}

/// The leaked types with their generics, in the order of the Referrer.
fn entries(referrer: &Referrer) -> Vec<(String, String)> {
    referrer
        .iter()
        .map(|(_, ty, context)| (ty, &context.generics))
        .map(|(ty, generics)| (s(ty), s(generics)))
        .collect()
}

fn e(ty: &str, generics: &str) -> (String, String) {
    (ty.to_string(), generics.to_string())
}

fn self_of(referrer: &Referrer, ix: usize) -> String {
    match common::entry(referrer, ix).self_ty.as_ref() {
        None => "-".to_string(),
        Some(SelfRequirement::Bound(path)) => format!("Self: {}", s(path)),
        Some(SelfRequirement::Type(ty)) => format!("Self = {}", s(ty)),
        Some(other) => format!("{:?}", other),
    }
}

/// The path of the module of the `ix`-th entry.
fn module_path_of(result: &LeakerResult, ix: usize) -> Option<Vec<String>> {
    Some(
        common::entry_module_path(result, ix)?
            .iter()
            .map(|i| i.to_string())
            .collect(),
    )
}

// ── what each scope declares ─────────────────────────────────────────────────────────────────

#[test]
fn generics_are_those_of_the_leaker() {
    let generics: Generics = parse_quote!(<T, U: ::core::marker::Copy>);
    let mut leaker = Leaker::new();
    let (context, value) = leaker.enter_generics(&generics, |view| {
        view.visit_type(&parse_quote!(Rel<T>));
        view.visit_type(&parse_quote!(Rel2<U>));
        view.visit_type(&parse_quote!(Rel3));
        view.visit_type(&parse_quote!(Rel4<T, U>));
        42
    });
    // The value of the closure is returned.
    assert_eq!(value, 42);
    let referrer = leaker.result().finish().unwrap();
    assert_eq!(
        entries(&referrer),
        [
            e("Rel < T >", "< T >"),
            e("Rel2 < U >", "< U : :: core :: marker :: Copy >"),
            e("Rel3", ""),
            e("Rel4 < T , U >", "< T , U : :: core :: marker :: Copy >"),
        ]
    );
    // The top-level context is that of the generics: the entry that uses all of them has it.
    let all = common::types(&referrer)
        .position(|ty| s(ty) == "Rel4 < T , U >")
        .unwrap();
    assert_eq!(common::entry_id(&referrer, all), context);
    assert!(common::context_ids(&referrer)
        .into_iter()
        .any(|c| c == context));
    // `Self` is declared by none of them.
    assert_eq!(self_of(&referrer, 0), "-");
}

#[test]
fn a_struct_an_enum_and_a_union_declare_self_as_the_item() {
    let generics: Generics = parse_quote!(<'a, T>);
    for (kind, name) in [("struct", "S"), ("enum", "E"), ("union", "U")] {
        let mut leaker = Leaker::new();
        let ty: Type = parse_quote!(Rel<Self, T>);
        let ident = ident(name);
        let (context, ()) = leaker.enter_type(&ident, &generics, |v| v.visit_type(&ty));
        let referrer = leaker.result().finish().unwrap();
        assert_eq!(
            entries(&referrer),
            [e("Rel < Self , T >", "< 'a , T >")],
            "{}",
            kind
        );
        assert_eq!(
            self_of(&referrer, 0),
            format!("Self = {} < 'a , T >", name),
            "{}",
            kind
        );
        // The top-level context has the generics of the item and, as an entry names `Self`, what
        // `Self` is: it is the context of the entry.
        assert_eq!(context, common::entry_id(&referrer, 0));
        assert_eq!(s(&common::entry(&referrer, 0).generics), "< 'a , T >");
        assert!(common::entry(&referrer, 0).self_ty.is_some());
    }
}

#[test]
fn an_impl_declares_self_as_the_self_type() {
    let generics: Generics = parse_quote!(<T>);
    let mut leaker = Leaker::new();
    let self_ty: Type = parse_quote!(::p::Wrap<T>);
    let (context, ()) = leaker.enter_impl(&generics, &self_ty, None, |view| {
        view.visit_type(&parse_quote!(Rel<Self>));
        view.visit_type(&parse_quote!(Rel2<T>));
    });
    let referrer = leaker.result().finish().unwrap();
    assert_eq!(
        entries(&referrer),
        [e("Rel < Self >", "< T >"), e("Rel2 < T >", "< T >")]
    );
    assert_eq!(self_of(&referrer, 0), "Self = :: p :: Wrap < T >");
    assert_eq!(self_of(&referrer, 1), "-");
    // The top-level context has the generics of the impl and what `Self` is: that of `Rel<Self>`.
    assert_eq!(context, common::entry_id(&referrer, 0));
}

#[test]
fn a_trait_declares_self_as_the_implementor() {
    let generics: Generics = parse_quote!(<T, const N: usize>);
    let mut leaker = Leaker::new();
    let (_, ()) = leaker.enter_trait(&ident("Tr"), &generics, |view| {
        view.visit_type(&parse_quote!(Rel<Self::Item, T>));
    });
    let referrer = leaker.result().finish().unwrap();
    assert_eq!(
        entries(&referrer),
        [e("Rel < Self :: Item , T >", "< T , const N : usize >")]
    );
    // The same as `intern_trait()` makes of the trait.
    assert_eq!(self_of(&referrer, 0), "Self: Tr < T , N >");
    let item: ItemTrait = parse_quote!(
        trait Tr<T, const N: usize> {
            fn f(&self) -> Rel<Self::Item, T>;
        }
    );
    let mut other = Leaker::new();
    other.intern_trait(&item);
    let other = other.result().finish().unwrap();
    assert_eq!(entries(&other), entries(&referrer));
    assert_eq!(self_of(&other, 0), self_of(&referrer, 0));
}

#[test]
fn a_module_declares_its_names_and_the_module() {
    let mut leaker = Leaker::new();
    let item: ItemStruct = parse_quote!(
        struct A {
            a: Rel,
            b: S<Rel2>,
            c: self::S,
            d: Other,
        }
    );
    let (module, ()) = leaker.enter_mod("my_module", [ident("S")], |view| {
        view.visit_item(&Item::Struct(item.clone()))
    });
    let result = leaker.result();
    let referrer = result.clone().finish().unwrap();
    // `S` is declared in the module, so it is not leaked, and nor is `self::S`.
    assert_eq!(
        entries(&referrer),
        [e("Other", ""), e("Rel", ""), e("Rel2", "")]
    );
    for ix in 0..3 {
        assert_eq!(
            common::in_module(common::entry(&referrer, ix)),
            Some(module)
        );
        assert_eq!(
            common::entry_module_hint(&result, ix).as_deref(),
            Some("my_module")
        );
        assert_eq!(common::entry_module_path(&result, ix), None);
    }
    assert!(common::context_ids(&referrer)
        .into_iter()
        .any(|c| c == module));
    // The hint is for debugging: any `Display` does.
    let mut leaker = Leaker::new();
    leaker.enter_mod(7, [], |view| view.visit_type(&parse_quote!(Rel)));
    let result = leaker.result();
    result.clone().finish().unwrap();
    assert_eq!(common::entry_module_hint(&result, 0).as_deref(), Some("7"));
}

#[test]
fn the_whole_item_methods_are_scopes_of_the_item() {
    // `intern_*()` analyze the item with its own generics as those of the Leaker, which is what
    // the scope of the item declares: only the fragments are analyzed there.
    let item: ItemStruct = parse_quote!(
        struct S<T> {
            a: Rel<T>,
            b: Wrap<Self>,
        }
    );
    let mut whole = Leaker::new();
    let top = whole.intern_struct(&item);
    let whole = whole.result().finish().unwrap();

    let mut scope = Leaker::new();
    let fields = item.fields.clone();
    let (top_scope, ()) = scope.enter_type(&item.ident, &item.generics, |view| {
        for field in &fields {
            view.visit_type(&field.ty);
        }
    });
    let scope = scope.result().finish().unwrap();
    assert_eq!(entries(&scope), entries(&whole));
    assert_eq!(top_scope, top);
    for ix in 0..2 {
        assert_eq!(self_of(&scope, ix), self_of(&whole, ix));
    }
}

// ── the entry points of the view ─────────────────────────────────────────────────────────────

#[test]
fn the_entry_points_of_the_view() {
    let generics: Generics = parse_quote!(<T: ::p::Bound<Rel>>);
    let mut leaker = Leaker::new();
    leaker.enter_generics(&generics, |view| {
        // A type.
        view.visit_type(&parse_quote!(Rel2<T>));
        // The arguments of a path, whose name is resolved elsewhere.
        view.visit_path_arguments(&PathArguments::AngleBracketed(parse_quote!(
            <Rel3, (T, Rel4)>
        )));
        // A path at the root (`Visit::visit_path()`): it is not checked nor leaked, its
        // arguments are.
        view.visit_path(&parse_quote!(::p::Tr::<Rel5>));
        // A where clause and a bound.
        view.visit_where_clause(&parse_quote!(where T: ::p::Tr<Rel6>));
        view.visit_type_param_bound(&parse_quote!(::p::Tr<Rel7>));
        // An item, whose generics are local ones.
        view.visit_item(&parse_quote!(
            fn f<V>(x: Rel8<V>) -> Rel9 {
                loop {}
            }
        ));
        // A signature (`Visit::visit_signature()`), whose generics are local ones as well.
        view.visit_signature(&parse_quote!(fn g<W>(x: Rel10<W>) -> Rel11));
    });
    let referrer = leaker.result().finish().unwrap();
    assert_eq!(
        entries(&referrer),
        [
            e("Rel2 < T >", "< T : :: p :: Bound < Rel > >"),
            e("Rel11", ""),
            e("Rel3", ""),
            e("Rel4", ""),
            e("Rel5", ""),
            e("Rel6", ""),
            e("Rel7", ""),
            e("Rel9", ""),
            e("Rel8 < V >", "< V >"),
            e("Rel10 < W >", "< W >"),
        ]
    );
}

#[test]
fn an_item_in_a_scope_has_local_generics() {
    // The generics of an item visited in a scope are local generics, as those of the items of a
    // module are: `intern_*()` makes the generics of the item those of the Leaker.
    let item: ItemStruct = parse_quote!(
        struct S<T> {
            a: Rel<T>,
        }
    );
    let mut leaker = Leaker::new();
    leaker.enter_generics(&Generics::default(), |view| {
        view.visit_item(&Item::Struct(item.clone()))
    });
    let referrer = leaker.result().finish().unwrap();
    assert_eq!(entries(&referrer), [e("Rel < T >", "< T >")]);
    // The top-level context has no generics here, whereas that of `intern_struct()` has them.
    let mut leaker = Leaker::new();
    let top = leaker.intern_struct(&item);
    let referrer = leaker.result().finish().unwrap();
    assert_eq!(common::entry_id(&referrer, 0), top);
    assert_eq!(s(&common::entry(&referrer, 0).generics), "< T >");
}

// ── `Self` ───────────────────────────────────────────────────────────────────────────────────

#[test]
fn self_has_no_meaning_in_a_module_or_generics() {
    let ty: Type = parse_quote!(Rel<Self>);
    let mut leaker = Leaker::new();
    leaker.enter_generics(&parse_quote!(<T>), |view| view.visit_type(&ty));
    leaker.enter_mod("m", [], |view| view.visit_type(&ty));
    // Not even where it would not be leaked, or in a bound.
    for ty in [
        parse_quote!(Self),
        parse_quote!(Self::Item),
        parse_quote!(&'static Self),
    ] {
        let ty: Type = ty;
        leaker.enter_generics(&Generics::default(), |view| view.visit_type(&ty));
    }
    leaker.enter_generics(&Generics::default(), |view| {
        view.visit_where_clause(&parse_quote!(where Self: ::core::marker::Sized))
    });
    let result = leaker.result();
    // Each call reports its error, in the order of the calls.
    let err = result.errors();
    assert_eq!(messages(err), ["`Self` has no meaning here"; 6].join("\n"));
    assert_eq!(
        selves_of(err),
        ["Self", "Self", "Self", "Self :: Item", "Self", "Self"]
    );
    assert!(
        err[0]
            .notes
            .first()
            .map(|(_, n)| n.as_str())
            .unwrap()
            .contains("`Self` has no meaning here"),
        "{:?}",
        err[0].notes.first().map(|(_, n)| n.as_str())
    );
    // No entry is left: the scopes recovered, and had none.
    assert!(result.into_referrer().is_empty());

    // The same in a module analyzed as a whole, and in a function of it (not in the items which
    // declare it).
    let m: ItemMod = parse_quote!(
        mod m {
            pub type A = Rel<Self>;
        }
    );
    let result = recorded(|leaker| {
        leaker.intern_mod(&m);
    });
    assert_eq!(messages(result.errors()), "`Self` has no meaning here");
    let m: ItemMod = parse_quote!(
        mod m {
            pub fn f() -> Rel<Self> {
                loop {}
            }
        }
    );
    let result = recorded(|leaker| {
        leaker.intern_mod(&m);
    });
    assert!(result.finish().is_err());
    let m: ItemMod = parse_quote!(
        mod m {
            pub struct S {
                a: Rel<Self>,
            }
            pub trait Tr {
                fn f(&self) -> Rel2<Self>;
            }
            pub struct W;
            impl W {
                pub fn f(&self) -> Rel3<Self> {
                    loop {}
                }
            }
        }
    );
    let mut leaker = Leaker::new();
    leaker.intern_mod(&m);
    assert_eq!(leaker.result().finish().unwrap().iter().count(), 3);
}

#[test]
fn self_is_kept_by_the_generics_in_a_scope_which_declares_it() {
    let impl_generics: Generics = parse_quote!(<T>);
    let method_generics: Generics = parse_quote!(<'a, U>);
    let mut leaker = Leaker::new();
    leaker.enter_impl(&impl_generics, &parse_quote!(::p::W<T>), None, |view| {
        view.enter_generics(&method_generics, |view| {
            view.visit_type(&parse_quote!(Rel<'a, Self, U>))
        });
    });
    let referrer = leaker.result().finish().unwrap();
    assert_eq!(
        entries(&referrer),
        [e("Rel < 'a , Self , U >", "< 'a , T , U >")]
    );
    assert_eq!(self_of(&referrer, 0), "Self = :: p :: W < T >");

    // A module in a scope is not in that `Self`.
    let mut leaker = Leaker::new();
    leaker.enter_impl(&impl_generics, &parse_quote!(::p::W<T>), None, |view| {
        view.enter_mod("m", [], |view| view.visit_type(&parse_quote!(Rel<Self>)));
    });
    // The error of the scope in the module is recorded once, for the scope around it too.
    let result = leaker.result();
    assert_eq!(messages(result.errors()), "`Self` has no meaning here");
}

#[test]
fn self_may_be_interned_in_a_scope() {
    let generics: Generics = parse_quote!(<T>);
    let ty: Type = parse_quote!(Rel<Self>);
    // (`enter_type()` is the same for a struct, an enum and a union.)
    let result = recorded(|leaker| {
        leaker.enter_type(&ident("S"), &generics, |view| view.visit_type(&ty));
    });
    assert!(result.errors().is_empty());
    let result = recorded(|leaker| {
        leaker.enter_impl(&generics, &parse_quote!(::p::W<T>), None, |view| {
            view.visit_type(&ty)
        });
    });
    assert!(result.errors().is_empty());
    // A type which can be decomposed is, and the others are not affected.
    let mut leaker = Leaker::new();
    leaker.enter_type(&ident("S"), &generics, |view| {
        view.visit_type(&parse_quote!((Rel, Self)));
        view.visit_type(&parse_quote!(Rel2<T>));
    });
    assert_eq!(
        entries(&leaker.result().finish().unwrap()),
        [e("Rel", ""), e("Rel2 < T >", "< T >")]
    );
    // The bounds of the generics a type uses may mention `Self` too.
    let bounded: Generics = parse_quote!(<T: ::core::cmp::PartialEq<Self>>);
    let result = recorded(|leaker| {
        leaker.enter_type(&ident("S"), &bounded, |view| {
            view.visit_type(&parse_quote!(Wrap<T>))
        });
    });
    assert!(result.errors().is_empty(), "{:?}", result.errors());
}

#[test]
fn self_is_accepted_in_the_whole_items() {
    // `Self` is no error in a whole item: the type that names it is an entry.
    let leaked = |f: &dyn Fn(&mut Leaker)| {
        let mut leaker = Leaker::new();
        f(&mut leaker);
        let result = leaker.result();
        assert!(result.errors().is_empty(), "{:?}", result.errors());
        entries(&result.finish().unwrap())
    };
    let item: ItemImpl = parse_quote! {
        impl<T> ::p::Tr for ::p::W<T> {
            fn f(&self) -> Rel<Self> { loop {} }
        }
    };
    assert_eq!(
        leaked(&|l| {
            l.intern_impl(&item);
        }),
        [e("Rel < Self >", "< T >"), e(":: p :: W < T >", "< T >")],
        "impl"
    );
    let item: ItemStruct = parse_quote!(
        struct S<T> {
            a: Rel<Self>,
            b: T,
        }
    );
    assert_eq!(
        leaked(&|l| {
            l.intern_struct(&item);
        }),
        [e("Rel < Self >", "< T >")],
        "struct"
    );
    let item: ItemEnum = parse_quote!(
        enum E<T> {
            A(Rel<Self>),
            B(T),
        }
    );
    assert_eq!(
        leaked(&|l| {
            l.intern_enum(&item);
        }),
        [e("Rel < Self >", "< T >")],
        "enum"
    );
    let item: ItemUnion = parse_quote!(union U { a: Rel<Self> });
    assert_eq!(
        leaked(&|l| {
            l.intern_union(&item);
        }),
        [e("Rel < Self >", "")],
        "union"
    );
    // `Self` in the default of a const parameter (an expression) is no error either.
    let item: ItemStruct = parse_quote!(
        pub struct S<const N: usize = { Self::K }>(u8);
    );
    assert_eq!(
        leaked(&|l| {
            l.intern_struct(&item);
        }),
        [e("u8", "")],
        "const default"
    );
}

// ── nested scopes ────────────────────────────────────────────────────────────────────────────

#[test]
fn a_module_in_a_module_in_a_scope() {
    // `enter_mod` > `enter_impl` > `visit_*`: the types are in the module, with the generics of
    // the impl (local ones) and what `Self` is.
    let generics: Generics = parse_quote!(<T>);
    let self_ty: Type = parse_quote!(::p::W<T>);
    let mut leaker = Leaker::new();
    let (outer, inner) = leaker.enter_mod("outer", [], |view| {
        view.enter_impl(&generics, &self_ty, None, |view| {
            view.visit_type(&parse_quote!(Rel<T>));
            view.visit_type(&parse_quote!(Rel2<Self>));
            view.visit_type(&parse_quote!(::q::Abs));
        });
        // A module in the module.
        view.enter_mod("inner", [], |view| view.visit_type(&parse_quote!(Rel3)))
            .0
    });
    let result = leaker.result();
    let referrer = result.clone().finish().unwrap();
    // The type of an external crate is an entry that does not depend on a module.
    let found: Vec<(String, String, String, Option<ContextId>)> = (0..referrer.iter().count())
        .map(|ix| {
            (
                s(common::types(&referrer).nth(ix).unwrap()),
                s(&common::entry(&referrer, ix).generics),
                self_of(&referrer, ix),
                common::in_module(common::entry(&referrer, ix)),
            )
        })
        .collect();
    let mut expected = vec![
        ("Rel < T >", "< T >", "-", Some(outer)),
        (
            "Rel2 < Self >",
            "< T >",
            "Self = :: p :: W < T >",
            Some(outer),
        ),
        ("Rel3", "", "-", Some(inner)),
        (":: q :: Abs", "", "-", None),
    ];
    let mut found = found;
    found.sort_by(|a, b| a.0.cmp(&b.0));
    expected.sort_by(|a, b| a.0.cmp(b.0));
    assert_eq!(found.len(), expected.len());
    for (found, expected) in found.iter().zip(&expected) {
        assert_eq!(found.0, expected.0);
        assert_eq!(found.1, expected.1, "{}", found.0);
        assert_eq!(found.2, expected.2, "{}", found.0);
        assert_eq!(found.3, expected.3, "{}", found.0);
    }
    // `inner` is entered in `outer`, which is the module of the call.
    let at = |text: &str| {
        common::types(&referrer)
            .position(|ty| s(ty) == text)
            .unwrap()
    };
    let (in_outer, in_inner) = (at("Rel < T >"), at("Rel3"));
    assert_eq!(
        common::entry_module_hint(&result, in_outer).as_deref(),
        Some("outer")
    );
    assert_eq!(
        common::entry_module_hint(&result, in_inner).as_deref(),
        Some("inner")
    );
    assert_eq!(common::entry_module_parent(&result, in_inner), Some(outer));
    assert_ne!(outer, inner);
}

#[test]
fn nested_modules_are_numbered_and_kept() {
    let mut leaker = Leaker::new();
    let (outer, (inner, deeper)) = leaker.enter_mod("outer", [], |view| {
        view.visit_type(&parse_quote!(RelO));
        view.enter_mod("inner", [], |view| {
            view.visit_type(&parse_quote!(RelI));
            view.enter_mod("deeper", [], |view| view.visit_type(&parse_quote!(Rel)))
                .0
        })
    });
    let result = leaker.result();
    let referrer = result.clone().finish().unwrap();
    // The module of each entry is the one it is written in, and each is a module of the Referrer.
    let at = |text: &str| {
        common::types(&referrer)
            .position(|ty| s(ty) == text)
            .unwrap()
    };
    for (text, module, hint) in [
        ("RelO", outer, "outer"),
        ("RelI", inner, "inner"),
        ("Rel", deeper, "deeper"),
    ] {
        let ix = at(text);
        assert_eq!(
            common::in_module(common::entry(&referrer, ix)),
            Some(module)
        );
        assert_eq!(
            common::entry_module_hint(&result, ix).as_deref(),
            Some(hint)
        );
        // A module entered by a scope has a hint, and no path.
        assert_eq!(common::entry_module_path(&result, ix), None);
    }
    // They are numbered in the order they are entered, from the outermost.
    assert_eq!((outer, inner, deeper), (id(0), id(1), id(2)));
}

#[test]
fn the_defaults_of_a_nested_scope_are_analyzed_with_the_parameters_before() {
    // As the defaults of the generics of the Leaker are: with the parameters before them as the
    // generics, so they are rewritten on their own where the Referrer omits the argument.
    let mut leaker = Leaker::new();
    let (_, (_, ())) = leaker.enter_generics(&parse_quote!(<T>), |view| {
        view.enter_generics(&parse_quote!(<A, B = Rel<A>>), |_| ())
    });
    let referrer = leaker.result().finish().unwrap();
    assert_eq!(entries(&referrer), [e("Rel < A >", "< A >")]);
    // The default is not among the generics, which are in the canonical form: the entries that use
    // all the generics of a scope have them so.
    let mut leaker = Leaker::new();
    leaker.enter_generics(&parse_quote!(<T>), |view| {
        view.enter_generics(&parse_quote!(<A, B = Rel<A>>), |view| {
            view.visit_type(&parse_quote!(Both<A, B>))
        });
        view.visit_type(&parse_quote!(One<T>));
    });
    assert_eq!(
        entries(&leaker.result().finish().unwrap()),
        [
            e("Both < A , B >", "< A , B >"),
            e("Rel < A >", "< A >"),
            e("One < T >", "< T >")
        ]
    );
    // The same for the generics of the Leaker.
    let mut other = Leaker::new();
    other.enter_generics(&parse_quote!(<A, B = Rel<A>>), |_| ());
    assert_eq!(
        entries(&other.result().finish().unwrap()),
        entries(&referrer)
    );
    // A default which is not internable is an error of the scope which declares it, which the
    // scope around has too (once).
    let mut leaker = Leaker::new();
    leaker.enter_generics(&Generics::default(), |view| {
        view.enter_generics(&parse_quote!(<A, B = Bad<'_, A>>), |_| ());
    });
    assert_eq!(
        messages(leaker.result().errors()),
        "use absolute path: 'Bad<'_, A>'"
    );
}

// ── recovery ─────────────────────────────────────────────────────────────────────────────────

/// What `enter_generics()` of the two scopes returns, as the ids of the Referrer, and the result.
struct Run {
    first: ContextId,
    last: ContextId,
    result: LeakerResult,
}

/// Enters a scope with `<A>`, which has a scope with `<B>` that succeeds, one with `<U>` which
/// does not if `fail`, and one with `<C>` that succeeds.
fn run(fail: bool) -> Run {
    let outer: Generics = parse_quote!(<T>);
    let mut leaker = Leaker::new();
    let (_, (first, last)) = leaker.enter_generics(&outer, |view| {
        view.visit_type(&parse_quote!(Rel<T>));
        let (first, ()) = view.enter_generics(&parse_quote!(<A>), |view| {
            view.visit_type(&parse_quote!(RelA<A>));
        });
        if fail {
            // A module, which takes an id, and two errors.
            view.enter_generics(&parse_quote!(<U>), |view| {
                view.visit_type(&parse_quote!(RelU<U>));
                view.enter_mod("lost", [], |view| view.visit_type(&parse_quote!(RelM)));
                view.visit_type(&parse_quote!(Bad1<'_>));
                view.visit_type(&parse_quote!(Bad2<'_, U>));
            });
        }
        let (last, ()) = view.enter_generics(&parse_quote!(<C>), |view| {
            view.visit_type(&parse_quote!(RelC<C>));
            view.enter_mod("kept", [], |view| view.visit_type(&parse_quote!(RelK)));
        });
        view.visit_type(&parse_quote!(Rel2));
        (first, last)
    });
    Run {
        first,
        last,
        result: leaker.result(),
    }
}

#[test]
fn a_failing_scope_keeps_its_result_and_the_outer_one_goes_on() {
    let ok = run(false);
    let failed = run(true);
    assert!(ok.result.errors().is_empty());
    let (ok_first, ok_last) = (ok.first, ok.last);
    // The errors of the scope are all recorded, once, with the spans (here, those of the paths),
    // in the list of the call around it.
    let message = messages(failed.result.errors());
    assert_eq!(message.lines().count(), 2, "{}", message);
    assert!(message.contains("Bad1"), "{}", message);
    assert!(message.contains("Bad2"), "{}", message);
    // Accepting the causes afterwards drops the errors, and does not change the Referrer.
    let mut accepted = failed.result.clone();
    accepted = accepted
        .accept_cause(&Cause::Type(parse_quote!(Bad1<'_>)))
        .accept_cause(&Cause::Type(parse_quote!(Bad2<'_, U>)));
    assert!(accepted.errors().is_empty());
    let accepted = accepted.finish().unwrap();
    let ok = ok.result.finish().unwrap();
    let failed_referrer = failed.result.into_referrer();
    // What it recorded is kept: the entries, the contexts and the modules, and so are the ids
    // that the scopes returned.
    assert_eq!(failed_referrer, accepted);
    assert_eq!(
        failed_referrer.to_token_stream().to_string(),
        accepted.to_token_stream().to_string()
    );
    assert_eq!(failed.first, ok_first);
    assert_ne!(failed.last, ok_last);
    let ids = |referrer: &Referrer| {
        common::context_ids(referrer)
            .into_iter()
            .map(ContextId::index)
            .collect::<Vec<_>>()
    };
    assert!(ids(&failed_referrer).len() > ids(&ok).len());
    // The ids which were returned are valid, and are the generics of their scopes: the entries
    // that use them are in those contexts.
    for (id, ty, generics) in [
        (failed.first, "RelA < A >", "< A >"),
        (failed.last, "RelC < C >", "< C >"),
    ] {
        let ix = common::types(&failed_referrer)
            .position(|found| s(found) == ty)
            .unwrap();
        assert_eq!(common::entry_id(&failed_referrer, ix), id, "{}", ty);
        assert_eq!(
            s(&common::entry(&failed_referrer, ix).generics),
            generics,
            "{}",
            ty
        );
    }
    let kept_types: Vec<String> = entries(&failed_referrer)
        .into_iter()
        .map(|(ty, _)| ty)
        .collect();
    assert!(
        kept_types.contains(&"RelU < U >".to_string()),
        "{:?}",
        kept_types
    );
    assert!(kept_types.contains(&"RelM".to_string()), "{:?}", kept_types);
    assert!(common::types(&failed_referrer).any(|ty| s(ty) == "RelK"));
}

#[test]
fn the_error_of_an_inner_scope_is_part_of_the_outer_one() {
    // The error of the inner scope is recorded once, in the list of the outermost call: the
    // closure cannot see it, and it is reported all the same.
    let mut leaker = Leaker::new();
    leaker.enter_generics(&Generics::default(), |view| {
        view.visit_type(&parse_quote!(Rel));
        view.enter_generics(&Generics::default(), |view| {
            view.visit_type(&parse_quote!(Bad<'_>))
        });
    });
    let result = leaker.result();
    assert_eq!(messages(result.errors()).lines().count(), 1);
    assert_eq!(messages(result.errors()), "use absolute path: 'Bad<'_>'");
    assert_eq!(result.into_referrer().iter().count(), 1);

    // With nothing else in the outer scope, it has the error all the same.
    let mut leaker = Leaker::new();
    leaker.enter_generics(&Generics::default(), |view| {
        view.enter_generics(&Generics::default(), |view| {
            view.visit_type(&parse_quote!(Bad<'_>))
        });
    });
    let result = leaker.result();
    assert_eq!(messages(result.errors()), "use absolute path: 'Bad<'_>'");
    assert!(result.into_referrer().is_empty());

    // The errors come in the order in which they were found, those of the inner scopes among the
    // errors of the outer one: before it entered them, in them, after.
    let mut leaker = Leaker::new();
    leaker.enter_generics(&Generics::default(), |view| {
        view.visit_type(&parse_quote!(Bad0<'_>));
        view.enter_generics(&Generics::default(), |view| {
            view.visit_type(&parse_quote!(Bad1<'_>));
            view.enter_generics(&Generics::default(), |view| {
                view.visit_type(&parse_quote!(Bad2<'_>))
            });
            view.visit_type(&parse_quote!(Bad3<'_>));
        });
        view.visit_type(&parse_quote!(Bad4<'_>));
    });
    let result = leaker.result();
    let heads: Vec<String> = result
        .errors()
        .iter()
        .map(|item| s(&item.cause).replace(" < '_ >", ""))
        .collect();
    assert_eq!(heads, ["Bad0", "Bad1", "Bad2", "Bad3", "Bad4"]);

    // A scope which succeeds inside one which fails keeps its context, and the scope that fails
    // keeps the entries it found.
    let mut leaker = Leaker::new();
    let mut inner = None;
    leaker.enter_generics(&Generics::default(), |view| {
        inner = Some(
            view.enter_generics(&parse_quote!(<T>), |view| {
                view.visit_type(&parse_quote!(Rel<T>))
            })
            .0,
        );
        view.visit_type(&parse_quote!(Bad<'_>));
    });
    // The next scope takes the next ids.
    let (next, ()) = leaker.enter_generics(&parse_quote!(<U>), |view| {
        view.visit_type(&parse_quote!(RelU<U>))
    });
    assert_ne!(next, inner.unwrap());
    let result = leaker.result();
    assert_eq!(result.errors().len(), 1);
    let referrer = result.into_referrer();
    assert_eq!(
        entries(&referrer),
        [e("Rel < T >", "< T >"), e("RelU < U >", "< U >")]
    );
    // The scope's generics are those of the entry `Rel<T>`.
    assert_eq!(common::entry_id(&referrer, 0), inner.unwrap());
}

/// A Leaker that allows the primitive types and the names of the standard prelude.
fn leaker_with_std() -> Leaker<impl Allow> {
    let config = Rules::new().primitive().prelude();
    Leaker::with_rule(config)
}

/// `Leaker::enter_generics()` docs: a scope keeps what it recorded when it has errors. The
/// top-level contexts of the scopes inside a scope that fails stay with it, so the Referrer has them
/// all, with the entries of the scope. Audit mutant `rb_leave_top_levels`.
#[test]
fn a_failed_scope_keeps_the_top_levels_of_its_inner_scopes() {
    let mut leaker = Leaker::new();
    let mut ids = Vec::new();
    let (top, ()) = leaker.enter_generics(&parse_quote!(<T>), |view| {
        let (failed, ()) = view.enter_generics(&parse_quote!(<U>), |view| {
            let (v, ()) = view.enter_generics(&parse_quote!(<V>), |view| {
                view.visit_type(&parse_quote!(Rel<V>))
            });
            let (v2, ()) = view.enter_generics(&parse_quote!(<V2>), |view| {
                view.visit_type(&parse_quote!(Rel<V2>))
            });
            ids.extend([v, v2]);
            view.visit_type(&parse_quote!(Rel2<'_>));
        });
        ids.push(failed);
    });
    ids.push(top);
    let result = leaker.result();
    assert_eq!(result.errors().len(), 1);
    let referrer = result.into_referrer();
    // The outermost context, the failed scope, and its inner scopes: the ids returned, and the
    // context of `Rel2<'_>`, which is copied as written and is a part, as its error is about it.
    assert_eq!(common::context_ids(&referrer).len(), 5);
    for id in &ids {
        assert!(common::context_ids(&referrer)
            .into_iter()
            .any(|context| context == *id));
    }
    assert_eq!(
        entries(&referrer),
        [e("Rel < V >", "< V >"), e("Rel < V2 >", "< V2 >")]
    );
}

/// `Leaker::enter_generics()` docs: what a failed inner scope found (a cycle of `Self` in its
/// bounds) is an error of the outermost call too. Audit mutant `rb_leave_env_errors`.
#[test]
fn the_cycle_of_a_failed_inner_scope_is_an_error_of_the_outer_one() {
    fn inner<A: Allow>(view: &mut View<'_, '_, A>) {
        // `Self` is the self type `W<V>`, which the copy writes through the Repeater because of
        // `Rel`, and the bound `V: Fn(Rel)` names it: a cycle of the inner scope.
        view.enter_impl(
            &parse_quote!(<V: Fn(Rel)>),
            &parse_quote!(W<V>),
            None,
            |view| {
                view.visit_where_clause(&parse_quote!(where Self: Sized));
            },
        );
    }
    let mut leaker = leaker_with_std();
    leaker.enter_generics(&parse_quote!(<T>), inner);
    let result = leaker.result();
    assert!(!result.errors().is_empty());
    let found = messages(result.errors());
    // The same scope on its own has the same error.
    let mut leaker = leaker_with_std();
    leaker.enter_generics(&Generics::default(), inner);
    assert_eq!(messages(leaker.result().errors()), found);
}

/// `Leaker::enter_generics()` docs: the decision that an entry of an external crate is not optional
/// (it is in a bound that would make a cycle) belongs to the scope that made it, and a failed
/// inner scope keeps it: the entry is one in the outer scope only where it is outside the
/// bounds of the failed scope. Audit mutant `rb_leave_not_optional`.
#[test]
fn a_failed_inner_scope_keeps_the_optional_entries_it_decided() {
    let mut leaker = leaker_with_std();
    leaker.enter_generics(&parse_quote!(<T: Fn(Rel)>), |view| {
        view.enter_generics(&parse_quote!(<T: Fn(Rel)>), |view| {
            view.visit_where_clause(&parse_quote!(where ::ext::W<T>: Sized));
            view.visit_type(&parse_quote!(Rel2<'_>));
        });
        view.visit_type(&parse_quote!(::ext::W<T>));
    });
    // The only error is that of the inner scope.
    let result = leaker.result();
    assert_eq!(result.errors().len(), 1);
    assert_eq!(
        result.errors()[0].cause,
        Cause::Type(parse_quote!(Rel2<'_>))
    );
    let referrer = result.into_referrer();
    assert_eq!(
        common::types(&referrer).map(s).collect::<Vec<_>>(),
        Vec::<String>::new()
    );
}

/// A failing outermost scope keeps its corrected result: the ids and the Referrer are the same
/// whatever is accepted afterwards, which drops the errors only.
#[test]
fn a_failing_outermost_scope_keeps_its_corrected_result() {
    let mut leaker = Leaker::new();
    leaker.enter_mod("first", [], |view| view.visit_type(&parse_quote!(Rel)));
    // Modules entered, ids taken, and types found by a call which fails.
    let (module, ()) = leaker.enter_mod("second", [ident("S")], |view| {
        view.enter_mod("inner", [], |view| view.visit_type(&parse_quote!(RelKept)));
        view.visit_type(&parse_quote!(Bad<'_>));
    });
    let item = leaker.intern_struct(&parse_quote!(
        struct S<T> {
            a: RelKept2,
            b: Bad<'_>,
        }
    ));
    let result = leaker.result();
    // Each of the two calls reports its error, in the order of the calls.
    let cause = Cause::Type(parse_quote!(Bad<'_>));
    assert_eq!(result.errors().len(), 2);
    assert!(result.errors().iter().all(|error| error.cause == cause));
    let tokens = result.clone().into_referrer().to_token_stream().to_string();
    let accepted = result.clone().accept_cause(&cause);
    assert!(accepted.errors().is_empty());
    assert_eq!(
        accepted.finish().unwrap().to_token_stream().to_string(),
        tokens
    );
    // The calls took the ids they take if the cause is accepted, and the next ones follow them.
    assert_eq!((module, item), (id(2), id(6)));
    // The entries of the calls are kept, and so are the modules they entered.
    assert!(tokens.contains("RelKept"), "{}", tokens);
    assert!(tokens.contains("RelKept2"), "{}", tokens);
}

#[test]
fn a_panic_after_a_recovery_rolls_back_the_scope_and_what_it_recovered() {
    // The scope that panics is rolled back whole, with the inner scope that recovered from an
    // error and the errors that it kept. The outer scope that catches the panic goes on, and its
    // errors are its own.
    let mut leaker = Leaker::new();
    leaker.enter_generics(&Generics::default(), |view| {
        view.visit_type(&parse_quote!(Kept));
        caught(|| {
            let _ = view.enter_generics(&parse_quote!(<T>), |view| {
                view.visit_type(&parse_quote!(RelLost<T>));
                let _ = view.enter_generics(&Generics::default(), |view| {
                    view.visit_type(&parse_quote!(Bad<'_>));
                });
                panic!("a closure panics");
            });
        });
        view.visit_type(&parse_quote!(Bad2<'_>));
    });
    let result = leaker.result();
    let err = result.errors();
    assert_eq!(err.len(), 1);
    assert!(messages(err).contains("Bad2"));
    let referrer = result.into_referrer();
    assert_eq!(entries(&referrer), [e("Kept", "")]);
    // Nothing of the scope that panicked is left, not its context either.
    assert!(!common::types(&referrer).any(|ty| s(ty).contains("RelLost") || s(ty).contains("Bad")));
    assert_eq!(common::context_ids(&referrer).len(), 1);
}

// ── the numbers of the modules ───────────────────────────────────────────────────────────────

fn modules_run() -> (Vec<ContextId>, String) {
    let mut leaker = Leaker::new();
    let mut ids = Vec::new();
    let m: ItemMod = parse_quote!(
        mod m {
            pub mod a {
                pub struct S(Rel);
            }
            pub struct T(Rel);
        }
    );
    ids.push(leaker.intern_mod(&m));
    // The same hint twice.
    for _ in 0..2 {
        let (module, inner) = leaker.enter_mod("same", [], |view| {
            view.visit_type(&parse_quote!(Rel));
            view.enter_mod("same", [], |view| view.visit_type(&parse_quote!(Rel)))
                .0
        });
        ids.push(module);
        ids.push(inner);
    }
    ids.push(leaker.intern_trait(&parse_quote!(
        trait Tr<T> {
            fn f(&self) -> Wrap<T>;
        }
    )));
    (
        ids,
        leaker
            .result()
            .finish()
            .unwrap()
            .to_token_stream()
            .to_string(),
    )
}

#[test]
fn the_numbers_of_the_modules_are_deterministic() {
    let (first, first_tokens) = modules_run();
    let (second, second_tokens) = modules_run();
    assert_eq!(first, second);
    assert_eq!(first_tokens, second_tokens);
    // Each module is numbered when it is entered: `intern_mod()` takes the first (`m`), then `a`
    // is found, and the types of the three are recorded at the end of the call.
    assert_eq!(first[0], id(0));
    // The two with the same hint are not the same module.
    let distinct: std::collections::BTreeSet<_> = first.iter().collect();
    assert_eq!(distinct.len(), first.len());
    // And they are not the module of another call.
    assert!(first[1] != first[3]);
}

#[test]
fn modules_are_numbered_in_the_order_they_are_entered() {
    let mut leaker = Leaker::new();
    let (a, b) = leaker.enter_mod("a", [], |view| view.enter_mod("b", [], |_| ()).0);
    assert_eq!(a, id(0));
    assert_eq!(b, id(1));
    let (c, ()) = leaker.enter_mod("c", [], |_| ());
    assert_eq!(c, id(2));
    assert!(leaker.result().errors().is_empty());
}

// ── the Referrer ─────────────────────────────────────────────────────────────────────────────

#[test]
fn module_path_and_hint() {
    let m: ItemMod = parse_quote!(
        mod m {
            pub struct A {
                y: Rel,
            }
            pub mod inner {
                pub mod deeper {
                    pub struct D {
                        y: Rel,
                    }
                }
                pub struct C {
                    y: Rel,
                }
            }
        }
    );
    let mut leaker = Leaker::new();
    let root = leaker.intern_mod(&m);
    let (hinted, ()) = leaker.enter_mod("hinted", [], |view| view.visit_type(&parse_quote!(Rel)));
    let result = leaker.result();
    let referrer = result.clone().finish().unwrap();
    let paths: Vec<Option<Vec<String>>> = (0..referrer.iter().count())
        .map(|ix| module_path_of(&result, ix))
        .collect();
    assert_eq!(
        paths,
        [
            Some(vec![]),
            Some(vec!["inner".to_string(), "deeper".to_string()]),
            Some(vec!["inner".to_string()]),
            // The one of `enter_mod()` has a hint and no path.
            None
        ]
    );
    // The module analyzed is the first, and the ones which have no entry are not kept (`m` has
    // one, as it is the top-level context).
    assert_eq!(common::in_module(common::entry(&referrer, 0)), Some(root));
    assert_eq!(common::entry_module_hint(&result, 0), None);
    assert_eq!(common::in_module(common::entry(&referrer, 3)), Some(hinted));
    assert_eq!(
        common::entry_module_hint(&result, 3).as_deref(),
        Some("hinted")
    );
    // A context of an entry is not a module.
    assert_ne!(common::entry_id(&referrer, 0), root);
}

#[test]
fn the_visitor_enters_the_module_by_its_id() {
    use syn::visit_mut::VisitMut;
    // The modules `a` and `b` have `Rel` each, entered by `enter_mod()`, and `empty` has none.
    let mut leaker = Leaker::new();
    let (a, ()) = leaker.enter_mod("a", [], |view| view.visit_type(&parse_quote!(Rel)));
    let (b, ()) = leaker.enter_mod("b", [], |view| view.visit_type(&parse_quote!(Rel)));
    let (empty, ()) = leaker.enter_mod("empty", [], |_| ());
    let referrer = leaker.result().finish().unwrap();
    assert_eq!(referrer.iter().count(), 2);
    let copy = |module: ContextId| {
        let mut visitor =
            referrer.visitor(module, |_, ix, _| parse_str(&format!("P{}", ix)).unwrap());
        let mut item: ItemStruct = parse_quote!(
            struct C {
                y: Rel,
            }
        );
        visitor.visit_item_struct_mut(&mut item);
        s(&item)
    };
    let in_a = common::in_module(common::entry(&referrer, 0)) == Some(a);
    let (first, second) = if in_a { (a, b) } else { (b, a) };
    assert_eq!(copy(first), "struct C { y : P0 , }");
    assert_eq!(copy(second), "struct C { y : P1 , }");
    // A module which is not one of those of the entries: only the entries with no module match
    // there, and `Rel` is in a module.
    assert_eq!(copy(empty), "struct C { y : Rel , }");
}

#[test]
fn the_visitor_finds_the_modules_found_in_a_module_by_their_names() {
    use syn::visit_mut::VisitMut;
    // `inner` is found in the module entered, so it is looked up by its name below it.
    let inner: ItemMod = parse_quote!(
        mod inner {
            pub struct C {
                y: Rel,
            }
        }
    );
    let mut leaker = Leaker::new();
    let (outer, ()) = leaker.enter_mod("outer", [], |view| {
        view.visit_item(&Item::Mod(inner.clone()))
    });
    let referrer = leaker.result().finish().unwrap();
    let copy = |item: &ItemMod, module: ContextId| {
        let mut visitor = referrer.visitor(module, |_, _, _| parse_quote!(P));
        let mut item = item.clone();
        visitor.visit_item_mod_mut(&mut item);
        s(&item)
    };
    let rewritten: ItemMod = parse_quote!(
        mod inner {
            pub struct C {
                y: P,
            }
        }
    );
    let rewritten = s(&rewritten);
    // `inner` is a module in `outer`, the module of the context: it is looked up by its name below
    // it.
    assert_eq!(copy(&inner, outer), rewritten);
    // A module of another name is not one that the Leaker saw below `outer`: no entry depends on
    // it, and the entry of `inner` is not found there.
    let other: ItemMod = parse_quote!(
        mod other {
            pub struct C {
                y: Rel,
            }
        }
    );
    assert_eq!(copy(&other, outer), s(&other));
    // A visited `mod` that none of the modules of the context is named like is the module itself
    // (F55): `mod outer` with the context `outer` is its items, so `inner` is looked up below it
    // by its name, as in `outer`. So is `inner` with the context of the module `inner`, which has
    // no module of that name: its `Rel` is the entry of the module.
    let wrapped =
        |text: &str| s(&parse_str::<ItemMod>(&format!("mod outer {{ {} }}", text)).unwrap());
    let wrap = |item: &ItemMod| -> ItemMod { parse_quote!(mod outer { #item }) };
    assert_eq!(copy(&wrap(&inner), outer), wrapped(&rewritten));
    let inner_id = common::in_module(common::entry(&referrer, 0)).unwrap();
    assert_ne!(inner_id, outer);
    assert_eq!(copy(&inner, inner_id), rewritten);
}

#[test]
fn modules_are_serialized() {
    let m: ItemMod = parse_quote!(
        mod m {
            pub mod inner {
                pub struct C {
                    y: Rel,
                }
            }
        }
    );
    let mut leaker = Leaker::new();
    let root = leaker.intern_mod(&m);
    let (hinted, ()) = leaker.enter_mod("with \"quotes\"", [], |view| {
        view.visit_type(&parse_quote!(Rel2));
        view.enter_mod("nested", [], |view| view.visit_type(&parse_quote!(Rel3)));
    });
    let result = leaker.result();
    let referrer = result.clone().finish().unwrap();
    let tokens = referrer.to_token_stream();
    let parsed: Referrer = parse2(tokens.clone()).unwrap();
    let parsed_result: LeakerResult = parse2(tokens.clone()).unwrap();
    assert_eq!(parsed, referrer);
    assert_eq!(parsed.to_token_stream().to_string(), tokens.to_string());
    // The modules, their paths and hints, and what depends on them are the same.
    let hinted_entry = common::types(&parsed)
        .position(|ty| s(ty) == "Rel2")
        .unwrap();
    assert_eq!(
        common::entry_module_hint(&parsed_result, hinted_entry).as_deref(),
        Some("with \"quotes\"")
    );
    assert_eq!(
        common::entry_module_path(&parsed_result, hinted_entry),
        None
    );
    assert_eq!(
        common::in_module(common::entry(&parsed, hinted_entry)),
        Some(hinted)
    );
    for ix in 0..referrer.iter().count() {
        let module = common::in_module(common::entry(&referrer, ix));
        assert_eq!(common::in_module(common::entry(&parsed, ix)), module);
        // The module's own context, with its id, path, hint and parent.
        assert_eq!(
            common::entry_module(&parsed_result, ix),
            common::entry_module(&result, ix)
        );
    }
    // The ids of the modules are written, and the parent of a nested one.
    let text = tokens.to_string();
    assert!(text.contains("mod { path = [] }"), "{}", text);
    assert!(text.contains("hint = \"nested\" , parent ="), "{}", text);
    // The module is used to find the entries after the round trip: the items of `m` are visited
    // in the context of `m`, and `inner` is looked up by its name below it.
    use syn::visit_mut::VisitMut;
    let mut item: ItemMod = parse_quote!(
        mod inner {
            pub struct C {
                y: Rel,
            }
        }
    );
    parsed
        .visitor(root, |_, _, _| parse_quote!(P))
        .visit_item_mod_mut(&mut item);
    assert_eq!(
        s(&item),
        s(&parse_quote!(
            mod inner {
                pub struct C {
                    y: P,
                }
            }
        ) as &ItemMod)
    );
}

#[test]
fn the_format_of_the_previous_version_is_rejected() {
    assert_eq!(Referrer::FORMAT_VERSION, 16);
    // What each format changed is told above `Referrer::FORMAT_VERSION`. The old bodies: the
    // format 1 had modules as `mod self ;`, the format 2 as `mod { path = [] } ;` (without the
    // `Self` and the module of a scope), the later ones the grammar of today.
    for version in 1..Referrer::FORMAT_VERSION {
        let old = match version {
            1 => "2 mod self ; (Rel) mod a :: b ; (Rel)",
            2 => "2 mod { path = [] } ; () mod 0 ; (Rel)",
            _ => "2 < T > ; () < T > ; (Rel)",
        };
        // The tag is checked first, with a body of the version or a plain one.
        for body in [old, "(MyType)"] {
            let text = format!("\"type-leak-format-{}\" {}", version, body);
            let err = parse_str::<Referrer>(&text).unwrap_err();
            assert_eq!(
                err.to_string(),
                format!(
                    "this Referrer was serialized by an incompatible type-leak (format {}, \
                     expected {}): the macros which emit and parse it must use the same version \
                     of type-leak",
                    version,
                    Referrer::FORMAT_VERSION
                ),
                "format {}: {}",
                version,
                body
            );
        }
    }
    let current = format!("\"type-leak-format-{}\" (MyType)", Referrer::FORMAT_VERSION);
    assert!(parse_str::<Referrer>(&current).is_ok());
}

// ── a View is only a restricted wrapper ──────────────────────────────────────────────────────

/// A function taking a view, as a caller of the scopes would write.
fn visit_all(view: &mut View, types: &[Type]) {
    for ty in types {
        view.visit_type(ty);
    }
}

#[test]
fn a_view_can_be_passed_around() {
    let types: Vec<Type> = vec![parse_quote!(Rel), parse_quote!(Rel2)];
    let mut leaker = Leaker::new();
    leaker.enter_mod("m", [], |view| visit_all(view, &types));
    assert_eq!(leaker.result().finish().unwrap().iter().count(), 2);
}

// ── a panic in a nested scope ────────────────────────────────────────────────────────────────

/// Runs `f`, which panics, as the closure around it catches the panic.
fn caught(f: impl FnOnce()) {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
    assert!(result.is_err());
}

#[test]
fn a_panic_caught_in_a_nested_scope_leaves_the_outer_scope_as_it_was() {
    // The nested scope declares a generic and `Self`, enters a module, and records types, a
    // context and an error, and then panics. The outer closure catches it and goes on: it is as if
    // the scope had never been entered.
    fn run(panic: bool) -> (ContextId, LeakerResult) {
        let mut leaker = Leaker::new();
        let (top, ()) = leaker.enter_generics(&parse_quote!(<T>), |view| {
            view.visit_type(&parse_quote!(Rel<T>));
            if panic {
                caught(|| {
                    let _ = view.enter_type(&ident("S"), &parse_quote!(<U>), |view| {
                        view.visit_type(&parse_quote!(RelU<U>));
                        view.visit_type(&parse_quote!(Rel2<Self>));
                        view.visit_type(&parse_quote!(Bad<'_>));
                        view.enter_mod("lost", [], |_| ());
                        panic!("a closure panics");
                    });
                });
            }
            // `U` is not a generic any more, and the module is not entered.
            view.visit_type(&parse_quote!(Wrap<U>));
            view.visit_type(&parse_quote!(Wrap2<T>));
        });
        let result = leaker.result();
        result.clone().finish().unwrap();
        (top, result)
    }
    let (top, result) = run(true);
    let (expected_top, expected) = run(false);
    let (referrer, expected) = (result.clone().into_referrer(), expected.into_referrer());
    assert_eq!(top, expected_top);
    assert_eq!(referrer, expected);
    assert_eq!(
        entries(&referrer),
        [
            e("Rel < T >", "< T >"),
            e("Wrap2 < T >", "< T >"),
            e("Wrap < U >", "")
        ]
    );
    // The top-level context is that of the generics, with no module.
    assert_eq!(common::entry_id(&referrer, 0), top);
    assert_eq!(common::entry_module(&result, 0), None);
    assert_eq!(
        referrer.to_token_stream().to_string(),
        expected.to_token_stream().to_string()
    );
}

#[test]
fn a_panic_caught_in_a_nested_scope_restores_what_the_scope_declares() {
    // `Self` is the one of the outer scope again.
    let mut leaker = Leaker::new();
    leaker.enter_type(&ident("O"), &Generics::default(), |view| {
        caught(|| {
            let _ = view.enter_impl(&Generics::default(), &parse_quote!(::p::W), None, |_| {
                panic!("a closure panics");
            });
        });
        view.visit_type(&parse_quote!(Rel2<Self>));
    });
    let referrer = leaker.result().finish().unwrap();
    assert_eq!(self_of(&referrer, 0), "Self = O");

    // Where `Self` has no meaning, it is not given one by the scope which panicked.
    let mut leaker = Leaker::new();
    leaker.enter_generics(&Generics::default(), |view| {
        caught(|| {
            let _ = view.enter_type(&ident("S"), &Generics::default(), |_| {
                panic!("a closure panics");
            });
        });
        view.visit_type(&parse_quote!(Rel<Self>));
    });
    // The errors are those of the outer scope: `Bad<'_>` of the scope which panicked is not one.
    let result = leaker.result();
    let err = result.errors();
    assert!(
        err.iter()
            .all(|item| item.reason == Reason::SelfWithoutMeaning),
        "{:?}",
        err
    );
    assert_eq!(err.len(), 1);
}

#[test]
fn a_panic_caught_in_a_nested_module_restores_the_module() {
    // The types of the outer module are in it, and the top-level context is its.
    let mut leaker = Leaker::new();
    let (outer, ()) = leaker.enter_mod("outer", [parse_quote!(S)], |view| {
        caught(|| {
            let _ = view.enter_mod("inner", [parse_quote!(Lost)], |view| {
                view.visit_type(&parse_quote!(Rel));
                panic!("a closure panics");
            });
        });
        // `S` is declared by the module, and `Lost` was by the module which is not entered.
        view.visit_type(&parse_quote!(S<Lost>));
    });
    let result = leaker.result();
    let referrer = result.clone().finish().unwrap();
    assert_eq!(entries(&referrer), [e("Lost", "")]);
    assert_eq!(
        common::in_module(common::entry(&referrer, 0)),
        Some(outer),
        "the type is in the outer module"
    );
    assert_eq!(
        common::entry_module_hint(&result, 0).as_deref(),
        Some("outer")
    );
    // (A module which panicked takes no number: see the layers of the item scopes, below.)
}

#[test]
fn a_panic_caught_in_the_outermost_scope_of_a_generics_scope_keeps_the_top_level_id() {
    // The top-level context of the outer scope is its generics, not the module of the nested one
    // which panicked (the module stack was not unwound).
    let mut leaker = Leaker::new();
    let (top, ()) = leaker.enter_generics(&parse_quote!(<T>), |view| {
        caught(|| {
            let _ = view.enter_mod("m", [], |_| panic!("a closure panics"));
        });
        view.visit_type(&parse_quote!(Rel<T>));
    });
    let result = leaker.result();
    let referrer = result.clone().finish().unwrap();
    assert_eq!(common::entry_id(&referrer, 0), top);
    assert_eq!(s(&common::entry(&referrer, 0).generics), "< T >");
    assert_eq!(common::entry_module(&result, 0), None);
    assert_eq!(common::in_module(common::entry(&referrer, 0)), None);
}

// ── a panic in an outermost scope ────────────────────────────────────────────────────────────

/// Runs `f`, which panics with the message `message`, and checks that the panic goes on unchanged.
fn caught_with(message: &str, f: impl FnOnce()) {
    let payload = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).unwrap_err();
    assert_eq!(
        payload.downcast_ref::<String>().map(|s| s.as_str()),
        Some(message)
    );
}

/// A Leaker with contexts from earlier calls: a trait, and a struct with other generics.
fn leaker_with_contexts() -> (Leaker, Vec<ContextId>) {
    let mut leaker = Leaker::new();
    let first = leaker.intern_trait(&parse_quote!(
        trait Tr<T> {
            type A: ::p::X<Rel<T>>;
        }
    ));
    let second = leaker.intern_struct(&parse_quote!(
        struct S<U>(Rel2<U>);
    ));
    (leaker, vec![first, second])
}

/// What a scope that panics leaves behind, if the table of contexts were not rolled back: a nested
/// module and nested generics with their own contexts.
fn record_and_panic(view: &mut View<'_, '_>, message: &str) {
    view.visit_type(&parse_quote!(RelLost<V>));
    view.enter_mod("lost", [], |_| ());
    view.enter_generics(&parse_quote!(<X, Y>), |_| ());
    panic!("{}", message);
}

/// The ids of the calls after the panic and the Referrer, with the panic (or without).
fn after_a_panic(panic: bool) -> (Vec<ContextId>, LeakerResult) {
    let (mut leaker, mut ids) = leaker_with_contexts();
    if panic {
        caught_with("an outermost closure panics", || {
            let _ = leaker.enter_generics(&parse_quote!(<V>), |view| {
                record_and_panic(view, "an outermost closure panics")
            });
        });
        caught_with("an outermost closure panics", || {
            let _ = leaker.enter_mod("lost", [], |view| {
                record_and_panic(view, "an outermost closure panics")
            });
        });
        caught_with("an outermost closure panics", || {
            let _ = leaker.enter_type(&ident("S3"), &parse_quote!(<V>), |view| {
                record_and_panic(view, "an outermost closure panics")
            });
        });
    }
    // New contexts get the numbers they would get without the panic.
    ids.push(leaker.intern_struct(&parse_quote!(
        struct S3<W>(Rel3<W>);
    )));
    ids.push(
        leaker
            .enter_mod("kept", [], |view| view.visit_type(&parse_quote!(Rel4)))
            .0,
    );
    // The context of the earlier trait is still there: the same generics share it.
    ids.push(leaker.intern_struct(&parse_quote!(
        struct Again<T>(Rel5<T>);
    )));
    let result = leaker.result();
    result.clone().finish().unwrap();
    (ids, result)
}

#[test]
fn a_panic_in_an_outermost_scope_keeps_the_earlier_contexts_and_ids() {
    let (ids, result) = after_a_panic(true);
    let (expected_ids, expected) = after_a_panic(false);
    let (referrer, expected) = (result.clone().into_referrer(), expected.into_referrer());
    assert_eq!(ids, expected_ids);
    assert_eq!(referrer, expected);
    assert_eq!(
        referrer.to_token_stream().to_string(),
        expected.to_token_stream().to_string()
    );
    // The earlier calls, then the new ones in order, and then `<T>` of the trait again.
    assert_eq!(ids, [id(0), id(1), id(2), id(3), id(0)]);
    // Each of the first three is the context of the entry that uses its generics, and the fourth
    // is the module of its entry.
    let at = |text: &str| {
        common::types(&referrer)
            .position(|ty| s(ty) == text)
            .unwrap()
    };
    for (id, ty, generics) in [
        (ids[0], "Rel < T >", "< T >"),
        (ids[1], "Rel2 < U >", "< U >"),
        (ids[2], "Rel3 < W >", "< W >"),
    ] {
        assert_eq!(common::entry_id(&referrer, at(ty)), id, "{}", ty);
        assert_eq!(s(&common::entry(&referrer, at(ty)).generics), generics);
    }
    assert_eq!(
        common::in_module(common::entry(&referrer, at("Rel4"))),
        Some(ids[3])
    );
    assert_eq!(
        common::entry_module_hint(&result, at("Rel4")).as_deref(),
        Some("kept")
    );
    assert_eq!(
        entries(&referrer),
        [
            e("Rel < T >", "< T >"),
            e("Rel5 < T >", "< T >"),
            e("Rel2 < U >", "< U >"),
            e("Rel3 < W >", "< W >"),
            e("Rel4", ""),
        ]
    );
}

/// A nested panic that nothing in the outermost scope catches ends at the Leaker. The payload is
/// the one of the closure, and the Leaker is as if the scope had not been entered.
fn after_a_nested_panic(panic: bool) -> (Vec<ContextId>, Referrer) {
    let (mut leaker, mut ids) = leaker_with_contexts();
    if panic {
        caught_with("a nested closure panics", || {
            let _ = leaker.enter_generics(&parse_quote!(<V>), |view| {
                view.visit_type(&parse_quote!(RelOuter<V>));
                view.enter_mod("lost", [], |_| ());
                let _ = view.enter_type(&ident("N"), &parse_quote!(<N>), |view| {
                    view.visit_type(&parse_quote!(RelNested<N>));
                    // Two levels down: the scopes around are left in turn.
                    let _ = view.enter_impl(
                        &parse_quote!(<I>),
                        &parse_quote!(::p::W<I>),
                        None,
                        |view| record_and_panic(view, "a nested closure panics"),
                    );
                    unreachable!("the panic goes through the scopes around");
                });
                unreachable!("the panic goes through the scopes around");
            });
        });
    }
    ids.push(leaker.intern_struct(&parse_quote!(
        struct S3<W>(Rel3<W>);
    )));
    ids.push(
        leaker
            .enter_generics(&parse_quote!(<V>), |view| {
                view.visit_type(&parse_quote!(RelOuter<V>))
            })
            .0,
    );
    (ids, leaker.result().finish().unwrap())
}

#[test]
fn a_nested_panic_not_caught_in_an_outermost_scope_ends_at_the_leaker_with_its_contexts() {
    let (ids, referrer) = after_a_nested_panic(true);
    let (expected_ids, expected) = after_a_nested_panic(false);
    assert_eq!(ids, expected_ids);
    assert_eq!(referrer, expected);
    assert_eq!(ids, [id(0), id(1), id(2), id(3)]);
    assert_eq!(
        entries(&referrer),
        [
            e("Rel < T >", "< T >"),
            e("Rel2 < U >", "< U >"),
            e("Rel3 < W >", "< W >"),
            e("RelOuter < V >", "< V >"),
        ]
    );
}

// ── `Self` in the generics of a scope ────────────────────────────────────────────────────────

/// The mentions of `Self` which the errors of `Reason::SelfWithoutMeaning` have, as tokens.
fn selves_of(err: &[ErrorItem]) -> Vec<String> {
    assert!(
        err.iter()
            .all(|item| item.reason == Reason::SelfWithoutMeaning),
        "{:?}",
        err
    );
    err.iter()
        .map(|item| item.cause.to_token_stream().to_string())
        .collect()
}

#[test]
fn self_in_the_generics_of_a_scope_which_has_none_is_an_error() {
    let with_where = |generics: &str, where_clause: &str| -> Generics {
        let mut generics: Generics = parse_str(generics).unwrap();
        generics.where_clause = Some(parse_str(where_clause).unwrap());
        generics
    };
    // The generics, the mention of `Self` in them, and whether a scope which declares `Self` is
    // fine with them (not with the default, which cannot depend on it).
    let cases: Vec<(Generics, &str, bool)> = vec![
        (parse_quote!(<T: ::p::Tr<Self>>), "Self", true),
        (parse_quote!(<T: ::p::Tr<Self::Item>>), "Self :: Item", true),
        (parse_quote!(<T: ::p::Tr<Item = Self>>), "Self", true),
        (parse_quote!(<T = Rel<Self>>), "Self", false),
        (with_where("<U>", "where U: ::p::Tr<Self>"), "Self", true),
        (with_where("<'a, T: 'a>", "where Self: 'a"), "Self", true),
        (
            with_where("<T>", "where Self: ::core::marker::Sized"),
            "Self",
            true,
        ),
    ];
    for (generics, mention, declared_ok) in &cases {
        let text = s(generics);
        // On the Leaker, with nothing visited.
        let result = recorded(|leaker| {
            leaker.enter_generics(generics, |_| ());
        });
        let err = result.errors();
        assert_eq!(messages(err), "`Self` has no meaning here", "{}", text);
        assert_eq!(selves_of(err), [mention.to_string()], "{}", text);
        // In a scope nested in one which has no `Self` either (the module, and the generics).
        let result = recorded(|leaker| {
            leaker.enter_mod("m", [], |view| {
                view.enter_generics(generics, |view| view.visit_type(&parse_quote!(Rel)));
            });
        });
        assert_eq!(
            selves_of(result.errors()),
            [mention.to_string()],
            "{}",
            text
        );
        let result = recorded(|leaker| {
            leaker.enter_generics(&Generics::default(), |view| {
                view.enter_generics(generics, |_| ());
            });
        });
        assert_eq!(
            selves_of(result.errors()),
            [mention.to_string()],
            "{}",
            text
        );
        // The scope fails, and keeps what it recorded: the one around goes on, with its error too.
        let result = recorded(|leaker| {
            leaker.enter_mod("m", [], |view| {
                view.enter_generics(generics, |view| view.visit_type(&parse_quote!(Wrap)));
                view.visit_type(&parse_quote!(Rel));
            });
        });
        assert_self_without_meaning(result.errors());
        assert_eq!(
            selves_of(result.errors()),
            [mention.to_string()],
            "{}",
            text
        );
        assert_eq!(
            entries(&result.into_referrer()),
            [e("Rel", ""), e("Wrap", "")],
            "{}",
            text
        );
        // The scopes which declare `Self` have a meaning for it, so they are fine.
        let result = recorded(|leaker| {
            leaker.enter_type(&ident("S"), generics, |_| ());
        });
        assert_eq!(result.errors().is_empty(), *declared_ok, "{}", text);
        let result = recorded(|leaker| {
            leaker.enter_trait(&ident("Tr"), generics, |_| ());
        });
        assert_eq!(result.errors().is_empty(), *declared_ok, "{}", text);
        let result = recorded(|leaker| {
            leaker.enter_impl(generics, &parse_quote!(::p::W), None, |_| ());
        });
        assert_eq!(result.errors().is_empty(), *declared_ok, "{}", text);
        // The generics nested in one of them keep it.
        let result = recorded(|leaker| {
            leaker.enter_type(&ident("S"), &Generics::default(), |view| {
                view.enter_generics(generics, |_| ());
            });
        });
        assert_eq!(result.errors().is_empty(), *declared_ok, "{}", text);
    }
    // Each mention is reported, with the others of the scope, at its own tokens.
    let result = recorded(|leaker| {
        leaker.enter_generics(
            &with_where(
                "<T: ::p::Tr<Self>, U: ::p::Tr<Self::X>>",
                "where Self: 'static",
            ),
            |_| (),
        );
    });
    let err = result.errors();
    assert_eq!(selves_of(err), ["Self", "Self :: X", "Self"]);
    assert_eq!(messages(err).lines().count(), 3);
    // It is the same for the generics of the View, and the outer scope has the error once.
    let result = recorded(|leaker| {
        leaker.enter_generics(&Generics::default(), |view| {
            view.enter_generics(&parse_quote!(<T: ::p::Tr<Self>>), |_| ());
        });
    });
    assert_eq!(selves_of(result.errors()), ["Self"]);
}

#[test]
fn self_in_a_default_where_a_scope_declares_it_is_the_conflict_of_the_default() {
    // Where `Self` has a meaning, a default which mentions it is a conflict of its own (the
    // implementation of the default is for any type); where it has none, it is `Self` which has
    // no meaning.
    let generics: Generics = parse_quote!(<T = Rel<Self>>);
    let result = recorded(|leaker| {
        leaker.enter_type(&ident("S"), &generics, |_| ());
    });
    let err = result.errors();
    assert_eq!(
        err[0].notes.first().map(|(_, n)| n.as_str()),
        Some("a default of a parameter cannot depend on `Self`")
    );
    assert_relative_path(err);
    let result = recorded(|leaker| {
        leaker.enter_generics(&generics, |_| ());
    });
    let err = result.errors();
    assert_self_without_meaning(err);
    assert_eq!(messages(err), "`Self` has no meaning here");
    // Also in a scope nested in one which declares it, and in the default of its own.
    let result = recorded(|leaker| {
        leaker.enter_type(&ident("S"), &Generics::default(), |view| {
            view.enter_generics(&generics, |_| ());
        });
    });
    assert_relative_path(result.errors());
    let result = recorded(|leaker| {
        leaker.enter_mod("m", [], |view| {
            view.enter_generics(&generics, |_| ());
        });
    });
    assert_self_without_meaning(result.errors());
}

// ── `Self` of a nested item ──────────────────────────────────────────────────────────────────

#[test]
fn a_nested_item_has_a_self_of_its_own() {
    let items: Vec<Item> = vec![
        parse_quote!(
            trait Tr {
                fn f(&self) -> Rel<Self>;
            }
        ),
        parse_quote!(
            struct X(Rel<Self>);
        ),
        parse_quote!(
            enum E {
                A(Rel<Self>),
            }
        ),
        parse_quote!(
            union U {
                a: Rel<Self>,
            }
        ),
        parse_quote!(
            impl ::p::W {
                fn f(&self) -> Rel<Self> {
                    loop {}
                }
            }
        ),
    ];
    for item in &items {
        let text = s(item);
        // In a scope of a type: nested in generics, in a module, directly, and inside several of
        // them.
        let mut leaker = Leaker::new();
        leaker.enter_type(&ident("S"), &Generics::default(), |view| {
            view.enter_generics(&Generics::default(), |view| view.visit_item(item));
            view.enter_mod("m", [], |view| view.visit_item(item));
            view.visit_item(item);
            view.enter_mod("m", [], |view| {
                view.enter_generics(&Generics::default(), |view| view.visit_item(item));
            });
        });
        let result = leaker.result();
        assert!(
            result.errors().is_empty(),
            "{}: {}",
            text,
            messages(result.errors())
        );
        let referrer = result.finish().unwrap();
        // (Each module is another one, so the type is an entry in each of them.)
        assert!(
            (0..referrer.iter().count()).any(|ix| common::entry(&referrer, ix).self_ty.is_some()),
            "{}",
            text
        );
        // The `Self` of the entries is the item's, not that of the scope around (`S`).
        assert!(
            (0..referrer.iter().count()).all(|ix| self_of(&referrer, ix) != "Self = S"),
            "{}",
            text
        );
        // In the other scopes.
        for scope in 0..3 {
            let result = recorded(|leaker| match scope {
                0 => {
                    leaker.enter_type(&ident("S"), &Generics::default(), |view| {
                        view.visit_item(item)
                    });
                }
                1 => {
                    leaker.enter_impl(&Generics::default(), &parse_quote!(::p::W), None, |view| {
                        view.visit_item(item)
                    });
                }
                _ => {
                    leaker.enter_trait(&ident("S"), &Generics::default(), |view| {
                        view.visit_item(item)
                    });
                }
            });
            assert!(result.errors().is_empty(), "{}: {:?}", text, result);
        }
    }
    // The `Self` of the scope itself is the scope's again after the nested items.
    let mut leaker = Leaker::new();
    leaker.enter_type(&ident("S"), &Generics::default(), |view| {
        view.visit_item(&items[0]);
        view.visit_type(&parse_quote!(Rel2<Self>));
    });
    let referrer = leaker.result().finish().unwrap();
    let rel2 = common::types(&referrer)
        .position(|ty| s(ty) == "Rel2 < Self >")
        .unwrap();
    assert_eq!(self_of(&referrer, rel2), "Self = S");
    // A function does not declare `Self`, so it is the one of the scope around.
    let mut leaker = Leaker::new();
    leaker.enter_impl(&Generics::default(), &parse_quote!(::p::W), None, |view| {
        view.visit_item(&parse_quote!(
            fn f() -> Rel<Self> {
                loop {}
            }
        ));
    });
    let referrer = leaker.result().finish().unwrap();
    assert_eq!(entries(&referrer), [e("Rel < Self >", "")]);
    assert_eq!(self_of(&referrer, 0), "Self = :: p :: W");
    // The items of a module analyzed as a whole have their own.
    let m: ItemMod = parse_quote!(
        mod m {
            pub struct S {
                a: Rel<Self>,
            }
            pub mod inner {
                pub trait Tr {
                    fn f(&self) -> Rel2<Self>;
                }
            }
        }
    );
    let mut leaker = Leaker::new();
    leaker.intern_mod(&m);
    assert_eq!(leaker.result().finish().unwrap().iter().count(), 2);
}

// ── `Self` of a trait ────────────────────────────────────────────────────────────────────────

#[test]
fn self_in_a_trait_is_the_implementor() {
    let item: ItemTrait = parse_quote!(
        trait Tr<T> {
            fn f(&self) -> Rel<Self>;
            fn g(&self, t: T) -> (Rel2, Self);
        }
    );
    let mut leaker = Leaker::new();
    leaker.intern_trait(&item);
    let referrer = leaker.result().finish().unwrap();
    assert_eq!(
        entries(&referrer),
        [e("Rel < Self >", "< T >"), e("Rel2", "")]
    );
    assert_eq!(self_of(&referrer, 0), "Self: Tr < T >");
    // What can be decomposed is, and the receivers are not affected.
    let item: ItemTrait = parse_quote!(
        trait Tr<T> {
            fn f(self) -> (Rel, Self);
            fn g(&self, t: &Self) -> Rel2<T>;
        }
    );
    let mut leaker = Leaker::new();
    leaker.intern_trait(&item);
    let referrer = leaker.result().finish().unwrap();
    assert_eq!(entries(&referrer), [e("Rel", ""), e("Rel2 < T >", "< T >")]);
    assert!(!referrer.has_self());
    // A bound of the generics which mentions `Self` is no error either.
    let item: ItemTrait = parse_quote!(
        trait Tr<T: ::core::cmp::PartialEq<Self>> {
            fn f(&self) -> Wrap<T>;
        }
    );
    let mut leaker = Leaker::new();
    leaker.intern_trait(&item);
    let result = leaker.result();
    assert!(result.finish().is_ok());
    // The same for the scope of the Leaker and of the View.
    let ty: Type = parse_quote!(Rel<Self, T>);
    let generics: Generics = parse_quote!(<T>);
    let result = recorded(|leaker| {
        leaker.enter_trait(&ident("Tr"), &generics, |view| view.visit_type(&ty));
    });
    assert!(result.errors().is_empty());
    let result = recorded(|leaker| {
        leaker.enter_generics(&Generics::default(), |view| {
            view.enter_trait(&ident("Tr"), &generics, |view| view.visit_type(&ty));
        });
    });
    assert!(result.errors().is_empty());
}

// ── a function as the analyzed item, and `Visit::visit_signature()` ──────────────────────────

#[test]
fn a_function_which_is_the_analyzed_item_is_analyzed_in_a_scope_of_its_generics() {
    // The function is the item analyzed: its generics are those of the Leaker (the scope declares
    // them), and the types of the signature are visited in it.
    let sig: Signature = parse_quote!(fn f<'b>(x: ::p::W<'b, dyn RelTr>) -> &'b ());
    let mut leaker = Leaker::new();
    let (top, ()) = enter_function(&mut leaker, &sig);
    let referrer = leaker.result().finish().unwrap();
    assert_eq!(
        entries(&referrer),
        [e(":: p :: W < 'b , dyn RelTr >", "< 'b >")]
    );
    assert_eq!(common::entry_id(&referrer, 0), top);
    // The lifetime is a generic of the scope: the signature is not compared with a declaration, so
    // it is not checked for E0195, which only the generics of a method or a nested function are.
    // Nor is it when the scope is nested in another one.
    let mut leaker = Leaker::new();
    leaker.enter_generics(&Generics::default(), |view| {
        view.enter_generics(&sig.generics, |view| visit_function(view, &sig));
    });
    assert!(leaker.result().errors().is_empty());
}

#[test]
fn the_generics_of_a_method_are_local_and_checked_for_e0195() {
    // The same signature as a method: the lifetime is local, which the Repeater's copy would turn
    // into an early-bound one where the trait's is late-bound (E0195).
    let sig: Signature = parse_quote!(fn f<'b>(x: ::p::W<'b, dyn RelTr>) -> &'b ());
    for scope in 1..5 {
        let result = recorded(|leaker| match scope {
            1 => {
                leaker.enter_impl(
                    &Generics::default(),
                    &parse_quote!(::p::W),
                    Some(&parse_quote!(::p::Tr)),
                    |view| {
                        let method: ImplItemFn = parse_quote!(
                            fn f<'b>(x: ::p::W<'b, dyn RelTr>) -> &'b () {
                                loop {}
                            }
                        );
                        view.visit_impl_item_fn(&method)
                    },
                );
            }
            2 => {
                leaker.enter_trait(&ident("Tr"), &Generics::default(), |view| {
                    let method: TraitItemFn = parse_quote!(
                        fn f<'b>(x: ::p::W<'b, dyn RelTr>) -> &'b ();
                    );
                    view.visit_trait_item_fn(&method)
                });
            }
            // The signature alone is the one of a method.
            3 => {
                leaker.enter_trait(&ident("Tr"), &Generics::default(), |view| {
                    view.visit_signature(&sig)
                });
            }
            // A method of an impl of a trait, nested in generics.
            _ => {
                leaker.enter_generics(&Generics::default(), |view| {
                    view.visit_item(&parse_quote!(
                        impl ::p::Tr for ::p::W {
                            fn f<'b>(x: ::p::W<'b, dyn RelTr>) -> &'b () {
                                loop {}
                            }
                        }
                    ))
                });
            }
        });
        let err = result.errors();
        assert!(!err.is_empty(), "{}", scope);
        assert!(to_syn_error(err).to_string().contains("E0195"), "{}", scope);
    }
    // A function nested in generics, and a method of an inherent impl, are no methods of a trait:
    // nothing to differ from (F22), but their generics are local too.
    let mut leaker = Leaker::new();
    leaker.enter_generics(&Generics::default(), |view| {
        view.visit_item(&parse_quote!(
            fn f<'b>(x: ::p::W<'b, dyn RelTr>) -> &'b () {
                loop {}
            }
        ))
    });
    assert_eq!(leaker.result().finish().unwrap().iter().count(), 1);
    let item: ItemMod = parse_quote! {
        mod m { fn f<'b>(x: ::p::W<'b, dyn RelTr>) -> &'b () { loop {} } }
    };
    let mut leaker = Leaker::new();
    leaker.intern_mod(&item);
    assert_eq!(leaker.result().finish().unwrap().iter().count(), 1);
    let mut leaker = Leaker::new();
    leaker.enter_generics(&Generics::default(), |view| {
        view.visit_item(&parse_quote!(
            impl ::p::W {
                fn f<'b>(x: ::p::W<'b, dyn RelTr>) -> &'b () {
                    loop {}
                }
            }
        ))
    });
    // The self type and the path.
    assert_eq!(leaker.result().finish().unwrap().iter().count(), 2);
    // The same method in `enter_impl()` given no trait (an inherent impl) has none to differ from.
    let result = recorded(|leaker| {
        leaker.enter_impl(&Generics::default(), &parse_quote!(::p::W), None, |view| {
            let method: ImplItemFn = parse_quote!(
                fn f<'b>(x: ::p::W<'b, dyn RelTr>) -> &'b () {
                    loop {}
                }
            );
            view.visit_impl_item_fn(&method)
        });
    });
    assert!(result.errors().is_empty());
    // Generics around do not declare those of the signature, whichever they are.
    let result = recorded(|leaker| {
        leaker.enter_generics(&parse_quote!(<'a>), |view| view.visit_signature(&sig));
    });
    assert!(!result.errors().is_empty());
    let result = recorded(|leaker| {
        leaker.enter_generics(&sig.generics, |view| view.visit_signature(&sig));
    });
    assert!(!result.errors().is_empty());
}

#[test]
fn visit_signature_is_visit_item_of_a_nested_function() {
    // The signature is analyzed like the one of a function which is nested in the scope: the
    // generics are local, and the bounds and the types are those of the signature.
    let item: ItemFn = parse_quote!(
        fn f<'a, T: ::p::Bound<Rel>>(x: &'a Rel2<T>, y: Rel3) -> Rel4<'a, T> {
            loop {}
        }
    );
    let by_signature = {
        let mut leaker = Leaker::new();
        leaker.enter_generics(&Generics::default(), |view| view.visit_signature(&item.sig));
        leaker.result().finish().unwrap()
    };
    let by_item = {
        let mut leaker = Leaker::new();
        leaker.enter_generics(&Generics::default(), |view| {
            view.visit_item(&Item::Fn(item.clone()))
        });
        leaker.result().finish().unwrap()
    };
    assert_eq!(by_signature, by_item);
    assert_eq!(by_signature.iter().count(), 4);
    // A relative bound is an error, like the bounds of a method.
    let bad: Signature = parse_quote!(fn f<T: Relative>(x: T));
    let mut leaker = Leaker::new();
    leaker.enter_generics(&Generics::default(), |view| view.visit_signature(&bad));
    let result = leaker.result();
    let err = result.errors();
    assert_eq!(messages(err), "use absolute path: 'Relative'");
    // The receiver is `Self`, which the scope has to declare.
    let method: Signature = parse_quote!(fn f(&self) -> Rel);
    let mut leaker = Leaker::new();
    leaker.enter_generics(&Generics::default(), |view| view.visit_signature(&method));
    let result = leaker.result();
    let err = result.errors();
    assert_self_without_meaning(err);
    let mut leaker = Leaker::new();
    leaker.enter_trait(&ident("Tr"), &Generics::default(), |view| {
        view.visit_signature(&method)
    });
    assert_eq!(leaker.result().finish().unwrap().iter().count(), 1);
}

#[test]
fn the_signature_of_a_foreign_function_declares_nothing() {
    // It is not a method: the lifetime of its signature is not declared, so it is not checked for
    // E0195 (as `Visit::visit_signature()` does for a method).
    let item: ItemMod = parse_quote!(
        mod m {
            extern "C" {
                fn f<'b>(x: ::p::W<'b, dyn RelTr>) -> &'b ();
            }
        }
    );
    let mut leaker = Leaker::new();
    leaker.intern_mod(&item);
    assert_eq!(
        entries(&leaker.result().finish().unwrap()),
        [e(":: p :: W < 'b , dyn RelTr >", ""), e("& 'b ()", "")]
    );
}

// ── `Reason::SelfWithoutMeaning` ─────────────────────────────────────────────────────────────

#[test]
fn self_without_meaning_is_an_error_of_its_own() {
    let mut leaker = Leaker::new();
    leaker.enter_generics(&Generics::default(), |view| {
        view.visit_type(&parse_quote!(Rel<Self>));
        view.visit_type(&parse_quote!(Wrap<Self::Item>));
    });
    let result = leaker.result();
    let err = result.errors();
    assert_eq!(selves_of(err), ["Self", "Self :: Item"]);
    // One error per mention, which says what is wrong, and why.
    assert_eq!(
        messages(err),
        "`Self` has no meaning here\n`Self` has no meaning here"
    );
    assert_eq!(
        err[0].notes.first().map(|(_, n)| n.as_str()),
        Some(
            "`Self` has no meaning here: it is declared only by a struct, an enum, a union, an impl \
             or a trait, not by a module or generics"
        )
    );
    let converted = to_syn_error(err);
    assert_eq!(converted.into_iter().count(), 2);
    assert_eq!(
        err.iter()
            .map(ErrorItem::to_compile_error)
            .collect::<proc_macro2::TokenStream>()
            .to_string(),
        to_syn_error(err).to_compile_error().to_string()
    );
    let _ = err[0].span();
    // A scope which has both kinds of errors returns both, in the order in which they were found:
    // the mention of `Self` in `Rel<Self>` is before the conflict of `Bad<'_>`, and after the one
    // of `Bad0<'_>`.
    let mut leaker = Leaker::new();
    leaker.enter_generics(&Generics::default(), |view| {
        view.visit_type(&parse_quote!(Bad0<'_>));
        view.visit_type(&parse_quote!(Rel<Self>));
        view.visit_type(&parse_quote!(Bad<'_>));
        view.visit_type(&parse_quote!(Wrap<Self::Item>));
    });
    let result = leaker.result();
    let err = result.errors();
    assert_eq!(
        err.iter()
            .map(|item| (item.reason, s(&item.cause)))
            .collect::<Vec<_>>(),
        [
            (Reason::RelativePath, "Bad0 < '_ >".to_string()),
            (Reason::SelfWithoutMeaning, "Self".to_string()),
            (Reason::RelativePath, "Bad < '_ >".to_string()),
            (Reason::SelfWithoutMeaning, "Self :: Item".to_string()),
        ]
    );
}

// ── the root of a trait bound ────────────────────────────────────────────────────────────────

#[test]
fn visit_path_at_the_root_analyzes_the_arguments_and_does_not_check_the_path() {
    // `Visit::visit_path()` outside a type: the path is not leaked, the types in its arguments are.
    let mut leaker = Leaker::new();
    leaker.enter_generics(&parse_quote!(<T>), |view| {
        view.visit_path(&parse_quote!(::p::Tr<Rel<T>>));
        view.visit_path(&parse_quote!(::core::marker::Copy));
        view.visit_path(&parse_quote!(::p::Tr<Rel2>));
    });
    assert_eq!(
        entries(&leaker.result().finish().unwrap()),
        [e("Rel < T >", "< T >"), e("Rel2", "")]
    );
    // It does not check it: a relative path, or `Self` where it has no meaning, is no error.
    let mut leaker = Leaker::new();
    leaker.enter_generics(&Generics::default(), |view| {
        view.visit_path(&parse_quote!(Tr<Rel>));
        view.visit_path(&parse_quote!(m::Tr<Rel2>));
        view.visit_path(&parse_quote!(self::Tr));
        view.visit_path(&parse_quote!(Self::X));
    });
    assert_eq!(
        entries(&leaker.result().finish().unwrap()),
        [e("Rel", ""), e("Rel2", "")]
    );
    let mut leaker = Leaker::new();
    leaker.enter_trait(&ident("Tr"), &Generics::default(), |view| {
        view.visit_path(&parse_quote!(Self::X))
    });
    assert!(leaker.result().finish().unwrap().is_empty());
}

#[test]
fn visit_type_param_bound_checks_the_root_of_a_trait_bound() {
    // A portable path is fine, and its arguments are analyzed.
    let mut leaker = Leaker::new();
    leaker.enter_generics(&parse_quote!(<T>), |view| {
        view.visit_type_param_bound(&parse_quote!(::p::Tr<Rel<T>>));
        view.visit_type_param_bound(&parse_quote!(::core::marker::Copy));
        view.visit_type_param_bound(&parse_quote!(::p::Tr<Rel2>));
    });
    assert_eq!(
        entries(&leaker.result().finish().unwrap()),
        [e("Rel < T >", "< T >"), e("Rel2", "")]
    );
    // A relative path is not, whatever it is.
    for path in [
        parse_quote!(Tr),
        parse_quote!(Tr<Rel>),
        parse_quote!(m::Tr),
        parse_quote!(self::Tr),
        parse_quote!(super::Tr),
    ] {
        let path: Path = path;
        let mut leaker = Leaker::new();
        leaker.enter_generics(&Generics::default(), |view| {
            view.visit_type_param_bound(&parse_quote!(#path))
        });
        let result = leaker.result();
        let err = result.errors();
        // The error is about the path of the trait, at the root.
        assert_eq!(
            messages(err),
            format!("use absolute path: '{}'", s(&path).replace(' ', "")),
            "{}",
            s(&path)
        );
        // The bound has no entry: the scope keeps the context of its generics only.
        if path.segments.last().unwrap().arguments.is_none() {
            assert!(result.into_referrer().is_empty(), "{}", s(&path));
        }
    }
    // `Self` at the root is an error where it has no meaning, and in the scope of an item it is a
    // path of the type context (a trait is not an associated type).
    let mut leaker = Leaker::new();
    leaker.enter_generics(&Generics::default(), |view| {
        view.visit_type_param_bound(&parse_quote!(Self::X))
    });
    let result = leaker.result();
    let err = result.errors();
    assert_self_without_meaning(err);
    let mut leaker = Leaker::new();
    leaker.enter_trait(&ident("Tr"), &Generics::default(), |view| {
        view.visit_type_param_bound(&parse_quote!(Self::X))
    });
    let result = leaker.result();
    let err = result.errors();
    assert!(messages(err).starts_with("use absolute path: 'Self::X'"));
}

#[test]
fn an_impl_checks_the_path_of_its_trait_on_entering_the_scope() {
    // The path of the trait of an impl is checked in the same way.
    let item: ItemImpl = parse_quote!(impl Tr for W {});
    let mut leaker = Leaker::new();
    leaker.intern_impl(&item);
    let result = leaker.result();
    let err = result.errors();
    assert_eq!(messages(err), "use absolute path: 'Tr'");
    let item: ItemImpl = parse_quote!(impl ::p::Tr for W {});
    let mut leaker = Leaker::new();
    leaker.intern_impl(&item);
    let result = leaker.result();
    assert!(result.finish().is_ok());
    // The check comes first, before what is visited of the impl, and the arguments of the trait
    // are analyzed once (`Rel` is one entry).
    let item: ItemImpl = parse_quote!(
        impl<T: Rel3> m::Tr<Rel> for W<T> {
            type A = Rel4<'_>;
        }
    );
    let mut leaker = Leaker::new();
    leaker.intern_impl(&item);
    let result = leaker.result();
    let err = result.errors();
    assert_eq!(
        messages(err),
        "use absolute path: 'm::Tr<Rel>'\nuse absolute path: 'Rel3'\n\
         use absolute path: 'Rel4<'_>'"
    );
    let item: ItemImpl = parse_quote!(impl ::p::Tr<Rel> for () {});
    let mut leaker = Leaker::new();
    leaker.intern_impl(&item);
    assert_eq!(entries(&leaker.result().finish().unwrap()), [e("Rel", "")]);
    // `enter_impl()` checks the trait it is given: the caller visits the arguments of the path.
    let item: ItemImpl = parse_quote!(impl Tr<Rel> for W {});
    let (_, trait_path, _) = item.trait_.as_ref().unwrap();
    let mut leaker = Leaker::new();
    leaker.enter_impl(&item.generics, &item.self_ty, Some(trait_path), |view| {
        view.visit_path(trait_path)
    });
    let result = leaker.result();
    let err = result.errors();
    assert_eq!(messages(err), "use absolute path: 'Tr<Rel>'");
    // Without a trait (an inherent impl), the caller visits the bound to check the path.
    let mut leaker = Leaker::new();
    leaker.enter_impl(&item.generics, &item.self_ty, None, |view| {
        view.visit_type_param_bound(&parse_quote!(#trait_path))
    });
    let result = leaker.result();
    let err = result.errors();
    assert_eq!(messages(err), "use absolute path: 'Tr<Rel>'");
    // The same path visited as `Visit::visit_path()` is not checked.
    let mut leaker = Leaker::new();
    leaker.enter_impl(&item.generics, &item.self_ty, None, |view| {
        view.visit_path(trait_path)
    });
    assert_eq!(entries(&leaker.result().finish().unwrap()), [e("Rel", "")]);
}

// ── the layers of the item scopes ────────────────────────────────────────────────────────────
//
// `enter_type()`, `enter_impl()`, `enter_trait()` and `enter_mod()` are the scope of the item (or
// of the module) around `enter_generics()` of its generics: what they return, record and roll back
// is what `enter_generics()` does, plus what the item declares (`Self`, or the module). The
// expected values below were taken from the scopes before they were composed that way, so they pin
// what has not changed: the ids (the same contexts, and no other one), the entries, the panics and
// what `Self` means. A scope that finds errors keeps its corrected result, and its errors are
// recorded once, in place, in the list of the outermost call (`Leaker::result()`).

/// The kinds of the scopes with a layer around their generics, and the module.
const KINDS: [&str; 6] = ["struct", "enum", "union", "impl", "trait", "mod"];

/// The kinds which declare `Self` (a module does not, and has no generics).
const ITEMS: [&str; 5] = ["struct", "enum", "union", "impl", "trait"];

/// The self type of an impl with the generics `generics`: `::p::W` with their parameters as
/// arguments.
fn self_type(generics: &Generics) -> Type {
    let args: Vec<String> = generics
        .params
        .iter()
        .map(|param| match param {
            GenericParam::Lifetime(param) => s(&param.lifetime),
            GenericParam::Type(param) => param.ident.to_string(),
            GenericParam::Const(param) => param.ident.to_string(),
        })
        .collect();
    if args.is_empty() {
        parse_quote!(::p::W)
    } else {
        parse_str(&format!("::p::W<{}>", args.join(", "))).unwrap()
    }
}

/// Enters the scope of `$kind` (one of [`KINDS`]) with the generics `$generics` on `$on`, which is
/// a `Leaker` or a `View`: the item is `S`, and the self type of the impl is that of
/// [`self_type()`], which is of the trait `::p::Tr` (its methods are compared with a trait's). A module has no generics: it is entered as `m`.
macro_rules! enter {
    ($on:expr, $kind:expr, $generics:expr, $f:expr) => {{
        let generics: &Generics = $generics;
        match $kind {
            "struct" | "enum" | "union" => $on.enter_type(&ident("S"), generics, $f),
            "impl" => {
                let trait_: Path = parse_quote!(::p::Tr);
                $on.enter_impl(generics, &self_type(generics), Some(&trait_), $f)
            }
            "trait" => $on.enter_trait(&ident("S"), generics, $f),
            "mod" => $on.enter_mod("m", [], $f),
            kind => panic!("{}", kind),
        }
    }};
}

/// The errors of a call, as a string: `ok` if there are none.
fn outcome(errors: &[ErrorItem]) -> String {
    if errors.is_empty() {
        "ok".to_string()
    } else if errors
        .iter()
        .all(|item| item.reason == Reason::SelfWithoutMeaning)
    {
        format!(
            "Self without meaning: {}",
            errors
                .iter()
                .map(|item| item.cause.to_token_stream().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )
    } else {
        messages(errors).replace('\n', " / ")
    }
}

/// Everything the Referrer says of the contexts of the entries and of the entries, with the ids
/// the calls returned. A context that no entry has (a module's own, or a scope's with no entry in
/// it) is not seen, but for its id, which the ids of the others skip, and for the hint and the path
/// of a module, which the entries in it tell.
fn dump(result: &LeakerResult, ids: &[(&str, ContextId)]) -> String {
    let referrer = &result.clone().into_referrer();
    let mut lines = Vec::new();
    for (name, id) in ids {
        lines.push(format!("id {} = {}", name, id.index()));
    }
    let entry_ids = common::serialized_ids(referrer).entries;
    for context in common::context_ids(referrer) {
        let ix = match entry_ids.iter().position(|entry| *entry == context) {
            Some(ix) => ix,
            None => continue,
        };
        let found = common::entry(referrer, ix);
        let self_ty = match found.self_ty.as_ref() {
            None => "-".to_string(),
            Some(SelfRequirement::Bound(path)) => format!("Self: {}", s(path)),
            Some(SelfRequirement::Type(ty)) => format!("Self = {}", s(ty)),
            Some(other) => format!("{:?}", other),
        };
        let module = match common::in_module(found) {
            None => "-".to_string(),
            Some(module) => module.index().to_string(),
        };
        let info = match (
            common::entry_module_hint(result, ix),
            common::entry_module_path(result, ix),
        ) {
            (None, None) => String::new(),
            (hint, path) => format!(
                " {:?} {:?}",
                hint,
                path.map(|p| p.iter().map(|i| i.to_string()).collect::<Vec<_>>())
            ),
        };
        lines.push(format!(
            "context {}: {} ; {} ; in module {}{}",
            context.index(),
            s(&found.generics),
            self_ty,
            module,
            info
        ));
    }
    for (ix, ty) in common::types(referrer).enumerate() {
        lines.push(format!(
            "entry {}: {} in {}",
            ix,
            s(ty),
            common::entry_id(referrer, ix)
        ));
    }
    lines.join("\n")
}

/// Asserts that `actual` is `expected` (see the module).
fn pin(name: &str, actual: &str, expected: &str) {
    assert_eq!(actual, expected.trim(), "{}", name);
}

// ── what a scope returns and records ─────────────────────────────────────────────────────────

/// What a scope of an item returns and records when it is the outermost one.
fn root_item(kind: &str) -> String {
    let generics: Generics = parse_quote!(<'a, T: ::p::Tr<Self>, U = Rel0>);
    let mut leaker = Leaker::new();
    let body = |view: &mut View| {
        view.visit_type(&parse_quote!(Rel<Self, T>));
        view.visit_type(&parse_quote!(Rel2<&'a U>));
        view.visit_type(&parse_quote!(Rel3));
        view.visit_type_param_bound(&parse_quote!(::p::Tr<Rel4>));
        view.visit_where_clause(&parse_quote!(where T: ::p::Tr2<Rel5>));
        7
    };
    let (id, value) = enter!(leaker, kind, &generics, body);
    assert_eq!(value, 7);
    let result = leaker.result();
    result.clone().finish().unwrap();
    dump(&result, &[("top", id)])
}

#[test]
fn the_outermost_scope_of_an_item_returns_the_generics_context() {
    // A struct, an enum and a union declare the same `Self`, and the same contexts.
    for kind in ["struct", "enum", "union"] {
        pin(&format!("root_item {}", kind), &root_item(kind), ROOT_ITEM);
    }
    pin("root_item impl", &root_item("impl"), ROOT_ITEM_IMPL);
    pin("root_item trait", &root_item("trait"), ROOT_ITEM_TRAIT);
}

const ROOT_ITEM: &str = r#"
id top = 0
context 0: < 'a , T : :: p :: Tr < Self > , U > ; Self = S < 'a , T , U > ; in module -
context 1: < 'a , U : 'a > ; - ; in module -
context 2:  ; - ; in module -
entry 0: Rel < Self , T > in 0
entry 1: Rel2 < & 'a U > in 1
entry 2: Rel0 in 2
entry 3: Rel3 in 2
entry 4: Rel4 in 2
entry 5: Rel5 in 2
"#;
const ROOT_ITEM_IMPL: &str = r#"
id top = 0
context 0: < 'a , T : :: p :: Tr < Self > , U > ; Self = :: p :: W < 'a , T , U > ; in module -
context 1: < 'a , U : 'a > ; - ; in module -
context 2:  ; - ; in module -
entry 0: Rel < Self , T > in 0
entry 1: Rel2 < & 'a U > in 1
entry 2: Rel0 in 2
entry 3: Rel3 in 2
entry 4: Rel4 in 2
entry 5: Rel5 in 2
"#;
const ROOT_ITEM_TRAIT: &str = r#"
id top = 0
context 0: < 'a , T : :: p :: Tr < Self > , U > ; Self: S < 'a , T , U > ; in module -
context 1: < 'a , U : 'a > ; - ; in module -
context 2:  ; - ; in module -
entry 0: Rel < Self , T > in 0
entry 1: Rel2 < & 'a U > in 1
entry 2: Rel0 in 2
entry 3: Rel3 in 2
entry 4: Rel4 in 2
entry 5: Rel5 in 2
"#;

/// The ids of several scopes in one: the contexts of the generics are those of the generics, and
/// the items and the modules add none, except `Self` if an entry of the scope names it.
fn nested_ids() -> String {
    let mut leaker = Leaker::new();
    let s_ident = ident("S");
    let mut ids = Vec::new();
    let (top, ()) = leaker.enter_generics(&parse_quote!(<X>), |view| {
        let generics: Generics = parse_quote!(<U>);
        let (a, ()) = view.enter_type(&s_ident, &generics, |view| {
            view.visit_type(&parse_quote!(Rel<Self, U>))
        });
        // The same generics, but another `Self`, which an entry names: another context. (`d`
        // below has no entry that names `Self`, so its context is that of the generics.)
        let (b, ()) = view.enter_type(&ident("E"), &generics, |view| {
            view.visit_type(&parse_quote!(Rel<Self, U>))
        });
        let (c, ()) = view.enter_type(&ident("Un"), &parse_quote!(<'a>), |view| {
            view.visit_type(&parse_quote!(Rel2<&'a ()>))
        });
        let (m, inner) = view.enter_mod("m", [ident("S")], |view| {
            view.visit_type(&parse_quote!(S<Rel3>));
            view.enter_type(&ident("S2"), &parse_quote!(<V>), |view| {
                view.visit_type(&parse_quote!(Rel4<Self, V>))
            })
            .0
        });
        let (m2, ()) = view.enter_mod("m2", [], |view| view.visit_type(&parse_quote!(Rel3)));
        // After the modules, the same generics are still the same context.
        let (d, ()) = view.enter_type(&s_ident, &generics, |_| ());
        let (e, ()) = view.enter_impl(&generics, &parse_quote!(::p::Wrap<U>), None, |view| {
            view.visit_type(&parse_quote!(Rel6<Self, U>))
        });
        let (f, ()) = view.enter_trait(&ident("Tr"), &generics, |view| {
            view.visit_type(&parse_quote!(Rel7<U>));
            // A trait inside a module inside a trait.
            view.enter_mod("m4", [], |view| {
                view.enter_trait(&ident("Tr2"), &parse_quote!(<W>), |view| {
                    view.visit_type(&parse_quote!(Rel8<Self, W>))
                });
            });
        });
        // A module has no generics of its own: the ones around stay in scope.
        let (m3, ()) = view.enter_mod("m3", [], |view| view.visit_type(&parse_quote!(Rel5<X>)));
        ids.extend([
            ("a", a),
            ("b", b),
            ("c", c),
            ("m", m),
            ("inner", inner),
            ("m2", m2),
            ("d", d),
            ("e", e),
            ("f", f),
            ("m3", m3),
        ]);
    });
    ids.push(("top", top));
    let result = leaker.result();
    result.clone().finish().unwrap();
    dump(&result, &ids)
}

const NESTED_IDS: &str = r#"
id a = 0
id b = 1
id c = 2
id m = 3
id inner = 4
id m2 = 5
id d = 6
id e = 7
id f = 10
id m3 = 11
id top = 15
context 0: < U > ; Self = S < U > ; in module -
context 1: < U > ; Self = E < U > ; in module -
context 2: < 'a > ; - ; in module -
context 4: < V > ; Self = S2 < V > ; in module 3 Some("m") None
context 6: < U > ; - ; in module -
context 7: < U > ; Self = :: p :: Wrap < U > ; in module -
context 9: < W > ; Self: Tr2 < W > ; in module 8 Some("m4") None
context 12:  ; - ; in module 3 Some("m") None
context 13:  ; - ; in module 5 Some("m2") None
context 14: < X > ; - ; in module 11 Some("m3") None
entry 0: Rel < Self , U > in 0
entry 1: Rel < Self , U > in 1
entry 2: Rel2 < & 'a () > in 2
entry 3: Rel4 < Self , V > in 4
entry 4: Rel7 < U > in 6
entry 5: Rel6 < Self , U > in 7
entry 6: Rel8 < Self , W > in 9
entry 7: Rel3 in 12
entry 8: Rel3 in 13
entry 9: Rel5 < X > in 14
"#;

#[test]
fn nested_scopes_of_items_and_modules_take_the_same_ids() {
    pin("nested_ids", &nested_ids(), NESTED_IDS);
}

// ── a scope which fails keeps its corrected result ───────────────────────────────────────────

/// The generics of a scope whose default is not internable (an error found when the scope is left).
fn bad_generics() -> Generics {
    parse_quote!(<A, B = Bad<'_, A>>)
}

/// Accepts the causes of the errors of the scope which fails in `failing()`.
fn accept_failing(result: LeakerResult) -> LeakerResult {
    result
        .accept_cause(&Cause::Type(parse_quote!(Bad1<'_>)))
        .accept_cause(&Cause::Type(parse_quote!(Bad<'_, A>)))
        // A module declares no `Self`.
        .accept_cause(&Cause::Type(parse_quote!(Self)))
}

/// Enters a module with a scope around which succeeds, one which does not if `fail` (`kind` is the
/// scope), and one which succeeds. Returns the result, and the ids `outer`, `first` and `last`.
fn failing(kind: &str, fail: bool) -> (LeakerResult, [(&'static str, ContextId); 3]) {
    let mut leaker = Leaker::new();
    let (outer, (first, last)) = leaker.enter_mod("outer", [], |view| {
        view.visit_type(&parse_quote!(Rel));
        let (first, ()) = view.enter_type(&ident("F"), &parse_quote!(<A>), |view| {
            view.visit_type(&parse_quote!(RelA<Self, A>));
        });
        if fail {
            // A module, which takes an id, a scope which succeeds, a type which does not, and
            // the generics, whose default does not either (a module has no generics: only the
            // type fails it).
            let body = |view: &mut View| {
                view.visit_type(&parse_quote!(RelU<Self, A>));
                view.enter_mod("lost", [], |view| view.visit_type(&parse_quote!(RelM)));
                view.enter_type(&ident("Lost"), &parse_quote!(<W>), |_| ());
                view.visit_type(&parse_quote!(Bad1<'_>));
            };
            enter!(view, kind, &bad_generics(), body);
        }
        let (last, ()) = view.enter_type(&ident("L"), &parse_quote!(<C>), |view| {
            view.visit_type(&parse_quote!(RelC<Self, C>));
            view.enter_mod("kept", [], |view| view.visit_type(&parse_quote!(RelK)));
        });
        (first, last)
    });
    (
        leaker.result(),
        [("outer", outer), ("first", first), ("last", last)],
    )
}

#[test]
fn a_failing_scope_of_an_item_or_a_module_keeps_its_corrected_result() {
    let (result, ids) = failing("struct", false);
    assert!(result.errors().is_empty());
    let ok = dump(&result, &ids);
    pin("failing ok", &ok, FAILING_OK);
    let mut errors = Vec::new();
    let mut kept = String::new();
    for kind in KINDS {
        let (result, ids) = failing(kind, true);
        let left = dump(&result, &ids);
        if kind == "struct" {
            kept = left.clone();
        }
        // The scope keeps what it recorded, and the ids it took (the module, the contexts of the
        // types and of the generics). It is more than there is without the scope. Accepting the
        // causes of its errors afterwards drops them and changes nothing else.
        let accepted = accept_failing(result.clone());
        assert!(accepted.errors().is_empty(), "{}", kind);
        assert_eq!(dump(&accepted, &ids), left, "{}", kind);
        assert_ne!(left, ok, "{}", kind);
        errors.push(messages(result.errors()));
    }
    // The errors are returned once: the type, and the default of the generics (a module has no
    // generics).
    pin("failing errors", &errors.join("\n--\n"), FAILING_ERRORS);
    pin("failing kept", &kept, FAILING_KEPT);
}

const FAILING_KEPT: &str = r#"
id outer = 0
id first = 1
id last = 6
context 1: < A > ; Self = F < A > ; in module 0 Some("outer") None
context 4: < A , B > ; Self = S < A , B > ; in module 0 Some("outer") None
context 6: < C > ; Self = L < C > ; in module 0 Some("outer") None
context 7:  ; - ; in module 0 Some("outer") None
context 8:  ; - ; in module 2 Some("lost") None
context 9:  ; - ; in module 5 Some("kept") None
entry 0: RelA < Self , A > in 1
entry 1: RelU < Self , A > in 4
entry 2: RelC < Self , C > in 6
entry 3: Rel in 7
entry 4: RelM in 8
entry 5: RelK in 9
"#;

const FAILING_OK: &str = r#"
id outer = 0
id first = 1
id last = 3
context 1: < A > ; Self = F < A > ; in module 0 Some("outer") None
context 3: < C > ; Self = L < C > ; in module 0 Some("outer") None
context 4:  ; - ; in module 0 Some("outer") None
context 5:  ; - ; in module 2 Some("kept") None
entry 0: RelA < Self , A > in 1
entry 1: RelC < Self , C > in 3
entry 2: Rel in 4
entry 3: RelK in 5
"#;
const FAILING_ERRORS: &str = r#"
use absolute path: 'Bad1<'_>'
use absolute path: 'Bad<'_, A>'
--
use absolute path: 'Bad1<'_>'
use absolute path: 'Bad<'_, A>'
--
use absolute path: 'Bad1<'_>'
use absolute path: 'Bad<'_, A>'
--
use absolute path: 'Bad1<'_>'
use absolute path: 'Bad<'_, A>'
--
use absolute path: 'Bad1<'_>'
use absolute path: 'Bad<'_, A>'
--
`Self` has no meaning here
use absolute path: 'Bad1<'_>'
"#;

/// A failing outermost scope keeps its corrected result: the ids and the Referrer are the same
/// whatever is accepted afterwards. The errors of the calls are in one list, in the order of the
/// calls.
#[test]
fn a_failing_outermost_scope_of_each_kind_keeps_its_corrected_result() {
    let mut leaker = Leaker::new();
    let mut ids = Vec::new();
    for kind in KINDS {
        let body = |view: &mut View| {
            view.visit_type(&parse_quote!(RelKept<Self, A>));
            view.enter_mod("inner", [], |view| view.visit_type(&parse_quote!(RelM)));
            view.visit_type(&parse_quote!(Bad1<'_>));
        };
        ids.push(enter!(leaker, kind, &bad_generics(), body).0.index());
    }
    // The module which is entered after them is the one after theirs.
    let (module, ()) = leaker.enter_mod("later", [], |_| ());
    let result = leaker.result();
    let tokens = result.clone().into_referrer().to_token_stream().to_string();
    // Accepting the causes drops the errors only.
    let accepted = accept_failing(result.clone());
    assert!(accepted.errors().is_empty());
    assert_eq!(
        accepted.into_referrer().to_token_stream().to_string(),
        tokens
    );
    // The ids and the module are those of the corrected results, whatever is accepted afterwards.
    assert_eq!(ids, [1, 1, 1, 8, 11, 13]);
    assert_eq!(module.index(), 17);
    pin(
        "outermost errors",
        &messages(result.errors()),
        OUTERMOST_ERRORS,
    );
}

const OUTERMOST_ERRORS: &str = r#"
use absolute path: 'Bad1<'_>'
use absolute path: 'Bad<'_, A>'
use absolute path: 'Bad1<'_>'
use absolute path: 'Bad<'_, A>'
use absolute path: 'Bad1<'_>'
use absolute path: 'Bad<'_, A>'
use absolute path: 'Bad1<'_>'
use absolute path: 'Bad<'_, A>'
use absolute path: 'Bad1<'_>'
use absolute path: 'Bad<'_, A>'
`Self` has no meaning here
use absolute path: 'Bad1<'_>'
"#;

// ── a panic in a scope ───────────────────────────────────────────────────────────────────────

#[test]
fn a_panic_in_an_outermost_scope_leaves_no_trace() {
    let reference = {
        let mut leaker = Leaker::new();
        leaker.enter_mod("kept", [], |_| ());
        leaker
            .result()
            .finish()
            .unwrap()
            .to_token_stream()
            .to_string()
    };
    let mut leaker = Leaker::new();
    for kind in KINDS {
        caught(|| {
            let body = |view: &mut View| {
                view.visit_type(&parse_quote!(RelLost<Self, T>));
                view.enter_mod("lost", [], |_| ());
                view.enter_type(&ident("Lost"), &parse_quote!(<W>), |_| ());
                panic!("a closure panics");
            };
            let _ = enter!(leaker, kind, &parse_quote!(<T>), body);
        });
    }
    let (module, ()) = leaker.enter_mod("kept", [], |_| ());
    assert_eq!(module.index(), 0);
    assert_eq!(
        leaker
            .result()
            .finish()
            .unwrap()
            .to_token_stream()
            .to_string(),
        reference
    );
}

/// A scope of kind `kind` which panics (if `panic`) in a scope of the kind `outer` (which is not
/// a module), which catches it.
fn panicking(outer: &str, kind: &str, panic: bool, mention_self: bool) -> String {
    let mut leaker = Leaker::new();
    let body = |view: &mut View| {
        view.visit_type(&parse_quote!(Rel<T>));
        if panic {
            caught(|| {
                let body = |view: &mut View| {
                    view.visit_type(&parse_quote!(RelU<Self, U>));
                    view.enter_mod("lost", [parse_quote!(Lost)], |_| ());
                    view.visit_type(&parse_quote!(Bad<'_>));
                    panic!("a closure panics");
                };
                let _ = enter!(view, kind, &parse_quote!(<U>), body);
            });
        }
        // `U` is not a generic any more, `Lost` is not declared, `Self` is the one of the outer
        // scope again.
        view.visit_type(&parse_quote!(Wrap<U, Lost>));
        if mention_self {
            view.visit_type(&parse_quote!(Wrap2<Self>));
        }
        view.enter_mod("kept", [], |view| view.visit_type(&parse_quote!(RelK)))
            .0
    };
    let (top, module) = enter!(leaker, outer, &parse_quote!(<T>), body);
    let result = leaker.result();
    let errors = result.errors().to_vec();
    let dumped = dump(&result, &[("top", top), ("module", module)]);
    if errors.is_empty() {
        dumped
    } else {
        // The scope keeps the corrected result. Its context is the one of the scope around, which
        // is not the same for every kind: the entries are.
        let entries: Vec<&str> = dumped
            .lines()
            .filter(|line| line.starts_with("entry "))
            .collect();
        format!("{}\n--\n{}", messages(&errors), entries.join("\n"))
    }
}

#[test]
fn a_panic_caught_in_a_scope_restores_the_scope_around() {
    for mention_self in [false, true] {
        // Whatever the scope around declares (`Self`), it is as if the scope which panicked had
        // not been entered.
        let mut first = None;
        for outer in ITEMS {
            let expected = panicking(outer, "struct", false, mention_self);
            for kind in KINDS {
                assert_eq!(
                    panicking(outer, kind, true, mention_self),
                    expected,
                    "{} {} {}",
                    outer,
                    kind,
                    mention_self
                );
            }
            if mention_self {
                // The context of `Wrap2<Self>` says what `Self` is in the scope around.
                let this = match outer {
                    "impl" => "Self = :: p :: W < T >",
                    "trait" => "Self: S < T >",
                    _ => "Self = S < T >",
                };
                pin(
                    &format!("panicking {}, mention_self", outer),
                    &expected,
                    &PANICKING_SELF.replace("{self}", this),
                );
            } else {
                assert_eq!(
                    *first.get_or_insert(expected.clone()),
                    expected,
                    "{}",
                    outer
                );
            }
        }
        if !mention_self {
            pin("panicking", &first.unwrap(), PANICKING);
        }
    }
}

const PANICKING: &str = r#"
id top = 1
id module = 0
context 1: < T > ; - ; in module -
context 2:  ; - ; in module -
context 3:  ; - ; in module 0 Some("kept") None
entry 0: Rel < T > in 1
entry 1: Wrap < U , Lost > in 2
entry 2: RelK in 3
"#;
const PANICKING_SELF: &str = r#"
id top = 3
id module = 0
context 1: < T > ; - ; in module -
context 2:  ; - ; in module -
context 3: < T > ; {self} ; in module -
context 4:  ; - ; in module 0 Some("kept") None
entry 0: Rel < T > in 1
entry 1: Wrap < U , Lost > in 2
entry 2: Wrap2 < Self > in 3
entry 3: RelK in 4
"#;

// ── the generics are declared once ───────────────────────────────────────────────────────────

#[test]
fn a_scope_of_an_item_or_a_module_adds_no_scope_of_its_own() {
    // A function which is the analyzed item is visited in the scope of its generics (`as_item()`),
    // whose generics are those of the Leaker if it is the outermost one: a module has none of its
    // own, and an item is the scope of its generics. The same function nested in a scope has local
    // generics (`nested()`). It is a method of a trait there: the copy of a function item may have
    // `'b` early-bound, and only that of a method is an error for a lifetime that no argument
    // constrains (E0195, F22), which shows that `'b` is local.
    let sig: Signature = parse_quote!(fn f<'b>(x: ::p::W<'b, dyn RelTr>) -> &'b ());
    let plain: Signature = parse_quote!(fn g(x: ::p::W<dyn RelTr>));
    let as_item = |v: &mut View<'_, '_>, sig: &Signature| visit_function(v, sig);
    let nested = |v: &mut View<'_, '_>, sig: &Signature| {
        v.visit_item(&parse_quote!(
            trait Nested {
                #sig;
            }
        ));
    };
    let b: Generics = parse_quote!(<'b>);
    let none = Generics::default();
    let mut lines = Vec::new();
    let mut check = |name: String, run: &dyn Fn(&mut Leaker)| {
        let mut leaker = Leaker::new();
        run(&mut leaker);
        let result = leaker.result();
        let outcome = outcome(result.errors());
        let count = result.into_referrer().iter().count();
        lines.push(format!("{}: {} ({})", name, outcome, count));
    };
    for kind in ["struct", "impl", "trait"] {
        check(format!("{} of the generics", kind), &|l| {
            enter!(l, kind, &b, |v| as_item(v, &sig));
        });
        check(format!("{} of others", kind), &|l| {
            enter!(l, kind, &none, |v| nested(v, &sig));
        });
        check(format!("{}, mod", kind), &|l| {
            enter!(l, kind, &b, |v| {
                v.enter_mod("m", [], |v| as_item(v, &sig));
            });
        });
        check(format!("generics, {} of the generics", kind), &|l| {
            l.enter_generics(&b, |v| {
                enter!(v, kind, &b, |v| nested(v, &sig));
            });
        });
        check(format!("mod, {} of the generics", kind), &|l| {
            l.enter_mod("m", [], |v| {
                enter!(v, kind, &b, |v| nested(v, &sig));
            });
        });
        check(format!("{}, {}, mod", kind, kind), &|l| {
            enter!(l, kind, &b, |v| {
                enter!(v, kind, &b, |v| {
                    v.enter_mod("m", [], |v| nested(v, &sig));
                });
            });
        });
    }
    check("mod".to_string(), &|l| {
        l.enter_mod("m", [], |v| nested(v, &sig));
    });
    check("mod, no generics".to_string(), &|l| {
        l.enter_mod("m", [], |v| as_item(v, &plain));
    });
    check("generics, mod".to_string(), &|l| {
        l.enter_generics(&b, |v| {
            v.enter_mod("m", [], |v| as_item(v, &sig));
        });
    });
    check("mod, mod, no generics".to_string(), &|l| {
        l.enter_mod("m", [], |v| {
            v.enter_mod("n", [], |v| as_item(v, &plain));
        });
    });
    check("mod, generics".to_string(), &|l| {
        l.enter_mod("m", [], |v| {
            v.enter_generics(&b, |v| nested(v, &sig));
        });
    });
    pin("signatures", &lines.join("\n"), SIGNATURES);
}

const SIGNATURES: &str = r#"
struct of the generics: ok (1)
struct of others: `RelTr` leaves the lifetime `'b` of the method unconstrained in the copy (E0195) (0)
struct, mod: ok (1)
generics, struct of the generics: `RelTr` leaves the lifetime `'b` of the method unconstrained in the copy (E0195) (0)
mod, struct of the generics: `RelTr` leaves the lifetime `'b` of the method unconstrained in the copy (E0195) (0)
struct, struct, mod: `RelTr` leaves the lifetime `'b` of the method unconstrained in the copy (E0195) (0)
impl of the generics: ok (1)
impl of others: `RelTr` leaves the lifetime `'b` of the method unconstrained in the copy (E0195) (0)
impl, mod: ok (1)
generics, impl of the generics: `RelTr` leaves the lifetime `'b` of the method unconstrained in the copy (E0195) (0)
mod, impl of the generics: `RelTr` leaves the lifetime `'b` of the method unconstrained in the copy (E0195) (0)
impl, impl, mod: `RelTr` leaves the lifetime `'b` of the method unconstrained in the copy (E0195) (0)
trait of the generics: ok (1)
trait of others: `RelTr` leaves the lifetime `'b` of the method unconstrained in the copy (E0195) (0)
trait, mod: ok (1)
generics, trait of the generics: `RelTr` leaves the lifetime `'b` of the method unconstrained in the copy (E0195) (0)
mod, trait of the generics: `RelTr` leaves the lifetime `'b` of the method unconstrained in the copy (E0195) (0)
trait, trait, mod: `RelTr` leaves the lifetime `'b` of the method unconstrained in the copy (E0195) (0)
mod: `RelTr` leaves the lifetime `'b` of the method unconstrained in the copy (E0195) (0)
mod, no generics: ok (1)
generics, mod: ok (1)
mod, mod, no generics: ok (1)
mod, generics: `RelTr` leaves the lifetime `'b` of the method unconstrained in the copy (E0195) (0)
"#;

// ── `Self` ───────────────────────────────────────────────────────────────────────────────────

#[test]
fn self_in_the_generics_of_a_scope() {
    let mut lines = Vec::new();
    let mut check = |name: String, run: &dyn Fn(&mut Leaker)| {
        let mut leaker = Leaker::new();
        run(&mut leaker);
        let result = leaker.result();
        let outcome = outcome(result.errors());
        let count = result.into_referrer().iter().count();
        lines.push(format!("{}: {} ({})", name, outcome, count));
    };
    let none = Generics::default();
    let mut bound: Generics = parse_quote!(<'a, T: ::p::Tr<Self>>);
    bound.where_clause = Some(parse_quote!(where Self: 'a));
    let default: Generics = parse_quote!(<T = Rel<Self>>);
    // It is not where nothing declares it.
    check("generics bound".to_string(), &|l| {
        l.enter_generics(&bound, |_| ());
    });
    check("generics default".to_string(), &|l| {
        l.enter_generics(&default, |_| ());
    });
    check("mod".to_string(), &|l| {
        l.enter_mod("m", [], |v| v.visit_type(&parse_quote!(Rel<Self>)));
    });
    check("mod, generics".to_string(), &|l| {
        l.enter_mod("m", [], |v| {
            v.enter_generics(&bound, |_| ());
        });
    });
    for kind in ["struct", "impl", "trait"] {
        // `Self` is declared by the item: it may be in its generics.
        check(format!("{} bound", kind), &|l| {
            enter!(l, kind, &bound, |_| ());
        });
        check(format!("{} default", kind), &|l| {
            enter!(l, kind, &default, |_| ());
        });
        check(format!("mod, {}", kind), &|l| {
            l.enter_mod("m", [], |v| {
                enter!(v, kind, &bound, |v| {
                    v.visit_type(&parse_quote!(Rel<Self>))
                });
            });
        });
        // In a scope nested in one which declares it, it is what it is there.
        check(format!("{}, generics", kind), &|l| {
            enter!(l, kind, &none, |v| {
                v.enter_generics(&bound, |v| v.visit_type(&parse_quote!(Rel<Self>)));
            });
        });
        check(format!("{}, mod, generics", kind), &|l| {
            enter!(l, kind, &none, |v| {
                v.enter_mod("m", [], |v| {
                    v.enter_generics(&bound, |_| ());
                });
            });
        });
        // The `Self` of the scope around is restored, and a module or an item sets its own.
        check(format!("{} in {}", kind, kind), &|l| {
            enter!(l, kind, &none, |v| {
                enter!(v, kind, &none, |v| {
                    v.visit_type(&parse_quote!(Rel<Self>))
                });
                enter!(v, kind, &none, |v| {
                    v.visit_type(&parse_quote!(Rel2<Self>))
                });
            });
        });
        check(format!("{}, then", kind), &|l| {
            enter!(l, kind, &none, |v| {
                enter!(v, kind, &none, |_| ());
                v.enter_mod("m", [], |_| ());
                v.visit_type(&parse_quote!(Rel<Self>))
            });
        });
        check(format!("{}, mod, item", kind), &|l| {
            enter!(l, kind, &none, |v| {
                v.enter_mod("m", [], |v| {
                    v.visit_item(&parse_quote!(
                        struct S {
                            a: Rel<Self>,
                        }
                    ));
                });
            });
        });
        check(format!("{}, item", kind), &|l| {
            enter!(l, kind, &none, |v| {
                v.visit_item(&parse_quote!(
                    struct S {
                        a: Rel<Self>,
                    }
                ));
            });
        });
    }
    pin("self", &lines.join("\n"), SELF);
}

const SELF: &str = r#"
generics bound: Self without meaning: Self, Self (0)
generics default: Self without meaning: Self (0)
mod: Self without meaning: Self (0)
mod, generics: Self without meaning: Self, Self (0)
struct bound: ok (0)
struct default: `Rel<Self>` is in the default of a parameter, and names `Self` (0)
mod, struct: ok (1)
struct, generics: ok (1)
struct, mod, generics: Self without meaning: Self, Self (0)
struct in struct: ok (2)
struct, then: ok (1)
struct, mod, item: ok (1)
struct, item: ok (1)
impl bound: ok (0)
impl default: `Rel<Self>` is in the default of a parameter, and names `Self` (0)
mod, impl: ok (1)
impl, generics: ok (1)
impl, mod, generics: Self without meaning: Self, Self (0)
impl in impl: ok (2)
impl, then: ok (1)
impl, mod, item: ok (1)
impl, item: ok (1)
trait bound: ok (0)
trait default: `Rel<Self>` is in the default of a parameter, and names `Self` (0)
mod, trait: ok (1)
trait, generics: ok (1)
trait, mod, generics: Self without meaning: Self, Self (0)
trait in trait: ok (2)
trait, then: ok (1)
trait, mod, item: ok (1)
trait, item: ok (1)
"#;

// ── the View forwards what the analysis overrides ────────────────────────────────────────────

/// The `Visit` methods of the type `ty` in `source`: written out, or named in the `forward!`
/// list.
fn visit_methods(source: &str, ty: &str) -> std::collections::BTreeSet<String> {
    use proc_macro2::{TokenStream, TokenTree};
    use std::collections::BTreeSet;

    fn idents(tokens: TokenStream, out: &mut BTreeSet<String>) {
        for token in tokens {
            match token {
                TokenTree::Ident(ident) if ident.to_string().starts_with("visit_") => {
                    out.insert(ident.to_string());
                }
                TokenTree::Group(group) => idents(group.stream(), out),
                _ => (),
            }
        }
    }
    let file = parse_file(source).unwrap();
    let mut out = BTreeSet::new();
    for item in file.items {
        let item = match item {
            Item::Impl(item) => item,
            _ => continue,
        };
        let is_visit = matches!(&item.trait_, Some((_, path, _))
            if path.segments.last().map_or(false, |seg| seg.ident == "Visit"));
        let is_ty = matches!(&*item.self_ty, Type::Path(path)
            if path.path.segments.last().map_or(false, |seg| seg.ident == ty));
        if !is_visit || !is_ty {
            continue;
        }
        for item in item.items {
            match item {
                ImplItem::Fn(method) => {
                    out.insert(method.sig.ident.to_string());
                }
                ImplItem::Macro(mac) => idents(mac.mac.tokens, &mut out),
                _ => (),
            }
        }
    }
    out
}

/// The view must forward exactly the methods the analysis overrides: a missing one would bypass
/// the analysis silently, and an extra one is a stale entry.
#[test]
fn the_view_forwards_exactly_the_methods_the_analysis_overrides() {
    let overridden = visit_methods(include_str!("../src/analyze.rs"), "AnalyzeVisitor");
    let forwarded = visit_methods(include_str!("../src/view.rs"), "View");
    assert!(!overridden.is_empty(), "the analysis was not found");
    assert!(
        !forwarded.is_empty(),
        "the forwarded methods were not found"
    );
    let missing: Vec<_> = overridden.difference(&forwarded).collect();
    let stale: Vec<_> = forwarded.difference(&overridden).collect();
    assert!(
        missing.is_empty(),
        "overridden by the analysis but not forwarded by the view: {:?}",
        missing
    );
    assert!(
        stale.is_empty(),
        "forwarded by the view but not overridden by the analysis: {:?}",
        stale
    );
}

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
    // The top-level context has the generics as declared, and takes its id after those of the
    // entries: the entry that uses all of them has a context of its own.
    let all = common::types(&referrer)
        .position(|ty| s(ty) == "Rel4 < T , U >")
        .unwrap();
    assert_eq!(context, id(4));
    assert_eq!(common::entry_id(&referrer, all), id(3));
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
        // The top-level context has the generics of the item as declared and, as an entry names
        // `Self`, what `Self` is: it is a context of its own, after the entry's.
        assert_eq!(common::entry_id(&referrer, 0), id(0));
        assert_eq!(context, id(1), "{}", kind);
        assert!(
            referrer.to_token_stream().to_string().ends_with(&format!(
                "declared < 'a , T > ; Self = {} < 'a , T > ; ()",
                name
            )),
            "{}",
            kind
        );
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
    // The top-level context has the generics of the impl as declared and what `Self` is, and is a
    // context of its own: not that of `Rel<Self>` (0), nor that of `Rel2<T>` (1).
    assert_eq!(common::entry_id(&referrer, 0), id(0));
    assert_eq!(common::entry_id(&referrer, 1), id(1));
    assert_eq!(context, id(2));
    assert!(referrer
        .to_token_stream()
        .to_string()
        .ends_with("declared < T > ; Self = :: p :: Wrap < T > ; ()"));
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
    // The generics of an item visited in a scope are local generics: `intern_*()` makes the
    // generics of the item those of the Leaker.
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
    // The top-level context has no generics here, whereas that of `intern_struct()` has them (as
    // declared), and is a context of its own, not that of the entry.
    let mut leaker = Leaker::new();
    let top = leaker.intern_struct(&item);
    let referrer = leaker.result().finish().unwrap();
    assert_eq!(top, id(1));
    assert_eq!(common::entry_id(&referrer, 0), id(0));
    assert_eq!(s(&common::entry(&referrer, 0).generics), "< T >");
}

// ── `Self` ───────────────────────────────────────────────────────────────────────────────────

#[test]
fn self_has_no_meaning_in_generics_and_in_the_items_that_do_not_declare_it() {
    let ty: Type = parse_quote!(Rel<Self>);
    let mut leaker = Leaker::new();
    leaker.enter_generics(&parse_quote!(<T>), |view| view.visit_type(&ty));
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
    assert_eq!(messages(err), ["`Self` has no meaning here"; 5].join("\n"));
    assert_eq!(
        selves_of(err),
        ["Self", "Self", "Self :: Item", "Self", "Self"]
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

    // The same in a type alias and in a function that a scope without generics visits (not in the
    // items which declare it).
    let nested = |items: Vec<Item>| {
        recorded(|leaker| {
            leaker.enter_generics(&Generics::default(), |view| {
                for item in &items {
                    view.visit_item(item);
                }
            });
        })
    };
    let result = nested(vec![parse_quote!(
        pub type A = Rel<Self>;
    )]);
    assert_eq!(messages(result.errors()), "`Self` has no meaning here");
    let result = nested(vec![parse_quote!(
        pub fn f() -> Rel<Self> {
            loop {}
        }
    )]);
    assert!(result.finish().is_err());
    let result = nested(vec![
        parse_quote!(
            pub struct S {
                a: Rel<Self>,
            }
        ),
        parse_quote!(
            pub trait Tr {
                fn f(&self) -> Rel2<Self>;
            }
        ),
        parse_quote!(
            pub struct W;
        ),
        parse_quote!(
            impl W {
                pub fn f(&self) -> Rel3<Self> {
                    loop {}
                }
            }
        ),
    ]);
    // Each `Self` is the one of its item, and the self type of the impl is an entry too.
    assert_eq!(
        entries(&result.finish().unwrap()),
        [
            e("Rel < Self >", ""),
            e("Rel2 < Self >", ""),
            e("W", ""),
            e("Rel3 < Self >", "")
        ]
    );
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
            // A scope in it, which takes an id, and two errors.
            view.enter_generics(&parse_quote!(<U>), |view| {
                view.visit_type(&parse_quote!(RelU<U>));
                view.enter_generics(&parse_quote!(<M>), |view| {
                    view.visit_type(&parse_quote!(RelM<M>))
                });
                view.visit_type(&parse_quote!(Bad1<'_>));
                view.visit_type(&parse_quote!(Bad2<'_, U>));
            });
        }
        let (last, ()) = view.enter_generics(&parse_quote!(<C>), |view| {
            view.visit_type(&parse_quote!(RelC<C>));
            view.enter_generics(&parse_quote!(<K>), |view| {
                view.visit_type(&parse_quote!(RelK<K>))
            });
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
    // What it recorded is kept: the entries and the contexts, and so are the ids that the scopes
    // returned.
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
    // The ids which were returned are valid, and are the generics of their scopes as declared:
    // contexts of their own, not those of the entries that use the same generics.
    let text = failed_referrer.to_token_stream().to_string();
    for (id, ty, generics) in [
        (failed.first, "RelA < A >", "< A >"),
        (failed.last, "RelC < C >", "< C >"),
    ] {
        let ix = common::types(&failed_referrer)
            .position(|found| s(found) == ty)
            .unwrap();
        assert!(
            common::context_ids(&failed_referrer).contains(&id),
            "{}",
            ty
        );
        assert_ne!(common::entry_id(&failed_referrer, ix), id, "{}", ty);
        assert!(
            text.contains(&format!("declared {} ;", generics)),
            "{}: {}",
            ty,
            text
        );
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
    assert!(
        kept_types.contains(&"RelM < M >".to_string()),
        "{:?}",
        kept_types
    );
    assert!(common::types(&failed_referrer).any(|ty| s(ty) == "RelK < K >"));
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
    // The scope's top-level context is a context of its own: it took its id when the scope was
    // left, before the context of the entry `Rel<T>`, which the outermost call records.
    assert_eq!(inner.unwrap(), id(0));
    assert_eq!(common::entry_id(&referrer, 0), id(1));
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
    // The outermost context, the failed scope, and its inner scopes: the ids returned; the
    // contexts of the two entries, which are not those of the scopes that made them; and the
    // context of `Rel2<'_>`, which is copied as written and is a part, as its error is about it.
    assert_eq!(common::context_ids(&referrer).len(), 7);
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
    leaker.enter_generics(&parse_quote!(<A>), |view| {
        view.visit_type(&parse_quote!(Rel))
    });
    // Scopes entered, ids taken, and types found by a call which fails.
    let (scope, ()) = leaker.enter_generics(&parse_quote!(<B>), |view| {
        view.enter_generics(&parse_quote!(<C>), |view| {
            view.visit_type(&parse_quote!(RelKept))
        });
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
    // The calls took the ids they take if the cause is accepted, and the next ones follow them: the
    // entries share the context 0, then come the scopes `<A>`, `<C>` and `<B>` (an inner scope is
    // left first), and the struct `<T>`.
    assert_eq!((scope, item), (id(3), id(4)));
    // The entries of the calls are kept, and so are the scopes they entered.
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
    // Nothing of the scope that panicked is left, not its context either: the contexts are
    // that of `Kept` and the outer scope's own.
    assert!(!common::types(&referrer).any(|ty| s(ty).contains("RelLost") || s(ty).contains("Bad")));
    assert_eq!(common::context_ids(&referrer), [id(0), id(1)]);
}

// ── the Referrer ─────────────────────────────────────────────────────────────────────────────

#[test]
fn the_format_of_the_previous_version_is_rejected() {
    assert_eq!(Referrer::FORMAT_VERSION, 16);
    // What each format changed is told above `Referrer::FORMAT_VERSION`.
    for version in 1..Referrer::FORMAT_VERSION {
        let old = "2 < T > ; () < T > ; (Rel)";
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
    leaker.enter_generics(&Generics::default(), |view| visit_all(view, &types));
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
    // The nested scope declares a generic and `Self`, enters a scope, and records types, a
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
                        view.enter_generics(&Generics::default(), |_| ());
                        panic!("a closure panics");
                    });
                });
            }
            // `U` is not a generic any more, and the scope is not entered.
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
    // The top-level context is the generics as declared: it is not the context of the entries that
    // use the same generics, and it comes after the context of `Wrap<U>`.
    assert_eq!(common::entry_id(&referrer, 0), id(0));
    assert_eq!(common::entry_id(&referrer, 2), id(1));
    assert_eq!(top, id(2));
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
fn a_panic_caught_in_the_outermost_scope_of_a_generics_scope_keeps_the_top_level_id() {
    // The top-level context of the outer scope is its generics as declared, not those of the
    // nested one which panicked, and the nested one took no id.
    let mut leaker = Leaker::new();
    let (top, ()) = leaker.enter_generics(&parse_quote!(<T>), |view| {
        caught(|| {
            let _ = view.enter_generics(&Generics::default(), |_| panic!("a closure panics"));
        });
        view.visit_type(&parse_quote!(Rel<T>));
    });
    let referrer = leaker.result().finish().unwrap();
    assert_eq!(common::entry_id(&referrer, 0), id(0));
    assert_eq!(top, id(1));
    assert_eq!(common::context_ids(&referrer), [id(0), id(1)]);
    assert_eq!(s(&common::entry(&referrer, 0).generics), "< T >");
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

/// What a scope that panics leaves behind, if the table of contexts were not rolled back: nested
/// scopes with their own contexts.
fn record_and_panic(view: &mut View<'_, '_>, message: &str) {
    view.visit_type(&parse_quote!(RelLost<V>));
    view.enter_generics(&Generics::default(), |_| ());
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
            let _ = leaker.enter_generics(&Generics::default(), |view| {
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
            .enter_generics(&Generics::default(), |view| {
                view.visit_type(&parse_quote!(Rel4))
            })
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
    assert_eq!(ids, [id(1), id(3), id(5), id(7), id(1)]);
    // Each of the first four is the generics of its call as declared, taken after the context of
    // the entry that uses the same generics (0, 2, 4 and 6).
    let at = |text: &str| {
        common::types(&referrer)
            .position(|ty| s(ty) == text)
            .unwrap()
    };
    for (top, entry, ty, generics) in [
        (ids[0], 0, "Rel < T >", "< T >"),
        (ids[1], 2, "Rel2 < U >", "< U >"),
        (ids[2], 4, "Rel3 < W >", "< W >"),
        (ids[3], 6, "Rel4", ""),
    ] {
        assert_eq!(common::entry_id(&referrer, at(ty)), id(entry), "{}", ty);
        assert_eq!(top, id(entry + 1), "{}", ty);
        assert_eq!(s(&common::entry(&referrer, at(ty)).generics), generics);
    }
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
                view.enter_generics(&Generics::default(), |_| ());
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
    // Each call takes the id of its generics after the context of its entry.
    assert_eq!(ids, [id(1), id(3), id(5), id(7)]);
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
        // In a scope nested in one which has no `Self` either, with a type visited.
        let result = recorded(|leaker| {
            leaker.enter_generics(&Generics::default(), |view| {
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
            leaker.enter_generics(&Generics::default(), |view| {
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
        leaker.enter_generics(&Generics::default(), |view| {
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
        // In a scope of a type: nested in generics, directly, and inside several of them.
        let mut leaker = Leaker::new();
        leaker.enter_type(&ident("S"), &Generics::default(), |view| {
            view.enter_generics(&Generics::default(), |view| view.visit_item(item));
            view.visit_item(item);
            view.enter_generics(&Generics::default(), |view| {
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
    // The scope's top-level context is the generics as declared: a context of its own, after the
    // entry's.
    assert_eq!(common::entry_id(&referrer, 0), id(0));
    assert_eq!(top, id(1));
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
    let item: Item = parse_quote!(
        extern "C" {
            fn f<'b>(x: ::p::W<'b, dyn RelTr>) -> &'b ();
        }
    );
    let mut leaker = Leaker::new();
    leaker.enter_generics(&Generics::default(), |view| view.visit_item(&item));
    assert_eq!(
        entries(&leaker.result().finish().unwrap()),
        [e("& 'b ()", ""), e(":: p :: W < 'b , dyn RelTr >", "")]
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
// `enter_type()`, `enter_impl()` and `enter_trait()` are the scope of the item around
// `enter_generics()` of its generics: what they return, record and roll back is what
// `enter_generics()` does, plus what the item declares (`Self`). The
// expected values below were taken from the scopes before they were composed that way, so they pin
// what has not changed: the entries, the panics and what `Self` means. The ids moved when the
// generics of a scope as declared became a context of their own (numbered when the scope is left:
// a nested one before the contexts of the entries, the outermost one after them). A scope that
// finds errors keeps its corrected result, and its errors are recorded once, in place, in the list
// of the outermost call (`Leaker::result()`).

/// The kinds of the scopes with a layer around their generics, and the generics alone.
const KINDS: [&str; 6] = ["struct", "enum", "union", "impl", "trait", "generics"];

/// The kinds which declare `Self` (the generics alone do not).
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
/// [`self_type()`], which is of the trait `::p::Tr` (its methods are compared with a trait's).
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
            "generics" => $on.enter_generics(generics, $f),
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
/// the calls returned. A context that no entry has (a scope's with no entry in it) is not seen, but
/// for its id, which the ids of the others skip.
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
        lines.push(format!(
            "context {}: {} ; {}",
            context.index(),
            s(&found.generics),
            self_ty
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
    let referrer = result.clone().finish().unwrap();
    // The top-level context has no entry, so its declaration is read from the last group.
    let text = referrer.to_token_stream().to_string();
    let declared = &text[text.rfind("declared").unwrap()..];
    format!("{}\ntop {}", dump(&result, &[("top", id)]), declared)
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
id top = 3
context 0: < 'a , T : :: p :: Tr < Self > , U > ; Self = S < 'a , T , U >
context 1: < 'a , U : 'a > ; -
context 2:  ; -
entry 0: Rel < Self , T > in 0
entry 1: Rel2 < & 'a U > in 1
entry 2: Rel0 in 2
entry 3: Rel3 in 2
entry 4: Rel4 in 2
entry 5: Rel5 in 2
top declared < 'a , T : :: p :: Tr < Self > , U = Rel0 > ; Self = S < 'a , T , U > ; ()
"#;
const ROOT_ITEM_IMPL: &str = r#"
id top = 3
context 0: < 'a , T : :: p :: Tr < Self > , U > ; Self = :: p :: W < 'a , T , U >
context 1: < 'a , U : 'a > ; -
context 2:  ; -
entry 0: Rel < Self , T > in 0
entry 1: Rel2 < & 'a U > in 1
entry 2: Rel0 in 2
entry 3: Rel3 in 2
entry 4: Rel4 in 2
entry 5: Rel5 in 2
top declared < 'a , T : :: p :: Tr < Self > , U = Rel0 > ; Self = :: p :: W < 'a , T , U > ; ()
"#;
const ROOT_ITEM_TRAIT: &str = r#"
id top = 3
context 0: < 'a , T : :: p :: Tr < Self > , U > ; Self: S < 'a , T , U >
context 1: < 'a , U : 'a > ; -
context 2:  ; -
entry 0: Rel < Self , T > in 0
entry 1: Rel2 < & 'a U > in 1
entry 2: Rel0 in 2
entry 3: Rel3 in 2
entry 4: Rel4 in 2
entry 5: Rel5 in 2
top declared < 'a , T : :: p :: Tr < Self > , U = Rel0 > ; Self : S < 'a , T , U > ; ()
"#;

/// The ids of several scopes in one: the generics of a scope as declared are a context of its own,
/// numbered when the scope is left; the contexts of the entries (with `Self` if the entry names
/// it) are numbered after them, when the outermost call is recorded, and the outermost scope takes
/// the last id.
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
        // below has no entry that names `Self`, so its context has none either.)
        let (b, ()) = view.enter_type(&ident("E"), &generics, |view| {
            view.visit_type(&parse_quote!(Rel<Self, U>))
        });
        let (c, ()) = view.enter_type(&ident("Un"), &parse_quote!(<'a>), |view| {
            view.visit_type(&parse_quote!(Rel2<&'a ()>))
        });
        let (m, inner) = view.enter_generics(&Generics::default(), |view| {
            view.visit_type(&parse_quote!(Rel3));
            view.enter_type(&ident("S2"), &parse_quote!(<V>), |view| {
                view.visit_type(&parse_quote!(Rel4<Self, V>))
            })
            .0
        });
        let (m2, ()) = view.enter_generics(&Generics::default(), |view| {
            view.visit_type(&parse_quote!(Rel3))
        });
        // After them, a scope is numbered as before.
        let (d, ()) = view.enter_type(&s_ident, &generics, |_| ());
        let (e, ()) = view.enter_impl(&generics, &parse_quote!(::p::Wrap<U>), None, |view| {
            view.visit_type(&parse_quote!(Rel6<Self, U>))
        });
        let (f, ()) = view.enter_trait(&ident("Tr"), &generics, |view| {
            view.visit_type(&parse_quote!(Rel7<U>));
            // A trait inside a scope inside a trait.
            view.enter_generics(&Generics::default(), |view| {
                view.enter_trait(&ident("Tr2"), &parse_quote!(<W>), |view| {
                    view.visit_type(&parse_quote!(Rel8<Self, W>))
                });
            });
        });
        // A scope without generics of its own: the ones around stay in scope.
        let (m3, ()) = view.enter_generics(&Generics::default(), |view| {
            view.visit_type(&parse_quote!(Rel5<X>))
        });
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
id m = 4
id inner = 3
id m2 = 4
id d = 5
id e = 6
id f = 9
id m3 = 4
id top = 19
context 10: < U > ; Self = S < U >
context 11: < U > ; Self = E < U >
context 12: < 'a > ; -
context 13:  ; -
context 14: < V > ; Self = S2 < V >
context 15: < U > ; Self = :: p :: Wrap < U >
context 16: < U > ; -
context 17: < W > ; Self: Tr2 < W >
context 18: < X > ; -
entry 0: Rel < Self , U > in 10
entry 1: Rel < Self , U > in 11
entry 2: Rel2 < & 'a () > in 12
entry 3: Rel3 in 13
entry 4: Rel4 < Self , V > in 14
entry 5: Rel6 < Self , U > in 15
entry 6: Rel7 < U > in 16
entry 7: Rel8 < Self , W > in 17
entry 8: Rel5 < X > in 18
"#;

#[test]
fn nested_scopes_of_items_and_generics_take_the_same_ids() {
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
        // The generics alone declare no `Self`.
        .accept_cause(&Cause::Type(parse_quote!(Self)))
}

/// Enters a scope with a scope around which succeeds, one which does not if `fail` (`kind` is the
/// scope), and one which succeeds. Returns the result, and the ids `outer`, `first` and `last`.
fn failing(kind: &str, fail: bool) -> (LeakerResult, [(&'static str, ContextId); 3]) {
    let mut leaker = Leaker::new();
    let (outer, (first, last)) = leaker.enter_generics(&Generics::default(), |view| {
        view.visit_type(&parse_quote!(Rel));
        let (first, ()) = view.enter_type(&ident("F"), &parse_quote!(<A>), |view| {
            view.visit_type(&parse_quote!(RelA<Self, A>));
        });
        if fail {
            // A scope, which takes an id, a scope which succeeds, a type which does not, and
            // the generics, whose default does not either.
            let body = |view: &mut View| {
                view.visit_type(&parse_quote!(RelU<Self, A>));
                view.enter_generics(&Generics::default(), |view| {
                    view.visit_type(&parse_quote!(RelM))
                });
                view.enter_type(&ident("Lost"), &parse_quote!(<W>), |_| ());
                view.visit_type(&parse_quote!(Bad1<'_>));
            };
            enter!(view, kind, &bad_generics(), body);
        }
        let (last, ()) = view.enter_type(&ident("L"), &parse_quote!(<C>), |view| {
            view.visit_type(&parse_quote!(RelC<Self, C>));
            view.enter_generics(&Generics::default(), |view| {
                view.visit_type(&parse_quote!(RelK))
            });
        });
        (first, last)
    });
    (
        leaker.result(),
        [("outer", outer), ("first", first), ("last", last)],
    )
}

#[test]
fn a_failing_scope_of_an_item_or_generics_keeps_its_corrected_result() {
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
        // The scope keeps what it recorded, and the ids it took (the contexts of the types and of
        // the generics). It is more than there is without the scope. Accepting the
        // causes of its errors afterwards drops them and changes nothing else.
        let accepted = accept_failing(result.clone());
        assert!(accepted.errors().is_empty(), "{}", kind);
        assert_eq!(dump(&accepted, &ids), left, "{}", kind);
        assert_ne!(left, ok, "{}", kind);
        errors.push(messages(result.errors()));
    }
    // The errors are returned once: the type, and the default of the generics.
    pin("failing errors", &errors.join("\n--\n"), FAILING_ERRORS);
    pin("failing kept", &kept, FAILING_KEPT);
}

const FAILING_KEPT: &str = r#"
id outer = 1
id first = 0
id last = 4
context 5:  ; -
context 6: < A > ; Self = F < A >
context 7: < A , B > ; Self = S < A , B >
context 8: < C > ; Self = L < C >
entry 0: Rel in 5
entry 1: RelK in 5
entry 2: RelM in 5
entry 3: RelA < Self , A > in 6
entry 4: RelU < Self , A > in 7
entry 5: RelC < Self , C > in 8
"#;

const FAILING_OK: &str = r#"
id outer = 1
id first = 0
id last = 2
context 3:  ; -
context 4: < A > ; Self = F < A >
context 5: < C > ; Self = L < C >
entry 0: Rel in 3
entry 1: RelK in 3
entry 2: RelA < Self , A > in 4
entry 3: RelC < Self , C > in 5
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
use absolute path: 'Bad<'_, A>'
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
            view.enter_generics(&Generics::default(), |view| {
                view.visit_type(&parse_quote!(RelM))
            });
            view.visit_type(&parse_quote!(Bad1<'_>));
        };
        ids.push(enter!(leaker, kind, &bad_generics(), body).0.index());
    }
    // The scope which is entered after them takes the id after theirs.
    let (later, ()) = leaker.enter_generics(&parse_quote!(<L>), |_| ());
    let result = leaker.result();
    let tokens = result.clone().into_referrer().to_token_stream().to_string();
    // Accepting the causes drops the errors only.
    let accepted = accept_failing(result.clone());
    assert!(accepted.errors().is_empty());
    assert_eq!(
        accepted.into_referrer().to_token_stream().to_string(),
        tokens
    );
    // The ids are those of the corrected results, whatever is accepted afterwards. The first
    // three declare the same generics and `Self`: one top-level context, numbered after the
    // scope inside and the contexts of the entries of the first.
    assert_eq!(ids, [3, 3, 3, 5, 7, 8]);
    assert_eq!(later.index(), 9);
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
use absolute path: 'Bad<'_, A>'
"#;

// ── a panic in a scope ───────────────────────────────────────────────────────────────────────

#[test]
fn a_panic_in_an_outermost_scope_leaves_no_trace() {
    let reference = {
        let mut leaker = Leaker::new();
        leaker.enter_generics(&parse_quote!(<K>), |_| ());
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
                view.enter_generics(&Generics::default(), |_| ());
                view.enter_type(&ident("Lost"), &parse_quote!(<W>), |_| ());
                panic!("a closure panics");
            };
            let _ = enter!(leaker, kind, &parse_quote!(<T>), body);
        });
    }
    let (kept, ()) = leaker.enter_generics(&parse_quote!(<K>), |_| ());
    assert_eq!(kept.index(), 0);
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
/// the generics alone), which catches it.
fn panicking(outer: &str, kind: &str, panic: bool, mention_self: bool) -> String {
    let mut leaker = Leaker::new();
    let body = |view: &mut View| {
        view.visit_type(&parse_quote!(Rel<T>));
        if panic {
            caught(|| {
                let body = |view: &mut View| {
                    view.visit_type(&parse_quote!(RelU<Self, U>));
                    view.enter_generics(&Generics::default(), |_| ());
                    view.visit_type(&parse_quote!(Bad<'_>));
                    panic!("a closure panics");
                };
                let _ = enter!(view, kind, &parse_quote!(<U>), body);
            });
        }
        // `U` is not a generic any more, `Self` is the one of the outer scope again.
        view.visit_type(&parse_quote!(Wrap<U, Lost>));
        if mention_self {
            view.visit_type(&parse_quote!(Wrap2<Self>));
        }
        view.enter_generics(&Generics::default(), |view| {
            view.visit_type(&parse_quote!(RelK))
        })
        .0
    };
    let (top, kept) = enter!(leaker, outer, &parse_quote!(<T>), body);
    let result = leaker.result();
    let errors = result.errors().to_vec();
    let dumped = dump(&result, &[("top", top), ("kept", kept)]);
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
id top = 3
id kept = 0
context 1: < T > ; -
context 2:  ; -
entry 0: Rel < T > in 1
entry 1: RelK in 2
entry 2: Wrap < U , Lost > in 2
"#;
const PANICKING_SELF: &str = r#"
id top = 4
id kept = 0
context 1: < T > ; -
context 2:  ; -
context 3: < T > ; {self}
entry 0: Rel < T > in 1
entry 1: RelK in 2
entry 2: Wrap < U , Lost > in 2
entry 3: Wrap2 < Self > in 3
"#;

// ── the generics are declared once ───────────────────────────────────────────────────────────

#[test]
fn a_scope_of_an_item_adds_no_scope_of_its_own() {
    // A function which is the analyzed item is visited in the scope of its generics (`as_item()`),
    // whose generics are those of the Leaker if it is the outermost one: an item is the scope of
    // its generics. The same function nested in a scope has local generics (`nested()`). It is a method of a trait there: the copy of a function item may have
    // `'b` early-bound, and only that of a method is an error for a lifetime that no argument
    // constrains (E0195, F22), which shows that `'b` is local.
    let sig: Signature = parse_quote!(fn f<'b>(x: ::p::W<'b, dyn RelTr>) -> &'b ());
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
        check(format!("generics, {} of the generics", kind), &|l| {
            l.enter_generics(&b, |v| {
                enter!(v, kind, &b, |v| nested(v, &sig));
            });
        });
    }
    pin("signatures", &lines.join("\n"), SIGNATURES);
}

const SIGNATURES: &str = r#"
struct of the generics: ok (1)
struct of others: `RelTr` leaves the lifetime `'b` of the method unconstrained in the copy (E0195) (0)
generics, struct of the generics: `RelTr` leaves the lifetime `'b` of the method unconstrained in the copy (E0195) (0)
impl of the generics: ok (1)
impl of others: `RelTr` leaves the lifetime `'b` of the method unconstrained in the copy (E0195) (0)
generics, impl of the generics: `RelTr` leaves the lifetime `'b` of the method unconstrained in the copy (E0195) (0)
trait of the generics: ok (1)
trait of others: `RelTr` leaves the lifetime `'b` of the method unconstrained in the copy (E0195) (0)
generics, trait of the generics: `RelTr` leaves the lifetime `'b` of the method unconstrained in the copy (E0195) (0)
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
    for kind in ["struct", "impl", "trait"] {
        // `Self` is declared by the item: it may be in its generics.
        check(format!("{} bound", kind), &|l| {
            enter!(l, kind, &bound, |_| ());
        });
        check(format!("{} default", kind), &|l| {
            enter!(l, kind, &default, |_| ());
        });
        // In a scope nested in one which declares it, it is what it is there.
        check(format!("{}, generics", kind), &|l| {
            enter!(l, kind, &none, |v| {
                v.enter_generics(&bound, |v| v.visit_type(&parse_quote!(Rel<Self>)));
            });
        });
        // The `Self` of the scope around is restored, and an item sets its own.
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
                v.visit_type(&parse_quote!(Rel<Self>))
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
struct bound: ok (0)
struct default: `Rel<Self>` is in the default of a parameter, and names `Self` (0)
struct, generics: ok (1)
struct in struct: ok (2)
struct, then: ok (1)
struct, item: ok (1)
impl bound: ok (0)
impl default: `Rel<Self>` is in the default of a parameter, and names `Self` (0)
impl, generics: ok (1)
impl in impl: ok (2)
impl, then: ok (1)
impl, item: ok (1)
trait bound: ok (0)
trait default: `Rel<Self>` is in the default of a parameter, and names `Self` (0)
trait, generics: ok (1)
trait in trait: ok (2)
trait, then: ok (1)
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

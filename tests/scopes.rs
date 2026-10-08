//! The scopes of the `Leaker` (`enter_*()`) and the `View` they give to their closure: what each
//! scope declares, nested scopes, `Self`, and the rollback of a scope which fails.

use syn::*;
use template_quote::{quote, ToTokens};
use type_leak::{ContextId, Leaker, NotInternableError, Referrer, SelfRequirement, View};

fn s(x: &impl ToTokens) -> String {
    quote!(#x).to_string()
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
        .iter_with_generics()
        .map(|(ty, generics)| (s(ty), s(generics)))
        .collect()
}

fn e(ty: &str, generics: &str) -> (String, String) {
    (ty.to_string(), generics.to_string())
}

fn self_of(referrer: &Referrer, ix: usize) -> String {
    match referrer.context_self(referrer.context(ix)) {
        None => "-".to_string(),
        Some(SelfRequirement::Bound(path)) => format!("Self: {}", s(path)),
        Some(SelfRequirement::Type(ty)) => format!("Self = {}", s(ty)),
        Some(other) => format!("{:?}", other),
    }
}

/// The path of the module of the `ix`-th entry.
fn module_path_of(referrer: &Referrer, ix: usize) -> Option<Vec<String>> {
    let module = referrer.context_module(referrer.context(ix))?;
    Some(
        referrer
            .module_path(module)?
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
    let (context, value) = leaker
        .enter_generics(&generics, |view| {
            view.visit_type(&parse_quote!(Rel<T>));
            view.visit_type(&parse_quote!(Rel2<U>));
            view.visit_type(&parse_quote!(Rel3));
            42
        })
        .unwrap();
    // The value of the closure is returned.
    assert_eq!(value, 42);
    let referrer = leaker.finish();
    assert_eq!(
        entries(&referrer),
        [
            e("Rel < T >", "< T >"),
            e("Rel2 < U >", "< U : :: core :: marker :: Copy >"),
            e("Rel3", ""),
        ]
    );
    // The top-level context is that of the generics, which no entry uses exactly.
    assert_eq!(s(referrer.context_generics(context)), s(&generics));
    assert!(!referrer
        .iter()
        .enumerate()
        .any(|(ix, _)| referrer.context(ix) == context));
    assert!(referrer.contexts().any(|c| c == context));
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
        let result = leaker.enter_type(&ident, &generics, true, |v| v.visit_type(&ty));
        let (context, ()) = result.unwrap();
        let referrer = leaker.finish();
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
        assert_eq!(s(referrer.context_generics(context)), "< 'a , T >");
        assert_eq!(context, referrer.context(0));
        assert_eq!(
            referrer.context_self(context),
            referrer.context_self(referrer.context(0))
        );
        assert!(referrer.context_self(context).is_some());
    }
}

#[test]
fn an_impl_declares_self_as_the_self_type() {
    let generics: Generics = parse_quote!(<T>);
    let mut leaker = Leaker::new();
    let self_ty: Type = parse_quote!(::p::Wrap<T>);
    let (context, ()) = leaker
        .enter_impl(&generics, &self_ty, true, |view| {
            view.visit_type(&parse_quote!(Rel<Self>));
            view.visit_type(&parse_quote!(Rel2<T>));
        })
        .unwrap();
    let referrer = leaker.finish();
    assert_eq!(
        entries(&referrer),
        [e("Rel < Self >", "< T >"), e("Rel2 < T >", "< T >")]
    );
    assert_eq!(self_of(&referrer, 0), "Self = :: p :: Wrap < T >");
    assert_eq!(self_of(&referrer, 1), "-");
    assert_eq!(s(referrer.context_generics(context)), "< T >");
}

#[test]
fn a_trait_declares_self_as_the_implementor() {
    let generics: Generics = parse_quote!(<T, const N: usize>);
    let mut leaker = Leaker::new();
    let (_, ()) = leaker
        .enter_trait(&ident("Tr"), &generics, true, |view| {
            view.visit_type(&parse_quote!(Rel<Self::Item, T>));
        })
        .unwrap();
    let referrer = leaker.finish();
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
    other.intern_trait(&item, true).unwrap();
    let other = other.finish();
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
    let (module, ()) = leaker
        .enter_mod("my_module", [ident("S")], |view| {
            view.visit_item(&Item::Struct(item.clone()))
        })
        .unwrap();
    let referrer = leaker.finish();
    // `S` is declared in the module, so it is not leaked, and nor is `self::S`.
    assert_eq!(
        entries(&referrer),
        [e("Other", ""), e("Rel", ""), e("Rel2", "")]
    );
    for ix in 0..3 {
        assert_eq!(referrer.context_module(referrer.context(ix)), Some(module));
    }
    assert_eq!(referrer.module_hint(module), Some("my_module"));
    assert_eq!(referrer.module_path(module), None);
    // The module is the top-level context, which has no generics.
    assert!(referrer.context_generics(module).params.is_empty());
    assert!(referrer.contexts().any(|c| c == module));
    // The hint is for debugging: any `Display` does.
    let mut leaker = Leaker::new();
    let (other, ()) = leaker.enter_mod(7, [], |_| ()).unwrap();
    assert_eq!(leaker.finish().module_hint(other), Some("7"));
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
    let top = whole.intern_struct(&item, true).unwrap();
    let whole = whole.finish();

    let mut scope = Leaker::new();
    let fields = item.fields.clone();
    let (top_scope, ()) = scope
        .enter_type(&item.ident, &item.generics, true, |view| {
            for field in &fields {
                view.visit_type(&field.ty);
            }
        })
        .unwrap();
    let scope = scope.finish();
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
    leaker
        .enter_generics(&generics, |view| {
            // A type.
            view.visit_type(&parse_quote!(Rel2<T>));
            // The arguments of a path, whose name is resolved elsewhere.
            view.visit_path_arguments(&PathArguments::AngleBracketed(parse_quote!(
                <Rel3, (T, Rel4)>
            )));
            // A path, as the root of a trait bound: it is not leaked, its arguments are.
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
            // A signature, whose generics are local ones as well.
            view.visit_signature(&parse_quote!(fn g<W>(x: Rel10<W>) -> Rel11));
        })
        .unwrap();
    let referrer = leaker.finish();
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
fn a_relative_trait_in_a_bound_is_an_error() {
    let mut leaker = Leaker::new();
    let result = leaker.enter_generics(&Generics::default(), |view| {
        view.visit_type_param_bound(&parse_quote!(Relative));
    });
    let err = result.err().unwrap();
    assert_eq!(err.to_string(), "use absolute path: 'Relative'");
    // Nothing is left of a failed scope.
    assert!(leaker.finish().is_empty());
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
    leaker
        .enter_generics(&Generics::default(), |view| {
            view.visit_item(&Item::Struct(item.clone()))
        })
        .unwrap();
    let referrer = leaker.finish();
    assert_eq!(entries(&referrer), [e("Rel < T >", "< T >")]);
    // The top-level context has no generics here, whereas that of `intern_struct()` has them.
    let mut leaker = Leaker::new();
    let top = leaker.intern_struct(&item, true).unwrap();
    let referrer = leaker.finish();
    assert_eq!(s(referrer.context_generics(top)), "< T >");
    assert_eq!(referrer.context(0), top);
}

// ── `Self` ───────────────────────────────────────────────────────────────────────────────────

#[test]
fn self_has_no_meaning_in_a_module_or_generics() {
    let ty: Type = parse_quote!(Rel<Self>);
    let mut leaker = Leaker::new();
    let err = leaker
        .enter_generics(&parse_quote!(<T>), |view| view.visit_type(&ty))
        .err()
        .unwrap();
    assert_eq!(err.to_string(), "`Self` has no meaning here");
    assert!(
        err.note().unwrap().contains("`Self` has no meaning here"),
        "{:?}",
        err.note()
    );
    let err = leaker
        .enter_mod("m", [], |view| view.visit_type(&ty))
        .err()
        .unwrap();
    assert_eq!(err.to_string(), "`Self` has no meaning here");
    // Not even where it would not be leaked, or in a bound.
    for ty in [
        parse_quote!(Self),
        parse_quote!(Self::Item),
        parse_quote!(&'static Self),
    ] {
        let ty: Type = ty;
        assert!(leaker
            .enter_generics(&Generics::default(), |view| view.visit_type(&ty))
            .is_err());
    }
    assert!(leaker
        .enter_generics(&Generics::default(), |view| {
            view.visit_where_clause(&parse_quote!(where Self: ::core::marker::Sized))
        })
        .is_err());
    // Nothing is left.
    assert!(leaker.finish().is_empty());

    // The same in a module analyzed as a whole, and in a function of it (not in the items which
    // declare it).
    let m: ItemMod = parse_quote!(
        mod m {
            pub type A = Rel<Self>;
        }
    );
    let err = Leaker::new().intern_mod(&m).err().unwrap();
    assert_eq!(err.to_string(), "`Self` has no meaning here");
    let m: ItemMod = parse_quote!(
        mod m {
            pub fn f() -> Rel<Self> {
                loop {}
            }
        }
    );
    assert!(Leaker::new().intern_mod(&m).is_err());
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
    leaker.intern_mod(&m).unwrap();
    assert_eq!(leaker.finish().iter().count(), 3);
}

#[test]
fn self_is_kept_by_the_generics_in_a_scope_which_declares_it() {
    let impl_generics: Generics = parse_quote!(<T>);
    let method_generics: Generics = parse_quote!(<'a, U>);
    let mut leaker = Leaker::new();
    leaker
        .enter_impl(&impl_generics, &parse_quote!(::p::W<T>), true, |view| {
            view.enter_generics(&method_generics, |view| {
                view.visit_type(&parse_quote!(Rel<'a, Self, U>))
            })
            .unwrap();
        })
        .unwrap();
    let referrer = leaker.finish();
    assert_eq!(
        entries(&referrer),
        [e("Rel < 'a , Self , U >", "< 'a , T , U >")]
    );
    assert_eq!(self_of(&referrer, 0), "Self = :: p :: W < T >");

    // A module in a scope is not in that `Self`.
    let mut leaker = Leaker::new();
    let result = leaker.enter_impl(&impl_generics, &parse_quote!(::p::W<T>), true, |view| {
        view.enter_mod("m", [], |view| view.visit_type(&parse_quote!(Rel<Self>)))
    });
    let (_, inner) = result.unwrap();
    assert!(inner.is_err());
}

#[test]
fn allow_self_decides_whether_self_may_be_interned() {
    let generics: Generics = parse_quote!(<T>);
    let ty: Type = parse_quote!(Rel<Self>);
    for allow_self in [true, false] {
        let mut leaker = Leaker::new();
        let result = leaker.enter_type(&ident("S"), &generics, allow_self, |view| {
            view.visit_type(&ty)
        });
        assert_eq!(result.is_ok(), allow_self);
        if let Err(err) = result {
            assert!(err
                .to_string()
                .starts_with("use absolute path: 'Rel < Self >'"));
            assert_eq!(
                err.note().as_deref(),
                Some("it cannot be leaked as a whole because of `Self`")
            );
        }
        // The same for the others which have it.
        let mut leaker = Leaker::new();
        assert_eq!(
            leaker
                .enter_type(&ident("E"), &generics, allow_self, |view| view
                    .visit_type(&ty))
                .is_ok(),
            allow_self
        );
        let mut leaker = Leaker::new();
        assert_eq!(
            leaker
                .enter_type(&ident("U"), &generics, allow_self, |view| view
                    .visit_type(&ty))
                .is_ok(),
            allow_self
        );
        let mut leaker = Leaker::new();
        assert_eq!(
            leaker
                .enter_impl(&generics, &parse_quote!(::p::W<T>), allow_self, |view| {
                    view.visit_type(&ty)
                })
                .is_ok(),
            allow_self
        );
    }
    // A type which can be decomposed is, and the others are not affected.
    let mut leaker = Leaker::new();
    leaker
        .enter_type(&ident("S"), &generics, false, |view| {
            view.visit_type(&parse_quote!((Rel, Self)));
            view.visit_type(&parse_quote!(Rel2<T>));
        })
        .unwrap();
    assert_eq!(
        entries(&leaker.finish()),
        [e("Rel", ""), e("Rel2 < T >", "< T >")]
    );
    // The bounds of the generics a type uses cannot mention `Self` either.
    let bounded: Generics = parse_quote!(<T: ::core::cmp::PartialEq<Self>>);
    let mut leaker = Leaker::new();
    let err = leaker
        .enter_type(&ident("S"), &bounded, false, |view| {
            view.visit_type(&parse_quote!(Wrap<T>))
        })
        .err()
        .unwrap();
    assert_eq!(
        err.note().as_deref(),
        Some("the bounds of its generics mention `Self`, which cannot be leaked")
    );
}

#[test]
fn allow_self_applies_to_the_whole_items() {
    let item: ItemImpl = parse_quote! {
        impl<T> ::p::Tr for ::p::W<T> {
            fn f(&self) -> Rel<Self> { loop {} }
        }
    };
    assert!(Leaker::new().intern_impl(&item, true).is_ok());
    assert!(Leaker::new().intern_impl(&item, false).is_err());
    let item: ItemStruct = parse_quote!(
        struct S<T> {
            a: Rel<Self>,
            b: T,
        }
    );
    assert!(Leaker::new().intern_struct(&item, true).is_ok());
    assert!(Leaker::new().intern_struct(&item, false).is_err());
    let item: ItemEnum = parse_quote!(
        enum E<T> {
            A(Rel<Self>),
            B(T),
        }
    );
    assert!(Leaker::new().intern_enum(&item, true).is_ok());
    assert!(Leaker::new().intern_enum(&item, false).is_err());
    let item: ItemUnion = parse_quote!(union U { a: Rel<Self> });
    assert!(Leaker::new().intern_union(&item, true).is_ok());
    assert!(Leaker::new().intern_union(&item, false).is_err());
}

// ── nested scopes ────────────────────────────────────────────────────────────────────────────

#[test]
fn a_module_in_a_module_in_a_scope() {
    // `enter_mod` > `enter_impl` > `visit_*`: the types are in the module, with the generics of
    // the impl (local ones) and what `Self` is.
    let generics: Generics = parse_quote!(<T>);
    let self_ty: Type = parse_quote!(::p::W<T>);
    let mut leaker = Leaker::new();
    let (outer, inner) = leaker
        .enter_mod("outer", [], |view| {
            view.enter_impl(&generics, &self_ty, true, |view| {
                view.visit_type(&parse_quote!(Rel<T>));
                view.visit_type(&parse_quote!(Rel2<Self>));
                view.visit_type(&parse_quote!(::q::Abs));
            })
            .unwrap();
            // A module in the module.
            view.enter_mod("inner", [], |view| view.visit_type(&parse_quote!(Rel3)))
                .unwrap()
                .0
        })
        .unwrap();
    let referrer = leaker.finish();
    // The type of an external crate is an optional entry, which does not depend on a module.
    let found: Vec<(String, String, String, Option<ContextId>, bool)> =
        (0..referrer.iter().count())
            .map(|ix| {
                (
                    s(referrer.iter().nth(ix).unwrap()),
                    s(referrer.generics(ix)),
                    self_of(&referrer, ix),
                    referrer.context_module(referrer.context(ix)),
                    referrer.must(ix),
                )
            })
            .collect();
    let mut expected = vec![
        ("Rel < T >", "< T >", "-", Some(outer), true),
        (
            "Rel2 < Self >",
            "< T >",
            "Self = :: p :: W < T >",
            Some(outer),
            true,
        ),
        ("Rel3", "", "-", Some(inner), true),
        (":: q :: Abs", "", "-", None, false),
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
        assert_eq!(found.4, expected.4, "{}", found.0);
    }
    // `inner` is entered in `outer`, which is the module of the call.
    assert_eq!(referrer.module_hint(outer), Some("outer"));
    assert_eq!(referrer.module_hint(inner), Some("inner"));
    assert_ne!(outer, inner);
}

#[test]
fn nested_modules_are_numbered_and_kept() {
    let mut leaker = Leaker::new();
    let (outer, (inner, deeper)) = leaker
        .enter_mod("outer", [], |view| {
            view.enter_mod("inner", [], |view| {
                view.enter_mod("deeper", [], |view| view.visit_type(&parse_quote!(Rel)))
                    .unwrap()
                    .0
            })
            .unwrap()
        })
        .unwrap();
    let referrer = leaker.finish();
    // The module of the entry is the innermost, and each is a module of the Referrer.
    assert_eq!(referrer.context_module(referrer.context(0)), Some(deeper));
    assert_eq!(referrer.module_hint(outer), Some("outer"));
    assert_eq!(referrer.module_hint(inner), Some("inner"));
    assert_eq!(referrer.module_hint(deeper), Some("deeper"));
    // A module entered by a scope has a hint, and no path.
    assert_eq!(referrer.module_path(deeper), None);
    // They are numbered in the order they are entered, from the outermost.
    assert_eq!((outer, inner, deeper), (id(0), id(1), id(2)));
}

#[test]
fn a_module_found_in_a_scope_has_a_path_relative_to_it() {
    let m: ItemMod = parse_quote!(
        mod inner {
            pub struct C {
                a: Rel,
            }
        }
    );
    let mut leaker = Leaker::new();
    let (outer, ()) = leaker
        .enter_mod("outer", [], |view| view.visit_item(&Item::Mod(m.clone())))
        .unwrap();
    let referrer = leaker.finish();
    let module = referrer.context_module(referrer.context(0)).unwrap();
    assert_ne!(module, outer);
    assert_eq!(referrer.module_path(module), Some(&[ident("inner")][..]));
    assert_eq!(referrer.module_hint(module), None);
    // The module it is in, whose hint is known, is kept too.
    assert_eq!(referrer.module_hint(outer), Some("outer"));
}

#[test]
fn the_defaults_of_a_nested_scope_are_analyzed_with_the_parameters_before() {
    // As the defaults of the generics of the Leaker are: with the parameters before them as the
    // generics, so they are rewritten on their own where the Referrer omits the argument.
    let mut leaker = Leaker::new();
    let (outer, (inner, ())) = leaker
        .enter_generics(&parse_quote!(<T>), |view| {
            view.enter_generics(&parse_quote!(<A, B = Rel<A>>), |_| ())
                .unwrap()
        })
        .unwrap();
    let referrer = leaker.finish();
    assert_eq!(entries(&referrer), [e("Rel < A >", "< A >")]);
    // The default is not among the generics, which are in the canonical form.
    assert_eq!(s(referrer.context_generics(inner)), "< A , B >");
    assert_eq!(s(referrer.context_generics(outer)), "< T >");
    // The same for the generics of the Leaker.
    let mut other = Leaker::new();
    other
        .enter_generics(&parse_quote!(<A, B = Rel<A>>), |_| ())
        .unwrap();
    assert_eq!(entries(&other.finish()), entries(&referrer));
    // A default which is not internable is an error of the scope which declares it.
    let mut leaker = Leaker::new();
    let (_, inner) = leaker
        .enter_generics(&Generics::default(), |view| {
            view.enter_generics(&parse_quote!(<A, B = Bad<'_, A>>), |_| ())
                .err()
        })
        .unwrap();
    assert!(inner.is_some());
}

// ── rollback ─────────────────────────────────────────────────────────────────────────────────

/// What `enter_generics()` of the two scopes returns, as the ids of the Referrer.
struct Run {
    first: ContextId,
    last: ContextId,
    referrer: Referrer,
    failure: Option<NotInternableError>,
}

/// Enters a scope with `<A>`, which has a scope with `<B>` that succeeds, one with `<U>` which
/// does not if `fail`, and one with `<C>` that succeeds.
fn run(fail: bool) -> Run {
    let outer: Generics = parse_quote!(<T>);
    let mut leaker = Leaker::new();
    let mut failure = None;
    let (_, (first, last)) = leaker
        .enter_generics(&outer, |view| {
            view.visit_type(&parse_quote!(Rel<T>));
            let (first, ()) = view
                .enter_generics(&parse_quote!(<A>), |view| {
                    view.visit_type(&parse_quote!(RelA<A>));
                })
                .unwrap();
            if fail {
                // A module, which takes an id, and two errors.
                failure = view
                    .enter_generics(&parse_quote!(<U>), |view| {
                        view.visit_type(&parse_quote!(RelU<U>));
                        view.enter_mod("lost", [], |view| view.visit_type(&parse_quote!(RelM)))
                            .unwrap();
                        view.visit_type(&parse_quote!(Bad1<'_>));
                        view.visit_type(&parse_quote!(Bad2<'_, U>));
                    })
                    .err();
            }
            let (last, ()) = view
                .enter_generics(&parse_quote!(<C>), |view| {
                    view.visit_type(&parse_quote!(RelC<C>));
                    view.enter_mod("kept", [], |view| view.visit_type(&parse_quote!(RelK)))
                        .unwrap();
                })
                .unwrap();
            view.visit_type(&parse_quote!(Rel2));
            (first, last)
        })
        .unwrap();
    Run {
        first,
        last,
        referrer: leaker.finish(),
        failure,
    }
}

#[test]
fn a_failing_scope_is_rolled_back_and_the_outer_one_goes_on() {
    let ok = run(false);
    let failed = run(true);
    assert!(ok.failure.is_none());
    // The errors of the scope are all returned, with the spans (here, those of the paths).
    let err = failed.failure.unwrap();
    let message = err.to_string();
    assert_eq!(message.lines().count(), 2, "{}", message);
    assert!(message.contains("Bad1"), "{}", message);
    assert!(message.contains("Bad2"), "{}", message);
    // Nothing of it is left: the entries, the contexts, and the modules (which are numbered as if
    // it was not there).
    assert_eq!(entries(&failed.referrer), entries(&ok.referrer));
    assert_eq!(failed.first, ok.first);
    assert_eq!(failed.last, ok.last);
    let ids = |referrer: &Referrer| {
        referrer
            .contexts()
            .map(ContextId::index)
            .collect::<Vec<_>>()
    };
    assert_eq!(ids(&failed.referrer), ids(&ok.referrer));
    assert_eq!(
        failed.referrer.to_token_stream().to_string(),
        ok.referrer.to_token_stream().to_string()
    );
    // The ids which were returned are valid, and are the generics of their scopes.
    assert_eq!(s(failed.referrer.context_generics(failed.first)), "< A >");
    assert_eq!(s(failed.referrer.context_generics(failed.last)), "< C >");
    assert!(!entries(&failed.referrer)
        .iter()
        .any(|(ty, _)| ty.contains("RelU") || ty.contains("RelM")));
    assert!(failed.referrer.iter().any(|ty| s(ty) == "RelK"));
}

#[test]
fn the_error_is_returned_once() {
    // The outer scope does not report the error of the inner one again, and it is not affected:
    // the caller decides whether to propagate it. If it does, the outer scope still succeeds, as it
    // is the closure which returns the error.
    let mut leaker = Leaker::new();
    let result = leaker.enter_generics(&Generics::default(), |view| {
        view.visit_type(&parse_quote!(Rel));
        view.enter_generics(&Generics::default(), |view| {
            view.visit_type(&parse_quote!(Bad<'_>))
        })
        .map(|_| ())
    });
    let (_, inner) = result.unwrap();
    let err = inner.err().unwrap();
    assert_eq!(err.to_string().lines().count(), 1);
    assert_eq!(leaker.finish().iter().count(), 1);

    // If the closure ignores it, the outer scope has no error either.
    let mut leaker = Leaker::new();
    let result = leaker.enter_generics(&Generics::default(), |view| {
        let _ = view.enter_generics(&Generics::default(), |view| {
            view.visit_type(&parse_quote!(Bad<'_>))
        });
    });
    assert!(result.is_ok());
    assert!(leaker.finish().is_empty());

    // An error of the outer scope itself is returned with its own errors, and not those of the
    // scopes which were rolled back.
    let mut leaker = Leaker::new();
    let result = leaker.enter_generics(&Generics::default(), |view| {
        let _ = view.enter_generics(&Generics::default(), |view| {
            view.visit_type(&parse_quote!(Bad1<'_>))
        });
        view.visit_type(&parse_quote!(Bad2<'_>));
    });
    let err = result.err().unwrap();
    assert_eq!(err.to_string().lines().count(), 1);
    assert!(err.to_string().contains("Bad2"));

    // A scope which succeeds inside one which fails is rolled back with it, with its context.
    let mut leaker = Leaker::new();
    let mut inner = None;
    let result = leaker.enter_generics(&Generics::default(), |view| {
        inner = Some(
            view.enter_generics(&parse_quote!(<T>), |view| {
                view.visit_type(&parse_quote!(Rel<T>))
            })
            .unwrap()
            .0,
        );
        view.visit_type(&parse_quote!(Bad<'_>));
    });
    assert!(result.is_err());
    // The Leaker is as new: the next scope takes the first ids.
    let (first, ()) = leaker
        .enter_generics(&parse_quote!(<U>), |view| {
            view.visit_type(&parse_quote!(RelU<U>))
        })
        .unwrap();
    assert_eq!(first, inner.unwrap());
    assert_eq!(first, id(0));
    let referrer = leaker.finish();
    assert_eq!(entries(&referrer), [e("RelU < U >", "< U >")]);
}

#[test]
fn a_failing_outermost_scope_leaves_no_trace() {
    let mut leaker = Leaker::new();
    leaker
        .enter_mod("kept", [], |view| view.visit_type(&parse_quote!(Rel)))
        .unwrap();
    let before = {
        let mut other = Leaker::new();
        other
            .enter_mod("kept", [], |view| view.visit_type(&parse_quote!(Rel)))
            .unwrap();
        other.finish().to_token_stream().to_string()
    };
    // Modules entered, ids taken, and types found by a call which fails.
    assert!(leaker
        .enter_mod("lost", [ident("S")], |view| {
            view.enter_mod("inner", [], |view| view.visit_type(&parse_quote!(RelLost)))
                .unwrap();
            view.visit_type(&parse_quote!(Bad<'_>));
        })
        .is_err());
    assert!(leaker
        .intern_struct(
            &parse_quote!(
                struct S<T> {
                    a: RelLost2,
                    b: Bad<'_>,
                }
            ),
            true
        )
        .is_err());
    assert_eq!(leaker.finish().to_token_stream().to_string(), before);
}

#[test]
fn a_panic_in_a_scope_leaves_no_trace() {
    let mut leaker = Leaker::new();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = leaker.enter_mod("lost", [], |view| {
            view.enter_mod("inner", [], |_| ()).unwrap();
            panic!("a closure panics");
        });
    }));
    assert!(result.is_err());
    // The module is not in the Leaker.
    let (module, ()) = leaker.enter_mod("kept", [], |_| ()).unwrap();
    assert_eq!(module, id(0));
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
    ids.push(leaker.intern_mod(&m).unwrap());
    // The same hint twice.
    for _ in 0..2 {
        let (module, inner) = leaker
            .enter_mod("same", [], |view| {
                view.visit_type(&parse_quote!(Rel));
                view.enter_mod("same", [], |view| view.visit_type(&parse_quote!(Rel)))
                    .unwrap()
                    .0
            })
            .unwrap();
        ids.push(module);
        ids.push(inner);
    }
    ids.push(
        leaker
            .intern_trait(
                &parse_quote!(
                    trait Tr<T> {
                        fn f(&self) -> Wrap<T>;
                    }
                ),
                true,
            )
            .unwrap(),
    );
    (ids, leaker.finish().to_token_stream().to_string())
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
    let (a, b) = leaker
        .enter_mod("a", [], |view| view.enter_mod("b", [], |_| ()).unwrap().0)
        .unwrap();
    assert_eq!(a, id(0));
    assert_eq!(b, id(1));
    let (c, ()) = leaker.enter_mod("c", [], |_| ()).unwrap();
    assert_eq!(c, id(2));
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
    let root = leaker.intern_mod(&m).unwrap();
    let (hinted, ()) = leaker
        .enter_mod("hinted", [], |view| view.visit_type(&parse_quote!(Rel)))
        .unwrap();
    let referrer = leaker.finish();
    let paths: Vec<Option<Vec<String>>> = (0..referrer.iter().count())
        .map(|ix| module_path_of(&referrer, ix))
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
    assert_eq!(referrer.module_path(root), Some(&[][..]));
    assert_eq!(referrer.module_hint(root), None);
    assert_eq!(referrer.module_hint(hinted), Some("hinted"));
    assert_eq!(referrer.module_path(hinted), None);
    // An id which is not a module has neither.
    assert_eq!(referrer.module_path(id(99)), None);
    assert_eq!(referrer.module_hint(id(99)), None);
    let ctx = referrer.context(0);
    assert_ne!(ctx, root);
    assert_eq!(referrer.module_path(ctx), None);
}

#[test]
fn the_visitor_enters_the_module_by_its_id() {
    use syn::visit_mut::VisitMut;
    // The modules `a` and `b` have `Rel` each, entered by `enter_mod()`, and `empty` has none.
    let mut leaker = Leaker::new();
    let (a, ()) = leaker
        .enter_mod("a", [], |view| view.visit_type(&parse_quote!(Rel)))
        .unwrap();
    let (b, ()) = leaker
        .enter_mod("b", [], |view| view.visit_type(&parse_quote!(Rel)))
        .unwrap();
    let (empty, ()) = leaker.enter_mod("empty", [], |_| ()).unwrap();
    let referrer = leaker.finish();
    assert_eq!(referrer.iter().count(), 2);
    let copy = |module: ContextId| {
        let mut visitor = referrer
            .clone()
            .visitor(module, |_, ix| parse_str(&format!("P{}", ix)).unwrap());
        let mut item: ItemStruct = parse_quote!(
            struct C {
                y: Rel,
            }
        );
        visitor.visit_item_struct_mut(&mut item);
        s(&item)
    };
    let in_a = referrer.context_module(referrer.context(0)) == Some(a);
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
    let (outer, ()) = leaker
        .enter_mod("outer", [], |view| {
            view.visit_item(&Item::Mod(inner.clone()))
        })
        .unwrap();
    let referrer = leaker.finish();
    let copy = |item: &ItemMod, module: ContextId| {
        let mut visitor = referrer.clone().visitor(module, |_, _| parse_quote!(P));
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
    let unchanged = s(&inner);
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
    // The visited `mod` item is never the module of the context. `outer` below `outer` is not a
    // module which the Leaker saw, so nothing in it is found, `inner` neither. Neither is `inner`
    // below `inner`, the module of the entry.
    let wrapped =
        |text: &str| s(&parse_str::<ItemMod>(&format!("mod outer {{ {} }}", text)).unwrap());
    let wrap = |item: &ItemMod| -> ItemMod { parse_quote!(mod outer { #item }) };
    assert_eq!(copy(&wrap(&inner), outer), wrapped(&unchanged));
    let inner_id = referrer.context_module(referrer.context(0)).unwrap();
    assert_ne!(inner_id, outer);
    assert_eq!(copy(&inner, inner_id), unchanged);
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
    let root = leaker.intern_mod(&m).unwrap();
    let (hinted, ()) = leaker
        .enter_mod("with \"quotes\"", [], |view| {
            view.visit_type(&parse_quote!(Rel2));
            view.enter_mod("nested", [], |view| view.visit_type(&parse_quote!(Rel3)))
                .unwrap();
        })
        .unwrap();
    let referrer = leaker.finish();
    let tokens = referrer.to_token_stream();
    let parsed: Referrer = parse2(tokens.clone()).unwrap();
    assert_eq!(parsed, referrer);
    assert_eq!(parsed.to_token_stream().to_string(), tokens.to_string());
    // The modules, their paths and hints, and what depends on them are the same.
    assert_eq!(parsed.module_hint(hinted), Some("with \"quotes\""));
    assert_eq!(parsed.module_path(hinted), None);
    for ix in 0..referrer.iter().count() {
        let module = referrer.context_module(referrer.context(ix));
        assert_eq!(parsed.context_module(parsed.context(ix)), module);
        if let Some(module) = module {
            assert_eq!(parsed.module_path(module), referrer.module_path(module));
            assert_eq!(parsed.module_hint(module), referrer.module_hint(module));
        }
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
        .clone()
        .visitor(root, |_, _| parse_quote!(P))
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
    assert_eq!(Referrer::FORMAT_VERSION, 3);
    // The serialized form of a module in the format 1.
    let old: proc_macro2::TokenStream =
        "\"type-leak-format-1\" 2 mod self ; (Rel) mod a :: b ; (Rel)"
            .parse()
            .unwrap();
    let err = parse2::<Referrer>(old).err().unwrap();
    assert!(err.to_string().contains("format 1, expected 3"), "{}", err);
    // The format 2 had the same grammar, but its top-level contexts did not tell `Self` or the
    // module of the scope, which the visitor needs.
    let old: proc_macro2::TokenStream =
        "\"type-leak-format-2\" 2 mod { path = [] } ; () mod 0 ; (Rel)"
            .parse()
            .unwrap();
    let err = parse2::<Referrer>(old).err().unwrap();
    assert!(err.to_string().contains("format 2, expected 3"), "{}", err);
    // And one without a module is too.
    let old: proc_macro2::TokenStream = "\"type-leak-format-1\" (MyType)".parse().unwrap();
    assert!(parse2::<Referrer>(old).is_err());
    let old: proc_macro2::TokenStream = "\"type-leak-format-2\" (MyType)".parse().unwrap();
    assert!(parse2::<Referrer>(old).is_err());
    let current = format!("\"type-leak-format-{}\" (MyType)", Referrer::FORMAT_VERSION);
    assert!(parse_str::<Referrer>(&current).is_ok());
}

#[test]
fn a_malformed_module_is_an_error() {
    let tag = format!("type-leak-format-{}", Referrer::FORMAT_VERSION);
    for text in [
        "mod { path = a } ; ()",
        "mod { unknown = 1 } ; ()",
        "mod { hint = 1 } ; ()",
        "mod x ; (Rel)",
    ] {
        assert!(
            parse_str::<Referrer>(&format!("\"{}\" {}", tag, text)).is_err(),
            "{}",
            text
        );
    }
    // The fields are optional, and may be in any order.
    let parsed: Referrer = parse_str(&format!(
        "\"{}\" 2 mod {{ parent = 1 , hint = \"x\" }} ; () mod {{}} ; ()",
        tag
    ))
    .unwrap();
    assert_eq!(parsed.module_hint(id(0)), Some("x"));
    assert_eq!(parsed.module_path(id(1)), None);
}

#[test]
fn inconsistent_module_data_is_an_error() {
    let tag = format!("type-leak-format-{}", Referrer::FORMAT_VERSION);
    let cases: &[(&str, &str)] = &[
        // A module which is not there.
        (
            "2 mod { path = [] } ; () mod 7 ; (Rel)",
            "there is no module 7",
        ),
        ("mod 3 ; (Rel)", "there is no module 3"),
        // A context which is not a module.
        ("2 () mod 0 ; (Rel)", "the context 0 is not a module"),
        (
            "2 < T > ; () mod 0 ; (Rel)",
            "the context 0 is not a module",
        ),
        // The parent.
        ("mod { parent = 0 } ; ()", "its own parent"),
        ("mod { parent = 5 } ; ()", "there is no module 5"),
        (
            "2 () mod { parent = 0 } ; ()",
            "the context 0 is not a module",
        ),
        (
            "2 mod { parent = 1 } ; () mod { parent = 0 } ; ()",
            "nested in itself",
        ),
        (
            "3 mod { parent = 1 } ; () mod { parent = 2 } ; () mod { parent = 0 } ; ()",
            "nested in itself",
        ),
        (
            "3 mod { } ; () mod { parent = 2 } ; () mod { parent = 1 } ; ()",
            "nested in itself",
        ),
        // A key which is written twice.
        (
            "mod { path = [a], path = [b] } ; ()",
            "`path` is written twice",
        ),
        (
            "mod { hint = \"a\", hint = \"b\" } ; ()",
            "`hint` is written twice",
        ),
        (
            "2 mod { } ; () mod { parent = 0, parent = 0 } ; ()",
            "`parent` is written twice",
        ),
        // What a module does not have.
        ("< T > ; mod { } ; (Rel<T>)", "which has no generics"),
        ("where T: Clone ; mod { } ; ()", "which has no generics"),
        ("Self = W ; mod { } ; ()", "where `Self` has no meaning"),
        ("Self : Tr ; mod { } ; ()", "where `Self` has no meaning"),
        ("<'a> ; Self : 'a ; mod { } ; ()", "which has no generics"),
        ("mod { } ; (Rel)", "which has no leaked type"),
    ];
    for (text, message) in cases {
        let parsed = parse_str::<Referrer>(&format!("\"{}\" {}", tag, text));
        let err = parsed.err().unwrap_or_else(|| panic!("accepted: {}", text));
        assert!(err.to_string().contains(message), "{}: {}", text, err);
    }
    // The consistent ones are fine: a module of its own, nested ones in either order of the ids,
    // and the entries which depend on a module.
    for text in [
        "mod { path = [] } ; ()",
        "2 mod { path = [] } ; () mod 0 ; (Rel)",
        "3 mod { path = [] } ; () mod { path = [a] , parent = 0 } ; () mod 1 ; (Rel)",
        "2 mod { parent = 1 } ; () mod { } ; ()",
    ] {
        let parsed = parse_str::<Referrer>(&format!("\"{}\" {}", tag, text));
        assert!(parsed.is_ok(), "{}: {:?}", text, parsed.err());
    }
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
    leaker
        .enter_mod("m", [], |view| visit_all(view, &types))
        .unwrap();
    assert_eq!(leaker.finish().iter().count(), 2);
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
    fn run(panic: bool) -> (ContextId, Referrer) {
        let mut leaker = Leaker::new();
        let (top, ()) = leaker
            .enter_generics(&parse_quote!(<T>), |view| {
                view.visit_type(&parse_quote!(Rel<T>));
                if panic {
                    caught(|| {
                        let _ = view.enter_type(&ident("S"), &parse_quote!(<U>), true, |view| {
                            view.visit_type(&parse_quote!(RelU<U>));
                            view.visit_type(&parse_quote!(Rel2<Self>));
                            view.visit_type(&parse_quote!(Bad<'_>));
                            view.enter_mod("lost", [], |_| ()).unwrap();
                            panic!("a closure panics");
                        });
                    });
                }
                // `U` is not a generic any more, and the module is not entered.
                view.visit_type(&parse_quote!(Wrap<U>));
                view.visit_type(&parse_quote!(Wrap2<T>));
            })
            .unwrap();
        (top, leaker.finish())
    }
    let (top, referrer) = run(true);
    let (expected_top, expected) = run(false);
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
    assert_eq!(s(referrer.context_generics(top)), "< T >");
    assert_eq!(referrer.module_hint(top), None);
    assert_eq!(
        referrer.to_token_stream().to_string(),
        expected.to_token_stream().to_string()
    );
}

#[test]
fn a_panic_caught_in_a_nested_scope_restores_what_the_scope_declares() {
    // `Self` is the one of the outer scope again, and so is whether it may be interned.
    let mut leaker = Leaker::new();
    let result = leaker.enter_type(&ident("O"), &Generics::default(), false, |view| {
        caught(|| {
            let _ = view.enter_impl(&Generics::default(), &parse_quote!(::p::W), true, |view| {
                view.visit_type(&parse_quote!(Rel<Self>));
                panic!("a closure panics");
            });
        });
        // `allow_self` of the outer scope is the one in effect, and `Self` is `O`.
        view.visit_type(&parse_quote!(Rel2<Self>));
    });
    assert!(result
        .unwrap_err()
        .to_string()
        .starts_with("use absolute path: 'Rel2 < Self >'"));
    let mut leaker = Leaker::new();
    leaker
        .enter_type(&ident("O"), &Generics::default(), true, |view| {
            caught(|| {
                let _ = view.enter_impl(&Generics::default(), &parse_quote!(::p::W), true, |_| {
                    panic!("a closure panics");
                });
            });
            view.visit_type(&parse_quote!(Rel2<Self>));
        })
        .unwrap();
    let referrer = leaker.finish();
    assert_eq!(self_of(&referrer, 0), "Self = O");

    // Where `Self` has no meaning, it is not given one by the scope which panicked.
    let mut leaker = Leaker::new();
    let result = leaker.enter_generics(&Generics::default(), |view| {
        caught(|| {
            let _ = view.enter_type(&ident("S"), &Generics::default(), true, |_| {
                panic!("a closure panics");
            });
        });
        view.visit_type(&parse_quote!(Rel<Self>));
    });
    // The errors are those of the outer scope: `Bad<'_>` of the scope which panicked is not one.
    match result.unwrap_err() {
        NotInternableError::SelfWithoutMeaning(selves) => assert_eq!(selves.len(), 1),
        other => panic!("{:?}", other),
    }
}

#[test]
fn a_panic_caught_in_a_nested_module_restores_the_module() {
    // The types of the outer module are in it, and the top-level context is its.
    let mut leaker = Leaker::new();
    let (outer, ()) = leaker
        .enter_mod("outer", [parse_quote!(S)], |view| {
            caught(|| {
                let _ = view.enter_mod("inner", [parse_quote!(Lost)], |view| {
                    view.visit_type(&parse_quote!(Rel));
                    panic!("a closure panics");
                });
            });
            // `S` is declared by the module, and `Lost` was by the module which is not entered.
            view.visit_type(&parse_quote!(S<Lost>));
        })
        .unwrap();
    let referrer = leaker.finish();
    assert_eq!(entries(&referrer), [e("Lost", "")]);
    assert_eq!(
        referrer.context_module(referrer.context(0)),
        Some(outer),
        "the type is in the outer module"
    );
    assert_eq!(referrer.module_hint(outer), Some("outer"));
    // The module which panicked took no number: the next one is the second.
    let mut leaker = Leaker::new();
    caught(|| {
        let _ = leaker.enter_mod("lost", [], |view| {
            view.enter_mod("lost too", [], |_| ()).unwrap();
            panic!("a closure panics");
        });
    });
    assert_eq!(leaker.enter_mod("kept", [], |_| ()).unwrap().0, id(0));
}

#[test]
fn a_panic_caught_in_the_outermost_scope_of_a_generics_scope_keeps_the_top_level_id() {
    // The top-level context of the outer scope is its generics, not the module of the nested one
    // which panicked (the module stack was not unwound).
    let mut leaker = Leaker::new();
    let (top, ()) = leaker
        .enter_generics(&parse_quote!(<T>), |view| {
            caught(|| {
                let _ = view.enter_mod("m", [], |_| panic!("a closure panics"));
            });
            view.visit_type(&parse_quote!(Rel<T>));
        })
        .unwrap();
    let referrer = leaker.finish();
    assert_eq!(s(referrer.context_generics(top)), "< T >");
    assert_eq!(referrer.module_hint(top), None);
    assert_eq!(referrer.module_path(top), None);
    assert_eq!(referrer.context_module(referrer.context(0)), None);
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
    let first = leaker
        .intern_trait(
            &parse_quote!(
                trait Tr<T> {
                    type A: ::p::X<Rel<T>>;
                }
            ),
            true,
        )
        .unwrap();
    let second = leaker
        .intern_struct(
            &parse_quote!(
                struct S<U>(Rel2<U>);
            ),
            true,
        )
        .unwrap();
    (leaker, vec![first, second])
}

/// What a scope that panics leaves behind, if the table of contexts were not rolled back: a nested
/// module and nested generics with their own contexts.
fn record_and_panic(view: &mut View<'_>, message: &str) {
    view.visit_type(&parse_quote!(RelLost<V>));
    view.enter_mod("lost", [], |_| ()).unwrap();
    view.enter_generics(&parse_quote!(<X, Y>), |_| ()).unwrap();
    panic!("{}", message);
}

/// The ids of the calls after the panic and the Referrer, with the panic (or without).
fn after_a_panic(panic: bool) -> (Vec<ContextId>, Referrer) {
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
            let _ = leaker.enter_type(&ident("S3"), &parse_quote!(<V>), true, |view| {
                record_and_panic(view, "an outermost closure panics")
            });
        });
    }
    // New contexts get the numbers they would get without the panic.
    ids.push(
        leaker
            .intern_struct(
                &parse_quote!(
                    struct S3<W>(Rel3<W>);
                ),
                true,
            )
            .unwrap(),
    );
    ids.push(
        leaker
            .enter_mod("kept", [], |view| view.visit_type(&parse_quote!(Rel4)))
            .unwrap()
            .0,
    );
    // The context of the earlier trait is still there: the same generics share it.
    ids.push(
        leaker
            .intern_struct(
                &parse_quote!(
                    struct Again<T>(Rel5<T>);
                ),
                true,
            )
            .unwrap(),
    );
    (ids, leaker.finish())
}

#[test]
fn a_panic_in_an_outermost_scope_keeps_the_earlier_contexts_and_ids() {
    let (ids, referrer) = after_a_panic(true);
    let (expected_ids, expected) = after_a_panic(false);
    assert_eq!(ids, expected_ids);
    assert_eq!(referrer, expected);
    assert_eq!(
        referrer.to_token_stream().to_string(),
        expected.to_token_stream().to_string()
    );
    // The earlier calls, then the new ones in order, and then `<T>` of the trait again.
    assert_eq!(ids, [id(0), id(1), id(2), id(3), id(0)]);
    assert_eq!(s(referrer.context_generics(ids[0])), "< T >");
    assert_eq!(s(referrer.context_generics(ids[1])), "< U >");
    assert_eq!(s(referrer.context_generics(ids[2])), "< W >");
    assert_eq!(referrer.module_hint(ids[3]), Some("kept"));
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
                view.enter_mod("lost", [], |_| ()).unwrap();
                let _ = view.enter_type(&ident("N"), &parse_quote!(<N>), true, |view| {
                    view.visit_type(&parse_quote!(RelNested<N>));
                    // Two levels down: the scopes around are left in turn.
                    let _ = view.enter_impl(
                        &parse_quote!(<I>),
                        &parse_quote!(::p::W<I>),
                        true,
                        |view| record_and_panic(view, "a nested closure panics"),
                    );
                    unreachable!("the panic goes through the scopes around");
                });
                unreachable!("the panic goes through the scopes around");
            });
        });
    }
    ids.push(
        leaker
            .intern_struct(
                &parse_quote!(
                    struct S3<W>(Rel3<W>);
                ),
                true,
            )
            .unwrap(),
    );
    ids.push(
        leaker
            .enter_generics(&parse_quote!(<V>), |view| {
                view.visit_type(&parse_quote!(RelOuter<V>))
            })
            .unwrap()
            .0,
    );
    (ids, leaker.finish())
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

/// The mentions of `Self` which `NotInternableError::SelfWithoutMeaning` has, as tokens.
fn selves_of(err: NotInternableError) -> Vec<String> {
    match err {
        NotInternableError::SelfWithoutMeaning(selves) => {
            selves.iter().map(|s| s.to_string()).collect()
        }
        other => panic!("{:?}", other),
    }
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
        let err = Leaker::new().enter_generics(generics, |_| ()).unwrap_err();
        assert_eq!(err.to_string(), "`Self` has no meaning here", "{}", text);
        assert_eq!(selves_of(err), [mention.to_string()], "{}", text);
        // In a scope nested in one which has no `Self` either (the module, and the generics).
        let nested = Leaker::new()
            .enter_mod("m", [], |view| {
                view.enter_generics(generics, |view| view.visit_type(&parse_quote!(Rel)))
                    .map(|_| ())
            })
            .unwrap()
            .1;
        assert_eq!(
            selves_of(nested.unwrap_err()),
            [mention.to_string()],
            "{}",
            text
        );
        let nested = Leaker::new()
            .enter_generics(&Generics::default(), |view| {
                view.enter_generics(generics, |_| ()).map(|_| ())
            })
            .unwrap()
            .1;
        assert_eq!(
            selves_of(nested.unwrap_err()),
            [mention.to_string()],
            "{}",
            text
        );
        // The scope fails, and what it recorded is rolled back: the one around goes on.
        let mut leaker = Leaker::new();
        leaker
            .enter_mod("m", [], |view| {
                assert!(view
                    .enter_generics(generics, |view| view.visit_type(&parse_quote!(Wrap)))
                    .is_err());
                view.visit_type(&parse_quote!(Rel));
            })
            .unwrap();
        assert_eq!(entries(&leaker.finish()), [e("Rel", "")], "{}", text);
        // The scopes which declare `Self` have a meaning for it, so they are fine.
        let mut leaker = Leaker::new();
        let ok = leaker.enter_type(&ident("S"), generics, true, |_| ());
        assert_eq!(ok.is_ok(), *declared_ok, "{}", text);
        let mut leaker = Leaker::new();
        let ok = leaker.enter_trait(&ident("Tr"), generics, true, |_| ());
        assert_eq!(ok.is_ok(), *declared_ok, "{}", text);
        let mut leaker = Leaker::new();
        let ok = leaker.enter_impl(generics, &parse_quote!(::p::W), true, |_| ());
        assert_eq!(ok.is_ok(), *declared_ok, "{}", text);
        // The generics nested in one of them keep it.
        let mut leaker = Leaker::new();
        let ok = leaker
            .enter_type(&ident("S"), &Generics::default(), false, |view| {
                view.enter_generics(generics, |_| ()).map(|_| ())
            })
            .unwrap()
            .1;
        assert_eq!(ok.is_ok(), *declared_ok, "{}", text);
    }
    // Each mention is reported, with the others of the scope, at its own tokens.
    let err = Leaker::new()
        .enter_generics(
            &with_where(
                "<T: ::p::Tr<Self>, U: ::p::Tr<Self::X>>",
                "where Self: 'static",
            ),
            |_| (),
        )
        .unwrap_err();
    assert_eq!(selves_of(err.clone()), ["Self", "Self :: X", "Self"]);
    assert_eq!(err.to_string().lines().count(), 3);
    // It is the same for the generics of the View, and the outer scope does not report it again.
    let mut leaker = Leaker::new();
    let (_, inner) = leaker
        .enter_generics(&Generics::default(), |view| {
            view.enter_generics(&parse_quote!(<T: ::p::Tr<Self>>), |_| ())
                .err()
        })
        .unwrap();
    assert_eq!(selves_of(inner.unwrap()), ["Self"]);
}

#[test]
fn self_in_a_default_where_a_scope_declares_it_is_the_conflict_of_the_default() {
    // Where `Self` has a meaning, a default which mentions it is a conflict of its own (the
    // implementation of the default is for any type); where it has none, it is `Self` which has
    // no meaning.
    let generics: Generics = parse_quote!(<T = Rel<Self>>);
    let err = Leaker::new()
        .enter_type(&ident("S"), &generics, true, |_| ())
        .unwrap_err();
    assert_eq!(
        err.note().as_deref(),
        Some("a default of a parameter cannot depend on `Self`")
    );
    assert!(matches!(err, NotInternableError::RelativePath(_)));
    let err = Leaker::new().enter_generics(&generics, |_| ()).unwrap_err();
    assert!(matches!(err, NotInternableError::SelfWithoutMeaning(_)));
    assert_eq!(err.to_string(), "`Self` has no meaning here");
    // Also in a scope nested in one which declares it, and in the default of its own.
    let err = Leaker::new()
        .enter_type(&ident("S"), &Generics::default(), true, |view| {
            view.enter_generics(&generics, |_| ()).map(|_| ())
        })
        .unwrap()
        .1
        .unwrap_err();
    assert!(matches!(err, NotInternableError::RelativePath(_)));
    let err = Leaker::new()
        .enter_mod("m", [], |view| {
            view.enter_generics(&generics, |_| ()).map(|_| ())
        })
        .unwrap()
        .1
        .unwrap_err();
    assert!(matches!(err, NotInternableError::SelfWithoutMeaning(_)));
}

// ── `allow_self` of a nested item ────────────────────────────────────────────────────────────

#[test]
fn a_nested_item_has_a_self_of_its_own_whatever_the_scope_around_says() {
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
        // In a scope which denies `Self`: nested in generics, in a module, directly, and inside
        // several of them.
        let mut leaker = Leaker::new();
        leaker
            .enter_type(&ident("S"), &Generics::default(), false, |view| {
                view.enter_generics(&Generics::default(), |view| view.visit_item(item))
                    .unwrap();
                view.enter_mod("m", [], |view| view.visit_item(item))
                    .unwrap();
                view.visit_item(item);
                view.enter_mod("m", [], |view| {
                    view.enter_generics(&Generics::default(), |view| view.visit_item(item))
                        .unwrap();
                })
                .unwrap();
            })
            .unwrap_or_else(|err| panic!("{}: {}", text, err));
        let referrer = leaker.finish();
        // (Each module is another one, so the type is an entry in each of them.)
        assert!(
            (0..referrer.iter().count())
                .any(|ix| referrer.context_self(referrer.context(ix)).is_some()),
            "{}",
            text
        );
        // In the others which deny it.
        for scope in 0..4 {
            let mut leaker = Leaker::new();
            let result = match scope {
                0 => leaker
                    .enter_type(&ident("S"), &Generics::default(), false, |view| {
                        view.visit_item(item)
                    })
                    .map(|_| ()),
                1 => leaker
                    .enter_type(&ident("S"), &Generics::default(), false, |view| {
                        view.visit_item(item)
                    })
                    .map(|_| ()),
                2 => leaker
                    .enter_impl(&Generics::default(), &parse_quote!(::p::W), false, |view| {
                        view.visit_item(item)
                    })
                    .map(|_| ()),
                _ => leaker
                    .enter_trait(&ident("S"), &Generics::default(), false, |view| {
                        view.visit_item(item)
                    })
                    .map(|_| ()),
            };
            assert!(result.is_ok(), "{}: {:?}", text, result);
        }
    }
    // The `Self` of the scope itself is still denied, also after the nested items.
    let mut leaker = Leaker::new();
    let result = leaker.enter_type(&ident("S"), &Generics::default(), false, |view| {
        view.visit_item(&items[0]);
        view.visit_type(&parse_quote!(Rel2<Self>));
    });
    assert!(result.is_err());
    // A function does not declare `Self`, so it is the one of the scope around which it denies.
    let mut leaker = Leaker::new();
    let result = leaker.enter_impl(&Generics::default(), &parse_quote!(::p::W), false, |view| {
        view.visit_signature(&parse_quote!(fn f() -> Rel<Self>));
    });
    assert!(result.is_err());
    // The items of a module analyzed as a whole have their own, as they have always had (the
    // module has no `allow_self`).
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
    leaker.intern_mod(&m).unwrap();
    assert_eq!(leaker.finish().iter().count(), 2);
}

#[test]
fn the_outermost_item_keeps_allow_self_but_the_ones_in_it_have_their_own() {
    // `allow_self` of `intern_*()` is that of the item itself, and not of the items in it.
    let item: ItemTrait = parse_quote!(
        trait Tr {
            fn f(&self) -> Rel<Self>;
            fn g(&self) -> Rel2;
        }
    );
    assert!(Leaker::new().intern_trait(&item, true).is_ok());
    assert!(Leaker::new().intern_trait(&item, false).is_err());
    let item: ItemStruct = parse_quote!(
        struct S {
            a: Rel<Self>,
        }
    );
    assert!(Leaker::new().intern_struct(&item, true).is_ok());
    assert!(Leaker::new().intern_struct(&item, false).is_err());
    // A trait in the module of the struct is a nested item, which is fine.
    let m: ItemMod = parse_quote!(
        mod m {
            pub trait Tr {
                fn f(&self) -> Rel<Self>;
            }
        }
    );
    assert!(Leaker::new().intern_mod(&m).is_ok());
}

// ── `allow_self` of a trait ──────────────────────────────────────────────────────────────────

#[test]
fn allow_self_applies_to_traits() {
    let item: ItemTrait = parse_quote!(
        trait Tr<T> {
            fn f(&self) -> Rel<Self>;
            fn g(&self, t: T) -> (Rel2, Self);
        }
    );
    let mut leaker = Leaker::new();
    leaker.intern_trait(&item, true).unwrap();
    let referrer = leaker.finish();
    assert_eq!(
        entries(&referrer),
        [e("Rel < Self >", "< T >"), e("Rel2", "")]
    );
    assert_eq!(self_of(&referrer, 0), "Self: Tr < T >");
    // `Rel<Self>` cannot be leaked without `Self`, as for a struct.
    let err = Leaker::new().intern_trait(&item, false).unwrap_err();
    assert!(err
        .to_string()
        .starts_with("use absolute path: 'Rel < Self >'"));
    assert_eq!(
        err.note().as_deref(),
        Some("it cannot be leaked as a whole because of `Self`")
    );
    // What can be decomposed is, and the receivers are not affected.
    let item: ItemTrait = parse_quote!(
        trait Tr<T> {
            fn f(self) -> (Rel, Self);
            fn g(&self, t: &Self) -> Rel2<T>;
        }
    );
    let mut leaker = Leaker::new();
    leaker.intern_trait(&item, false).unwrap();
    let referrer = leaker.finish();
    assert_eq!(entries(&referrer), [e("Rel", ""), e("Rel2 < T >", "< T >")]);
    assert!(!referrer.has_self());
    // A bound of the generics which mentions `Self` cannot be leaked either.
    let item: ItemTrait = parse_quote!(
        trait Tr<T: ::core::cmp::PartialEq<Self>> {
            fn f(&self) -> Wrap<T>;
        }
    );
    assert!(Leaker::new().intern_trait(&item, true).is_ok());
    let err = Leaker::new().intern_trait(&item, false).unwrap_err();
    assert_eq!(
        err.note().as_deref(),
        Some("the bounds of its generics mention `Self`, which cannot be leaked")
    );
    // The same for the scope of the Leaker and of the View.
    let ty: Type = parse_quote!(Rel<Self, T>);
    let generics: Generics = parse_quote!(<T>);
    for allow_self in [true, false] {
        let mut leaker = Leaker::new();
        let result = leaker.enter_trait(&ident("Tr"), &generics, allow_self, |view| {
            view.visit_type(&ty)
        });
        assert_eq!(result.is_ok(), allow_self);
        let mut leaker = Leaker::new();
        let (_, inner) = leaker
            .enter_generics(&Generics::default(), |view| {
                view.enter_trait(&ident("Tr"), &generics, allow_self, |view| {
                    view.visit_type(&ty)
                })
                .map(|_| ())
            })
            .unwrap();
        assert_eq!(inner.is_ok(), allow_self);
    }
}

// ── `visit_signature()` ──────────────────────────────────────────────────────────────────────

#[test]
fn visit_signature_declares_the_generics_of_the_signature_once() {
    // A function analyzed with its generics as those of the Leaker: it is not declared again, so
    // the lifetime is the Leaker's, and the signature is compared with no declaration.
    let sig: Signature = parse_quote!(fn f<'b>(x: ::p::W<'b, dyn RelTr>) -> &'b ());
    let mut leaker = Leaker::new();
    let (top, ()) = leaker
        .enter_generics(&sig.generics, |view| view.visit_signature(&sig))
        .unwrap();
    let referrer = leaker.finish();
    assert_eq!(
        entries(&referrer),
        [e(":: p :: W < 'b , dyn RelTr >", "< 'b >")]
    );
    assert_eq!(s(referrer.context_generics(top)), "< 'b >");
    // The same signature with its generics local ones, as for a method, which the Repeater's copy
    // would turn into an early-bound lifetime where the trait's is late-bound (E0195).
    for scope in 0..3 {
        let mut leaker = Leaker::new();
        let result = match scope {
            0 => leaker.enter_generics(&Generics::default(), |view| view.visit_signature(&sig)),
            1 => leaker.enter_impl(&Generics::default(), &parse_quote!(::p::W), true, |view| {
                view.visit_signature(&sig)
            }),
            _ => leaker.enter_trait(&ident("Tr"), &Generics::default(), true, |view| {
                view.visit_signature(&sig)
            }),
        };
        let err = result.unwrap_err();
        assert!(
            syn::Error::from(err).to_string().contains("E0195"),
            "{}",
            scope
        );
    }
    // Any other generics around do not declare those of the signature.
    let mut leaker = Leaker::new();
    let result = leaker.enter_generics(&parse_quote!(<'a>), |view| view.visit_signature(&sig));
    assert!(result.is_err());
    // Nor does a nested scope with them, and the outermost one is not compared.
    let mut leaker = Leaker::new();
    let result = leaker.enter_generics(&Generics::default(), |view| {
        view.enter_generics(&sig.generics, |view| view.visit_signature(&sig))
            .map(|_| ())
    });
    // (A nested scope has local generics, which are compared with a declaration.)
    assert!(result.unwrap().1.is_err());
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
        leaker
            .enter_generics(&Generics::default(), |view| view.visit_signature(&item.sig))
            .unwrap();
        leaker.finish()
    };
    let by_item = {
        let mut leaker = Leaker::new();
        leaker
            .enter_generics(&Generics::default(), |view| {
                view.visit_item(&Item::Fn(item.clone()))
            })
            .unwrap();
        leaker.finish()
    };
    assert_eq!(by_signature, by_item);
    assert_eq!(by_signature.iter().count(), 4);
    // A relative bound is an error, like the bounds of a method.
    let bad: Signature = parse_quote!(fn f<T: Relative>(x: T));
    let err = Leaker::new()
        .enter_generics(&Generics::default(), |view| view.visit_signature(&bad))
        .unwrap_err();
    assert_eq!(err.to_string(), "use absolute path: 'Relative'");
    // The receiver is `Self`, which the scope has to declare.
    let method: Signature = parse_quote!(fn f(&self) -> Rel);
    let err = Leaker::new()
        .enter_generics(&Generics::default(), |view| view.visit_signature(&method))
        .unwrap_err();
    assert!(matches!(err, NotInternableError::SelfWithoutMeaning(_)));
    let mut leaker = Leaker::new();
    leaker
        .enter_trait(&ident("Tr"), &Generics::default(), true, |view| {
            view.visit_signature(&method)
        })
        .unwrap();
    assert_eq!(leaker.finish().iter().count(), 1);
}

// ── `NotInternableError::SelfWithoutMeaning` ─────────────────────────────────────────────────

#[test]
fn self_without_meaning_is_an_error_of_its_own() {
    use template_quote::ToTokens;
    let err = Leaker::new()
        .enter_generics(&Generics::default(), |view| {
            view.visit_type(&parse_quote!(Rel<Self>));
            view.visit_type(&parse_quote!(Wrap<Self::Item>));
        })
        .unwrap_err();
    assert_eq!(selves_of(err.clone()), ["Self", "Self :: Item"]);
    // One error per mention, which says what is wrong, and why.
    assert_eq!(
        err.to_string(),
        "`Self` has no meaning here\n`Self` has no meaning here"
    );
    assert_eq!(
        err.note().as_deref(),
        Some(
            "`Self` has no meaning here: it is declared only by a struct, an enum, a union, an impl \
             or a trait, not by a module or generics"
        )
    );
    let converted = syn::Error::from(err.clone());
    assert_eq!(converted.into_iter().count(), 2);
    assert_eq!(
        err.to_token_stream().to_string(),
        syn::Error::from(err.clone()).to_compile_error().to_string()
    );
    let _ = err.span();
    // A scope which has both kinds of errors returns the conflicts of the paths: the others are
    // found after they are fixed.
    let err = Leaker::new()
        .enter_generics(&Generics::default(), |view| {
            view.visit_type(&parse_quote!(Rel<Self>));
            view.visit_type(&parse_quote!(Bad<'_>));
        })
        .unwrap_err();
    assert!(matches!(err, NotInternableError::RelativePath(_)));
    assert!(err.to_string().contains("Bad"));
    let err = Leaker::new()
        .enter_generics(&Generics::default(), |view| {
            view.visit_type(&parse_quote!(Rel<Self>));
        })
        .unwrap_err();
    assert!(matches!(err, NotInternableError::SelfWithoutMeaning(_)));
}

// ── the root of a trait bound ────────────────────────────────────────────────────────────────

#[test]
fn visit_path_checks_the_path_like_a_bound() {
    // A portable path is fine, and its arguments are analyzed.
    let mut leaker = Leaker::new();
    leaker
        .enter_generics(&parse_quote!(<T>), |view| {
            view.visit_path(&parse_quote!(::p::Tr<Rel<T>>));
            view.visit_path(&parse_quote!(::core::marker::Copy));
            view.visit_path(&parse_quote!(::p::Tr<Rel2>));
        })
        .unwrap();
    assert_eq!(
        entries(&leaker.finish()),
        [e("Rel < T >", "< T >"), e("Rel2", "")]
    );
    // A relative path is not, as in `visit_type_param_bound()`.
    for path in [
        parse_quote!(Tr),
        parse_quote!(Tr<Rel>),
        parse_quote!(m::Tr),
        parse_quote!(self::Tr),
        parse_quote!(super::Tr),
    ] {
        let path: Path = path;
        let by_path = Leaker::new()
            .enter_generics(&Generics::default(), |view| view.visit_path(&path))
            .unwrap_err();
        let by_bound = Leaker::new()
            .enter_generics(&Generics::default(), |view| {
                view.visit_type_param_bound(&parse_quote!(#path))
            })
            .unwrap_err();
        assert!(
            by_path.to_string().starts_with("use absolute path"),
            "{}",
            s(&path)
        );
        assert_eq!(
            by_path.to_string().lines().next(),
            by_bound.to_string().lines().next(),
            "{}",
            s(&path)
        );
    }
    // The error is about the path of the trait, at the root.
    let err = Leaker::new()
        .enter_generics(&Generics::default(), |view| {
            view.visit_path(&parse_quote!(Tr<Rel>))
        })
        .unwrap_err();
    assert_eq!(err.to_string(), "use absolute path: 'Tr < Rel >'");
    // `Self` at the root is an error where it has no meaning, and in the scope of an item it is a
    // path of the type context (a trait is not an associated type).
    let err = Leaker::new()
        .enter_generics(&Generics::default(), |view| {
            view.visit_path(&parse_quote!(Self::X))
        })
        .unwrap_err();
    assert!(matches!(err, NotInternableError::SelfWithoutMeaning(_)));
    let err = Leaker::new()
        .enter_generics(&Generics::default(), |view| {
            view.visit_type_param_bound(&parse_quote!(Self::X))
        })
        .unwrap_err();
    assert!(matches!(err, NotInternableError::SelfWithoutMeaning(_)));
    let err = Leaker::new()
        .enter_trait(&ident("Tr"), &Generics::default(), true, |view| {
            view.visit_path(&parse_quote!(Self::X))
        })
        .unwrap_err();
    assert!(err
        .to_string()
        .starts_with("use absolute path: 'Self :: X'"));
    // The path of the trait of an impl is checked in the same way.
    let item: ItemImpl = parse_quote!(impl Tr for W {});
    let err = Leaker::new().intern_impl(&item, true).unwrap_err();
    assert_eq!(err.to_string(), "use absolute path: 'Tr'");
    let item: ItemImpl = parse_quote!(impl ::p::Tr for W {});
    assert!(Leaker::new().intern_impl(&item, true).is_ok());
}

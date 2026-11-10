//! Tests of `Referrer::complete_args()` and of the generics as declared that the top-level context
//! of a scope keeps for it: where they are kept, how they tell two scopes apart, and how they are
//! written as tokens and read back. The arguments that it completes, and the `Substitute` that it
//! returns, are also tested in `tests/encode.rs`, with the errors.

mod common;

use common::{s, Recovered};
use syn::parse::Parser;
use syn::visit_mut::VisitMut;
use syn::*;
use template_quote::quote;
use type_leak::*;

fn repeater() -> Path {
    parse_quote!(crate::Repeater)
}

fn arguments(texts: &[&str]) -> Vec<GenericArgument> {
    texts.iter().map(|text| parse_str(text).unwrap()).collect()
}

/// The arguments of an impl of the scope `context` of `generics`, completed for the Repeater
/// `crate::Repeater` and the id 9, as text.
fn completed(
    referrer: &Referrer,
    context: ContextId,
    generics: &Generics,
    given: &[&str],
) -> std::result::Result<Vec<String>, EncodeError> {
    common::completed_args_for(
        referrer,
        context,
        generics,
        &arguments(given),
        &common::projection(repeater(), 9),
    )
    .map(|completed| completed.iter().map(s).collect())
}

/// The entry `Rel<T>`-like type that the `s` of the text of its tokens names.
fn entry_of(referrer: &Referrer, text: &str) -> EntryId {
    let at = common::types(referrer)
        .position(|ty| s(ty) == text)
        .unwrap_or_else(|| panic!("no entry `{}`", text));
    common::entry_id_at(referrer, at)
}

/// `<Self as crate::Repeater<args, KEY>>::Type` of the entry with the text `entry`.
fn projection(referrer: &Referrer, entry: &str, args: &str) -> String {
    let key = common::key(9, entry_of(referrer, entry));
    let args: Type = parse_str(args).unwrap();
    let ty: Type = parse_quote!(<Self as crate::Repeater<#args, #key>>::Type);
    s(&ty)
}

#[test]
fn complete_args_completes_the_arguments_of_an_analyzed_trait() {
    let tr: ItemTrait = parse_quote! {
        trait Tr<'a, T: 'a, const N: usize = 3, U = Rel<T>, V = Vec<U>> {
            fn f(&'a self, t: T) -> RelOut;
        }
    };
    let mut leaker = Leaker::with_rule(Rules::new().prelude());
    let context = leaker.intern_trait(&tr);
    let referrer = leaker.result().finish().unwrap();
    let rel = |t: &str| {
        // The entry `Rel<T>` has the generics `T` (its bound `T: 'a` is encoded) and the lifetime.
        projection(
            &referrer,
            "Rel < T >",
            &format!("(&'static (), {t}, &'static {t})", t = t),
        )
    };
    // (case, the arguments given, the arguments completed)
    for (case, given, expected) in [
        (
            "the defaults are left out",
            &["'static", "u8"][..],
            vec![
                "'static".to_string(),
                "u8".to_string(),
                "3".to_string(),
                rel("u8"),
                format!("Vec < {} >", rel("u8")),
            ],
        ),
        (
            "a default is given, so the next one uses it",
            &["'static", "u8", "5", "i32"][..],
            vec![
                "'static".to_string(),
                "u8".to_string(),
                "5".to_string(),
                "i32".to_string(),
                "Vec < i32 >".to_string(),
            ],
        ),
        (
            "every argument is given",
            &["'x", "u8", "5", "i32", "i64"][..],
            ["'x", "u8", "5", "i32", "i64"].map(String::from).to_vec(),
        ),
        (
            "a binding is skipped, and the kinds go apart",
            &["u8", "'static", "Item = u16"][..],
            vec![
                "'static".to_string(),
                "u8".to_string(),
                "3".to_string(),
                rel("u8"),
                format!("Vec < {} >", rel("u8")),
            ],
        ),
    ] {
        assert_eq!(
            completed(&referrer, context, &tr.generics, given).unwrap(),
            expected,
            "{}",
            case
        );
    }
    // The errors are `EncodeError`s, from the declaration.
    for (case, given) in [
        ("a type for the lifetime", &["u8", "u8"][..]),
        ("nothing for `T`", &["'static"][..]),
        (
            "one too many",
            &["'static", "u8", "5", "i32", "i64", "u8"][..],
        ),
    ] {
        assert!(
            completed(&referrer, context, &tr.generics, given).is_err(),
            "{}",
            case
        );
    }
}

/// The declared order of the parameters is kept: the generics of the context are sorted by name
/// (lifetimes, types, consts), which is not the order that the arguments of an impl are written in.
#[test]
fn complete_args_takes_the_arguments_in_the_declared_order() {
    let tr: ItemTrait = parse_quote! {
        trait Tr<B, const N: usize, 'a, A: 'a> { fn f(&'a self, a: A, b: B) -> Rel<A, B>; }
    };
    // A lifetime parameter after a type is not valid Rust, but the declaration is kept as written:
    // the kinds are matched apart, the order inside a kind is the declared one.
    let tr: ItemTrait = ItemTrait {
        generics: parse_quote!(<'a, B, A: 'a, const N: usize>),
        ..tr
    };
    let mut leaker = Leaker::new();
    let context = leaker.intern_trait(&tr);
    let referrer = leaker.result().finish().unwrap();
    // The generics of the context are `<'a, A: 'a, B>` (`N` is not used by the entry).
    let entry = common::entry(&referrer, 0);
    let names: Vec<String> = entry.generics.params.iter().map(s).collect();
    assert_eq!(names, ["'a", "A : 'a", "B"]);
    let completed = completed(&referrer, context, &tr.generics, &["'x", "u8", "u16", "3"]).unwrap();
    assert_eq!(completed, ["'x", "u8", "u16", "3"]);
    // `B` is the first type, `A` the second: not the order of the generics of the context.
    let mut sig: Signature = parse_quote!(fn f(&'a self, a: A, b: B));
    let mut substitute = referrer
        .complete_args(
            context,
            &arguments(&["'x", "u8", "u16", "3"]),
            &Projection::new(repeater(), 9, ImplSelf::Param(parse_quote!(__S))),
        )
        .unwrap();
    substitute.visit_signature_mut(&mut sig);
    let expected: Signature = parse_quote!(fn f(&'x self, a: u16, b: u8));
    assert_eq!(s(&sig), s(&expected));
}

/// Two traits whose generics are equal once canonical, and whose declarations differ, are two
/// contexts, so each keeps its own declaration. The same declaration is one context. The contexts
/// of the entries keep their ids: the entries of the traits that have the same context share it.
#[test]
fn two_traits_with_equal_canonical_generics_and_other_declarations_are_two_contexts() {
    let traits: Vec<ItemTrait> = vec![
        parse_quote!(
            trait A<T, U = u8> {
                fn f(&self) -> Rel<T>;
            }
        ),
        parse_quote!(
            trait B<T, U = u16> {
                fn f(&self) -> Rel<T>;
            }
        ),
        parse_quote!(
            trait C<B, A> {
                fn f(&self) -> Rel<B>;
            }
        ),
        parse_quote!(
            trait D<A, B> {
                fn f(&self) -> Rel<B>;
            }
        ),
        // The same declaration as the first one, in another trait.
        parse_quote!(
            trait E<T, U = u8> {
                fn g(&self) -> Rel<T>;
            }
        ),
    ];
    let mut leaker = Leaker::with_rule(Rules::new().primitive());
    let contexts: Vec<ContextId> = traits.iter().map(|tr| leaker.intern_trait(tr)).collect();
    // The same trait again gives the same context.
    assert_eq!(leaker.intern_trait(&traits[1]), contexts[1]);
    let referrer = leaker.result().finish().unwrap();
    let [a, b, c, d, e] = [0, 1, 2, 3, 4].map(|i| contexts[i]);
    assert_ne!(a, b);
    assert_ne!(c, d);
    assert_eq!(a, e);
    for (x, y) in [(a, c), (a, d), (b, c), (b, d)] {
        assert_ne!(x, y);
    }
    // `Rel<T>` of the first two, and `Rel<B>`, `Rel<B>` of the next ones: the entries use the
    // generics `<T>` and `<B>`, which no top level is.
    let entry_contexts = common::context_ids(&referrer);
    for top_level in [a, b, c, d] {
        assert!(!common::serialized_ids(&referrer)
            .entries
            .contains(&top_level));
        assert!(entry_contexts.contains(&top_level));
    }
    // Each completes its own defaults, before and after the tokens.
    let tokens = quote!(#referrer);
    let parsed: Referrer = parse2(tokens).unwrap();
    assert_eq!(parsed, referrer);
    for referrer in [&referrer, &parsed] {
        assert_eq!(
            completed(referrer, a, &traits[0].generics, &["u32"]).unwrap(),
            ["u32", "u8"]
        );
        assert_eq!(
            completed(referrer, b, &traits[1].generics, &["u32"]).unwrap(),
            ["u32", "u16"]
        );
        assert_eq!(
            completed(referrer, e, &traits[4].generics, &["u32"]).unwrap(),
            ["u32", "u8"]
        );
        // `<B, A>` and `<A, B>`: the first argument is for the first declared parameter.
        let written: Signature = parse_quote!(fn f(a: A, b: B));
        for (context, expected) in [(c, "fn f(a: u16, b: u8)"), (d, "fn f(a: u8, b: u16)")] {
            let mut sig = written.clone();
            referrer
                .complete_args(
                    context,
                    &arguments(&["u8", "u16"]),
                    &Projection::new(repeater(), 9, ImplSelf::Param(parse_quote!(__S))),
                )
                .unwrap()
                .visit_signature_mut(&mut sig);
            assert_eq!(s(&sig), s(&parse_str::<Signature>(expected).unwrap()));
        }
    }
}

/// The top-level context of a scope has its own id, after the contexts of the entries of its call,
/// which keep the ids they have without it. Scopes with the same declaration share it.
#[test]
fn the_context_of_an_entry_keeps_its_id_and_the_top_level_comes_after() {
    let mut leaker = Leaker::new();
    let first = leaker.intern_trait(&parse_quote!(
        trait Tr<T> {
            fn f(&self) -> Rel<T>;
        }
    ));
    let second = leaker.intern_struct(&parse_quote!(
        struct S<T>(Rel<T>, Other);
    ));
    let referrer = leaker.result().finish().unwrap();
    // `Rel<T>` has the context `<T>` (0), the generics of both items, which is no top level: it
    // keeps its id. The top level of the trait is the next (1), and the struct, declared the same,
    // has it too. `Other` has the context 2, with no generics.
    let entries: Vec<usize> = common::serialized_ids(&referrer)
        .entries
        .iter()
        .map(|id| id.index())
        .collect();
    assert_eq!(entries, [0, 2]);
    assert_eq!((first.index(), second.index()), (1, 1));
}

/// Every scope of generics has its top-level context declared: a struct, an enum, a union, an impl,
/// `enter_generics()`, and a scope inside another one.
#[test]
fn every_scope_of_generics_completes_arguments() {
    let given = arguments(&["u8"]);
    let defaulted: Generics = parse_quote!(<T, U = Vec<T>>);
    let complete = |referrer: &Referrer, context: ContextId| -> Vec<String> {
        common::completed_args_for(
            referrer,
            context,
            &defaulted,
            &given,
            &common::projection(repeater(), 9),
        )
        .unwrap()
        .iter()
        .map(s)
        .collect()
    };
    let mut leaker = Leaker::with_rule(Rules::new().prelude());
    let contexts = vec![
        (
            "struct",
            leaker.intern_struct(&parse_quote!(
                struct S<T, U = Vec<T>>(Rel<T>, U);
            )),
        ),
        (
            "enum",
            leaker.intern_enum(&parse_quote!(
                enum E<T, U = Vec<T>> {
                    A(Rel<T>, U),
                }
            )),
        ),
        (
            "union",
            leaker.intern_union(
                &parse_quote!(union U<T: Copy, U: Copy = Vec<T>> { a: Rel<T>, b: U }),
            ),
        ),
        ("generics", leaker.enter_generics(&defaulted, |_| ()).0),
        (
            "type",
            leaker.enter_type(&parse_quote!(S2), &defaulted, |_| ()).0,
        ),
        (
            "trait",
            leaker.enter_trait(&parse_quote!(Tr), &defaulted, |_| ()).0,
        ),
        (
            "impl",
            leaker
                .enter_impl(&defaulted, &parse_quote!(Host<T, U>), None, |_| ())
                .0,
        ),
        (
            "a trait in a scope",
            leaker
                .enter_generics(&Generics::default(), |view| {
                    view.enter_trait(&parse_quote!(Tr), &defaulted, |_| ()).0
                })
                .1,
        ),
    ];
    let referrer = leaker.result().finish().unwrap();
    for (case, context) in contexts {
        // `Vec` is allowed, so the default is as written, with `T` replaced.
        assert_eq!(
            complete(&referrer, context),
            ["u8", "Vec < u8 >"],
            "{}",
            case
        );
    }
}

/// A default that names `Self` (`U = Rel<Self>`) is a known limit: the Leaker reports it, and the
/// completion leaves it as written, with no panic.
#[test]
fn a_default_that_names_self_is_left_as_written() {
    let tr: ItemTrait = parse_quote!(
        trait Tr<T, U = Rel<Self>> {
            fn f(&self, u: U);
        }
    );
    let mut leaker = Leaker::new();
    let context = leaker.intern_trait(&tr);
    let result = leaker.result();
    assert_eq!(result.errors().len(), 1);
    assert_eq!(result.errors()[0].reason, Reason::SelfInDefault);
    let referrer = result.recovered();
    assert_eq!(
        completed(&referrer, context, &tr.generics, &["u8"]).unwrap(),
        ["u8", "Rel < Self >"]
    );
    // A given argument is as given.
    assert_eq!(
        completed(&referrer, context, &tr.generics, &["u8", "u16"]).unwrap(),
        ["u8", "u16"]
    );
}

fn declared_tr() -> ItemTrait {
    parse_quote! {
        pub trait Tr<'a, K: 'a + ::core::clone::Clone, const N: usize = 3, D = Wrap<K>>
        where
            K: ::core::fmt::Debug,
        {
            fn get(&'a self, key: &'a K) -> D;
        }
    }
}

/// The generics as declared are written in place of the canonical ones, which the parser computes:
/// the Referrer is read back equal, and completes the same.
#[test]
fn the_declaration_is_written_as_tokens_and_read_back() {
    let tr = declared_tr();
    let mut leaker = Leaker::new();
    let context = leaker.intern_trait(&tr);
    let referrer = leaker.result().finish().unwrap();
    let text = s(&referrer);
    // The top-level context is the second group: only the declaration (as written), then no
    // entry. The canonical generics (sorted, with the where predicate moved to the bound, no
    // defaults) are not written, as they are for the context of the entry in the first group.
    assert!(
        text.ends_with(
            "(Wrap < K >) \
             declared < 'a , K : 'a + :: core :: clone :: Clone , const N : usize = 3 , D = Wrap < K > > \
             where K : :: core :: fmt :: Debug , ; ()"
        ),
        "{}",
        text
    );
    assert!(!text.contains("< 'a , D ,"), "{}", text);
    let parsed: Referrer = parse2(quote!(#referrer)).unwrap();
    assert_eq!(parsed, referrer);
    for referrer in [&referrer, &parsed] {
        let completed = completed(referrer, context, &tr.generics, &["'static", "u8"]).unwrap();
        assert_eq!(completed[..3], ["'static", "u8", "3"]);
        assert!(completed[3].contains("Repeater"), "{}", completed[3]);
    }
    // The context id is written as an integer, and the Referrer follows the one that the Leaker
    // returned.
    let tokens = quote!(#referrer, #context);
    let (parsed, id): (Referrer, ContextId) = (|input: syn::parse::ParseStream| {
        let referrer = input.parse()?;
        input.parse::<Token![,]>()?;
        Ok((referrer, input.parse()?))
    })
    .parse2(tokens)
    .unwrap();
    assert_eq!(parsed, referrer);
    assert_eq!(id, context);

    // A trait with no generics has the declaration `declared < > ;`: it completes nothing, and
    // it is not a module.
    let mut leaker = Leaker::new();
    let context = leaker.intern_trait(&parse_quote!(
        trait Plain {
            fn f(&self) -> Rel;
        }
    ));
    let referrer = leaker.result().finish().unwrap();
    assert!(s(&referrer).contains("declared < > ;"), "{}", s(&referrer));
    let parsed: Referrer = parse2(quote!(#referrer)).unwrap();
    assert_eq!(parsed, referrer);
    assert_eq!(
        completed(&parsed, context, &Generics::default(), &[]).unwrap(),
        Vec::<String>::new()
    );
}

/// A `LeakerResult` keeps the declaration of a scope that has errors, with its errors.
#[test]
fn the_declaration_of_a_scope_with_errors_is_carried_by_the_result() {
    let tr: ItemTrait = parse_quote! {
        trait Tr<T, D = Wrap<T>> { fn f(&self, t: T, w: Bad<'_>) -> D; }
    };
    let mut leaker = Leaker::new();
    let context = leaker.intern_trait(&tr);
    let result = leaker.result();
    assert_eq!(result.errors().len(), 1);
    let parsed: LeakerResult = parse2(quote!(#result)).unwrap();
    assert_eq!(parsed.errors().len(), 1);
    let referrer = result.recovered();
    let read = parsed.recovered();
    assert_eq!(read, referrer);
    assert_eq!(
        completed(&read, context, &tr.generics, &["u8"]).unwrap(),
        completed(&referrer, context, &tr.generics, &["u8"]).unwrap()
    );
    // A `Referrer` that was serialized alone reads as a `LeakerResult` with no error, and keeps
    // the declaration too.
    let referrer = Leaker::new().result().finish().unwrap();
    let parsed: LeakerResult = parse2(quote!(#referrer)).unwrap();
    assert!(parsed.errors().is_empty());
}

/// The parser refuses a declaration that is not well formed.
#[test]
fn a_malformed_declaration_is_refused() {
    let tag = format!("\"type-leak-format-{}\"", Referrer::FORMAT_VERSION);
    // (case, the group, part of the message)
    for (case, group, message) in [
        (
            "no angle bracket",
            "declared ;  ()",
            "expected `<` after `declared`",
        ),
        (
            "the generics and the same generics as declared",
            "< T > ; declared < T > ; ()",
            "has both `< generics >` and `declared < .. >`",
        ),
        (
            "the generics and other generics as declared",
            "< T > ; declared < U > ; ()",
            "has both `< generics >` and `declared < .. >`",
        ),
        (
            "a where clause and the generics as declared",
            "where T : Clone ; declared < T > ; ()",
            "has both `< generics >` and `declared < .. >`",
        ),
        (
            "the generics after the generics as declared",
            "declared < T > ; < T > ; ()",
            "expected",
        ),
        (
            "a module word after the generics as declared",
            "declared < > ; mod { path = [] } ; ()",
            "expected parentheses",
        ),
        (
            "a declaration twice",
            "declared < > ; declared < > ; ()",
            "expected",
        ),
        (
            "a declaration after `Self`",
            "Self ; declared < > ; ()",
            "expected",
        ),
        (
            "the declaration in a second group of a declared context",
            "2 declared < > ; ( ) # 0 declared < > ; ( )",
            "declared already",
        ),
    ] {
        let tokens: proc_macro2::TokenStream = format!("{} {}", tag, group).parse().unwrap();
        let error = parse2::<Referrer>(tokens).expect_err(case);
        assert!(
            error.to_string().contains(message),
            "{}: `{}` has no `{}`",
            case,
            error,
            message
        );
    }
    // The same groups, well formed, are read.
    for group in [
        "declared < > ; ()",
        "< T > ; ()",
        "declared < T = u8 > ; ()",
        "declared < U , T : Clone > ; ()",
        "declared < U , T > where T : Clone ; ()",
    ] {
        let tokens: proc_macro2::TokenStream = format!("{} {}", tag, group).parse().unwrap();
        parse2::<Referrer>(tokens).unwrap_or_else(|error| panic!("{}: {}", group, error));
    }
    // The generics of the context are computed from the generics as declared: sorted, the where
    // predicate moved to the bound, no default.
    let tokens: proc_macro2::TokenStream = format!(
        "{} declared < U = u8 , T > where T : Clone ; ( Wrap < T > )",
        tag
    )
    .parse()
    .unwrap();
    let referrer: Referrer = parse2(tokens).unwrap();
    assert_eq!(
        s(&common::entry(&referrer, 0).generics),
        "< T : Clone , U >"
    );
}

/// The self type that `<_ as Repeater<..>>::Type` is written for.
fn projected_self(ty: &Type) -> Type {
    match ty {
        Type::Path(TypePath {
            qself: Some(qself), ..
        }) => (*qself.ty).clone(),
        other => panic!("not a projection: {}", s(other)),
    }
}

/// The default that names an entry (the shape of `trait Store<'a, K: 'a, D = Page<K>>` of
/// `examples/two-macros`) is written for the self type of the impl that `repeater_impls()` wrote
/// for that entry: `Self` for a parameter, the marker for a marker, and the primitive type of the
/// bucket for a bucket. The impl and the written type have one self type.
#[test]
fn a_default_is_written_for_the_self_type_of_the_impl_of_its_entry() {
    let tr: ItemTrait = parse_quote! {
        trait Store<'a, K: 'a, D = Page<K>> {
            fn get(&self, key: &'a K) -> D;
            fn keep(&'a self) -> &'a K;
        }
    };
    let mut leaker = Leaker::new();
    let context = leaker.intern_trait(&tr);
    let result = leaker.result();
    let referrer = result.clone().finish().unwrap();
    let page = entry_of(&referrer, "Page < K >");
    let given: Vec<GenericArgument> = vec![parse_quote!('static), parse_quote!(u8)];
    let param: Ident = parse_quote!(__S);
    let marker: Type = parse_quote!(crate::Marker);
    let bucket = common::bucket_self(&referrer, 7, page).unwrap();
    for (name, impl_self, expected_self) in [
        ("param", ImplSelf::Param(param.clone()), parse_quote!(Self)),
        ("marker", ImplSelf::Marker(marker.clone()), marker.clone()),
        (
            "bucket",
            ImplSelf::Bucket {
                param: param.clone(),
            },
            bucket,
        ),
    ] {
        let projection = Projection::new(repeater(), 7, impl_self);
        // The impl of `Page<K>`, in crate A.
        let impls = result.repeater_impls(&projection).unwrap();
        let imp = impls.iter().find(|r| r.entry == page).unwrap();
        // The omitted `D` of `impl Store<'static, u8> for ..`, in crate B.
        let completed =
            common::completed_args_for(&referrer, context, &tr.generics, &given, &projection)
                .unwrap();
        let written = match &completed[2] {
            GenericArgument::Type(ty) => ty.clone(),
            other => panic!("{}: not a type: {}", name, s(other)),
        };
        let key = projection.key(page);
        let expected: Type = parse_quote! {
            <#expected_self as crate::Repeater<(&'static (), u8, &'static u8), #key>>::Type
        };
        assert_eq!(s(&written), s(&expected), "{}", name);
        // The impl is for the type that the copy writes. A parameter is every type, `Self` too.
        match name {
            "param" => assert_eq!(s(&imp.self_ty), s(&param), "{}", name),
            _ => assert_eq!(s(&imp.self_ty), s(&projected_self(&written)), "{}", name),
        }
    }

    // A bucket writes `Self` for an entry that needs it: the impl is for the parameter, though
    // `Rel<T>` names no `Self` (its generics do, `T: AsRef<Self>`).
    let tr: ItemTrait = parse_quote! {
        trait Tr<T: AsRef<Self>, U = Rel<T>> { fn f(&self, t: T) -> U; }
    };
    let mut leaker = Leaker::with_rule(Rules::new().prelude());
    let context = leaker.intern_trait(&tr);
    let result = leaker.result();
    let referrer = result.clone().finish().unwrap();
    let rel = entry_of(&referrer, "Rel < T >");
    assert_eq!(common::bucket_self(&referrer, 7, rel), None);
    let projection = Projection::new(
        repeater(),
        7,
        ImplSelf::Bucket {
            param: param.clone(),
        },
    );
    let impls = result.repeater_impls(&projection).unwrap();
    let completed = common::completed_args_for(
        &referrer,
        context,
        &tr.generics,
        &[parse_quote!(String)],
        &projection,
    )
    .unwrap();
    let key = projection.key(rel);
    let expected: Type = parse_quote!(<Self as crate::Repeater<(String,), #key>>::Type);
    assert_eq!(s(&completed[1]), s(&expected));
    assert_eq!(s(&impls[0].self_ty), s(&param));
}

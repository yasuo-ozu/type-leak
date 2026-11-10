//! `ErrorItem::paths`: each relative path of an error as the cause of its kind.
//!
//! - A path in a type is a `Cause::Type`, the trait of a bound or of a qualified path a
//!   `Cause::Trait`, a path in a value a `Cause::Expr`, the path of a macro a `Cause::Macro`; all
//!   without generic arguments.
//! - A path whose kind is not known (the root of a `use` in a value) is a `Cause::Unreadable`.
//! - The kinds survive the serialization of the error.

use syn::parse::Parser;
use syn::*;
use template_quote::ToTokens;
use type_leak::*;

/// The paths of the errors of `item` as `kind(path)`, the paths of an error joined by a space.
fn paths_of(result: &LeakerResult) -> Vec<Vec<String>> {
    result
        .errors()
        .iter()
        .map(|error| error.paths.iter().map(show).collect())
        .collect()
}

fn show(cause: &Cause) -> String {
    let (kind, text) = match cause {
        Cause::Type(_) => ("type", cause.to_token_stream().to_string()),
        Cause::Trait(_) => ("trait", cause.to_token_stream().to_string()),
        Cause::Expr(_) => ("expr", cause.to_token_stream().to_string()),
        Cause::Macro(_) => ("macro", cause.to_token_stream().to_string()),
        Cause::Unreadable(tokens) => ("unreadable", tokens.to_string()),
        other => panic!("{:?}", other),
    };
    format!("{}({})", kind, text)
}

fn values() -> Leaker<Rules> {
    Leaker::from_config(Rules::new(), Options::new().check_values())
}

#[test]
fn a_path_in_a_type_is_a_type_and_the_trait_of_a_bound_a_trait() {
    let item: ItemTrait = parse_quote!(
        trait Tr: Clone {
            fn f(&self, a: Vec<Rel<'_>>);
        }
    );
    let mut leaker = Leaker::new();
    leaker.intern_trait(&item, true);
    let result = leaker.result();
    assert_eq!(
        paths_of(&result),
        [vec!["trait(Clone)"], vec!["type(Vec)", "type(Rel)"]]
    );
}

#[test]
fn the_trait_of_a_bound_and_a_trait_object_is_a_trait_and_its_arguments_are_types() {
    let item: ItemStruct = parse_quote!(
        struct S<T: RelTr<Arg>> {
            a: ::p::W<'_, dyn OtherTr<Item = Assoc>>,
            b: T,
        }
    );
    let mut leaker = Leaker::new();
    leaker.intern_struct(&item, true);
    let result = leaker.result();
    assert_eq!(
        paths_of(&result),
        [
            vec!["trait(RelTr)", "type(Arg)"],
            vec!["trait(OtherTr)", "type(Assoc)"]
        ]
    );
}

#[test]
fn the_trait_of_a_qualified_path_is_a_trait_and_its_type_a_type() {
    let item: ItemStruct = parse_quote!(
        struct S {
            a: Wrap<'_, <Rel as m::Tr<u8>>::X>,
        }
    );
    let mut leaker = Leaker::new();
    leaker.intern_struct(&item, true);
    let result = leaker.result();
    assert_eq!(
        paths_of(&result),
        [vec![
            "type(Wrap)",
            "trait(m :: Tr)",
            "type(Rel)",
            "type(u8)"
        ]]
    );
}

#[test]
fn the_path_of_a_macro_is_a_macro() {
    let item: ItemStruct = parse_quote!(
        struct S {
            a: Wrap<'_, m!(u8)>,
        }
    );
    let mut leaker = Leaker::new();
    leaker.intern_struct(&item, true);
    let result = leaker.result();
    assert_eq!(paths_of(&result), [vec!["type(Wrap)", "macro(m !)"]]);
}

#[test]
fn a_path_in_a_value_is_an_expression_and_its_arguments_are_types() {
    let item: ItemConst = parse_quote!(
        const C: u8 = rel::f::<Arg>(1);
    );
    let mut leaker = values();
    leaker.enter_generics(&Generics::default(), |view| {
        view.visit_item(&Item::Const(item));
    });
    let result = leaker.result();
    assert_eq!(paths_of(&result), [vec!["expr(rel :: f)", "type(Arg)"]]);
}

#[test]
fn the_root_of_a_use_in_a_value_has_no_known_kind() {
    let item: ItemFn = parse_quote!(
        fn f() {
            use inner::X;
        }
    );
    let mut leaker = values();
    leaker.enter_generics(&Generics::default(), |view| {
        view.visit_item(&Item::Fn(item));
    });
    let result = leaker.result();
    assert_eq!(result.errors().len(), 1);
    assert_eq!(paths_of(&result), [vec!["unreadable(inner)"]]);
}

#[test]
fn the_kinds_are_written_and_read_back() {
    let item: ItemTrait = parse_quote!(
        trait Tr: Clone {
            fn f(&self, a: Wrap<'_, <Rel as m::Tr<u8>>::X>, b: Wrap<'_, m!(u8)>);
        }
    );
    let mut leaker = Leaker::new();
    leaker.intern_trait(&item, true);
    let result = leaker.result();
    assert!(result.errors().len() >= 3);
    for error in result.errors() {
        let tokens = error.to_token_stream();
        let read: ErrorItem = (|input: parse::ParseStream| input.parse())
            .parse2(tokens)
            .unwrap();
        assert_eq!(read.paths, error.paths, "{}", error.message);
    }
}

#[test]
fn a_path_entry_with_arguments_or_that_is_no_path_is_refused() {
    let read = |source: &str| parse_str::<ErrorItem>(source);
    assert!(read(r#"{ type ( A ) RelativePath "m" [] paths [ type ( a ) ] }"#).is_ok());
    assert!(read(r#"{ type ( A ) RelativePath "m" [] paths [ type ( a < b > ) ] }"#).is_err());
    assert!(
        read(r#"{ type ( A ) RelativePath "m" [] paths [ unreadable ( a < b > ) ] }"#).is_err()
    );
    assert!(read(r#"{ type ( A ) RelativePath "m" [] paths [ unreadable ( & a ) ] }"#).is_err());
    assert!(read(r#"{ type ( A ) RelativePath "m" [] paths [ lifetime ( 'a ) ] }"#).is_err());
}

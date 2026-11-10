//! The errors of the encoders and of the helpers that encode (`EncodeError`): their text, the
//! conversion to `syn::Error`, and the `std::error::Error` impls.
//!
//! The text of `UnsupportedConstType`, `UnexpectedArgument` and `MissingArgument` is also pinned in
//! `tests/encode.rs`. The spans of the errors are checked in `tests/crates/spans`, which
//! `tests/spans.rs` runs: they need a feature of `proc-macro2` that the tests of this package
//! cannot have.

use syn::*;
use type_leak::*;

fn slots() -> RepeaterSlots {
    RepeaterSlots::new(parse_quote!(crate::Repeater))
}

/// The error of encoding the arguments in `args` for the parameters in `generics`.
fn args_error(generics: &str, args: &str) -> EncodeError {
    let generics: Generics = parse_str(generics).unwrap();
    let args: AngleBracketedGenericArguments = parse_str(args).unwrap();
    slots()
        .encode_generics_args_to_ty(&args.args, &generics.params)
        .unwrap_err()
}

/// The error of encoding `generics` itself.
fn params_error(generics: &str) -> EncodeError {
    let generics: Generics = parse_str(generics).unwrap();
    slots()
        .encode_generics_params_to_ty(&generics.params)
        .unwrap_err()
}

/// The error of the Repeater impl of the entry `Wrap<T, Self>` for a marker, which cannot name
/// `Self`. The source says where the entry is written.
fn needs_self_error(source: &str) -> EncodeError {
    let item: ItemTrait = parse_str(source).unwrap();
    let mut leaker = Leaker::new();
    leaker.intern_trait(&item, true);
    let referrer = leaker.result().finish().unwrap();
    let wrap = referrer
        .iter()
        .position(|ty| template_quote::quote!(#ty).to_string() == "Wrap < T , Self >")
        .unwrap();
    let marker: Type = parse_quote!(crate::Marker);
    referrer
        .repeater_impl(wrap, ImplSelf::Marker(&marker))
        .unwrap_err()
}

const NEEDS_SELF_SOURCE: &str = "trait Tr<'a, T> {\n    fn f(&self) -> Wrap<T, Self>;\n}";

/// Every variant.
fn errors() -> Vec<EncodeError> {
    vec![
        params_error("<const N: MyIdx>"),
        args_error("<T>", "<u8, u16>"),
        args_error("<'a, T>", "<u8>"),
        args_error("<'a, T: ?Sized + 'a>", "<'x, dyn Tr>"),
        needs_self_error(NEEDS_SELF_SOURCE),
    ]
}

#[test]
fn the_text_of_every_variant() {
    let all = errors();
    let texts: Vec<String> = all.iter().map(|err| err.to_string()).collect();
    assert_eq!(
        texts,
        [
            "unsupported type of const parameter: 'MyIdx'",
            "unexpected generic argument: 'u16'",
            "missing generic argument for ''a'",
            "the trait object 'dyn Tr' has no lifetime bound, which defaults to different \
             lifetimes in the encoding of a type with a lifetime bound; write it (e.g. `+ \
             'static`)",
            "the type 'Wrap < T , Self >' names `Self`, so its Repeater impl cannot be for a \
             marker type; use `ImplSelf::Param`",
        ]
    );
    assert!(matches!(all[0], EncodeError::UnsupportedConstType(_)));
    assert!(matches!(all[1], EncodeError::UnexpectedArgument(_)));
    assert!(matches!(all[2], EncodeError::MissingArgument(_)));
    assert!(matches!(all[3], EncodeError::UnboundedTraitObject(_)));
    assert!(matches!(all[4], EncodeError::NeedsSelf(_)));
}

#[test]
fn the_text_of_an_unbounded_trait_object_is_that_of_its_type() {
    // The type is shown as it is written: parenthesized, or with other bounds.
    let generics = "<'a, T: ?Sized + 'a>";
    for (arg, shown) in [
        ("dyn Tr", "dyn Tr"),
        ("(dyn Tr)", "(dyn Tr)"),
        ("dyn Tr + Send", "dyn Tr + Send"),
    ] {
        let err = args_error(generics, &format!("<'x, {}>", arg));
        assert_eq!(
            err.to_string(),
            format!(
                "the trait object '{}' has no lifetime bound, which defaults to different \
                 lifetimes in the encoding of a type with a lifetime bound; write it (e.g. `+ \
                 'static`)",
                shown
            )
        );
    }
}

#[test]
fn an_encode_error_is_a_syn_error_with_its_text() {
    for err in errors() {
        let text = err.to_string();
        let converted: syn::Error = err.into();
        assert_eq!(converted.to_string(), text);
        // One error, and it renders as `compile_error!`.
        assert_eq!(converted.clone().into_iter().count(), 1);
        let tokens = converted.to_compile_error().to_string();
        assert!(tokens.contains("compile_error"), "{}", tokens);
        assert!(tokens.contains(&format!("{:?}", text)), "{}", tokens);
    }
}

fn is_error<E: std::error::Error>(_: &E) {}

#[test]
fn both_errors_are_std_errors() {
    // `EncodeError`: it can be boxed, `?`-converted into a `Box<dyn Error>`, and has no source.
    for err in errors() {
        is_error(&err);
        let text = err.to_string();
        let debug = format!("{:?}", err);
        let boxed: Box<dyn std::error::Error> = Box::new(err.clone());
        assert_eq!(boxed.to_string(), text);
        assert!(boxed.source().is_none());
        // The derived `Debug` names the variant.
        assert!(debug.starts_with(&text_variant(&err)), "{}", debug);
        let result: std::result::Result<(), Box<dyn std::error::Error>> = (|| Err(err.clone())?)();
        assert_eq!(result.unwrap_err().to_string(), text);
    }
    // `ErrorItem`, one of the errors of the Leaker (it returns a `Vec` of them).
    let item: ItemTrait = parse_quote!(
        trait Tr: Clone {}
    );
    let mut leaker = Leaker::new();
    leaker.intern_trait(&item, true);
    let result = leaker.result();
    let err = result.errors();
    let err = err[0].clone();
    is_error(&err);
    let boxed: Box<dyn std::error::Error> = Box::new(err.clone());
    assert_eq!(boxed.to_string(), "use absolute path: 'Clone'");
    assert!(boxed.source().is_none());
    let result: std::result::Result<(), Box<dyn std::error::Error>> = (|| Err(err.clone())?)();
    assert_eq!(
        result.unwrap_err().to_string(),
        "use absolute path: 'Clone'"
    );
}

/// The name of the variant of `err`.
fn text_variant(err: &EncodeError) -> String {
    match err {
        EncodeError::UnsupportedConstType(_) => "UnsupportedConstType(",
        EncodeError::UnexpectedArgument(_) => "UnexpectedArgument(",
        EncodeError::MissingArgument(_) => "MissingArgument(",
        EncodeError::UnboundedTraitObject(_) => "UnboundedTraitObject(",
        EncodeError::NeedsSelf(_) => "NeedsSelf(",
        _ => unreachable!(),
    }
    .to_string()
}

//! The spans of `EncodeError`, and of the `syn::Error` made of it.
//!
//! The code is parsed from a string, so that it has a place (see `tests/spans.rs`), and the span of
//! the error is compared with the place of the part of the string that is at fault. The text of
//! the errors is in `tests/encode_errors.rs` of `type-leak`.

mod support;

use support::assert_span_at;
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
    leaker.intern_trait(&item, true).unwrap();
    let referrer = leaker.finish();
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

/// Every variant, with the source it is parsed from and the part of it that is at fault.
fn errors() -> Vec<(EncodeError, &'static str, &'static str, usize)> {
    vec![
        (
            params_error("<const N: MyIdx>"),
            "<const N: MyIdx>",
            "MyIdx",
            0,
        ),
        (args_error("<T>", "<u8, u16>"), "<u8, u16>", "u16", 0),
        (args_error("<'a, T>", "<u8>"), "<'a, T>", "'a", 0),
        (
            args_error("<'a, T: ?Sized + 'a>", "<'x, dyn Tr>"),
            "<'x, dyn Tr>",
            "dyn Tr",
            0,
        ),
        (
            needs_self_error(NEEDS_SELF_SOURCE),
            NEEDS_SELF_SOURCE,
            "Wrap<T, Self>",
            0,
        ),
    ]
}

#[test]
fn the_span_of_every_variant_is_the_part_that_is_at_fault() {
    let all = errors();
    assert!(matches!(all[0].0, EncodeError::UnsupportedConstType(_)));
    assert!(matches!(all[1].0, EncodeError::UnexpectedArgument(_)));
    assert!(matches!(all[2].0, EncodeError::MissingArgument(_)));
    assert!(matches!(all[3].0, EncodeError::UnboundedTraitObject(_)));
    assert!(matches!(all[4].0, EncodeError::NeedsSelf(_)));
    for (err, source, needle, nth) in all {
        assert_span_at(err.span(), source, needle, nth);
    }
}

#[test]
fn an_encode_error_is_a_syn_error_at_its_span() {
    for (err, source, needle, nth) in errors() {
        let text = err.to_string();
        let converted: syn::Error = err.into();
        assert_eq!(converted.to_string(), text);
        assert_span_at(converted.span(), source, needle, nth);
    }
}

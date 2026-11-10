//! The spans of `EncodeError`, and of the `syn::Error` made of it.
//!
//! The code is parsed from a string, so that it has a place (see `tests/diagnostic.rs`), and the
//! span of the error is compared with the place of the part of the string that is at fault. The text
//! of the errors is in `tests/encode.rs` of `type-leak`.

mod support;

use support::assert_span_at;
use syn::*;
use template_quote::ToTokens;
use type_leak::*;

fn repeater() -> Path {
    parse_quote!(crate::Repeater)
}

/// The error of encoding the arguments in `args` for the parameters in `generics`: completing
/// them, then writing them at a use site of an entry that uses every generic.
fn args_error(generics: &str, args: &str) -> EncodeError {
    let generics: Generics = parse_str(generics).unwrap();
    let args: AngleBracketedGenericArguments = parse_str(args).unwrap();
    let empty = Leaker::new().result().into_referrer();
    // The only context of a Referrer with no entry.
    let context: ContextId = parse_str("0").unwrap();
    let completed =
        match empty.complete_args(context, &generics.params, &args.args, |_, _, _, _| {
            unreachable!()
        }) {
            Ok(completed) => completed,
            Err(error) => return error,
        };
    let used = generics.params.iter().map(|param| match param {
        GenericParam::Lifetime(param) => param.lifetime.to_token_stream(),
        GenericParam::Type(param) => param.ident.to_token_stream(),
        GenericParam::Const(param) => param.ident.to_token_stream(),
    });
    let item: ItemTrait = parse_quote!(trait Tr #generics { fn f(&self) -> Wrap<#(#used),*>; });
    let mut leaker = Leaker::new();
    leaker.intern_trait(&item, true);
    let referrer = leaker.result().finish().unwrap();
    let (_, _, context) = referrer.iter().next().unwrap();
    let self_ty: Type = parse_quote!(Self);
    UseSite::substituting(&generics.params, &completed)
        .generic_args(&context.generics)
        .and_then(|args| {
            encode_generics_args_with_params(
                &args,
                &context.generics.params,
                Some((&self_ty, &context.self_outlives)),
                slot_const(&repeater()),
            )
        })
        .unwrap_err()
}

/// The error of encoding `generics` itself: the const parameter of the first one.
fn params_error(generics: &str) -> EncodeError {
    let generics: Generics = parse_str(generics).unwrap();
    let param = generics
        .params
        .iter()
        .find_map(|param| match param {
            GenericParam::Const(param) => Some(param),
            _ => None,
        })
        .unwrap();
    encode_slot_const(&repeater(), &parse_quote!(N), &param.ty)
        .unwrap_err()
}

/// The error of `repeater_impls()` for a marker, which cannot name `Self`: the entry `Wrap<T,
/// Self>`. The source says where the entry is written.
fn needs_self_error(source: &str) -> EncodeError {
    let item: ItemTrait = parse_str(source).unwrap();
    let mut leaker = Leaker::new();
    leaker.intern_trait(&item, true);
    let result = leaker.result();
    let marker: Type = parse_quote!(crate::Marker);
    // The call fails for the first entry that names `Self`, which is `Wrap<T, Self>` here.
    let error = result
        .repeater_impls(&parse_quote!(crate::Repeater), 7, ImplSelf::Marker(&marker), array_const)
        .err()
        .unwrap();
    assert!(matches!(&error, EncodeError::NeedsSelf(ty)
        if template_quote::quote!(#ty).to_string() == "Wrap < T , Self >"));
    error
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

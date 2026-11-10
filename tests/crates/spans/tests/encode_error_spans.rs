//! The spans of `EncodeError`, and of the `syn::Error` made of it.
//!
//! The code is parsed from a string, so that it has a place (see `tests/diagnostic.rs`), and the
//! span of the error is compared with the place of the part of the string that is at fault. The text
//! of the errors is in `tests/referrer/encode.rs` of `type-leak`.

mod support;

use support::assert_span_at;
use syn::*;
use type_leak::*;

fn projection(impl_self: ImplSelf) -> Projection {
    Projection::new(parse_quote!(crate::Repeater), 7, impl_self)
}

/// The error of completing the arguments in `args` for the parameters in `generics`.
fn args_error(generics: &str, args: &str) -> EncodeError {
    let generics: Generics = parse_str(generics).unwrap();
    let args: AngleBracketedGenericArguments = parse_str(args).unwrap();
    // The scope of the generics, where every path is written as it is: the defaults are what they say.
    let mut leaker = Leaker::with_rule(|_: &Cause, _: Place| true);
    let (context, ()) = leaker.enter_generics(&generics, |_| ());
    let referrer = leaker.result().finish().unwrap();
    referrer
        .complete_args(
            context,
            &args.args,
            &projection(ImplSelf::Param(parse_quote!(__S))),
        )
        .unwrap_err()
}

/// The error of writing the path through the Repeater for an entry that needs `Self: 'a`, for the
/// type in `source` as `Self`, a trait object without a lifetime bound.
fn unbounded_self_error(source: &str) -> EncodeError {
    let item: ItemTrait = parse_quote!(
        trait Tr<'a> {
            fn f(&self, x: &'a Wrap<Self>);
        }
    );
    let mut leaker = Leaker::new();
    leaker.intern_trait(&item);
    let referrer = leaker.result().finish().unwrap();
    let self_ty: Type = parse_str(source).unwrap();
    let (ix, _, context) = referrer.iter().next().unwrap();
    projection(ImplSelf::Param(parse_quote!(__S)))
        .of(&self_ty, context, ix)
        .unwrap_err()
}

/// The error of encoding the const parameter `N` of `generics` for the entry `Wrap<{ N }>`.
fn params_error(generics: &str) -> EncodeError {
    let generics: Generics = parse_str(generics).unwrap();
    let mut leaker = Leaker::new();
    leaker.enter_generics(&generics, |view| {
        view.visit_type(&parse_quote!(Wrap<{ N }>))
    });
    let referrer = leaker.result().finish().unwrap();
    let (entry, _, context) = referrer.iter().next().unwrap();
    projection(ImplSelf::Param(parse_quote!(__S)))
        .of(&parse_quote!(Self), context, entry)
        .unwrap_err()
}

/// The error of `repeater_impls()` for a marker, which cannot name `Self`: the entry `Wrap<T,
/// Self>`. The source says where the entry is written.
fn needs_self_error(source: &str) -> EncodeError {
    let item: ItemTrait = parse_str(source).unwrap();
    let mut leaker = Leaker::new();
    leaker.intern_trait(&item);
    let result = leaker.result();
    let marker = projection(ImplSelf::Marker(parse_quote!(crate::Marker)));
    // The call fails for the first entry that names `Self`, which is `Wrap<T, Self>` here.
    let error = result.repeater_impls(&marker).err().unwrap();
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
        (unbounded_self_error("dyn Tr"), "dyn Tr", "dyn Tr", 0),
        (
            needs_self_error(NEEDS_SELF_SOURCE),
            NEEDS_SELF_SOURCE,
            "Wrap<T, Self>",
            0,
        ),
        (
            args_error("<T, U = T::Item>", "<::std::vec::IntoIter<u8>>"),
            "<T, U = T::Item>",
            "T::Item",
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
    assert!(matches!(all[5].0, EncodeError::AmbiguousAssociatedType(_)));
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

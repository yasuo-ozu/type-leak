//! The names that `LeakerConfig::allow_primitive()` and `LeakerConfig::allow_std_prelude()` allow.
//!
//! An allowed path is the same on both sides, so it is not leaked: a bound that names it is fine,
//! and a type that names it has no entry. Without the option the same name is a relative path:
//! an error in a bound or a const expression, and an entry in a type. Dropping a name from a list
//! makes the Leaker leak a type that the Referrer's crate has anyway, so everything compiles still,
//! and only the entries tell. Each name is checked here, in a bound, in a const expression and in
//! a type, with the option and without it.

use std::collections::BTreeSet;
use syn::*;
use template_quote::quote;
use type_leak::*;

/// The names that `allow_primitive()` allows.
const PRIMITIVES: [&str; 17] = [
    "bool", "char", "str", "u8", "u16", "u32", "u64", "u128", "usize", "i8", "i16", "i32", "i64",
    "i128", "isize", "f32", "f64",
];

/// The names that `allow_std_prelude()` allows.
const PRELUDE: [&str; 31] = [
    "Box",
    "String",
    "Vec",
    "Option",
    "Result",
    "Copy",
    "Send",
    "Sized",
    "Sync",
    "Unpin",
    "Drop",
    "Fn",
    "FnMut",
    "FnOnce",
    "Clone",
    "PartialEq",
    "PartialOrd",
    "Eq",
    "Ord",
    "AsRef",
    "AsMut",
    "Into",
    "From",
    "Default",
    "Iterator",
    "IntoIterator",
    "Extend",
    "FromIterator",
    "ToOwned",
    "ToString",
    "drop",
];

/// The names of the prelude of the 2021 edition: the crate may be of the 2018, so they are not
/// allowed.
const PRELUDE_2021: [&str; 6] = ["TryFrom", "TryInto", "Some", "None", "Ok", "Err"];

fn config(primitive: bool, prelude: bool) -> LeakerConfig {
    let mut config = LeakerConfig::new();
    if primitive {
        config.allow_primitive();
    }
    if prelude {
        config.allow_std_prelude();
    }
    config
}

/// Where the name is written, and the headline of the error where that is one.
struct Form {
    label: &'static str,
    source: String,
    /// The headline of the error without the option.
    headline: Option<String>,
    /// The entry without the option, if it is no error.
    entry: Option<String>,
}

fn forms(name: &str) -> Vec<Form> {
    vec![
        Form {
            label: "supertrait",
            source: format!("trait Tr: {} {{}}", name),
            headline: Some(format!("use absolute path: '{}'", name)),
            entry: None,
        },
        Form {
            label: "const expression",
            source: format!("trait Tr: ::p::Sup<{{ {}::X }}> {{}}", name),
            headline: Some(format!("use absolute path: '{} :: X'", name)),
            entry: None,
        },
        Form {
            label: "type",
            source: format!("trait Tr {{ fn f(&self, x: {}<()>); }}", name),
            headline: None,
            entry: Some(format!("{} < () >", name)),
        },
    ]
}

/// The entries of the Referrer, or the headline of the error.
fn analyze(config: LeakerConfig, source: &str) -> std::result::Result<Vec<String>, String> {
    let item: ItemTrait = parse_str(source).unwrap();
    let mut leaker = Leaker::from_config(config);
    leaker
        .intern_trait(&item, true)
        .map_err(|err| err.to_string())?;
    Ok(leaker
        .finish()
        .iter()
        .map(|ty| quote!(#ty).to_string())
        .collect())
}

/// `names` are allowed by the config `allowing`, and not by the other config, nor by none.
fn check(names: &[&str], allowing: impl Fn() -> LeakerConfig, others: &[LeakerConfig]) {
    for name in names {
        for form in forms(name) {
            let context = format!("{} in {}: {}", name, form.label, form.source);
            // Allowed: no error, and nothing leaked.
            assert_eq!(
                analyze(allowing(), &form.source),
                Ok(Vec::new()),
                "{}",
                context
            );
            // Not allowed: an error, or an entry for the type.
            for config in others.iter().cloned().chain([LeakerConfig::new()]) {
                let result = analyze(config, &form.source);
                match (&form.headline, &form.entry) {
                    (Some(headline), _) => assert_eq!(
                        result.as_ref().err(),
                        Some(headline),
                        "{} without the option",
                        context
                    ),
                    (_, Some(entry)) => assert_eq!(
                        result.as_ref().ok(),
                        Some(&vec![entry.clone()]),
                        "{} without the option",
                        context
                    ),
                    _ => unreachable!(),
                }
            }
        }
    }
}

#[test]
fn allow_primitive_allows_every_primitive_type() {
    assert_eq!(PRIMITIVES.iter().collect::<BTreeSet<_>>().len(), 17);
    check(&PRIMITIVES, || config(true, false), &[config(false, true)]);
}

#[test]
fn allow_std_prelude_allows_every_name_of_the_prelude() {
    assert_eq!(PRELUDE.iter().collect::<BTreeSet<_>>().len(), 31);
    check(&PRELUDE, || config(false, true), &[config(true, false)]);
}

#[test]
fn the_names_of_the_2021_prelude_are_not_allowed() {
    // Not by either option, nor by both: an edition 2018 crate has no such name.
    let both = || config(true, true);
    for name in PRELUDE_2021 {
        for form in forms(name) {
            let result = analyze(both(), &form.source);
            match (&form.headline, &form.entry) {
                (Some(headline), _) => assert_eq!(result.err().as_ref(), Some(headline)),
                (_, Some(entry)) => assert_eq!(result.ok(), Some(vec![entry.clone()])),
                _ => unreachable!(),
            }
        }
    }
}

#[test]
fn the_options_allow_those_names_and_no_others() {
    // The config says it: the paths are the names, one segment each, in the order of the lists.
    let paths = |config: LeakerConfig| -> Vec<String> {
        config
            .allowed_paths
            .iter()
            .map(|path| quote!(#path).to_string())
            .collect()
    };
    assert_eq!(paths(config(true, false)), PRIMITIVES);
    assert_eq!(paths(config(false, true)), PRELUDE);
    let both = paths(config(true, true));
    assert_eq!(both.len(), PRIMITIVES.len() + PRELUDE.len());
    // `allow_crate()` adds `crate`.
    let mut config = LeakerConfig::new();
    config.allow_crate();
    assert_eq!(paths(config), ["crate"]);
}

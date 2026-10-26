//! The names that `LeakerConfig::allow_primitive()` and `LeakerConfig::allow_prelude()` allow.
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

/// The names that `allow_prelude()` allows, with their absolute paths: those of the prelude of
/// every edition (rustc 1.56).
const PRELUDE: [(&str, &str); 36] = [
    ("Copy", "::core::marker::Copy"),
    ("Send", "::core::marker::Send"),
    ("Sized", "::core::marker::Sized"),
    ("Sync", "::core::marker::Sync"),
    ("Unpin", "::core::marker::Unpin"),
    ("Drop", "::core::ops::Drop"),
    ("Fn", "::core::ops::Fn"),
    ("FnMut", "::core::ops::FnMut"),
    ("FnOnce", "::core::ops::FnOnce"),
    ("drop", "::core::mem::drop"),
    ("Box", "::std::boxed::Box"),
    ("ToOwned", "::std::borrow::ToOwned"),
    ("Clone", "::core::clone::Clone"),
    ("PartialEq", "::core::cmp::PartialEq"),
    ("PartialOrd", "::core::cmp::PartialOrd"),
    ("Eq", "::core::cmp::Eq"),
    ("Ord", "::core::cmp::Ord"),
    ("AsRef", "::core::convert::AsRef"),
    ("AsMut", "::core::convert::AsMut"),
    ("Into", "::core::convert::Into"),
    ("From", "::core::convert::From"),
    ("Default", "::core::default::Default"),
    ("Iterator", "::core::iter::Iterator"),
    ("Extend", "::core::iter::Extend"),
    ("IntoIterator", "::core::iter::IntoIterator"),
    ("DoubleEndedIterator", "::core::iter::DoubleEndedIterator"),
    ("ExactSizeIterator", "::core::iter::ExactSizeIterator"),
    ("Option", "::core::option::Option"),
    ("Some", "::core::option::Option::Some"),
    ("None", "::core::option::Option::None"),
    ("Result", "::core::result::Result"),
    ("Ok", "::core::result::Result::Ok"),
    ("Err", "::core::result::Result::Err"),
    ("String", "::std::string::String"),
    ("ToString", "::std::string::ToString"),
    ("Vec", "::std::vec::Vec"),
];

/// The names of a later prelude (2021, 2024, rustc 1.80) and their absolute paths: the crate of the
/// copy may be of another edition or version, so no option allows them.
const PRELUDE_LATER: [(&str, &str); 9] = [
    ("TryFrom", "::core::convert::TryFrom"),
    ("TryInto", "::core::convert::TryInto"),
    ("FromIterator", "::core::iter::FromIterator"),
    ("Future", "::core::future::Future"),
    ("IntoFuture", "::core::future::IntoFuture"),
    ("size_of", "::core::mem::size_of"),
    ("size_of_val", "::core::mem::size_of_val"),
    ("align_of", "::core::mem::align_of"),
    ("align_of_val", "::core::mem::align_of_val"),
];

fn config(primitive: bool, prelude: bool) -> LeakerConfig {
    let mut config = LeakerConfig::new();
    if primitive {
        config.allow_primitive();
    }
    if prelude {
        config.allow_prelude();
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
    leaker.intern_trait(&item, true);
    let referrer = leaker.result().finish().map_err(|err| {
        err.iter()
            .map(|item| item.message.as_str())
            .collect::<Vec<_>>()
            .join("\n")
    })?;
    Ok(referrer.iter().map(|ty| quote!(#ty).to_string()).collect())
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
fn allow_prelude_allows_every_name_of_the_prelude() {
    let names: Vec<&str> = PRELUDE.iter().map(|(name, _)| *name).collect();
    assert_eq!(names.iter().collect::<BTreeSet<_>>().len(), 36);
    check(&names, || config(false, true), &[config(true, false)]);
}

#[test]
fn the_names_of_a_later_prelude_are_not_allowed() {
    // Not by either option, nor by both: a crate of an earlier edition has no such name.
    let both = || config(true, true);
    for (name, _) in PRELUDE_LATER {
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

/// The help gives the absolute path of every name that `allow_prelude()` allows or leaves out
/// (`DoubleEndedIterator` had "e.g. `::my_crate::DoubleEndedIterator`"), and of a primitive type.
#[test]
fn the_help_gives_the_absolute_path_of_every_name_of_the_prelude_and_the_primitives() {
    let primitives = PRIMITIVES.map(|name| (name, format!("::core::primitive::{}", name)));
    let known = PRELUDE
        .iter()
        .chain(PRELUDE_LATER.iter())
        .map(|(name, path)| (*name, path.to_string()))
        .chain(primitives);
    for (name, path) in known {
        let item: ItemTrait = parse_str(&format!("trait Tr: {} {{}}", name)).unwrap();
        let mut leaker = Leaker::new();
        leaker.intern_trait(&item, true);
        let result = leaker.result();
        let help = result.errors()[0].help_text();
        assert_eq!(
            help,
            Some(format!("use `{}` instead of `{}`", path, name).as_str()),
            "{}",
            name
        );
    }
}

#[test]
fn the_options_allow_those_names_and_no_others() {
    // Another name is a relative path still, with both options.
    for name in ["Foo", "Local", "Rel", "tryfrom"] {
        for form in forms(name) {
            let result = analyze(config(true, true), &form.source);
            match (&form.headline, &form.entry) {
                (Some(headline), _) => assert_eq!(result.err().as_ref(), Some(headline)),
                (_, Some(entry)) => assert_eq!(result.ok(), Some(vec![entry.clone()])),
                _ => unreachable!(),
            }
        }
    }
}

#[test]
fn allow_crate_and_allow_std_crates_allow_the_paths_that_start_with_their_roots() {
    // In a supertrait and in a type: the root decides, and the rest of the path does not.
    type Allowing = fn(&mut LeakerConfig);
    let analyze_with = |config: Allowing, source: &str| {
        let mut allowed = LeakerConfig::new();
        config(&mut allowed);
        analyze(allowed, source)
    };
    let roots: [(&str, Allowing, &[&str]); 2] = [
        (
            "crate",
            |config| {
                config.allow_crate();
            },
            &["crate::Foo", "crate::a::b::Foo"],
        ),
        (
            "std crates",
            |config| {
                config.allow_std_crates();
            },
            &["std::fmt::Debug", "core::fmt::Debug", "alloc::vec::Vec"],
        ),
    ];
    for (label, config, paths) in roots {
        for path in paths {
            for source in [
                format!("trait Tr: {} {{}}", path),
                format!("trait Tr {{ fn f(&self, x: {}<()>); }}", path),
            ] {
                // Allowed: no error, and nothing leaked.
                assert_eq!(analyze_with(config, &source), Ok(Vec::new()), "{}", source);
                // Not by another root, nor by none.
                let none = analyze_with(|_| (), &source);
                assert!(
                    none != Ok(Vec::new()),
                    "{} ({}) is leaked or an error",
                    source,
                    label
                );
            }
        }
    }
    // A root decides by its first segment only: `my::core::X` is not `core::X`, and the absolute
    // `::core::X` is context-free whatever is allowed.
    let mut config = LeakerConfig::new();
    config.allow_std_crates().allow_crate();
    let relative = analyze(
        config.clone(),
        "trait Tr { fn f(&self, x: my::core::X<()>); }",
    );
    assert_eq!(relative, Ok(vec!["my :: core :: X < () >".to_string()]));
    let absolute = analyze(LeakerConfig::new(), "trait Tr: ::core::fmt::Debug {}");
    assert_eq!(absolute, Ok(Vec::new()));
}

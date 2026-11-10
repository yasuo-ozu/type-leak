//! The absolute paths that the help of an error suggests (`ABSOLUTE_PATHS` of `src/error.rs`).
//!
//! A relative path that cannot be leaked (a bound, a const expression, a type with an elided
//! lifetime) is an error with the headline `use absolute path: '..'`, and for a name of the
//! standard prelude or a primitive type, a help that says what to write: ``use `::std::vec::Vec`
//! instead of `Vec` ``. The help is the fix that a user copies, so each of the 54 names has to give
//! the right path, and the path has to be valid.
//!
//! The table is written once, by the macro `absolute_paths!`, which makes both what the tests
//! compare with (the bare name and the absolute path as a string) and the code that uses the
//! path, as a type, a trait bound or a value: a path with a typo fails to compile, instead of
//! being suggested to a user. The table of the library cannot be read by the tests (it is
//! private), so the first test asserts the help of each name, and fails for a name whose path
//! differs from the one here.

use syn::*;
use type_leak::*;

/// `name = path, kind [what follows the path]`: `kind` is how the path is used, to check that it
/// is what the name is: `ty` (a type), `trait` (a bound), or `value` (a function or a constructor).
macro_rules! absolute_paths {
    ($($name:ident = $($seg:ident)::+, $kind:ident [$($rest:tt)*];)*) => {
        /// The bare names, and the absolute path of each as the help writes it.
        fn table() -> Vec<(&'static str, String)> {
            vec![$((
                stringify!($name),
                format!("::{}", [$(stringify!($seg)),+].join("::")),
            )),*]
        }

        /// Every path used as what the name is. It only has to compile.
        #[test]
        #[allow(unused, drop_bounds)]
        fn every_absolute_path_is_valid() {
            $( absolute_paths!(@check $kind ($($seg)::+) [$($rest)*]); )*
        }
    };
    (@check ty ($($seg:ident)::+) [$($rest:tt)*]) => {{
        fn check(_: &::$($seg)::+ $($rest)*) {}
    }};
    (@check trait ($($seg:ident)::+) [$($rest:tt)*]) => {{
        fn check<T: ::$($seg)::+ $($rest)*>() {}
    }};
    (@check value ($($seg:ident)::+) [$($rest:tt)*]) => {{
        let _ = ::$($seg)::+ $($rest)*;
    }};
}

absolute_paths! {
    Box = std::boxed::Box, ty [<u8>];
    String = std::string::String, ty [];
    Vec = std::vec::Vec, ty [<u8>];
    Option = core::option::Option, ty [<u8>];
    Result = core::result::Result, ty [<u8, u8>];
    Copy = core::marker::Copy, trait [];
    Send = core::marker::Send, trait [];
    Sized = core::marker::Sized, trait [];
    Sync = core::marker::Sync, trait [];
    Unpin = core::marker::Unpin, trait [];
    Drop = core::ops::Drop, trait [];
    Fn = core::ops::Fn, trait [(u8)];
    FnMut = core::ops::FnMut, trait [(u8)];
    FnOnce = core::ops::FnOnce, trait [(u8)];
    Clone = core::clone::Clone, trait [];
    PartialEq = core::cmp::PartialEq, trait [];
    PartialOrd = core::cmp::PartialOrd, trait [];
    Eq = core::cmp::Eq, trait [];
    Ord = core::cmp::Ord, trait [];
    AsRef = core::convert::AsRef, trait [<str>];
    AsMut = core::convert::AsMut, trait [<str>];
    Into = core::convert::Into, trait [<u8>];
    From = core::convert::From, trait [<u8>];
    Default = core::default::Default, trait [];
    Iterator = core::iter::Iterator, trait [];
    IntoIterator = core::iter::IntoIterator, trait [];
    Extend = core::iter::Extend, trait [<u8>];
    FromIterator = core::iter::FromIterator, trait [<u8>];
    ToOwned = std::borrow::ToOwned, trait [];
    ToString = std::string::ToString, trait [];
    drop = core::mem::drop, value [::<u8>];
    TryFrom = core::convert::TryFrom, trait [<u8>];
    TryInto = core::convert::TryInto, trait [<u8>];
    Some = core::option::Option::Some, value [::<u8>];
    None = core::option::Option::None, value [::<u8>];
    Ok = core::result::Result::Ok, value [::<u8, u8>];
    Err = core::result::Result::Err, value [::<u8, u8>];
    bool = core::primitive::bool, ty [];
    char = core::primitive::char, ty [];
    str = core::primitive::str, ty [];
    u8 = core::primitive::u8, ty [];
    u16 = core::primitive::u16, ty [];
    u32 = core::primitive::u32, ty [];
    u64 = core::primitive::u64, ty [];
    u128 = core::primitive::u128, ty [];
    usize = core::primitive::usize, ty [];
    i8 = core::primitive::i8, ty [];
    i16 = core::primitive::i16, ty [];
    i32 = core::primitive::i32, ty [];
    i64 = core::primitive::i64, ty [];
    i128 = core::primitive::i128, ty [];
    isize = core::primitive::isize, ty [];
    f32 = core::primitive::f32, ty [];
    f64 = core::primitive::f64, ty [];
}

/// The item that has the bare `name` where it cannot be leaked, the headline of its error, and
/// the name of the way it is written. The name is the first segment of the offending path.
fn forms(name: &str) -> Vec<(&'static str, String, String)> {
    vec![
        // The value of a const argument of a bound.
        (
            "const expression",
            format!("trait Tr: ::p::Sup<{{ {} }}> {{}}", name),
            format!("use absolute path: '{}'", name),
        ),
        // The path of a const expression with more segments.
        (
            "path of a const expression",
            format!("trait Tr: ::p::Sup<{{ {}::X }}> {{}}", name),
            format!("use absolute path: '{} :: X'", name),
        ),
        // The supertrait.
        (
            "supertrait",
            format!("trait Tr: {} {{}}", name),
            format!("use absolute path: '{}'", name),
        ),
        // A type with an elided lifetime in an argument, which cannot be leaked as a whole.
        (
            "type with an elided lifetime",
            format!("trait Tr {{ fn f(&self, x: {}<&'_ ()>); }}", name),
            format!("use absolute path: '{} < & '_ () >'", name),
        ),
    ]
}

#[test]
fn a_bare_name_gets_the_help_with_its_absolute_path() {
    let table = table();
    assert_eq!(table.len(), 54);
    for (name, absolute) in &table {
        let help = format!("use `{}` instead of `{}`", absolute, name);
        for (form, source, message) in forms(name) {
            let item: ItemTrait = parse_str(&source).unwrap();
            let mut leaker = Leaker::new();
            leaker.intern_trait(&item, true);
            let result = leaker.result();
            let err = result.errors();
            assert!(!err.is_empty(), "`{}` is accepted: {}", name, source);
            let context = format!("{} as a {}: {}", name, form, source);
            let items = err;
            assert_eq!(items.len(), 1, "{}", context);
            assert_eq!(items[0].message, message, "{}", context);
            // It is the help of the error, and of the first one. It is no note: the notes say
            // why.
            assert_eq!(items[0].help_text(), Some(help.as_str()), "{}", context);
            assert_eq!(err[0].help_text(), Some(help.as_str()), "{}", context);
            assert!(
                items[0].notes.iter().all(|(_, note)| *note != help),
                "{}",
                context
            );
            // And it is in the `syn::Error`, which stable Rust gets: the notes follow the headline,
            // then the help.
            let rendered = to_syn_error(err).to_string();
            assert!(rendered.starts_with(&message), "{}: {}", context, rendered);
            assert!(
                rendered.ends_with(&format!("\n  = help: {}\n", help)),
                "{}: {}",
                context,
                rendered
            );
        }
    }
}

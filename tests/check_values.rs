//! `check_values` and `check_bodies`: the Leaker reads the values of an item (the default of an
//! associated const, the value of a const or a static) with the first and the bodies of its
//! functions with the second, and rejects a path that depends on where it is written. A value is
//! copied as written, so it leaks nothing but the types in a type position (a cast, a `let`, a closure, a turbofish, a qualified path), which the
//! Visitor replaces: see `types_in_type_positions_are_leaked()` and, for the copy,
//! `tests/findings.rs` (F40).
//!
//! The check must not reject what means the same everywhere, so it follows the scopes of a body:
//! bindings, parameters, closure parameters, patterns, and the items and `use` declarations of a
//! block are local, and so are the generics and `Self` in scope. Absolute paths and allowed paths
//! are not relative. Most of the tests below are tables of bodies, each accepted or rejected.

mod common;

use common::{starts_with, to_syn_error, Recovered};
use syn::*;
use template_quote::quote;
use type_leak::*;

/// The rules of the tests below, which all read the values and the bodies (`check_values` and
/// `check_bodies`) unless they say not.
fn config() -> Rules<impl Allow + Clone> {
    Rules::new().primitive().prelude()
}

/// The messages of the errors of a trait whose method `f` has the body `body`, or none. The trait
/// has the generics `T` and `N`, and the method `U` and the argument `p: u8`.
fn body_messages(body: &str) -> Vec<String> {
    let source = format!(
        "trait Tr<T: ::core::default::Default, const N: usize> {{
            fn g(&self) {{}}
            fn f<U: ::core::default::Default>(&self, p: u8) {{ {} }}
        }}",
        body
    );
    messages_of(config(), &source)
}

fn messages_of<A: Allow>(config: A, source: &str) -> Vec<String> {
    let item: ItemTrait = parse_str(source).unwrap_or_else(|e| panic!("{}: {}", source, e));
    let mut leaker = Leaker::with_rule(config);
    leaker.check_values = true;
    leaker.check_bodies = true;
    leaker.intern_trait(&item);
    leaker
        .result()
        .errors()
        .iter()
        .map(|item| item.message.clone())
        .collect()
}

fn rejected(path: &str) -> Vec<String> {
    vec![format!("use absolute path: '{}'", path)]
}

#[test]
fn bindings_parameters_and_generics_are_local() {
    for body in [
        // the argument, a `let`, a closure and its parameters
        "let x: u8 = p + 1; let _ = x;",
        "let c = |a: u8, b| a + b; let _ = c(1, 2);",
        "let _ = || -> u8 { p };",
        "let c = |(a, b): (u8, u8)| a + b; let _ = c((1, 2));",
        // shadowing: the initializer sees the binding it shadows
        "let x = 1u8; let x = x + 1; let _ = x;",
        "let x = 1u8; { let x = x + 1; let _ = x; } let _ = x;",
        // generics of the trait and of the method, `Self`, and `Self::` paths
        "let _ = T::default(); let _ = U::default(); let _ = [0u8; N];",
        "let v: Vec<U> = Vec::new(); let _ = v;",
        "let _ = <T as ::core::default::Default>::default();",
        "let _ = Self::g(self); let _ = <Self>::g(self); let _ = <Self as ::p::Other>::h(self);",
        // patterns
        "let (a, b) = (1u8, 2u8); let _ = a + b;",
        "let [a, b] = [1u8, 2]; let _ = (a, b);",
        "let r = &p; let _ = *r as u32;",
        "let r#type = 1u8; let _ = r#type;",
        // control flow
        "match ::core::option::Option::Some(p) { ::core::option::Option::Some(v) if v > 0 => { let _ = v; } ::core::option::Option::Some(_) | ::core::option::Option::None => {} }",
        "if let ::core::option::Option::Some(x) = ::core::option::Option::Some(p) { let _ = x; }",
        "while let ::core::option::Option::Some(x) = ::core::option::Option::Some(p) { let _ = x; break; }",
        "for (i, x) in [1u8, 2].iter().enumerate() { let _ = (i, x); }",
        "'a: loop { break 'a; }",
        "unsafe { let _ = p; }",
        // literals, casts and the types of the primitives and the prelude
        "let _: [u8; 2] = [0; 2]; let _ = <Vec<u8>>::new(); let _ = p as u32;",
        // a struct literal with a shorthand, a struct pattern
        "struct P { a: u8, b: u8 } let a = 1; let _ = P { a, b: 2 };",
        "let Pt { x, y: yy } = Pt { x: 1, y: 2 }; let _ = (x, yy); struct Pt { x: u8, y: u8 }",
    ] {
        assert_eq!(body_messages(body), Vec::<String>::new(), "{}", body);
    }
}

#[test]
fn the_items_of_a_block_are_local() {
    for body in [
        "fn helper(a: u8) -> u8 { a } let _ = helper(1);",
        "struct L(u8); let _ = L(1);",
        "const C: u8 = 1; static S: u8 = 2; let _ = C + S;",
        // an item is visible from the start of its block
        "let _ = D + 1; const D: u8 = 1;",
        "let _ = helper(); fn helper() -> u8 { 1 }",
        // an item with generics and an impl of its own
        "struct L<V>(V); impl<V> L<V> { fn g(v: V) -> V { v } } let _ = L::g(1u8);",
        "fn id<V>(v: V) -> V { let w: V = v; w } let _ = id(1);",
        // a nested module declares its own names
        "mod inner { pub fn a() {} fn b() { a(); } } inner::a();",
        "mod inner { pub struct S; pub mod deeper { pub fn f() {} } } let _ = inner::S; inner::deeper::f();",
        // `use` of an absolute path declares the name
        "use ::core::cmp::max; let _ = max(1, 2);",
        "use ::core::cmp::{self, min as lo}; let _ = (cmp::min(1, 2), lo(1, 2));",
        "use ::core::fmt::Write as _;",
        // a nested function sees the items around it
        "fn a() -> u8 { 1 } fn b() -> u8 { a() } let _ = b();",
        // the arguments of a nested function are its own
        "fn h(q: u8) -> u8 { q } let _ = h(p);",
    ] {
        assert_eq!(body_messages(body), Vec::<String>::new(), "{}", body);
    }
}

#[test]
fn a_name_is_local_only_where_it_is_in_scope() {
    for (body, path) in [
        ("{ let z = 1u8; } let _ = z;", "z"),
        ("let c = |a: u8| a; let _ = a;", "a"),
        ("match p { 0 => { let z = 1u8; let _ = z; } _ => { let _ = z; } }", "z"),
        (
            "if let ::core::option::Option::Some(x) = ::core::option::Option::Some(p) { let _ = x; } else { let _ = x; }",
            "x",
        ),
        ("for i in 0..2u8 { let _ = i; } let _ = i;", "i"),
        // the initializer does not see the binding it makes
        ("let q = q;", "q"),
        // a nested function does not see the bindings or the arguments around it
        ("fn inner() { let _ = p; } inner();", "p"),
        ("let outer = 1u8; fn inner() { let _ = outer; } inner();", "outer"),
        // the items of a block are not visible outside it
        ("{ fn helper() {} } helper();", "helper"),
        // `Self` and the generics are not names to declare: another name is relative
        ("let _ = V::default();", "V::default"),
    ] {
        assert_eq!(body_messages(body), rejected(path), "{}", body);
    }
}

#[test]
fn a_relative_path_is_rejected_wherever_it_is() {
    for (body, path) in [
        ("rel();", "rel"),
        ("let _ = Rel::new();", "Rel::new"),
        ("let _ = size_of::<u8>();", "size_of::<u8>"),
        ("let _ = <u8 as Rel>::f();", "Rel"),
        // the arguments of a path whose own path is relative are not read
        ("let _ = Vec2::<Rel>::new();", "Vec2::<Rel>::new"),
        ("let _ = Rel { a: 1 };", "Rel"),
        ("let Rel { a } = x; let _ = a;", "Rel"),
        ("let _ = || Rel::new();", "Rel::new"),
        ("let _ = [Rel::new()];", "Rel::new"),
        ("let _ = crate::f();", "crate::f"),
        ("let _ = super::f();", "super::f"),
        ("let _ = self::f();", "self::f"),
        ("fn inner() -> Rel { todo() } ", "Rel"),
        ("struct L(Rel);", "Rel"),
        ("impl Rel for L {}", "Rel"),
    ] {
        let got = body_messages(body);
        assert!(
            got.contains(&format!("use absolute path: '{}'", path)),
            "{}: {:?}",
            body,
            got
        );
    }
}

/// The entries of a trait whose method `f` has the body `body` (see `body_messages()`).
fn body_entries(body: &str) -> Vec<String> {
    let source = format!(
        "trait Tr<T: ::core::default::Default, const N: usize> {{
            fn g(&self) {{}}
            fn f<U: ::core::default::Default>(&self, p: u8) {{ {} }}
        }}",
        body
    );
    let item: ItemTrait = parse_str(&source).unwrap_or_else(|e| panic!("{}: {}", source, e));
    let mut leaker = Leaker::with_rule(config());
    leaker.check_values = true;
    leaker.check_bodies = true;
    leaker.intern_trait(&item);
    let referrer = leaker
        .result()
        .finish()
        .unwrap_or_else(|err| panic!("{}: {}", body, to_syn_error(&err)));
    let mut entries: Vec<String> = common::types(&referrer)
        .map(|ty| quote!(#ty).to_string())
        .collect();
    entries.sort();
    entries
}

/// A type in a type position of a value is an entry, not an error: the cases that
/// `a_relative_path_is_rejected_wherever_it_is()` used to hold (F40 of `tests/findings.rs`).
#[test]
fn types_in_type_positions_are_leaked() {
    for (body, entries) in [
        (
            "let _ = ::core::mem::size_of::<inner::Local>();",
            vec!["inner :: Local"],
        ),
        (
            "let _: Rel2 = ::core::default::Default::default();",
            vec!["Rel2"],
        ),
        // a bare identifier is read as a type, as in an enum discriminant
        ("let _ = Vec::<Rel>::new();", vec!["Rel"]),
        ("let _ = Self::g(self).h::<Rel>();", vec!["Rel"]),
        (
            "let v: ::std::vec::Vec<Rel> = ::std::vec::Vec::new(); let _ = v;",
            vec![":: std :: vec :: Vec < Rel >"],
        ),
        ("let _ = p as Rel;", vec!["Rel"]),
        (
            "let _ = <Rel as ::core::default::Default>::default();",
            vec!["Rel"],
        ),
        // the generics in scope make the context of the entry, and need no entry of their own
        (
            "let _: inner::Wrap<(T, U, [u8; N])> = ::core::unimplemented!();",
            vec!["inner :: Wrap < (T , U , [u8 ; N]) >"],
        ),
        ("let _: Vec<(T, U)> = ::core::unimplemented!();", vec![]),
    ] {
        assert_eq!(body_entries(body), entries, "{}", body);
    }
}

#[test]
fn every_relative_path_of_a_value_is_reported() {
    assert_eq!(
        body_messages("let a = Rel::new(); let b = other(); let _ = (a, b);"),
        [
            "use absolute path: 'Rel::new'",
            "use absolute path: 'other'"
        ]
    );
}

#[test]
fn the_std_prelude_and_the_primitives_are_allowed_only_if_asked() {
    let source = "trait Tr { fn f(&self, p: u8) -> Vec<u8> { let v: Vec<u8> = Vec::new(); v } }";
    // Allowed by both options.
    assert_eq!(messages_of(config(), source), Vec::<String>::new());
    // Not by none: the primitives and the prelude are relative names, as in a type. The path
    // `Vec::new` is rejected; the type `Vec<u8>` of the `let` is leaked, as the one of the
    // signature is.
    let none = Rules::new();
    assert_eq!(
        messages_of(none.clone(), source),
        ["use absolute path: 'Vec::new'"]
    );
    fn entries<A: Allow>(config: A, source: &str) -> Vec<String> {
        let item: ItemTrait = parse_str(source).unwrap();
        let mut leaker = Leaker::with_rule(config);
        leaker.check_values = true;
        leaker.check_bodies = true;
        leaker.intern_trait(&item);
        let mut entries: Vec<String> = leaker
            .result()
            .finish()
            .unwrap()
            .iter()
            .map(|(_, ty, _)| quote!(#ty).to_string())
            .collect();
        entries.sort();
        entries
    }
    let types = "trait Tr { fn f(&self, p: u8) -> Vec<u8> { let v: Vec<u8> = ::core::unimplemented!(); v } }";
    assert_eq!(entries(none, types), ["Vec < u8 >", "u8"]);
    // Each option allows its own names: `Vec` or `u8`.
    let only_primitive = || Rules::new().primitive();
    assert_eq!(
        messages_of(only_primitive(), source),
        ["use absolute path: 'Vec::new'"]
    );
    assert_eq!(entries(only_primitive(), types), ["Vec < u8 >"]);
    let only_prelude = || Rules::new().prelude();
    assert_eq!(messages_of(only_prelude(), source), Vec::<String>::new());
    assert_eq!(entries(only_prelude(), types), ["Vec < u8 >", "u8"]);
}

/// The trait's own name is a relative path, like any other name of the module of the item.
#[test]
fn the_name_of_the_item_itself_is_relative() {
    assert_eq!(
        messages_of(
            config(),
            "trait Tr { fn g(&self); fn f(&self) { Tr::g(self) } }"
        ),
        rejected("Tr::g")
    );
}

/// `Some`, `None`, `Ok` and `Err` are in the prelude of every edition, as `drop` is: `Rules::prelude()`
/// allows them in an expression and a pattern, and without it they are relative paths.
#[test]
fn the_variants_of_the_prelude_are_allowed_in_a_value() {
    for body in [
        "let _ = Some(p);",
        "let _ = None::<u8>;",
        "let _: Result<u8, u8> = Ok(p);",
        "let _ = Err::<u8, u8>(p);",
        "match Some(p) { Some(v) => { let _ = v; } None => {} }",
        "match Ok::<u8, u8>(p) { Ok(v) | Err(v) => { let _ = v; } }",
        "if let Some(x) = Some(p) { let _ = x; } else { drop(p); }",
        "let _ = Option::Some(p);",
    ] {
        assert_eq!(body_messages(body), Vec::<String>::new(), "{}", body);
    }
    let none = Rules::new();
    for (path, body) in [
        ("Some", "let _ = Some(1u8);"),
        ("None::<u8>", "let _ = None::<u8>;"),
        ("Ok::<u8, u8>", "let _ = Ok::<u8, u8>(1);"),
        ("Err::<u8, u8>", "let _ = Err::<u8, u8>(1);"),
    ] {
        let source = format!("trait Tr {{ fn f(&self) {{ {} }} }}", body);
        let got = messages_of(none.clone(), &source);
        assert_eq!(got, rejected(path), "{}", body);
    }
}

#[test]
fn the_names_of_a_later_prelude_are_relative_in_a_value() {
    // Like in a type: a crate of an earlier edition has no `TryFrom` (see `allow_prelude`), and
    // `size_of` is in the prelude only since rustc 1.80.
    for (body, path, first, absolute) in [
        (
            "let _: u32 = TryFrom::try_from(p).unwrap();",
            "TryFrom::try_from",
            "TryFrom",
            "::core::convert::TryFrom",
        ),
        (
            "let _ = size_of::<u8>();",
            "size_of::<u8>",
            "size_of",
            "::core::mem::size_of",
        ),
    ] {
        assert_eq!(body_messages(body), rejected(path), "{}", body);
        let item: ItemTrait =
            parse_str(&format!("trait Tr {{ fn f(&self, p: u8) {{ {} }} }}", body)).unwrap();
        let mut leaker = Leaker::with_rule(config());
        leaker.check_values = true;
        leaker.check_bodies = true;
        leaker.intern_trait(&item);
        let result = leaker.result();
        assert_eq!(
            result.errors()[0].help_text(),
            Some(format!("use `{}` instead of `{}`", absolute, first).as_str())
        );
        // An allowed path is accepted.
        let config = config().or_if(starts_with(parse_str(first).unwrap()));
        let mut leaker = Leaker::with_rule(config);
        leaker.check_values = true;
        leaker.check_bodies = true;
        leaker.intern_trait(&item);
        assert!(leaker.result().finish().is_ok(), "{}", body);
    }
}

#[test]
fn a_primitive_name_in_a_value_is_assumed() {
    let item: ItemTrait = parse_quote!(
        trait Tr {
            const N: u8 = 1 as u8;
        }
    );
    let mut leaker = Leaker::with_rule(config());
    leaker.check_values = true;
    leaker.intern_trait(&item);
    assert!(
        leaker.assertions().to_string().contains("u8"),
        "{}",
        leaker.assertions()
    );
    assert!(leaker.result().errors().is_empty());
}

// ── macros ───────────────────────────────────────────────────────────────────────────────────

#[test]
fn a_macro_is_checked_by_its_path_and_its_arguments() {
    for body in [
        "::std::println!(\"{}\", p);",
        "::std::println!(\"{p}\");",
        "::std::println!(\"{p:?} {p:>5}\");",
        "let w = 5; ::std::println!(\"{:w$} {:>1$}\", 1, 2);",
        "::std::println!(\"{{FOO}}\");",
        "::std::println!(\"{x} {}\", p, x = 1);",
        "let _ = ::std::format!(\"{}\", p);",
        "let _ = ::std::vec![0u8; 3];",
        "let _ = ::std::vec![p, 2];",
        "::core::assert_eq!(p, 1, \"message {}\", p);",
        "let _ = ::core::matches!(p, 1 | 2);",
        "let _ = ::core::matches!(::core::option::Option::Some(p), ::core::option::Option::Some(x) if x > 0);",
        "if ::core::cfg!(unix) {}",
        "let _ = ::core::stringify!(anything at all);",
        "::core::todo!();",
        "::core::unreachable!(\"{}\", p);",
    ] {
        assert_eq!(body_messages(body), Vec::<String>::new(), "{}", body);
    }
}

#[test]
fn a_relative_macro_path_says_what_to_write() {
    let item: ItemTrait = parse_quote!(
        trait Tr {
            fn f(&self) {
                println!("hi");
            }
        }
    );
    let mut leaker = Leaker::with_rule(config());
    leaker.check_values = true;
    leaker.check_bodies = true;
    leaker.intern_trait(&item);
    let result = leaker.result();
    let err = result.errors();
    let items = &err;
    assert_eq!(items.len(), 1);
    assert_eq!(items[0].message, "use absolute path: 'println!'");
    assert_eq!(
        items[0].help_text(),
        Some("use `::std::println!` instead of `println!`")
    );
    // A macro of the crate: how to write it.
    let item: ItemTrait = parse_quote!(
        trait Tr {
            fn f(&self) {
                my_mac!(1);
            }
        }
    );
    let mut leaker = Leaker::with_rule(config());
    leaker.check_values = true;
    leaker.check_bodies = true;
    leaker.intern_trait(&item);
    let result = leaker.result();
    let err = result.errors();
    assert_eq!(
        err[0].help_text(),
        Some("write `my_mac!` with an absolute path (e.g. `::my_crate::my_mac!`)")
    );
}

#[test]
fn a_name_in_a_format_string_is_a_path() {
    assert_eq!(
        body_messages("::std::println!(\"{FOO} {p}\");"),
        rejected("FOO")
    );
    assert_eq!(
        body_messages("::std::println!(\"{:width$}\", 1);"),
        rejected("width")
    );
    // An argument of the macro by name is no path.
    assert_eq!(
        body_messages("::std::println!(\"{FOO}\", FOO = 1);"),
        Vec::<String>::new()
    );
}

/// The format string of a macro of the standard library is the argument at its place; any string
/// literal of a macro of the user may be one.
#[test]
fn only_a_format_string_captures_names() {
    for body in [
        // not format strings: `{FOO}` is text
        "::core::assert_eq!(\"{FOO}\", \"{FOO}\");",
        "::core::assert_eq!(p, 1, \"{p}\");",
        "::core::assert!(p > 0, \"{p}\");",
        "::std::dbg!(\"{FOO}\");",
        "let _ = ::core::concat!(\"{FOO}\", \"b\");",
        "::std::println!(\"{}\", \"{FOO}\");",
        "::core::write!(::std::io::stdout(), \"{p}\").unwrap();",
        "::core::panic!(\"{p}\");",
        "::core::todo!(\"{p}\");",
    ] {
        assert_eq!(body_messages(body), Vec::<String>::new(), "{}", body);
    }
    for (body, name) in [
        ("::core::assert_eq!(1, 1, \"{FOO}\");", "FOO"),
        ("::core::assert!(true, \"{FOO}\");", "FOO"),
        (
            "::core::write!(::std::io::stdout(), \"{FOO}\").unwrap();",
            "FOO",
        ),
        ("::core::panic!(\"{FOO}\");", "FOO"),
        ("::std::eprintln!(\"{FOO}\");", "FOO"),
        ("let _ = ::std::format!(\"{FOO:?}\");", "FOO"),
        // a macro of the user: every literal may be a format string
        ("::p::mac!(1, \"{FOO}\");", "FOO"),
    ] {
        assert_eq!(body_messages(body), rejected(name), "{}", body);
    }
}

#[test]
fn what_the_leaker_cannot_read_is_rejected() {
    // A macro whose arguments are not expressions, a `macro_rules!`, and a glob import.
    for (body, headline) in [
        (
            "::p::mac!(a b c);",
            "cannot analyze `::p::mac!(a b c)` in the default body of `f`",
        ),
        (
            "macro_rules! m { () => {} }",
            "cannot analyze `macro_rules ! m { () => { } }` in the default body of `f`",
        ),
        (
            "use ::core::fmt::*;",
            "cannot analyze `use::core::fmt::*;` in the default body of `f`",
        ),
    ] {
        assert_eq!(body_messages(body), [headline], "{}", body);
    }
    let item: ItemTrait = parse_quote!(
        trait Tr {
            fn f(&self) {
                ::p::mac!(a b c);
            }
        }
    );
    let mut leaker = Leaker::with_rule(config());
    leaker.check_values = true;
    leaker.check_bodies = true;
    leaker.intern_trait(&item);
    let result = leaker.result();
    let err = result.errors();
    assert_eq!(
        err[0].notes.first().map(|(_, n)| n.as_str()),
        Some(
            "the arguments of the macro are not expressions. Every path in the default body of \
             `f` must be readable, since the copy writes it as it is"
        )
    );
    assert_eq!(
        err[0].help_text(),
        Some("write it with paths and expressions only, or move it out of the item")
    );
}

// ── the items ────────────────────────────────────────────────────────────────────────────────

#[test]
fn an_associated_const_default_is_read() {
    for source in [
        "trait Tr { const A: usize = 1 + 2; }",
        "trait Tr { const A: usize = 1; const B: usize = Self::A * 2; }",
        "trait Tr<const N: usize> { const A: usize = N; }",
        "trait Tr<T: ::p::HasLen> { const A: usize = T::LEN; }",
        "trait Tr<T: ::p::HasLen> { const A: usize = <T as ::p::HasLen>::LEN; }",
        "trait Tr { const A: usize = ::core::mem::size_of::<u64>(); }",
        "trait Tr { const A: Option<u8> = ::core::option::Option::None; }",
        "trait Tr { const A: u8; }",
    ] {
        assert_eq!(
            messages_of(config(), source),
            Vec::<String>::new(),
            "{}",
            source
        );
    }
    for (source, path) in [
        ("trait Tr { const A: usize = inner::X; }", "inner::X"),
        ("trait Tr { const A: Rel = Rel::new(); }", "Rel::new"),
        ("trait Tr { const A: usize = <u8 as Rel>::N; }", "Rel"),
        (
            "trait Tr { const A: usize = { let x = 1; x + rel() }; }",
            "rel",
        ),
    ] {
        assert_eq!(messages_of(config(), source), rejected(path), "{}", source);
    }
}

/// The note says which value the path is in.
#[test]
fn the_note_says_which_value() {
    let notes = |source: &str| -> Vec<String> {
        let file: File = parse_str(source).unwrap();
        let mut leaker = Leaker::with_rule(config());
        leaker.check_values = true;
        leaker.check_bodies = true;
        for item in &file.items {
            match item {
                Item::Trait(i) => {
                    leaker.intern_trait(i);
                }
                Item::Impl(i) => {
                    leaker.intern_impl(i);
                }
                _ => {
                    leaker.enter_generics(&Generics::default(), |view| view.visit_item(item));
                }
            }
        }
        let result = leaker.result();
        assert!(!result.errors().is_empty(), "accepted: {}", source);
        result
            .errors()
            .iter()
            .map(|item| {
                item.notes
                    .first()
                    .map(|(_, n)| n.as_str())
                    .unwrap()
                    .to_string()
            })
            .collect()
    };
    let note = |what: &str| format!("it is in {}, which the copy writes as it is", what);
    assert_eq!(
        notes("trait Tr { const A: usize = r1(); fn f(&self) { r2() } fn g(&self); }"),
        [
            note("the default value of `A`"),
            note("the default body of `f`")
        ]
    );
    assert_eq!(
        notes("impl ::p::Tr for ::p::S { const A: usize = r1(); fn f(&self) { r2() } }"),
        [note("the value of `A`"), note("the body of `f`")]
    );
    assert_eq!(
        notes("pub const A: usize = r1(); pub static B: usize = r2(); pub fn f() { r3() }"),
        [
            note("the value of `A`"),
            note("the value of `B`"),
            note("the body of `f`")
        ]
    );
}

#[test]
fn an_impl_is_read() {
    let item: ItemImpl = parse_quote! {
        impl<T: ::core::default::Default> ::p::Tr for ::p::S<T> {
            const N: usize = 1;
            fn f(&self) -> T { let t = T::default(); t }
            fn g(&self) -> usize { Self::N }
        }
    };
    let mut leaker = Leaker::with_rule(config());
    leaker.check_values = true;
    leaker.check_bodies = true;
    leaker.intern_impl(&item);
    assert!(leaker.result().errors().is_empty());

    let item: ItemImpl = parse_quote! {
        impl ::p::Tr for ::p::S {
            const N: usize = Rel::N;
            fn f(&self) -> u8 { Rel::f() }
        }
    };
    let mut leaker = Leaker::with_rule(config());
    leaker.check_values = true;
    leaker.check_bodies = true;
    leaker.intern_impl(&item);
    let result = leaker.result();
    let messages: Vec<String> = result
        .errors()
        .iter()
        .map(|item| item.message.clone())
        .collect();
    assert_eq!(
        messages,
        ["use absolute path: 'Rel::N'", "use absolute path: 'Rel::f'"]
    );
}

#[test]
fn a_nested_item_is_read_through_the_view() {
    let mut leaker = Leaker::with_rule(config());
    leaker.check_values = true;
    leaker.check_bodies = true;
    leaker.enter_generics(&parse_quote!(<T: ::core::default::Default>), |view| {
        view.visit_item(&parse_quote!(
            const N: usize = 1;
        ));
        view.visit_item(&parse_quote!(
            fn f() -> T {
                T::default()
            }
        ));
    });
    assert!(leaker.result().errors().is_empty());
    let mut leaker = Leaker::with_rule(config());
    leaker.check_values = true;
    leaker.check_bodies = true;
    leaker.enter_generics(&parse_quote!(<T>), |view| {
        view.visit_item(&parse_quote!(
            fn f<V>() -> usize {
                Rel::f::<T, V>()
            }
        ));
    });
    let result = leaker.result();
    let messages: Vec<String> = result
        .errors()
        .iter()
        .map(|item| item.message.clone())
        .collect();
    assert_eq!(messages, ["use absolute path: 'Rel::f::<T, V>'"]);
}

/// A scope with a bad value has an error, as for a bad type, and keeps what it found: the list of
/// the call has it once.
#[test]
fn a_bad_value_is_an_error_of_its_scope_and_the_one_around_it() {
    let mut leaker = Leaker::with_rule(config());
    leaker.check_values = true;
    leaker.enter_generics(&Generics::default(), |view| {
        view.visit_type(&parse_quote!(Wrap<u8>));
        view.enter_generics(&Generics::default(), |view| {
            view.visit_type(&parse_quote!(Other<u8>));
            view.visit_item(&parse_quote!(
                const N: usize = rel();
            ));
        });
    });
    let result = leaker.result();
    assert_eq!(result.errors().len(), 1);
    assert_eq!(result.errors()[0].message, "use absolute path: 'rel'");
    // `Other<u8>` of the scope with the bad value stays, like `Wrap<u8>`.
    let entries: Vec<String> = result
        .recovered()
        .iter()
        .map(|(_, ty, _)| quote!(#ty).to_string())
        .collect();
    assert_eq!(entries, ["Other < u8 >", "Wrap < u8 >"]);
}

/// `Self` where it has no meaning is still `SelfWithoutMeaning`, also in a value.
#[test]
fn self_without_meaning_in_a_value() {
    let item: Item = parse_quote!(
        pub const X: usize = Self::N;
    );
    fn analyze<A: Allow>(leaker: &mut Leaker<A>, item: &Item) {
        leaker.enter_generics(&Generics::default(), |view| view.visit_item(item));
    }
    let mut leaker = Leaker::with_rule(config());
    leaker.check_values = true;
    analyze(&mut leaker, &item);
    let result = leaker.result();
    let err = result.errors();
    assert!(
        !err.is_empty()
            && err
                .iter()
                .all(|item| item.reason == Reason::SelfWithoutMeaning),
        "{:?}",
        err
    );
    // The value is not read without the option.
    let plain = Rules::new().primitive();
    let mut leaker = Leaker::with_rule(plain);
    analyze(&mut leaker, &item);
    let result = leaker.result();
    assert!(result.finish().is_ok());
}

// ── what the option does not change ──────────────────────────────────────────────────────────

#[test]
fn the_types_are_leaked_as_ever() {
    let source: ItemTrait = parse_quote! {
        trait Tr<T> {
            const A: usize = 1;
            fn f(&self, t: T) -> Wrap<T> { ::core::unimplemented!() }
            fn g(&self) -> Rel;
        }
    };
    fn entries<A: Allow>(config: A, check_values: bool, source: &ItemTrait) -> Vec<String> {
        let mut leaker = Leaker::with_rule(config);
        leaker.check_values = check_values;
        leaker.check_bodies = check_values;
        leaker.intern_trait(source);
        leaker
            .result()
            .finish()
            .unwrap()
            .iter()
            .map(|(_, ty, _)| quote!(#ty).to_string())
            .collect()
    }
    let without = Rules::new().primitive().prelude();
    assert_eq!(
        entries(config(), true, &source),
        entries(without, false, &source)
    );
    let mut found = entries(config(), true, &source);
    found.sort();
    assert_eq!(found, ["Rel", "Wrap < T >"]);
}

/// Code as it is written: nothing in it is relative.
#[test]
fn ordinary_bodies_are_accepted() {
    let item: ItemTrait = parse_quote! {
        pub trait Shape<T: ::core::clone::Clone + ::core::fmt::Debug> where T: 'static {
            const SIDES: usize = 0;
            const NAME: &'static str = "shape";
            fn area(&self) -> f64;
            fn describe(&self) -> String {
                let mut s = String::new();
                s.push_str(Self::NAME);
                for i in 0..Self::SIDES {
                    s.push_str(&i.to_string());
                }
                let n = self.area() as i64;
                if n > 0 { s.push('+') } else { s.push('-') }
                s
            }
            fn longest<'a>(&self, a: &'a str, b: &'a str) -> &'a str {
                if a.len() > b.len() { a } else { b }
            }
            fn sum<I: Iterator<Item = u32>>(&self, it: I) -> u32 {
                it.fold(0, |acc, x| acc + x)
            }
            fn parse(&self, text: &str) -> Result<u8, ::core::num::ParseIntError> {
                let value: u8 = text.trim().parse()?;
                Ok(value)
            }
            fn first(&self, items: &[T]) -> Option<T> {
                let mut found = None;
                for item in items {
                    if found.is_none() {
                        found = Some(item.clone());
                    }
                }
                found
            }
            fn nested(&self) -> Vec<Vec<u8>> {
                let make = |n: usize| -> Vec<u8> { vec![0; n] };
                (0..3).map(make).collect()
            }
        }
    };
    // The prelude of the 2021 edition (`Ok`, `Some`) and the macro `vec!` are not among the allowed
    // names: allow them one by one.
    let names: Vec<Path> = ["Ok", "Some", "None", "vec"]
        .iter()
        .map(|name| Ident::new(name, proc_macro2::Span::call_site()).into())
        .collect();
    let config = config().or_if(move |cause: &Cause, place: Place| {
        names
            .iter()
            .any(|name| starts_with(name.clone())(cause, place))
    });
    let mut leaker = Leaker::with_rule(config);
    leaker.check_values = true;
    leaker.check_bodies = true;
    leaker.intern_trait(&item);
    // The types of the signatures are in the prelude: nothing is leaked.
    assert_eq!(leaker.result().finish().unwrap().iter().count(), 0);
}

/// `check_values` and `check_bodies` are off by default, whatever rules the Leaker is given.
#[test]
fn check_values_is_off_by_default() {
    let item: ItemTrait = parse_quote!(
        trait Tr {
            const A: usize = rel();
            fn f(&self) {
                rel()
            }
        }
    );
    let mut leaker = Leaker::new();
    assert!(!leaker.check_values && !leaker.check_bodies);
    leaker.intern_trait(&item);
    let result = leaker.result();
    assert!(result.finish().is_ok());
    let mut leaker = Leaker::with_rule(Rules::new());
    assert!(!leaker.check_values && !leaker.check_bodies);
    leaker.intern_trait(&item);
    let result = leaker.result();
    assert!(result.finish().is_ok());
    let mut leaker = Leaker::with_rule(Rules::<()>::default());
    leaker.intern_trait(&item);
    let result = leaker.result();
    assert!(result.finish().is_ok());
    // Nothing is leaked where the types are allowed: like method bodies, the defaults of associated
    // consts are not analyzed.
    let mut leaker = Leaker::with_rule(Rules::new().primitive().prelude());
    leaker.intern_trait(&item);
    assert!(leaker.result().finish().unwrap().is_empty());
    let mut leaker = Leaker::with_rule(Rules::new());
    leaker.check_values = true;
    leaker.check_bodies = true;
    leaker.intern_trait(&item);
    let result = leaker.result();
    assert!(result.finish().is_err());
}

/// `check_values` is read by each outermost call, so a Leaker can switch it on and off between
/// calls: only a call made while it is on reads the values (its errors and its entries).
#[test]
fn check_values_is_read_by_each_call() {
    let item = |name: &str, suffix: &str| -> ItemTrait {
        parse_str(&format!(
            "trait {} {{ fn f(&self) {{ let _: Entry{} = rel{}(); }} }}",
            name, suffix, suffix
        ))
        .unwrap()
    };
    let mut leaker = Leaker::with_rule(config());
    assert!(!leaker.check_values && !leaker.check_bodies);
    leaker.intern_trait(&item("Off0", "A"));
    leaker.check_values = true;
    leaker.check_bodies = true;
    leaker.intern_trait(&item("On", "B"));
    leaker.check_values = false;
    leaker.check_bodies = false;
    leaker.intern_trait(&item("Off1", "C"));
    leaker.check_values = true;
    leaker.check_bodies = true;
    leaker.enter_generics(&Generics::default(), |view| {
        view.visit_item(&Item::Trait(item("OnInGenerics", "D")));
    });
    leaker.check_values = false;
    leaker.check_bodies = false;
    leaker.enter_generics(&Generics::default(), |view| {
        view.visit_item(&Item::Trait(item("OffInGenerics", "E")));
    });
    let result = leaker.result();
    // Only the two calls that read the values have errors and entries of them.
    let mut errors: Vec<String> = result
        .errors()
        .iter()
        .map(|error| error.message.clone())
        .collect();
    errors.sort();
    assert_eq!(
        errors,
        ["use absolute path: 'relB'", "use absolute path: 'relD'"]
    );
    let referrer = result.recovered();
    let mut entries: Vec<String> = common::types(&referrer)
        .map(|ty| quote!(#ty).to_string())
        .collect();
    entries.sort();
    assert_eq!(entries, ["EntryB", "EntryD"]);
}

// ── the two switches ─────────────────────────────────────────────────────────────────────────

/// The notes of the errors of the items of `source`, analyzed with `check_values` and
/// `check_bodies` as given.
fn notes_with(source: &str, values: bool, bodies: bool) -> Vec<String> {
    let file: File = parse_str(source).unwrap();
    let mut leaker = Leaker::with_rule(config());
    leaker.check_values = values;
    leaker.check_bodies = bodies;
    for item in &file.items {
        match item {
            Item::Trait(i) => {
                leaker.intern_trait(i);
            }
            Item::Impl(i) => {
                leaker.intern_impl(i);
            }
            _ => {
                leaker.enter_generics(&Generics::default(), |view| view.visit_item(item));
            }
        }
    }
    let result = leaker.result();
    result
        .errors()
        .iter()
        .map(|item| {
            let (_, note) = item.notes.first().unwrap();
            note.clone()
        })
        .collect()
}

/// `check_values` reads the values (the default of an associated const, the value of an impl
/// const, of a const and of a static) and `check_bodies` the bodies of the functions (the default
/// body of a method, an impl method, a function): each reads only its own, in the order of the
/// items.
#[test]
fn check_values_reads_the_values_and_check_bodies_the_bodies() {
    let note = |what: &str| format!("it is in {}, which the copy writes as it is", what);
    // (the items, what the values give, what the bodies give)
    let cases = [
        (
            "trait Tr { const N: usize = inner::N; fn f() -> usize { inner::N } }",
            vec![note("the default value of `N`")],
            vec![note("the default body of `f`")],
        ),
        (
            "impl ::p::Tr for ::p::S { const N: usize = r1(); fn f() { r2() } }",
            vec![note("the value of `N`")],
            vec![note("the body of `f`")],
        ),
        (
            "impl ::p::S { const N: usize = r1(); fn f() { r2() } }",
            vec![note("the value of `N`")],
            vec![note("the body of `f`")],
        ),
        (
            "pub static S: usize = r1(); pub const C: usize = r2(); pub fn g() { r3() }",
            vec![note("the value of `S`"), note("the value of `C`")],
            vec![note("the body of `g`")],
        ),
        (
            "trait Tr { const N: usize = r1(); fn f() { r2() } fn g() { r3() } }",
            vec![note("the default value of `N`")],
            vec![
                note("the default body of `f`"),
                note("the default body of `g`"),
            ],
        ),
    ];
    for (source, values, bodies) in cases {
        let both: Vec<String> = notes_with(source, true, true);
        // The notes come in the order of the items: the values first here, the bodies after.
        let mut expected = values.clone();
        expected.extend(bodies.clone());
        let mut sorted = both.clone();
        sorted.sort();
        expected.sort();
        assert_eq!(sorted, expected, "both, {}", source);
        assert_eq!(
            notes_with(source, true, false),
            values,
            "values, {}",
            source
        );
        assert_eq!(
            notes_with(source, false, true),
            bodies,
            "bodies, {}",
            source
        );
        assert_eq!(
            notes_with(source, false, false),
            Vec::<String>::new(),
            "neither, {}",
            source
        );
    }
}

/// A type in a type position is an entry in a value with `check_values` and in a body with
/// `check_bodies`, and in neither without them.
#[test]
fn the_types_of_a_value_and_of_a_body_are_leaked_by_their_own_switch() {
    let item: ItemTrait = parse_quote! {
        trait Tr {
            const N: usize = ::core::mem::size_of::<inner::Local>();
            fn f(&self) -> usize { ::core::mem::size_of::<inner::Other>() }
        }
    };
    let entries = |values: bool, bodies: bool| -> Vec<String> {
        let mut leaker = Leaker::with_rule(config());
        leaker.check_values = values;
        leaker.check_bodies = bodies;
        leaker.intern_trait(&item);
        let referrer = leaker.result().finish().unwrap();
        let mut entries: Vec<String> = common::types(&referrer)
            .map(|ty| quote!(#ty).to_string())
            .collect();
        entries.sort();
        entries
    };
    assert_eq!(entries(false, false), Vec::<String>::new());
    assert_eq!(entries(true, false), ["inner :: Local"]);
    assert_eq!(entries(false, true), ["inner :: Other"]);
    assert_eq!(entries(true, true), ["inner :: Local", "inner :: Other"]);
}

/// What is inside a type is read whatever the switches say: a value or a body of a nested item in
/// the length of an array makes the whole array an entry (F101), and an enum discriminant is read
/// as a value.
#[test]
fn what_a_type_holds_is_read_with_neither_switch() {
    let item: ItemStruct = parse_quote! {
        struct S([u8; { const X: usize = rel::N; fn g() -> usize { rel::M } static Y: usize = rel::K; 3 }]);
    };
    let enum_item: ItemEnum = parse_quote! { enum E { A = inner::X } };
    for (values, bodies) in [(false, false), (true, false), (false, true), (true, true)] {
        let mut leaker = Leaker::with_rule(config());
        leaker.check_values = values;
        leaker.check_bodies = bodies;
        leaker.intern_struct(&item);
        leaker.intern_enum(&enum_item);
        let result = leaker.result();
        let messages: Vec<&str> = result.errors().iter().map(|e| e.message.as_str()).collect();
        let what = format!("values {}, bodies {}", values, bodies);
        // Only the discriminant is an error: the array is the entry, with no error.
        assert_eq!(messages, ["use absolute path: 'inner::X'"], "{}", what);
        let referrer = result.recovered();
        let entries: Vec<String> = common::types(&referrer)
            .map(|ty| quote!(#ty).to_string())
            .collect();
        assert_eq!(entries.len(), 1, "{}", what);
        assert!(entries[0].starts_with("[u8 ;"), "{}: {}", what, entries[0]);
    }
}

/// `Rules::values()` allows what is in a body as well as what is in a value: both are `in_value`.
/// It has an effect only where the matching switch reads them.
#[test]
fn the_values_rule_allows_the_bodies_too() {
    let item: ItemTrait = parse_quote! {
        trait Tr { const N: usize = inner::N; fn f() -> usize { inner::N } }
    };
    let count = |rule: Rules<()>, values: bool, bodies: bool| {
        let mut leaker = Leaker::with_rule(rule);
        leaker.check_values = values;
        leaker.check_bodies = bodies;
        leaker.intern_trait(&item);
        leaker.result().errors().len()
    };
    for (values, bodies, read) in [
        (false, false, 0),
        (true, false, 1),
        (false, true, 1),
        (true, true, 2),
    ] {
        assert_eq!(count(Rules::new(), values, bodies), read);
        assert_eq!(count(Rules::new().values(), values, bodies), 0);
    }
}

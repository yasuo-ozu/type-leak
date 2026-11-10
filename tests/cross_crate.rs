//! Uses `type-leak` across crate boundaries through a proc macro, as a downstream crate would.
//!
//! - `tests/crates/macro`: the proc macro using `type-leak` (`Leaker`, `Referrer` and
//!   the const slots of the Repeater).
//! - `tests/crates/leaker`: the Leaker's crate, defining traits with relative types.
//! - `tests/crates/referrer`: the Referrer's crate, implementing the traits for wrappers which
//!   forward to `Inner`, and copying items of every kind (struct, enum, union, impl, module, trait)
//!   and fragments of impls (the scope API), with a witness for each copied type.
//! - `tests/crates/ext`: a crate which the Leaker's crate depends on and the Referrer's does not,
//!   for the strict Referrer.
//!
//! The copies are compared with the originals when the Referrer's crate is compiled (a witness of
//! the Leaker's crate returns the original type, and the Referrer's crate returns the rewritten
//! one from it), so a rewrite which does not give the original type fails the build of this test.
//! The tests here read what the macros reported and what they wrote.

use syn::parse_quote;
use type_leak::{EntryId, ImplSelf, Projection};
use type_leak_test_leaker::m::{
    Braced9246, Consts, Defaulted, Defaults, DollarCrate, Edge, Holder, Inner, Plain, RelDefault,
    RelType,
};
use type_leak_test_referrer::{Concrete, Explicit, Generic, Generic2, Omitted, G9246};

#[test]
fn concrete_args() {
    let c = Concrete(Inner);
    assert_eq!(c.relative().0, 42);
    assert_eq!(
        c.array().iter().map(|r| r.0).collect::<Vec<_>>(),
        vec![1, 2, 3]
    );
    let (flag, letter) = c.flags();
    assert!(flag.get());
    assert_eq!(letter.get(), 'x');
    assert_eq!(c.value(5u8), (5u8, -3));
}

#[test]
fn generic_args() {
    let g = Generic::<false, 2>(Inner);
    assert_eq!(g.array().len(), 2);
    let (flag, letter) = g.flags();
    assert!(!flag.get());
    assert_eq!(letter.get(), 'y');
    assert_eq!(g.value(7u16), (7u16, 3));
}

#[test]
fn local_generics() {
    let c = Concrete(Inner);
    assert!(c.local_flag::<true>().get());
    assert!(!c.local_flag::<false>().get());
    assert_eq!(
        c.local_array::<4>().iter().map(|r| r.0).collect::<Vec<_>>(),
        vec![4, 4, 4, 4]
    );
    let n = 9;
    assert_eq!(c.holder(Holder(&n)), 9);
}

#[test]
fn omitted_default_args() {
    let c = Concrete(Inner);
    assert_eq!(c.default_array().len(), 2);
    assert!(c.default_flag().get());

    let g = Generic::<false, 5>(Inner);
    assert_eq!(g.default_array().len(), 5);
    assert!(g.default_flag().get());
}

#[test]
fn omitted_relative_default() {
    // The omitted `U` is the Leaker's `RelDefault<u8>`, which the Referrer's crate cannot name.
    let r: RelDefault<u8> = <Omitted as Defaulted<u8>>::get(&Omitted(Inner), 7);
    assert_eq!(r, RelDefault(7));
    // The default mentions the argument given for `T`.
    let r: RelDefault<&str> =
        <Generic2<&str> as Defaulted<&str>>::get(&Generic2(Inner, Default::default()), "x");
    assert_eq!(r, RelDefault("x"));
    // An explicit argument is used as it is.
    assert_eq!(
        <Explicit as Defaulted<u8, u32>>::get(&Explicit(Inner), 5),
        105
    );
}

/// Fixture 9246 (F73): the use site gives `{ M }` for `N`, and the default `[u8; { N }]` of `U` is
/// completed as `[u8; { M }]`. The Referrer's crate does not build with `[u8; { { M } }]`.
#[test]
fn braced_argument_for_a_braced_default() {
    let g = G9246::<3>(Inner);
    let got: [u8; 3] = <G9246<3> as Braced9246<3>>::get(&g);
    assert_eq!(got, [3, 3, 3]);
}

#[test]
fn edge_cases() {
    let c = Concrete(Inner);
    let pair = Edge::<u8>::e_where(&c, 7u8);
    assert_eq!((pair.0, pair.1), (7, 7));
    assert_eq!(Edge::<u8>::e_where_rel(&c, 9u32), 9);
    assert_eq!(Edge::<u8>::e_ref(&c).0, 2);
    assert_eq!(Edge::<u8>::e_pat(&c, RelType(0), 4), 5);
    assert_eq!(Edge::<u8>::e_hrtb(&c, |r: &RelType| r.0 * 10), 30);
    assert_eq!(Edge::<u8>::e_elided(&c, &RelType(6)), 6);
    assert_eq!(
        Edge::<u8>::e_impl(&c).map(|r| r.0).collect::<Vec<_>>(),
        vec![4]
    );
    assert_eq!(Edge::<u8>::e_box(Box::new(Concrete(Inner))).0, 1);
    assert_eq!(c.p().0, 5);
}

/// The `$crate` paths of `DollarCrate` (a supertrait, types, an array length and a bound), which
/// the Referrer's crate copies, resolve to the Leaker's crate.
///
/// The exact results of the Leaker for such paths (and for the absolute paths of the external
/// crates, which are optional entries) are checked when the Leaker's crate is compiled, by
/// `check_leak!` in `tests/crates/leaker`.
#[test]
fn dollar_crate_paths() {
    use type_leak_test_leaker::m_portable::{Portable, Wrap};
    let c = Concrete(Inner);
    let _: Portable = c.d_ret();
    assert_eq!(c.d_arg(Wrap(4)), 5);
    assert_eq!(c.d_arr(), [7, 7]);
    assert_eq!(c.d_rel(Portable).0, 11);
}

/// `Self`, its associated types (also as a GAT and in a bound) and the generics of the trait in
/// the leaked types of `type_leak_test_leaker::s`.
#[test]
fn self_and_associated_types() {
    use type_leak_test_leaker::s::{Assoc, SelfSized, Wrap};
    let c = Concrete(Inner);
    // `Wrap<Self>` where `Self: Sized`, implemented by the Referrer's crate.
    let me: Wrap<Concrete> = c.me();
    let _ = me;
    assert_eq!(c.plain(), Wrap(1));
    // `Self::Item` is `u8` for `Concrete`.
    let item: Wrap<u8> = c.item();
    assert_eq!(item, Wrap(8));
    assert_eq!(c.u(3u8), Wrap(3u8));
    let v = 5u16;
    let g: Wrap<&u16> = c.gat(&v);
    assert_eq!(g, Wrap(&5));
}

/// Fixture 9248 (F76): the trait `Clash9248<__S>` has a generic named like the parameter that the
/// Repeater's impl declares for the implementor. The Leaker's crate does not build with
/// `impl<__S, __S: ?Sized + ..>` (E0403); the Referrer's crate implements the trait for `Concrete`.
#[test]
fn a_generic_named_like_the_parameter_of_the_implementor() {
    use type_leak_test_leaker::s::{Clash9248, Wrap};
    let c = Concrete(Inner);
    let got: Wrap<(u8, u16)> = <Concrete as Clash9248<u16>>::f(&c, 5);
    assert_eq!(got, Wrap((9, 5)));
}

/// Fixture 9252 (F90): `Proj9252<T: Iterator>` has `T::Item` in a signature, and the Referrer's
/// crate implements it for `::std::vec::IntoIter<u8>`. `<IntoIter<u8>>::Item`, which the macro wrote
/// before the fix, is E0223 ("ambiguous associated type") there.
#[test]
fn the_associated_type_of_a_concrete_argument_is_written_with_its_trait() {
    use type_leak_test_leaker::s::{Proj9252, Wrap};
    let c = Concrete(Inner);
    let got: Option<u8> = <Concrete as Proj9252<::std::vec::IntoIter<u8>>>::f(&c, Wrap(7));
    assert_eq!(got, Some(7));
}

/// Fixture 9253: `Where9253<T, D = Wrap<T>> where T: Iterator` has `T::Item` in a signature, and
/// the Referrer's crate implements it for `::std::vec::IntoIter<u8>`, leaving `D` out. The
/// arguments are completed from the declaration that the Referrer carries: `D` is `Wrap<IntoIter<u8>>`
/// through the Repeater, and the bound of the where clause gives the trait of `T::Item`. Before,
/// the macro knew only the bounds of the parameters themselves, and `<IntoIter<u8>>::Item` is E0223.
#[test]
fn the_declaration_of_the_trait_completes_a_default_and_a_where_clause_at_the_use_site() {
    use type_leak_test_leaker::s::{Where9253, Wrap};
    let c = Concrete(Inner);
    let item: Option<u8> = <Concrete as Where9253<::std::vec::IntoIter<u8>>>::f(&c, Wrap(7));
    assert_eq!(item, Some(7));
    let wrapped: Wrap<::std::vec::IntoIter<u8>> =
        <Concrete as Where9253<::std::vec::IntoIter<u8>>>::d(&c, vec![1u8, 2].into_iter());
    assert_eq!(wrapped.0.collect::<Vec<u8>>(), [1, 2]);
}

#[test]
fn lifetimes_trait_objects_and_projections() {
    use type_leak_test_leaker::s::{DynRoot, Out, RelImpl, Wrap, Q};
    let c = Concrete(Inner);
    let x = 4u64;
    assert_eq!(Out::f(&c, &x), Wrap(&4));
    assert_eq!(c.boxed().rel(), 31);
    assert_eq!(c.borrowed(&RelImpl), 32);
    assert_eq!(c.lent().0.rel(), 31);
    // `<T>::Item` with `T` = `Range<u32>`.
    assert_eq!(Q::<::core::ops::Range<u32>>::f(&c, 6), Wrap(6));
}

/// A trait object without a lifetime bound, as an argument of a path with a lifetime (`Ref<'a, dyn
/// RelTr>`), takes the lifetime from the path, so it is leaked together with the path if it has to
/// be (it names a relative trait). Otherwise only what is inside it is leaked, and the copy keeps
/// the trait object where it is.
#[test]
fn trait_object_inheriting_the_path_lifetime() {
    use std::cell::RefCell;
    use type_leak_test_leaker::s::{PathDyn, RelImpl, RelTr};
    let c = Concrete(Inner);
    let cell = RefCell::new(RelImpl);
    let r = ::core::cell::Ref::map(cell.borrow(), |r| r as &dyn RelTr);
    assert_eq!(c.path(r), 32);
    let r = ::core::cell::Ref::map(cell.borrow(), |r| r as &dyn RelTr);
    assert_eq!(c.paren(r), 33);
    let r = ::core::cell::Ref::map(cell.borrow(), |r| r as &(dyn RelTr + '_));
    assert_eq!(c.bound(r), 34);
    let boxed: Box<dyn RelTr> = Box::new(RelImpl);
    let cell = RefCell::new(boxed);
    assert_eq!(c.boxed(cell.borrow()), 35);
    let cell = RefCell::new(Box::new(RelImpl));
    let r = ::core::cell::Ref::map(cell.borrow(), |b| {
        b as &dyn ::core::ops::Deref<Target = RelImpl>
    });
    assert_eq!(c.inside(r), 36);
}

/// A bare identifier as an argument, which syn parses as a type, may be a const: it is not leaked
/// as a type, whatever it is named, but the type containing it is leaked as a whole.
#[test]
fn bare_const_argument() {
    use type_leak_test_leaker::s::{ConstArg, Len, SIZE};
    assert_eq!(Concrete(Inner).bare().n(), SIZE);
    assert_eq!(Concrete(Inner).bare_mixed_case().n(), Len);
}

/// The types in an array length or in a const argument are not leaked alone, whatever the
/// implementor is (a projection is not permitted in an anonymous constant with its generics).
#[test]
fn types_in_expressions() {
    use type_leak_test_leaker::s::ExprTypes;
    use type_leak_test_referrer::GenericExpr;
    let c = Concrete(Inner);
    assert_eq!(c.expr_array(), Some([1, 2, 3]));
    assert_eq!(c.expr_arg().n(), 3);
    let g = GenericExpr::<true>(Inner);
    assert_eq!(g.expr_array(), Some([1, 2, 3]));
    assert_eq!(g.expr_arg().n(), 3);
}

/// The types of external crates (`::std::vec::Vec<..>`, which the Leaker does not allow) are
/// optional entries: the Referrer's crate, which leaves them as written (`Trusting`) and the one
/// which replaces them by the Repeater's projections (`Strict`) both compile and forward.
#[test]
fn external_crate_types() {
    use type_leak_test_leaker::m::External;
    use type_leak_test_referrer::{Strict, Trusting};

    fn check<E: External<u8>>(e: E) -> E {
        assert_eq!(
            e.wrapped().iter().map(|(r,)| r.0).collect::<Vec<_>>(),
            vec![1, 2]
        );
        assert_eq!(e.generic(vec![1u8, 2, 3]), 3);
        assert_eq!(e.borrowed(&"xyz".to_string()), "yz");
        let nested = e.nested();
        assert_eq!((nested.0.clone(), (nested.1).0), (vec![3, 4], 5));
        assert_eq!(e.dynamic(Box::new(|r| r.0 * 2)), 14);
        e
    }
    assert_eq!(
        <Trusting as External<u8>>::receiver(Box::new(check(Trusting(Inner)))),
        6
    );
    assert_eq!(
        <Strict as External<u8>>::receiver(Box::new(check(Strict(Inner)))),
        6
    );
}

// ---------------------------------------------------------------------------------------------
// Items of every kind, copied by the Referrer (`#[prop_leak]`, see `tests/crates/macro/prop.rs`).
// ---------------------------------------------------------------------------------------------

use std::marker::PhantomData;

/// Removes the whitespace of tokens printed by a compiler, which depends on its version.
fn squash(text: &str) -> String {
    text.split_whitespace().collect()
}

/// The `KEY` of the entry `entry` of the Leaker `id`, as the tokens write it (and `squash()` leaves it).
fn key_text(id: u64, entry: usize) -> String {
    let entry: EntryId = parse_quote!(#entry);
    format!(
        "{}u128",
        Projection::new(
            parse_quote!(Repeater),
            id,
            ImplSelf::Param(parse_quote!(__S))
        )
        .key(entry)
    )
}

/// `Projection::key()` in a type, where the argument of a path is a const expression.
const fn key(id: u64, entry: usize) -> u128 {
    ((id as u128) << 64) | entry as u128
}

#[test]
fn the_const_key_is_the_key_of_the_library() {
    for (id, index) in [(0, 0), (7, 3), (9209, 2), (u64::MAX, usize::MAX)] {
        let entry: EntryId = parse_quote!(#index);
        assert_eq!(
            key(id, index),
            Projection::new(
                parse_quote!(Repeater),
                id,
                ImplSelf::Param(parse_quote!(__S))
            )
            .key(entry)
        );
    }
}

/// The `(id, entry)` of each `Repeater<(), KEY>` in the squashed `text`, in order.
fn repeater_keys(text: &str) -> Vec<(u64, usize)> {
    text.split("Repeater<(),")
        .skip(1)
        .map(|rest| {
            let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
            assert!(rest[digits.len()..].starts_with("u128>"), "{}", rest);
            let key: u128 = digits.parse().unwrap();
            ((key >> 64) as u64, (key & u128::from(u64::MAX)) as usize)
        })
        .collect()
}

/// A type equality witness: compiles only if both are the same type.
fn same<T>(_: &T, _: &T) {}

/// What a `#[prop_leak]` macro reported: `<id>|<kind>|ok|entries=E positions=P checked=C`, or
/// `<id>|<kind>|rej|<message>`. A `#[prop_leak_accept]` macro reports `acc` for an item that the
/// Leaker rejected and accepted with `LeakerResult::allow()` of the causes of its errors.
#[derive(Debug, PartialEq)]
enum Reported {
    Ok {
        kind: &'static str,
        entries: usize,
        positions: usize,
        checked: usize,
    },
    Allowed {
        kind: &'static str,
        entries: usize,
        positions: usize,
        checked: usize,
    },
    Rejected {
        kind: &'static str,
        message: String,
    },
}

fn ok(kind: &'static str, entries: usize, positions: usize, checked: usize) -> Reported {
    Reported::Ok {
        kind,
        entries,
        positions,
        checked,
    }
}

fn rejected(kind: &'static str, message: &str) -> Reported {
    Reported::Rejected {
        kind,
        message: message.to_string(),
    }
}

fn reported(id: u32, text: &'static str) -> Reported {
    let mut parts = text.splitn(4, '|');
    assert_eq!(parts.next(), Some(id.to_string().as_str()), "{}", text);
    let kind = parts.next().unwrap();
    let kind = [
        "struct",
        "enum",
        "union",
        "impl-inherent",
        "impl-trait",
        "trait",
    ]
    .iter()
    .copied()
    .find(|k| *k == kind)
    .unwrap_or_else(|| panic!("unknown kind in {}", text));
    let (status, rest) = (parts.next().unwrap(), parts.next().unwrap_or(""));
    if status == "rej" {
        return rejected(kind, &squash(rest));
    }
    assert!(status == "ok" || status == "acc", "{}", text);
    let number = |key: &str| -> usize {
        rest.split(' ')
            .find_map(|word| word.strip_prefix(key))
            .unwrap_or_else(|| panic!("no `{}` in {}", key, text))
            .parse()
            .unwrap()
    };
    let (entries, positions, checked) =
        (number("entries="), number("positions="), number("checked="));
    if status == "acc" {
        return Reported::Allowed {
            kind,
            entries,
            positions,
            checked,
        };
    }
    ok(kind, entries, positions, checked)
}

fn allowed(kind: &'static str, entries: usize, positions: usize, checked: usize) -> Reported {
    Reported::Allowed {
        kind,
        entries,
        positions,
        checked,
    }
}

/// The errors of the report `text` of `__tl_errors_<id>!()`: the reason, the kind of the cause and
/// its tokens without whitespace.
fn errors(text: &str) -> Vec<(String, String, String)> {
    text.split('\u{1d}')
        .map(|error| {
            let fields: Vec<&str> = error.split('\u{1f}').collect();
            (
                fields[3].to_string(),
                fields[4].to_string(),
                squash(fields[1]),
            )
        })
        .collect()
}

fn error(reason: &str, cause: &str, tokens: &str) -> (String, String, String) {
    (reason.to_string(), cause.to_string(), tokens.to_string())
}

/// A struct, an enum, a union, an impl, a module and a trait are each analyzed by the matching
/// `intern_*` and accepted (or rejected) as the Leaker's crate says, and
/// the Referrer's crate checked every copied type against the original: `checked` is the number of
/// type positions of the item which have a witness, or in a module name what it declares and are
/// compared with the original written for the Referrer's crate. The fixtures of some findings
/// (F51, F56, F68 to F70) only report, and are rows too.
#[test]
fn every_item_kind_is_leaked_across_crates() {
    use type_leak_test_leaker::*;
    let table = [
        // structs: `Self`, a `Self` that is not named, and an optional entry (a type of another
        // crate)
        (9001, __tl_report_9001!(), ok("struct", 5, 5, 5)),
        (9002, __tl_report_9002!(), ok("struct", 3, 3, 3)),
        (9004, __tl_report_9004!(), ok("struct", 3, 3, 3)),
        // a where clause on a type that is not a parameter, in the context of its entry (`Vec<T>` of
        // `::std` is no entry, F71: the predicate `Vec<T>: Clone` must stay as written, or the impl of
        // the entry does not hold in the copy)
        (9005, __tl_report_9005!(), ok("struct", 1, 1, 1)),
        // enums
        (9011, __tl_report_9011!(), ok("enum", 4, 5, 5)),
        (9012, __tl_report_9012!(), ok("enum", 2, 2, 2)),
        // unions
        (9021, __tl_report_9021!(), ok("union", 3, 3, 3)),
        (9022, __tl_report_9022!(), ok("union", 2, 2, 2)),
        // impls: inherent, of a trait (also with a relative argument), and one that names no `Self`
        (9031, __tl_report_9031!(), ok("impl-inherent", 6, 6, 6)),
        (9032, __tl_report_9032!(), ok("impl-trait", 2, 5, 5)),
        (9033, __tl_report_9033!(), ok("impl-inherent", 2, 2, 2)),
        (9034, __tl_report_9034!(), ok("impl-trait", 2, 5, 5)),
        // traits: `Self` not named, and an unsized implementor
        (9051, __tl_report_9051!(), ok("trait", 3, 3, 3)),
        (9053, __tl_report_9053!(), ok("trait", 4, 6, 6)),
        // a bare identifier that is a const (`LEN`): the path that holds it is an entry as a whole,
        // also with `Self` in it; a bare identifier that is a type is leaked alone where the path
        // cannot be
        (9081, __tl_report_9081!(), ok("struct", 2, 2, 2)),
        (9083, __tl_report_9083!(), ok("trait", 3, 3, 3)),
        (9084, __tl_report_9084!(), ok("trait", 1, 5, 1)),
        // items nested in a module, which name their own `Self`
        (9061, __tl_report_9061!(), ok("struct", 5, 5, 5)),
        (9062, __tl_report_9062!(), ok("enum", 4, 4, 4)),
        (9063, __tl_report_9063!(), ok("trait", 5, 5, 5)),
        // structs, an enum and a union nested in a module, with a bound that names their own
        // `Self` (`T: AsRef<Self>`), also in the where clause: the entry that does not name `Self`
        // is for any type, and the copy passes its own; one that names it too is rejected
        (9111, __tl_report_9111!(), ok("struct", 3, 3, 3)),
        (9112, __tl_report_9112!(), ok("struct", 1, 2, 2)),
        (9113, __tl_report_9113!(), ok("enum", 2, 1, 1)),
        (9114, __tl_report_9114!(), ok("union", 2, 2, 2)),
        (
            9115,
            __tl_report_9115!(),
            rejected(
                "struct",
                "`Wrap<Self>`names`Self`andhasagenericwhoseboundmentions`Self`",
            ),
        ),
        // the entry is in a bound of another generic, in a where predicate, keeps a lifetime, and
        // is for a `Self` that is outlived or bounded by `Sized`
        (9116, __tl_report_9116!(), ok("struct", 1, 2, 2)),
        (9117, __tl_report_9117!(), ok("struct", 1, 1, 1)),
        (9118, __tl_report_9118!(), ok("struct", 1, 2, 2)),
        (9119, __tl_report_9119!(), ok("struct", 1, 2, 2)),
        (9120, __tl_report_9120!(), ok("struct", 1, 2, 2)),
        // the entry for any type is the last field of a struct or a field of an enum, with a bound
        // that needs no `Sized` or that does: the Repeater's impl is `?Sized` in the first case
        // only
        (9125, __tl_report_9125!(), ok("struct", 2, 2, 2)),
        (9126, __tl_report_9126!(), ok("enum", 1, 2, 2)),
        (9127, __tl_report_9127!(), ok("struct", 1, 2, 2)),
        (9128, __tl_report_9128!(), ok("struct", 1, 2, 2)),
        // an absolute type that names their own `Self` is no entry (the copy has it as written);
        // the relative types beside it are
        (9121, __tl_report_9121!(), ok("struct", 2, 3, 3)),
        (9122, __tl_report_9122!(), ok("struct", 1, 2, 2)),
        (9124, __tl_report_9124!(), ok("enum", 2, 3, 3)),
        // the defaults of a nested trait whose generic has a bound that names its `Self`
        (9131, __tl_report_9131!(), ok("trait", 1, 3, 3)),
        (9132, __tl_report_9132!(), ok("trait", 2, 5, 5)),
        // the same, not declared again: the use site that leaves the default out completes it (F65)
        (9224, __tl_report_9224!(), ok("trait", 1, 3, 3)),
        // a braced const parameter in a type that is no entry, at a use site (F66)
        (9225, __tl_report_9225!(), ok("struct", 1, 3, 3)),
        (9226, __tl_report_9226!(), ok("struct", 1, 3, 3)),
        (9227, __tl_report_9227!(), ok("struct", 1, 4, 4)),
        // the default of a parameter of a struct and an enum nested in a module, when a bound of
        // an earlier parameter names their `Self`: the entry has no `Self`
        (9151, __tl_report_9151!(), ok("struct", 1, 3, 3)),
        (9152, __tl_report_9152!(), ok("enum", 1, 3, 3)),
        (9153, __tl_report_9153!(), ok("struct", 2, 4, 4)),
        (9154, __tl_report_9154!(), ok("struct", 2, 5, 5)),
        // a type that names their `Self` too (the macro accepts the `NestedSelfInBounds` error,
        // `nested_self_entries = true`): for any type, with `Self` in the type; and the control
        // without a bound that names `Self`
        (9161, __tl_report_9161!(), ok("struct", 2, 2, 2)),
        (9162, __tl_report_9162!(), ok("enum", 2, 3, 3)),
        (9163, __tl_report_9163!(), ok("struct", 2, 3, 3)),
        (9164, __tl_report_9164!(), ok("struct", 2, 2, 2)),
        // a bound of a generic that names it, next to a bound that fixes an associated type, is left
        // out of the Repeater's impls of the entries: through an alias that hides the projection
        // (also for a strict Referrer, which replaces the alias too), written, through a struct, a
        // supertrait or an associated type of a trait, in a struct, for the bound that names the
        // leaked type itself, and for the impl of F19; a rigid `Item` keeps the bound; a bound of
        // another generic is not left out, and the Leaker rejects it
        (9171, __tl_report_9171!(), ok("trait", 2, 0, 0)),
        (9172, __tl_report_9172!(), ok("trait", 2, 0, 0)),
        (9173, __tl_report_9173!(), ok("trait", 1, 0, 0)),
        (9174, __tl_report_9174!(), ok("trait", 2, 0, 0)),
        (9175, __tl_report_9175!(), ok("trait", 2, 0, 0)),
        (9176, __tl_report_9176!(), ok("trait", 1, 0, 0)),
        (9177, __tl_report_9177!(), ok("struct", 3, 1, 1)),
        (9178, __tl_report_9178!(), ok("struct", 1, 1, 1)),
        (9179, __tl_report_9179!(), ok("impl-inherent", 1, 2, 2)),
        (9180, __tl_report_9180!(), ok("trait", 1, 0, 0)),
        (9182, __tl_report_9182!(), ok("impl-inherent", 3, 2, 2)),
        (9183, __tl_report_9183!(), ok("struct", 3, 3, 3)),
        (
            9181,
            __tl_report_9181!(),
            rejected(
                "trait",
                "`Wrap<U>`isinaboundorwherepredicate,andaboundofitsgenericsnames`T::Item`",
            ),
        ),
        // `Self: 'x` brings `Self` into the context of an entry that does not name it: the bound
        // that is left out stays out, for a trait and for an impl (F47)
        (9200, __tl_report_9200!(), ok("trait", 2, 0, 0)),
        (9201, __tl_report_9201!(), ok("impl-trait", 3, 2, 2)),
        // the types in the values of an item (`check_values`): the consts and the default
        // methods of a trait
        (9141, __tl_report_9141!(), ok("trait", 5, 6, 6)),
        // scopes that share names (the findings of a downstream crate): methods that declare the
        // same generic with different bounds, a generic named like a relative type, a path ending
        // with the name of a generic, a type and a const parameter of one name; a bound that names
        // a parameter or a lifetime the entry does not use; a where clause; a cyclic bound; a
        // `for<'b>` binder
        (9071, __tl_report_9071!(), ok("trait", 7, 13, 13)),
        (9072, __tl_report_9072!(), ok("struct", 1, 2, 2)),
        (9073, __tl_report_9073!(), ok("trait", 1, 2, 2)),
        (9074, __tl_report_9074!(), ok("struct", 1, 2, 2)),
        (9075, __tl_report_9075!(), ok("struct", 1, 1, 1)),
        (
            9076,
            __tl_report_9076!(),
            rejected(
                "struct",
                "`Wrap<T>`isinaboundorwherepredicate,andaboundofitsgenericsnamesaleakedtype",
            ),
        ),
        (9077, __tl_report_9077!(), ok("struct", 1, 1, 1)),
        // paths that end with the name of a generic (`self::U`, `crate::U`), in a struct and in a
        // trait
        (9091, __tl_report_9091!(), ok("struct", 4, 5, 5)),
        (9092, __tl_report_9092!(), ok("trait", 5, 8, 8)),
        // F51: a nested struct whose bound names a leaked type that names `Self`, though the macro
        // accepts the `NestedSelfInBounds` error (`nested_self_entries = true`), is rejected, as
        // `intern_struct()` rejects it. The Referrer's macro expands to nothing (`kinds`, like the
        // other rejected items).
        (
            9206,
            __tl_report_9206!(),
            rejected(
                "struct",
                "`Wrap<Self>`isinaboundorwherepredicate,andaboundofitsgenericsnamesaleakedtype",
            ),
        ),
        // F56: a macro and a struct literal that are relative paths, in a discriminant and in the
        // default of a const parameter, are rejected (the Leaker did not report them, and the copy
        // could not resolve `rel_two!` and `RelPoint`).
        (
            9217,
            __tl_report_9217!(),
            rejected("enum", "useabsolutepath:'rel_two!'"),
        ),
        (
            9218,
            __tl_report_9218!(),
            rejected("struct", "useabsolutepath:'RelPoint'"),
        ),
        // F85: a relative constant in a value in the default of a type parameter is rejected with
        // `check_values` (the Leaker did not read it, and the copy could not resolve `REL9250`).
        (
            9250,
            __tl_report_9250!(),
            rejected("struct", "useabsolutepath:'REL9250'"),
        ),
        // F89: the argument of a generic associated type replaces its parameter in the relations of its
        // where clause, also when it is no path (`&'x u8`) and a generic of the method has the name.
        (9251, __tl_report_9251!(), ok("trait", 1, 2, 2)),
        // F68: the self type of an impl is a type of the Leaker's crate by an absolute path, a bound
        // of a generic names an optional type and another `Self`, and a predicate is on an entry
        // that uses the generic: the self type is no entry, and every copy writes it as it is.
        (9230, __tl_report_9230!(), ok("impl-inherent", 1, 2, 2)),
        // F69: the copy is used at an instantiation that gives the type parameter a type with a
        // lifetime, where it is in the inputs of a function pointer or `Fn(..)` whose output has an
        // elided lifetime: a struct, the arguments of a trait's method, and the default of a
        // parameter that `complete_args()` completes. The elided one is named.
        (9231, __tl_report_9231!(), ok("struct", 1, 4, 4)),
        (9232, __tl_report_9232!(), ok("trait", 1, 3, 2)),
        (9233, __tl_report_9233!(), ok("struct", 1, 4, 4)),
        // F70: the same where an entry is in the inputs (`Vec<T>` of `::std`, an optional entry that
        // a strict Referrer replaces by a path that names the lifetime of the site).
        (9234, __tl_report_9234!(), ok("struct", 2, 2, 2)),
        (9235, __tl_report_9235!(), ok("struct", 2, 2, 2)),
        // F71: a type of a crate that the rule does not allow, in a where predicate that the context of
        // an entry carries, is no entry (the predicate does not hold without itself): in the bound of
        // a predicate on `u8`, nested, held in a field too, with no other entry, in the predicate of a
        // method, of the trait and of the impl. The last has a type that names no generic: its
        // context is empty, and it stays an entry (2 entries: it and the struct that has the predicate).
        (9237, __tl_report_9237!(), ok("struct", 1, 1, 1)),
        (9238, __tl_report_9238!(), ok("struct", 1, 1, 1)),
        (9239, __tl_report_9239!(), ok("struct", 1, 2, 2)),
        (9240, __tl_report_9240!(), ok("struct", 0, 1, 1)),
        (9241, __tl_report_9241!(), ok("trait", 1, 2, 2)),
        (9242, __tl_report_9242!(), ok("trait", 1, 2, 2)),
        (9243, __tl_report_9243!(), ok("impl-inherent", 2, 3, 3)),
        (9244, __tl_report_9244!(), ok("struct", 2, 1, 1)),
    ];
    for (id, text, expected) in table {
        assert_eq!(reported(id, text), expected, "item {}: {}", id, text);
    }
}

/// The copy of a struct, an enum, a union and a module has the types of the original: the fields
/// of the original go into the copy and back (the Referrer's crate cannot name `Rel`, `Wrap` and
/// `Pair`, which `type_leak_test_leaker::kinds` declares).
#[test]
fn copies_of_items_hold_the_original_types() {
    use type_leak_test_leaker::kinds as l;
    use type_leak_test_referrer::kinds as r;

    // A struct that does not name `Self`.
    let rel = l::Rel(7);
    let copy = r::Plain::<u8> {
        rel,
        wrapped: l::Wrap(PhantomData),
        pair: l::Pair(3u8, rel),
    };
    same(&copy.rel, &rel);
    same(&copy.pair, &l::Pair(0u8, rel));
    let back = l::Plain::<u8> {
        rel: copy.rel,
        wrapped: copy.wrapped,
        pair: copy.pair,
    };
    assert_eq!(back.pair, l::Pair(3, l::Rel(7)));

    // With `Self` in it: the node holds `Wrap<Node>`, the Leaker's.
    let node = r::Node::<u8> {
        rel,
        wrapped: l::Wrap(PhantomData),
        pair: l::Pair(rel, 4u8),
        borrowed: &rel,
        next: Some(Box::new(l::Wrap::<l::Node<'_, u8>>(PhantomData))),
    };
    assert_eq!(node.pair.1, 4);
    assert_eq!(node.borrowed, &l::Rel(7));
    assert!(node.next.is_some());

    // An enum: each variant has the original types.
    let choices = [
        r::Choice::<u8>::Rel(rel),
        r::Choice::Wrapped(l::Wrap(PhantomData)),
        r::Choice::Pair {
            pair: l::Pair(rel, 9),
            flag: true,
        },
        r::Choice::Me(l::Wrap::<l::Choice<u8>>(PhantomData)),
        r::Choice::Nothing,
    ];
    let seen: Vec<_> = choices
        .iter()
        .map(|c| match c {
            r::Choice::Rel(x) => x.0 as usize,
            r::Choice::Wrapped(_) => 100,
            r::Choice::Pair { pair, flag } => pair.1 as usize + *flag as usize,
            r::Choice::Me(_) => 200,
            r::Choice::Nothing => 300,
        })
        .collect();
    assert_eq!(seen, vec![7, 100, 10, 200, 300]);
    let n = 3u8;
    let lt = r::Borrow::Lt(l::Lt(&n));
    assert!(matches!(lt, r::Borrow::Lt(l::Lt(&3))));

    // A union of the original types, written and read.
    let bits = r::Bits::<u8> { rel };
    assert_eq!(unsafe { bits.rel }, l::Rel(7));
    let bits = r::BitsSelf {
        me: l::Wrap::<l::BitsSelf>(PhantomData),
    };
    let _ = unsafe { bits.me };
}

/// A nested struct, enum and trait (the scope `nested`) name their own `Self` in relative types
/// (`Wrap<Self>`), and the copy holds the Leaker's: the closure that the test macro gives to
/// `Visitor::with_nested_self()` returns `::type_leak_test_leaker::kinds::Links<T>` for the `Self`
/// of the struct, which the copy's own `Links` could not stand for (the Repeater's impl is for the
/// Leaker's one).
/// The Referrer's crate compares each position with a witness of the original when it compiles; here
/// the fields go in and out of the Leaker's types, which `same()` and the assignments check.
#[test]
fn nested_items_name_their_own_self() {
    use type_leak_test_leaker::kinds as l;
    use type_leak_test_referrer::kinds as r;

    // The struct: `Wrap<Self>` is `Wrap<l::Links<u8>>`, not `Wrap<r::Links<u8>>`.
    let rel = l::Rel(5);
    let copy = r::Links::<u8> {
        rel,
        wrapped: l::Wrap(PhantomData),
        me: l::Wrap::<l::Links<u8>>(PhantomData),
        pair: l::Pair(rel, l::Wrap::<l::Links<u8>>(PhantomData)),
        next: Some(Box::new(l::Wrap::<l::Links<u8>>(PhantomData))),
    };
    same(&copy.me, &l::Wrap::<l::Links<u8>>(PhantomData));
    same(&copy.rel, &rel);
    let original = l::Links::<u8> {
        rel: copy.rel,
        wrapped: copy.wrapped,
        me: copy.me,
        pair: copy.pair,
        next: copy.next,
    };
    assert_eq!(original.pair.0, rel);
    assert!(original.next.is_some());

    // The enum.
    let steps = [
        r::Step::<u8>::Rel(rel),
        r::Step::Wrapped(l::Wrap(PhantomData)),
        r::Step::Me(l::Wrap::<l::Step<u8>>(PhantomData)),
        r::Step::Pair {
            pair: l::Pair(l::Wrap::<l::Step<u8>>(PhantomData), 3),
        },
        r::Step::Nothing,
    ];
    let seen: Vec<_> = steps
        .iter()
        .map(|step| match step {
            r::Step::Rel(x) => x.0 as usize,
            r::Step::Wrapped(_) => 100,
            r::Step::Me(_) => 200,
            r::Step::Pair { pair } => pair.1 as usize,
            r::Step::Nothing => 300,
        })
        .collect();
    assert_eq!(seen, vec![5, 100, 200, 3, 300]);

    // The trait: the host implements the Leaker's trait with the signatures of the copy, whose
    // `Wrap<Self>` is the Leaker's `Wrap<Host9063>`.
    fn implemented_for<T: l::Walk<u8>>() {}
    implemented_for::<r::Host9063>();
    let host = r::Host9063;
    fn me<T: l::Walk<u8> + ?Sized>(_: &T) -> l::Wrap<T> {
        l::Wrap(PhantomData)
    }
    same(&me(&host), &l::Wrap::<r::Host9063>(PhantomData));
}

/// A struct, an enum and a union nested in a module, whose generic has a bound that names their
/// own `Self` (`T: AsRef<Self>`, `where T: AsRef<(Self, u8)>`), use the generic in a relative type
/// (`Wrap<T>`) that does not name `Self`. The copy declares a type of its own, which has the bound
/// for itself, so the visitor's `<Self as Repeater<..>>::Type` is right, with no closure of
/// `with_nested_self()`: the Repeater's impl of the entry is for any type that the bound holds for. The field is the
/// Leaker's `Wrap<T>`; had the macro passed the Leaker's type, the bound would not hold for it
/// (E0277). The Referrer's crate compiles only if rustc accepts the copy and the witnesses.
#[test]
fn nested_types_with_a_bound_on_their_own_self_keep_the_original_field_types() {
    use std::collections::BTreeMap;
    use type_leak_test_leaker::base::B2;
    use type_leak_test_leaker::kinds as l;
    use type_leak_test_referrer::kinds as r;

    // The struct, with the bound on the parameter. The bound holds for the copy's own `Owned`.
    struct Owner;
    impl AsRef<r::Owned<Owner>> for Owner {
        fn as_ref(&self) -> &r::Owned<Owner> {
            unimplemented!()
        }
    }
    let rel = l::Rel(5);
    let copy = r::Owned::<Owner> {
        rel,
        wrapped: l::Wrap(PhantomData),
        pair: l::Pair(rel, Owner),
    };
    same(&copy.rel, &rel);
    same(&copy.wrapped, &l::Wrap::<Owner>(PhantomData));
    same(&copy.pair.0, &rel);
    assert_eq!(copy.pair.0, rel);

    // The where predicate, with `Self` in a tuple.
    struct Tag;
    impl AsRef<(r::Tagged<Tag>, u8)> for Tag {
        fn as_ref(&self) -> &(r::Tagged<Tag>, u8) {
            unimplemented!()
        }
    }
    let tagged = r::Tagged::<Tag>(l::Wrap(PhantomData), 7);
    same(&tagged.0, &l::Wrap::<Tag>(PhantomData));
    assert_eq!(tagged.1, 7);

    // The enum of the property test: a lifetime, a const parameter, `Self` twice in a bound that
    // also names a relative type (`Lt<'a, Rel>`, which the copy writes through the Repeater too).
    struct Owner3;
    type Key = r::Reported<'static, Owner3, 3>;
    impl AsRef<BTreeMap<(Key, Key), l::Lt<'static, l::Rel>>> for Owner3 {
        fn as_ref(&self) -> &BTreeMap<(Key, Key), l::Lt<'static, l::Rel>> {
            unimplemented!()
        }
    }
    static PAIR: B2<Owner3, Result<u32, u16>> = B2(Owner3, Ok(9));
    let steps: [Key; 2] = [r::Reported::V0(&PAIR), r::Reported::V1];
    match &steps[0] {
        r::Reported::V0(held) => {
            same::<B2<Owner3, Result<u32, u16>>>(held, &PAIR);
            assert_eq!(held.1, Ok(9));
        }
        r::Reported::V1 => unreachable!(),
    }
    assert!(matches!(steps[1], r::Reported::V1));

    // The union.
    #[derive(Clone, Copy)]
    struct Owner4;
    impl AsRef<r::Overlaid<Owner4>> for Owner4 {
        fn as_ref(&self) -> &r::Overlaid<Owner4> {
            unimplemented!()
        }
    }
    let overlaid = r::Overlaid::<Owner4> {
        wrapped: l::Wrap(PhantomData),
    };
    same(
        unsafe { &overlaid.wrapped },
        &l::Wrap::<Owner4>(PhantomData),
    );
}

/// An absolute type that names the own `Self` of a struct or an enum nested in a module
/// (`Option<::std::boxed::Box<Self>>`, `::type_leak_test_leaker::kinds::Pair<PhantomData<Self>, T>`)
/// is no entry: the copy has it as written, and `Self` in it is the copy's own type, whether the
/// Referrer replaces the optional entries (`strict`) or not. Before, a strict Referrer replaced it
/// with the Leaker's type, and a recursive type was no longer recursive: `List { next:
/// Option<Box<leaker::List>> }`. The relative types beside it (`Rel`, `Wrap<T>`) are entries, and
/// replaced. The Referrer's crate compiles only if the copies, which each position is compared
/// with a witness of the original for, hold what the witnesses say; here the fields go in and out
/// of the copy's own types, which `same()` and the assignments check.
#[test]
fn an_absolute_type_that_names_the_own_self_is_the_copys_in_a_strict_copy() {
    use type_leak_test_leaker::kinds as l;
    use type_leak_test_referrer::kinds as r;

    // A recursive type (strict): the box holds the copy's `List`.
    let rel = l::Rel(1);
    let list = r::List {
        next: Some(Box::new(r::List { next: None, rel })),
        rel,
    };
    same::<Option<Box<r::List>>>(&list.next, &None);
    same(&list.rel, &rel);
    assert_eq!(list.next.as_ref().map(|next| next.rel), Some(rel));

    // The bound holds for the copy's own type, not the Leaker's `Pathed<Owner>`, which `Owner`
    // has no `AsRef` for. The pair holds the copy's type for `Self`.
    struct Owner;
    impl AsRef<r::Pathed<Owner>> for Owner {
        fn as_ref(&self) -> &r::Pathed<Owner> {
            unimplemented!()
        }
    }
    let pathed = r::Pathed::<Owner> {
        path: l::Pair(PhantomData::<r::Pathed<Owner>>, Owner),
        wrapped: l::Wrap(PhantomData),
        rel,
    };
    same(&pathed.path.0, &PhantomData::<r::Pathed<Owner>>);
    same(&pathed.wrapped, &l::Wrap::<Owner>(PhantomData));
    same(&pathed.rel, &rel);

    // The enum: the vector holds boxes of the copy's `Tree`.
    let tree = r::Tree::<u8>::Branch(vec![
        Box::new(r::Tree::Leaf(rel)),
        Box::new(r::Tree::Tagged(l::Pair(
            PhantomData::<r::Tree<u8>>,
            l::Wrap(PhantomData),
        ))),
    ]);
    match &tree {
        r::Tree::Branch(children) => {
            same::<Vec<Box<r::Tree<u8>>>>(children, &Vec::new());
            assert_eq!(children.len(), 2);
            match &*children[1] {
                r::Tree::Tagged(pair) => {
                    same(&pair.0, &PhantomData::<r::Tree<u8>>);
                    same(&pair.1, &l::Wrap::<u8>(PhantomData));
                }
                _ => unreachable!(),
            }
        }
        _ => unreachable!(),
    }
}

/// The default of a parameter of a trait nested in a module, when a generic has a bound that names
/// the trait's own `Self` (`trait SelfDefault<T: AsRef<Self>, U = Defd<T>>`). The entry `Defd<T>` is
/// for any type that the bound holds for, as it is for the trait itself, and the Visitor left the
/// default as written (`Defd<T>`, a relative path that the Referrer's crate cannot resolve) when it
/// was for the implementors of the trait. A default exists only in a declaration, so the Referrer's
/// crate declares the copy (`SelfDefaultCopy`), with the default written through the host, a type
/// that has the bound.
///
/// An impl of the copy that leaves the parameter out takes the default, and its method has the
/// type of the original, `Defd<Owner>`: rustc compares the signature with the trait's (E0053), so
/// the default is exactly that type. The macro compares it with a witness as well (the position of
/// the default).
#[test]
fn the_default_of_a_nested_trait_that_names_its_self_is_written_through_the_host() {
    use type_leak_test_leaker::kinds as l;
    use type_leak_test_referrer::kinds as r;

    // The bound holds for the implementor of the copy (`T: AsRef<Self>`), and the default is written
    // for the host, which implements the Leaker's trait for the types that have the bound for it:
    // `Owner` has it for the host, and for each implementor here.
    struct Owner;
    macro_rules! implementors {
        ($($name:ident)*) => {$(
            struct $name;
            impl AsRef<$name> for Owner {
                fn as_ref(&self) -> &$name {
                    unimplemented!()
                }
            }
        )*};
    }
    implementors!(Defaulted Given Two TwoFirst TwoGiven);
    impl AsRef<r::Host9131> for Owner {
        fn as_ref(&self) -> &r::Host9131 {
            unimplemented!()
        }
    }
    impl AsRef<r::Host9132> for Owner {
        fn as_ref(&self) -> &r::Host9132 {
            unimplemented!()
        }
    }
    // `U` is the default: `Defd<Owner>`.
    impl r::SelfDefaultCopy<Owner> for Defaulted {
        fn get(&self, _t: Owner, u: l::Defd<Owner>) {
            same(&u, &l::Defd::<Owner>(PhantomData));
        }
    }
    // The default is replaced when it is given, too.
    impl r::SelfDefaultCopy<Owner, u8> for Given {
        fn get(&self, _t: Owner, u: u8) {
            assert_eq!(u, 7);
        }
    }
    r::SelfDefaultCopy::<Owner>::get(&Defaulted, Owner, l::Defd(PhantomData));
    r::SelfDefaultCopy::<Owner, u8>::get(&Given, Owner, 7);

    // Two defaults (strict), the second with an entry of its own: `Defd2<Rel, Owner>`.
    impl r::SelfDefaultsCopy<Owner> for Two {
        fn get(&self, _t: Owner, u: l::Defd<Owner>, v: l::Defd2<l::Rel, Owner>) {
            same(&u, &l::Defd::<Owner>(PhantomData));
            same(&v, &l::Defd2(l::Rel(0), Owner));
        }
    }
    impl r::SelfDefaultsCopy<Owner, u8> for TwoFirst {
        fn get(&self, _t: Owner, u: u8, v: l::Defd2<l::Rel, Owner>) {
            assert_eq!(u, 1);
            same(&v, &l::Defd2(l::Rel(0), Owner));
        }
    }
    impl r::SelfDefaultsCopy<Owner, u8, u16> for TwoGiven {
        fn get(&self, _t: Owner, u: u8, v: u16) {
            assert_eq!((u, v), (1, 2));
        }
    }
    r::SelfDefaultsCopy::<Owner>::get(
        &Two,
        Owner,
        l::Defd(PhantomData),
        l::Defd2(l::Rel(5), Owner),
    );
    r::SelfDefaultsCopy::<Owner, u8>::get(&TwoFirst, Owner, 1, l::Defd2(l::Rel(5), Owner));
    r::SelfDefaultsCopy::<Owner, u8, u16>::get(&TwoGiven, Owner, 1, 2);
}

/// The default of a parameter of a struct and an enum nested in a module, when a generic has a bound
/// that names their own `Self` (`struct S<T: Clone + AsRef<Self>, U = Defd<T>>`; F41 of
/// `tests/findings.rs`). The Leaker used to reject it. The copy cannot write `Self` in a default
/// (E0735), so the entry `Defd<T>` has no `Self` and none of the bounds that name it: the Repeater's
/// impl is `impl<T: Clone, __S: ?Sized> Repeater<(T,), ..> for __S` and the copy declares an ordinary
/// default through it, `U = <() as Repeater<(T,), ..>>::Type`. rustc checks a default that uses a
/// generic where it is used, so the default must be the type of the original there: an impl of
/// the copy for `S<Owner>` is for `S<Owner, Defd<Owner>>`, and a function over the one is a function
/// over the other.
#[test]
fn the_default_of_a_nested_struct_whose_bound_names_its_self_is_the_original_type() {
    use type_leak_test_leaker::kinds as l;
    use type_leak_test_referrer::kinds as r;

    #[derive(Clone)]
    struct Owner;
    // The bound holds for the copy's own types, with the default and with a given argument.
    impl AsRef<r::SelfBoundDefault<Owner>> for Owner {
        fn as_ref(&self) -> &r::SelfBoundDefault<Owner> {
            unimplemented!()
        }
    }
    #[derive(Clone)]
    struct Given;
    impl AsRef<r::SelfBoundDefault<Given, u8>> for Given {
        fn as_ref(&self) -> &r::SelfBoundDefault<Given, u8> {
            unimplemented!()
        }
    }
    impl AsRef<r::SelfBoundDefaultEnum<Owner>> for Owner {
        fn as_ref(&self) -> &r::SelfBoundDefaultEnum<Owner> {
            unimplemented!()
        }
    }
    impl AsRef<r::DefaultAndField<Owner>> for Owner {
        fn as_ref(&self) -> &r::DefaultAndField<Owner> {
            unimplemented!()
        }
    }
    impl AsRef<r::TwoDefaults<Owner>> for Owner {
        fn as_ref(&self) -> &r::TwoDefaults<Owner> {
            unimplemented!()
        }
    }

    // The omitted argument is the Leaker's `Defd<Owner>`: a function over the copy with the default
    // is one over the copy with the type written, in both directions.
    fn default_is_explicit(
        f: fn(r::SelfBoundDefault<Owner>),
    ) -> fn(r::SelfBoundDefault<Owner, l::Defd<Owner>>) {
        f
    }
    fn explicit_is_default(
        f: fn(r::SelfBoundDefault<Owner, l::Defd<Owner>>),
    ) -> fn(r::SelfBoundDefault<Owner>) {
        f
    }
    fn copy(_: r::SelfBoundDefault<Owner>) {}
    default_is_explicit(copy)(r::SelfBoundDefault(Owner, l::Defd(PhantomData)));
    explicit_is_default(copy)(r::SelfBoundDefault(Owner, l::Defd(PhantomData)));
    // A given argument replaces the default.
    let given: r::SelfBoundDefault<Given, u8> = r::SelfBoundDefault(Given, 7);
    assert_eq!(given.1, 7);

    // The enum.
    fn enum_default_is_explicit(
        f: fn(r::SelfBoundDefaultEnum<Owner>),
    ) -> fn(r::SelfBoundDefaultEnum<Owner, l::Defd<Owner>>) {
        f
    }
    fn copy_enum(e: r::SelfBoundDefaultEnum<Owner>) {
        match e {
            r::SelfBoundDefaultEnum::Plain(_) => (),
            r::SelfBoundDefaultEnum::Defaulted(wrapped) => same(&wrapped, &l::Defd(PhantomData)),
        }
    }
    enum_default_is_explicit(copy_enum)(r::SelfBoundDefaultEnum::Defaulted(l::Defd(PhantomData)));

    // The default and a field of the same type: the field is the entry for any type, where the
    // copy passes its own type, and it is `Defd<Owner>` as the default is.
    let both = r::DefaultAndField::<Owner>(Owner, l::Defd(PhantomData), l::Defd(PhantomData));
    same(&both.1, &both.2);
    same(&both.2, &l::Defd::<Owner>(PhantomData));

    // Two defaults, the second names the first: `V = Defd2<Rel, U>` with `U = Defd<T>`.
    let two = r::TwoDefaults::<Owner>(
        Owner,
        l::Defd(PhantomData),
        l::Defd2(l::Rel(3), l::Defd(PhantomData)),
    );
    same(&two.1, &l::Defd::<Owner>(PhantomData));
    same(&two.2, &l::Defd2(l::Rel(3), l::Defd::<Owner>(PhantomData)));
}

/// A type that names the own `Self` of a struct, an enum or a union nested in a module, in an item
/// whose generic has a bound that names it too (`struct S<T: AsRef<Self>> { link: Link<Self, T> }`),
/// when the macro accepts the `NestedSelfInBounds` error (F42 of `tests/findings.rs`). The entry is
/// for any type that the bound holds for, and the copy passes its own type as the implementor: the
/// field is `Link<S<T>, T>` with the copy's `S`, as in the original. Without a bound that names
/// `Self` (`Unbound`), the entry is for the Leaker's type as before, so the field holds the
/// Leaker's.
#[test]
fn a_type_that_names_the_own_self_with_a_bound_that_names_it_is_the_copys_own() {
    use type_leak_test_leaker::kinds as l;
    use type_leak_test_referrer::kinds as r;

    let rel = l::Rel(1);
    // The struct: the first argument of the field is the copy's own type.
    struct Owner;
    impl AsRef<r::Linked<Owner>> for Owner {
        fn as_ref(&self) -> &r::Linked<Owner> {
            unimplemented!()
        }
    }
    let linked = r::Linked::<Owner> {
        link: l::Link(PhantomData, Owner),
        rel,
    };
    same(
        &linked.link,
        &l::Link::<r::Linked<Owner>, Owner>(PhantomData, Owner),
    );
    same(&linked.rel, &rel);

    // The enum: `Link<Self, Rel>` is `Link<Variant<Owner2>, Rel>`, and `T` is not in the impl.
    struct Owner2;
    impl AsRef<r::Variant<Owner2>> for Owner2 {
        fn as_ref(&self) -> &r::Variant<Owner2> {
            unimplemented!()
        }
    }
    let variant = r::Variant::<Owner2>::Me(l::Link(PhantomData, rel));
    match &variant {
        r::Variant::Me(link) => same(
            link,
            &l::Link::<r::Variant<Owner2>, l::Rel>(PhantomData, rel),
        ),
        _ => unreachable!(),
    }

    // The bound names `Self` among other types (`(Self, u8)`): the vector holds the copy's type.
    struct Owner3;
    impl AsRef<(r::Tuple<Owner3>, u8)> for Owner3 {
        fn as_ref(&self) -> &(r::Tuple<Owner3>, u8) {
            unimplemented!()
        }
    }
    let tuple = r::Tuple::<Owner3>(vec![l::Link(PhantomData, 1)], Owner3, rel);
    same::<Vec<l::Link<r::Tuple<Owner3>, u8>>>(&tuple.0, &Vec::new());

    // No bound names `Self`: the entry is for the Leaker's type, so the field holds the Leaker's
    // `Unbound`, not the copy's.
    let unbound = r::Unbound::<u8> {
        link: l::Link(PhantomData, 0),
        rel,
    };
    same(
        &unbound.link,
        &l::Link::<l::Unbound<u8>, u8>(PhantomData, 0),
    );
}

/// The types in the values of an item, with `check_values` (F40 of
/// `tests/findings.rs`, the finding 14 of a downstream crate): `trait ValueTypes { const N: usize =
/// size_of::<vals::Local>(); .. }`. The Referrer's crate declares the copy (`ValueTypesCopy`), and
/// names none of the types of the defaults, so it compiles only if every one of them is replaced
/// by a path through the Repeater (a `let`, a turbofish, a qualified path, a closure, a cast, a
/// type in an array, the generics of the method and of the trait). The two traits compute the same.
#[test]
fn the_types_in_the_values_of_a_trait_are_carried() {
    use type_leak_test_leaker::kinds as l;
    use type_leak_test_referrer::kinds as r;

    struct Original;
    impl l::ValueTypes<u16> for Original {}
    struct Copy;
    impl r::ValueTypesCopy<u16> for Copy {}

    // The const: seven bytes, and three of them in an array.
    assert_eq!(<Original as l::ValueTypes<u16>>::N, 7);
    assert_eq!(<Copy as r::ValueTypesCopy<u16>>::N, 7);
    assert_eq!(<Original as l::ValueTypes<u16>>::M, 21);
    assert_eq!(<Copy as r::ValueTypesCopy<u16>>::M, 21);

    // The default methods: `7 + 0 + 7 + 7`, the closure `7 + 1`, and the sizes of the generics
    // (`u16` is the trait's `T`, `u8` the method's `U`).
    assert_eq!(l::ValueTypes::<u16>::size(&Original), 21);
    assert_eq!(r::ValueTypesCopy::<u16>::size(&Copy), 21);
    assert_eq!(l::ValueTypes::<u16>::closure(&Original), 8);
    assert_eq!(r::ValueTypesCopy::<u16>::closure(&Copy), 8);
    let original = l::ValueTypes::<u16>::generic(&Original, 1u8);
    let copy = r::ValueTypesCopy::<u16>::generic(&Copy, 1u8);
    assert_eq!(original, 1 + 4 * 100 + 2 * 10_000);
    assert_eq!(copy, original);
    let original = l::ValueTypes::<u16>::generic(&Original, 1u64);
    assert_eq!(original, 8 + 16 * 100 + 2 * 10_000);
    assert_eq!(r::ValueTypesCopy::<u16>::generic(&Copy, 1u64), original);
}

/// The scopes of an item that share names (the findings A, 1, 2, 3, 4 and 7 of a downstream crate,
/// `tests/downstream_findings.rs`): the Repeater's impl of each entry has the generics of its own
/// scope, so the Referrer's crate compiles, and the signatures of the copy have the original types
/// (a function pointer of each method, with the original types, is the method of the copy).
#[test]
fn scopes_that_share_names_keep_their_own_generics() {
    use type_leak_test_leaker::kinds as l;
    use type_leak_test_referrer::kinds as r;

    // `with_bound<X: Clone>` and `without_bound<X>` share the name `X`; `generic_local<Local>` has
    // a generic named like the type `Local` that `type_local` returns; `segment<U>` returns
    // `inner::U`, which is no generic; `type_n<N>` and `const_n<const N: usize>` share `N`.
    type H = r::Host9071;
    fn implemented_for<T: l::SameNames>() {}
    implemented_for::<H>();
    let _: fn(&H, l::Cloned<u8>) -> l::Cloned<u8> = <H as l::SameNames>::with_bound::<u8>;
    let _: fn(&H, l::Wrap<String>) -> l::Wrap<String> =
        <H as l::SameNames>::without_bound::<String>;
    let _: fn(&H, l::Local) -> l::Local = <H as l::SameNames>::generic_local::<l::Local>;
    let _: fn(&H, u8) -> u8 = <H as l::SameNames>::generic_local::<u8>;
    let _: fn(&H) -> l::Local = <H as l::SameNames>::type_local;
    let _: fn(&H, l::Wrap<u8>) -> l::inner::U = <H as l::SameNames>::segment::<u8>;
    let _: fn(&H, l::Wrap<u8>) -> l::Wrap<u8> = <H as l::SameNames>::type_n::<u8>;
    let _: fn(&H, l::Arr<3>) -> l::Arr<3> = <H as l::SameNames>::const_n::<3>;

    // The bound of `T` names `V`.
    let copy = r::ForeignBound::<u8, u32>(l::Wrap(PhantomData), 3);
    same(&copy.0, &l::Wrap::<u8>(PhantomData));
    assert_eq!(copy.1, 3);
    // The bound of a method generic names the parameter of the trait.
    fn convert_of<T>() -> fn(&r::Host9073, l::Wrap<u8>) -> l::Wrap<u8>
    where
        u8: Into<T>,
    {
        <r::Host9073 as l::ForeignBoundTrait<T>>::convert::<u8>
    }
    let _ = convert_of::<u32>();
    // The bound `T: 'a`.
    let n = 5u8;
    let copy = r::OutlivesBound::<u8>(l::Wrap(PhantomData), &n);
    same(&copy.0, &l::Wrap::<u8>(PhantomData));
    assert_eq!(*copy.1, 5);
    // The where clause of the item.
    let copy = r::WhereBound::<u8>(l::Cloned(1));
    same(&copy.0, &l::Cloned(1u8));
    // A binder lifetime that only the entry names.
    let seen = std::cell::Cell::new(0u8);
    let copy = r::Binder(|lt: l::Lt<'_, u8>| seen.set(*lt.0));
    (copy.0)(l::Lt(&n));
    assert_eq!(seen.get(), 5);
}

/// Names that a generic shares with a path or with a declared item (`tests/shadowed_names.rs`): the
/// Referrer's crate compiles only if the visitor replaced `self::U` and `crate::U` next to a generic
/// `U` (neither type exists there), and `crate::base::B1<X>` of a generic `X` in a module that
/// declares an `X` (the Referrer's crate has no `base`).
#[test]
fn names_shared_with_a_generic_keep_the_original_types() {
    use type_leak_test_leaker as l;
    use type_leak_test_referrer::kinds as r;

    let copy = r::PathsLikeGeneric::<u8> {
        generic: 1,
        here: l::kinds::U(2),
        root: l::U(3),
        wrapped: l::kinds::Wrap(PhantomData),
        both: l::kinds::Pair(l::U(4), 5),
    };
    assert_eq!(
        (
            copy.generic,
            copy.here.0,
            copy.root.0,
            copy.both.0 .0,
            copy.both.1
        ),
        (1, 2, 3, 4, 5)
    );
    same(&copy.wrapped, &l::kinds::Wrap::<l::kinds::U>(PhantomData));

    // A host implements the trait with the signatures of the copy.
    type H = r::Host9092;
    fn implemented_for<T: l::kinds::PathsLikeGenericTrait<u8>>() {}
    implemented_for::<H>();
    let _: fn(&H, u16, l::kinds::U) -> l::U =
        <H as l::kinds::PathsLikeGenericTrait<u8>>::here::<u16>;
    let _: fn(&H, l::kinds::Wrap<u16>) -> l::kinds::Wrap<l::kinds::U> =
        <H as l::kinds::PathsLikeGenericTrait<u8>>::wrapped::<u16>;
    let _: fn(&H, u8, l::kinds::U) -> l::kinds::Pair<l::U, u8> =
        <H as l::kinds::PathsLikeGenericTrait<u8>>::trait_generic;
}

/// A trait is implemented for an unsized type (`dyn Host`) in the Referrer's crate, with the
/// signatures it rewrote: the Repeater's impls of the entries that name `Self` must be for
/// `__S: ?Sized`, or the Referrer's crate would not compile. (The Leaker's crate cannot know the
/// implementors.)
#[test]
fn a_trait_with_an_unsized_implementor() {
    use type_leak_test_leaker::kinds::{Unsized, Wrap};
    use type_leak_test_referrer::kinds::Host9053;

    fn implemented_for<T: ?Sized + Unsized>() {}
    implemented_for::<dyn Host9053>();

    fn me<T: ?Sized>(_: &T) -> Wrap<T> {
        Wrap(PhantomData)
    }
    struct Local;
    impl Host9053 for Local {}
    let dynamic: &dyn Host9053 = &Local;
    same(&me(dynamic), &Wrap::<dyn Host9053>(PhantomData));
}

// ---------------------------------------------------------------------------------------------
// Conflicts accepted with `LeakerResult::allow()` (`#[prop_leak_accept]`).
// ---------------------------------------------------------------------------------------------

/// The Leaker rejects an item whose part cannot be leaked, and accepts it when the cause of the
/// error is allowed: that part is copied as written, whole, and the rest is leaked as usual. The
/// Referrer's crate names what the copy names (`accepted`, in `tests/crates/referrer`), and
/// implements the traits for hosts, or declares the copy, so it compiles only if the part resolves
/// to the same item and everything else is replaced. The Leaker's own tests (`tests/allow.rs`) read
/// the entries; these are about the copy.
#[test]
fn accepted_conflicts_are_copied_as_written() {
    use type_leak_test_leaker::kinds as l;
    use type_leak_test_leaker::*;
    use type_leak_test_referrer::accepted as r;

    // The reports: `acc`, the entries (the part that conflicts has none), and the errors of the
    // first analysis (the `Reason`, then the kind of the cause).
    let table = [
        // (a) `Held<'b>` leaves the lifetime of the method unconstrained (E0195 in the host's
        // impl): copied as written. Nothing is leaked (`u8` is allowed). Both types are compared.
        (9184, __tl_report_9184!(), allowed("trait", 0, 2, 2)),
        // (b) the parameter of `f` has an elided lifetime in the generics of a path, and `g`'s
        // `dyn PlainTr` is an entry (the path around it is allowed by the Leaker). Only `g`'s
        // parameter is compared (`f`'s has an elided lifetime, which a witness cannot name).
        (9185, __tl_report_9185!(), allowed("trait", 1, 2, 1)),
        // (c) `Wrap<Self>`, where `Self` has no meaning, is copied as written; `Rel` is the entry.
        (9186, __tl_report_9186!(), allowed("struct", 1, 2, 2)),
        // (d) the same trait object in `f` and `g`: in `f` it is in the part that is copied as
        // written (the path with the elided lifetime), in `g` it is the entry. The Visitor finds
        // the entry by its tokens and its context, and must leave the part alone.
        (9187, __tl_report_9187!(), allowed("trait", 1, 2, 1)),
        // (e) `Wrap<Y>` is an entry in `a1`, and in the bound of `ai` it is in the part that is
        // copied as written (`impl Into<Wrap<Y>>`): the same tokens and context.
        (9188, __tl_report_9188!(), allowed("trait", 2, 2, 1)),
        // (f) the same as (d), with a relative type beside the trait object in the path of `f`: it
        // is an entry, and replaced, whereas the Referrer's crate does not name it.
        (9189, __tl_report_9189!(), allowed("trait", 2, 2, 1)),
    ];
    for (id, text, expected) in table {
        assert_eq!(reported(id, text), expected, "item {}: {}", id, text);
    }

    // The errors that were allowed.
    assert_eq!(
        errors(__tl_errors_9184!()),
        [error("UnconstrainedLifetime", "Type", "Held<'b>")]
    );
    assert_eq!(
        errors(__tl_errors_9185!()),
        [error("RelativePath", "Trait", "ElidedTr")]
    );
    assert_eq!(
        errors(__tl_errors_9186!()),
        [error("SelfWithoutMeaning", "Type", "Self")]
    );
    assert_eq!(
        errors(__tl_errors_9187!()),
        [error("RelativePath", "Trait", "SharedTr")]
    );
    assert_eq!(
        errors(__tl_errors_9188!()),
        [error("UnprovableBound", "Type", "Wrap<Y>")]
    );
    assert_eq!(
        errors(__tl_errors_9189!()),
        [error("RelativePath", "Trait", "SharedTr")]
    );

    // (a) The host implements the trait, with `Held<'b>` in the signature as written.
    fn held_input<T: l::HeldInput>() {}
    held_input::<r::Host9184>();
    // (b) `f` keeps its elided lifetime and its trait object, and `g` takes the entry, which is
    // `dyn PlainTr`.
    fn elided_and_plain<T: l::ElidedAndPlain>() {}
    elided_and_plain::<r::Host9185>();
    // (d) `f` keeps its elided lifetime and its trait object as written, whatever `g` has.
    fn elided_and_bare<T: l::ElidedAndBare>() {}
    elided_and_bare::<r::Host9187>();
    // (e) `ai` keeps `impl Into<Wrap<Y>>`, and `a1` and the bound have the entries.
    fn entry_and_bound<T: l::EntryAndBound>() {}
    entry_and_bound::<r::Host9188>();
    // (f) `f` keeps `dyn SharedTr`, and the relative type beside it is an entry, like `g`'s.
    fn elided_with_sibling<T: l::ElidedWithSibling>() {}
    elided_with_sibling::<r::Host9189>();
    // (c) The copy holds the original types: `Self` is replaced by the Visitor with the Leaker's
    // type, as it is for any struct.
    let copy = r::NoSelfMeaning {
        rel: l::Rel(2),
        me: l::Wrap(PhantomData),
    };
    same(&copy.rel, &l::Rel(0));
    same(&copy.me, &l::Wrap::<l::NoSelfMeaning>(PhantomData));
}

/// The Leaker's macro passes its errors on (`#[prop_leak_result]`: the serialized `LeakerResult`),
/// and the Referrer's macro, in the crate `accepted` of `tests/crates/referrer`, drops them by what
/// they are: `allow(trait Clone)`, `allow(std_crates)`, and the causes written out,
/// `allow(trait PlainTr, type Held<'_>)`. If one remained, the macro would report it and the crate
/// would not build (`tests/diagnostic.rs` has that); so it is enough to see that the crate exists
/// and that the copy keeps the parts as written, with every other type replaced and checked.
#[test]
fn errors_the_referrers_macro_allows_are_copied_as_written() {
    use type_leak_test_leaker::kinds as l;
    use type_leak_test_leaker::*;
    use type_leak_test_referrer::accepted as r;

    // The reports of the Leaker's macro: the same entries as if it had accepted the errors (the
    // parts that conflict have none). Every type position is compared with its original, but the
    // one of `ExplicitCauses` that has an elided lifetime (`Held<'_>`), which a witness cannot
    // name.
    let table = [
        (9190, __tl_report_9190!(), allowed("trait", 2, 4, 4)),
        (9191, __tl_report_9191!(), allowed("trait", 2, 4, 4)),
        (9192, __tl_report_9192!(), allowed("trait", 2, 4, 3)),
    ];
    for (id, text, expected) in table {
        assert_eq!(reported(id, text), expected, "item {}: {}", id, text);
    }

    // The errors of the Leaker's analysis: every one has the reason `RelativePath`.
    let relative = |cause: &str, tokens: &[&str]| -> Vec<(String, String, String)> {
        tokens
            .iter()
            .map(|tokens| {
                (
                    "RelativePath".to_string(),
                    cause.to_string(),
                    tokens.to_string(),
                )
            })
            .collect()
    };
    assert_eq!(
        errors(__tl_errors_9190!()),
        relative("Trait", &["Clone", "Clone", "Clone"])
    );
    assert_eq!(
        errors(__tl_errors_9191!()),
        relative("Trait", &["core::fmt::Debug", "std::fmt::Display"])
    );
    let mut explicit = relative("Trait", &["PlainTr", "PlainTr"]);
    explicit.insert(
        1,
        (
            "RelativePath".to_string(),
            "Type".to_string(),
            "Held<'_>".to_string(),
        ),
    );
    assert_eq!(errors(__tl_errors_9192!()), explicit);

    // The hosts implement the Leaker's traits, whose signatures hold the original types.
    fn prelude_bounds<T: l::PreludeBounds<u8>>() {}
    prelude_bounds::<r::Host9190>();
    fn std_bounds<T: l::StdBounds<u8>>() {}
    std_bounds::<r::Host9191>();
    struct Plain;
    impl l::PlainTr for Plain {}
    fn explicit_causes<T: l::PlainTr, H: l::ExplicitCauses<T>>() {}
    explicit_causes::<Plain, r::Host9192>();
}

// ---------------------------------------------------------------------------------------------
// A crate without the dependency of the crate of an entry.
// ---------------------------------------------------------------------------------------------

/// The type of a crate which the Referrer's crate has no dependency on is an entry, unless the
/// Leaker allows the crate. After a Leaker that allows it the copy names the crate (the relative
/// type in it is replaced), and after the default one the copy replaces the whole type, so that the
/// crate is not named: the text of the two copies of the same items differs in that. Only the
/// second can be compiled where the crate is not a dependency (this crate has none on it): the
/// module `trusting` of `tests/diagnostic.rs` builds both in a crate that has none, and runs the
/// tests of the second.
#[test]
fn a_leaker_that_allows_a_crate_has_the_copy_name_it_and_the_default_one_does_not() {
    use type_leak_test_referrer::{ALLOWED_COPY, REPLACED_COPY};
    let (allowed, replaced) = (squash(ALLOWED_COPY), squash(REPLACED_COPY));

    // Allowed: `Ext<(RelX,)>` names the crate with the relative type replaced (the entry 0), and
    // `Ext<u8>` is as written.
    assert!(allowed.contains("x:::type_leak_test_ext::Ext<(<Wrapperas$crate::Repeater<(),"));
    assert!(allowed.contains("u128>>::Type,)>,y:::type_leak_test_ext::Ext<u8>"));
    let keys = repeater_keys(&allowed);
    assert_eq!(keys.len(), 1, "{}", allowed);
    let (_, entry) = keys[0];
    assert_eq!(entry, 0);
    assert_eq!(allowed.matches("type_leak_test_ext").count(), 2);
    assert_eq!(allowed.matches("Repeater<").count(), 1);

    // Replaced: both types of the other crate are replaced as a whole, each by its own entry (0 and
    // 1), and the crate is not named at all.
    assert!(!replaced.contains("type_leak_test_ext"));
    assert!(replaced.contains("x:<Wrapperas$crate::Repeater<(),"));
    assert!(replaced.contains("u128>>::Type,y:<Wrapperas$crate::Repeater<(),"));
    assert!(replaced.ends_with("u128>>::Type){}"));
    let (replaced_id, _) = repeater_keys(&replaced)[0];
    assert_eq!(
        repeater_keys(&replaced),
        [(replaced_id, 0), (replaced_id, 1)]
    );
    assert_eq!(replaced.matches("Repeater<").count(), 2);
    assert_ne!(allowed, replaced);
}

// ---------------------------------------------------------------------------------------------
// Fragments of impls, analyzed with the scope API.
// ---------------------------------------------------------------------------------------------

/// `coinduction`'s shape: `enter_impl(..)` with `visit_path_arguments()`, the
/// Repeater's impls for a marker (`ImplSelf::Marker`) of the entries, and the Referrer's copy of the self type, the trait and its
/// arguments with `Referrer::visitor(context, ..)` and `Projection::of()` with the generics
/// passed by their own names. The Referrer's crate checks the copy: each type is the
/// original (a witness), and rustc finds
/// the Leaker's impl of the trait for the rewritten self type.
#[test]
fn the_scope_api_with_a_marker() {
    // Three required entries: `Rel3` (no generics), `Rel<T>` and `&'a Rel2` (the lifetime encoded
    // in the first argument of the Repeater).
    assert_eq!(
        type_leak_test_leaker::__scope_report_7001!(),
        "7001|entries=3 impls=3 types=4"
    );
    let (self_ty, trait_path) = (
        squash(type_leak_test_referrer::scoped::SCOPE_7001_SELF),
        squash(type_leak_test_referrer::scoped::SCOPE_7001_TRAIT),
    );
    let marker = "<::type_leak_test_leaker::scoped::M7001as::type_leak_test_leaker::Repeater<";
    // `T` of the self type stays (a generic of the impl), `Rel3` goes through the marker.
    assert_eq!(
        self_ty,
        format!(
            "::type_leak_test_leaker::scoped::Holder<T,{}(),{}>>::Type>",
            marker,
            key_text(7001, 0)
        )
    );
    assert_eq!(
        trait_path,
        format!(
            "::type_leak_test_leaker::scoped::Tr<{m}(T,),{k1}>>::Type,{m}(&'a(),),{k2}>>::Type>",
            m = marker,
            k1 = key_text(7001, 1),
            k2 = key_text(7001, 2)
        )
    );
    // The impls: the entry without generics is plain, the others declare what they use (and the
    // bound of `T`, but not the lifetime of the reference, which the first argument encodes).
    assert_eq!(
        type_leak_test_leaker::__scope_impls_7001!()
            .split(";;")
            .map(squash)
            .collect::<Vec<_>>(),
        [
            format!("implcrate::Repeater<(),{}>forM7001", key_text(7001, 0)),
            format!(
                "impl<T:::core::clone::Clone>crate::Repeater<(T,),{}>forM7001",
                key_text(7001, 1)
            ),
            format!(
                "impl<'a>crate::Repeater<(&'a(),),{}>forM7001",
                key_text(7001, 2)
            ),
        ]
    );
    // The types that were visited, one by one.
    assert_eq!(
        type_leak_test_referrer::scoped::SCOPE_7001_TYPES
            .split(";;")
            .map(squash)
            .collect::<Vec<_>>(),
        vec![
            "T".to_string(), // never replaced: it is the generic of the impl
            format!("{}(),{}>>::Type", marker, key_text(7001, 0)),
            format!("{}(T,),{}>>::Type", marker, key_text(7001, 1)),
            format!("{}(&'a(),),{}>>::Type", marker, key_text(7001, 2)),
        ]
    );
}

/// `newer-type`'s shape: `Self` is allowed, and the Repeater's impl of an entry that names it is for
/// the self type (`ImplSelf::Param`). A marker is refused then (the call of `repeater_impls()`
/// fails), so the impl of the one that does not is for a parameter too, and `Projection::of()`
/// writes both for `Self`.
#[test]
fn the_scope_api_with_self() {
    assert_eq!(
        type_leak_test_leaker::__scope_report_7002!(),
        "7002|entries=2 impls=2 types=4"
    );
    let trait_path = squash(type_leak_test_referrer::scoped::SCOPE_7002_TRAIT);
    // `Wrap<Self>` is in the impl of the self type itself, `Rel<T>` in a blanket impl for `Self`.
    assert_eq!(
        trait_path,
        format!(
            "::type_leak_test_leaker::scoped::Tr<\
             <::type_leak_test_leaker::scoped::Holder<T,u8>as::type_leak_test_leaker::Repeater<(T,),{}>>::Type,\
             <::type_leak_test_leaker::scoped::Holder<T,u8>as::type_leak_test_leaker::Repeater<(T,),{}>>::Type>",
            key_text(7002, 0),
            key_text(7002, 1)
        )
    );
    assert_eq!(
        type_leak_test_leaker::__scope_impls_7002!()
            .split(";;")
            .map(squash)
            .collect::<Vec<_>>(),
        [
            format!(
                "impl<T:::core::clone::Clone>crate::Repeater<(T,),{}>forcrate::scoped::Holder<T,u8>",
                key_text(7002, 0)
            ),
            format!(
                "impl<T:::core::clone::Clone,__S:?::core::marker::Sized>crate::Repeater<(T,),{}>for__S",
                key_text(7002, 1)
            ),
        ]
    );
    // The self type has nothing to replace.
    assert_eq!(
        squash(type_leak_test_referrer::scoped::SCOPE_7002_SELF),
        "::type_leak_test_leaker::scoped::Holder<T,u8>"
    );
}

/// Entries which need no generics are plain: the impl for the marker declares nothing. The type of
/// another crate is an entry, and has its impl; with a Leaker that allows the absolute paths
/// (`external`) it is none, and the copy writes it as it is.
#[test]
fn the_scope_api_writes_the_impls_that_are_needed() {
    assert_eq!(
        type_leak_test_leaker::__scope_report_7003!(),
        "7003|entries=2 impls=2 types=4"
    );
    assert_eq!(
        type_leak_test_leaker::__scope_report_7004!(),
        "7004|entries=1 impls=1 types=4"
    );
    assert_eq!(
        type_leak_test_leaker::__scope_report_7005!(),
        "7005|entries=2 impls=2 types=4"
    );
    // A where clause on a type that is not a parameter (`Vec<T>: Clone`) is in the context of the
    // entry that needs it (`Cl<T>` is well-formed only with it), so the impl of the Repeater has it.
    assert_eq!(
        type_leak_test_leaker::__scope_report_7006!(),
        "7006|entries=1 impls=1 types=4"
    );
    assert_eq!(
        squash(type_leak_test_leaker::__scope_impls_7006!()),
        format!(
            "impl<T>crate::Repeater<(T,),{}>forM7006where::std::vec::Vec<T>:::core::clone::Clone",
            key_text(7006, 0)
        )
    );
    // With `external`, the type of the other crate is no entry: only `Rel<u8>` has an impl.
    assert_eq!(
        type_leak_test_leaker::__scope_report_7007!(),
        "7007|entries=1 impls=1 types=4"
    );
    // (The module `trusting` of `tests/diagnostic.rs` applies 7005 in a crate that has no
    // dependency on the crate of the second entry.)
}

/// The errors of the Leaker (`Vec<ErrorItem>`) through a macro: one item for each offending path,
/// in order, with its message, the tokens, the source text at its span and the notes with theirs.
/// The macro reported them as a const, and as many `compile_error!`s.
#[test]
fn errors_are_reported_item_by_item_through_a_macro() {
    use type_leak_test_leaker::scoped::*;
    type Row = (String, String, String, Vec<(String, String)>);
    type Rows = &'static [(
        &'static str,
        &'static str,
        &'static str,
        &'static [(&'static str, &'static str)],
    )];
    fn rows(rows: Rows) -> Vec<Row> {
        rows.iter()
            .map(|(message, tokens, source, notes)| {
                (
                    squash(message),
                    squash(tokens),
                    source.to_string(),
                    notes
                        .iter()
                        .map(|(note, at)| (note.to_string(), at.to_string()))
                        .collect(),
                )
            })
            .collect()
    }
    let because = |what: &str| format!("it cannot be leaked as a whole because of `{}`", what);

    // Three paths with an elided lifetime: the self type's argument first, then the trait's. The
    // note is at the lifetime, the error at the path (its first segment, or the whole path on a
    // compiler which joins spans).
    let multi = rows(SCOPE_ERRORS_MULTI);
    assert_eq!(multi.len(), 3);
    let expected = [
        ("useabsolutepath:'Three<'_>'", "Three<'_>", "Three"),
        ("useabsolutepath:'One<'_>'", "One<'_>", "One"),
        ("useabsolutepath:'Two<'_,T>'", "Two<'_,T>", "Two"),
    ];
    for ((message, tokens, source, notes), (e_message, e_tokens, e_source)) in
        multi.iter().zip(expected)
    {
        assert_eq!(message, e_message);
        assert_eq!(tokens, e_tokens);
        assert!(source.starts_with(e_source), "{:?}", source);
        assert_eq!(notes, &[(because("'_"), "'_".to_string())]);
    }
    assert_eq!(SCOPE_ERRORS_MULTI_COMPILE_ERRORS, 3);

    // `Self` where no scope declares it: one item for each mention, at the mention.
    let without = rows(SCOPE_ERRORS_SELF_WITHOUT_MEANING);
    assert_eq!(without.len(), 2);
    let note = "`Self` has no meaning here: it is declared only by a struct, an enum, a union, an impl or a trait, not by a module or generics";
    for (row, tokens) in without.iter().zip(["Self", "Self::Item"]) {
        assert_eq!(row.0, "`Self`hasnomeaninghere");
        assert_eq!(row.1, tokens);
        assert!(row.2.starts_with("Self"), "{:?}", row.2);
        assert_eq!(row.3.len(), 1);
        assert_eq!(row.3[0].0, note);
        assert!(row.3[0].1.starts_with("Self"));
    }
    assert_eq!(SCOPE_ERRORS_SELF_WITHOUT_MEANING_COMPILE_ERRORS, 2);
}

// ---------------------------------------------------------------------------------------------
// The parts and the optional entries that the Visitor must tell apart (tests/findings.rs F18, F43
// to F45).
// ---------------------------------------------------------------------------------------------

/// F18: `Held<'b>` is the only input of `f` that holds the method lifetime, written with an
/// absolute path of the Leaker's crate: an optional entry. A strict Referrer used to replace it,
/// and the impl of the host had `'b` early-bound where the trait has it late-bound (E0195, in
/// `tests/crates/referrer`, where it is invoked as `strict`). The Leaker keeps it as written now,
/// for every Referrer: the only entry is `Rel`.
#[test]
fn an_optional_entry_that_holds_a_method_lifetime_is_copied_as_written() {
    use type_leak_test_leaker::kinds as l;
    use type_leak_test_leaker::*;
    use type_leak_test_referrer::kinds as r;

    assert_eq!(reported(9194, __tl_report_9194!()), ok("trait", 1, 5, 5));
    fn holds_lifetime<T: l::OptionalHoldsLifetime>() {}
    holds_lifetime::<r::Host9194>();
}

/// F44 and F45: parts that the Visitor must leave as written, with a trait object inside whose
/// lifetime comes from the path around it (`dyn SharedTr + 'a`), and the entry `dyn SharedTr`
/// (`+ 'static`) elsewhere in the same item. The copy compiles in the Referrer's crate only if the
/// part is not rewritten. The conflicts are about `Self`: named by a predicate of the trait, and
/// named by the bound of an impl whose self type holds the generic. The struct nested in a module
/// whose bound names its `Self` (9198) is no part any more: it is an entry for any type, with the
/// risk `NestedSelfInBounds`, which the accepted copy writes with its own type.
#[test]
fn parts_that_name_self_are_left_as_written() {
    use type_leak_test_leaker::kinds as l;
    use type_leak_test_leaker::*;
    use type_leak_test_referrer::accepted as r;

    let table = [
        (9196, __tl_report_9196!(), allowed("trait", 1, 1, 1)),
        (9197, __tl_report_9197!(), allowed("impl-inherent", 1, 2, 2)),
        (9198, __tl_report_9198!(), allowed("struct", 2, 2, 2)),
    ];
    for (id, text, expected) in table {
        assert_eq!(reported(id, text), expected, "item {}: {}", id, text);
    }

    // The errors that were allowed: the reason, the kind of the cause and its tokens.
    let obj = "::type_leak_test_leaker::kinds::ObjOf<'a,T,dynSharedTr>";
    assert_eq!(
        errors(__tl_errors_9196!()),
        [error(
            "SelfInTraitPredicate",
            "Type",
            "::type_leak_test_leaker::kinds::ObjOf<'a,Self,dynSharedTr>"
        )]
    );
    assert_eq!(
        errors(__tl_errors_9197!()),
        [
            error("SelfInImplSelfType", "Type", obj),
            error("UnprovableBound", "Type", "Self")
        ]
    );
    assert_eq!(
        errors(__tl_errors_9198!()),
        [error(
            "NestedSelfInBounds",
            "Type",
            "::type_leak_test_leaker::kinds::ObjOf<'a,(Self,T),dynSharedTr>"
        )]
    );

    // The hosts implement the traits, with `ObjOf<'a, .., dyn SharedTr + 'a>` in the signature as
    // written, and the entry in `g`.
    fn predicate<T: l::PredicateObject<'static>>() {}
    predicate::<r::Host9196>();
}

/// F43: the impl nested in the scope of a trait, whose generic has a bound that names the `Self` of
/// the trait, has a type with no context. It is copied as written, and the entry `dyn SharedTr` of
/// the other type is replaced.
#[test]
fn a_part_with_no_context_is_left_as_written() {
    use type_leak_test_leaker::*;
    use type_leak_test_referrer::accepted as r;

    assert_eq!(
        __nested_report_9199!(),
        "9199|entries=1|errors=UnnameableSelf"
    );
    assert_eq!(
        squash(r::NESTED_9199_TYPES),
        squash(&format!(
            "::type_leak_test_leaker::kinds::ObjOf<'a, T, dyn SharedTr> ;; \
             ::type_leak_test_leaker::kinds::Bare<<() as ::type_leak_test_leaker::Repeater<(), \
             {}>>::Type>",
            key_text(9199, 0)
        ))
    );
}

/// F48 (E0195 and E0106): two items analyzed together, a trait and a struct. The method that has
/// the fault keeps the type as written, where the module that copies the trait names it
/// (`accepted`); the other methods and the struct, copied in a module that names neither `Held` nor
/// `Wrap` (`unnamed`), have the entry.
#[test]
fn a_type_copied_as_written_for_a_signature_is_left_in_a_signature_like_it_only() {
    use std::marker::PhantomData;
    use type_leak_test_leaker::kinds as l;
    use type_leak_test_leaker::*;
    use type_leak_test_referrer::{accepted, unnamed};

    assert_eq!(
        __shared_report_9202!(),
        "9202|entries=1|errors=UnconstrainedLifetime"
    );
    assert_eq!(
        __shared_report_9203!(),
        "9203|entries=2|errors=RelativePath"
    );

    // The copies of the traits are implementations of the Leaker's: rustc compares the signatures.
    fn held<T: l::HeldInTwoMethods>() {}
    held::<accepted::Host9202x0>();
    fn elided<T: l::ElidedInOneMethod<'static, u8>>() {}
    elided::<accepted::Host9203x0>();

    // The fields of the structs are the original types.
    let n = 5;
    let copy = unnamed::HeldHolder { held: l::Held(&n) };
    same(&copy.held, &l::Held(&n));
    let copy = unnamed::ElidedHolder::<u8>(PhantomData, l::Wrap(PhantomData));
    same(&copy.1, &l::Wrap::<u8>(PhantomData));
}

/// F49 (E0106): a lifetime that another input keeps. `f` keeps the entry `Wrap<T>` as written,
/// where `'a` is also in `x`; `g`, `h` and `k` and the struct have the entry (the struct is copied
/// in the module that names neither `Held` nor `Wrap`).
#[test]
fn a_lifetime_that_another_input_keeps_breaks_the_elision_of_the_copy() {
    use std::marker::PhantomData;
    use type_leak_test_leaker::kinds as l;
    use type_leak_test_leaker::*;
    use type_leak_test_referrer::{accepted, unnamed};

    assert_eq!(
        __shared_report_9204!(),
        "9204|entries=2|errors=RelativePath"
    );

    // The copy of the trait is an implementation of the Leaker's: rustc compares the signatures.
    fn kept<T: l::KeptInOtherInput<'static, u8>>() {}
    kept::<accepted::Host9204x0>();

    let copy = unnamed::KeptHolder::<u8>(PhantomData, l::Wrap(PhantomData));
    same(&copy.1, &l::Wrap::<u8>(PhantomData));
}

/// F72 (E0195): three traits analyzed together. `f` has `Held<'b>` as the only input that names the
/// lifetime of its output: it is copied as written, in the module that names `Held` (`accepted`).
/// `g` and `h` keep it in another input (`&'b u8`, `Option<&'b u8>`), so `Held` is an entry there,
/// written through the Repeater in a module that names no `Held` (`unnamed`): the part of `f` was
/// copied as written in any signature whose output names the lifetime. The third trait has a
/// method with two culprits (`Held<'b>` in each input) and another with one, all copied as
/// written.
#[test]
fn a_type_copied_as_written_for_a_lifetime_is_left_only_where_the_copy_leaves_it_unconstrained() {
    use type_leak_test_leaker::kinds as l;
    use type_leak_test_leaker::*;
    use type_leak_test_referrer::{accepted, unnamed};

    assert_eq!(
        __shared_report_9245!(),
        "9245|entries=1|errors=UnconstrainedLifetime,UnconstrainedLifetime,\
         UnconstrainedLifetime,UnconstrainedLifetime"
    );

    // The copies of the traits are implementations of the Leaker's: rustc compares the signatures.
    fn culprit<T: l::HeldCulprit>() {}
    culprit::<accepted::Host9245x0>();
    fn kept<T: l::HeldKept>() {}
    kept::<unnamed::Host9245x1>();
    fn twice<T: l::HeldCulpritTwice>() {}
    twice::<accepted::Host9245x2>();
}

/// F50 (`stages = true`): `core::mem::size_of::<*const Rel>()` in a default method. The path of the
/// value is written as it is and `Rel` in its arguments is an entry, so `Rules::std_crates()` given
/// to the result after the analysis drops the only error, as the config that allows the crates has
/// none: the copy is made in the `clean` crate too, which names no `Rel` (it would be E0412 if the
/// copy wrote `Rel` as written).
#[test]
fn a_rule_after_the_analysis_drops_the_error_for_the_path_of_a_value_and_leaks_its_arguments() {
    use type_leak_test_leaker::kinds as l;
    use type_leak_test_leaker::*;
    use type_leak_test_referrer::stage_copies as r;

    // What the stages leave: the rule matters, one error without it, none in the config (`Rel` is
    // an entry) and none after it in the result (the path of the value is dropped, `Rel` is an
    // entry too): the same errors, the same Referrer.
    let fields: Vec<&str> = __tl_stages_std_9205!().split('\u{1f}').collect();
    assert_eq!(fields[..4], ["1", "1", "0", "0"]);
    assert_eq!(fields[4], "");
    assert_eq!(fields[5], "");
    assert_eq!(fields[6..], ["1", "1"]);

    // The copy after the rule is a clean one: no error is left.
    let report = __tl_st_std_res_report_9205!();
    let status: Vec<&str> = report
        .split('\u{1e}')
        .next()
        .unwrap()
        .splitn(4, '|')
        .collect();
    assert_eq!(status[..3], ["9205", "trait", "ok"]);

    // The copy is an implementation of the Leaker's trait, and its default method is the original.
    fn implements<T: l::ValuePathArguments>() {}
    implements::<r::Host9205>();
    struct Copy;
    impl r::ValuePathArgumentsCopy for Copy {}
    use r::ValuePathArgumentsCopy;
    assert_eq!(Copy.size(), core::mem::size_of::<*const l::Rel>());
}

/// F52: the optional entries in the where predicate of a trait stay as written beside an unrelated
/// error whose part copies `Self::A0` as written, as they do without the error. The strict Referrer
/// (`accepted`) has no entry for them, and the host implements the trait.
#[test]
fn optional_entries_in_a_predicate_stay_as_written_beside_an_unrelated_error() {
    use type_leak_test_leaker::*;
    assert_eq!(
        reported(9207, __tl_report_9207!()),
        allowed("trait", 0, 0, 0)
    );
    // The one error is the unrelated one, accepted by its cause: the bound copied as written.
    assert_eq!(
        errors(__tl_errors_9207!()),
        [error("RelativePath", "Trait", "GenTr<Self::A0>")]
    );
}

/// Issue 3 of `tests/downstream_findings.rs`: `None`, `Some`, `Ok` and `Err` in the values of a
/// trait and `DoubleEndedIterator + ExactSizeIterator` in a bound are names of the prelude of every
/// edition, so `Rules::prelude()` accepts them (it rejected the trait before: the Referrer's crate
/// had no `PreludeNamesCopy`, which also fails to build the test). The copy writes them as they
/// are, replaces the relative type of the values, and computes what the original does.
#[test]
fn the_names_of_the_prelude_of_every_edition_are_accepted_in_a_trait() {
    use type_leak_test_leaker::kinds as l;
    use type_leak_test_leaker::*;
    use type_leak_test_referrer::kinds as r;

    assert_eq!(
        reported(9208, __tl_report_9208!()),
        ok("trait", 1, 7, 7),
        "{}",
        __tl_report_9208!()
    );

    struct Original;
    impl l::PreludeNames for Original {}
    struct Copy;
    impl r::PreludeNamesCopy for Copy {}

    assert_eq!(<Original as l::PreludeNames>::NOTHING, None);
    assert_eq!(<Copy as r::PreludeNamesCopy>::NOTHING, None);
    for n in [0u8, 1] {
        let original = l::PreludeNames::pick(&Original, n);
        assert_eq!(original.is_some(), n > 0);
        assert_eq!(r::PreludeNamesCopy::pick(&Copy, n), original);
        let original = l::PreludeNames::check(&Original, n);
        assert_eq!(original.is_ok(), n > 0);
        assert_eq!(r::PreludeNamesCopy::check(&Copy, n), original);
    }
    let items = [1u8, 2, 3];
    assert_eq!(l::PreludeNames::count(&Original, items.iter()), 6);
    assert_eq!(r::PreludeNamesCopy::count(&Copy, items.iter()), 6);
}

/// Fixture 9249 (F83): the default methods of `BlockNames` declare `Block9249` with `use
/// a::b::{self}` and `len9249` with a `const fn`, in a block. The types in the body are the local
/// ones, and the copy keeps them. The Referrer replaced them by the entries of the parameters
/// (`Block9249`, `[u8; len9249()]`), which is E0308 in the Referrer's crate (the struct for the
/// enum, an array of 2 for one of 3). The original and the copy compute the same.
#[test]
fn a_block_that_imports_with_self_or_declares_a_fn_hides_the_entries_that_name_it() {
    use type_leak_test_leaker::kinds as l;
    use type_leak_test_leaker::*;
    use type_leak_test_referrer::kinds as r;

    assert_eq!(
        reported(9249, __tl_report_9249!()),
        ok("trait", 2, 4, 4),
        "{}",
        __tl_report_9249!()
    );

    struct Original;
    impl l::BlockNames for Original {}
    struct Copy;
    impl r::BlockNamesCopy for Copy {}

    assert!(l::BlockNames::imported(&Original, l::Block9249(0)));
    assert!(r::BlockNamesCopy::imported(&Copy, l::Block9249(0)));
    assert_eq!(l::BlockNames::declared(&Original, [0; 2]), 3);
    assert_eq!(r::BlockNamesCopy::declared(&Copy, [0; 2]), 3);
}

/// The impls of the entries that do not name `Self` are for a primitive type that the id and the
/// index choose (`ImplSelf::Bucket`, which `prop_leak` uses for every item), the one that names it
/// is for the implementor. The copy compiles only if it writes the same type.
#[test]
fn the_entries_that_do_not_name_self_are_implemented_for_a_bucket() {
    use std::any::type_name;
    use type_leak_test_leaker as l;
    use type_leak_test_leaker::*;

    assert_eq!(
        reported(9209, __tl_report_9209!()),
        ok("trait", 5, 6, 6),
        "{}",
        __tl_report_9209!()
    );
    // The entries `Defd2<Rel, u8>` and `Rel`, that no generic is used in, are for the primitive
    // type that `Projection::of()` writes for `ImplSelf::Bucket` (the bucket is part of the format, so these
    // do not change without a bump of `Referrer::FORMAT_VERSION`). `Wrap<Self>` has none.
    assert_eq!(
        type_name::<<u64 as Repeater<(), { key(9209, 1) }>>::Type>(),
        type_name::<l::kinds::Defd2<l::kinds::Rel, u8>>()
    );
    assert_eq!(
        type_name::<<str as Repeater<(), { key(9209, 2) }>>::Type>(),
        type_name::<l::kinds::Rel>()
    );
}

/// The impls of the entries that name no `Self`, nor do their generics, are for a marker type of the
/// Leaker's crate (`ImplSelf::Marker`), the entry whose generics name `Self` (the default `Defd<T>`
/// of `trait MarkerDefault<T: AsRef<Self>, U = Defd<T>>`, `context_self()` is `None`) for the
/// implementor, which is the host in the copy. The Referrer's crate has an impl of the copy that
/// takes the default (`impl MarkerDefaultCopy<Owner9210> for Host9210`): it compiles only if the
/// default is written through the host (`Owner9210: AsRef<Host9210>`), not through the marker
/// (E0277, `Owner9210: AsRef<Marker9210>`).
#[test]
fn a_marker_is_for_the_entries_that_do_not_need_self_and_the_host_for_the_others() {
    use std::any::type_name;
    use type_leak_test_leaker as l;
    use type_leak_test_leaker::*;

    assert_eq!(
        reported(9210, __tl_report_9210!()),
        ok("trait", 2, 4, 4),
        "{}",
        __tl_report_9210!()
    );
    // `Rel` is for the marker.
    assert_eq!(
        type_name::<<l::kinds::Marker9210 as Repeater<(), { key(9210, 0) }>>::Type>(),
        type_name::<l::kinds::Rel>()
    );
}

/// F56: the bindings in an enum discriminant (`let`, a `match` arm, a pattern, an item of the
/// block) and in the default of a const parameter are local names, not relative paths. The Leaker
/// rejected them, and the Referrer's crate had no copy of either (which fails to build this test).
/// The copies write the expressions as they are and compute what the originals do.
#[test]
fn the_bindings_in_a_discriminant_and_in_the_default_of_a_const_parameter_are_local() {
    use type_leak_test_leaker::kinds as l;
    use type_leak_test_leaker::*;
    use type_leak_test_referrer::kinds as r;

    assert_eq!(
        reported(9215, __tl_report_9215!()),
        ok("enum", 0, 0, 0),
        "{}",
        __tl_report_9215!()
    );
    assert_eq!(
        reported(9216, __tl_report_9216!()),
        ok("struct", 0, 1, 1),
        "{}",
        __tl_report_9216!()
    );
    let original = [
        l::DiscriminantBindings::Let as isize,
        l::DiscriminantBindings::Arm as isize,
        l::DiscriminantBindings::Pair as isize,
        l::DiscriminantBindings::Typed as isize,
        l::DiscriminantBindings::Item as isize,
    ];
    assert_eq!(original, [3, 6, 12, 7, 42]);
    let copy = [
        r::DiscriminantBindings::Let as isize,
        r::DiscriminantBindings::Arm as isize,
        r::DiscriminantBindings::Pair as isize,
        r::DiscriminantBindings::Typed as isize,
        r::DiscriminantBindings::Item as isize,
    ];
    assert_eq!(copy, original);

    // The default of `N` is 4 in both: `DefaultBinding` is `DefaultBinding<4>`.
    let original: l::DefaultBinding = l::DefaultBinding([1; 4]);
    let copy: r::DefaultBinding = r::DefaultBinding([1; 4]);
    assert_eq!(original.0.len(), 4);
    assert_eq!(copy.0, original.0);
}

/// Flat recovery: a type whose own path conflicts is written as it is, and the types in it are
/// leaked as usual. `Two<T, Leaf>` is a conflict (`SelfInImplSelfType` in an impl whose bound names
/// `Self`), copied with `Two` as written, and `Leaf` in it is the entry that `g` has too: the
/// Referrer's crate replaces it (`Two<T, <..>::Type>`).
#[test]
fn a_type_whose_own_path_conflicts_is_written_as_it_is_and_the_types_in_it_are_leaked() {
    use type_leak_test_leaker::*;

    assert_eq!(
        reported(9220, __tl_report_9220!()),
        allowed("impl-inherent", 1, 2, 2)
    );
    let reasons = |text: &'static str| -> Vec<String> {
        errors(text)
            .into_iter()
            .map(|(reason, _, _)| reason)
            .collect()
    };
    assert_eq!(
        reasons(__tl_errors_9220!()),
        ["SelfInImplSelfType", "UnprovableBound"]
    );
}

/// Flat recovery: the types in a type written as it is that hold the lifetime of a method are
/// written as they are too. `Two<Leaf, *mut Held<'b>>` is the only input of `f` that names `'b`
/// (`UnconstrainedLifetime`), and `Held<'b>` in it is leaked as an entry, so the copy had `'b` in a
/// projection only: early-bound where the trait of the Leaker's crate has it late-bound (E0195 in
/// the impl of the host). Each type on every way to the lifetime that is leaked whole is an error,
/// and written as it is, `Leaf` alone is an entry. The host implements the trait of the Leaker's
/// crate.
#[test]
fn the_types_that_hold_a_method_lifetime_in_a_type_written_as_it_is_are_written_as_they_are() {
    use type_leak_test_leaker::kinds as l;
    use type_leak_test_leaker::*;
    use type_leak_test_referrer::accepted as r;

    assert_eq!(
        reported(9221, __tl_report_9221!()),
        allowed("trait", 1, 6, 6),
        "{}",
        __tl_report_9221!()
    );
    let causes = |text: &'static str| -> Vec<(String, String)> {
        errors(text)
            .into_iter()
            .map(|(reason, _, tokens)| (reason, tokens))
            .collect()
    };
    let unconstrained = |causes: &[&str]| -> Vec<(String, String)> {
        causes
            .iter()
            .map(|cause| ("UnconstrainedLifetime".to_string(), cause.to_string()))
            .collect()
    };
    assert_eq!(
        causes(__tl_errors_9221!()),
        unconstrained(&[
            "Two<Leaf,*mutHeld<'b>>",
            "Held<'b>",
            "Two<Two<Leaf,Held<'c>>,Leaf>",
            "Two<Leaf,Held<'c>>",
            "Held<'c>",
            "Two<Held<'d>,*mutHeld<'d>>",
            "Held<'d>",
            "Held<'d>",
        ])
    );
    fn hosts<T: l::NestedMethodLifetime>() {}
    hosts::<r::Host9221>();

    // An optional entry on the way to the lifetime is written as it is for a strict Referrer too.
    assert_eq!(
        reported(9222, __tl_report_9222!()),
        allowed("trait", 1, 2, 2),
        "{}",
        __tl_report_9222!()
    );
    assert_eq!(
        causes(__tl_errors_9222!()),
        unconstrained(&[
            "Two<Leaf,::type_leak_test_leaker::kinds::Wrap<Held<'b>>>",
            "Held<'b>",
        ])
    );
    fn strict_hosts<T: l::NestedOptionalLifetime>() {}
    strict_hosts::<r::Host9222>();
}

/// F64: `Wrap<Self>` in a predicate of a trait, and `Wrap<V>` in the predicate of a method that
/// equates the `Item` of a generic of the trait, next to a bound of the trait that names the
/// generic through a type alias (`U: Fn(WIt<U>)`, `WIt<U>` is `<U as Iterator>::Item`). The
/// Repeater's impl of the entry needs the bound, which rustc proves in the environment of the copy
/// only as written, not normalized (E0277 "expected an `Fn(u8)` closure, found `U`", in the impl
/// of the host in the Referrer's crate, which does not build). The Leaker cannot tell the alias
/// from a struct, so it rejects both entries, and the Referrer's macro expands to nothing.
#[test]
fn a_bound_that_names_its_generic_through_an_alias_is_rejected_where_the_entry_keeps_it() {
    use type_leak_test_leaker::*;

    let headline = |ty: &str, generic: &str| {
        format!(
            "`{}` is in a bound or where predicate, and a bound of its generics names `{}` in the \
             arguments of `Fn (::type_leak_test_leaker::kinds::WIt<{}>) -> u8`, which may hide a \
             projection",
            ty, generic, generic
        )
    };
    assert_eq!(
        reported(9223, __tl_report_9223!()),
        rejected(
            "trait",
            &squash(&format!(
                "{} {}",
                headline("Wrap<Self>", "U"),
                headline("Wrap<V>", "V")
            ))
        ),
        "{}",
        __tl_report_9223!()
    );
}

/// F67: a trait object bounded by the lifetime of a method (`Box<dyn Fn(Rel) + 'x>`) is the only
/// input that holds it, in the arguments of a type of a crate that the Leaker does not allow
/// (`Vec`, `Option`, a tuple, a second `Box`). The Leaker made the box an entry beside the type
/// around it, so the copy had `'x` in a projection only: early-bound where the trait has it
/// late-bound (E0195 in the impl of the host, in `tests/crates/referrer`, where it is invoked as
/// `strict`). Every optional type on the way to the lifetime is written as it is now, and the only
/// entry is `Rel`. The host implements the trait of the Leaker's crate.
#[test]
fn the_optional_types_around_an_object_bounded_by_a_method_lifetime_are_written_as_they_are() {
    use type_leak_test_leaker::kinds as l;
    use type_leak_test_leaker::*;
    use type_leak_test_referrer::kinds as r;

    assert_eq!(
        reported(9228, __tl_report_9228!()),
        ok("trait", 1, 2, 2),
        "{}",
        __tl_report_9228!()
    );
    assert_eq!(
        reported(9229, __tl_report_9229!()),
        ok("trait", 1, 6, 6),
        "{}",
        __tl_report_9229!()
    );
    fn in_vec<T: l::ObjectInVec>() {}
    in_vec::<r::Host9228>();
    fn under_types<T: l::ObjectUnderTypes>() {}
    under_types::<r::Host9229>();
}

/// Fixture 9236: const generics of type `u64` and `u128` in the arguments of the Repeater. The
/// slots put a `u128` into `KEY` (so the entry that uses only it has the argument `(*const dyn
/// Repeater<(), M, Type = ()>,)`) and a `u64` into the slot `U64`; the Referrer's crate compiles
/// the copy, which writes the same arguments.
#[test]
fn const_generics_of_type_u64_and_u128_use_the_slots() {
    use std::any::type_name;
    use type_leak_test_leaker as l;
    use type_leak_test_leaker::*;

    assert_eq!(
        reported(9236, __tl_report_9236!()),
        ok("struct", 4, 4, 4),
        "{}",
        __tl_report_9236!()
    );
    // The entries are for the marker, in the order of the fields. `N` is `5` and `M` is `7`; the
    // first entry has the argument of `M` before that of `N`.
    type Marker = l::kinds::Marker9236;
    assert_eq!(
        type_name::<
            <Marker as Repeater<
                (
                    *const dyn Repeater<(), 7, Type = ()>,
                    *const dyn Repeater<(), 0, false, '\0', 0, 0, 0, 5, Type = ()>,
                ),
                { key(9236, 0) },
            >>::Type,
        >(),
        type_name::<l::kinds::Keyed<5, 7>>()
    );
    assert_eq!(
        type_name::<
            <Marker as Repeater<
                (*const dyn Repeater<(), 0, false, '\0', 0, 0, 0, 5, Type = ()>,),
                { key(9236, 1) },
            >>::Type,
        >(),
        type_name::<l::kinds::Wrap<l::kinds::Keyed<5, 0>>>()
    );
    assert_eq!(
        type_name::<
            <Marker as Repeater<(*const dyn Repeater<(), 7, Type = ()>,), { key(9236, 2) }>>::Type,
        >(),
        type_name::<l::kinds::Wrap<l::kinds::Keyed<0, 7>>>()
    );
}

// ---------------------------------------------------------------------------------------------
// `LeakerResult::ignore()`.
// ---------------------------------------------------------------------------------------------

/// `LeakerResult::ignore()` in the macros of both crates (fixtures 9260 and 9261).
///
/// 9260: the Referrer's macro says `ignore(type Two<Leaf, Held<'_>>)`. That one call drops the
/// error of the type and the error inside it (`Held<'_>`), and the copy writes the whole type as it
/// is, `Leaf` in it too, while `y: Leaf` is an entry: the host implements the trait only if the
/// signature has the types of the original.
///
/// 9261: the field names the `Self` of a nested struct and uses a generic whose bound names it
/// (`NestedSelfInBounds`), and the type of the field needs `Clone` of its first parameter. The
/// entry for any type would have an impl that does not prove it (E0277 in the Leaker's crate: the
/// crate does not build if the macro takes the risk with `allow()` and not `ignore()`). The Leaker's
/// macro removes the entry, so there is no impl, and the copy has the field as written, with the
/// `Self` of the copy.
#[test]
fn ignore_writes_a_whole_part_as_written_and_removes_the_entry_of_a_risk() {
    use type_leak_test_leaker::kinds as l;
    use type_leak_test_leaker::*;
    use type_leak_test_referrer::accepted as r;

    assert_eq!(
        reported(9260, __tl_report_9260!()),
        // (`x` has an elided lifetime, which a witness cannot name.)
        allowed("trait", 2, 3, 2)
    );
    assert_eq!(
        errors(__tl_errors_9260!()),
        [
            error("RelativePath", "Type", "Two<Leaf,Held<'_>>"),
            error("RelativePath", "Type", "Held<'_>")
        ]
    );
    fn ignored_flat<T: l::IgnoredFlat>() {}
    ignored_flat::<r::Host9260>();

    // The entry of the risk is gone: one entry, `rel`, and its index is that of the entry that is
    // left.
    assert_eq!(reported(9261, __tl_report_9261!()), ok("struct", 1, 2, 2));
    // The fields of the copy are the original types, with the `Self` of the copy: this compiles.
    #[allow(dead_code)]
    fn field<T: AsRef<r::IgnoredEntry<T>>>(
        copy: &r::IgnoredEntry<T>,
    ) -> &l::NeedsClone<r::IgnoredEntry<T>, T> {
        &copy.needs
    }
    #[allow(dead_code)]
    fn rel<T: AsRef<r::IgnoredEntry<T>>>(copy: &r::IgnoredEntry<T>) -> &l::Rel {
        &copy.rel
    }
}

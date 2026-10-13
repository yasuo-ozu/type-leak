//! Uses `type-leak` across crate boundaries through a proc macro, as a downstream crate would.
//!
//! - `tests/crates/macro`: the proc macro using `type-leak` (`Leaker`, `Referrer` and
//!   `RepeaterSlots`).
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

use type_leak_test_leaker::m::{
    Consts, Defaulted, Defaults, DollarCrate, Edge, Holder, Inner, Plain, RelDefault, RelType,
};
use type_leak_test_referrer::{Concrete, Explicit, Generic, Generic2, Omitted};

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

/// A type equality witness: compiles only if both are the same type.
fn same<T>(_: &T, _: &T) {}

/// What a `#[prop_leak]` macro reported: `<id>|<kind>|ok|entries=E positions=P checked=C`, or
/// `<id>|<kind>|rej|<message>`.
#[derive(Debug, PartialEq)]
enum Reported {
    Ok {
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
        "mod",
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
    assert_eq!(status, "ok", "{}", text);
    let number = |key: &str| -> usize {
        rest.split(' ')
            .find_map(|word| word.strip_prefix(key))
            .unwrap_or_else(|| panic!("no `{}` in {}", key, text))
            .parse()
            .unwrap()
    };
    ok(
        kind,
        number("entries="),
        number("positions="),
        number("checked="),
    )
}

/// A struct, an enum, a union, an impl, a module and a trait are each analyzed by the matching
/// `intern_*` and accepted (or rejected, for a `Self` that is denied) as the Leaker's crate says, and
/// the Referrer's crate checked every copied type against the original: `checked` is the number of
/// type positions of the item which have a witness (all but the ones of a module that name what it
/// declares, which the copy declares again).
#[test]
fn every_item_kind_is_leaked_across_crates() {
    use type_leak_test_leaker::*;
    let table = [
        // structs: `Self`, a denied `Self` that is not named, a denied `Self` that is named, and an
        // optional entry (a type of another crate)
        (
            9001,
            reported(9001, __tl_report_9001!()),
            ok("struct", 5, 5, 5),
        ),
        (
            9002,
            reported(9002, __tl_report_9002!()),
            ok("struct", 3, 3, 3),
        ),
        (
            9003,
            reported(9003, __tl_report_9003!()),
            rejected("struct", "useabsolutepath:'Wrap<Self>'"),
        ),
        (
            9004,
            reported(9004, __tl_report_9004!()),
            ok("struct", 3, 3, 3),
        ),
        // a where clause on a type that is not a parameter, in the context of its entry
        (
            9005,
            reported(9005, __tl_report_9005!()),
            ok("struct", 2, 1, 1),
        ),
        // enums
        (
            9011,
            reported(9011, __tl_report_9011!()),
            ok("enum", 4, 5, 5),
        ),
        (
            9012,
            reported(9012, __tl_report_9012!()),
            ok("enum", 2, 2, 2),
        ),
        // unions
        (
            9021,
            reported(9021, __tl_report_9021!()),
            ok("union", 3, 3, 3),
        ),
        (
            9022,
            reported(9022, __tl_report_9022!()),
            ok("union", 2, 2, 2),
        ),
        // impls: inherent, of a trait (also with a relative argument), and with `Self` denied
        (
            9031,
            reported(9031, __tl_report_9031!()),
            ok("impl-inherent", 6, 6, 6),
        ),
        (
            9032,
            reported(9032, __tl_report_9032!()),
            ok("impl-trait", 2, 5, 5),
        ),
        (
            9033,
            reported(9033, __tl_report_9033!()),
            ok("impl-inherent", 2, 2, 2),
        ),
        (
            9034,
            reported(9034, __tl_report_9034!()),
            ok("impl-trait", 2, 5, 5),
        ),
        // a module (its own `Local` is declared in it, so the types that name it have no witness)
        (
            9041,
            reported(9041, __tl_report_9041!()),
            ok("mod", 4, 6, 5),
        ),
        // a bare identifier that is a const, in a module, beside a name that the module declares: an
        // absolute `use` declares it, so it is copied as written; one imported from outside is
        // rejected
        (
            9101,
            reported(9101, __tl_report_9101!()),
            ok("mod", 2, 4, 3),
        ),
        (
            9102,
            reported(9102, __tl_report_9102!()),
            rejected(
                "mod",
                "`LEN4`maybeaconst,andthetypethatholdsitcannotbeleakedasawhole",
            ),
        ),
        // traits: `Self` denied and not named, denied and named, and an unsized implementor
        (
            9051,
            reported(9051, __tl_report_9051!()),
            ok("trait", 3, 3, 3),
        ),
        (
            9052,
            reported(9052, __tl_report_9052!()),
            rejected("trait", "useabsolutepath:'Wrap<Self>'"),
        ),
        (
            9053,
            reported(9053, __tl_report_9053!()),
            ok("trait", 4, 6, 6),
        ),
        // a bare identifier that is a const (`LEN`): the path that holds it is an entry as a whole,
        // also with `Self` in it; with `Self` denied the Leaker rejects it, as the const would not
        // be a type; a bare identifier that is a type is leaked alone where the path cannot be
        (
            9081,
            reported(9081, __tl_report_9081!()),
            ok("struct", 2, 2, 2),
        ),
        (
            9082,
            reported(9082, __tl_report_9082!()),
            rejected(
                "struct",
                "`LEN`maybeaconst,andthetypethatholdsitcannotbeleakedasawhole",
            ),
        ),
        (
            9083,
            reported(9083, __tl_report_9083!()),
            ok("trait", 3, 3, 3),
        ),
        (
            9084,
            reported(9084, __tl_report_9084!()),
            ok("trait", 1, 5, 1),
        ),
        // items nested in a module, which name their own `Self`
        (
            9061,
            reported(9061, __tl_report_9061!()),
            ok("struct", 5, 5, 5),
        ),
        (
            9062,
            reported(9062, __tl_report_9062!()),
            ok("enum", 4, 4, 4),
        ),
        (
            9063,
            reported(9063, __tl_report_9063!()),
            ok("trait", 5, 5, 5),
        ),
        // structs, an enum and a union nested in a module, with a bound that names their own
        // `Self` (`T: AsRef<Self>`), also in the where clause: the entry that does not name `Self`
        // is for any type, and the copy passes its own; one that names it too is rejected
        (
            9111,
            reported(9111, __tl_report_9111!()),
            ok("struct", 3, 3, 3),
        ),
        (
            9112,
            reported(9112, __tl_report_9112!()),
            ok("struct", 1, 2, 2),
        ),
        (
            9113,
            reported(9113, __tl_report_9113!()),
            ok("enum", 2, 1, 1),
        ),
        (
            9114,
            reported(9114, __tl_report_9114!()),
            ok("union", 2, 2, 2),
        ),
        (
            9115,
            reported(9115, __tl_report_9115!()),
            rejected(
                "struct",
                "`Wrap<Self>`hasagenericwhoseboundmentions`Self`,whichcannotbeleaked",
            ),
        ),
        // the entry is in a bound of another generic, in a where predicate, keeps a lifetime, and
        // is for a `Self` that is outlived or bounded by `Sized`
        (
            9116,
            reported(9116, __tl_report_9116!()),
            ok("struct", 1, 2, 2),
        ),
        (
            9117,
            reported(9117, __tl_report_9117!()),
            ok("struct", 1, 1, 1),
        ),
        (
            9118,
            reported(9118, __tl_report_9118!()),
            ok("struct", 1, 2, 2),
        ),
        (
            9119,
            reported(9119, __tl_report_9119!()),
            ok("struct", 1, 2, 2),
        ),
        (
            9120,
            reported(9120, __tl_report_9120!()),
            ok("struct", 1, 2, 2),
        ),
        // the entry for any type is the last field of a struct or a field of an enum, with a bound
        // that needs no `Sized` (9125 to 9127) or that does (9128): the Repeater's impl is
        // `?Sized` in the first case only
        (
            9125,
            reported(9125, __tl_report_9125!()),
            ok("struct", 2, 2, 2),
        ),
        (
            9126,
            reported(9126, __tl_report_9126!()),
            ok("enum", 1, 2, 2),
        ),
        (
            9127,
            reported(9127, __tl_report_9127!()),
            ok("struct", 1, 2, 2),
        ),
        (
            9128,
            reported(9128, __tl_report_9128!()),
            ok("struct", 1, 2, 2),
        ),
        // an absolute type that names their own `Self` is no entry (the copy has it as written);
        // the relative types beside it are
        (
            9121,
            reported(9121, __tl_report_9121!()),
            ok("struct", 2, 3, 3),
        ),
        (
            9122,
            reported(9122, __tl_report_9122!()),
            ok("struct", 1, 2, 2),
        ),
        (
            9123,
            reported(9123, __tl_report_9123!()),
            ok("struct", 2, 3, 3),
        ),
        (
            9124,
            reported(9124, __tl_report_9124!()),
            ok("enum", 2, 3, 3),
        ),
        // the defaults of a nested trait whose generic has a bound that names its `Self`
        (
            9131,
            reported(9131, __tl_report_9131!()),
            ok("trait", 1, 3, 3),
        ),
        (
            9132,
            reported(9132, __tl_report_9132!()),
            ok("trait", 2, 5, 5),
        ),
        // the types in the values of an item (`check_values()`): the consts and the default
        // methods of a trait, and the const, the static and the function of a module
        (
            9141,
            reported(9141, __tl_report_9141!()),
            ok("trait", 5, 6, 6),
        ),
        (
            9142,
            reported(9142, __tl_report_9142!()),
            ok("mod", 2, 3, 3),
        ),
        // scopes that share names (the findings of a downstream crate): methods that declare the
        // same generic with different bounds, a generic named like a relative type, a path ending
        // with the name of a generic, a type and a const parameter of one name; a bound that names
        // a parameter or a lifetime the entry does not use; a where clause; a cyclic bound; a
        // `for<'b>` binder
        (
            9071,
            reported(9071, __tl_report_9071!()),
            ok("trait", 7, 13, 13),
        ),
        (
            9072,
            reported(9072, __tl_report_9072!()),
            ok("struct", 1, 2, 2),
        ),
        (
            9073,
            reported(9073, __tl_report_9073!()),
            ok("trait", 1, 2, 2),
        ),
        (
            9074,
            reported(9074, __tl_report_9074!()),
            ok("struct", 1, 2, 2),
        ),
        (
            9075,
            reported(9075, __tl_report_9075!()),
            ok("struct", 1, 1, 1),
        ),
        (
            9076,
            reported(9076, __tl_report_9076!()),
            rejected(
                "struct",
                "`Wrap<T>`isinaboundorwherepredicate,andaboundofitsgenericsnamesaleakedtype",
            ),
        ),
        (
            9077,
            reported(9077, __tl_report_9077!()),
            ok("struct", 1, 1, 1),
        ),
        // paths that end with the name of a generic (`self::U`, `crate::U`), in a struct and in a
        // trait, and a module that declares `X` next to a struct whose generic is `X`
        (
            9091,
            reported(9091, __tl_report_9091!()),
            ok("struct", 4, 5, 5),
        ),
        (
            9092,
            reported(9092, __tl_report_9092!()),
            ok("trait", 5, 8, 8),
        ),
        (
            9093,
            reported(9093, __tl_report_9093!()),
            ok("mod", 3, 5, 5),
        ),
    ];
    for (id, got, expected) in table {
        assert_eq!(got, expected, "item {}", id);
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

    // A module: what it declares (`Local`) is its own, what it uses from outside is the original.
    let holds = r::nested::Holds::<u8> {
        rel,
        wrapped: l::Wrap(PhantomData),
        pair: l::Pair(rel, 6),
        local: r::nested::Local(11),
    };
    assert_eq!((holds.pair.1, holds.local.0), (6, 11));
    same(&holds.rel, &rel);
    assert_eq!(r::nested::deeper::Deep(rel).0, rel);
}

/// A struct, an enum and a trait nested in a module name their own `Self` in relative types
/// (`Wrap<Self>`), and the copy holds the Leaker's: `Visitor::with_nested_self()` writes
/// `<::type_leak_test_leaker::kinds::Links<T> as Repeater<..>>::Type` for the `Self` of the struct,
/// which the copy's own `Links` could not stand for (the Repeater's impl is for the Leaker's one).
/// The Referrer's crate compares each position with a witness of the original when it compiles; here
/// the fields go in and out of the Leaker's types, which `same()` and the assignments check.
#[test]
fn items_nested_in_a_module_name_their_own_self() {
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
/// for itself, so `Visitor::with_nested_self()` lets the macro write `<Self as Repeater<..>>::Type`:
/// the Repeater's impl of the entry is for any type that the bound holds for. The field is the
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
    // has no `AsRef` for. The pair holds the copy's type for `Self` (strict), and so does the
    // one of a Referrer that trusts the names of the crates.
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
    struct Owner2;
    impl AsRef<r::PathedTrusting<Owner2>> for Owner2 {
        fn as_ref(&self) -> &r::PathedTrusting<Owner2> {
            unimplemented!()
        }
    }
    let trusting = r::PathedTrusting::<Owner2> {
        path: l::Pair(PhantomData::<r::PathedTrusting<Owner2>>, Owner2),
        wrapped: l::Wrap(PhantomData),
        rel,
    };
    same(&trusting.path.0, &PhantomData::<r::PathedTrusting<Owner2>>);

    // The enum (strict): the vector holds boxes of the copy's `Tree`.
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
/// for the implementors of the trait, and the Visitor left the default as written (`Defd<T>`, a
/// relative path that the Referrer's crate cannot resolve). A default exists only in a declaration,
/// so the Referrer's crate declares the copy (`SelfDefaultCopy`), with the default written through
/// the host, a type that implements the Leaker's trait: `<Host as Repeater<(T, ()), ID, 0>>::Type`.
/// The default cannot pass the defaulted parameter (`(T, U)`) to the Repeater, which names itself
/// (E0128), so it passes `()`: the host implements the Leaker's trait for any `U`.
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

/// The types in the values of an item, with `LeakerConfig::check_values()` (F40 of
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

/// The same for the const, the static and the function of a module. The module declares `Declared`,
/// which the copy declares again (`size_of::<Declared>()` is its own); the type that it does not
/// declare is replaced.
#[test]
fn the_types_in_the_values_of_a_module_are_carried() {
    use type_leak_test_leaker::kinds as l;
    use type_leak_test_referrer::kinds as r;

    assert_eq!(l::values_in_module::SIZE, 1 + 7);
    assert_eq!(r::values_in_module::SIZE, l::values_in_module::SIZE);
    assert_eq!(l::values_in_module::TAG, 2);
    assert_eq!(r::values_in_module::TAG, l::values_in_module::TAG);
    assert_eq!(l::values_in_module::total(), 7 + 1 + 7);
    assert_eq!(r::values_in_module::total(), l::values_in_module::total());
}

/// The scopes of an item that share names (the findings A, 1, 2, 3, 4 and 7 of a downstream crate,
/// `tests/downstream_findings.rs`): the Repeater's impl of each entry has the generics of its own
/// scope, so the Referrer's crate compiles, and the signatures of the copy have the original types
/// (a function pointer of each method, with the original types, is the method of the copy).
#[test]
fn scopes_that_share_names_keep_their_own_generics() {
    use type_leak_test_leaker::kinds as l;
    use type_leak_test_referrer::kinds as r;

    // 9071: `with_bound<X: Clone>` and `without_bound<X>` share the name `X`; `generic_local<Local>`
    // has a generic named like the type `Local` that `type_local` returns; `segment<U>` returns
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

    // 9072: the bound of `T` names `V`.
    let copy = r::ForeignBound::<u8, u32>(l::Wrap(PhantomData), 3);
    same(&copy.0, &l::Wrap::<u8>(PhantomData));
    assert_eq!(copy.1, 3);
    // 9073: the bound of a method generic names the parameter of the trait.
    fn convert_of<T>() -> fn(&r::Host9073, l::Wrap<u8>) -> l::Wrap<u8>
    where
        u8: Into<T>,
    {
        <r::Host9073 as l::ForeignBoundTrait<T>>::convert::<u8>
    }
    let _ = convert_of::<u32>();
    // 9074: `T: 'a`.
    let n = 5u8;
    let copy = r::OutlivesBound::<u8>(l::Wrap(PhantomData), &n);
    same(&copy.0, &l::Wrap::<u8>(PhantomData));
    assert_eq!(*copy.1, 5);
    // 9075: the where clause of the item.
    let copy = r::WhereBound::<u8>(l::Cloned(1));
    same(&copy.0, &l::Cloned(1u8));
    // 9077: a binder lifetime that only the entry names.
    let seen = std::cell::Cell::new(0u8);
    let copy = r::Binder(|lt: l::Lt<'_, u8>| seen.set(*lt.0));
    (copy.0)(l::Lt(&n));
    assert_eq!(seen.get(), 5);
}

/// Names that a generic shares with a path or with a declared item (`tests/shadowed_names.rs`): the
/// Referrer's crate compiles only if the visitor replaced `self::U` and `crate::U` next to a generic
/// `U` (9091, 9092; neither type exists there), and `crate::base::B1<X>` of a generic `X` in a
/// module that declares an `X` (9093; the Referrer's crate has no `base`).
#[test]
fn names_shared_with_a_generic_keep_the_original_types() {
    use type_leak_test_leaker as l;
    use type_leak_test_referrer::kinds as r;

    // 9091
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

    // 9092: a host implements the trait with the signatures of the copy.
    type H = r::Host9092;
    fn implemented_for<T: l::kinds::PathsLikeGenericTrait<u8>>() {}
    implemented_for::<H>();
    let _: fn(&H, u16, l::kinds::U) -> l::U =
        <H as l::kinds::PathsLikeGenericTrait<u8>>::here::<u16>;
    let _: fn(&H, l::kinds::Wrap<u16>) -> l::kinds::Wrap<l::kinds::U> =
        <H as l::kinds::PathsLikeGenericTrait<u8>>::wrapped::<u16>;
    let _: fn(&H, u8, l::kinds::U) -> l::kinds::Pair<l::U, u8> =
        <H as l::kinds::PathsLikeGenericTrait<u8>>::trait_generic;

    // 9093: the generic `X` of the copy is the parameter of the Leaker's `B1<X>`; the declared `X`
    // is the Leaker's.
    use r::declared_like_generic as m;
    let generic = m::Generic::<u8>(l::base::B1(PhantomData), l::base::B2(1u8, 2u8));
    same(&generic.0, &l::base::B1::<u8>(PhantomData));
    assert_eq!((generic.1 .0, generic.1 .1), (1, 2));
    let declared = m::Declared(l::base::B1(PhantomData));
    same(
        &declared.0,
        &l::base::B1::<l::kinds::declared_like_generic::X>(PhantomData),
    );
    let deeper = m::inner::Deeper::<u16>(l::base::B1(PhantomData));
    same(&deeper.0, &l::base::B1::<u16>(PhantomData));
    assert_eq!(m::X(7).0, 7);
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
// A strict Referrer, in a crate without the dependency of an optional entry.
// ---------------------------------------------------------------------------------------------

/// The type of a crate which the Referrer's crate has no dependency on is an optional entry. The
/// trusting Referrer leaves it as written (and replaces what is relative in it), and the strict one
/// replaces the whole type: the text of the two copies of the same items differs in that. Only the
/// strict one can be compiled where the crate is not a dependency (this crate has none on it):
/// `tests/trusting.rs` builds both in a crate that has none, and runs the tests of the strict one.
#[test]
fn a_strict_referrer_replaces_the_optional_entries_and_a_trusting_one_does_not() {
    use type_leak_test_referrer::{STRICT_COPY, TRUSTING_COPY};
    let (trusting, strict) = (squash(TRUSTING_COPY), squash(STRICT_COPY));

    // Trusting: the type of the other crate stays, with the relative type in it (`RelX`, the
    // required entry 0) replaced; the plain `Ext<u8>` is not touched.
    assert!(trusting.contains("x:::type_leak_test_ext::Ext<(<Wrapperas$crate::Repeater<(),"));
    assert!(trusting.contains("0usize>>::Type,)>"));
    assert!(trusting.contains("y:::type_leak_test_ext::Ext<u8>"));
    assert_eq!(trusting.matches("type_leak_test_ext").count(), 2);
    assert_eq!(trusting.matches("Repeater<").count(), 1);

    // Strict: both types of the other crate are replaced as a whole, each by its own entry (1 and
    // 2, after the required `RelX`), and the crate is not named at all.
    assert!(!strict.contains("type_leak_test_ext"));
    assert!(strict.contains("x:<Wrapperas$crate::Repeater<(),"));
    assert!(strict.contains("1usize>>::Type,y:<Wrapperas$crate::Repeater<(),"));
    assert!(strict.ends_with("2usize>>::Type){}"));
    assert_eq!(strict.matches("Repeater<").count(), 2);
    assert_ne!(trusting, strict);
}

// ---------------------------------------------------------------------------------------------
// Fragments of impls, analyzed with the scope API.
// ---------------------------------------------------------------------------------------------

/// `coinduction`'s shape: `enter_impl(.., false, ..)` with `visit_path_arguments()`, the
/// Repeater's impls for a marker (`ImplSelf::Marker`) of the required entries only
/// (`has_required()`, `required()`), and the Referrer's copy of the self type, the trait and its
/// arguments with `Referrer::visitor(context, ..)` and `entry_args_with(ix, &UseSite::identity(),
/// ..)`. The Referrer's crate checks the copy: each type is the original (a witness), and rustc finds
/// the Leaker's impl of the trait for the rewritten self type.
#[test]
fn the_scope_api_with_a_marker() {
    // Three required entries: `Rel3` (no generics), `Rel<T>` and `&'a Rel2` (the lifetime encoded
    // in the first argument of the Repeater).
    assert_eq!(
        type_leak_test_leaker::__scope_report_7001!(),
        "7001|entries=3 required=3 impls=3 types=4"
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
            "::type_leak_test_leaker::scoped::Holder<T,{}(),7001u64,0usize>>::Type>",
            marker
        )
    );
    assert_eq!(
        trait_path,
        format!(
            "::type_leak_test_leaker::scoped::Tr<{m}(T,),7001u64,1usize>>::Type,{m}(&'a(),),7001u64,2usize>>::Type>",
            m = marker
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
            "implcrate::Repeater<(),7001u64,0usize>forM7001",
            "impl<T:::core::clone::Clone>crate::Repeater<(T,),7001u64,1usize>forM7001",
            "impl<'a>crate::Repeater<(&'a(),),7001u64,2usize>forM7001",
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
            format!("{}(),7001u64,0usize>>::Type", marker),
            format!("{}(T,),7001u64,1usize>>::Type", marker),
            format!("{}(&'a(),),7001u64,2usize>>::Type", marker),
        ]
    );
}

/// `newer-type`'s shape: `Self` is allowed, and the Repeater's impl of an entry that names it is for
/// the self type (`ImplSelf::Param`), whereas the one that does not is for the marker.
#[test]
fn the_scope_api_with_self() {
    assert_eq!(
        type_leak_test_leaker::__scope_report_7002!(),
        "7002|entries=2 required=2 impls=2 types=4"
    );
    let trait_path = squash(type_leak_test_referrer::scoped::SCOPE_7002_TRAIT);
    // `Wrap<Self>` is in the impl of the self type itself, `Rel<T>` in the marker's.
    assert_eq!(
        trait_path,
        concat!(
            "::type_leak_test_leaker::scoped::Tr<",
            "<::type_leak_test_leaker::scoped::Holder<T,u8>as::type_leak_test_leaker::Repeater<(T,),7002u64,0usize>>::Type,",
            "<::type_leak_test_leaker::scoped::M7002as::type_leak_test_leaker::Repeater<(T,),7002u64,1usize>>::Type>"
        )
    );
    assert_eq!(
        type_leak_test_leaker::__scope_impls_7002!()
            .split(";;")
            .map(squash)
            .collect::<Vec<_>>(),
        [
            "impl<T:::core::clone::Clone>crate::Repeater<(T,),7002u64,0usize>forcrate::scoped::Holder<T,u8>",
            "impl<T:::core::clone::Clone>crate::Repeater<(T,),7002u64,1usize>forM7002",
        ]
    );
    // The self type has nothing to replace.
    assert_eq!(
        squash(type_leak_test_referrer::scoped::SCOPE_7002_SELF),
        "::type_leak_test_leaker::scoped::Holder<T,u8>"
    );
}

/// Entries which need no generics are plain: the impl for the marker declares nothing. With an
/// optional entry only, `has_required()` is `false` and the macro writes no impl; with the impl of
/// the optional entry too (`all`), the strict Referrer replaces it where its crate has no dependency
/// on the crate of the type.
#[test]
fn the_scope_api_writes_the_impls_that_are_needed() {
    assert_eq!(
        type_leak_test_leaker::__scope_report_7003!(),
        "7003|entries=2 required=2 impls=2 types=4"
    );
    assert_eq!(
        type_leak_test_leaker::__scope_report_7004!(),
        "7004|entries=1 required=0 impls=0 types=4"
    );
    assert_eq!(
        type_leak_test_leaker::__scope_report_7005!(),
        "7005|entries=2 required=1 impls=2 types=4"
    );
    // A where clause on a type that is not a parameter (`Vec<T>: Clone`) is in the context of the
    // entry that needs it (`Cl<T>` is well-formed only with it), so the impl of the Repeater has it.
    assert_eq!(
        type_leak_test_leaker::__scope_report_7006!(),
        "7006|entries=1 required=1 impls=1 types=4"
    );
    assert_eq!(
        squash(type_leak_test_leaker::__scope_impls_7006!()),
        "impl<T>crate::Repeater<(T,),7006u64,0usize>forM7006where::std::vec::Vec<T>:::core::clone::Clone"
    );
    // Without `all`, a required entry next to an optional one gets the impl of the required one only.
    assert_eq!(
        type_leak_test_leaker::__scope_report_7007!(),
        "7007|entries=2 required=1 impls=1 types=4"
    );
    // (`tests/trusting.rs` applies 7005 as a strict Referrer, in a crate that has no dependency on
    // the crate of the optional entry.)
}

/// `NotInternableError::items()` through a macro: one item for each offending path, in order, with
/// its message, the tokens, the source text at its span and the notes with theirs. The macro
/// reported them as a const, and as many `compile_error!`s.
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

    // `Self` is denied for the impl.
    let denied = rows(SCOPE_ERRORS_SELF_DENIED);
    assert_eq!(denied.len(), 1);
    assert_eq!(denied[0].0, "useabsolutepath:'Wrap<Self>'");
    assert_eq!(denied[0].1, "Wrap<Self>");
    assert!(denied[0].2.starts_with("Wrap"), "{:?}", denied[0].2);
    assert_eq!(denied[0].3, vec![(because("Self"), "Self".to_string())]);
    assert_eq!(SCOPE_ERRORS_SELF_DENIED_COMPILE_ERRORS, 1);

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

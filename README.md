# type-leak crate [![Latest Version]][crates.io] [![Documentation]][docs.rs] [![GitHub Actions]][actions]

[Latest Version]: https://img.shields.io/crates/v/type-leak.svg
[crates.io]: https://crates.io/crates/type-leak
[Documentation]: https://img.shields.io/docsrs/type-leak
[docs.rs]: https://docs.rs/type-leak/latest/type_leak/
[GitHub Actions]: https://github.com/yasuo-ozu/type-leak/actions/workflows/rust.yml/badge.svg
[actions]: https://github.com/yasuo-ozu/type-leak/actions/workflows/rust.yml

<p align="center"><img src="https://raw.githubusercontent.com/yasuo-ozu/type-leak/main/logo.png" alt="type-leak" width="240"></p>

A path in Rust means something only where it is written. Say a proc macro copies
`fn get(&self) -> MyType` from crate A into crate B. In crate B, the copy looks up `MyType` again.
It finds nothing, or another type.

`type-leak` solves this for proc-macro authors. In crate A, it finds the types that depend on where
they are written. Your macro writes an impl of one shared trait, the Repeater, for each of them. In
crate B, `type-leak` rewrites the copy to reach each type through that trait:
`<Self as Repeater<..>>::Type`.

# How it works

```text
 Crate A (defines the item)                   Crate B (copies the item)
 trait Tr { fn f(&self) -> MyType; }

 1. Analyze the item:                         3. Rewrite the copy:
    Leaker --> Referrer ------ as tokens --->    fn f(&self) -> MyType
                                                 becomes
 2. Write the Repeater impls:                    fn f(&self) -> <Self as Repeater<..>>::Type
    impl Repeater<..> for .. {                                         |
        type Type = MyType;  <-----------------------------------------+
    }
```

- **Crate A** defines the item, and **crate B** copies it, each with a proc macro. The **copy** is
  what crate B's macro writes.
- An **entry** is a type of the item that depends on where it is written (`MyType`, `MyWrap<T>`,
  `Wrap<Self>`). `type-leak` *leaks* it: crate B reaches it through the Repeater. By default every
  relative path is an entry, even `u8`, and so is a type of another crate (`::serde::Value`). Give
  the Leaker [`Rules`] to have the copy write such paths as they are, if they mean the same in
  crate B.
- The **Leaker** analyzes the item in crate A. Its result holds the **Referrer**, the list of the
  entries. Crate A's macro writes one Repeater impl for each entry, and passes the Referrer to
  crate B as tokens. There, the Referrer rewrites the copy.
- The **Repeater** is the one trait that both crates see. Define it once, at the root of crate A
  ([`repeater_definition()`] writes it). It cannot be in the proc-macro crate: its impls are for
  any type, and only a local trait allows that.

  ```rust,ignore
  pub trait Repeater<T: ?Sized, const KEY: u128, /* const slots */> {
      type Type: ?Sized;
  }
  ```

  `T` encodes the generics that the entry uses. `KEY` holds the Leaker's id and the [`EntryId`] of
  the entry, so each entry has its own impl.
- A [`Projection`] says how to write the Repeater: its path, the Leaker's id, and what the impls
  are for. Both crates need the same id and `impl_self`.

# Quick start

```rust
use syn::visit_mut::VisitMut;
use syn::{parse_quote, ItemImpl, ItemTrait};
use type_leak::{ImplSelf, Leaker, Projection};

let item: ItemTrait = parse_quote! {
    pub trait MyTrait<T> {
        fn f(&self, t: T) -> MyWrap<T>;
        fn g(&self) -> MyType;
    }
};

// Crate A: analyze the trait. Keep its `ContextId`: crate B needs it.
let mut leaker = Leaker::new();
let top_level = leaker.intern_trait(&item);
let result = leaker.result();

// Write one Repeater impl for each entry. The Repeater is `path::Repeater`, `42` is the id of
// this Leaker (each Leaker of crate A needs its own), and the impls are for any type `__S`.
let projection = Projection::new(
    parse_quote!(path::Repeater),
    42,
    ImplSelf::Param(parse_quote!(__S)),
);
let impls: Vec<ItemImpl> = result
    .repeater_impls(&projection)
    .unwrap()
    .into_iter()
    .map(|r| parse_quote!(#r))
    .collect();
// The entries are `MyWrap<T>` (0) and `MyType` (1). The `KEY` tells them apart.
let (wrap, my_type) = (projection.key(parse_quote!(0)), projection.key(parse_quote!(1)));
assert_eq!(impls.len(), 2);
assert_eq!(impls[1], parse_quote! {
    impl<__S: ?::core::marker::Sized> path::Repeater<(), #my_type> for __S {
        type Type = MyType;
    }
});
let referrer = result.finish().unwrap(); // `Ok`: there are no errors.

// Crate B: copy the methods into `impl<U> MyTrait<U> for Foo`.
let mut imp: ItemImpl = parse_quote! {
    impl<U> MyTrait<U> for Foo {
        fn f(&self, t: T) -> MyWrap<T> { todo!() }
        fn g(&self) -> MyType { todo!() }
    }
};
// Replace each entry with its path through the Repeater.
referrer.visitor(top_level, &projection).visit_item_impl_mut(&mut imp);
// Write the arguments of the impl into its items: the trait's `T` is `U` here.
let mut substitute = referrer
    .complete_args(top_level, &[parse_quote!(U)], &projection)
    .unwrap();
substitute.visit_item_impl_mut(&mut imp);

assert_eq!(
    imp,
    parse_quote! {
        impl<U> MyTrait<U> for Foo {
            fn f(&self, t: U) -> <Self as path::Repeater<(U,), #wrap>>::Type { todo!() }
            fn g(&self) -> <Self as path::Repeater<(), #my_type>>::Type { todo!() }
        }
    }
);
```

The visitor replaces only the entries. [`Referrer::complete_args()`] pairs the parameters of the
trait with the arguments of the impl, and uses the defaults for missing arguments. The
[`Substitute`] that it returns writes the arguments into the items (`T` and `(T,)` become `U` and
`(U,)`). Skip this step if the copy declares the generics of the trait itself, with the same names
and bounds (`impl<T> MyTrait<T> for Foo`).

# Errors and recovery

Some parts cannot be leaked. `MyType<'_>` is one: its path depends on where it is written, and its
elided lifetime cannot be leaked. The Leaker records an error for it and goes on, and the copy
writes that part as written. Each error has a help that says what to write instead.

[`finish()`] returns the Referrer only when no error remains. So your macro either:

- reports the errors, with `finish()?` (one `syn::Error` for all of them) or with
  [`ErrorItem::emit()`] for each error; or
- accepts them first, with [`allow()`], [`ignore()`] or [`ignore_if()`]. You promise that the part
  compiles as written in crate B.

See [errors and recovery] and the [limitations] in the docs of the `Leaker`.

# In real macros

Crate A's macro cannot see the input of crate B's macro. So it usually exports a `macro_rules!`
for crate B to call. That macro adds the Referrer, the `ContextId`, the Leaker's id and
`$crate::Repeater` to the input (as tokens: `ToTokens` and `Parse`), and calls crate B's proc
macro. Both macros must use the same version of `type-leak`. [`examples/two-macros`] is a
complete, tested setup.

# Learn more

- [`Leaker`]: the `intern_*` methods analyze a whole item ([`intern_trait()`]), and the `enter_*`
  methods parts of one ([`enter_impl()`]). [`Leaker::with_rule()`] takes the [`Rules`].
- [`LeakerResult`]: [`repeater_impls()`] in crate A, and [`finish()`] for the Referrer.
- [`Referrer`]: its [`Visitor`] rewrites the copy. [`complete_args()`], [`Substitute`] and
  [`substitute_args()`] write the arguments of an impl.
- [`Projection`] and [`repeater_definition()`]: the Repeater.
- [`ErrorItem`] and [`LeakerResultError`]: the errors.
- [`examples/two-macros`]: a complete, tested pair of proc macros.

[`Leaker`]: https://docs.rs/type-leak/latest/type_leak/struct.Leaker.html
[`intern_trait()`]: https://docs.rs/type-leak/latest/type_leak/struct.Leaker.html#method.intern_trait
[`enter_impl()`]: https://docs.rs/type-leak/latest/type_leak/struct.Leaker.html#method.enter_impl
[`Leaker::with_rule()`]: https://docs.rs/type-leak/latest/type_leak/struct.Leaker.html#method.with_rule
[errors and recovery]: https://docs.rs/type-leak/latest/type_leak/struct.Leaker.html#errors-and-recovery
[limitations]: https://docs.rs/type-leak/latest/type_leak/struct.Leaker.html#limitations
[`Rules`]: https://docs.rs/type-leak/latest/type_leak/struct.Rules.html
[`LeakerResult`]: https://docs.rs/type-leak/latest/type_leak/struct.LeakerResult.html
[`repeater_impls()`]: https://docs.rs/type-leak/latest/type_leak/struct.LeakerResult.html#method.repeater_impls
[`finish()`]: https://docs.rs/type-leak/latest/type_leak/struct.LeakerResult.html#method.finish
[`allow()`]: https://docs.rs/type-leak/latest/type_leak/struct.LeakerResult.html#method.allow
[`ignore()`]: https://docs.rs/type-leak/latest/type_leak/struct.LeakerResult.html#method.ignore
[`ignore_if()`]: https://docs.rs/type-leak/latest/type_leak/struct.LeakerResult.html#method.ignore_if
[`LeakerResultError`]: https://docs.rs/type-leak/latest/type_leak/struct.LeakerResultError.html
[`ErrorItem`]: https://docs.rs/type-leak/latest/type_leak/struct.ErrorItem.html
[`ErrorItem::emit()`]: https://docs.rs/type-leak/latest/type_leak/struct.ErrorItem.html#method.emit
[`Referrer`]: https://docs.rs/type-leak/latest/type_leak/struct.Referrer.html
[`complete_args()`]: https://docs.rs/type-leak/latest/type_leak/struct.Referrer.html#method.complete_args
[`Referrer::complete_args()`]: https://docs.rs/type-leak/latest/type_leak/struct.Referrer.html#method.complete_args
[`Visitor`]: https://docs.rs/type-leak/latest/type_leak/struct.Visitor.html
[`Substitute`]: https://docs.rs/type-leak/latest/type_leak/struct.Substitute.html
[`substitute_args()`]: https://docs.rs/type-leak/latest/type_leak/fn.substitute_args.html
[`EntryId`]: https://docs.rs/type-leak/latest/type_leak/struct.EntryId.html
[`Projection`]: https://docs.rs/type-leak/latest/type_leak/struct.Projection.html
[`repeater_definition()`]: https://docs.rs/type-leak/latest/type_leak/fn.repeater_definition.html
[`examples/two-macros`]: https://github.com/yasuo-ozu/type-leak/tree/main/examples/two-macros

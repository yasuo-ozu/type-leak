# type-leak crate [![Latest Version]][crates.io] [![Documentation]][docs.rs] [![GitHub Actions]][actions]

[Latest Version]: https://img.shields.io/crates/v/type-leak.svg
[crates.io]: https://crates.io/crates/type-leak
[Documentation]: https://img.shields.io/docsrs/type-leak
[docs.rs]: https://docs.rs/type-leak/latest/type_leak/
[GitHub Actions]: https://github.com/yasuo-ozu/type-leak/actions/workflows/rust.yml/badge.svg
[actions]: https://github.com/yasuo-ozu/type-leak/actions/workflows/rust.yml

<p align="center"><img src="https://raw.githubusercontent.com/yasuo-ozu/type-leak/main/logo.png" alt="type-leak" width="240"></p>

A name in Rust means something only where it is written. When a proc macro copies `fn get(&self) -> MyType` from crate A into code that crate B expands, the copy looks up `MyType` in crate B and finds nothing, or the wrong type.

`type-leak` fixes this for proc-macro authors. In crate A, it finds the types that depend on where they are written, and describes for each a small impl of a shared trait, the *Repeater*, which your macro emits. In crate B, it rewrites the copy so that each such type is written `<Self as Repeater<..>>::Type`.

# How it works

```text
 Crate A (defines the item)                   Crate B (copies the item)
 trait Tr { fn f(&self) -> MyType; }

 1. Analyze the item:                         3. Rewrite the copy:
    Leaker --> Referrer ------ as tokens --->    fn f(&self) -> MyType
                                                 becomes
 2. Emit the Repeater impls:                     fn f(&self) -> <Self as Repeater<..>>::Type
    impl Repeater<..> for .. {                                         |
        type Type = MyType;  <-----------------------------------------+
    }
```

- The **Leaker** analyzes the item. Its result, the **Referrer**, lists the *entries*: the types that depend on where they are written (`MyType`, `Wrap<Self>`, `MyWrap<T>`). It travels to crate B as tokens.
- The **Repeater** is a trait defined once at the root of crate A, not in the proc-macro crate (`repeater_definition()` generates it):

  ```rust,ignore
  pub trait Repeater<T: ?Sized, const KEY: u128, /* one const slot per primitive const type except usize and u128 */> {
      type Type: ?Sized;
  }
  ```

  `KEY` = `Projection::key(ENTRY)`: its high half is the Leaker's id (the `id` of the `Projection` that both crates build, which `repeater_impls` and `Projection::of` take), its low half is the `EntryId` of the entry, and `T` encodes the generics the entry uses.

# Quick start

```rust
use syn::visit_mut::VisitMut;
use syn::{parse_quote, ItemImpl, ItemTrait, Path};
use type_leak::{ImplSelf, Leaker, Projection};

let item: ItemTrait = parse_quote! {
    pub trait MyLeaker<T> {
        fn f(&self, t: T) -> MyWrap<T>;
        fn g(&self) -> MyType;
    }
};

// Crate A: analyze the trait, and write one Repeater impl per entry (`42` is the ID).
let mut leaker = Leaker::new();
let top_level = leaker.intern_trait(&item);
let result = leaker.result();
let projection = Projection::new(parse_quote!(path::Repeater), 42, ImplSelf::Param(parse_quote!(__S)));
let impls: Vec<ItemImpl> = result
    .repeater_impls(&projection)
    .unwrap()
    .into_iter()
    .map(|r| parse_quote!(#r))
    .collect();
assert_eq!(impls.len(), 2);
let referrer = result.finish().unwrap();

// Crate B: copy the methods of the trait into `impl<U> MyLeaker<U> for Foo`. The visitor writes
// each entry as `<Self as Repeater<..>>::Type`, by the names of its own generics.
let mut imp: ItemImpl = parse_quote! {
    impl<U> MyLeaker<U> for Foo {
        fn f(&self, t: T) -> MyWrap<T> { todo!() }
        fn g(&self) -> MyType { todo!() }
    }
};
referrer.visitor(top_level, &projection).visit_item_impl_mut(&mut imp);

// Write the arguments of the impl into its items, not into its header, which declares the
// generics itself: the trait's `T` is `U`.
let mut substitute = referrer
    .complete_args(top_level, &[parse_quote!(U)], &projection)
    .unwrap();
substitute.visit_item_impl_mut(&mut imp);

let (wrap, my_type) = (projection.key(parse_quote!(0)), projection.key(parse_quote!(1)));
assert_eq!(
    imp,
    parse_quote! {
        impl<U> MyLeaker<U> for Foo {
            fn f(&self, t: U) -> <Self as path::Repeater<(U,), #wrap>>::Type { todo!() }
            fn g(&self) -> <Self as path::Repeater<(), #my_type>>::Type { todo!() }
        }
    }
);
```

The visitor rewrites only the entries. The rest of the copy is yours to write: `Referrer::complete_args` takes the arguments of the impl, completes the omitted ones with the defaults of the parameters of the trait (a leaked type in a default is written through the Repeater too), and returns a `Substitute`. That visitor writes the arguments into the types and the lifetimes of the items you give it (`&'a self`, `T: 'a`, `where Self: 'a`), also into the arguments of the paths through the Repeater (`(T,)` is `(U,)`). Give it the items or the whole impl: it leaves the header, which declares the generics itself, and the items nested in a body. `substitute_args` does the same for single arguments, with a closure that answers each name (`Self` too). An associated type of a type that is not a type parameter (`T::Item` for `T` as `IntoIter<u8>`) needs its trait: `complete_args` and `Substitute` write `<IntoIter<u8> as Iterator>::Item` from the bound of the parameter, inline or in the where clause, and `substitute_args` refuses it (`EncodeError::AmbiguousAssociatedType`) unless the copy has `<T as Iterator>::Item`.

If the item has a part that cannot be leaked (`MyType<'_>`), the `Leaker` copies it as written and records an error: `finish()` is then `Err` until every error is accepted with `allow()` (the part is copied as written) or `ignore()`. The `Err` is a `LeakerResultError`: it reads as a slice of `ErrorItem`s, writes a `compile_error!` for each with `ToTokens`, and becomes one `syn::Error` with `?`. Report the errors one by one with `ErrorItem::emit()` (on nightly, a diagnostic with the notes and the help at their own spans), or with `ErrorItem::to_compile_error()`.

In real macros, crate A passes `referrer` and `top_level` to crate B as tokens (`ToTokens` / `Parse`), usually through an exported `macro_rules!`. [`examples/two-macros`](https://github.com/yasuo-ozu/type-leak/tree/main/examples/two-macros) is a complete, tested setup.

# Learn more

- [API docs](https://docs.rs/type-leak/latest/type_leak/). Start with:
  - [`Leaker`](https://docs.rs/type-leak/latest/type_leak/struct.Leaker.html), which analyzes an item: [`intern_trait`](https://docs.rs/type-leak/latest/type_leak/struct.Leaker.html#method.intern_trait) and the other `intern_*` methods for a whole item, [`enter_impl`](https://docs.rs/type-leak/latest/type_leak/struct.Leaker.html#method.enter_impl) and the other `enter_*` methods for parts of one.
  - [`LeakerResult`](https://docs.rs/type-leak/latest/type_leak/struct.LeakerResult.html), what `Leaker::result()` returns: the `Referrer` and the errors, and [`repeater_impls`](https://docs.rs/type-leak/latest/type_leak/struct.LeakerResult.html#method.repeater_impls) in crate A.
  - [`Referrer`](https://docs.rs/type-leak/latest/type_leak/struct.Referrer.html): [`visitor`](https://docs.rs/type-leak/latest/type_leak/struct.Referrer.html#method.visitor) in crate B, which writes the path through the Repeater ([`Projection::of`](https://docs.rs/type-leak/latest/type_leak/struct.Projection.html#method.of)) for each entry, and [`complete_args`](https://docs.rs/type-leak/latest/type_leak/struct.Referrer.html#method.complete_args), [`Substitute`](https://docs.rs/type-leak/latest/type_leak/struct.Substitute.html) and [`substitute_args`](https://docs.rs/type-leak/latest/type_leak/fn.substitute_args.html) for the arguments of an impl. An entry is named by its [`EntryId`](https://docs.rs/type-leak/latest/type_leak/struct.EntryId.html).
  - [`repeater_definition`](https://docs.rs/type-leak/latest/type_leak/fn.repeater_definition.html), which generates the Repeater trait, and [`Projection`](https://docs.rs/type-leak/latest/type_leak/struct.Projection.html), which says how both crates name it (its path, the Leaker's id, what its impls are for) and gives its `KEY`.
  - [`ErrorItem`](https://docs.rs/type-leak/latest/type_leak/struct.ErrorItem.html), the error of the `Leaker`. The `Leaker` [recovers from errors](https://docs.rs/type-leak/latest/type_leak/struct.Leaker.html#errors-and-recovery) and goes on. [`LeakerResultError`](https://docs.rs/type-leak/latest/type_leak/struct.LeakerResultError.html) holds the errors that `finish()` returns.
- [Limitations of the `Leaker`](https://docs.rs/type-leak/latest/type_leak/struct.Leaker.html#limitations): what cannot be leaked. An error carries a `help` that says what to write instead.
- [`examples/two-macros`](https://github.com/yasuo-ozu/type-leak/tree/main/examples/two-macros): a complete, tested pair of proc macros.

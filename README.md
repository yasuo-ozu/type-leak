# type-leak crate [![Latest Version]][crates.io] [![Documentation]][docs.rs] [![GitHub Actions]][actions]

[Latest Version]: https://img.shields.io/crates/v/type-leak.svg
[crates.io]: https://crates.io/crates/type-leak
[Documentation]: https://img.shields.io/docsrs/type-leak
[docs.rs]: https://docs.rs/type-leak/latest/type-leak/
[GitHub Actions]: https://github.com/yasuo-ozu/type-leak/actions/workflows/rust.yml/badge.svg
[actions]: https://github.com/yasuo-ozu/type-leak/actions/workflows/rust.yml

A name in Rust means something only where it is written. When a proc macro copies `fn get(&self) -> MyType` from crate A into code that crate B expands, the copy looks up `MyType` in crate B, and finds nothing or the wrong type.

`type-leak` fixes this for proc-macro authors. In crate A, it finds the types that depend on where they are written and emits a small impl of a shared trait, the *Repeater*, for each. In crate B, it rewrites the copy so that each such type is written `<Self as Repeater<..>>::Type`, which resolves to the original type.

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

- The **Leaker** analyzes the item. Its result, the **Referrer**, lists the *entries*: the types whose meaning depends on where they are written (`MyType`, `Self`, `MyWrap<T>`). It travels to crate B as tokens.
- The **Repeater** is a trait defined once at the root of crate A (not in the proc-macro crate). `repeater_definition()` generates it:

  ```rust,ignore
  pub trait Repeater<T: ?Sized, const ID: u64, const NTH: usize, /* one const slot per primitive type */> {
      type Type: ?Sized;
  }
  ```

  `NTH` is the entry, `ID` identifies the analyzed item (e.g. a hash of the crate and item names), and `T` encodes the generics the entry uses.

# Quick start

```rust
use syn::visit_mut::VisitMut;
use syn::{parse_quote, GenericArgument, Ident, ItemImpl, ItemTrait, Signature};
use type_leak::{ImplSelf, Leaker, UseSite};

let item: ItemTrait = parse_quote! {
    pub trait MyLeaker<T> {
        fn f(&self, t: T) -> MyWrap<T>;
        fn g(&self) -> MyType;
    }
};

// Crate A, step 1: analyze the trait.
let mut leaker = Leaker::new();
let top_level = leaker.intern_trait(&item, true).unwrap();
let referrer = leaker.finish();

// Crate A, step 2: one Repeater impl per entry (`42` is the ID).
let self_param: Ident = parse_quote!(__S);
let impls: Vec<ItemImpl> = (0..referrer.iter().count())
    .map(|ix| {
        let r = referrer.repeater_impl(ix, ImplSelf::Param(&self_param)).unwrap();
        let (impl_generics, _, where_clause) = r.generics.split_for_impl();
        let (args, self_ty, ty) = (&r.args, &r.self_ty, &r.ty);
        parse_quote! {
            impl #impl_generics path::Repeater<#args, 42u64, #ix> for #self_ty #where_clause {
                type Type = #ty;
            }
        }
    })
    .collect();
assert_eq!(impls.len(), 2);

// Crate B, step 3: rewrite a copy of `f` for `impl MyLeaker<u8> for ..`.
let trait_args: Vec<GenericArgument> = vec![parse_quote!(u8)];
let site = UseSite::substituting(&item.generics.params, &trait_args);
let mut sig: Signature = parse_quote!(fn f(&self, t: T) -> MyWrap<T>);
referrer
    .clone()
    .visitor(top_level, |_ty, ix| {
        let args = referrer.entry_args(ix, &site).unwrap();
        parse_quote!(<Self as path::Repeater<#args, 42u64, #ix>>::Type)
    })
    .visit_signature_mut(&mut sig);
assert_eq!(
    sig,
    parse_quote!(fn f(&self, t: T) -> <Self as path::Repeater<(u8,), 42u64, 0usize>>::Type)
);
```

Only the entries are rewritten: `t: T` is left for you to write as `u8`.

In real macros, crate A passes `referrer` and `top_level` to crate B as tokens (`ToTokens` / `Parse`), usually through a `macro_rules!` it exports. [`examples/two-macros`](https://github.com/yasuo-ozu/type-leak/tree/main/examples/two-macros) is a complete, tested setup.

# Learn more

- [API docs](https://docs.rs/type-leak/latest/type_leak/): every item has examples. Start with `Leaker` (analysis: `intern_*`, `enter_*`) and `Referrer` (`repeater_impl`, `visitor`, `entry_args`).
- [Rules](https://docs.rs/type-leak/latest/type_leak/rules/index.html): the error messages and how to fix them, what cannot be leaked, and the exact rules.

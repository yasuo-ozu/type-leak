# type-leak crate [![Latest Version]][crates.io] [![Documentation]][docs.rs] [![GitHub Actions]][actions]

[Latest Version]: https://img.shields.io/crates/v/type-leak.svg
[crates.io]: https://crates.io/crates/type-leak
[Documentation]: https://img.shields.io/docsrs/type-leak
[docs.rs]: https://docs.rs/type-leak/latest/type-leak/
[GitHub Actions]: https://github.com/yasuo-ozu/type-leak/actions/workflows/rust.yml/badge.svg
[actions]: https://github.com/yasuo-ozu/type-leak/actions/workflows/rust.yml

# Overview

A name in Rust means something only where it is written. If a macro copies `fn get(&self) -> MyType` from one crate into code that another crate expands, the copy looks up `MyType` in the wrong place: it is not found, or it is another type. `Self` and the generic parameters of the item have the same problem.

`type-leak` is for proc-macro authors who copy an item (a trait, a struct, an impl, ...) across a crate boundary. It finds the types of the item that depend on where they are written, describes one small impl of a shared trait (the *Repeater*) for each, and rewrites the copy so that each of them is written `<Self as Repeater<..>>::Type`. That path resolves wherever the Repeater is visible, so the two crates share nothing else. The exact rules (contexts, `Self`, lifetimes, bounds, modules, external crates, const generics) are in the [`rules`][rules] module of the API docs.

# How it works

```text
 Crate A (defines the item)                   Crate B (copies the item)
 --------------------------                   -------------------------
 trait Tr { fn f(&self) -> MyType; }

 1. Analyze the item:                         3. Rewrite the copy:
    Leaker --> Referrer ------ as tokens --->    Referrer::visitor(..) turns
                                                   fn f(&self) -> MyType
 2. Emit the Repeater impl:                      into
    impl Repeater<..> for .. {                     fn f(&self)
        type Type = MyType;  <--------------+        -> <Self as Repeater<..>>::Type
    }                                       |
                                            +-- the path resolves through the impl
```

- **Leaker**: the `Leaker` value that analyzes the item, in crate A.
- **Leaked type**: a type whose meaning depends on where it is written: a relative path (`MyType`), `Self`, or a type that uses a generic of the item (`MyWrap<T>`). An *entry* is a leaked type with its own Repeater impl. The Leaker keeps as few as it can: if `Rel` is an entry, `Vec<(Rel,)>` needs none (but `Vec<Rel>` does: a bare `Rel` may be a const, see [Bare identifiers]).
- **Referrer**: the result of `Leaker::finish()`. It holds the entries and travels to crate B as tokens, where it rewrites copies.
- **Repeater**: the one trait both crates see. `impl Repeater<ARGS, ID, NTH> for .. { type Type = ..; }` turns the index `NTH` of an entry back into its type. `ID` identifies the Leaker: a number that is unique for each analyzed item (a proc macro does not know the path of the item, so hash the name of its crate and of the item), and `ARGS` encodes the generics the entry uses.
- **Context**: what gives an entry its meaning: the generics it uses, what `Self` is, the module. A `ContextId` names one. The id returned for an item is its *top-level* context.

Define the Repeater once, at the root of crate A (the crate that writes the impls). Crate A implements it for any `Self`, which the orphan rule (E0210) allows only for a trait of the crate itself: so it is not in the crate of your proc macros, which can export only macros, and not in a third crate. Emit it with `repeater_definition()`, which generates:

```rust,ignore
pub trait Repeater<
    T: ?Sized,
    const ID: u64,
    const NTH: usize,
    /* const BOOL: bool = false, const U8: u8 = 0, ...: one slot per primitive type */
> {
    type Type: ?Sized;
}
```

Impls and paths write only the first three arguments (`Repeater<ARGS, ID, NTH>`); the slots appear only inside the encoding of a const parameter that is not `usize` ([Const generics]). The slots need Rust 1.59. If every const parameter is `usize` or `u64`, the first three parameters alone are enough.

# Quick start

Steps 1 and 2 run in crate A, and step 3 runs in crate B.

```rust
use syn::visit_mut::VisitMut;
use syn::{parse_quote, GenericArgument, Ident, ItemImpl, ItemTrait, Signature};
use type_leak::{ImplSelf, Leaker, UseSite};

// `MyWrap` and `MyType` are relative paths, so they are leaked types.
let item: ItemTrait = parse_quote! {
    pub trait MyLeaker<T> {
        fn f(&self, t: T) -> MyWrap<T>;
        fn g(&self) -> MyType;
    }
};

// 1. Analyze the trait. `top_level` names its scope.
let mut leaker = Leaker::new();
let top_level = leaker.intern_trait(&item, true).unwrap();
let referrer = leaker.finish();
assert_eq!(referrer.required().count(), 2);

// 2. One Repeater impl per entry. `42` identifies this Leaker, `ix` the entry.
let self_param: Ident = parse_quote!(__S);
let impls: Vec<ItemImpl> = referrer
    .required()
    .map(|(ix, _)| {
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
let expected: ItemImpl = parse_quote! {
    impl<T, __S: ?::core::marker::Sized> path::Repeater<(T,), 42u64, 0usize> for __S {
        type Type = MyWrap<T>;
    }
};
assert_eq!(impls[0], expected);

// 3. Copy `f` into `impl MyLeaker<u8> for ..`. Each leaked type becomes a path through the
// Repeater, with the generics of the type as this impl gives them (`site`). `top_level` tells the
// visitor the generics, `Self` and module of the copy. The visitor replaces leaked types only: the
// rest of the signature (`t: T`) is yours to write for `u8`.
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
let expected: Signature =
    parse_quote!(fn f(&self, t: T) -> <Self as path::Repeater<(u8,), 42u64, 0usize>>::Type);
assert_eq!(sig, expected);
```

In a real macro, `Referrer` and `ContextId` convert to and from tokens (`ToTokens`, `Parse`). Both crates must use the same version of `type-leak`.

# Two macros, end to end

[`examples/two-macros`](https://github.com/yasuo-ozu/type-leak/tree/main/examples/two-macros) is a complete setup, built and tested with this crate: `#[export_trait]` in crate A analyzes a trait (a type in a sibling module, a generic wrapper, `Self`) and `#[forward(..)]` in crate B implements it for a wrapper. What the quick start does not show:

- **Crate A cannot call crate B's macro**: it does not know B's input. So `#[export_trait]` writes a `macro_rules!` that B calls. It adds `$crate::Repeater`, the id, `#referrer`, `#top_level` and the item to B's input, and calls a proc macro that parses them back.
- **Emit the impls of all the entries** (`0..referrer.iter().count()`), not only of `required()`. Crate B may use a strict Referrer, which replaces the optional entries too, and the flag is not serialized.
- **The copy is yours to write**: the visitor replaces leaked types only. For `impl MyTrait<u8> for ..`, write `u8` for `T` in the rest of the signature yourself.
- **Report each error at its span** with `NotInternableError::items()`. `cargo check --features bad` in the `repo` crate shows it.

# Common tasks

Each item named here has examples in the [API docs](https://docs.rs/type-leak/latest/type_leak/).

- **Analyze an item.** [`Leaker::intern_struct`][intern] (also `intern_enum`, `intern_union`, `intern_impl`, `intern_trait`) takes `(&item, allow_self)`, and `intern_mod` takes the module (see `Referrer::module_path()` for the nested modules). It returns the top-level `ContextId` or a `NotInternableError`. Keep the id: `Referrer::visitor()` needs it. With `allow_self = false`, a type that names `Self` is split into parts instead of leaked whole: use it if your Repeater impls are for a marker type, which cannot name `Self` ([Self rules]).
- **Analyze part of an item.** [`Leaker::enter_impl`][enter] (also `enter_generics`, `enter_type`, `enter_trait`, `enter_mod`) runs a closure on a `View` of a scope whose generics and `Self` you declare: `view.visit_type(..)`, `visit_path_arguments(..)`, `visit_where_clause(..)`. A scope that fails forgets what it found, and the scope around it can go on ([Scopes]).
- **Emit the Repeater impls.** For each entry (`Referrer::iter()`; `required()` is the minimum for a non-strict Referrer), [`Referrer::repeater_impl`][impl] gives the generics, the type, the first trait argument and the module of the impl (write it in that module). `ImplSelf::Param` writes it for any implementor (or for `Self`), `ImplSelf::Marker` for a marker type. `Referrer::is_plain(ix)` is true if `impl Repeater<(), ID, IX> for Marker` is enough. A const parameter that is not `usize` needs `repeater_definition()` and `RepeaterSlots` ([Const generics]).
- **Rewrite a copy.** [`Referrer::visitor`][visitor] returns a `VisitMut` for the copy. Get the arguments of each path from `Referrer::entry_args(ix, &UseSite)`: `UseSite::identity()` where the copy declares the generics of the item itself, `UseSite::substituting(..)` for `impl MyLeaker<u8> for ..`. Visit a whole item or signature, not a type cut out of another. Omitted defaults of the trait's parameters: `Referrer::complete_args()`. Repeater impls for a marker type: `Visitor::with_self()`. Copying a module with structs that name their own `Self`: `Visitor::with_nested_self()`. If the copy declares the generics with other bounds, call `Visitor::with_outermost_generics_as_leakers()` ([Contexts]).
- **Report errors.** [`NotInternableError::items`][items] gives one `ErrorItem` (span, message, notes) per offending path. The error also converts into a `syn::Error`, and its `ToTokens` writes `compile_error!` (or nightly diagnostics).
- **Configure.** [`LeakerConfig`][config] lists paths that are not leaked: `allow_primitive()`, `allow_std_prelude()` (without it, `trait Tr: Clone` is an error), your own crates. `Referrer::allow_ambiguous_crate_name(false)` makes the Referrer strict about absolute paths of other crates ([External crates]).

# Errors and how to fix them

The Leaker returns the first group itself. The `rustc` errors come from the generated code, if a rule was not met.

| Message or error | Fix | Rule |
| --- | --- | --- |
| `use absolute path: 'Clone'` | A relative path that cannot be leaked (a trait in a bound, `MyType<'_>`, `impl Trait`): write it in full, `::core::clone::Clone`, or allow it. The notes say why. | [Contexts] |
| `` `Self` has no meaning here `` | Enter a scope that declares `Self` (`enter_type`, `enter_impl`, `enter_trait`). | [Self rules] |
| `` `Self::A` cannot be written in the impl of the Repeater `` | Name the trait: `<Self as ::path::Trait>::A`. | [Self rules] |
| `` the bound of `T` mentions a `Self` which has no name here `` | Write the bound without `Self`, or the type with an absolute path. | [Self rules] |
| `` `S<T>` is in the self type of the impl, and a bound of its generics names `Self` `` | Write the self type with an absolute path, or the bounds without `Self`. | [Self rules] |
| `` `Wrap<T>` has a generic whose bound mentions `Self`, which cannot be leaked `` | Write the bound without `Self`, or use `allow_self = true`. | [Self rules] |
| `` `Wrap<T>` is in the default of a parameter, and names `Self` `` | Write the default without `Self`. | [Self rules] |
| `` `Wrap<T>` is in a bound or where predicate, and a bound of its generics names .. `` | Write the type of the bound with an absolute path, or use the type outside predicates. | [Bounds] |
| `` `Wrap<'x>` leaves the lifetime `'x` of the method unconstrained in the copy (E0195) `` | Keep `'x` in another argument, or make the path absolute. | [Method lifetimes] |
| rustc E0309 | Write the bound: `T: 'a`, or `where Self: 'a`. | [Outlives bounds] |
| rustc E0106 | Name the lifetimes, give the method a receiver, or make the path absolute. | [Elided lifetimes] |
| rustc E0581, E0582 | Keep an input that names the `for<'x>` lifetime. | [Binder lifetimes] |
| rustc E0478 on `dyn Trait + 'a` | Write `&'a dyn Trait`, or make the trait path absolute. | [Trait objects] |
| rustc E0277, E0271, E0275, E0284 in a bound | The bound names a type that rustc normalizes in another form. | [Bounds] |
| a constant is read as a type | Write `{ ::my_crate::MY_CONST }`. | [Bare identifiers] |
| a type is not replaced in the copy | Visit the whole item, and declare the generics as the entry has them. | [Contexts] |
| a type that names `Self` is not replaced in a struct, trait or impl nested in a module | Call `Visitor::with_nested_self()` and write the type of the item in the Leaker's crate. | [Self rules] |
| "incompatible type-leak" when parsing | Use the same version of `type-leak` on both sides. | |
| panic in `visitor()` or `expand()` | Pass the `ContextId` that the analysis returned for the item. | [Contexts] |

# What cannot be leaked

The Leaker reads only the item you give it, not the definitions it names. So some code needs a hint, or another form.

- A relative path with an elided lifetime (`MyType<'_>`), `impl Trait`, a trait or a value in a bound: write an absolute path ([Contexts]).
- A path with a lifetime argument records no `T: 'a`: write the bound ([Outlives bounds]).
- An output that relies on elision or names a `for<'x>` lifetime needs an input that keeps it ([Elided lifetimes], [Binder lifetimes]).
- A `dyn Trait` without a lifetime bound is never leaked alone ([Trait objects]).
- A method lifetime that only a leaked type mentions breaks a trait impl ([Method lifetimes]).
- A leaked type in a bound or where predicate fails if rustc cannot prove the bounds of its generics ([Bounds]).
- `Self::A` cannot be in an entry if `Self` is a type, and some bounds that name `Self` cannot be written ([Self rules]).
- A bare identifier argument may be a const and may have to be treated as a type ([Bare identifiers]).
- An absolute path of another crate is trusted only if both sides have that crate ([External crates]).
- A shadowed primitive name (`type u8 = i32;`) breaks the copy: emit `Leaker::assertions()` to check ([Bare identifiers]).

# MSRV

The library supports Rust 1.56 (`rust-version` in `Cargo.toml`). Check it with `cargo build --lib`. The tests and the test crates need a newer Rust.

The current dependencies do not build on 1.56: `syn` 2.0.87 needs 1.61, the latest `num-traits` (used by `safegraph`) needs 1.60, and `unicode-width` 0.1.14 needs 1.66 without saying so. Cargo of that age cannot pick versions by `rust-version`, so generate the lockfile with a recent Cargo. `Cargo.lock` is not part of the repository or the package. Cargo 1.56 reads lockfile versions up to 3: if yours says `version = 4`, change it to `3`.

The simplest way, which CI uses (see `.github/workflows/rust.yml`), is a Cargo 1.84 or later that resolves by `rust-version`, with `unicode-width` pinned:

```sh
CARGO_RESOLVER_INCOMPATIBLE_RUST_VERSIONS=fallback cargo +stable generate-lockfile
cargo +stable update -p unicode-width --precise 0.1.11
cargo +1.56 build --lib
```

Or pin these versions by hand. They are known to build with Rust 1.56. The other packages are the latest compatible ones: `safegraph` 0.3.0, `template-quote` 0.4.4, `autocfg` 1.5.1, `version_check` 0.9.5, `proc-macro-error` 1.0.4 and the `syn` 1.0.109 it uses.

```sh
cargo generate-lockfile
cargo update syn@2 --precise 2.0.56
cargo update proc-macro2 --precise 1.0.86
cargo update quote --precise 1.0.36
cargo update unicode-ident --precise 1.0.12
cargo update unicode-width --precise 0.1.11
cargo update num-traits --precise 0.2.15
cargo +1.56 build --lib
```

# Application

Proc-macro crates that use `type-leak`: [sumtype](https://crates.io/crates/sumtype), [flat_enum](https://crates.io/crates/flat_enum), [parametrized](https://crates.io/crates/parametrized).

[rules]: https://docs.rs/type-leak/latest/type_leak/rules/index.html
[Contexts]: https://docs.rs/type-leak/latest/type_leak/rules/index.html#contexts
[Scopes]: https://docs.rs/type-leak/latest/type_leak/rules/index.html#scopes
[Self rules]: https://docs.rs/type-leak/latest/type_leak/rules/index.html#self-rules
[Outlives bounds]: https://docs.rs/type-leak/latest/type_leak/rules/index.html#outlives-bounds
[Binder lifetimes]: https://docs.rs/type-leak/latest/type_leak/rules/index.html#binder-lifetimes
[Elided lifetimes]: https://docs.rs/type-leak/latest/type_leak/rules/index.html#elided-lifetimes
[Trait objects]: https://docs.rs/type-leak/latest/type_leak/rules/index.html#trait-objects
[Bounds]: https://docs.rs/type-leak/latest/type_leak/rules/index.html#bounds-that-name-leaked-types
[Method lifetimes]: https://docs.rs/type-leak/latest/type_leak/rules/index.html#method-lifetimes
[Bare identifiers]: https://docs.rs/type-leak/latest/type_leak/rules/index.html#bare-identifiers-and-consts
[External crates]: https://docs.rs/type-leak/latest/type_leak/rules/index.html#external-crates
[Const generics]: https://docs.rs/type-leak/latest/type_leak/rules/index.html#const-generics
[intern]: https://docs.rs/type-leak/latest/type_leak/struct.Leaker.html#method.intern_struct
[enter]: https://docs.rs/type-leak/latest/type_leak/struct.Leaker.html#method.enter_impl
[impl]: https://docs.rs/type-leak/latest/type_leak/struct.Referrer.html#method.repeater_impl
[visitor]: https://docs.rs/type-leak/latest/type_leak/struct.Referrer.html#method.visitor
[items]: https://docs.rs/type-leak/latest/type_leak/enum.NotInternableError.html#method.items
[config]: https://docs.rs/type-leak/latest/type_leak/struct.LeakerConfig.html

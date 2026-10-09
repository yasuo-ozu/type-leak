# type-leak crate [![Latest Version]][crates.io] [![Documentation]][docs.rs] [![GitHub Actions]][actions]

[Latest Version]: https://img.shields.io/crates/v/type-leak.svg
[crates.io]: https://crates.io/crates/type-leak
[Documentation]: https://img.shields.io/docsrs/type-leak
[docs.rs]: https://docs.rs/type-leak/latest/type-leak/
[GitHub Actions]: https://github.com/yasuo-ozu/type-leak/actions/workflows/rust.yml/badge.svg
[actions]: https://github.com/yasuo-ozu/type-leak/actions/workflows/rust.yml

# Overview

In Rust, a type is only valid where it is written. A relative path, `Self` or a generic parameter means something only in its own crate and item. A macro that copies such a type into another place can produce a type that no longer resolves.

`type-leak` fixes this in three steps. It finds the types of an item that depend on where they are written. It lets you emit one impl of a shared trait, the Repeater, for each of them. It rewrites a copy of the item so that each such type becomes `<Self as Repeater<..>>::Type`. That path is valid wherever the Repeater is visible, so the copy works in another crate.

# Glossary

- **Leaker**: the item whose types are copied, such as a trait, a struct or an impl. The `Leaker` type analyzes it.
- **Leaked type**: a type in the Leaker whose meaning depends on where it is written. A relative path (`MyType`), `Self` and a generic of the Leaker (`MyWrap<T>`) make a type leaked.
- **Entry**: a leaked type that has its own index and its own Repeater impl. The Leaker keeps as few entries as it can. If `Rel` is an entry, `Vec<Rel>` needs none.
- **Context**: what gives an entry its meaning. It is the generics the entry uses, what `Self` means if the entry names `Self`, and the module if its path is relative. The same tokens in two contexts are two entries. A `ContextId` names a context.
- **Repeater**: the one trait that both sides can see. Its impls, written in the Leaker's crate, turn an entry index back into a type. `ID` identifies the Leaker, `NTH` is the entry index, and `T` encodes the generics of the context.
- **Referrer**: the side that receives the copy, and the `Referrer` value that `Leaker::finish()` returns for it. It holds the entries and rewrites copies.
- **Copy**: the item or signature that a macro copies from the Leaker into the Referrer's crate. Its leaked types must be replaced.
- **Scope**: a part of the Leaker that is analyzed with known generics, `Self` and module. See [Scopes](#scopes).

```rust,ignore
// Defined once in the user crate. `repeater_definition()` can generate it.
pub trait Repeater<T: ?Sized, const ID: u64, const NTH: usize> {
    type Type: ?Sized;
}
```

# Walkthrough

The three steps below show both sides in one place. The first two run in the Leaker's crate. The third runs in the Referrer's crate. A macro passes the `Referrer` and the top-level `ContextId` from the first side to the second as tokens.

```rust
use syn::visit_mut::VisitMut;
use syn::{parse_quote, GenericArgument, ItemImpl, ItemTrait, Signature};
use type_leak::{Leaker, RepeaterSlots};

// 1. Analyze the trait. `MyWrap<T>` and `MyType` are leaked types.
let item: ItemTrait = parse_quote! {
    pub trait MyLeaker<T> {
        fn f(&self, t: T) -> MyWrap<T>;
        fn g(&self) -> MyType;
    }
};
let mut leaker = Leaker::new();
let top_level = leaker.intern_trait(&item, false).unwrap();
let referrer = leaker.finish();
assert_eq!(referrer.iter().count(), 2);
assert_eq!(referrer.context_generics(top_level).params.len(), 1);

// 2. Emit one Repeater impl per entry. It declares `impl_generics()` and
// encodes the generics the entry uses in its trait arguments.
let slots = RepeaterSlots::new(parse_quote!(path::Repeater));
let id = 42u64;
let impls: Vec<ItemImpl> = referrer
    .iter()
    .enumerate()
    .map(|(ix, ty)| {
        let context = referrer.context(ix);
        let used = referrer.context_generics(context);
        let encoded = slots.encode_generics_params_to_ty(&used.params).unwrap();
        let declared = referrer.impl_generics(context);
        let params = declared.params.iter();
        parse_quote! {
            impl<#(#params,)* __S: ?Sized> path::Repeater<#encoded, #id, #ix> for __S {
                type Type = #ty;
            }
        }
    })
    .collect();
let expected: ItemImpl = parse_quote! {
    impl<T, __S: ?Sized> path::Repeater<(T,), 42u64, 0usize> for __S {
        type Type = MyWrap<T>;
    }
};
assert_eq!(impls[0], expected);

// 3. In the Referrer's crate, copy `f` into `impl MyLeaker<u8> for ..`. `top_level` is the scope
// of the copy: it gives the visitor the generics (`T`), `Self` and the module.
// Each leaked type becomes a path through the Repeater, with its generics as arguments.
let trait_args: Vec<GenericArgument> = vec![parse_quote!(u8)];
let mut sig: Signature = parse_quote!(fn f(&self, t: T) -> MyWrap<T>);
let mut visitor = referrer.clone().visitor(top_level, |_ty, ix| {
    let args = referrer.generic_args(ix, &item.generics.params, &trait_args).unwrap();
    let used = &referrer.generics(ix).params;
    let encoded = slots.encode_generics_args_to_ty(&args, used).unwrap();
    parse_quote!(<Self as path::Repeater<#encoded, #id, #ix>>::Type)
});
visitor.visit_signature_mut(&mut sig);
let expected: Signature = parse_quote!(
    fn f(&self, t: T) -> <Self as path::Repeater<(u8,), 42u64, 0usize>>::Type
);
assert_eq!(sig, expected);
```

Notes:

- `Leaker::new()` allows no path. `Leaker::from_config()` takes a `LeakerConfig` that allows some, such as the standard prelude, so they are not leaked.
- Each `intern_*` method returns the top-level `ContextId` of the item, or a `NotInternableError` that lists the paths that cannot be leaked. The error converts into a `syn::Error`, and its `ToTokens` impl renders `compile_error!`.
- `intern_struct()`, `intern_enum()`, `intern_union()`, `intern_impl()` and `intern_trait()` take `(&item, allow_self)`. `Self` is the type for the first four and the implementor for a trait. `intern_mod()` takes only the module. See [Self rules](#self-rules) for `allow_self`.
- An impl declares only the generics its type uses. `MyWrap<T>` uses `T`, and `MyType` uses none. The encoding is `&'a ()` for a lifetime `'a`, `T` for a type `T` and `*const T` for a type `T: ?Sized`, because only the last element of a tuple may be unsized. A generic that the copy declares itself, such as one of a method, is passed by its own name.
- If the Referrer omits an argument with a default, complete the arguments with `Referrer::complete_args()` first. A default is analyzed with the parameters before it, so it cannot name a later one or `Self`.
- For the arguments of the trait itself (`u8` of `impl MyLeaker<u8>`), use `Referrer::context_generics(top_level)` and `Referrer::context_generic_args()`. The generics of a context are sorted, so they are matched by name and not by position. Pass `top_level` along with the `Referrer` (`quote!(#referrer, #top_level)`), because the Referrer does not know which context is the top-level one, and `Referrer::visitor()` needs it. A `ContextId` is written and parsed as an integer.
- Do not encode the generics of the top-level context for every entry. They have only the bounds the trait writes, so rustc rejects them (E0309) for an entry such as `&'a Rel<T>`. Encode the generics of each entry, as above.
- A lifetime bound of an entry (`T: 'a`) is encoded in the trait arguments, so the impl must not declare it. Use `Referrer::impl_generics()`, or `without_encoded_bounds()` for other generics. See [Outlives bounds](#outlives-bounds).
- `Referrer::visitor(context, ..)` is the rewriter of a copy. `context` is the scope of the copy, the top-level `ContextId` of the call that analyzed what the copy is made from, and the visited syntax is inside it. `Referrer::expand(context, ty, ..)` rewrites one type instead of a whole item. Give them the whole type or item. See [Outlives bounds](#outlives-bounds).
- `Referrer::visitor()` and `Referrer::expand()` panic if `context` is not one of the `Referrer::contexts()`.

## Const generics

A const parameter is treated like a type parameter. The Repeater needs a type that carries its value. `encode_generics_params_to_ty()` encodes `const N: usize` as `[(); N]`. For other primitive types, define the Repeater with `repeater_definition()`. It adds a defaulted const slot for each primitive type, and `RepeaterSlots` encodes values into them. The generated Repeater needs Rust 1.59 and the 2018 edition, while `type-leak` itself supports 1.56.

## Contexts

The context of an entry decides which Repeater impl it needs.

- The generics are only those the type uses. They are the ones the type names, those that the bounds of these generics name, and, if the type names `Self`, those in the description of `Self`. A where predicate on one generic (`U: Into<T>`) is a bound of it, as if it was written on `U`: it is in the context of what uses `U`, not of what uses `T`, and so is the same one written inline. A predicate on another type (`Vec<T>: Clone`) belongs to each generic it names.
- The same type found by several scopes is one entry if the contexts are equal, whatever the other generics of the scopes are.
- Equal declarations give equal contexts. Parameters are sorted by kind and name, bounds are sorted and moved from the where clause to their parameter, and defaults are removed. The output type of the sugar of `Fn` is parenthesized where a `+` would join it (`T: Fn() -> (&dyn Tr) + Clone`), whichever way the source wrote it.
- The Leaker numbers a context when it first appears, from 0. It numbers a module when it is entered, a nested scope when it is left, and the types of the outermost scope when it is left. It never renumbers them, except for the contexts of a scope that fails and is rolled back. So `Referrer::contexts()` can have gaps. It has at least one context. With no other, it is context 0 of the types that use no generics.
- The top-level context is the one of the generics of the item or scope itself. It differs from an entry's context if the entry uses fewer generics, as `Rel` does in `trait Tr<T> { fn f(&self) -> Rel; }`. The Referrer keeps it even if it has no entry.
- The top-level context also records what `Self` is, if an entry names it or has a bound rewritten for it ([Self rules](#self-rules)) (`Referrer::context_self()`), and the module the scope was entered in (`Referrer::context_module()`). So scopes with the same generics but another `Self` return other ids, and `Referrer::visitor(context, ..)` knows the generics, `Self` and module of the copy from the id alone.
- A type is replaced if its context, computed from the generics in scope, is an entry's. The generics in scope are those of `context`, then those that the visited syntax declares. A name that neither declares is not a generic, so it is not matched: the Referrer does not guess which generics are the Leaker's. If no entry has exactly the context, one that has bounds on `Self` alone besides (`where Self: Sized`) is taken, since a copy where `Self` is a type parameter (`Visitor::with_self()`) cannot write them. An outermost item that declares exactly the generics of `context` (the copy of the item itself) does not declare them again. If the copy declares them with other bounds (`impl<T: Clone> MyLeaker<T> for ..`), call `Visitor::with_outermost_generics_as_leakers()`.
- `Referrer::expand()` knows only the generics of `context`. It does not find a type that names a generic of a method: visit the signature.

## Which types become entries

`Leaker::finish()` calls `Leaker::reduce_roots()`. It picks the fewest entries from which every leaked type can be built, because each entry needs an impl. It is a minimum vertex cut of the graph of the types, and the result is deterministic. So `Rel` alone covers `Vec<(Rel,)>`, `Option<(Rel,)>` and `Box<(Rel,)>`.

Some types are always entries as a whole, never replaced by their parts. They are a reference with a lifetime around an entry (`&'a Rel`), a path with a lifetime argument, a path with a bare identifier argument (`Wrap<SIZE>`), and a type with a type in an expression (`[u8; size_of::<Big>()]`). A trait object without a lifetime bound inside such a reference or path is one too. See [Trait objects](#trait-objects).

# Scopes

`intern_*` analyzes a whole item. To analyze a fragment, whose generics and `Self` the caller knows, enter a scope. `Leaker::enter_impl()`, `enter_type()`, `enter_trait()`, `enter_generics()` and `enter_mod()` take a closure. The closure gets a `View`. A `View` has `visit_item()`, `visit_type()`, `visit_path_arguments()`, `visit_where_clause()` and `visit_type_param_bound()`. It also has the same `enter_*` methods for nested scopes. A scope returns its top-level `ContextId` and the value of the closure.

```rust
use syn::{parse_quote, ItemImpl, Type};
use type_leak::Leaker;

// Analyze only the arguments of the self type and of the trait of an impl.
let item: ItemImpl = parse_quote! {
    impl<T: ::path::Bound> ::path::MyTrait<Rel<T>> for MyType<T, Rel2> {}
};
let (_, trait_path, _) = item.trait_.as_ref().unwrap();
let self_args = match &*item.self_ty {
    Type::Path(path) => &path.path.segments.last().unwrap().arguments,
    _ => unreachable!(),
};
let mut leaker = Leaker::new();
leaker
    .enter_impl(&item.generics, &item.self_ty, true, |view| {
        view.visit_path_arguments(self_args);
        for seg in &trait_path.segments {
            view.visit_path_arguments(&seg.arguments);
        }
    })
    .unwrap();
// `Rel<T>` uses `T`, and `Rel2` uses nothing.
assert_eq!(leaker.finish().iter().count(), 2);
```

- The generics of a scope are those of the Leaker. In a nested scope they are local generics, and the generics around it stay in scope.
- `View` implements `syn::visit::Visit`, so a `syn::visit` walk of an item analyzes it. `Visit::visit_signature()` is a method's signature, with local generics (and the check of [Method lifetimes](#method-lifetimes)). `Visit::visit_path()` at the root analyzes only the arguments of a path and does not check the path: to analyze a trait bound with its root checked, use `visit_type_param_bound()`.
- `check()` finds what `visit_type()` would find in a type here (the causes that force it to be leaked and those that forbid it, and whether it is an optional entry), with the generics around, `Self` and the module, and records nothing.
- A function that is itself the analyzed item has no local generics: enter its generics with `enter_generics(&sig.generics, ..)` and visit its input and output types and its where clause with `visit_type()` and `visit_where_clause()`.
- `Self` is what the scope declares. In `enter_generics()` and `enter_mod()` it has no meaning, so naming it is an error. An item visited with `visit_item()` has a `Self` of its own.
- Scopes with equal generics return the same id, if they are in the same module and, when an entry names `Self`, declare the same `Self`.
- The generics of a scope are not analyzed. To copy their bounds or where clause, visit them with `visit_type_param_bound()` and `visit_where_clause()`. The default of a type parameter is analyzed with the earlier parameters as the generics. The defaults of const parameters are analyzed with the syntax that visits them.
- A scope is all-or-nothing. If the syntax it visits has an error, it rolls back everything it recorded, including its inner scopes. It returns its own errors with spans, in order. It never returns those of a failed inner scope, with or without `?`. Ids returned earlier stay valid. The scope around it can go on. A panic in the closure rolls the scope back too, and an outer scope may catch it with `catch_unwind()` and go on.

## Modules

`Leaker::intern_mod()` and `Leaker::enter_mod(hint, declared, ..)` analyze a module. The items in it are nested items, so their generics are local. `hint` is only for debugging.

- `declared` lists the names declared in the module. They are copied with it, so they are not leaked. Give them first. A name counts as declared if it is declared directly (unless a generic parameter shadows it), named by `self::` or `super::` without leaving the analyzed module, or imported with an absolute path. `use super::X` and `use crate::X` are leaked.
- A relative path means what it means in its module. So an entry is keyed by its module if its path is relative, if it names an item that the module declares (`crate::base::B1<R0>` with `R0` declared in the module: `crate::base::B1` cannot be leaked alone), if a path in what `Self` means or in a bound of its generics is neither absolute nor starts with `crate::` or `$crate::`, or if it is a type macro. Other entries are shared by all modules.
- Emit the Repeater impl of a keyed entry inside its module. `Referrer::context_module()` tells which module it is. `Referrer::module_path()` gives its path relative to the module of `intern_mod()` (`[]` for that module). A module found by `visit_item()` has such a path. A module entered by `enter_mod()` has no path, only its hint. The impl is in the Leaker's module, so an entry that names a declared item (`R0` above) names the Leaker's item, not the one the copy of the module declares again.

To rewrite a copy, tell the visitor where it is: `Referrer::visitor(context, ..)`. An entry with no module matches in every module. An entry of a module matches only there.

- The visited syntax is inside the module of `context`, which is `context` itself if it is a module, and else `Referrer::context_module(context)`. A `mod` item is a module nested in it, found by name, never the module itself. If the Leaker saw no such module, it counts as a module no entry depends on, so only entries with no module match in it. To rewrite a copy of a whole module, visit its items, or a `File` of them. A `File` hides the entries of no module that name what it declares, as a module does. Items visited one by one do not know their siblings.
- A context with no module (`intern_trait()`, `enter_generics()`) matches only entries with no module. A `mod` item there is a module found with no module around it, which is found by its name.
- The module of a scope entered inside a module is that module, so a copy in the scope is looked up there. The top-level context of a nested scope has only the generics of that scope, not those around it.
- Two `intern_mod()` calls return the ids of two modules. Rewrite the copy of each with its own id.

# Limitations

Each entry gives the rule, a short example and what to write instead.

## Outlives bounds

A reference `&'a X` is valid only if `X: 'a` holds. The Leaker records this bound for an entry, and the encoded arguments carry it: `T: 'a` as `&'a T`, `'b: 'a` as `&'a &'b ()`, and `Self: 'a` as `&'a Self`. `Self::A: 'a` is a where predicate of the context. The Leaker also records the bounds you write. It cannot read the definition of a path, so a path with a lifetime argument records nothing, even if its definition needs `T: 'a`. A guessed bound would make the copy stricter than the original for a path that needs none. Without the bound, rustc rejects the Repeater impl (E0309).

```rust,ignore
fn f<'a, T>(x: Rel<'a, T>);      // rejected: `struct Rel<'a, T>(&'a T)` needs `T: 'a`
fn f<'a, T: 'a>(x: Rel<'a, T>);  // good
```

Instead, write the bound: `T: 'a`, or `where Self: 'a` for `Rel<'a, Self>`. A struct that holds a leaked type under such a path writes it too: `struct F<'a, T: 'a>(Ref<'a, Rel2<T>>)`. The copy must write the same bound, or the entry is not found.

- The lifetimes around a type count. `Rel2<T>` in `&'a Vec<Rel2<T>>` is an entry with the generics `<'a, T: 'a>`. A reference, tuple, array, slice, pointer, path argument, function pointer or trait object passes the lifetime on. An associated type path (`<X as Tr>::A`, `T::A<X>`), `impl Trait` and an array length do not. The same type under another lifetime is another entry. rustc infers `T: 'a` for the original `struct F<'a, T>(&'a Vec<Rel2<T>>)`, but not for a copy that uses `<S as Repeater<..>>::Type` (E0309). So give `expand()` and `visitor()` the whole type or item, not a type cut out of another.
- A trait object argument counts too, also if the object has its own bound. `&'a dyn Iterator<Item = Rel2<T>>` needs `T: 'a`.
- Only a lifetime parameter counts as `'a`. `'static`, `'_` and the lifetime of a `for<..>` binder are never recorded, and `T: 'static` is not encoded. A binder around the type does not stop the lifetimes outside it: `&'a for<'x> fn(&'x X)` needs `X: 'a`.
- `where Self: 'a` and `where Self: Sized + 'a` count as `Self: 'a`. Only a bound by a lifetime parameter moves into `Referrer::context_self_outlives()`. `Self: 'static`, a bound under `for<..>` and `Self::Item: 'a` stay where predicates. A `Self: 'a` that only the where clause of the trait writes is not in the copy of a method. For a lifetime that the copy declares (`impl<'a> MyLeaker<'a> for ..`), write `where Self: 'a` in the copy again.
- `&'a Self::A`, an associated type of the implementor of a trait (`Self::A` in a type, with no trait written, and no arguments), records the where predicate `Self::A: 'a` besides `Self: 'a`, as if you wrote it. The original gets it from the well-formedness of its signature (`fn f<'a>(&'a self, x: Rel<&'a Self::A>)`), which the Repeater's impl does not have. The Leaker records no `T: 'a` for the arguments of the trait, though rustc proves `<Self as Tr<T>>::A: 'a` through them: the original does not know them either, so a copy that needs them would not compile. The predicate is in the impl of the Repeater (`where <__S>::A: 'a`), and in the copy it follows from the normalized type. rustc reads `where <` as generics of the where clause, so write it after another predicate, or parenthesize it: `(<__S>::A): 'a`. Other projections (`T::A`, `<T as Tr<'b>>::A`, `Self::A<'b>`) get no such predicate: write `where T::A: 'a` if the original needs it.
- A type with `'static` needs no bound. The original already declares `T: 'static`, because rustc infers none for a struct, and so must the copy.
- The value of an associated type or GAT, such as `type A<'a, T> = Ref<'a, Rel2<T>>;`, is rejected like the original. Write `type A<'a, T: 'a>`.

## Binder lifetimes

The lifetimes of a `for<..>` binder are generics of the types inside it. `Rel<'x>` in `for<'x> fn(Rel<'x>)` is leaked with `<'x>`. The copy fails if the output names `'x` and no input keeps `'x` as written. The output is a return type, the output of `Fn(A) -> R`, or an associated type binding such as `Iterator<Item = R>`. An elided lifetime in the output is `'x` if that is the only lifetime of the inputs (`for<'x> fn(Rel<'x>) -> &u8`). rustc does not count a lifetime as constrained if it appears only in an associated type path such as `<T as Tr<'x>>::A`. The errors are E0581 for a function pointer and E0582 for `Fn`.

An input that keeps `'x` is an anchor. With no anchor, the Leaker leaks the whole function pointer or trait object. That is an error if it cannot, as in a bound, in a where predicate, with a trait object under an elided reference (see [Elided lifetimes](#elided-lifetimes)) or in `impl for<'x> Fn(..)`.

```rust,ignore
F: for<'x> Fn(Rel<'x>) -> &'x u8                // error: no anchor
F: for<'x> Fn(&'x u8, Rel<'x>) -> &'x u8        // good: `&'x u8` anchors `'x`
F: for<'x> Fn(::my_crate::Rel<'x>) -> &'x u8    // good: nothing is leaked
```

Instead, keep an anchor, or make the path absolute.

- An associated type path (`<T as Tr<'x>>::A`, `T::A<'x>`), an array length, `impl Trait` and a type macro do not anchor. The rest of a type does, however deep, as rustc checks. That includes `Box<dyn Iterator<Item = &'x u8>>`, `Box<dyn Tr + 'x>` and `Box<dyn Fn() -> &'x u8>`.
- A type that is leaked as a whole keeps no anchor. So does an external type that the Leaker does not allow, such as `::other::Wrap<&'x u8>`, since the Referrer may replace it. A `Parser<'x>` next to it cannot be leaked alone. Occurrences in `::core` paths, in allowed paths and in types that are not leaked stay as written.
- Only the associated type bindings and the output of the bound itself are its output. `T: for<'x> Tr<u8, Output = &'x u8>` is rejected. The bounded type and the arguments of a bound anchor the output of that bound only.
- A type that names `'x` is leaked only if its own path depends on the context. So `(&'x str, Parser<'x>)` keeps `&'x str`, and only `Parser<'x>` is leaked.
- A binder inside an input is not probed again. A type that uses its lifetime makes the inner function pointer or trait object count as leaked, which loses its anchors. This is cautious.
- A `::core` or allowed path is assumed to be a struct, an enum or a union. A type alias that drops the lifetime would not anchor, so allow only crates without such aliases.

## Elided lifetimes

An elided lifetime in the output of a signature is the only lifetime of its inputs: the elided one of `&Rel<T>` in `Fn(&Rel<T>) -> &u8`. A leaked type that has lifetimes among its generics names them in the arguments of its projection (`<S as Repeater<(&'a (), T, &'a T), ..>>::Type`), so the inputs of the copy have more lifetimes, and rustc cannot tell which one the output has (E0106). `Rel<T>` has `<'a, T: 'a>` where the Leaker has `'a` and `T: 'a`. The signature is the output of `Fn(..) -> R`, of a function pointer, or of a method with no receiver that is a reference (`&self`, `self: &Self` and `self: Pin<&mut Self>` take the lifetime of the output). If the inputs show no lifetime, a path hides the one that the output has (`fn(Ref<u8>, Rel<T>) -> &u8` with `Ref<'a, T>`): the Leaker cannot see it, and takes it as the lifetime that elision gives, so a leaked type that adds lifetimes breaks it as well.

The Leaker leaks the whole function pointer or trait object instead of the type in its inputs. Its impl of the Repeater keeps the elision as written. That is an error if it cannot, as in a bound, in `impl Fn(..)`, in a method, or in a trait object under an elided reference, which takes its lifetime from it.

```rust,ignore
fn f(&self, x: Box<dyn Fn(&Rel<T>) -> &u8>);    // good: the trait object is leaked, with `'a, T`
fn f(&self, x: &'a dyn Fn(&Rel<T>) -> &u8);     // good: the whole reference is leaked
fn f(&self, x: &dyn Fn(&Rel<T>) -> &u8);        // error: elided reference
fn g(x: &Rel<T>) -> &u8;                        // error: no receiver
fn g<'x>(x: &'x Rel<T>) -> &'x u8;              // good
F: Fn(&Rel<T>) -> &u8                           // error: a bound
F: for<'x> Fn(&'x Rel<T>) -> &'x u8             // good
fn f(&self, x: fn(Ref<u8>, Rel<T>) -> &u8);     // good: the function pointer is leaked, `Ref` hides a lifetime
F: Fn(Ref<u8>, Rel<T>) -> &u8                   // error: a bound, and `Ref` hides a lifetime
F: Fn(Ref<'_, u8>, Rel<T>) -> &u8               // error: a bound
struct S<'a> { f: fn(&Self) -> &u8, g: &'a u8 } // good: the Referrer names the lifetime (see below)
struct S<'a, F: Fn(&Self) -> &u8>(&'a F);       // good: the same
```

Instead, name the lifetimes of the inputs and of the output, give the method a receiver, or make the path absolute.

- Nothing is added if the inputs keep the lifetimes the entry has: the only lifetime of the inputs is `'a` (`Fn(&'a Rel<T>) -> &u8`). Nor if the output does not rely on elision: it has no lifetime or names one (`Fn(&Rel<T>) -> Rel<T>`, `for<'x> Fn(&'x Rel<T>) -> &'x u8`). `Rel<T>` where `T` has no lifetime bound is leaked alone as usual.
- A function pointer or `Fn(..)` inside an input has its own signature: its lifetimes do not count for the one around, and the whole inner one is the entry if it adds lifetimes to the outer one. That is how rustc 1.64 and later count (checked: 1.64, 1.65, 1.70, stable and nightly pass, 1.63 fails). Older ones also count the lifetimes that the inner one does not bind, so the copy of `Fn(&Box<dyn for<'x> Fn(&'x Rel<T>) -> &'x u8>) -> &u8` does not compile there.
- The elided lifetimes inside a function pointer or `Fn(..)` are its own, so they do not stop it from being leaked as a whole. They stop the types inside it from being leaked alone: `Rel<&Rel2>` in `Fn(Rel<&Rel2>)` is an error, like any type with an elided lifetime.
- `Self` and `Self::A` in the inputs add lifetimes if the copy writes them as a type that has some. `Visitor::with_self()` writes `Self` of a struct, an enum or a union as the type itself (`::path::S<'a>`), and `Self::A` as `<::path::S<'a>>::A`. rustc counts no lifetime for `Self`, so `fn(&Self) -> &u8` is valid in `struct S<'a>`, and the plain copy `fn(&::path::S<'a>) -> &u8` would be E0106. The Referrer names the lifetime that elision gives instead, with a fresh binder, in the inputs and in the output: `for<'__tl_e0> fn(&'__tl_e0 ::path::S<'a>) -> &'__tl_e0 u8`, and the same for `Fn(..)` in a type, in a bound and in a where predicate. If the inputs name their lifetime (`for<'x> fn(&'x Self) -> &u8`), the output takes that name. So nothing is leaked, and these are fine in a bound and with `allow_self = false`. The lifetime that the inputs keep is not added (`fn(&'a Self) -> &u8` in `struct S<'a>`). The copy of an impl keeps `Self`, and so does that of a trait, whose impl of the Repeater writes `Self::A` as `<__S>::A` (`Referrer::iter_with_self()`), so they are not counted; a trait with a host type that has lifetimes (`with_self(Host<'h>)`) gets the binder like a struct. Where the Referrer cannot name the lifetime (a path hides it, or the signature is a method's, which has no binder), `Self` counts as a leaked type with lifetimes: the function pointer or trait object is leaked as a whole, with `Self` as written, since the impl of the Repeater for the type keeps it. That is an error with `allow_self = false`, in a bound, in a method, and in a trait object under an elided reference (see [Self rules](#self-rules)).
- A lifetime that a path hides (`Ref<u8>` for `Ref<'a, u8>`, `::core::slice::Iter<u8>`) is not seen by the Leaker. In the inputs it is taken as the lifetime that elision gives, when the output is elided and the inputs show no lifetime (see above). Write `Ref<'_, u8>` to make it visible, for the same result (and the Referrer can then name it). A hidden lifetime in the output (`fn(&Rel<T>) -> Ref<u8>`) is not seen: the copy fails if `Rel<T>` adds lifetimes, so write `Ref<'_, u8>` there.
- A function that is itself the analyzed item has no signature to the Leaker, which sees only the types you visit: write the lifetimes out.

## Trait objects

A `dyn Trait` without a lifetime bound gets its bound from its surroundings. Behind `&'a`, it is the lifetime of the reference, or the bound of the trait if it has one. As an argument of a path, it is a bound written in the definition (`Ref<'a, T: ?Sized + 'a>`). A bound that rustc only infers from a field does not count, and the default is `'static`. The Leaker cannot see these definitions, so it never leaks such a trait object alone. Writing the default out (`dyn RelTrait + 'a`) is rejected by rustc (E0478) for a trait with a bound of its own.

- If the trait object must be leaked itself, the whole reference or path is leaked, with the lifetime among its generics. That is the case for a relative trait (`&'a dyn RelTrait`) and for a bare identifier that may be a const. For a path, `'a` must be its only lifetime argument.
- Otherwise the trait object stays in the copy, and what is inside it is leaked (`Rel` of `&'a dyn ::path::Tr<Item = Rel>`).
- A lifetime bound you write changes this. With a lifetime of the Leaker, the whole reference is leaked. With a local lifetime, the trait object alone is leaked with its bound.
- These are errors: an elided lifetime (`&dyn RelTrait`, `Ref<'_, dyn RelTrait>`), several lifetime arguments, and a trait object in a bound or where predicate (`T: ::path::Tr<'a, dyn RelTrait>`). `Wrap<'static, dyn RelTrait>` and `Ref<'a, Box<dyn RelTrait>>` are fine.

```rust,ignore
fn f(x: &dyn RelTrait);          // error: elided lifetime
fn f<'a>(x: &'a dyn RelTrait);   // good: the whole reference is leaked
```

Instead, name the lifetime, make the trait path absolute, or write the bound: `&'a (dyn RelTrait + 'a)`.

## Bounds that name leaked types

A bound or where predicate is rewritten like any other type: `T: Fn(Rel)` becomes `T: Fn(<() as Repeater<..>>::Type)`. That is a problem if a path through the Repeater is in a predicate of the item and its impl needs a bound of the item that the environment has in another form. rustc builds the environment of the item from its predicates, and normalizes the path there, while the bounds are not normalized yet. The impl of the Repeater for `Wrap<T>` needs `T: Fn(Rel)`, and the environment has `T: Fn(<() as Repeater<..>>::Type)`: rustc cannot prove it (E0277 "expected an `Fn(Rel)` closure, found `T`", E0271, E0275 or E0284). The copy cannot have the original bound, the only link between the crates being the Repeater, so the Leaker reports an error. A type in a field or in a signature is fine, as it is normalized when the environment is built. The bounds of an `impl Trait` in an argument (`x: impl Into<Wrap<T>>` is `x: X` with `X: Into<Wrap<T>>`) are predicates of the function, and so are those of a generic. The bounds of one in the return type are not: rustc proves them where the body is.

```rust,ignore
struct S<T: Fn(Rel)>(Wrap<T>) where Wrap<T>: Sized;  // error: `Wrap<T>` is in a predicate, `T: Fn(Rel)` names `Rel`
struct S<T: Fn(Rel)>(Wrap<T>);                       // good
struct S<T: Fn(Rel)>(Wrap<T>) where T: Sized;        // good
```

A bound is *unstable* if the environment has it in another form than the impl that rustc proves needs it. That is so if it names:

- a leaked type (`T: Fn(Rel)`), which the copy writes as a path through the Repeater;
- `Self`, if `Self` is the self type of an impl and that is leaked (`X: Fn(Self) -> u8`): the copy writes the self type through the Repeater, and the impl of an entry has the bound with the self type of the Leaker;
- `Self::A` (`X: Fn(&Self::A) -> u8`): the impl of the Repeater for the implementors has `<__S as Tr>::A`, and the environment of the copy has `<Host as Tr>::A`, which rustc resolves by the impl of the trait;
- an associated type that rustc normalizes: the `Item` of a generic that an `Iterator<Item = ..>` bound or predicate in scope equates (`T: Fn(T::Item)` next to `T: Iterator<Item = u8>`, also in an enclosing item), the `Output` of `Fn(..)`, a projection of a type that an impl defines (`<Vec<u8> as IntoIterator>::Item`, `<u8 as Tr>::A`). The `Item` of a generic that nothing equates is rigid, and stable. A supertrait may fix it (`trait WIter: Iterator<Item = u8>`, then `T: WIter + Fn(T::Item)` is `T: WIter + Fn(u8)`), and the Leaker cannot read it, so a projection (`T::Item`, `<T as Iterator>::Item`) of a generic with a bound of a trait that is not one of the standard library counts as one that rustc normalizes. A bound of the standard library (`Iterator`, `Clone`, `Fn(..)`, any path in `::core`, `::std` or `::alloc`) does not. That rejects `T: Tr + Fn(T::Out)` with `trait Tr { type Out; }`, which would compile. A type alias that hides the projection is not seen.

It is an error for a type in a bound or where predicate of the item or of a method, in these cases:

- It is a leaked type (`where Wrap<T>: Sized`, `U: Into<Wrap<T>>`, `x: impl Into<Wrap<T>>`), and a bound of a generic that it uses is unstable. A type that names itself is one: `T: Into<Wrap<T>>`. A bound that is stable (`T: Clone`) is fine, and so is a predicate on another type than a generic (`Vec<Rel2<T>>: Clone`), which rustc proves by an impl. The generics of the type are those of its context, so a bound of the enclosing item counts only if the type uses the generic: `T: Fn(T::Item)` of an impl, and `T: Iterator<Item = Wrap<T>>` of its method.
- It is `Self::A`, an associated type of `Self`, and a bound of the generics of the trait (or of the type of `Self`) is unstable. The copy for a concrete implementor resolves it by the impl of the trait. In an impl, the impl of the trait has all the generics of the impl, not only those that `Self` uses: `impl<T> Tr<T> for u8 where T: Fn(Rel) { fn m<X>(&self) where X: Into<Self::A>; }` is an error, and so is `impl<T: Fn(Rel)> Tr<T> for u8` with `u8: Into<Self::A>`.
- It is `Self`, in an impl whose self type is leaked, and a bound of the generics of a type in the self type that the copy writes through the Repeater is unstable. The copy writes the self type through the Repeater, and `Self` stands for it: rustc normalizes it where the predicate names it, with the impls of the Repeater for those types, which need the bounds of the generics that they use, not those of the other generics of the impl. `impl<T: Fn(Rel2), V: AsRef<Self>> ::p::W<Rel<T>, T, V>` is an error, and `impl<T: Fn(Rel2), U, V: AsRef<Self>> ::p::W<T, Rel<U>, V>` is fine. The Leaker does not choose a type of the self type as an entry for its parts if its generics have an unstable bound: `impl<T: Fn(Rel2), U> Tr<U> for ::p::W<T, Rel<U>> where Self: Clone` has the entries `Rel2` and `Rel<U>`, not `::p::W<T, Rel<U>>`, whose bound `T: Fn(Rel2)` is unstable. It is an error only for a type that must be an entry.
- It is a leaked type that names `Self` of a trait, in a predicate of the trait itself (its where clause, a bound of its generics or a supertrait), not in one of a method: `trait Tr where Rel<Self>: Sized`. The Repeater's impl for it is for the implementors of the trait, so proving that the host of the copy implements the trait needs the predicate, which names the type.

The error names the type and the bound. Instead, write the type of the bound with an absolute path (`T: Fn(::my_crate::Rel)`), write the bound without what it names, or use the type outside of the predicates (in a field or a signature).

Two things are not errors. A type that is an entry only because its parts need it (`(Rel, T, Rel2)` in `T: Fn((Rel, T, Rel2))`) is not one when that makes a cycle: its parts are the entries. An optional entry (see [External crates](#external-crates)) in such a place is not one either, but copied as written, even if the Referrer would replace it. That holds for an optional entry that the bound names (`T: Fn(::ext::X)` and `Wrap<T>: Sized`), for the optional types of a self type that an unstable bound names, and for an optional entry whose own context is unstable (`&'a ::ext::W<T, ::ext::B0>` in `T: Fn(&'a ::ext::W<T, ::ext::B0>)`, where the predicate that names `'a` is part of its context).

## Method lifetimes

A leaked type may name a lifetime `'b` of a method: a path (`Wrap<'b>`), or a trait object that takes `'b` from the reference or path around it (`&'b dyn RelTrait`). In the trait, `'b` is late-bound: an argument uses it and no bound names it. In the copy, only `<Self as Repeater<..>>::Type` holds it, and that does not constrain it. So rustc makes it early-bound, and rejects the copy because the two differ (E0195). The Leaker reports an error if all of these hold. A type that is leaked as a whole names `'b`: a path with a relative name, or a trait object. The return type names `'b`, and no bound or where predicate does. Some argument constrains `'b` in the original, and none does in the copy, because the types leaked as a whole are projections there.

```rust,ignore
fn f<'b>(&self, x: &'b dyn RelTrait) -> &'b u8;           // error
fn f<'b>(&self, x: Wrap<'b>) -> &'b u8;                   // error: `Wrap` is relative
fn f<'b>(&'b self, x: &'b dyn RelTrait) -> &'b u8;        // good: `&'b self` keeps `'b`
fn f<'b>(&self, x: &'b dyn RelTrait, y: &'b u8) -> &'b u8; // good: `y` keeps `'b`
fn f<'b>(&self, x: Wrap<'b>, y: &'b u8) -> &'b u8;        // good: `y` keeps `'b`
```

Instead, keep `'b` anchored, make the trait or path absolute (`::my_crate::RelTrait`, `::my_crate::Wrap<'b>`), or write the bound (`&'b (dyn RelTrait + 'b)`). A bound on `'b` (`'c: 'b`, `T: 'b`, `where Self: 'b`) makes it early-bound in both, so it is fine. So is a function whose only mention of `'b` is the return type. A lifetime of the Leaker and one of the outermost function are not affected.

An argument keeps `'b` if it is the receiver (`&'b self`), or holds it in a type that is not leaked, however deep: a reference, tuple, array, slice, pointer, function pointer, or a path argument (`Option<&'b str>`). That includes an external type that the Leaker does not allow, which the Referrer leaves as written unless it replaces it. An associated type path (`T::A<'b>`) and an array length do not hold it, as rustc sees it.

A trait object that takes `'b` from the reference around it is more careful. A reference to a trait object keeps it only if the trait object is certainly not leaked as a whole. That is true for `&'b dyn ::core::fmt::Debug`, and for a trait of an allowed crate. It is not certain for a relative trait, a bare identifier that may be a const, an external type that is not allowed, a path of the analyzed module, and a type that may be leaked (`Vec<&'b u8>`).

## Bare identifiers and consts

A bare identifier argument (`MY_CONST` in `::path::MyType<MY_CONST>`) parses as a type, so it may be a type or a const. Its position decides. A const cannot be an entry on its own, so the type that holds it is leaked as a whole. If that is impossible, the identifier is leaked as a type, and the copy fails to compile if it is a const. This happens when the type has an elided lifetime or `impl Trait`, when its path is declared in the module of `intern_mod()`, and in a trait bound such as `T: ::path::MyTrait<MY_CONST>`. An identifier in a tuple, an array or a binding (`Item = X`) is a type.

```rust,ignore
fn f(x: ::path::MyType<'_, MY_CONST>);       // error: treated as a type
fn f(x: ::path::MyType<{ ::my_crate::MY_CONST }>);  // good: not leaked
```

Instead, write the const in braces with an absolute path.

- A const value can only be leaked as part of a type. A relative const in a trait bound is rejected, because a const cannot be recovered from an associated type without `generic_const_exprs`.
- A type in an expression (`[u8; size_of::<MyType>()]`) is not leaked alone. Its copy would put an associated type path in an anonymous constant. The type that holds it is leaked as a whole.
- Primitive names (`u8` in `const N: u8`) are copied as they are. This assumes that nothing shadows them (`type u8 = i32;`). Emit `Leaker::assertions()` in the Leaker's type context to check it at compile time.
- The type of a const parameter must be a primitive type or a path that does not depend on the context, such as an absolute path to an alias. `RepeaterSlots` cannot encode an alias.

## External crates

An absolute path such as `::serde::Wrap<(Rel,)>` names a crate by name only. It means the same type in the Referrer's crate only if that crate has the very same crate under that name. The Leaker cannot tell. So it leaks such a type as an optional entry (`Referrer::must()` is `false`), after the required ones. `Referrer::allow_ambiguous_crate_name()` is the Referrer's choice, and it is not serialized.

- `true` (the default): the optional entry is left as written, and the types in it are looked up as usual (`Rel` is replaced). Use it when both sides are built from the same dependency graph. A crate of the same name with another version or origin is taken for the same one, and the copy then fails or means another type.
- `false`: the optional entry is replaced like a required one, through the Repeater. An absolute type that contains a relative one (`::serde::Wrap<(Rel,)>`) is replaced as a whole. Use it if the Referrer's crate may have other dependencies than the Leaker's. The impl of an optional entry is needed only if some Referrer replaces it.

```rust,ignore
fn f(x: ::serde::Wrap<(Rel,)>);  // `Rel` is an entry. The whole type is an optional entry.
```

These absolute paths are not optional entries: `::core::..`, `$crate::..`, and paths that start with one of `LeakerConfig::allowed_paths` (`parse_quote!(::serde)` allows `::serde::..` and not `serde::..`). You promise that allowed crates are the same on both sides. `::std` and `::alloc` are like any other crate unless allowed. `Referrer::external_crates_require()` lists the crates an entry names.

Only types are optional entries. An absolute path in a bound, a value, the type of a const parameter, the type of a receiver (`self: ::std::boxed::Box<Self>`) and a type that cannot be leaked (`::serde::Wrap<'_>`) is copied as written, and that is not an error. The required entries are those of a Leaker that allows every crate, and they come first. Replacing a type can lose an anchor or a method lifetime, so they may differ in that case. An optional type inside a required one is not an entry. One that contains required ones is, and they are entries of their own.

## Self rules

A type that names `Self` is part of a context in which `Self` has a meaning: the implementor of a trait (`SelfRequirement::Bound`), or a type (`SelfRequirement::Type`). Bounds and where predicates of the generics may name `Self` too. The Repeater impl is written for what `Self` is, so `Referrer::context_self()` tells which form to use.

```rust,ignore
// `Rel<Self>` in `trait MyLeaker<T, U>`: for the implementors of the trait
impl<T, U, __S: ?Sized + MyLeaker<T, U>> path::Repeater<(T, U), 42, 2> for __S {
    type Type = path::Rel<__S>;
}
// `Rel<Self>` in `struct MyStruct<T>`: for the type
impl<T> path::Repeater<(T,), 42, 3> for MyStruct<T> {
    type Type = path::Rel<Self>;
}
```

- In the impl for the implementor, `Self` is `__S`. Take the type and generics of a `Bound` entry from `Referrer::iter_with_self()` with `__S`. `Self::Item` becomes `<__S>::Item`, and `Self: Sized` becomes `__S: Sized`.
- The copy uses `<Self as Repeater<..>>::Type`, where `Self` is the implementor. So the impl in the Referrer's crate must provide the associated types the type uses (`type Item = u8;`).
- With `allow_self = true`, a type that names `Self` is one entry, written for the implementor or the type.
- `Self::A` cannot be in an entry if `Self` is a type (an impl, a struct, an enum or a union): the Repeater's impl for the type cannot write it, because it has no trait to take `A` from (E0220). So a type that names it is never leaked whole, only its parts are, and that is an error where it must be (`Rel<Self::A>`). The same holds if a bound of the generics of an entry names it. Write the trait, `<Self as ::path::Trait>::A`. A type that needs no leaking is copied as written, so `Vec<Self::A>` is fine, and so is `Self::A` in a trait, whose Repeater impl is for the implementor.
- With `allow_self = false`, such a type is split into parts. That is an error if it cannot be (`Rel<Self>`, `Self::Item`), or if the bounds of the generics it uses name `Self`. Use it if the Repeater impls are for a marker type, which cannot name `Self`. `Visitor::with_self()` then replaces `Self` in copies in the same way. A type that replaces `Self` is parenthesized where its precedence matters: as the bounded type of a where predicate (`(&fn(u8)): 'a`) and as the pointee of a reference or a pointer (`&(dyn Tr + Send)`).
- `Self` has no meaning in `enter_generics()`, `enter_mod()` and the generics of a scope. Naming it there is an error.
- The self type of an impl is written by the copy, and `Self` stands for it, so an entry in it cannot depend on `Self`. It does if a bound that the impl declares names `Self` and the entry uses that generic. `impl<T> S<T> where T: AsRef<Option<Self>>` has the entry `S<T>`, whose context has that bound for `Self` = `S<T>`. The Repeater impl of the entry is for `Self`, so the copy would write `<S<T> as Repeater<..>>::Type` where `S<T>` is the very path it is writing: the Referrer has no type to write for it, whatever its macro does. The same holds for a type in the self type (`impl<T: AsRef<Self>> ::path::W<Rel<T>>`). A generic that no entry in the self type uses may have the bound (`impl<T: AsRef<Self>, U> ::path::W<T, Rel<U>>`). It is an error for `intern_impl()` and for an impl that `View::visit_item()` or `intern_mod()` visits outside of another item. (The generics of an impl nested in an item are local, and nothing that uses them is leaked.) The headline is "`S<T>` is in the self type of the impl, and a bound of its generics names `Self`". Rewriting the bound for the self type does not help: it would name the leaked entry in its own bound ([Bounds that name leaked types](#bounds-that-name-leaked-types)). Write the self type with an absolute path (`::my_crate::S<T>`), or the bounds without `Self`. An optional entry in the self type (`::serde::W<T>`) is copied as written, a strict Referrer too. Not affected: a bound of an outer scope that names its own `Self`, a bound or predicate of a method, and `enter_impl()`, whose caller says what the self type is and visits only fragments.
- The bounds and where predicates of a generic are written for the `Self` of the scope that declares it. A type that uses the generic where `Self` is something else (in an impl, a struct, an enum, a union or a trait nested in the scope) gets them rewritten for that `Self`, and its entry does not depend on the other one: `T: ::p::Tr<Self>` of `struct S<T: ::p::Tr<Self>>` is `T: ::p::Tr<S<T>>` in `impl W { fn f(x: Rel<T>); }`. That fails with an error if `Self` is the implementor of a trait (it has no name), or if the bound names an associated item of `Self` (`Self::X`, ambiguous for a type). The headline of the error is about the bound, not the path of the type ("the bound of `T` mentions a `Self` which has no name here"), and `CheckResult::headline` has it for `View::check()`. The scope that declares the bound records its `Self` in its top-level context, so `Referrer::visitor(context, ..)` knows it. The copy must nest the same scopes, as the Leaker did: the Referrer takes `Self` of a nested struct, enum, union, trait or impl from the item it visits, and computes the same contexts. Where `Self` is the same, nothing is rewritten, except for the default of a type parameter of a scope: it has no `Self`, so the bounds around it are rewritten for the `Self` that declared them (or fail) whatever `Self` the scope has. `Referrer::visitor(context, ..)` does the same for the defaults of the scopes it visits. A nested function, type alias, trait alias, const or static has no `Self` of its own, in the Leaker and in the Referrer.
- `Self: 'a` is encoded by the `*_with_self()` encoders, with `Referrer::context_self_outlives()`. The impl does not declare it. See [Outlives bounds](#outlives-bounds).

# MSRV

The library supports Rust 1.56 (`rust-version` in `Cargo.toml`). Check it with `cargo build --lib`. The tests and the test crates need a newer Rust.

The current dependencies do not build on Rust 1.56. `syn` 2.0.87 needs Rust 1.61. The latest `num-traits`, which `safegraph` uses, needs 1.60. `unicode-width` 0.1.14 needs 1.66 without saying so. Cargo of that age cannot pick versions by `rust-version`, so generate the lockfile with a recent Cargo. `Cargo.lock` is not part of the repository or the package. Cargo 1.56 reads lockfile versions up to 3. If yours says `version = 4`, change it to `3`.

The simplest way, which CI uses (see `.github/workflows/rust.yml`), is a Cargo 1.84 or later that resolves by `rust-version`, with `unicode-width` pinned:

```sh
CARGO_RESOLVER_INCOMPATIBLE_RUST_VERSIONS=fallback cargo +stable generate-lockfile
cargo +stable update -p unicode-width --precise 0.1.11
cargo +1.56 build --lib
```

Or pin these versions by hand. They are known to build with Rust 1.56.

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

The other packages are the latest compatible ones: `safegraph` 0.3.0, `template-quote` 0.4.4, `autocfg` 1.5.1, `version_check` 0.9.5, `proc-macro-error` 1.0.4 and the `syn` 1.0.109 it uses.

# Application

- [sumtype](https://crates.io/crates/sumtype)
- [flat_enum](https://crates.io/crates/flat_enum)
- [parametrized](https://crates.io/crates/parametrized)

The exact rules of the analysis, one section for each topic.

Each section gives the rules, a short example, and what to write instead. The [crate
documentation](crate) is the user guide: start there. It defines the terms used here: the *Leaker*
(analysis) and *Referrer* (copy) sides, *leaked types* and *entries*, the *Repeater*, and the
*context* of an entry ([How it works](crate#how-it-works)).

# Encoding of the generics

The Repeater has one type argument, `T`, for the generics of an entry. An impl declares only the generics its type uses: `MyWrap<T>` uses `T`, and `MyType` uses none.

- A lifetime `'a` is encoded as `&'a ()`, and a type `T` as `T`.
- A type `T: ?Sized` is encoded as `*const T`, because only the last element of a tuple may be unsized.
- A const parameter `const N: usize` is encoded as `[(); N]`. See [Const generics](#const-generics) for other types.
- The lifetime bounds follow the parameters: `T: 'a` as `&'a T`, `'b: 'a` as `&'a &'b ()`, and `Self: 'a` as `&'a Self` ([Outlives bounds](#outlives-bounds)).
- The impl must not declare a bound that the encoding implies. Use `Referrer::impl_generics()` for the generics of a context, or `without_encoded_bounds()` for other generics.
- A generic that the copy declares itself, such as one of a method, is passed by its own name.
- Encode the generics of each entry. Do not encode the generics of the top-level context for every entry: they have only the bounds the trait writes, so rustc rejects them (E0309) for an entry such as `&'a Rel<T>`.
- If the copy omits an argument that has a default, complete the arguments with `Referrer::complete_args()` first. A default is analyzed with the parameters before it, so it cannot name a later one or `Self`.
- For the arguments of the trait itself (`u8` of `impl MyLeaker<u8>`), use `Referrer::context_generics(top_level)` and `Referrer::context_generic_args()`. The generics of a context are sorted, so they are matched by name and not by position.
- `encode_generics_args_to_ty()` is the counterpart of `encode_generics_params_to_ty()` for arguments. It is for generics that have no const parameter or only `usize` ones.

`Referrer::repeater_impl()` and `Referrer::entry_args()` follow these rules, so a macro that uses them cannot disagree with itself. The example below does the same by hand with the encoders.

```rust
use syn::visit_mut::VisitMut;
use syn::{parse_quote, GenericArgument, ItemImpl, ItemTrait, Signature};
use type_leak::{Leaker, RepeaterSlots};

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

// One Repeater impl per entry. It declares `impl_generics()` and encodes the generics the entry
// uses in its trait arguments.
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

// The copy of `f` in `impl MyLeaker<u8> for ..`. Each leaked type becomes a path through the
// Repeater, with its generics as arguments.
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

# Const generics

A const parameter is treated like a type parameter. The Repeater needs a type that carries its value.

- `encode_generics_params_to_ty()` encodes `const N: usize` as `[(); N]`.
- For other primitive types, define the Repeater with `repeater_definition()`. It adds a defaulted const slot for each primitive type, and `RepeaterSlots` encodes values into them. A `u64` value uses the `ID` slot.
- The generated Repeater needs Rust 1.59 and the 2018 edition. `type-leak` itself supports 1.56.
- Without the slots, the Repeater works if every const parameter is `usize` or `u64`.
- `repeater_impl_with()` and `entry_args_with()` take a closure for the const parameters, such as `|value, param| slots.encode_const(value, &param.ty)`.

# Contexts

The context of an entry decides which Repeater impl it needs. A `ContextId` names a context. Entries with the same generics, `Self` and module share one.

What a context holds:

```rust
use syn::{parse_quote, ItemTrait};
use type_leak::Leaker;

let item: ItemTrait = parse_quote! {
    trait Tr<T, U> where U: ::core::convert::Into<T> {
        fn f(&self) -> Rel;
        fn g(&self) -> Wrap<T>;
        fn h(&self) -> Wrap<U>;
    }
};
let mut leaker = Leaker::new();
leaker.intern_trait(&item, true).unwrap();
let referrer = leaker.finish();
// `Rel` uses no generic. `Wrap<T>` uses `T`. `Wrap<U>` uses `U` and, through the bound
// `U: Into<T>`, `T`.
let used: Vec<usize> = (0..3).map(|ix| referrer.generics(ix).params.len()).collect();
assert_eq!(used, [0, 1, 2]);
```

- The generics are only those the type uses. They are the generics the type names, the generics that the bounds of those generics name, and, if the type names `Self`, the generics in the description of `Self`.
- A where predicate on one generic (`U: Into<T>`) is a bound of that generic, as if it was written inline on `U`. It is in the context of what uses `U`, not of what uses `T`.
- A where predicate on another type (`Vec<T>: Clone`) belongs to each generic it names.

When two contexts are equal:

- Equal declarations give equal contexts. Parameters are sorted by kind and name. Bounds are sorted and moved from the where clause to their parameter. Defaults are removed.
- The output type of the sugar of `Fn` is parenthesized where a `+` would join it (`T: Fn() -> (&dyn Tr) + Clone`), whichever way the source wrote it.
- The same type found by several scopes is one entry if the contexts are equal, whatever the other generics of the scopes are.
- Scopes with the same generics but another `Self` return other ids.

How contexts are numbered:

- The Leaker numbers a context when it first appears, from 0. It numbers a module when it is entered, a nested scope when it is left, and the types of the outermost scope when it is left.
- It never renumbers them, except for the contexts of a scope that fails and is rolled back. So `Referrer::contexts()` can have gaps.
- There is always at least one context. With no other, it is context 0 of the types that use no generics.

The top-level context:

- It is the context of the generics of the item or scope itself.
- It differs from the context of an entry if the entry uses fewer generics, as `Rel` does in `trait Tr<T> { fn f(&self) -> Rel; }`. The Referrer keeps the top-level context even if it has no entry.
- It records what `Self` is if an entry names it, or has a bound rewritten for it ([Self rules](#self-rules)): `Referrer::context_self()`.
- It records the module the scope was entered in: `Referrer::context_module()`.
- For the trait that `intern_trait()` analyzes, if an entry names `Self`, it also records the generic associated types that have outlives bounds (`type G<'b> where Self: 'b;`). So `Referrer::visitor()` finds the entries that name them in a signature alone, as in the trait ([Outlives bounds](#outlives-bounds)).
- So `Referrer::visitor(context, ..)` knows the generics, `Self` and module of the copy from the id alone.

How the Referrer finds an entry:

- A type is replaced if its context, computed from the generics in scope, is the context of an entry. The generics in scope are those of `context`, then those that the visited syntax declares.
- A name that neither declares is not a generic, so it is not matched. The Referrer does not guess which generics are the Leaker's.
- If no entry has exactly the context, an entry that has bounds on `Self` alone besides (`where Self: Sized`) is taken. A copy where `Self` is a type parameter (`Visitor::with_self()`) cannot write them.
- An outermost item that declares exactly the generics of `context` (the copy of the item itself) does not declare them again.
- If the copy declares them with other bounds (`impl<T: Clone> MyLeaker<T> for ..`), call `Visitor::with_outermost_generics_as_leakers()`.
- `Referrer::expand()` knows only the generics of `context`. It does not find a type that names a generic of a method: visit the signature.

# Which types become entries

`Leaker::finish()` calls `Leaker::reduce_roots()`. It picks the fewest entries from which every leaked type can be built, because each entry needs an impl. That is a minimum vertex cut of the graph of the types, and the result is deterministic. So `Rel` alone covers `Vec<(Rel,)>`, `Option<(Rel,)>` and `Box<(Rel,)>`.

Some types are always entries as a whole. They are never replaced by their parts:

- a reference with a lifetime around an entry (`&'a Rel`);
- a path with a lifetime argument;
- a path with a bare identifier argument (`Wrap<SIZE>`);
- a type with a type in an expression (`[u8; size_of::<Big>()]`);
- a trait object without a lifetime bound inside such a reference or path ([Trait objects](#trait-objects)).

# Scopes

`intern_*` analyzes a whole item. To analyze a fragment whose generics and `Self` the caller knows, enter a scope: `Leaker::enter_impl()`, `enter_type()`, `enter_trait()`, `enter_generics()` or `enter_mod()`. The closure gets a `View`, and the scope returns its top-level `ContextId` and the value of the closure.

A `View` has `visit_item()`, `visit_type()`, `visit_path_arguments()`, `visit_where_clause()` and `visit_type_param_bound()`. It has the same `enter_*` methods for nested scopes, and `check()`.

- The generics of a scope are the generics of the Leaker. In a nested scope they are local generics, and the generics around it stay in scope.
- `Self` is what the scope declares. In `enter_generics()` and `enter_mod()` it has no meaning, so naming it is an error. An item visited with `visit_item()` has a `Self` of its own.
- Scopes with equal generics return the same id, if they are in the same module and, when an entry names `Self`, declare the same `Self`.
- The generics of a scope are not analyzed. To copy their bounds or where clause, visit them with `visit_type_param_bound()` and `visit_where_clause()`.
- The default of a type parameter is analyzed with the earlier parameters as the generics. The defaults of const parameters are analyzed with the syntax that visits them.
- A function that is itself the analyzed item has no local generics. Enter its generics with `enter_generics(&sig.generics, ..)`, and visit its input and output types and its where clause with `visit_type()` and `visit_where_clause()`.
- `View` implements `syn::visit::Visit`, so a `syn::visit` walk of an item analyzes it. `Visit::visit_signature()` is a method's signature, with local generics and the check of [Method lifetimes](#method-lifetimes). A method of an inherent impl that is walked as a whole does not get that check.
- `Visit::visit_path()` at the root analyzes only the arguments of a path and does not check the path. To analyze a trait bound with its root checked, use `visit_type_param_bound()`.
- `check()` finds what `visit_type()` would find in a type here, and records nothing. It reports the causes that force the type to be leaked, those that forbid it, and whether it is an optional entry. It uses the generics around, `Self` and the module.

A scope is all-or-nothing:

- If the syntax it visits has an error, it rolls back everything it recorded, including its inner scopes. It returns its own errors with spans, in order.
- It never returns the errors of a failed inner scope, with or without `?`.
- Ids returned earlier stay valid. The scope around it can go on.
- A panic in the closure rolls the scope back too. An outer scope may catch it with `catch_unwind()` and go on.

# Modules

`Leaker::intern_mod()` and `Leaker::enter_mod(hint, declared, ..)` analyze a module. The items in it are nested items, so their generics are local. `hint` is only for debugging. Module ids come from a counter, so equal calls give equal ids.

Declared names:

- `declared` lists the names declared in the module. Give them first.
- They are copied with the module, so they are not leaked.
- A name counts as declared if it is declared directly (unless a generic parameter shadows it), if it is named by `self::` or `super::` without leaving the analyzed module, or if it is imported with an absolute path.
- `use super::X` and `use crate::X` are leaked.

Entries of a module:

- An entry is keyed by its module in these cases. Its path is relative. It names an item that the module declares (`crate::base::B1<R0>` with `R0` declared in the module: `crate::base::B1` cannot be leaked alone). A path in what `Self` means, or in a bound of its generics, is neither absolute nor starts with `crate::` or `$crate::`. It is a type macro.
- Other entries are shared by all modules.
- Emit the Repeater impl of a keyed entry inside its module. `Referrer::context_module()` (`RepeaterImpl::module`) tells which module it is. For `intern_mod()`, follow `Referrer::module_path()` through the `mod` items you analyzed to find it. A module that `enter_mod()` made is the one you entered.
- `Referrer::module_path()` gives the path of a module relative to the module of `intern_mod()` (`[]` for that module). A module found by `visit_item()` has such a path. A module entered by `enter_mod()` has no path, only its hint (`Referrer::module_hint()`).
- The impl is in the Leaker's module. So an entry that names a declared item (`R0` above) names the Leaker's item, not the one that the copy of the module declares again.

Rewriting a copy of a module, with `Referrer::visitor(context, ..)`:

- An entry with no module matches in every module. An entry of a module matches only there.
- The visited syntax is inside the module of `context`. That is `context` itself if it is a module, and else `Referrer::context_module(context)`.
- A `mod` item is a module nested in it, found by name, never the module itself. If the Leaker saw no such module, it counts as a module no entry depends on, so only entries with no module match in it.
- To rewrite a copy of a whole module, visit its items, or a `File` of them. A `File` hides the entries of no module that name what it declares, as a module does. Items visited one by one do not know their siblings.
- A context with no module (`intern_trait()`, `enter_generics()`) matches only entries with no module. A `mod` item there is a module found with no module around it, which is found by its name.
- The module of a scope entered inside a module is that module, so a copy in the scope is looked up there. The top-level context of a nested scope has only the generics of that scope, not those around it.
- Two `intern_mod()` calls return the ids of two modules. Rewrite the copy of each with its own id.

# Self rules

A type that names `Self` belongs to a context where `Self` has a meaning: the implementor of a trait (`SelfRequirement::Bound`), or a type (`SelfRequirement::Type`). The bounds and where predicates of its generics may name `Self` too. The Repeater impl is written for what `Self` is, so `Referrer::context_self()` tells which form to use.

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

With `allow_self = true`:

- A type that names `Self` is one entry, written for the implementor or the type.
- In the impl for the implementor, `Self` is `__S`. Take the type and generics of a `Bound` entry from `Referrer::iter_with_self()` with `__S` (`Referrer::repeater_impl()` does it). `Self::Item` becomes `<__S>::Item`, and `Self: Sized` becomes `__S: Sized`.
- The copy uses `<Self as Repeater<..>>::Type`, where `Self` is the implementor. So the impl in the other crate must provide the associated types that the type uses (`type Item = u8;`).
- `Self::A` cannot be in an entry if `Self` is a type (an impl, a struct, an enum or a union). The Repeater impl for the type has no trait to take `A` from (E0220). So a type that names it is never leaked whole, only its parts are. That is an error where the whole type must be leaked (`Rel<Self::A>`). The same holds if a bound of the generics of an entry names it.
- Instead, name the trait: `<Self as ::path::Trait>::A`. A type that needs no leaking is copied as written, so `Vec<Self::A>` is fine. `Self::A` in a trait is fine too, because the Repeater impl of a trait is for the implementor.

With `allow_self = false`:

- A type that names `Self` is split into parts. That is an error if it cannot be (`Rel<Self>`, `Self::Item`), or if the bounds of the generics it uses name `Self`.
- Use it if the Repeater impls are for a marker type, which cannot name `Self`. `Visitor::with_self()` then replaces `Self` in copies in the same way.
- A type that replaces `Self` is parenthesized where its precedence matters: as the bounded type of a where predicate (`(&fn(u8)): 'a`), and as the pointee of a reference or a pointer (`&(dyn Tr + Send)`).

Where `Self` has no meaning:

- `Self` has no meaning in `enter_generics()`, `enter_mod()` and the generics of a scope. Naming it there is an error (`NotInternableError::SelfWithoutMeaning`).

The self type of an impl:

- The copy of an impl writes the self type, and `Self` stands for it. So an entry in the self type must not depend on `Self`. It does if a bound that the impl declares names `Self` and the entry uses that generic.
- `impl<T> S<T> where T: AsRef<Option<Self>>` has the entry `S<T>`, whose context has that bound for `Self` = `S<T>`. The Repeater impl of the entry is for `Self`, so the copy would write `<S<T> as Repeater<..>>::Type` where `S<T>` is the very path it is writing. The Referrer has no type to write for it, whatever its macro does.
- The same holds for a type in the self type (`impl<T: AsRef<Self>> ::path::W<Rel<T>>`). A generic that no entry in the self type uses may have the bound (`impl<T: AsRef<Self>, U> ::path::W<T, Rel<U>>`).
- It is an error for `intern_impl()`, and for an impl that `View::visit_item()` or `intern_mod()` visits outside of another item. The headline is "`S<T>` is in the self type of the impl, and a bound of its generics names `Self`". The generics of an impl nested in an item are local, and nothing that uses them is leaked.
- Rewriting the bound for the self type does not help: it would name the leaked entry in its own bound ([Bounds that name leaked types](#bounds-that-name-leaked-types)).
- Instead, write the self type with an absolute path (`::my_crate::S<T>`), or the bounds without `Self`.
- An optional entry in the self type (`::serde::W<T>`) is copied as written, also by a strict Referrer.
- Not affected: a bound of an outer scope that names its own `Self`, a bound or predicate of a method, and `enter_impl()`, whose caller says what the self type is and visits only fragments.

Bounds written for another `Self`:

- The bounds and where predicates of a generic are written for the `Self` of the scope that declares it.
- A type that uses the generic where `Self` is something else (in an impl, a struct, an enum, a union or a trait nested in the scope) gets them rewritten for that `Self`, and its entry does not depend on the other one. `T: ::p::Tr<Self>` of `struct S<T: ::p::Tr<Self>>` is `T: ::p::Tr<S<T>>` in `impl W { fn f(x: Rel<T>); }`.
- That is an error if `Self` is the implementor of a trait (it has no name), or if the bound names an associated item of `Self` (`Self::X`, which is ambiguous for a type). The headline is about the bound, not about the path of the type: "the bound of `T` mentions a `Self` which has no name here". `CheckResult::headline` has it for `View::check()`.
- The scope that declares the bound records its `Self` in its top-level context, so `Referrer::visitor(context, ..)` knows it.
- The copy must nest the same scopes as the Leaker did. The Referrer takes `Self` of a nested struct, enum, union, trait or impl from the item it visits, and computes the same contexts.
- Where `Self` is the same, nothing is rewritten. The exception is the default of a type parameter of a scope: it has no `Self`, so the bounds around it are rewritten for the `Self` that declared them (or fail), whatever `Self` the scope has. `Referrer::visitor(context, ..)` does the same for the defaults of the scopes it visits.
- A nested function, type alias, trait alias, const or static has no `Self` of its own, in the Leaker and in the Referrer.

The own `Self` of a nested item:

- A struct, enum, union, trait or impl that `View::visit_item()` or `intern_mod()` visits has a `Self` of its own, and the Leaker records the entries that depend on it, whatever `allow_self` the scope around has (`Rel2<Self>` in `struct S<T> { c: Vec<Rel2<Self>> }`, or `Rel<T>` where a bound of `T` names `Self`). The `Self` of the scope around is hidden in the item.
- The Repeater impl of such an entry is for the Leaker's own type (`impl<T> Repeater<(T,), ID, IX> for S<T>`, in the module of the Leaker's crate). The copy declares a `S` of its own, so it cannot name that type as `Self`.
- So `Visitor::with_nested_self(|nested, leaked, ix| ..)` writes the replacement of these entries. `nested` (`NestedSelf`) is the item: what `Self` is (`Referrer::context_self()` of the entries), its name, and the `mod` items around it. The macro knows where the item is in the Leaker's crate, and writes `<ITS_TYPE as Repeater<ARGS, ID, IX>>::Type`. `ARGS` is `Referrer::entry_args()` at a `UseSite` with `with_self(ITS_TYPE)`. For a nested trait, `Self` is the type that implements the Leaker's trait (a host).
- The visitor finds these entries the way the Leaker did: it keeps a layer for each nested item, with the `Self` of the item, and computes the same contexts. An entry of the `Self` of the context, or of a scope around that the nested item hides, is not found in it.
- Without `with_nested_self()` they are not found, whatever `type_fn` is, and the copy keeps `Rel2<Self>`. It does not compile in the other crate.
- A nested item allows its `Self` even if the scope around has `allow_self = false`. The scope's own `Self` is denied as usual, so no entry names it, and the visitor finds none.
- The copy must nest the items as the Leaker did. For a scope that is not a module (`enter_generics()`, `enter_type()`..), an item is nested in a function body of the copy; the outermost item of the copy is the scope itself.

`Self: 'a`:

- It is encoded by the `*_with_self()` encoders, with `Referrer::context_self_outlives()`. The impl does not declare it. See [Outlives bounds](#outlives-bounds).

# Outlives bounds

A reference `&'a X` is valid only if `X: 'a`. So an entry under a reference needs that bound in its Repeater impl. The Leaker handles it like this:

- It records the bound for an entry, and the encoded arguments carry it: `T: 'a` as `&'a T`, `'b: 'a` as `&'a &'b ()`, and `Self: 'a` as `&'a Self`. The predicates `Self::A: 'a`, `<Self as Tr>::A: 'a` and `T::A: 'a` are where predicates of the context.
- It records the bounds you write.
- It cannot read the definition of a path. So a path with a lifetime argument records nothing, even if its definition needs `T: 'a`. The generic associated types of the trait that `intern_trait()` analyzes are the exception (see below). A guessed bound would make the copy stricter than the original for a path that needs none.
- Without the bound, rustc rejects the Repeater impl (E0309).

```rust,ignore
fn f<'a, T>(x: Rel<'a, T>);      // rejected: `struct Rel<'a, T>(&'a T)` needs `T: 'a`
fn f<'a, T: 'a>(x: Rel<'a, T>);  // good
```

Instead, write the bound: `T: 'a`, or `where Self: 'a` for `Rel<'a, Self>`. A struct that holds a leaked type under such a path writes it too: `struct F<'a, T: 'a>(Ref<'a, Rel2<T>>)`. The copy must write the same bound, or the entry is not found.

Which lifetimes count:

- The lifetimes around a type count. `Rel2<T>` in `&'a Vec<Rel2<T>>` is an entry with the generics `<'a, T: 'a>`.
- A reference, tuple, array, slice, pointer, path argument, function pointer or trait object passes the lifetime on. An associated type path (`<X as Tr>::A`, `T::A<X>`), `impl Trait` and an array length do not.
- The same type under another lifetime is another entry.
- rustc infers `T: 'a` for the original `struct F<'a, T>(&'a Vec<Rel2<T>>)`, but not for a copy that uses `<S as Repeater<..>>::Type` (E0309). So give `expand()` and `visitor()` the whole type or item, not a type cut out of another.
- A trait object argument counts too, also if the object has its own bound. `&'a dyn Iterator<Item = Rel2<T>>` needs `T: 'a`.
- Only a lifetime parameter counts as `'a`. `'static`, `'_` and the lifetime of a `for<..>` binder are never recorded, and `T: 'static` is not encoded.
- A binder around the type does not stop the lifetimes outside it: `&'a for<'x> fn(&'x X)` needs `X: 'a`.
- A type with `'static` needs no bound. The original already declares `T: 'static`, because rustc infers none for a struct, and so must the copy.
- The value of an associated type or GAT, such as `type A<'a, T> = Ref<'a, Rel2<T>>;`, is rejected like the original. Write `type A<'a, T: 'a>`.

`Self: 'a`:

- `where Self: 'a` and `where Self: Sized + 'a` count as `Self: 'a`.
- Only a bound by a lifetime parameter moves into `Referrer::context_self_outlives()`. `Self: 'static`, a bound under `for<..>` and `Self::Item: 'a` stay where predicates.
- A `Self: 'a` that only the where clause of the trait writes is not in the copy of a method.
- For a lifetime that the copy declares (`impl<'a> MyLeaker<'a> for ..`), write `where Self: 'a` in the copy again.

## Associated types of `Self`

This is about `&'a Self::A`, `&'a <Self as Tr<'b, U>>::A` and `&'a Self::A<'b>`: an associated type of the implementor of a trait. `Self` is a type parameter here, as `T` is in the next section.

- The Leaker records the where predicate `Self::A: 'a` as it is written (`<Self as Tr<'b, U>>::A: 'a`, `Self::A<'b>: 'a`), as if you wrote it. It does not record `Self: 'a`, nor `'b: 'a` or `U: 'a` for the arguments of the trait.
- The original gets the predicate, and nothing of `Self` or of the arguments, from the well-formedness of its signature (`fn f<'a>(&'a self, x: Rel<&'a Self::A>)`). The Repeater's impl does not have that.
- A copy that asked for more would be stricter than the original, and rustc rejects it. `&'a &'b ()` for `'b: 'a` is E0491 (E0477 on nightly) in an impl for a host type. `&'a T` is E0309. So is `&'a Gen<H>` in an impl for a generic host whose `H::Item` the original needs (E0309 on `H`).
- `Self: 'a` is recorded where the type mentions `Self` apart from the associated type (`&'a (Self, Self::A)`), and where a where clause writes it (`where Self: 'a`).
- The predicate is in the impl of the Repeater (`where <__S as Tr<'b, U>>::A: 'a`, `where <__S>::A: 'a`). In the copy it follows from the normalized type.
- rustc reads `where <` as generics of the where clause. So write `<__S>::A: 'a` after another predicate, or parenthesize it: `(<__S>::A): 'a`. `Referrer::repeater_impl()` does it for you.
- If a where clause of the context equates the associated type (`where Self: Iterator<Item = X>`), rustc normalizes the path to `X`. The Leaker records what `X` mentions, as for a generic below (`T: 'a` for `Item = (Vec<T>, u8)`, nothing for `Item = u8`), and no predicate.
- A supertrait that equates it (`trait Tr<T>: Iterator<Item = (Vec<T>, u8)>`) is not seen. The copy of an impl has no supertraits, so the context of the Referrer could not be the same, and the Repeater's impl is E0309 for `T`. Write `T: 'a` on the trait: it is recorded, and the impl compiles.

## Associated types of a generic

This is about `&'a T::A` and `&'a <T as Tr<'b>>::A`: an associated type of a generic (`T` is a type parameter in scope).

- The Leaker records the where predicate `T::A: 'a`, as it does for `Self::A`. It records no `T: 'a` and no bound for the arguments of the trait. It is also the predicate that `where T::A: 'a` writes.
- The original gets `T::A: 'a`, and not `T: 'a`, from the well-formedness of its signature (`trait Tr<'a, T: Iterator> { fn f(&self, x: Rel<&'a T::Item>); }`). So an impl of the trait for another type, whose signature has the encoded `T: 'a`, would not compile (E0309).
- `T::A<'b>` and a path on a projection (`<T::A as Tr2>::B`) count too.
- The rest of a type is as before: `T` of `&'a (T, T::A)` still records `T: 'a`.
- If a bound of the context equates the associated type (`T: Iterator<Item = X>`, or `F: Fn(A) -> B` for `F::Output`), rustc normalizes the path to `X`. What `&'a T::Item` requires is then that of `X`. The Leaker records what `X` mentions (`T: 'a` for `Item = (Wrap<T>, u8)`, `U: 'a` for `Item = U`, nothing for `Item = u8`) and no predicate. The environment of the Repeater's impl would have the predicate as `X: 'a`, not as the bounds of the parts of `X`.
- These cases record the parts of the type instead of the predicate: a path that names a lifetime that is no generic of the context (the one of a `for<..>` binder, `'_`, an elided one), `impl Trait`, a function pointer or `Fn(..)`, and an associated type of a type that is no generic (`<Vec<T> as IntoIterator>::Item`, which rustc normalizes).
- A struct that holds the copy must write the predicate too: `struct S<'a, T: Iterator>(Rel<&'a T::Item>) where T::Item: 'a`. rustc infers it for the original, but not through the projection of the Repeater (E0309).

## Generic associated types of the analyzed trait

`Leaker::intern_trait()` reads the generic associated types (GATs) of the trait with their outlives bounds: `type G<'b> where Self: 'b;`, `type G<'b, U: 'b>;`, `where 'a: 'b`, `where Self::A: 'b`.

- A projection that names a GAT (`Self::G<'x>`, `<Self as Tr<..>>::G<'x>`), under a reference or not, requires those bounds. The parameters of the GAT take the arguments of the projection, and those of the trait take the arguments of the trait in a qualified path.
- `Self::G<'x>` with `where Self: 'b` records `Self: 'x` (encoded as `&'x Self`). `T: 'b` records `T: 'x`. `'a: 'b` records `'a: 'x` (`&'x &'a ()`). `Self::A: 'b` records the where predicate `Self::A: 'x`.
- These come in addition to the predicate `Self::G<'x>: 'x` that a reference around the projection records. They are recorded as if the method wrote them.
- The original gets them from the well-formedness of its signature (`fn m<'x>(&'x self, a: Rel<&'x Self::G<'x>>)`). The Repeater's impl does not have that, and without them it is E0309 on `Self`.
- An argument that is no lifetime of the context (`'static`, `'_`, the lifetime of a `for<..>` binder) records nothing.
- A where clause that is no outlives (`where Self: Clone`) is for the method to write, as rustc requires.
- `Leaker::enter_trait()` does not see the items of the trait, so it reads no GAT.
- The copy has to prove what is recorded, as for a bound you write. In an impl for another type, whose `G<'x>` rustc normalizes to a type that does not name them, `T: 'x` and `'a: 'x` need `where T: 'x` and `where 'a: 'x` on the method (E0309, or E0478 otherwise). `Self: 'x` needs `where Self: 'x` for a generic host (`impl<H> Tr for Gen<H>`) where the receiver is not `&'x self`.
- The Leaker does not see the GAT of a supertrait (`trait Tr: Ext`, `Self::G<'x>`) or of the trait of a generic (`T::G<'x>` with `T: Ext`). It records the predicate only, and the Repeater's impl is E0309 on `Self` or `T` if its where clause needs more. Write the bound on the method (`where Self: 'x`, `where T: 'x`): it is recorded, and the impl compiles.
- Recording `T: 'x` for every path that has arguments would be stricter than the original where the type has no such where clause. So the Leaker does not.

# Binder lifetimes

The lifetimes of a `for<..>` binder are generics of the types inside it. `Rel<'x>` in `for<'x> fn(Rel<'x>)` is leaked with `<'x>`.

The copy fails if the output names `'x` and no input keeps `'x` as written.

- The output is a return type, the output of `Fn(A) -> R`, or an associated type binding such as `Iterator<Item = R>`.
- An elided lifetime in the output is `'x` if that is the only lifetime of the inputs (`for<'x> fn(Rel<'x>) -> &u8`).
- rustc does not count a lifetime as constrained if it appears only in an associated type path such as `<T as Tr<'x>>::A`.
- The errors are E0581 for a function pointer and E0582 for `Fn`.

An input that keeps `'x` is an *anchor*. With no anchor, the Leaker leaks the whole function pointer or trait object. That is an error if it cannot: in a bound, in a where predicate, with a trait object under an elided reference ([Elided lifetimes](#elided-lifetimes)), or in `impl for<'x> Fn(..)`.

```rust,ignore
F: for<'x> Fn(Rel<'x>) -> &'x u8                // error: no anchor
F: for<'x> Fn(&'x u8, Rel<'x>) -> &'x u8        // good: `&'x u8` anchors `'x`
F: for<'x> Fn(::my_crate::Rel<'x>) -> &'x u8    // good: nothing is leaked
```

Instead, keep an anchor, or make the path absolute.

What is an anchor:

- An associated type path (`<T as Tr<'x>>::A`, `T::A<'x>`), an array length, `impl Trait` and a type macro do not anchor.
- The rest of a type does, however deep, as rustc checks. That includes `Box<dyn Iterator<Item = &'x u8>>`, `Box<dyn Tr + 'x>` and `Box<dyn Fn() -> &'x u8>`.
- A type that is leaked as a whole keeps no anchor. So does an external type that the Leaker does not allow, such as `::other::Wrap<&'x u8>`, since the Referrer may replace it. A `Parser<'x>` next to it cannot be leaked alone.
- Occurrences in `::core` paths, in allowed paths and in types that are not leaked stay as written.
- A `::core` or allowed path is assumed to be a struct, an enum or a union. A type alias that drops the lifetime would not anchor, so allow only crates without such aliases.

What is the output of a bound:

- Only the associated type bindings and the output of the bound itself are its output. `T: for<'x> Tr<u8, Output = &'x u8>` is rejected.
- The bounded type and the arguments of a bound anchor the output of that bound only.

More rules:

- A type that names `'x` is leaked only if its own path depends on the context. So `(&'x str, Parser<'x>)` keeps `&'x str`, and only `Parser<'x>` is leaked.
- A binder inside an input is not probed again. A type that uses its lifetime makes the inner function pointer or trait object count as leaked, which loses its anchors. This is cautious.

# Elided lifetimes

An elided lifetime in the output of a signature is the only lifetime of its inputs: the elided one of `&Rel<T>` in `Fn(&Rel<T>) -> &u8`. A leaked type that has lifetimes among its generics names them in the arguments of its projection (`<S as Repeater<(&'a (), T, &'a T), ..>>::Type`). So the inputs of the copy have more lifetimes, and rustc cannot tell which one the output has (E0106). `Rel<T>` has `<'a, T: 'a>` where the Leaker has `'a` and `T: 'a`.

This affects the output of `Fn(..) -> R`, of a function pointer, and of a method that has no reference receiver. A method with `&self`, `self: &Self` or `self: Pin<&mut Self>` is not affected: its output takes the lifetime of the receiver.

If the inputs show no lifetime, a path may hide the one that the output has (`fn(Ref<u8>, Rel<T>) -> &u8` with `Ref<'a, T>`). The Leaker cannot see it. It takes it as the lifetime that elision gives, so a leaked type that adds lifetimes breaks it as well.

The Leaker leaks the whole function pointer or trait object instead of the type in its inputs. Its impl of the Repeater keeps the elision as written. That is an error if it cannot: in a bound, in `impl Fn(..)`, in a method, or in a trait object under an elided reference, which takes its lifetime from it.

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

When nothing is added:

- Nothing is added if the inputs keep the lifetimes the entry has: the only lifetime of the inputs is `'a` (`Fn(&'a Rel<T>) -> &u8`).
- Nothing is added if the output does not rely on elision: it has no lifetime or names one (`Fn(&Rel<T>) -> Rel<T>`, `for<'x> Fn(&'x Rel<T>) -> &'x u8`).
- `Rel<T>` where `T` has no lifetime bound is leaked alone as usual.

Nested signatures:

- A function pointer or `Fn(..)` inside an input has its own signature. Its lifetimes do not count for the one around, and the whole inner one is the entry if it adds lifetimes to the outer one.
- That is how rustc 1.64 and later count (checked with 1.64, 1.65, 1.70, stable and nightly; 1.63 fails). Older ones also count the lifetimes that the inner one does not bind, so the copy of `Fn(&Box<dyn for<'x> Fn(&'x Rel<T>) -> &'x u8>) -> &u8` does not compile there.
- The elided lifetimes inside a function pointer or `Fn(..)` are its own, so they do not stop it from being leaked as a whole. They stop the types inside it from being leaked alone: `Rel<&Rel2>` in `Fn(Rel<&Rel2>)` is an error, like any type with an elided lifetime.

Hidden lifetimes:

- A lifetime that a path hides (`Ref<u8>` for `Ref<'a, u8>`, `::core::slice::Iter<u8>`) is not seen by the Leaker.
- In the inputs, it is taken as the lifetime that elision gives, when the output is elided and the inputs show no lifetime. Write `Ref<'_, u8>` to make it visible, for the same result. The Referrer can then name it.
- A hidden lifetime in the output (`fn(&Rel<T>) -> Ref<u8>`) is not seen. The copy fails if `Rel<T>` adds lifetimes, so write `Ref<'_, u8>` there.

A function that is itself the analyzed item has no signature to the Leaker, which sees only the types you visit. Write the lifetimes out.

`Self` in the inputs:

- `Self` and `Self::A` in the inputs add lifetimes if the copy writes them as a type that has some. `Visitor::with_self()` writes `Self` of a struct, an enum or a union as the type itself (`::path::S<'a>`), and `Self::A` as `<::path::S<'a>>::A`.
- rustc counts no lifetime for `Self`, so `fn(&Self) -> &u8` is valid in `struct S<'a>`. The plain copy `fn(&::path::S<'a>) -> &u8` would be E0106.
- So the Referrer names the lifetime that elision gives, with a fresh binder, in the inputs and in the output: `for<'__tl_e0> fn(&'__tl_e0 ::path::S<'a>) -> &'__tl_e0 u8`. It does the same for `Fn(..)` in a type, in a bound and in a where predicate. Nothing is leaked, and these are fine in a bound and with `allow_self = false`.
- If the inputs name their lifetime (`for<'x> fn(&'x Self) -> &u8`), the output takes that name.
- The lifetime that the inputs keep is not added (`fn(&'a Self) -> &u8` in `struct S<'a>`).
- The copy of an impl keeps `Self`, and so does that of a trait, whose impl of the Repeater writes `Self::A` as `<__S>::A` (`Referrer::iter_with_self()`). So they are not counted. A trait with a host type that has lifetimes (`with_self(Host<'h>)`) gets the binder like a struct.
- Where the Referrer cannot name the lifetime (a path hides it, or the signature is a method's, which has no binder), `Self` counts as a leaked type with lifetimes. The function pointer or trait object is leaked as a whole, with `Self` as written, since the impl of the Repeater for the type keeps it. That is an error with `allow_self = false`, in a bound, in a method, and in a trait object under an elided reference ([Self rules](#self-rules)).

# Trait objects

A `dyn Trait` without a lifetime bound gets its bound from its surroundings.

- Behind `&'a`, it is the lifetime of the reference, or the bound of the trait if it has one.
- As an argument of a path, it is a bound written in the definition (`Ref<'a, T: ?Sized + 'a>`). A bound that rustc only infers from a field does not count, and the default is `'static`.
- The Leaker cannot see these definitions, so it never leaks such a trait object alone.
- Writing the default out (`dyn RelTrait + 'a`) is rejected by rustc (E0478) for a trait with a bound of its own.

What the Leaker does:

- If the trait object must be leaked itself, the whole reference or path is leaked, with the lifetime among its generics. That is the case for a relative trait (`&'a dyn RelTrait`) and for a bare identifier that may be a const. For a path, `'a` must be its only lifetime argument.
- Otherwise the trait object stays in the copy, and what is inside it is leaked (`Rel` of `&'a dyn ::path::Tr<Item = Rel>`).
- A lifetime bound you write changes this. With a lifetime of the Leaker, the whole reference is leaked. With a local lifetime, the trait object alone is leaked with its bound.
- These are errors: an elided lifetime (`&dyn RelTrait`, `Ref<'_, dyn RelTrait>`), several lifetime arguments, and a trait object in a bound or where predicate (`T: ::path::Tr<'a, dyn RelTrait>`). `Wrap<'static, dyn RelTrait>` and `Ref<'a, Box<dyn RelTrait>>` are fine.

```rust,ignore
fn f(x: &dyn RelTrait);          // error: elided lifetime
fn f<'a>(x: &'a dyn RelTrait);   // good: the whole reference is leaked
```

Instead, name the lifetime, make the trait path absolute, or write the bound: `&'a (dyn RelTrait + 'a)`.

# Bounds that name leaked types

A bound or where predicate is rewritten like any other type: `T: Fn(Rel)` becomes `T: Fn(<() as Repeater<..>>::Type)`. That is a problem if a path through the Repeater is in a predicate of the item, and the Repeater's impl needs a bound of the item that the environment has in another form.

- rustc builds the environment of the item from its predicates, and normalizes the path there, while the bounds are not normalized yet.
- The impl of the Repeater for `Wrap<T>` needs `T: Fn(Rel)`, and the environment has `T: Fn(<() as Repeater<..>>::Type)`. rustc cannot prove it (E0277 "expected an `Fn(Rel)` closure, found `T`", E0271, E0275 or E0284).
- The copy cannot have the original bound, the only link between the crates being the Repeater. So the Leaker reports an error.
- A type in a field or in a signature is fine, as it is normalized when the environment is built.
- The bounds of an `impl Trait` in an argument (`x: impl Into<Wrap<T>>` is `x: X` with `X: Into<Wrap<T>>`) are predicates of the function, and so are those of a generic. The bounds of one in the return type are not: rustc proves them where the body is.

```rust,ignore
struct S<T: Fn(Rel)>(Wrap<T>) where Wrap<T>: Sized;  // error: `Wrap<T>` is in a predicate, `T: Fn(Rel)` names `Rel`
struct S<T: Fn(Rel)>(Wrap<T>);                       // good
struct S<T: Fn(Rel)>(Wrap<T>) where T: Sized;        // good
```

A bound is *unstable* if the environment has it in another form than the impl that rustc proves needs it. That is so if it names:

- a leaked type (`T: Fn(Rel)`), which the copy writes as a path through the Repeater;
- `Self`, if `Self` is the self type of an impl and that is leaked (`X: Fn(Self) -> u8`). The copy writes the self type through the Repeater, and the impl of an entry has the bound with the self type of the Leaker;
- `Self::A` (`X: Fn(&Self::A) -> u8`). The impl of the Repeater for the implementors has `<__S as Tr>::A`, and the environment of the copy has `<Host as Tr>::A`, which rustc resolves by the impl of the trait;
- an associated type that rustc normalizes. That is the `Item` of a generic that an `Iterator<Item = ..>` bound or predicate in scope equates (`T: Fn(T::Item)` next to `T: Iterator<Item = u8>`, also in an enclosing item), the `Output` of `Fn(..)`, and a projection of a type that an impl defines (`<Vec<u8> as IntoIterator>::Item`, `<u8 as Tr>::A`).

About associated types that rustc normalizes:

- The `Item` of a generic that nothing equates is rigid, and stable.
- A supertrait may fix it (`trait WIter: Iterator<Item = u8>`, then `T: WIter + Fn(T::Item)` is `T: WIter + Fn(u8)`). The Leaker cannot read it. So a projection (`T::Item`, `<T as Iterator>::Item`) of a generic with a bound of a trait that is not one of the standard library counts as one that rustc normalizes.
- A bound of the standard library (`Iterator`, `Clone`, `Fn(..)`, any path in `::core`, `::std` or `::alloc`) does not make the projection count. A bound of any other trait does. So `T: Tr + Fn(T::Out)` with `trait Tr { type Out; }` is rejected, although it would compile.
- A type alias that hides the projection is not seen.

A where predicate that is not a bound of a generic (`Vec<T::Item>: Default`, `u8: Into<T::Item>`) is proved by an impl. So in it only the last kind counts, and only the projection of a generic that a bound or predicate in scope equates (or that a supertrait may fix). The same holds in the bounded type of a predicate on a generic (`T::Item: Clone`).

- The predicate that equates it may be of an inner item. In `trait Tr<T: Iterator> where Vec<T::Item>: Default`, the method `fn m(&self) where T: Iterator<Item = Rel<T>>` has the entry `Rel<T>`, whose context has the predicate of the trait. The Repeater's impl of the entry needs `Vec<T::Item>: Default`, and rustc normalizes `T::Item` to the path through that impl, which needs the predicate (E0275).
- It counts whatever the predicate equates the `Item` to. `Item = u8` compiles with `Vec<T::Item>: Default`, but not with `T::Item: Into<T>` (the environment has the predicate with the projection, and the impl needs it normalized). The Leaker cannot tell, so it rejects both.
- `Self::A` and the projection of a type that an impl defines (`<Vec<T> as IntoIterator>::Item`) do not count there, as an impl proves them.

It is an error for a type in a bound or where predicate of the item or of a method, in these cases:

- The type is a leaked type (`where Wrap<T>: Sized`, `U: Into<Wrap<T>>`, `x: impl Into<Wrap<T>>`), and a bound of a generic that it uses is unstable. A type that names itself is one: `T: Into<Wrap<T>>`. A stable bound (`T: Clone`) is fine, and so is a predicate on another type than a generic (`Vec<Rel2<T>>: Clone`), which rustc proves by an impl.
  - The generics of the type are those of its context. So a bound of the enclosing item counts only if the type uses the generic: `T: Fn(T::Item)` of an impl, and `T: Iterator<Item = Wrap<T>>` of its method.
  - The same holds for a where predicate of the enclosing item that is not a bound of a generic and names its `Item` (`Vec<T::Item>: Default`), if the type uses the generic and a bound or predicate in scope, also one of a method, equates the `Item`.
- The type is `Self::A`, an associated type of `Self`, and a bound of the generics of the trait (or of the type of `Self`) is unstable. The copy for a concrete implementor resolves it by the impl of the trait. In an impl, the impl of the trait has all the generics of the impl, not only those that `Self` uses. `impl<T> Tr<T> for u8 where T: Fn(Rel) { fn m<X>(&self) where X: Into<Self::A>; }` is an error, and so is `impl<T: Fn(Rel)> Tr<T> for u8` with `u8: Into<Self::A>`.
- The type is `Self`, in an impl whose self type is leaked, and a bound of the generics of a type in the self type that the copy writes through the Repeater is unstable. The copy writes the self type through the Repeater, and `Self` stands for it. rustc normalizes it where the predicate names it, with the impls of the Repeater for those types, which need the bounds of the generics that they use, not those of the other generics of the impl. `impl<T: Fn(Rel2), V: AsRef<Self>> ::p::W<Rel<T>, T, V>` is an error, and `impl<T: Fn(Rel2), U, V: AsRef<Self>> ::p::W<T, Rel<U>, V>` is fine.
  - The Leaker does not choose a type of the self type as an entry for its parts if its generics have an unstable bound. `impl<T: Fn(Rel2), U> Tr<U> for ::p::W<T, Rel<U>> where Self: Clone` has the entries `Rel2` and `Rel<U>`, not `::p::W<T, Rel<U>>`, whose bound `T: Fn(Rel2)` is unstable. It is an error only for a type that must be an entry.
- The type is a leaked type that names `Self` of a trait, in a predicate of the trait itself (its where clause, a bound of its generics or a supertrait), not in one of a method: `trait Tr where Rel<Self>: Sized`. The Repeater's impl for it is for the implementors of the trait, so proving that the host of the copy implements the trait needs the predicate, which names the type.

The error names the type and the bound. Instead, write the type of the bound with an absolute path (`T: Fn(::my_crate::Rel)`), write the bound without what it names, or use the type outside of the predicates (in a field or a signature).

Two things are not errors:

- A type that is an entry only because its parts need it (`(Rel, T, Rel2)` in `T: Fn((Rel, T, Rel2))`) is not one when that makes a cycle. Its parts are the entries.
- An optional entry ([External crates](#external-crates)) in such a place is not an entry either. It is copied as written, even if the Referrer would replace it. That holds for an optional entry that the bound names (`T: Fn(::ext::X)` and `Wrap<T>: Sized`), for the optional types of a self type that an unstable bound names, and for an optional entry whose own context is unstable (`&'a ::ext::W<T, ::ext::B0>` in `T: Fn(&'a ::ext::W<T, ::ext::B0>)`, where the predicate that names `'a` is part of its context).

# Method lifetimes

A leaked type may name a lifetime `'b` of a method: a path (`Wrap<'b>`), or a trait object that takes `'b` from the reference or path around it (`&'b dyn RelTrait`).

- In the trait, `'b` is late-bound: an argument uses it and no bound names it.
- In the copy, only `<Self as Repeater<..>>::Type` holds it, and that does not constrain it. So rustc makes it early-bound, and rejects the copy because the two differ (E0195).
- That affects a method of a trait, and a method of an impl of a trait: rustc compares the signature with the trait's.
- A free function and a method of an inherent impl have no signature to differ from. The copy compiles with `'b` early-bound, and the Leaker does not report an error for them (also in a module).

The Leaker reports an error if all of these hold:

- A type that is leaked as a whole names `'b`: a path with a relative name, or a trait object.
- The return type names `'b`, and no bound or where predicate does.
- Some argument constrains `'b` in the original, and none does in the copy, because the types leaked as a whole are projections there.

```rust,ignore
fn f<'b>(&self, x: &'b dyn RelTrait) -> &'b u8;           // error
fn f<'b>(&self, x: Wrap<'b>) -> &'b u8;                   // error: `Wrap` is relative
fn f<'b>(&'b self, x: &'b dyn RelTrait) -> &'b u8;        // good: `&'b self` keeps `'b`
fn f<'b>(&self, x: &'b dyn RelTrait, y: &'b u8) -> &'b u8; // good: `y` keeps `'b`
fn f<'b>(&self, x: Wrap<'b>, y: &'b u8) -> &'b u8;        // good: `y` keeps `'b`
```

Instead, keep `'b` anchored, make the trait or path absolute (`::my_crate::RelTrait`, `::my_crate::Wrap<'b>`), or write the bound (`&'b (dyn RelTrait + 'b)`).

- A bound on `'b` (`'c: 'b`, `T: 'b`, `where Self: 'b`) makes it early-bound in both, so it is fine. So is a function whose only mention of `'b` is the return type.
- A lifetime of the Leaker and one of the outermost function are not affected.

What keeps `'b`:

- An argument keeps `'b` if it is the receiver (`&'b self`), or holds it in a type that is not leaked, however deep: a reference, tuple, array, slice, pointer, function pointer, or a path argument (`Option<&'b str>`).
- That includes an external type that the Leaker does not allow, which the Referrer leaves as written unless it replaces it.
- An associated type path (`T::A<'b>`) and an array length do not hold it, as rustc sees it.
- A reference to a trait object that takes `'b` from the reference keeps it only if the trait object is certainly not leaked as a whole. That is true for `&'b dyn ::core::fmt::Debug`, and for a trait of an allowed crate.
- It is not certain for a relative trait, a bare identifier that may be a const, an external type that is not allowed, a path of the analyzed module, and a type that may be leaked (`Vec<&'b u8>`).

# Bare identifiers and consts

A bare identifier argument (`MY_CONST` in `::path::MyType<MY_CONST>`) parses as a type, so it may be a type or a const. Its position decides.

- A const cannot be an entry on its own, so the type that holds it is leaked as a whole.
- If that is impossible, the identifier is leaked as a type, and the copy fails to compile if it is a const. This happens when the type has an elided lifetime or `impl Trait`, when its path is declared in the module of `intern_mod()`, and in a trait bound such as `T: ::path::MyTrait<MY_CONST>`.
- An identifier in a tuple, an array or a binding (`Item = X`) is a type.

```rust,ignore
fn f(x: ::path::MyType<'_, MY_CONST>);       // error: treated as a type
fn f(x: ::path::MyType<{ ::my_crate::MY_CONST }>);  // good: not leaked
```

Instead, write the const in braces with an absolute path.

More rules about consts:

- A const value can only be leaked as part of a type. A relative const in a trait bound is rejected, because a const cannot be recovered from an associated type without `generic_const_exprs`.
- A type in an expression (`[u8; size_of::<MyType>()]`) is not leaked alone. Its copy would put an associated type path in an anonymous constant. The type that holds it is leaked as a whole.
- Primitive names (`u8` in `const N: u8`) are copied as they are. This assumes that nothing shadows them (`type u8 = i32;`). Emit `Leaker::assertions()` in the Leaker's type context to check it at compile time.
- The type of a const parameter must be a primitive type, or a path that does not depend on the context, such as an absolute path to an alias. `RepeaterSlots` cannot encode an alias.

# External crates

An absolute path such as `::serde::Wrap<(Rel,)>` names a crate by name only. It means the same type in the other crate only if that crate has the very same crate under that name. The Leaker cannot tell. So it leaks such a type as an *optional entry* (`Referrer::must()` is `false`), after the required ones.

`Referrer::allow_ambiguous_crate_name()` is the Referrer's choice. It is not serialized.

- `true` (the default): the optional entry is left as written, and the types in it are looked up as usual (`Rel` is replaced). Use it when both sides are built from the same dependency graph. A crate of the same name with another version or origin is taken for the same one, and the copy then fails or means another type.
- `false`: the optional entry is replaced like a required one, through the Repeater. An absolute type that contains a relative one (`::serde::Wrap<(Rel,)>`) is replaced as a whole. Use it if the other crate may have other dependencies than the Leaker's. The impl of an optional entry is needed only if some Referrer replaces it. The flag is not serialized, so the crate that writes the impls cannot know: it writes the impls of all the entries (`Referrer::iter()`), not only of `Referrer::required()`, or a strict Referrer in another crate replaces a type that has no impl (E0277).

```rust,ignore
fn f(x: ::serde::Wrap<(Rel,)>);  // `Rel` is an entry. The whole type is an optional entry.
```

These absolute paths are not optional entries:

- `::core::..` and `$crate::..`;
- paths that start with one of `LeakerConfig::allowed_paths`. `parse_quote!(::serde)` allows `::serde::..` and not `serde::..`. You promise that allowed crates are the same on both sides.
- `::std` and `::alloc` are like any other crate unless allowed.

`Referrer::external_crates_require()` lists the crates an entry names.

More rules:

- Only types are optional entries. An absolute path in a bound, a value, the type of a const parameter, the type of a receiver (`self: ::std::boxed::Box<Self>`) and a type that cannot be leaked (`::serde::Wrap<'_>`) is copied as written. That is not an error.
- The required entries are those of a Leaker that allows every crate, and they come first.
- Replacing a type can lose an anchor or a method lifetime, so the required entries may differ in that case.
- An optional type inside a required one is not an entry. One that contains required ones is, and they are entries of their own.

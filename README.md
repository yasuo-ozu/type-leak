# type-leak crate [![Latest Version]][crates.io] [![Documentation]][docs.rs] [![GitHub Actions]][actions]

[Latest Version]: https://img.shields.io/crates/v/type-leak.svg
[crates.io]: https://crates.io/crates/type-leak
[Documentation]: https://img.shields.io/docsrs/type-leak
[docs.rs]: https://docs.rs/type-leak/latest/type-leak/
[GitHub Actions]: https://github.com/yasuo-ozu/type-leak/actions/workflows/rust.yml/badge.svg
[actions]: https://github.com/yasuo-ozu/type-leak/actions/workflows/rust.yml

In Rust, type references depend on the type context in which they appear. For example, relative paths rely on the location of the type context and the imported namespaces. Even absolute paths depend on the crate itself and the crates it depends on. This becomes problematic when defining macros that copy user-written type references as token sequences into another type context. In short, type references written by the user are valid in their original type context but may not be valid in the type context into which they are copied.

`type-leak` is a concept (and library) that helps you rewrite type references when copying user-defined type definitions within the macros you create. This is crucial because type references in Rust depend on the context in which they are defined. When you copy these definitions into a different context, the type references they contain may no longer be valid. `type-leak` addresses this problem by handling the rewriting process. Three key elements are involved:

1. The **Leaker** is a user-defined type (such as an enum or struct) or a trait definition intended to be copied into another type context. It must be located where it can be referenced from the Referrer (described below) via absolute paths, relative paths, or type aliases. If the Leaker is a trait definition, you must also specify a leaker struct type that has the same type parameters as the leaker trait.

2. The **Repeater** is a trait that passes type information from the Leaker to the Referrer. It must be accessible from both the Leaker and the Referrer via absolute or relative paths. This trait is defined in the `type-leak` crate. The Repeater has a `const` generic parameter that uniquely identifies each type reference in the Leaker.

3. The **Referrer** is the definition into which the Leaker's type references are copied. The `type-leak` crate parses the Referrer's definition and replaces the type references inside it as needed. The Referrer also records the location of the Leaker and the type arguments passed to it.

The three components are generally defined in different crates, so `type-leak` enables copying type context across crate boundaries.

# Core mechanism

The definition of a "repeater" is like:

```rust,ignore
pub trait Repeater<const INDEX: usize> {
    type Type: ?Sized;
}
```

This trait is used by both "leaker" and "referrer". The leaker type is like:

```rust,ignore
enum MyLeaker<T, U, ...> {
    Variant1(path::to::MyType1),
    Variant2(path::to::MyType2),
}
```

Or if the leaker is a trait definition, the leaker type is like:

```rust,ignore
trait MyLeaker<T, U, ..> {
    fn trait_fn(input: path::to::MyType1) -> path::to::MyType2;
}

struct MyLeakerType<T, U, ...>(::core::convert::Infallible);
```

`type-leak` automatically analyzes the definition of `MyLeakerType` or `MyLeaker` to find all type references that depend on the type context of that definition. For each such type reference, it assigns a unique ID (simply an incrementing integer) so they can be distinguished through a common Repeater.

Given a Referrer, `type-leak` also analyzes its type references in the same way and generates new type references that depend only on the path of the Repeater trait.

Example of leaking a type in the leaker's type context:

```rust,ignore
impl<T, U, ..> path::to::Repeater<0> for MyLeakerType<T, U, ..> {
    type Type = path::to::MyType1;
}

impl<T, U, ..> path::to::Repeater<1> for MyLeakerType<T, U, ..> {
    type Type = path::to::MyType2;
}

// MyLeakerType definition is placed here
```

Example of a reference in the referrer's type context:

```rust,ignore
<MyLeakerType<T, U, ..> as path::to::Repeater<0>>::Type
```

# Application

- [sumtype](https://crates.io/crates/sumtype)
- [flat_enum](https://crates.io/crates/flat_enum)
- [parametrized](https://crates.io/crates/parametrized)


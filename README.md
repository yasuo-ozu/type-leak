# type-leak crate [![Latest Version]][crates.io] [![Documentation]][docs.rs] [![GitHub Actions]][actions]

[Latest Version]: https://img.shields.io/crates/v/type-leak.svg
[crates.io]: https://crates.io/crates/type-leak
[Documentation]: https://img.shields.io/docsrs/type-leak
[docs.rs]: https://docs.rs/type-leak/latest/type-leak/
[GitHub Actions]: https://github.com/yasuo-ozu/type-leak/actions/workflows/rust.yml/badge.svg
[actions]: https://github.com/yasuo-ozu/type-leak/actions/workflows/rust.yml

In Rust, type references depend on the type context in which they are described. For example, relative paths rely on the location of the type context and the imported namespaces. Even absolute paths depend on the crate itself and other crates that the crate depends on. This fact becomes problematic when defining macros that copy user-written type references as token sequences into another type context. In other words, type references written by the user are valid in their original type context but are not necessarily valid in the type context to which they are copied.

`type-leak` is a concept (or library) that assists you in rewriting type references when copying user-defined type definitions within the macros you create. This is crucial because type references in Rust depend on the context in which they are defined. When you copy these definitions into a different context, their type references might not be valid anymore. "Type-leak" helps address this problem by handling the rewriting process. Three key elements are involved:

1. The "Leaker" is a user-defined type (such as an enum or struct) or trait definition that is intended to be copied into another type context. It must be located in a position that can be referenced from the Referrer (which we will describe next) via absolute paths, relative paths, or through type aliases. If "Leaker" is an trait definition, you should specify leaker type, which has the same type parameters with the leaker definition.

2. The "Repeater" is a trait that facilitates passing type information from the Leaker to the Referrer. It must be accessible from both the Leaker and the Referrer using absolute or relative paths. This trait is defined in the type-leak crate. The repeater has a const generic, which distinguishes a type referrence in the leaker.

3. The "Referrer" is the definition into which the type references written in the Leaker are copied. The type-leak crate parses the definition of the Referrer and replaces the type references inside it as necessary. Referrer also knows the location of leaker, and type arguments passed to that.

The three components are generally defined in different crates, and thus `type-leak` allows copying type context beyond crate boundary.

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

Or if the leaker is an trait definition, the leaker type is like:

```rust,ignore
trait MyLeaker<T, U, ..> {
    fn trait_fn(input: path::to::MyType1) -> path::to::MyType2;
}

struct MyLeakerType<T, U, ...>(::core::convert::Infallible);
```

`type-leak` automatically analyze definition of `MyLeakerType` or `MyLeaker` to find all type references which depends on the typa context of this definition. For each type references it assigns unique IDs (of just an incremented variable) to be distinguished through a common repeater.

With given referrer, `type-leak` also analyze type referrences in the same way, and generates new type referrences which only depends on the path of repeater trait.

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

Example of referrence in the referrer's type context:

```rust,ignore
<MyLeakerType<T, U, ..> as path::to::Repeater<0>>::Type
```

# Application

- [sumtype](https://crates.io/crates/sumtype)
- [flat_enum](https://crates.io/crates/flat_enum)
- [parametrized](https://crates.io/crates/parametrized)


#[type_leak_test_macro::leak_strict(serialized)]
pub trait Prelude: Clone {}

#[type_leak_test_macro::leak_strict(serialized)]
pub trait Bound: for<'a> Fn(&'a u8) {}

#[type_leak_test_macro::leak_strict(serialized)]
pub trait Nested {
    fn f(&self, x: Box<dyn RelA + RelB<'_>>);
}

// Several offending paths in one item: each is an error of its own, at its own path.
#[type_leak_test_macro::leak_strict(serialized)]
pub trait Multi: Clone + Send {
    fn f(&self, a: MyType<'_>, b: Other<'_, ::core::primitive::u8>);
    fn g<T: Sync>(&self, t: T) -> Vec<T>;
}

// `Self` where no scope declares it: one error for each mention.
#[type_leak_test_macro::leak_generics(serialized)]
pub trait NoSelf {
    fn f(&self, a: &'static Self) -> Self::Item;
}

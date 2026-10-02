#[type_leak_test_macro::leak_strict]
pub trait Prelude: Clone {}

#[type_leak_test_macro::leak_strict]
pub trait Bound: for<'a> Fn(&'a u8) {}

#[type_leak_test_macro::leak_strict]
pub trait Nested {
    fn f(&self, x: Box<dyn RelA + RelB<'_>>);
}

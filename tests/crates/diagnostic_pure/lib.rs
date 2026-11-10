// The macro builds the tokens of `ErrorItem::to_compile_error()` for each error and drops them:
// the crate compiles, on every toolchain (the nightly diagnostic is for `emit()`).
#[type_leak_test_macro::leak_discard]
pub trait Prelude: Clone {}

#[type_leak_test_macro::leak_discard]
pub trait Multi: Clone + Send {
    fn f(&self, a: Vec<u8>);
}

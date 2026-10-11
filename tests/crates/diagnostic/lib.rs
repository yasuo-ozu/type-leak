#[type_leak_test_macro::leak_strict]
pub trait Prelude: Clone {}

#[type_leak_test_macro::leak_strict]
pub trait Bound: for<'a> Fn(&'a u8) {}

#[type_leak_test_macro::leak_strict]
pub trait Nested {
    fn f(&self, x: Box<dyn RelA + RelB<'_>>);
}

// Several offending paths in one item: each is an error of its own, at its own path.
#[type_leak_test_macro::leak_strict]
pub trait Multi: Clone + Send {
    fn f(&self, a: MyType<'_>, b: Other<'_, ::core::primitive::u8>);
    fn g<T: Sync>(&self, t: T) -> Vec<T>;
}

// `Self` where no scope declares it: one error for each mention.
#[type_leak_test_macro::leak_generics]
pub trait NoSelf {
    fn f(&self, a: &'static Self) -> Self::Item;
}

// The Referrer is serialized with the format of the previous version of `type-leak`, of the next
// one, with no tag, and with something that is no tag. The current format is no error.
#[type_leak_test_macro::leak_tagged(tag = "current")]
pub trait TagCurrent {
    fn f(&self) -> Rel;
}
__tagged_TagCurrent!();

#[type_leak_test_macro::leak_tagged(tag = "stale")]
pub trait TagStale {
    fn f(&self) -> Rel;
}
__tagged_TagStale!();

#[type_leak_test_macro::leak_tagged(tag = "future")]
pub trait TagFuture {
    fn f(&self) -> Rel;
}
__tagged_TagFuture!();

#[type_leak_test_macro::leak_tagged(tag = "none")]
pub trait TagNone {
    fn f(&self) -> Rel;
}
__tagged_TagNone!();

#[type_leak_test_macro::leak_tagged(tag = "garbled")]
pub trait TagGarbled {
    fn f(&self) -> Rel;
}
__tagged_TagGarbled!();

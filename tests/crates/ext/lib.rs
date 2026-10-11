//! A crate which the Leaker's crate (`type-leak-test-leaker`) names with an absolute path
//! (`::type_leak_test_ext::Ext<..>`) and the Referrer's crate (`type-leak-test-referrer`) does not
//! depend on: the type of it is an optional entry, which only a strict Referrer can copy there.

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Ext<T>(pub T);

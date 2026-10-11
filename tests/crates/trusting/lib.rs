//! A Referrer's crate which implements the Leaker's trait `ExtDep`, whose signatures name a type of
//! the crate `type_leak_test_ext`, which this crate does not depend on.
//!
//! The type is an optional entry (an absolute path of an external crate). A Referrer which trusts
//! the names of the crates leaves it as written, and rustc cannot find the crate here (E0433).
//! With the feature `strict` the macro replaces it with the Repeater's projection, and the crate
//! compiles (and its tests, which run only with the feature, work). It then copies the items of
//! the Leaker's crate that name the type too, as a strict Referrer does.

pub struct NoDependency(pub type_leak_test_leaker::x::Inner);

#[cfg(not(feature = "strict"))]
type_leak_test_leaker::x::ExtDep!(impl ExtDep for NoDependency => 0);

#[cfg(feature = "strict")]
type_leak_test_leaker::x::ExtDep!(strict impl ExtDep for NoDependency => 0);

/// A struct with the type of the crate as fields (`#[prop_leak]`): its copy has the original types.
#[cfg(feature = "strict")]
pub mod kinds {
    type_leak_test_leaker::__tl_prop_9004!(path = ::type_leak_test_leaker::kinds, strict);
}

/// A fragment of an impl, analyzed with the scope API (the type of the crate is an optional entry
/// next to a required one, and the Leaker's macro wrote the impl of both).
#[cfg(feature = "strict")]
pub mod scoped {
    type_leak_test_leaker::__scope_apply_7005!(path = ::type_leak_test_leaker::scoped, strict);
}

#[cfg(all(test, feature = "strict"))]
mod tests {
    use super::*;
    use type_leak_test_leaker::__ext::Ext;
    use type_leak_test_leaker::kinds as l;
    use type_leak_test_leaker::x::{ExtDep, RelX};

    fn squash(text: &str) -> String {
        text.split_whitespace().collect()
    }

    /// The strict impl, in a crate without the dependency, works: it forwards, and the types it
    /// takes and returns are the original ones.
    #[test]
    fn the_strict_impl_forwards() {
        let strict = NoDependency(type_leak_test_leaker::x::Inner);
        let wrapped: Ext<(RelX,)> = strict.wrapped();
        assert_eq!((wrapped.0).0 .0, 21);
        assert_eq!(strict.plain(Ext(4)), 5);
    }

    /// The copy of a struct has the original types in its fields, the type of the crate included.
    #[test]
    fn the_strict_copy_of_a_struct_has_the_original_types() {
        let rel = l::Rel(7);
        let holds = kinds::Holds {
            wrapped: Ext((rel,)),
            plain: Ext(5u8),
            rel,
        };
        assert_eq!(holds.wrapped, Ext((l::Rel(7),)));
        assert_eq!(holds.plain, Ext(5));
    }

    /// The copy of the types of a fragment does not name the crate: the optional entry is replaced.
    #[test]
    fn the_strict_copy_of_a_fragment_does_not_name_the_crate() {
        let trait_path = squash(scoped::SCOPE_7005_TRAIT);
        assert!(!trait_path.contains("type_leak_test_ext"), "{}", trait_path);
        // The optional entry is the entry 1, after the required one.
        assert!(
            trait_path.contains("7005u64,1usize>>::Type,"),
            "{}",
            trait_path
        );
        assert!(
            trait_path.ends_with("7005u64,0usize>>::Type>"),
            "{}",
            trait_path
        );
    }
}

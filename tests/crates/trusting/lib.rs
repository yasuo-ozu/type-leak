//! A Referrer's crate which implements the Leaker's traits `ExtDepAllowed` and `ExtDep`, whose
//! signatures name a type of the crate `type_leak_test_ext`, which this crate does not depend on.
//!
//! The type is an entry unless the Leaker allows the crate. `ExtDepAllowed` is analyzed by a Leaker
//! that allows it: the copy leaves it as written, and rustc cannot find the crate here (E0433).
//! With the feature `replace` the crate implements `ExtDep`, whose Leaker does not allow the
//! crate: the macro replaces the type with the Repeater's projection, and the crate compiles (and
//! its tests, which run only with the feature, work). It then copies the items of the Leaker's
//! crate that name the type too.

pub struct NoDependency(pub type_leak_test_leaker::x::Inner);

#[cfg(not(feature = "replace"))]
type_leak_test_leaker::x::ExtDepAllowed!(impl ExtDepAllowed for NoDependency => 0);

#[cfg(feature = "replace")]
type_leak_test_leaker::x::ExtDep!(impl ExtDep for NoDependency => 0);

/// A struct with the type of the crate as fields (`#[prop_leak]`): its copy has the original types.
#[cfg(feature = "replace")]
pub mod kinds {
    type_leak_test_leaker::__tl_prop_9004!(path = ::type_leak_test_leaker::kinds);
}

/// A fragment of an impl, analyzed with the scope API (the type of the crate is an entry next to
/// another one, and the Leaker's macro wrote the impl of both).
#[cfg(feature = "replace")]
pub mod scoped {
    type_leak_test_leaker::__scope_apply_7005!(path = ::type_leak_test_leaker::scoped);
}

#[cfg(all(test, feature = "replace"))]
mod tests {
    use super::*;
    use type_leak_test_leaker::__ext::Ext;
    use type_leak_test_leaker::kinds as l;
    use type_leak_test_leaker::x::{ExtDep, RelX};

    fn squash(text: &str) -> String {
        text.split_whitespace().collect()
    }

    /// The `KEY` of the Repeater (`type_leak::repeater_key(id, entry)`) as the tokens write it.
    fn key(id: u64, entry: usize) -> String {
        format!("{}u128", (u128::from(id) << 64) | entry as u128)
    }

    /// The impl, in a crate without the dependency, works: it forwards, and the types it takes and
    /// returns are the original ones.
    #[test]
    fn the_impl_forwards() {
        let strict = NoDependency(type_leak_test_leaker::x::Inner);
        let wrapped: Ext<(RelX,)> = strict.wrapped();
        assert_eq!((wrapped.0).0 .0, 21);
        assert_eq!(strict.plain(Ext(4)), 5);
    }

    /// The copy of a struct has the original types in its fields, the type of the crate included.
    #[test]
    fn the_copy_of_a_struct_has_the_original_types() {
        let rel = l::Rel(7);
        let holds = kinds::Holds {
            wrapped: Ext((rel,)),
            plain: Ext(5u8),
            rel,
        };
        assert_eq!(holds.wrapped, Ext((l::Rel(7),)));
        assert_eq!(holds.plain, Ext(5));
    }

    /// The copy of the types of a fragment does not name the crate: the entry is replaced.
    #[test]
    fn the_copy_of_a_fragment_does_not_name_the_crate() {
        let trait_path = squash(scoped::SCOPE_7005_TRAIT);
        assert!(!trait_path.contains("type_leak_test_ext"), "{}", trait_path);
        // The type of the crate is the entry 1, after the other.
        assert!(
            trait_path.contains(&format!("{}>>::Type,", key(7005, 1))),
            "{}",
            trait_path
        );
        assert!(
            trait_path.ends_with(&format!("{}>>::Type>", key(7005, 0))),
            "{}",
            trait_path
        );
    }
}

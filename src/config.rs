//! Configuration of the [`Leaker`](crate::Leaker).

use syn::*;

/// Configuration for building a [`Leaker`](crate::Leaker).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct LeakerConfig {
    pub allowed_paths: Vec<Path>,
    pub self_ty_can_be_interned: bool,
}

impl Default for LeakerConfig {
    fn default() -> Self {
        Self {
            allowed_paths: Vec::new(),
            self_ty_can_be_interned: true,
        }
    }
}

impl LeakerConfig {
    pub fn new() -> Self {
        Default::default()
    }

    /// Allow primitive types from `::core::primitive`.
    pub fn allow_primitive(&mut self) -> &mut Self {
        self.allowed_paths.extend([
            parse_quote!(bool),
            parse_quote!(char),
            parse_quote!(str),
            parse_quote!(u8),
            parse_quote!(u16),
            parse_quote!(u32),
            parse_quote!(u64),
            parse_quote!(u128),
            parse_quote!(usize),
            parse_quote!(i8),
            parse_quote!(i16),
            parse_quote!(i32),
            parse_quote!(i64),
            parse_quote!(i128),
            parse_quote!(isize),
            parse_quote!(f32),
            parse_quote!(f64),
        ]);
        self
    }

    /// Allow items in the standard prelude (`std::prelude::v1`).
    pub fn allow_std_prelude(&mut self) -> &mut Self {
        self.allowed_paths.extend([
            parse_quote!(Box),
            parse_quote!(String),
            parse_quote!(Vec),
            parse_quote!(Option),
            parse_quote!(Result),
            parse_quote!(Copy),
            parse_quote!(Send),
            parse_quote!(Sized),
            parse_quote!(Sync),
            parse_quote!(Unpin),
            parse_quote!(Drop),
            parse_quote!(Fn),
            parse_quote!(FnMut),
            parse_quote!(FnOnce),
            parse_quote!(Clone),
            parse_quote!(PartialEq),
            parse_quote!(PartialOrd),
            parse_quote!(Eq),
            parse_quote!(Ord),
            parse_quote!(AsRef),
            parse_quote!(AsMut),
            parse_quote!(Into),
            parse_quote!(From),
            parse_quote!(Default),
            parse_quote!(Iterator),
            parse_quote!(IntoIterator),
            parse_quote!(Extend),
            parse_quote!(FromIterator),
            parse_quote!(ToOwned),
            parse_quote!(ToString),
            parse_quote!(drop),
        ]);
        self
    }

    /// Allow paths starting with `crate`.
    pub fn allow_crate(&mut self) -> &mut Self {
        self.allowed_paths.push(parse_quote!(crate));
        self
    }
}

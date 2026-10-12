#![doc = include_str!("../README.md")]
#![doc(html_logo_url = "https://raw.githubusercontent.com/yasuo-ozu/type-leak/main/logo.png")]
#![doc(html_favicon_url = "https://raw.githubusercontent.com/yasuo-ozu/type-leak/main/logo.png")]
#![cfg_attr(type_leak_diagnostic, feature(proc_macro_diagnostic))]

#[cfg(type_leak_diagnostic)]
extern crate proc_macro;

mod analysis;
mod analyze;
mod config;
mod context;
mod elided;
mod encode;
mod error;
mod graph;
mod helpers;
mod leaker;
mod referrer;
mod repeater;
mod unstable;
mod util;
mod view;

pub use syn;

pub use crate::analysis::CheckResult;
pub use crate::config::LeakerConfig;
pub use crate::context::{ContextId, SelfRequirement};
pub use crate::encode::{
    encode_generics_args_to_ty, encode_generics_args_to_ty_with,
    encode_generics_args_to_ty_with_self, encode_generics_params_to_ty,
    encode_generics_params_to_ty_with, encode_generics_params_to_ty_with_self,
    without_encoded_bounds, EncodeError,
};
pub use crate::error::{ErrorItem, NotInternableError};
pub use crate::helpers::{ImplSelf, RepeaterImpl, UseSite};
pub use crate::leaker::Leaker;
pub use crate::referrer::{NestedSelf, Referrer, Visitor};
pub use crate::repeater::{repeater_definition, RepeaterSlots};
pub use crate::view::View;

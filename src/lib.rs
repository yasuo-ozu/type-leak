#![doc = include_str!("../README.md")]
#![cfg_attr(type_leak_diagnostic, feature(proc_macro_diagnostic))]

#[cfg(type_leak_diagnostic)]
extern crate proc_macro;

mod check;
mod config;
mod encode;
mod error;
mod graph;
mod leaker;
mod referrer;
mod repeater;
mod util;

pub use syn;

pub use crate::check::CheckResult;
pub use crate::config::LeakerConfig;
#[allow(deprecated)]
pub use crate::encode::encode_generics_args_to_ty;
pub use crate::encode::{
    encode_generics_args_to_ty_with, encode_generics_params_to_ty,
    encode_generics_params_to_ty_with, EncodeError,
};
pub use crate::error::NotInternableError;
pub use crate::leaker::Leaker;
pub use crate::referrer::{Referrer, Visitor};
pub use crate::repeater::{repeater_definition, RepeaterSlots};

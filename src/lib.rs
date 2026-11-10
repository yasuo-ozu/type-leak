#![doc = include_str!("../README.md")]
#![doc(html_logo_url = "https://raw.githubusercontent.com/yasuo-ozu/type-leak/main/logo.png")]
#![doc(html_favicon_url = "https://raw.githubusercontent.com/yasuo-ozu/type-leak/main/logo.png")]
#![cfg_attr(type_leak_diagnostic, feature(proc_macro_diagnostic))]

#[cfg(type_leak_diagnostic)]
extern crate proc_macro;

mod analysis;
mod analyze;
mod context;
mod elided;
mod encode;
mod error;
mod graph;
mod leaker;
mod referrer;
mod repeater;
mod result;
mod unstable;
mod util;
mod view;

pub use syn;

pub use crate::analysis::CheckResult;
pub use crate::context::{Context, ContextId, SelfRequirement};
pub use crate::encode::{substitute_args, EncodeError, Substitute};
pub use crate::error::{Cause, ErrorItem, Reason};
pub use crate::leaker::Leaker;
pub use crate::referrer::{EntryId, Referrer, Visitor};
pub use crate::repeater::{repeater_definition, ImplSelf, Projection, RepeaterImpl};
pub use crate::result::{Allow, LeakerResult, LeakerResultError, Place, Rules};
pub use crate::view::View;

#![doc = include_str!("../README.md")]
#![doc(html_logo_url = "https://raw.githubusercontent.com/yasuo-ozu/type-leak/main/logo.png")]
#![doc(html_favicon_url = "https://raw.githubusercontent.com/yasuo-ozu/type-leak/main/logo.png")]
#![cfg_attr(type_leak_diagnostic, feature(proc_macro_diagnostic))]

#[cfg(type_leak_diagnostic)]
extern crate proc_macro;

mod allow;
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
mod unstable;
mod util;
mod view;

pub use syn;

pub use crate::allow::{Allow, Place, Rules};
pub use crate::analysis::CheckResult;
pub use crate::context::{Context, ContextId, Gat, ModuleScope, SelfRequirement};
pub use crate::encode::{array_const, complete_params, substitute_args, EncodeError, Substitute};
pub use crate::error::{to_syn_error, Cause, ErrorItem, Reason};
pub use crate::leaker::{Leaker, LeakerResult};
pub use crate::referrer::{EntryId, NestedSelf, Referrer, Visitor};
pub use crate::repeater::{
    repeater_definition, repeater_key, repeater_projection, slot_const, ImplSelf, RepeaterImpl,
};
pub use crate::view::View;
